"""Device-output metadata, generated-code protocol, and optional real CUDA tests.

The C++ harness validates emitted scalar/index code on CPU, not GPU execution.
Only the explicitly marked CuPy class provides evidence about CUDA launches.
"""
import ctypes
import gc
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm, OutputNode, OutputRecipe, TensorSpec, TermContribution,
    describe_coordinates,
)
from lqcdweave.contraction.results.api import bind_output_assembly
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.output import (
    PackedOutputs, _BoundLayoutPackedOutputs, _OutputOwnerSnapshot, bind_output_layout,
)
from lqcdweave.contraction.execution import output_program as program


def _fixture(root, *arrays):
    coords, _ = describe_coordinates()
    ids = tuple(f't{index}' for index in range(len(arrays)))
    specs, terms = {}, []
    for key, array in zip(ids, arrays, strict=True):
        specs[key] = TensorSpec(array.shape, array.dtype)
        labels = ''.join(chr(ord('a') + i) for i in range(array.ndim))
        terms.append(ContractionTerm((key,), f'{labels}->{labels}'))
    return OutputRecipe(root, ids, coords), SimpleNamespace(terms=tuple(terms), specs=specs)


def _term(index, coefficient=1):
    return OutputNode('term', TermContribution(f't{index}', coefficient))


def _compare_tree(actual, expected):
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _compare_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        for left, right in zip(actual, expected, strict=True):
            _compare_tree(left, right)
    else:
        assert np.asarray(actual).dtype == np.asarray(expected).dtype
        assert np.asarray(actual).shape == np.asarray(expected).shape
        np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-7)


class _RecordingBackend:
    """Protocol double: deliberately does not claim to compute CUDA results."""
    __name__ = 'cupy'
    ndarray = np.ndarray

    def __init__(self):
        self.uploads, self.allocations, self.kernels, self.launches = [], [], [], []

    def asarray(self, value):
        self.uploads.append(value)
        return np.array(value, copy=True)

    def empty(self, shape, dtype):
        self.allocations.append((shape, np.dtype(dtype)))
        return np.empty(shape, dtype=dtype)

    def ElementwiseKernel(self, inputs, outputs, operation, name, **kwargs):
        self.kernels.append((inputs, outputs, operation, name, kwargs))
        def launch(*arguments, size):
            self.launches.append((name, size))
            arguments[-1].fill(0)
        return launch


class _CountingArray(np.ndarray):
    """Array double that distinguishes snapshot owners from term views."""
    view_shape_reads = 0
    view_dtype_reads = 0
    owner_shape_reads = 0
    owner_dtype_reads = 0
    corrupt_views = False

    def __array_finalize__(self, source):
        self._snapshot_owner = False

    @property
    def shape(self):
        if getattr(self, '_snapshot_owner', False):
            type(self).owner_shape_reads += 1
        else:
            type(self).view_shape_reads += 1
        if type(self).corrupt_views and not getattr(self, '_snapshot_owner', False):
            return (999,)
        return np.ndarray.shape.__get__(self)

    @property
    def dtype(self):
        if getattr(self, '_snapshot_owner', False):
            type(self).owner_dtype_reads += 1
        else:
            type(self).view_dtype_reads += 1
        if type(self).corrupt_views and not getattr(self, '_snapshot_owner', False):
            return np.dtype('float32')
        return np.ndarray.dtype.__get__(self)

    @classmethod
    def reset_counts(cls):
        cls.view_shape_reads = cls.view_dtype_reads = 0
        cls.owner_shape_reads = cls.owner_dtype_reads = 0


class _CountingBackend(_RecordingBackend):
    ndarray = _CountingArray

    def empty(self, shape, dtype):
        self.allocations.append((shape, np.dtype(dtype)))
        owner = np.empty(shape, dtype=dtype).view(_CountingArray)
        owner._snapshot_owner = True
        return owner


def _native_snapshot_fixture(*, debug=False):
    values = (np.arange(4, dtype='complex128'), np.arange(4, dtype='complex128') + 1)
    root = OutputNode('add', children=(_term(0), _term(1)))
    recipe, problem = _fixture(root, *values)
    backend = _CountingBackend()
    bound = program.bind_device_output_assembly(recipe, problem, backend, debug=debug)
    layout = bind_output_layout(recipe.term_ids, problem.specs, backend=backend, validate=False)
    packed = layout.copy(dict(zip(recipe.term_ids, values, strict=True)), backend)
    return values, recipe, problem, backend, bound, layout, packed


class DeviceOutputMetadataTests(unittest.TestCase):
    def test_expression_metadata_preserves_constructor_repr_weakrefs_and_copy(self):
        import copy
        from dataclasses import replace
        import inspect

        self.assertEqual(tuple(inspect.signature(program._Expression).parameters),
                         ('kind', 'shape', 'probe', 'children', 'value'))
        leaf = program._Expression('constant', (), 1., value=1.)
        borrowed = weakref.ref(leaf)
        self.assertIs(borrowed(), leaf)
        self.assertEqual(repr(leaf),
            "_Expression(kind='constant', shape=(), probe=1.0, children=(), value=1.0)")
        changed = replace(leaf, probe=np.float32(.5), value=np.float32(.5))
        self.assertEqual(changed.dtype, np.dtype('float32'))
        self.assertEqual(changed.work, 1)
        root = program._Expression('add', (), 2., (leaf, leaf))
        cloned = copy.deepcopy(root)
        self.assertEqual(root.work, 3)
        self.assertEqual(cloned.work, root.work)
        self.assertEqual(cloned.dtype, root.dtype)
        self.assertIs(cloned.children[0], cloned.children[1])
        self.assertIsNot(cloned.children[0], leaf)
        self.assertNotEqual(root, cloned)
        self.assertEqual(len({root, cloned}), 2)

        class Extension(program._Expression):
            pass

        extended = Extension('constant', (), 3., value=3.)
        extended.annotation = 'extension metadata'
        combined = program._Expression('add', (), 6., (extended, extended))
        self.assertEqual(extended.annotation, 'extension metadata')
        self.assertEqual(combined.work, 3)
        del root, leaf
        gc.collect()
        self.assertIsNone(borrowed())

    def test_kernel_identity_framing_and_legacy_fallback_keep_code_metadata(self):
        import ast
        import hashlib
        import struct

        real_sha = hashlib.sha256
        traces = []

        class HashTrace:
            def __init__(self, value=b''):
                self.parts = [value] if value else []
                self.digest = real_sha(value)
                traces.append(self)

            def update(self, value):
                self.parts.append(value)
                self.digest.update(value)

            def hexdigest(self):
                return self.digest.hexdigest()

        class CustomString(str):
            def __repr__(self):
                return 'custom_code_repr'

        def factory(*args, **kwargs):
            return args, kwargs

        operations = ('', 'x0', 'a\nb\n', '"double"', "single ' quote", 'back\\slash',
                      'control' + chr(0), 'unicode α', 'surrogate ' + chr(0xd800),
                      CustomString('code'), None, 'x' * 65537 + '\n')
        layouts = (('raw complex128 x0', 'complex128 output', ('--std=c++17', '--fmad=false'), True),
                   (CustomString('params'), 'out', (), False),
                   ('p', 'o', ('nested', np.int64(3)), False))
        for version in (1, 2, 4):
            for params, output, options, native_metadata in layouts:
                for operation in operations:
                    with self.subTest(version=version, layout=repr((params, output, options)),
                                      operation=repr(operation)[:40]):
                        traces.clear()
                        program._output_kernel.cache_clear()
                        with patch.object(hashlib, 'sha256', HashTrace):
                            args, kwargs = program._output_kernel(
                                factory, params, output, operation, options, version)
                        self.assertEqual(len(traces), 1)
                        wire = b''.join(traces[0].parts)
                        if version == 2 and native_metadata and type(operation) is str:
                            header = b'ED.output-kernel' + bytes([0]) + b'v2' + bytes([0])
                            self.assertTrue(wire.startswith(header))
                            cursor = len(header)
                            for value in (params, output, operation, program._PREAMBLE):
                                size, = struct.unpack_from('<Q', wire, cursor)
                                cursor += 8
                                actual = wire[cursor:cursor + size].decode('utf-8', 'surrogatepass')
                                self.assertEqual(actual, value)
                                cursor += size
                            size, = struct.unpack_from('<Q', wire, cursor)
                            cursor += 8
                            metadata = ast.literal_eval(wire[cursor:cursor + size].decode())
                            self.assertEqual(metadata, (options, version))
                            self.assertEqual(cursor + size, len(wire))
                        else:
                            expected = repr((params, output, operation, program._PREAMBLE, options, version)).encode()
                            self.assertEqual(wire, expected)
                        self.assertEqual(args, (params, output, operation,
                            f'ed_output_v{version}_{real_sha(wire).hexdigest()[:24]}'))
                        self.assertEqual(kwargs, dict(preamble=program._PREAMBLE, options=options))
        # Plain concatenation would confuse these two parameter boundaries.
        names = [program._output_kernel(factory, p, o, 'body', (), 2)[0][3]
                 for p, o in (('a', 'bc'), ('ab', 'c'))]
        self.assertNotEqual(*names)
        program._output_kernel.cache_clear()

    def test_shared_output_nodes_compile_once_without_retaining_identity_lookup(self):
        term = _term(0)
        shared = OutputNode('add', children=(term, term))
        root = OutputNode('tuple', children=(shared, shared))
        recipe, problem = _fixture(root, np.ones(2, dtype='complex128'))
        original = program._Description._compile
        compiled = []

        def record(description, node):
            compiled.append(id(node))
            return original(description, node)

        with patch.object(program._Description, '_compile', record):
            description = program._Description(recipe, problem)
        self.assertEqual(compiled.count(id(shared)), 1)
        self.assertEqual(compiled.count(id(term)), 1)
        self.assertIs(description.fields[0][1], description.fields[1][1])
        self.assertFalse(any(type(value) is dict and id(shared) in value
                             for value in vars(description).values()))

    def test_failed_output_construction_releases_identity_lookup_and_restores_error_policy(self):
        shared = OutputNode('add', children=(_term(0), _term(0)))
        invalid = OutputNode('constant', np.float16(1))
        root = OutputNode('tuple', children=(shared, invalid))
        recipe, problem = _fixture(root, np.ones(2, dtype='complex128'))
        structure = program._Description._structure
        compile_node = program._Description._compile
        instances, populated_before_failure = [], []

        def record_structure(description, node, path):
            if not path:
                instances.append(description)
            return structure(description, node, path)

        def record_compile(description, node):
            if node is invalid:
                populated_before_failure.append(any(type(value) is dict and id(shared) in value
                    for value in vars(description).values()))
            return compile_node(description, node)

        with np.errstate(all='raise'):
            policy, callback = np.geterr(), np.geterrcall()
            with patch.object(program._Description, '_structure', record_structure), \
                 patch.object(program._Description, '_compile', record_compile):
                with self.assertRaisesRegex(TypeError, 'does not support dtype float16'):
                    program._Description(recipe, problem)
            self.assertEqual(np.geterr(), policy)
            self.assertIs(np.geterrcall(), callback)
        self.assertEqual(populated_before_failure, [True])
        self.assertEqual(len(instances), 1)
        self.assertFalse(any(type(value) is dict and id(shared) in value
                             for value in vars(instances[0]).values()))

    def test_repeated_native_literal_text_preserves_signed_zero_and_nonfinite_values(self):
        complex_values = tuple(complex(real, imag) for real, imag in
            ((0., 0.), (-0., 0.), (0., -0.), (-0., -0.)))
        complex_literals = tuple('complex<double>(' + real + ', ' + imag + ')' for real, imag in (
            ('double(0x0.0p+0)', 'double(0x0.0p+0)'),
            ('double(-0x0.0p+0)', 'double(0x0.0p+0)'),
            ('double(0x0.0p+0)', 'double(-0x0.0p+0)'),
            ('double(-0x0.0p+0)', 'double(-0x0.0p+0)')))
        cases = (
            ((0, 1, 0, 1), 'long long', ('0LL', '1LL', '0LL', '1LL')),
            ((0., -0., 0., -0., float('inf'), -float('inf'), float('nan'), float('nan')),
             'double', ('double(0x0.0p+0)', 'double(-0x0.0p+0)',
                        'double(0x0.0p+0)', 'double(-0x0.0p+0)',
                        'double(INFINITY)', 'double(-INFINITY)', 'double(NAN)', 'double(NAN)')),
            (complex_values + complex_values[::-1], 'complex<double>',
             complex_literals + complex_literals[::-1]),
        )
        for values, ctype, expected in cases:
            with self.subTest(ctype=ctype):
                root = OutputNode('array', children=tuple(OutputNode('constant', value) for value in values),
                                  shape=(len(values),))
                recipe, problem = _fixture(root)
                backend = _RecordingBackend()
                program.bind_device_output_assembly(recipe, problem, backend)
                self.assertEqual(len(backend.kernels), 1)
                actual = tuple(line.split(' = ', 1)[1][:-1]
                    for line in backend.kernels[0][2].splitlines() if line.startswith(f'const {ctype} v'))
                self.assertEqual(actual, expected)

    def test_literal_text_storage_is_bounded_and_opaque_conversions_are_not_retained(self):
        class Probe:
            calls = 0
            def __float__(self):
                self.calls += 1
                return float(self.calls)

        emitter = program._Emitter(SimpleNamespace())
        value = Probe()
        borrowed = weakref.ref(value)
        node = SimpleNamespace(kind='constant', dtype=np.dtype('float64'), value=value)
        emitter.emit(node, 'i')
        emitter.emit(node, 'i')
        self.assertEqual(value.calls, 2)
        self.assertIn('double(0x1.0000000000000p+0)', emitter.lines[0])
        self.assertIn('double(0x1.0000000000000p+1)', emitter.lines[1])
        del node, value
        gc.collect()
        self.assertIsNone(borrowed())
        for index in range(512):
            emitter.emit(SimpleNamespace(kind='constant', dtype=np.dtype('float64'),
                                         value=float(index + 2)), 'i')
        self.assertLessEqual(len(getattr(emitter, 'literal_text', {})), 128)

    def test_opaque_probe_conversion_order_and_rejection_are_preserved(self):
        for operation in ('constant', 'add', 'multiply'):
            for unsupported in (False, True):
                with self.subTest(operation=operation, unsupported=unsupported):
                    events = []

                    class Probe:
                        calls = 0

                        def __array__(self, dtype=None, copy=None):
                            self.calls += 1
                            events.append(('array', self.calls))
                            return np.ones((), dtype=('float16' if unsupported else
                                'float32' if self.calls == 1 else 'float64'))

                    probe = Probe()

                    class Operand:
                        shape = ()

                        def __init__(self, name):
                            self.name = name
                            self.probe = self

                        @property
                        def work(self):
                            events.append(('work', self.name))
                            return 1

                        def __add__(self, other):
                            return probe

                        def __mul__(self, other):
                            return probe

                    def build():
                        if operation == 'constant':
                            return program._constant(probe)
                        return program._binary(operation, Operand('left'), Operand('right'))

                    if unsupported:
                        with self.assertRaisesRegex(TypeError, 'does not support dtype float16'):
                            build()
                        self.assertEqual(events, [('array', 1)])
                    else:
                        expression = build()
                        self.assertEqual(expression.dtype, np.dtype('float64'))
                        expected = [('array', 1)]
                        if operation != 'constant':
                            expected.extend((('work', 'left'), ('work', 'right')))
                        expected.append(('array', 2))
                        self.assertEqual(events, expected)

    def test_shared_dependency_arguments_keep_first_use_order_and_unique_sources(self):
        recipe, problem = _fixture(_term(0), np.ones(2), np.ones(()), np.ones((2, 3)))
        description = program._Description(recipe, problem)
        description.constants = (np.ones((3,), dtype='complex128'),)
        description.temp_specs = (((4,), np.dtype('float32')),)

        def node(kind, value=None, children=()):
            return program._Expression(kind, (), 0, children, value)

        shared = node('add', children=(node('input', 2), node('temporary', 0)))
        root = node('add', children=(shared, node('literal_array', 0), shared,
                                     node('input', 2), node('input', 0)))
        self.assertEqual(description._arguments(root), (
            (('input', 2), (2, 3), np.dtype('float64')),
            (('temporary', 0), (4,), np.dtype('float32')),
            (('literal_array', 0), (3,), np.dtype('complex128')),
            (('input', 0), (2,), np.dtype('float64')),
        ))

    def test_indirect_snapshot_tables_use_current_owners_and_read_positions_once(self):
        class PointerArray(np.ndarray):
            @property
            def data(self):
                return SimpleNamespace(ptr=self.ctypes.data)

        class PointerBackend(_RecordingBackend):
            ndarray = PointerArray

            def empty(self, shape, dtype):
                return super().empty(shape, dtype).view(PointerArray)

            def asarray(self, value):
                return super().asarray(value).view(PointerArray)

        count = 220
        original_positions = _OutputOwnerSnapshot.positions.fget
        for shape in ((), (2, 3)):
            with self.subTest(shape=shape):
                values = tuple(np.ones(shape, dtype='complex128') for _ in range(count))
                root = OutputNode('diagram_sum', tuple((f't{i}', .5j) for i in range(count)))
                recipe, problem = _fixture(root, *values)
                backend = PointerBackend()
                bound = program.bind_device_output_assembly(recipe, problem, backend)
                self.assertEqual(bound.description.indirect_stages, [True])
                layout = bind_output_layout(recipe.term_ids, problem.specs, validate=False)
                first = np.empty((2 * count, *shape), dtype='complex128').view(PointerArray)[::2]
                second = np.empty_like(first, shape=(2 * count, *shape))[::-2]
                if shape:
                    second = second[..., ::-1]

                def check_table(owner):
                    expected = np.empty((count, 1 + len(shape)), dtype='uint64')
                    for lane in range(count):
                        value = owner[lane:lane + 1].reshape(()) if not shape else owner[lane]
                        expected[lane, 0] = value.ctypes.data
                        expected.view('int64')[lane, 1:] = value.strides
                    np.testing.assert_array_equal(bound.call_owners[-2], expected.ravel())
                    np.testing.assert_array_equal(bound.call_owners[-1], expected.ravel())

                for owner in (first, second, first):
                    snapshot = _OutputOwnerSnapshot((owner,), layout)
                    reads = []
                    def positions(value):
                        reads.append(1)
                        return original_positions(value)
                    with patch.object(_OutputOwnerSnapshot, 'positions', property(positions)):
                        bound(snapshot)
                    self.assertEqual(len(reads), 2)  # Sequence length and this call's table.
                    check_table(owner)
                    self.assertEqual(bound.last_stage_launches, 1)

                snapshot = _OutputOwnerSnapshot((second,), layout)
                with execution_debug(True):
                    bound(snapshot)
                check_table(second)
                bound(tuple(snapshot))
                check_table(second)

    def test_constant_literals_preserve_signed_zero_subnormal_and_nonfinite_values(self):
        cases = (
            (np.float32(0.), 'float(0x0.0p+0)'),
            (np.float32(-0.), 'float(-0x0.0p+0)'),
            (np.float64(0.), 'double(0x0.0p+0)'),
            (np.float64(-0.), 'double(-0x0.0p+0)'),
            (np.float32(2. ** -149), 'float(0x1.0000000000000p-149)'),
            (np.float64(2. ** -1074), 'double(0x0.0000000000001p-1022)'),
            (np.float32(np.inf), 'float(INFINITY)'),
            (np.float64(-np.inf), 'double(-INFINITY)'),
            (np.float64(np.nan), 'double(NAN)'),
            (np.complex64(complex(-0., np.inf)),
             'complex<float>(float(-0x0.0p+0), float(INFINITY))'),
            (np.complex128(complex(np.nan, -np.inf)),
             'complex<double>(double(NAN), double(-INFINITY))'),
        )
        for value, literal in cases:
            with self.subTest(literal=literal):
                recipe, problem = _fixture(OutputNode('constant', value))
                backend = _RecordingBackend()
                bound = program.bind_device_output_assembly(recipe, problem, backend, debug=False)
                self.assertEqual(bound.fields, (((), (), value.dtype.str),))
                self.assertIn(literal, backend.kernels[0][2])

    def test_probe_binding_restores_caller_error_policy_on_success_and_failure(self):
        import warnings

        class ErrorSink:
            def __init__(self):
                self.events = []

            def __call__(self, *args):
                self.events.append(args)

            def write(self, message):
                self.events.append(message)

        def constant(value):
            return OutputNode('constant', value)

        cases = (
            ('overflow', OutputNode('multiply', children=(constant(np.float64(1e308)),
                                                         constant(np.float64(1e308)))), None),
            ('invalid_add', OutputNode('add', children=(constant(np.float64(np.inf)),
                                                      constant(np.float64(-np.inf)))), None),
            ('invalid_product', OutputNode('multiply', children=(constant(np.float64(0)),
                                                                constant(np.float64(np.inf)))), None),
            ('zero_inverse', OutputNode('power', -1, (
                OutputNode('zeros', 'float64', shape=(2,)),)), None),
            ('bad_shape', OutputNode('add', children=(_term(0), _term(1))), ValueError),
            ('unsupported_dtype', constant(np.float16(1)), TypeError),
            ('python_zero_inverse', OutputNode('power', -1, (constant(0.0),)), ZeroDivisionError),
            ('python_overflow', OutputNode('power', 2, (constant(1e308),)), OverflowError),
        )
        for mode in ('ignore', 'raise', 'warn', 'call', 'print', 'log'):
            sink = ErrorSink()
            with np.errstate(all=mode, call=sink), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                policy = np.geterr().copy()
                test = self

                class Backend(_RecordingBackend):
                    def ElementwiseKernel(self, *args, **kwargs):
                        # Kernel creation is outside the metadata probe scope.
                        test.assertEqual(np.geterr(), policy)
                        test.assertIs(np.geterrcall(), sink)
                        return super().ElementwiseKernel(*args, **kwargs)

                for name, root, error in cases:
                    with self.subTest(mode=mode, case=name):
                        values = (np.ones(2), np.ones(3)) if name == 'bad_shape' else ()
                        recipe, problem = _fixture(root, *values)
                        backend = Backend()
                        if error is None:
                            bound = program.bind_device_output_assembly(recipe, problem, backend)
                            self.assertEqual(program.describe_device_output_fields(recipe, problem),
                                             bound.fields)
                            self.assertTrue(backend.kernels)
                        else:
                            with self.assertRaises(error):
                                program.bind_device_output_assembly(recipe, problem, backend)
                            with self.assertRaises(error):
                                program.describe_device_output_fields(recipe, problem)
                            self.assertEqual(backend.kernels, [])
                        self.assertEqual(np.geterr(), policy)
                        self.assertIs(np.geterrcall(), sink)
                self.assertEqual(captured, [])
                self.assertEqual(sink.events, [])


    def test_combined_offline_reservation_uses_one_description(self):
        root = OutputNode('dict', keys=('matrix', 'vector'), children=(_term(0), _term(1)))
        recipe, problem = _fixture(root, np.ones((2, 3), dtype='complex128'),
                                   np.ones((4,), dtype='float32'))
        with patch.object(program, '_Description', wraps=program._Description) as describe:
            result = program.describe_device_output_reservation(recipe, problem)
            self.assertEqual(describe.call_count, 1)
        self.assertEqual(result, dict(fields=((('matrix',), (2, 3), '<c16'),
                                             (('vector',), (4,), '<f4')),
                                     device_bytes=112, device_owner_bytes=(96, 16), metadata_bytes=0))
        self.assertEqual(result['device_bytes'], program.estimate_device_output_bytes(recipe, problem))
        self.assertEqual(result['metadata_bytes'], program.estimate_device_output_metadata_bytes(recipe, problem))

    def test_scalarization_and_numpy_weak_promotion_match_R(self):
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            for shape in ((), (3,)):
                for coefficient in (1, 2.5, -0.5j):
                    value = np.ones(shape, dtype=dtype)
                    recipe, problem = _fixture(_term(0, coefficient), value)
                    expected = np.asarray(bind_output_assembly(recipe)((value,)))
                    self.assertEqual(program.describe_device_output_fields(recipe, problem),
                                     (((), expected.shape, expected.dtype.str),))
                    bound = program.bind_device_output_assembly(
                        recipe, problem, _RecordingBackend(), debug=False)
                    self.assertEqual(bound.fields, (((), expected.shape, expected.dtype.str),))

    def test_scalar_pairs_and_array_broadcast_shapes_match_R(self):
        shapes = (((), ()), ((), (2, 3)), ((2, 3), ()),
                  ((2, 1), (1, 3)), ((0, 1), (1, 3)),
                  ((3,), (3,)), ((2, 3), (2, 3)))
        for errors in ('ignore', 'raise'):
            for dtype in ('float64', 'complex128'):
                for left, right in shapes:
                    for kind in ('add', 'multiply'):
                        with self.subTest(errors=errors, dtype=dtype, shapes=(left, right), kind=kind):
                            values = (np.full(left, .75, dtype=dtype),
                                      np.full(right, 1.25, dtype=dtype))
                            root = OutputNode(kind, children=(_term(0, -.5j), _term(1)))
                            recipe, problem = _fixture(root, *values)
                            with np.errstate(all=errors):
                                expected = np.asarray(bind_output_assembly(recipe)(values))
                                for debug in (False, True):
                                    bound = program.bind_device_output_assembly(
                                        recipe, problem, _RecordingBackend(), debug=debug)
                                    self.assertEqual(bound.fields,
                                                     (((), expected.shape, expected.dtype.str),))
        values = (np.ones(2), np.ones(3))
        recipe, problem = _fixture(OutputNode('add', children=(_term(0), _term(1))), *values)
        with self.assertRaises(ValueError):
            bind_output_assembly(recipe)(values)
        with self.assertRaises(ValueError):
            program.describe_device_output_fields(recipe, problem)

    def test_indexed_numpy_scalar_remains_strong_and_reshape_zero_dim_array(self):
        value = np.array([2], dtype='float32')
        for child in (OutputNode('select', (0,), (_term(0),)),
                      OutputNode('reshape', children=(_term(0),), shape=())):
            root = OutputNode('add', children=(child, OutputNode('constant', 1.5)))
            recipe, problem = _fixture(root, value)
            result = bind_output_assembly(recipe)((value,))
            self.assertEqual(program.describe_device_output_fields(recipe, problem)[0][2],
                             np.asarray(result).dtype.str)

    def test_nested_containers_and_uniform_arrays(self):
        value = np.arange(3, dtype='complex64')
        uniform = OutputNode('array', children=(_term(0), _term(0, 2)), shape=(1, 2))
        root = OutputNode('dict', keys=('series', 'rest'), children=(uniform,
            OutputNode('tuple', children=(OutputNode('list'), OutputNode('zeros', 'complex128', shape=(0, 2))))))
        recipe, problem = _fixture(root, value)
        self.assertEqual(program.describe_device_output_fields(recipe, problem), (
            (('series',), (1, 2, 3), '<c8'), (('rest', 1), (0, 2), '<c16')))
        self.assertEqual(program.estimate_device_output_bytes(recipe, problem), 48)

    def test_rejects_unsupported_dtype_ragged_and_scanned_recipe(self):
        for dtype in ('float16', 'longdouble'):
            value = np.ones((3,), dtype=dtype)
            recipe, problem = _fixture(_term(0), value)
            with self.subTest(dtype=dtype):
                if dtype == 'longdouble' and value.dtype == np.dtype('float64'):
                    # Windows may alias longdouble to supported binary64.
                    self.assertEqual(program.describe_device_output_fields(recipe, problem)[0][2],
                                     np.dtype('float64').str)
                else:
                    with self.assertRaisesRegex(TypeError, 'dtype'):
                        program.describe_device_output_fields(recipe, problem)
        for root, values in (
            (OutputNode('array', children=(_term(0), _term(1)), shape=(2,)),
             (np.ones(2), np.ones(3))),
            (OutputNode('array', children=(OutputNode('constant_array', np.ones(2), shape=(2,)),), shape=(1,)), ())):
            recipe, problem = _fixture(root, *values)
            with self.assertRaisesRegex(TypeError, 'ragged|object'):
                program.describe_device_output_fields(recipe, problem)
        coords, _ = describe_coordinates(time_axes={'t': (0,)})
        recipe = OutputRecipe(OutputNode('constant', 1.), (), coords)
        with self.assertRaisesRegex(ValueError, 'fixed single-point'):
            program.describe_device_output_fields(recipe, SimpleNamespace(terms=(), specs={}))

    def test_legacy_stable_wide_scalar_array_promotion_is_supported(self):
        for dtype in ('float64', 'complex128'):
            vector = np.ones(3, dtype=dtype)
            scalar = np.array(1.2 + 0.7j, dtype='complex128')
            recipe, problem = _fixture(OutputNode('add', children=(_term(0), _term(1))), vector, scalar)
            expected = bind_output_assembly(recipe)((vector, scalar))
            self.assertEqual(program.describe_device_output_fields(recipe, problem)[0][2], expected.dtype.str)
        if np.lib.NumpyVersion(np.__version__) < '2.0.0':
            vector = np.ones(3, dtype='float32')
            scalar = np.array(1.2, dtype='float64')
            recipe, problem = _fixture(OutputNode('add', children=(_term(0), _term(1))), vector, scalar)
            with self.assertRaisesRegex(TypeError, 'value-dependent'):
                program.describe_device_output_fields(recipe, problem)

    def test_python_integer_scalar_terms_and_legacy_zero_dim_ambiguity_are_rejected(self):
        scalar, vector = np.array(1000, dtype='int64'), np.ones(3, dtype='int8')
        recipe, problem = _fixture(OutputNode('multiply', children=(_term(0), _term(1))), scalar, vector)
        with self.assertRaisesRegex(TypeError, 'floating/complex scalar terms'):
            program.describe_device_output_fields(recipe, problem)
        if np.lib.NumpyVersion(np.__version__) < '2.0.0':
            scalar = np.array([1e200], dtype='float64')
            vector = np.ones(3, dtype='float32')
            zero_dim = OutputNode('reshape', children=(_term(0),), shape=())
            recipe, problem = _fixture(OutputNode('multiply', children=(zero_dim, _term(1))), scalar, vector)
            with self.assertRaisesRegex(TypeError, 'value-dependent'):
                program.describe_device_output_fields(recipe, problem)

    def test_large_signature_uses_one_metadata_upload_and_retains_both_owners(self):
        class DeviceDouble(np.ndarray):
            @property
            def data(self):
                return SimpleNamespace(ptr=self.ctypes.data)
        values = tuple(np.ones((2, 3), dtype='complex128').view(DeviceDouble) for _ in range(100))
        root = OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(len(values))))
        recipe, problem = _fixture(root, *values)
        backend = _RecordingBackend()
        bound = program.bind_device_output_assembly(recipe, problem, backend)
        self.assertEqual(backend.kernels[0][0], 'raw uint64 addresses')
        self.assertEqual(program.estimate_device_output_metadata_bytes(recipe, problem), 2400)
        self.assertEqual(program.estimate_device_output_bytes(recipe, problem), 2496)
        bound(values)
        self.assertEqual(len(backend.uploads), 1)
        self.assertEqual(len(backend.launches), 1)
        self.assertEqual(len(bound.call_owners), 2)
        self.assertIs(bound.call_owners[0], backend.uploads[0])
        first_owners = bound.call_owners
        bound(values)
        self.assertIsNot(first_owners[1], bound.call_owners[1])
        self.assertEqual(first_owners[0][0], values[0].data.ptr)

    def test_long_ordered_sum_segments_and_reserves_reusable_scratch_offline(self):
        coordinates, _ = describe_coordinates()
        term = ContractionTerm(('input',), 'a->a')
        for count in (1024, 23056):
            ids = tuple(f't{i}' for i in range(count))
            recipe = OutputRecipe(OutputNode('diagram_sum', tuple((key, 1) for key in ids)), ids, coordinates)
            problem = SimpleNamespace(terms=(term,) * count, specs={'input': TensorSpec((2,), 'complex128')})
            report = program.describe_device_output_program(recipe, problem)
            self.assertTrue(report['supported'])
            self.assertEqual(report['field_count'], 1)
            self.assertLessEqual(max(report['stage_node_counts']), 8192)
            self.assertEqual(report['device_bytes'], program.estimate_device_output_bytes(recipe, problem))
            if count == 1024:
                self.assertEqual(report['stage_count'], 1)
                self.assertEqual(report['peak_scratch_bytes'], 0)
            else:
                self.assertGreater(report['stage_count'], 1)
                self.assertLess(report['stage_count'], 32)
                self.assertEqual(report['scratch_slot_count'], 2)
                self.assertEqual(report['peak_scratch_bytes'], 64)
                self.assertGreater(report['metadata_bytes'], 0)

    def test_hot_path_uses_metadata_only_without_value_reads_or_synchronization(self):
        class MetadataOnly:
            def __init__(self, shape, dtype, pointer=4096):
                self.shape, self.dtype = shape, np.dtype(dtype)
                self.strides = tuple(int(np.prod(shape[axis + 1:])) * self.dtype.itemsize
                                     for axis in range(len(shape)))
                self.data = SimpleNamespace(ptr=pointer)
            def forbidden(self, *args, **kwargs):
                raise AssertionError('GPU values must not be read or synchronized on this hot path')
            __array__ = __int__ = __float__ = __complex__ = item = get = tolist = forbidden
        class MetadataBackend(_RecordingBackend):
            ndarray = MetadataOnly
            def empty(self, shape, dtype):
                return MetadataOnly(shape, dtype)
            def ElementwiseKernel(self, inputs, outputs, operation, name, **kwargs):
                self.kernels.append((inputs, outputs, operation, name, kwargs))
                def launch(*args, size):
                    self.launches.append((name, size))
                return launch
        for count in (1, 100):
            host = tuple(np.ones((2, 3), dtype='complex128') for _ in range(count))
            root = OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(count)))
            recipe, problem = _fixture(root, *host)
            backend = MetadataBackend()
            values = tuple(MetadataOnly((2, 3), 'complex128', 4096 + 128 * i) for i in range(count))
            bound = program.bind_device_output_assembly(recipe, problem, backend)
            result = bound(values)
            self.assertEqual(result.shape, (2, 3))
            self.assertEqual(len(backend.launches), 1)
            self.assertEqual(len(backend.uploads), 0 if count == 1 else 1)

    def test_native_snapshot_first_call_scans_terms_then_reuses_owner_schema(self):
        values, _, _, _, bound, layout, packed = _native_snapshot_fixture()
        self.assertIs(type(packed), _BoundLayoutPackedOutputs)
        self.assertIs(packed.provenance, layout)

        _CountingArray.reset_counts()
        bound(packed)
        self.assertEqual(_CountingArray.view_shape_reads, len(values))
        self.assertEqual(_CountingArray.view_dtype_reads, len(values))
        self.assertEqual(len(bound._owner_layout_schemas), 1)

        _CountingArray.reset_counts()
        _CountingArray.corrupt_views = True
        try:
            bound(packed)
        finally:
            _CountingArray.corrupt_views = False
        self.assertEqual(_CountingArray.view_shape_reads, 0)
        self.assertEqual(_CountingArray.view_dtype_reads, 0)
        self.assertEqual(_CountingArray.owner_shape_reads, 1)
        self.assertEqual(_CountingArray.owner_dtype_reads, 1)

        _CountingArray.reset_counts()
        _CountingArray.corrupt_views = True
        try:
            with execution_debug(True):
                with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                    bound(packed)
        finally:
            _CountingArray.corrupt_views = False
        self.assertGreaterEqual(_CountingArray.view_shape_reads, 1)

    def test_debug_and_external_values_keep_full_term_abi_checks(self):
        values, _, _, _, bound, _, packed = _native_snapshot_fixture(debug=True)
        _CountingArray.reset_counts()
        _CountingArray.corrupt_views = True
        try:
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                bound(packed)
        finally:
            _CountingArray.corrupt_views = False
        self.assertGreaterEqual(_CountingArray.view_shape_reads, 1)

        _, _, _, _, cached_bound, _, native = _native_snapshot_fixture()
        cached_bound(native)
        external_values = tuple(native)
        external_packed = PackedOutputs(native.owners, native.positions)
        for external in (external_values, external_packed):
            _CountingArray.reset_counts()
            cached_bound(external)
            self.assertEqual(_CountingArray.view_shape_reads, len(values))
            self.assertEqual(_CountingArray.view_dtype_reads, len(values))

    def test_layout_identity_and_expired_weakref_never_false_hit(self):
        values, recipe, problem, backend, bound, first_layout, first = _native_snapshot_fixture()
        bound(first)

        second_layout = bind_output_layout(recipe.term_ids, problem.specs,
                                           backend=backend, validate=False)
        self.assertIsNot(first_layout, second_layout)
        self.assertEqual(first_layout, second_layout)
        second = second_layout.copy(dict(zip(recipe.term_ids, values, strict=True)), backend)
        _CountingArray.reset_counts()
        bound(second)
        self.assertEqual(_CountingArray.view_shape_reads, len(values))
        self.assertEqual(_CountingArray.view_dtype_reads, len(values))

        stale_layout = second_layout
        stale_reference = weakref.ref(stale_layout)
        del second, second_layout, stale_layout
        gc.collect()
        self.assertIsNone(stale_reference())

        fresh_layout = bind_output_layout(recipe.term_ids, problem.specs,
                                          backend=backend, validate=False)
        fresh = fresh_layout.copy(dict(zip(recipe.term_ids, values, strict=True)), backend)
        # Model id reuse with a stale weak identity at the new layout's key.
        bound._owner_layout_schemas[id(fresh_layout)] = (
            stale_reference, ((1, (999,), np.dtype('float32')),))
        _CountingArray.reset_counts()
        bound(fresh)
        self.assertEqual(_CountingArray.view_shape_reads, len(values))
        self.assertEqual(_CountingArray.view_dtype_reads, len(values))
        self.assertIs(bound._owner_layout_schemas[id(fresh_layout)][0](), fresh_layout)

    def test_schema_cache_is_bounded_and_holds_no_layouts_or_owner_arrays(self):
        values, recipe, problem, backend, bound, _, _ = _native_snapshot_fixture()
        layout_refs, owner_refs = [], []
        for _ in range(program._MAX_OUTPUT_LAYOUT_SCHEMAS + 4):
            layout = bind_output_layout(recipe.term_ids, problem.specs,
                                        backend=backend, validate=False)
            packed = layout.copy(dict(zip(recipe.term_ids, values, strict=True)), backend)
            layout_refs.append(weakref.ref(layout))
            owner_refs.append(weakref.ref(packed.owners[0]))
            bound(packed)
            del packed, layout

        gc.collect()
        self.assertLessEqual(len(bound._owner_layout_schemas), 16)
        self.assertTrue(all(reference() is None for reference in layout_refs))
        self.assertTrue(all(reference() is None for reference in owner_refs))
        for layout_reference, schema in bound._owner_layout_schemas.values():
            self.assertIsNone(layout_reference())
            self.assertIsInstance(schema, tuple)
            self.assertTrue(all(isinstance(shape, tuple) and isinstance(dtype, np.dtype)
                                for _, shape, dtype in schema))

    def test_segmented_runtime_reuses_scratch_and_counts_only_actual_launches(self):
        from unittest.mock import patch
        for shape in ((2,), (0, 2)):
            value = np.ones(shape, dtype='complex128')
            root = OutputNode('diagram_sum', tuple(('t0', 1) for _ in range(100)))
            recipe, problem = _fixture(root, value)
            backend = _RecordingBackend()
            with patch.object(program, '_MAX_STAGE_NODES', 64):
                bound = program.bind_device_output_assembly(recipe, problem, backend)
                self.assertGreater(bound.stage_count, 1)
                self.assertEqual(bound.description.scratch_slots.__len__(), 2)
                result = bound((value,))
            self.assertEqual(result.shape, shape)
            self.assertEqual(len(backend.allocations), 3)  # Two scratch owners and one final field.
            self.assertEqual(len(bound.call_owners), 2)
            self.assertEqual(bound.last_stage_launches, bound.stage_count if value.size else 0)
            self.assertEqual(len(backend.launches), bound.last_stage_launches)
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                bound(())
            self.assertEqual(bound.last_stage_launches, 0)

    def test_invalid_matrix_shapes_fail_before_backend_work(self):
        value = np.ones((2, 3))
        for root in (OutputNode('reshape', children=(_term(0),), shape=(7,)),
                     OutputNode('transpose', (1, 0), (_term(0),), shape=(2, 3)),
                     OutputNode('select', (2,), (_term(0),)),
                     OutputNode('scatter', ((0,),), (_term(0),), shape=(3, 3))):
            recipe, problem = _fixture(root, value)
            backend = _RecordingBackend()
            with self.assertRaises(ValueError):
                program.bind_device_output_assembly(recipe, problem, backend)
            self.assertEqual(backend.allocations, [])
            self.assertEqual(backend.uploads, [])

    def test_binding_and_repeated_calls_have_one_kernel_per_field_and_no_uploads(self):
        value = np.arange(4, dtype='complex64')
        literal = OutputNode('constant_array', np.ones(4, dtype='complex64'), shape=(4,))
        root = OutputNode('add', children=(OutputNode('power', 2, (_term(0, 2j),)), literal))
        recipe, problem = _fixture(root, value)
        backend = _RecordingBackend()
        bound = program.bind_device_output_assembly(recipe, problem, backend)
        self.assertEqual(len(backend.kernels), 1)
        self.assertEqual(len(backend.uploads), 1)
        self.assertEqual(backend.kernels[0][4]['options'], ('--std=c++17', '--fmad=false'))
        self.assertIn('ed_output_round', backend.kernels[0][2])
        first, second = bound((value,)), bound((value + 1,))
        self.assertIsNot(first, second)
        self.assertEqual(len(backend.uploads), 1)
        self.assertEqual(len(backend.launches), 2)
        self.assertEqual(len(backend.allocations), 2)
        self.assertEqual(bound.nbytes, 64)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            bound(())
        with self.assertRaisesRegex(TypeError, 'CuPy term arrays'):
            bound((3.,))
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            bound((value.astype('complex128'),))

    def test_backend_policy_and_code_cache_are_internal_and_bounded(self):
        value = np.arange(4, dtype='float64')
        backend = _RecordingBackend()
        backend.cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=False))
        recipe, problem = _fixture(_term(0, .7), value)
        first = program.bind_device_output_assembly(recipe, problem, backend)
        # Fresh cfg/window descriptions and values must not invalidate code.
        recipe2, problem2 = _fixture(_term(0, .7), value + 1)
        second = program.bind_device_output_assembly(recipe2, problem2, backend)
        self.assertIs(first.programs[0][0], second.programs[0][0])
        self.assertEqual(len(backend.kernels), 1)
        # A different numerical coefficient must not reuse the old code.
        changed, changed_problem = _fixture(_term(0, .8), value)
        third = program.bind_device_output_assembly(changed, changed_problem, backend)
        self.assertIsNot(first.programs[0][0], third.programs[0][0])
        backend.cuda.runtime.is_hip = True
        hip = program.bind_device_output_assembly(recipe, problem, backend)
        self.assertEqual(backend.kernels[-1][4]['options'], ('--std=c++17', '-ffp-contract=off'))
        self.assertIsNot(first.programs[0][0], hip.programs[0][0])
        self.assertEqual(program._output_kernel.cache_info().maxsize, 16)

    def test_compile_error_is_reported_with_source_without_retry_or_fallback(self):
        from unittest.mock import patch
        class CompileFailure(Exception):
            code = 'bad generated source'
        class FailingBackend(_RecordingBackend):
            cuda = SimpleNamespace(compiler=SimpleNamespace(CompileException=CompileFailure))
            def ElementwiseKernel(self, *args, **kwargs):
                def launch(*args, **kwargs):
                    self.launches.append(1)
                    raise CompileFailure('syntax error')
                return launch
        backend = FailingBackend()
        recipe, problem = _fixture(_term(0, .7), np.ones(2))
        bound = program.bind_device_output_assembly(recipe, problem, backend)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tempfile, 'tempdir', directory):
                with self.assertRaisesRegex(RuntimeError, 'GPU output kernel compilation failed') as caught:
                    bound((np.ones(2),))
            logs = list(Path(directory).glob('lqcdweave-output-compile-*.log'))
            self.assertEqual(len(logs), 1)
            diagnostic = logs[0].read_text(encoding='utf-8')
            self.assertIn('bad generated source', diagnostic)
            self.assertIn('--std=c++17', diagnostic)
            self.assertIn(str(logs[0]), str(caught.exception))
            self.assertIsInstance(caught.exception.__cause__, CompileFailure)
        self.assertEqual(backend.launches, [1])

    def test_cpu_backend_and_invalid_debug_never_fall_back(self):
        recipe, problem = _fixture(OutputNode('constant', 1.))
        with self.assertRaisesRegex(TypeError, 'CuPy'):
            program.bind_device_output_assembly(recipe, problem, np)
        with self.assertRaisesRegex(TypeError, 'debug must be bool'):
            program.bind_device_output_assembly(recipe, problem, _RecordingBackend(), debug=1)

    def test_zero_size_scatter_and_fields_allocate_without_launch(self):
        value = np.empty((0, 2), dtype='complex64')
        root = OutputNode('scatter', ((),), (_term(0),), shape=(3, 2))
        recipe, problem = _fixture(root, value)
        backend = _RecordingBackend()
        bound = program.bind_device_output_assembly(recipe, problem, backend)
        self.assertIn('= -1LL;', backend.kernels[0][2])
        self.assertEqual(bound((value,)).shape, (3, 2))
        recipe, problem = _fixture(OutputNode('zeros', 'float64', shape=(0, 3)))
        backend = _RecordingBackend()
        result = program.bind_device_output_assembly(recipe, problem, backend)(())
        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(backend.launches, [])


def _five_operator_nested_fixture():
    """Representative meson55-style layout using repository builders, not the external script."""
    import sympy as sp
    from test_structured_correlator import fixture, task
    from lqcdweave.contraction.binding.api import Meson
    from lqcdweave.contracts.elemental import ElementalComponent
    from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
    from lqcdweave.operators.correlator.api import (
        Hadron, HadronIrrep, Tag, Flavor, gen_correlator,
        remove_disconneted_diagram, split_meson_mode_pairs,
    )
    from lqcdweave.contracts.correlator.api import DiagramBindings
    from lqcdweave.contraction.expansion.api import describe_meson_matrix, bind_meson_matrix_catalog
    _, elemental, propagator = fixture()
    component = ElementalComponent(0)
    names = tuple(f'op{i}' for i in range(5))
    operators = {name: Operator.from_terms(name, (
        OperatorTerm(5, component, 0, 1 + (i + 1) * .2j),
        OperatorTerm(1, component, 0, (i + 1) * .1 - .4j))) for i, name in enumerate(names)}
    vertices = {}
    def bind(row):
        key = row.hadron_name, row.dagger
        if key not in vertices:
            vertices[key] = Meson(elemental, operators[key[0]], key[1], phase_convention='anchor')
        return vertices[key]
    def split(request):
        return split_meson_mode_pairs(request, partitions=(0, 1, 2, 3))
    full = split(task(structured=True, flavor='I0', names=names))
    connected = remove_disconneted_diagram(full, [r'S^q_\mathrm{local}'])
    i1 = split(task(structured=True, flavor='I1', names=names))
    axis = [Hadron(HadronIrrep(name, (0, 0, 0), 'A_1', -1, Tag(0, 0))[0],
                   (Flavor('uu') + Flavor('dd')) / sp.sqrt(2)) for name in names]
    loops = {name: split(gen_correlator([axis.copy()], [time], [dagger], structured=True, degenerate=True))
             for name, time, dagger in (('source', 1, True), ('sink', 0, False))}
    request = {'I0': {'two_point': full, 'connected': connected, 'disconnected': full - connected},
               'I1': {'two_point': i1}, 'one_point': loops}
    bindings = DiagramBindings(vertex_map=bind, configuration='cfg', usedNe=3,
        propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator})
    coordinates, _ = describe_coordinates(time_axes={0: 1, 1: 0})
    catalog = describe_meson_matrix(request, bindings=bindings, representative_times={0: 1, 1: 0})
    return bind_meson_matrix_catalog(catalog, bindings=bindings, coordinates=coordinates)


def _run_host_generated(recipe, problem, values, *, indirect=False, optimization="-O2"):
    """Execute exact emitted field bodies with a host C++ ABI for codegen QA."""
    description = program._Description(recipe, problem)
    fields = [None] * len(description.fields)
    scratch = tuple(np.empty(size, dtype=dtype) for size, dtype in description.scratch_slots)
    temporaries = tuple(scratch[description.temp_slots[index]][:int(np.prod(shape))].reshape(shape)
                        for index, (shape, _) in enumerate(description.temp_specs))
    def argument(kind, index):
        return values[index] if kind == 'input' else (temporaries[index]
            if kind == 'temporary' else description.constants[index])
    with tempfile.TemporaryDirectory() as directory:
        for number, stage in enumerate(description.stages):
            expression = stage.expression
            use_indirect = indirect and description.indirect_stages[number]
            emitter = program._Emitter(description, indirect=use_indirect)
            variable = emitter.emit(expression, 'i')
            emitter.line(f'output[i] = {variable};')
            arguments = tuple(emitter.arguments.items())
            parameters = ('const unsigned long long* addresses' if use_indirect else
                ', '.join(f'const {program._CTYPE[spec[2]]}* {spec[0]}' for _, spec in arguments))
            if parameters:
                parameters += ', '
            parameters += f'{program._CTYPE[expression.dtype]}* output'
            source = ('#include <complex>\n#include <cmath>\n'
                '#define __device__\nusing std::complex; using std::pow;\n'
                + program._PREAMBLE + f'\nextern "C" void run({parameters}) {{\n'
                + f'for (long long i=0;i<{np.prod(expression.shape, dtype=np.int64)};++i) {{\n'
                + '\n'.join(emitter.lines) + '\n}}\n')
            cpp, library = Path(directory) / f'{number}.cc', Path(directory) / f'{number}.so'
            cpp.write_text(source)
            subprocess.run([shutil.which('c++'), '-std=c++17', optimization, '-shared', '-fPIC',
                            '-ffp-contract=off', str(cpp), '-o', str(library)],
                           check=True, capture_output=True, text=True)
            function = ctypes.CDLL(str(library)).run
            function.argtypes = [ctypes.c_void_p] * (2 if use_indirect else len(arguments) + 1)
            function.restype = None
            arrays = [np.ascontiguousarray(argument(kind, index))
                      for (kind, index), _ in arguments]
            owners = []
            if use_indirect:
                table = np.zeros(description.table_nbytes // 8, dtype='uint64')
                for (kind, index), shape, dtype, offset in description.table_layout:
                    array = argument(kind, index)
                    owners.append(array)
                    table[offset] = array.ctypes.data
                    table.view('int64')[offset + 1:offset + 1 + len(shape)] = array.strides
                arrays = [table]
            kind, target = stage.target
            output = temporaries[target] if kind == 'temporary' else np.empty(expression.shape, dtype=expression.dtype)
            function(*(array.ctypes.data for array in (*arrays, output)))
            if kind == 'field':
                fields[target] = output
    return program._restore(description.structure, fields)


@unittest.skipUnless(shutil.which('c++'), 'host C++ compiler unavailable; this is not a GPU test')
class GeneratedOutputCodeTests(unittest.TestCase):
    def test_repeated_constant_zero_signs_and_subnormals_match_independent_bits(self):
        cases = (
            (0., -0., 2. ** -1074, -(2. ** -1074), 0., -0.),
            tuple(complex(real, imag) for real, imag in (
                (0., 0.), (-0., 0.), (0., -0.), (-0., -0.),
                (2. ** -1074, -(2. ** -1074)), (-0., 0.), (0., -0.))),
        )
        for values in cases:
            expected = np.asarray(values)
            with self.subTest(dtype=expected.dtype):
                root = OutputNode('array', children=tuple(OutputNode('constant', value) for value in values),
                                  shape=(len(values),))
                recipe, problem = _fixture(root)
                actual = _run_host_generated(recipe, problem, ())
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(actual.view('uint64'), expected.view('uint64'))

    def test_disconnected_outer_product_transpose_scatter_select_against_R_and_independent_fixture(self):
        left = np.arange(4, dtype='float32').reshape(2, 2) + 0.3j
        right = np.arange(6, dtype='float32').reshape(3, 2) - 0.7j
        l = OutputNode('reshape', children=(_term(0, -0.5j),), shape=(2, 1, 2, 1))
        r = OutputNode('reshape', children=(_term(1, 2),), shape=(1, 3, 1, 2))
        product = OutputNode('multiply', children=(l, r))
        transpose = OutputNode('transpose', (2, 3, 0, 1), (product,), shape=(2, 2, 2, 3))
        scatter = OutputNode('scatter', ((3, 1), (0, 4)), (transpose,), shape=(5, 5, 2, 3))
        root = OutputNode('dict', keys=('matrix', 'selected'), children=(scatter,
            OutputNode('select', (3, 4), (scatter,))))
        recipe, problem = _fixture(root, left, right)
        actual = _run_host_generated(recipe, problem, (left, right))
        reference = bind_output_assembly(recipe)((left, right))
        _compare_tree(actual, reference)
        expected = np.zeros((5, 5, 2, 3), dtype='complex64')
        block = ((-0.5j * left)[:, None, :, None] * (2 * right)[None, :, None, :]).transpose(2, 3, 0, 1)
        expected[np.ix_((3, 1), (0, 4))] = block
        np.testing.assert_allclose(actual['matrix'], expected, rtol=4e-6)
        np.testing.assert_allclose(actual['selected'], expected[3, 4], rtol=4e-6)

    def test_ordered_cancellation_scalar_promotion_arrays_and_integer_powers(self):
        a = np.array([1e8, 1e8], dtype='float32')
        b = np.array([-1e8, -1e8], dtype='float32')
        scalar = np.array(1.3 + 0.2j, dtype='complex64')
        add = OutputNode('diagram_sum', (('t0', 1), ('t1', 1)))
        cancellation = OutputNode('add', children=(add, OutputNode('constant', 1.)))
        powers = OutputNode('array', children=tuple(OutputNode('power', exponent, (_term(2),))
            for exponent in (-3, -1, 0, 1, 2, 3)), shape=(2, 3))
        root = OutputNode('tuple', children=(cancellation, powers))
        recipe, problem = _fixture(root, a, b, scalar)
        actual = _run_host_generated(recipe, problem, (a, b, scalar))
        _compare_tree(actual, bind_output_assembly(recipe)((a, b, scalar)))
        np.testing.assert_array_equal(actual[0], np.ones(2, dtype='float32'))
        self.assertEqual(actual[1].dtype, np.dtype('complex128'))

    def test_indirect_pointer_table_handles_negative_strides(self):
        values = tuple((np.arange(12).reshape(3, 4) * (1 + 0.3j) + i)[::-1, ::2]
                       for i in range(100))
        root = OutputNode('diagram_sum', tuple((f't{i}', (-1) ** i * 0.5j) for i in range(len(values))))
        recipe, problem = _fixture(root, *values)
        actual = _run_host_generated(recipe, problem, values, indirect=True)
        _compare_tree(actual, bind_output_assembly(recipe)(values))

    def test_genuine_three_block_structured_I0_recipe_compiles_without_pair_temporaries(self):
        try:
            from test_structured_correlator import fixture, task
        except ModuleNotFoundError as error:
            self.skipTest(f'optional symbolic fixture unavailable: {error}')
        from lqcdweave.contracts.correlator.api import DiagramBindings
        from lqcdweave.contraction.expansion.api import describe_meson_matrix, bind_meson_matrix_catalog
        from lqcdweave.operators.correlator.api import split_meson_mode_pairs
        bind, _, propagator = fixture()
        bindings = DiagramBindings(vertex_map=bind, configuration='cfg', usedNe=3,
            propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator})
        request = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=(0, 1, 2, 3))
        expanded = bind_meson_matrix_catalog(describe_meson_matrix(request, bindings=bindings), bindings=bindings)
        description = program._Description(expanded.output_recipe, expanded.problem)
        self.assertEqual(len(description.term_specs), 99)
        self.assertEqual(description.indirect_fields, [True])
        self.assertEqual(description.table_nbytes, 2232)
        rng = np.random.default_rng(126)
        values = tuple((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
                       for shape, dtype in description.term_specs)
        actual = _run_host_generated(expanded.output_recipe, expanded.problem, values, indirect=True)
        _compare_tree(actual, bind_output_assembly(expanded.output_recipe)(values))
        backend = _RecordingBackend()
        program.bind_device_output_assembly(expanded.output_recipe, expanded.problem, backend)
        self.assertEqual(len(backend.kernels), 1)
        self.assertEqual(backend.allocations, [])

    def test_five_operator_nested_meson_layout_against_complete_cpu_recipe(self):
        try:
            expanded = _five_operator_nested_fixture()
        except ModuleNotFoundError as error:
            self.skipTest(f'optional symbolic fixture unavailable: {error}')
        description = program._Description(expanded.output_recipe, expanded.problem)
        self.assertEqual(len(description.term_specs), 108)
        self.assertEqual([len(term.inputs) for term in expanded.problem.terms].count(6), 81)
        self.assertEqual([len(term.inputs) for term in expanded.problem.terms].count(3), 27)
        self.assertEqual(len(expanded.problem.specs), 56)
        self.assertEqual([expression.shape for _, expression in description.fields],
                         [(5, 5, 6, 6)] * 4 + [(5, 6)] * 2)
        self.assertEqual(description.indirect_fields, [True] * 4 + [False] * 2)
        self.assertEqual(description.table_nbytes, 2232)
        # Execute the full six-input/three-input physical graph on the fixture's
        # synthetic elemental/propagator arrays; no external data or job.
        from lqcdweave.contraction.planning.api import plan_matrix_actions, CostProfile
        from lqcdweave.contraction.execution.api import replay_scene_storage
        inputs = expanded.inputs
        plan = plan_matrix_actions(inputs.problem, inputs.matrix_kernels, inputs.blocks, inputs.preparations,
            profile=CostProfile(), memory_limit=1 << 30, reserved_host_bytes=inputs.assembly_host_bytes)
        values = replay_scene_storage(plan, inputs.loader, io_overlap=False)
        actual = _run_host_generated(expanded.output_recipe, expanded.problem, values, indirect=True)
        expected = bind_output_assembly(expanded.output_recipe)(values)
        _compare_tree(actual, expected)
        for path, _ in description.fields:
            a, b = actual, expected
            for key in path:
                a, b = a[key], b[key]
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
        # Check the requested connected/disconnected identity independently of
        # field metadata and include both one-point leaves in the comparison.
        np.testing.assert_allclose(actual['I0']['two_point'],
            actual['I0']['connected'] + actual['I0']['disconnected'], rtol=3e-13, atol=3e-13)
        backend = _RecordingBackend()
        bound = program.bind_device_output_assembly(expanded.output_recipe, expanded.problem, backend)
        # Six independently owned output fields share four distinct code bodies.
        self.assertEqual(bound.stage_count, 6)
        self.assertEqual(len(backend.kernels), 4)
        self.assertEqual(backend.allocations, [])

    def test_segmented_scalar_probe_kind_and_wide_uniform_array(self):
        from unittest.mock import patch
        scalar = np.array(1.25 + .3j, dtype='complex64')
        small_sum = OutputNode('diagram_sum', tuple(('t0', (-1) ** i * (1 + .2j)) for i in range(21)))
        children = tuple(OutputNode('power', 2, (small_sum,)) for _ in range(8))
        recipe, problem = _fixture(OutputNode('array', children=children, shape=(2, 4)), scalar)
        with patch.object(program, '_MAX_STAGE_NODES', 24):
            report = program.describe_device_output_program(recipe, problem)
            self.assertGreater(report['stage_count'], 1)
            self.assertLessEqual(max(report['stage_node_counts']), 24)
            actual = _run_host_generated(recipe, problem, (scalar,), indirect=True)
        expected = bind_output_assembly(recipe)((scalar,))
        self.assertEqual(actual.dtype, np.dtype('complex128'))
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_empty_scatter_zero_and_literal_arrays(self):
        value = np.empty((0, 2), dtype='complex128')
        literal = np.arange(6, dtype='float64').reshape(2, 3)
        root = OutputNode('list', children=(
            OutputNode('scatter', ((),), (_term(0),), shape=(3, 2)),
            OutputNode('constant_array', literal, shape=literal.shape),
            OutputNode('zeros', 'complex64', shape=())))
        recipe, problem = _fixture(root, value)
        actual = _run_host_generated(recipe, problem, (value,))
        _compare_tree(actual, bind_output_assembly(recipe)((value,)))


try:
    import cupy as cp
    _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp, _GPU_AVAILABLE = None, False


@unittest.skipUnless(_GPU_AVAILABLE, 'real CuPy CUDA device is unavailable')
class RealCuPyOutputTests(unittest.TestCase):
    def test_native_layout_snapshot_reuse_matches_cpu_and_preserves_tuple_order(self):
        left = np.array([1.25 - 0.5j, -2 + 3j, 0.75 + 1.5j], dtype='complex128')
        right = np.array([2 + 0.25j, -1.5j, 4 - 2j], dtype='complex128')
        recipe, problem = _fixture(OutputNode('add', children=(_term(0, .5j), _term(1, -2))),
                                   left, right)
        device_inputs = dict(zip(recipe.term_ids,
                                 (cp.asarray(left), cp.asarray(right)), strict=True))
        layout = bind_output_layout(recipe.term_ids, problem.specs, backend=cp, validate=False)
        packed = layout.copy(device_inputs, cp)
        self.assertIs(type(packed), _BoundLayoutPackedOutputs)
        self.assertIsInstance(packed, tuple)
        self.assertIs(packed.provenance, layout)
        self.assertEqual(packed.positions, ((0, 0), (0, 1)))
        for actual, expected in zip(packed, (left, right), strict=True):
            np.testing.assert_array_equal(cp.asnumpy(actual), expected)

        bound = program.bind_device_output_assembly(recipe, problem, cp)
        reference = bind_output_assembly(recipe)
        first = bound(packed)
        second = bound(packed)
        expected = reference((left, right))
        np.testing.assert_allclose(cp.asnumpy(first), expected, rtol=2e-13, atol=1e-13)
        np.testing.assert_allclose(cp.asnumpy(second), expected, rtol=2e-13, atol=1e-13)
        self.assertIsInstance(first, cp.ndarray)
        self.assertIsInstance(second, cp.ndarray)

    def test_hex_coefficients_reuse_compiled_kernel_with_fresh_values(self):
        from unittest.mock import patch
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            with self.subTest(dtype=dtype):
                host = np.array([1.25, -2, .5], dtype=dtype)
                coefficient = .7 + .3j if np.dtype(dtype).kind == 'c' else .7
                recipe, problem = _fixture(_term(0, coefficient), host)
                first = program.bind_device_output_assembly(recipe, problem, cp)
                for delta in (0, 1):
                    fresh_recipe, fresh_problem = _fixture(_term(0, coefficient), host + delta)
                    again = program.bind_device_output_assembly(fresh_recipe, fresh_problem, cp)
                    self.assertIs(first.programs[0][0], again.programs[0][0])
                    result = again((cp.asarray(host + delta),))
                    _compare_tree(cp.asnumpy(result), bind_output_assembly(recipe)((host + delta,)))
                # Once compiled, identical signatures must not call NVRTC again.
                with patch.object(cp.cuda.compiler, 'compile_using_nvrtc',
                                  side_effect=AssertionError('unexpected repeated compilation')):
                    _compare_tree(cp.asnumpy(first((cp.asarray(host),))),
                                  bind_output_assembly(recipe)((host,)))

    def test_strided_terms_repeated_points_and_nested_matrix_match_cpu(self):
        left = (np.arange(24).reshape(4, 6)[::2, ::2] * (1 + 0.2j)).astype('complex64')
        right = np.array([2 + 3j], dtype='complex64')
        root = OutputNode('dict', keys=('selected', 'sum'), children=(
            OutputNode('select', (1,), (OutputNode('transpose', (1, 0), (_term(0, -0.25j),), shape=(3, 2)),)),
            OutputNode('add', children=(_term(0), _term(1, 2)))))
        recipe, problem = _fixture(root, left, right)
        bound = program.bind_device_output_assembly(recipe, problem, cp, debug=True)
        reference = bind_output_assembly(recipe)
        for delta in (0, 1):
            device_left = cp.asarray(np.pad(left + delta, ((0, 0), (0, 3))))[:, ::2]
            # Preserve the numerical left fixture while exercising a strided GPU view.
            device_left[...] = cp.asarray(left + delta)
            device_right = cp.asarray(right + delta)
            actual = bound((device_left, device_right))
            host = {key: cp.asnumpy(value) for key, value in actual.items()}
            _compare_tree(host, reference((left + delta, right + delta)))
            self.assertTrue(all(isinstance(value, cp.ndarray) for value in actual.values()))

    def test_large_signature_indirect_device_addresses_and_negative_strides(self):
        host = tuple((np.arange(12).reshape(3, 4) * (1 + 0.2j) + i)[::-1, ::2] for i in range(100))
        device = tuple(cp.asarray(np.ascontiguousarray(value[::-1, :]))[::-1, :] for value in host)
        recipe, problem = _fixture(OutputNode('diagram_sum', tuple((f't{i}', 0.5j) for i in range(100))), *host)
        bound = program.bind_device_output_assembly(recipe, problem, cp)
        actual = bound(device)
        np.testing.assert_allclose(cp.asnumpy(actual), bind_output_assembly(recipe)(host), rtol=2e-13, atol=1e-13)
        self.assertEqual(bound.metadata_bytes, 2400)
        self.assertEqual(len(bound.call_owners), 2)

    def test_segmented_indirect_stages_reuse_device_scratch_in_order(self):
        from unittest.mock import patch
        count = 400
        host = tuple(np.array([i + .25j, -i + .7j], dtype='complex128') for i in range(count))
        child = OutputNode('diagram_sum', tuple((f't{i}', (-1) ** i * (.5 + .3j)) for i in range(count)))
        root = OutputNode('dict', keys=('sum', 'selected'), children=(child, OutputNode('select', (0,), (child,))))
        recipe, problem = _fixture(root, *host)
        device = tuple(cp.asarray(value) for value in host)
        with patch.object(program, '_MAX_STAGE_NODES', 512):
            bound = program.bind_device_output_assembly(recipe, problem, cp)
            actual = bound(device)
        _compare_tree({key: cp.asnumpy(value) for key, value in actual.items()},
                      bind_output_assembly(recipe)(host))
        self.assertGreater(bound.stage_count, 2)
        self.assertEqual(bound.last_stage_launches, bound.stage_count)
        self.assertEqual(bound.scratch_bytes, 64)
        self.assertGreater(bound.metadata_bytes, 0)
        self.assertEqual(len(bound.call_owners), 4)  # Two scratch owners and one host/device metadata pair.

    def test_outer_product_scatter_and_powers_use_real_device_kernel(self):
        left, right = np.array([1 + 2j, 3 - 1j]), np.array([2 - 1j, 4 + 2j, 5])
        l = OutputNode('reshape', children=(_term(0),), shape=(2, 1))
        r = OutputNode('reshape', children=(_term(1),), shape=(1, 3))
        child = OutputNode('power', 3, (OutputNode('multiply', children=(l, r)),))
        root = OutputNode('scatter', ((3, 0), (4, 2, 0)), (child,), shape=(4, 5))
        recipe, problem = _fixture(root, left, right)
        actual = program.bind_device_output_assembly(recipe, problem, cp)((cp.asarray(left), cp.asarray(right)))
        np.testing.assert_allclose(cp.asnumpy(actual), bind_output_assembly(recipe)((left, right)), rtol=2e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
