"""Pointer-only frozen preparation ABI: CPU protocol checks and optional GPU oracles."""
import gc
import importlib
import re
from itertools import product
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.execution.preparation_kernel import (
    FusedSpinPreparation, _MAX_ARGUMENT_BYTES, _MAX_VARIANTS, _SpinLeaf,
    _raw_spin_kernel,
)


def _spin_preparation(spin, *, block="raw", conjugate=False, permutation=(), profile=()):
    recipe = TensorPreparation("spin_spatial", (block,),
        tuple(ArrayConstant.from_array(value) for value in (spin, *profile)),
        conjugate=conjugate, permutation=permutation,
        configuration="cfg" if profile else None)
    return SourcePreparation((block,), recipe=recipe)


def _outer_oracle(spin, raw, *, conjugate=False, permutation=()):
    value = raw.conj() if conjugate else raw
    value = value.transpose(permutation) if permutation else value
    dtype = np.result_type(spin.dtype, value.dtype)
    output = np.empty((4, 4, *value.shape), dtype=dtype)
    spin, value = spin.astype(dtype), value.astype(dtype)
    for a, b in product(range(4), repeat=2):
        for index in np.ndindex(value.shape):
            output[(a, b, *index)] = spin[a, b] * value[index]
    return output


def _profile_oracle(spin, raw, left, right, weights, *, conjugate=False, permutation=()):
    """Independent scalar reference with the established strong-dtype stages."""
    d1 = np.result_type(weights.dtype, left.dtype)
    d2 = np.result_type(d1, raw.dtype)
    d3 = np.result_type(d2, right.dtype)
    projected = np.empty(raw.shape, np.complex128)
    for row, col in np.ndindex(raw.shape):
        total = np.complex128(0)
        for p in range(len(weights)):
            first = d1.type(d1.type(weights[p]) * d1.type(left[p, row].conjugate()))
            second = d2.type(d2.type(first) * d2.type(raw[row, col]))
            third = d3.type(d3.type(second) * d3.type(right[p, col]))
            total = np.complex128(total + np.complex128(third))
        projected[row, col] = total
    return _outer_oracle(spin, projected, conjugate=conjugate, permutation=permutation)


def _many_leaf_program(backend, count, *, prefer_raw):
    spin = np.eye(4, dtype=np.float32)
    recipe = _spin_preparation(spin).recipe
    leaf_spec = TensorSpec((4, 4, 2, 3), np.float32)
    leaves = tuple(_SpinLeaf(recipe, (spin,), leaf_spec) for _ in range(count))
    spec = TensorSpec((count, *leaf_spec.shape), np.float32)
    return FusedSpinPreparation(backend, leaves, spec, stacked=True,
                                prefer_raw=prefer_raw)


class _RawRecorder:
    """Records ABI metadata only, so tests do not create false array retention."""
    def __init__(self):
        self.launches = []

    def __call__(self, grid, block, arguments):
        self.launches.append((grid, block, len(arguments),
                              tuple(type(value).__name__ for value in arguments)))


def _protocol_backend(*, raw=True):
    backend = SimpleNamespace(__name__="cupy", result_type=np.result_type,
        asarray=Mock(side_effect=lambda value: np.array(value, copy=True)),
        empty=Mock(wraps=np.empty), multiply=Mock(wraps=np.multiply),
        zeros_like=Mock(wraps=np.zeros_like), synchronize=Mock())
    backend.ElementwiseKernel = Mock(return_value=Mock())
    sources, kernels = [], []
    if raw:
        def factory(source, name, **kwargs):
            sources.append(source)
            kernel = _RawRecorder()
            kernels.append(kernel)
            return kernel
        backend.RawKernel = factory
    return backend, sources, kernels


class RawPreparationProtocolTests(unittest.TestCase):
    def tearDown(self):
        _raw_spin_kernel.cache_clear()

    def test_raw_is_limited_to_frozen_supported_programs_and_uses_pointer_abi(self):
        spin = np.eye(4, dtype=np.float32)
        spec = TensorSpec((4, 4, 2, 3), np.float32)
        prep = _spin_preparation(spin)

        mutable_backend, _, mutable_kernels = _protocol_backend()
        mutable = bind_prepared_loader(None, {"x": prep}, {"x": spec}, backend=mutable_backend)
        mutable.materialize("x", None, {"raw": np.ones((2, 3), np.float32)})
        self.assertEqual(mutable_kernels, [])
        mutable_backend.ElementwiseKernel.assert_called_once()

        missing_backend, _, missing_kernels = _protocol_backend(raw=False)
        frozen_missing = bind_prepared_loader(None, MappingProxyType({"x": prep}),
            MappingProxyType({"x": spec}), backend=missing_backend)
        frozen_missing.materialize("x", None, {"raw": np.ones((2, 3), np.float32)})
        self.assertEqual(missing_kernels, [])
        missing_backend.ElementwiseKernel.assert_called_once()

        backend, sources, kernels = _protocol_backend()
        loader = bind_prepared_loader(None, MappingProxyType({"x": prep}),
            MappingProxyType({"x": spec}), backend=backend)
        loader._bind_fused_structure("x", set())
        backend.empty.reset_mock()
        backend.asarray.reset_mock()
        raw = np.arange(6, dtype=np.float32).reshape(2, 3)
        first = loader.materialize("x", None, {"raw": raw})
        raw[:] = 4
        second = loader.materialize("x", None, {"raw": raw})
        self.assertEqual(len(kernels), 1)
        self.assertEqual(len(kernels[0].launches), 2)
        self.assertEqual(backend.empty.call_count, 2)
        backend.asarray.assert_not_called()
        backend.ElementwiseKernel.assert_not_called()
        backend.synchronize.assert_not_called()
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertIn("#include <cupy/complex.cuh>", source)
        self.assertIn("volatile T re", source)
        self.assertIn("ed_view_x0", source)
        self.assertIn("byte_offset", source)
        declarations, function = source.split(
            'extern "C" __global__ void ed_prepare_spin_raw(', 1)
        self.assertNotIn("ed_arg_", declarations)
        signature, body = function.split(") {", 1)
        self.assertIn("ed_arg_x0", signature)
        self.assertIn("ed_arg_s0", signature)
        self.assertIn("ed_arg_output", signature)
        for name in ("x0", "s0", "output"):
            self.assertIn(f"ed_view_{name} {name}{{", body)
            self.assertIn(f"ed_arg_{name}", body)
        self.assertEqual(len(signature.split(",")), 5)
        self.assertNotIn("CArray", source)
        self.assertNotIn("shape[", source)
        self.assertEqual(len(loader._fused_bindings["x"][0]._variants), 1)
        # The fake launch records no values; generated programs retain no input blocks.
        input_ref = weakref.ref(raw)
        del raw, first, second
        gc.collect()
        self.assertIsNone(input_ref())

    def test_identical_leaf_arithmetic_and_indexers_are_shared_in_raw_source(self):
        backend, sources, _ = _protocol_backend()
        program = _many_leaf_program(backend, 32, prefer_raw=True)
        inputs = tuple(np.full((2, 3), lane + 1, np.float32) for lane in range(32))
        program.materialize({}, inputs=inputs)
        source = sources[0]
        # One original arithmetic body, retaining every independent input slot.
        self.assertEqual(source.count('float prepared ='), 1)
        self.assertEqual(source.count('struct ed_view_'), 3)  # raw, spin, output
        self.assertIn('using ed_view_x31 = ed_view_x0;', source)
        self.assertIn('ed_selected_x0', source)
        self.assertIn('x31.pointer', source)
        self.assertIn('(offset + lane)', source)

    def test_shared_raw_arithmetic_separates_conjugation_cast_and_layout(self):
        backend, sources, _ = _protocol_backend()
        spin = np.eye(4, dtype=np.complex64)
        base = _spin_preparation(spin).recipe
        conjugated = _spin_preparation(spin, conjugate=True).recipe
        leaf_spec = TensorSpec((4, 4, 2, 3), np.complex64)
        leaves = tuple(_SpinLeaf(recipe, (spin,), leaf_spec, casts=casts)
                       for recipe, casts in ((base, ()), (conjugated, ()),
                                             (base, (np.dtype('complex128'),)), (base, ())))
        program = FusedSpinPreparation(backend, leaves,
            TensorSpec((4, *leaf_spec.shape), np.complex64), stacked=True, prefer_raw=True)
        a = np.ones((2, 3), np.float32)
        program.materialize({}, inputs=(a, a, a, a))
        self.assertEqual(sources[0].count('complex<float> prepared ='), 3)
        self.assertIn('lane == 0 || lane == 3', sources[0])
        # Same numerical shape with a negative stride is a different program.
        program.materialize({}, inputs=(a, a, a, a[:, ::-1]))
        self.assertEqual(sources[1].count('complex<float> prepared ='), 4)
        self.assertNotIn('ed_selected_', sources[1])

    def test_shared_raw_arithmetic_keeps_distinct_compressed_spin_permutations(self):
        backend, sources, _ = _protocol_backend()
        columns = ((0, 1, 2, 3), (3, 2, 1, 0))
        spins = tuple(np.eye(4, dtype=np.complex64)[:, order] for order in columns)
        recipes = tuple(TensorPreparation('spin_spatial', ('raw',),
            (ArrayConstant.from_array(spin),), spin_columns=order)
            for spin, order in zip(spins, columns, strict=True))
        spec = TensorSpec((4, 2, 3), np.complex64)
        leaves = tuple(_SpinLeaf(recipes[i], (spins[i],), spec) for i in (0, 1, 0, 1))
        program = FusedSpinPreparation(backend, leaves,
            TensorSpec((4, *spec.shape), np.complex64), stacked=True, prefer_raw=True)
        value = np.ones((2, 3), np.complex64)
        program.materialize({}, inputs=(value,) * 4)
        source = sources[0]
        self.assertEqual(source.count('complex<float> prepared ='), 2)
        self.assertIn('lane == 0 || lane == 2', source)
        self.assertIn('lane == 1 || lane == 3', source)
        self.assertIn('ed_spin_index[4] = {0, 5, 10, 15}', source)
        self.assertIn('ed_spin_index[4] = {3, 6, 9, 12}', source)
        self.assertNotIn('s < 16', source)

    def test_stride_variants_are_bounded_and_do_not_capture_input_arrays(self):
        backend, _, kernels = _protocol_backend()
        spin = np.eye(4, dtype=np.float64)
        loader = bind_prepared_loader(None, MappingProxyType({"x": _spin_preparation(spin)}),
            MappingProxyType({"x": TensorSpec((4, 4, 2, 3), np.float64)}), backend=backend)
        program = loader._bind_fused_structure("x", set())[0]
        references = []
        for step in range(_MAX_VARIANTS + 5):
            base = np.ones((2, 3 * (step + 1)), dtype=np.float64)
            raw = base[:, ::step + 1]
            references.append(weakref.ref(raw))
            loader.materialize("x", None, {"raw": raw})
            del raw, base
        gc.collect()
        self.assertLessEqual(len(program._variants), _MAX_VARIANTS)
        self.assertLessEqual(_raw_spin_kernel.cache_info().currsize, 64)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertGreater(len(kernels), _MAX_VARIANTS)

    def test_warm_call_cache_skips_leaf_metadata_rebuild_for_fresh_inputs(self):
        for prefer_raw in (False, True):
            with self.subTest(prefer_raw=prefer_raw):
                backend, _, _ = _protocol_backend()
                program = _many_leaf_program(backend, 17, prefer_raw=prefer_raw)
                inputs = tuple(np.ones((2, 3), dtype=np.float32) for _ in range(17))
                first = program.materialize({}, inputs=inputs)
                # The kernels remain fixed, but each invocation resolves fresh
                # inputs and allocates its own result. Rebuilding a leaf would
                # repeat dtype promotion here.
                fresh = tuple(np.full((2, 3), 4, dtype=np.float32) for _ in range(17))
                with patch('lqcdweave.contraction.execution.preparation_kernel.np.result_type',
                           side_effect=AssertionError('leaf metadata rebuilt')):
                    second = program.materialize({}, inputs=fresh)
                self.assertIsNot(first, second)
                self.assertEqual(len(program._calls), 1)
                with execution_debug(True), patch(
                        'lqcdweave.contraction.execution.preparation_kernel.np.result_type',
                        wraps=np.result_type) as promote:
                    program.materialize({}, inputs=fresh)
                self.assertEqual(promote.call_count, 17)

    def test_repeated_current_inputs_read_each_layout_once_and_keep_lane_order(self):
        class CountingInput:
            def __init__(self, value):
                self.value = value
                self.reads = {'dtype': 0, 'shape': 0, 'strides': 0}

            def __getattr__(self, name):
                if name in self.reads:
                    self.reads[name] += 1
                return getattr(self.value, name)

        backend, _, _ = _protocol_backend()
        program = _many_leaf_program(backend, 32, prefer_raw=True)
        a = CountingInput(np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2])
        b = CountingInput(np.ones((2, 3), np.float32)[:, ::-1])
        inputs = (a, a, b, b) * 8
        program.materialize({}, inputs=inputs)
        for value in (a, b):
            value.reads = dict.fromkeys(value.reads, 0)
        program.materialize({}, inputs=inputs)
        self.assertEqual(a.reads, {'dtype': 1, 'shape': 1, 'strides': 1})
        self.assertEqual(b.reads, a.reads)
        signature = next(iter(program._calls))[0]
        expected = tuple((np.dtype(np.float32), value.value.shape, value.value.strides) for value in inputs)
        self.assertEqual(signature, expected)
        # Debug continues to read every occurrence, even on the warm path.
        for value in (a, b):
            value.reads = dict.fromkeys(value.reads, 0)
        with execution_debug(True):
            program.materialize({}, inputs=inputs)
        self.assertGreaterEqual(a.reads['strides'], 16)
        self.assertGreaterEqual(b.reads['strides'], 16)

    def test_repeated_layouts_refresh_strides_between_calls_and_match_scalar_oracle(self):
        backend, _, _ = _protocol_backend()
        delivered = []

        def factory(source, name, **kwargs):
            signature, body = source.split('ed_prepare_spin_raw(', 1)[1].split(') {', 1)
            names = tuple(parameter.rsplit(' ', 1)[-1] for parameter in signature.split(', '))
            aliases = re.findall(r'ed_view_x(\d+) x\d+\{reinterpret_cast<const char\*>\((ed_arg_x\d+)\)\};', body)
            def launch(grid, block, arguments):
                spatial, start, result = arguments[-3:]
                values = dict(zip(names, arguments, strict=True))
                spin = values['ed_arg_s0']
                current = tuple(values[alias] for _, alias in aliases)
                delivered.append(tuple(id(value) for value in current))
                for lane, value in enumerate(current):
                    result[int(start) + lane] = _outer_oracle(spin, value)
            return launch

        backend.RawKernel = factory
        program = _many_leaf_program(backend, 32, prefer_raw=True)
        a = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
        b = np.arange(6, dtype=np.float32).reshape(2, 3)[:, ::-1]
        references = [weakref.ref(a), weakref.ref(b)]
        inputs = (a, a, b, b) * 8
        for stride_change in (False, True):
            with self.subTest(stride_change=stride_change):
                if stride_change:
                    a.strides = (12, 4)
                actual = program.materialize({}, inputs=inputs)
                expected = np.stack([_outer_oracle(np.eye(4, dtype=np.float32), value) for value in inputs])
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(delivered[-1], tuple(id(value) for value in inputs))
                with execution_debug(True):
                    audited = program.materialize({}, inputs=inputs)
                np.testing.assert_array_equal(audited, expected)
        self.assertEqual(len(program._calls), 4)  # two layouts, compact and debug ABIs
        unsupported = np.ones((2, 3), np.float16)
        self.assertIsNone(program.materialize({}, inputs=(unsupported,) * 32))
        self.assertEqual(len(program._calls), 4)
        del inputs, a, b, actual, audited, expected, unsupported
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_repeated_raw_input_slots_follow_current_alias_pattern_without_retaining_owners(self):
        for count in (32, 33):
            with self.subTest(lanes=count):
                backend, sources, kernels = _protocol_backend()
                program = _many_leaf_program(backend, count, prefer_raw=True)
                a, b = np.ones((2, 3), np.float32), np.full((2, 3), 2, np.float32)
                refs = [weakref.ref(a), weakref.ref(b)]
                for pattern in ((a, a, b, b), (a, a, b, a)):
                    inputs = (pattern * ((count + 3) // 4))[:count]
                    program.materialize({}, inputs=inputs)
                self.assertEqual(len(program._calls), 2)
                self.assertTrue(all(key[3] is not None for key in program._calls))
                # Each full chunk now passes two current inputs, one spin,
                # two scalars and output, while keeping all 32 leaf proxies.
                full_chunks = [source for source in sources if 'if (i >= 192LL)' in source]
                self.assertEqual(len(full_chunks), 2)
                for source in full_chunks:
                    signature, body = source.split('ed_prepare_spin_raw(', 1)[1].split(') {', 1)
                    self.assertEqual(len(re.findall(r'ed_arg_x\d+', signature)), 2)
                    self.assertEqual(len(re.findall(r'ed_view_x\d+ x\d+\{', body)), 32)
                self.assertTrue(all(entry[2] in (5, 6) for kernel in kernels for entry in kernel.launches))
                # Equal layouts and values cannot make distinct owners alias.
                inputs = tuple(np.full((2, 3), 7, np.float32) for _ in range(count))
                program.materialize({}, inputs=inputs)
                self.assertEqual(len(program._calls), 3)
                self.assertTrue(any(key[3] is None for key in program._calls))
                self.assertTrue(any(entry[2] == 36 for kernel in kernels for entry in kernel.launches))
                del inputs, pattern, a, b
                gc.collect()
                self.assertTrue(all(reference() is None for reference in refs))

    def test_raw_warm_call_passes_original_pointer_without_transpose_view(self):
        class NoTranspose(np.ndarray):
            def transpose(self, *args, **kwargs):
                raise AssertionError('warm raw path created a transpose view')

        backend, _, kernels = _protocol_backend()
        spin = np.eye(4, dtype=np.float32)
        recipe = _spin_preparation(spin, permutation=(1, 0)).recipe
        spec = TensorSpec((4, 4, 3, 2), np.float32)
        program = FusedSpinPreparation.spin(backend, recipe, (spin,), spec, prefer_raw=True)
        program.materialize({}, inputs=(np.ones((2, 3), dtype=np.float32),))
        fresh = np.full((2, 3), 7, dtype=np.float32).view(NoTranspose)
        program.materialize({}, inputs=(fresh,))
        self.assertEqual(len(kernels), 1)
        self.assertEqual(len(kernels[0].launches), 2)
        self.assertEqual(kernels[0].launches[-1][-1][0], 'NoTranspose')

    def test_private_call_cache_separates_dtype_and_stride_and_rejects_unsupported_inputs(self):
        backend, _, _ = _protocol_backend()
        program = _many_leaf_program(backend, 4, prefer_raw=True)
        for dtype in (np.float32, np.float64):
            for reverse in (False, True):
                value = np.empty((2, 3), dtype=dtype)
                if reverse:
                    value = value[:, ::-1]
                self.assertIsNotNone(program.materialize({}, inputs=(value,) * 4))
        self.assertEqual(len(program._calls), 4)
        for key in program._calls:
            self.assertTrue(all(item[0].metadata is None for item in key[0]))
        for dtype in (np.float16, np.int32, np.dtype(np.float32).newbyteorder('S')):
            value = np.empty((2, 3), dtype=dtype)
            self.assertIsNone(program.materialize({}, inputs=(value,) * 4))
        self.assertEqual(len(program._calls), 4)

    def test_input_dtype_metadata_does_not_keep_numeric_owners_in_private_cache(self):
        for count in (4, 32):
            with self.subTest(count=count):
                backend, _, _ = _protocol_backend()
                program = _many_leaf_program(backend, count, prefer_raw=True)
                owner = np.ones((2, 3))
                owner_ref = weakref.ref(owner)
                dtype = np.dtype(np.float32, metadata={'owner': owner})
                value = np.empty((2, 3), dtype=dtype)
                program.materialize({}, inputs=(value,) * count)
                del value, dtype, owner
                gc.collect()
                self.assertIsNone(owner_ref())

    def test_bound_raw_arguments_use_fresh_inputs_output_and_chunk_offsets(self):
        backend, _, _ = _protocol_backend()
        launches = []
        def factory(*args, **kwargs):
            def launch(grid, block, arguments):
                spatial, start, result = arguments[-3:]
                # All leaves share one spin owner: its one argument follows x0.
                count = len(arguments) - 4
                launches.append((grid, block, int(start), count))
                spin = arguments[1]
                for lane, raw in enumerate((arguments[0], *arguments[2:-3])):
                    result[int(start)+lane] = _outer_oracle(spin, raw)
            return launch
        backend.RawKernel = factory
        program = _many_leaf_program(backend, 65, prefer_raw=True)
        outputs = []
        references = []
        for scale in (1, 7):
            inputs = tuple(np.full((2, 3), scale*(i+1), np.float32) for i in range(65))
            references.extend(weakref.ref(value) for value in inputs)
            result = program.materialize({}, inputs=inputs)
            expected = np.stack([_outer_oracle(leaf.constants[0], value)
                                 for leaf, value in zip(program.leaves, inputs)])
            np.testing.assert_array_equal(result, expected)
            outputs.append(result)
            del inputs, expected
        self.assertIsNot(outputs[0], outputs[1])
        self.assertEqual([entry[2:] for entry in launches], [(0, 32), (32, 32), (64, 1)]*2)
        self.assertEqual(len(program._calls), 1)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_raw_constant_aliases_use_identity_and_keep_each_leaf_proxy(self):
        backend, sources, kernels = _protocol_backend()
        spin = np.eye(4, dtype=np.float32)
        equal_spin = spin.copy()
        recipe = _spin_preparation(spin).recipe
        leaf_spec = TensorSpec((4, 4, 2, 3), np.float32)
        leaves = tuple(_SpinLeaf(recipe, (constant,), leaf_spec)
                       for constant in (spin, spin, equal_spin))
        program = FusedSpinPreparation(backend, leaves,
            TensorSpec((3, 4, 4, 2, 3), np.float32), stacked=True, prefer_raw=True)
        for scale in (1, 7):
            inputs = tuple(np.full((2, 3), scale*(lane+1), np.float32)
                           for lane in range(3))
            program.materialize({}, inputs=inputs)
        # Three independent inputs, two constant owners, two scalars and output.
        self.assertEqual([entry[2] for entry in kernels[0].launches], [8, 8])
        signature, body = sources[0].split('ed_prepare_spin_raw(', 1)[1].split(') {', 1)
        self.assertIn('ed_arg_s0', signature)
        self.assertNotIn('ed_arg_s1', signature)
        self.assertIn('ed_arg_s2', signature)
        self.assertIn('ed_view_s1 s1{reinterpret_cast<const char*>(ed_arg_s0)};', body)
        self.assertIn('ed_view_s2 s2{reinterpret_cast<const char*>(ed_arg_s2)};', body)

    def test_raw_profile_constants_can_alias_across_leaves_and_roles(self):
        backend, sources, kernels = _protocol_backend()
        spin = np.eye(4, dtype=np.float32)
        profile = np.ones((2, 3), dtype=np.float64)
        weights = np.ones(2, dtype=np.float32)
        recipe = _spin_preparation(spin, profile=(profile, profile, weights)).recipe
        leaf_spec = TensorSpec((4, 4, 3, 3), np.complex128)
        leaf = _SpinLeaf(recipe, (spin, profile, profile, weights), leaf_spec)
        program = FusedSpinPreparation(backend, (leaf, leaf),
            TensorSpec((2, 4, 4, 3, 3), np.complex128), stacked=True, prefer_raw=True)
        program.materialize({}, inputs=(np.ones((3, 3)), np.full((3, 3), 7.0)))
        self.assertEqual(kernels[0].launches[0][2], 8)
        signature, body = sources[0].split('ed_prepare_spin_raw(', 1)[1].split(') {', 1)
        self.assertIn('ed_arg_l0', signature)
        self.assertNotIn('ed_arg_r0', signature)
        self.assertNotIn('ed_arg_l1', signature)
        self.assertIn('ed_view_r0 r0{reinterpret_cast<const char*>(ed_arg_l0)};', body)
        self.assertIn('ed_view_l1 l1{reinterpret_cast<const char*>(ed_arg_l0)};', body)

    def test_raw_warm_noncontiguous_allocation_rebinds_legacy_without_extra_owner(self):
        backend, _, kernels = _protocol_backend()
        program = _many_leaf_program(backend, 17, prefer_raw=True)
        inputs = tuple(np.ones((2, 3), dtype=np.float32) for _ in range(17))
        program.materialize({}, inputs=inputs)
        destination = np.empty(program.spec.shape, dtype=np.float32, order='F')
        backend.empty.reset_mock()
        backend.empty.side_effect = lambda *args, **kwargs: destination
        result = program.materialize({}, inputs=inputs)
        self.assertIs(result, destination)
        backend.empty.assert_called_once()
        self.assertEqual(sum(len(kernel.launches) for kernel in kernels), 1)
        calls = backend.ElementwiseKernel.return_value.call_args_list
        self.assertEqual([(len(call.args) - 3) // 2 for call in calls], [8, 8, 1])

    def test_raw_thirty_two_lane_and_legacy_eight_lane_boundaries(self):
        for count in (9, 17, 31, 32, 33, 65):
            with self.subTest(count=count):
                backend, _, kernels = _protocol_backend()
                program = _many_leaf_program(backend, count, prefer_raw=True)
                inputs = tuple(np.full((2, 3), i + 1, dtype=np.float32)
                               for i in range(count))
                result = program.materialize({}, inputs=inputs)
                chunks = [len(arguments) - 4
                          for kernel in kernels for _, _, _, arguments
                          in kernel.launches]
                self.assertEqual(chunks, [min(32, count - offset)
                                          for offset in range(0, count, 32)])
                self.assertEqual(sum(len(kernel.launches) for kernel in kernels),
                                 (count + 31) // 32)
                self.assertTrue(all(argument_count * 8 < _MAX_ARGUMENT_BYTES
                    for kernel in kernels
                    for _, _, argument_count, _ in kernel.launches))
                self.assertEqual(backend.empty.call_count, 1)
                backend.asarray.assert_not_called()
                backend.synchronize.assert_not_called()
                backend.ElementwiseKernel.assert_not_called()
                refs = [weakref.ref(value) for value in inputs]
                del inputs, result
                gc.collect()
                self.assertTrue(all(reference() is None for reference in refs))

                legacy_backend, _, legacy_kernels = _protocol_backend(raw=False)
                legacy = _many_leaf_program(legacy_backend, count, prefer_raw=False)
                legacy_inputs = tuple(np.full((2, 3), i + 1, dtype=np.float32)
                                      for i in range(count))
                legacy.materialize({}, inputs=legacy_inputs)
                calls = legacy_backend.ElementwiseKernel.return_value.call_args_list
                sizes = [(len(call.args) - 3) // 2 for call in calls]
                self.assertEqual(sizes, [min(8, count - offset)
                                         for offset in range(0, count, 8)])
                self.assertEqual(len(legacy_kernels), 0)
                legacy_backend.ElementwiseKernel.assert_called()
                self.assertTrue(all(64 + 16 * 5 + size * 2 * (16 + 16 * 2)
                                    < _MAX_ARGUMENT_BYTES for size in sizes))

        # A raw-capable binder must return to the same legacy limit when the
        # caller supplies a non-C destination.
        backend, _, kernels = _protocol_backend()
        program = _many_leaf_program(backend, 17, prefer_raw=True)
        inputs = tuple(np.ones((2, 3), dtype=np.float32) for _ in range(17))
        target = np.empty(program.spec.shape, dtype=np.float32, order="F")
        program.materialize({}, target, inputs=inputs)
        self.assertEqual(kernels, [])
        calls = backend.ElementwiseKernel.return_value.call_args_list
        self.assertEqual([(len(call.args) - 3) // 2 for call in calls], [8, 8, 1])

    def test_noncontiguous_fallback_retains_legacy_argument_byte_guard(self):
        backend, _, kernels = _protocol_backend()
        spin = np.eye(4, dtype=np.float32)
        # This protocol test exercises the parameter guard without requiring
        # NumPy/CuPy to allocate a rank above their supported maximum.
        raw = SimpleNamespace(shape=(2,) + (1,) * 61, dtype=np.dtype(np.float32),
            ndim=62, strides=(4,) * 62, flags=SimpleNamespace(c_contiguous=True))
        recipe = _spin_preparation(spin).recipe
        leaf_spec = TensorSpec((4, 4, *raw.shape), np.float32)
        leaves = tuple(_SpinLeaf(recipe, (spin,), leaf_spec) for _ in range(3))
        spec = TensorSpec((3, *leaf_spec.shape), np.float32)
        program = FusedSpinPreparation(backend, leaves, spec, stacked=True,
                                       prefer_raw=True)
        destination = SimpleNamespace(shape=spec.shape, dtype=np.dtype(np.float32),
            ndim=len(spec.shape), strides=(0,) * len(spec.shape),
            flags=SimpleNamespace(c_contiguous=False))
        result = program.materialize({}, destination,
                                    inputs=(raw, raw, raw))
        self.assertIs(result, destination)
        self.assertEqual(kernels, [])
        calls = backend.ElementwiseKernel.return_value.call_args_list
        self.assertEqual([(len(call.args) - 3) // 2 for call in calls], [1, 1, 1])


class _CuPyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module("cupy")
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no device")
        except Exception as exc:
            raise unittest.SkipTest(f"CuPy device unavailable: {exc}")

    def tearDown(self):
        _raw_spin_kernel.cache_clear()

    def _tracked_raw(self, cp):
        factory = cp.RawKernel
        kernels = []

        class _Tracked:
            def __init__(self, kernel):
                self.kernel = kernel
                self.calls = 0
                self.argument_counts = []

            def __call__(self, grid, block, arguments):
                self.calls += 1
                self.argument_counts.append(len(arguments))
                self.kernel(grid, block, arguments)

        def build(source, name, **kwargs):
            kernel = _Tracked(factory(source, name, **kwargs))
            kernels.append(kernel)
            return kernel
        return build, kernels

    def test_call_cache_switches_current_dtype_and_strides_with_fresh_values(self):
        cp = self.cp
        spin = np.arange(16, dtype=np.float64).reshape(4, 4) / 7 + 0.3j
        recipe = _spin_preparation(spin, conjugate=True, permutation=(1, 0)).recipe
        spec = TensorSpec((4, 4, 3, 2), np.complex128)
        constants = (cp.asarray(spin),)
        raw_program = FusedSpinPreparation.spin(cp, recipe, constants, spec, prefer_raw=True)
        legacy_program = FusedSpinPreparation.spin(cp, recipe, constants, spec, prefer_raw=False)
        for repeat in range(2):
            for dtype in (np.float32, np.float64, np.complex64, np.complex128):
                for reversed_input in (False, True):
                    with self.subTest(repeat=repeat, dtype=dtype, reverse=reversed_input):
                        host = (np.arange(6).reshape(2, 3) + repeat + 1).astype(dtype)
                        if np.dtype(dtype).kind == 'c':
                            host += np.asarray(0.2j, dtype=dtype)
                        value = cp.asarray(host)
                        if reversed_input:
                            host, value = host[:, ::-1], value[:, ::-1]
                        actual = cp.asnumpy(raw_program.materialize({}, inputs=(value,)))
                        legacy = cp.asnumpy(legacy_program.materialize({}, inputs=(value,)))
                        expected = _outer_oracle(spin, host, conjugate=True, permutation=(1, 0))
                        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
                        np.testing.assert_array_equal(actual, legacy)
        self.assertEqual(len(raw_program._calls), 8)
        self.assertTrue(all(key[0][0][0].metadata is None for key in raw_program._calls))
        if (not cp.cuda.runtime.is_hip
                and callable(getattr(cp.cuda.function, 'CNumpyArray', None))):
            self.assertTrue(all(call.native_arguments is not None
                                for _, _, calls in raw_program._calls.values() for call in calls))

    def test_native_parameter_slots_follow_current_stream_and_chunk_offsets(self):
        cp = self.cp
        if cp.cuda.runtime.is_hip or not callable(getattr(cp.cuda.function, 'CNumpyArray', None)):
            self.skipTest('native parameter slots require CUDA and CuPy CNumpyArray')
        rng = np.random.default_rng(9703)
        spin = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(np.complex64)
        left = rng.normal(size=(2, 3)).astype(np.float32)
        right = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(np.complex64)
        weights = np.array([0.7, -0.2], dtype=np.float64)
        recipe = _spin_preparation(spin, conjugate=True, permutation=(1, 0),
                                   profile=(left, right, weights)).recipe
        constants = tuple(cp.asarray(value) for value in (spin, left, right, weights))
        leaf_spec = TensorSpec((4, 4, 3, 3), np.complex128)
        leaves = tuple(_SpinLeaf(recipe, constants, leaf_spec,
                                casts=(np.dtype(np.complex64), np.dtype(np.complex128)))
                       for _ in range(33))
        program = FusedSpinPreparation(cp, leaves,
            TensorSpec((33, *leaf_spec.shape), np.complex128), stacked=True, prefer_raw=True)
        constants_ready = cp.cuda.Event()
        constants_ready.record()
        pending = []
        bound_storage = None
        for repeat in range(2):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.wait_event(constants_ready)
                bases = tuple((rng.normal(size=(3, 3)) + repeat + index / 7).astype(np.float32)
                              for index in range(33))
                inputs = tuple(cp.asarray(value)[::-1, ::-1] for value in bases)
                actual = program.materialize({}, inputs=inputs)
                bindings = next(iter(program._calls.values()))[2]
                self.assertEqual(len(bindings), 2)
                self.assertTrue(all(call.native_arguments is not None for call in bindings))
                storage = tuple(call.native_arguments._local.storage for call in bindings)
                if bound_storage is None:
                    bound_storage = storage
                else:
                    self.assertTrue(all(a is b for a, b in zip(storage, bound_storage)))
                for call, (words, _) in zip(bindings, storage):
                    slots = call.native_arguments
                    if slots._clear_after_launch:
                        self.assertEqual(words[slots.output_position], 0)
                        self.assertTrue(all(words[position] == 0 for position, _ in slots.positions))
                    else:
                        # The native driver consumes pointer values during
                        # launch and keeps reusable integer slots, not owners.
                        self.assertEqual(words[slots.output_position], actual.data.ptr)
                        self.assertTrue(all(words[position] == inputs[index].data.ptr
                                            for position, index in slots.positions))
            expected = np.stack([_profile_oracle(spin, value[::-1, ::-1], left, right,
                weights, conjugate=True, permutation=(1, 0)).astype(np.complex64).astype(np.complex128)
                for value in bases])
            pending.append((stream, inputs, actual, expected))
        # Synchronize only at this test's observation boundary. Two current
        # streams share host slots but each enqueue must capture its own values.
        for stream, inputs, actual, expected in pending:
            stream.synchronize()
            np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=3e-7, atol=3e-7)

    def test_wide_repeated_current_arrays_match_debug_legacy_and_scalar_oracle(self):
        cp = self.cp
        rng = np.random.default_rng(9704)
        spin = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(np.complex64)
        recipe = _spin_preparation(spin, conjugate=True, permutation=(1, 0)).recipe
        alternate = _spin_preparation(spin).recipe
        constants = (cp.asarray(spin),)
        leaf_spec = TensorSpec((4, 4, 3, 3), np.complex64)
        constants_ready = cp.cuda.Event()
        constants_ready.record()
        for count in (32, 33):
            with self.subTest(lanes=count):
                leaves = tuple(_SpinLeaf(recipe if lane % 2 == 0 else alternate, constants,
                                        leaf_spec, casts=(np.dtype(np.complex128),)) for lane in range(count))
                spec = TensorSpec((count, *leaf_spec.shape), np.complex128)
                program = FusedSpinPreparation(cp, leaves, spec, stacked=True, prefer_raw=True)
                legacy = FusedSpinPreparation(cp, leaves, spec, stacked=True, prefer_raw=False)
                pending = []
                for repeat in range(3):
                    hosts = (rng.normal(size=(3, 3)).astype(np.float32),
                             (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))).astype(np.complex64))
                    stream = cp.cuda.Stream(non_blocking=True)
                    with stream:
                        stream.wait_event(constants_ready)
                        a, b = (cp.asarray(value) for value in hosts)
                        if repeat % 2:
                            a = cp.asfortranarray(a)
                            b = b[:, ::-1]
                            host_inputs = (np.asfortranarray(hosts[0]), hosts[1][:, ::-1])
                        else:
                            a = a[::-1, ::-1]
                            b = cp.asfortranarray(b)
                            host_inputs = (hosts[0][::-1, ::-1], np.asfortranarray(hosts[1]))
                        # Preserve repeated lanes with distinct dtype/stride layouts.
                        # The first two entries trigger call-local layout reuse.
                        inputs = ((a, a, b, b) * ((count + 3) // 4))[:count]
                        cold = program.materialize({}, inputs=inputs)
                        warm = program.materialize({}, inputs=inputs)
                        with execution_debug(True):
                            audited = program.materialize({}, inputs=inputs)
                        legacy_result = legacy.materialize({}, inputs=inputs)
                    expected = np.stack([_outer_oracle(spin, host_inputs[0 if lane % 4 < 2 else 1],
                        conjugate=lane % 2 == 0, permutation=(1, 0) if lane % 2 == 0 else ()
                        ).astype(np.complex128) for lane in range(count)])
                    pending.append((stream, inputs, cold, warm, audited, legacy_result, expected))
                self.assertEqual(len(program._calls), 4)  # two layouts plus independent debug slots
                self.assertTrue(all(len(key[0]) == count for key in program._calls))
                # Observation boundaries only: queued streams retain their own
                # current owners while sharing the fixed host argument slots.
                for stream, inputs, cold, warm, audited, legacy_result, expected in pending:
                    stream.synchronize()
                    actual = cp.asnumpy(cold)
                    np.testing.assert_array_equal(cp.asnumpy(warm), actual)
                    np.testing.assert_array_equal(cp.asnumpy(audited), actual)
                    np.testing.assert_array_equal(cp.asnumpy(legacy_result), actual)
                    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)

    def test_frozen_raw_profile_dagger_negative_stride_matches_debug_and_oracle(self):
        cp = self.cp
        rng = np.random.default_rng(9701)
        profiles = (
            (np.complex64, np.float32, np.complex64, np.float64, np.float32),
            (np.complex128, np.complex64, np.float32, np.complex128, np.complex128),
            (np.float32, np.complex128, np.complex64, np.float64, np.complex128),
        )
        for types in profiles:
            shapes = ((4, 4), (3, 2), (3, 3), (3, 2), (3,))
            values = []
            for shape, dtype in zip(shapes, types):
                value = rng.normal(size=shape)
                if np.dtype(dtype).kind == "c":
                    value = value + 1j * rng.normal(size=shape)
                values.append(value.astype(dtype))
            spin, raw, left, right, weights = values
            raw_base = raw
            raw = raw_base[::-1, ::-1]
            axes = (1, 0)
            profile = (left, right, weights)
            prep = _spin_preparation(spin, conjugate=True, permutation=axes, profile=profile)
            expected = _profile_oracle(spin, raw, *profile, conjugate=True, permutation=axes)
            preps = MappingProxyType({"x": prep})
            specs = MappingProxyType({"x": TensorSpec(expected.shape, expected.dtype)})
            loader = bind_prepared_loader(None, preps, specs, backend=cp)
            device_raw = cp.asarray(raw_base)[::-1, ::-1]
            build, kernels = self._tracked_raw(cp)
            with patch.object(cp, "RawKernel", side_effect=build), \
                 patch.object(cp, "asarray", wraps=cp.asarray) as upload, \
                 patch.object(cp, "empty", wraps=cp.empty) as allocate:
                actual = loader.materialize("x", None, {"raw": device_raw})
                upload_count = upload.call_count
                self.assertGreater(upload_count, 0)  # recipe constants upload once
                self.assertEqual(allocate.call_count, 1)  # prepared output only
                with execution_debug(True):
                    debug = loader.materialize("x", None, {"raw": device_raw})
                self.assertEqual(len(kernels), 1)
                self.assertEqual(kernels[0].calls, 1)
                self.assertEqual(upload.call_count, upload_count)
            np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=7e-6, atol=7e-6)
            np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(debug), rtol=7e-6, atol=7e-6)

    def test_fresh_values_nested_cast_scalar_empty_and_zero_profile_rows(self):
        cp = self.cp
        spin = (np.arange(16, dtype=np.float64).reshape(4, 4) / 13).astype(np.float64)
        raw_host = np.array([[1.00000007, -2.5]], dtype=np.float64)
        prep = _spin_preparation(spin)
        preps = MappingProxyType({
            "x": prep,
            "inner": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("x", "x"))),
            "outer": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("inner", "inner"))),
        })
        specs = MappingProxyType({
            "x": TensorSpec((4, 4, 1, 2), np.float64),
            "inner": TensorSpec((2, 4, 4, 1, 2), np.float32),
            "outer": TensorSpec((2, 2, 4, 4, 1, 2), np.complex128),
        })
        loader = bind_prepared_loader(None, preps, specs, backend=cp)
        raw = cp.asarray(raw_host)
        build, kernels = self._tracked_raw(cp)
        with patch.object(cp, "RawKernel", side_effect=build):
            first = loader.materialize("outer", None, {"raw": raw})
            raw[...] = cp.asarray(raw_host * 3)
            second = loader.materialize("outer", None, {"raw": raw})
            self.assertGreaterEqual(len(kernels), 1)
        def nested_reference(value):
            base = _outer_oracle(spin, value)
            inner = np.stack([base, base]).astype(np.float32)
            return np.stack([inner, inner]).astype(np.complex128)
        expected_first = nested_reference(raw_host)
        expected_second = nested_reference(raw_host * 3)
        np.testing.assert_array_equal(cp.asnumpy(first), expected_first)
        np.testing.assert_array_equal(cp.asnumpy(second), expected_second)
        self.assertNotEqual(first.data.ptr, second.data.ptr)

        for value in (np.asarray(2.5, dtype=np.float32),
                      np.empty((0, 3), dtype=np.complex64)):
            scalar_prep = _spin_preparation(np.eye(4, dtype=np.complex64), conjugate=True)
            value_expected = _outer_oracle(np.eye(4, dtype=np.complex64), value, conjugate=True)
            scalar_loader = bind_prepared_loader(None, MappingProxyType({"x": scalar_prep}),
                MappingProxyType({"x": TensorSpec(value_expected.shape, value_expected.dtype)}), backend=cp)
            device_value = cp.asarray(value)
            with patch.object(cp, "RawKernel", side_effect=build):
                actual = scalar_loader.materialize("x", None, {"raw": device_value})
            np.testing.assert_array_equal(cp.asnumpy(actual), value_expected)

        zero_profile = (np.empty((0, 2), np.complex64), np.empty((0, 3), np.float32),
                        np.empty((0,), np.float64))
        zero_raw = np.arange(6, dtype=np.complex64).reshape(2, 3)
        zero_prep = _spin_preparation(np.eye(4, dtype=np.complex64), profile=zero_profile)
        zero_loader = bind_prepared_loader(None, MappingProxyType({"x": zero_prep}),
            MappingProxyType({"x": TensorSpec((4, 4, 2, 3), np.complex128)}), backend=cp)
        with patch.object(cp, "RawKernel", side_effect=build):
            zero_actual = zero_loader.materialize("x", None, {"raw": cp.asarray(zero_raw)})
        np.testing.assert_array_equal(cp.asnumpy(zero_actual), np.zeros((4, 4, 2, 3), np.complex128))

    def test_thirty_two_lane_profile_chunks_match_legacy_and_scalar_oracle(self):
        cp = self.cp
        _raw_spin_kernel.cache_clear()
        rng = np.random.default_rng(9702)
        left_host = (rng.normal(size=(4, 3)) * np.array([[1e4], [1e-4], [1], [-1e4]])).astype(np.complex64)
        right_host = (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))).astype(np.complex128)
        weights_host = np.array([1.0000001, -1.0, 1e-7, 0.5], dtype=np.float32)
        spin_seed = np.eye(4, dtype=np.float32)
        recipe = _spin_preparation(spin_seed, conjugate=True, permutation=(1, 0),
                                   profile=(left_host, right_host, weights_host)).recipe
        left_device, right_device, weights_device = (
            cp.asarray(left_host), cp.asarray(right_host), cp.asarray(weights_host))
        leaf_spec = TensorSpec((4, 4, 3, 3), np.complex128)

        for count in (9, 17, 31, 32, 33, 65):
            with self.subTest(count=count):
                leaves, inputs, expected = [], [], []
                for index in range(count):
                    spin_dtype = (np.complex64, np.float32, np.complex128)[index % 3]
                    spin_values = rng.normal(size=(4, 4))
                    if np.dtype(spin_dtype).kind == "c":
                        spin_values = spin_values + 1j * rng.normal(size=(4, 4))
                    spin_host = spin_values.astype(spin_dtype)
                    spin_device = cp.asarray(spin_host)

                    raw_dtype = (np.complex64, np.float32, np.complex128)[index % 3]
                    raw_values = rng.normal(size=(3, 3))
                    if np.dtype(raw_dtype).kind == "c":
                        raw_values = raw_values + 1j * rng.normal(size=(3, 3))
                    raw_base = raw_values.astype(raw_dtype)
                    layout = index % 4
                    if layout == 0:
                        raw_host = raw_base[::-1, ::-1]
                        raw_device = cp.asarray(raw_base)[::-1, ::-1]
                    elif layout == 1:
                        raw_host = np.asfortranarray(raw_base)
                        raw_device = cp.asfortranarray(cp.asarray(raw_base))
                    elif layout == 2:
                        raw_host = np.broadcast_to(raw_base[:1, :], (3, 3))
                        raw_device = cp.broadcast_to(cp.asarray(raw_base[:1, :]), (3, 3))
                    else:
                        raw_host = raw_base
                        raw_device = cp.asarray(raw_base)
                    leaves.append(_SpinLeaf(recipe,
                        (spin_device, left_device, right_device, weights_device), leaf_spec,
                        casts=(np.dtype(np.complex64), np.dtype(np.complex128))))
                    inputs.append(raw_device)
                    projected = _profile_oracle(spin_host, raw_host, left_host, right_host,
                        weights_host, conjugate=True, permutation=(1, 0))
                    expected.append(projected.astype(np.complex64).astype(np.complex128))

                spec = TensorSpec((count, *leaf_spec.shape), np.complex128)
                raw_program = FusedSpinPreparation(cp, leaves, spec, stacked=True,
                                                   prefer_raw=True)
                legacy_program = FusedSpinPreparation(cp, leaves, spec, stacked=True,
                                                      prefer_raw=False)
                build, kernels = self._tracked_raw(cp)
                with patch.object(cp, "RawKernel", side_effect=build):
                    actual = raw_program.materialize({}, inputs=tuple(inputs))
                    warm = raw_program.materialize({}, inputs=tuple(inputs))
                self.assertIsNot(actual, warm)
                self.assertEqual(sum(kernel.calls for kernel in kernels), 2 * ((count + 31) // 32))
                # Distinct spin owners stay distinct, while the three shared
                # profile arrays are each passed once per chunk, cold and warm.
                expected_counts = [2*min(32, count-offset)+6
                                   for offset in range(0, count, 32)] * 2
                self.assertCountEqual([n for kernel in kernels for n in kernel.argument_counts],
                                      expected_counts)
                legacy = legacy_program.materialize({}, inputs=tuple(inputs))
                actual_host, legacy_host = cp.asnumpy(actual), cp.asnumpy(legacy)
                expected_host = np.stack(expected)
                np.testing.assert_array_equal(cp.asnumpy(warm), actual_host)
                np.testing.assert_array_equal(actual_host, legacy_host)
                np.testing.assert_allclose(actual_host, expected_host, rtol=2e-7, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
