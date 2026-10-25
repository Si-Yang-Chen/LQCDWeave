"""Generated-kernel CPU oracles and separate optional real CUDA/HIP checks.

The C++ harness executes the emitted element body with host strided arrays. It
checks generated indexing/arithmetic and dispatch/allocation/upload counts, not
CuPy compilation, CUDA/HIP device behavior, concurrency or performance.
"""
import ctypes
import gc
import hashlib
from itertools import product
from pathlib import Path
import shutil
import subprocess
import tempfile
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
    FusedSpinPreparation, _MAX_VARIANTS, _ROUND_PREAMBLE,
)
from lqcdweave.contraction.execution.operations.output import bind_output_layout
from lqcdweave.contraction.execution.operations.output_kernel import _copy_kernel
from correlator.execution.test_prebound_preparation import _spin_preparation, _outer_oracle


_TYPES = {'float32': 'float', 'float64': 'double', 'complex64': 'complex<float>',
          'complex128': 'complex<double>', 'int64': 'long long', 'int32': 'int',
          'uint64': 'unsigned long long', 'uint32': 'unsigned int', 'bool': 'bool'}
_HOST_PREAMBLE = r'''
#include <cstdint>
// Match CuPy's const-reference constructors (std::complex<float/double> take
// values and would incorrectly accept volatile lvalues at this ABI boundary).
template <typename T> struct complex {
    T real_value, imag_value;
    complex() = default;
    complex(const T& real): real_value(real), imag_value(0) {}
    complex(const T& real, const T& imag): real_value(real), imag_value(imag) {}
    template <typename U> complex(const complex<U>& other):
        real_value(other.real()), imag_value(other.imag()) {}
    T real() const { return real_value; }
    T imag() const { return imag_value; }
};
template <typename T> complex<T> conj(const complex<T>& z) {
    return complex<T>(z.real(), -z.imag());
}
template <typename T> complex<T> operator+(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real() + b.real(), a.imag() + b.imag());
}
template <typename T> complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    T real = a.real() * b.real() - a.imag() * b.imag();
    T imag = a.real() * b.imag() + a.imag() * b.real();
    return complex<T>(real, imag);
}
#define __device__
template <typename T> struct View {
    void* data; long long ndim; const long long* shape; const long long* strides;
    T& operator[](long long index) const {
        long long offset = 0;
        for (long long axis = ndim - 1; axis >= 0; --axis) {
            offset += (index % shape[axis]) * strides[axis];
            index /= shape[axis];
        }
        return *reinterpret_cast<T*>(static_cast<char*>(data) + offset);
    }
};
'''


class _HostKernel:
    def __init__(self, owner, inputs, outputs, operation, name, *, preamble=''):
        self.owner, self.parameters = owner, (inputs + ', ' + outputs).split(', ')
        self.operation, self.name, self.preamble = operation, name, preamble
        owner.created += 1

    def __call__(self, *args, size):
        self.owner.launches += 1
        arrays, scalars, definitions = [], [], []
        for parameter, value in zip(self.parameters, args):
            parts = parameter.split()
            dtype, name = parts[-2:]
            if parts[0] == 'raw':
                index = len(arrays)
                actual = value.dtype.name if dtype == 'T' else dtype
                assert actual == value.dtype.name, (parameter, value.dtype)
                definitions.append(f'View<{_TYPES[actual]}> {name}'
                    f'{{data[{index}], ndim[{index}], shape[{index}], strides[{index}]}};')
                arrays.append(value)
            else:
                definitions.append(f'long long {name} = scalar[{len(scalars)}];')
                scalars.append(int(value))
        source = (_HOST_PREAMBLE + self.preamble + '\nextern "C" void run(void** data, '
            'long long* ndim, long long** shape, long long** strides, long long* scalar, long long size) {\n'
            + '\n'.join(definitions) + '\nfor (long long i = 0; i < size; ++i) {\n'
            + self.operation + '\n}\n}')
        digest = hashlib.sha256(source.encode()).hexdigest()
        function = self.owner.compiled.get(digest)
        if function is None:
            path = Path(self.owner.directory.name) / digest
            path.with_suffix('.cpp').write_text(source)
            subprocess.run([self.owner.compiler, '-shared', '-fPIC', '-std=c++11', '-O2',
                '-ffp-contract=fast', str(path.with_suffix('.cpp')), '-o', str(path.with_suffix('.so'))],
                check=True, capture_output=True, text=True)
            function = ctypes.CDLL(str(path.with_suffix('.so'))).run
            function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
            self.owner.compiled[digest] = function
        integer = ctypes.c_int64
        shapes = [(integer * len(a.shape))(*a.shape) for a in arrays]
        strides = [(integer * len(a.strides))(*a.strides) for a in arrays]
        function((ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays)),
            (integer * len(arrays))(*(a.ndim for a in arrays)),
            (ctypes.POINTER(integer) * len(arrays))(*shapes),
            (ctypes.POINTER(integer) * len(arrays))(*strides),
            (integer * len(scalars))(*scalars), size)


class _HostKernelBackend:
    __name__ = 'cupy'
    ndarray = np.ndarray
    result_type = staticmethod(np.result_type)

    def __init__(self):
        self.compiler = shutil.which('g++') or shutil.which('clang++')
        if self.compiler is None:
            raise unittest.SkipTest('host C++ compiler unavailable; emitted-kernel oracle not run')
        self.directory = tempfile.TemporaryDirectory(prefix='ed-fused-kernel-test-')
        self.compiled = {}
        self.created = self.launches = 0
        self.asarray = Mock(side_effect=lambda value: np.array(value, copy=True))
        self.empty = Mock(wraps=np.empty)
        self.multiply = Mock(wraps=np.multiply)
        self.zeros_like = Mock(wraps=np.zeros_like)
        self.einsum = Mock(side_effect=AssertionError('unexpected einsum'))

    def ElementwiseKernel(self, *args, **kwargs):
        return _HostKernel(self, *args, **kwargs)

    def close(self):
        _copy_kernel.cache_clear()
        self.compiled.clear()
        self.directory.cleanup()


def _profile_oracle(spin, raw, left, right, weights, *, conjugate=False, permutation=()):
    """Scalar loop with explicit strong-ndarray CuPy promotion at every stage."""
    d1 = np.result_type(weights.dtype, left.dtype)
    d2, d3 = np.result_type(d1, raw.dtype), np.result_type(d1, raw.dtype, right.dtype)
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


def _old_expression(backend, spin, raw, profile=(), *, conjugate=False, permutation=()):
    """Preserved pre-fusion expression; optional GPU gate runs this on CuPy."""
    value = raw
    if profile:
        left, right, weights = profile
        value = backend.zeros_like(raw, dtype=complex)
        for coefficient, l, r in zip(weights, left, right, strict=True):
            value += coefficient * l.conj()[:, None] * raw * r[None, :]
    if conjugate:
        value = value.conj()
    if permutation:
        value = value.transpose(permutation)
    return backend.einsum('ab,...->ab...', spin, value, optimize=False)


def _loader(spin, raw, backend, *, profile=(), conjugate=False, permutation=()):
    shape = tuple(raw.shape[axis] for axis in permutation) if permutation else raw.shape
    dtype = np.result_type(spin.dtype, np.complex128 if profile else raw.dtype)
    return bind_prepared_loader(None, {'x': _spin_preparation(spin, profile=profile,
        conjugate=conjugate, permutation=permutation)},
        {'x': TensorSpec((4, 4, *shape), dtype)}, backend=backend)


def _structural_loader(backend):
    spins = (np.eye(4, dtype=np.complex128) * (1.00000007 + .3j),
             np.ones((4, 4), np.complex128) * (.7 - .4j))
    preps, specs = {}, {}
    for point in range(2):
        raw = f'raw-{point}'
        for lane, spin in enumerate(spins):
            key = f'{point}-{lane}'
            preps[key] = _spin_preparation(spin, block=raw, conjugate=True, permutation=(1, 0))
            specs[key] = TensorSpec((4, 4, 2, 3), np.complex128)
        inner, outer = f'inner-{point}', f'outer-{point}'
        preps[inner] = SourcePreparation((raw,), recipe=TensorPreparation(
            'stack', tuple(f'{point}-{lane}' for lane in range(2))))
        specs[inner] = TensorSpec((2, 4, 4, 2, 3), np.complex64)
        preps[outer] = SourcePreparation((raw,), recipe=TensorPreparation('stack', (inner,)))
        specs[outer] = TensorSpec((1, 2, 4, 4, 2, 3), np.complex128)
    return bind_prepared_loader(None, MappingProxyType(preps), MappingProxyType(specs),
                                backend=backend), spins


class FrozenStructureBindingTests(unittest.TestCase):
    """Binding/ownership oracle with NumPy; this does not execute GPU code."""

    def setUp(self):
        self.backend = SimpleNamespace(__name__='cupy', result_type=np.result_type,
            asarray=Mock(side_effect=lambda value: np.array(value, copy=True)),
            empty=np.empty, multiply=np.multiply, zeros_like=np.zeros_like,
            ElementwiseKernel=Mock(side_effect=AssertionError('GPU code in binding test')))

        def scalar_materialize(program, blocks, destination=None, *, inputs=None):
            inputs = tuple(blocks[leaf.recipe.inputs[0]] for leaf in program.leaves) if inputs is None else inputs
            values = []
            for leaf, raw in zip(program.leaves, inputs, strict=True):
                value = _outer_oracle(leaf.constants[0], raw,
                    conjugate=leaf.recipe.conjugate, permutation=leaf.recipe.permutation)
                for dtype in leaf.casts:
                    value = value.astype(dtype)
                values.append(value)
            result = (np.stack(values).reshape(program.spec.shape) if program.stacked else values[0])
            if destination is not None:
                destination[...] = result
                return destination
            return result

        oracle = patch.object(FusedSpinPreparation, 'materialize', scalar_materialize)
        oracle.start()
        self.addCleanup(oracle.stop)

    def test_frozen_structures_share_binding_without_capturing_current_arrays(self):
        loader, spins = _structural_loader(self.backend)
        refs = []
        with patch.object(loader, '_constants', wraps=loader._constants) as constants:
            for point in range(2):
                raw = (np.arange(6).reshape(3, 2) + (point + .5j)).astype(np.complex64)[::-1]
                with patch.object(loader, '_bind_spin', side_effect=AssertionError('per-ID bind')):
                    actual = loader.materialize(f'outer-{point}', None, {f'raw-{point}': raw})
                expected = np.stack([_outer_oracle(spin, raw, conjugate=True, permutation=(1, 0))
                                     for spin in spins]).astype(np.complex64)[None].astype(np.complex128)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                # Debug takes the established independent binding path.
                with execution_debug(True):
                    debug = loader.materialize(f'outer-{point}', None, {f'raw-{point}': raw})
                np.testing.assert_array_equal(actual, debug)
                refs.extend(weakref.ref(value) for value in (raw, actual, debug))
                del raw, actual, debug
            # Debug fills per-ID constants too; the normal second point does not.
            self.assertEqual(constants.call_count, 6)
        self.assertIs(loader._fused_bindings['outer-0'][0], loader._fused_bindings['outer-1'][0])
        self.assertEqual(len(loader._fused_structures), 4)
        self.assertEqual(self.backend.asarray.call_count, 2)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_frozen_unsupported_structure_falls_back_and_cycles_still_fail(self):
        preps = {'x': SourcePreparation(('raw',), recipe=TensorPreparation('identity', ('raw',))),
                 's': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x',)))}
        specs = {'x': TensorSpec((2,), np.float64), 's': TensorSpec((1, 2), np.float64)}
        loader = bind_prepared_loader(None, MappingProxyType(preps), MappingProxyType(specs),
                                      backend=self.backend)
        np.testing.assert_array_equal(loader.materialize('s', None, {'raw': np.array([2., 3.])}), [[2., 3.]])
        self.assertIsNone(loader._fused_bindings['s'])
        cyclic = {'s': SourcePreparation((), recipe=TensorPreparation('stack', ('s',)))}
        loader = bind_prepared_loader(None, MappingProxyType(cyclic),
            MappingProxyType({'s': TensorSpec((1,), np.float64)}), backend=self.backend)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            loader.materialize('s', None, {})

    def test_frozen_structure_index_is_bounded_and_distinguishes_constants(self):
        preps = {str(i): _spin_preparation(np.full((4, 4), i + .5)) for i in range(514)}
        specs = {key: TensorSpec((4, 4, 1), np.float64) for key in preps}
        loader = bind_prepared_loader(None, MappingProxyType(preps), MappingProxyType(specs),
                                      backend=self.backend)
        programs = [loader._bind_fused_structure(key, set())[0] for key in preps]
        self.assertEqual(len(loader._fused_structures), 512)
        self.assertEqual(len({id(program) for program in programs}), 514)


class FusedPreparationHostTests(unittest.TestCase):
    def setUp(self):
        self.backend = _HostKernelBackend()
        self.addCleanup(self.backend.close)

    def test_emitted_body_matches_scalar_oracle_with_strides_mixed_precision_and_dagger(self):
        rng = np.random.default_rng(852)
        combinations = ((np.float32, np.float64, np.float32, np.float32, np.float32),
                        (np.complex64, np.complex64, np.complex64, np.complex64, np.complex128),
                        (np.complex128, np.complex64, np.float64, np.complex128, np.complex64),
                        (np.float32, np.float32, np.float32, np.float32, np.float32))
        for types, dagger in product(combinations, (False, True)):
            arrays = []
            for shape, dtype in zip(((4, 4), (3, 2), (4, 3), (4, 2), (4,)), types):
                value = rng.normal(size=shape)
                if np.dtype(dtype).kind == 'c':
                    value = value + 1j * rng.normal(size=shape)
                arrays.append(value.astype(dtype))
            spin, raw, left, right, weights = arrays
            raw = raw[::-1, ::-1]
            axes = (1, 0) if dagger else ()
            expected = _profile_oracle(spin, raw, left, right, weights,
                                       conjugate=dagger, permutation=axes)
            loader = _loader(spin, raw, self.backend, profile=(left, right, weights),
                             conjugate=dagger, permutation=axes)
            with execution_debug(True):
                actual = loader.materialize('x', None, {'raw': raw})
            tolerance = 4e-6 if any(np.dtype(t).itemsize <= 8 for t in types) else 3e-14
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            self.assertEqual(actual.dtype, np.dtype(np.complex128))
        self.assertEqual(self.backend.launches, 8)
        self.assertEqual(self.backend.empty.call_count, 8)
        self.backend.multiply.assert_not_called()
        self.backend.zeros_like.assert_not_called()

    def test_product_rounding_and_ordered_complex128_accumulation_are_observable(self):
        spin = np.ones((4, 4), np.float32)
        raw = np.array([[1.00000007]], np.float64)
        left = np.array([[1], [1 - 2**-23], [1]], np.float32)
        weights = np.array([-1, 1 + 2**-23, 2**-30], np.float32)
        right = np.ones((3, 1), np.float32)
        loader = _loader(spin, raw, self.backend, profile=(left, right, weights))
        actual = loader.materialize('x', None, {'raw': raw})
        expected = _profile_oracle(spin, raw, left, right, weights)
        np.testing.assert_array_equal(actual, expected)
        regrouped = raw[0, 0] * np.sum(weights.astype(np.float64) * left[:, 0].astype(np.float64))
        self.assertNotEqual(actual[0, 0, 0, 0], regrouped)
        # Large cancellation exposes a changed profile sum order.
        weights = np.array([1e16, 1, -1e16, 3], np.float64)
        left = right = np.ones((4, 1), np.float64)
        loader = _loader(spin, np.ones((1, 1)), self.backend, profile=(left, right, weights))
        actual = loader.materialize('x', None, {'raw': np.ones((1, 1))})
        np.testing.assert_array_equal(actual, np.full((4, 4, 1, 1), 3, np.complex128))
        self.assertIn('volatile T re', _ROUND_PREAMBLE)

    def test_scalar_empty_general_rank_and_zero_profile_rows(self):
        spin = np.arange(16, dtype=np.float32).reshape(4, 4)
        for shape, axes in (((), ()), ((0, 2), (1, 0)), ((2, 3, 2), (2, 0, 1))):
            raw = np.arange(np.prod(shape, dtype=int), dtype=np.float64).reshape(shape)
            if raw.ndim:
                raw = raw[::-1]
            loader = _loader(spin, raw, self.backend, conjugate=True, permutation=axes)
            actual = loader.materialize('x', None, {'raw': raw})
            np.testing.assert_array_equal(actual, _outer_oracle(spin, raw, conjugate=True, permutation=axes))
        raw = np.ones((2, 3), np.complex64)
        profile = (np.empty((0, 2), np.float32), np.empty((0, 3), np.float32), np.empty(0, np.float64))
        loader = _loader(spin, raw, self.backend, profile=profile)
        np.testing.assert_array_equal(loader.materialize('x', None, {'raw': raw}), np.zeros((4, 4, 2, 3)))
        self.assertEqual(self.backend.launches, 3)

    def test_compatible_nested_stacks_use_one_owner_and_preserve_each_cast(self):
        raw = np.array([[1.00000007]], np.float64)
        spin = np.ones((4, 4), np.float64)
        preps = {'x': _spin_preparation(spin)}
        specs = {'x': TensorSpec((4, 4, 1, 1), np.float64)}
        preps['inner'] = SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x', 'x')))
        preps['outer'] = SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('inner', 'inner')))
        specs['inner'] = TensorSpec((2, 4, 4, 1, 1), np.float32)
        specs['outer'] = TensorSpec((2, 2, 4, 4, 1, 1), np.float64)
        loader = bind_prepared_loader(None, preps, specs, backend=self.backend)
        with execution_debug(True):
            actual = loader.materialize('outer', None, {'raw': raw})
        expected = np.stack([np.stack([_outer_oracle(spin, raw)] * 2).astype(np.float32)] * 2).astype(np.float64)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.backend.empty.call_count, 1)
        self.assertEqual(self.backend.asarray.call_count, 1)
        self.assertEqual(self.backend.launches, 1)
        self.backend.multiply.assert_not_called()

    def test_large_stack_is_chunked_without_pointer_uploads_or_intermediate_owners(self):
        spin, raw = np.eye(4, dtype=np.complex64), np.ones((2, 3), np.complex64)
        preps = {'x': _spin_preparation(spin),
            's': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x',) * 19))}
        specs = {'x': TensorSpec((4, 4, 2, 3), np.complex64),
                 's': TensorSpec((19, 4, 4, 2, 3), np.complex64)}
        loader = bind_prepared_loader(None, preps, specs, backend=self.backend)
        for run in range(2):
            actual = loader.materialize('s', None, {'raw': raw})
            np.testing.assert_array_equal(actual, np.stack([_outer_oracle(spin, raw)] * 19))
        self.assertEqual(self.backend.launches, 6)
        self.assertEqual(self.backend.asarray.call_count, 1)
        self.assertEqual(self.backend.empty.call_count, 2)
        self.assertEqual(self.backend.created, 2)

    def test_mutable_records_specs_raw_and_backend_do_not_reuse_numerical_values(self):
        raw = np.ones((2, 3), np.float64)
        preps = {'x': _spin_preparation(np.eye(4)),
            's': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x',)))}
        specs = {'x': TensorSpec((4, 4, 2, 3), np.float64),
                 's': TensorSpec((1, 4, 4, 2, 3), np.float64)}
        loader = bind_prepared_loader(None, preps, specs, backend=self.backend)
        first = loader.materialize('s', None, {'raw': raw})
        preps['x'] = _spin_preparation(np.full((4, 4), 2.0))
        raw[:] = 3
        second = loader.materialize('s', None, {'raw': raw})
        np.testing.assert_array_equal(second, np.full(specs['s'].shape, 6.0))
        np.testing.assert_array_equal(first, np.eye(4)[None, :, :, None, None] * np.ones(specs['s'].shape))
        specs['s'] = TensorSpec((1, 4, 4, 2, 3), np.float32)
        third = loader.materialize('s', None, {'raw': raw})
        self.assertEqual(third.dtype, np.dtype(np.float32))
        other = _HostKernelBackend()
        self.addCleanup(other.close)
        rebound = loader.with_backend(other)
        np.testing.assert_array_equal(rebound.materialize('s', None, {'raw': raw}), third)
        self.assertEqual(other.asarray.call_count, 1)
        self.assertIsNot(loader._bound_preparations['s'].fused, rebound._bound_preparations['s'].fused)
        refs = [weakref.ref(a) for a in (raw, first, second, third)]
        del raw, first, second, third
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_profile_record_replacement_and_unsupported_shapes_keep_reference_semantics(self):
        spin, raw = np.ones((4, 4)), np.arange(6.0).reshape(2, 3)
        profile = (np.ones((2, 2)), np.ones((2, 3)), np.array([0.5, 1.5]))
        loader = _loader(spin, raw, self.backend, profile=profile)
        first = loader.materialize('x', None, {'raw': raw})
        new_profile = (*profile[:2], np.array([-3.0, 0.25]))
        loader.preparations['x'] = _spin_preparation(spin, profile=new_profile)
        actual = loader.materialize('x', None, {'raw': raw})
        np.testing.assert_array_equal(actual, _profile_oracle(spin, raw, *new_profile))
        np.testing.assert_array_equal(first, _profile_oracle(spin, raw, *profile))
        # Ordinary NumPy/CuPy broadcasting also allows one selected profile row
        # to broadcast over the raw mode extent. Unsupported fusion preserves it.
        broadcast_profile = (np.ones((1, 1)), np.ones((1, 3)), np.ones(1))
        fallback = _loader(spin, raw, self.backend, profile=broadcast_profile)
        count = self.backend.launches
        actual = fallback.materialize('x', None, {'raw': raw})
        np.testing.assert_array_equal(actual, _old_expression(np, spin, raw, broadcast_profile))
        self.assertEqual(self.backend.launches, count)
        # Recipe construction rejects inconsistent profile counts before any
        # kernel can index them; strict zip's original error is not discarded.
        with self.assertRaisesRegex(ValueError, 'compatible rows'):
            _spin_preparation(spin, profile=(np.ones((2, 2)), np.ones((1, 3)), np.ones(2)))

    def test_unsupported_argument_envelope_falls_back_before_allocating_owner(self):
        spin, raw = np.eye(4), np.ones((2, 3))
        loader = _loader(spin, raw, self.backend)
        with patch('lqcdweave.contraction.execution.preparation_kernel._MAX_ARGUMENT_BYTES', 1):
            actual = loader.materialize('x', None, {'raw': raw})
        np.testing.assert_array_equal(actual, _outer_oracle(spin, raw))
        self.assertEqual(self.backend.launches, 0)
        self.backend.empty.assert_not_called()
        self.assertEqual(self.backend.multiply.call_count, 1)

    def test_kernel_variants_are_bounded_and_missing_protocol_preserves_fallback(self):
        spin = np.eye(4, dtype=np.float64)
        raw = np.ones((2, 3), np.float64)
        loader = _loader(spin, raw, self.backend)
        for step in range(_MAX_VARIANTS + 3):
            storage = np.ones((2, 3 * (step + 1)))
            loader.materialize('x', None, {'raw': storage[:, ::(step + 1)]})
        self.assertEqual(len(loader._bound_preparations['x'].fused._variants), _MAX_VARIANTS)
        backend = SimpleNamespace(asarray=np.asarray, empty=np.empty, multiply=Mock(wraps=np.multiply),
                                  zeros_like=np.zeros_like, result_type=np.result_type)
        fallback = _loader(spin, raw, backend)
        np.testing.assert_array_equal(fallback.materialize('x', None, {'raw': raw}), _outer_oracle(spin, raw))
        self.assertEqual(backend.multiply.call_count, 1)


class FusedOutputHostTests(unittest.TestCase):
    def setUp(self):
        self.backend = _HostKernelBackend()
        self.addCleanup(self.backend.close)

    def test_output_group_launches_are_bounded_and_snapshot_is_owned(self):
        arrays = {str(i): (np.arange(24).reshape(4, 6)[:, ::2] + i)[::-1] for i in range(35)}
        arrays['scalar'] = np.array(7.5)
        arrays['empty'] = np.empty((0, 2), np.complex64)
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        layout = bind_output_layout(tuple(arrays), catalog)
        expected = tuple(np.array(value, copy=True) for value in arrays.values())
        actual = layout.copy(arrays, self.backend)
        self.assertEqual(self.backend.launches, 4)
        self.assertEqual(self.backend.empty.call_count, 3)
        self.backend.asarray.assert_not_called()
        for value in arrays.values():
            value[...] = 0
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)
        self.assertEqual(len(actual.owners), 3)
        self.assertFalse(np.shares_memory(actual[0], actual[1]))
        self.assertIs(actual[0].base, actual[1].base)

    def test_mixed_groups_strides_scalars_and_changed_inputs_match_scalar_snapshot(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128, np.int32, np.bool_):
            with self.subTest(dtype=dtype):
                arrays = {'a': np.arange(12).reshape(3, 4).astype(dtype).T[::-1],
                          'b': np.ones((4, 3), dtype=dtype)[:, ::-1]}
                layout = bind_output_layout(('b', 'a', 'b'),
                    {k: TensorSpec(v.shape, v.dtype) for k, v in arrays.items()})
                for run in range(2):
                    result = layout.copy(arrays, self.backend)
                    for index, key in enumerate(('b', 'a', 'b')):
                        expected = np.empty(arrays[key].shape, dtype=dtype)
                        for coordinate in np.ndindex(expected.shape):
                            expected[coordinate] = arrays[key][coordinate]
                        np.testing.assert_array_equal(result[index], expected)
                    arrays['a'][...] = run + 3
        self.assertEqual(self.backend.launches, 12)
        self.assertEqual(self.backend.empty.call_count, 12)
        self.backend.asarray.assert_not_called()

    def test_saved_output_shape_dtype_boundary_and_custom_backend_fallback(self):
        layout = bind_output_layout(('a', 'b'), {key: TensorSpec((2,), np.float32) for key in ('a', 'b')})
        arrays = {'a': np.ones(2, np.float32), 'b': np.ones(3, np.float32)}
        with self.assertRaisesRegex(ValueError, 'saved output layout'):
            layout.copy(arrays, self.backend)
        arrays['b'] = np.ones(2, np.float64)
        with self.assertRaisesRegex(ValueError, 'saved output layout'):
            layout.copy(arrays, self.backend)
        arrays['b'] = np.full(2, 9, np.float32)
        result = layout.copy(arrays, np)
        np.testing.assert_array_equal(result, [[1, 1], [9, 9]])
        self.assertEqual(self.backend.launches, 0)


class _CuPyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.backend = cupy


class FusedPreparationCuPyTests(_CuPyCase):
    def test_gpu_frozen_structure_shares_program_with_current_inputs_and_debug_oracle(self):
        cp = self.backend
        loader, spins = _structural_loader(cp)
        with patch.object(loader, '_constants', wraps=loader._constants) as constants:
            for point in range(2):
                host = (np.arange(6).reshape(3, 2) + (point + .5j)).astype(np.complex64)
                raw = cp.asarray(host)[::-1]
                with patch.object(loader, '_bind_spin', side_effect=AssertionError('per-ID bind')):
                    actual = loader.materialize(f'outer-{point}', None, {f'raw-{point}': raw})
                expected = np.stack([_outer_oracle(spin, host[::-1], conjugate=True, permutation=(1, 0))
                                     for spin in spins]).astype(np.complex64)[None].astype(np.complex128)
                np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-6, atol=2e-6)
                with execution_debug(True):
                    debug = loader.materialize(f'outer-{point}', None, {f'raw-{point}': raw})
                np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(debug))
            self.assertEqual(constants.call_count, 6)
        self.assertIs(loader._fused_bindings['outer-0'][0], loader._fused_bindings['outer-1'][0])
        self.assertEqual(len(loader._fused_structures), 4)

    def test_profiles_match_original_gpu_expression_with_strides_dtypes_and_dagger(self):
        cp = self.backend
        rng = np.random.default_rng(729)
        for spin_dtype, raw_dtype, left_dtype, right_dtype, weight_dtype, dagger in product(
                (np.complex64, np.complex128), (np.float32, np.complex64, np.complex128),
                (np.float32, np.complex64), (np.float64, np.complex128),
                (np.float32, np.complex128), (False, True)):
            host = []
            for shape, dtype in zip(((4, 4), (3, 2), (3, 3), (3, 2), (3,)),
                                    (spin_dtype, raw_dtype, left_dtype, right_dtype, weight_dtype)):
                value = rng.normal(size=shape)
                if np.dtype(dtype).kind == 'c':
                    value = value + 1j * rng.normal(size=shape)
                host.append(value.astype(dtype))
            spin, raw, left, right, weights = host
            device_raw = cp.asarray(raw)[::-1, ::-1]
            axes = (1, 0) if dagger else ()
            loader = _loader(spin, device_raw, cp, profile=(left, right, weights),
                             conjugate=dagger, permutation=axes)
            constants = tuple(cp.asarray(value) for value in (spin, left, right, weights))
            expected = _old_expression(cp, constants[0], device_raw, constants[1:],
                                       conjugate=dagger, permutation=axes)
            with execution_debug(True):
                actual = loader.materialize('x', None, {'raw': device_raw})
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), rtol=4e-13, atol=4e-13)
            independent = _profile_oracle(spin, raw[::-1, ::-1], left, right, weights,
                                          conjugate=dagger, permutation=axes)
            np.testing.assert_allclose(cp.asnumpy(actual), independent, rtol=5e-6, atol=5e-6)

    def test_gpu_intermediate_rounding_cancellation_empty_and_scalar(self):
        cp = self.backend
        spin = np.ones((4, 4), np.float32)
        raw = np.array([[1.00000007]], np.float64)
        left = np.array([[1], [1 - 2**-23], [1]], np.float32)
        right = np.ones((3, 1), np.float32)
        weights = np.array([-1, 1 + 2**-23, 2**-30], np.float32)
        loader = _loader(spin, raw, cp, profile=(left, right, weights))
        d_raw = cp.asarray(raw)
        actual = loader.materialize('x', None, {'raw': d_raw})
        expected = _old_expression(cp, cp.asarray(spin), d_raw,
                                  tuple(cp.asarray(x) for x in (left, right, weights)))
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
        for value in (np.array(2 + 3j), np.empty((0, 3), np.complex64)):
            loader = _loader(spin, value, cp, conjugate=True)
            actual = loader.materialize('x', None, {'raw': cp.asarray(value)})
            np.testing.assert_array_equal(cp.asnumpy(actual), _outer_oracle(spin, value, conjugate=True))

    def test_gpu_nested_stack_preserves_narrow_intermediate_and_changed_record(self):
        cp = self.backend
        spin, raw = np.ones((4, 4)), np.array([[1.00000007]])
        preps = {'x': _spin_preparation(spin),
            'inner': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x', 'x'))),
            'outer': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('inner', 'inner')))}
        specs = {'x': TensorSpec((4, 4, 1, 1), np.float64),
                 'inner': TensorSpec((2, 4, 4, 1, 1), np.float32),
                 'outer': TensorSpec((2, 2, 4, 4, 1, 1), np.float64)}
        loader = bind_prepared_loader(None, preps, specs, backend=cp)
        device_raw = cp.asarray(raw)
        first = loader.materialize('outer', None, {'raw': device_raw})
        original = _old_expression(cp, cp.asarray(spin), device_raw)
        expected = cp.stack([cp.stack([original] * 2).astype(cp.float32)] * 2).astype(cp.float64)
        np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(expected))
        preps['x'] = _spin_preparation(2 * spin)
        actual = loader.materialize('outer', None, {'raw': device_raw})
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected) * 2)
        np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(expected))

    def test_gpu_stacked_profile_uses_one_kernel_allocation_and_no_repeat_uploads(self):
        cp = self.backend
        spin, raw = np.eye(4, dtype=np.complex64), np.ones((2, 3), np.complex64)
        profile = (np.ones((2, 2), np.complex64), np.ones((2, 3), np.float32),
                   np.array([1.3, -0.7], np.float64))
        preps = {'x': _spin_preparation(spin, profile=profile),
            's': SourcePreparation(('raw',), recipe=TensorPreparation('stack', ('x',) * 3))}
        specs = {'x': TensorSpec((4, 4, 2, 3), np.complex128),
                 's': TensorSpec((3, 4, 4, 2, 3), np.complex128)}
        loader = bind_prepared_loader(None, preps, specs, backend=cp)
        device_raw = cp.asarray(raw)
        kernels, factory = [], cp.ElementwiseKernel
        def tracked_kernel(*args, **kwargs):
            wrapped = Mock(wraps=factory(*args, **kwargs))
            kernels.append(wrapped)
            return wrapped
        with patch.object(cp, 'ElementwiseKernel', side_effect=tracked_kernel), \
                patch.object(cp, 'empty', wraps=cp.empty) as allocate, \
                patch.object(cp, 'asarray', wraps=cp.asarray) as upload:
            first = loader.materialize('s', None, {'raw': device_raw})
            first_uploads = upload.call_count
            self.assertEqual(first_uploads, 4)
            second = loader.materialize('s', None, {'raw': device_raw})
            self.assertEqual(allocate.call_count, 2)
            self.assertEqual(upload.call_count, first_uploads)
        self.assertEqual(len(loader._bound_preparations['s'].fused._variants), 1)
        self.assertEqual(sum(kernel.call_count for kernel in kernels), 2)
        expected = np.stack([_profile_oracle(spin, raw, *profile)] * 3)
        np.testing.assert_allclose(cp.asnumpy(first), expected, rtol=4e-13, atol=4e-13)
        np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(second))
        device_raw[:] = 0
        np.testing.assert_allclose(cp.asnumpy(first), expected, rtol=4e-13, atol=4e-13)


class FusedOutputCuPyTests(_CuPyCase):
    def test_gpu_packed_snapshot_strides_scalars_empty_and_many_lanes(self):
        cp = self.backend
        arrays = {str(i): (cp.arange(24, dtype=cp.complex128).reshape(4, 6) + i)[:, ::2][::-1]
                  for i in range(35)}
        arrays['scalar'], arrays['empty'] = cp.asarray(3.5), cp.empty((0, 2), cp.float32)
        layout = bind_output_layout(tuple(arrays), {k: TensorSpec(v.shape, v.dtype) for k, v in arrays.items()})
        expected = tuple(cp.asnumpy(value) for value in arrays.values())
        with patch.object(cp, 'asarray', side_effect=AssertionError('pointer-table upload')), \
                patch.object(cp, 'empty', wraps=cp.empty) as allocate:
            actual = layout.copy(arrays, cp)
        self.assertEqual(allocate.call_count, 3)
        for value in arrays.values():
            value[...] = 0
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(cp.asnumpy(value), reference)


if __name__ == '__main__':
    unittest.main()
