"""Grouped copies preserve the saved packing allowance and current inputs."""
from dataclasses import replace
import gc
import importlib
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide
from lqcdweave.contraction.execution.operations.packing_kernel import (
    _raw_kernel, bind_gemm_group_packing,
)


def descriptors(dtype=np.float64, *, count=3, parts=2, permuted=False, mixed=False):
    catalog, sides = {}, []
    permutation = (1, 0, 2) if permuted else (0, 1)
    source_shape = (3, 2, 2) if permuted else (3, 2)
    merged = (2, 3, 2 * parts) if permuted else (3, 2 * parts)
    for child in range(count):
        keys = (f'a{child}', f'b{child}',
                *(f'part{part}_{child}' for part in range(2, parts)))
        for index, key in enumerate(keys):
            catalog[key] = TensorSpec(source_shape,
                np.float32 if mixed and index == 0 else dtype)
        sides.append(_BatchSide(tuple((key, permutation) for key in keys),
            merged, len(merged) - 1, False, original_packer=lambda *args: None,
            input_layouts=(permutation,) * len(keys)))
    matrix_shape = (merged[0] * merged[1], merged[2]) if permuted else merged
    return tuple(sides), catalog, matrix_shape


class GroupPackingProtocolTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()

    def backend(self):
        self.sources, self.calls, self.allocations = [], [], []
        def factory(source, name, **kwargs):
            self.sources.append(source)
            def kernel(grid, block, arguments):
                self.calls.append((grid, block, tuple(id(a) for a in arguments), arguments[-1].shape))
            return kernel
        def empty(shape, **kwargs):
            self.allocations.append(shape)
            return np.empty(shape, **kwargs)
        return SimpleNamespace(__name__='cupy', RawKernel=factory, empty=empty)

    def test_one_launch_exact_workspace_and_no_array_cache(self):
        backend = self.backend()
        sides, catalog, shape = descriptors()
        pack = bind_gemm_group_packing(backend, sides, catalog, shape, np.float64)
        arrays = {key: np.ones(spec.shape) for key, spec in catalog.items()}
        references = tuple(weakref.ref(value) for value in arrays.values())
        first = pack(arrays)
        self.assertEqual(self.allocations, [(3, 3, 4)])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0][2]), 7)
        self.assertEqual(first[0].base.nbytes, sum(np.prod(side.shape) * 8 for side in sides))
        self.assertTrue(all(np.shares_memory(first[0], value.base) for value in first))
        first_ids = self.calls[0][2]
        del arrays, first
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        fresh = {key: np.full(spec.shape, 7.) for key, spec in catalog.items()}
        pack(fresh)
        self.assertEqual(len(self.sources), 1)
        self.assertNotEqual(first_ids[:-1], self.calls[-1][2][:-1])

    def test_any_existing_contiguous_owner_keeps_zero_copy(self):
        backend = self.backend()
        sides, catalog, shape = descriptors()
        arrays = {key: np.ones(spec.shape) for key, spec in catalog.items()}
        owner = np.arange(12.).reshape(3, 4)
        arrays['a2'], arrays['b2'] = owner[:, :2], owner[:, 2:]
        pack = bind_gemm_group_packing(backend, sides, catalog, shape, np.float64)
        self.assertIsNone(pack(arrays))
        self.assertEqual(self.allocations, [])
        self.assertEqual(self.calls, [])

    def test_prepared_or_unsupported_descriptor_falls_back(self):
        backend = self.backend()
        sides, catalog, shape = descriptors()
        self.assertIsNone(bind_gemm_group_packing(backend,
            (replace(sides[0], prepared=True), *sides[1:]), catalog, shape, np.float64))
        complex_sides, specs, shape = descriptors(np.complex64)
        self.assertIsNone(bind_gemm_group_packing(backend, complex_sides, specs, shape, np.float64))
        self.assertEqual(self.sources, [])

    def test_wide_inputs_split_within_children_and_share_one_fresh_workspace(self):
        for count, parts in ((3, 11), (4, 9), (21, 3)):
            with self.subTest(count=count, parts=parts):
                backend = self.backend()
                sides, specs, shape = descriptors(count=count, parts=parts)
                pack = bind_gemm_group_packing(backend, sides, specs, shape, np.float64)
                retained = None
                for scale in (1, 7):
                    arrays = {key: np.full(spec.shape, index + scale, dtype=spec.dtype)
                              for index, (key, spec) in enumerate(specs.items())}
                    before = len(self.calls)
                    packed = pack.pack_buffer(arrays)
                    calls = self.calls[before:]
                    inputs = tuple(id(arrays[key]) for side in sides for key, _ in side.operands)
                    self.assertEqual(len(calls), (len(inputs) + 31) // 32)
                    self.assertEqual(self.allocations[-1], (count, *shape))
                    self.assertEqual(packed.nbytes, sum(np.prod(side.shape) * 8 for side in sides))
                    self.assertEqual(tuple(arg for call in calls for arg in call[2][:-1]), inputs)
                    self.assertTrue(all(len(call[2]) <= 33 and call[2][-1] == id(packed) for call in calls))
                    if retained is not None:
                        self.assertFalse(np.shares_memory(packed, retained))
                    retained = packed

    def test_thirtytwo_inputs_fit_one_launch_and_exact_workspace(self):
        backend = self.backend()
        sides, catalog, shape = descriptors(count=4, parts=8)
        self.assertEqual(sum(len(side.operands) for side in sides), 32)
        pack = bind_gemm_group_packing(backend, sides, catalog, shape, np.float64)
        arrays = {key: np.full(spec.shape, index, dtype=spec.dtype)
                  for index, (key, spec) in enumerate(catalog.items())}

        packed = pack(arrays)

        self.assertEqual(self.allocations, [(4, 3, 16)])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2], tuple(
            id(arrays[key]) for side in sides for key, _ in side.operands
        ) + (id(packed[0].base),))
        self.assertEqual(self.calls[0][3], (4, 3, 16))
        self.assertEqual(len(self.calls[0][2]), 33)
        self.assertEqual(packed[0].base.shape, (4, 3, 16))

    def test_stride_variants_are_bounded(self):
        backend = self.backend()
        sides, catalog, shape = descriptors()
        pack = bind_gemm_group_packing(backend, sides, catalog, shape, np.float64)
        for stride in range(1, 10):
            arrays = {key: np.ones((3, 2 * stride))[:, ::stride] for key in catalog}
            pack(arrays)
        self.assertEqual(len(self.sources), 9)
        hits = _raw_kernel.cache_info().hits
        pack({key: np.ones((3, 2)) for key in catalog})
        # Evicted layout rebinds; the outer code cache can reuse its kernel.
        self.assertEqual(_raw_kernel.cache_info().hits, hits + 1)

    def test_group_stride_mapping_occurs_only_on_layout_miss(self):
        indexed = []

        class Steps(tuple):
            def __getitem__(self, axis):
                indexed.append(axis)
                return super().__getitem__(axis)

        class Strided(np.ndarray):
            @property
            def strides(self):
                return Steps(super().strides)

        backend = self.backend()
        sides, catalog, shape = descriptors(permuted=True)
        pack = bind_gemm_group_packing(backend, sides, catalog, shape, np.float64)

        def inputs(reverse=False):
            return {key: (np.ones(spec.shape)[::-1] if reverse else np.ones(spec.shape)).view(Strided)
                    for key, spec in catalog.items()}

        pack(inputs())
        self.assertEqual(indexed, [1, 0, 2] * 6)
        indexed.clear()
        pack(inputs())
        self.assertEqual(indexed, [])
        self.assertEqual(len(self.sources), 1)
        pack(inputs(reverse=True))
        self.assertEqual(indexed, [1, 0, 2] * 6)
        self.assertEqual(len(self.sources), 2)
        self.assertIn('(-32LL)', self.sources[1])


class GroupPackingCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(f'CUDA unavailable: {error}') from error

    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_permutation_negative_stride_mixed_dtype_and_fresh_inputs(self):
        cp = self.cp
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for count, parts in ((3, 2), (4, 8), (3, 11), (21, 3)):
                with self.subTest(dtype=dtype, count=count, parts=parts):
                    sides, catalog, shape = descriptors(
                        dtype, count=count, parts=parts, permuted=True, mixed=True)
                    pack = bind_gemm_group_packing(cp, sides, catalog, shape, dtype)
                    for scale in (1, 7):
                        arrays, host = {}, {}
                        for index, (key, spec) in enumerate(catalog.items()):
                            value = (np.arange(12.).reshape(spec.shape) + index / 8) * scale
                            if spec.dtype.kind == 'c':
                                value = value + 1j * value / 9
                            value = value.astype(spec.dtype)
                            arrays[key] = cp.asarray(value)[::-1]
                            host[key] = value[::-1]
                        packed = pack(arrays)
                        for child, side in enumerate(sides):
                            expected = np.concatenate(tuple(host[key].transpose((1, 0, 2))
                                for key, _ in side.operands), axis=2).astype(dtype).reshape(shape)
                            np.testing.assert_array_equal(cp.asnumpy(packed[child]), expected)

    def test_group_executes_one_packing_launch_and_debug_keeps_child_path(self):
        from unittest.mock import patch
        from correlator.execution.test_gemm_group_device import _group
        from lqcdweave.contraction.execution.operations.gemm_group import bind_dense_gemm_group
        from lqcdweave.contraction.execution.diagnostics import execution_debug
        cp = self.cp
        group, catalog = _group(np.complex128)
        host = {key: (np.arange(np.prod(spec.shape)).reshape(spec.shape) + index / 7
                       + 1j / 11).astype(spec.dtype)
                for index, (key, spec) in enumerate(catalog.items()) if key.startswith(('left', 'right'))}
        calls = []
        original = cp.RawKernel
        def factory(source, name, **kwargs):
            kernel = original(source, name, **kwargs)
            def call(grid, block, arguments):
                if name == 'ed_dense_batch_pack_raw':
                    calls.append((len(arguments) - 1, arguments[-1].shape))
                return kernel(grid, block, arguments)
            return call
        with patch.object(cp, 'RawKernel', side_effect=factory):
            run = bind_dense_gemm_group(group, catalog, cp)
            for debug in (False, True):
                calls.clear()
                arrays = {key: cp.asarray(value) for key, value in host.items()}
                with execution_debug(debug):
                    cube = run(arrays, device=True)
                self.assertEqual(len(calls), 3 if debug else 1)
                for child, batch in enumerate(group.batches):
                    expected = host[batch.left[0].tensor] @ np.concatenate(
                        tuple(host[o.tensor] for o in batch.right), axis=1)
                    np.testing.assert_allclose(cp.asnumpy(cube[child]), expected, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
