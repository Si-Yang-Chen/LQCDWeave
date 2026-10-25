"""Independent dense-matrix references for retained sparse point outputs."""

import unittest
from types import SimpleNamespace

import numpy as np

from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge,
)

try:
    import cupy as cp
    _DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
except Exception:
    cp, _DEVICE_COUNT = None, 0


def retained_case(dtype=np.complex128, points=5, outer=(2, 3), repeats=2, nc=3):
    rng = np.random.default_rng(6100301)

    def random(shape):
        real = rng.normal(size=shape)
        if np.dtype(dtype).kind == "c":
            real = real + 1j * rng.normal(size=shape)
        return real.astype(dtype)

    left = random((outer[0], 4, points, nc))
    right = random((4, points, nc, outer[1]))
    gamma = random((4, 4))
    selected = np.tile(np.arange(points)[::-1], repeats)
    pairs = np.column_stack((selected, selected))
    pairs = np.concatenate((pairs, [[0, points - 1]]))
    record = {"type": "sparse", "indices": pairs,
              "values": random((len(pairs), nc, nc))}
    return left, right, gamma, record


def dense_reference(left, right, gamma, record, *, transpose=False):
    """Assemble only Nc x Nc blocks, then independently multiply dense matrices."""
    points = left.shape[2]
    nc = left.shape[-1]
    blocks = np.zeros((points, nc, nc), dtype=np.complex128)
    for (p, q), value in zip(record["indices"], record["values"], strict=True):
        if p == q:
            blocks[p] += value.T if transpose else value
    spin = gamma.T if transpose else gamma
    result = np.zeros((left.shape[0], right.shape[-1], points), dtype=np.complex128)
    for point in range(points):
        result[:, :, point] = (
            left[:, :, point, :].reshape(left.shape[0], -1)
            @ np.kron(spin, blocks[point])
            @ right[:, point, :, :].reshape(-1, right.shape[-1])
        )
    return result


class _RetainedChecks:
    backend = np

    def check_case(self, *, dtype=np.complex128, transpose=False,
                   equation="aipc,jqdb->pab", same_point=True, two_labels=False):
        left, right, gamma, record = retained_case(dtype)
        # Deliberately non-contiguous input views exercise CuPy stride handling.
        left = left[::-1]
        right = right[..., ::-1]
        coefficient = 0.6 + 0.4j
        expected = coefficient * dense_reference(
            left, right, gamma, record, transpose=transpose,
        )
        if two_labels:
            diagonal = np.zeros(expected.shape + (expected.shape[-1],), complex)
            for point in range(expected.shape[-1]):
                diagonal[:, :, point, point] = expected[:, :, point]
            expected = diagonal.transpose(2, 0, 1, 3)
        else:
            expected = expected.transpose(2, 0, 1)
        empty = {"type": "sparse", "indices": np.empty((0, 2), np.int64),
                 "values": np.empty((0, 3, 3), dtype=dtype)}
        actual = contract_p2p_bridge(
            self.backend.asarray(left), self.backend.asarray(right),
            ((gamma, coefficient, record), (gamma, 2, empty)),
            used_np=left.shape[2], backend=self.backend,
            same_point=same_point, transpose_bridge=transpose, equation=equation,
        )
        if self.backend is not np:
            actual = self.backend.asnumpy(actual)
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        tolerance = 3e-5 if np.dtype(dtype).itemsize == 8 else 1e-11
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    def test_duplicate_unsorted_rows_and_empty_part(self):
        self.check_case()

    def test_transpose_retains_both_point_labels_on_diagonal(self):
        self.check_case(transpose=True, equation="aipc,jqdb->pabq", two_labels=True)

    def test_shared_point_label_without_same_point_flag(self):
        self.check_case(equation="aipc,jpdb->pab", same_point=False)

    def test_complex64_and_real_inputs_promote_output_to_complex128(self):
        for dtype in (np.complex64, np.float64):
            with self.subTest(dtype=dtype):
                self.check_case(dtype=dtype)

    def test_mixed_precision_preserves_output_precision(self):
        left, right, gamma, record = retained_case()
        left, gamma = left.astype(np.complex64), gamma.astype(np.complex64)
        expected = dense_reference(left, right, gamma, record).transpose(2, 0, 1)
        actual = contract_p2p_bridge(
            self.backend.asarray(left), self.backend.asarray(right),
            ((gamma, 1, record),), used_np=left.shape[2], backend=self.backend,
            same_point=True, equation="aipc,jqdb->pab",
        )
        if self.backend is not np:
            actual = self.backend.asnumpy(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)

    def test_other_spin_color_extents_preserve_the_generic_fallback(self):
        rng = np.random.default_rng(6100302)
        left = rng.normal(size=(2, 2, 3, 2)).astype(np.complex128)
        right = rng.normal(size=(3, 3, 2, 2)).astype(np.complex128)
        gamma = rng.normal(size=(2, 3)).astype(np.complex128)
        blocks = rng.normal(size=(3, 2, 2)).astype(np.complex128)
        record = {"type": "sparse", "indices": np.column_stack((np.arange(3), np.arange(3))),
                  "values": blocks}
        expected = np.stack([
            left[:, :, p, :].reshape(2, 4) @ np.kron(gamma, blocks[p])
            @ right[:, p, :, :].reshape(6, 2)
            for p in range(3)
        ])
        actual = contract_p2p_bridge(
            self.backend.asarray(left), self.backend.asarray(right),
            ((gamma, 1, record),), used_np=3, backend=self.backend,
            same_point=True, equation="aipc,jqdb->pab",
        )
        if self.backend is not np:
            actual = self.backend.asnumpy(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)

    def test_runtime_colors_independent_input_precisions_and_lifetime(self):
        coefficient = -0.3 + 0.7j
        for nc in (1, 2, 3, 4, 5):
            for dtypes in ((np.complex64,)*4, (np.complex128,)*4,
                           (np.complex64,np.complex128,np.float32,np.complex64),
                           (np.float32,np.float64,np.complex64,np.complex128)):
                for transpose in (False, True):
                    with self.subTest(nc=nc, dtypes=dtypes, transpose=transpose):
                        left, right, gamma, record = retained_case(nc=nc)
                        left = (left.real if np.dtype(dtypes[0]).kind != 'c' else left).astype(dtypes[0])[::-1]
                        right = (right.real if np.dtype(dtypes[1]).kind != 'c' else right).astype(dtypes[1])[...,::-1]
                        gamma = (gamma.real if np.dtype(dtypes[2]).kind != 'c' else gamma).astype(dtypes[2])[:,::-1]
                        values = record['values']
                        record['values'] = (values.real if np.dtype(dtypes[3]).kind != 'c' else values).astype(dtypes[3])[:,::-1,:]
                        expected = coefficient * dense_reference(left, right, gamma, record, transpose=transpose)
                        operands = tuple(map(self.backend.asarray, (left, right)))
                        actual = contract_p2p_bridge(*operands, ((gamma,coefficient,record),),
                            used_np=left.shape[2], backend=self.backend, same_point=True,
                            transpose_bridge=transpose, equation='aipc,jqdb->pab')
                        result = self.backend.asnumpy(actual) if self.backend is not np else actual
                        self.assertEqual(result.dtype, np.dtype(np.complex128))
                        np.testing.assert_allclose(result, expected.transpose(2,0,1), rtol=4e-5, atol=4e-5)
                        before = result.copy()
                        for value in operands: value[...] = 0
                        newer = contract_p2p_bridge(*operands, ((gamma,coefficient,record),),
                            used_np=left.shape[2], backend=self.backend, same_point=True,
                            transpose_bridge=transpose, equation='aipc,jqdb->pab')
                        self.assertFalse(self.backend.shares_memory(actual, newer))
                        result = self.backend.asnumpy(actual) if self.backend is not np else actual
                        np.testing.assert_array_equal(result, before)


class RetainedSparseCPUChecks(_RetainedChecks, unittest.TestCase):
    def test_hip_backend_keeps_the_existing_accumulation_path(self):
        class HIPProxy:
            __name__ = "cupy"
            cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=True))

            def __getattr__(self, name):
                if name == "ElementwiseKernel":
                    raise AssertionError("HIP should retain the existing array backend path")
                if name == "asnumpy":
                    return np.asarray
                return getattr(np, name)

        self.backend = HIPProxy()
        self.check_case(transpose=True, equation="aipc,jqdb->pabq", two_labels=True)


@unittest.skipUnless(cp is not None and _DEVICE_COUNT > 0, "requires a working CUDA device")
class RetainedSparseGPUChecks(_RetainedChecks, unittest.TestCase):
    backend = cp

    def test_transient_no_search_backend_wrappers(self):
        from lqcdweave.contraction.execution.direct import _NoSearchBackend
        for _ in range(2):
            self.backend = _NoSearchBackend(cp)
            self.check_case()

    def test_complex128_fused_path_does_not_gather_endpoints(self):
        from lqcdweave.contraction.execution.direct import _NoSearchBackend

        class FusedBackend(_NoSearchBackend):
            def take(self, *args, **kwargs):
                raise AssertionError("fused retained bridge must load resident endpoints directly")

        self.backend = FusedBackend(cp)
        self.check_case(transpose=True, equation="aipc,jqdb->pabq", two_labels=True)

    def test_runtime_colors_and_types_do_not_gather_or_convert_endpoints(self):
        from lqcdweave.contraction.execution.direct import _NoSearchBackend
        class FusedBackend(_NoSearchBackend):
            def take(self, *args, **kwargs):
                raise AssertionError('fused bridge must not gather endpoints')
            def einsum(self, *arguments, **kwargs):
                if isinstance(arguments[0], str):
                    raise AssertionError('fused retained bridge must use integer-axis einsum')
                operand_count = (len(arguments) - 1) // 2
                if operand_count == 4:
                    raise AssertionError('fused bridge must not form a contribution tensor')
                return cp.einsum(*arguments, **kwargs)
        self.backend = FusedBackend(cp)
        self.test_runtime_colors_independent_input_precisions_and_lifetime()
