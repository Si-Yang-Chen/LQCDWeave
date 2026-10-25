"""Generated address arithmetic and wide accumulation; CUDA is a separate gate."""
import ctypes
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.reduction import bind_precise_reduction
from lqcdweave.contraction.execution.reduction_kernel import _kernel, bind_raw_reduction
from lqcdweave.contracts.correlator.api import parse_axis_pattern


class _SourceBackend:
    """Interpret emitted loads on the CPU, without claiming CUDA validation."""
    __name__ = 'cupy'
    einsum = staticmethod(np.einsum)
    empty = staticmethod(np.empty)

    def __init__(self):
        self.launches = 0
        self.sources = []
        self.fail_at = None
        self.synchronize = Mock()
        self.cuda = SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize))

    def RawKernel(self, source, name, **kwargs):
        self.sources.append(source)

        def kernel(grid, threads, args):
            self.launches += 1
            if self.launches == self.fail_at:
                raise RuntimeError('submission failed')
            if name in ('ed_precise_finish', 'ed_precise_matrix_finish'):
                partials, *constants, result = args
                blocks = int(constants[0]) if constants else int(re.search(r'i < (\d+)LL', source)[1])
                used = partials.reshape(-1)[:result.size * blocks].reshape(result.size, blocks)
                result[...] = used.sum(axis=1, dtype=partials.dtype).reshape(result.shape)
                return
            if name in ('ed_precise_matrix_partial', 'ed_precise_matrix_tile_partial'):
                tile = name == 'ed_precise_matrix_tile_partial'
                if tile:
                    a, b, k, m, n, blocks, lm, lk, rk, rn, partials = args
                    assert grid == (blocks, (n+1)//2, (m+1)//2)
                else:
                    a, b, k, n, blocks, lm, lk, rk, rn, partials = args
                    m = grid[2]
                    assert grid == (blocks, n, m)
                partials = partials.reshape(-1)[:m*n*blocks].reshape(m*n,blocks)
                offsets = re.findall(r'p([01]) \+ ([^)]*)\)\)', source)
                assert len(offsets) == (4 if tile else 2)
                expressions = [compile(expr, '<matrix CUDA offset>', 'eval') for _,expr in offsets]
                for row in range(m):
                    for col in range(n):
                        selected = ((expressions[row%2], expressions[2+col%2]) if tile else expressions)
                        base_row, base_col = (row//2*2, col//2*2) if tile else (row, col)
                        for split in range(blocks):
                            terms = []
                            for start in range(split*threads[0], k, blocks*threads[0]):
                                for i in range(start,min(k,start+threads[0])):
                                    term = partials.dtype.type(1)
                                    for value,expr in zip((a,b),selected,strict=True):
                                        offset=eval(expr,{'__builtins__':{}},dict(row=base_row,col=base_col,i=i,lm=int(lm),lk=int(lk),rk=int(rk),rn=int(rn)))
                                        raw=ctypes.string_at(value.ctypes.data+offset,value.itemsize)
                                        term*=partials.dtype.type(np.frombuffer(raw,dtype=value.dtype)[0])
                                    terms.append(term)
                            partials[row*n+col,split]=np.sum(terms,dtype=partials.dtype)
                return
            *values, partials = args
            size = int(re.search(r'i < (\d+)LL', source)[1])
            blocks = int(re.search(r'blockIdx.x / (\d+)LL', source)[1])
            outputs = grid[0]//blocks
            partials = partials.reshape(-1)[:outputs*blocks].reshape(outputs,blocks)
            offsets = re.findall(r'p\d+ \+ \((.*?)\)\)\);', source)
            self_test = len(offsets) == len(values)
            if not self_test:
                raise AssertionError('unrecognized source loads')
            expressions = [compile(expr.replace('LL', '').replace('/', '//'), '<CUDA offset>', 'eval')
                           for expr in offsets]
            for o in range(partials.shape[0]):
                for b in range(blocks):
                    terms = []
                    for i in range(b * threads[0], size, blocks * threads[0]):
                        for index in range(i, min(size, i + threads[0])):
                            term = partials.dtype.type(1)
                            for value, expr in zip(values, expressions, strict=True):
                                offset = eval(expr, {'__builtins__': {}}, {'o': o, 'i': index})
                                raw = ctypes.string_at(value.ctypes.data + offset, value.itemsize)
                                term *= partials.dtype.type(np.frombuffer(raw, dtype=value.dtype)[0])
                            terms.append(term)
                    partials[o, b] = np.sum(terms, dtype=partials.dtype)
        return kernel


class RawPreciseReductionTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_double_sources_preserve_input_mantissa_and_output_precision(self):
        for dtype in (np.float64,np.complex128):
            a=np.full((2,513),1.+2.**-40,dtype=dtype)
            b=np.ones_like(a)
            if dtype==np.complex128: a+=1j*(1.+2.**-39);b-=.25j
            a=a[:,::-1]
            backend=_SourceBackend()
            allowance=a.nbytes+b.nbytes+2*a.itemsize
            run=bind_raw_reduction(parse_axis_pattern('bk,bk->b'),(a.shape,b.shape),a.dtype,backend,allowance)
            value=run(a,b)
            reference=np.einsum('bk,bk->b',a,b)
            np.testing.assert_allclose(value,reference,rtol=1e-14,atol=1e-13)
            self.assertEqual(value.dtype,np.dtype(dtype))
            self.assertLessEqual(run.workspace_bytes,allowance)
            self.assertNotIn('complex<float>', ''.join(backend.sources))
            self.assertNotIn('float(total_', ''.join(backend.sources))

    def test_emitted_offsets_cover_negative_strides_diagonals_free_axes_and_fresh_inputs(self):
        rng = np.random.default_rng(18)
        for dtype in (np.float32, np.complex64):
            a = rng.normal(size=(5, 5, 49)).astype(dtype)[::-1, ::-1, ::-1]
            b = rng.normal(size=(3, 49, 5)).astype(dtype)[:, ::-1, :]
            if np.dtype(dtype).kind == 'c':
                a = a + 1j * a
                b = b - .5j * b
                a, b = a[::-1], b[:, ::-1]
            backend = _SourceBackend()
            equation = 'iik,oki->o'
            run = bind_precise_reduction(equation, (a.shape, b.shape), (a.dtype, b.dtype), backend)
            self.assertTrue(run.raw_reduction)
            self.assertLessEqual(run.workspace_bytes, run.workspace_allowance)
            wide = np.complex128 if np.dtype(dtype).kind == 'c' else np.float64
            for left, right in ((a, b), (2 * a, b), (a.copy(), b.copy())):
                reference = np.einsum(equation, left, right, dtype=wide).astype(dtype)
                np.testing.assert_allclose(run(left, right), reference, rtol=1e-6, atol=1e-6)
            self.assertEqual(backend.launches, 6)
            with self.assertRaisesRegex(ValueError, 'bound geometry'):
                run(a.astype(wide), b)

    def test_wide_product_survives_cancellation_and_failure_synchronizes(self):
        a = np.tile(np.array([128.125, 128.125, -128.125, -128.125], np.float32), 200)
        a[0] = np.nextafter(a[0], np.float32(np.inf))
        b = np.full(a.shape, 1.125, np.float32)
        backend = _SourceBackend()
        run = bind_precise_reduction('i,i->', (a.shape, b.shape), (a.dtype, b.dtype), backend)
        self.assertEqual(run(a, b), np.einsum('i,i->', a, b, dtype=np.float64).astype(np.float32))
        for relative in (1, 2):
            backend.fail_at = backend.launches + relative
            with self.assertRaisesRegex(RuntimeError, 'submission failed'):
                run(a, b)
        self.assertEqual(backend.synchronize.call_count, 2)
        backend.fail_at = None
        self.assertEqual(run(a, b).dtype, np.dtype(np.float32))

    def test_debug_uses_original_bounded_tiled_oracle(self):
        backend = _SourceBackend()
        with execution_debug(True):
            run = bind_precise_reduction('i,i->', ((1024,),)*2, (np.complex64,)*2, backend)
        self.assertFalse(getattr(run, 'raw_reduction', False))
        self.assertGreater(run.tile_count, 1)
        self.assertEqual(backend.sources, [])


class RawPreciseReductionCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.cp = cp

    def test_cuda_compilation_wide_cancellation_diagonal_and_noncontiguous_inputs(self):
        cp = self.cp
        rng = np.random.default_rng(28)
        for dtype in (np.float32, np.complex64):
            for equation, shapes in (('i,i->', ((800,),)*2),
                                     ('iik,oki->o', ((5, 5, 49), (3, 49, 5))),
                                     ('iijj->', ((4, 4, 50, 50),))):
                host = [rng.normal(size=shape).astype(dtype) for shape in shapes]
                if len(host) == 2 and len(shapes[0]) == 1:
                    host[0] = np.tile(np.array([128.125, 128.125, -128.125, -128.125], dtype), 200)
                    host[0][0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
                    host[1].fill(1.125)
                values = [cp.asarray(value)[..., ::-1] for value in host]
                run = bind_precise_reduction(equation, tuple(v.shape for v in values),
                                             tuple(v.dtype for v in values), cp)
                self.assertTrue(run.raw_reduction)
                reference = np.einsum(equation, *(v[..., ::-1] for v in host),
                    dtype=np.complex128 if np.dtype(dtype).kind == 'c' else np.float64).astype(dtype)
                np.testing.assert_allclose(cp.asnumpy(run(*values)), reference, rtol=1e-6, atol=1e-6)

    def test_cuda_warp_reduction_handles_partial_warps_and_multiple_partial_loads(self):
        cp = self.cp
        rng = np.random.default_rng(78)
        # The last input warp may be incomplete. Large reductions also make
        # each thread of the finish kernel load more than one partial sum.
        for size in (257, 513, 134400, 230400):
            for dtype in (np.float32, np.complex64):
                host = [rng.normal(size=size).astype(dtype) for _ in range(2)]
                if dtype == np.complex64:
                    host = [(v + 1j * rng.normal(size=size).astype(np.float32))
                            for v in host]
                wide = np.complex128 if dtype == np.complex64 else np.float64
                reference = np.einsum('i,i->', *host, dtype=wide).astype(dtype)
                values = tuple(cp.asarray(v) for v in host)
                run = bind_precise_reduction('i,i->', (host[0].shape,) * 2,
                                             (np.dtype(dtype),) * 2, cp)
                self.assertTrue(run.raw_reduction)
                np.testing.assert_allclose(cp.asnumpy(run(*values)), reference,
                                           rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
