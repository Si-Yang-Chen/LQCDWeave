"""Prebound reference reductions against independent scalar summation."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.correlator.api import TensorSpec, parse_axis_pattern
from lqcdweave.contracts.correlator.precision import minimum_contraction_dtype
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.contraction.execution.operations.reference_matrix import bind_reference_matrix
from correlator.execution.test_dense_atomic_kernel import _loop_reference

CASES=(('aij,jk->ki',((2,3,4),(4,5))),
       ('ij,ajk->ki',((3,4),(2,4,5))),
       ('aij,bjk->ik',((2,3,4),(2,4,5))),
       ('ikjl,jlmn->nmki',((2,3,4,5),(4,5,2,3))))

def fake():
    return SimpleNamespace(__name__='cupy',transpose=np.transpose,reshape=np.reshape,
                           matmul=np.matmul,dot=np.dot,einsum=np.einsum)

class ReferenceMatrixTests(unittest.TestCase):
    def test_uniform_inputs_and_single_sided_sums(self):
        rng=np.random.default_rng(833)
        for equation,shapes in CASES:
            for dtypes in ((np.complex128,np.complex128),(np.float64,np.complex128),
                           (np.float32,np.float64),(np.complex64,np.complex64)):
                with self.subTest(equation=equation,dtypes=dtypes):
                    arrays=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(dtype)
                        if np.dtype(dtype).kind=='c' else rng.normal(size=shape).astype(dtype)
                        for shape,dtype in zip(shapes,dtypes))
                    arrays=tuple(a[...,::-1] for a in arrays)
                    dtype=minimum_contraction_dtype(dtypes)
                    arrays=tuple(a.astype(dtype) for a in arrays)
                    catalog={k:TensorSpec(a.shape,a.dtype) for k,a in zip(('a','b'),arrays)}
                    action=SimpleNamespace(core=None,equation=parse_axis_pattern(equation),
                                           operands=('a','b'),wide_accumulation=False)
                    run=bind_dense_action(action,catalog,fake())
                    value,owned=run(arrays)
                    tolerance=3e-5 if dtype in (np.dtype('float32'),np.dtype('complex64')) else 2e-12
                    np.testing.assert_allclose(value,_loop_reference(equation,arrays),rtol=tolerance,atol=tolerance)
                    extension=bind_reference_matrix(parse_axis_pattern(equation),
                        tuple(catalog[k] for k in ('a','b')),fake())
                    if extension is not None:
                        self.assertTrue(owned)
                    saved=tuple(a.copy() for a in arrays)
                    value[...] = 0
                    for a,b in zip(arrays,saved): np.testing.assert_array_equal(a,b)

    def test_unsupported_and_workspace_insufficient_retain_reference(self):
        for equation,shapes,dtypes in (
            ('ii,ij->j',((3,3),(3,4)),(np.complex128,)*2),
            ('ij,jk->ik',((1,4),(4,3)),(np.float64,np.complex128)),
            ('ij,jk->ik',((2,40),(40,2)),(np.float64,np.complex128)),
            ('i,j->ij',((3,),(4,)),(np.complex128,)*2)):
            self.assertIsNone(bind_reference_matrix(parse_axis_pattern(equation),
                tuple(TensorSpec(s,d) for s,d in zip(shapes,dtypes)),fake()))

class ReferenceMatrixCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp=importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no CUDA device')
        except (ImportError,RuntimeError) as error:
            raise unittest.SkipTest(str(error))

    def test_actual_device_prebound_sums_casts_and_fresh_storage(self):
        cp=self.cp
        rng=np.random.default_rng(733)
        for equation,shapes in CASES:
            for dtypes in ((np.complex128,)*2,(np.float64,np.complex128)):
                host=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype(d)
                    if np.dtype(d).kind=='c' else rng.normal(size=s).astype(d)
                    for s,d in zip(shapes,dtypes))
                specs=tuple(TensorSpec(s,d) for s,d in zip(shapes,dtypes))
                run=bind_reference_matrix(parse_axis_pattern(equation),specs,cp)
                if run is None: continue  # Workspace fallback is separately tested.
                for scale in (1,3):
                    inputs=tuple(cp.asarray(a*scale)[...,::-1] for a in host)
                    expected=_loop_reference(equation,tuple((a*scale)[...,::-1] for a in host))
                    with patch.object(cp,'einsum',side_effect=AssertionError('reparsed equation')):
                        actual=run(*inputs)
                    np.testing.assert_allclose(cp.asnumpy(actual),expected,rtol=2e-12,atol=2e-12)
                    inputs[0][...] = 0
                    np.testing.assert_allclose(cp.asnumpy(actual),expected,rtol=2e-12,atol=2e-12)
