"""Independent dense oracle and owned result tests for propagation."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.production import PerambulatorGenerator
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis


class Handle:
    def __init__(self, data, **attributes):
        self.data = data
        self.__dict__.update(attributes)
        self.keys = []

    def load(self, key):
        self.keys.append(key)
        return self.data


class BlockOnlyHandle:
    """Reuses one read buffer and forbids whole-basis loading."""

    def __init__(self, data, max_block):
        self.data = data
        self.Ne = data.shape[1]
        self.basis_id = "explicit-block-basis"
        self.scratch = np.empty((max_block, *data.shape[2:]), dtype=data.dtype)
        self.reads = []

    def load(self, key):
        raise AssertionError("whole-basis load is forbidden")

    def read_modes(self, key, time, start, stop):
        self.reads.append((key, time, start, stop))
        self.scratch[:] = np.nan
        self.scratch[:stop-start] = self.data[time, start:stop]
        return self.scratch[:stop-start]


class CountingBackend(DenseDiracBackend):
    def __init__(self):
        super().__init__()
        self.loads = 0
        self.calls = 0
        self.rhs_count = 0

    def load(self, gauge, parameters):
        self.loads += 1
        super().load(gauge, parameters)

    def solve_into(self, rhs, solution, *, tol, maxiter):
        self.calls += 1
        self.rhs_count += len(rhs)
        super().solve_into(rhs, solution, tol=tol, maxiter=maxiter)

    def stout_smear(self, gauge, *, nstep, rho, dir_ignore):
        transform = getattr(gauge, "stoutSmear", None)
        if not callable(transform):
            raise NotImplementedError("test gauge does not expose stoutSmear")
        transform(nstep, rho, dir_ignore)
        return gauge


class PropagatorTests(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 1, 1, 2, 4, 3)
        rng = np.random.default_rng(83)
        self.matrix = 5*np.eye(48) + .1*(rng.normal(size=(48,48))+1j*rng.normal(size=(48,48)))
        self.vsrc = rng.normal(size=(2,2,1,1,2,3))+1j*rng.normal(size=(2,2,1,1,2,3))
        self.vsink = rng.normal(size=(2,3,1,1,2,3))+1j*rng.normal(size=(2,3,1,1,2,3))
        self.psrc = np.array([[[1,0,0]], [[0,0,0]]])
        self.psink = np.array([[[1,0,0],[0,0,0]],[[0,0,0],[1,0,0]]])

    def generator(self, *, src=None, vector_sink=None, point_sink=None, **options):
        self.src_basis = src or VectorBasis(Handle(self.vsrc, Ne=2, basis_id="source"))
        self.sink_v_basis = vector_sink or VectorBasis(Handle(self.vsink, Ne=3, basis_id="sink-v"))
        self.sink_p_basis = point_sink or PointBasis(Handle(self.psink, Np=2, basis_id="sink-p"))
        self.sink_bases = [self.sink_v_basis, self.sink_p_basis]
        kwargs = dict(latt_size=(2,1,1,2), gauge_field=Handle(self.matrix), mass=.2,
                      tol=1e-11, maxiter=1000, backend=CountingBackend())
        kwargs.update(options)
        return PerambulatorGenerator(**kwargs)

    def embedding(self, kind, source):
        v = self.vsrc if source else self.vsink
        points = self.psrc if source else self.psink
        size = v.shape[1] if kind=='V' else 3*points.shape[1]
        out = np.zeros((48, 2*4*size), complex)
        for t in range(2):
            for s in range(4):
                for k in range(size):
                    col = (t*4+s)*size+k
                    for x in range(2):
                        for c in range(3):
                            row = np.ravel_multi_index((t,0,0,x,s,c),self.shape)
                            if kind=='V':
                                out[row,col] = v[t,k,0,0,x,c]
                            else:
                                p, color = divmod(k,3)
                                out[row,col] = int(points[t,p,0]==x and color==c)
        return out

    def reference(self, product, t_src):
        snk = self.embedding(product[0],False)
        src = self.embedding(product[2],True)
        projected = snk.conj().T @ np.linalg.inv(self.matrix) @ src
        ns = self.vsrc.shape[1] if product[2]=='V' else 3*self.psrc.shape[1]
        nk = self.vsink.shape[1] if product[0]=='V' else 3*self.psink.shape[1]
        selected = projected[:,t_src*4*ns:(t_src+1)*4*ns].reshape(2,4,nk,4,ns).transpose(0,1,3,2,4)
        if product=='PSV':
            return selected.reshape(2,4,4,self.psink.shape[1],3,ns)
        if product=='PSP':
            return selected.reshape(2,4,4,self.psink.shape[1],3,self.psrc.shape[1],3)
        if product=='VSP':
            return selected.reshape(2,4,4,self.vsink.shape[1],self.psrc.shape[1],3)
        return selected

    def test_products_match_inverse_with_shared_solves_and_mrhs_remainder(self):
        for mrhs in (False,True):
            with self.subTest(mrhs=mrhs):
                gen = self.generator(MRHS=mrhs,rhs_batch_size=5,sink_ne_block_size=2)
                gen.load('cfg')
                for t in range(2):
                    vector, point = gen.calc(t, src=self.src_basis, sink=self.sink_bases)
                    for result in (vector, point):
                        np.testing.assert_allclose(result.values, self.reference(result.product,t), atol=2e-14)
                self.assertEqual(gen.backend.loads,1)
                self.assertEqual(gen.backend.rhs_count,2*(2*4))
                self.assertEqual(gen.backend.calls,2*2 if mrhs else 16)
                self.assertTrue(all(x<1e-11 for x in gen.backend.last_relative_residuals))

                point_src = PointBasis(Handle(self.psrc, Np=1, basis_id="source-p"))
                for t in range(2):
                    vector, point = gen.calc(t, src=point_src, sink=self.sink_bases)
                    for result in (vector, point):
                        np.testing.assert_allclose(result.values, self.reference(result.product,t), atol=2e-14)
                self.assertEqual(gen.backend.rhs_count,2*(2*4+12))

    def test_streamed_modes_match_inverse_without_whole_basis_loading(self):
        for time_block in (1, 2):
            with self.subTest(time_block=time_block):
                src = BlockOnlyHandle(self.vsrc, 1)
                snk = BlockOnlyHandle(self.vsink, 2)
                gen = self.generator(src=VectorBasis(src), vector_sink=VectorBasis(snk),
                                     stream_eigenvectors=True, sink_ne_block_size=2,
                                     sink_time_block_size=time_block)
                gen.load('cfg')
                self.assertEqual(src.reads, [])
                self.assertEqual(snk.reads, [])
                for time in range(2):
                    result = gen.calc(time, src=self.src_basis, sink=[self.sink_v_basis])[0]
                    np.testing.assert_allclose(result.values,self.reference('VSV',time),atol=2e-14)
                self.assertTrue(src.reads and snk.reads)
                self.assertTrue(all(stop-start == 1 for _,_,start,stop in src.reads))
                self.assertTrue(all(stop-start <= 2 for _,_,start,stop in snk.reads))

    def test_streamed_time_block_tail_preserves_all_sink_times(self):
        class IdentityBackend:
            array_module = np
            empty = staticmethod(np.empty)
            zeros = staticmethod(np.zeros)
            def load(self, gauge, parameters):
                return None
            def solve_into(self, rhs, solution, *, tol, maxiter):
                solution[...] = rhs
        rng = np.random.default_rng(203)
        src = rng.normal(size=(3,2,1,1,2,3)) + 1j*rng.normal(size=(3,2,1,1,2,3))
        snk = rng.normal(size=(3,3,1,1,2,3)) + 1j*rng.normal(size=(3,3,1,1,2,3))
        src_basis = VectorBasis(BlockOnlyHandle(src,1))
        sink_basis = VectorBasis(BlockOnlyHandle(snk,2))
        gen = PerambulatorGenerator((2,1,1,3),Handle(None),.2,1e-10,10,
            backend=IdentityBackend(),stream_eigenvectors=True,sink_ne_block_size=2,
            sink_time_block_size=2)
        gen.load("cfg")
        for t_src in range(3):
            actual = gen.calc(t_src,src=src_basis,sink=[sink_basis])[0].values
            expected = np.zeros_like(actual)
            overlap = snk[t_src].reshape(3,-1).conj() @ src[t_src].reshape(2,-1).T
            for spin in range(4):
                expected[t_src,spin,spin] = overlap
            np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_streamed_sink_cache_is_bounded_and_reuses_records(self):
        for budget in (0,192,576):
            with self.subTest(cache_bytes=budget):
                src = BlockOnlyHandle(self.vsrc,1)
                snk = BlockOnlyHandle(self.vsink,2)
                gen = self.generator(src=VectorBasis(src),vector_sink=VectorBasis(snk),
                    stream_eigenvectors=True,sink_ne_block_size=2,
                    sink_time_block_size=2,sink_cache_limit_bytes=budget)
                gen.load('cfg')
                np.testing.assert_allclose(gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])[0].values,
                                           self.reference('VSV',0),atol=2e-14)
                self.assertLessEqual(len(snk.reads),32)
                self.assertLessEqual(gen.sink_cache_used_bytes,budget)
                np.testing.assert_allclose(gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])[0].values,
                                           self.reference('VSV',1),atol=2e-14)
                self.assertLessEqual(len(snk.reads),64)
                self.assertLessEqual(gen.sink_cache_used_bytes,budget)
                gen.load('cfg')
                self.assertEqual(gen.sink_cache_used_bytes,0)
                self.assertEqual(len(gen._sink_cache),0)
        with self.assertRaisesRegex(ValueError,'require stream_eigenvectors'):
            self.generator(sink_cache_limit_bytes=1)

    def test_streamed_read_failure_drops_cached_blocks(self):
        src = BlockOnlyHandle(self.vsrc,1)
        snk = BlockOnlyHandle(self.vsink,2)
        gen = self.generator(src=VectorBasis(src),vector_sink=VectorBasis(snk),
            stream_eigenvectors=True,sink_ne_block_size=2,sink_cache_limit_bytes=576)
        gen.load('cfg')
        gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertEqual(gen.sink_cache_used_bytes,576)
        with patch.object(src,'read_modes',side_effect=RuntimeError('source read failed')):
            with self.assertRaisesRegex(RuntimeError,'source read failed'):
                gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertEqual(gen.sink_cache_used_bytes,0)
        self.assertEqual(gen.last_metrics,{})

    def test_streamed_input_scratch_limit_is_checked_when_bases_are_staged(self):
        src = BlockOnlyHandle(self.vsrc, 1)
        snk = BlockOnlyHandle(self.vsink, 2)
        gen = self.generator(src=VectorBasis(src),vector_sink=VectorBasis(snk),
                             stream_eigenvectors=True,sink_ne_block_size=2,
                             vector_scratch_limit_bytes=288)
        gen.load('cfg')
        gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertEqual(gen.vector_scratch_bytes,288)

        too_small = self.generator(src=VectorBasis(BlockOnlyHandle(self.vsrc,1)),
            vector_sink=VectorBasis(BlockOnlyHandle(self.vsink,2)),stream_eigenvectors=True,
            sink_ne_block_size=2,vector_scratch_limit_bytes=287)
        too_small.load('cfg')
        with self.assertRaisesRegex(ValueError,"scratch exceeds"):
            too_small.calc(0,src=self.src_basis,sink=[self.sink_v_basis])
        with self.assertRaisesRegex(ValueError,"require stream_eigenvectors"):
            self.generator(vector_scratch_limit_bytes=288)

    def test_results_own_arrays_and_bind_the_actual_endpoints(self):
        gen=self.generator(contract_prec='<c8',sink_ne_block_size=2)
        gen.load('cfg')
        first=gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])[0]
        saved=first.values.copy()
        self.assertTrue(first.values.flags.owndata)
        self.assertEqual(first.source,gen.endpoint(self.src_basis,source=True))
        self.assertEqual(first.sink,gen.endpoint(self.sink_v_basis))
        self.assertEqual(first.at(1).sink_time,1)
        second=gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])[0]
        np.testing.assert_array_equal(first.values,saved)
        np.testing.assert_allclose(second.values,self.reference('VSV',1),rtol=2e-6,atol=2e-6)
        description=gen.describe()
        self.assertNotIn('products',description)
        self.assertNotIn('endpoints',description)

    def test_raw_solution_can_be_projected_without_a_second_solve(self):
        gen=self.generator()
        gen.load('cfg')
        solution=gen.calc(0,src=self.src_basis,sink=None)[0]
        self.assertTrue(solution.values.flags.owndata)
        before=gen.backend.rhs_count
        projected=solution.project(sink=self.sink_bases)
        self.assertEqual(gen.backend.rhs_count,before)
        for result in projected:
            np.testing.assert_allclose(result.values,self.reference(result.product,0),atol=2e-14)

    def test_count_precision_and_operator_description(self):
        basis=VectorBasis(Handle(self.vsrc,Ne=2,basis_id='basis'),count=1)
        gen=self.generator(src=basis,vector_sink=basis,contract_prec='<c8')
        gen.load('cfg')
        result=gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])[0]
        self.assertEqual(result.values.dtype,np.dtype('complex64'))
        self.assertEqual(gen._rhs.dtype,np.dtype('complex128'))
        expected=self.vsrc[:,:1].astype(np.complex64).astype(np.complex128)
        inverse=np.linalg.inv(self.matrix).reshape(*self.shape,*self.shape)
        value=0j
        for x in range(2):
            for c in range(3):
                for y in range(2):
                    for d in range(3):
                        value+=expected[0,0,0,0,x,c].conjugate()*inverse[0,0,0,x,2,c,1,0,0,y,3,d]*expected[1,0,0,0,y,d]
        np.testing.assert_allclose(result.values[0,2,3,0,0],value,rtol=2e-6,atol=1e-7)
        description=gen.describe()
        description['parameters']['mass']=999
        self.assertEqual(gen.describe()['parameters']['mass'],.2)

    def test_failed_solve_does_not_publish_metrics(self):
        gen=self.generator()
        gen.load('cfg')
        def fail(*args,**kwargs): raise RuntimeError('solver failed')
        gen.backend.solve_into=fail
        with self.assertRaisesRegex(RuntimeError,'solver failed'):
            gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertEqual(gen.last_metrics,{})

    def test_stout_marks_gauge_dirty_and_matches_independent_solve(self):
        class Gauge:
            def __init__(self,matrix): self.matrix=matrix.copy()
            def __array__(self,dtype=None,copy=None): return np.asarray(self.matrix,dtype=dtype)
            def stoutSmear(self,nstep,rho,dir_ignore): self.matrix*=2
        payload=Gauge(self.matrix)
        gen=self.generator(gauge_field=Handle(payload,gauge_id='gauge'))
        with self.assertRaises(RuntimeError): gen.stout_smear(1,.1)
        gen.load('cfg')
        before=gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])[0].values.copy()
        gen.stout_smear(1,.1)
        after=gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])[0].values
        np.testing.assert_allclose(after,before/2,atol=1e-14)
        self.assertEqual(gen.backend.loads,2)
        self.assertEqual(len(gen.describe()['gauge_transforms']),1)
        def fail(*args): raise RuntimeError('in-place failure')
        payload.stoutSmear=fail
        with self.assertRaises(RuntimeError): gen.stout_smear(1,.1)
        with self.assertRaises(RuntimeError): gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])

    def test_constructor_and_calc_validation(self):
        for kwargs in ({'latt_size':(2,1,0,2)},{'latt_size':(2.,1,1,2)},
                       {'rhs_batch_size':0},{'sink_ne_block_size':1.2},
                       {'tol':np.nan},{'t_boundary':0},{'contract_prec':'float32'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): self.generator(**kwargs)
        with self.assertRaises(ValueError): self.generator(backend=None)
        gen=self.generator(); gen.load('cfg')
        for t in (-1,2,True,1.5):
            with self.assertRaises((ValueError,KeyError)):
                gen.calc(t,src=self.src_basis,sink=[self.sink_v_basis])
        with self.assertRaises(TypeError): gen.calc(0)
        with self.assertRaises(TypeError): gen.calc(0,src=self.src_basis,sink=[])
        self.assertEqual(gen.backend.calls,0)
        self.assertEqual(gen.backend.loads,0)

    def test_dense_singular_and_dimension_guard(self):
        gen=self.generator(gauge_field=Handle(np.zeros((48,48))))
        gen.load('cfg')
        with self.assertRaises(np.linalg.LinAlgError):
            gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])
        gen=self.generator(backend=DenseDiracBackend(max_dimension=12))
        gen.load('cfg')
        with self.assertRaises(ValueError): gen.calc(0,src=self.src_basis,sink=[self.sink_v_basis])

    def test_constructor_signature_contains_only_operator_and_solver_options(self):
        signature=inspect.signature(PerambulatorGenerator)
        self.assertEqual(list(signature.parameters),[
            'latt_size','gauge_field','mass','tol','maxiter','xi_0','nu','clover_coeff_t',
            'clover_coeff_r','t_boundary','multigrid','contract_prec','backend','MRHS',
            'rhs_batch_size','stream_eigenvectors','sink_ne_block_size','sink_time_block_size',
            'sink_cache_limit_bytes','vector_scratch_limit_bytes','source_selection',
            'sink_selection','measure','gamma5','gamma5_hermitian'])
        self.assertEqual(signature.parameters['t_boundary'].default,1)
        self.assertEqual(signature.parameters['backend'].kind,inspect.Parameter.KEYWORD_ONLY)
        for name in ('source_selection', 'sink_selection'):
            self.assertIsNone(signature.parameters[name].default)
            self.assertEqual(signature.parameters[name].kind,inspect.Parameter.KEYWORD_ONLY)


if __name__=='__main__':
    unittest.main()
