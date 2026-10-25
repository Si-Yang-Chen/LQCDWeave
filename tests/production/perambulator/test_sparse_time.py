"""Selected physical solves/projections against independent full-lattice inverses."""
import itertools
import unittest
import numpy as np

from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.perambulator import PerambulatorGenerator, VectorBasis, PointBasis


class _Handle:
    def __init__(self, data, **metadata):
        self.data = data
        self.__dict__.update(metadata)
        self.loads = []
    def load(self, key):
        self.loads.append(key)
        return self.data


class _Stream(_Handle):
    def __init__(self, data, selection):
        super().__init__(data, Ne=data.shape[1], time_selection=selection)
        self.reads = []
        self.scratch = np.empty((2, *data.shape[2:]), complex)
    def load(self, key):
        raise AssertionError('streaming must not load a whole basis')
    def read_modes(self, key, time, start, stop):
        self.reads.append((key, time, start, stop))
        self.scratch.fill(np.nan)
        self.scratch[:stop-start] = self.data[self.time_selection.slot(time), start:stop]
        return self.scratch[:stop-start]


class _Backend(DenseDiracBackend):
    def __init__(self):
        super().__init__()
        self.rhs_shapes = []
        self.projection_shapes = []
    def solve_into(self, rhs, solution, **kwargs):
        self.rhs_shapes.append(rhs.shape)
        return super().solve_into(rhs, solution, **kwargs)


class SparsePhysicalPerambulatorTests(unittest.TestCase):
    def setUp(self):
        self.lattice = (2, 1, 1, 5)
        self.field_shape = (5, 1, 1, 2, 4, 3)
        rng = np.random.default_rng(671)
        self.matrix = 6*np.eye(120) + .03*(rng.normal(size=(120,120)) + 1j*rng.normal(size=(120,120)))
        self.inverse = np.linalg.inv(self.matrix)
        self.vsrc = rng.normal(size=(5,2,1,1,2,3)) + 1j*rng.normal(size=(5,2,1,1,2,3))
        self.vsnk = rng.normal(size=(5,3,1,1,2,3)) + 1j*rng.normal(size=(5,3,1,1,2,3))
        self.psrc = np.zeros((5,1,3), np.int32)
        self.psrc[:,0,0] = [1,0,1,0,1]
        self.psnk = np.zeros((5,2,3), np.int32)
        self.psnk[:,0,0] = [0,1,0,1,0]
        self.psnk[:,1,0] = 1-self.psnk[:,0,0]
        self.source = TimeSelectionRecord(5, (3,1))
        self.sink = TimeSelectionRecord(5, (4,0,2))

    def _embedding(self, kind, source):
        vectors = self.vsrc if source else self.vsnk
        points = self.psrc if source else self.psnk
        count = vectors.shape[1] if kind == 'V' else 3*points.shape[1]
        result = np.zeros((120, 5*4*count), complex)
        for t,s,k,x,c in itertools.product(range(5),range(4),range(count),range(2),range(3)):
            value = (vectors[t,k,0,0,x,c] if kind == 'V' else
                     int(points[t,k//3,0] == x and k%3 == c))
            row = np.ravel_multi_index((t,0,0,x,s,c), self.field_shape)
            result[row,(t*4+s)*count+k] = value
        return result

    def _reference(self, product, source_time):
        left, right = self._embedding(product[0],False), self._embedding(product[2],True)
        matrix = left.conj().T @ self.inverse @ right
        ns, nk = right.shape[1]//20, left.shape[1]//20
        full = matrix[:,source_time*4*ns:(source_time+1)*4*ns].reshape(5,4,nk,4,ns).transpose(0,1,3,2,4)
        if product == 'PSV': full = full.reshape(5,4,4,2,3,2)
        if product == 'VSP': full = full.reshape(5,4,4,3,1,3)
        if product == 'PSP': full = full.reshape(5,4,4,2,3,1,3)
        return full[list(self.sink.times)]

    def _generator(self, stream=False, compact=True, source_handle=None,
                   vector_sink_handle=None, sink_vector_count=None,
                   source_selection=None, sink_selection=None, **options):
        src_map = TimeSelectionRecord(5,(1,3)) if compact else TimeSelectionRecord(5,tuple(range(5)))
        snk_map = TimeSelectionRecord(5,(2,4,0)) if compact else TimeSelectionRecord(5,tuple(range(5)))
        def basis(values, selection):
            data = values[list(selection.times)]
            return (_Stream(data,selection) if stream else
                    _Handle(data,Ne=data.shape[1],time_selection=selection))
        source_handle = source_handle or basis(self.vsrc,src_map)
        vector_sink_handle = vector_sink_handle or basis(self.vsnk,snk_map)
        self.src_basis = VectorBasis(source_handle)
        self.point_src_basis = PointBasis(_Handle(
            self.psrc[list(src_map.times)],Np=self.psrc.shape[1],time_selection=src_map))
        self.sink_v_basis = VectorBasis(vector_sink_handle,count=sink_vector_count)
        self.sink_p_basis = PointBasis(_Handle(
            self.psnk[list(snk_map.times)],Np=self.psnk.shape[1],time_selection=snk_map))
        self.sink_bases = [self.sink_v_basis,self.sink_p_basis]
        kwargs = dict(latt_size=self.lattice, gauge_field=_Handle(self.matrix),
            mass=.2,tol=1e-11,maxiter=1000,backend=_Backend(),
            stream_eigenvectors=stream,
            source_selection=self.source if source_selection is None else source_selection,
            sink_selection=self.sink if sink_selection is None else sink_selection,
            MRHS=True,rhs_batch_size=5,sink_ne_block_size=2,
            sink_time_block_size=2 if stream else 1,sink_cache_limit_bytes=384 if stream else 0)
        kwargs.update(options)
        return PerambulatorGenerator(**kwargs)

    def test_all_products_compact_and_full_inputs_match_full_physical_reference(self):
        references = {(p,t):self._reference(p,t) for p in ('VSV','PSV','VSP','PSP') for t in self.source.times}
        for stream,compact in itertools.product((False,True),repeat=2):
            with self.subTest(stream=stream,compact=compact):
                gen = self._generator(stream,compact)
                gen.load('cfg')
                for time in self.source.times:
                    vector_results = gen.calc(time,src=self.src_basis,sink=self.sink_bases)
                    point_results = gen.calc(time,src=self.point_src_basis,sink=self.sink_bases)
                    for result in (*vector_results,*point_results):
                        np.testing.assert_allclose(result.values,references[result.product,time],rtol=2e-13,atol=3e-14)
                        self.assertEqual(result.values.shape[0],3)
                self.assertTrue(all(shape[1] == 5 for shape in gen.backend.rhs_shapes))
                self.assertEqual(sum(shape[0] for shape in gen.backend.rhs_shapes),40)
                self.assertEqual(gen._rhs.shape[1],5)
                self.assertEqual(gen._solution.shape[1],5)
                self.assertEqual(gen.sink_times,(4,0,2))
                self.assertEqual(gen.source_selection.document,self.source.document)
                if stream:
                    self.assertEqual({t for _,t,_,_ in self.sink_v_basis.handle.reads},{4,0,2})
                    self.assertEqual({t for _,t,_,_ in self.src_basis.handle.reads},{3,1})
                    self.assertLessEqual(gen.sink_cache_used_bytes,384)

    def test_missing_coverage_and_unselected_source_fail_before_solve(self):
        gen = self._generator()
        gen.load('cfg')
        with self.assertRaises((ValueError,KeyError)):
            gen.calc(4,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertFalse(gen.backend.rhs_shapes)
        self.src_basis.handle.time_selection = TimeSelectionRecord(5,(1,2))
        with self.assertRaises((ValueError,KeyError)):
            gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])
        self.assertTrue(gen.describe()['loaded'])
        self.assertFalse(gen.backend.rhs_shapes)
        gen = self._generator(source_handle=_Handle(self.vsrc[[3,1]],Ne=2))
        gen.load('cfg')
        with self.assertRaisesRegex(ValueError,'time shape'):
            gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])
        gen = self._generator(vector_sink_handle=_Handle(self.vsnk[[4,0,2]],Ne=3,
                                                    time_selection=TimeSelectionRecord(6,(4,0,2))))
        gen.load('cfg')
        with self.assertRaises(ValueError):
            gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])

    def test_only_selected_eager_records_are_read_and_projected(self):
        class Recorded:
            def __init__(self,array):
                self.array,self.shape,self.dtype = array,array.shape,array.dtype
                self.reads = []
            def __getitem__(self,key):
                self.reads.append(key[0])
                return self.array[key]
        source = Recorded(self.vsrc.copy())
        sink = Recorded(self.vsnk.copy())
        source.array[[0,2,4]] = np.nan
        sink.array[[1,3]] = np.nan
        gen = self._generator(source_handle=_Handle(source,Ne=2),
                              vector_sink_handle=_Handle(sink,Ne=3))
        gen.load('cfg')
        actual=gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])[0]
        np.testing.assert_allclose(actual.values, self._reference('VSV',3),atol=3e-14)
        self.assertEqual(source.reads,[3,3,1,1])
        self.assertEqual(sink.reads,[4,4,4,0,0,0,2,2,2])
        # The contraction input time extent is compact: a full-time projection
        # followed by output cropping cannot satisfy this instrumentation.
        np.testing.assert_allclose(gen.calc(1,src=self.src_basis,sink=[self.sink_v_basis])[0].values,
                                   self._reference('VSV',1),atol=3e-14)


    def test_shared_basis_with_distinct_endpoints_and_small_sink_workspace(self):
        shared = _Handle(self.vsrc,Ne=2)
        gen = self._generator(source_handle=shared,vector_sink_handle=shared,sink_vector_count=2)
        gen.load('cfg')
        result=gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])[0]
        self.assertEqual(shared.loads,['cfg','cfg'])
        saved = self.vsnk
        self.vsnk = self.vsrc
        try:
            expected = self._reference('VSV',3)
        finally:
            self.vsnk = saved
        np.testing.assert_allclose(result.values,expected,atol=3e-14)
        gen = self._generator(stream=True,sink_selection=TimeSelectionRecord(5,(4,)),
                              sink_time_block_size=4)
        gen.load('cfg')
        result=gen.calc(3,src=self.src_basis,sink=[self.sink_v_basis])[0]
        # One source vector and a two-mode sink block need three scratch modes.
        self.assertEqual(gen.vector_scratch_bytes,3*2*3*16)
        self.assertEqual(result.values.shape[0],1)
