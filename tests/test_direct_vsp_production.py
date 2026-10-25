"""Direct sink-vector/source-point projection against a dense inverse oracle."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.production import PerambulatorGenerator
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis


class _Handle:
    def __init__(self, data, **metadata):
        self.data = data
        self.__dict__.update(metadata)

    def load(self, key):
        return self.data


class _BlockHandle(_Handle):
    def __init__(self, data, max_block):
        super().__init__(data,Ne=data.shape[1],basis_id="sink-basis-actual")
        self.scratch = np.empty((max_block,*data.shape[2:]),dtype=data.dtype)
        self.reads=[]

    def load(self, key):
        raise AssertionError("streamed VSP must not load the whole sink basis")

    def read_modes(self, key, time, start, stop):
        self.reads.append((time,start,stop))
        self.scratch[:stop-start]=self.data[time,start:stop]
        return self.scratch[:stop-start]


class _InverseBackend:
    array_module = np

    def __init__(self, matrix):
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        self.solve_sizes = []
        self.syncs = 0

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype)

    def load(self, gauge, parameters):
        np.testing.assert_array_equal(np.asarray(gauge), self.matrix)

    def solve_into(self, rhs, solution, *, tol, maxiter):
        self.solve_sizes.append(len(rhs))
        solution[...] = (self.inverse @ np.asarray(rhs).reshape(len(rhs),-1).T).T.reshape(rhs.shape)

    def synchronize(self):
        self.syncs += 1


class DirectVSPProductionTests(unittest.TestCase):
    def setUp(self):
        self.lattice=(2,1,1,3)
        self.field_shape=(3,1,1,2,4,3)
        self.dimension=int(np.prod(self.field_shape))
        rng=np.random.default_rng(419)
        self.matrix=5.0*np.eye(self.dimension)+.08*(rng.normal(size=(self.dimension,self.dimension))
            +1j*rng.normal(size=(self.dimension,self.dimension)))
        self.sink_modes=rng.normal(size=(3,3,1,1,2,3))+1j*rng.normal(size=(3,3,1,1,2,3))
        self.source_points=np.asarray([
            [[1,0,0],[0,0,0]],
            [[0,0,0],[1,0,0]],
            [[1,0,0],[0,0,0]],
        ],dtype=np.int64)
        self.sink_points=np.asarray([[[0,0,0]],[[1,0,0]],[[0,0,0]]],dtype=np.int64)

    def _generator(self, *, streamed=False, batch_size=5, time_block=1, cache_bytes=0):
        sink_handle=(_BlockHandle(self.sink_modes,2) if streamed else
                     _Handle(self.sink_modes,Ne=3,basis_id="sink-basis-actual"))
        source_basis=PointBasis(_Handle(self.source_points,Np=2))
        sink_vector=VectorBasis(sink_handle,count=3)
        sink_point=PointBasis(_Handle(self.sink_points,Np=1),count=1)
        backend=_InverseBackend(self.matrix)
        generator=PerambulatorGenerator(self.lattice,_Handle(self.matrix),mass=.3,tol=1e-12,
            maxiter=80,backend=backend,MRHS=True,rhs_batch_size=batch_size,
            sink_ne_block_size=2,stream_eigenvectors=streamed,
            sink_time_block_size=time_block,sink_cache_limit_bytes=cache_bytes)
        return generator,source_basis,sink_vector,sink_point,backend,sink_handle

    def _source_rhs(self, source_time, point, spin, color):
        field=np.zeros(self.field_shape,dtype=np.complex128)
        x,y,z=self.source_points[source_time,point]
        field[source_time,z,y,x,spin,color]=1
        return field.reshape(-1)

    def _sink_vector(self, sink_time, spin, mode):
        field=np.zeros(self.field_shape,dtype=np.complex128)
        field[sink_time,...,spin,:]=self.sink_modes[sink_time,mode]
        return field.reshape(-1)

    def _independent_vsp(self, source_time):
        inverse=np.linalg.inv(self.matrix)
        result=np.empty((3,4,4,3,2,3),dtype=np.complex128)
        for sink_time in range(3):
            for sink_spin in range(4):
                for source_spin in range(4):
                    for mode in range(3):
                        sink=self._sink_vector(sink_time,sink_spin,mode)
                        for point in range(2):
                            for color in range(3):
                                rhs=self._source_rhs(source_time,point,source_spin,color)
                                result[sink_time,sink_spin,source_spin,mode,point,color]=np.vdot(sink,inverse@rhs)
        return result

    def _independent_psp(self, source_time):
        inverse=np.linalg.inv(self.matrix)
        result=np.empty((3,4,4,1,3,2,3),dtype=np.complex128)
        for sink_time in range(3):
            for sink_spin in range(4):
                for sink_point,(x,y,z) in enumerate(self.sink_points[sink_time]):
                    for source_spin in range(4):
                        for source_point in range(2):
                            for source_color in range(3):
                                rhs=self._source_rhs(source_time,source_point,source_spin,source_color)
                                solution=(inverse@rhs).reshape(self.field_shape)
                                result[sink_time,sink_spin,source_spin,sink_point,:,source_point,source_color]=\
                                    solution[sink_time,z,y,x,sink_spin,:]
        return result

    def test_vsp_uses_only_the_explicit_sink_vector_basis(self):
        generator,source,sink_vector,_,backend,_=self._generator()
        generator.load("cfg")
        result=generator.calc(1,src=source,sink=[sink_vector])[0]
        self.assertEqual(result.product,"VSP")
        self.assertEqual(result.values.shape,(3,4,4,3,2,3))
        np.testing.assert_allclose(result.values,self._independent_vsp(1),atol=2e-14,rtol=2e-14)
        self.assertEqual(sum(backend.solve_sizes),24)
        self.assertEqual(result.source.kind,'P')
        self.assertEqual(result.sink.kind,'V')

    def test_vsp_and_psp_share_one_point_source_solve_loop(self):
        generator,source,sink_vector,sink_point,backend,_=self._generator()
        generator.load("cfg")
        vsp,psp=generator.calc(0,src=source,sink=[sink_vector,sink_point])
        self.assertEqual((vsp.product,psp.product),('VSP','PSP'))
        np.testing.assert_allclose(vsp.values,self._independent_vsp(0),atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(psp.values,self._independent_psp(0),atol=2e-14,rtol=2e-14)
        self.assertEqual(sum(backend.solve_sizes),24)

    def test_streamed_tail_blocks_preserve_all_physical_times(self):
        for streamed in (False,True):
            with self.subTest(streamed=streamed):
                generator,source,sink_vector,_,_,sink_handle=self._generator(
                    streamed=streamed,batch_size=7,time_block=2 if streamed else 1)
                generator.load("cfg")
                actual=generator.calc(2,src=source,sink=[sink_vector])[0]
                np.testing.assert_allclose(actual.values,self._independent_vsp(2),atol=3e-14,rtol=3e-14)
                if streamed:
                    self.assertEqual(set(sink_handle.reads),{
                        (time,start,stop) for time in range(3) for start,stop in ((0,2),(2,3))})
                    self.assertEqual(generator.vector_scratch_bytes,
                        2*1*1*2*3*np.dtype(generator.contract_prec).itemsize)

    def test_streamed_sink_cache_respects_its_memory_limit(self):
        generator,source,sink_vector,_,_,sink=self._generator(
            streamed=True,time_block=2,cache_bytes=1024)
        generator.load("cfg")
        result=generator.calc(0,src=source,sink=[sink_vector])[0]
        np.testing.assert_allclose(result.values,self._independent_vsp(0),atol=3e-14,rtol=3e-14)
        self.assertLessEqual(generator.sink_cache_used_bytes,1024)
        self.assertLessEqual(len(sink.reads),24)

    def test_owned_result_survives_later_calc_and_stream_failure(self):
        generator,source,sink_vector,_,_,sink=self._generator(streamed=True,cache_bytes=0)
        generator.load("cfg")
        first=generator.calc(0,src=source,sink=[sink_vector])[0]
        saved=first.values.copy()
        second=generator.calc(1,src=source,sink=[sink_vector])[0]
        self.assertTrue(first.values.flags.owndata)
        np.testing.assert_array_equal(first.values,saved)
        np.testing.assert_allclose(second.values,self._independent_vsp(1),atol=3e-14,rtol=3e-14)
        with patch.object(sink,"read_modes",side_effect=RuntimeError("sink read failed")):
            with self.assertRaisesRegex(RuntimeError,"sink read failed"):
                generator.calc(2,src=source,sink=[sink_vector])
        self.assertEqual(generator.last_metrics,{})
        np.testing.assert_array_equal(first.values,saved)


if __name__ == "__main__":
    unittest.main()
