"""Module tests split from test_baryon_elemental.py; shared numerical fixtures stay unchanged."""

from test_baryon_elemental import (
    BaryonElementalGenerator,
    BorrowedModes,
    Handle,
    _BaryonProductionFixture,
    determinant_reference,
    generate_baryon_elemental,
    np,
    patch,
    project_su3,
    reference_matrix,
    stout_smear,
    tracemalloc,
)
from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator


class BaryonProductionTests(_BaryonProductionFixture):
    def test_paths_all_legs_against_determinants_all_times(self):
        gen = self.generator()
        for t in range(self.sizes[3]):
            actual = gen.calc_path_blocks(t)["vvv"]
            for pi,paths in enumerate(self.paths):
                for mi,mom in enumerate(self.momenta):
                    np.testing.assert_allclose(actual[pi,mi], determinant_reference(self.u,self.vectors,paths,t,mom), atol=2e-12)

    def test_streaming_tiles_and_borrowed_output(self):
        gen = self.generator()
        expected = gen.calc_path_blocks(0)["vvv"].copy()
        buffer = gen.calc_path_blocks(1)["vvv"]
        self.assertIs(buffer,gen.calc_path_blocks(0)["vvv"])
        np.testing.assert_allclose(buffer,expected)
        stream = self.generator(cache_bytes=0)
        assembled = np.zeros_like(expected)
        def write(pi,mi,sector,slices,value):
            self.assertEqual(sector,"vvv")
            self.assertTrue(all(n<=b for n,b in zip(value.shape,stream.ne_block_size)))
            assembled[(pi,mi,*slices)] = value
        self.assertIsNone(stream.calc_path_blocks(0,write=write))
        self.assertNotIn("vvv",stream._workspace._buffers)
        np.testing.assert_allclose(assembled,expected)

    def test_complex_projection_cancellation_and_streaming(self):
        gen = self.generator()
        raw = gen.calc_path_blocks(0)["vvv"].copy()
        rows = [{"id":0,"terms":[{"paths":self.paths[0],"coefficient":[.2,.7]},
                                  {"paths":self.paths[1],"coefficient":[-.4,.1]}]},
                {"id":1,"terms":[{"paths":self.paths[2],"coefficient":[1,0]},
                                  {"paths":self.paths[2],"coefficient":[-1,0]}]}]
        expected = np.stack(((.2+.7j)*raw[0]+(-.4+.1j)*raw[1],np.zeros_like(raw[0])))
        np.testing.assert_allclose(gen.calc_path_blocks(0,rows=rows)["vvv"],expected,atol=1e-12)
        assembled = np.zeros_like(expected)
        gen.calc_path_blocks(0,rows=rows,write=lambda ri,mi,se,sl,v: assembled.__setitem__((ri,mi,*sl),v))
        np.testing.assert_allclose(assembled,expected,atol=1e-12)

    def test_distance_recipe_and_local_limit(self):
        gen = self.generator(paths=[((),(),())])
        result = gen.calc_path_blocks(0,distances=[(0,0,0),(1,0,2)])["vvv"]
        transformed = []
        for v,d in zip(self.vectors,(1,0,2)):
            value = v.copy()
            if d:
                shift = sum(reference_matrix(self.u,(axis,)*d,0,phase="none") for axis in range(6))/6
                value[0] = (shift @ v[0].reshape(v.shape[1],-1).T).T.reshape(v[0].shape)
            transformed.append(value)
        for mi,mom in enumerate(self.momenta):
            np.testing.assert_allclose(result[1,mi],determinant_reference(self.u,transformed,((),(),()),0,mom),atol=1e-12)
        np.testing.assert_allclose(result[0],gen.calc_path_blocks(0)["vvv"][0])

    def test_gauge_covariance_and_slot_antisymmetry(self):
        rng = np.random.default_rng(85)
        raw = rng.normal(size=(*self.u.shape[:4],3,3))+1j*rng.normal(size=(*self.u.shape[:4],3,3))
        omega = np.linalg.qr(raw)[0]
        omega[..., :, 0] /= np.linalg.det(omega)[...,None]
        gauge = self.u.copy()
        for direction,axis in enumerate((3,2,1,0)):
            end = np.roll(omega,-1,axis=axis)
            gauge[...,direction,:,:] = omega @ self.u[...,direction,:,:] @ end.conj().swapaxes(-1,-2)
        vec = [np.einsum("tzyxab,tezyxb->tezyxa",omega,v) for v in self.vectors]
        actual = BaryonElementalGenerator(self.sizes,Handle(gauge),tuple(Handle(v) for v in vec),
            paths=self.paths,momentum_list=self.momenta,ne_block_size=2).load("c").calc_path_blocks(1)["vvv"]
        expected = self.generator().calc_path_blocks(1)["vvv"].copy()
        np.testing.assert_allclose(actual,expected,atol=3e-12)
        swapped = BaryonElementalGenerator(self.sizes,Handle(self.u),
            tuple(Handle(self.vectors[i]) for i in (1,0,2)),
            paths=[(p[1],p[0],p[2]) for p in self.paths],momentum_list=self.momenta).load("c").calc_path_blocks(1)["vvv"]
        np.testing.assert_allclose(swapped,-expected.swapaxes(-3,-2),atol=3e-12)

    def test_shared_scratch_inputs_and_reload(self):
        scratch = np.empty_like(self.vectors[1])
        class Scratch:
            def __init__(self, value): self.value = value
            def load(self,key):
                scratch[:] = self.value*(2 if key=="other" else 1)
                return scratch
        handles = tuple(Scratch(self.vectors[1]*(i+1)) for i in range(3))
        gen = BaryonElementalGenerator(self.sizes,Handle(self.u),handles,usedNe=(1,2,3)).load("c")
        expected = determinant_reference(self.u,[self.vectors[1][:,:n]*(i+1) for i,n in enumerate((1,2,3))],((),(),()),0,(0,0,0))
        before = gen.calc_path_blocks(0)["vvv"].copy()
        np.testing.assert_allclose(before[0,0],expected,atol=1e-12)
        gen.load("other")
        np.testing.assert_allclose(gen.calc_path_blocks(0)["vvv"],8*before,atol=1e-11)
        shared = BaryonElementalGenerator(self.sizes,Handle(self.u),Handle(self.vectors[1]),usedNe=(1,2,3)).load("c")
        self.assertIs(shared._vectors[0],shared._vectors[2])

    def test_streamed_three_legs_own_borrowed_blocks(self):
        shared = BorrowedModes(self.vectors[1],capacity=2)
        streamed = BaryonElementalGenerator(self.sizes,Handle(self.u),shared,
            usedNe=(1,2,1),paths=self.paths,momentum_list=self.momenta,
            ne_block_size=(1,2,1),spatial_block_size=5,
            stream_eigenvectors=True,cache_bytes=64*1024**2).load("c")
        reference = BaryonElementalGenerator(self.sizes,Handle(self.u),Handle(self.vectors[1]),
            usedNe=(1,2,1),paths=self.paths,momentum_list=self.momenta,
            ne_block_size=(1,2,1),spatial_block_size=5).load("c")
        self.assertTrue(all(handle is shared for handle in streamed._vectors))
        self.assertEqual(len({id(block) for block in streamed._vector_blocks}),3)
        self.assertTrue(shared.reads == [])
        for t in range(self.sizes[3]):
            np.testing.assert_allclose(streamed.calc_path_blocks(t)["vvv"],reference.calc_path_blocks(t)["vvv"],atol=2e-12)
        self.assertTrue(shared.reads)
        self.assertTrue(all(stop-start <= 2 for _,_,start,stop in shared.reads))
        projected = [{"id":0,"terms":[{"paths":self.paths[1],"coefficient":[.3,.2]},
                                       {"paths":self.paths[2],"coefficient":[-.4,0]}]}]
        np.testing.assert_allclose(streamed.calc_path_blocks(1,rows=projected)["vvv"],
                                   reference.calc_path_blocks(1,rows=projected)["vvv"],atol=2e-12)
        np.testing.assert_allclose(streamed.calc_path_blocks(1,distances=[(0,0,0),(1,0,2)])["vvv"],
                                   reference.calc_path_blocks(1,distances=[(0,0,0),(1,0,2)])["vvv"],atol=2e-12)
        expected = reference.calc_path_blocks(0)["vvv"].copy()
        assembled = np.zeros_like(expected)
        self.assertIsNone(streamed.calc_path_blocks(0,write=lambda pi,mi,se,sl,v:
            assembled.__setitem__((pi,mi,*sl),v)))
        np.testing.assert_allclose(assembled,expected,atol=2e-12)

    def test_streamed_three_leg_limit_and_failed_read_requires_reload(self):
        handles = tuple(BorrowedModes(v,capacity=2) for v in self.vectors)
        options = dict(usedNe=(2,3,2),ne_block_size=(1,2,1),
                       stream_eigenvectors=True,paths=[((),(),())])
        generator = BaryonElementalGenerator(self.sizes,Handle(self.u),handles,
                                              **options).load("c")
        self.assertEqual(generator.vector_scratch_bytes,
                         4*np.prod(self.sizes[:3])*3*16)
        with self.assertRaisesRegex(ValueError,"scratch exceeds"):
            BaryonElementalGenerator(self.sizes,Handle(self.u),handles,
                **options,vector_scratch_limit_bytes=generator.vector_scratch_bytes-1).load("c")
        with self.assertRaisesRegex(ValueError,"requires stream_eigenvectors"):
            BaryonElementalGenerator(self.sizes,Handle(self.u),Handle(self.vectors[0]),
                vector_scratch_limit_bytes=1)
        with patch.object(handles[1],"read_modes",side_effect=RuntimeError("read failed")):
            with self.assertRaisesRegex(RuntimeError,"read failed"):
                generator.calc_path_blocks(0)["vvv"]
        with self.assertRaisesRegex(ValueError,"load inputs"):
            generator.calc_path_blocks(0)["vvv"]
        generator.load("c")
        np.testing.assert_allclose(generator.calc_path_blocks(0)["vvv"],
            BaryonElementalGenerator(self.sizes,Handle(self.u),
                tuple(Handle(v) for v in self.vectors),usedNe=(2,3,2),
                ne_block_size=(1,2,1),paths=[((),(),())]).load("c").calc_path_blocks(0)["vvv"],atol=2e-12)

    def test_streamed_load_peak_excludes_full_mode_snapshot(self):
        sizes = (8,4,4,8)
        lx,ly,lz,lt = sizes
        gauge = np.broadcast_to(np.eye(3),
            (lt,lz,ly,lx,4,3,3)).astype(np.complex128,copy=True)
        values = np.ones((lt,16,lz,ly,lx,3),dtype=np.complex128)
        kwargs = dict(paths=[((),(),())],usedNe=16,ne_block_size=2)
        eager = BaryonElementalGenerator(sizes,Handle(gauge),Handle(values),**kwargs)
        streamed = BaryonElementalGenerator(sizes,Handle(gauge),BorrowedModes(values,2),
            **kwargs,stream_eigenvectors=True)
        def load_peak(generator):
            tracemalloc.start()
            try:
                generator.load("cfg")
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        eager_peak = load_peak(eager)
        streamed_peak = load_peak(streamed)
        self.assertGreater(eager_peak-streamed_peak,values.nbytes//2)
        self.assertEqual(streamed.vector_scratch_bytes,6*lz*ly*lx*3*16)

    def test_gauge_preparation_invalidates_call_cache(self):
        gen = self.generator()
        gen.calc_path_blocks(0)["vvv"]
        gen.stout_smear(1,.08,matrix_batch_size=7)
        gen.project_SU3(matrix_batch_size=9)
        links = stout_smear(self.u.transpose(4,0,1,2,3,5,6).copy(),1,.08,matrix_batch_size=7)
        links[:3] = project_su3(links[:3],matrix_batch_size=9)
        reference = links.transpose(1,2,3,4,0,5,6)
        np.testing.assert_allclose(gen.calc_path_blocks(0)["vvv"][1,1],determinant_reference(reference,self.vectors,self.paths[1],0,self.momenta[1]),atol=2e-12)

    def test_validation_none_phase_and_kernel_out(self):
        gen = self.generator(phase_convention="none")
        actual = gen.calc_path_blocks(0)["vvv"]
        np.testing.assert_allclose(actual[:,0],actual[:,1])
        with self.assertRaises(ValueError): self.generator(usedNe=9)
        with self.assertRaises(ValueError): self.generator(paths=[((6,),(),())])
        with self.assertRaises(ValueError): self.generator(phase_convention="midpoint")
        with self.assertRaises(ValueError): gen.calc_path_blocks(-1)["vvv"]
        with self.assertRaises(ValueError): gen.calc_path_blocks(0,("p2p",))
        with self.assertRaises(ValueError): gen.calc_path_blocks(0,rows=[{"id":0,"terms":[]}])["vvv"]
        self.assertFalse(hasattr(gen,"calc_deriv"))
        out = np.empty((2,3,2),complex)
        self.assertIs(generate_baryon_elemental([v[0] for v in self.vectors],(1,0,0),self.sizes[:3],out=out),out)
        np.testing.assert_allclose(out,determinant_reference(self.u,self.vectors,((),(),()),0,(1,0,0)),atol=1e-12)

    def test_cupy_paths_projection_and_streaming(self):
        try:
            import cupy as cp
            available = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            available = False
        if not available:
            self.skipTest("requires a working CUDA device")
        gpu = self.generator(array_module=cp,cache_bytes=512)
        expected = self.generator().calc_path_blocks(1)["vvv"].copy()
        np.testing.assert_allclose(cp.asnumpy(gpu.calc_path_blocks(1)["vvv"]),expected,atol=3e-12)
        assembled = np.empty_like(expected)
        gpu.calc_path_blocks(1,write=lambda pi,mi,se,sl,v: assembled.__setitem__((pi,mi,*sl),cp.asnumpy(v)))
        np.testing.assert_allclose(assembled,expected,atol=3e-12)
        rows = [{"id":0,"terms":[{"paths":self.paths[1],"coefficient":[.3,.2]},
                                  {"paths":self.paths[2],"coefficient":[-.4,0]}]}]
        np.testing.assert_allclose(cp.asnumpy(gpu.calc_path_blocks(1,rows=rows)["vvv"])[0],
                                   (.3+.2j)*expected[1]-.4*expected[2],atol=3e-12)
        shared = BorrowedModes(self.vectors[1],capacity=2)
        streamed = BaryonElementalGenerator(self.sizes,Handle(self.u),shared,
            usedNe=(1,2,1),paths=self.paths,momentum_list=self.momenta,
            ne_block_size=(1,2,1),spatial_block_size=5,array_module=cp,
            stream_eigenvectors=True).load("c")
        cpu = BaryonElementalGenerator(self.sizes,Handle(self.u),Handle(self.vectors[1]),
            usedNe=(1,2,1),paths=self.paths,momentum_list=self.momenta,
            ne_block_size=(1,2,1),spatial_block_size=5).load("c")
        np.testing.assert_allclose(cp.asnumpy(streamed.calc_path_blocks(1)["vvv"]),cpu.calc_path_blocks(1)["vvv"],atol=3e-12)
