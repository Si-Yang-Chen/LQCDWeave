"""CuPy correctness and bounded GPU efficiency checks for elemental features.

Run on an allocated CUDA GPU. Set ED_RUN_GPU_BENCHMARK=1 to include the
throughput/block-size/memory sweep. CPU and GPU receive identical inputs.
"""

import json
import os
from pathlib import Path
from time import perf_counter
import unittest

import numpy as np

try:
    import cupy as cp
    try:
        _DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    except Exception:
        _DEVICE_COUNT = 0
except Exception:
    cp, _DEVICE_COUNT = None, 0

from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron import OperatorPath
from lqcdweave.production.elemental import (
    ElementalGenerator, generate_baryon_elemental, re_combine,
)
from lqcdweave.production.gauge import project_su3, stout_smear
from lqcdweave.sampling.mode_weights import dilution_pair_weights

from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


_WEAK_FIELD_DIR = Path(__file__).resolve().parent / "data" / "weak_field"


class Handle:
    def __init__(self, value):
        self.value = value

    def load(self, key):
        return self.value


def _fixture():
    rng = np.random.default_rng(20260926)
    sizes = (4,3,2,3)
    lx,ly,lz,lt = sizes
    raw = rng.normal(size=(lt,lz,ly,lx,4,3,3)) + 1j*rng.normal(size=(lt,lz,ly,lx,4,3,3))
    u,_ = np.linalg.qr(raw)
    u *= np.exp(-1j*np.angle(np.linalg.det(u))/3)[...,None,None]
    v = rng.normal(size=(lt,5,lz,ly,lx,3)) + 1j*rng.normal(size=(lt,5,lz,ly,lx,3))
    p = np.array([[(t%lx,0,0),((t+1)%lx,1,0),(0,2,1)] for t in range(lt)])
    return sizes,u,v,p


def _generator(xp, data, *, paths=None, order=0, mode="calc_disp", points=True,
               boundary="periodic", momenta=((0,0,0),(1,-1,0)), block=2, **kwargs):
    sizes,u,v,p = data
    return ElementalGenerator(sizes,Handle(u),Handle(v),point=Handle(p) if points else None,
        paths=paths,path_version="v2" if paths is not None else "legacy",num_nabla=order,
        calc_mode=mode,momentum_list=momenta,usedNe=5,usedNp=3 if points else None,
        path_boundary=boundary,ne_block_size=block,array_module=xp,**kwargs).load("gpu-test")


@unittest.skipUnless(cp is not None and _DEVICE_COUNT > 0, "requires a working CUDA device")
class ElementalCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _fixture()

    @staticmethod
    def _assert_backend_equal(cpu, gpu, *, rtol=3e-11, atol=3e-12):
        if isinstance(cpu, dict):
            for sector in cpu:
                if sector == "p2p":
                    for rowc,rowg in zip(cpu[sector],gpu[sector]):
                        for vc,vg in zip(rowc,rowg):
                            np.testing.assert_array_equal(cp.asnumpy(vg.rows),vc.rows)
                            np.testing.assert_array_equal(cp.asnumpy(vg.columns),vc.columns)
                            np.testing.assert_allclose(cp.asnumpy(vg.values),vc.values,rtol=rtol,atol=atol)
                else:
                    np.testing.assert_allclose(cp.asnumpy(gpu[sector]),cpu[sector],rtol=rtol,atol=atol)
            return
        np.testing.assert_allclose(cp.asnumpy(gpu),cpu,rtol=rtol,atol=atol)

    def test_all_endpoint_sectors_spatial_and_temporal_paths(self):
        paths = [(),(0,),(3,),(0,1,3,4),(6,),(7,0),(0,3)]
        cpu = _generator(np,self.data,paths=paths)
        gpu = _generator(cp,self.data,paths=paths)
        for t in range(self.data[0][3]):
            self._assert_backend_equal(cpu.calc_path_blocks(t,("v2v","v2p","p2v","p2p")),
                                      gpu.calc_path_blocks(t,("v2v","v2p","p2v","p2p")))

    def test_public_complex_point_path_combination(self):
        from lqcdweave.production.elemental.api import ElementalGenerator as TableGenerator
        sizes, u, v, p = self.data
        # Local and closed paths overlap at identical point pairs; both real
        # and imaginary contributions must accumulate on CUDA/HIP.
        table = [dict(component='sum', path=path, coefficient=coefficient)
                 for path, coefficient in (((), .3+.7j), ((0, 3), -.2+.4j), ((0,), .6-.1j))]
        options = dict(momentum_dict={0:(0, 0, 0)}, operator_table=table,
                       points=p, blocks=('v2v', 'v2p', 'p2v', 'p2p'))
        cpu = TableGenerator(u, v, **options).calc_path_blocks(0)
        gpu = TableGenerator(cp.asarray(u), cp.asarray(v), array_module=cp, **options).calc_path_blocks(0)
        self._assert_backend_equal(cpu, gpu)

    def test_point_only_and_streaming_borrowed_buffers(self):
        sizes,u,v,p=self.data
        cpu=ElementalGenerator(sizes,Handle(u),point=Handle(p),paths=[(),(0,)],path_version="v2",
                               momentum_list=((0,0,0),),array_module=np).load("point-only")
        gpu=ElementalGenerator(sizes,Handle(u),point=Handle(p),paths=[(),(0,)],path_version="v2",
                               momentum_list=((0,0,0),),array_module=cp).load("point-only")
        a=cpu.calc_path_blocks(0,("p2p",)); b=gpu.calc_path_blocks(0,("p2p",))
        self._assert_backend_equal(a,b)
        cpu=_generator(np,self.data,paths=[(),(0,),(6,)])
        gpu=_generator(cp,self.data,paths=[(),(0,),(6,)])
        expected=cpu.calc_path_blocks(0,("v2v","p2p"))
        first=gpu.calc_path_blocks(0,("v2v","p2p"))
        borrowed=first["v2v"]
        saved=cp.array(borrowed,copy=True)
        gathered={}
        def write(pi,mi,sector,slices,value):
            if sector=="p2p":
                gathered[pi,mi]=value
            else:
                gathered.setdefault(sector,cp.zeros_like(expected[sector]))[(pi,mi)+slices]=value
        gpu.calc_path_blocks(0,("v2v","p2p"),write=write)
        np.testing.assert_allclose(cp.asnumpy(gathered["v2v"]),expected["v2v"])
        for pi,row in enumerate(expected["p2p"]):
            for mi,value in enumerate(row):
                np.testing.assert_allclose(cp.asnumpy(gathered[pi,mi].values),value.values)
        gpu.calc_path_blocks(1)["v2v"]
        self.assertIs(borrowed,gpu.calc_path_blocks(2)["v2v"])
        self.assertFalse(bool(cp.array_equal(borrowed,saved)))

    def test_legacy_and_v2_path_enumeration(self):
        a = _generator(np,self.data,order=2)
        b = _generator(cp,self.data,order=2)
        self.assertEqual(a.num_disp,37)
        self._assert_backend_equal(a.calc_path_blocks(1)["v2v"],b.calc_path_blocks(1)["v2v"])
        paths = [GaugePath.from_index(i) for i in range(sum(8**k for k in range(3)))]
        a = _generator(np,self.data,paths=paths)
        b = _generator(cp,self.data,paths=paths)
        self.assertEqual(a.num_disp,73)
        self._assert_backend_equal(a.calc_path_blocks(1)["v2v"],b.calc_path_blocks(1)["v2v"])

    def test_derivative_and_projection(self):
        a = _generator(np,self.data,order=2,mode="calc_deriv")
        b = _generator(cp,self.data,order=2,mode="calc_deriv")
        self._assert_backend_equal(a.calc_path_blocks(1)["v2v"],b.calc_path_blocks(1)["v2v"])
        rows = [{"id":0,"terms":[{"path":[0,1],"coefficient":[.3,.8]},{"path":[3],"coefficient":[-.2,.7]}]},
                {"id":1,"terms":[{"path":[],"coefficient":[1,0]}]}]
        self._assert_backend_equal(_generator(np,self.data,paths=[()]).calc_path_blocks(1,rows=rows)["v2v"],
                                  _generator(cp,self.data,paths=[()]).calc_path_blocks(1,rows=rows)["v2v"])

    def test_existing_elementals_and_operator_consumers_on_gpu(self):
        paths=[GaugePath(),GaugePath((0,)),GaugePath((6,))]
        sectors={k:[] for k in ("v2v","v2p","p2v","p2p")}
        for xp in (np,cp):
            generator=_generator(xp,self.data,paths=paths)
            values={k:[] for k in sectors}
            for t in range(self.data[0][3]):
                data=generator.calc_path_blocks(t,tuple(sectors))
                for key in sectors:
                    values[key].append([xp.stack([value.to_dense() for value in row]) if key=="p2p" else row
                                        for row in data[key]])
            packed={key:xp.stack([xp.stack([values[key][t][i] for t in range(len(values[key]))],axis=1)
                                   for i in range(len(paths))]) for key in sectors}
            reader=ArrayElemental(packed,[ElementalComponent.path(path) for path in paths],phase_convention="midpoint",cfg="gpu-test")
            operator=OperatorPath("gpu-operator",[(15,paths[1],1,.4+.2j)])
            sink=Meson(reader,operator,array_module=xp).load("gpu-test")
            source=Meson(reader,operator,True,array_module=xp).load("gpu-test")
            found={"sink_v2v":sink.get(1),"source_v2v":source.get(1),
                   "sink_v2p":sink.get_v2p(1),"sink_p2v":sink.get_p2v(1),
                   "sink_p2p":sink.get_p2p(1),"source_v2p":source.get_v2p(1),
                   "source_p2v":source.get_p2v(1),"source_p2p":source.get_p2p(1)}
            if xp is np:
                cpu=found
            else:
                for key,value in found.items():
                    np.testing.assert_allclose(cp.asnumpy(value),cpu[key],rtol=4e-11,atol=4e-12)

    def test_open_temporal_boundary(self):
        cpu = _generator(np,self.data,paths=[(7,),(6,),(7,6)],boundary="open")
        gpu = _generator(cp,self.data,paths=[(7,),(6,),(7,6)],boundary="open")
        self._assert_backend_equal(cpu.calc_path_blocks(0,("v2v","p2v","v2p","p2p")),
                                   gpu.calc_path_blocks(0,("v2v","p2v","v2p","p2p")))

    def test_gauge_preparation_and_stout(self):
        sizes,u,v,p = self.data
        cpu = project_su3(u.transpose(4,0,1,2,3,5,6),matrix_batch_size=5)
        gpu = project_su3(cp.asarray(u.transpose(4,0,1,2,3,5,6)),matrix_batch_size=5,array_module=cp)
        self._assert_backend_equal(cpu,gpu,rtol=2e-12,atol=2e-12)
        cpu = stout_smear(u.transpose(4,0,1,2,3,5,6),1,.07,matrix_batch_size=5)
        gpu = stout_smear(cp.asarray(u.transpose(4,0,1,2,3,5,6)),1,.07,matrix_batch_size=5,array_module=cp)
        self._assert_backend_equal(cpu,gpu,rtol=3e-12,atol=3e-12)
        np.testing.assert_array_equal(cp.asnumpy(gpu[3]),u.transpose(4,0,1,2,3,5,6)[3])

    @unittest.skipUnless(
        os.environ.get("ED_WEAK_FIELD_DIR") or _WEAK_FIELD_DIR.is_dir(),
        "weak_field reference fixture is not available",
    )
    def test_weak_field_all_times_against_stored_reference(self):
        directory=Path(os.environ.get("ED_WEAK_FIELD_DIR", _WEAK_FIELD_DIR))
        sizes=(4,4,4,8)
        gauge=np.load(directory/"weak_field.npy",mmap_mode="r").reshape(4,8,4,4,4,3,3).transpose(1,2,3,4,0,5,6)
        # The archived CPU production consumed complex64 vector working buffers.
        vectors=np.load(directory/"weak_field.eigenvector.input.npy",mmap_mode="r").astype(np.complex64)
        deriv_mom=((0,0,0),(0,0,1),(0,1,1),(1,1,1),(0,0,2),(0,1,2),(1,1,2))
        expected=np.load(directory/"weak_field.elemental.npy",mmap_mode="r")
        cpu=ElementalGenerator(sizes,Handle(gauge),Handle(vectors),num_nabla=2,momentum_list=deriv_mom,
                               calc_mode="calc_deriv",ne_block_size=20,array_module=np).load("weak_field")
        gpu=ElementalGenerator(sizes,Handle(gauge),Handle(vectors),num_nabla=2,momentum_list=deriv_mom,
                               calc_mode="calc_deriv",ne_block_size=20,array_module=cp).load("weak_field")
        cp.cuda.runtime.deviceSynchronize()
        for t in range(8):
            actual=cp.asnumpy(gpu.calc_path_blocks(t)["v2v"])
            np.testing.assert_allclose(actual,expected[:,:,t],atol=2e-12,rtol=2e-11)
            np.testing.assert_allclose(actual,cpu.calc_path_blocks(t)["v2v"],atol=2e-12,rtol=2e-11)

    def test_baryon_sparse_writer_blending_and_recombine(self):
        sizes,u,v,p = self.data
        baryon_cpu = generate_baryon_elemental((v[0,:2],v[1,:3],v[2,:4]),(1,0,-1),sizes[:3],block_size=3)
        baryon_gpu = generate_baryon_elemental(tuple(cp.asarray(x) for x in (v[0,:2],v[1,:3],v[2,:4])),
                    (1,0,-1),sizes[:3],block_size=3,backend=cp)
        self._assert_backend_equal(baryon_cpu,baryon_gpu)
        rows=[[(.3+.8j,0),(2j,2)],[(-1,1)]]
        array=np.arange(36).reshape(3,3,4).astype(complex)
        self._assert_backend_equal(re_combine(array,rows,1),re_combine(cp.asarray(array),rows,1,array_module=cp))
        base = _generator(np,self.data,paths=[()]).calc_path_blocks(1)["v2v"]
        weighted = _generator(np,self.data,paths=[()],is_blending=True,dilution=([10],[5])).calc_path_blocks(1)["v2v"]
        np.testing.assert_allclose(weighted,base*dilution_pair_weights([10],[5]))
        baseg = _generator(cp,self.data,paths=[()]).calc_path_blocks(1)["v2v"]
        weightedg = _generator(cp,self.data,paths=[()],is_blending=True,dilution=([10],[5])).calc_path_blocks(1)["v2v"]
        self._assert_backend_equal(weighted,weightedg)
        self.assertEqual(baseg.shape,weightedg.shape)

    @unittest.skipUnless(os.environ.get("ED_RUN_GPU_BENCHMARK") == "1", "set ED_RUN_GPU_BENCHMARK=1 on the GPU node")
    def test_gpu_efficiency_and_memory(self):
        """Track all-time weak_field elemental throughput on CPU and GPU."""
        cp.cuda.runtime.deviceSynchronize()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"unknown")
        if isinstance(name,bytes):
            name=name.decode(errors="replace")
        directory=Path(os.environ.get("ED_WEAK_FIELD_DIR",_WEAK_FIELD_DIR))
        raw=np.load(directory/"weak_field.npy",mmap_mode="r")
        lx,ly,lz,lt=raw.shape[:3]+(raw.shape[4],)
        gauge=raw.reshape(4,lt,lz,ly,lx,3,3).transpose(1,2,3,4,0,5,6)
        vectors=np.load(directory/"weak_field.eigenvector.input.npy",mmap_mode="r").astype(np.complex64)
        ne=vectors.shape[1]
        momenta=((0,0,0),(0,0,1),(0,1,1),(1,1,1),(0,0,2),(0,1,2),(1,1,2))
        rows=[]
        for batch in (8,16,20):
            generator=ElementalGenerator((lx,ly,lz,lt),Handle(gauge),Handle(vectors),
                num_nabla=2,calc_mode="calc_deriv",momentum_list=momenta,
                ne_block_size=batch,array_module=cp).load("weak_field")
            for t in range(lt):
                generator.calc_path_blocks(t)["v2v"]
            cp.cuda.runtime.deviceSynchronize()
            samples=[]
            for _ in range(4):
                start=perf_counter()
                for t in range(lt):
                    value=generator.calc_path_blocks(t)["v2v"]
                cp.cuda.runtime.deviceSynchronize()
                samples.append(perf_counter()-start)
                if not bool(cp.isfinite(value).all()):
                    self.fail("nonfinite result during benchmark")
            cpu_generator=ElementalGenerator((lx,ly,lz,lt),Handle(gauge),Handle(vectors),
                num_nabla=2,calc_mode="calc_deriv",momentum_list=momenta,
                ne_block_size=batch,array_module=np).load("weak_field")
            for t in range(lt):
                cpu_generator.calc_path_blocks(t)["v2v"]
            cpu_samples=[]
            for _ in range(4):
                start=perf_counter()
                for t in range(lt):
                    cpu_value=cpu_generator.calc_path_blocks(t)["v2v"]
                cpu_samples.append(perf_counter()-start)
                if not np.isfinite(cpu_value).all():
                    self.fail("nonfinite result during CPU benchmark reference")
            pool=cp.get_default_memory_pool()
            contracted_pairs=lt*lx*ly*lz*ne*ne*len(momenta)
            rows.append({"ne_block_size":batch,"median_seconds":float(np.median(samples)),
                "samples_seconds":samples,"live_pool_bytes":int(pool.used_bytes()),
                "pool_reserved_bytes":int(pool.total_bytes()),
                "cpu_median_seconds":float(np.median(cpu_samples)),
                "cpu_samples_seconds":cpu_samples,
                "gpu_speedup_over_same_node_cpu":float(np.median(cpu_samples)/np.median(samples)),
                "site_mode_pairs_per_second":float(contracted_pairs/np.median(samples)),
                "cpu_site_mode_pairs_per_second":float(contracted_pairs/np.median(cpu_samples))})
        free_bytes,total_bytes=cp.cuda.runtime.memGetInfo()
        report={"gpu":name,"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
                "fixture":"weak_field","shape":{"lattice":[lx,ly,lz,lt],"Ne":ne,"momenta":len(momenta),"num_nabla":2},
                "sector":"derivative_elemental_all_times","iterations_per_batch":4,"device_total_bytes":int(total_bytes),
                "device_free_bytes_after_sweep":int(free_bytes),"results":rows}
        print("ELEMENTAL_GPU_BENCHMARK="+json.dumps(report,sort_keys=True))
        self.assertTrue(all(row["median_seconds"]>0 for row in rows))
        self.assertTrue(all(row["pool_reserved_bytes"]>0 for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
