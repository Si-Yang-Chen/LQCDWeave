"""Revised physical scope, independent scalar traces and physical source means."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/"tools/campaigns"))
import light_charm_spd as campaign
import light_charm_spd_production as production
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.operators.single_hadron.api import meson55_preset, mom_dict_to_list, momDict_meson55
from lqcdweave.contraction.correlator import calculate, load_correlator


def test_scalar_full_disconnected_and_loops_against_independent_traces(tmp_path):
    rng=np.random.default_rng(8192)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    table=meson55_preset(version="v2")
    elemental=ArrayElemental({"v2v":random((55,33,3,1,1))},tuple(r["component"] for r in table),
        cfg="toy",momentum_list=mom_dict_to_list(momDict_meson55),phase_convention="midpoint")
    light=ArrayPropagator({"vsv":random((3,3,4,4,1,1))},cfg="toy",time_convention="absolute")
    charm=ArrayPropagator({"vsv":random((3,3,4,4,1,1))},cfg="toy",time_convention="absolute")
    channel=dict(next(c for c in campaign.channels() if c["name"]=="p2_0_A1_Pp"))
    channel["operators"]=[o for o in channel["operators"] if o["name"]=="op_00_identity_A1"]
    assert len(channel["operators"])==1
    requests,fields,tags=campaign.build_task([channel],light_bins=((0,),),charm_bins=((0,),))
    assert set(requests)=={"I0_light_Cp","I1_light_Cp","charmonium_Cp","D"}
    planned=calculate(requests,vertex_map=campaign.vertex_binding(elemental,tags),
        propagator_map=production.propagators(light,charm),configuration="toy",
        time_axes={"sink":1,"source":0},execute=False,optimize="anytime",kernel_batching=True,
        search=dict(root_width=1,branch_width=1,storage_branch_width=1,
                    include_ties=False,component_workers=1,deadline_seconds=.2),
        backend="numpy",memory_limit=1<<28,host_memory_limit=1<<28)
    pairs=[dict(source=0,sink=1),dict(source=0,sink=2),dict(source=1,sink=2)]
    with patch("lqcdweave.contraction.correlator.find_contraction_path",
            side_effect=AssertionError("fixed replay searched")), patch(
            "lqcdweave.contraction.correlator.expand_correlator",
            side_effect=AssertionError("fixed replay expanded")):
        reference=planned.run(time_pairs=pairs,batch="full").value
        with patch("lqcdweave.contraction.correlator_coordinates.select_coordinate_replay",
                side_effect=AssertionError("window replay expanded coordinate graphs")):
            replay=planned.run(output=tmp_path/"pairs",time_pairs=pairs,
                physical_times=(0,1,2),physical_lt=3,time_roles=("sink","source"))
        from lqcdweave.storage.dataset import DatasetReader
        with patch('lqcdweave.storage.dataset.DatasetReader',wraps=DatasetReader) as opened:
            actual=production.read_pair_results(tmp_path/"pairs","toy",requests)
        assert opened.call_count==1
        assert replay.report['coordinate_count']==len(pairs)
        assert replay.report['peak_bound_windows']<=2
        for group,channels in reference.items():
            for name,rows in channels.items():
                for row,outputs in rows.items():
                    for field,value in outputs.items():
                        np.testing.assert_allclose(actual[group][name][row][field],value,
                            rtol=2e-12,atol=2e-10)
    def matrix(reader,s,t):
        return reader.block(s,t,"vsv")[:,:,0,0]
    # Local identity gamma; random complex spatial kernel has source conjugation.
    for i,pair in enumerate(pairs):
        s,t=pair["sink"],pair["source"]
        a=elemental.sectors["v2v"][0,0,s,0,0]*np.eye(4)
        b=elemental.sectors["v2v"][0,0,t,0,0].conjugate()*np.eye(4)
        conn=-np.trace(a@matrix(light,s,t)@b@matrix(light,t,s))
        disc=2*np.trace(a@matrix(light,s,s))*np.trace(b@matrix(light,t,t))
        light0=actual["I0_light_Cp"][channel["name"]]["row_0"]
        np.testing.assert_allclose(light0["two_point"][0,0,i],conn+disc,rtol=2e-12,atol=2e-10)
        np.testing.assert_allclose(light0["disconnected"][0,0,i],disc,rtol=2e-12,atol=2e-10)
        # One-point Wick sign is fixed by the same physical insertion convention.
        np.testing.assert_allclose(light0["vacuum_sink"][0,i],-np.sqrt(2)*np.trace(a@matrix(light,s,s)),rtol=2e-12,atol=2e-10)
        np.testing.assert_allclose(light0["vacuum_source"][0,i],-np.sqrt(2)*np.trace(b@matrix(light,t,t)),rtol=2e-12,atol=2e-10)
        cc=actual["charmonium_Cp"][channel["name"]]["row_0"]["two_point"][0,0,i]
        np.testing.assert_allclose(cc,-np.trace(a@matrix(charm,s,t)@b@matrix(charm,t,s)),rtol=2e-12,atol=2e-10)
    assert planned.value is None
    planned.close()


def test_source_average_counts_wrapping_and_scalar_loop_order(tmp_path):
    times=tuple((78+16*b+o)%128 for b in range(8) for o in (0,1,4,9,11))
    selection=TimeSelectionRecord(128,times)
    pairs=campaign.periodic_pairs(times)
    raw=np.array([10*p["source"]+p["sink"]+1j*(i%7) for i,p in enumerate(pairs)])
    sink=np.array([100+p["sink"] for p in pairs],dtype=complex)
    source=np.array([200+p["source"] for p in pairs],dtype=complex)
    values={"I0_light_Cp":{"p2_0_A1_Pp":{"row_0":{
        "two_point":raw[None,None,:],"disconnected":(raw*3)[None,None,:],
        "vacuum_sink":sink[None,:],"vacuum_source":source[None,:]}}}}
    production.save_source_average(tmp_path,"cfg",values,selection,
        metadata={"vacuum_subtracted":False},provenance={})
    saved=load_correlator(tmp_path/"source_average",configuration="cfg")
    counts=np.zeros(128,dtype=int);sums=np.zeros(128,dtype=complex)
    for i,p in enumerate(pairs):
        dt=(times[p["sink"]]-times[p["source"]])%128
        counts[dt]+=1;sums[dt]+=raw[i]
    expected=np.full(128,np.nan+0j);covered=counts>0
    expected[covered]=sums[covered]/counts[covered]
    fields=saved.value["I0_light_Cp"]["p2_0_A1_Pp"]["row_0"]
    np.testing.assert_allclose(fields["two_point"][0,0],expected[1:33],equal_nan=True)
    np.testing.assert_allclose(fields["disconnected"][0,0],3*expected[1:33],equal_nan=True)
    np.testing.assert_array_equal(saved.reduction["counts"],counts)
    assert counts.sum()==400 and counts[32]==40 and not counts[0]
    loops=load_correlator(tmp_path/"scalar_loops",configuration="cfg").value
    loops=loops["I0_light_Cp"]["p2_0_A1_Pp"]["row_0"]
    np.testing.assert_array_equal(loops["vacuum_sink"][0],100+np.arange(40))
    np.testing.assert_array_equal(loops["vacuum_source"][0],200+np.arange(40))


def test_dispatch_two_cfgs_share_one_selection_and_bind_current_inputs(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from lqcdweave.adapters import mpi_dispatch
    rng=np.random.default_rng(236)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    times=tuple((78+16*b+o)%128 for b in range(8) for o in (0,1,4,9,11))
    selection=TimeSelectionRecord(128,times)
    table=meson55_preset(version="v2")
    data={}
    for cfg in ("a","b"):
        e=ArrayElemental({"v2v":random((55,33,40,1,1))},tuple(r["component"] for r in table),
            cfg=cfg,data_identity=f"elemental-{cfg}",momentum_list=mom_dict_to_list(momDict_meson55),
            phase_convention="midpoint")
        l=ArrayPropagator({"vsv":random((40,40,4,4,1,1))},cfg=cfg,data_identity=f"light-{cfg}",time_convention="absolute")
        c=ArrayPropagator({"vsv":random((40,40,4,4,1,1))},cfg=cfg,data_identity=f"charm-{cfg}",time_convention="absolute")
        data[cfg]=(e,l,c)
    @contextmanager
    def inputs(_base,cfg,_scratch):
        yield *data[cfg],selection,{"configuration":cfg}
    class Queue:
        comm=SimpleNamespace(Get_size=lambda:1)
        def __iter__(self):
            yield from ("a","b")
    finished=[]
    monkeypatch.setattr(mpi_dispatch,"Dispatch",lambda *_:Queue())
    monkeypatch.setattr(mpi_dispatch,"combine",lambda path,cfg:finished.append(cfg))
    monkeypatch.setattr(production,"inputs",inputs)
    from lqcdweave.contraction import correlator_coordinates
    monkeypatch.setattr(correlator_coordinates,"select_coordinate_replay",
        lambda *_args,**_kwargs: (_ for _ in ()).throw(AssertionError("full coordinate expansion")))
    channel=dict(next(c for c in campaign.channels() if c["name"]=="p2_0_A1_Pp"))
    channel["operators"]=[o for o in channel["operators"] if o["name"]=="op_00_identity_A1"]
    original_build=campaign.build_task
    monkeypatch.setattr(campaign,"build_task",lambda **kw:original_build([channel],
        light_bins=((0,),),charm_bins=((0,),),**kw))
    original_plan=campaign.plan;selected=[]
    def plan(*args,**kwargs):
        selected.append(kwargs["configuration"])
        return original_plan(*args,**kwargs)
    monkeypatch.setattr(campaign,"plan",plan)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        monkeypatch.setenv(key,"1")
    args=SimpleNamespace(scratch=tmp_path/"scratch",output=tmp_path/"out",data_root=tmp_path,
        cfg_file=tmp_path/"cfg.txt",suffix="test",backend="numpy",host_gib=1,device_gib=1)
    production.run(args)
    assert selected==["a"] and finished==["a","b"]
    pairs=campaign.periodic_pairs(times)
    for cfg,(e,l,c) in data.items():
        expected={name:np.zeros(128,dtype=complex) for name in
            ('charmonium_Cp','I1_light_Cp','I0_light_Cp','D','disconnected')}
        counts=np.zeros(128,dtype=int)
        for p in pairs:
            s,t=p["sink"],p["source"]
            a=e.sectors["v2v"][0,0,s,0,0]*np.eye(4)
            b=e.sectors["v2v"][0,0,t,0,0].conjugate()*np.eye(4)
            dt=(times[s]-times[t])%128
            light_conn=-np.trace(a@l.block(s,t,"vsv")[:,:,0,0]@b@l.block(t,s,"vsv")[:,:,0,0])
            disc=2*np.trace(a@l.block(s,s,"vsv")[:,:,0,0])*np.trace(b@l.block(t,t,"vsv")[:,:,0,0])
            expected['charmonium_Cp'][dt]-=np.trace(a@c.block(s,t,"vsv")[:,:,0,0]@b@c.block(t,s,"vsv")[:,:,0,0])
            expected['I1_light_Cp'][dt]+=light_conn
            expected['I0_light_Cp'][dt]+=light_conn+disc
            expected['disconnected'][dt]+=disc
            expected['D'][dt]-=np.trace(a@c.block(s,t,"vsv")[:,:,0,0]@b@l.block(t,s,"vsv")[:,:,0,0])
            counts[dt]+=1
        saved_result=load_correlator(args.output/"source_average",configuration=cfg)
        saved=saved_result.value
        np.testing.assert_array_equal(saved_result.reduction['counts'],counts)
        for group in ('charmonium_Cp','I1_light_Cp','I0_light_Cp','D'):
            actual=saved[group][channel["name"]]["row_0"]["two_point"][0,0]
            np.testing.assert_allclose(actual,expected[group][1:33]/counts[1:33],rtol=2e-12,atol=2e-10)
        actual=saved['I0_light_Cp'][channel['name']]['row_0']['disconnected'][0,0]
        np.testing.assert_allclose(actual,expected['disconnected'][1:33]/counts[1:33],rtol=2e-12,atol=2e-10)
        loops=load_correlator(args.output/'scalar_loops',configuration=cfg).value
        for role in ('sink','source'):
            expected_loop=[]
            for slot in range(40):
                spatial=e.sectors['v2v'][0,0,slot,0,0]
                spatial=spatial.conjugate() if role=='source' else spatial
                expected_loop.append(-np.sqrt(2)*spatial*np.trace(l.block(slot,slot,'vsv')[:,:,0,0]))
            np.testing.assert_allclose(loops['I0_light_Cp'][channel['name']]['row_0'][f'vacuum_{role}'][0],
                expected_loop,rtol=2e-12,atol=2e-10)
        statistics=saved_result.metadata['fixed_replay']
        assert statistics['coordinate_count']==400 and statistics['peak_bound_windows']<=2
        assert not list(args.scratch.glob('pairs-*'))


def test_scalar_disconnected_is_not_obtained_by_subtracting_large_full_values():
    table=meson55_preset(version="v2")
    e=ArrayElemental({"v2v":np.ones((55,33,2,1,1),dtype=np.complex64)},
        tuple(r["component"] for r in table),cfg="toy",
        momentum_list=mom_dict_to_list(momDict_meson55),phase_convention="midpoint")
    arrays=np.empty((2,2,4,4,1,1),dtype=np.complex64)
    for a in range(2):
        for b in range(2):
            arrays[a,b,:,:,0,0]=np.eye(4)*(1 if a==b else 1e6)
    p=ArrayPropagator({"vsv":arrays},cfg="toy",time_convention="absolute")
    channel=dict(next(c for c in campaign.channels() if c["name"]=="p2_0_A1_Pp"))
    channel["operators"]=[o for o in channel["operators"] if o["name"]=="op_00_identity_A1"]
    task,_,tags=campaign.build_task([channel],light_bins=((0,),),charm_bins=((0,),))
    result=calculate(task,vertex_map=campaign.vertex_binding(e,tags),
        propagator_map=production.propagators(p,p),configuration="toy",
        time_axes={"sink":1,"source":0},optimize="none",backend="numpy",
        memory_limit=1<<28,host_memory_limit=1<<28)
    scalar=result.value["I0_light_Cp"][channel["name"]]["row_0"]
    assert abs(scalar["two_point"][0,0])>1e12
    np.testing.assert_allclose(scalar["disconnected"],[[32.]],rtol=1e-6,atol=1e-6)
