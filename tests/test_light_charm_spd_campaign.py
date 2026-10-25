"""Joint flavor/radial campaign checked against independent dense Wick traces."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/campaigns'))
import light_charm_spd as campaign
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.operators.single_hadron.api import meson55_preset, mom_dict_to_list, momDict_meson55
from lqcdweave.operators.correlator.api import Hadron, Flavor, Tag
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.contraction.correlator import calculate


def test_periodic_slot_pairs_equal_physical_dt32():
    times=tuple((78+16*block+offset)%128 for block in range(8) for offset in (0,1,4,9,11))
    pairs=campaign.periodic_pairs(times)
    assert len(pairs)==400
    assert len({(p['sink'],p['source']) for p in pairs})==400
    assert pairs[-10]==dict(sink=0,source=39)
    assert pairs[-1]==dict(sink=9,source=39)
    assert {(times[(s+10)%40]-times[s])%128 for s in range(40)}=={32}


def test_joint_flavor_mixing_loops_radials_and_dagger():
    rng=np.random.default_rng(79138)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    table=meson55_preset(version='v2')
    elemental=ArrayElemental({'v2v':random((55,33,2,3,3))},tuple(r['component'] for r in table),
        cfg='toy',momentum_list=mom_dict_to_list(momDict_meson55),phase_convention='midpoint')
    light=ArrayPropagator({'vsv':random((2,2,4,4,2,2))},cfg='toy',time_convention='absolute')
    charm=ArrayPropagator({'vsv':random((2,2,4,4,3,3))},cfg='toy',time_convention='absolute')
    channel=dict(next(c for c in campaign.channels() if c['name']=='p2_0_T1_Pm'))
    channel['rows']=1
    channel['operators']=[o for o in channel['operators'] if o['name'] in
        ('op_00_gamma4gammai_T1','op_25_gammai_T1')]
    assert len(channel['operators'])==2
    assert {o['reflection'] for o in channel['operators']}=={-1,1}
    assert {o['c_label'] for o in channel['operators']}=={-1,1}
    requests,fields,tags=campaign.build_task([channel],light_bins=((0,),(1,)),charm_bins=((0,),(1,2)))
    props={}
    for flavor,reader in (('u',light),('d',light),('q',light),('c',charm)):
        props[f'S^{flavor}']=reader;props[rf'S^{flavor}_\mathrm{{local}}']=reader
    result=calculate(requests,vertex_map=campaign.vertex_binding(elemental,tags),
        propagator_map=props,configuration='toy',time_axes={'sink':1,'source':0},
        optimize='none',backend='numpy',memory_limit=1<<30,host_memory_limit=1<<30).value
    from lqcdweave.contracts.correlator.api import DiagramBindings, describe_coordinates
    from lqcdweave.contracts.execution import TermValue
    from lqcdweave.contraction.expansion.api import describe_meson_matrix, bind_meson_matrix_catalog
    from lqcdweave.contraction.planning.api import plan_matrix_actions, CostProfile
    from lqcdweave.contraction.execution.api import replay_scene_storage
    from lqcdweave.contraction.results.api import assemble_correlator
    bindings=DiagramBindings(vertex_map=campaign.vertex_binding(elemental,tags),
        propagator_map=props,configuration='toy')
    coordinates,_=describe_coordinates(time_axes={'sink':1,'source':0})
    with patch('lqcdweave.contraction.expansion.structured._Compiler.task',
               side_effect=AssertionError('scalar fallback')):
        catalog=describe_meson_matrix(requests,bindings=bindings,
            representative_times={'sink':1,'source':0})
        expanded=bind_meson_matrix_catalog(catalog,bindings=bindings,coordinates=coordinates)
    inputs=expanded.inputs
    native_plan=plan_matrix_actions(inputs.problem,inputs.matrix_kernels,inputs.blocks,inputs.preparations,
        profile=CostProfile(),memory_limit=1<<30,reserved_host_bytes=inputs.assembly_host_bytes)
    values=replay_scene_storage(native_plan,inputs.loader,io_overlap=False)
    native=assemble_correlator(expanded,tuple(TermValue(term,coordinates.points[0].key,value)
        for term,value in zip(inputs.term_ids,values,strict=True)))
    mesons=RecipeVertex(elemental,version='v2')
    def dense(value,left,right):
        v=value[:,:,left,:][:,:,:,right]
        return v.transpose(0,2,1,3).reshape(4*len(left),4*len(right))
    def vertex(record,dagger,time):
        op=record['operator']
        symbol=mesons.symbol(op['name'],channel['momentum'],irrep=channel['irrep'],
            copy_index=op['copy_index'],tag=Tag(0,'sink'))[0]
        if dagger:
            symbol=Hadron(symbol,Flavor('uc')).conjugate().irrep_row
        ne_left=3 if record['kind']=='charm' else 2
        ne_right=2 if record['kind']=='light' else 3
        shape=(4*ne_right,4*ne_left) if dagger else (4*ne_left,4*ne_right)
        value=np.zeros(shape,dtype=np.complex128)
        kernel=mesons(symbol).load('toy').get(time)
        for term in record['terms']:
            left,right=term['mode_indices']
            a,b=(right,left) if dagger else (left,right)
            na,nb=(ne_right,ne_left) if dagger else (ne_left,ne_right)
            ai=[spin*na+i for spin in range(4) for i in a]
            bi=[spin*nb+i for spin in range(4) for i in b]
            value[np.ix_(ai,bi)]+=complex(term['coefficient'])*dense(kernel,a,b)
        return value
    def propagator(reader,sink,source):
        ne=reader.sectors['vsv'].shape[-1]
        return dense(reader.block(sink,source,'vsv'),tuple(range(ne)),tuple(range(ne)))
    for path,field in fields.items():
        actual=result[path[0]][path[1]][path[2]]['two_point']
        records=field['operators'];expected=np.zeros_like(actual)
        for i,a in enumerate(records):
            A=vertex(a,False,1)
            for j,b in enumerate(records):
                B=vertex(b,True,0)
                if a['kind']==b['kind']:
                    forward=charm if a['kind'] in ('D','charm') else light
                    backward=charm if a['kind']=='charm' else light
                    expected[i,j]-=np.trace(A@propagator(forward,1,0)@B@propagator(backward,0,1))
                if a['flavor']=='I0' and b['flavor']=='I0':
                    pa=charm if a['kind']=='charm' else light
                    pb=charm if b['kind']=='charm' else light
                    weight=(np.sqrt(2) if a['flavor']=='I0' else 1)*(np.sqrt(2) if b['flavor']=='I0' else 1)
                    expected[i,j]+=weight*np.trace(A@propagator(pa,1,1))*np.trace(B@propagator(pb,0,0))
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-10)
        np.testing.assert_allclose(native[path[0]][path[1]][path[2]]['two_point'],expected,rtol=2e-12,atol=2e-10)


def test_complete_radial_space_has_correct_c_eigenvalues():
    for count in (2,4,5):
        for c_factor in (-1,1):
            bases=[]
            for c_parity in (1,-1):
                choices=campaign.c_radials(count,c_factor,c_parity)
                for _,exchange,terms in choices:
                    value=np.zeros(count*count)
                    for index,weight in terms:
                        value[index]=float(weight)
                    np.testing.assert_allclose(c_factor*value.reshape(count,count).T.ravel(),c_parity*value)
                    assert exchange==c_factor*c_parity
                    bases.append(value)
            transform=np.stack(bases)
            assert transform.shape==(count*count,count*count)
            np.testing.assert_allclose(transform@transform.T,np.eye(count*count),rtol=1e-14,atol=1e-14)
