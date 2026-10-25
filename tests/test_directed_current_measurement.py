"""Linked-current row writes over actual fields and independent dense matrices."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import test_project_source_provenance as source_fixture

from production.elemental.test_elemental import fixture, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.operators.single_hadron.api import (
    ConservedVectorInsertion, validate_one_link_current_term, gamma,
)
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, inspect_dataset, register_npy,
)
from lqcdweave.planning.production import directed_current_dataset_document
from lqcdweave.contraction.directed_current_measurement import (
    plan_indexed_directed_current_pair,
    write_directed_current_pair_rows,
)
from lqcdweave.contraction import directed_current_measurement as workflow
from lqcdweave.runtime.datasets import open_dataset_view
from lqcdweave.adapters.directed_current_dataset import load_registered_directed_current
from lqcdweave.contraction.directed_current import resolve_directed_current_raw, resolve_current_term_endpoints
from fixtures.c_group_production import publish_directed_current



def axis(name, values):
    return dict(name=name, coordinates=list(values))


def flatten(value):
    return value.transpose(0,2,1,3).reshape(value.shape[0]*value.shape[2],-1)


class DirectedCurrentMeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.index=DatasetIndex(self.root/'index')
        self.sizes,self.links,self.vectors,_=fixture()
        self.other_links=np.roll(self.links,1,axis=0).copy()
        self.cfg=dict(ensemble='synthetic',configuration='cfg',lattice=list(self.sizes))
        names=('time','z','y','x','direction','color_row','color_column')
        gaxes=[axis(n,range(k)) for n,k in zip(names,self.links.shape)]
        gaxes[0]=axis('time',[2,0,1])
        self.gspec=self.spec('gauge',gaxes)
        self.gid=self.save('gauge',self.gspec,self.links[[2,0,1]])
        self.other_gid=self.save('other-gauge',self.gspec,self.other_links[[2,0,1]])
        names=('time','mode','z','y','x','color')
        baxes=[axis(n,range(k)) for n,k in zip(names,self.vectors.shape)]
        baxes[:2]=[axis('time',[1,2,0]),axis('mode',[4,1,9])]
        self.bspec=self.spec('explicit_basis',baxes)
        self.bid=self.save('basis',self.bspec,self.vectors[[1,2,0]])
        self.momenta=[((1,-1,0),(0,0,0)),((0,0,0),(1,0,1))]
        self.first=self.current('first',self.links,self.gid,self.momenta[0])
        self.second=self.current('second',self.other_links,self.other_gid,self.momenta[1])
        rng=np.random.default_rng(191)
        dimension=int(np.prod(self.sizes[:3]))*3*4
        self.full_s=rng.normal(size=(3,3,dimension,dimension))+1j*rng.normal(size=(3,3,dimension,dimension))
        self.tau=np.empty((3,3,4,4,3,3),complex)
        for source in range(3):
            for sink in range(3):
                left=np.kron(np.eye(4),self.vectors[sink].reshape(3,-1).conj())
                right=np.kron(np.eye(4),self.vectors[source].reshape(3,-1).T)
                self.tau[source,sink]=(left@self.full_s[source,sink]@right).reshape(4,3,4,3).transpose(0,2,1,3)
        binding=dict(data_id=self.bid,indices=[4,1,9])
        self.vspec=self.spec('vsv',[axis('source_time',[2,0,1]),axis('sink_time',[1,2,0]),
            axis('sink_spin',range(4)),axis('source_spin',range(4)),
            axis('sink_mode',[4,1,9]),axis('source_mode',[4,1,9])],
            bases=dict(sink_mode=binding,source_mode=binding),parents=[self.gid,self.bid],
            definition=dict(gauge_id=self.gid,dirac=dict(action='synthetic_dense_matrix')),
            conventions=dict(normalization='unscaled',spin_basis='degrand_rossi',temporal_boundary='periodic'))
        self.vid=self.save('vsv',self.vspec,self.tau[[2,0,1]][:,[1,2,0]])
        temporal=ConservedVectorInsertion(3,wilson_r=.7).terms
        self.first_terms=(replace(temporal[0],coefficient=.4+.7j,normalization=.8-.2j,
            bar_offset=(0,0,0,1),field_offset=(0,0,0,2),link_origin_offset=(0,0,0,1)),
            replace(temporal[1],coefficient=-.3+.1j,normalization=.5+.4j))
        self.second_terms=tuple(replace(t,coefficient=t.coefficient*(.6-.2j),normalization=.7+.3j)
            for t in ConservedVectorInsertion(1,wilson_r=1.3).terms)

    def spec(self,kind,axes,**extra):
        doc=dict(schema=1,kind=kind,configuration=self.cfg,axes=axes,dtype='complex128',
            bases={},definition={},conventions={},parents=[]); doc.update(extra)
        return DatasetSpec(doc)

    def save(self,name,spec,values):
        source=self.root/(name+'.npy'); np.save(source,values)
        identity=register_npy(self.root/name,source,spec,provenance=dict(test='independent_explicit_arrays'))
        self.index.register(self.root/name)
        return identity

    def current(self,name,links,gid,momenta,boundary='periodic',basis_id=None):
        with self.index.open(gid) as gauge,self.index.open(
                self.bid if basis_id is None else basis_id) as basis:
            _,document=directed_current_dataset_document(gauge.spec,basis.spec,
                gauge_id=gid,basis_id=(self.bid if basis_id is None else basis_id),
                momentum_list=momenta,boundary=boundary)
        values=np.empty((3,8,len(momenta),3,3),complex)
        for time in range(3):
            for direction in range(8):
                field=(time+(1 if direction==6 else -1 if direction==7 else 0))%3
                for mi,momentum in enumerate(momenta):
                    values[time,direction,mi]=self.vectors[time].reshape(3,-1).conj() @ reference_matrix(
                        links,(direction,),time,momentum,boundary=boundary) @ self.vectors[field].reshape(3,-1).T
        return self.save(name,DatasetSpec(document),values)

    def write_pair(self, name='out', *, first_id=None, second_id=None, vsv_id=None,
                   first_terms=None, second_terms=None, first_momentum=1,
                   second_momentum=1, usedNe=2, first_times=None, second_times=None,
                   source_policy='allow_unknown'):
        first_id = self.first if first_id is None else first_id
        second_id = self.second if second_id is None else second_id
        vsv_id = self.vid if vsv_id is None else vsv_id
        first_terms = self.first_terms if first_terms is None else first_terms
        second_terms = self.second_terms if second_terms is None else second_terms
        preview = plan_indexed_directed_current_pair(
            self.index, first_id, second_id, vsv_id, first_terms, second_terms,
            first_momentum=first_momentum, second_momentum=second_momentum,
            usedNe=usedNe, first_times=first_times, second_times=second_times,
            source_policy=source_policy,
        )
        if preview.get('action') == 'blocked':
            raise ValueError('strict source policy blocks current pair inputs')
        stages = preview['stages']
        with ExitStack() as stack:
            inputs = []
            for role, identity in zip(
                    ('first_current', 'second_current', 'vsv'),
                    (first_id, second_id, vsv_id), strict=True):
                axes = ('sink_mode', 'source_mode', 'momentum') if role != 'vsv' else (
                    'sink_mode', 'source_mode')
                inputs.append(stack.enter_context(open_dataset_view(
                    self.index, stages[role]['spec'], data_id=identity,
                    subset_axes=axes,
                )))
            destination = self.root / name / 'measurement'
            writer = stack.enter_context(DatasetWriter(
                destination, stages['measurement']['spec'],
                provenance={'test': 'directed-current-low-level-writer'},
            ))
            write_directed_current_pair_rows(*inputs, writer)
            identity = writer.publish()
        self.index.register(destination)
        return {'datasets': {'measurement': identity}}

    def test_strict_source_policy_blocks_legacy_current_inputs_before_output(self):
        preview=plan_indexed_directed_current_pair(self.index,self.first,self.second,self.vid,
            self.first_terms,self.second_terms,first_momentum=1,second_momentum=1,
            usedNe=2,source_policy='require_known')
        self.assertEqual(preview['action'],'blocked')
        self.assertTrue(preview['source_reasons'])


    def vertex(self,term,anchor,links,momentum,ne):
        bar=(anchor+term.bar_offset[3])%3; field=(anchor+term.field_offset[3])%3
        direction=(6 if term.link=='forward' else 7) if term.direction==3 else (
            term.direction if term.link=='forward' else term.direction+3)
        spatial=self.vectors[bar,:ne].reshape(ne,-1).conj() @ reference_matrix(
            links,(direction,),bar,momentum) @ self.vectors[field,:ne].reshape(ne,-1).T
        spin=term.wilson_r*np.eye(4)+(-1 if term.link=='forward' else 1)*gamma(term.gamma_index)
        return bar,field,flatten(spin[:,:,None,None]*spatial[None,None])

    def expected(self,first_time,second_time,first_terms=None,ne=2):
        total=0j
        for ta in (self.first_terms if first_terms is None else first_terms):
            ba,fa,a=self.vertex(ta,first_time,self.links,self.momenta[0][1],ne)
            for tb in self.second_terms:
                bb,fb,b=self.vertex(tb,second_time,self.other_links,self.momenta[1][1],ne)
                total+=ta.coefficient*ta.normalization*tb.coefficient*tb.normalization*np.trace(
                    a.T@flatten(self.tau[fb,ba,...,:ne,:ne])@b.T@flatten(self.tau[fa,bb,...,:ne,:ne]))
        return total

    def assert_reference(self,result,first_terms=None,ne=2,rtol=5e-12):
        with self.index.open(result['datasets']['measurement']) as output:
            for first in output.spec.coordinates('first_anchor_time'):
                for second in output.spec.coordinates('second_anchor_time'):
                    np.testing.assert_allclose(output.read(first_anchor_time=first,second_anchor_time=second),
                        self.expected(first,second,first_terms,ne),rtol=rtol,atol=rtol*1e6)

    def test_all_times_complex_terms_actual_inputs_and_no_production(self):
        with patch.object(workflow,'load_registered_directed_current',side_effect=AssertionError('preview computes')):
            preview=plan_indexed_directed_current_pair(self.index,self.first,self.second,self.vid,
                self.first_terms,self.second_terms,first_momentum=1,second_momentum=1,usedNe=2)
        self.assertEqual(preview['stages']['measurement']['action'],'compute')
        before=[(self.root/(name+'.npy')).read_bytes() for name in ('first','second','vsv')]
        result=self.write_pair()
        self.assert_reference(result)
        self.assertEqual(before,[(self.root/(name+'.npy')).read_bytes() for name in ('first','second','vsv')])
        with self.index.open(result['datasets']['measurement']) as output:
            self.assertEqual(output.spec.coordinates('first_anchor_time'),[0,1,2])
            self.assertEqual(output.spec.coordinates('second_anchor_time'),[0,1,2])
            document=output.spec.document
            self.assertEqual(document['parents'],[self.first,self.second,self.vid])
            self.assertEqual(document['definition']['first_operator']['terms'][0]['coefficient'],[.4,.7])
            self.assertEqual(document['definition']['first_operator']['terms'][0]['field_offset'],[0,0,0,2])
            views=document['definition']['input_views']
            self.assertEqual(views['first_current']['bases']['sink_mode']['indices'],[4,1])
            self.assertEqual(views['first_current']['definition']['gauge_id'],self.gid)
            self.assertEqual(views['second_current']['definition']['gauge_id'],self.other_gid)
            self.assertEqual(views['vsv']['definition']['gauge_id'],self.gid)
            self.assertEqual(document['conventions']['wick_sign'],'not_applied')
    def test_ordered_time_subsets_are_preserved_by_the_public_row_writer(self):
        first_times = [2, 0]
        second_times = [1, 0]
        result = self.write_pair(first_times=first_times, second_times=second_times)
        self.assert_reference(result)
        with self.index.open(result['datasets']['measurement']) as output:
            first_coordinates = output.spec.coordinates('first_anchor_time')
            second_coordinates = output.spec.coordinates('second_anchor_time')
            self.assertEqual(first_coordinates, [2, 0])
            self.assertEqual(second_coordinates, [1, 0])
            self.assertNotIn(1, first_coordinates)
            self.assertNotIn(2, second_coordinates)



    def test_reordered_output_anchors_component_first_complex64_input(self):
        with self.index.open(self.first) as current:
            document=current.spec.document
            values=np.stack([current.read(time=t) for t in (2,0,1)])[:,::-1]
        axes={a['name']:a for a in document['axes']}
        axes['time']['coordinates']=[2,0,1]
        axes['component']['coordinates'].reverse()
        document['axes']=[axes[name] for name in ('component','momentum','time','sink_mode','source_mode')]
        document['dtype']='complex64'
        imported=self.save('imported',DatasetSpec(document),values.transpose(1,2,0,3,4).astype(np.complex64))
        result=self.write_pair('ordered',first_id=imported,first_times=[2,0,1],
            second_times=[1,0,2])
        self.assert_reference(result,rtol=4e-6)
        with self.index.open(result['datasets']['measurement']) as output:
            self.assertEqual(output.spec.coordinates('first_anchor_time'),[2,0,1])
            self.assertEqual(output.spec.coordinates('second_anchor_time'),[1,0,2])

    def test_open_temporal_crossings_refused_and_explicit_valid_window_preserved(self):
        first=self.current('open-first',self.links,self.gid,self.momenta[0],'open')
        second=self.current('open-second',self.other_links,self.other_gid,self.momenta[1],'open')
        document=self.vspec.document; document['conventions']['temporal_boundary']='open'
        vid=self.save('open-vsv',DatasetSpec(document),self.tau[[2,0,1]][:,[1,2,0]])
        with self.assertRaisesRegex(IndexError,'open temporal boundary'):
            plan_indexed_directed_current_pair(self.index,first,second,vid,
                self.first_terms,self.second_terms,first_momentum=1,second_momentum=1,usedNe=2)
        result=self.write_pair('window',first_id=first,second_id=second,vsv_id=vid,
            first_times=[0])
        self.assert_reference(result)
        with self.index.open(result['datasets']['measurement']) as output:
            self.assertEqual(output.spec.coordinates('first_anchor_time'),[0])
            self.assertEqual(output.spec.coordinates('second_anchor_time'),[0,1,2])

    def test_missing_or_incompatible_metadata_and_invalid_recipes_fail_before_output(self):
        variants=[dict(usedNe=4),dict(first_momentum=True),dict(first_momentum=4),
                  dict(first_times=[]),dict(first_times=[True]),dict(second_times=[0,0])]
        for options in variants:
            with self.subTest(options=options),self.assertRaises((ValueError,TypeError)):
                plan_indexed_directed_current_pair(self.index,self.first,self.second,self.vid,
                    self.first_terms,self.second_terms,**options)
        bad_terms=[(),(replace(self.first_terms[0],field_offset=(0,0,0,3)),),
                   (replace(self.first_terms[0],link_dagger=True),)]
        for terms in bad_terms:
            with self.subTest(terms=terms),self.assertRaises(ValueError):
                plan_indexed_directed_current_pair(self.index,self.first,self.second,self.vid,
                    terms,self.second_terms)
        document=self.vspec.document
        for binding in document['bases'].values(): binding['data_id']='unrelated-basis'
        document['parents'].append('unrelated-basis')
        bad=self.save('wrong-basis',DatasetSpec(document),self.tau[[2,0,1]][:,[1,2,0]])
        with self.assertRaisesRegex(ValueError,'basis identity'):
            plan_indexed_directed_current_pair(self.index,self.first,self.second,bad,
                self.first_terms,self.second_terms)

    def test_wrong_actual_parent_writer_refused_before_adaptation(self):
        preview=plan_indexed_directed_current_pair(self.index,self.first,self.second,self.vid,
            self.first_terms,self.second_terms,first_momentum=1,second_momentum=1,usedNe=2)
        document=preview['stages']['measurement']['spec'].document
        document['parents'][0]=self.second
        document['definition']['input_roles']['first_current']=self.second
        with ExitStack() as stack:
            inputs=[]
            for role,identity in zip(('first_current','second_current','vsv'),(self.first,self.second,self.vid)):
                inputs.append(stack.enter_context(open_dataset_view(self.index,preview['stages'][role]['spec'],
                    data_id=identity,subset_axes=('sink_mode','source_mode','momentum') if role!='vsv' else
                        ('sink_mode','source_mode'))))
            writer=stack.enter_context(DatasetWriter(self.root/'wrong-parent',DatasetSpec(document),provenance={}))
            with patch.object(workflow,'load_registered_directed_current',side_effect=AssertionError('adapted')):
                with self.assertRaisesRegex(ValueError,'actual input views'):
                    write_directed_current_pair_rows(*inputs,writer)
            self.assertEqual(writer.missing,(0,1,2))

    def test_nonfinite_output_never_published(self):
        with patch.object(workflow,'contract_directed_current_pair_v2v',return_value=dict(value=complex(np.inf))):
            with self.assertRaisesRegex(FloatingPointError,'nonfinite'):
                self.write_pair()
        output = self.root / 'out' / 'measurement'
        self.assertNotEqual(inspect_dataset(output, cfg='cfg')['state'], 'complete')
        self.assertFalse((output / 'cfg.npy').exists())
        self.assertFalse((output / 'cfg.h5').exists())

    def test_public_current_producer_handoff_to_pair_row_writer(self):
        produced=publish_directed_current(
            self.index,self.root,self.gid,self.bid,output_name='production',
            momentum_list=self.momenta[0],usedNe=2)
        result=self.write_pair('produced-trace',first_id=produced)
        self.assert_reference(result)

    def test_raw_resolver_rejects_forged_one_link_field_endpoint(self):
        term=self.first_terms[0]
        with self.index.open(self.first) as first,self.index.open(self.vid) as vsv:
            adapted=load_registered_directed_current(first,vsv)
        endpoints=resolve_current_term_endpoints(term,anchor_time=0,temporal_extent=3,boundary='periodic')
        endpoints['field_time']=0  # +t at bar time 1 actually ends at time 2.
        with self.assertRaisesRegex(ValueError,'raw link endpoints'):
            resolve_directed_current_raw(adapted['raw'],adapted['contract'],term,
                endpoints=endpoints,source_ne=2,sink_ne=2)
