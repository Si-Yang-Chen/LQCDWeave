"""Direct eight-link row writing and independent site-oracle checks."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from production.elemental.test_elemental import fixture, reference_matrix, Handle
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, inspect_dataset, register_npy,
)
from lqcdweave.runtime.datasets import open_dataset_view
from lqcdweave.production.elemental.planning import plan_indexed_directed_current
from fixtures.elemental_script import produce_current
from lqcdweave.planning.production import directed_current_dataset_document
from lqcdweave.adapters.directed_current_dataset import load_registered_directed_current
from lqcdweave.contraction.directed_current import contract_directed_current_pair_v2v
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion
from lqcdweave.operators.single_hadron.api import gamma
from test_project_source_provenance import import_provenance

from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class DirectedCurrentWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = DatasetIndex(self.root / 'index')
        self.sizes, self.links, self.vectors, _ = fixture()
        self.cfg = dict(ensemble='synthetic', configuration='cfg', lattice=list(self.sizes))
        names = ('time', 'z', 'y', 'x', 'direction', 'color_row', 'color_column')
        gaxes = [axis(n, range(k)) for n,k in zip(names,self.links.shape)]
        gaxes[0] = axis('time', [2,0,1])
        self.gspec = self.spec('gauge',gaxes)
        self.gid = self.save('gauge',self.gspec,self.links[[2,0,1]])
        names = ('time', 'mode', 'z', 'y', 'x', 'color')
        baxes = [axis(n, range(k)) for n,k in zip(names,self.vectors.shape)]
        baxes[:2] = [axis('time',[1,2,0]),axis('mode',[4,1,9])]
        self.bspec = self.spec('explicit_basis',baxes)
        self.bid = self.save('basis',self.bspec,self.vectors[[1,2,0]])
        self.momenta = ((1,-1,0),(0,0,0))

    def spec(self, kind, axes, **extra):
        doc = dict(schema=1, kind=kind, configuration=self.cfg, axes=axes,
            dtype='complex128', bases={}, definition={}, conventions={}, parents=[])
        doc.update(extra)
        return DatasetSpec(doc)

    def save(self, name, spec, values):
        source = self.root / (name+'.npy')
        np.save(source, values)
        identity = register_npy(self.root/name,source,spec,provenance=dict(test='independent'))
        self.index.register(self.root/name)
        return identity

    def write_current(self, name='out', **options):
        preview = plan_indexed_directed_current(
            self.index, self.gid, self.bid, momentum_list=self.momenta, **options)
        if preview.get('action') == 'blocked':
            raise ValueError('strict source policy blocks current inputs')
        stages = preview['stages']
        with ExitStack() as stack:
            gauge = stack.enter_context(open_dataset_view(
                self.index, stages['gauge']['spec'], data_id=self.gid))
            basis = stack.enter_context(open_dataset_view(
                self.index, stages['basis']['spec'], data_id=self.bid,
                subset_axes=('mode',)))
            destination = self.root/name/'current'
            writer = stack.enter_context(DatasetWriter(
                destination, stages['current']['spec'],
                provenance={'test': 'independent-eight-link-row-writer'},
            ))
            produce_current(gauge, basis, writer)
            identity = writer.publish()
        self.index.register(destination)
        return {'datasets': {'current': identity}}

    def reference(self, direction, time, momentum, ne=2, boundary='periodic'):
        field_time = (time + (1 if direction==6 else -1 if direction==7 else 0)) % 3
        left = self.vectors[time,:ne].reshape(ne,-1)
        right = self.vectors[field_time,:ne].reshape(ne,-1)
        matrix = reference_matrix(self.links,(direction,),time,momentum,boundary=boundary)
        return left.conj() @ matrix @ right.T

    def assert_reference(self, result, boundary='periodic', ne=2):
        with self.index.open(result['datasets']['current']) as output:
            self.assertEqual(output.spec.coordinates('time'),[0,1,2])
            self.assertEqual(output.spec.document['parents'],[self.gid,self.bid])
            self.assertEqual(output.spec.document['bases']['sink_mode'],
                dict(data_id=self.bid,indices=[4,1,9][:ne]))
            for time in range(3):
                for direction in range(8):
                    for mi,momentum in enumerate(self.momenta):
                        np.testing.assert_allclose(output.read(time=time)[direction,mi],
                            self.reference(direction,time,momentum,ne,boundary),atol=3e-12,rtol=3e-12)

    def test_periodic_open_all_times_actual_views_and_read_only_preview(self):
        for boundary in ('periodic','open'):
            with self.subTest(boundary=boundary):
                with patch.object(ElementalGenerator,'load',side_effect=AssertionError('preview computes')):
                    preview = plan_indexed_directed_current(self.index,self.gid,self.bid,
                        usedNe=2,momentum_list=self.momenta,boundary=boundary)
                self.assertEqual(preview['stages']['current']['action'],'compute')
                result = self.write_current(boundary,usedNe=2,boundary=boundary)
                self.assert_reference(result,boundary)
                with self.index.open(result['datasets']['current']) as current:
                    doc = current.spec.document['definition']
                    self.assertEqual(doc['input_views']['basis']['axes'][1]['coordinates'],[4,1])

    def test_mode_prefix_changes_planned_definition(self):
        first = plan_indexed_directed_current(self.index,self.gid,self.bid,
            momentum_list=self.momenta,usedNe=2)
        smaller_plan = plan_indexed_directed_current(self.index,self.gid,self.bid,
            momentum_list=self.momenta,usedNe=1)
        self.assertNotEqual(first['stages']['current']['spec'].definition_id,
                            smaller_plan['stages']['current']['spec'].definition_id)
        smaller = self.write_current('smaller',usedNe=1)
        self.assert_reference(smaller,ne=1)

    def test_shared_source_policy_uses_actual_inputs(self):
        known = import_provenance()

        def register_known(name, spec, values):
            source = self.root / (name + "-known.npy")
            np.save(source, values, allow_pickle=False)
            directory = self.root / (name + "-known")
            identity = register_npy(directory, source, spec, provenance=known,
                                    provenance_mode="external_import")
            self.index.register(directory)
            return identity

        gauge_id = register_known("gauge", self.gspec, self.links[[2, 0, 1]])
        basis_id = register_known("basis", self.bspec, self.vectors[[1, 2, 0]])
        preview = plan_indexed_directed_current(self.index, gauge_id, basis_id,
            usedNe=2, momentum_list=self.momenta, source_policy="require_known")
        self.assertEqual(preview["source_status"], "known")
        self.assertEqual(set(preview["source_origins"]), {gauge_id, basis_id})

        self.assertEqual(preview['stages']['current']['spec'].document['parents'],
                         [gauge_id, basis_id])

    def test_strict_unknown_actual_source_is_blocked_by_planning(self):
        preview = plan_indexed_directed_current(self.index,self.gid,self.bid,
            momentum_list=self.momenta,usedNe=2,source_policy='require_known')
        self.assertEqual(preview['action'],'blocked')
        self.assertTrue(preview['source_reasons'])

    def test_bad_requests_are_refused_during_planning(self):
        for options in (dict(boundary='unknown'),dict(usedNe=4),dict(usedNe=True),
                        dict(momentum_list=((0,0,0),(0,0,0))),dict(momentum_list=((True,0,0),))):
            with self.subTest(options=options), self.assertRaises((ValueError,TypeError)):
                plan_indexed_directed_current(self.index,self.gid,self.bid,**options)
        wrong=self.bspec.document
        wrong['configuration']['configuration']='different'
        bad=self.save('wrong',DatasetSpec(wrong),self.vectors[[1,2,0]])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            plan_indexed_directed_current(self.index,self.gid,bad)

    def test_direct_writer_rejects_different_actual_parent(self):
        doc=self.gspec.document; doc['definition']['variant']='independent'
        duplicate=self.save('variant',DatasetSpec(doc),self.links[[2,0,1]])
        with self.index.open(self.gid) as gauge,self.index.open(self.bid) as basis:
            _,document=directed_current_dataset_document(gauge.spec,basis.spec,
                gauge_id=duplicate,basis_id=self.bid,momentum_list=self.momenta)
            with DatasetWriter(self.root/'wrong-parent',DatasetSpec(document),provenance={}) as writer:
                with patch.object(ElementalGenerator,'load',side_effect=AssertionError('computed')):
                    with self.assertRaisesRegex(ValueError,'actual input views'):
                        produce_current(gauge,basis,writer)
                self.assertEqual(writer.missing,(0,1,2))

    def test_nonfinite_output_is_not_published(self):
        with patch.object(ElementalGenerator,'calc_path_blocks',
                          return_value={"v2v":np.full((8,2,2,2),np.nan,dtype=complex)}):
            with self.assertRaisesRegex(FloatingPointError,'nonfinite'):
                self.write_current(usedNe=2)
        output = self.root / 'out' / 'current'
        self.assertNotEqual(inspect_dataset(output, cfg='cfg')['state'], 'complete')
        self.assertFalse((output / 'cfg.npy').exists())
        self.assertFalse((output / 'cfg.h5').exists())

    def test_published_current_handoff_matches_independent_temporal_pair_trace(self):
        result=self.write_current(usedNe=2)
        rng=np.random.default_rng(301)
        values=rng.normal(size=(3,3,4,4,2,2))+1j*rng.normal(size=(3,3,4,4,2,2))
        binding=dict(data_id=self.bid,indices=[4,1])
        spec=self.spec('vsv',[axis('source_time',[2,0,1]),axis('sink_time',[1,2,0]),
            axis('sink_spin',range(4)),axis('source_spin',range(4)),
            axis('sink_mode',[4,1]),axis('source_mode',[4,1])],
            bases=dict(sink_mode=binding,source_mode=binding),parents=[self.bid,self.gid],
            definition=dict(gauge_id=self.gid),conventions=dict(normalization='unscaled',
                spin_basis='degrand_rossi',temporal_boundary='periodic'))
        vid=self.save('vsv',spec,values[[2,0,1]][:,[1,2,0]])
        terms=[replace(t,coefficient=t.coefficient*(.4+.7j))
               for t in ConservedVectorInsertion(3).terms]
        def flatten(value):
            return value.transpose(0,2,1,3).reshape(8,8)
        def vertex(term,time):
            bar=(time+term.bar_offset[3])%3
            field=(time+term.field_offset[3])%3
            direction=6 if term.link=='forward' else 7
            spatial=self.reference(direction,bar,self.momenta[0])
            spin=np.eye(4)-gamma(8) if term.link=='forward' else np.eye(4)+gamma(8)
            return bar,field,flatten(spin[:,:,None,None]*spatial[None,None])
        with self.index.open(result['datasets']['current']) as current,self.index.open(vid) as vsv:
            adapted=load_registered_directed_current(current,vsv)
            for first in range(3):
                for second in range(3):
                    expected=0j
                    for ta in terms:
                        ba,fa,a=vertex(ta,first)
                        for tb in terms:
                            bb,fb,b=vertex(tb,second)
                            expected += ta.coefficient*tb.coefficient*np.trace(
                                a.T@flatten(values[fb,ba])@b.T@flatten(values[fa,bb]))
                    actual=contract_directed_current_pair_v2v(terms,adapted['raw'],adapted['contract'],
                        terms,adapted['raw'],adapted['contract'],adapted['vsv'][0],
                        first_anchor_time=first,second_anchor_time=second,
                        first_field_ne=2,first_bar_ne=2,second_field_ne=2,second_bar_ne=2,
                        first_momentum=0,second_momentum=0)
                    np.testing.assert_allclose(actual['value'],expected,rtol=4e-12,atol=4e-9)

    def test_path_entry_retains_eight_directions_and_rejects_bad_times(self):
        generator=ElementalGenerator(self.sizes,Handle(self.links),Handle(self.vectors),
            paths=[(d,) for d in range(8)],path_version="v2",
            phase_convention="midpoint",momentum_list=self.momenta,usedNe=2).load('cfg')
        for time in range(3):
            row=generator.calc_path_blocks(time)["v2v"]
            for direction in range(8):
                for mi,momentum in enumerate(self.momenta):
                    np.testing.assert_allclose(row[direction,mi],
                        self.reference(direction,time,momentum,2,"periodic"),rtol=4e-12,atol=4e-12)
        for time in (-1,3,True):
            with self.subTest(time=time),self.assertRaises((ValueError,TypeError)):
                generator.calc_path_blocks(time)
