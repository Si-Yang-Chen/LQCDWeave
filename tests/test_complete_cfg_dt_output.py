"""Packed nonzero physical dt, compact loops and offline resource accounting."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import test_complete_cfg_output
from lqcdweave.contraction.correlator import calculate,save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.correlator_output import load_correlator_output
from lqcdweave.storage.dataset import read_dataset_metadata,DatasetReader,DatasetWriter


class PackedDtTests(unittest.TestCase):
    def test_internal_output_scans_are_debug_only(self):
        from lqcdweave.storage.correlator_results import PackedDtOutputWriter
        from lqcdweave.contraction.correlator_cfg import CfgSources
        helper,selected,coords,e,p=self.selected()
        complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',
            physical_lt=128,output_layout='dt_packed')
        expected=complete_calculation(selected,coordinates=coords,
            readers=selected.inputs.loader.input_readers).run().value
        point={'two_point':expected['two_point'][...,0]}
        binding=CfgSources({},(4,19),128)
        normal=PackedDtOutputWriter(complete.plan,'a','unused',binding)
        diagnostic=PackedDtOutputWriter(complete.plan,'a','unused',binding,debug=True)
        with patch('lqcdweave.storage._cfg_correlator_output.np.isfinite',
                   side_effect=AssertionError('normal output scanned')):
            normal.consume(0,point)
            with self.assertRaisesRegex(AssertionError,'normal output scanned'):
                diagnostic.consume(0,point)
        diagnostic=PackedDtOutputWriter(complete.plan,'a','unused',binding,debug=True)
        diagnostic.consume(0,point)
        np.testing.assert_array_equal(normal.sums[('two_point',)],diagnostic.sums[('two_point',)])

    def selected(self,loops=False):
        helper=test_complete_cfg_output.CompleteOutputTests()
        selected,coords,e,p=helper.selected()
        if loops:
            from lqcdweave.contracts.correlator.api import OutputNode
            root=selected.expanded.output_recipe.root
            recipe=replace(selected.expanded.output_recipe,root=replace(root,
                keys=(*root.keys,'vacuum_sink','vacuum_source'),
                children=(*root.children,OutputNode('constant_array',np.array([1+2j,3+4j]),shape=(2,)),
                          OutputNode('constant_array',np.array([5+6j,7+8j]),shape=(2,)))))
            expanded=replace(selected.expanded,output_recipe=recipe)
            selected=replace(selected,expanded=expanded,path=replace(selected.path,expanded=expanded))
        return helper,selected,coords,e,p

    def test_saved_whole_dt_and_compact_loops_without_runtime_planning(self):
        helper,selected,coords,e,p=self.selected(loops=True)
        expected=complete_calculation(selected,coordinates=coords,readers=selected.inputs.loader.input_readers).run().value
        records=[];closed=[]
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128,
                output_layout='dt_packed',dt_coordinates=tuple(range(1,128)))
            archive=Path(directory)/'plan.json.gz';save_calculation_plan(archive,complete)
            factory=helper.factory(complete.plan,e,p,records,closed,physical=True)
            with ExitStack() as blocked:
                for name in ('lqcdweave.contraction.correlator.expand_correlator',
                    'lqcdweave.contraction.correlator.find_contraction_path',
                    'lqcdweave.contraction.planning.api.audit_cfg_plan',
                    'lqcdweave.contraction.correlator_dt_output.configure_packed_dt'):
                    blocked.enter_context(patch(name,side_effect=AssertionError(name)))
                result=calculate(plan_archive=archive,configuration=iter(('a','b')),data=factory,output=directory)
            self.assertEqual(result.report['completed'],('a','b'))
            for cfg,scale in [('a',1),('b',16)]:
                destination=Path(directory)
                actual=load_correlator_output(destination,('two_point',),dt=True,configuration=cfg)
                self.assertEqual(actual.shape[-1],127)
                np.testing.assert_allclose(actual[...,14],expected['two_point'][...,0]*scale,rtol=3e-12,atol=3e-12)
                np.testing.assert_allclose(actual[...,112],expected['two_point'][...,1]*scale,rtol=3e-12,atol=3e-12)
                self.assertTrue(np.isnan(actual[...,0]).all())
                for field in ('vacuum_sink','vacuum_source'):
                    np.testing.assert_array_equal(load_correlator_output(destination,(field,),configuration=cfg),expected[field])
                spec=read_dataset_metadata(destination,cfg=cfg)['spec']
                definition=spec.document['definition']
                self.assertEqual(spec.shape,())
                self.assertEqual(spec.axis_names,())
                self.assertIsNone(spec.document['dtype'])
                self.assertEqual(definition['coordinate_shape'],list(complete.plan.coordinate_shape))
                self.assertEqual(definition['coordinates'],[dict(instance.coordinate) for instance in complete.plan.instances])
                self.assertEqual(definition['physical_dt'],list(range(1,128)))
                self.assertEqual(definition['measured_pair_counts'][15],1)
                self.assertEqual(definition['measured_pair_counts'][113],1)
                with DatasetReader(destination,cfg=cfg) as reader:
                    for field in definition['fields']:
                        role='dt' if field['averaged_dt'] else 'one_point'
                        stored=reader.read_field((role,*field['path']))
                        extent=(len(definition['physical_dt']) if field['averaged_dt']
                            else len(definition['physical_times']))
                        self.assertEqual(stored.shape,(*field['shape'],extent))
                        self.assertEqual(stored.dtype.str,field['dtype'])
            self.assertEqual({path.name for path in Path(directory).glob('*.h5')},{'a.h5','b.h5'})
            self.assertEqual(closed,['a','b'])

    def test_final_resource_audit_and_invalid_coordinates(self):
        from lqcdweave.contraction.planning.api import audit_cfg_plan
        helper,selected,coords,e,p=self.selected()
        base=complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128)
        complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128,output_layout='dt_packed')
        self.assertLessEqual(audit_cfg_plan(complete.plan)['peak_host_bytes'],complete.plan.peak_host_bytes)
        with self.assertRaisesRegex(ValueError,'packing or reconstruction'):
            audit_cfg_plan(replace(complete.plan,output_host_bytes=base.plan.output_host_bytes))
        for coordinates in ((0,), (128,), (1,1), ()):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128,
                        output_layout='dt_packed',dt_coordinates=coordinates)

    def test_write_failure_never_publishes_root(self):
        helper,selected,coords,e,p=self.selected(loops=True);closed=[]
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128,
                output_layout='dt_packed',output=directory)
            complete=replace(complete,options=dict(complete.options,configuration='a',
                source_factory=helper.factory(complete.plan,e,p,[],closed,physical=True)))
            from lqcdweave.storage.dataset import read_dataset_metadata
            def fail(writer,path,array,**options):
                raise OSError('disk full')
            with patch.object(DatasetWriter,'write_field',fail):
                with self.assertRaisesRegex(OSError,'disk full'): complete.run()
            with self.assertRaises(FileNotFoundError): read_dataset_metadata(directory,cfg='a')
            self.assertFalse((Path(directory)/'a.h5').exists())
            self.assertEqual(closed,['a'])


if __name__=='__main__': unittest.main()
