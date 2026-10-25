"""Complete delivery, native numerical reference, lazy cfgs and publication."""
from contextlib import contextmanager,ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_structured_correlator import fixture,task
from correlator.execution.test_cfg_runtime import SafeReader
from lqcdweave.contraction.correlator import calculate,save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation,CfgSources
from lqcdweave.contracts.correlator.api import describe_coordinates


# Correctness checks require a feasible route, not production-scale optimization.
_TEST_SEARCH = dict(root_width=1, branch_width=1, max_states=1000, component_workers=1)


class CompleteOutputTests(unittest.TestCase):
    def selected(self,request=None):
        bind,e,p=fixture();request={'two_point':task(structured=True,flavor='I0')} if request is None else request
        mapping={'S^q':p,r'S^q_\mathrm{local}':p}
        selected=calculate(request,vertex_map=bind,propagator_map=mapping,configuration='cfg',usedNe=3,
            time_axes={0:1,1:0},optimize='meson_matrix',search=_TEST_SEARCH,memory_limit=1<<30,execute=False)
        coords,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1}])
        return selected,coords,e,p

    def factory(self,plan,e,p,records,closed,*,physical=False):
        @contextmanager
        def bind(cfg):
            prop=SafeReader(p,cfg,1 if cfg=='a' else 2,records)
            elem=SafeReader(e,cfg,1 if cfg=='a' else 2,records)
            readers={r.reader_role:prop if r.family=='propagator' else elem
                for primary in plan.requests.values() for r in (primary,*primary.aliases)}
            try: yield CfgSources(readers,(4,19) if physical else (),128 if physical else None,
                                 dict(test='independent synthetic cfg binding'))
            finally: closed.append(cfg)
        return bind

    def test_output_pack_and_dt128_are_lossless_and_saved_before_load(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output
        selected,coords,e,p=self.selected();records=[];closed=[]
        expected=complete_calculation(selected,coordinates=coords,readers=selected.inputs.loader.input_readers).run().value['two_point']
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',
                output=directory,physical_lt=128)
            archive=Path(directory)/'complete.json.gz';save_calculation_plan(archive,complete)
            factory=self.factory(complete.plan,e,p,records,closed,physical=True)
            with ExitStack() as blocked:
                for name in ('lqcdweave.contraction.correlator.expand_correlator',
                    'lqcdweave.contraction.correlator.find_contraction_path',
                    'lqcdweave.contraction.planning.api.audit_cfg_plan'):
                    blocked.enter_context(patch(name,side_effect=AssertionError(name)))
                output=calculate(plan_archive=archive,configuration=iter(('a','b')),data=factory,output=directory)
            for cfg,scale in [('a',1),('b',16)]:
                raw=load_correlator_output(Path(directory),('two_point',),configuration=cfg)
                dt=load_correlator_output(Path(directory),('two_point',),dt=True,configuration=cfg)
                np.testing.assert_allclose(raw,expected*scale,rtol=3e-12,atol=3e-12)
                np.testing.assert_allclose(dt[...,15],expected[...,0]*scale,rtol=3e-12,atol=3e-12)
                np.testing.assert_allclose(dt[...,113],expected[...,1]*scale,rtol=3e-12,atol=3e-12)
                self.assertTrue(np.isnan(dt[...,0]).all())
            self.assertEqual(output.report['completed'],('a','b'))
            self.assertEqual(closed,['a','b'])
            self.assertGreater(complete.plan.output_host_bytes,complete.plan.output_words*8)
            for name in ('peak_host_bytes','output_host_bytes','prefetch_host_bytes'):
                self.assertEqual(complete.plan.statistics[name],getattr(complete.plan,name))

    def test_memory_iterator_returns_one_cfg_and_close_releases_claimed_readers(self):
        selected,coords,e,p=self.selected();records=[];closed=[];claims=[]
        complete=complete_calculation(selected,coordinates=coords)
        def configurations():
            for cfg in ('a','b','c','d'):
                claims.append(cfg);yield cfg
        complete=replace(complete,options=dict(complete.options,configuration=configurations(),
            source_factory=self.factory(complete.plan,e,p,records,closed),prefetch_host_bytes=0))
        running=complete.run();self.assertEqual(claims,[])
        cfg,value=next(running.value);self.assertEqual(cfg,'a');self.assertEqual(claims,['a'])
        self.assertEqual(value['two_point'].shape[-1],2)
        running.value.close();self.assertEqual(closed,['a']);self.assertEqual(claims,['a'])

    def test_write_failure_never_publishes_complete_cfg(self):
        from lqcdweave.storage.dataset import DatasetWriter
        selected,coords,e,p=self.selected();records=[];closed=[]
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',output=directory,
                source_factory=None)
            complete=replace(complete,options=dict(complete.options,configuration='a',
                source_factory=self.factory(complete.plan,e,p,records,closed),prefetch_host_bytes=0))
            from lqcdweave.storage.dataset import read_dataset_metadata
            original=DatasetWriter.write_field
            calls=0
            def failing(writer,path,value,**options):
                nonlocal calls
                calls+=1
                if calls==2: raise OSError('slow disk failed')
                return original(writer,path,value,**options)
            with patch.object(DatasetWriter,'write_field',failing):
                with self.assertRaisesRegex(OSError,'slow disk failed'): complete.run()
            with self.assertRaises(FileNotFoundError): read_dataset_metadata(directory,cfg='a')
            self.assertFalse((Path(directory)/'a.h5').exists())
            self.assertEqual(closed,['a'])

    def test_ordinary_contraction_uses_same_dataset_delivery(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output
        bind,e,p=fixture();request=task(flavor='charged')
        options=dict(vertex_map=bind,propagator_map={'S^q':p},configuration='cfg',usedNe=3,
                     optimize='none',time_pairs=[{0:1,1:0},{0:0,1:1}],memory_limit=1<<30)
        expected=calculate(request,**options).value
        with tempfile.TemporaryDirectory() as directory:
            complete=calculate(request,output=directory,execute=False,**options)
            archive=Path(directory)/'ordinary.json.gz';save_calculation_plan(archive,complete)
            restored=calculate(plan_archive=archive,configuration='cfg',vertex_map=e,
                propagator_map={'S^q':p},output=directory,memory_limit=1<<30)
            actual=load_correlator_output(Path(directory),configuration='cfg')
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
            self.assertEqual(restored.report['completed'],('cfg',))

    def test_final_audit_counts_saved_output_and_dt_buffers(self):
        from lqcdweave.contraction.planning.api import audit_cfg_plan
        selected,coords,e,p=self.selected()
        complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',physical_lt=128)
        self.assertLessEqual(audit_cfg_plan(complete.plan)['peak_host_bytes'],complete.plan.peak_host_bytes)
        with self.assertRaisesRegex(ValueError,'packing or reconstruction'):
            audit_cfg_plan(replace(complete.plan,output_host_bytes=complete.plan.output_words*8))

    def test_single_pair_scan_uses_shared_catalog_and_retains_scan_axis(self):
        bind,e,p=fixture()
        with patch('lqcdweave.contraction.expansion.matrix_workflow.expand_matrix_rows',
                   side_effect=AssertionError('old row expansion')):
            complete=calculate(task(structured=True),vertex_map=bind,propagator_map={'S^q':p},
                configuration='cfg',usedNe=3,time_pairs=[{0:1,1:0}],optimize='meson_matrix',search=_TEST_SEARCH,
                memory_limit=1<<30,execute=False)
        self.assertEqual(complete.plan.coordinate_shape,(1,))
        self.assertEqual(complete.run().value.shape,(2,2,1))

    def test_packing_preserves_mixed_scalar_precision_without_numeric_cast(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output
        from lqcdweave.contracts.correlator.api import OutputNode
        selected,coords,e,p=self.selected(task(structured=True))
        # This test adds literal fields after planning to exercise lossless
        # packing, independently of the default contraction/output dtype.
        recipe=replace(selected.expanded.output_recipe,dtype=None,root=OutputNode('dict',
            keys=('answer','integer','single'),children=(selected.expanded.output_recipe.root,
                OutputNode('constant',np.int64(2**60+3)),
                OutputNode('constant_array',np.array([.2,.3],dtype=np.float32),shape=(2,)))))
        expanded=replace(selected.expanded,output_recipe=recipe)
        selected=replace(selected,expanded=expanded,path=replace(selected.path,expanded=expanded))
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',output=directory,
                readers=selected.inputs.loader.input_readers)
            complete.run()
            integer=load_correlator_output(Path(directory),('integer',),configuration='cfg')
            single=load_correlator_output(Path(directory),('single',),configuration='cfg')
            self.assertEqual(integer.dtype,np.int64);self.assertTrue(np.all(integer==2**60+3))
            self.assertEqual(single.dtype,np.float32)
            np.testing.assert_array_equal(single,np.array([[.2,.2],[.3,.3]],dtype=np.float32))

    def test_four_partitions_connected_disconnected_and_measured_dt_means(self):
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.single_hadron.api import Operator,OperatorTerm
        from lqcdweave.operators.correlator.api import remove_disconneted_diagram
        from lqcdweave.contraction.correlator_output import load_correlator_output
        from lqcdweave.storage.dataset import read_dataset_metadata
        rng=np.random.default_rng(731)
        def random(shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
        component=ElementalComponent(0)
        e=ArrayElemental({'v2v':random((1,1,3,7,7))},(component,),
            phase_convention='anchor',cfg='cfg',data_identity='three-times-elemental')
        p=ArrayPropagator({'vsv':random((3,3,4,4,7,7))},time_convention='absolute',
            cfg='cfg',data_identity='three-times-propagator')
        op=Operator.from_terms('a',(OperatorTerm(5,component,0,.6-.25j),))
        vertices={dagger:Meson(e,op,dagger,phase_convention='anchor') for dagger in (False,True)}
        def fields(structured):
            full=task(structured=structured,flavor='I0',names=('a',))
            connected=remove_disconneted_diagram(full,[r'S^q_\mathrm{local}'])
            return {'two_point':full,'connected':connected,'disconnected':full-connected}
        pairs=[{0:1,1:0},{0:2,1:1},{0:2,1:0}]
        options=dict(vertex_map=lambda row:vertices[row.dagger],
            propagator_map={'S^q':p,r'S^q_\mathrm{local}':p},configuration='cfg',usedNe=7,
            partitions=(0,1,3,4,7),memory_limit=1<<30)
        expected=calculate(fields(False),optimize='none',time_pairs=pairs,**options).value
        selected=calculate(fields(True),optimize='meson_matrix',search=_TEST_SEARCH,time_axes=pairs[0],execute=False,**options)
        coords,_=describe_coordinates(time_pairs=pairs)
        binding=CfgSources(selected.inputs.loader.input_readers,(2,7,12),128,
            lattice=(16,16,16,128),ensemble='independent-small-reference')
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',output=directory,
                physical_lt=128,readers=binding)
            complete.run()
            destination=Path(directory)
            for field in expected:
                raw=load_correlator_output(destination,(field,),configuration='cfg')
                dt=load_correlator_output(destination,(field,),dt=True,configuration='cfg')
                np.testing.assert_allclose(raw,expected[field],rtol=3e-12,atol=3e-12)
                np.testing.assert_allclose(dt[...,5],expected[field][...,:2].mean(axis=-1),rtol=3e-12,atol=3e-12)
                np.testing.assert_allclose(dt[...,10],expected[field][...,2],rtol=3e-12,atol=3e-12)
                self.assertTrue(np.isnan(dt[...,0]).all())
            counts=read_dataset_metadata(destination,cfg='cfg')['spec'].document['definition']['measured_pair_counts']
            self.assertEqual(counts[5],2);self.assertEqual(counts[10],1)

    def test_repeated_points_keep_distinct_dataset_occurrences(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output
        selected,coords,e,p=self.selected()
        coords,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:1,1:0}])
        with tempfile.TemporaryDirectory() as directory:
            complete=complete_calculation(selected,coordinates=coords,output_mode='dataset',output=directory,
                readers=selected.inputs.loader.input_readers)
            complete.run()
            raw=load_correlator_output(Path(directory),('two_point',),configuration='cfg')
            np.testing.assert_array_equal(raw[...,0],raw[...,1])

    def test_archive_rejects_planning_changes_before_opening_sources(self):
        for override in ({'time_pairs':[{0:1,1:0}]},{'usedNe':120},{'partitions':[0,16,70]}):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError,'fixes planning options'):
                    calculate(plan_archive='unused.json.gz',configuration='cfg',**override)


if __name__=='__main__': unittest.main()
