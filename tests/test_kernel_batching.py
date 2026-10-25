"""The kernel-batching policy is independent of coordinate batches and replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,ContractionTerm,PlannedContraction,SourcePreparation,SourceSpec,TensorSpec,
    validate_executable_resources,
)
from lqcdweave.contraction.planning.api import CostProfile,solve_storage
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
from lqcdweave.contraction.correlator import calculate,save_calculation_plan,load_calculation_plan


def grid(backend='numpy'):
    shapes={'a0':(2,3),'a1':(4,3),'b0':(3,2),'b1':(3,5)}
    specs={key:TensorSpec(shape,'complex64') for key,shape in shapes.items()}
    terms=tuple(ContractionTerm((f'a{i}',f'b{j}'),'ij,jk->ik') for i in range(2) for j in range(2))
    problem=ContractionProblem(specs,(*terms,terms[0]),
        {key:SourceSpec(key,0,read_bytes=0) for key in specs},backend=backend)
    return problem,{key:SourcePreparation(()) for key in specs}


class KernelBatchingTests(unittest.TestCase):
    def solve(self,enabled=None,*,backend='numpy'):
        problem,preps=grid(backend)
        options={} if enabled is None else {'kernel_batching':enabled}
        return solve_storage(problem,{},preps,memory_limit=1<<28,profile=CostProfile(),
            component_workers=1,root_width=1,branch_width=1,
            deadline_seconds=2,max_states=10000,**options)

    def test_omitted_switch_disables_batching_in_frontend_and_solver(self):
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('default must skip discovery')):
            result=self.solve()
            selected=self.native()
        self.assertEqual(result.plan.kernel_batches,())
        self.assertEqual(result.statistics['dense_kernel_lowering']['reason'],'disabled')
        self.assertFalse(selected.options['kernel_batching'])
        self.assertFalse(selected.prediction['kernel_batching'])
        self.assertEqual(selected.plan.kernel_batches,())

    def test_disabled_skips_discovery_preserves_shared_results_and_matches_reference(self):
        enabled=self.solve(True)
        self.assertTrue(enabled.plan.kernel_batches)
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('disabled policy must skip discovery')):
            disabled=self.solve(False)
        self.assertEqual(disabled.plan.kernel_batches,())
        self.assertEqual(disabled.statistics['dense_kernel_lowering']['reason'],'disabled')
        self.assertEqual(disabled.plan.state.results,enabled.plan.state.results)
        self.assertEqual(disabled.plan.state.results[0],disabled.plan.state.results[-1])
        paid=lambda plan:{(a.tensor,a.operands,a.equation) for a in plan.state.actions
                          if a.kind=='contract' and not a.reused}
        self.assertEqual(paid(disabled.plan),paid(enabled.plan))
        validate_executable_resources(disabled.plan)
        rng=np.random.default_rng(461)
        arrays={key:(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)).astype(s.dtype)
                for key,s in disabled.plan.state.problem.specs.items()}
        class Loader:
            def materialize(self,key,*args):return arrays[key]
            def close_all(self):pass
        values=[]
        for plan in (enabled.plan,disabled.plan):
            values.append(replay_scene_storage(plan,Loader()))
        for term,on,off in zip(disabled.plan.state.problem.terms,*values,strict=True):
            expected=arrays[term.inputs[0]]@arrays[term.inputs[1]]
            np.testing.assert_allclose(off,expected,rtol=3e-6,atol=3e-6)
            np.testing.assert_allclose(on,off,rtol=3e-6,atol=3e-6)

    def test_gpu_metadata_disables_cartesian_and_independent_gemm_grouping(self):
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('no GPU batch discovery')):
            selected=self.solve(False,backend='cupy')
        self.assertEqual(selected.plan.kernel_batches,())
        path=PlannedContraction(None,None,None,{'kernel_batching':False},0.)
        with patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                   side_effect=AssertionError('no GPU GEMM grouping')):
            self.assertIs(lower_independent_gemm_path(path,target_runtime='cuda'),path)

    def test_parallel_component_delivery_keeps_batching_disabled(self):
        from correlator.planning.test_parallel_components import fixture
        problem,blocks,preps=fixture()
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('parallel delivery must skip discovery')):
            selected=solve_storage(problem,blocks,preps,memory_limit=1<<28,profile=CostProfile(),
                kernel_batching=False,component_workers=2,root_width=1,branch_width=1,
                deadline_seconds=10,max_states=10000)
        self.assertEqual(selected.statistics['component_count'],2)
        self.assertEqual(selected.plan.kernel_batches,())
        self.assertEqual(selected.statistics['dense_kernel_lowering']['reason'],'disabled')

    def native(self,enabled=None,**options):
        from test_structured_correlator import fixture,task
        bind,elemental,propagator=fixture()
        parameters=dict(vertex_map=bind,propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator},
            configuration='cfg',usedNe=3,time_axes={0:1,1:0},memory_limit=1<<28,
            profile=CostProfile(),search={'component_workers':1,'root_width':1,'branch_width':1,
            'deadline_seconds':2},execute=False)
        parameters.update(options)
        if enabled is not None:
            parameters['kernel_batching']=enabled
        selected=calculate(task(structured=True,flavor='I0'),**parameters)
        return selected

    def test_frontend_default_and_disabled_numerics_and_fixed_archive(self):
        from lqcdweave.contraction.expansion.api import input_readers
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        enabled=self.native(True)
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('no batch discovery')):
            disabled=self.native(False)
        self.assertFalse(disabled.options['kernel_batching'])
        self.assertFalse(disabled.prediction['kernel_batching'])
        self.assertFalse(disabled.plan.kernel_batches)
        expected=disabled.run().value
        np.testing.assert_allclose(enabled.run().value,expected,rtol=2e-11,atol=2e-11)
        readers=input_readers(disabled.inputs)
        complete=complete_calculation(disabled,readers=readers)
        self.assertFalse(complete.plan.statistics['kernel_batching'])
        with TemporaryDirectory() as directory:
            fixed=Path(directory)/'fixed.json.gz'
            cfg=Path(directory)/'cfg.json.gz'
            save_calculation_plan(fixed,disabled)
            save_calculation_plan(cfg,complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('archive must not replan')),\
                 patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                       side_effect=AssertionError('archive must not discover batches')):
                restored=load_calculation_plan(fixed,expected=disabled)
                self.assertFalse(restored.options['kernel_batching'])
                self.assertFalse(restored.plan.kernel_batches)
                np.testing.assert_array_equal(restored.run().value,expected)
                restored=load_calculation_plan(cfg,configuration='cfg',readers=readers)
                self.assertFalse(restored.options['kernel_batching'])
                self.assertTrue(all(not t.plan.kernel_batches for t in restored.plan.templates))
                np.testing.assert_array_equal(restored.run().value,expected)

    def test_coordinate_batches_keep_disabled_policy_for_all_children(self):
        for batch in (1,'auto'):
            with self.subTest(batch=batch),patch(
                    'lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                    side_effect=AssertionError('coordinate batches must preserve policy')):
                selected=self.native(False,time_axes=None,time_pairs=[{0:1,1:0},{0:0,1:1}],batch=batch)
            self.assertTrue(selected.path.batch_paths)
            self.assertFalse(selected.prediction['kernel_batching'])
            for child in selected.path.batch_paths:
                self.assertFalse(child.prediction['kernel_batching'])
                self.assertFalse(child.plan.kernel_batches)
            self.assertEqual(selected.run().value.shape[-1],2)

    def test_archive_replay_preserves_enabled_policy_despite_new_default(self):
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contraction.expansion.api import input_readers
        selected=self.native(True)
        readers=input_readers(selected.inputs)
        complete=complete_calculation(selected,readers=readers)
        expected=complete.run().value
        with TemporaryDirectory() as directory:
            archive=Path(directory)/'enabled.json.gz'
            save_calculation_plan(archive,complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('archive must not replan')):
                restored=calculate(plan_archive=archive,configuration='cfg',
                    vertex_map=readers,
                    propagator_map=selected.expanded.bindings.propagator_map,execute=False)
            self.assertTrue(restored.options['kernel_batching'])
            self.assertTrue(restored.plan.statistics['kernel_batching'])
            np.testing.assert_array_equal(restored.run().value,expected)

    def test_direct_and_recursive_complete_frontend_preserve_policy(self):
        direct=self.native(False,optimize='none',search=None)
        self.assertFalse(direct.prediction['kernel_batching'])
        self.assertFalse(direct.plan.kernel_batches)
        complete=self.native(False,optimize='meson_matrix',time_axes=None,
            time_pairs=[{0:1,1:0},{0:0,1:1}])
        self.assertFalse(complete.options['kernel_batching'])
        self.assertFalse(complete.plan.statistics['kernel_batching'])
        self.assertTrue(all(not t.plan.kernel_batches for t in complete.plan.templates))

    def test_invalid_switch_and_archive_override_fail_before_input_access(self):
        for value in (None,0,1,'false',np.bool_(False)):
            with self.subTest(value=value),self.assertRaisesRegex(TypeError,'kernel_batching'):
                calculate(1,kernel_batching=value,execute=False)
        with self.assertRaisesRegex(ValueError,'kernel_batching'):
            calculate(plan_archive='must-not-open.json',configuration='cfg',
                kernel_batching=True,execute=False)


if __name__=='__main__':unittest.main()
