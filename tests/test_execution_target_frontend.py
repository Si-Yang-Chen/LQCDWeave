"""Offline target selection and replay retain the public calculation interface."""
import builtins
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
import tempfile
from io import StringIO
import unittest
from unittest.mock import patch
import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.memory import TargetHardwareMemory, MemoryLimits
from lqcdweave.contraction.performance.api import register_execution_target, CostProfile
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.storage.contraction_plans.api import load_cfg_plan, PlanArchiveError


def inputs():
    binding, elemental, propagator = fixture()
    return dict(correlator=task(structured=True, flavor='I0'), vertex_map=binding,
        propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
        configuration='cfg', usedNe=3, time_axes={0:1,1:0}, execute=False,
        search={'max_states':128,'deadline_seconds':2})


class TargetFrontendTests(unittest.TestCase):
    def test_legacy_memory_description_preserves_supplied_hip_build(self):
        from types import SimpleNamespace
        from lqcdweave.contraction.correlator_target import resolve_planning_target
        backend=SimpleNamespace(__name__='cupy',cuda=SimpleNamespace(
            runtime=SimpleNamespace(is_hip=True)))
        memory=TargetHardwareMemory(host_available_bytes=1<<28,
            device_total_bytes=1<<28,device_free_bytes=1<<28)
        with patch('lqcdweave.contraction.correlator_target.observe_execution_target',
                   side_effect=AssertionError('offline device probe')):
            target,_,_,summary=resolve_planning_target(memory,backend,
                memory_limit=1<<26,host_memory_limit=1<<27,profile=CostProfile())
        self.assertEqual(target.runtime,'hip')
        self.assertEqual(summary['runtime'],'hip')

    def test_live_calibration_uses_remaining_resources_and_cache(self):
        from lqcdweave.contraction.correlator_target import calibrate_if_needed
        from lqcdweave.adapters.execution_target import observe_execution_target
        from lqcdweave.contraction.performance.api import resolve_target_profile
        from lqcdweave.contraction.execution.api import probe_compute_profile
        limits=MemoryLimits(1<<20,None)
        target=observe_execution_target(np,limits)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ',{'LQCDWEAVE_PROFILE_DIR':directory}):
            document=resolve_target_profile(target)
            with patch('lqcdweave.contraction.correlator_target.resolve_diagram_memory_limits',
                       return_value=MemoryLimits(32768,None)), patch(
                       'lqcdweave.contraction.execution.api.probe_compute_profile',
                       wraps=probe_compute_profile) as probe:
                measured=calibrate_if_needed(target,limits,document,backend=np,
                    dtypes=('complex128',),profile=None,reserved_host_bytes=1024)
                self.assertEqual(probe.call_args.kwargs['max_bytes'],32768-1024)
            self.assertEqual(measured['calibration_status'],'partially-calibrated')
            with patch('lqcdweave.contraction.execution.api.probe_compute_profile',
                       side_effect=AssertionError('cache hit measured again')):
                cached=calibrate_if_needed(target,limits,document,backend=np,
                    dtypes=('complex128',),profile=None)
            self.assertEqual(cached['profile'],measured['profile'])
            self.assertNotIn('contract',cached['coverage']['missing_operations'])
            self.assertIn('preparation',cached['coverage']['missing_operations'])
            offline=resolve_target_profile(replace(target,hardware_fingerprint='unmeasured'))
            self.assertEqual(offline['profile'],measured['profile'])

    def test_offline_gpu_has_no_runtime_or_resource_observation(self):
        target=ExecutionTarget('test-gpu','cupy','cuda',TargetHardwareMemory(
            host_available_bytes=1<<29,device_total_bytes=1<<28,device_free_bytes=1<<28))
        original=builtins.__import__
        def guard(name,*args,**kwargs):
            if name=='cupy' or name.startswith('cupy.'):
                raise AssertionError('offline planning imported CuPy')
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'LQCDWEAVE_PROFILE_DIR':directory}):
            register_execution_target(target)
            with ExitStack() as stack:
                stack.enter_context(patch('builtins.__import__',side_effect=guard))
                stack.enter_context(patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',side_effect=AssertionError('local resources')))
                stack.enter_context(patch('lqcdweave.contraction.correlator_target.observe_execution_target',side_effect=AssertionError('local device')))
                stack.enter_context(patch('lqcdweave.contraction.execution.api.probe_compute_profile',side_effect=AssertionError('measurement')))
                summary_output=StringIO()
                with patch('lqcdweave.contraction.correlator_target.logging.getLogger') as logger:
                    logger.return_value.isEnabledFor.return_value=False
                    with redirect_stdout(summary_output):
                        planned=calculate(**inputs(),target_hardware='test-gpu',backend=np,memory_limit=1<<26,host_memory_limit=1<<27)
                self.assertIn('test-gpu [cupy/cuda]',summary_output.getvalue())
                self.assertIn('device budget=67108864 bytes',summary_output.getvalue())
                self.assertIn('uncalibrated',summary_output.getvalue())
                self.assertEqual(planned.plan.state.problem.backend,'cupy')
                self.assertEqual(planned.plan.memory_limit,1<<26)
                self.assertEqual(planned.prediction['execution_target']['name'],'test-gpu')
                complete=complete_calculation(planned)
                path=Path(directory)/'complete.json.gz';save_calculation_plan(path,complete)
                restored=load_cfg_plan(path)
                self.assertEqual(restored.execution_target,target)
                self.assertEqual(restored.backend,'cupy')
                self.assertGreater(restored.peak_device_bytes,0)
                self.assertEqual(restored.target_summary['device_memory_limit'],1<<26)
                self.assertNotIn('debug',restored.target_summary)

    def test_named_target_ignores_planner_resources_and_unknown_fails_early(self):
        target=ExecutionTarget('cpu-offline','numpy','cpu',TargetHardwareMemory(host_available_bytes=1<<28))
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'LQCDWEAVE_PROFILE_DIR':directory}):
            register_execution_target(target)
            with patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',return_value=1):
                one=calculate(**inputs(),target_hardware='cpu-offline')
            with patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',return_value=1<<40):
                two=calculate(**inputs(),target_hardware='cpu-offline')
            self.assertEqual(one.plan.memory_limit,two.plan.memory_limit)
            self.assertEqual(one.plan.state.actions,two.plan.state.actions)
            self.assertEqual(one.path.profile,two.path.profile)
            with patch('lqcdweave.contraction.correlator.expand_correlator',side_effect=AssertionError('expansion')):
                with self.assertRaisesRegex(ValueError,'not found'):
                    calculate(**inputs(),target_hardware='unknown-target')

    def test_run_inherits_debug_and_does_not_recalibrate_or_search(self):
        planned=calculate(**inputs(),debug=True,profile=CostProfile(),memory_limit=1<<28)
        with patch('lqcdweave.contraction.correlator.find_contraction_path',side_effect=AssertionError('search')), patch('lqcdweave.contraction.execution.api.probe_compute_profile',side_effect=AssertionError('calibration')):
            normal=planned.run(debug=False)
            debug=planned.run()
        self.assertTrue(debug.options['debug'])
        self.assertFalse(normal.options['debug'])
        np.testing.assert_allclose(normal.value,debug.value,rtol=1e-12,atol=1e-12)


if __name__=='__main__': unittest.main()
