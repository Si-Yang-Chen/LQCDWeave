"""Public complete plan entry does not require a correlator after loading."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_structured_correlator import fixture,task
from lqcdweave.contraction.correlator import calculate,save_calculation_plan


class CompleteFrontendTests(unittest.TestCase):
    def inputs(self):
        bind,elemental,propagator=fixture()
        return bind,elemental,propagator,{'S^q':propagator,r'S^q_\mathrm{local}':propagator}

    def test_selected_path_saved_once_and_direct_archive_runs_without_expansion(self):
        bind,elemental,propagator,mapping=self.inputs()
        points=[{0:1,1:0},{0:0,1:1}]
        request=task(structured=True,flavor='I0')
        selected=calculate(request,vertex_map=bind,propagator_map=mapping,configuration='cfg',
            usedNe=3,optimize='meson_matrix',time_pairs=points,memory_limit=1<<30,execute=False)
        expected=calculate(request,vertex_map=bind,propagator_map=mapping,configuration='cfg',
            usedNe=3,optimize='none',time_pairs=points,memory_limit=1<<30).value
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'cfg-plan.json.gz'
            save_calculation_plan(archive,selected)
            bind2,elemental2,propagator2,mapping2=self.inputs()
            with ExitStack() as blocked:
                for name in (
                    'lqcdweave.contraction.correlator.expand_correlator',
                    'lqcdweave.contraction.correlator.find_contraction_path',
                    'lqcdweave.contraction.correlator._resolve_cost_profile',
                    'lqcdweave.contraction.planning.api.plan_scene_storage',
                    'lqcdweave.contraction.planning.api.rebind_scene_path',
                    'lqcdweave.contraction.planning.api.audit_cfg_plan',
                    'lqcdweave.contraction.planning.api.estimate_matrix_action',
                ):
                    blocked.enter_context(patch(name,side_effect=AssertionError(name)))
                result=calculate(plan_archive=archive,configuration='cfg',
                    propagator_map=mapping2,vertex_map=elemental2,memory_limit=1<<30)
        np.testing.assert_allclose(result.value,expected,rtol=3e-12,atol=3e-12)
        self.assertIn('loading_seconds',result.options)
        self.assertNotIn('predicted_seconds',result.report)

    def test_unrestorable_callback_is_rejected_during_offline_completion(self):
        from dataclasses import replace
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        bind,elemental,propagator,mapping=self.inputs()
        selected=calculate(task(),vertex_map=bind,propagator_map=mapping,
            configuration='cfg',usedNe=3,optimize='none',execute=False,memory_limit=1<<30)
        prep=next(iter(selected.inputs.loader.preparations))
        selected.inputs.loader.preparations[prep]=replace(selected.inputs.loader.preparations[prep],recipe=None)
        with self.assertRaisesRegex(ValueError,'described preparations'):
            complete_calculation(selected)

    def test_new_process_uses_only_archive_and_reader_capabilities(self):
        import subprocess,sys
        bind,elemental,propagator,mapping=self.inputs()
        selected=calculate(task(structured=True,flavor='I0'),vertex_map=bind,
            propagator_map=mapping,configuration='cfg',usedNe=3,optimize='meson_matrix',
            time_pairs=[{0:1,1:0},{0:0,1:1}],memory_limit=1<<30,execute=False)
        expected=selected.run().value
        script='''
import sys,numpy as np
from unittest.mock import patch
from contextlib import ExitStack
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contraction.correlator import calculate
rng=np.random.default_rng(771)
def random(shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
e=ArrayElemental({'v2v':random((1,1,2,3,3))},(ElementalComponent(0),),
    phase_convention='anchor',cfg='cfg',data_identity='structured-elemental')
p=ArrayPropagator({'vsv':random((2,2,4,4,3,3))},
    time_convention='absolute',cfg='cfg',data_identity='structured-propagator')
with ExitStack() as blocked:
    for name in ('lqcdweave.contraction.correlator.expand_correlator',
                 'lqcdweave.contraction.correlator.find_contraction_path',
                 'lqcdweave.contraction.correlator._resolve_cost_profile',
                 'lqcdweave.contraction.planning.api.plan_scene_storage',
                 'lqcdweave.contraction.planning.api.rebind_scene_path',
                 'lqcdweave.contraction.planning.api.audit_cfg_plan',
                 'lqcdweave.contraction.planning.api.estimate_matrix_action'):
        blocked.enter_context(patch(name,side_effect=AssertionError(name)))
    r=calculate(plan_archive=sys.argv[1],configuration='cfg',vertex_map=e,
        propagator_map={'S^q':p,r'S^q_\\mathrm{local}':p},memory_limit=1<<30)
np.save(sys.argv[2],r.value,allow_pickle=False)
'''
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'complete.json.gz'; output=Path(directory)/'values.npy'
            save_calculation_plan(archive,selected)
            result=subprocess.run([sys.executable,'-X','utf8','-c',script,str(archive),str(output)],
                text=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,0,result.stderr)
            np.testing.assert_allclose(np.load(output,allow_pickle=False),expected,rtol=3e-12,atol=3e-12)


if __name__=='__main__': unittest.main()
