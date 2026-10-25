"""Selected-route tiling preserves contracts and audits physical budgets."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,ContractionProblem,SourceSpec,SourcePreparation,TensorSpec,
    FactorizedWideReduction,AxisPattern,
)
from lqcdweave.contraction.planning.api import ContractionState,CostProfile
from lqcdweave.contraction.planning.factorized_reduction import describe_factorized_action,replay_factorized_action,factorized_result_name
from lqcdweave.contraction.planning.factorized_tiles import retile_factorized_reductions
from lqcdweave.contraction.planning.storage import plan_scene_storage
from lqcdweave.storage.contraction_plans.api import encode_plan_value,decode_plan_value


def fixture(*,limit=1<<26,shared=1):
    equation='lak,lbk,ca,db->cld';shapes=((shared,192,256),(shared,192,256),(1,192),(1,192))
    specs={str(i):TensorSpec(s,'complex64') for i,s in enumerate(shapes)}
    term=ContractionTerm(tuple(specs),equation);l,a,k=term.subscripts.inputs[0];b=term.subscripts.inputs[1][1]
    term=replace(term,sum_barriers=(((k,),(a,b)),((a,),(k,b)),((b,),(a,k))))
    problem=ContractionProblem(specs,(term,),{key:SourceSpec(key,0,read_bytes=0) for key in specs},backend='cupy')
    profile=CostProfile();state=ContractionState.initial(problem)
    for key in specs:state=state.load(key,profile,limit)
    action,_,_,_=describe_factorized_action(state,0,(0,1,2,3),(0,1),profile=profile,
        output_order=term.subscripts.output,tile_elements=4096)
    state=replay_factorized_action(state,action,profile,memory_limit=limit)
    for key in sorted(state.loaded-set(state.results)):state=state.release(key)
    preps={key:SourcePreparation((),dtype='complex64') for key in specs}
    plan=plan_scene_storage(state,{},preps,profile=profile,memory_limit=limit,host_memory_limit=limit)
    assert plan.feasible
    return plan,profile


class FactorizedTileTests(unittest.TestCase):
    def test_public_archive_rejects_explicit_retile_override(self):
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.calculation import Calculation
        with patch('lqcdweave.contraction.correlator.load_calculation_plan',return_value=Calculation()) as load:
            calculate(plan_archive='fixed',factorized_retile=None,execute=False)
            self.assertEqual(load.call_count,1)
            for override in (False,True):
                with self.assertRaisesRegex(ValueError,'factorized_retile'):
                    calculate(plan_archive='fixed',factorized_retile=override,execute=False)
            self.assertEqual(load.call_count,1)
        for bad in (1,0,'auto'):
            with self.assertRaisesRegex(TypeError,'factorized_retile'):
                calculate(plan_archive='fixed',factorized_retile=bad,execute=False)

    def test_closed_option_does_not_call_retile(self):
        from types import SimpleNamespace
        from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
        old,profile=fixture()
        path=SimpleNamespace(batch_paths=(),plan=old,profile=profile,prediction={'kernel_batching':False})
        with patch('lqcdweave.contraction.planning.factorized_tiles.retile_factorized_reductions',side_effect=AssertionError):
            self.assertIs(lower_independent_gemm_path(path,target_runtime='cuda',factorized_retile=False),path)

    def test_whole_width_is_named_saved_and_charged(self):
        old,profile=fixture();new,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda')
        self.assertTrue(stats['applied']);self.assertTrue(new.feasible)
        before=next(a for a in old.state.actions if a.core is not None);after=next(a for a in new.state.actions if a.core is not None)
        self.assertEqual(before.core.tile_width,4096);self.assertEqual(after.core.tile_width,36864)
        self.assertEqual(after.core.tiles,1);self.assertEqual(after.core.kernel_calls,2)
        self.assertEqual(before.operands,after.operands);self.assertEqual(before.equation,after.equation)
        self.assertNotEqual(before.tensor,after.tensor)
        self.assertGreater(after.core.partial_bytes,before.core.partial_bytes)
        self.assertLess(new.seconds,old.seconds)
        self.assertEqual(type(new.state.pending),tuple);self.assertEqual(type(new.state.results),tuple)
        restored=decode_plan_value(encode_plan_value(after.core));self.assertEqual(restored,after.core)
        self.assertEqual(factorized_result_name(after.operands,restored),after.tensor)
        self.assertEqual(decode_plan_value(encode_plan_value(new.state)),new.state)

    def test_tight_budget_selects_medium_width_and_really_fits(self):
        old,profile=fixture(limit=2<<20)
        new,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda')
        self.assertTrue(stats['applied']);self.assertTrue(new.feasible)
        core=next(a.core for a in new.state.actions if a.core is not None)
        self.assertEqual(core.tile_width,8192)
        self.assertLessEqual(new.peak_device_bytes,2<<20)

    def test_retained_lanes_are_included_in_partial_limit(self):
        old,profile=fixture(limit=1<<28,shared=64)
        new,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda')
        self.assertTrue(stats['applied'])
        core=next(a.core for a in new.state.actions if a.core is not None)
        self.assertEqual(core.tile_width,8192)
        self.assertEqual(core.partial_bytes,8<<20)

    def test_cpu_unknown_runtime_and_explicit_plan_keep_original(self):
        old,profile=fixture()
        for runtime in ('cpu','hip',None):
            self.assertIs(retile_factorized_reductions(old,profile=profile,target_runtime=runtime)[0],old)
        explicit=replace(old,explicit_resources=True)
        self.assertIs(retile_factorized_reductions(explicit,profile=profile,target_runtime='cuda')[0],explicit)
        limited,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda',max_tile_elements=4096)
        self.assertIs(limited,old)

    def test_contract_error_is_not_hidden_as_budget_fallback(self):
        old,profile=fixture();actions=tuple(replace(a,core=replace(a.core,output_dtype='complex128')) if a.core is not None else a for a in old.state.actions)
        invalid=replace(old,state=old.state.with_action_history(actions))
        with self.assertRaisesRegex(ValueError,'invalid saved core'):
            retile_factorized_reductions(invalid,profile=profile,target_runtime='cuda')

    def test_failed_final_physical_audit_keeps_original_plan(self):
        old,profile=fixture()
        with patch('lqcdweave.contraction.planning.storage.plan_scene_storage',
                   return_value=replace(old,feasible=False)):
            new,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda')
        self.assertIs(new,old)
        self.assertEqual(stats['reason'],'no_physical_cost_improvement')

    def test_new_bound_and_original_untiled_semantics(self):
        old,_=fixture();core=next(a.core for a in old.state.actions if a.core is not None)
        self.assertEqual(replace(core,tile_elements=65536).tile_width,36864)
        for width in (0,65537,True,1.5):
            with self.assertRaises(ValueError):replace(core,tile_elements=width)
        with self.assertRaises(ValueError):replace(core,tile_elements=None)
