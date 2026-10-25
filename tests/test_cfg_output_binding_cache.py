"""A fixed cfg handle reuses output code, never numerical evaluations."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np

from correlator.planning import test_cfg_plan
from lqcdweave.contraction.correlator_cfg import CompleteCalculation,bind_point_assemblers
from lqcdweave.contraction.results import binding_cache


class OutputBindingCacheTests(unittest.TestCase):
    def test_plan_identity_and_debug_partition_without_value_reuse(self):
        plan=test_cfg_plan.CfgPlanTests().build()
        calculation=CompleteCalculation(plan)
        with patch.object(binding_cache,'bind_output_assembly',wraps=binding_cache.bind_output_assembly) as bind:
            first=bind_point_assemblers(calculation)
            self.assertIs(first,bind_point_assemblers(replace(calculation,options={'configuration':'other'})))
            self.assertEqual(bind.call_count,len(plan.templates))
            debug=bind_point_assemblers(replace(calculation,options={'debug':True}))
            self.assertIsNot(debug,first)
            self.assertEqual(bind.call_count,2*len(plan.templates))
            changed=replace(calculation,plan=replace(plan))
            self.assertIsNot(bind_point_assemblers(changed),first)
            self.assertEqual(bind.call_count,3*len(plan.templates))
        # The real assemblers are closures of output recipes, not cached values.
        state=plan.templates[0].plan.state
        values=tuple(np.full(state.catalog[key].shape,i+1,dtype=state.catalog[key].dtype)
                     for i,key in enumerate(state.results))
        a=first[0](values)
        b=first[0](tuple(2*v for v in values))
        def leaves(value):
            if isinstance(value,dict):return [a for v in value.values() for a in leaves(v)]
            if isinstance(value,(list,tuple)):return [a for v in value for a in leaves(v)]
            return [np.asarray(value)]
        self.assertTrue(any(not np.array_equal(x,y) for x,y in zip(leaves(a),leaves(b))))
        self.assertEqual(len(calculation._output_bindings._binding),2)


if __name__=='__main__':unittest.main()
