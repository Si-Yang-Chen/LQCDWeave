"""A shared axis program must preserve the complete computation identity."""
from dataclasses import replace
from unittest.mock import patch
import unittest

from lqcdweave.contracts.correlator.api import Action, AxisPattern
from lqcdweave.contraction.planning import cfg_plan


class ExtendedAction(Action):
    pass


class ExtendedAxes(AxisPattern):
    pass


class CfgIdentityTemplateTests(unittest.TestCase):
    def test_shared_programs_preserve_axis_key_equivalence_for_all_id_orders(self):
        patterns = (
            AxisPattern(((0, 1), (1, 2)), (0, 2)),
            AxisPattern(((1, 2), (0, 1)), (0, 2)),
            AxisPattern(((7, 4), (4, 9)), (7, 9)),
            AxisPattern(((0, 1), (0, 1)), ()),
            AxisPattern(((0, 1), (1, 0)), ()),
            AxisPattern(((0, 1), (0, 1)), (0, 1)),
            AxisPattern(((0, 1),), (1, 0)),
            AxisPattern(((0, 1),), (0, 1)),
        )
        actions = tuple(Action('contract', f'out-{copy}-{i}',
            tuple(f'in-{copy}-{j}' for j in range(len(pattern.inputs))), pattern)
            for copy in range(3) for i, pattern in enumerate(patterns))
        programs = cfg_plan._bind_computation_identities(actions, debug=False)
        reference_to_bound, bound_to_reference = {}, {}
        for action, program in zip(actions, programs):
            for ids in ((3,), (9,)) if len(action.operands) == 1 else ((2, 7), (7, 2), (4, 4)):
                reference = cfg_plan._computation_identity(action, ids)
                bound = program(ids)
                self.assertEqual(reference_to_bound.setdefault(reference, bound), bound)
                self.assertEqual(bound_to_reference.setdefault(bound, reference), reference)
        self.assertIs(programs[0], programs[len(patterns)])
        self.assertIsNot(programs[0], programs[1])

    def test_debug_extensions_and_special_metadata_keep_original_binding_path(self):
        equation = AxisPattern(((0, 1), (1, 2)), (0, 2))
        first = Action('contract', 'c0', ('a', 'b'), equation)
        second = replace(first, tensor='c1', operands=('different-a', 'different-b'))
        extended = ExtendedAction('contract', 'c2', ('a', 'b'), equation)
        axes_extended = replace(first, tensor='c3', equation=ExtendedAxes(equation.inputs, equation.output))
        special = replace(first, used_np=1, term_index=7)
        actions = (Action('load', 'a'), first, second, extended, extended,
                   axes_extended, axes_extended, special, replace(special, term_index=8))
        for debug, expected_calls in ((False, 7), (True, 8)):
            with self.subTest(debug=debug):
                with patch.object(cfg_plan, '_bind_computation_identity',
                                  wraps=cfg_plan._bind_computation_identity) as bind:
                    programs = cfg_plan._bind_computation_identities(actions, debug=debug)
                self.assertEqual(bind.call_count, expected_calls)
                self.assertIsNone(programs[0])
                if not debug:
                    self.assertIs(programs[1], programs[2])
                self.assertIsNot(programs[3], programs[4])
                self.assertIsNot(programs[5], programs[6])
                self.assertNotEqual(programs[-2]((1, 2)), programs[-1]((1, 2)))

    def test_new_call_rebuilds_the_axis_registry_without_retaining_an_old_program(self):
        action = Action('contract', 'c', ('a', 'b'), AxisPattern(((0, 1), (1, 2)), (0, 2)))
        first = cfg_plan._bind_computation_identities((action,), debug=False)[0]
        changed = replace(action, equation=AxisPattern(((0, 1), (1, 0)), ()))
        programs = cfg_plan._bind_computation_identities((changed, action), debug=False)
        self.assertIsNot(first, programs[1])
        self.assertNotEqual(programs[0]((2, 3)), programs[1]((2, 3)))


if __name__ == '__main__':
    unittest.main()
