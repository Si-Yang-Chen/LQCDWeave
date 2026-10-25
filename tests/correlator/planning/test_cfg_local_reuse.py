"""Local equality proofs must survive actual ID collisions and overwrites."""
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import Action, AxisPattern, describe_coordinates
from lqcdweave.contraction.planning import cfg_plan


class ExtendedAction(Action):
    pass


class ExtendedAxes(AxisPattern):
    pass


class CfgLocalReuseTests(unittest.TestCase):
    @staticmethod
    def proof(actions, preparations=('a', 'b', 'c'), *, debug=False):
        return cfg_plan._bind_local_computation_reuses(actions,
            cfg_plan._bind_computation_identities(actions, debug=debug),
            tuple(cfg_plan._bind_operands(action.operands) for action in actions),
            preparations, debug=debug)

    @staticmethod
    def trace():
        mm = AxisPattern(((0, 1), (1, 2)), (0, 2))
        ab = Action('contract', 'x', ('a', 'b'), mm, reused=True)
        return (
            Action('load', 'a'), Action('load', 'b'), Action('load', 'c'), ab,
            replace(ab, equation=AxisPattern(((7, 4), (4, 9)), (7, 9)), reused=False),
            replace(ab, tensor='y', operands=('b', 'a'),
                    equation=AxisPattern(((1, 2), (0, 1)), (0, 2))),
            replace(ab, tensor='y'),
            replace(ab, operands=('c', 'b')), ab, ab,
            Action('contract', 'a', ('c',), AxisPattern(((0, 1),), (1, 0))),
            ab, Action('load', 'a'), ab,
            replace(ab, tensor='y', core='opaque-core'),
            replace(ab, tensor='y'), replace(ab, tensor='y'),
        )

    def test_formal_equalities_match_reference_for_every_actual_input_id_order(self):
        actions = self.trace()
        proof = self.proof(actions)
        self.assertEqual(proof, (False, False, False, False, True, False, True,
                                False, False, True, False, False, False, True,
                                False, False, True))
        # Retain the reference identity dictionary across coordinates. Formal
        # input names can collapse onto one actual ID, or reverse their order.
        expected_ids, actual_ids = {}, {}
        expected_next = actual_next = 100
        for inputs in product((1, 2, 3), repeat=3):
            expected = dict(zip(('a', 'b', 'c'), inputs))
            actual = dict(expected)
            for action, repeat in zip(actions, proof, strict=True):
                if action.kind == 'load':
                    continue
                key = cfg_plan._computation_identity(action, tuple(expected[k] for k in action.operands))
                if key not in expected_ids:
                    expected_ids[key] = expected_next
                    expected_next += 1
                expected[action.tensor] = expected_ids[key]
                if not repeat:
                    key = cfg_plan._computation_identity(action, tuple(actual[k] for k in action.operands))
                    if key not in actual_ids:
                        actual_ids[key] = actual_next
                        actual_next += 1
                    actual[action.tensor] = actual_ids[key]
                self.assertEqual(actual, expected)
        self.assertEqual(actual_ids, expected_ids)

    def test_special_and_extension_writes_invalidate_a_previous_output(self):
        first = self.trace()[3]
        special = (
            replace(first, used_np=1), replace(first, join_id='join'),
            replace(first, core='core'),
            ExtendedAction('contract', 'x', first.operands, first.equation),
            replace(first, equation=ExtendedAxes(first.equation.inputs, first.equation.output)),
            replace(first, operands=('a', 'b', 'c'),
                    equation=AxisPattern(((0, 1), (1, 2), (2, 3)), (0, 3))),
        )
        for other in special:
            with self.subTest(action=other):
                self.assertEqual(self.proof((first, first, other, first, first)),
                                 (False, True, False, False, True))

    def test_self_overwrites_and_downstream_consumers_do_not_reuse_old_values(self):
        transpose = Action('contract', 'a', ('a',), AxisPattern(((0, 1),), (1, 0)))
        consumer = self.trace()[3]
        actions = (consumer, transpose, consumer, transpose, consumer, consumer)
        self.assertEqual(self.proof(actions), (False, False, False, False, False, True))

    def test_native_wide_value_identity_preserves_precision_and_ordered_core_roles(self):
        from correlator.execution.test_factorized_wide_reduction import tiled_fixture
        for loop in (False, True):
            _, _, action, _ = tiled_fixture(loop=loop)
            other = replace(action, term_index=29, positions=tuple(reversed(action.positions)))
            self.assertEqual(self.proof((action, other), preparations=action.operands), (False, True))
            axes = AxisPattern(tuple(tuple(axis+100 for axis in labels) for labels in action.equation.inputs),
                               tuple(axis+100 for axis in action.equation.output))
            renamed = replace(other, equation=axes, core=replace(other.core, equation=axes))
            self.assertEqual(self.proof((action, renamed), preparations=action.operands), (False, True))
            reordered = replace(action, operands=tuple(reversed(action.operands)))
            self.assertEqual(self.proof((action, reordered), preparations=action.operands), (False, False))
            different = replace(action, core=replace(action.core, tile_elements=3))
            self.assertEqual(self.proof((action, different, action), preparations=action.operands),
                             (False, False, False))
            precision = replace(action, core=replace(action.core,
                output_dtype='complex128' if action.core.output_dtype is None else None))
            self.assertEqual(self.proof((action, precision, action), preparations=action.operands),
                             (False, False, False))
        narrow = self.trace()[3]
        wide = replace(narrow, wide_accumulation=True)
        self.assertEqual(self.proof((wide, replace(wide, term_index=9), narrow, wide)),
                         (False, True, False, False))

    def test_repeated_three_and_four_factor_wide_values_have_one_live_owner(self):
        from correlator.execution.test_factorized_wide_reduction import tiled_fixture
        from lqcdweave.contracts.correlator.api import ArrayConstant, SourcePreparation, TensorPreparation
        from lqcdweave.contraction.planning.api import ContractionState
        from lqcdweave.contraction.planning.factorized_reduction import describe_factorized_action, replay_factorized_action
        from lqcdweave.contraction.planning.storage import plan_scene_storage
        from lqcdweave.contraction.expansion.api import restore_input_loader
        from lqcdweave.contraction.execution.api import execute_cfg_plan
        for loop in (False, True):
            before, _, original, profile = tiled_fixture(loop=loop)
            term = before.problem.terms[0]
            problem = replace(before.problem, backend='numpy', terms=(term, term),
                sources={key: replace(source, read_bytes=0) for key, source in before.problem.sources.items()})
            state = ContractionState.initial(problem)
            for key in problem.specs:
                state = state.load(key, profile, 1 << 25)
            for scene in range(2):
                action, _, _, _ = describe_factorized_action(state, scene, original.positions,
                    original.core.large_operands, profile=profile,
                    output_order=term.subscripts.output, tile_elements=4)
                state = replay_factorized_action(state, action, profile)
            rng = np.random.default_rng(261049)
            arrays = {key: (rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)).astype('complex64')
                      for key, spec in problem.specs.items()}
            preparations = {key: SourcePreparation((), dtype='complex64', recipe=TensorPreparation(
                'constant', constants=(ArrayConstant.from_array(value),))) for key, value in arrays.items()}
            selected = plan_scene_storage(state, {}, preparations, profile=profile,
                                          memory_limit=1 << 25, host_memory_limit=1 << 25)
            points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in range(3)])
            equation = term.subscripts.normalized()
            args = tuple(item for key, axes in zip(term.inputs, equation.inputs, strict=True)
                         for item in (arrays[key].astype('complex128'), axes))+(equation.output,)
            reference = np.einsum(*args, optimize=True)
            for debug in (False, True):
                with self.subTest(loop=loop, debug=debug), patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
                    plan = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preparations,
                                                   problem.specs, debug=debug)
                cfg_plan.audit_cfg_plan(plan)
                collected = []
                @contextmanager
                def factory(configuration):
                    yield restore_input_loader(plan.requests, {}, plan.preparations, plan.specs,
                                               configuration=configuration)
                execute_cfg_plan(plan, ('cfg',), factory,
                                 lambda _cfg, _index, values: collected.extend(values) if values is not None else None)
                self.assertEqual(len(collected), 6)
                for value in collected:
                    np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-6)

    def test_debug_disables_the_proof_without_invoking_identity_hooks(self):
        actions = self.trace()
        hooks = tuple(lambda *_: self.fail('debug proof called an identity hook') for _ in actions)
        self.assertEqual(cfg_plan._bind_local_computation_reuses(actions, hooks, hooks,
                         ('a', 'b', 'c'), debug=True), (False,) * len(actions))

    def test_full_plan_events_ids_and_numerics_match_the_original_per_action_path(self):
        from correlator.planning.test_cfg_plan import CfgPlanTests
        from lqcdweave.contraction.execution.api import execute_cfg_plan
        from lqcdweave.contraction.expansion.api import restore_input_loader

        selected, preparations, specs = CfgPlanTests.dense_fixture()
        repeated = tuple(replace(action, reused=not action.reused)
                         for action in selected.state.actions if action.kind == 'contract')
        selected = replace(selected, state=replace(selected.state,
                           actions=(*selected.state.actions, *repeated)))
        points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in range(3)])
        for slots in (1, 1_000_000):
            with self.subTest(binding_slots=slots), patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', slots):
                with patch.object(cfg_plan, '_bind_local_computation_reuses',
                                  side_effect=lambda actions, *_args, **_kw: (False,) * len(actions)):
                    expected = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preparations, specs)
                actual = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preparations, specs)
            self.assertEqual(replace(actual, statistics={}), replace(expected, statistics={}))
            cfg_plan.audit_cfg_plan(actual)
            self.assertTrue(all(actual.instances[0].reused.array()[-len(repeated):]))
            collected = {}

            @contextmanager
            def factory(configuration):
                yield restore_input_loader(actual.requests, {}, actual.preparations,
                                           actual.specs, configuration=configuration)

            execute_cfg_plan(actual, ('cfg',), factory,
                             lambda _cfg, index, values: collected.update({index: values}))
            inputs = {key: prep.recipe.constants[0].array() for key, prep in preparations.items()}
            reference = {}
            for action in selected.state.actions:
                if action.kind == 'contract':
                    numbers = {axis: i for i, axis in enumerate(dict.fromkeys(
                        axis for axes in action.equation.inputs for axis in axes))}
                    args = []
                    for key, axes in zip(action.operands, action.equation.inputs, strict=True):
                        args.extend((inputs[key] if key in inputs else reference[key],
                                     [numbers[axis] for axis in axes]))
                    reference[action.tensor] = np.einsum(*args,
                        [numbers[axis] for axis in action.equation.output])
            for index in range(3):
                for key, value in zip(selected.state.results, collected[index], strict=True):
                    np.testing.assert_allclose(value, reference[key], rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
