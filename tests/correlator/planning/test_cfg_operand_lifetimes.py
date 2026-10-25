"""Compact input-use binding preserves exact releases, including actual aliases."""
from contextlib import contextmanager
from dataclasses import replace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.api import solve_storage


class ExtendedAction(Action):
    pass


class CfgOperandLifetimeTests(unittest.TestCase):
    @staticmethod
    def fixture():
        values = (np.arange(9).reshape(3, 3) + 1j*np.arange(9, 18).reshape(3, 3))
        spec = TensorSpec(values.shape, values.dtype)
        specs = dict(a=spec, b=spec)
        prep = SourcePreparation((), recipe=TensorPreparation('constant',
                                 constants=(ArrayConstant.from_array(values),)))
        preps = dict(a=prep, b=prep)
        problem = ContractionProblem(specs, (
            ContractionTerm(('a', 'b'), 'ij,jk->ik'),
            ContractionTerm(('b',), 'ij->ji'),
        ), {key: SourceSpec(key, 0, read_bytes=0) for key in specs})
        selected = solve_storage(problem, {}, preps, optimizer='anytime', memory_limit=1 << 20,
                                 deadline_seconds=2.0, component_workers=1).plan
        repeated = tuple(replace(action, reused=not action.reused)
                         for action in selected.state.actions if action.kind == 'contract')
        selected = replace(selected, state=replace(selected.state,
                           actions=(*selected.state.actions, *repeated)))
        return selected, preps, specs, values

    @staticmethod
    def bound(selected, preps, specs, *, debug=False):
        return cfg_plan._bind_cfg_template(selected, None, {}, preps, specs,
                                            output_mode='dataset', debug=debug)

    def test_compact_uses_match_original_release_order_with_aliases_and_windows(self):
        selected, preps, specs, _ = self.fixture()
        points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in range(4)])
        for window in (1, 3):
            compact = self.bound(selected, preps, specs)
            self.assertFalse(compact.keywords['fine_operand_uses'])
            original = self.bound(selected, preps, specs)
            original.keywords['fine_operand_uses'] = True
            actual = compact(points.points, reuse_window=window)
            expected = original(points.points, reuse_window=window)
            self.assertEqual(replace(actual, statistics={}), replace(expected, statistics={}))
            positions = {key:i for i, key in enumerate(actual.templates[0].keys)}
            for instance in actual.instances:
                ids = instance.node_ids.array()
                self.assertEqual(ids[positions['a']], ids[positions['b']])
            cfg_plan.audit_cfg_plan(actual)

    def test_proven_repeats_do_not_extract_operands_again(self):
        selected, preps, specs, _ = self.fixture()
        bound = self.bound(selected, preps, specs)
        flags = bound.keywords['local_reuses']
        self.assertTrue(any(flags))
        def unexpected(_):
            self.fail('a proven local repeat extracted operands')
        bound.keywords['operand_binders'] = tuple(unexpected if flag else bind
            for flag, bind in zip(flags, bound.keywords['operand_binders'], strict=True))
        points, _ = describe_coordinates(time_pairs=[{0:0, 1:0}, {0:1, 1:0}])
        cfg_plan.audit_cfg_plan(bound(points.points, reuse_window=2))

    def test_debug_and_unproven_redefinitions_keep_per_action_uses(self):
        selected, preps, specs, _ = self.fixture()
        debug = self.bound(selected, preps, specs, debug=True)
        self.assertTrue(debug.keywords['fine_operand_uses'])
        self.assertEqual(tuple(debug.keywords['binding_indices']),
                         tuple(range(len(debug.keywords['actions']))))
        actions = list(selected.state.actions)
        last = actions[-1]
        actions[-1] = ExtendedAction(**vars(last))
        selected = replace(selected, state=replace(selected.state, actions=actions))
        bound = self.bound(selected, preps, specs)
        self.assertTrue(bound.keywords['fine_operand_uses'])
        self.assertEqual(tuple(bound.keywords['binding_indices']),
                         tuple(range(len(bound.keywords['actions']))))
        points, _ = describe_coordinates()
        cfg_plan.audit_cfg_plan(bound(points.points, reuse_window=1))

    def test_skipped_repeats_are_not_inspected_by_the_coordinate_binding_loop(self):
        selected, preps, specs, _ = self.fixture()
        bound = self.bound(selected, preps, specs)
        flags = bound.keywords['local_reuses']
        self.assertTrue(any(flags))
        self.assertEqual(tuple(bound.keywords['binding_indices']),
                         tuple(i for i, flag in enumerate(flags) if not flag))

        class ResourceFieldsOnly:
            def __init__(self, action):
                self.tensor, self.seconds = action.tensor, action.seconds

            @property
            def kind(self):
                raise AssertionError('coordinate binding inspected a proven repeated Action')

        bound.keywords['actions'] = tuple(ResourceFieldsOnly(action) if flag else action
            for action, flag in zip(bound.keywords['actions'], flags, strict=True))
        points, _ = describe_coordinates(time_pairs=[{0:0}, {0:1}, {0:0}])
        actual = bound(points.points, reuse_window=2)
        original = self.bound(selected, preps, specs)
        original.keywords['fine_operand_uses'] = True
        expected = original(points.points, reuse_window=2)
        self.assertEqual(replace(actual, statistics={}), replace(expected, statistics={}))
        cfg_plan.audit_cfg_plan(actual)

    def test_repeated_result_names_sharing_one_actual_value_keep_releases_and_outputs(self):
        from lqcdweave.contraction.execution.api import execute_cfg_plan
        from lqcdweave.contraction.expansion.api import restore_input_loader

        selected, preps, specs, values = self.fixture()
        product = next(action for action in selected.state.actions
                       if action.kind == 'contract' and len(action.operands) == 2)
        alias = replace(product, tensor='same_product')
        selected = replace(selected, state=replace(selected.state,
            actions=(*selected.state.actions, alias, product, alias),
            catalog=dict(selected.state.catalog, same_product=selected.state.catalog[product.tensor]),
            results=(*selected.state.results, 'same_product')))
        points, _ = describe_coordinates(time_pairs=[{0:0}, {0:1}, {0:0}])
        for window in (1, 2, 3):
            with self.subTest(reuse_window=window):
                original = self.bound(selected, preps, specs)
                original.keywords['fine_operand_uses'] = True
                expected = original(points.points, reuse_window=window)
                actual = self.bound(selected, preps, specs)(points.points, reuse_window=window)
                self.assertEqual(replace(actual, statistics={}), replace(expected, statistics={}))
                cfg_plan.audit_cfg_plan(actual)
                positions = {key:i for i, key in enumerate(actual.templates[0].keys)}
                for instance in actual.instances:
                    ids = instance.node_ids.array()
                    self.assertEqual(ids[positions[product.tensor]], ids[positions['same_product']])
                collected = {}

                @contextmanager
                def factory(configuration):
                    yield restore_input_loader(actual.requests, {}, actual.preparations,
                                               actual.specs, configuration=configuration)

                execute_cfg_plan(actual, ('cfg',), factory,
                    lambda _cfg, index, result: collected.update({index: result}))
                for index in range(3):
                    np.testing.assert_allclose(collected[index][-1], values @ values,
                                               rtol=3e-12, atol=3e-12)

    def test_default_and_debug_execution_match_independent_matrix_products(self):
        from lqcdweave.contraction.execution.api import execute_cfg_plan
        from lqcdweave.contraction.expansion.api import restore_input_loader
        selected, preps, specs, values = self.fixture()
        points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in range(3)])
        for debug in (False, True):
            with self.subTest(debug=debug):
                plan = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preps, specs, debug=debug)
                results = {}
                @contextmanager
                def factory(configuration):
                    yield restore_input_loader(plan.requests, {}, plan.preparations,
                                               plan.specs, configuration=configuration)
                execute_cfg_plan(plan, ('cfg',), factory,
                    lambda _cfg, index, result: results.update({index: result}), debug=debug)
                for index in range(3):
                    np.testing.assert_allclose(results[index][0], values @ values, rtol=3e-12, atol=3e-12)
                    np.testing.assert_array_equal(results[index][1], values.T)


if __name__ == '__main__':
    unittest.main()
