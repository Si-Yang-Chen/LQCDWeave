"""Behavioral checks for dense-batch graph discovery and producer indexing."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from math import prod
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import Action, AxisPattern, TensorSpec
from lqcdweave.contraction.planning import dense_batches
from lqcdweave.contraction.planning.api import CostProfile


def _entry(action_index, left, right):
    return action_index, ((left, right),)


def _load_existing_producer_layout_fixture():
    """Load the existing integration fixture under a non-test module name."""
    path = Path(__file__).with_name("test_dense_batches.py")
    name = "_dense_batch_graph_fixture_helpers"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DenseBatchGraphTests(unittest.TestCase):
    def test_discovery_derives_layout_once_per_action_and_keeps_no_cross_call_cache(self):
        catalog = {key: TensorSpec((2, 3), np.complex128)
                   for key in ('a0', 'a1', 'b0', 'b1')}
        actions = []
        for i in range(2):
            for j in range(2):
                key = f'c{i}{j}'
                catalog[key] = TensorSpec((), np.complex128)
                actions.append(Action('contract', key, (f'a{i}', f'b{j}'),
                    AxisPattern(((0, 1), (0, 1)), ()), seconds=1.0))
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                state = SimpleNamespace(actions=tuple(actions), catalog=catalog,
                    results=tuple(action.tensor for action in actions),
                    problem=SimpleNamespace(backend=backend))
                expected = dense_batches.describe_dense_batch(
                    actions, tuple(range(4)), catalog, CostProfile(), backend=backend)
                with patch.object(dense_batches, '_descriptor', wraps=dense_batches._descriptor) as derive:
                    self.assertEqual(dense_batches.find_dense_batches(state, CostProfile()), (expected,))
                self.assertEqual(derive.call_count, len(actions))
                # A later call must see the new equations, even with the same
                # action names, shapes and catalog object.
                state.actions = tuple(replace(action,
                    equation=AxisPattern(((0, 1), (2, 3)), (0, 1, 2, 3))) for action in actions)
                self.assertEqual(dense_batches.find_dense_batches(state, CostProfile()), ())

    def test_components_use_tensor_and_permutation_identity_and_keep_duplicate_edges(self):
        left = dense_batches.DenseBatchOperand("same-tensor", (0, 1))
        transposed = dense_batches.DenseBatchOperand("same-tensor", (1, 0))
        other_left = dense_batches.DenseBatchOperand("other-left", (0, 1))
        isolated_left = dense_batches.DenseBatchOperand("isolated-left", (0, 1))
        right0 = dense_batches.DenseBatchOperand("right-0", (0, 1))
        right1 = dense_batches.DenseBatchOperand("right-1", (0, 1))
        right2 = dense_batches.DenseBatchOperand("right-2", (0, 1))
        right3 = dense_batches.DenseBatchOperand("right-3", (0, 1))

        entries = [
            _entry(17, left, right0),
            _entry(3, left, right1),
            _entry(8, transposed, right2),
            _entry(23, other_left, right1),
            _entry(5, left, right0),
            _entry(14, isolated_left, right3),
        ]

        components = tuple(dense_batches._connected_indices(entries))

        self.assertEqual(components, ((3, 5, 17, 23), (8,), (14,)))
        flattened = tuple(index for component in components for index in component)
        self.assertEqual(sorted(flattened), sorted(index for index, _ in entries))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_star_is_complete_and_missing_edge_only_allows_existing_subgrid(self):
        hub = dense_batches.DenseBatchOperand("hub", (0, 1))
        star = [
            _entry(2 * i + 101, hub, dense_batches.DenseBatchOperand(f"leaf-{i}", (0, 1)))
            for i in range(512)
        ]
        components = tuple(dense_batches._connected_indices(star))
        expected = tuple(sorted(index for index, _ in star))
        self.assertEqual(components, (expected,))
        self.assertEqual(len(components[0]), len(star))

        catalog = {
            "a0": TensorSpec((2, 3), np.complex128),
            "a1": TensorSpec((4, 3), np.complex128),
            "b0": TensorSpec((3, 5), np.complex128),
            "b1": TensorSpec((3, 7), np.complex128),
        }
        actions = []
        outputs = []
        for i, j in ((0, 0), (0, 1), (1, 0)):
            output = f"c{i}{j}"
            catalog[output] = TensorSpec((catalog[f"a{i}"].shape[0], catalog[f"b{j}"].shape[1]),
                                         np.complex128)
            outputs.append(output)
            actions.append(Action("contract", output, (f"a{i}", f"b{j}"),
                AxisPattern(((0, 1), (1, 2)), (0, 2)), seconds=1.0))
        state = SimpleNamespace(actions=tuple(actions), catalog=catalog, results=tuple(outputs),
                                problem=SimpleNamespace(backend="numpy"))

        batches = dense_batches.find_dense_batches(state, CostProfile())
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].action_indices, (0, 2))
        self.assertEqual({result.tensor for result in batches[0].results}, {'c00', 'c10'})

    def test_producer_index_is_paid_tensor_specific_and_rejects_multiple_producers(self):
        helpers = _load_existing_producer_layout_fixture()
        original_eligible = dense_batches.dense_batch_producer_eligible
        original_solve = helpers.solve_storage
        captured = []
        eligibility_calls = []

        def capture_solve(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            captured.append(result)
            return result

        def inspect_paid_producers(operand, actions, catalog):
            paid = tuple(actions)
            self.assertTrue(paid)
            self.assertEqual({action.tensor for action in paid}, {operand.tensor})
            self.assertTrue(all(action.kind == "contract" and not action.reused for action in paid))
            eligibility_calls.append((operand, paid))
            return original_eligible(operand, paid, catalog)

        with patch.object(helpers, "solve_storage", side_effect=capture_solve), \
             patch.object(dense_batches, "dense_batch_producer_eligible",
                          side_effect=inspect_paid_producers):
            helpers.DenseBatchIntegrationTests().test_producer_layout_eliminates_single_side_pack_and_preserves_logical_axes()

        self.assertGreaterEqual(len(captured), 1)
        self.assertEqual(len(eligibility_calls), 1)
        result = captured[0]
        plan = result.plan
        batch = plan.kernel_batches[0]
        self.assertEqual(result.statistics["dense_kernel_lowering"]["producer_layouts"], 1)
        self.assertEqual(batch.prepared_sides, (True, False))
        self.assertEqual(batch.left, (eligibility_calls[0][0],))

        expected_packing_bytes = sum(
            prod(plan.state.catalog[operand.tensor].shape) * plan.state.catalog[operand.tensor].dtype.itemsize
            for side, prepared in zip((batch.left, batch.right), batch.prepared_sides)
            if not prepared for operand in side)
        self.assertEqual(batch.packing_bytes, expected_packing_bytes)
        self.assertEqual(result.statistics["dense_kernel_lowering"]["packing_bytes"], batch.packing_bytes)
        prepared_inputs = {batch.left[0].tensor: batch.left[0].permutation}
        expected_batch = dense_batches.describe_dense_batch(
            plan.state.actions, batch.action_indices, plan.state.catalog, CostProfile(),
            backend="cupy", prepared_inputs=prepared_inputs)
        self.assertEqual(batch, expected_batch)
        self.assertLess(batch.seconds, sum(plan.state.actions[i].seconds for i in batch.action_indices))

        producer = eligibility_calls[0][1][0]
        duplicated_state = replace(plan.state, actions=plan.state.actions + (producer,))
        duplicate_calls = []

        def reject_multiple_producers(operand, actions, catalog):
            paid = tuple(actions)
            accepted = original_eligible(operand, paid, catalog)
            if operand.tensor == producer.tensor:
                duplicate_calls.append((operand, paid, accepted))
                self.assertEqual(len(paid), 2)
                self.assertTrue(all(action.tensor == operand.tensor and action.kind == "contract"
                                    and not action.reused for action in paid))
                self.assertFalse(accepted)
            return accepted

        with patch.object(dense_batches, "dense_batch_producer_eligible",
                          side_effect=reject_multiple_producers):
            duplicate_batches = dense_batches.find_dense_batches(duplicated_state, CostProfile())

        self.assertEqual(len(duplicate_calls), 1)
        self.assertEqual(len(duplicate_batches), 1)
        duplicate_batch = duplicate_batches[0]
        self.assertEqual(duplicate_batch.left, (duplicate_calls[0][0],))
        self.assertEqual(duplicate_batch.prepared_sides, (False, False))


if __name__ == "__main__":
    unittest.main()
