"""Storage replay uses saved preparations without a physical materializer."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    DiagramBindings,
    DiagramInputs,
    ExpandedCorrelator,
    OutputNode,
    OutputRecipe,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorPreparation,
    TensorSpec,
    describe_coordinates,
    validate_executable_resources,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.storage_runtime import (
    replay_scene_storage,
    replay_storage_batches,
    bind_prepared_loader,
)
from lqcdweave.contraction.planning.api import plan_direct_execution


class _SavedRecipeLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.closed = []

    def read_block(self, key, block):
        return self.arrays[key]

    def close_block(self, key, value):
        self.closed.append(key)

    # Deliberately no materialize(): the saved identity recipes own that step.


def _execution_case():
    left = np.array([[1 + 2j, 3 - 1j, 2j], [4, 2 + 3j, -1j]], dtype=np.complex128)
    right = np.array([[2 - 1j, 1, 3j, 2], [1j, -2, 3 + 1j, 4],
                      [2, 1 + 2j, -1, 1j]], dtype=np.complex128)
    arrays = {"left": left, "right": right}
    specs = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
    sources = {key: SourceSpec(key, value.nbytes) for key, value in arrays.items()}
    problem = ContractionProblem(
        specs, (ContractionTerm(("left", "right"), "ij,jk->ik"),), sources,
    )
    blocks = {
        key: StorageBlock("fixture", key, value.nbytes, host_bytes=value.nbytes)
        for key, value in arrays.items()
    }
    preparations = {
        key: SourcePreparation((key,), recipe=TensorPreparation("identity", (key,)))
        for key in arrays
    }
    coordinates, batch = describe_coordinates()
    loader = _SavedRecipeLoader(arrays)
    inputs = DiagramInputs(problem, blocks, preparations, loader, coordinates, batch, ("product",))
    expanded = ExpandedCorrelator(
        "prepared-storage-test", inputs,
        OutputRecipe(OutputNode("constant", 0), ("product",), coordinates),
        DiagramBindings(), coordinates, batch,
    )
    path = plan_direct_execution(expanded, memory_limit=1 << 20)
    expected = left @ right
    return path.plan, loader, expected


class UnifiedStorageBindingTests(unittest.TestCase):
    def test_fixed_replay_binds_preparation_once_and_reads_fresh_values(self):
        plan, loader, expected = _execution_case()
        binding = 'lqcdweave.contraction.execution.storage_runtime.bind_prepared_loader'
        with patch(binding, wraps=bind_prepared_loader) as bind:
            first = replay_scene_storage(plan, loader, backend=np, io_overlap=False)
            self.assertEqual(bind.call_count, 1)
            loader.arrays['left'] *= 2
            second = replay_scene_storage(plan, loader, backend=np, io_overlap=False)
            self.assertEqual(bind.call_count, 2)
        np.testing.assert_array_equal(first[0], expected)
        np.testing.assert_array_equal(second[0], 2 * expected)

    def test_saved_preparations_execute_without_physical_materializer(self):
        plan, loader, expected = _execution_case()
        report = replay_scene_storage(plan, loader, backend=np, io_overlap=False)
        np.testing.assert_allclose(report[0], expected)
        self.assertEqual(set(loader.closed), {"left", "right"})

    def test_normal_audits_once_and_debug_rechecks_without_changing_results(self):
        plan, loader, expected = _execution_case()
        validate = "lqcdweave.contraction.execution.storage_runtime.validate_executable_resources"
        with patch(validate, wraps=validate_executable_resources) as audit:
            normal = replay_scene_storage(plan, loader, backend=np, io_overlap=False)
            self.assertEqual(audit.call_count, 1)

        debug_loader = _SavedRecipeLoader(loader.arrays)
        with patch(validate, wraps=validate_executable_resources) as audit:
            with execution_debug(True):
                debug = replay_scene_storage(plan, debug_loader, backend=np, io_overlap=False)
            self.assertEqual(audit.call_count, 2)

        np.testing.assert_allclose(normal[0], expected)
        np.testing.assert_allclose(debug[0], expected)
        np.testing.assert_array_equal(normal[0], debug[0])

    def test_batch_entry_does_not_repeat_its_initial_resource_audit(self):
        plan, loader, expected = _execution_case()
        validate = "lqcdweave.contraction.execution.storage_runtime.validate_executable_resources"
        results = []
        with patch(validate, wraps=validate_executable_resources) as audit:
            replay_storage_batches(
                (plan,), loader, lambda _index, report: results.append(report[0]),
                backend=np, prejoined=True, io_overlap=False,
            )
        self.assertEqual(audit.call_count, 1)
        np.testing.assert_allclose(results[0], expected)


if __name__ == "__main__":
    unittest.main()
