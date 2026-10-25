"""Optional debug audits stay out of the default execution path."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
)
from lqcdweave.contraction.execution import domain, scene_runtime
from lqcdweave.contraction.execution.diagnostics import (
    debug_enabled,
    execution_debug,
)
from lqcdweave.contraction.execution.storage_runtime import _bind_loader_for_plan
from lqcdweave.contraction.planning.api import find_contraction_path


class _ArrayLoader:
    def __init__(self, arrays, *, reject_materialize=False):
        self.arrays = arrays
        self.reject_materialize = reject_materialize

    def read_block(self, key, block):
        return self.arrays[key]

    def materialize(self, key, source, raw_blocks):
        if self.reject_materialize:
            raise AssertionError("a complete preparation recipe should be used")
        return raw_blocks[source.region]

    def close_block(self, key, value):
        pass

    @property
    def input_readers(self):
        return {}

    def can_prefetch_block(self, key, block):
        return False

    def prefetch_host_bytes(self, key, block):
        return 0

    def prefetch_block(self, key, block):
        raise AssertionError("prefetch is disabled")

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        pass


def _make_dense_execution(*, optimize="anytime", prepared=False):
    left = np.array([[1 + 2j, 3 - 1j, 2j], [4, 2 + 3j, -1j]], dtype=np.complex128)
    right = np.array([[2 - 1j, 1, 3j, 2], [1j, -2, 3 + 1j, 4],
                      [2, 1 + 2j, -1, 1j]], dtype=np.complex128)
    arrays = {"left": left, "right": right}
    specs = {
        "left": TensorSpec(left.shape, left.dtype),
        "right": TensorSpec(right.shape, right.dtype),
    }
    sources = {
        key: SourceSpec(key, value.nbytes)
        for key, value in (("left", left), ("right", right))
    }
    problem = ContractionProblem(
        specs,
        (ContractionTerm(("left", "right"), "ij,jk->ik"),),
        sources,
    )
    blocks = {
        key: StorageBlock(
            "fixture", key, value.nbytes, host_bytes=value.nbytes,
        )
        for key, value in (("left", left), ("right", right))
    }
    preparations = {
        key: SourcePreparation(
            (key,),
            recipe=TensorPreparation("identity", (key,)) if prepared else None,
        )
        for key in ("left", "right")
    }
    coordinates, batch = describe_coordinates()
    loader = _ArrayLoader(arrays, reject_materialize=prepared)
    inputs = DiagramInputs(
        problem, blocks, preparations, loader, coordinates, batch, ("product",),
    )
    expanded = ExpandedCorrelator(
        "debug-test", inputs,
        OutputRecipe(OutputNode("constant", 0), ("product",), coordinates),
        DiagramBindings(), coordinates, batch,
    )
    path = find_contraction_path(
        expanded, memory_limit=1 << 20, optimize=optimize,
    )
    expected = np.zeros((left.shape[0], right.shape[1]), dtype=np.complex128)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            for shared in range(left.shape[1]):
                expected[row, column] += left[row, shared] * right[shared, column]
    return expanded, path, loader, expected


class DebugExecutionTests(unittest.TestCase):
    def test_default_skips_deep_binding_audit_and_debug_preserves_dense_result(self):
        expanded, path, _loader, expected = _make_dense_execution()

        with patch.object(domain, "validate_execution_binding") as validate:
            normal = domain.execute_contraction_path(expanded, path, debug=False)
            validate.assert_not_called()
            audited = domain.execute_contraction_path(expanded, path, debug=True)
            validate.assert_called_once_with(expanded, path)

        np.testing.assert_allclose(normal.term_values[0].value, expected)
        np.testing.assert_allclose(audited.term_values[0].value, expected)
        np.testing.assert_array_equal(normal.term_values[0].value, audited.term_values[0].value)

    def test_direct_audit_is_debug_only_and_fully_described_recipe_binds_once(self):
        expanded, path, _loader, expected = _make_dense_execution(
            optimize="none", prepared=True,
        )

        with patch.object(domain, "validate_execution_binding") as domain_validate, \
             patch("lqcdweave.contraction.execution.direct.validate_execution_binding") as direct_validate, \
             patch("lqcdweave.contraction.execution.direct._bind_loader_for_plan",
                   wraps=_bind_loader_for_plan) as bind:
            normal = domain.execute_contraction_path(expanded, path, debug=False)
            domain_validate.assert_not_called()
            direct_validate.assert_not_called()
            self.assertEqual(bind.call_count, 1)

            bind.reset_mock()
            audited = domain.execute_contraction_path(expanded, path, debug=True)
            domain_validate.assert_called_once_with(expanded, path)
            direct_validate.assert_called_once_with(expanded, path)
            self.assertEqual(bind.call_count, 1)

        np.testing.assert_allclose(normal.term_values[0].value, expected)
        np.testing.assert_allclose(audited.term_values[0].value, expected)

    def test_context_restores_after_nested_exception_and_rejects_non_bool(self):
        self.assertFalse(debug_enabled())
        with execution_debug(True):
            self.assertTrue(debug_enabled())
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with execution_debug(False):
                    self.assertFalse(debug_enabled())
                    raise RuntimeError("injected")
            self.assertTrue(debug_enabled())
        self.assertFalse(debug_enabled())
        with self.assertRaisesRegex(TypeError, "enabled must be bool"):
            with execution_debug(1):
                pass
        self.assertFalse(debug_enabled())

    def test_debug_does_not_change_measurement_or_add_synchronization(self):
        @dataclass
        class Report:
            execution_profile: object = None

        class Backend:
            def __init__(self):
                self.synchronizations = 0
                self.stream = SimpleNamespace(synchronize=self.synchronize)
                self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)

            @property
            def __name__(self):
                return "cupy"

            def synchronize(self):
                self.synchronizations += 1

        class Measurement:
            def __init__(self, backend):
                self.backend = backend
                self.result = {"measured": True}

            def __enter__(self):
                self.backend.cuda.get_current_stream().synchronize()
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.backend.cuda.get_current_stream().synchronize()
                return False

        state = SimpleNamespace(actions=(), problem=SimpleNamespace(sources={}, specs={}))
        loader = object()
        for debug in (False, True):
            for measure in (False, True):
                backend = Backend()
                source_execution = SimpleNamespace(
                    validate_state=Mock(), plan=SimpleNamespace(kernel_batches=()),
                )
                with patch.object(scene_runtime, "_replay", return_value=Report()) as replay, \
                     patch.object(scene_runtime, "ExecutionMeasurement", Measurement):
                    with execution_debug(debug):
                        report = scene_runtime.replay(
                            state, loader, backend=backend, measure=measure,
                            source_execution=source_execution,
                        )
                self.assertEqual(replay.call_args.kwargs.get("measure", False), measure)
                self.assertEqual(source_execution.validate_state.call_count, int(debug))
                self.assertEqual(backend.synchronizations, 2 if measure else 0)
                self.assertEqual(report.execution_profile is not None, measure)

    def test_standalone_replay_keeps_resource_validation_in_both_modes(self):
        state = SimpleNamespace(actions=(), problem=SimpleNamespace(sources={}, specs={}))
        for debug in (False, True):
            with patch.object(scene_runtime, "validate_executable_resources") as validate, \
                 patch.object(scene_runtime, "_replay", return_value=()):
                with execution_debug(debug):
                    scene_runtime.replay(state, object(), source_execution=None)
                validate.assert_called_once_with(state)


if __name__ == "__main__":
    unittest.main()
