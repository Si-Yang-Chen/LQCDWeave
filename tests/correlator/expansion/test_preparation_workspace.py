"""Preparation workspace bounds for dense diagram batches."""

import math
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion.batching import _BatchTaskLoader
from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval


class PreparationWorkspaceTests(unittest.TestCase):
    @staticmethod
    def _case(dtype=np.complex128):
        rng = np.random.default_rng(20261005)

        def random(shape):
            values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            return values.astype(dtype)

        component = ElementalComponent(0)
        raw_elemental = random((1, 1, 2, 1, 1))
        raw_propagator = random((2, 2, 4, 4, 1, 1))
        elemental = ArrayElemental(
            {"v2v": raw_elemental}, (component,), phase_convention="anchor",
            cfg="workspace-test", data_identity="workspace-test/elemental",
        )
        propagator = ArrayPropagator(
            {"vsv": raw_propagator}, time_convention="absolute",
            cfg="workspace-test", data_identity="workspace-test/propagator",
        )
        operator = Operator.from_terms("local", (OperatorTerm(0, component, 0),))
        quark = Propagator("u", Tag(0, 0), Tag(1, 1))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["sink", "source"], [None, quark],
        )
        vertices = {
            "sink": Meson(elemental, operator, phase_convention="anchor"),
            "source": Meson(elemental, operator, source=True, phase_convention="anchor"),
        }
        return diagram, elemental, propagator, quark, vertices, raw_elemental

    def _batched(self, width, *, dtype=np.complex128):
        diagram, elemental, propagator, quark, vertices, raw_elemental = self._case(dtype)
        calculation = calc_diagram_eval(
            diagram, propagator_map={quark: propagator}, vertex_map=vertices,
            usedNe=1, time_axes={"sink": 0, "source": range(2)},
            batch={"axis": "source", "size": width}, optimize="none",
            execute=False,
        )
        return calculation, elemental, propagator, quark, vertices, raw_elemental

    def test_dynamic_workspace_covers_batch_buffers_and_keeps_lane_cost(self):
        outputs = {}
        for width in (1, 2):
            with self.subTest(width=width):
                calculation, _, _, _, _, raw_elemental = self._batched(width)
                self.assertTrue(calculation.inputs.task_batches)
                profile = calculation.path.profile
                self.assertIsNotNone(profile)
                saw_dynamic_prop = False
                saw_dynamic_dagger = False
                saw_static_elemental = False
                saw_static_prop = False
                saw_static_dagger = False

                for task in calculation.inputs.task_batches:
                    if width == 1:
                        self.assertEqual(task.coordinate_batch.actual_size, 1)
                        self.assertFalse(any(
                            binding.batch_dependent
                            for binding in task.batch_bindings.values()
                        ))
                    for tensor_id, spec in task.problem.specs.items():
                        description = spec.description
                        if description is None:
                            continue
                        prep = task.preparations[tensor_id]
                        target_bytes = math.prod(spec.shape) * spec.dtype.itemsize
                        is_dynamic = description.axes[:1] == ("coordinate_batch",)
                        if width == 1:
                            self.assertFalse(is_dynamic)
                        if description.role == "elemental" and not is_dynamic:
                            saw_static_elemental = True
                            transform = dict(description.transform)
                            lane_transforms = transform.get("lane_transforms", ())
                            is_dagger = bool(transform.get("dagger")) or any(
                                dict(lane_transform).get("dagger", False)
                                for lane_transform in lane_transforms
                            )
                            if width == 1 and is_dagger:
                                saw_static_dagger = True
                                raw_bytes = math.prod(raw_elemental.shape[3:]) * raw_elemental.dtype.itemsize
                                expected_workspace = target_bytes + raw_bytes
                            else:
                                expected_workspace = target_bytes
                            self.assertEqual(prep.workspace_host_bytes, expected_workspace)
                            self.assertEqual(prep.kernel_calls, 1)
                            expected_seconds = profile.preparation_seconds(
                                0, 0, kernel_calls=1, dtype=spec.dtype.name,
                            )
                            self.assertEqual(prep.seconds(profile), expected_seconds)
                            continue
                        if width == 1 and description.role == "propagator":
                            saw_static_prop = True
                            self.assertEqual(prep.workspace_host_bytes, 0)
                            self.assertEqual(prep.kernel_calls, 0)
                            expected_seconds = profile.preparation_seconds(
                                0, 0, kernel_calls=0, dtype=spec.dtype.name,
                            )
                            self.assertEqual(prep.seconds(profile), expected_seconds)
                            continue
                        if not is_dynamic:
                            continue
                        if description.role == "propagator":
                            saw_dynamic_prop = True
                            self.assertEqual(prep.workspace_host_bytes, target_bytes)
                            self.assertEqual(prep.kernel_calls, 0)
                            expected_seconds = profile.preparation_seconds(
                                0, 0, kernel_calls=0, dtype=spec.dtype.name,
                            )
                            self.assertEqual(prep.seconds(profile), expected_seconds)
                        elif description.role == "elemental":
                            batch_transform = dict(description.transform)
                            lane_transforms = batch_transform.get("lane_transforms", ())
                            if not any(dict(transform).get("dagger") for transform in lane_transforms):
                                continue
                            saw_dynamic_dagger = True
                            lane_bytes = math.prod(spec.shape[1:]) * spec.dtype.itemsize
                            raw_bytes = math.prod(raw_elemental.shape[3:]) * raw_elemental.dtype.itemsize
                            expected_workspace = max(target_bytes, lane_bytes + raw_bytes)
                            self.assertEqual(prep.workspace_host_bytes, expected_workspace)
                            self.assertEqual(prep.kernel_calls, 1)
                            expected_seconds = profile.preparation_seconds(
                                0, 0, kernel_calls=1, dtype=spec.dtype.name,
                            )
                            self.assertEqual(prep.seconds(profile), expected_seconds)

                self.assertTrue(saw_static_elemental)
                if width == 1:
                    self.assertTrue(saw_static_dagger)
                    self.assertTrue(saw_static_prop)
                    self.assertFalse(saw_dynamic_prop)
                    self.assertFalse(saw_dynamic_dagger)
                else:
                    self.assertTrue(saw_dynamic_prop)
                    self.assertTrue(saw_dynamic_dagger)
                outputs[width] = calculation.run().value

        scalar_values = []
        for source_time in (0, 1):
            diagram, _, propagator, quark, vertices, _ = self._case()
            scalar_values.append(calc_diagram_eval(
                diagram, propagator_map={quark: propagator}, vertex_map=vertices,
                usedNe=1, time_axes={"sink": 0, "source": source_time},
                optimize="none",
            ).value)
        expected = np.asarray(scalar_values)
        np.testing.assert_allclose(outputs[1], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(outputs[2], expected, rtol=1e-6, atol=1e-6)

    def test_opaque_batch_rejects_implicit_precision_conversion_before_reads(self):
        # Opaque lane materializers cannot be rewritten to a new compute dtype.
        # Keep the precision policy explicit rather than silently widening inputs.
        with patch.object(Meson, "describe_factor_preparation", return_value=None), \
             patch.object(ArrayElemental, "block", side_effect=AssertionError("read input")), \
             patch.object(ArrayPropagator, "block", side_effect=AssertionError("read input")):
            for width in (1, 2):
                with self.subTest(width=width):
                    with self.assertRaisesRegex(NotImplementedError, "described preparation recipe"):
                        self._batched(width, dtype=np.complex64)

    def test_dynamic_lane_temporary_is_released_before_next_materializer(self):
        loader = _BatchTaskLoader({"blocks": {}, "block_by_identity": {}})
        lane_spec = TensorSpec((1,), np.complex128)
        batch_spec = TensorSpec((2, 1), np.complex128)
        first_value = []

        def materialize(raw):
            if first_value:
                self.assertIsNone(first_value[0]())
            value = np.asarray(raw, dtype=np.complex128).copy()
            if not first_value:
                first_value.append(weakref.ref(value))
            return value

        loader.register(
            0,
            specs={"tensor:0": batch_spec},
            lane_specs=(lane_spec,),
            role_entries={0: (("raw", 0, materialize), ("raw", 1, materialize))},
            dynamic_roles={0},
            sparse_records={},
        )
        with loader.activate(0):
            result = loader.materialize(
                "tensor:0", None, {"raw": (np.asarray([1]), np.asarray([2]))},
            )

        np.testing.assert_array_equal(result, np.asarray([[1], [2]], dtype=np.complex128))
        self.assertIsNone(first_value[0]())


if __name__ == "__main__":
    unittest.main()
