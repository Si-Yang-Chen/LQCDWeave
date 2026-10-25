"""Execution-side binding and evaluation of saved source preparations."""
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.preparation import (
    PreparedSourceLoader, bind_prepared_loader,
)
from lqcdweave.contraction.expansion.saved_inputs import restore_input_loader


class _PhysicalLoader:
    def __init__(self, *, constant_values=None, opaque=None):
        self.input_readers = {}
        self.constant_values = constant_values or {}
        self.opaque = opaque
        self.materialized = []

    def materialize(self, key, source, raw_blocks):
        self.materialized.append((key, source, raw_blocks))
        return self.opaque

    def read_block(self, key, block):
        raise AssertionError("test should supply raw blocks directly")

    def close_block(self, key, value):
        pass

    def can_prefetch_block(self, key, block):
        return False

    def prefetch_host_bytes(self, key, block):
        raise AssertionError("prefetch is not available")

    def prefetch_block(self, key, block):
        raise AssertionError("prefetch is not available")

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        pass


class SourcePreparationExecutionTests(unittest.TestCase):
    @staticmethod
    def _constant_preparation(values):
        return SourcePreparation((), recipe=TensorPreparation(
            "constant", constants=(ArrayConstant.from_array(values),)))

    def test_frozen_restoration_keeps_closure_and_explicit_plan_values(self):
        old_value = np.array([1.0, -2.0])
        closure_value = np.array([3.0, 4.0])
        plan_value = np.array([7.0, -5.0])
        restored_preparations = MappingProxyType({
            "value": self._constant_preparation(old_value),
            "closure": self._constant_preparation(closure_value),
        })
        restored_specs = MappingProxyType({
            key: TensorSpec(value.shape, value.dtype)
            for key, value in (("value", old_value), ("closure", closure_value))
        })

        physical = restore_input_loader({}, {}, restored_preparations, restored_specs)
        self.assertIs(physical.preparations, restored_preparations)
        self.assertIs(physical.specs, restored_specs)

        plan_preparations = MappingProxyType({
            "value": self._constant_preparation(plan_value),
        })
        plan_specs = MappingProxyType({
            "value": TensorSpec(plan_value.shape, plan_value.dtype),
        })
        loader = bind_prepared_loader(physical, plan_preparations, plan_specs)

        self.assertIsInstance(loader.preparations, MappingProxyType)
        self.assertIsInstance(loader.specs, MappingProxyType)
        self.assertTrue(loader._frozen)
        self.assertIn("closure", loader.preparations)
        self.assertIs(loader.preparations["value"], plan_preparations["value"])
        np.testing.assert_array_equal(loader.materialize("closure", None, {}), closure_value)
        np.testing.assert_array_equal(loader.materialize("value", None, {}), plan_value)

    def test_mutable_restoration_is_a_snapshot_and_keeps_invalidation(self):
        first_value = np.array([1.5, -2.5])
        second_value = np.array([8.0, 9.0])
        mutable_preparations = {"value": self._constant_preparation(first_value)}
        mutable_specs = {"value": TensorSpec(first_value.shape, first_value.dtype)}
        physical = restore_input_loader({}, {}, mutable_preparations, mutable_specs)

        self.assertIsNot(physical.preparations, mutable_preparations)
        self.assertIsNot(physical.specs, mutable_specs)
        mutable_preparations["value"] = self._constant_preparation(second_value)
        mutable_specs["value"] = TensorSpec(second_value.shape, second_value.dtype)
        self.assertIsNot(physical.preparations["value"], mutable_preparations["value"])

        loader = bind_prepared_loader(physical, physical.preparations, physical.specs)
        self.assertFalse(loader._frozen)
        np.testing.assert_array_equal(loader.materialize("value", None, {}), first_value)

        physical.preparations["value"] = self._constant_preparation(second_value)
        physical.specs["value"] = TensorSpec(second_value.shape, second_value.dtype)
        np.testing.assert_array_equal(loader.materialize("value", None, {}), second_value)

    def test_equal_constants_share_upload_across_residency_ids(self):
        values = np.array([1+2j, 3-4j])
        preps = {str(index): SourcePreparation((), recipe=TensorPreparation(
            'constant', constants=(ArrayConstant.from_array(values.copy()),)))
            for index in range(4)}
        for cached in (None, {key: (values.copy(),) for key in preps}):
            with self.subTest(cached=cached is not None):
                backend = SimpleNamespace(asarray=Mock(wraps=np.asarray))
                loader = bind_prepared_loader(_PhysicalLoader(constant_values=cached),
                                             preps, {}, backend=backend)
                outputs = [loader.materialize(key, None, {}) for key in preps]
                self.assertEqual(backend.asarray.call_count, 1)
                self.assertTrue(all(value is outputs[0] for value in outputs))
                np.testing.assert_array_equal(outputs[0], values)
                other = SimpleNamespace(asarray=Mock(side_effect=lambda value: np.array(value, copy=True)))
                rebound = loader.with_backend(other)
                self.assertIsNot(rebound.materialize('0', None, {}), outputs[0])

    def test_spin_preparation_runs_from_pure_recipe(self):
        spin = np.array([[0, 1j, 0, 0], [1, 0, 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128)
        recipe = TensorPreparation(
            "spin_spatial", ("raw",), (ArrayConstant.from_array(spin),),
            conjugate=True, permutation=(1, 0),
        )
        raw = (np.arange(12).reshape(4, 3) + 1j).astype(np.complex128)
        prepared = SourcePreparation(("raw",), recipe=recipe)
        spec = TensorSpec((4, 4, 3, 4), np.complex128)
        loader = bind_prepared_loader(_PhysicalLoader(), {"value": prepared}, {"value": spec})

        actual = loader.materialize("value", None, {"raw": raw})
        expected = np.einsum("ab,...->ab...", spin, raw.transpose(1, 0).conj(), optimize=False)
        np.testing.assert_allclose(actual, expected)

    def test_recipe_less_source_uses_original_physical_materializer(self):
        token = object()
        physical = _PhysicalLoader(opaque=token)
        loader = bind_prepared_loader(
            physical,
            {"value": SourcePreparation(("raw",), recipe=None)},
            {"value": TensorSpec((2,), np.float64)},
        )

        actual = loader.materialize("value", "source-description", {"raw": np.ones(2)})
        self.assertIs(actual, token)
        self.assertEqual(len(physical.materialized), 1)
        self.assertEqual(physical.materialized[0][0:2], ("value", "source-description"))

    def test_same_backend_reuses_binding_and_other_backend_gets_new_cache(self):
        constant = np.array([2.0, -3.0], dtype=np.float64)
        recipe = TensorPreparation("constant", constants=(ArrayConstant.from_array(constant),))
        preps = {"value": SourcePreparation((), recipe=recipe)}
        specs = {"value": TensorSpec(constant.shape, constant.dtype)}
        physical = _PhysicalLoader(constant_values={"value": (constant,)})
        original = bind_prepared_loader(physical, preps, specs)
        self.assertIs(bind_prepared_loader(original, preps, specs), original)
        self.assertIs(original.with_backend(np), original)

        backend = SimpleNamespace(
            asarray=Mock(wraps=np.asarray), empty=np.empty,
            zeros_like=np.zeros_like, einsum=np.einsum,
        )
        rebound = bind_prepared_loader(original, preps, specs, backend=backend)
        self.assertIsNot(rebound.constants, original.constants)
        np.testing.assert_array_equal(rebound.materialize("value", None, {}), constant)
        self.assertTrue(backend.asarray.called)


if __name__ == "__main__":
    unittest.main()
