"""Runtime memory observations and budget delegation tests."""

from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.contracts.memory import (
    BudgetError,
    BudgetExceededError,
    MemoryEstimate,
    MemoryLimits,
)
from lqcdweave.runtime import memory as runtime_memory


class RuntimeMemoryTests(unittest.TestCase):
    @staticmethod
    def _budget(*, host=None, device=None, fraction=0.1, minimum=1_000):
        return SimpleNamespace(
            host_bytes=host,
            device_bytes=device,
            reserve_fraction=fraction,
            min_reserve_bytes=minimum,
            workspace_bytes=0,
        )

    @staticmethod
    def _cupy_module():
        return ModuleType("cupy")

    def test_cpu_default_uses_fresh_host_observation_and_default_fraction(self):
        with patch.object(runtime_memory, "host_available_memory", return_value=10_000) as host:
            limits = runtime_memory.resolve_memory_limits(None, backend=np)

        host.assert_called_once_with()
        self.assertEqual(limits.host_available_bytes, 10_000)
        self.assertEqual(limits.host_bytes, 7_000)
        self.assertIsNone(limits.device_bytes)

    def test_gpu_reserve_and_explicit_cap_match_saved_budget_policy(self):
        backend = self._cupy_module()
        with (
            patch.object(runtime_memory, "host_available_memory", return_value=10_000) as host,
            patch.object(
                runtime_memory,
                "device_memory_info",
                side_effect=AssertionError("injected device observations must not be probed"),
            ) as device,
        ):
            default = runtime_memory.resolve_memory_limits(
                self._budget(), backend=backend, device_info=(6_000, 10_000),
            )
            capped = runtime_memory.resolve_memory_limits(
                self._budget(host=8_000, device=4_000, fraction=0.1, minimum=1_500),
                backend=backend,
                host_available=10_000,
                device_info=(6_000, 10_000),
            )

        host.assert_called_once_with()
        device.assert_not_called()
        self.assertEqual(default.device_reserve_bytes, 1_000)
        self.assertEqual(default.device_bytes, 5_000)
        self.assertEqual(capped.host_bytes, 7_000)
        self.assertEqual(capped.device_reserve_bytes, 1_500)
        self.assertEqual(capped.device_bytes, 4_000)

    def test_unknown_host_uses_explicit_cap_and_errors_without_one(self):
        with patch.object(runtime_memory, "host_available_memory", return_value=None) as host:
            capped = runtime_memory.resolve_memory_limits(
                self._budget(host=1_234), backend=np,
            )
            with self.assertRaisesRegex(BudgetError, "cannot query available host memory"):
                runtime_memory.resolve_memory_limits(self._budget(), backend=np)

        self.assertEqual(capped.host_available_bytes, None)
        self.assertEqual(capped.host_bytes, 1_234)
        self.assertEqual(host.call_count, 2)

    def test_missing_observations_are_queried_again_on_each_resolution(self):
        backend = self._cupy_module()
        with (
            patch.object(runtime_memory, "host_available_memory", side_effect=(10_000, 11_000)) as host,
            patch.object(runtime_memory, "device_memory_info", side_effect=((6_000, 10_000), (5_000, 10_000))) as device,
        ):
            first = runtime_memory.resolve_memory_limits(self._budget(), backend=backend)
            second = runtime_memory.resolve_memory_limits(self._budget(), backend=backend)

        self.assertEqual(host.call_count, 2)
        self.assertEqual(device.call_count, 2)
        self.assertEqual((first.host_available_bytes, first.device_free_bytes), (10_000, 6_000))
        self.assertEqual((second.host_available_bytes, second.device_free_bytes), (11_000, 5_000))

    def test_actual_available_bytes_returns_fresh_observations(self):
        backend = self._cupy_module()
        with (
            patch.object(runtime_memory, "host_available_memory", side_effect=(300, 250)),
            patch.object(runtime_memory, "device_memory_info", side_effect=((90, 100), (80, 100))),
        ):
            first = runtime_memory.actual_available_bytes(backend)
            second = runtime_memory.actual_available_bytes(backend)

        self.assertEqual(first, (300, 90))
        self.assertEqual(second, (250, 80))

    def test_gpu_query_failure_is_a_budget_error(self):
        backend = self._cupy_module()
        backend.cuda = SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=Mock(side_effect=OSError("device unavailable")))
        )
        with patch.object(runtime_memory, "host_available_memory", return_value=1_000):
            with self.assertRaisesRegex(BudgetError, "unable to query CuPy device memory"):
                runtime_memory.resolve_memory_limits(self._budget(), backend=backend)

    def test_backend_normalization_and_pure_resolver_delegation(self):
        normalized = ModuleType("numpy")
        expected = MemoryLimits(host_bytes=700, device_bytes=None, host_available_bytes=1_000)
        with (
            patch.object(runtime_memory.importlib, "import_module", return_value=normalized) as importer,
            patch.object(runtime_memory._planning_memory, "resolve_budget_limits", return_value=expected) as resolve,
        ):
            result = runtime_memory.resolve_memory_limits(
                self._budget(), backend="NumPy", host_available=1_000,
            )

        self.assertIs(result, expected)
        importer.assert_called_once_with("numpy")
        resolve.assert_called_once_with(
            self._budget(), target="cpu", host_available=1_000, device_info=None,
        )
        with self.assertRaisesRegex(ValueError, "backend must be NumPy or CuPy"):
            runtime_memory.resolve_memory_limits(self._budget(), backend="cuda")

    def test_ensure_within_budget_preserves_c_types_and_error_category(self):
        estimate = MemoryEstimate(input_bytes=100)
        limits = MemoryLimits(host_bytes=99, device_bytes=None)
        with self.assertRaisesRegex(BudgetExceededError, "resolved host budget is 99 bytes"):
            runtime_memory.ensure_within_budget(
                estimate, self._budget(), backend=np, limits=limits,
            )
        self.assertIs(
            runtime_memory.ensure_within_budget(
                MemoryEstimate(input_bytes=50), self._budget(), backend=np,
                limits=limits,
            ),
            limits,
        )
        with self.assertRaisesRegex(TypeError, "estimate must be a MemoryEstimate"):
            runtime_memory.ensure_within_budget(object(), self._budget(), limits=limits)

    def test_fresh_execution_limits_cpu_keeps_cpu_resolution_path(self):
        budget = self._budget(host=2_000)
        expected = MemoryLimits(host_bytes=2_000, device_bytes=None)
        with (
            patch.object(runtime_memory, "resolve_memory_limits", return_value=expected) as resolve,
            patch.object(
                runtime_memory,
                "device_memory_info",
                side_effect=AssertionError("CPU fresh guard must not query device memory"),
            ) as device,
        ):
            actual = runtime_memory.fresh_execution_limits(budget, backend=np)

        self.assertIs(actual, expected)
        resolve.assert_called_once_with(budget, np)
        device.assert_not_called()

    def test_fresh_execution_limits_samples_once_per_call_and_preserves_pool_fields(self):
        backend = self._cupy_module()
        driver_query = Mock(side_effect=((6_000, 10_000), (3_000, 10_000)))
        pool_free = Mock(side_effect=(1_000, 500))
        pool_query = Mock(return_value=SimpleNamespace(free_bytes=pool_free))
        backend.cuda = SimpleNamespace(runtime=SimpleNamespace(memGetInfo=driver_query))
        backend.get_default_memory_pool = pool_query
        budget = self._budget(device=5_500)

        with (
            patch.object(runtime_memory, "host_available_memory", side_effect=(10_000, 12_000)) as host,
            patch.object(
                runtime_memory,
                "actual_available_bytes",
                side_effect=AssertionError("fresh execution limits must not resample observations"),
            ) as actual_available,
        ):
            first = runtime_memory.fresh_execution_limits(budget, backend)
            second = runtime_memory.fresh_execution_limits(budget, backend)

        self.assertEqual(driver_query.call_count, 2)
        self.assertEqual(pool_query.call_count, 2)
        self.assertEqual(pool_free.call_count, 2)
        self.assertEqual(host.call_count, 2)
        actual_available.assert_not_called()
        self.assertEqual(first.host_available_bytes, 10_000)
        self.assertEqual(first.device_driver_free_bytes, 6_000)
        self.assertEqual(first.device_pool_free_bytes, 1_000)
        self.assertEqual(first.device_free_bytes, 7_000)
        self.assertEqual(first.device_bytes, 5_500)
        self.assertEqual(second.host_available_bytes, 12_000)
        self.assertEqual(second.device_driver_free_bytes, 3_000)
        self.assertEqual(second.device_pool_free_bytes, 500)
        self.assertEqual(second.device_free_bytes, 3_500)
        self.assertEqual(second.device_bytes, 2_500)

    def test_fresh_execution_limits_pool_query_failure_uses_zero_pool_bytes(self):
        backend = self._cupy_module()
        driver_query = Mock(return_value=(5_000, 10_000))
        pool_free = Mock(side_effect=RuntimeError("pool unavailable"))
        backend.cuda = SimpleNamespace(runtime=SimpleNamespace(memGetInfo=driver_query))
        backend.get_default_memory_pool = Mock(return_value=SimpleNamespace(free_bytes=pool_free))

        with patch.object(runtime_memory, "host_available_memory", return_value=10_000):
            limits = runtime_memory.fresh_execution_limits(self._budget(), backend)

        self.assertEqual(driver_query.call_count, 1)
        self.assertEqual(backend.get_default_memory_pool.call_count, 1)
        self.assertEqual(limits.device_driver_free_bytes, 5_000)
        self.assertEqual(limits.device_pool_free_bytes, 0)
        self.assertEqual(limits.device_free_bytes, 5_000)
        self.assertEqual(limits.device_bytes, 4_000)

    def test_fresh_execution_limits_driver_failure_uses_canonical_budget_error(self):
        backend = self._cupy_module()
        driver_query = Mock(side_effect=OSError("driver unavailable"))
        backend.cuda = SimpleNamespace(runtime=SimpleNamespace(memGetInfo=driver_query))
        backend.get_default_memory_pool = Mock()

        with self.assertRaises(BudgetError) as captured:
            runtime_memory.fresh_execution_limits(self._budget(), backend)

        self.assertIs(type(captured.exception), BudgetError)
        self.assertNotIsInstance(captured.exception, BudgetExceededError)
        self.assertEqual(driver_query.call_count, 1)
        backend.get_default_memory_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
