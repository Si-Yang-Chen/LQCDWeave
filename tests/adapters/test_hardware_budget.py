"""Target and local memory resolution before Diagram planning."""

import unittest
from unittest.mock import patch

from lqcdweave.adapters.hardware_budget import (
    DEFAULT_PLANNING_FRACTION,
    resolve_diagram_memory_limits,
)
from lqcdweave.contracts.memory import BudgetError, TargetHardwareMemory


class HardwareBudgetTests(unittest.TestCase):
    def test_target_device_free_includes_existing_use_and_default_is_eighty_percent(self):
        target = TargetHardwareMemory(
            host_available_bytes=10_000,
            device_total_bytes=16_000,
            device_free_bytes=8_000,
        )
        self.assertEqual(target.device_used_bytes, 8_000)
        with patch("lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes",
                   side_effect=AssertionError("offline planning probed local host")), \
             patch("lqcdweave.runtime.memory.device_memory_info",
                   side_effect=AssertionError("offline planning probed local device")):
            limits = resolve_diagram_memory_limits(
                backend="cupy", execution_mode="offline", target=target,
            )
        self.assertEqual(limits.host_bytes, int(10_000 * DEFAULT_PLANNING_FRACTION))
        self.assertEqual(limits.device_bytes, int(8_000 * DEFAULT_PLANNING_FRACTION))
        self.assertEqual(limits.device_free_bytes, 8_000)
        self.assertEqual(limits.device_total_bytes, 16_000)
        self.assertIsNone(limits.device_reserve_bytes)

    def test_offline_device_used_form_derives_free_and_checks_consistency(self):
        target = TargetHardwareMemory(device_total_bytes=12_000, device_used_bytes=5_000)
        self.assertEqual(target.device_free_bytes, 7_000)
        self.assertEqual(target.device_used_bytes, 5_000)
        with self.assertRaisesRegex(ValueError, "sum to device_total_bytes"):
            TargetHardwareMemory(device_total_bytes=12_000,
                                 device_free_bytes=8_000, device_used_bytes=5_000)
        with self.assertRaisesRegex(ValueError, "required with device free/used"):
            TargetHardwareMemory(device_free_bytes=2_000)

    def test_explicit_limits_replace_fraction_but_cannot_exceed_known_capacity(self):
        target = TargetHardwareMemory(host_available_bytes=10_000,
                                      device_total_bytes=10_000,
                                      device_free_bytes=6_000)
        limits = resolve_diagram_memory_limits(
            backend="cupy", execution_mode="offline", target=target,
            memory_limit=7_000, host_memory_limit=9_000,
        )
        self.assertEqual(limits.device_bytes, 6_000)
        self.assertEqual(limits.host_bytes, 9_000)

    def test_offline_explicit_budget_is_sufficient_only_for_its_resource(self):
        with patch("lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes",
                   side_effect=AssertionError("offline planning probed local host")), \
             patch("lqcdweave.runtime.memory.device_memory_info",
                   side_effect=AssertionError("offline planning probed local device")):
            cpu = resolve_diagram_memory_limits(
                backend="numpy", execution_mode="offline", memory_limit=2_000,
            )
            gpu = resolve_diagram_memory_limits(
                backend="cupy", execution_mode="offline", memory_limit=3_000,
                host_memory_limit=1_000,
            )
            with self.assertRaisesRegex(BudgetError, "target memory specification"):
                resolve_diagram_memory_limits(backend="numpy", execution_mode="offline")
            with self.assertRaisesRegex(BudgetError, "target memory specification"):
                resolve_diagram_memory_limits(backend="cupy", execution_mode="offline")
        self.assertEqual(cpu.host_bytes, 2_000)
        self.assertEqual(gpu.device_bytes, 3_000)
        self.assertEqual(gpu.host_bytes, 1_000)

    def test_local_host_uses_tightest_system_cgroup_and_rlimit_headroom(self):
        with patch("lqcdweave.runtime.memory.host_available_memory",
                   return_value=10_000), \
             patch("lqcdweave.adapters.hardware_budget._cgroup_memory_headroom_bytes",
                   return_value=7_000), \
             patch("lqcdweave.adapters.hardware_budget._rlimit_address_space_headroom_bytes",
                   return_value=5_000):
            limits = resolve_diagram_memory_limits(backend="numpy")
        self.assertEqual(limits.host_available_bytes, 5_000)
        self.assertEqual(limits.host_bytes, 4_000)

    def test_local_gpu_free_memory_is_not_reduced_by_used_memory_again(self):
        with patch("lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes",
                   return_value=None), \
             patch("lqcdweave.runtime.memory.device_memory_info",
                   return_value=(6_000, 10_000)) as query:
            limits = resolve_diagram_memory_limits(backend="cupy")
            explicit = resolve_diagram_memory_limits(backend="cupy", memory_limit=5_500)
        self.assertEqual(limits.device_bytes, 4_800)
        self.assertEqual(limits.device_free_bytes, 6_000)
        self.assertEqual(limits.device_total_bytes, 10_000)
        self.assertEqual(explicit.device_bytes, 5_500)
        self.assertEqual(query.call_count, 2)

    def test_invalid_backend_mode_and_fractional_byte_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "numpy or cupy"):
            resolve_diagram_memory_limits(backend="torch", memory_limit=100)
        with self.assertRaisesRegex(ValueError, "local.*offline"):
            resolve_diagram_memory_limits(execution_mode="cluster")
        with self.assertRaisesRegex(TypeError, "nonnegative integer"):
            TargetHardwareMemory(host_available_bytes=10.5)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            resolve_diagram_memory_limits(backend="numpy", memory_limit=0)


if __name__ == "__main__":
    unittest.main()
