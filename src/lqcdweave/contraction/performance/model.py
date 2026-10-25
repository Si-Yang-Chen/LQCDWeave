"""Compute, input-preparation, and snapshot cost model."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping as MappingABC
import math
import numpy as np

@dataclass(frozen=True)
class CostProfile:
    """Provisional model for modeled computation and preparation work."""

    snapshot_call_seconds: float = 1e-6
    snapshot_seconds_per_byte: float = 1e-10
    kernel_call_seconds: float = 2e-5
    kernel_call_seconds_float32: float = 1e-5
    kernel_call_seconds_float64: float = 1.5e-5
    kernel_call_seconds_complex64: float = 2e-5
    kernel_call_seconds_complex128: float = 2.5e-5
    flop_seconds: float = 1e-12
    traffic_seconds_per_byte: float = 5e-12
    contract_scale: float = 1.0
    sparse_bridge_scale: float = 1.0
    sparse_kernel_call_scale: float = 1.0
    sparse_flop_seconds: float = 1e-12
    sparse_cycle_flop_seconds: float = 1e-12
    sparse_traffic_seconds_per_byte: float = 5e-12
    baryon_join_scale: float = 1.0
    version: str = "compute-only-v1"
    hardware_fingerprint: str = "unmeasured"
    sample_count: int = 0
    preparation_call_seconds: float = 0.0
    preparation_kernel_call_seconds: float | None = None
    preparation_flop_seconds: float | None = None
    preparation_traffic_seconds_per_byte: float | None = None

    def __post_init__(self):
        for name in ('preparation_call_seconds', 'preparation_kernel_call_seconds',
                     'preparation_flop_seconds', 'preparation_traffic_seconds_per_byte'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and non-negative')
        if self.preparation_call_seconds is None:
            raise ValueError('preparation_call_seconds cannot be None')
        for name in (
            "snapshot_call_seconds",
            "snapshot_seconds_per_byte",
            "kernel_call_seconds",
            "kernel_call_seconds_float32",
            "kernel_call_seconds_float64",
            "kernel_call_seconds_complex64",
            "kernel_call_seconds_complex128",
            "flop_seconds",
            "traffic_seconds_per_byte",
            "contract_scale",
            "sparse_bridge_scale",
            "sparse_kernel_call_scale",
            "sparse_flop_seconds",
            "sparse_cycle_flop_seconds",
            "sparse_traffic_seconds_per_byte",
            "baryon_join_scale",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def preparation_seconds(self, flops, traffic_bytes, *, kernel_calls=0, dtype=None):
        """Estimate modeled source materialization work."""
        call = (self.contract_scale * self.kernel_call_for_dtype(dtype)
                if self.preparation_kernel_call_seconds is None else self.preparation_kernel_call_seconds)
        flop = (self.contract_scale * self.flop_seconds
                if self.preparation_flop_seconds is None else self.preparation_flop_seconds)
        traffic = (self.contract_scale * self.traffic_seconds_per_byte
                   if self.preparation_traffic_seconds_per_byte is None else self.preparation_traffic_seconds_per_byte)
        return self.preparation_call_seconds + call * kernel_calls + flop * flops + traffic * traffic_bytes

    def snapshot_seconds(self, nbytes: int) -> float:
        """Estimate the owned snapshot copy used during input preparation."""
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("snapshot bytes must be a non-negative integer")
        return (self.snapshot_call_seconds + nbytes * self.snapshot_seconds_per_byte) if nbytes else 0.0

    def gemm_group_seconds(self, matrix_shape, dtype, *, batch_count, packing_seconds):
        """Provisional dense estimate: pointer preparation plus one batched call.

        This is not dedicated batched-GEMM calibration. Device pointer writes
        and reads count as computation traffic; no host/device upload is used.
        """
        m, k, n = matrix_shape
        itemsize = np.dtype(dtype).itemsize
        traffic = batch_count * ((m*k + k*n + m*n)*itemsize + 48)
        return packing_seconds + self.contract_seconds(
            batch_count*m*k*n, traffic, dtype=dtype, kernel_calls=2)

    def kernel_call_for_dtype(self, dtype=None) -> float:
        if dtype is None:
            return self.kernel_call_seconds
        field = f"kernel_call_seconds_{np.dtype(dtype).name}"
        return getattr(self, field, self.kernel_call_seconds)

    def contract_seconds(
        self,
        flops: int,
        traffic_bytes: int,
        *,
        kind: str = "contract",
        dtype=None,
        kernel_calls: int = 1,
        sparse_records: int = 1,
    ) -> float:
        if isinstance(kernel_calls, bool) or not isinstance(kernel_calls, int) or kernel_calls < 0:
            raise ValueError("kernel_calls must be a non-negative integer")
        if sparse_records not in (1, 2):
            raise ValueError("sparse_records must be one or two")
        scales = {
            "contract": self.contract_scale,
            "sparse_bridge": self.sparse_bridge_scale,
            "baryon_join": self.baryon_join_scale,
        }
        if kind not in scales:
            raise ValueError(f"unsupported contraction cost kind {kind!r}")
        call_cost = kernel_calls * self.kernel_call_for_dtype(dtype)
        flop_cost = flops * self.flop_seconds
        traffic_cost = traffic_bytes * self.traffic_seconds_per_byte
        if kind == "sparse_bridge":
            call_cost *= self.sparse_kernel_call_scale
            flop_cost = flops * (
                self.sparse_cycle_flop_seconds if sparse_records == 2 else self.sparse_flop_seconds
            )
            traffic_cost = traffic_bytes * self.sparse_traffic_seconds_per_byte
        return scales[kind] * (call_cost + flop_cost + traffic_cost)


def cost_profile_calibration_status(profile: CostProfile, provenance=None) -> str:
    """Summarize recorded calibration evidence without treating it as validation."""
    if not isinstance(profile, CostProfile):
        raise TypeError("profile must be CostProfile")
    metadata = provenance if isinstance(provenance, MappingABC) else {}
    recorded = metadata.get("calibration_status")
    if isinstance(recorded, str) and recorded.strip():
        return recorded
    operation = metadata.get("operation")
    if operation == "weighted-merge" or profile.version == "weighted-merge-v1":
        return "merged"
    if operation == "fit":
        return "fit-recorded"
    if profile.sample_count > 0 and profile.hardware_fingerprint != "unmeasured":
        return "has-samples"
    if profile.sample_count == 0 and profile.hardware_fingerprint == "unmeasured":
        return "unmeasured"
    return "unknown"


__all__ = ["CostProfile", "cost_profile_calibration_status"]

