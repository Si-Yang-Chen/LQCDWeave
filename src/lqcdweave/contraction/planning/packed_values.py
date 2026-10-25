"""Metadata-only admission of optional operand packing residency."""
from dataclasses import replace


_MAX_PACKED_VALUE_BYTES = 256 * 1024**2


def admit_packed_values(plan):
    """Reserve bounded cache storage without making a feasible path fail.

    Keep three quarters of spare device capacity available for IO/outputs.
    Opaque preparation callbacks and CPU plans retain their existing envelope.
    No values, devices, layout kernels or new contraction choices enter P.
    """
    if (not plan.feasible or plan.state.problem.backend != 'cupy'
            or plan.packed_value_cache_device_bytes
            or any(prep.recipe is None for prep in plan.preparations.values())):
        return plan
    capacity = min(_MAX_PACKED_VALUE_BYTES,
                   max(0, plan.memory_limit - plan.peak_device_bytes) // 4)
    if not capacity:
        return plan
    return replace(plan, packed_value_cache_device_bytes=capacity,
        peak_device_bytes=plan.peak_device_bytes + capacity,
        memory=tuple(replace(point, packed_value_cache_device_bytes=capacity)
                     for point in plan.memory))
