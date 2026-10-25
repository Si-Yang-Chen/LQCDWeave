"""Shared byte geometry for CuPy's pooled device allocation requests."""

from ..memory import DEFAULT_PLANNING_FRACTION

DEVICE_ALLOCATION_ALIGNMENT = 512
_CUPY_POOL_HEADROOM_BYTES = 1024 ** 3


def device_allocation_bytes(nbytes):
    """Round one allocation owner, never a collection of result views.

    CuPy's default device pool rounds individual requests to 512-byte units.
    This accounts for that granularity only, not pool fragmentation, retained
    split blocks, CUDA libraries or allocations owned by an external caller.
    """
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError('allocation bytes must be a nonnegative integer')
    return (nbytes + DEVICE_ALLOCATION_ALIGNMENT - 1) // DEVICE_ALLOCATION_ALIGNMENT * DEVICE_ALLOCATION_ALIGNMENT


def device_allocator_headroom(memory_limit, *, backend='cupy', device_name=None):
    """Describe an empirical allowance inside the unchanged pool hard limit.

    One GiB is an empirical default allowance, not a general bound on
    fragmentation or a calibration of the current device. Small budgets cap
    it by the existing default planning slack. Measurement-specific evidence
    belongs to the calling project; this helper does not probe a GPU.
    """
    if type(memory_limit) is not int or memory_limit <= 0:
        raise ValueError('memory_limit must be positive integer bytes')
    if backend not in ('numpy', 'cupy'):
        raise ValueError('backend must identify numpy or cupy')
    if device_name is not None and not isinstance(device_name, str):
        raise TypeError('device_name must be a string or None')
    allowance = memory_limit - int(memory_limit * DEFAULT_PLANNING_FRACTION)
    reserve = min(_CUPY_POOL_HEADROOM_BYTES, allowance) if backend == 'cupy' else 0
    status = ('not_applicable' if backend == 'numpy' else
              'small_budget_fallback' if reserve < _CUPY_POOL_HEADROOM_BYTES else
              'empirical_default')
    return dict(policy='cupy-pool-headroom-v1', reserve_bytes=reserve,
        hard_limit_bytes=memory_limit, numerical_limit_bytes=memory_limit-reserve,
        device_name=device_name, calibration_status=status,
        default_planning_fraction=DEFAULT_PLANNING_FRACTION,
        small_budget_policy='cap the allowance by existing default planning slack',
        source=dict(kind='library_default', runtime_calibrated=False,
            empirical_allowance_bytes=_CUPY_POOL_HEADROOM_BYTES),
        limitation='empirical allocator allowance, not a proof or upper bound on fragmentation')


__all__ = ['DEVICE_ALLOCATION_ALIGNMENT', 'device_allocation_bytes', 'device_allocator_headroom']
