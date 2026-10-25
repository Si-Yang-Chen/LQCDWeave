"""Resolve declared execution targets before handing metadata to planning."""
from dataclasses import replace
import logging

from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.performance.api import (
    load_execution_target, resolve_target_profile,
)
from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
from lqcdweave.adapters.execution_target import observe_execution_target, declared_backend_runtime


def resolve_planning_target(target_hardware, backend, *, memory_limit, host_memory_limit, profile):
    target = load_execution_target(target_hardware) if isinstance(target_hardware, str) else target_hardware
    supplied_backend = backend if isinstance(backend, str) else getattr(backend, '__name__', None)
    target_backend = target.backend if isinstance(target, ExecutionTarget) else supplied_backend or 'numpy'
    memory = target.memory if isinstance(target, ExecutionTarget) else target
    limits = resolve_diagram_memory_limits(backend=target_backend,
        memory_limit=memory_limit, host_memory_limit=host_memory_limit,
        execution_mode='offline' if memory is not None else 'local', target=memory)
    if limits.host_bytes is None:
        raise ValueError('execution target is missing a host memory budget; supply host_memory_limit or register it')
    if isinstance(target, TargetHardwareMemory):
        target = ExecutionTarget('declared-' + target_backend, target_backend,
            declared_backend_runtime(backend or target_backend), target)
    elif target is None:
        target = observe_execution_target(backend or target_backend, limits)
    document = resolve_target_profile(target, profile)
    summary = target_summary(target, limits, document)
    message = (f"Execution target: {summary['name']} [{summary['backend']}/{summary['runtime']}] | "
        f"host budget={summary['host_memory_limit']} bytes | "
        f"device budget={summary['device_memory_limit']} bytes | "
        f"model={summary['profile_source']} ({summary['calibration_status']})")
    logger = logging.getLogger('lqcdweave')
    if logger.isEnabledFor(logging.INFO):
        logger.info(message)
    else:
        print(message, flush=True)
    return target, limits, document, summary


def target_summary(target, limits, document):
    return dict(name=target.name, backend=target.backend, runtime=target.runtime,
        host_memory_limit=limits.host_bytes, device_memory_limit=limits.device_bytes,
        profile_source=document['source'], calibration_status=document['calibration_status'],
        coverage=document.get('coverage', {}))


def calibrate_if_needed(target, limits, document, *, backend, dtypes, profile,
                        reserved_host_bytes=0, reserved_device_bytes=0):
    if profile is not None:
        return document
    from lqcdweave.contraction.performance.api import load_target_calibration, calibrate_target_profile
    from lqcdweave.contraction.execution.api import probe_compute_profile
    from lqcdweave.adapters.execution_target import check_execution_target
    backend = check_execution_target(target, backend)
    observed = observe_execution_target(backend, limits)
    if target.hardware_fingerprint not in ('unmeasured', observed.hardware_fingerprint):
        raise ValueError(f'calibration must run on the registered execution target {target.name!r}')
    # A declared target is only calibrated on the matching runtime. Cache keys
    # use actual hardware/software, not the arbitrary user-facing alias.
    key = observed.hardware_fingerprint + ':unified-device-v1'
    cached = load_target_calibration(target, cache_key=key)
    covered = () if cached is None else cached.get('provenance', {}).get('covered_dtypes', ())
    def with_coverage(result):
        calibrated = tuple(result.get('provenance', {}).get('covered_operations', ()))
        modeled = ('contract', 'snapshot', 'preparation', 'packing', 'accumulation',
                   'baryon_join', 'sparse_bridge')
        return dict(result, coverage=dict(document.get('coverage', {}),
            profile_source=result['source'], calibration_status=result['calibration_status'],
            calibrated_operations=calibrated,
            missing_operations=tuple(op for op in modeled if op not in calibrated)))
    if cached is not None and set(dtypes) <= set(covered):
        return with_coverage(dict(document, **cached))
    provenance = document.get('provenance', {})
    if ({'contract', 'snapshot'} <= set(provenance.get('covered_operations', ()))
            and set(dtypes) <= set(provenance.get('covered_dtypes', ()))):
        return document
    available = resolve_diagram_memory_limits(backend=backend,
        memory_limit=limits.device_bytes if target.backend=='cupy' else limits.host_bytes,
        host_memory_limit=limits.host_bytes, execution_mode='local')
    cap = min([64 << 20] + [max(0, v-r) for v,r in (
        (available.host_bytes,reserved_host_bytes),(available.device_bytes,reserved_device_bytes)) if v is not None])
    requested = tuple(sorted(set(dtypes) | set(covered))) or ('complex128',)
    supported = tuple(dtype for dtype in requested if dtype in {'float32','float64','complex64','complex128'})
    if not supported:
        return dict(document, calibration=dict(calibrated=False, missing=('dtype probe',)))
    report = probe_compute_profile(backend, dtypes=supported, max_bytes=cap)
    return with_coverage(calibrate_target_profile(
        replace(target, hardware_fingerprint=observed.hardware_fingerprint),
        document, report, cache_key=key))


__all__ = ['resolve_planning_target', 'target_summary', 'calibrate_if_needed']
