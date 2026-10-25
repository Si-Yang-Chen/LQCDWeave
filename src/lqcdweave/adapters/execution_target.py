"""Observe an explicitly selected runtime; offline target lookup lives in M."""
import hashlib
import importlib
import json
import platform
import re
from functools import lru_cache

from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.memory import TargetHardwareMemory, MemoryLimits


def resolve_array_backend(backend):
    name = backend if isinstance(backend, str) else getattr(backend, '__name__', None)
    if name not in ('numpy', 'cupy'):
        raise ValueError('execution backend must be numpy or cupy')
    return importlib.import_module(name) if isinstance(backend, str) else backend


def declared_backend_runtime(backend):
    """Read an already supplied module's build flag without probing a device."""
    name = backend if isinstance(backend, str) else getattr(backend, '__name__', None)
    if name == 'numpy':
        return 'cpu'
    if name != 'cupy':
        raise ValueError('execution backend must be numpy or cupy')
    return 'hip' if not isinstance(backend, str) and backend.cuda.runtime.is_hip else 'cuda'


def observe_execution_target(backend, limits):
    backend = resolve_array_backend(backend)
    name = backend.__name__
    software = [('backend', str(getattr(backend, '__version__', 'unknown')))]
    hardware = dict(backend=name, machine=platform.machine(), processor=platform.processor())
    runtime = 'cpu'
    device_name = hardware['processor'] or hardware['machine'] or 'cpu'
    if name == 'cupy':
        api = backend.cuda.runtime
        properties = api.getDeviceProperties(api.getDevice())
        device_name = properties.get('name', 'accelerator')
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8')
        runtime = 'hip' if api.is_hip else 'cuda'
        software += [('runtime', str(api.runtimeGetVersion())), ('driver', str(api.driverGetVersion()))]
        hardware.update(device=str(device_name), runtime=runtime)
    hardware['software'] = software
    fingerprint = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()
    label = re.sub(r'[^A-Za-z0-9._-]+', '-', str(device_name)).strip('-') or runtime
    memory = TargetHardwareMemory(host_available_bytes=limits.host_bytes,
        device_total_bytes=limits.device_total_bytes or limits.device_bytes,
        device_free_bytes=limits.device_bytes)
    return ExecutionTarget(label, name, runtime, memory,
                           hardware_fingerprint=fingerprint, software=tuple(software))


@lru_cache(maxsize=16)
def _checked_fingerprint(backend, device, context, version, runtime, driver, machine, processor):
    return observe_execution_target(backend, MemoryLimits(None, None)).hardware_fingerprint


def _current_fingerprint(backend):
    """Reuse immutable hardware facts, never available memory observations."""
    try:
        hash(backend)
    except TypeError:
        return observe_execution_target(backend, MemoryLimits(None, None)).hardware_fingerprint
    if backend.__name__ == 'cupy':
        api = backend.cuda.runtime
        context = getattr(getattr(backend.cuda, 'driver', None), 'ctxGetCurrent', None)
        # CuPy's HIP driver exports ctxGetCurrent, but implements it as an
        # unsupported stub. Without a context key, observe hardware afresh.
        if api.is_hip or not callable(context):
            return observe_execution_target(backend, MemoryLimits(None, None)).hardware_fingerprint
        identity = (api.getDevice(), context(), api.runtimeGetVersion(), api.driverGetVersion())
    else:
        identity = (None, None, None, None)
    return _checked_fingerprint(backend, identity[0], identity[1],
        str(getattr(backend, '__version__', 'unknown')), identity[2], identity[3],
        platform.machine(), platform.processor())


def check_execution_target(target, backend):
    """One startup compatibility check, never an implicit backend substitution."""
    backend = resolve_array_backend(backend)
    if target is not None:
        if backend.__name__ != target.backend:
            raise ValueError(f'execution backend differs from target {target.name!r}')
        if target.backend == 'cupy':
            actual = 'hip' if backend.cuda.runtime.is_hip else 'cuda'
            if actual != target.runtime:
                raise ValueError(f'execution runtime differs from target {target.name!r}')
        versions = dict(target.software)
        declared = versions.get('backend')
        if declared is not None and declared != str(getattr(backend, '__version__', 'unknown')):
            raise ValueError(f'execution software differs from target {target.name!r}')
        if target.hardware_fingerprint != 'unmeasured':
            if _current_fingerprint(backend) != target.hardware_fingerprint:
                raise ValueError(f'execution hardware differs from target {target.name!r}')
    return backend


__all__ = ['resolve_array_backend', 'declared_backend_runtime',
           'observe_execution_target', 'check_execution_target']
