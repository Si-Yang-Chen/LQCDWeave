"""Resolve explicit and locally observable memory limits before planning.

This adapter only resolves capacities. Scene/workflow planners remain
responsible for counting their declared reserved, retained, and workspace
bytes inside the returned limits.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import os
import re
import numbers

from lqcdweave.contracts.memory import (
    BudgetError,
    DEFAULT_PLANNING_FRACTION,
    MemoryLimits,
    TargetHardwareMemory,
)
from lqcdweave.runtime import memory as runtime_memory


def _backend_name(backend):
    if backend is None:
        return "numpy"
    if isinstance(backend, str):
        name = backend.lower()
    else:
        name = str(getattr(backend, "__name__", "")).lower()
    if name not in ("numpy", "cupy"):
        raise ValueError("backend must identify numpy or cupy")
    return name


def _bytes(value, name, *, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer number of bytes")
    value = int(value)
    if value < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer number of bytes")
    return value


def _effective_local_host_available_bytes():
    """Take the tightest readable system, cgroup, and address-space headroom."""
    candidates = []
    system = runtime_memory.host_available_memory()
    if system is not None:
        candidates.append(_bytes(system, "system available memory"))
    cgroup = _cgroup_memory_headroom_bytes()
    if cgroup is not None:
        candidates.append(_bytes(cgroup, "cgroup memory headroom"))
    address_space = _rlimit_address_space_headroom_bytes()
    if address_space is not None:
        candidates.append(_bytes(address_space, "RLIMIT_AS headroom"))
    return min(candidates) if candidates else None


def _unescape_mount_field(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_mount_candidates():
    """Yield mounted cgroup directories for this process and their ancestors."""
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="ascii")
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return ()

    return _parse_cgroup_mounts(os.getpid(), memberships, mount_lines)


@lru_cache(maxsize=8)
def _parse_cgroup_mounts(pid, memberships, mount_lines):
    """Cache topology only; membership/mount changes select a new entry."""
    v2_path = None
    v1_path = None
    for line in memberships.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        if fields[0] == "0" and fields[1] == "":
            v2_path = fields[2]
        elif "memory" in fields[1].split(","):
            v1_path = fields[2]

    result = []
    for line in mount_lines.splitlines():
        if " - " not in line:
            continue
        before, after = line.split(" - ", 1)
        left, right = before.split(), after.split()
        if len(left) < 5 or len(right) < 3:
            continue
        filesystem = right[0]
        super_options = set(right[-1].split(","))
        if filesystem == "cgroup2" and v2_path is not None:
            member = v2_path
        elif filesystem == "cgroup" and "memory" in super_options and v1_path is not None:
            member = v1_path
        else:
            continue
        mount_root = Path(_unescape_mount_field(left[3]))
        mount_point = Path(_unescape_mount_field(left[4]))
        member_path = Path(member)
        try:
            relative = member_path.relative_to(mount_root)
        except ValueError:
            relative = Path(str(member).lstrip("/"))
        current = (mount_point / relative).resolve()
        mount_point = mount_point.resolve()
        while True:
            if current == mount_point or mount_point in current.parents:
                result.append((filesystem, current))
            if current == mount_point or current.parent == current:
                break
            current = current.parent

    # Common container mounts may be readable even when procfs is hidden.
    for path, filesystem in ((Path("/sys/fs/cgroup"), "cgroup2"),
                             (Path("/sys/fs/cgroup/memory"), "cgroup")):
        if path.is_dir():
            result.append((filesystem, path))
    return tuple(dict.fromkeys(result))


def _read_integer_file(path):
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if value in ("", "max"):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0 or parsed >= (1 << 60):
        return None
    return parsed


def _cgroup_memory_headroom_bytes():
    observations = []
    for filesystem, directory in _cgroup_mount_candidates():
        if filesystem == "cgroup2":
            limit_name, current_name = "memory.max", "memory.current"
        else:
            limit_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        limit = _read_integer_file(directory / limit_name)
        current = _read_integer_file(directory / current_name)
        if limit is not None and current is not None:
            observations.append(max(0, limit - current))
    return min(observations) if observations else None


def _process_virtual_memory_bytes():
    try:
        import psutil

        memory = psutil.Process().memory_info()
        value = getattr(memory, "vms", None)
        if value is not None:
            return int(value)
    except Exception:
        pass
    if os.name == "posix":
        try:
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmSize:"):
                    value_kib = int(line.split()[1])
                    return value_kib * 1024
        except (OSError, UnicodeError, ValueError, IndexError):
            pass
    return None


def _rlimit_address_space_headroom_bytes():
    """Return finite RLIMIT_AS headroom, or None when no finite limit applies."""
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    infinity = getattr(resource, "RLIM_INFINITY", -1)
    if soft in (None, infinity, -1) or soft < 0:
        return None
    current = _process_virtual_memory_bytes()
    if current is None:
        # A finite limit with no readable current usage cannot be turned into
        # safe remaining headroom.
        return 0
    return max(0, int(soft) - current)


def _observed_device_memory(backend, device_info):
    if device_info is None:
        free, total = runtime_memory.device_memory_info(backend)
        if free is not None and total is not None:
            free = min(total, free + runtime_memory.reusable_device_bytes(backend))
    else:
        try:
            free, total = device_info
        except (TypeError, ValueError) as exc:
            raise ValueError("device_info must be a (free_bytes, total_bytes) pair") from exc
        free = _bytes(free, "device free bytes")
        total = _bytes(total, "device total bytes")
    if free is None or total is None:
        return None, None
    free = _bytes(free, "device free bytes")
    total = _bytes(total, "device total bytes")
    if free > total:
        raise ValueError("device free bytes cannot exceed total bytes")
    return free, total


def _resolved_limit(explicit, available, *, name, required):
    if explicit is not None:
        explicit = _bytes(explicit, name, allow_zero=False)
    if available is not None:
        available = _bytes(available, f"available {name}")
        cap = min(explicit, available) if explicit is not None else int(
            available * DEFAULT_PLANNING_FRACTION
        )
        if cap <= 0 and required:
            raise BudgetError(f"no usable {name} is available for planning")
        return cap
    if explicit is not None:
        return explicit
    if required:
        raise BudgetError(
            f"cannot resolve {name}; supply an explicit budget or target memory specification"
        )
    return None


def resolve_diagram_memory_limits(
    *,
    backend="numpy",
    memory_limit=None,
    host_memory_limit=None,
    execution_mode="local",
    target=None,
    host_available=None,
    device_info=None,
):
    """Resolve planning limits without guessing offline target capacities.

    Explicit limits replace the 80% default and are clamped to known effective
    availability. In local mode host availability is the minimum of system
    free memory, readable cgroup headroom, and finite process address-space
    headroom. In offline mode only ``target`` is consulted; an explicit budget
    can supply its own limit when that target resource is otherwise unknown.

    Device ``free`` already excludes current allocations. It is used directly;
    ``total - free`` is not subtracted again. The returned cap leaves 20% of
    known availability as general slack by default. Scene-level reserved bytes
    remain part of the existing plan peak and must not be passed here as a
    second subtraction.
    """
    requested_backend = backend
    backend = _backend_name(backend)
    if execution_mode not in ("local", "offline"):
        raise ValueError("execution_mode must be 'local' or 'offline'")
    if execution_mode == "local" and target is not None:
        raise ValueError("target is only valid in offline planning mode")
    if execution_mode == "offline" and (host_available is not None or device_info is not None):
        raise ValueError("offline target observations belong in TargetHardwareMemory")
    if target is not None and not isinstance(target, TargetHardwareMemory):
        raise TypeError("target must be TargetHardwareMemory or None")

    if backend == "numpy":
        explicit_host = [value for value in (memory_limit, host_memory_limit) if value is not None]
        host_cap = min(_bytes(value, "host memory limit", allow_zero=False)
                       for value in explicit_host) if explicit_host else None
        device_cap = None
    else:
        host_cap = (None if host_memory_limit is None else
                    _bytes(host_memory_limit, "host memory limit", allow_zero=False))
        device_cap = (None if memory_limit is None else
                      _bytes(memory_limit, "device memory limit", allow_zero=False))

    if execution_mode == "local":
        observed_host = (_bytes(host_available, "host available memory")
                         if host_available is not None
                         else _effective_local_host_available_bytes())
        if backend == "cupy":
            observed_free, observed_total = _observed_device_memory(
                requested_backend if requested_backend is not None else backend, device_info)
        else:
            observed_free = observed_total = None
    else:
        observed_host = target.host_available_bytes if target is not None else None
        if backend == "cupy":
            observed_free = target.device_free_bytes if target is not None else None
            observed_total = target.device_total_bytes if target is not None else None
        else:
            observed_free = observed_total = None

    resolved_host = _resolved_limit(
        host_cap, observed_host, name="host memory budget",
        required=(backend == "numpy" or host_cap is not None or observed_host is not None),
    )
    resolved_device = None
    if backend == "cupy":
        resolved_device = _resolved_limit(
            device_cap, observed_free, name="device memory budget", required=True,
        )
    return MemoryLimits(
        host_bytes=resolved_host,
        device_bytes=resolved_device,
        host_available_bytes=observed_host,
        device_free_bytes=observed_free,
        device_total_bytes=observed_total,
        device_reserve_bytes=None,
    )


__all__ = ["DEFAULT_PLANNING_FRACTION", "resolve_diagram_memory_limits"]
