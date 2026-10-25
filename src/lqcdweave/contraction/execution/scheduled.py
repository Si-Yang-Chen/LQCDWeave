"""Synchronous execution for persisted, memory-bounded contraction batches.

``execute_bundle`` receives already compiled batches and
only binds fresh values for one configuration; this boundary is intentional so
that a cfg loop cannot silently perform planner work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import numbers
import time
from typing import Any

import numpy as np

from lqcdweave.contracts.memory import (
    BudgetExceededError,
    MemoryEstimate,
    MemoryLimits,
)
from .errors import (
    PlanMismatchError,
    SourceIdentityError,
    UnsupportedLoaderError,
)
from lqcdweave.runtime.memory import (
    fresh_execution_limits as _fresh_runtime_limits,
)
from ..planning.api import estimate_plan_memory
from lqcdweave.contracts.correlator.api import ContractionTerm, TensorSpec
from .executor import execute_plan, prepare_plan_kernels


def _metadata_mapping(workload: Any) -> dict[str, Any]:
    metadata = getattr(workload, "metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("workload.metadata must be a mapping")
    return dict(metadata)


def _workload_parts(workload: Any) -> tuple[dict[Any, TensorSpec], tuple[ContractionTerm, ...], dict[Any, Any]]:
    specs = getattr(workload, "specs", None)
    terms = getattr(workload, "terms", None)
    sources = getattr(workload, "input_sources", {})
    if not isinstance(specs, Mapping):
        raise TypeError("workload.specs must be a mapping")
    if terms is None:
        raise TypeError("workload.terms must be a sequence")
    terms = tuple(terms)
    normalized_specs: dict[Any, TensorSpec] = {}
    for key, spec in specs.items():
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"workload spec for {key!r} must be a TensorSpec")
        normalized_specs[key] = spec
    for index, term in enumerate(terms):
        if not isinstance(term, ContractionTerm):
            raise TypeError(f"workload term {index} must be a ContractionTerm")
        missing = [key for key in term.inputs if key not in normalized_specs]
        if missing:
            raise PlanMismatchError(f"workload term {index} has unknown input key(s): {missing!r}")
    if sources is None:
        sources = {}
    if not isinstance(sources, Mapping):
        raise TypeError("workload.input_sources must be a mapping")
    normalized_sources = dict(sources)
    unknown_sources = [key for key in normalized_sources if key not in normalized_specs]
    if unknown_sources:
        raise PlanMismatchError(f"input_sources contain unknown key(s): {unknown_sources!r}")
    return normalized_specs, terms, normalized_sources


def _freeze_identity(value: Any) -> str:
    """Return a deterministic comparison key for JSON-like source metadata."""

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(value) for value in item]
        return item

    try:
        return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _term_output_bytes(plan: Any) -> int:
    total = 0
    for term in getattr(plan, "terms", ()):
        total += math.prod(term.output_shape) * np.dtype(term.output_dtype).itemsize
    return int(total)


def _runtime_estimate(bundle: Any, batch: Any, *, total_outputs: int | None = None) -> MemoryEstimate:
    """Recompute a batch estimate from its immutable plan at execution time."""

    workload = bundle.workload
    metadata = _metadata_mapping(workload)
    assembly = metadata.get("assembly_extra_bytes", 0)
    if isinstance(assembly, bool) or not isinstance(assembly, numbers.Integral) or assembly < 0:
        raise PlanMismatchError("workload metadata assembly_extra_bytes is invalid")
    if total_outputs is None:
        total_outputs = sum(_term_output_bytes(item.plan) for item in bundle.batches)
    return estimate_plan_memory(
        batch.plan,
        batch.plan.specs,
        budget=bundle.budget,
        output_bytes=total_outputs,
        assembly_extra_bytes=int(assembly),
        snapshot_inputs=True,
    )


@dataclass(frozen=True, slots=True)
class PreparedBundle:
    """Validated, numerical-free execution metadata reusable across cfgs."""

    bundle: Any
    estimates: tuple[MemoryEstimate, ...]
    last_use: Mapping[Any, int]
    total_output_bytes: int
    kernels: tuple[Any, ...] = ()


def prepare_bundle_execution(bundle: Any, *, backend: Any = None) -> PreparedBundle:
    """Validate a bundle and precompute cfg-independent runtime metadata once."""

    xp = _backend_module(backend)
    workload = getattr(bundle, "workload", None)
    batches = tuple(getattr(bundle, "batches", ()))
    if workload is None or not batches:
        raise TypeError("bundle must contain a workload and at least one precompiled batch")
    specs, _, _ = _workload_parts(workload)
    last_use: dict[Any, int] = {}
    for index, batch in enumerate(batches):
        if not hasattr(batch, "plan") or not hasattr(batch, "term_indices"):
            raise TypeError(f"bundle batch {index} is malformed")
        if not hasattr(batch, "memory_estimate"):
            raise TypeError(f"bundle batch {index} has no memory estimate")
        for key in batch.plan.specs:
            if key not in specs:
                raise PlanMismatchError(f"batch {index} refers to unknown input {key!r}")
            last_use[key] = index
    total_outputs = sum(_term_output_bytes(batch.plan) for batch in batches)
    estimates = tuple(
        _runtime_estimate(bundle, batch, total_outputs=total_outputs) for batch in batches
    )
    kernels = ()
    if getattr(xp, "__name__", "").lower() == "cupy":
        kernels = tuple(prepare_plan_kernels(batch.plan) for batch in batches)
    return PreparedBundle(
        bundle=bundle,
        estimates=estimates,
        last_use=dict(last_use),
        total_output_bytes=total_outputs,
        kernels=kernels,
    )


def _backend_module(backend: Any) -> Any:
    if backend is None:
        return np
    if isinstance(backend, str):
        if backend.lower() not in ("numpy", "cupy"):
            raise ValueError("backend must be NumPy or CuPy")
        import importlib
        return importlib.import_module(backend.lower())
    name = getattr(backend, "__name__", "").lower()
    if name not in ("numpy", "cupy"):
        raise ValueError("backend must be NumPy or CuPy")
    return backend


def _is_array(value: Any, xp: Any) -> bool:
    return isinstance(value, getattr(xp, "ndarray", ()))


def _invoke_loader(loader: Any, cfg: Any, key: Any, source: Any) -> Any:
    if callable(loader):
        return loader(cfg, key, source)
    method = getattr(loader, "load", None)
    if callable(method):
        return method(cfg, key, source)
    raise UnsupportedLoaderError(
        "loader must be callable as loader(cfg, key, source) or expose load(cfg, key, source)"
    )


def _verify_loader_identity(loader: Any, cfg: Any, key: Any, source: Any) -> None:
    for name in ("source_identity", "identity"):
        method = getattr(loader, name, None)
        if not callable(method):
            continue
        try:
            actual = method(cfg, key, source)
        except TypeError:
            actual = method(cfg, key)
        if _freeze_identity(actual) != _freeze_identity(source):
            raise SourceIdentityError(
                f"loader source identity for {key!r} does not match the persisted plan"
            )
        return


def _unwrap_loaded(value: Any, source: Any, key: Any) -> Any:
    # A loader may return (array, provenance) or {value/array, source} to make
    # its source identity explicit.  A plain backend array remains valid.
    if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "shape"):
        actual_source = value[1]
        if _freeze_identity(actual_source) != _freeze_identity(source):
            raise SourceIdentityError(f"loaded source identity for {key!r} does not match the plan")
        return value[0]
    if isinstance(value, Mapping) and ("value" in value or "array" in value):
        actual_source = value.get("source", value.get("identity", source))
        if _freeze_identity(actual_source) != _freeze_identity(source):
            raise SourceIdentityError(f"loaded source identity for {key!r} does not match the plan")
        return value.get("value", value.get("array"))
    return value


def _snapshot_input(value: Any, spec: TensorSpec, xp: Any, key: Any) -> Any:
    if not _is_array(value, xp):
        raise TypeError(f"loader returned a non-{xp.__name__} array for input {key!r}")
    if tuple(value.shape) != tuple(spec.shape):
        raise PlanMismatchError(
            f"input {key!r} has shape {tuple(value.shape)!r}, expected {tuple(spec.shape)!r}"
        )
    if np.dtype(value.dtype) != np.dtype(spec.dtype):
        raise PlanMismatchError(
            f"input {key!r} has dtype {np.dtype(value.dtype)!r}, expected {np.dtype(spec.dtype)!r}"
        )
    try:
        copied = xp.array(value, copy=True)
    except Exception as exc:
        raise UnsupportedLoaderError(f"unable to snapshot loader result for input {key!r}") from exc
    if not _is_array(copied, xp):
        raise UnsupportedLoaderError(f"backend snapshot for input {key!r} is not an ndarray")
    return copied


def _host_required_bytes(estimate: MemoryEstimate, budget: Any, xp: Any) -> int:
    staging = getattr(budget, "host_staging_bytes", 0)
    if isinstance(staging, bool) or not isinstance(staging, numbers.Integral) or staging < 0:
        raise TypeError("budget.host_staging_bytes must be a non-negative integer")
    if getattr(xp, "__name__", "").lower() == "cupy":
        # Device intermediates are not host-resident, but declared loader
        # staging and the input snapshots are.
        return estimate.input_bytes + estimate.snapshot_bytes + int(staging)
    return estimate.logical_peak_bytes + int(staging)


def _runtime_guard(
    estimate: MemoryEstimate,
    budget: Any,
    xp: Any,
    *,
    extra_cached_bytes: int = 0,
    resident_bytes: int = 0,
    host_resident_bytes: int | None = None,
    startup_limits: MemoryLimits | None = None,
) -> MemoryLimits:
    if (
        isinstance(extra_cached_bytes, bool)
        or not isinstance(extra_cached_bytes, numbers.Integral)
        or extra_cached_bytes < 0
    ):
        raise TypeError("extra_cached_bytes must be a non-negative integer")
    if extra_cached_bytes:
        estimate = MemoryEstimate(
            input_bytes=estimate.input_bytes + int(extra_cached_bytes),
            snapshot_bytes=estimate.snapshot_bytes,
            intermediate_bytes=estimate.intermediate_bytes,
            output_bytes=estimate.output_bytes,
            assembly_extra_bytes=estimate.assembly_extra_bytes,
            workspace_bytes=estimate.workspace_bytes,
        )
    if isinstance(resident_bytes, bool) or not isinstance(resident_bytes, numbers.Integral) or resident_bytes < 0:
        raise TypeError("resident_bytes must be a non-negative integer")
    if host_resident_bytes is None:
        host_resident_bytes = int(resident_bytes)
    if (
        isinstance(host_resident_bytes, bool)
        or not isinstance(host_resident_bytes, numbers.Integral)
        or host_resident_bytes < 0
    ):
        raise TypeError("host_resident_bytes must be a non-negative integer")
    limits = _fresh_runtime_limits(budget, xp)
    startup = limits if startup_limits is None else startup_limits
    # First enforce the full logical model against the startup cap.  This is a
    # plan-level bound and remains stable while the process allocates arrays.
    static_limit = startup.limit_for(xp)
    if static_limit is not None and estimate.logical_peak_bytes > static_limit:
        raise BudgetExceededError(
            f"batch requires {estimate.logical_peak_bytes} logical bytes, "
            f"startup {'device' if getattr(xp, '__name__', '') == 'cupy' else 'host'} budget is {static_limit}"
        )
    incremental = max(estimate.logical_peak_bytes - int(resident_bytes), 0)
    current_limit = limits.limit_for(xp)
    if current_limit is not None and incremental > current_limit:
        raise BudgetExceededError(
            f"batch needs {incremental} additional logical bytes, "
            f"current {'device' if getattr(xp, '__name__', '') == 'cupy' else 'host'} allowance is {current_limit}"
        )
    # Host staging remains a separate constraint even for device execution.
    host_required = _host_required_bytes(estimate, budget, xp)
    host_incremental = max(host_required - int(host_resident_bytes), 0)
    startup_host_limit = startup.host_bytes
    if startup_host_limit is not None and host_required > startup_host_limit:
        raise BudgetExceededError(
            f"batch requires {host_required} host/staging bytes, startup host budget is {startup_host_limit}"
        )
    if limits.host_bytes is not None and host_incremental > limits.host_bytes:
        raise BudgetExceededError(
            f"batch needs {host_incremental} additional host/staging bytes, host allowance is {limits.host_bytes}"
        )
    return limits


@dataclass(frozen=True, slots=True)
class BundleExecutionStats:
    """Timing, cache and memory observations from one cfg execution."""

    timings: Mapping[str, float]
    bytes_read: int
    bytes_bound: int
    cache_hits: int
    cache_misses: int
    logical_peak_bytes: int
    actual_free_bytes: int | None = None
    actual_host_available_bytes: int | None = None
    batches: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "timings": dict(self.timings),
            "timing_semantics": (
                "read/bind are host call spans; asynchronous getter/copy GPU work "
                "may complete during execute; total is synchronous"
            ),
            "bytes_read": self.bytes_read,
            "loader_bytes": self.bytes_read,
            "bytes_bound": self.bytes_bound,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "logical_peak_bytes": self.logical_peak_bytes,
            "actual_free_bytes": self.actual_free_bytes,
            "actual_host_available_bytes": self.actual_host_available_bytes,
            "batches": [dict(item) for item in self.batches],
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


@dataclass(frozen=True, slots=True)
class BundleExecutionResult:
    """Raw term values and stats; frontend assembly happens at a higher layer."""

    values: tuple[Any, ...]
    stats: BundleExecutionStats

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.stats, BundleExecutionStats):
            raise TypeError("stats must be BundleExecutionStats")


def execute_bundle(
    bundle: Any,
    cfg: Any,
    loader: Any,
    backend: Any = None,
    *,
    prepared: PreparedBundle | None = None,
) -> BundleExecutionResult:
    """Execute all precompiled batches for one cfg synchronously.

    Loader arrays are copied immediately after each request.  The per-cfg cache
    retains a key only until its last scheduled use, so no numerical value can
    cross a call to this function or a different configuration.
    """

    if prepared is None:
        prepared = prepare_bundle_execution(bundle, backend=backend)
    elif not isinstance(prepared, PreparedBundle) or prepared.bundle is not bundle:
        raise PlanMismatchError("prepared execution metadata belongs to a different bundle")
    workload = bundle.workload
    budget = bundle.budget
    batches = tuple(bundle.batches)
    specs, terms, sources = _workload_parts(workload)
    xp = _backend_module(backend)
    estimates = prepared.estimates
    last_use = dict(prepared.last_use)

    cache: dict[Any, Any] = {}
    values: list[Any] = [None] * len(terms)
    bytes_read = bytes_bound = cache_hits = cache_misses = 0
    logical_peak = 0
    timing = {"read": 0.0, "bind": 0.0, "execute": 0.0, "total": 0.0}
    batch_stats: list[Mapping[str, Any]] = []
    actual_free: int | None = None
    actual_host: int | None = None
    startup_limits = _fresh_runtime_limits(budget, xp)
    started = time.perf_counter()

    for batch_index, batch in enumerate(batches):
        # Do not trust a serialized/stale memory_estimate supplied by a caller;
        # use the validated, immutable plan-derived estimate prepared once.
        estimate = estimates[batch_index]
        logical_peak = max(logical_peak, estimate.logical_peak_bytes)
        current_keys = set(batch.plan.specs)
        cached_extra = sum(
            int(value.nbytes) for key, value in cache.items() if key not in current_keys
        )
        resident = sum(int(value.nbytes) for value in cache.values()) + sum(
            int(value.nbytes) for value in values if value is not None
        )
        limits = _runtime_guard(
            estimate, budget, xp, extra_cached_bytes=cached_extra,
            resident_bytes=resident,
            host_resident_bytes=resident if getattr(xp, "__name__", "").lower() == "numpy" else 0,
            startup_limits=startup_limits,
        )
        if limits.device_free_bytes is not None:
            actual_free = (
                limits.device_driver_free_bytes
                if limits.device_driver_free_bytes is not None
                else limits.device_free_bytes
            )
        if limits.host_available_bytes is not None:
            actual_host = limits.host_available_bytes

        bindings: dict[Any, Any] = {}
        read_start = time.perf_counter()
        for key in batch.plan.specs:
            if key in cache:
                cache_hits += 1
                bindings[key] = cache[key]
                continue
            cache_misses += 1
            source = sources.get(key)
            if source is not None and not isinstance(source, Mapping):
                raise SourceIdentityError(f"source descriptor for {key!r} must be a mapping")
            _verify_loader_identity(loader, cfg, key, source)
            loaded = _invoke_loader(loader, cfg, key, source)
            read_elapsed = time.perf_counter() - read_start
            timing["read"] += read_elapsed
            bind_start = time.perf_counter()
            loaded = _unwrap_loaded(loaded, source, key)
            bytes_read += int(getattr(loaded, "nbytes", 0))
            snapshot = _snapshot_input(loaded, specs[key], xp, key)
            cache[key] = snapshot
            bindings[key] = snapshot
            bytes_bound += int(snapshot.nbytes)
            timing["bind"] += time.perf_counter() - bind_start
            read_start = time.perf_counter()
        timing["read"] += time.perf_counter() - read_start

        # Loader scratch may be released only after the call returns.  Repeat
        # the guard now, before any contraction kernel is launched.
        retained_extra = sum(
            int(value.nbytes) for key, value in cache.items() if key not in current_keys
        )
        resident = sum(int(value.nbytes) for value in cache.values()) + sum(
            int(value.nbytes) for value in values if value is not None
        )
        limits = _runtime_guard(
            estimate, budget, xp, extra_cached_bytes=retained_extra,
            resident_bytes=resident,
            host_resident_bytes=resident if getattr(xp, "__name__", "").lower() == "numpy" else 0,
            startup_limits=startup_limits,
        )
        if limits.device_free_bytes is not None:
            actual_free = (
                limits.device_driver_free_bytes
                if limits.device_driver_free_bytes is not None
                else limits.device_free_bytes
            )
        if limits.host_available_bytes is not None:
            actual_host = limits.host_available_bytes

        execute_start = time.perf_counter()
        try:
            result = execute_plan(
                batch.plan, bindings, backend=xp,
                kernels=prepared.kernels[batch_index] if prepared.kernels else None,
            )
        except (MemoryError, RuntimeError) as exc:
            # A backend OOM is terminal for this cfg.  Do not retry because a
            # retry can observe a partially populated loader or output set.
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower() or "memory" in str(exc).lower():
                raise BudgetExceededError(f"batch {batch_index} exceeded runtime memory") from exc
            raise
        timing["execute"] += time.perf_counter() - execute_start
        term_indices = tuple(int(index) for index in batch.term_indices)
        if len(term_indices) != len(result.values):
            raise PlanMismatchError(
                f"batch {batch_index} returned {len(result.values)} values for {len(term_indices)} terms"
            )
        for term_index, value in zip(term_indices, result.values, strict=True):
            if term_index < 0 or term_index >= len(values) or values[term_index] is not None:
                raise PlanMismatchError(f"batch {batch_index} returned an invalid term index {term_index}")
            if not _is_array(value, xp):
                raise PlanMismatchError(f"batch {batch_index} returned a non-backend output")
            # execute_plan returns independently owned outputs and synchronizes
            # CuPy work before returning.  Retaining that object avoids an
            # unaccounted second output copy.
            values[term_index] = value

        next_keys = set(batches[batch_index + 1].plan.specs) if batch_index + 1 < len(batches) else set()
        for key in tuple(cache):
            # Retain only values needed by the immediately following batch.
            # A non-adjacent future use is reloaded, which bounds residency and
            # keeps the cache strictly within this cfg execution.
            if last_use.get(key, batch_index) <= batch_index or key not in next_keys:
                del cache[key]
        del bindings
        if "loaded" in locals():
            del loaded
        if "snapshot" in locals():
            del snapshot
        del result
        batch_stats.append({
            "index": batch_index,
            "term_indices": term_indices,
            "logical_peak_bytes": estimate.logical_peak_bytes,
            "cache_size": len(cache),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        })

    if any(value is None for value in values):
        raise PlanMismatchError("bundle batches did not produce every workload term")
    timing["total"] = time.perf_counter() - started
    return BundleExecutionResult(
        values=tuple(values),
        stats=BundleExecutionStats(
            timings=timing,
            bytes_read=bytes_read,
            bytes_bound=bytes_bound,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            logical_peak_bytes=logical_peak,
            actual_free_bytes=actual_free,
            actual_host_available_bytes=actual_host,
            batches=tuple(batch_stats),
        ),
    )


__all__ = [
    "BundleExecutionResult",
    "BundleExecutionStats",
    "execute_bundle",
    "PreparedBundle",
    "prepare_bundle_execution",
]
