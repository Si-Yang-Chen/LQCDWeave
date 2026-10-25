"""One explicit execution policy for tensor, diagram and scene frontends.

Physics and loading stay in the frontends. A session owns structural plans,
never input values or getter objects. All services receive an explicit backend.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Hashable, Mapping, Sequence
import operator
from typing import Any

import numpy as np

from lqcdweave.contracts.correlator.api import (
    BaryonTerm,
    PlanBundle,
    TensorSpec,
    ExecutionStats,
    validate_frontend_workload,
)
from lqcdweave.contraction.results.api import assemble_workload_outputs
from lqcdweave.contraction.execution.api import execute_plan, ExecutionBindingCache
from lqcdweave.contraction.planning.api import (
    ContractionPlanCache, compile_plan_cached, compile_baryon_terms_cached,
)
from lqcdweave.contraction.binding.api import read_simple_diagram_inputs, read_baryon_diagram_inputs
from lqcdweave.contraction.binding.api import normalize_array_precision
from lqcdweave.contraction.expansion.api import (
    expand_simple_diagrams, validate_baryon_diagram_batch, expand_baryon_diagrams,
)
from lqcdweave.contraction.results.api import (
    assemble_simple_diagram_outputs, assemble_baryon_diagram_outputs,
)
from lqcdweave.contraction.calculation import Calculation
from lqcdweave.contraction.calculation_compat import CfgExecutionResult, session_result


# Relocated unchanged from execution/baryon.py: compilation belongs to W/P.
def _normalise_terms(terms: Sequence[BaryonTerm]) -> tuple[BaryonTerm, ...]:
    try:
        normalized = tuple(terms)
    except TypeError as exc:
        raise TypeError("terms must be a sequence of BaryonTerm objects") from exc
    for index, term in enumerate(normalized):
        if not isinstance(term, BaryonTerm):
            raise TypeError(f"term {index} must be a BaryonTerm")
    return normalized


def _required_keys(terms: Sequence[BaryonTerm]) -> tuple[Hashable, ...]:
    required: list[Hashable] = []
    seen: set[Hashable] = set()

    def add(key: Hashable) -> None:
        if key not in seen:
            seen.add(key)
            required.append(key)

    for term in terms:
        add(term.source.spin)
        add(term.source.elemental)
        add(term.sink.spin)
        add(term.sink.elemental)
        for key in term.propagators:
            add(key)
    return tuple(required)


def _specs_from_bindings(
    bindings: Mapping[Hashable, Any], terms: Sequence[BaryonTerm]
) -> dict[Hashable, TensorSpec]:
    required = _required_keys(terms)
    missing = [key for key in required if key not in bindings]
    extra = [key for key in bindings if key not in required]
    if missing:
        raise KeyError(f"missing bindings for input key(s): {missing!r}")
    if extra:
        raise KeyError(f"bindings contain unknown input key(s): {extra!r}")

    specs: dict[Hashable, TensorSpec] = {}
    for key in required:
        value = bindings[key]
        try:
            shape = tuple(operator.index(dimension) for dimension in value.shape)
            dtype = np.dtype(value.dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                f"binding {key!r} must expose an ndarray-like shape and dtype"
            ) from exc
        specs[key] = TensorSpec(shape, dtype)
    return specs


def execute_baryon_terms(
    bindings: Mapping[Hashable, Any],
    terms: Sequence[BaryonTerm],
    *,
    backend: Any = None,
    plan_cache: ContractionPlanCache | None = None,
    optimize: Any = "greedy",
    execution_cache=None,
) -> "Calculation":
    """Execute explicit baryon terms; callers assemble returned term values.

    This workflow convenience snapshots no inputs. The supplied bindings
    are borrowed for the synchronous X call, while the optional cache retains
    only a structure-only compiled plan.
    """

    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping of input keys to arrays")
    normalized_terms = _normalise_terms(terms)
    specs = _specs_from_bindings(bindings, normalized_terms)
    specs, bindings = normalize_array_precision(specs, bindings)
    plan, execute_bindings = compile_baryon_terms_cached(
        plan_cache,
        specs,
        normalized_terms,
        bindings,
        optimize=optimize,
    )
    return _run_ir(plan, execute_bindings, backend=backend, execution_cache=execution_cache)


def _run_ir(plan, bindings, *, backend, execution_cache, assemble=None, pair=False, empty_stats=None):
    """Replay the already selected IR; retain bindings on W, never in its cache."""
    def replay(*, backend=None):
        selected_backend = backend if backend is not None else original_backend
        return _run_ir(plan, bindings, backend=selected_backend, execution_cache=execution_cache,
            assemble=assemble, pair=pair, empty_stats=empty_stats)
    original_backend = backend
    if plan is None:
        values, stats = (), empty_stats
    else:
        execution = execute_plan(plan, bindings, backend=backend, execution_cache=execution_cache)
        values, stats = execution.values, execution.stats
    value = assemble(values) if assemble is not None else values
    return session_result(value, stats, replay, array=assemble is not None, pair=pair, plan=plan)


class ContractionSession:
    """Shared synchronous execution with a bounded, structure-only cache.

    ``optimize`` is ``'greedy'``; the former joint policy is retired.
    Every method uses this session's explicit NumPy or CuPy backend.
    Omitting the backend selects NumPy.

    Use separate sessions for concurrent evaluations. A session never changes
    physical coefficients, point-sector selection, gamma rules or time maps.
    """

    def __init__(self, *, optimize="greedy", backend=None, cache_entries=8):
        if optimize == "joint":
            raise ValueError("session optimize='joint' has been retired; use 'greedy'")
        if optimize != "greedy":
            raise ValueError("session optimize must be 'greedy'")
        xp = np if backend is None else backend
        if isinstance(xp, str):
            if xp not in ("numpy", "cupy"):
                raise ValueError("session backend must be NumPy or CuPy")
            xp = importlib.import_module(xp)
        if xp is not np and (
            getattr(xp, "__name__", None) != "cupy" or xp is not importlib.import_module("cupy")
        ):
            raise ValueError("session backend must be NumPy or CuPy")
        self.backend = xp
        self.optimize = optimize
        self.plan_cache = ContractionPlanCache(max_entries=cache_entries)
        self._bundle = None
        self._prepared_bundle = None
        self._execution_bindings = ExecutionBindingCache()

    @classmethod
    def from_plan(
        cls,
        plan,
        *,
        backend=None,
        load_plan: Callable | None = None,
        validate_bundle: Callable | None = None,
    ):
        """Bind an immutable bundle once; subsequent cfg calls never plan.

        A persisted session accepts only ``execute_cfg``. The ordinary dynamic
        methods reject use rather than silently rebuilding an incompatible DAG.
        Archive reading and validation are supplied by the storage owner so
        this execution policy does not depend on persistence implementation.
        """
        if isinstance(plan, (str, Path)):
            if not callable(load_plan):
                raise TypeError("archive paths require the storage load_plan callback")
            plan = load_plan(plan)
        bundle = plan
        if not isinstance(bundle, PlanBundle):
            raise TypeError("plan must be a PlanBundle or archive path")
        if not callable(validate_bundle):
            raise TypeError("from_plan requires the storage validate_bundle callback")
        validate_bundle(bundle)
        validate_frontend_workload(bundle.workload)
        return cls._from_validated_plan(bundle, backend=backend, validated_bundle=bundle)

    @classmethod
    def _from_validated_plan(cls, bundle, *, backend=None, validated_bundle):
        """Prepare the exact bundle whose DAG and frontend were just accepted.

        This private handoff is scoped to one open/bind operation. Public
        Python inputs continue through ``from_plan`` and its validators.
        """
        if not isinstance(bundle, PlanBundle) or bundle is not validated_bundle:
            raise ValueError("validated bundle does not match the session bundle")
        session = cls(backend=backend)
        fingerprint = bundle.performance_fingerprint
        if fingerprint is not None:
            from lqcdweave.runtime.environment import runtime_fingerprint
            if dict(fingerprint) != runtime_fingerprint(session.backend):
                raise ValueError(
                    "saved performance profile differs from this runtime; "
                    "regenerate or recalibrate the bundle offline"
                )
        session._bundle = bundle
        from lqcdweave.contraction.execution.api import prepare_bundle_execution
        session._prepared_bundle = prepare_bundle_execution(bundle, backend=session.backend)
        return session

    @property
    def bundle(self):
        """The read-only saved plan, or None for a dynamic session."""
        return self._bundle

    def execute_cfg(self, cfg, loader):
        """Bind this cfg, execute saved batches, and assemble saved outputs.

        ``loader(cfg, key, source_metadata)`` returns one declared ndarray.
        Its internal staging allocation must obey the workload's budget;
        opaque third-party loaders cannot be constrained by this API.
        """
        from lqcdweave.contraction.execution.api import execute_bundle

        if self._bundle is None:
            raise ValueError("execute_cfg requires ContractionSession.from_plan")
        started = perf_counter()
        execution = execute_bundle(
            self._bundle, cfg, loader, backend=self.backend, prepared=self._prepared_bundle,
        )
        assembly_started = perf_counter()
        assembled = assemble_workload_outputs(
            self._bundle.workload, execution.values, self.backend,
        )
        # Meson stacking owns independent storage; generic executor outputs
        # already own theirs. Slices in the tuple refer to disjoint scalars.
        values = tuple(assembled)
        stats = execution.stats.as_dict()
        timings = dict(stats.get("timings", {}))
        timings["assembly_seconds"] = perf_counter() - assembly_started
        timings["session_seconds"] = perf_counter() - started
        stats["timings"] = timings
        def replay(*, backend=None):
            if backend is not None and backend is not self.backend and backend != self.backend.__name__:
                raise ValueError('persisted session replay uses its selected backend')
            return self.execute_cfg(cfg, loader)
        return session_result(values, stats, replay, plan=self._bundle)

    def _require_dynamic(self):
        if self._bundle is not None:
            raise ValueError("persisted sessions use execute_cfg; dynamic replanning is disabled")

    def clear(self):
        """Drop structural plans and reset hit/miss counters."""
        self.plan_cache.clear()
        self._execution_bindings = ExecutionBindingCache()

    def tensors(self, bindings, terms):
        """Evaluate explicit IR terms with fresh arrays; return Calculation."""
        self._require_dynamic()
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping of keys to arrays")
        specs = {}
        for key, value in bindings.items():
            if not isinstance(value, self.backend.ndarray):
                raise TypeError("tensor bindings must use the session backend")
            specs[key] = TensorSpec(value.shape, value.dtype)
        specs, bindings = normalize_array_precision(specs, bindings)
        plan, rebound = compile_plan_cached(
            self.plan_cache, specs, tuple(terms), bindings, optimize=self.optimize
        )
        return _run_ir(plan, rebound, backend=self.backend, execution_cache=self._execution_bindings)

    def baryon_terms(self, bindings, terms):
        """Evaluate factorized three-line baryon terms, with explicit slots."""
        self._require_dynamic()
        return execute_baryon_terms(
            bindings, terms, backend=self.backend,
            plan_cache=self.plan_cache, optimize=self.optimize,
            execution_cache=self._execution_bindings,
        )

    @staticmethod
    def _empty_execution_stats():
        return ExecutionStats(
            executed_operations=0, total_operations=0, unique_operations=0,
            shared_operations=0, peak_live_bytes=0, peak_intermediate_bytes=0,
            input_bytes=0, output_bytes=0, scratch_bytes=None,
        )

    def baryon_diagrams(self, diagrams, times, vertices, propagators, *, coefficients=None, return_stats=False):
        """Execute two-vertex VSV baryons through public module services."""
        self._require_dynamic()
        diagrams, times, vertices = tuple(diagrams), tuple(times), tuple(vertices)
        topologies, times, coefficients = validate_baryon_diagram_batch(
            diagrams, times, len(vertices), coefficients=coefficients,
        )
        specs, bindings, input_keys = read_baryon_diagram_inputs(
            topologies, times, vertices, propagators, backend=self.backend,
        )
        terms = expand_baryon_diagrams(topologies, input_keys, coefficients)
        plan, rebound = None, {}
        if terms:
            specs, bindings = normalize_array_precision(specs, bindings)
            plan, rebound = compile_baryon_terms_cached(
                self.plan_cache, specs, terms, bindings, optimize=self.optimize,
            )
        return _run_ir(plan, rebound, backend=self.backend, execution_cache=self._execution_bindings,
            assemble=lambda values: assemble_baryon_diagram_outputs(values, self.backend),
            pair=return_stats, empty_stats=self._empty_execution_stats() if plan is None else None)

    def diagrams(self, diagrams, times, vertices, propagators, *, return_stats=False):
        """Execute scalar-time graphs with the original getter conventions."""
        self._require_dynamic()
        diagrams = tuple(diagrams)
        specs, bindings, occurrence_keys = read_simple_diagram_inputs(
            diagrams, times, vertices, propagators, backend=self.backend,
        )
        terms, recipes = expand_simple_diagrams(diagrams, occurrence_keys, specs)
        plan, rebound = None, {}
        if terms:
            specs, bindings = normalize_array_precision(specs, bindings)
            plan, rebound = compile_plan_cached(
                self.plan_cache, specs, terms, bindings, optimize=self.optimize,
            )
        return _run_ir(plan, rebound, backend=self.backend, execution_cache=self._execution_bindings,
            assemble=lambda values: assemble_simple_diagram_outputs(values, recipes, self.backend),
            pair=return_stats, empty_stats=self._empty_execution_stats() if plan is None else None)


__all__ = ["ContractionSession", "CfgExecutionResult"]
