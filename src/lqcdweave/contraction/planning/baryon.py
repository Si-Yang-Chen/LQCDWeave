"""Pure structural compilation of explicit three-line baryon terms."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    BaryonTerm,
    ContractionPlan,
    ContractionTerm,
    TensorSpec,
)
from .cache import ContractionPlanCache, compile_plan_cached
from .compiler import compile_plan


_SOURCE_SPIN_LABELS = (0, 1, 2)
_SOURCE_MODE_LABELS = (3, 4, 5)
_SINK_SPIN_LABELS = (6, 7, 8)
_SINK_MODE_LABELS = (9, 10, 11)
_VERTEX_RANK = 3
_PROPAGATOR_RANK = 4


def _check_hashable_key(key: Hashable, *, name: str) -> None:
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError(f"{name} key {key!r} is not hashable") from exc


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


def _normalise_specs(
    specs: Mapping[Hashable, TensorSpec], terms: Sequence[BaryonTerm]
) -> dict[Hashable, TensorSpec]:
    if not isinstance(specs, Mapping):
        raise TypeError("specs must be a mapping of binding keys to TensorSpec objects")
    normalized: dict[Hashable, TensorSpec] = {}
    for key, spec in specs.items():
        _check_hashable_key(key, name="input")
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"spec for input key {key!r} must be a TensorSpec")
        normalized[key] = spec

    required = _required_keys(terms)
    missing = [key for key in required if key not in normalized]
    extra = [key for key in normalized if key not in required]
    if missing:
        raise KeyError(f"missing TensorSpec(s) for input key(s): {missing!r}")
    if extra:
        raise KeyError(f"TensorSpec mapping contains unknown input key(s): {extra!r}")
    return normalized


def _validate_baryon_shapes(
    specs: Mapping[Hashable, TensorSpec], terms: Sequence[BaryonTerm]
) -> None:
    """Validate ranks and all four dimensions at every propagator endpoint."""

    def require_rank(key: Hashable, rank: int, role: str) -> TensorSpec:
        spec = specs[key]
        if len(spec.shape) != rank:
            raise ValueError(
                f"{role} key {key!r} must have rank {rank}, got rank {len(spec.shape)}"
            )
        return spec

    for term_index, term in enumerate(terms):
        source_spin = require_rank(term.source.spin, _VERTEX_RANK, f"term {term_index} source spin")
        source_elemental = require_rank(
            term.source.elemental, _VERTEX_RANK, f"term {term_index} source elemental"
        )
        sink_spin = require_rank(term.sink.spin, _VERTEX_RANK, f"term {term_index} sink spin")
        sink_elemental = require_rank(
            term.sink.elemental, _VERTEX_RANK, f"term {term_index} sink elemental"
        )
        propagator_specs = tuple(
            require_rank(key, _PROPAGATOR_RANK, f"term {term_index} propagator {q}")
            for q, key in enumerate(term.propagators)
        )

        for q, (source_slot, propagator) in enumerate(
            zip(term.source_slots, propagator_specs, strict=True)
        ):
            checks = (
                (
                    sink_spin.shape[q],
                    propagator.shape[0],
                    f"term {term_index} sink spin slot {q} to propagator {q} sink-spin axis",
                ),
                (
                    source_spin.shape[source_slot],
                    propagator.shape[1],
                    f"term {term_index} source spin slot {source_slot} to propagator {q} source-spin axis",
                ),
                (
                    sink_elemental.shape[q],
                    propagator.shape[2],
                    f"term {term_index} sink elemental slot {q} to propagator {q} sink-mode axis",
                ),
                (
                    source_elemental.shape[source_slot],
                    propagator.shape[3],
                    f"term {term_index} source elemental slot {source_slot} to propagator {q} source-mode axis",
                ),
            )
            for left_dimension, right_dimension, description in checks:
                if left_dimension != right_dimension:
                    raise ValueError(
                        f"{description} has incompatible dimensions "
                        f"{left_dimension} and {right_dimension}"
                    )


def _equation_for(term: BaryonTerm) -> AxisPattern:
    propagator_labels = tuple(
        (_SINK_SPIN_LABELS[sink_slot], _SOURCE_SPIN_LABELS[source_slot],
         _SINK_MODE_LABELS[sink_slot], _SOURCE_MODE_LABELS[source_slot])
        for sink_slot, source_slot in enumerate(term.source_slots)
    )
    return AxisPattern((_SOURCE_SPIN_LABELS, _SOURCE_MODE_LABELS,
                        _SINK_SPIN_LABELS, _SINK_MODE_LABELS, *propagator_labels), ())


def _contraction_terms(terms: Sequence[BaryonTerm]) -> tuple[ContractionTerm, ...]:
    return tuple(
        ContractionTerm(
            (
                term.source.spin,
                term.source.elemental,
                term.sink.spin,
                term.sink.elemental,
                *term.propagators,
            ),
            _equation_for(term),
            coefficient=term.coefficient,
        )
        for term in terms
    )


def _prepare(
    specs: Mapping[Hashable, TensorSpec], terms: Sequence[BaryonTerm]
) -> tuple[dict[Hashable, TensorSpec], tuple[BaryonTerm, ...], tuple[ContractionTerm, ...]]:
    normalized_terms = _normalise_terms(terms)
    normalized_specs = _normalise_specs(specs, normalized_terms)
    _validate_baryon_shapes(normalized_specs, normalized_terms)
    return normalized_specs, normalized_terms, _contraction_terms(normalized_terms)


def compile_baryon_terms(
    specs: Mapping[Hashable, TensorSpec],
    terms: Sequence[BaryonTerm],
    *,
    optimize: Any = "greedy",
) -> ContractionPlan:
    """Compile baryon terms into the shared contraction DAG."""

    normalized_specs, _, contraction_terms = _prepare(specs, terms)
    return compile_plan(normalized_specs, contraction_terms, optimize=optimize)


def compile_baryon_terms_cached(
    cache: ContractionPlanCache | None,
    specs: Mapping[Hashable, TensorSpec],
    terms: Sequence[BaryonTerm],
    bindings: Mapping[Hashable, Any],
    *,
    optimize: Any = "greedy",
) -> tuple[ContractionPlan, Mapping[Hashable, Any]]:
    """Compile or reuse a baryon plan without inspecting numerical bindings."""

    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping of input keys to arrays")
    normalized_terms = _normalise_terms(terms)
    normalized_specs, _, contraction_terms = _prepare(specs, normalized_terms)
    required = _required_keys(normalized_terms)
    missing = [key for key in required if key not in bindings]
    extra = [key for key in bindings if key not in required]
    if missing:
        raise KeyError(f"missing bindings for input key(s): {missing!r}")
    if extra:
        raise KeyError(f"bindings contain unknown input key(s): {extra!r}")
    return compile_plan_cached(
        cache,
        normalized_specs,
        contraction_terms,
        bindings,
        optimize=optimize,
    )


__all__ = ["compile_baryon_terms", "compile_baryon_terms_cached"]

