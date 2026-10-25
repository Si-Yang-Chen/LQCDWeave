"""Bounded structural cache for compiled contraction plans.

The cache deliberately stores only :class:`ContractionPlan` objects and their
shape/index signatures.  Numerical bindings, loader objects, and source/time
identities remain owned by the current adapter call.  Each lookup canonicalizes
the caller's input keys by first appearance, so a plan can be reused after a
configuration reload or a time change while still distinguishing changed
shapes, dtypes, equations, coefficients, and sharing patterns.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
import numbers
from typing import Any, Callable

import numpy as np

from .compiler import compile_plan
from lqcdweave.contracts.correlator.api import ContractionPlan, ContractionTerm, TensorSpec


def _freeze(value: Any) -> Hashable:
    """Make a small structural option or coefficient hashable and typed."""

    if isinstance(value, np.generic):
        return ("numpy-scalar", value.dtype.str, value.tobytes())
    if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
        return (type(value).__module__, type(value).__qualname__, value)
    if isinstance(value, numbers.Number):
        return (type(value).__module__, type(value).__qualname__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    ((_freeze(key), _freeze(item)) for key, item in value.items()),
                    key=repr,
                )
            ),
        )
    raise TypeError(
        "unsupported structural cache option "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonicalize(
    specs: Mapping[Hashable, TensorSpec],
    terms: Sequence[ContractionTerm],
    bindings: Mapping[Hashable, Any],
    *,
    optimize: Any,
):
    """Return canonical structural inputs, fresh bindings, and a cache key."""

    actual_to_canonical: dict[Hashable, int] = {}
    actual_order: list[Hashable] = []

    def canonical(key: Hashable) -> int:
        try:
            return actual_to_canonical[key]
        except KeyError:
            value = len(actual_order)
            actual_to_canonical[key] = value
            actual_order.append(key)
            return value

    # First appearance in ordered term inputs is the only identity convention
    # used by this cache.  Unreferenced specs are retained afterwards so their
    # shape and dtype cannot silently change a cache entry.
    for term in terms:
        for key in term.inputs:
            canonical(key)
    for key in specs:
        canonical(key)

    canonical_specs = {
        actual_to_canonical[key]: TensorSpec(specs[key].shape, specs[key].dtype)
        for key in actual_order
        if key in specs
    }
    canonical_terms = tuple(
        ContractionTerm(
            tuple(actual_to_canonical[key] for key in term.inputs),
            term.subscripts,
            coefficient=term.coefficient,
            sum_barriers=term.sum_barriers,
        )
        for term in terms
    )
    structural_specs = tuple(
        (
            actual_to_canonical[key],
            tuple(specs[key].shape),
            np.dtype(specs[key].dtype),
        )
        for key in actual_order
        if key in specs
    )
    structural_terms = tuple(
        (
            tuple(actual_to_canonical[key] for key in term.inputs),
            term.subscripts,
            _freeze(term.coefficient),
            term.sum_barriers,
        )
        for term in terms
    )
    cache_key = (
        "contraction-plan-v1",
        _freeze(optimize),
        structural_specs,
        structural_terms,
    )
    canonical_bindings = {
        actual_to_canonical[key]: value
        for key, value in bindings.items()
        if key in actual_to_canonical
    }
    return canonical_specs, canonical_terms, canonical_bindings, cache_key


class ContractionPlanCache:
    """A bounded LRU cache containing structural contraction plans only."""

    def __init__(self, max_entries: int = 8):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be a positive integer")
        if max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, ContractionPlan] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict[str, int]:
        """Return counters suitable for a benchmark or validation report."""

        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._entries),
            "max_entries": self.max_entries,
        }

    def clear(self) -> None:
        """Drop all structural plans and reset hit/miss counters."""

        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def _get_or_compile(
        self,
        key: Hashable,
        factory: Callable[[], ContractionPlan],
    ) -> ContractionPlan:
        try:
            plan = self._entries.pop(key)
        except KeyError:
            self.misses += 1
            plan = factory()
        else:
            self.hits += 1
        self._entries[key] = plan
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return plan


def compile_plan_cached(
    plan_cache: ContractionPlanCache | None,
    specs: Mapping[Hashable, TensorSpec],
    terms: Sequence[ContractionTerm],
    bindings: Mapping[Hashable, Any],
    *,
    optimize: Any = "greedy",
):
    """Compile or reuse a structure-only plan and return current bindings."""

    if plan_cache is None:
        return compile_plan(specs, tuple(terms), optimize=optimize), bindings
    if not isinstance(plan_cache, ContractionPlanCache):
        raise TypeError("plan_cache must be a ContractionPlanCache or None")
    missing = [key for key in specs if key not in bindings]
    extra = [key for key in bindings if key not in specs]
    if missing:
        raise KeyError(f"missing bindings for input key(s): {missing!r}")
    if extra:
        raise KeyError(f"bindings contain unknown input key(s): {extra!r}")

    canonical_specs, canonical_terms, canonical_bindings, cache_key = _canonicalize(
        specs, tuple(terms), bindings, optimize=optimize
    )
    plan = plan_cache._get_or_compile(
        cache_key,
        lambda: compile_plan(canonical_specs, canonical_terms, optimize=optimize),
    )
    return plan, canonical_bindings


__all__ = ["ContractionPlanCache", "compile_plan_cached"]
