"""Portable structural workload descriptions for contraction planning.

The classes in this module are deliberately independent of numerical
bindings.  A :class:`WorkloadSpec` contains shapes, dtypes, explicit
contraction terms, and declarative source metadata only.  It can therefore be
constructed once and reused for every configuration in a dispatch queue.

The implementation keeps the in-memory objects immutable as well as JSON
serialisable.  ``MappingProxyType`` and tuples are used for nested metadata so
that a caller cannot mutate a plan after its structural signature has been
computed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import numbers
import operator
from types import MappingProxyType
from typing import Any

import numpy as np

from .ir import ContractionPlan, ContractionTerm, TensorSpec


DEFAULT_CONVENTION = "explicit-v1"
DEFAULT_RESERVE_FRACTION = 0.10
DEFAULT_MIN_RESERVE_BYTES = 1 << 30


def _path_text(path: tuple[str, ...]) -> str:
    return "" if not path else " at " + "".join(path)


def _freeze_json(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Return an immutable, strict-JSON representation of ``value``.

    JSON metadata is intentionally narrower than the numerical coefficient
    field.  NumPy arrays, scalar objects, bytes, sets, and non-finite floats
    are rejected instead of being stringified or silently converted.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"JSON metadata requires finite floats{_path_text(path)}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON metadata keys must be strings, got {key!r}"
                    f"{_path_text(path)}"
                )
            frozen[key] = _freeze_json(item, path=path + (f"[{key!r}]",))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=path + (f"[{index}]",)) for index, item in enumerate(value))
    raise TypeError(
        f"value of type {type(value).__module__}.{type(value).__qualname__} is not JSON metadata"
        f"{_path_text(path)}"
    )


def _thaw_json(value: Any) -> Any:
    """Convert immutable metadata back to ordinary JSON-compatible objects."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _encode_scalar(value: Any) -> dict[str, Any]:
    """Encode a supported scalar without losing dtype or IEEE bit patterns."""

    if isinstance(value, np.generic):
        dtype = np.dtype(value.dtype)
        if dtype.kind not in "biufc":
            raise TypeError(f"unsupported NumPy coefficient dtype {dtype}")
        raw = np.asarray(value, dtype=dtype).reshape(()).tobytes().hex()
        return {"type": "numpy", "dtype": dtype.str, "hex": raw}
    if isinstance(value, bool):
        return {"type": "python-bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        # A decimal string avoids JSON parser limits and preserves arbitrary
        # precision Python integers.
        return {"type": "python-int", "value": str(value)}
    if isinstance(value, float):
        raw = np.asarray(value, dtype=np.dtype("<f8")).reshape(()).tobytes().hex()
        return {"type": "python-float", "dtype": "<f8", "hex": raw}
    if isinstance(value, complex):
        raw = np.asarray(value, dtype=np.dtype("<c16")).reshape(()).tobytes().hex()
        return {"type": "python-complex", "dtype": "<c16", "hex": raw}
    raise TypeError(
        "contraction coefficients must be Python or NumPy real/complex scalar "
        f"values, got {type(value).__module__}.{type(value).__qualname__}"
    )


def _decode_scalar(payload: Any) -> Any:
    """Decode a validated scalar payload from a plan archive."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("type"), str):
        raise ValueError("invalid typed coefficient payload")
    kind = payload["type"]
    if kind == "python-bool":
        if set(payload) != {"type", "value"} or not isinstance(payload["value"], bool):
            raise ValueError("invalid Python bool coefficient payload")
        return payload["value"]
    if kind == "python-int":
        if set(payload) != {"type", "value"} or not isinstance(payload["value"], str):
            raise ValueError("invalid Python int coefficient payload")
        try:
            return int(payload["value"], 10)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Python int coefficient payload") from exc
    if kind not in {"numpy", "python-float", "python-complex"}:
        raise ValueError(f"unknown typed coefficient kind {kind!r}")
    if not isinstance(payload.get("dtype"), str) or not isinstance(payload.get("hex"), str):
        raise ValueError("invalid binary coefficient payload")
    if kind == "python-float" and payload["dtype"] != "<f8":
        raise ValueError("Python float coefficients must use <f8")
    if kind == "python-complex" and payload["dtype"] != "<c16":
        raise ValueError("Python complex coefficients must use <c16")
    try:
        dtype = np.dtype(payload["dtype"])
        raw = bytes.fromhex(payload["hex"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid binary coefficient payload") from exc
    if dtype.kind not in "biufc" or len(raw) != dtype.itemsize:
        raise ValueError("invalid coefficient dtype or byte length")
    if set(payload) != {"type", "dtype", "hex"}:
        raise ValueError("unexpected fields in coefficient payload")
    try:
        result = np.frombuffer(raw, dtype=dtype, count=1)[0]
    except (TypeError, ValueError) as exc:
        raise ValueError("unable to decode coefficient payload") from exc
    if kind == "python-float":
        return float(result)
    if kind == "python-complex":
        return complex(result)
    return result


def _normalise_groups(groups: Any, term_count: int) -> tuple[tuple[int, ...], ...]:
    if groups is None:
        return tuple((index,) for index in range(term_count))
    if isinstance(groups, (str, bytes)):
        raise TypeError("output_groups must be a sequence of term-index sequences")
    try:
        normalised = tuple(
            tuple(
                (_raise_bool_index(index) if isinstance(index, bool) else operator.index(index))
                for index in group
            )
            for group in groups
        )
    except TypeError as exc:
        raise TypeError("output_groups must be a sequence of term-index sequences") from exc
    if any(not group for group in normalised):
        raise ValueError("output_groups cannot contain empty groups")
    seen: set[int] = set()
    for group in normalised:
        for index in group:
            if index < 0 or index >= term_count:
                raise ValueError(f"output group term index {index} is outside the term list")
            if index in seen:
                raise ValueError(f"term index {index} occurs in more than one output group")
            seen.add(index)
    expected = set(range(term_count))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise ValueError("output_groups must cover every term exactly once (" + ", ".join(detail) + ")")
    return normalised


def _raise_bool_index(value: bool) -> int:
    raise TypeError(f"boolean {value!r} is not a term index")


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Host/device logical-memory limits used during plan selection.

    ``reserve_fraction`` and ``min_reserve_bytes`` describe the default device
    reserve policy: available device memory is reduced by
    ``max(total * reserve_fraction, min_reserve_bytes)`` when a runtime knows
    the device's total memory.  Explicit ``device_bytes`` and ``host_bytes``
    remain hard upper bounds for execution budgets.
    """

    host_bytes: int | None = None
    device_bytes: int | None = None
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION
    min_reserve_bytes: int = DEFAULT_MIN_RESERVE_BYTES
    workspace_bytes: int = 0
    host_staging_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("host_bytes", "device_bytes"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    raise TypeError(f"{name} must be a positive integer or None")
                if int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer or None")
                object.__setattr__(self, name, int(value))
        fraction = self.reserve_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
            raise TypeError("reserve_fraction must be a finite real number")
        fraction = float(fraction)
        if not math.isfinite(fraction) or fraction < 0.0 or fraction >= 1.0:
            raise ValueError("reserve_fraction must satisfy 0 <= reserve_fraction < 1")
        object.__setattr__(self, "reserve_fraction", fraction)
        minimum = self.min_reserve_bytes
        if isinstance(minimum, bool) or not isinstance(minimum, numbers.Integral):
            raise TypeError("min_reserve_bytes must be a non-negative integer")
        if int(minimum) < 0:
            raise ValueError("min_reserve_bytes must be a non-negative integer")
        object.__setattr__(self, "min_reserve_bytes", int(minimum))
        for name in ("workspace_bytes", "host_staging_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be a non-negative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))

    def device_reserve(self, total_bytes: int) -> int:
        """Return the default reserve for a device with ``total_bytes``."""

        if isinstance(total_bytes, bool) or not isinstance(total_bytes, numbers.Integral):
            raise TypeError("total_bytes must be a positive integer")
        if int(total_bytes) <= 0:
            raise ValueError("total_bytes must be a positive integer")
        return max(int(math.ceil(int(total_bytes) * self.reserve_fraction)), self.min_reserve_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "host_bytes": self.host_bytes,
            "device_bytes": self.device_bytes,
            "reserve_fraction": self.reserve_fraction,
            "min_reserve_bytes": self.min_reserve_bytes,
            "workspace_bytes": self.workspace_bytes,
            "host_staging_bytes": self.host_staging_bytes,
        }


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Immutable, configuration-independent contraction workload description."""

    specs: Mapping[str, TensorSpec]
    terms: tuple[ContractionTerm, ...]
    input_sources: Mapping[str, Any] = field(default_factory=dict)
    convention: str = DEFAULT_CONVENTION
    output_groups: tuple[tuple[int, ...], ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.specs, Mapping):
            raise TypeError("specs must be a mapping of portable string keys to TensorSpec")
        specs: dict[str, TensorSpec] = {}
        for key, spec in self.specs.items():
            if not isinstance(key, str):
                raise TypeError(f"workload input keys must be strings, got {key!r}")
            if not isinstance(spec, TensorSpec):
                raise TypeError(f"spec for input key {key!r} must be a TensorSpec")
            normalized_spec = TensorSpec(tuple(spec.shape), np.dtype(spec.dtype))
            if normalized_spec.dtype.kind not in "biufc":
                raise TypeError(
                    f"spec for input key {key!r} must use a numeric dtype, "
                    f"got {normalized_spec.dtype}"
                )
            specs[key] = normalized_spec
        object.__setattr__(self, "specs", MappingProxyType(specs))

        try:
            terms = tuple(self.terms)
        except TypeError as exc:
            raise TypeError("terms must be an iterable of ContractionTerm objects") from exc
        normalized_terms: list[ContractionTerm] = []
        for term_index, term in enumerate(terms):
            if not isinstance(term, ContractionTerm):
                raise TypeError(f"term {term_index} must be a ContractionTerm")
            for key in term.inputs:
                if not isinstance(key, str):
                    raise TypeError(
                        f"term {term_index} input keys must be strings, got {key!r}"
                    )
                if key not in specs:
                    raise KeyError(f"term {term_index} refers to unknown input key {key!r}")
            # Validate coefficients now, before an object can be archived.
            _encode_scalar(term.coefficient)
            normalized_terms.append(
                ContractionTerm(tuple(term.inputs), term.subscripts, coefficient=term.coefficient)
            )
        object.__setattr__(self, "terms", tuple(normalized_terms))

        if not isinstance(self.input_sources, Mapping):
            raise TypeError("input_sources must be a mapping")
        sources: dict[str, Any] = {}
        for key, source in self.input_sources.items():
            if not isinstance(key, str):
                raise TypeError(f"input source keys must be strings, got {key!r}")
            if key not in specs:
                raise KeyError(f"input source refers to unknown input key {key!r}")
            sources[key] = _freeze_json(source, path=(f"input_sources[{key!r}]",))
        object.__setattr__(self, "input_sources", MappingProxyType(sources))

        if not isinstance(self.convention, str) or not self.convention:
            raise TypeError("convention must be a non-empty string")
        object.__setattr__(self, "output_groups", _normalise_groups(self.output_groups, len(normalized_terms)))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a JSON mapping")
        frozen_metadata = _freeze_json(self.metadata, path=("metadata",))
        recipes = frozen_metadata.get("output_recipes")
        if recipes is not None:
            if not isinstance(recipes, tuple) or any(not isinstance(recipe, tuple) for recipe in recipes):
                raise TypeError("metadata.output_recipes must be a list of term-index lists")
            for recipe in recipes:
                for index in recipe:
                    if isinstance(index, bool) or not isinstance(index, int):
                        raise TypeError("metadata.output_recipes indices must be integers")
                    if index < 0 or index >= len(normalized_terms):
                        raise ValueError(
                            f"metadata.output_recipes index {index} is outside the term list"
                        )
        object.__setattr__(self, "metadata", frozen_metadata)

    @property
    def output_recipes(self) -> tuple[tuple[int, ...], ...]:
        """Return optional adapter recipes from metadata as immutable tuples."""

        recipes = self.metadata.get("output_recipes", ())
        if not recipes:
            return ()
        if not isinstance(recipes, tuple):
            raise ValueError("metadata.output_recipes must be a list of term-index lists")
        result: list[tuple[int, ...]] = []
        for recipe in recipes:
            if not isinstance(recipe, tuple):
                raise ValueError("metadata.output_recipes must be a list of term-index lists")
            result.append(tuple(operator.index(index) for index in recipe))
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        """Return a detached summary suitable for diagnostics."""

        from .equations import parse_term

        return {
            "specs": {
                key: {"shape": list(spec.shape), "dtype": spec.dtype.str}
                for key, spec in self.specs.items()
            },
            "terms": [
                {
                    "inputs": list(term.inputs),
                    "subscripts": parse_term(term, self.specs, index).equation.as_dict(),
                    "coefficient": _encode_scalar(term.coefficient),
                }
                for index, term in enumerate(self.terms)
            ],
            "input_sources": _thaw_json(self.input_sources),
            "convention": self.convention,
            "output_groups": [list(group) for group in self.output_groups],
            "metadata": _thaw_json(self.metadata),
        }


def validate_frontend_workload(workload):
    """Check saved assembly metadata before any numerical loader is called."""
    adapter = workload.metadata.get("adapter")
    if adapter is None:
        return
    if adapter != "meson-simple-v1" or workload.convention != adapter:
        raise ValueError("unsupported saved frontend adapter or convention")
    recipes = workload.metadata.get("output_recipes")
    if not isinstance(recipes, (tuple, list)):
        raise ValueError("saved Meson workload requires ordered output recipes")
    seen = []
    for recipe in recipes:
        if not isinstance(recipe, (tuple, list)):
            raise ValueError("output recipe must be a sequence of term indices")
        for index in recipe:
            if type(index) is not int or not 0 <= index < len(workload.terms):
                raise ValueError("invalid output recipe term reference")
            seen.append(index)
    if sorted(seen) != list(range(len(workload.terms))):
        raise ValueError("Meson recipes must assemble every term exactly once")
    from .equations import parse_term
    if any(parse_term(term, workload.specs, index).output_labels for index, term in enumerate(workload.terms)):
        raise ValueError("saved Meson term outputs must be scalar")
    extra = workload.metadata.get("assembly_extra_bytes")
    if type(extra) is not int or extra < 48 * len(recipes):
        raise ValueError("saved Meson assembly memory reserve is insufficient")


@dataclass(frozen=True, slots=True)
class PlanBatch:
    """One immutable compiled DAG and the source terms it evaluates."""

    plan: ContractionPlan
    term_indices: tuple[int, ...]
    memory_estimate: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ContractionPlan):
            raise TypeError("PlanBatch.plan must be a ContractionPlan")
        try:
            indices = tuple(
                _raise_bool_index(index) if isinstance(index, bool) else operator.index(index)
                for index in self.term_indices
            )
        except TypeError as exc:
            raise TypeError("PlanBatch.term_indices must be integer indices") from exc
        if len(indices) != len(self.plan.terms):
            raise ValueError(
                "PlanBatch.term_indices must contain one source index per compiled term"
            )
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("PlanBatch.term_indices must be distinct non-negative integers")
        object.__setattr__(self, "term_indices", indices)
        if not isinstance(self.memory_estimate, Mapping):
            raise TypeError("PlanBatch.memory_estimate must be a JSON mapping")
        object.__setattr__(
            self,
            "memory_estimate",
            _freeze_json(self.memory_estimate, path=("memory_estimate",)),
        )


@dataclass(frozen=True, slots=True)
class PlanBundle:
    """A reusable workload plus one or more already-compiled DAG batches."""

    workload: WorkloadSpec
    batches: tuple[PlanBatch, ...]
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.workload, WorkloadSpec):
            raise TypeError("PlanBundle.workload must be a WorkloadSpec")
        try:
            batches = tuple(self.batches)
        except TypeError as exc:
            raise TypeError("PlanBundle.batches must be an iterable of PlanBatch objects") from exc
        if any(not isinstance(batch, PlanBatch) for batch in batches):
            raise TypeError("PlanBundle.batches must contain PlanBatch objects")
        if not isinstance(self.budget, ExecutionBudget):
            raise TypeError("PlanBundle.budget must be an ExecutionBudget")
        seen: set[int] = set()
        term_count = len(self.workload.terms)
        for batch_index, batch in enumerate(batches):
            for term_index in batch.term_indices:
                if term_index >= term_count:
                    raise ValueError(
                        f"batch {batch_index} refers to term index {term_index} outside workload"
                    )
                if term_index in seen:
                    raise ValueError(f"term index {term_index} occurs in multiple plan batches")
                seen.add(term_index)
        expected = set(range(term_count))
        if seen != expected:
            raise ValueError(
                "PlanBundle batches must cover every workload term exactly once "
                f"(missing={sorted(expected - seen)})"
            )
        object.__setattr__(self, "batches", batches)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PlanBundle.metadata must be a JSON mapping")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(self.metadata, path=("bundle_metadata",)),
        )

    @property
    def signature(self) -> str:
        """Return the SHA-256 structural signature computed by the archive code."""

        # Keep the data classes independent from the archive's I/O machinery;
        # importing lazily also avoids a module cycle during package startup.
        from .signatures import structural_signature

        return structural_signature(self)

    @property
    def performance_fingerprint(self) -> Mapping[str, Any] | None:
        """Return a detached view of optional runtime fingerprint metadata."""

        value = self.metadata.get("performance_fingerprint")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("PlanBundle.metadata.performance_fingerprint must be a mapping")
        return value


__all__ = [
    "DEFAULT_CONVENTION",
    "DEFAULT_MIN_RESERVE_BYTES",
    "DEFAULT_RESERVE_FRACTION",
    "ExecutionBudget",
    "PlanBatch",
    "PlanBundle",
    "WorkloadSpec",
]

# Public pure serialization helpers shared with the archive boundary.
encode_scalar = _encode_scalar
decode_scalar = _decode_scalar
thaw_json = _thaw_json
