"""Immutable intermediate representation for shared tensor contractions.

The objects in this module deliberately contain no numerical arrays.  A
compiled plan describes input specifications, explicit-index operations and
term outputs; :func:`lattice.contraction.execute_plan` supplies the arrays for
one synchronous execution.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
import numbers
import operator
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from .semantics import TensorDescription
from .indices import AxisPattern, _axes, parse_axis_pattern


def _normalise_shape(shape: Any) -> tuple[int, ...]:
    """Return a validated tuple of non-negative Python integers."""

    if isinstance(shape, (str, bytes)):
        raise TypeError("tensor shape must be an iterable of dimensions")
    try:
        dimensions = tuple(operator.index(dimension) for dimension in shape)
    except TypeError as exc:
        raise TypeError("tensor shape must be an iterable of dimensions") from exc
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"tensor dimensions must be non-negative, got {dimensions!r}")
    return dimensions


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Shape and dtype contract for one named input tensor.

    ``dtype`` is normalized with :func:`numpy.dtype`, and ``shape`` is copied
    into an immutable tuple.  A scalar tensor therefore has ``shape == ()``.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    description: TensorDescription | None = field(default=None, kw_only=True, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _normalise_shape(self.shape))
        try:
            normalized_dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid tensor dtype {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", normalized_dtype)
        if self.description is not None:
            if not isinstance(self.description, TensorDescription):
                raise TypeError("TensorSpec.description must be a pure TensorDescription")
            if self.description.axes and len(self.description.axes) != len(self.shape):
                raise ValueError("named logical axes must match TensorSpec rank")

    @property
    def axes(self) -> tuple[str, ...]:
        return () if self.description is None else self.description.axes

    @property
    def data_identity(self) -> str | None:
        return None if self.description is None else self.description.data_identity

    @property
    def selection_identity(self) -> str | None:
        return None if self.description is None else self.description.selection_identity


@dataclass(frozen=True, slots=True)
class ContractionTerm:
    """One contraction term with explicit input/output axis relations.

    ``subscripts`` accepts an integer AxisPattern or a caller-provided string
    with an explicit ``->`` output. Parsing produces an integer term. A
    term's coefficient is intentionally kept outside operation nodes and is
    applied only while assembling the execution result.
    """

    inputs: tuple[Hashable, ...]
    subscripts: str | AxisPattern
    coefficient: Any = 1
    sum_barriers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    kernel: object = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        try:
            normalized_inputs = tuple(self.inputs)
        except TypeError as exc:
            raise TypeError("term inputs must be an iterable of hashable keys") from exc
        for key in normalized_inputs:
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(f"term input key {key!r} is not hashable") from exc
        object.__setattr__(self, "inputs", normalized_inputs)
        if not isinstance(self.subscripts, (str, AxisPattern)):
            raise TypeError("term subscripts must be a string or AxisPattern")
        if isinstance(self.subscripts, str) and self.subscripts.count("->") != 1:
            raise ValueError("term subscripts require one explicit '->' output")
        object.__setattr__(self, "subscripts", parse_axis_pattern(self.subscripts))
        barriers = tuple((tuple(trigger), tuple(dependent)) for trigger, dependent in self.sum_barriers)
        reduced = set(label for axes in self.subscripts.inputs for label in axes) - set(self.subscripts.output)
        if any(not trigger or not dependent or set(trigger) & set(dependent)
               or not set((*trigger, *dependent)) <= reduced
               for trigger, dependent in barriers):
            raise ValueError('sum barriers require disjoint nonempty reduced axis groups')
        object.__setattr__(self, 'sum_barriers', barriers)
        # NumPy scalar numbers are accepted; strings, datetimes, arrays and
        # symbolic objects are rejected here so malformed coefficients cannot
        # reach execution.
        coefficient = self.coefficient
        if not isinstance(coefficient, (numbers.Number, np.bool_)):
            raise TypeError("term coefficient must be a scalar number")


@dataclass(frozen=True, slots=True)
class InputRef:
    """Reference to an input binding in an operation."""

    key: Hashable

    def __post_init__(self) -> None:
        try:
            hash(self.key)
        except TypeError as exc:
            raise TypeError(f"input key {self.key!r} is not hashable") from exc


@dataclass(frozen=True, slots=True)
class NodeRef:
    """Reference to an operation result in a compiled plan."""

    index: int

    def __post_init__(self) -> None:
        try:
            normalized_index = operator.index(self.index)
        except TypeError as exc:
            raise TypeError("operation reference index must be an integer") from exc
        if normalized_index < 0:
            raise ValueError("operation reference index must be non-negative")
        object.__setattr__(self, "index", normalized_index)


Ref: TypeAlias = InputRef | NodeRef


@dataclass(frozen=True, slots=True)
class Operation:
    """A unary or binary explicit-index NumPy operation.

    ``input_subscripts`` and ``output_subscript`` contain integer axis IDs.
    The compiler guarantees that there are one or two dependencies;
    execution consequently never invokes an unrestricted global contraction.
    """

    index: int
    inputs: tuple[Ref, ...]
    input_subscripts: tuple[tuple[int, ...], ...]
    output_subscript: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: np.dtype

    def __post_init__(self) -> None:
        try:
            normalized_index = operator.index(self.index)
        except TypeError as exc:
            raise TypeError("operation index must be an integer") from exc
        if normalized_index < 0:
            raise ValueError("operation index must be non-negative")
        object.__setattr__(self, "index", normalized_index)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        pattern = AxisPattern(self.input_subscripts, self.output_subscript)
        object.__setattr__(self, "input_subscripts", pattern.inputs)
        object.__setattr__(self, "output_subscript", pattern.output)
        if len(self.inputs) not in (1, 2):
            raise ValueError("contraction operations must be unary or binary")
        if len(self.inputs) != len(self.input_subscripts):
            raise ValueError("one input subscript is required per operation dependency")
        object.__setattr__(self, "output_shape", _normalise_shape(self.output_shape))
        try:
            object.__setattr__(self, "output_dtype", np.dtype(self.output_dtype))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid operation dtype {self.output_dtype!r}") from exc

    @property
    def equation(self) -> AxisPattern:
        """The integer relation; execution adapts it to the selected backend."""
        return AxisPattern(self.input_subscripts, self.output_subscript)


@dataclass(frozen=True, slots=True)
class CompiledTerm:
    """Compiled output reference and coefficient for one source term."""

    output: Ref
    output_subscript: tuple[int, ...]
    coefficient: Any
    output_shape: tuple[int, ...]
    output_dtype: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_subscript", _axes(self.output_subscript))
        object.__setattr__(self, "output_shape", _normalise_shape(self.output_shape))
        try:
            object.__setattr__(self, "output_dtype", np.dtype(self.output_dtype))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid compiled term dtype {self.output_dtype!r}") from exc


class _MappingView(Mapping[str, Any]):
    """Small immutable, callable mapping used for summaries and statistics.

    Calling the view returns an ordinary dictionary.  This permits both
    ``plan.summary["unique_operations"]`` and ``plan.summary()`` without
    exposing mutable state in the plan itself.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = MappingProxyType(dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return repr(dict(self._data))


@dataclass(frozen=True, slots=True)
class ContractionPlan:
    """Immutable shared contraction DAG produced by :func:`compile_plan`."""

    specs: Mapping[Hashable, TensorSpec]
    operations: tuple[Operation, ...]
    terms: tuple[CompiledTerm, ...]
    optimize: Any = "greedy"
    operation_use_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        specs = dict(self.specs)
        for key, spec in specs.items():
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(f"input key {key!r} is not hashable") from exc
            if not isinstance(spec, TensorSpec):
                raise TypeError(f"spec for {key!r} must be a TensorSpec")
        object.__setattr__(self, "specs", MappingProxyType(specs))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "terms", tuple(self.terms))
        uses = tuple(operator.index(count) for count in self.operation_use_counts)
        if uses and len(uses) != len(self.operations):
            raise ValueError("one operation use count is required per operation")
        if not uses:
            uses = tuple(1 for _ in self.operations)
        if any(count < 1 for count in uses):
            raise ValueError("operation use counts must be positive")
        object.__setattr__(self, "operation_use_counts", uses)
        for expected_index, operation in enumerate(self.operations):
            if operation.index != expected_index:
                raise ValueError("operation indices must be contiguous and creation ordered")

    @property
    def total_operations(self) -> int:
        """Number of operations before exact common-operation elimination."""

        return int(sum(self.operation_use_counts))

    @property
    def unique_operations(self) -> int:
        """Number of operation nodes retained by the shared DAG."""

        return len(self.operations)

    @property
    def shared_operations(self) -> int:
        """Number of retained nodes referenced by more than one source operation."""

        return sum(count > 1 for count in self.operation_use_counts)

    @property
    def eliminated_operations(self) -> int:
        """Number of exact duplicate operation occurrences removed by CSE."""

        return self.total_operations - self.unique_operations

    @property
    def summary(self) -> _MappingView:
        """Deterministic structural summary (also callable for a plain dict)."""

        return _MappingView(
            {
                "terms": len(self.terms),
                "total_operations": self.total_operations,
                "unique_operations": self.unique_operations,
                "shared_operations": self.shared_operations,
                "eliminated_operations": self.eliminated_operations,
                "optimize": repr(self.optimize),
            }
        )


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """Synchronous logical-memory and operation metrics.

    ``peak_live_bytes`` includes plan-owned intermediate arrays and copied
    output arrays that are still logically live.  User-owned input binding
    memory and NumPy's allocator or temporary scratch buffers are excluded;
    ``input_bytes`` reports the former separately for reference.  No timing or
    RSS measurement is implied by these metrics.
    """

    executed_operations: int
    total_operations: int
    unique_operations: int
    shared_operations: int
    peak_live_bytes: int
    peak_intermediate_bytes: int
    input_bytes: int
    output_bytes: int
    scratch_bytes: int | None = None

    @property
    def peak_logical_live_bytes(self) -> int:
        """Alias spelling used by the architecture contract."""

        return self.peak_live_bytes

    @property
    def operation_count(self) -> int:
        """Alias for the number of operation nodes executed."""

        return self.executed_operations

    def as_dict(self) -> dict[str, Any]:
        return {
            "executed_operations": self.executed_operations,
            "total_operations": self.total_operations,
            "unique_operations": self.unique_operations,
            "shared_operations": self.shared_operations,
            "peak_live_bytes": self.peak_live_bytes,
            "peak_logical_live_bytes": self.peak_logical_live_bytes,
            "peak_intermediate_bytes": self.peak_intermediate_bytes,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "scratch_bytes": self.scratch_bytes,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Numerical term values and metrics from one plan execution."""

    values: tuple[Any, ...]
    stats: ExecutionStats

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.stats, ExecutionStats):
            raise TypeError("execution stats must be an ExecutionStats instance")
