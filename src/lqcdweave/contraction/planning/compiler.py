"""Compile explicit-index contraction terms into a shared unary/binary DAG."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import operator
from typing import Any

import numpy as np
import opt_einsum

from lqcdweave.contracts.correlator.api import (
    CompiledTerm,
    ContractionPlan,
    ContractionTerm,
    InputRef,
    NodeRef,
    Operation,
    Ref,
    TensorSpec,
    AxisPattern,
    parse_term,
)




@dataclass(frozen=True, slots=True)
class _State:
    ref: Ref
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: np.dtype






def _normalised_pattern(input_subscripts: tuple[tuple[int, ...], ...], output_subscript: tuple[int, ...]) -> AxisPattern:
    """Canonicalize dummy labels while preserving all explicit positions."""

    return AxisPattern(input_subscripts, output_subscript).normalized()


def _retained_labels(
    selected: Sequence[_State],
    remaining: Sequence[_State],
    output_labels: tuple[int, ...],
) -> tuple[int, ...]:
    """Determine labels which a unary/binary step must pass onward."""

    needed = set(output_labels)
    for state in remaining:
        needed.update(state.labels)
    result: list[int] = []
    for state in selected:
        for label in state.labels:
            if label in needed and label not in result:
                result.append(label)
    return tuple(result)


def compile_plan(
    specs: Mapping[Hashable, TensorSpec],
    terms: Sequence[ContractionTerm],
    *,
    optimize: Any = "greedy",
) -> ContractionPlan:
    """Compile explicit-index terms into a deterministic shared contraction DAG.

    The planner passes integer sets to :mod:`opt_einsum.paths`, then lowers
    each unary or binary step to a local integer relation. Exact CSE uses ordered
    dependency references and a position-preserving dummy-label normalization;
    it never infers graph, conjugation, fermionic or operand-permutation
    equivalence.
    """

    if not isinstance(specs, Mapping):
        raise TypeError("specs must be a mapping of input keys to TensorSpec objects")
    normalized_specs: dict[Hashable, TensorSpec] = {}
    for key, spec in specs.items():
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"input key {key!r} is not hashable") from exc
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"spec for input key {key!r} must be a TensorSpec")
        normalized_specs[key] = spec
    try:
        normalized_terms = tuple(terms)
    except TypeError as exc:
        raise TypeError("terms must be a sequence of ContractionTerm objects") from exc

    parsed_terms = tuple(
        parse_term(term, normalized_specs, term_index) for term_index, term in enumerate(normalized_terms)
    )
    if (any(term.sum_barriers for term in normalized_terms)
            and any(spec.dtype in (np.dtype('float32'), np.dtype('complex64'))
                    for spec in normalized_specs.values())):
        raise NotImplementedError('sum barriers require the ordinary scene planner and fixed Action replay')

    operations: list[Operation] = []
    operation_use_counts: list[int] = []
    operation_keys: dict[tuple[Any, ...], int] = {}

    def intern(
        refs: tuple[Ref, ...],
        input_subscripts: tuple[tuple[int, ...], ...],
        output_subscript: tuple[int, ...],
        output_shape: tuple[int, ...],
        output_dtype: np.dtype,
    ) -> NodeRef:
        pattern = _normalised_pattern(input_subscripts, output_subscript)
        key = (refs, pattern)
        operation_index = operation_keys.get(key)
        if operation_index is not None:
            operation_use_counts[operation_index] += 1
            return NodeRef(operation_index)
        operation_index = len(operations)
        operation_keys[key] = operation_index
        operations.append(
            Operation(
                index=operation_index,
                inputs=refs,
                input_subscripts=input_subscripts,
                output_subscript=output_subscript,
                output_shape=output_shape,
                output_dtype=output_dtype,
            )
        )
        operation_use_counts.append(1)
        return NodeRef(operation_index)

    compiled_terms: list[CompiledTerm] = []
    for term_index, parsed in enumerate(parsed_terms):
        states: list[_State] = [
            _State(
                ref=InputRef(key),
                labels=labels,
                shape=specs[key].shape,
                dtype=specs[key].dtype,
            )
            for key, labels in zip(parsed.input_keys, parsed.input_labels, strict=True)
        ]
        try:
            # Call the path algorithms with integer sets directly. The
            # contract_path wrapper builds string equations and PathInfo,
            # neither of which is needed by the compiler. Preserve its path
            # choice and operand ordering, including explicit paths.
            count = len(parsed.input_keys)
            option = "auto" if optimize is True else optimize
            if option is False:
                path = [tuple(range(count))]
            elif not isinstance(option, (str, opt_einsum.paths.PathOptimizer)):
                path = option
            elif count <= 2:
                path = [tuple(range(count))]
            else:
                optimizer = (opt_einsum.paths.get_path_fn(option) if isinstance(option, str) else option)
                path = optimizer(
                    [frozenset(axes) for axes in parsed.input_labels],
                    frozenset(parsed.output_labels),
                    {axis: max(size, 1) for axis, size in parsed.label_dimensions.items()},
                    None,
                )
        except Exception as exc:
            raise ValueError(f"term {term_index}: unable to construct contraction path") from exc

        for step_index, raw_step in enumerate(path):
            try:
                step = tuple(operator.index(index) for index in raw_step)
            except TypeError as exc:
                raise ValueError(f"term {term_index}: invalid path step {raw_step!r}") from exc
            if len(step) not in (1, 2):
                raise ValueError(
                    f"term {term_index}: path step {step_index} has {len(step)} operands; "
                    "only unary and binary lowering is supported"
                )
            if len(set(step)) != len(step) or any(index < 0 or index >= len(states) for index in step):
                raise ValueError(f"term {term_index}: invalid path operand indices {step!r}")
            selected = tuple(states[index] for index in step)
            remaining = tuple(state for index, state in enumerate(states) if index not in step)
            output_labels = _retained_labels(selected, remaining, parsed.output_labels)
            output_shape = tuple(
                next(state.shape[state.labels.index(label)] for state in selected if label in state.labels)
                for label in output_labels
            )
            output_dtype = np.result_type(*(state.dtype for state in selected))
            node_ref = intern(
                tuple(state.ref for state in selected),
                tuple(state.labels for state in selected),
                output_labels,
                output_shape,
                output_dtype,
            )
            states = list(remaining) + [
                _State(ref=node_ref, labels=output_labels, shape=output_shape, dtype=output_dtype)
            ]

        if len(states) != 1:  # pragma: no cover - opt_einsum always completes a valid path
            raise ValueError(f"term {term_index}: contraction path did not produce one result")
        state = states[0]
        if state.labels != parsed.output_labels or not isinstance(state.ref, NodeRef):
            output_dtype = state.dtype
            state = _State(
                ref=intern(
                    (state.ref,),
                    (state.labels,),
                    parsed.output_labels,
                    parsed.output_shape,
                    output_dtype,
                ),
                labels=parsed.output_labels,
                shape=parsed.output_shape,
                dtype=output_dtype,
            )
        coefficient_dtype = np.result_type(state.dtype, parsed.source.coefficient)
        compiled_terms.append(
            CompiledTerm(
                output=state.ref,
                output_subscript=parsed.output_labels,
                coefficient=parsed.source.coefficient,
                output_shape=parsed.output_shape,
                output_dtype=coefficient_dtype,
            )
        )

    return ContractionPlan(
        specs=normalized_specs,
        operations=tuple(operations),
        terms=tuple(compiled_terms),
        optimize=optimize,
        operation_use_counts=tuple(operation_use_counts),
    )
