"""Versioned persistence for pure contraction plans and storage schedules.

All supported payloads use one versioned JSON envelope and one explicit
type whitelist. The module stores descriptions and explicit ArrayConstant
records, never production arrays, loaders, adapters, callbacks or backend modules. Loading
validates the requested task and can rebind a fixed contraction order, but
does not search for a new order or execute numerical work.
"""

from __future__ import annotations
from lqcdweave.contracts.correlator.api import validate_executable_resources

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import hmac
from inspect import Parameter, signature
import json
import math
import numbers
from dataclasses import fields, replace, MISSING
from functools import lru_cache
import gzip
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from lqcdweave.contracts.correlator import api as contract_api
from .archive_io import _atomic_archive_file

if TYPE_CHECKING:
    from lqcdweave.contracts.memory import TargetHardwareMemory
    from lqcdweave.contraction.performance.api import CostProfile


ARCHIVE_FORMAT = "easy-distillation.contraction-plan"
ARCHIVE_VERSION = 8

_PAYLOAD_PLAN_BUNDLE = "plan-bundle"
_PAYLOAD_STORAGE_SCHEDULE = "storage-schedule"

ExecutionBudget = contract_api.ExecutionBudget
PlanBatch = contract_api.PlanBatch
PlanBundle = contract_api.PlanBundle
WorkloadSpec = contract_api.WorkloadSpec
ContractionPlan = contract_api.ContractionPlan
ContractionTerm = contract_api.ContractionTerm
InputRef = contract_api.InputRef
NodeRef = contract_api.NodeRef
Operation = contract_api.Operation
CompiledTerm = contract_api.CompiledTerm
Ref = contract_api.Ref
TensorSpec = contract_api.TensorSpec
_decode_scalar = contract_api.decode_scalar
_encode_scalar = contract_api.encode_scalar
_thaw_json = contract_api.thaw_json


class PlanArchiveError(ValueError):
    """Raised when a plan archive is malformed, stale, or incompatible."""


def _fail(message: str) -> PlanArchiveError:
    return PlanArchiveError(message)


def _json_bytes(value: Any) -> bytes:
    try:
        return contract_api.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PlanArchiveError(f"archive contains a non-JSON value: {exc}") from exc


def _digest(value: Any) -> str:
    try:
        return contract_api.json_digest(value)
    except (TypeError, ValueError) as exc:
        raise PlanArchiveError(f"archive contains a non-JSON value: {exc}") from exc


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not permitted")


def _strict_object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise _fail(f"{name} keys must be strings")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"unexpected={extra}")
        raise _fail(f"{name} has invalid fields ({', '.join(detail)})")


def _dtype_from_text(value: Any, *, name: str) -> np.dtype:
    if not isinstance(value, str):
        raise _fail(f"{name} dtype must be a string")
    try:
        return np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} has invalid dtype {value!r}") from exc


def _shape_from_json(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise _fail(f"{name} shape must be a JSON list")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise _fail(f"{name} shape dimensions must be non-negative integers")
        shape.append(dimension)
    return tuple(shape)


def _decode_ref(value: Any, *, name: str) -> Ref:
    obj = _strict_object(value, name=name)
    kind = obj.get("type")
    if kind == "input":
        _strict_keys(obj, {"type", "key"}, name=name)
        if not isinstance(obj["key"], str):
            raise _fail(f"{name} input key must be a string")
        return InputRef(obj["key"])
    if kind == "node":
        _strict_keys(obj, {"type", "index"}, name=name)
        index = obj["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise _fail(f"{name} node index must be a non-negative integer")
        return NodeRef(index)
    raise _fail(f"{name} has unknown reference type {kind!r}")


def _encode_workload(workload: WorkloadSpec) -> dict[str, Any]:
    try:
        payload = contract_api.encode_workload(workload)
        for item, (key, spec) in zip(payload["specs"], workload.specs.items(), strict=True):
            if item["key"] != key:
                raise PlanArchiveError("workload spec order changed during archive encoding")
            item["description"] = (
                None if spec.description is None else _encode_plan_value(spec.description)
            )
        return payload
    except (TypeError, ValueError) as exc:
        raise PlanArchiveError(str(exc)) from exc


def _decode_workload(value: Any) -> WorkloadSpec:
    obj = _strict_object(value, name="workload")
    _strict_keys(
        obj,
        {"specs", "terms", "input_sources", "convention", "output_groups", "metadata"},
        name="workload",
    )
    raw_specs = obj["specs"]
    if not isinstance(raw_specs, list):
        raise _fail("workload.specs must be a JSON list")
    specs: dict[str, TensorSpec] = {}
    for index, raw in enumerate(raw_specs):
        item = _strict_object(raw, name=f"workload.specs[{index}]")
        if set(item) not in ({"key", "shape", "dtype"}, {"key", "shape", "dtype", "description"}):
            _strict_keys(item, {"key", "shape", "dtype"}, name=f"workload.specs[{index}]")
        key = item["key"]
        if not isinstance(key, str) or key in specs:
            raise _fail(f"workload.specs[{index}] has a duplicate or invalid key")
        description = item.get("description")
        if description is not None:
            description = _decode_plan_value(description)
            if not isinstance(description, contract_api.TensorDescription):
                raise _fail(f"workload.specs[{index}].description has an invalid type")
        specs[key] = TensorSpec(
            _shape_from_json(item["shape"], name=f"workload.specs[{index}]"),
            _dtype_from_text(item["dtype"], name=f"workload.specs[{index}]"),
            description=description,
        )

    raw_terms = obj["terms"]
    if not isinstance(raw_terms, list):
        raise _fail("workload.terms must be a JSON list")
    terms: list[ContractionTerm] = []
    for index, raw in enumerate(raw_terms):
        item = _strict_object(raw, name=f"workload.terms[{index}]")
        _strict_keys(item, {"inputs", "subscripts", "coefficient"}, name=f"workload.terms[{index}]")
        inputs = item["inputs"]
        if not isinstance(inputs, list) or any(not isinstance(key, str) for key in inputs):
            raise _fail(f"workload.terms[{index}].inputs must be a list of strings")
        try:
            coefficient = _decode_scalar(item["coefficient"])
            pattern = _decode_axis_pattern(item["subscripts"], name=f"workload.terms[{index}].subscripts")
            terms.append(ContractionTerm(tuple(inputs), pattern, coefficient))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _fail(f"invalid workload term {index}: {exc}") from exc

    sources = _strict_object(obj["input_sources"], name="workload.input_sources")
    metadata = _strict_object(obj["metadata"], name="workload.metadata")
    groups = obj["output_groups"]
    if not isinstance(groups, list):
        raise _fail("workload.output_groups must be a JSON list")
    try:
        normalized_groups = tuple(
            tuple(index for index in group)
            for group in groups
        )
    except TypeError as exc:
        raise _fail("workload.output_groups must contain lists") from exc
    if any(
        not isinstance(group, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in group)
        for group in groups
    ):
        raise _fail("workload.output_groups must contain lists of integers")
    try:
        return WorkloadSpec(
            specs=specs,
            terms=tuple(terms),
            input_sources=dict(sources),
            convention=obj["convention"],
            output_groups=normalized_groups,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise _fail(f"invalid workload: {exc}") from exc


def _decode_axes(value, *, name):
    if not isinstance(value, list) or any(type(axis) is not int or axis < 0 for axis in value):
        raise _fail(f"{name} must be a list of non-negative integer axis IDs")
    return tuple(value)


def _decode_axis_pattern(value, *, name):
    obj = _strict_object(value, name=name)
    _strict_keys(obj, {"inputs", "output"}, name=name)
    if not isinstance(obj["inputs"], list):
        raise _fail(f"{name}.inputs must be a list of axis lists")
    return contract_api.AxisPattern(
        tuple(_decode_axes(axes, name=f"{name}.inputs[{index}]") for index, axes in enumerate(obj["inputs"])),
        _decode_axes(obj["output"], name=f"{name}.output"),
    )


def _encode_plan(plan: ContractionPlan) -> dict[str, Any]:
    try:
        payload = contract_api.encode_plan(plan)
        for item, (key, spec) in zip(payload["specs"], plan.specs.items(), strict=True):
            if item["key"] != key:
                raise PlanArchiveError("compiled spec order changed during archive encoding")
            item["description"] = (
                None if spec.description is None else _encode_plan_value(spec.description)
            )
        return payload
    except (TypeError, ValueError) as exc:
        raise PlanArchiveError(str(exc)) from exc


def _decode_plan(value: Any, *, name: str) -> ContractionPlan:
    obj = _strict_object(value, name=name)
    _strict_keys(
        obj,
        {"specs", "operations", "terms", "optimize", "operation_use_counts"},
        name=name,
    )
    if not isinstance(obj["optimize"], str) or not obj["optimize"]:
        raise _fail(f"{name}.optimize must be a non-empty string")

    raw_specs = obj["specs"]
    if not isinstance(raw_specs, list):
        raise _fail(f"{name}.specs must be a JSON list")
    specs: dict[str, TensorSpec] = {}
    for index, raw in enumerate(raw_specs):
        item = _strict_object(raw, name=f"{name}.specs[{index}]")
        if set(item) not in ({"key", "shape", "dtype"}, {"key", "shape", "dtype", "description"}):
            _strict_keys(item, {"key", "shape", "dtype"}, name=f"{name}.specs[{index}]")
        key = item["key"]
        if not isinstance(key, str) or key in specs:
            raise _fail(f"{name}.specs[{index}] has a duplicate or invalid key")
        description = item.get("description")
        if description is not None:
            description = _decode_plan_value(description)
            if not isinstance(description, contract_api.TensorDescription):
                raise _fail(f"{name}.specs[{index}].description has an invalid type")
        specs[key] = TensorSpec(
            _shape_from_json(item["shape"], name=f"{name}.specs[{index}]"),
            _dtype_from_text(item["dtype"], name=f"{name}.specs[{index}]"),
            description=description,
        )

    operations_raw = obj["operations"]
    if not isinstance(operations_raw, list):
        raise _fail(f"{name}.operations must be a JSON list")
    operations: list[Operation] = []
    for index, raw in enumerate(operations_raw):
        item = _strict_object(raw, name=f"{name}.operations[{index}]")
        _strict_keys(
            item,
            {
                "index",
                "inputs",
                "input_subscripts",
                "output_subscript",
                "output_shape",
                "output_dtype",
            },
            name=f"{name}.operations[{index}]",
        )
        if isinstance(item["index"], bool) or not isinstance(item["index"], int):
            raise _fail(f"{name}.operations[{index}].index must be an integer")
        inputs_raw = item["inputs"]
        subscripts_raw = item["input_subscripts"]
        if not isinstance(inputs_raw, list) or not isinstance(subscripts_raw, list):
            raise _fail(f"{name}.operations[{index}] inputs/subscripts must be lists")
        try:
            refs = tuple(
                _decode_ref(ref, name=f"{name}.operations[{index}].inputs[{ref_index}]")
                for ref_index, ref in enumerate(inputs_raw)
            )
            operations.append(
                Operation(
                    index=item["index"],
                    inputs=refs,
                    input_subscripts=tuple(_decode_axes(axes, name=f"{name}.operations[{index}].input_subscripts") for axes in subscripts_raw),
                    output_subscript=_decode_axes(item["output_subscript"], name=f"{name}.operations[{index}].output_subscript"),
                    output_shape=_shape_from_json(
                        item["output_shape"], name=f"{name}.operations[{index}]"
                    ),
                    output_dtype=_dtype_from_text(
                        item["output_dtype"], name=f"{name}.operations[{index}]"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise _fail(f"invalid {name}.operations[{index}]: {exc}") from exc

    terms_raw = obj["terms"]
    if not isinstance(terms_raw, list):
        raise _fail(f"{name}.terms must be a JSON list")
    compiled_terms: list[CompiledTerm] = []
    for index, raw in enumerate(terms_raw):
        item = _strict_object(raw, name=f"{name}.terms[{index}]")
        _strict_keys(
            item,
            {"output", "output_subscript", "coefficient", "output_shape", "output_dtype"},
            name=f"{name}.terms[{index}]",
        )
        try:
            compiled_terms.append(
                CompiledTerm(
                    output=_decode_ref(item["output"], name=f"{name}.terms[{index}].output"),
                    output_subscript=_decode_axes(item["output_subscript"], name=f"{name}.terms[{index}].output_subscript"),
                    coefficient=_decode_scalar(item["coefficient"]),
                    output_shape=_shape_from_json(
                        item["output_shape"], name=f"{name}.terms[{index}]"
                    ),
                    output_dtype=_dtype_from_text(
                        item["output_dtype"], name=f"{name}.terms[{index}]"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise _fail(f"invalid {name}.terms[{index}]: {exc}") from exc

    uses = obj["operation_use_counts"]
    if not isinstance(uses, list) or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in uses
    ):
        raise _fail(f"{name}.operation_use_counts must be a list of positive integers")
    try:
        return ContractionPlan(
            specs=specs,
            operations=tuple(operations),
            terms=tuple(compiled_terms),
            optimize=obj["optimize"],
            operation_use_counts=tuple(uses),
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"invalid {name}: {exc}") from exc


def _encode_budget(budget: ExecutionBudget) -> dict[str, Any]:
    return budget.as_dict()


def _decode_budget(value: Any) -> ExecutionBudget:
    obj = _strict_object(value, name="budget")
    _strict_keys(
        obj,
        {
            "host_bytes",
            "device_bytes",
            "reserve_fraction",
            "min_reserve_bytes",
            "workspace_bytes",
            "host_staging_bytes",
        },
        name="budget",
    )
    try:
        return ExecutionBudget(
            host_bytes=obj["host_bytes"],
            device_bytes=obj["device_bytes"],
            reserve_fraction=obj["reserve_fraction"],
            min_reserve_bytes=obj["min_reserve_bytes"],
            workspace_bytes=obj["workspace_bytes"],
            host_staging_bytes=obj["host_staging_bytes"],
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"invalid budget: {exc}") from exc


def _plan_ref_info(
    ref: Ref,
    specs: Mapping[str, TensorSpec],
    operations: Sequence[Operation],
) -> tuple[tuple[int, ...], np.dtype, tuple[int, ...] | None]:
    if isinstance(ref, InputRef):
        if not isinstance(ref.key, str) or ref.key not in specs:
            raise _fail(f"operation references unknown input key {ref.key!r}")
        return tuple(specs[ref.key].shape), np.dtype(specs[ref.key].dtype), None
    if not isinstance(ref, NodeRef) or ref.index >= len(operations):
        raise _fail("operation references an unknown node")
    operation = operations[ref.index]
    return tuple(operation.output_shape), np.dtype(operation.output_dtype), operation.output_subscript


def _parse_equation(equation: str, *, name: str) -> tuple[tuple[str, ...], str]:
    if not isinstance(equation, str):
        raise _fail(f"{name} must be a string")
    cleaned = "".join(equation.split())
    if cleaned.count("->") != 1 or "..." in cleaned:
        raise _fail(f"{name} is not an explicit-index equation")
    left, output = cleaned.split("->")
    inputs = tuple(left.split(","))
    if not inputs or any(
        not labels.isascii() or (bool(labels) and not labels.isalpha()) for labels in inputs
    ):
        raise _fail(f"{name} has invalid input labels")
    if not output.isascii() or not output.isalpha() and output:
        raise _fail(f"{name} has invalid output labels")
    if len(set(output)) != len(output):
        raise _fail(f"{name} output labels must be unique")
    return inputs, output


def _validate_plan_against_workload(
    plan: ContractionPlan,
    workload: WorkloadSpec,
    term_indices: tuple[int, ...],
    *,
    name: str,
) -> None:
    if not isinstance(plan.optimize, str) or not plan.optimize:
        raise _fail(f"{name}.optimize must be a non-empty string")
    if len(term_indices) != len(plan.terms):
        raise _fail(f"{name} has one term index required per compiled term")
    required_keys: set[str] = set()
    for source_index in term_indices:
        if source_index < 0 or source_index >= len(workload.terms):
            raise _fail(f"{name} term index {source_index} is outside workload")
        required_keys.update(workload.terms[source_index].inputs)
    if set(plan.specs) != required_keys:
        raise _fail(
            f"{name}.specs keys must equal the inputs used by its term batch "
            f"(expected={sorted(required_keys)}, actual={sorted(plan.specs)})"
        )
    for key in required_keys:
        spec = workload.specs[key]
        actual = plan.specs.get(key)
        if actual is None or tuple(actual.shape) != tuple(spec.shape) or np.dtype(actual.dtype) != np.dtype(spec.dtype):
            raise _fail(f"{name}.specs[{key!r}] does not match workload shape/dtype")

    for operation_index, operation in enumerate(plan.operations):
        if operation.index != operation_index:
            raise _fail(f"{name}.operations indices must be contiguous and ordered")
        if len(operation.inputs) not in (1, 2):
            raise _fail(f"{name}.operations[{operation_index}] must be unary or binary")
        if len(operation.inputs) != len(operation.input_subscripts):
            raise _fail(f"{name}.operations[{operation_index}] has mismatched inputs/subscripts")
        input_infos: list[tuple[tuple[int, ...], np.dtype, tuple[int, ...] | None]] = []
        for ref_index, ref in enumerate(operation.inputs):
            if isinstance(ref, NodeRef) and ref.index >= operation_index:
                raise _fail(f"{name}.operations[{operation_index}] is not topologically ordered")
            shape, dtype, labels = _plan_ref_info(ref, plan.specs, plan.operations)
            labels_text = operation.input_subscripts[ref_index]
            if len(labels_text) != len(shape):
                raise _fail(f"{name}.operations[{operation_index}] input rank mismatch")
            input_infos.append((shape, dtype, labels))
        output_subscript = operation.output_subscript
        if len(set(output_subscript)) != len(output_subscript):
            raise _fail(f"{name}.operations[{operation_index}] output labels must be unique")
        dimensions: dict[int, int] = {}
        for (shape, _dtype, _labels), labels_text in zip(
            input_infos, operation.input_subscripts, strict=True
        ):
            local: dict[int, int] = {}
            for label, dimension in zip(labels_text, shape, strict=True):
                previous = local.get(label)
                if previous is not None and previous != dimension:
                    raise _fail(f"{name}.operations[{operation_index}] repeats a label at two sizes")
                local[label] = dimension
                previous = dimensions.get(label)
                if previous is not None and previous != dimension:
                    raise _fail(f"{name}.operations[{operation_index}] has incompatible label sizes")
                dimensions[label] = dimension
        if any(label not in dimensions for label in output_subscript):
            raise _fail(f"{name}.operations[{operation_index}] output uses an unknown label")
        expected_shape = tuple(dimensions[label] for label in output_subscript)
        if tuple(operation.output_shape) != expected_shape:
            raise _fail(f"{name}.operations[{operation_index}] output shape is inconsistent")
        try:
            expected_dtype = np.result_type(*(dtype for _shape, dtype, _labels in input_infos))
        except TypeError as exc:
            raise _fail(f"{name}.operations[{operation_index}] has invalid input dtypes") from exc
        if np.dtype(operation.output_dtype) != np.dtype(expected_dtype):
            raise _fail(f"{name}.operations[{operation_index}] output dtype is inconsistent")

    uses = tuple(plan.operation_use_counts)
    if len(uses) != len(plan.operations) or any(count < 1 for count in uses):
        raise _fail(f"{name}.operation_use_counts is inconsistent")

    for local_index, (term, source_index) in enumerate(zip(plan.terms, term_indices, strict=True)):
        if source_index < 0 or source_index >= len(workload.terms):
            raise _fail(f"{name} term index {source_index} is outside workload")
        source = workload.terms[source_index]
        parsed = contract_api.parse_term(source, workload.specs, source_index)
        input_labels, output_labels = parsed.input_labels, parsed.output_labels
        if len(input_labels) != len(source.inputs):
            raise _fail(f"workload.terms[{source_index}] input count does not match equation")
        dimensions: dict[int, int] = {}
        for key, labels in zip(source.inputs, input_labels, strict=True):
            shape = workload.specs[key].shape
            if len(labels) != len(shape):
                raise _fail(f"workload.terms[{source_index}] input rank mismatch")
            local: dict[int, int] = {}
            for label, dimension in zip(labels, shape, strict=True):
                if label in local and local[label] != dimension:
                    raise _fail(f"workload.terms[{source_index}] repeats a label at two sizes")
                local[label] = dimension
                if label in dimensions and dimensions[label] != dimension:
                    raise _fail(f"workload.terms[{source_index}] has incompatible label sizes")
                dimensions[label] = dimension
        if any(label not in dimensions for label in output_labels):
            raise _fail(f"workload.terms[{source_index}] output uses an unknown label")
        expected_shape = tuple(dimensions[label] for label in output_labels)
        if term.output_subscript != output_labels or tuple(term.output_shape) != expected_shape:
            raise _fail(f"{name}.terms[{local_index}] output shape/labels do not match source term")
        if _encode_scalar(term.coefficient) != _encode_scalar(source.coefficient):
            raise _fail(f"{name}.terms[{local_index}] coefficient does not match source term")
        if isinstance(term.output, NodeRef):
            if term.output.index >= len(plan.operations):
                raise _fail(f"{name}.terms[{local_index}] references an unknown node")
            output_operation = plan.operations[term.output.index]
            ref_shape = output_operation.output_shape
            ref_dtype = output_operation.output_dtype
        elif isinstance(term.output, InputRef):
            if term.output.key not in plan.specs:
                raise _fail(f"{name}.terms[{local_index}] references an unknown input")
            ref_shape = plan.specs[term.output.key].shape
            ref_dtype = plan.specs[term.output.key].dtype
        else:
            raise _fail(f"{name}.terms[{local_index}] has an invalid output reference")
        if tuple(term.output_shape) != tuple(ref_shape):
            raise _fail(f"{name}.terms[{local_index}] output shape does not match its node")
        try:
            expected_dtype = np.result_type(ref_dtype, term.coefficient)
        except TypeError as exc:
            raise _fail(f"{name}.terms[{local_index}] has invalid coefficient dtype") from exc
        if np.dtype(term.output_dtype) != np.dtype(expected_dtype):
            raise _fail(f"{name}.terms[{local_index}] output dtype is inconsistent")


def _validate_bundle(bundle: PlanBundle) -> None:
    seen: set[int] = set()
    for index, batch in enumerate(bundle.batches):
        _validate_plan_against_workload(
            batch.plan,
            bundle.workload,
            batch.term_indices,
            name=f"batches[{index}].plan",
        )
        if any(term_index in seen for term_index in batch.term_indices):
            raise _fail("plan batches contain duplicate term indices")
        seen.update(batch.term_indices)
    if seen != set(range(len(bundle.workload.terms))):
        raise _fail("plan batches do not cover every workload term")


def validate_bundle(bundle: PlanBundle) -> None:
    """Validate an in-memory bundle without compiling or searching paths.

    Session constructors use this entry point for a plan supplied directly by
    Python code.  Archive loading calls the same validator after rebuilding
    objects from JSON, so the two paths enforce identical DAG invariants.
    """

    if not isinstance(bundle, PlanBundle):
        raise TypeError("validate_bundle expects a PlanBundle")
    _validate_bundle(bundle)


def structural_signature(bundle: PlanBundle) -> str:
    """Return a deterministic digest covering all executable structure."""

    try:
        return contract_api.structural_signature(bundle)
    except TypeError:
        raise
    except ValueError as exc:
        raise PlanArchiveError(str(exc)) from exc


def _archive_payload(bundle: PlanBundle) -> dict[str, Any]:
    """Return the integer plan-bundle payload for the current archive envelope."""

    return {
        "signature": structural_signature(bundle),
        "workload": _encode_workload(bundle.workload),
        "batches": [
            {
                "term_indices": list(batch.term_indices),
                "memory_estimate": _thaw_json(batch.memory_estimate),
                "plan": _encode_plan(batch.plan),
            }
            for batch in bundle.batches
        ],
        "budget": _encode_budget(bundle.budget),
        "metadata": _thaw_json(bundle.metadata),
    }


def _make_envelope(payload_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload_type not in {_PAYLOAD_PLAN_BUNDLE, _PAYLOAD_STORAGE_SCHEDULE, "correlator-plan", "complete-cfg-plan"}:
        raise PlanArchiveError(f"unknown archive payload type {payload_type!r}")
    body = {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "payload_type": payload_type,
        "payload": dict(payload),
    }
    return body


def _decode_envelope(document: Any, *, expected_type: str) -> Mapping[str, Any]:
    obj = _strict_object(document, name="archive")
    _strict_keys(
        obj,
        {"format", "version", "payload_type", "payload"},
        name="archive",
    )
    if obj["format"] != ARCHIVE_FORMAT:
        raise PlanArchiveError(f"unsupported plan archive format {obj['format']!r}")
    # v5 generic DAGs already use integer axes. Earlier Scene/cfg payloads
    # require explicit offline conversion before any runtime objects are bound.
    versions = {5, 6, 7, ARCHIVE_VERSION} if expected_type == _PAYLOAD_PLAN_BUNDLE else {ARCHIVE_VERSION}
    if type(obj["version"]) is not int or obj["version"] not in versions:
        raise PlanArchiveError(
            f"unsupported plan archive version {obj['version']!r}; expected {ARCHIVE_VERSION}; "
            + ("legacy archives require explicit offline migration" if expected_type == _PAYLOAD_PLAN_BUNDLE
             else "generate a new plan; old matrix plans have no migrator")
        )
    if obj["payload_type"] != expected_type:
        raise PlanArchiveError(
            f"archive payload type {obj['payload_type']!r} is not {expected_type!r}"
        )
    return _strict_object(obj["payload"], name="archive.payload")


def _read_document(source: Any) -> Any:
    """Read path, UTF-8 text, or bytes, accepting gzip by magic bytes."""

    display = "<bytes>"
    with ExitStack() as stack:
        if isinstance(source, bytes):
            raw = stack.enter_context(io.BytesIO(source))
        elif isinstance(source, str) and source.lstrip().startswith("{"):
            raw = stack.enter_context(io.BytesIO(source.encode("utf-8")))
        elif isinstance(source, (str, os.PathLike)):
            display = str(source)
            try:
                raw = stack.enter_context(Path(source).open("rb"))
            except OSError as exc:
                raise PlanArchiveError(f"unable to read plan archive {source}: {exc}") from exc
        else:
            raise TypeError("archive source must be a path, UTF-8 JSON text, or bytes")
        compressed = False
        try:
            compressed = raw.read(2) == b"\x1f\x8b"
            raw.seek(0)
            stream = (stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                      if compressed else raw)
            # json.load still owns one complete text string while parsing;
            # avoid also retaining the compressed and decompressed byte copies.
            text = stack.enter_context(io.TextIOWrapper(
                stream, encoding="utf-8-sig", newline=""))
            return json.load(text, parse_constant=_reject_constant)
        except (OSError, EOFError) as exc:
            message = "invalid compressed plan archive" if compressed else "unable to read plan archive"
            raise PlanArchiveError(f"{message} {display}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise PlanArchiveError(f"invalid plan archive JSON: {exc}") from exc


def _write_document(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    """Publish a JSON document atomically; ``.gz`` uses deterministic gzip."""
    destination = Path(path)
    encoded = _json_bytes(document)
    if destination.suffix.lower() == ".gz":
        encoded = gzip.compress(encoded, mtime=0)
    with _atomic_archive_file(destination) as handle:
        handle.write(encoded)
    return destination


def save_cfg_plan(path,plan,*,debug=False):
    from .cfg_archive import save_cfg_plan as save
    return save(path,plan,debug=debug)


def load_cfg_plan(source):
    from .cfg_archive import load_cfg_plan as load
    return load(source)


def migrate_plan_bundle(source, destination):
    """Explicitly convert a v4 compiled DAG to integer v5 without searching."""
    from .migration import migrate_plan_bundle as migrate
    return migrate(source, destination)


def migrate_plan_archive(source, destination):
    """Explicitly convert retired axes or a v8 storage dataset field offline."""
    from .migration import migrate_plan_archive as migrate
    return migrate(source, destination)


def save_plan(bundle: PlanBundle, path: str | os.PathLike[str]) -> Path:
    """Atomically write an integer PlanBundle in the current archive envelope."""

    if not isinstance(bundle, PlanBundle):
        raise TypeError("save_plan expects a PlanBundle")
    _validate_bundle(bundle)
    return _write_document(path, _make_envelope(_PAYLOAD_PLAN_BUNDLE, _archive_payload(bundle)))


def _decode_plan_bundle_payload(
    payload: Mapping[str, Any],
    *,
    expected_signature: str | None,
    expected_workload: WorkloadSpec | None,
) -> PlanBundle:
    _strict_keys(
        payload,
        {"signature", "workload", "batches", "budget", "metadata"},
        name="archive payload",
    )
    if not isinstance(payload["signature"], str):
        raise PlanArchiveError("archive signature must be a string")

    workload = _decode_workload(payload["workload"])
    batches_raw = payload["batches"]
    if not isinstance(batches_raw, list):
        raise PlanArchiveError("archive.batches must be a JSON list")
    batches: list[PlanBatch] = []
    for index, raw_batch in enumerate(batches_raw):
        batch = _strict_object(raw_batch, name=f"archive.batches[{index}]")
        _strict_keys(
            batch,
            {"term_indices", "memory_estimate", "plan"},
            name=f"archive.batches[{index}]",
        )
        indices = batch["term_indices"]
        if not isinstance(indices, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in indices
        ):
            raise PlanArchiveError(f"archive.batches[{index}].term_indices must be integer list")
        memory = _strict_object(
            batch["memory_estimate"], name=f"archive.batches[{index}].memory_estimate"
        )
        plan = _decode_plan(batch["plan"], name=f"archive.batches[{index}].plan")
        try:
            batches.append(
                PlanBatch(plan=plan, term_indices=tuple(indices), memory_estimate=dict(memory))
            )
        except (TypeError, ValueError) as exc:
            raise PlanArchiveError(f"invalid archive batch {index}: {exc}") from exc

    metadata = _strict_object(payload["metadata"], name="archive.metadata")
    budget = _decode_budget(payload["budget"])
    try:
        bundle = PlanBundle(
            workload=workload,
            batches=tuple(batches),
            budget=budget,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise PlanArchiveError(f"invalid plan bundle: {exc}") from exc
    _validate_bundle(bundle)
    actual_signature = structural_signature(bundle)
    if not hmac.compare_digest(actual_signature, payload["signature"]):
        raise PlanArchiveError("plan archive structural signature mismatch")
    if expected_signature is not None:
        if not isinstance(expected_signature, str):
            raise TypeError("expected_signature must be a string or None")
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise PlanArchiveError("loaded plan does not match expected structural signature")
    if expected_workload is not None:
        if not isinstance(expected_workload, WorkloadSpec):
            raise TypeError("expected_workload must be a WorkloadSpec or None")
        if _digest(_encode_workload(expected_workload)) != _digest(_encode_workload(bundle.workload)):
            raise PlanArchiveError("loaded plan workload does not match expected workload")
    return bundle


def load_plan(
    path: str | os.PathLike[str] | bytes,
    *,
    expected_signature: str | None = None,
    expected_workload: WorkloadSpec | None = None,
) -> PlanBundle:
    """Load and validate a plan archive without invoking a planner."""

    document = _read_document(path)
    obj = _strict_object(document, name="archive")
    payload = _decode_envelope(obj, expected_type=_PAYLOAD_PLAN_BUNDLE)
    return _decode_plan_bundle_payload(
        payload,
        expected_signature=expected_signature,
        expected_workload=expected_workload,
    )


def _plan_value_types() -> tuple[type, ...]:
    """Return the cross-module public whitelist for fixed path payloads."""

    from lqcdweave.contraction.planning.api import ContractionState
    from lqcdweave.contraction.performance.api import CostProfile

    names = (
        "AxisPattern", "TensorSpec", "TensorDescription", "ContractionTerm", "SourceSpec", "Action", "Factor",
        "ResourceTransition",
        "PendingTerm", "ContractionProblem", "SparseBridgeSpec", "BaryonJoinSpec",
        "CoreComponent", "TensorCoreSpec", "MesonWholeCore", "MesonWholeKernel",
        "VertexKernelDescription",
        "StorageBlock", "SourcePreparation", "StorageEvent", "MemoryPoint",
        "ArrayConstant", "TensorPreparation", "SourceReadRequest",
        "CfgPlan", "CfgTemplate", "CfgInstance", "CfgOutputField", "MatrixKernelSpec", "FactorizedWideReduction", "ExecutionTarget",
        "DenseBatchOperand", "DenseBatchResult", "DenseKernelBatch", "DenseGemmGroup",
        "ScalarReductionGroup", "CartesianReductionGroup",
        "SceneStoragePlan", "CoordinatePoint", "CoordinateSpec", "BatchSpec",
        "CoordinateBatch", "TermContribution", "OutputNode", "OutputRecipe",
        "WorkloadSpec", "PlanBatch", "PlanBundle",
        "ExecutionBudget", "DiagramBindings", "TensorAxisMetadata",
        "PhysicalSlabBinding", "BatchTensorBinding",
    )
    records = [getattr(contract_api, name) for name in names]
    from lqcdweave.contracts.elemental import ElementalComponent, BaryonComponent, GaugePath
    from lqcdweave.contracts.memory import TargetHardwareMemory
    records.extend((ElementalComponent, BaryonComponent, GaugePath, TargetHardwareMemory))
    records.extend((ContractionState, CostProfile))
    return tuple(records)


@lru_cache(maxsize=1)
def _plan_type_maps() -> tuple[dict[type, str], dict[str, type], dict[str, type]]:
    classes = _plan_value_types()
    by_type: dict[type, str] = {}
    by_name: dict[str, type] = {}
    legacy_by_name: dict[str, type] = {}
    for cls in classes:
        type_id = f"{cls.__module__}.{cls.__qualname__}"
        if type_id in by_name or cls.__name__ in legacy_by_name:
            raise RuntimeError(f"plan archive whitelist type name collision: {cls.__name__}")
        by_type[cls] = type_id
        by_name[type_id] = cls
        legacy_by_name[cls.__name__] = cls
    return by_type, by_name, legacy_by_name


@lru_cache(maxsize=128)
def _plan_record_schema(cls):
    """Bind only immutable field metadata for whitelisted contract classes."""
    record_fields = fields(cls)
    names = tuple(item.name for item in record_fields)
    required = frozenset(item.name for item in record_fields
                         if item.default is MISSING and item.default_factory is MISSING)
    compatible = set()
    if cls is contract_api.CfgTemplate:
        compatible.update(("preparation_closure", "preparation_specs"))
    if cls is contract_api.SourcePreparation:
        compatible.add("recipe")
    if cls is contract_api.SourceReadRequest:
        compatible.update(("coordinate_arguments", "coordinate_tables"))
    if cls is contract_api.TensorAxisMetadata:
        compatible.update(("point_diagonal", "structure"))
    if cls is contract_api.TensorPreparation:
        compatible.update(("spatial_structure", "compute_dtype", "raw_index", "spin_columns", "index_permutations", "spatial_only", "coordinate_constants", "spatial_support"))
    if cls is contract_api.OutputRecipe:
        compatible.update(("dtype", "intermediate_dtype"))
    if cls is contract_api.ContractionTerm:
        compatible.add('sum_barriers')
        compatible.add('kernel')
    if cls is contract_api.ContractionProblem:
        compatible.add('meson_cores')
    if cls is contract_api.FactorizedWideReduction:
        compatible.add('tile_elements')
        compatible.add('output_dtype')
        compatible.add('gram_strategy')
    if cls is contract_api.Action:
        compatible.update(("resource", "wide_accumulation"))
    if cls is contract_api.MemoryPoint:
        # Frozen v1/v2 routes predate these independently added zero charges.
        # Keep this migration explicit; other optional fields stay required.
        compatible.update(("temporary_disk_bytes", "packed_value_cache_device_bytes",
                           "allocation_rounding_device_bytes", "scalar_reduction_device_bytes",
                           "cartesian_reduction_device_bytes"))
    if cls is contract_api.SceneStoragePlan:
        compatible.update(("peak_disk_bytes", "disk_limit_bytes", "explicit_resources", "kernel_batches",
                           "packed_value_cache_device_bytes", "scalar_reduction_groups", "cartesian_reduction_groups"))
    if cls is contract_api.DenseKernelBatch:
        compatible.update(("packing_dtype", "prepared_sides", "stacked_sides"))
    if cls is contract_api.DenseBatchResult:
        compatible.add("squeeze_axes")
    if cls in (contract_api.Action, contract_api.SparseBridgeSpec):
        compatible.add("used_np_right")
    if cls in (contract_api.Action, contract_api.BaryonJoinSpec):
        compatible.update(("sparse_spin", "core"))
        compatible.add('kernel_description')
    if cls is contract_api.BaryonJoinSpec:
        compatible.add("batch_label")
    integer_fields = {
        contract_api.Factor: ("labels",),
        contract_api.PendingTerm: ("output",),
        contract_api.BaryonJoinSpec: ("spin_labels", "mode_labels"),
    }
    return names, frozenset(names), required, frozenset(compatible), integer_fields.get(cls, ())


_PLAN_CONTAINERS = {
    "python.tuple": tuple, "python.list": list,
    "python.set": set, "python.frozenset": frozenset,
    # Names used by the pre-migration snapshot codec.
    "tuple": tuple, "list": list, "set": set, "frozenset": frozenset,
}
_PLAN_SCALAR_TYPES = (str, bool, int, float)
_PLAN_LITERAL_TYPES = (str, bool, int)
_PLAN_TAG_FIELDS = frozenset(("type", "value"))
_PLAN_SHARED_FIELDS = frozenset(("id", "record"))


def _encode_plan_value(value: Any, *, depth: int = 0, shared=None) -> Any:
    if depth > 128:
        raise PlanArchiveError("plan value nesting exceeds the supported limit")
    value_type = type(value)
    if value is None or value_type in _PLAN_LITERAL_TYPES:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise PlanArchiveError("plan values cannot contain non-finite floats")
        return value
    if isinstance(value, np.dtype):
        return {"type": "numpy.dtype", "value": value.str}
    if isinstance(value, np.generic):
        dtype = np.dtype(value.dtype)
        if dtype.kind not in "biufc":
            raise PlanArchiveError(f"unsupported NumPy scalar dtype {dtype}")
        raw = np.asarray(value, dtype=dtype).reshape(()).tobytes().hex()
        return {"type": "numpy.scalar", "value": {"dtype": dtype.str, "hex": raw}}
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise PlanArchiveError("plan values cannot contain non-finite complex numbers")
        return {"type": "python.complex", "value": [value.real, value.imag]}

    by_type, _by_name, _legacy_by_name = _plan_type_maps()
    # Membership already establishes this fixed dataclass whitelist; do not
    # rediscover dataclass metadata for every record and container.
    if value_type in by_type:
        share=shared is not None and type(value).__dataclass_params__.frozen
        if share and id(value) in shared:
            identity,complete=shared[id(value)]
            if not complete: raise PlanArchiveError('cyclic shared plan record')
            return {'type':'archive.reference','value':identity}
        if share:
            identity=len(shared);shared[id(value)]=(identity,False)
        record={
            "type": by_type[type(value)],
            "value": {
                name: _encode_plan_value(getattr(value, name), depth=depth + 1,shared=shared)
                for name in _plan_record_schema(type(value))[0]
            },
        }
        if share:
            shared[id(value)]=(identity,True)
            return {'type':'archive.shared','value':{'id':identity,'record':record}}
        return record
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "value": [
                [
                    _encode_plan_value(key, depth=depth + 1,shared=shared),
                    _encode_plan_value(item, depth=depth + 1,shared=shared),
                ]
                for key, item in value.items()
            ],
        }
    if type(value) in (tuple, list):
        return {
            "type": f"python.{type(value).__name__}",
            "value": [_encode_plan_value(item, depth=depth + 1,shared=shared) for item in value],
        }
    if type(value) in (set, frozenset):
        # Set ordering must remain independent of object reference traversal.
        items = [_encode_plan_value(item, depth=depth + 1) for item in value]
        items.sort(key=_json_bytes)
        return {"type": f"python.{type(value).__name__}", "value": items}
    raise PlanArchiveError(f"unsupported plan value type {type(value).__module__}.{type(value).__qualname__}")


def _decode_plan_value(value: Any, *, legacy: bool = False, depth: int = 0,shared=None,
                       _consume=False) -> Any:
    if depth > 128:
        raise PlanArchiveError("plan value nesting exceeds the supported limit")
    value_type = type(value)
    if value is None or value_type in _PLAN_SCALAR_TYPES:
        if value_type is float and not math.isfinite(value):
            raise PlanArchiveError("plan archive contains a non-finite float")
        return value
    if ((value_type is not dict and not isinstance(value, Mapping))
            or set(value) != _PLAN_TAG_FIELDS):
        raise PlanArchiveError("invalid encoded plan value")
    kind, payload = value["type"], value["value"]
    if not isinstance(kind, str):
        raise PlanArchiveError("encoded plan type must be a string")
    if kind in ('archive.compact-v1','archive.compact-v2'):
        from .compact import decode
        return decode(payload, depth=depth,allow_defaults=kind=='archive.compact-v2',
                      _consume=_consume)
    if kind=='archive.reference':
        if (shared is None or type(payload) is not int or payload not in shared
                or shared[payload] is None):
            raise PlanArchiveError('invalid or forward shared plan reference')
        return shared[payload]
    if kind=='archive.shared':
        if (shared is None or (type(payload) is not dict and not isinstance(payload,Mapping))
                or set(payload)!=_PLAN_SHARED_FIELDS
                or type(payload['id']) is not int or payload['id']!=len(shared)):
            raise PlanArchiveError('invalid shared plan record')
        definition=payload['record']
        if ((type(definition) is not dict and not isinstance(definition,Mapping))
                or set(definition)!=_PLAN_TAG_FIELDS):
            raise PlanArchiveError('invalid shared contract definition')
        _,types,legacy_types=_plan_type_maps()
        cls=types.get(definition['type']) if isinstance(definition['type'],str) else None
        if cls is None and legacy and isinstance(definition['type'],str):
            cls=legacy_types.get(definition['type'])
        if cls is None or not cls.__dataclass_params__.frozen:
            raise PlanArchiveError('shared plan definitions must be immutable contract records')
        identity=payload['id'];shared[identity]=None
        record=_decode_plan_value(definition,legacy=legacy,depth=depth,shared=shared)
        # The definition's frozen whitelist class was checked before decoding;
        # the constructor above returns that same contract type.
        shared[identity]=record
        return record
    if kind in ("numpy.dtype", "dtype"):
        if not isinstance(payload, str):
            raise PlanArchiveError("encoded dtype must contain text")
        try:
            return np.dtype(payload)
        except (TypeError, ValueError) as exc:
            raise PlanArchiveError(f"invalid dtype in plan archive: {payload!r}") from exc
    if kind == "numpy.scalar":
        obj = _strict_object(payload, name="numpy scalar")
        _strict_keys(obj, {"dtype", "hex"}, name="numpy scalar")
        dtype = _dtype_from_text(obj["dtype"], name="numpy scalar")
        if dtype.kind not in "biufc" or not isinstance(obj["hex"], str):
            raise PlanArchiveError("invalid NumPy scalar payload")
        try:
            raw = bytes.fromhex(obj["hex"])
            if len(raw) != dtype.itemsize:
                raise ValueError("wrong byte count")
            return np.frombuffer(raw, dtype=dtype, count=1)[0]
        except (TypeError, ValueError) as exc:
            raise PlanArchiveError("invalid NumPy scalar payload") from exc
    if kind in ("python.complex", "complex"):
        if not isinstance(payload, list) or len(payload) != 2:
            raise PlanArchiveError("invalid complex plan value")
        try:
            result = complex(float(payload[0]), float(payload[1]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlanArchiveError("invalid complex plan value") from exc
        if not np.isfinite(result.real) or not np.isfinite(result.imag):
            raise PlanArchiveError("plan archive contains a non-finite complex value")
        return result
    if kind == "mapping":
        if not isinstance(payload, list):
            raise PlanArchiveError("encoded mapping must contain key/value pairs")
        result = {}
        for index, pair in enumerate(payload):
            if not isinstance(pair, list) or len(pair) != 2:
                raise PlanArchiveError(f"encoded mapping pair {index} is malformed")
            key = pair[0]
            if depth >= 128 or (key is not None and type(key) not in _PLAN_LITERAL_TYPES):
                key = _decode_plan_value(key, legacy=legacy, depth=depth + 1,shared=shared)
            item = pair[1]
            if depth >= 128 or (item is not None and type(item) not in _PLAN_LITERAL_TYPES):
                item = _decode_plan_value(item, legacy=legacy, depth=depth + 1,shared=shared)
            try:
                if key in result:
                    raise PlanArchiveError("encoded mapping contains duplicate keys")
                result[key] = item
            except TypeError as exc:
                raise PlanArchiveError("encoded mapping key is not hashable") from exc
        return result
    if kind in _PLAN_CONTAINERS:
        if not isinstance(payload, list):
            raise PlanArchiveError(f"encoded {kind} must contain a list")
        if depth < 128:
            items = [item if item is None or type(item) in _PLAN_LITERAL_TYPES
                     else _decode_plan_value(item, legacy=legacy, depth=depth + 1,shared=shared)
                     for item in payload]
        else:
            # Even literal children must cross the original depth boundary.
            items = [_decode_plan_value(item, legacy=legacy, depth=depth + 1,shared=shared)
                     for item in payload]
        try:
            return _PLAN_CONTAINERS[kind](items)
        except TypeError as exc:
            raise PlanArchiveError(f"invalid encoded {kind}") from exc

    by_type, by_name, legacy_by_name = _plan_type_maps()
    del by_type
    cls = by_name.get(kind)
    used_legacy_name = False
    if cls is None and legacy:
        cls = legacy_by_name.get(kind)
        used_legacy_name = cls is not None
    if cls is None:
        raise PlanArchiveError(f"unknown plan archive type {kind!r}")
    names, supported, required, compatible_defaults, integer_fields = _plan_record_schema(cls)
    if type(payload) is not dict and not isinstance(payload, Mapping):
        raise PlanArchiveError(f"{kind} fields must be a JSON object")
    actual = set(payload)
    if actual != supported:
        missing_compatible_defaults = (
            actual <= supported and supported - actual <= compatible_defaults
        )
        if not (
            missing_compatible_defaults
            or (used_legacy_name and required <= actual <= supported)
        ):
            raise PlanArchiveError(f"{kind} fields do not match the supported schema")
    decoded = {}
    for name in names:
        if name not in payload:
            continue
        item = payload[name]
        decoded[name] = (item if depth < 128 and (item is None or type(item) in _PLAN_LITERAL_TYPES)
                         else _decode_plan_value(item, legacy=legacy, depth=depth + 1,shared=shared))
    return _construct_plan_record(cls, decoded, kind, integer_fields)


@lru_cache(maxsize=128)
def _plan_record_validator(cls, integer_fields):
    """Bind archive-only guards once; ordinary constructors remain mandatory."""
    term = cls is contract_api.ContractionTerm
    action = cls is contract_api.Action
    if not integer_fields and not term and not action:
        return None

    def validate(decoded, kind):
        for name in integer_fields:
            if not isinstance(decoded.get(name), tuple) or any(type(axis) is not int or axis < 0 for axis in decoded[name]):
                raise PlanArchiveError(f"{kind}.{name} must contain integer axes")
        if term and not isinstance(decoded.get("subscripts"), contract_api.AxisPattern):
            raise PlanArchiveError("archived ContractionTerm requires integer AxisPattern")
        if action and decoded.get("equation") is not None and not isinstance(decoded.get("equation"), contract_api.AxisPattern):
            raise PlanArchiveError("archived Action requires integer AxisPattern or None")
    return validate


def _bind_positional_record_constructor(cls, names, kind, integer_fields):
    """Bind only the exact positional prefix/keyword tail of a known schema."""
    try:
        parameters = tuple(signature(cls).parameters.values())
    except (TypeError, ValueError):
        return None
    if integer_fields and cls in (contract_api.Action, contract_api.ContractionTerm):
        return None
    count = len(names)
    if (tuple(parameter.name for parameter in parameters[:count]) != names
            or any(parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                   for parameter in parameters[:count])
            or any(parameter.default is Parameter.empty for parameter in parameters[count:])):
        return None
    positional = next((index for index, parameter in enumerate(parameters[:count])
                       if parameter.kind is Parameter.KEYWORD_ONLY), count)
    keyword_names = names[positional:]
    validate = None
    if cls is contract_api.ContractionTerm:
        pattern_index = names.index('subscripts')
        def validate(values):
            if not isinstance(values[pattern_index], contract_api.AxisPattern):
                raise PlanArchiveError('archived ContractionTerm requires integer AxisPattern')
    elif cls is contract_api.Action:
        pattern_index = names.index('equation')
        def validate(values):
            value = values[pattern_index]
            if value is not None and not isinstance(value, contract_api.AxisPattern):
                raise PlanArchiveError('archived Action requires integer AxisPattern or None')
    elif integer_fields:
        axes = tuple((name, names.index(name)) for name in integer_fields)
        def validate(values):
            for name, index in axes:
                value = values[index]
                if not isinstance(value, tuple) or any(type(axis) is not int or axis < 0 for axis in value):
                    raise PlanArchiveError(f'{kind}.{name} must contain integer axes')

    if validate is None and not keyword_names:
        def construct(values):
            try:
                return cls(*values)
            except (TypeError, ValueError, KeyError) as exc:
                raise PlanArchiveError(f'invalid {kind} plan value: {exc}') from exc
        return construct

    def construct(values):
        if validate is not None:
            validate(values)
        try:
            if keyword_names:
                return cls(*values[:positional], **dict(zip(keyword_names, values[positional:])))
            return cls(*values)
        except (TypeError, ValueError, KeyError) as exc:
            raise PlanArchiveError(f'invalid {kind} plan value: {exc}') from exc
    return construct


def _construct_plan_record(cls, decoded, kind, integer_fields):
    # Constructors accept external strings for user convenience. Archives
    # must not use that boundary to silently upgrade retired string records.
    validate = _plan_record_validator(cls, integer_fields)
    if validate is not None:
        validate(decoded, kind)
    try:
        return cls(**decoded)
    except (TypeError, ValueError, KeyError) as exc:
        raise PlanArchiveError(f"invalid {kind} plan value: {exc}") from exc


def _encode_archive_value(value):
    from .compact import encode
    return encode(value)


def encode_plan_value(value: Any) -> Any:
    """Encode an explicitly whitelisted pure contract or planning record."""

    return _encode_plan_value(value)


def decode_plan_value(value: Any) -> Any:
    """Decode an explicitly whitelisted record without importing archive names."""

    return _decode_plan_value(value, legacy=True)


def _strict_json_value(value: Any, *, name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return {key: _strict_json_value(item, name=name) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json_value(item, name=name) for item in value]
    raise TypeError(f"{name} must contain only finite strict-JSON values")


def _problem_payload(problem: Any) -> dict[str, Any]:
    """Describe semantic contraction inputs without backend or array values."""

    if not isinstance(problem, contract_api.ContractionProblem):
        raise TypeError("expected input must contain a ContractionProblem")
    return {
        "backend": problem.backend,
        "specs": [
            {
                "key": key,
                "shape": list(spec.shape),
                "dtype": spec.dtype.str,
                "description": None if spec.description is None else _encode_plan_value(spec.description),
            }
            for key, spec in problem.specs.items()
        ],
        "terms": [
            {
                "inputs": list(term.inputs),
                "subscripts": term.subscripts.as_dict(),
                "coefficient": contract_api.encode_scalar(term.coefficient),
            }
            for term in problem.terms
        ],
        "sources": [
            {
                "key": key,
                # Keep every caller-supplied identity and selection field.  The
                # physical dataset path is carried separately by StorageBlock.
                "metadata": contract_api.thaw_json(source.metadata),
                "snapshot_bytes": (
                    source.snapshot_bytes if key in problem.specs
                    else source.snapshot_bytes or source.region_bytes
                ),
            }
            for key, source in problem.sources.items()
        ],
        "sparse_bridges": [_encode_plan_value(item) for item in problem.sparse_bridges],
        "baryon_joins": [_encode_plan_value(item) for item in problem.baryon_joins],
    }


def _description_invariants(description: Any) -> tuple[Any, ...] | None:
    if description is None:
        return None
    if not isinstance(description, contract_api.TensorDescription):
        raise PlanArchiveError("TensorSpec description is not a public TensorDescription")
    return (
        description.axes,
        description.role,
        _encode_plan_value(description.transform),
        _encode_plan_value(description.semantics),
        description.data_identity is None,
        description.selection_identity is None,
    )


def _source_binding_topology(specs: Sequence[tuple[str, Any]]) -> tuple[Any, ...]:
    labels: dict[str, int] = {}
    next_label = 0
    result = []
    for key, description in specs:
        if description is None:
            result.append((key, None))
            continue
        local = []
        for binding in description.source_bindings:
            if binding not in labels:
                labels[binding] = next_label
                next_label += 1
            local.append(labels[binding])
        result.append((key, tuple(local)))
    return tuple(result)


def _problem_path_invariants(problem: Any) -> dict[str, Any]:
    if not isinstance(problem, contract_api.ContractionProblem):
        raise TypeError("problem must be a ContractionProblem")
    descriptions = []
    for key, spec in problem.specs.items():
        descriptions.append((key, spec.description))
    terms, joins = contract_api.normalized_scene_relations(problem.terms, problem.baryon_joins)
    return {
        "backend": problem.backend,
        "specs": [
            {
                "key": key,
                "shape": list(spec.shape),
                "dtype": spec.dtype.str,
                "description_invariants": _description_invariants(spec.description),
            }
            for key, spec in problem.specs.items()
        ],
        "source_binding_topology": _source_binding_topology(descriptions),
        "terms": [
            {
                "inputs": list(term.inputs),
                "subscripts": term.subscripts.as_dict(),
                "coefficient": contract_api.encode_scalar(term.coefficient),
            }
            for term in terms
        ],
        "source_keys_and_snapshots": [
            (key, source.snapshot_bytes) for key, source in problem.sources.items()
        ],
        "sparse_bridges": [_encode_plan_value(item) for item in problem.sparse_bridges],
        "baryon_joins": [_encode_plan_value(item) for item in joins],
    }


def _problem_payload_compatible(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allow_identity_rebind: bool,
) -> tuple[bool, bool]:
    """Check invariant physics/task structure and explicit source rebinding."""

    if _digest(_problem_path_invariants_from_payload(saved)) != _digest(
        _problem_path_invariants_from_payload(current)
    ):
        return False, False
    saved_sources = {item["key"]: item for item in saved.get("sources", ())}
    current_sources = {item["key"]: item for item in current.get("sources", ())}
    saved_specs = {item["key"]: item for item in saved.get("specs", ())}
    current_specs = {item["key"]: item for item in current.get("specs", ())}
    if set(saved_sources) != set(current_sources) or set(saved_specs) != set(current_specs):
        return False, False
    identity_changed: set[str] = set()
    for key in saved_specs:
        old_raw, new_raw = saved_specs[key].get("description"), current_specs[key].get("description")
        if old_raw is None or new_raw is None:
            if old_raw != new_raw:
                return False, False
            continue
        try:
            old = _decode_plan_value(old_raw)
            new = _decode_plan_value(new_raw)
        except PlanArchiveError:
            return False, False
        if not isinstance(old, contract_api.TensorDescription) or not isinstance(new, contract_api.TensorDescription):
            return False, False
        if (old.data_identity, old.selection_identity, dict(old.bindings), old.source_bindings) != (
            new.data_identity, new.selection_identity, dict(new.bindings), new.source_bindings
        ):
            identity_changed.add(key)
    if identity_changed and not allow_identity_rebind:
        return False, False
    for key, old_source in saved_sources.items():
        new_source = current_sources[key]
        if old_source.get("snapshot_bytes") != new_source.get("snapshot_bytes"):
            return False, False
        if old_source.get("metadata") != new_source.get("metadata"):
            # SourceSpec metadata is preserved in the file. It may change only
            # when the current expanded task explicitly supplies a TensorDescription
            # whose identity/binding fields changed while its invariants matched.
            if not allow_identity_rebind or key not in identity_changed:
                return False, False
    return True, bool(identity_changed)


def _assert_saved_sum_barriers(saved, current):
    """Compare the authoritative C records, including old archive defaults.

    Historical semantic summary/hash records omitted sum_barriers. Preserve
    that wire format, but never let an explicit expected task silently replace
    the saved precision boundary. Missing fields in genuine old C records
    decode as the empty contract and are compatible only with the same one.
    """
    if not (saved._has_sum_barriers or current._has_sum_barriers):
        return
    old_terms, _ = contract_api.normalized_scene_relations(saved.terms)
    new_terms, _ = contract_api.normalized_scene_relations(current.terms)
    if tuple(term.sum_barriers for term in old_terms) != tuple(term.sum_barriers for term in new_terms):
        raise PlanArchiveError('sum barriers differ from the saved precision contract; create a new plan')


def _assert_saved_term_kernels(saved, current):
    """Check the authoritative records omitted by historical hash summaries."""
    if len(saved.terms) != len(current.terms):
        raise PlanArchiveError('term kernels differ from the saved plan; create a new plan')
    checked = set()
    for old, new in zip(saved.terms, current.terms, strict=True):
        previous, actual = old.kernel, new.kernel
        if previous is actual:
            continue
        pair = id(previous), id(actual)
        if pair in checked:
            continue
        if previous != actual:
            raise PlanArchiveError('term kernels differ from the saved plan; create a new plan')
        # Both term tuples keep these objects alive throughout the comparison.
        if len(checked) < 128:
            checked.add(pair)


def _problem_path_invariants_from_payload(problem: Mapping[str, Any], *, normalize_axes=True) -> dict[str, Any]:
    specs = []
    descriptions = []
    for item in problem.get("specs", ()):
        raw = item.get("description")
        desc = None if raw is None else _decode_plan_value(raw)
        if desc is not None and not isinstance(desc, contract_api.TensorDescription):
            raise PlanArchiveError("saved tensor description has an invalid type")
        key = item["key"]
        descriptions.append((key, desc))
        specs.append({
            "key": key,
            "shape": item["shape"],
            "dtype": item["dtype"],
            "description_invariants": _description_invariants(desc),
        })
    sparse_bridges = []
    for raw in problem.get("sparse_bridges", ()):
        bridge = _decode_plan_value(raw, legacy=True)
        if not isinstance(bridge, contract_api.SparseBridgeSpec):
            raise PlanArchiveError("saved sparse bridge has an invalid type")
        sparse_bridges.append(_encode_plan_value(bridge))
    terms, joins = problem.get("terms"), problem.get("baryon_joins")
    if normalize_axes:
        decoded_terms = tuple(ContractionTerm(tuple(term["inputs"]),
            _decode_axis_pattern(term["subscripts"], name="saved scene equation"),
            _decode_scalar(term["coefficient"])) for term in terms)
        normalized, normalized_joins = contract_api.normalized_scene_relations(decoded_terms,
            tuple(_decode_plan_value(join, legacy=True) for join in joins))
        terms = [{"inputs": list(term.inputs), "subscripts": term.subscripts.as_dict(),
                  "coefficient": contract_api.encode_scalar(term.coefficient)} for term in normalized]
        joins = [_encode_plan_value(join) for join in normalized_joins]
    return {
        "backend": problem.get("backend"),
        "specs": specs,
        "source_binding_topology": _source_binding_topology(descriptions),
        "terms": terms,
        "source_keys_and_snapshots": [
            (item["key"], item.get("snapshot_bytes")) for item in problem.get("sources", ())
        ],
        "sparse_bridges": sparse_bridges,
        "baryon_joins": joins,
    }


def _expected_context(expected: Any) -> tuple[tuple[Any, ...], Any | None]:
    if isinstance(expected, contract_api.ExpandedCorrelator):
        if expected.problem is None or expected.inputs is None:
            raise PlanArchiveError("a saved contraction path requires a populated ExpandedCorrelator")
        inputs = expected.inputs
        problems = tuple(item.problem for item in inputs.task_batches) if inputs.task_batches else (expected.problem,)
        if any(not isinstance(problem, contract_api.ContractionProblem) for problem in problems):
            raise PlanArchiveError("expanded task inputs must include physical contraction problems")
        return problems, expected
    if isinstance(expected, contract_api.ContractionProblem):
        return (expected,), None
    if isinstance(expected, (tuple, list)) and expected and all(
        isinstance(item, contract_api.ContractionProblem) for item in expected
    ):
        return tuple(expected), None
    raise TypeError(
        "expected must be a ContractionProblem, ExpandedCorrelator, or sequence of ContractionProblems"
    )


def _problem_shapes(problems: Sequence[Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for problem in problems:
        for key, spec in problem.specs.items():
            shape = list(spec.shape)
            if key in shapes and shapes[key] != shape:
                raise PlanArchiveError(f"input {key!r} has inconsistent shapes across batches")
            shapes[key] = shape
    return shapes


def _normalize_shapes(shapes: Any, *, name: str = "shapes") -> dict[str, list[int]]:
    if not isinstance(shapes, Mapping):
        raise TypeError(f"{name} must be a mapping of input keys to shapes")
    result = {}
    for key, raw in shapes.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be nonempty strings")
        if isinstance(raw, Mapping):
            if set(raw) != {"shape"}:
                raise ValueError(f"{name}[{key!r}] objects may contain only a shape field")
            raw = raw["shape"]
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"{name}[{key!r}] must be an integer shape sequence")
        try:
            shape = tuple(raw)
        except TypeError as exc:
            raise TypeError(f"{name}[{key!r}] must be an integer shape sequence") from exc
        if any(isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < 0 for size in shape):
            raise ValueError(f"{name}[{key!r}] dimensions must be nonnegative integers")
        result[key] = [int(size) for size in shape]
    return result


def _hardware_payload(hardware: Any) -> Any:
    from lqcdweave.contracts.memory import TargetHardwareMemory

    if isinstance(hardware, TargetHardwareMemory):
        return {field.name: getattr(hardware, field.name) for field in fields(hardware)}
    if not isinstance(hardware, Mapping):
        raise TypeError("hardware must be a JSON mapping or TargetHardwareMemory")
    return _strict_json_value(hardware, name="hardware")


def _profile_payload(profile: Any) -> Any:
    from lqcdweave.contraction.performance.api import CostProfile

    if profile is None:
        return None
    if not isinstance(profile, CostProfile):
        raise TypeError("profile must be a CostProfile or None")
    return _encode_plan_value(profile)


def _profile_source_payload(source: Any) -> Any:
    if source is None:
        return None
    return _strict_json_value(source, name="profile_source")


def _assert_expected_semantics(
    saved_semantics: Any,
    expected: Any,
    *,
    saved_shapes: Mapping[str, list[int]],
    supplied_shapes: Mapping[str, list[int]] | None,
    batch_count: int,
    allow_identity_rebind: bool | None = None,
) -> tuple[tuple[Any, ...], Any | None]:
    problems, expanded = _expected_context(expected)
    if len(problems) != batch_count:
        raise PlanArchiveError(
            f"expected task has {len(problems)} problem batches; archive has {batch_count}"
        )
    if not isinstance(saved_semantics, list) or len(saved_semantics) != len(problems):
        raise PlanArchiveError("saved input semantic descriptors are incomplete")
    allow_rebind = expanded is not None if allow_identity_rebind is None else allow_identity_rebind
    for index, (saved, problem) in enumerate(zip(saved_semantics, problems, strict=True)):
        compatible, _identity_changed = _problem_payload_compatible(
            _strict_object(saved, name=f"saved semantics[{index}]"),
            _problem_payload(problem),
            allow_identity_rebind=allow_rebind,
        )
        if not compatible:
            raise PlanArchiveError(
                "input semantic identity or contraction topology differs from the saved plan"
            )
    current_shapes = _problem_shapes(problems)
    if current_shapes != dict(saved_shapes):
        raise PlanArchiveError("input shapes differ from saved plan")
    if supplied_shapes is not None and dict(supplied_shapes) != current_shapes:
        raise PlanArchiveError("supplied input shapes differ from the current task")
    return problems, expanded


def _validate_hardware_payload(value: Any) -> Any:
    payload = _hardware_payload(value)
    if payload is None or payload == {}:
        raise ValueError("hardware must declare a target identity or memory capacity")
    if isinstance(payload, Mapping) and payload and all(item is None for item in payload.values()):
        raise ValueError("hardware must declare at least one non-unknown target fact")
    return payload


def _schedule_shapes(plans: Sequence[Any]) -> dict[str, list[int]]:
    problems = []
    for index, plan in enumerate(plans):
        if not isinstance(plan, contract_api.SceneStoragePlan):
            raise TypeError(f"plans[{index}] must be a SceneStoragePlan")
        problems.append(plan.state.problem)
    return _problem_shapes(problems)


def _validate_schedule(plans: Sequence[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not plans:
        raise ValueError("at least one storage plan is required")
    if any(not isinstance(plan, contract_api.SceneStoragePlan) for plan in plans):
        raise TypeError("storage schedules require SceneStoragePlan values")
    has_temporary_move = any(
        action.kind in ("spill_disk", "restore_host")
        for plan in plans for action in plan.state.actions
    )
    if has_temporary_move and len(plans) != 1:
        raise NotImplementedError("temporary array resource archives support one non-batched path")
    opened: set[str] = set()
    resident: set[str] = set()
    blocks: dict[str, Any] = {}
    preparations: dict[str, Any] = {}
    for plan_index, plan in enumerate(plans):
        if not isinstance(plan, contract_api.SceneStoragePlan):
            raise TypeError(f"plans[{plan_index}] must be a SceneStoragePlan")
        validate_executable_resources(plan, allow_temporary_array=True)
        if not plan.feasible or not plan.state.done:
            raise ValueError("only complete feasible storage schedules can be saved")
        for key, block in plan.blocks.items():
            if not isinstance(key, str) or not isinstance(block, contract_api.StorageBlock):
                raise TypeError("storage plans require string keys and StorageBlock values")
            previous = blocks.get(key)
            if previous is not None and previous != block:
                raise ValueError(f"shared storage block {key!r} has inconsistent definitions")
            blocks[key] = block
        for key, preparation in plan.preparations.items():
            if not isinstance(key, str) or not isinstance(preparation, contract_api.SourcePreparation):
                raise TypeError("storage plans require string keys and SourcePreparation values")
            preparations.setdefault(key, preparation)
            if any(block_key not in plan.blocks for block_key in preparation.blocks):
                raise ValueError(f"source preparation {key!r} references an unknown block")
        for event in plan.events:
            if event.kind == "read":
                if event.block in opened or event.block not in plan.blocks:
                    raise ValueError("storage schedule contains a repeated or unknown block read")
                opened.add(event.block)
                resident.add(event.block)
            elif event.kind == "release":
                if event.block not in resident:
                    raise ValueError("storage schedule releases a block that is not resident")
                resident.remove(event.block)
            else:
                raise ValueError(f"unknown storage event {event.kind!r}")
    if resident:
        raise ValueError("storage schedule leaves physical blocks open")
    return blocks, preparations


def _storage_schedule_payload(
    plans: Sequence[Any],
    *,
    shapes: Mapping[str, list[int]],
    hardware: Any,
    metadata: Any,
    profile_source: Any,
) -> dict[str, Any]:
    blocks, _preparations = _validate_schedule(plans)
    plan_records = [
        _encode_archive_value(replace(plan, blocks={}))
        for plan in plans
    ]
    semantics = [_problem_payload(plan.state.problem) for plan in plans]
    payload = {
        "shapes": dict(shapes),
        "semantics": semantics,
        "hardware": hardware,
        "profile_source": profile_source,
        "blocks": _encode_archive_value(blocks),
        "plans": plan_records,
        "metadata": metadata,
    }
    return payload


def save_storage_schedule(
    path: str | os.PathLike[str],
    plans: Sequence[contract_api.SceneStoragePlan],
    *,
    shapes: Mapping[str, Any],
    hardware: Mapping[str, Any] | TargetHardwareMemory,
    metadata: Mapping[str, Any] | None = None,
    profile_source: Any = None,
) -> Path:
    """Save a fixed, already-audited storage schedule in the shared envelope.

    ``shapes`` must list every tensor input.  The archive also stores complete
    SourceSpec metadata and the contraction problem for each scheduled batch.
    """

    normalized_plans = tuple(plans)
    if not normalized_plans:
        raise ValueError("at least one storage plan is required")
    derived_shapes = _schedule_shapes(normalized_plans)
    normalized_shapes = _normalize_shapes(shapes)
    if normalized_shapes != derived_shapes:
        raise ValueError("shapes must exactly cover the storage plan tensor inputs")
    target = _validate_hardware_payload(hardware)
    if metadata is None:
        normalized_metadata = {}
    else:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON mapping or None")
        normalized_metadata = _strict_json_value(metadata, name="metadata")
    normalized_source = _profile_source_payload(profile_source)
    payload = _storage_schedule_payload(
        normalized_plans,
        shapes=normalized_shapes,
        hardware=target,
        metadata=normalized_metadata,
        profile_source=normalized_source,
    )
    return _write_document(
        path,
        _make_envelope(_PAYLOAD_STORAGE_SCHEDULE, payload),
    )


def _rebind_storage_plans(
    plans: Sequence[Any],
    *,
    blocks: Mapping[str, Any],
    preparations: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    plans = tuple(plans)
    if not isinstance(blocks, Mapping):
        raise TypeError("blocks must be a mapping")
    if isinstance(preparations, Mapping):
        preparation_maps = tuple(preparations for _ in plans)
    else:
        try:
            preparation_maps = tuple(preparations)
        except TypeError as exc:
            raise TypeError("preparations must be a mapping or one mapping per plan") from exc
        if len(preparation_maps) != len(plans) or any(not isinstance(item, Mapping) for item in preparation_maps):
            raise TypeError("preparations must provide one mapping per saved plan")
    if any(not isinstance(key, str) or not isinstance(value, contract_api.StorageBlock)
           for key, value in blocks.items()):
        raise TypeError("blocks must map string keys to StorageBlock values")
    if any(not isinstance(key, str) or not isinstance(value, contract_api.SourcePreparation)
           for preparation_map in preparation_maps for key, value in preparation_map.items()):
        raise TypeError("preparations must map string keys to SourcePreparation values")

    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for plan, current_preparations in zip(plans, preparation_maps, strict=True):
        if set(plan.preparations) - set(current_preparations):
            raise PlanArchiveError("current preparations do not cover every saved logical source")
        for key, old_preparation in plan.preparations.items():
            new_preparation = current_preparations[key]
            if len(old_preparation.blocks) != len(new_preparation.blocks):
                raise PlanArchiveError("input dependency shape changed; storage plan must be rebound")
            for before, after in zip(old_preparation.blocks, new_preparation.blocks, strict=True):
                if mapping.setdefault(before, after) != after or reverse.setdefault(after, before) != before:
                    raise PlanArchiveError("physical input sharing layout changed")
                if before not in plan.blocks or after not in blocks:
                    raise PlanArchiveError("storage preparation references an unknown physical block")
                old_block, new_block = plan.blocks[before], blocks[after]
                for field in fields(contract_api.StorageBlock):
                    if field.name not in {"dataset", "selection"} and getattr(old_block, field.name) != getattr(new_block, field.name):
                        raise PlanArchiveError(
                            f"physical block capacity or read contract changed at {key!r}; replan storage"
                        )
            if replace(new_preparation, blocks=old_preparation.blocks) != old_preparation:
                raise PlanArchiveError(
                    f"source preparation changed at {key!r}; replan storage"
                )

    rebound = []
    for plan, current_preparations in zip(plans, preparation_maps, strict=True):
        events = tuple(replace(event, block=mapping[event.block]) for event in plan.events)
        rebound.append(replace(
            plan,
            blocks=dict(blocks),
            preparations={key: current_preparations[key] for key in plan.preparations},
            events=events,
        ))
    return tuple(rebound)


def rebind_storage_schedule(
    plans: Sequence[contract_api.SceneStoragePlan],
    *,
    blocks: Mapping[str, contract_api.StorageBlock],
    preparations: Mapping[str, contract_api.SourcePreparation]
        | Sequence[Mapping[str, contract_api.SourcePreparation]],
) -> tuple[contract_api.SceneStoragePlan, ...]:
    """Rebind physical dataset identities while preserving a fixed schedule."""

    return _rebind_storage_plans(tuple(plans), blocks=blocks, preparations=preparations)


def load_storage_schedule(
    path: str | os.PathLike[str] | bytes,
    *,
    shapes: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | TargetHardwareMemory,
    expected: contract_api.ContractionProblem | contract_api.ExpandedCorrelator | Sequence[contract_api.ContractionProblem],
    blocks: Mapping[str, contract_api.StorageBlock] | None = None,
    preparations: Mapping[str, contract_api.SourcePreparation]
        | Sequence[Mapping[str, contract_api.SourcePreparation]] | None = None,
    profile: CostProfile | None = None,
    profile_source: Any = None,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Validate and load a fixed schedule; current inputs must be explicit.

    ``expected`` supplies current shapes, algebra, and full SourceSpec metadata.
    Its ExpandedCorrelator explicitly supplies current identity bindings and
    current blocks/preparations when those keyword arguments are omitted.
    Physics, source sharing, capacities and preparation recipes must still
    match. A bare problem requires unchanged identities. No schedule is searched.
    """

    if expected is None:
        raise ValueError("expected current task semantics are required for safe plan reuse")
    problems, expanded = _expected_context(expected)
    inputs = None if expanded is None else expanded.inputs
    if (blocks is None) != (preparations is None):
        raise ValueError("blocks and preparations must be supplied together")
    if blocks is None and inputs is not None:
        blocks = inputs.blocks
        preparations = (
            tuple(item.preparations for item in inputs.task_batches)
            if inputs.task_batches else inputs.preparations
        )
    if blocks is None or preparations is None:
        raise ValueError("current physical blocks and preparations are required for rebinding")

    document = _read_document(path)
    payload = _decode_envelope(document, expected_type=_PAYLOAD_STORAGE_SCHEDULE)
    _strict_keys(
        payload,
        {"shapes", "semantics", "hardware", "profile_source", "blocks", "plans", "metadata"},
        name="storage schedule payload",
    )
    saved_shapes = _normalize_shapes(payload["shapes"], name="saved shapes")
    supplied_shapes = None if shapes is None else _normalize_shapes(shapes)
    _assert_expected_semantics(
        payload["semantics"],
        expected,
        saved_shapes=saved_shapes,
        supplied_shapes=supplied_shapes,
        batch_count=len(payload["plans"]) if isinstance(payload["plans"], list) else -1,
        allow_identity_rebind=expanded is not None,
    )
    target = _validate_hardware_payload(hardware)
    if target != _strict_json_value(payload["hardware"], name="saved hardware"):
        raise PlanArchiveError("hardware specifications differ from saved schedule")
    saved_profile_source = _strict_json_value(payload["profile_source"], name="profile_source")
    if profile_source is not None and _profile_source_payload(profile_source) != saved_profile_source:
        raise PlanArchiveError("profile source differs from the saved storage schedule")
    plans_raw = payload["plans"]
    if not isinstance(plans_raw, list):
        raise PlanArchiveError("storage schedule plans must be a list")
    saved_blocks = _decode_plan_value(payload["blocks"])
    if not isinstance(saved_blocks, Mapping):
        raise PlanArchiveError("storage schedule blocks must be a mapping")
    plans = []
    for index, value in enumerate(plans_raw):
        plan = _decode_plan_value(value)
        if not isinstance(plan, contract_api.SceneStoragePlan):
            raise PlanArchiveError(f"storage schedule plan {index} has an invalid type")
        plans.append(replace(plan, blocks=dict(saved_blocks)))
    plans = tuple(plans)
    for plan, problem in zip(plans, problems, strict=True):
        _assert_saved_sum_barriers(plan.state.problem, problem)
        _assert_saved_term_kernels(plan.state.problem, problem)
    _validate_schedule(plans)
    if profile is not None:
        expected_profile = _profile_payload(profile)
        if any(_profile_payload(plan.cost_profile) != expected_profile for plan in plans):
            raise PlanArchiveError("profile changed; create a new plan and save a new archive")
    rebound = _rebind_storage_plans(plans, blocks=blocks, preparations=preparations)
    metadata = _strict_json_value(payload["metadata"], name="metadata")
    if not isinstance(metadata, Mapping):
        raise PlanArchiveError("storage schedule metadata must be a mapping")
    return rebound, metadata


def _task_context_payload(expanded: Any, inputs: Any | None = None) -> dict[str, Any]:
    if not isinstance(expanded, contract_api.ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    selected_inputs = expanded.inputs if inputs is None else inputs
    problem = expanded.problem if selected_inputs is None else selected_inputs.problem
    return {
        "problem": None if problem is None else _problem_payload(problem),
        "coordinates": _encode_plan_value(expanded.coordinates),
        "batch": _encode_plan_value(expanded.batch),
        "term_ids": [] if selected_inputs is None else list(selected_inputs.term_ids),
        "coordinate_batch": None if selected_inputs is None else _encode_plan_value(selected_inputs.coordinate_batch),
        "batch_topology": None if selected_inputs is None else _encode_plan_value(selected_inputs.batch_topology),
        "batch_output_axes": [] if selected_inputs is None else list(selected_inputs.batch_output_axes),
    }


def _batch_shape_key(inputs: Any) -> tuple[Any, ...]:
    problem = inputs.problem
    batch_size = None if inputs.coordinate_batch is None else inputs.coordinate_batch.actual_size
    return (
        inputs.batch_topology,
        batch_size,
        problem.backend,
        tuple((key, spec.shape, spec.dtype.str) for key, spec in problem.specs.items()),
        tuple((term.inputs, term.subscripts.normalized(), term.coefficient, term.kernel)
              for term in problem.terms),
        tuple((key, source.snapshot_bytes) for key, source in problem.sources.items()),
    )


def _batch_key_without_kernels(key):
    """The earlier three-field term key is valid only for ordinary terms."""
    if any(term[3] is not None for term in key[4]):
        return None
    return (*key[:4], tuple(term[:3] for term in key[4]), key[5])


def _task_inputs(expanded: Any) -> tuple[Any, ...]:
    if not isinstance(expanded, contract_api.ExpandedCorrelator):
        raise TypeError("expected must be an ExpandedCorrelator")
    if expanded.inputs is None:
        return ()
    inputs = expanded.inputs
    return tuple(inputs.task_batches) if inputs.task_batches else (inputs,)


def _path_records(path: Any) -> tuple[Any, ...]:
    if not isinstance(path, contract_api.PlannedContraction):
        raise TypeError("path must be a PlannedContraction")
    return tuple(path.batch_paths) if path.batch_paths else (path,)


def _default_target(path: Any, expanded: Any) -> dict[str, Any]:
    profile = path.profile
    return {
        "execution_mode": path.execution_mode,
        "backend": None if expanded.problem is None else expanded.problem.backend,
        "hardware_fingerprint": None if profile is None else profile.hardware_fingerprint,
        "memory_limit": path.memory_limit,
        "host_memory_limit": path.host_memory_limit,
    }


def save_contraction_plan(
    path: str | os.PathLike[str],
    plan: contract_api.PlannedContraction,
    *,
    hardware: Mapping[str, Any] | TargetHardwareMemory | None = None,
    profile_source: Any = None,
    metadata: Mapping[str, Any] | None = None,
    storage_policy: Any = None,
    optimize: Any = None,
    reserved_host_bytes: int | None = None,
    reserved_device_bytes: int | None = None,
) -> Path:
    """Persist a selected PlannedContraction and its explicit provenance.

    Only the fixed operation/storage plans and semantic signatures are saved.
    Current adapters/loaders, Calculation values/reports, and search objects are
    omitted. Workflow-specific options are supplied as explicit archive fields.
    """

    if not isinstance(plan, contract_api.PlannedContraction):
        raise TypeError("plan must be a PlannedContraction")
    planned = plan
    expanded = planned.expanded
    if not isinstance(expanded, contract_api.ExpandedCorrelator):
        raise TypeError("path must contain an ExpandedCorrelator")
    selected_paths = _path_records(planned)
    if any(item.execution_mode not in {"planned", "direct", "empty"} for item in selected_paths):
        raise ValueError("unsupported path execution mode")
    has_temporary_move = any(
        item.plan is not None and any(action.kind in ("spill_disk", "restore_host")
                                      for action in item.plan.state.actions)
        for item in selected_paths
    )
    if has_temporary_move and (
        len(selected_paths) != 1 or planned.execution_mode != "planned"
        or selected_paths[0].execution_mode != "planned"
        or bool(expanded.inputs and expanded.inputs.task_batches)
    ):
        raise NotImplementedError("temporary array archives support one planned non-batched path")
    for index, item in enumerate(selected_paths):
        if item.plan is not None and not isinstance(item.plan, contract_api.SceneStoragePlan):
            raise TypeError(f"path batch {index} must contain a SceneStoragePlan or None")
        if item.plan is not None:
            validate_executable_resources(item.plan, allow_temporary_array=True)
        if item.execution_mode in {"planned", "direct"} and item.plan is None:
            raise ValueError(f"path batch {index} has no audited storage schedule")
        if item.plan is not None and (not item.plan.feasible or not item.plan.state.done):
            raise ValueError(f"path batch {index} is not complete and feasible")
    retention_modes = {
        item.plan.retention_planned for item in selected_paths if item.plan is not None
    }
    if len(retention_modes) > 1:
        raise PlanArchiveError("path batches disagree about logical tensor retention")

    target = hardware
    if target is None:
        target = _default_target(planned, expanded)
    target = _validate_hardware_payload(target)
    archive_profile = planned.profile
    if archive_profile is None:
        archive_profile = next(
            (item.plan.cost_profile for item in selected_paths if item.plan is not None
             and item.plan.cost_profile is not None),
            None,
        )
    if planned.execution_mode in {"planned", "direct"} and archive_profile is None:
        raise PlanArchiveError("audited paths must carry their CostProfile for persistence")
    model_payload = _profile_payload(archive_profile)
    source_payload = _profile_source_payload(profile_source)
    prediction_payload = _strict_json_value(planned.prediction, name="prediction")
    if not isinstance(prediction_payload, Mapping):
        raise PlanArchiveError("path prediction must be a JSON object")
    if metadata is None:
        metadata_payload = {}
    else:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        metadata_payload = _strict_json_value(metadata, name="metadata")

    saved_keys = tuple(planned.batch_shape_keys)
    base_reserved_host = reserved_host_bytes
    if base_reserved_host is None:
        assembly_host = 0 if expanded.inputs is None else expanded.inputs.assembly_host_bytes
        base_reserved_host = max(0, planned.reserved_host_bytes - assembly_host)
    base_reserved_device = reserved_device_bytes
    if base_reserved_device is None:
        assembly_device = 0 if expanded.inputs is None else expanded.inputs.assembly_device_bytes
        base_reserved_device = max(0, planned.reserved_device_bytes - assembly_device)
    task_batches = _task_inputs(expanded)
    if not task_batches and planned.execution_mode == "empty":
        task_batches = (None,)
    if len(task_batches) != len(selected_paths):
        raise PlanArchiveError("path batch count differs from the expanded task")
    if saved_keys and len(saved_keys) != len(selected_paths):
        raise PlanArchiveError("path batch shape keys do not match the selected batches")
    if not saved_keys and expanded.inputs and expanded.inputs.task_batches:
        saved_keys = tuple(_batch_shape_key(item) for item in task_batches)
    records = []
    for index, (child_path, child_inputs) in enumerate(zip(selected_paths, task_batches, strict=True)):
        if child_path.plan is not None and (
            child_path.plan.reserved_host_bytes != base_reserved_host + child_inputs.assembly_host_bytes
            or child_path.plan.reserved_device_bytes != base_reserved_device + child_inputs.assembly_device_bytes
        ):
            raise PlanArchiveError(
                f"path batch {index} reserve differs from its saved storage schedule"
            )
        child_expanded = replace(expanded, inputs=child_inputs)
        records.append({
            "task": _task_context_payload(child_expanded),
            "batch_shape_key": _encode_plan_value(saved_keys[index]) if saved_keys else None,
            "execution_mode": child_path.execution_mode,
            "memory_limit": child_path.memory_limit,
            "host_memory_limit": child_path.host_memory_limit,
            "reserved_host_bytes": base_reserved_host,
            "reserved_device_bytes": base_reserved_device,
            "storage_plan": None if child_path.plan is None else _encode_archive_value(child_path.plan),
        })
    payload = {
        "path_signature": (
            _digest(None) if expanded.problem is None
            else _digest(_problem_path_invariants(expanded.problem))
        ),
        "task": _task_context_payload(expanded),
        "execution_mode": planned.execution_mode,
        "memory_limit": planned.memory_limit,
        "host_memory_limit": planned.host_memory_limit,
        "reserved_host_bytes": base_reserved_host,
        "reserved_device_bytes": base_reserved_device,
        "batch_shape_keys": _encode_plan_value(saved_keys),
        "profile": model_payload,
        "profile_source": source_payload,
        "prediction": prediction_payload,
        "hardware": target,
        "retention_planned": bool(next(iter(retention_modes), False)),
        "storage_policy": storage_policy,
        "optimize": optimize,
        "plans": records,
        "metadata": metadata_payload,
    }
    return _write_document(path, _make_envelope("correlator-plan", payload))


def _validate_plan_archive_budget(
    memory_limit: Any,
    host_memory_limit: Any,
    reserved_host_bytes: Any,
    reserved_device_bytes: Any,
) -> None:
    if memory_limit is not None and (type(memory_limit) is not int or memory_limit <= 0):
        raise ValueError("memory_limit must be a positive integer number of bytes")
    if host_memory_limit is not None and (
        type(host_memory_limit) is not int or host_memory_limit <= 0
    ):
        raise ValueError("host_memory_limit must be a positive integer or None")
    for value in (reserved_host_bytes, reserved_device_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("memory reserves must be nonnegative integer bytes")


def load_contraction_plan(
    source: str | os.PathLike[str] | bytes,
    *,
    expected: contract_api.ExpandedCorrelator,
    hardware: Mapping[str, Any] | TargetHardwareMemory,
    profile: CostProfile | None = None,
    profile_source: Any = None,
    memory_limit: int | None = None,
    host_memory_limit: int | None = None,
    reserved_host_bytes: int | None = None,
    reserved_device_bytes: int | None = None,
) -> contract_api.PlannedContraction:
    """Load a selected path and its saved storage schedule without replanning.

    Current readers and physical identities may be rebound when the saved
    source shapes, preparation declarations, coordinate layout, and storage
    capacities still match. Any change that needs a new schedule requires a
    newly planned archive.
    """

    if not isinstance(expected, contract_api.ExpandedCorrelator):
        raise TypeError("expected must be a current ExpandedCorrelator")
    document = _read_document(source)
    payload = _decode_envelope(document, expected_type="correlator-plan")
    payload_keys = {
        "path_signature", "task", "execution_mode", "memory_limit", "host_memory_limit",
        "reserved_host_bytes", "reserved_device_bytes", "batch_shape_keys", "profile",
        "profile_source", "hardware", "retention_planned", "storage_policy", "optimize",
        "plans", "metadata",
    }
    # Older archives did not persist a prediction. Keep them readable with an
    # explicit unavailable marker; all newly saved archives carry the original.
    has_saved_prediction = "prediction" in payload
    _strict_keys(
        payload,
        payload_keys | ({"prediction"} if has_saved_prediction else set()),
        name="correlator path payload",
    )
    if type(payload["retention_planned"]) is not bool:
        raise PlanArchiveError("retention_planned must be a boolean")

    current_path_signature = (
        _digest(None) if expected.problem is None
        else _digest(_problem_path_invariants(expected.problem))
    )
    if not hmac.compare_digest(current_path_signature, payload["path_signature"]):
        raise PlanArchiveError(
            "current input shapes, source identity, axes, or contraction semantics differ from the saved path"
        )
    target = _validate_hardware_payload(hardware)
    if target != _strict_json_value(payload["hardware"], name="saved hardware"):
        raise PlanArchiveError("target hardware differs from the saved plan")

    archived_profile = None if payload["profile"] is None else _decode_plan_value(payload["profile"])
    if profile is None:
        profile = archived_profile
    if profile is not None:
        from lqcdweave.contraction.performance.api import CostProfile
        if not isinstance(profile, CostProfile):
            raise TypeError("profile must be a CostProfile or None")
    elif payload["execution_mode"] in {"planned", "direct"}:
        from lqcdweave.contraction.performance.api import CostProfile
        profile = CostProfile()

    saved_profile_source = _strict_json_value(payload["profile_source"], name="profile_source")
    if profile_source is not None:
        if _profile_source_payload(profile_source) != saved_profile_source:
            raise PlanArchiveError("profile source changed; create a new plan")
    if _profile_payload(profile) != payload["profile"]:
        raise PlanArchiveError("cost profile changed; create a new plan")

    saved_prediction = (
        _strict_json_value(payload["prediction"], name="prediction")
        if has_saved_prediction
        else {"execution_mode": payload["execution_mode"], "predicted_seconds": None}
    )
    if not isinstance(saved_prediction, Mapping):
        raise PlanArchiveError("saved prediction must be a JSON object")
    saved_prediction = dict(saved_prediction)

    def task_layout(task: Any) -> dict[str, Any]:
        task = _strict_object(task, name="saved task context")
        return {key: value for key, value in task.items() if key != "problem"}

    if task_layout(_task_context_payload(expected)) != task_layout(payload["task"]):
        raise PlanArchiveError(
            "coordinates or batch structure changed; create a new plan"
        )

    records = payload["plans"]
    if not isinstance(records, list) or not records:
        raise PlanArchiveError("correlator archive must contain one or more fixed path records")
    if expected.inputs is not None and expected.inputs.batch_candidates and not expected.inputs.task_batches:
        raise PlanArchiveError("selected batch structure changed; create a new plan")
    current_inputs = _task_inputs(expected)
    current_keys = tuple(_batch_shape_key(item) for item in current_inputs)
    saved_batch_keys = _decode_plan_value(payload["batch_shape_keys"])
    expected_batch_keys = current_keys if expected.inputs and expected.inputs.task_batches else ()
    legacy_batch_keys = None
    if isinstance(saved_batch_keys, tuple) and saved_batch_keys != expected_batch_keys and expected_batch_keys:
        older = tuple(_batch_key_without_kernels(key) for key in expected_batch_keys)
        if all(key is not None for key in older) and saved_batch_keys == older:
            legacy_batch_keys = older
    if not isinstance(saved_batch_keys, tuple) or (saved_batch_keys != expected_batch_keys
                                                   and legacy_batch_keys is None):
        raise PlanArchiveError("coordinate batch structure changed; create a new plan")
    decoded_records = []
    for index, raw in enumerate(records):
        item = _strict_object(raw, name=f"correlator path records[{index}]")
        _strict_keys(
            item,
            {"task", "batch_shape_key", "execution_mode", "memory_limit", "host_memory_limit",
             "reserved_host_bytes", "reserved_device_bytes", "storage_plan"},
            name=f"correlator path records[{index}]",
        )
        decoded_records.append(item)

    saved_modes = {item["execution_mode"] for item in decoded_records}
    if expected.problem is None:
        if expected.inputs is not None or saved_modes != {"empty"} or len(decoded_records) != 1:
            raise PlanArchiveError("empty expansion is incompatible with the saved path")
        if decoded_records[0]["storage_plan"] is not None:
            raise PlanArchiveError("empty path archive unexpectedly contains a storage schedule")
        if _task_context_payload(expected) != payload["task"]:
            raise PlanArchiveError("empty expansion coordinates differ from the saved archive")
        return contract_api.PlannedContraction(
            expected, None, None, saved_prediction,
            0.0, "empty", None, None, None,
        )
    if expected.inputs is None:
        raise PlanArchiveError("expected nonempty expansion has no DiagramInputs")

    batch_mode = bool(expected.inputs.task_batches)
    if len(decoded_records) != len(current_inputs) or (not batch_mode and len(decoded_records) != 1):
        raise PlanArchiveError("saved path records do not cover the current coordinate batches")
    templates = decoded_records
    for index, (record, child_inputs) in enumerate(zip(templates, current_inputs, strict=True)):
        expected_key = None if not batch_mode else _encode_plan_value(current_keys[index])
        older_key = (None if legacy_batch_keys is None else
                     _encode_plan_value(legacy_batch_keys[index]))
        if record["batch_shape_key"] != expected_key and (
                legacy_batch_keys is None or record["batch_shape_key"] != older_key):
            raise PlanArchiveError("coordinate batch order or shape changed; create a new plan")
        child_expanded = replace(expected, inputs=child_inputs)
        if task_layout(_task_context_payload(child_expanded)) != task_layout(record["task"]):
            raise PlanArchiveError(
                f"coordinate batch {index} changed; create a new plan"
            )

    modes = {item["execution_mode"] for item in templates}
    if len(modes) != 1 or next(iter(modes)) not in {"planned", "direct"}:
        raise PlanArchiveError("saved path records use inconsistent execution modes")
    execution_mode = next(iter(modes))
    saved_reserved_host = payload["reserved_host_bytes"]
    saved_reserved_device = payload["reserved_device_bytes"]
    if reserved_host_bytes is not None and reserved_host_bytes != saved_reserved_host:
        raise PlanArchiveError("memory reserve changed; create a new plan")
    if reserved_device_bytes is not None and reserved_device_bytes != saved_reserved_device:
        raise PlanArchiveError("memory reserve changed; create a new plan")
    reserved_host_bytes = saved_reserved_host
    reserved_device_bytes = saved_reserved_device
    root_memory_limit = payload["memory_limit"] if memory_limit is None else memory_limit
    root_host_memory_limit = (
        payload["host_memory_limit"] if host_memory_limit is None else host_memory_limit
    )
    _validate_plan_archive_budget(
        root_memory_limit, root_host_memory_limit,
        reserved_host_bytes, reserved_device_bytes,
    )
    if root_memory_limit is None:
        raise PlanArchiveError("a finite current memory_limit is required to load this plan")
    current_children = tuple(replace(expected, inputs=item) for item in current_inputs)
    saved_plans = []
    for index, (record, child_expanded) in enumerate(zip(templates, current_children, strict=True)):
        if record["execution_mode"] != execution_mode:
            raise PlanArchiveError("saved path mode differs across coordinate batches")
        compatible, _identity_changed = _problem_payload_compatible(
            _strict_object(record["task"]["problem"], name=f"saved batch {index} problem"),
            _problem_payload(child_expanded.problem),
            allow_identity_rebind=True,
        )
        if not compatible:
            raise PlanArchiveError(
                f"coordinate batch {index} changed input identity or contraction semantics"
            )
        saved_plan = None if record["storage_plan"] is None else _decode_plan_value(record["storage_plan"])
        if saved_plan is not None and not isinstance(saved_plan, contract_api.SceneStoragePlan):
            raise PlanArchiveError(f"saved coordinate batch {index} has an invalid storage plan")
        if execution_mode in {"planned", "direct"} and saved_plan is None:
            raise PlanArchiveError(f"saved coordinate batch {index} has no storage plan")
        if saved_plan is not None:
            if not saved_plan.feasible or not saved_plan.state.done:
                raise PlanArchiveError(f"saved coordinate batch {index} is incomplete or infeasible")
            validate_executable_resources(saved_plan, allow_temporary_array=True)
            _assert_saved_sum_barriers(saved_plan.state.problem, child_expanded.problem)
            _assert_saved_term_kernels(saved_plan.state.problem, child_expanded.problem)
        saved_plans.append(saved_plan)

    has_temporary_move = any(
        saved_plan is not None and any(action.kind in ("spill_disk", "restore_host")
                                       for action in saved_plan.state.actions)
        for saved_plan in saved_plans
    )
    if has_temporary_move and (batch_mode or execution_mode != "planned" or len(saved_plans) != 1):
        raise PlanArchiveError("temporary array archives support one planned non-batched path")
    current_preparations = tuple(item.preparations for item in current_inputs)
    try:
        rebound_plans = _rebind_storage_plans(
            tuple(saved_plans),
            blocks=expected.inputs.blocks,
            preparations=current_preparations,
        )
    except (PlanArchiveError, ValueError) as exc:
        raise PlanArchiveError(
            f"physical storage identity or preparation layout changed; create a new plan: {exc}"
        ) from exc

    checked_plans = []
    for index, (record, child_inputs, plan) in enumerate(
        zip(templates, current_inputs, rebound_plans, strict=True)
    ):
        record_host_reserve = record["reserved_host_bytes"]
        record_device_reserve = record["reserved_device_bytes"]
        if (record_host_reserve != saved_reserved_host
                or record_device_reserve != saved_reserved_device
                or plan.reserved_host_bytes != record_host_reserve + child_inputs.assembly_host_bytes
                or plan.reserved_device_bytes != record_device_reserve + child_inputs.assembly_device_bytes):
            raise PlanArchiveError(
                f"coordinate batch {index} reserve differs from the saved schedule; create a new plan"
            )
        current_memory_limit = root_memory_limit if memory_limit is not None else record["memory_limit"]
        current_host_limit = (
            root_host_memory_limit if host_memory_limit is not None else record["host_memory_limit"]
        )
        if current_memory_limit is None:
            raise PlanArchiveError(
                f"coordinate batch {index} has no finite saved memory budget"
            )
        _validate_plan_archive_budget(
            current_memory_limit, current_host_limit,
            record_host_reserve, record_device_reserve,
        )
        backend = child_inputs.problem.backend
        active_peak = plan.peak_device_bytes if backend == "cupy" else plan.peak_host_bytes
        if (active_peak > current_memory_limit
                or (current_host_limit is not None and plan.peak_host_bytes > current_host_limit)):
            raise PlanArchiveError("saved path exceeds the current memory budget")
        checked_plans.append(replace(
            plan,
            memory_limit=current_memory_limit,
            host_memory_limit=current_host_limit,
        ))
    rebound_plans = tuple(checked_plans)
    planning_seconds = 0.0

    if execution_mode in {"planned", "direct"}:
        if expected.problem is None or expected.inputs is None:
            raise PlanArchiveError("expected expansion has no contraction inputs")
        child_paths = tuple(
            contract_api.PlannedContraction(
                child_expanded,
                storage_plan,
                None,
                saved_prediction,
                0.0,
                execution_mode,
                storage_plan.memory_limit,
                storage_plan.host_memory_limit,
                id(storage_plan),
                profile,
                storage_plan.reserved_host_bytes,
                storage_plan.reserved_device_bytes,
            )
            for child_expanded, child_inputs, storage_plan in zip(
                current_children, current_inputs, rebound_plans, strict=True
            )
        )
        first = child_paths[0]
        batch_keys = tuple(current_keys)
        return contract_api.PlannedContraction(
            expected,
            first.plan,
            None,
            saved_prediction,
            planning_seconds,
            execution_mode,
            first.plan.memory_limit,
            first.plan.host_memory_limit,
            id(first.plan),
            profile,
            first.plan.reserved_host_bytes,
            first.plan.reserved_device_bytes,
            batch_paths=child_paths if batch_mode else (),
            batch_shape_keys=batch_keys if batch_mode else (),
        )

    return contract_api.PlannedContraction(
        expected,
        None,
        None,
        saved_prediction,
        planning_seconds,
        execution_mode,
        memory_limit,
        host_memory_limit,
        None,
        profile,
        reserved_host_bytes,
        reserved_device_bytes,
    )


__all__ = [
    "migrate_plan_archive",
    "migrate_plan_bundle",
    "save_cfg_plan", "load_cfg_plan",
    "ARCHIVE_FORMAT",
    "ARCHIVE_VERSION",
    "PlanArchiveError",
    "decode_plan_value",
    "encode_plan_value",
    "load_contraction_plan",
    "load_plan",
    "load_storage_schedule",
    "rebind_storage_schedule",
    "save_contraction_plan",
    "save_plan",
    "save_storage_schedule",
    "structural_signature",
    "validate_bundle",
]
