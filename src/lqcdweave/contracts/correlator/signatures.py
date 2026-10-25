"""Canonical structural encodings and signatures for portable plan records.

This module contains pure JSON transformations only.  File publication,
archive persistence and format handling belong to ``storage.contraction_plans``.
Compiled DAG signatures encode integer axis relations. Older string-based
PlanBundle signatures require explicit offline conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from .ir import ContractionPlan, InputRef, NodeRef, Ref
from .equations import parse_term
from .workload import PlanBundle, WorkloadSpec, encode_scalar, thaw_json


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic strict UTF-8 JSON bytes for an archive value."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not representable as strict JSON: {exc}") from exc
    return text.encode("utf-8")


def json_digest(value: Any) -> str:
    """Return SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def encode_ref(ref: Ref) -> dict[str, Any]:
    """Encode one explicit input or operation reference."""

    if isinstance(ref, InputRef):
        if not isinstance(ref.key, str):
            raise TypeError("archive input references require portable string keys")
        return {"type": "input", "key": ref.key}
    if isinstance(ref, NodeRef):
        return {"type": "node", "index": ref.index}
    raise TypeError(f"unsupported reference object {type(ref).__name__}")


def encode_workload(workload: WorkloadSpec) -> dict[str, Any]:
    """Return the established portable WorkloadSpec payload."""

    if not isinstance(workload, WorkloadSpec):
        raise TypeError("encode_workload expects a WorkloadSpec")
    return {
        "specs": [
            {"key": key, "shape": list(spec.shape), "dtype": spec.dtype.str}
            for key, spec in workload.specs.items()
        ],
        "terms": [
            {
                "inputs": list(term.inputs),
                "subscripts": parse_term(term, workload.specs, index).equation.as_dict(),
                "coefficient": encode_scalar(term.coefficient),
            }
            for index, term in enumerate(workload.terms)
        ],
        "input_sources": thaw_json(workload.input_sources),
        "convention": workload.convention,
        "output_groups": [list(group) for group in workload.output_groups],
        "metadata": thaw_json(workload.metadata),
    }


def encode_plan(plan: ContractionPlan) -> dict[str, Any]:
    """Return the established explicit-index compiled DAG payload."""

    if not isinstance(plan, ContractionPlan):
        raise TypeError("encode_plan expects a ContractionPlan")
    if not isinstance(plan.optimize, str):
        raise TypeError("compiled plan optimize option must be a string for persistence")
    return {
        "specs": [
            {"key": key, "shape": list(spec.shape), "dtype": spec.dtype.str}
            for key, spec in plan.specs.items()
        ],
        "operations": [
            {
                "index": operation.index,
                "inputs": [encode_ref(ref) for ref in operation.inputs],
                "input_subscripts": [list(axes) for axes in operation.input_subscripts],
                "output_subscript": list(operation.output_subscript),
                "output_shape": list(operation.output_shape),
                "output_dtype": operation.output_dtype.str,
            }
            for operation in plan.operations
        ],
        "terms": [
            {
                "output": encode_ref(term.output),
                "output_subscript": list(term.output_subscript),
                "coefficient": encode_scalar(term.coefficient),
                "output_shape": list(term.output_shape),
                "output_dtype": term.output_dtype.str,
            }
            for term in plan.terms
        ],
        "optimize": plan.optimize,
        "operation_use_counts": list(plan.operation_use_counts),
    }


def structural_payload(bundle: PlanBundle) -> dict[str, Any]:
    """Build the historical structure-only signature payload for a bundle."""

    if not isinstance(bundle, PlanBundle):
        raise TypeError("structural_payload expects a PlanBundle")
    workload = encode_workload(bundle.workload)
    # Timing and calibration fingerprints describe the choice process, not the
    # executable contraction structure.  Keep this exclusion stable for v1
    # archives and for callers that persisted ``PlanBundle.signature``.
    metadata = workload["metadata"]
    if isinstance(metadata, Mapping):
        metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"performance_fingerprint", "calibration"}
        }
    workload["metadata"] = metadata
    return {
        "workload": workload,
        "batches": [
            {
                "term_indices": list(batch.term_indices),
                "plan": encode_plan(batch.plan),
            }
            for batch in bundle.batches
        ],
    }


def structural_signature(bundle: PlanBundle) -> str:
    """Return the structural signature of an integer compiled PlanBundle."""

    if not isinstance(bundle, PlanBundle):
        raise TypeError("structural_signature expects a PlanBundle")
    return json_digest(structural_payload(bundle))


__all__ = [
    "canonical_json_bytes",
    "encode_plan",
    "encode_ref",
    "encode_workload",
    "json_digest",
    "structural_payload",
    "structural_signature",
]
