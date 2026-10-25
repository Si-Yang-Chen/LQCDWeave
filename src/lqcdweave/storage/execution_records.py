"""Execution attempts and scientific validation, separate from data placement."""
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import uuid
from lqcdweave.contracts.dataset import canonical_json, stable_digest
from lqcdweave.contracts.execution import normalize_execution_parameters
from lqcdweave.contracts.source import (validate_external_import_registration, validate_local_source_declaration, validate_local_source_event, validate_local_spatial_point_spec)

_EXECUTION_FIELDS = frozenset({
    "schema", "definition_id", "data_id", "history_complete", "attempts",
    "scientific_validation",
})


_ATTEMPT_FIELDS_V1 = frozenset({
    "run_id", "producer_attempt_id", "writer_session_id", "stage", "origin",
    "status", "started_at", "ended_at", "failure", "recovered_from",
})


_ATTEMPT_FIELDS_V4 = _ATTEMPT_FIELDS_V1 | {"execution_parameters"}


_UNKNOWN_VALIDATION = {"status": "unknown"}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_timestamp(value, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("attempt timestamps must be UTC ISO-8601 strings")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("attempt timestamps must be UTC ISO-8601 strings") from exc


def _nonempty_json_mapping(value, field):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"scientific validation {field} must be a nonempty JSON mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"scientific validation {field} keys must be nonempty strings")
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scientific validation {field} must contain legal JSON values") from exc
    if not normalized:
        raise ValueError(f"scientific validation {field} must be nonempty")
    if _contains_unknown_value(normalized):
        raise ValueError(f"scientific validation {field} cannot contain unknown null values")
    return normalized


def _contains_unknown_value(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_unknown_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unknown_value(item) for item in value)
    return False


def _normalize_scientific_validation(value, spec):
    if value is None:
        return dict(_UNKNOWN_VALIDATION)
    if not isinstance(value, Mapping):
        raise TypeError("scientific_validation must be a JSON mapping")
    status = value.get("status")
    if status == "unknown":
        if set(value) != {"status"}:
            raise ValueError("unknown scientific validation accepts only its status")
        return dict(_UNKNOWN_VALIDATION)
    required = {"status", "method", "criteria", "checked_coverage", "evidence"}
    if status not in ("passed", "failed") or set(value) != required:
        raise ValueError("known scientific validation needs status, method, criteria, checked_coverage and evidence")
    method = value["method"]
    if not isinstance(method, str) or not method.strip():
        raise ValueError("scientific validation method must be a nonempty string")
    criteria = _nonempty_json_mapping(value["criteria"], "criteria")
    evidence = _nonempty_json_mapping(value["evidence"], "evidence")
    coverage = value["checked_coverage"]
    if not isinstance(coverage, Mapping) or not coverage:
        raise ValueError("checked_coverage must map named axes to explicit coordinate lists")
    if any(not isinstance(axis, str) for axis in coverage):
        raise ValueError("checked_coverage axis names must be strings")
    names = set(spec.axis_names)
    if set(coverage) - names:
        raise ValueError("checked_coverage names an axis absent from the DatasetSpec")
    normalized_coverage = {}
    for axis in spec.axis_names:
        if axis not in coverage:
            continue
        coordinates = coverage[axis]
        if not isinstance(coordinates, (list, tuple)) or not coordinates:
            raise ValueError("checked_coverage coordinates must be nonempty ordered lists")
        keys = [canonical_json(item) for item in coordinates]
        available = {canonical_json(item) for item in spec.coordinates(axis)}
        if len(set(keys)) != len(keys) or any(key not in available for key in keys):
            raise ValueError("checked_coverage coordinates must be unique labels present in the DatasetSpec")
        normalized_coverage[axis] = json.loads(canonical_json(list(coordinates)))
    result = dict(status=status, method=method.strip(), criteria=criteria,
                  checked_coverage=normalized_coverage, evidence=evidence)
    # The complete result must itself be representable by the canonical JSON contract.
    try:
        return json.loads(canonical_json(result))
    except (TypeError, ValueError) as exc:
        raise ValueError("scientific_validation must contain legal JSON values") from exc


def _validate_attempt(attempt, *, draft, execution_schema, _validated_parameters=False):
    fields = {1: _ATTEMPT_FIELDS_V1,
              4: _ATTEMPT_FIELDS_V4, 5: _ATTEMPT_FIELDS_V4}.get(execution_schema)
    if fields is None:
        raise ValueError("invalid dataset attempt schema")
    if not isinstance(attempt, dict) or set(attempt) != fields:
        raise ValueError("invalid dataset attempt record")
    for name in ("run_id", "producer_attempt_id", "stage", "recovered_from"):
        value = attempt[name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"attempt {name} must be a nonempty string or null")
    if not isinstance(attempt["writer_session_id"], str) or not attempt["writer_session_id"]:
        raise ValueError("attempt writer_session_id must be a nonempty string")
    if attempt["origin"] not in ("dataset_graph", "direct_writer"):
        raise ValueError("invalid dataset attempt origin")
    if attempt["origin"] == "direct_writer":
        if any(attempt[name] is not None for name in ("run_id", "producer_attempt_id", "stage")):
            raise ValueError("direct writer attempts cannot claim runtime IDs")
    elif not attempt["run_id"] or not attempt["stage"]:
        raise ValueError("dataset graph attempts require run_id and stage")
    if attempt["producer_attempt_id"] is not None and attempt["origin"] != "dataset_graph":
        raise ValueError("only runtime attempts can identify a producer attempt")
    if execution_schema in (4, 5):
        parameters = attempt["execution_parameters"]
        if not _validated_parameters:
            parameters = normalize_execution_parameters(parameters)
            if canonical_json(parameters) != canonical_json(attempt["execution_parameters"]):
                raise ValueError("execution parameters are not canonical JSON")
        if parameters is not None:
            if attempt["origin"] != "dataset_graph":
                raise ValueError("execution parameters are only valid for graph attempts")
            if not attempt["producer_attempt_id"]:
                raise ValueError("execution parameters require a bound producer attempt")
    _validate_timestamp(attempt["started_at"])
    _validate_timestamp(attempt["ended_at"], optional=attempt["ended_at"] is None)
    status = attempt["status"]
    allowed = {"running", "partial", "failed", "interrupted", "complete"}
    if status not in allowed:
        raise ValueError("invalid dataset attempt status")
    if status == "running":
        if not draft or attempt["ended_at"] is not None or attempt["failure"] is not None:
            raise ValueError("only the latest draft attempt may be running")
    elif attempt["ended_at"] is None:
        raise ValueError("finished dataset attempts require ended_at")
    failure = attempt["failure"]
    if status == "failed":
        if (not isinstance(failure, dict) or set(failure) != {"type", "message"} or
                not all(isinstance(failure[key], str) and failure[key] for key in failure)):
            raise ValueError("failed attempts require an exception type and message")
    elif status == "interrupted":
        if failure is not None:
            raise ValueError("interrupted attempts cannot claim a failure")
    elif failure is not None:
        raise ValueError("nonfailed attempts cannot carry a failure")


def _validate_execution(execution, spec, *, data_id, draft, provenance=None):
    if not isinstance(execution, dict):
        raise ValueError("invalid dataset execution record")
    if type(execution.get("schema")) is not int or execution["schema"] not in (1, 3, 4, 5):
        raise ValueError("unsupported dataset execution schema")
    schema = execution["schema"]
    if schema == 3:
        if draft:
            raise ValueError("invalid external registration execution record")
        validate_external_import_registration(provenance, execution,
            data_id=data_id, definition_id=spec.definition_id)
        return
    if schema == 5:
        extra = "local_source_declaration" if draft else "local_source_event"
        if set(execution) != _EXECUTION_FIELDS | {extra}:
            raise ValueError("invalid schema-5 local source execution record")
    elif set(execution) != _EXECUTION_FIELDS:
        raise ValueError("invalid dataset execution record")
    if execution["definition_id"] != spec.definition_id:
        raise ValueError("dataset execution definition does not match its DatasetSpec")
    if execution["data_id"] != data_id:
        raise ValueError("dataset execution actual ID does not match its registration")
    if type(execution["history_complete"]) is not bool:
        raise ValueError("dataset execution history_complete must be bool")
    attempts = execution["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("dataset execution needs at least one attempt")
    session_ids = []
    for index, attempt in enumerate(attempts):
        _validate_attempt(attempt, draft=draft and index == len(attempts) - 1,
                          execution_schema=schema)
        session_ids.append(attempt["writer_session_id"])
        if index < len(attempts) - 1 and attempt["status"] in ("running", "complete"):
            raise ValueError("only the latest attempt may be running or complete")
        if attempt["status"] == "interrupted" and index < len(attempts)-1:
            if (index + 1 >= len(attempts) or
                    attempts[index + 1]["recovered_from"] != attempt["writer_session_id"]):
                raise ValueError("an interrupted attempt must be followed by its recovery attempt")
        if attempt["recovered_from"] is not None:
            if (index == 0 or attempts[index - 1]["status"] not in ("interrupted", "failed") or
                    attempts[index - 1]["writer_session_id"] != attempt["recovered_from"]):
                raise ValueError("recovery attempt must identify the interrupted writer session")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("dataset writer session IDs must be unique")
    if draft:
        if data_id is not None or attempts[-1]["status"] not in (
                "running", "partial", "failed", "interrupted"):
            raise ValueError("invalid partial dataset execution state")
    elif attempts[-1]["status"] != "complete":
        raise ValueError("published dataset execution must end with a complete attempt")
    validation = _normalize_scientific_validation(execution["scientific_validation"], spec)
    if validation != execution["scientific_validation"]:
        raise ValueError("scientific validation is not canonical")
    if validation["status"] == "failed" and not draft:
        raise ValueError("failed scientific validation cannot be published")
    if schema == 5:
        if draft:
            declaration = validate_local_source_declaration(
                execution["local_source_declaration"])
            validate_local_spatial_point_spec(spec.document, declaration)
            if any(attempt["origin"] != "dataset_graph" or
                   attempt["stage"] != "points"
                   for attempt in attempts):
                raise ValueError("local point sampling requires an input-free points graph stage")
        else:
            if "local_source_declaration" in execution:
                raise ValueError("published local source cannot retain a draft declaration")
            validate_local_source_event(execution["local_source_event"],
                data_id=data_id, definition_id=spec.definition_id,
                execution=execution, spec_document=spec.document)


def _execution_record(spec, *, history_complete=True, schema=1,
                      local_source_declaration=None):
    if schema not in (1, 4, 5):
        raise ValueError("unsupported dataset execution schema")
    result = dict(schema=schema, definition_id=spec.definition_id, data_id=None,
                history_complete=history_complete, attempts=[],
                scientific_validation=dict(_UNKNOWN_VALIDATION))
    if schema == 5:
        declaration = validate_local_source_declaration(local_source_declaration)
        validate_local_spatial_point_spec(spec.document, declaration)
        result["local_source_declaration"] = declaration
    elif local_source_declaration is not None:
        raise ValueError("local source declaration requires execution schema 5")
    return result


def _normalize_attempt_context(context):
    if context is None:
        return dict(run_id=None, producer_attempt_id=None, stage=None,
                    origin="direct_writer")
    if (not isinstance(context, Mapping) or
            set(context) not in ({"run_id", "producer_attempt_id", "stage"},
                                {"run_id", "producer_attempt_id", "stage",
                                 "local_source_declaration"})):
        raise ValueError("attempt_context requires run_id, producer_attempt_id and stage")
    result = dict(context)
    for name in ("run_id", "producer_attempt_id", "stage"):
        value = result[name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"attempt_context {name} must be a nonempty string or null")
    if result["run_id"] is None:
        if "local_source_declaration" in result:
            raise ValueError("local source declaration requires a points graph stage")
        if result["producer_attempt_id"] is not None or result["stage"] is not None:
            raise ValueError("runtime stage metadata requires a run_id")
        return dict(result, origin="direct_writer")
    if result["stage"] is None:
        raise ValueError("runtime dataset attempts require a stage")
    if "local_source_declaration" in result:
        if result["stage"] != "points" or result["producer_attempt_id"] is not None:
            raise ValueError("local source declaration requires an unbound points graph attempt")
        result["local_source_declaration"] = validate_local_source_declaration(
            result["local_source_declaration"])
    return dict(result, origin="dataset_graph")


def _new_attempt(context, *, recovered_from=None):
    context = dict(context)
    attempt = dict(run_id=context["run_id"],
                producer_attempt_id=context["producer_attempt_id"],
                writer_session_id=uuid.uuid4().hex,
                stage=context["stage"], origin=context["origin"], status="running",
                started_at=_utc_now(), ended_at=None, failure=None,
                recovered_from=recovered_from)
    return attempt


def _execution_schema4(execution):
    """Add per-attempt execution parameter slots to an earlier history."""
    schema = execution["schema"]
    if schema != 1:
        raise ValueError("only schema-1 attempt histories can be upgraded to schema 4")
    upgraded = json.loads(canonical_json(execution))
    for attempt in upgraded["attempts"]:
        attempt["execution_parameters"] = None
    upgraded["schema"] = 4
    return upgraded


def _published_record(identity, execution):
    actual_id = stable_digest(identity)
    final_execution = json.loads(canonical_json(execution))
    final_execution["data_id"] = actual_id
    attempt = final_execution["attempts"][-1]
    if attempt["status"] != "running":
        raise ValueError("only a running attempt can publish a dataset")
    attempt["status"] = "complete"
    attempt["ended_at"] = _utc_now()
    if final_execution["schema"] == 5:
        if not attempt["producer_attempt_id"]:
            raise ValueError("local source event producer_attempt_id must be nonempty")
        declaration = final_execution.pop("local_source_declaration")
        final_execution["local_source_event"] = dict(
            declaration, data_id=actual_id,
            definition_id=final_execution["definition_id"],
            run_id=attempt["run_id"], stage=attempt["stage"],
            producer_attempt_id=attempt["producer_attempt_id"],
            writer_session_id=attempt["writer_session_id"])
    # The writer accepted disk history on resume and each public update at its
    # boundary. Publication only changes its private current attempt.
    _validate_attempt(attempt, draft=False, execution_schema=final_execution["schema"],
                      _validated_parameters=True)
    if final_execution["scientific_validation"]["status"] == "failed":
        raise ValueError("failed scientific validation cannot be published")
    return dict(identity=identity, data_id=actual_id, execution=final_execution)

