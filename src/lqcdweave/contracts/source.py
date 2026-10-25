"""Explicit knownness contracts for registered external source origins."""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import re

from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.contracts.time_selection import dataset_time_selection


_ORIGIN_FIELDS = frozenset({
    "source_system", "dataset_id", "revision", "producer", "producer_revision",
})
_EXTERNAL_IMPORT_FIELDS = frozenset({"schema", "kind", "origin"})
_EXTERNAL_IMPORT_INPUT_FIELDS = _EXTERNAL_IMPORT_FIELDS | {"registration_runtime"}


def _normalize_knownness(value, field):
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON mapping")
    status = value.get("status")
    if status == "known":
        if set(value) != {"status", "value"}:
            raise ValueError(f"known {field} must contain exactly status and value")
        item = value["value"]
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"known {field} value must be a nonempty string")
        return dict(status="known", value=item)
    if status == "unknown":
        if set(value) != {"status", "reason"}:
            raise ValueError(f"unknown {field} must contain exactly status and reason")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"unknown {field} reason must be a nonempty string")
        return dict(status="unknown", reason=reason)
    raise ValueError(f"{field} status must be 'known' or 'unknown'")


def normalize_external_import_provenance(value):
    """Validate the explicit registration input envelope and detach JSON data.

    This function is intended for newly requested external registrations. Old
    unversioned provenance remains readable as-is and is never upgraded by
    inspecting its keys.
    """
    if not isinstance(value, Mapping) or set(value) != _EXTERNAL_IMPORT_INPUT_FIELDS:
        raise ValueError("external import provenance has an invalid exact field set")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("external import provenance schema must be integer 1")
    if value["kind"] != "external_import":
        raise ValueError("external import provenance kind must be 'external_import'")
    origin = value["origin"]
    if not isinstance(origin, Mapping) or set(origin) != _ORIGIN_FIELDS:
        raise ValueError("external import origin has an invalid exact field set")
    normalized_origin = {
        name: _normalize_knownness(origin[name], f"origin.{name}")
        for name in sorted(_ORIGIN_FIELDS)
    }
    runtime = validate_registration_runtime(value["registration_runtime"])
    result = dict(schema=1, kind="external_import", origin=normalized_origin,
                  registration_runtime=runtime)
    try:
        return json.loads(canonical_json(result))
    except (TypeError, ValueError) as exc:
        raise ValueError("external import provenance must contain legal JSON values") from exc


def external_import_identity(value):
    """Return the strict origin-only provenance stored in dataset identity."""
    if not isinstance(value, Mapping) or set(value) != _EXTERNAL_IMPORT_INPUT_FIELDS:
        raise ValueError("external import provenance has an invalid exact field set")
    normalized = normalize_external_import_provenance(value)
    return dict(schema=1, kind="external_import", origin=normalized["origin"]), normalized["registration_runtime"]


def validate_external_import_identity(value):
    """Validate the identity portion stored in a new external import record."""
    if not isinstance(value, Mapping) or set(value) != _EXTERNAL_IMPORT_FIELDS:
        raise ValueError("external import identity has an invalid exact field set")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("external import identity schema must be integer 1")
    if value["kind"] != "external_import":
        raise ValueError("external import identity kind must be 'external_import'")
    origin = value["origin"]
    if not isinstance(origin, Mapping) or set(origin) != _ORIGIN_FIELDS:
        raise ValueError("external import origin has an invalid exact field set")
    normalized = dict(schema=1, kind="external_import", origin={
        name: _normalize_knownness(origin[name], f"origin.{name}")
        for name in sorted(_ORIGIN_FIELDS)
    })
    if canonical_json(normalized) != canonical_json(dict(value)):
        raise ValueError("external import identity is not canonical")
    return json.loads(canonical_json(normalized))


_REGISTRATION_EXECUTION_FIELDS = frozenset({
    "schema", "definition_id", "data_id", "registration_event",
    "scientific_validation",
})
_REGISTRATION_EVENT_FIELDS = frozenset({
    "operation", "started_at", "completed_at", "runtime",
})
_REGISTRATION_OPERATIONS = frozenset({
    "register_npy", "register_hdf5", "register_binary", "register_npy_timeslices",
})
_LOCAL_SOURCE_DECLARATION_FIELDS = frozenset({
    "schema", "kind", "operation", "producer", "algorithm", "producer_revision",
})
_LOCAL_SOURCE_EVENT_FIELDS = _LOCAL_SOURCE_DECLARATION_FIELDS | {
    "data_id", "definition_id", "run_id", "stage", "producer_attempt_id",
    "writer_session_id",
}
_LOCAL_POINT_OPERATION = "sample_spatial_points"
_LOCAL_POINT_PRODUCER = "lqcdweave.sampling.spatial.points.sample_spatial_points"
_LOCAL_POINT_ALGORITHM = "numpy.PCG64/partial-fisher-yates-v1"
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)


def _registration_timestamp(value):
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise ValueError("external registration timestamps must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("external registration timestamps must be UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("external registration timestamps must be UTC ISO-8601")
    return parsed


def validate_external_import_registration(provenance, execution, *, data_id,
                                          definition_id):
    """Validate the shared schema-3 registration event contract.

    The caller supplies the actual and definition IDs from the enclosing
    dataset identity.
    """
    identity = validate_external_import_identity(provenance)
    if not isinstance(execution, Mapping) or set(execution) != _REGISTRATION_EXECUTION_FIELDS:
        raise ValueError("invalid external registration execution record")
    if type(execution["schema"]) is not int or execution["schema"] != 3:
        raise ValueError("external registration execution schema must be integer 3")
    if (not isinstance(data_id, str) or not data_id or
            not isinstance(definition_id, str) or not definition_id or
            not isinstance(execution["data_id"], str) or
            execution["data_id"] != data_id):
        raise ValueError("registration actual ID does not match its verified dataset")
    if (not isinstance(execution["definition_id"], str) or
            execution["definition_id"] != definition_id):
        raise ValueError("registration definition does not match its DatasetSpec")

    event = execution["registration_event"]
    if not isinstance(event, Mapping) or set(event) != _REGISTRATION_EVENT_FIELDS:
        raise ValueError("invalid external registration event")
    if (not isinstance(event["operation"], str) or
            event["operation"] not in _REGISTRATION_OPERATIONS):
        raise ValueError("invalid external registration operation")
    started_at = _registration_timestamp(event["started_at"])
    completed_at = _registration_timestamp(event["completed_at"])
    if completed_at < started_at:
        raise ValueError("external registration completed_at precedes started_at")
    runtime = validate_registration_runtime(event["runtime"])
    if canonical_json(runtime) != canonical_json(event["runtime"]):
        raise ValueError("external registration runtime is not canonical")
    validation = execution["scientific_validation"]
    if (not isinstance(validation, Mapping) or set(validation) != {"status"} or
            validation["status"] != "unknown"):
        raise ValueError("external registration cannot claim scientific validation")
    return dict(identity=identity, registration_event=dict(
        operation=event["operation"], started_at=event["started_at"],
        completed_at=event["completed_at"], runtime=runtime))


def normalize_local_source_declaration(operation, producer_revision):
    """Build the one supported local-source declaration from runtime facts.

    The caller selects only the fixed sampler operation. Runtime supplies its
    source revision; arbitrary producer/origin mappings are not accepted.
    """
    if operation != _LOCAL_POINT_OPERATION:
        raise ValueError("unsupported local source operation")
    if (not isinstance(producer_revision, str) or
            not re.fullmatch(r"[0-9a-f]{64}", producer_revision)):
        raise ValueError("local producer revision must be a lowercase SHA-256 digest")
    return dict(schema=1, kind="local_generated", operation=_LOCAL_POINT_OPERATION,
        producer=_LOCAL_POINT_PRODUCER, algorithm=_LOCAL_POINT_ALGORITHM,
        producer_revision=producer_revision)


def validate_local_source_declaration(value):
    """Validate the fixed local point-sampler declaration without bindings."""
    if not isinstance(value, Mapping) or set(value) != _LOCAL_SOURCE_DECLARATION_FIELDS:
        raise ValueError("local source declaration has an invalid exact field set")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("local source declaration schema must be integer 1")
    if value["kind"] != "local_generated":
        raise ValueError("local source declaration kind must be local_generated")
    expected = normalize_local_source_declaration(
        value["operation"], value["producer_revision"])
    if canonical_json(dict(value)) != canonical_json(expected):
        raise ValueError("local source declaration does not match the fixed sampler contract")
    return expected


def validate_local_spatial_point_spec(spec_document, declaration=None):
    """Validate the exact sampled-point product eligible for a schema-5 event."""
    declaration = (None if declaration is None else
                   validate_local_source_declaration(declaration))
    if not isinstance(spec_document, Mapping):
        raise ValueError("local generated point source needs a DatasetSpec")
    required = {"schema", "kind", "configuration", "axes", "dtype", "bases",
                "definition", "conventions", "parents"}
    if set(spec_document) != required or type(spec_document.get("schema")) is not int or spec_document["schema"] != 1:
        raise ValueError("local generated point source has an invalid DatasetSpec")
    if spec_document["kind"] != "spatial_points" or spec_document["dtype"] != "int64":
        raise ValueError("local source operation only supports int64 spatial_points")
    configuration = spec_document["configuration"]
    if not isinstance(configuration, Mapping) or set(configuration) != {
            "ensemble", "configuration", "lattice"}:
        raise ValueError("local generated point configuration is invalid")
    definition = spec_document["definition"]
    if not isinstance(definition, Mapping) or set(definition) != {"realization"}:
        raise ValueError("local generated point definition must contain its realization")
    realization = definition["realization"]
    if not isinstance(realization, Mapping) or set(realization) != {
            "schema", "kind", "design", "seed", "algorithm"}:
        raise ValueError("local generated point realization is invalid")
    design = realization["design"]
    if (type(realization["schema"]) is not int or realization["schema"] != 1 or
            realization["kind"] != "local_spatial_sampling" or
            not isinstance(design, Mapping) or set(design) != {
                "lattice_size", "num_points", "temporal_policy"}):
        raise ValueError("local generated point design is invalid")
    lattice = design["lattice_size"]
    count = design["num_points"]
    if (not isinstance(lattice, list) or len(lattice) != 4 or
            any(type(item) is not int or item < 1 for item in lattice) or
            type(count) is not int or count < 1 or
            count > lattice[0] * lattice[1] * lattice[2] or
            not (design["temporal_policy"] in ("independent", "same") or
                 (type(design["temporal_policy"]) is int and
                  1 <= design["temporal_policy"] <= lattice[3]))):
        raise ValueError("local generated point design has invalid geometry or policy")
    if configuration["lattice"] != lattice:
        raise ValueError("local generated point design differs from its configuration")
    seed = realization["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("local generated point seed must be a nonnegative integer")
    if (realization["algorithm"] != _LOCAL_POINT_ALGORITHM or
            (declaration is not None and
             realization["algorithm"] != declaration["algorithm"])):
        raise ValueError("local generated point algorithm differs from its source declaration")
    selection = dataset_time_selection(spec_document)
    expected_axes = [
        dict(name="time", coordinates=list(range(lattice[3] if selection is None else selection.Nt))),
        dict(name="point", coordinates=list(range(count))),
        dict(name="xyz", coordinates=["x", "y", "z"]),
    ]
    expected_conventions = {"representation": "coordinate_delta", "normalization": "unit_delta"}
    if selection is not None:
        expected_conventions["time_coordinates"] = spec_document["conventions"]["time_coordinates"]
    if (canonical_json(spec_document["axes"]) != canonical_json(expected_axes) or
            spec_document["bases"] != {} or spec_document["parents"] != [] or
            canonical_json(spec_document["conventions"]) != canonical_json(expected_conventions)):
        raise ValueError("local generated point axes, parents or conventions are invalid")
    return declaration


def validate_local_source_event(event, *, data_id, definition_id,
                                execution, spec_document):
    """Bind a completed local sampler event to one actual schema-5 attempt."""
    if not isinstance(event, Mapping) or set(event) != _LOCAL_SOURCE_EVENT_FIELDS:
        raise ValueError("local source event has an invalid exact field set")
    declaration = validate_local_source_declaration({
        key: event[key] for key in _LOCAL_SOURCE_DECLARATION_FIELDS
    })
    validate_local_spatial_point_spec(spec_document, declaration)
    for field, expected in (("data_id", data_id), ("definition_id", definition_id)):
        if (not isinstance(expected, str) or not expected or
                not isinstance(event[field], str) or event[field] != expected):
            raise ValueError(f"local source event {field} does not match its dataset")
    for field in ("run_id", "producer_attempt_id", "writer_session_id"):
        if not isinstance(event[field], str) or not event[field]:
            raise ValueError(f"local source event {field} must be nonempty")
    if event["stage"] != "points":
        raise ValueError("local point source event must bind the points graph stage")
    if (not isinstance(execution, Mapping) or
            type(execution.get("schema")) is not int or execution["schema"] != 5 or
            not isinstance(execution.get("attempts"), list) or not execution["attempts"]):
        raise ValueError("local source event needs a schema-5 execution history")
    attempt = execution["attempts"][-1]
    if (attempt.get("status") != "complete" or
            attempt.get("origin") != "dataset_graph" or
            attempt.get("stage") != event["stage"] or
            attempt.get("run_id") != event["run_id"] or
            attempt.get("producer_attempt_id") != event["producer_attempt_id"] or
            attempt.get("writer_session_id") != event["writer_session_id"]):
        raise ValueError("local source event does not identify the completed producer attempt")
    return dict(status="recorded", origin=dict(kind="local_generated",
        operation=declaration["operation"],
        producer=dict(status="known", value=declaration["producer"]),
        algorithm=dict(status="known", value=declaration["algorithm"]),
        producer_revision=dict(status="known", value=declaration["producer_revision"])))


def source_origin_from_registration(provenance, execution, *, data_id=None,
                                   definition_id=None, spec_document=None):
    """Parse a recorded source claim after its registration envelope validates."""
    if execution is None:
        return dict(status="unknown", reason="legacy-unstructured-provenance")
    if not isinstance(execution, Mapping):
        return dict(status="unknown", reason="legacy-unstructured-provenance")
    schema = execution.get("schema")
    if type(schema) is int and schema in (1, 2, 4):
        if schema == 4 and "registration_event" in execution:
            raise ValueError("unsupported external registration execution schema")
        return dict(status="unknown", reason="legacy-unstructured-provenance")
    if type(schema) is int and schema == 5:
        if "local_source_event" not in execution:
            raise ValueError("schema-5 execution has no local source event")
        return validate_local_source_event(execution["local_source_event"],
            data_id=data_id, definition_id=definition_id,
            execution=execution, spec_document=spec_document)
    if type(schema) is not int or schema != 3:
        raise ValueError("unsupported external registration execution schema")
    validated = validate_external_import_registration(provenance, execution,
        data_id=data_id, definition_id=definition_id)
    return dict(status="recorded", origin=validated["identity"]["origin"])


def source_origin_is_fully_known(source_origin):
    """Return whether a parsed registered source has every required known field."""
    if not isinstance(source_origin, Mapping) or set(source_origin) != {"status", "origin"}:
        return False
    origin = source_origin["origin"]
    if source_origin["status"] != "recorded" or not isinstance(origin, Mapping):
        return False
    if origin.get("kind") == "local_generated":
        fields = {"kind", "operation", "producer", "algorithm", "producer_revision"}
        return (set(origin) == fields and origin["operation"] == _LOCAL_POINT_OPERATION and
            all(isinstance(origin[name], Mapping) and origin[name].get("status") == "known"
                and set(origin[name]) == {"status", "value"} and
                isinstance(origin[name].get("value"), str) and origin[name]["value"].strip()
                for name in ("producer", "algorithm", "producer_revision")) and
            origin["producer"]["value"] == _LOCAL_POINT_PRODUCER and
            origin["algorithm"]["value"] == _LOCAL_POINT_ALGORITHM and
            re.fullmatch(r"[0-9a-f]{64}", origin["producer_revision"]["value"]) is not None)
    return (set(origin) == _ORIGIN_FIELDS and
            all(isinstance(origin[name], Mapping) and origin[name].get("status") == "known"
                and set(origin[name]) == {"status", "value"} and
                isinstance(origin[name].get("value"), str) and origin[name]["value"].strip()
                for name in _ORIGIN_FIELDS))


def validate_registration_runtime(value):
    """Validate a runtime snapshot embedded in a registration event."""
    if not isinstance(value, Mapping):
        raise TypeError("registration runtime must be a JSON mapping")
    status = value.get("status")
    if status == "known":
        if set(value) != {"status", "value"} or not isinstance(value["value"], Mapping):
            raise ValueError("known registration runtime needs a mapping value")
        try:
            normalized = json.loads(canonical_json(dict(value["value"])))
        except (TypeError, ValueError) as exc:
            raise ValueError("registration runtime must contain legal JSON values") from exc
        if not normalized:
            raise ValueError("known registration runtime must be nonempty")
        return dict(status="known", value=normalized)
    if status == "unknown":
        if set(value) != {"status", "reason"}:
            raise ValueError("unknown registration runtime needs status and reason only")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("registration runtime unknown reason must be nonempty")
        return dict(status="unknown", reason=reason)
    raise ValueError("registration runtime status must be 'known' or 'unknown'")
