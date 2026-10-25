"""Explicit offline conversion of string-based compiled DAG archives.

This changes representation only. It does not expand tasks, choose paths,
read tensor data, estimate costs, or arrange storage.
"""
from copy import deepcopy
from dataclasses import fields, replace
import hmac

from . import api


def _signature_payload(payload):
    # Copied from C signatures.structural_payload's established bundle
    # encoding. Operate on JSON so v4 can be checked before constructing v5 IR.
    workload = deepcopy(payload["workload"])
    for spec in workload["specs"]:
        spec.pop("description", None)
    workload["metadata"] = {
        key: value for key, value in workload["metadata"].items()
        if key not in {"performance_fingerprint", "calibration"}
    }
    batches = []
    for batch in payload["batches"]:
        plan = deepcopy(batch["plan"])
        for spec in plan["specs"]:
            spec.pop("description", None)
        batches.append({"term_indices": batch["term_indices"], "plan": plan})
    return {"workload": workload, "batches": batches}


def _axes(value):
    if not isinstance(value, str) or not value.isascii() or (value and not value.isalpha()):
        raise api.PlanArchiveError("legacy compiled labels must be ASCII letters")
    return list(map(ord, value))


def migrate_plan_bundle(source, destination):
    """Validate a v4 bundle and publish the same DAG in the current format."""
    document = api._strict_object(api._read_document(source), name="archive")
    api._strict_keys(document, {"format", "version", "payload_type", "payload"}, name="archive")
    if document["payload_type"] == 'complete-cfg-plan':
        raise api.PlanArchiveError('old matrix/complete cfg plans have no migrator; generate a new plan')
    if (document["format"] != api.ARCHIVE_FORMAT or type(document["version"]) is not int
            or document["version"] != 4 or document["payload_type"] != "plan-bundle"):
        raise api.PlanArchiveError("offline conversion requires a v4 plan-bundle archive")
    payload = document["payload"]
    try:
        api._strict_keys(payload, {"signature", "workload", "batches", "budget", "metadata"}, name="archive payload")
        signature = api._digest(_signature_payload(payload))
        if not isinstance(payload["signature"], str) or not hmac.compare_digest(signature, payload["signature"]):
            raise api.PlanArchiveError("legacy plan archive structural signature mismatch")
        for term in payload["workload"]["terms"]:
            inputs, output = api._parse_equation(term["subscripts"], name="legacy workload equation")
            term["subscripts"] = {"inputs": [_axes(axes) for axes in inputs], "output": _axes(output)}
        for batch in payload["batches"]:
            plan = batch["plan"]
            for operation in plan["operations"]:
                operation["input_subscripts"] = [_axes(axes) for axes in operation["input_subscripts"]]
                operation["output_subscript"] = _axes(operation["output_subscript"])
            for term in plan["terms"]:
                term["output_subscript"] = _axes(term["output_subscript"])
        payload["signature"] = api._digest(_signature_payload(payload))
        bundle = api._decode_plan_bundle_payload(payload, expected_signature=None, expected_workload=None)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, api.PlanArchiveError):
            raise
        raise api.PlanArchiveError(f"invalid legacy plan bundle: {exc}") from exc
    return api.save_plan(bundle, destination)


def _pattern(value):
    return api._encode_plan_value(api.contract_api.parse_axis_pattern(value))


def _axis_tuple(value):
    return api._encode_plan_value(api.contract_api.axis_ids(value))


def _tuple_items(value):
    if isinstance(value, dict) and value.get("type") in ("tuple", "python.tuple"):
        return value["value"]
    return None


def _topology(value):
    """Convert only declared Diagram, structured and matrix topology records."""
    parts = _tuple_items(value)
    if not parts:
        return value
    if parts[0] == "structured-v1" and len(parts) == 3:
        parts[1] = _topology(parts[1])
        return value
    if parts[0] == "meson-rows-v1" and len(parts) == 2:
        rows = _tuple_items(parts[1])
        if rows is None:
            raise api.PlanArchiveError("invalid legacy meson topology")
        for row in rows:
            fields = _tuple_items(row)
            if fields is None or len(fields) != 2:
                raise api.PlanArchiveError("invalid legacy meson topology row")
            if isinstance(fields[1], str):
                fields[1] = api._encode_plan_value(api.contract_api.parse_axis_pattern(fields[1]).normalized())
        return value
    rows = _tuple_items(parts[0])
    if rows is None:
        return value
    for row in rows:
        fields = _tuple_items(row)
        if fields and len(fields) == 2 and isinstance(fields[0], str) and type(fields[1]) is int:
            fields[0] = api._encode_plan_value(api.contract_api.parse_axis_pattern(fields[0]).normalized())
    return value


def _batch_key(value):
    parts = _tuple_items(value)
    if parts is None or len(parts) != 6:
        raise api.PlanArchiveError("invalid legacy batch compatibility key")
    parts[0] = _topology(parts[0])
    rows = _tuple_items(parts[4])
    if rows is None:
        raise api.PlanArchiveError("invalid legacy batch term rows")
    for row in rows:
        fields = _tuple_items(row)
        if fields is None or len(fields) not in (3, 4):
            raise api.PlanArchiveError("invalid legacy batch term key")
        if isinstance(fields[1], str):
            fields[1] = api._encode_plan_value(api.contract_api.parse_axis_pattern(fields[1]).normalized())
        if len(fields) == 3:
            fields.append(None)
    return value


def _records(value):
    """Change declared connection fields, leaving source/physical text intact."""
    if isinstance(value, list):
        return [_records(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("type")
    payload = value.get("value")
    if isinstance(kind, str) and isinstance(payload, dict):
        name = kind.rsplit(".", 1)[-1]
        fields = {
            "Factor": ("labels",), "PendingTerm": ("output",),
            "BaryonJoinSpec": ("spin_labels", "mode_labels"),
        }.get(name, ())
        for field in fields:
            if isinstance(payload.get(field), str):
                payload[field] = _axis_tuple(payload[field])
        if name == "ContractionTerm" and isinstance(payload.get("subscripts"), str):
            payload["subscripts"] = _pattern(payload["subscripts"])
        if name == "Action" and isinstance(payload.get("equation"), str):
            payload["equation"] = _pattern(payload["equation"]) if payload["equation"] else None
        if "topology_key" in payload:
            payload["topology_key"] = _topology(payload["topology_key"])
    return {key: _records(item) for key, item in value.items()}


def _problem(problem):
    if problem is None:
        return
    for term in problem["terms"]:
        if isinstance(term["subscripts"], str):
            term["subscripts"] = api.contract_api.parse_axis_pattern(term["subscripts"]).as_dict()


def _task(task):
    _problem(task["problem"])
    task["batch_topology"] = _topology(task["batch_topology"])


def _migrate_storage_dataset_field(document, destination):
    """Rename only the retired StorageBlock field in a generic v8 schedule."""
    payload = api._decode_envelope(document, expected_type="storage-schedule")
    api._strict_keys(payload,
        {"shapes", "semantics", "hardware", "profile_source", "blocks", "plans", "metadata"},
        name="storage schedule payload")
    block_type = api.contract_api.StorageBlock
    tag = f"{block_type.__module__}.{block_type.__name__}"
    old_fields = {field.name for field in fields(block_type)} - {"dataset"} | {"artifact"}
    renamed = 0
    # Iterate the freshly read JSON in place. Do not duplicate the large plan,
    # change opaque identities/metadata, or rewrite similarly named user fields.
    stack = [iter((payload["blocks"], payload["plans"]))]
    end = object()
    while stack:
        item = next(stack[-1], end)
        if item is end:
            stack.pop()
        elif isinstance(item, list):
            stack.append(iter(item))
        elif isinstance(item, dict):
            if item.get("type") == tag:
                value = item.get("value")
                if not isinstance(value, dict):
                    raise api.PlanArchiveError("StorageBlock fields must be a JSON object")
                if "artifact" in value:
                    if set(item) != {"type", "value"} or set(value) != old_fields:
                        raise api.PlanArchiveError("retired StorageBlock fields do not match the supported schema")
                    value["dataset"] = value.pop("artifact")
                    renamed += 1
            stack.append(iter(item.values()))
    if not renamed:
        raise api.PlanArchiveError("storage schedule has no retired StorageBlock artifact field")
    blocks = api._decode_plan_value(payload["blocks"])
    if not isinstance(blocks, dict) or not isinstance(payload["plans"], list):
        raise api.PlanArchiveError("invalid storage schedule blocks or plans")
    plans = []
    for item in payload["plans"]:
        plan = api._decode_plan_value(item)
        if not isinstance(plan, api.contract_api.SceneStoragePlan):
            raise api.PlanArchiveError("storage schedule has an invalid plan type")
        plans.append(replace(plan, blocks=blocks))
    api._validate_schedule(tuple(plans))
    return api._write_document(destination, document)


def migrate_plan_archive(source, destination):
    """Convert retired axes or a v8 storage dataset field, explicitly offline.

    No array reads, task expansion, path search, replay or resource scheduling
    occur. Storage events, kernel paths, coefficients and data IDs stay fixed.
    """
    document = api._strict_object(api._read_document(source), name="archive")
    api._strict_keys(document, {"format", "version", "payload_type", "payload"}, name="archive")
    if document["payload_type"] == 'complete-cfg-plan':
        raise api.PlanArchiveError('old matrix/complete cfg plans have no migrator; generate a new plan')
    if (document["format"] == api.ARCHIVE_FORMAT
            and type(document["version"]) is int and document["version"] == api.ARCHIVE_VERSION
            and document["payload_type"] == "storage-schedule"):
        return _migrate_storage_dataset_field(document, destination)
    if (document["format"] != api.ARCHIVE_FORMAT or type(document["version"]) is not int
            or document["version"] not in (4, 5)):
        raise api.PlanArchiveError("offline conversion requires a v4/v5 plan archive")
    kind = document["payload_type"]
    if kind == "plan-bundle":
        if document["version"] == 4:
            return migrate_plan_bundle(source, destination)
        bundle = api.load_plan(source)
        return api.save_plan(bundle, destination)
    if kind not in {"correlator-plan", "storage-schedule"}:
        raise api.PlanArchiveError("unsupported legacy plan payload type")
    payload = document["payload"]
    try:
        if kind == "correlator-plan":
            legacy_signature = api._digest(None if payload["task"]["problem"] is None else
                api._problem_path_invariants_from_payload(payload["task"]["problem"], normalize_axes=False))
            if not isinstance(payload["path_signature"], str) or not hmac.compare_digest(legacy_signature, payload["path_signature"]):
                raise api.PlanArchiveError("legacy path signature mismatch")
        payload = _records(payload)
        if kind == "storage-schedule":
            for problem in payload["semantics"]:
                _problem(problem)
            blocks = api._decode_plan_value(payload["blocks"], legacy=True)
            api._validate_schedule(tuple(replace(api._decode_plan_value(item, legacy=True), blocks=blocks)
                                         for item in payload["plans"]))
        else:
            _task(payload["task"])
            keys = _tuple_items(payload["batch_shape_keys"])
            if keys is None:
                raise api.PlanArchiveError("invalid legacy batch compatibility keys")
            for key in keys:
                _batch_key(key)
            for row in payload["plans"]:
                _task(row["task"])
                if row["batch_shape_key"] is not None:
                    _batch_key(row["batch_shape_key"])
                if row["storage_plan"] is not None:
                    api._decode_plan_value(row["storage_plan"], legacy=True)
            payload["path_signature"] = api._digest(None if payload["task"]["problem"] is None else
                api._problem_path_invariants_from_payload(payload["task"]["problem"]))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, api.PlanArchiveError):
            raise
        raise api.PlanArchiveError(f"invalid legacy plan archive: {exc}") from exc
    return api._write_document(destination, api._make_envelope(kind, payload))
