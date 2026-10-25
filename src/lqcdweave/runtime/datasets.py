"""Compose verified storage observations with pure dataset requirement planning."""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import uuid

from lqcdweave.contracts.dataset import (
    DatasetRequest, DatasetSpec, canonical_json,
)
from lqcdweave.contracts.source import source_origin_is_fully_known
from lqcdweave.planning.dataset import plan_requirement, plan_dataset_graph
from lqcdweave.runtime.dataset_stage import run_dataset_graph
from lqcdweave.storage.dataset import (
    DatasetView as StorageDatasetView, read_json_records, write_json_records,
)


@contextmanager
def _index_view(index):
    """Reuse a live storage transaction without storing lock state on the index."""
    if getattr(index, "is_transaction_view", False):
        yield index
        return
    transaction = getattr(index, "transaction", None)
    if transaction is None:
        yield index
        return
    with transaction() as view:
        yield view


def plan_dataset_request(index, spec, *, subset_axes=(), data_id=None,
                          project_scoped=False):
    """Query explicit registered locations without producing missing data."""
    if project_scoped:
        if data_id is None:
            raise ValueError("project-scoped planning requires a selected actual dataset ID")
        try:
            with index.open(data_id) as reader:
                observation = reader.observation()
        except KeyError:
            observation = None
        observations = () if observation is None else (observation,)
    else:
        observations = index.observations()
    return plan_requirement(spec, observations, subset_axes=subset_axes,
                            data_id=data_id)


class DatasetView(StorageDatasetView):
    """Scientifically admitted storage view; selection and ownership are inherited."""

    def __init__(self, reader, spec, *, subset_axes=()):
        coverage = getattr(reader, "coverage_observation", reader.observation)()
        plan = plan_requirement(spec, (coverage,), subset_axes=subset_axes,
                                data_id=reader.data_id)
        if plan["action"] != "reuse":
            raise ValueError("reader does not cover the requested scientific dataset view")
        super().__init__(reader, spec)


def open_dataset_view(index, spec, *, subset_axes=(), data_id=None,
                       project_scoped=False):
    """Open a complete compatible view or report blocked/missing coverage.

    A compute plan is available through plan_dataset_request; it cannot be
    opened as if the whole requested coverage were already complete.
    """
    plan = plan_dataset_request(index, spec, subset_axes=subset_axes,
                                 data_id=data_id,
                                 project_scoped=project_scoped)
    if plan["action"] != "reuse":
        raise ValueError(f"dataset request {plan['action']}: {plan['reasons']}; "
                         f"missing first-axis positions {plan['missing']}")
    reader = index.open_selected(plan["data_id"],
                                 spec.coordinates(spec.axis_names[0]))
    try:
        return DatasetView(reader, spec, subset_axes=subset_axes)
    except BaseException:
        reader.close()
        raise


def run_dataset_requests(requests, index, outputs, provenance, producers, *,
                          resume=False, data_ids=None, materialize_reused_outputs=(),
                          execution_parameters=None, candidate_data_ids=None):
    """Compose a pure request graph with verified views and stage execution.

    Producers are explicit name-to-callable bindings; each callable writes only
    writer.missing rows and must consume borrowed buffers before returning.
    The runtime owns publication and index registration. No file discovery or
    numerical producer selection is inferred from dataset kind or filenames.
    ``execution_parameters`` maps request names to JSON objects recorded only
    when the corresponding producer actually runs.
    ``candidate_data_ids`` optionally restricts automatic full-dataset
    candidates to that explicit tuple. Direct request pins, graph outputs and
    verified resume paths retain their ordinary behavior; omission preserves
    the global-index candidate query.
    """
    def open_view(spec, identity, *, subset_axes):
        return open_dataset_view(index, spec, subset_axes=subset_axes,
                                  data_id=identity,
                                  project_scoped=(candidate_data_ids is not None))
    options = dict(resume=resume, data_ids=data_ids,
                   materialize_reused_outputs=materialize_reused_outputs)
    if execution_parameters is not None:
        options["execution_parameters"] = execution_parameters
    if candidate_data_ids is not None:
        options["candidate_data_ids"] = candidate_data_ids
    return run_dataset_graph(requests, index, outputs, provenance, producers,
                              open_view, **options)


def _source_reference_ids(spec):
    """Collect only contract-declared actual IDs from one dataset spec."""
    document = spec.document
    references = set(document["parents"])
    references.update(basis["data_id"] for basis in document["bases"].values()
                      if basis["data_id"] is not None)

    def walk_declared(value):
        if isinstance(value, dict):
            roles = value.get("input_roles")
            if isinstance(roles, dict):
                references.update(identity for identity in roles.values()
                                  if isinstance(identity, str) and identity)
            views = value.get("input_views")
            if isinstance(views, dict):
                for view in views.values():
                    if isinstance(view, dict) and set(view) == {
                            "schema", "kind", "configuration", "axes", "dtype",
                            "bases", "definition", "conventions", "parents"}:
                        walk_spec(DatasetSpec(view))
                    else:
                        walk_declared(view)
            for key, child in value.items():
                if key not in ("input_roles", "input_views"):
                    walk_declared(child)
        elif isinstance(value, list):
            for child in value:
                walk_declared(child)

    def walk_spec(embedded):
        nested = embedded.document
        references.update(nested["parents"])
        references.update(basis["data_id"] for basis in nested["bases"].values()
                          if basis["data_id"] is not None)
        walk_declared(nested["definition"])

    walk_declared(document["definition"])
    references.discard("")
    return tuple(sorted(references))


def _source_lineage(index, roots):
    """Walk declared lineage from current dataset metadata."""
    reasons = []
    origins = {}
    completed = set()
    active = []

    def visit(identity):
        if identity in active:
            reasons.append("actual dataset parent cycle: " + " -> ".join(active + [identity]))
            return
        if identity in completed:
            return
        try:
            metadata = _source_metadata_record(index, identity)
        except Exception as exc:
            reasons.append(
                f"actual source metadata {identity!r} cannot be read ({type(exc).__name__})")
            completed.add(identity)
            return
        origin = dict(metadata["source_origin"])
        execution = metadata["execution"]
        spec = metadata["spec"]
        origins[identity] = origin
        source_event = (execution is not None and
                        type(execution.get("schema")) is int and
                        execution.get("schema") in (3, 5))
        if source_event and not source_origin_is_fully_known(origin):
            reasons.append(f"external source {identity!r} has unknown origin fields")
        try:
            dependencies = _source_reference_ids(spec)
        except Exception as exc:
            reasons.append(
                f"actual source metadata {identity!r} has invalid declared references ({type(exc).__name__})")
            dependencies = ()
        if not dependencies and (not source_event or not source_origin_is_fully_known(origin)):
            reasons.append(f"terminal actual dataset {identity!r} has no fully known external origin")
        active.append(identity)
        for parent in dependencies:
            visit(parent)
        active.pop()
        completed.add(identity)

    for identity in roots:
        visit(identity)
    return origins, tuple(dict.fromkeys(reasons))

def _source_request_bindings(preview, requests):
    """Keep each root request's resolved ID and view local."""
    stages = preview.get("stages", {})
    bindings = []
    for name, request in requests.items():
        if request.dependencies:
            continue
        stage = stages.get(name, {})
        identity = request.data_id or stage.get("data_id")
        if identity is None:
            continue
        spec = stage.get("spec")
        if not isinstance(spec, DatasetSpec):
            try:
                spec = request.resolve({})
            except Exception:
                spec = None
        bindings.append(dict(
            role=name,
            data_id=identity,
            request_spec=spec,
            subset_axes=request.subset_axes,
        ))
    return tuple(bindings)


def _source_metadata_record(index, identity):
    """Read the current source declaration and its dataset specification."""
    observe = getattr(index, "observe_metadata", None)
    if not callable(observe):
        raise TypeError("storage index has no public source metadata observation")
    metadata = observe(identity)
    if not isinstance(metadata, Mapping):
        raise TypeError("source metadata observation must be a mapping")
    if metadata.get("state") != "metadata":
        raise ValueError("source metadata observation has the wrong state")
    if metadata.get("data_id") != identity:
        raise ValueError("source metadata observation has the wrong actual dataset ID")
    if not isinstance(metadata.get("spec"), DatasetSpec):
        raise ValueError("source metadata observation has no DatasetSpec")
    if "execution" not in metadata:
        raise ValueError("source metadata observation has no execution record")
    if metadata["execution"] is not None and not isinstance(metadata["execution"], Mapping):
        raise ValueError("source metadata observation has an invalid execution record")
    if not isinstance(metadata.get("source_origin"), Mapping):
        raise ValueError("source metadata observation has no source-origin record")
    return metadata


def apply_source_policy(preview, requests, index, *, source_policy="allow_unknown"):
    """Attach an explicit source-lineage result to a dataset graph plan.

    Roots are dependency-free requests with a pinned or planned actual ID.
    Declared lineage follows DatasetSpec parents,
    bases, explicit input roles and embedded input views, never arbitrary IDs.
    """
    if source_policy not in ("allow_unknown", "require_known"):
        raise ValueError("source_policy must be 'allow_unknown' or 'require_known'")
    if source_policy == "allow_unknown":
        return preview
    bindings = _source_request_bindings(preview, requests)
    invalid_ids = [binding for binding in bindings
                   if not isinstance(binding["data_id"], str) or
                   not binding["data_id"]]
    reasons = []
    origins = {}
    if invalid_ids:
        reasons.extend(
            f"source request {binding['role']!r} has no valid actual dataset ID"
            for binding in invalid_ids)

    source_origins, source_reasons = _source_lineage(
        index, tuple(sorted({binding["data_id"] for binding in bindings
                             if isinstance(binding["data_id"], str) and
                             binding["data_id"]})))
    origins.update(source_origins)
    reasons.extend(source_reasons)
    if not bindings:
        reasons.insert(0, "no explicitly pinned external source roots are present")
    result = dict(preview)
    result["source_policy"] = source_policy
    result["source_origins"] = origins
    result["source_status"] = "unknown" if reasons else "known"
    if reasons:
        result["action"] = "blocked"
        result["source_reasons"] = tuple(dict.fromkeys(reasons))
        result["reasons"] = tuple(preview.get("reasons", ())) + result["source_reasons"]
    return result

def validate_project_id(project_id):
    """Validate an optional project receipt label before workflow side effects."""
    if project_id is not None and (not isinstance(project_id, str) or not project_id.strip()):
        raise ValueError("project_id must be an explicit nonempty string")
    return project_id


def _effective_receipt_ids(requests, data_ids):
    if data_ids is None:
        supplied = {}
    elif not isinstance(data_ids, Mapping):
        raise TypeError("data_ids must map request names to actual dataset IDs")
    else:
        supplied = dict(data_ids)
    unknown = set(supplied) - set(requests)
    if unknown:
        raise ValueError(f"data_ids names unknown request {sorted(unknown)[0]!r}")
    identities = {}
    for name, request in requests.items():
        if not isinstance(name, str) or not name or not isinstance(request, DatasetRequest):
            raise TypeError("receipt requests must map nonempty names to DatasetRequest values")
        pinned = request.data_id
        actual = supplied.get(name, pinned)
        if pinned is not None and actual is not None and pinned != actual:
            raise ValueError(f"receipt actual IDs conflict for request {name!r}")
        if not isinstance(actual, str) or not actual:
            raise ValueError(f"receipt request {name!r} needs a complete actual dataset ID")
        identities[name] = actual
    return identities


def _receipt_path(path):
    if not isinstance(path, (str, os.PathLike)) or isinstance(path, bytes):
        raise TypeError("receipt path must be an explicit string or Path")
    return Path(path).expanduser().resolve(strict=False)


def _receipt_path_key(path):
    return os.path.normcase(os.path.normpath(str(path)))


def _detach_receipt_metadata(value):
    """Copy receipt/planner metadata without retaining readers or array views."""
    if isinstance(value, DatasetSpec):
        return DatasetSpec(value.document)
    if isinstance(value, Mapping):
        return {key: _detach_receipt_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_receipt_metadata(item) for item in value)
    if isinstance(value, list):
        return [_detach_receipt_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"receipt result contains unsupported metadata {type(value).__name__}")


def _collect_receipt_graph(requests, index, data_ids, reader_cache,
                           reader_stack, *, failure_message):
    identities = _effective_receipt_ids(requests, data_ids)
    observations = {}
    for identity in sorted(set(identities.values())):
        reader = reader_cache.get(identity)
        if reader is None:
            reader = reader_stack.enter_context(index.open(identity))
            if reader.data_id != identity:
                raise ValueError("receipt reader has the wrong actual dataset ID")
            reader_cache[identity] = reader
        observations[identity] = reader.observation()

    graph = plan_dataset_graph(requests, tuple(observations.values()),
                                data_ids=identities)
    if any(graph["stages"][name]["action"] != "reuse" for name in graph["order"]):
        raise ValueError(failure_message)

    source_origins = {}
    for name in graph["order"]:
        request = requests[name]
        stage = graph["stages"][name]
        identity = identities[name]
        spec = stage["spec"]
        if stage["data_id"] != identity:
            raise ValueError("dataset graph receipt actual ID differs from the verified plan")
        reader = reader_cache[identity]
        DatasetView(reader, spec, subset_axes=request.subset_axes)
        source_origins[name] = dict(reader.source_origin)

    return dict(
        identities=identities,
        graph=graph,
        dataset_observations=tuple(observations.values()),
        source_origins=source_origins,
    )


def _save_project_receipt(directory, requests, index, data_ids, project_id):
    if validate_project_id(project_id) is None:
        raise ValueError("project_id must be an explicit nonempty string")
    with ExitStack() as reader_stack:
        collected = _collect_receipt_graph(
            requests, index, data_ids, {}, reader_stack,
            failure_message="only a complete compatible graph can produce a project receipt")
        identities, graph = collected["identities"], collected["graph"]
        records = []
        for name in graph["order"]:
            request = requests[name]
            spec = graph["stages"][name]["spec"]
            records.append(dict(schema=4, kind="dataset_graph_receipt", node=name,
                project_id=project_id, request=request.template,
                subset_axes=list(request.subset_axes), spec=spec.document,
                data_id=identities[name], requires=list(graph["stages"][name]["requires"]),
                source_origin=collected["source_origins"][name]))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("graph-" + uuid.uuid4().hex + ".jsonl.gz")
    bump = getattr(index, "bump_publication_revision", None)
    if bump is not None:
        bump()
    write_json_records(path, records)
    return path


def _read_dataset_request_receipt(path, index, *, expected_project_id,
                                   reader_cache, reader_stack):
    records = list(read_json_records(path))
    if not records:
        raise ValueError("dataset graph receipt cannot be empty")
    schemas = {record.get("schema") if isinstance(record, Mapping) else None
               for record in records}
    if len(schemas) != 1:
        raise ValueError("dataset graph receipt records must use one schema")
    schema = next(iter(schemas))
    if type(schema) is not int or schema not in (1, 4):
        raise ValueError("unsupported dataset graph receipt schema")
    base_fields = {"schema", "kind", "node", "request", "subset_axes",
                   "spec", "data_id", "requires"}
    expected_fields = (base_fields if schema == 1 else
        base_fields | {"project_id", "source_origin"})
    project_id = None
    names = []
    requests = {}
    data_ids = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError("dataset graph receipt record has an invalid exact field set")
        if (type(record["schema"]) is not int or record["schema"] != schema or
                record["kind"] != "dataset_graph_receipt"):
            raise ValueError("invalid dataset graph receipt node")
        name = record["node"]
        if not isinstance(name, str) or not name or name in requests:
            raise ValueError("dataset graph receipt node names must be unique nonempty strings")
        names.append(name)
        identity = record["data_id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("dataset graph receipt needs an actual dataset ID")
        data_ids[name] = identity
        subset_axes = record["subset_axes"]
        requires = record["requires"]
        if (not isinstance(subset_axes, list) or
                any(not isinstance(axis, str) or not axis for axis in subset_axes) or
                len(set(subset_axes)) != len(subset_axes) or
                not isinstance(requires, list) or
                any(not isinstance(parent, str) or not parent for parent in requires) or
                len(set(requires)) != len(requires)):
            raise ValueError("invalid dataset graph receipt axes or dependency list")
        requests[name] = DatasetRequest(record["request"], subset_axes=subset_axes,
            data_id=identity)
        if list(requests[name].dependencies) != requires:
            raise ValueError("receipt dependencies differ from the request template")
        if schema == 4:
            current_project = record["project_id"]
            if not isinstance(current_project, str) or not current_project.strip():
                raise ValueError("project receipt needs a nonempty project_id")
            if (expected_project_id is not None and
                    current_project != expected_project_id):
                raise ValueError("dataset graph receipt does not match the expected project_id")
            if project_id is None:
                project_id = current_project
            elif project_id != current_project:
                raise ValueError("project receipt nodes disagree on project_id")
    if expected_project_id is not None:
        if schema != 4 or project_id != expected_project_id:
            raise ValueError("dataset graph receipt does not match the expected project_id")

    collected = _collect_receipt_graph(
        requests, index, data_ids, reader_cache, reader_stack,
        failure_message="only a complete compatible graph can produce a project receipt")
    identities, graph = collected["identities"], collected["graph"]
    if tuple(names) != graph["order"]:
        raise ValueError("dataset graph receipt nodes are not in canonical dependency order")
    for record in records:
        name = record["node"]
        stage = graph["stages"][name]
        if canonical_json(stage["spec"].document) != canonical_json(record["spec"]):
            raise ValueError("dataset graph receipt spec differs from the resolved request")
        if list(stage["requires"]) != record["requires"]:
            raise ValueError("dataset graph receipt dependencies differ from the resolved request")
        source_origin = collected["source_origins"][name]
        if schema == 4 and canonical_json(source_origin) != canonical_json(record["source_origin"]):
            raise ValueError("project receipt source origin differs from the indexed dataset")

    detached_graph = _detach_receipt_metadata(graph)
    full_observations = tuple(_detach_receipt_metadata(item)
                              for item in collected["dataset_observations"])
    return dict(schema=schema, project_id=project_id, records=records,
        project_id_verified=(expected_project_id is not None),
        source_origins=_detach_receipt_metadata(collected["source_origins"]),
        receipt_path=str(path), resolved_graph=detached_graph,
        dataset_observations=full_observations)


def read_dataset_request_receipt(path, index, *, expected_project_id=None):
    """Revalidate one graph receipt and return its verified planner graph.

    The project association is checked when ``expected_project_id`` is given.
    Receipt structure and every referenced dataset are revalidated before
    returning detached metadata.
    """
    normalized_path = _receipt_path(path)
    validate_project_id(expected_project_id)
    with _index_view(index) as index_view:
        with ExitStack() as reader_stack:
            return _read_dataset_request_receipt(
                normalized_path, index_view, expected_project_id=expected_project_id,
                reader_cache={}, reader_stack=reader_stack)


def read_dataset_request_receipts(paths, index, *, expected_project_id=None):
    """Read distinct explicit receipts with a reader cache scoped to this call."""
    validate_project_id(expected_project_id)
    if (not isinstance(paths, Sequence) or
            isinstance(paths, (str, bytes, os.PathLike))):
        raise TypeError("paths must be an explicit sequence of receipt paths")
    normalized = []
    seen = set()
    for path in paths:
        resolved = _receipt_path(path)
        key = _receipt_path_key(resolved)
        if key not in seen:
            seen.add(key)
            normalized.append((resolved, key))

    with _index_view(index) as index_view:
        with ExitStack() as reader_stack:
            reader_cache = {}
            return tuple(
                _read_dataset_request_receipt(
                    path, index_view, expected_project_id=expected_project_id,
                    reader_cache=reader_cache, reader_stack=reader_stack)
                for path, key in normalized)


def _save_dataset_request_receipt_in_transaction(directory, requests, index,
                                                  data_ids, *, project_id=None):
    """Save immutable project definitions, actual inputs and output references.

    Every referenced dataset is first revalidated through the public index.
    A new graph realization gets a new receipt, preserving earlier definitions.
    """
    validate_project_id(project_id)
    if project_id is not None:
        return _save_project_receipt(directory, requests, index, data_ids, project_id)
    observations = index.observations()
    graph = plan_dataset_graph(requests, observations, data_ids=data_ids)
    if any(stage["action"] != "reuse" for stage in graph["stages"].values()):
        raise ValueError("only a complete compatible graph can produce a receipt")
    records = [dict(schema=1, kind="dataset_graph_receipt", node=name,
        request=requests[name].template, subset_axes=list(requests[name].subset_axes),
        spec=graph["stages"][name]["spec"].document,
        data_id=graph["bindings"][name], requires=list(graph["stages"][name]["requires"]))
        for name in graph["order"]]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("graph-" + uuid.uuid4().hex + ".jsonl.gz")
    bump = getattr(index, "bump_publication_revision", None)
    if bump is not None:
        bump()
    write_json_records(path, records)
    return path


def save_dataset_request_receipt(directory, requests, index, data_ids, *,
                                  project_id=None):
    """Publish a receipt under the same index gate used by retention plans."""
    with _index_view(index) as index_view:
        return _save_dataset_request_receipt_in_transaction(
            directory, requests, index_view, data_ids, project_id=project_id)
