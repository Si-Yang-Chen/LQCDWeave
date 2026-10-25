"""Project receipt coverage and sequential execution over explicit graphs."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path

from lqcdweave.planning.dataset import plan_dataset_retention
from lqcdweave.contracts.dataset import (
    DatasetRequest, DatasetSpec, canonical_json,
)
from lqcdweave.contracts.execution import normalize_execution_parameters
from lqcdweave.planning.dataset import plan_dataset_graph


@contextmanager
def _retention_index_view(index):
    if getattr(index, "is_transaction_view", False):
        yield index
        return
    transaction = getattr(index, "transaction", None)
    if transaction is None:
        raise TypeError("retention requires a DatasetIndex transaction service")
    with transaction() as view:
        yield view


def _validate_query_arguments(requests, expected_project_id):
    if not isinstance(requests, Mapping):
        raise TypeError("requests must map names to DatasetRequest instances")
    for name, request in requests.items():
        if not isinstance(name, str) or not name:
            raise ValueError("request names must be nonempty strings")
        if not isinstance(request, DatasetRequest):
            raise TypeError(f"request {name!r} must be a DatasetRequest")
    if (not isinstance(expected_project_id, str) or
            not expected_project_id.strip()):
        raise ValueError("expected_project_id must be an explicit nonempty string")


def _receipt_observations(receipts):
    """Collect the complete datasets named by a validated receipt batch."""
    data_ids = set()
    observations = {}
    for receipt in receipts:
        data_ids.update(record["data_id"] for record in receipt["records"])
        for observation in receipt["dataset_observations"]:
            observations.setdefault(observation["data_id"], observation)
    return data_ids, tuple(observations[key] for key in sorted(observations))


def _read_project_receipt_scope(requests, index, receipts, *,
                                expected_project_id):
    _validate_query_arguments(requests, expected_project_id)

    # Import at call time so the shared receipt reader remains the sole parser
    # and can be supplied by the workflow integration boundary.
    from lqcdweave.runtime.datasets import read_dataset_request_receipts

    verified_receipts = read_dataset_request_receipts(
        receipts, index, expected_project_id=expected_project_id,
    )
    verified_receipts = tuple(verified_receipts)
    data_ids, observations = _receipt_observations(verified_receipts)
    return dict(receipts=verified_receipts, data_ids=data_ids,
                observations=observations)


def _plan_project_graph(requests, index, scope, *, data_ids=None,
                        extra_full_ids=(), extra_full_observations=()):
    candidate_ids = set(scope["data_ids"]) | set(extra_full_ids)
    observations_by_id = {
        item["data_id"]: item for item in scope["observations"]
    }
    for observation in extra_full_observations:
        observations_by_id.setdefault(observation["data_id"], observation)
    full_observations = tuple(observations_by_id[key]
                              for key in sorted(observations_by_id))
    return plan_dataset_graph(
        requests, full_observations, data_ids=data_ids,
        candidate_data_ids=tuple(sorted(candidate_ids)))


def _project_plan_metadata(graph, expected_project_id, receipts):
    return dict(graph,
        project_id=expected_project_id,
        project_id_verified=bool(receipts),
    )


def plan_project_dataset_requests(requests, index, receipts, *,
                                   expected_project_id):
    """Plan arbitrary requests using only explicitly supplied project receipts.

    Receipt files are first revalidated by the shared workflow batch reader.
    Receipt nodes supply complete candidates. No global index enumeration or
    receipt discovery is performed.
    """
    scope = _read_project_receipt_scope(requests, index, receipts,
        expected_project_id=expected_project_id)
    graph = _plan_project_graph(requests, index, scope)

    # The final loop iteration already planned with every currently
    # authorized exact selected view. The public planner owns compute/blocked,
    # axis coverage and dependency propagation semantics.
    return _project_plan_metadata(graph, expected_project_id, scope["receipts"])


def run_project_dataset_requests(requests, index, outputs, provenance, producers,
                                  receipts, *, expected_project_id,
                                  source_policy="allow_unknown", resume=False,
                                  data_ids=None, materialize_reused_outputs=(),
                                  execution_parameters=None):
    """Execute one graph using only the explicitly supplied project candidates.

    Receipt contents are freshly revalidated before graph execution. Candidates
    are limited to datasets named by those receipts;
    explicit request/data_ids pins remain caller-authorized inputs. The
    standard graph receipt is written separately with
    :func:`save_dataset_request_receipt` after this function succeeds.
    """
    scope = _read_project_receipt_scope(requests, index, receipts,
        expected_project_id=expected_project_id)

    if data_ids is None:
        explicit_ids = {}
    elif isinstance(data_ids, Mapping):
        explicit_ids = dict(data_ids)
    else:
        raise TypeError("data_ids must map request names to actual dataset IDs")
    explicit_full_ids = set()
    receipt_ids = set(scope["data_ids"])
    extra_full_observations = []
    for name, request in requests.items():
        identity = explicit_ids.get(name, request.data_id)
        if (name in explicit_ids and request.data_id is not None and
                explicit_ids[name] != request.data_id):
            raise ValueError(f"request {name!r} has conflicting actual dataset IDs")
        if identity is not None and identity not in receipt_ids:
            explicit_full_ids.add(identity)

    for identity in sorted(explicit_full_ids):
        with index.open(identity) as reader:
            extra_full_observations.append(reader.observation())

    graph = _plan_project_graph(requests, index, scope,
        data_ids=explicit_ids,
        extra_full_ids=explicit_full_ids,
        extra_full_observations=extra_full_observations)

    from lqcdweave.runtime.datasets import apply_source_policy

    preview = apply_source_policy(graph, requests, index,
                                  source_policy=source_policy)
    if preview.get("action") == "blocked" and preview.get("source_reasons"):
        raise ValueError("strict source policy blocked project dataset execution: " +
                         "; ".join(preview["source_reasons"]))
    for name, request in requests.items():
        if (request.data_id is not None or name in explicit_ids) and \
                preview["stages"][name]["action"] != "reuse":
            raise ValueError(f"fixed input {name!r} lacks compatible complete coverage")

    from lqcdweave.runtime.datasets import run_dataset_requests

    return run_dataset_requests(requests, index, outputs, provenance, producers,
        resume=resume, data_ids=data_ids,
        materialize_reused_outputs=materialize_reused_outputs,
        execution_parameters=execution_parameters,
        candidate_data_ids=tuple(sorted(scope["data_ids"])))


_SEQUENCE_REQUIRED_STEP_FIELDS = frozenset({
    "requests", "outputs", "provenance", "producers", "receipt_directory",
})
_SEQUENCE_OPTIONAL_STEP_FIELDS = frozenset({
    "data_ids", "materialize_reused_outputs", "execution_parameters",
})


def _sequence_path(value, owner):
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
        raise TypeError(f"{owner} must be an explicit text path")
    raw = os.fspath(value)
    if isinstance(raw, bytes) or not raw or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(f"{owner} must be a nonempty text path")
    return Path(value)


def _sequence_path_key(path):
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def _validate_sequence_step(step, position, expected_project_id, resume,
                            source_policy, seen_output_paths,
                            initial_receipt_paths):
    if not isinstance(step, Mapping):
        raise TypeError(f"step builder {position} must return a mapping")
    missing = _SEQUENCE_REQUIRED_STEP_FIELDS - set(step)
    unknown = set(step) - _SEQUENCE_REQUIRED_STEP_FIELDS - _SEQUENCE_OPTIONAL_STEP_FIELDS
    if missing:
        raise ValueError(f"step {position} is missing fields: {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"step {position} has unsupported fields: {sorted(unknown)!r}")

    requests = step["requests"]
    _validate_query_arguments(requests, expected_project_id)
    if not requests:
        raise ValueError(f"step {position} requests must not be empty")
    request_names = set(requests)
    for name, request in requests.items():
        if not set(request.dependencies).issubset(request_names):
            missing_parent = sorted(set(request.dependencies) - request_names)[0]
            raise ValueError(
                f"step {position} request {name!r} references unknown request {missing_parent!r}")

    outputs = step["outputs"]
    provenance = step["provenance"]
    producers = step["producers"]
    for field, value in (("outputs", outputs), ("provenance", provenance),
                         ("producers", producers)):
        if not isinstance(value, Mapping):
            raise TypeError(f"step {position} {field} must be a mapping")
        if any(not isinstance(name, str) or not name for name in value):
            raise ValueError(f"step {position} {field} names must be nonempty strings")
        unknown_names = set(value) - request_names
        if unknown_names:
            raise ValueError(
                f"step {position} {field} names unknown request {sorted(unknown_names)[0]!r}")

    data_ids = step.get("data_ids")
    if data_ids is None:
        data_ids = {}
    elif not isinstance(data_ids, Mapping):
        raise TypeError(f"step {position} data_ids must be a mapping")
    else:
        data_ids = dict(data_ids)
    for name, identity in data_ids.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"step {position} data_ids names must be nonempty strings")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"step {position} data_ids for {name!r} must be a nonempty actual ID")
    unknown_ids = set(data_ids) - request_names
    if unknown_ids:
        raise ValueError(
            f"step {position} data_ids names unknown request {sorted(unknown_ids)[0]!r}")
    for name, identity in data_ids.items():
        request_identity = requests[name].data_id
        if request_identity is not None and request_identity != identity:
            raise ValueError(f"step {position} request {name!r} has conflicting actual IDs")

    materialize = step.get("materialize_reused_outputs", ())
    if not isinstance(materialize, tuple):
        raise TypeError(f"step {position} materialize_reused_outputs must be a tuple")
    if (any(not isinstance(name, str) or not name for name in materialize) or
            len(set(materialize)) != len(materialize)):
        raise ValueError(f"step {position} materialize_reused_outputs must contain unique request names")
    for name in materialize:
        if name not in requests or name not in outputs:
            raise ValueError(
                f"step {position} materialized request {name!r} needs an output path")

    execution_parameters = normalize_execution_parameters(step.get("execution_parameters"))
    if execution_parameters is not None:
        unknown_parameters = set(execution_parameters) - request_names
        if unknown_parameters:
            raise ValueError(
                f"step {position} execution parameters name unknown request {sorted(unknown_parameters)[0]!r}")
        for name, parameters in execution_parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"step {position} execution parameter names must be nonempty strings")
            if not isinstance(parameters, dict):
                raise TypeError(
                    f"step {position} execution parameters for {name!r} must be a JSON object")

    output_paths = {}
    for name, value in outputs.items():
        path = _sequence_path(value, f"step {position} output {name!r}")
        key = _sequence_path_key(path)
        if key in seen_output_paths:
            previous = seen_output_paths[key]
            raise ValueError(
                f"sequence output path for {name!r} duplicates {previous}")
        seen_output_paths[key] = f"step {position} output {name!r}"
        output_paths[name] = path

    for name, producer in producers.items():
        if not callable(producer):
            raise TypeError(f"step {position} producer for {name!r} must be callable")
    for name, metadata in provenance.items():
        if not isinstance(metadata, Mapping):
            raise TypeError(f"step {position} provenance for {name!r} must be a mapping")
        # Validate JSON serializability now, before any earlier graph can run.
        canonical_json(dict(metadata))

    for name in producers:
        if name not in output_paths or name not in provenance:
            raise ValueError(
                f"step {position} producer {name!r} needs explicit output and provenance")

    receipt_directory = _sequence_path(
        step["receipt_directory"], f"step {position} receipt_directory").expanduser()
    receipt_key = _sequence_path_key(receipt_directory)
    if receipt_key in initial_receipt_paths:
        raise ValueError(
            f"step {position} receipt_directory is an initial receipt file path")
    if receipt_directory.exists() and not receipt_directory.is_dir():
        raise NotADirectoryError(str(receipt_directory))

    # Reuse the canonical pure DAG planner for structural checks. No storage
    # observations means this cannot resolve candidates or authorize reuse.
    topology_requests = {
        name: DatasetRequest(request.template, subset_axes=request.subset_axes)
        for name, request in requests.items()
    }
    plan_dataset_graph(topology_requests, (), candidate_data_ids=())

    return dict(requests=dict(requests), outputs=output_paths,
        provenance=dict(provenance), producers=dict(producers),
        receipt_directory=receipt_directory, data_ids=data_ids,
        materialize_reused_outputs=materialize,
        execution_parameters=execution_parameters,
        source_policy=source_policy, resume=resume)


def run_project_dataset_sequence(index, step_builders, *, expected_project_id,
                                  initial_receipts=(),
                                  source_policy="allow_unknown", resume=False):
    """Run an explicit finite sequence of existing project-scoped dataset graphs.

    Builders are zero-argument callables that return the ordinary per-graph
    arguments plus ``receipt_directory``. They may inspect registered metadata
    and construct producers, but must not execute producers or write files.
    Every builder and static step contract is checked before the first graph is
    executed. Candidate resolution remains fresh and graph-local: only a
    successfully executed graph's standard receipt, revalidated through the
    public reader, is added to the next graph's explicit scope.
    """
    from lqcdweave.runtime.datasets import (
        read_dataset_request_receipts, validate_project_id,
    )

    validate_project_id(expected_project_id)
    if expected_project_id is None:
        raise ValueError("project sequence requires an explicit expected_project_id")
    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if source_policy not in ("allow_unknown", "require_known"):
        raise ValueError("source_policy must be 'allow_unknown' or 'require_known'")
    if (not isinstance(step_builders, Sequence) or
            isinstance(step_builders, (str, bytes, bytearray))):
        raise TypeError("step_builders must be an explicit finite sequence")
    if not step_builders:
        raise ValueError("step_builders must contain at least one graph")
    if any(not callable(builder) for builder in step_builders):
        raise TypeError("each project sequence step must be a zero-argument builder")

    # Reopen initial receipts before builders inspect graph-specific datasets
    # or any graph can write output.
    initial = read_dataset_request_receipts(
        initial_receipts, index, expected_project_id=expected_project_id)
    receipt_paths = [receipt["receipt_path"] for receipt in initial]

    seen_output_paths = {}
    initial_receipt_path_keys = {
        _sequence_path_key(Path(receipt["receipt_path"])) for receipt in initial
    }
    prepared = []
    for position, builder in enumerate(step_builders):
        raw_step = builder()
        prepared.append(_validate_sequence_step(
            raw_step, position, expected_project_id, resume, source_policy,
            seen_output_paths, initial_receipt_path_keys))

    results = []
    for step in prepared:
        identities = run_project_dataset_requests(
            step["requests"], index, step["outputs"], step["provenance"],
            step["producers"], tuple(receipt_paths),
            expected_project_id=expected_project_id,
            source_policy=step["source_policy"], resume=step["resume"],
            data_ids=step["data_ids"],
            materialize_reused_outputs=step["materialize_reused_outputs"],
            execution_parameters=step["execution_parameters"])

        from lqcdweave.runtime.datasets import (
            read_dataset_request_receipt, save_dataset_request_receipt,
        )

        receipt_path = save_dataset_request_receipt(
            step["receipt_directory"], step["requests"], index, identities,
            project_id=expected_project_id)
        verified = read_dataset_request_receipt(
            receipt_path, index, expected_project_id=expected_project_id)
        receipt_key = os.path.normcase(os.path.normpath(str(
            Path(verified["receipt_path"]).expanduser().resolve(strict=False))))
        if not any(_sequence_path_key(Path(path)) == receipt_key
                   for path in receipt_paths):
            receipt_paths.append(verified["receipt_path"])
        results.append(dict(datasets=identities,
                            receipt=verified["receipt_path"]))

    return tuple(results)


def _retention_receipt_inputs(receipts):
    if (not isinstance(receipts, Sequence) or
            isinstance(receipts, (str, bytes, os.PathLike))):
        raise TypeError("receipts must be an explicit sequence of receipt paths")
    paths = []
    keys = set()
    for value in receipts:
        if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
            raise TypeError("receipt paths must be explicit text paths")
        path = Path(value).expanduser().resolve(strict=False)
        key = os.path.normcase(os.path.normpath(str(path)))
        if key not in keys:
            keys.add(key)
            paths.append(path)
    return tuple(paths)


def _retention_candidates(candidates, index):
    if (not isinstance(candidates, Sequence) or
            isinstance(candidates, (str, bytes, os.PathLike))):
        raise TypeError("candidates must be an explicit sequence of actual-ID locations")
    normalized = []
    seen = set()
    for item in candidates:
        if not isinstance(item, Mapping) or set(item) != {"data_id", "location"}:
            raise ValueError("each retention candidate needs exactly data_id and location")
        identity, raw_location = item["data_id"], item["location"]
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("retention candidates require an actual dataset ID")
        if not isinstance(raw_location, (str, os.PathLike)) or isinstance(raw_location, bytes):
            raise TypeError("retention candidate locations must be explicit text paths")
        location = index.canonical_registered_location(identity, raw_location)
        key = (identity, location)
        if key in seen:
            raise ValueError("retention candidates cannot repeat an actual ID/location pair")
        seen.add(key)
        normalized.append(dict(data_id=identity, location=location))
    return tuple(normalized)


def _build_project_retention_plan(index, *, candidates, receipts,
                                  expected_project_id,
                                  pinned_actual_ids, complete_index_inventory,
                                  complete_receipt_inventory):
    if complete_index_inventory is not True or complete_receipt_inventory is not True:
        raise ValueError("retention requires explicit complete index and receipt inventories")
    if not isinstance(expected_project_id, str) or not expected_project_id.strip():
        raise ValueError("retention requires an explicit project ID")
    if (not isinstance(pinned_actual_ids, (tuple, list, set, frozenset)) or
            any(not isinstance(identity, str) or len(identity) != 64
                for identity in pinned_actual_ids)):
        raise ValueError("pinned_actual_ids must be explicit actual dataset IDs")
    normalized_receipts = _retention_receipt_inputs(receipts)
    candidate_records = _retention_candidates(candidates, index)

    # _retention_candidates asks storage to validate raw path components before
    # normalization. These canonical records now distinguish retained source
    # locations from exact, active managed copies eligible for deletion.
    candidate_reasons = {}
    for candidate in candidate_records:
        identity, location = candidate["data_id"], candidate["location"]
        try:
            description = index.describe_managed_location(identity, location)
        except ValueError as exc:
            message = str(exc)
            if ("configured managed cache root" in message or
                    "direct nonsymlink" in message or
                    "not owned and active" in message):
                candidate_reasons[(identity, location)] = {
                    "not_an_owned_managed_cache_copy"}
            else:
                raise
        else:
            if description["state"] != "active":
                candidate_reasons[(identity, location)] = {
                    "retirement_already_in_progress"}
            else:
                candidate_reasons[(identity, location)] = set()

    inventory = index.metadata_inventory()
    revision = index.publication_revision

    from lqcdweave.runtime.datasets import read_dataset_request_receipts
    verified_receipts = read_dataset_request_receipts(
        normalized_receipts, index, expected_project_id=expected_project_id)

    plan = plan_dataset_retention(
        inventory=inventory, revision=revision, candidate_records=candidate_records,
        candidate_reasons=candidate_reasons, verified_receipts=verified_receipts,
        expected_project_id=expected_project_id, pinned_actual_ids=pinned_actual_ids)
    for item in plan["decisions"]:
        if item["action"] == "quarantine_and_delete":
            index.verify_managed_location(item["data_id"], item["location"])
    return plan


def plan_project_dataset_retention(index, candidates, receipts, *,
                                    expected_project_id,
                                    pinned_actual_ids=(),
                                    complete_index_inventory=False,
                                    complete_receipt_inventory=False):
    """Build a dry-run-only retention plan from explicit complete inventories.

    Unmanaged/external locations, pins, receipt nodes and parents of retained
    datasets are preserved. The caller must provide every project receipt;
    storage never scans project directories.
    """
    with _retention_index_view(index) as view:
        return _build_project_retention_plan(view, candidates=candidates,
            receipts=receipts, expected_project_id=expected_project_id,
            pinned_actual_ids=pinned_actual_ids,
            complete_index_inventory=complete_index_inventory,
            complete_receipt_inventory=complete_receipt_inventory)


def apply_project_dataset_retention(index, plan):
    """Apply a fresh explicit plan; stale revisions require a new dry run."""
    if not isinstance(plan, Mapping) or plan.get("schema") != 1:
        raise ValueError("invalid project dataset retention plan")
    required = {"schema", "publication_revision", "project_id", "receipts",
        "pinned_actual_ids", "candidates", "inventory", "decisions",
        "complete_index_inventory", "complete_receipt_inventory"}
    if set(plan) != required:
        raise ValueError("project dataset retention plan has an invalid field set")
    receipt_paths = list(plan["receipts"])
    with _retention_index_view(index) as view:
        fresh = _build_project_retention_plan(view,
            candidates=plan["candidates"], receipts=receipt_paths,
            expected_project_id=plan["project_id"],
            pinned_actual_ids=plan["pinned_actual_ids"],
            complete_index_inventory=plan["complete_index_inventory"],
            complete_receipt_inventory=plan["complete_receipt_inventory"])
        if (view.publication_revision != plan["publication_revision"] or
                canonical_json(fresh) != canonical_json(dict(plan))):
            raise ValueError("retention plan snapshot is stale; replan is required")
        result = []
        for item in fresh["decisions"]:
            if item["action"] != "quarantine_and_delete":
                result.append(item)
                continue
            result.append(view.retire_managed_location(
                item["data_id"], item["location"],
                expected_revision=view.publication_revision))
        return dict(decisions=tuple(result),
                    publication_revision=view.publication_revision)
