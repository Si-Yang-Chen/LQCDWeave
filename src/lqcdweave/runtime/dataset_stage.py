"""Validate an already published stage before a workflow reuses it."""

from contextlib import ExitStack
from collections.abc import Mapping
import uuid

from lqcdweave.planning.dataset import plan_stage, plan_dataset_graph, plan_requirement
from lqcdweave.storage.dataset import DatasetWriter, inspect_dataset
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.execution import normalize_execution_parameters


def stage_plan(directory, spec, provenance):
    """Inspect one explicit output and return its storage work decision."""
    return plan_stage(spec, provenance, inspect_dataset(directory, cfg=spec.document['configuration']['configuration']))


def completed_dataset(directory, spec, provenance):
    """Return a complete compatible dataset ID, or None if unpublished.

    A published registration or declared array-layout mismatch is an error.
    DatasetWriter handles partial cfg locks, array layout and completed coordinates.
    """
    decision = stage_plan(directory, spec, provenance)
    return decision["data_id"] if decision["action"] == "reuse" else None


def _plan_observations(requests, index, pinned, candidate_data_ids=None):
    """Build observations for the global index or an explicit candidate set."""
    if candidate_data_ids is None:
        return index.observations()
    full_ids = set(candidate_data_ids)
    for name, request in requests.items():
        identity = pinned.get(name, request.data_id)
        if request.data_id is not None and identity != request.data_id:
            raise ValueError(f"request {name!r} has conflicting actual dataset IDs")
        if identity is not None:
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"request {name!r} has an invalid actual dataset ID")
            full_ids.add(identity)
    observations = []
    for identity in sorted(full_ids):
        with index.open(identity) as reader:
            observations.append(reader.observation())
    return tuple(observations)


def _normalize_graph_execution_parameters(requests, value):
    """Validate and detach optional request-name to JSON-object settings."""
    normalized = normalize_execution_parameters(value)
    if normalized is None:
        return None
    if not isinstance(requests, Mapping):
        raise TypeError("execution parameters require a request mapping")
    unknown = set(normalized) - set(requests)
    if unknown:
        raise ValueError(f"execution parameters name unknown requests: {sorted(unknown)!r}")
    for name, parameters in normalized.items():
        if not isinstance(name, str) or not name:
            raise ValueError("execution parameter request names must be nonempty strings")
        if not isinstance(parameters, dict):
            raise TypeError(f"execution parameters for {name!r} must be a JSON object")
    return normalized


def run_dataset_graph(requests, index, outputs, provenance, producers, open_view,
                       *, resume=False, data_ids=None,
                       materialize_reused_outputs=(), execution_parameters=None,
                       candidate_data_ids=None):
    """Execute a dependency graph through the existing dataset lifecycle.

    All paths, per-stage provenance and numerical producers are explicit.
    ``open_view(spec, data_id, subset_axes=...)`` is injected by the calling frontend
    and receives the requested scientific subset axes.
    A producer receives ``(spec, inputs, writer)`` and synchronously writes
    blocks requested by writer.missing. Completion is determined exclusively
    by writer.publish and index registration, never by a producer return value.
    ``materialize_reused_outputs`` explicitly names reused nodes whose
    actual dataset should also be materialized at ``outputs[name]``. Other
    output mappings retain their existing workflow meaning.

    Only one compatible complete source may supply covered first-axis rows;
    remaining rows are produced under the same fully resolved request. Its
    actual ID and copied request positions are recorded in execution provenance.
    Partial outputs are resumed only at their explicit path, with exact spec
    and provenance. No additional queue, lock or persistent task ledger is made.
    ``execution_parameters`` maps requested stage names to detached JSON
    objects. Values are validated before observing or opening any datasets
    and are passed to storage only when that stage's producer will run.
    ``candidate_data_ids`` optionally replaces global candidate enumeration
    for unpinned requests with a fixed explicit full-dataset set. A direct pin
    is available only to its own request. Datasets actually published while
    this graph runs can supply later unpinned requests. The default ``None``
    preserves global-index planning.
    """
    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if candidate_data_ids is not None:
        if not isinstance(candidate_data_ids, tuple):
            raise TypeError("candidate_data_ids must be a tuple of actual dataset IDs")
        if (any(not isinstance(identity, str) or not identity
                for identity in candidate_data_ids) or
                len(set(candidate_data_ids)) != len(candidate_data_ids)):
            raise ValueError("candidate_data_ids must contain unique nonempty actual IDs")
        automatic_candidate_ids = set(candidate_data_ids)
    else:
        automatic_candidate_ids = None
    normalized_execution_parameters = _normalize_graph_execution_parameters(
        requests, execution_parameters)
    if not isinstance(materialize_reused_outputs, tuple):
        raise TypeError("materialize_reused_outputs must be a tuple of request names")
    if any(not isinstance(name, str) or not name for name in materialize_reused_outputs):
        raise ValueError("materialize_reused_outputs must contain nonempty request names")
    if materialize_reused_outputs and (
            not isinstance(requests, Mapping) or not isinstance(outputs, Mapping)):
        raise TypeError("materialize_reused_outputs requires request and output mappings")
    if len(set(materialize_reused_outputs)) != len(materialize_reused_outputs):
        raise ValueError("materialize_reused_outputs request names must be unique")
    for name in materialize_reused_outputs:
        if name not in requests or name not in outputs:
            raise ValueError(
                f"materialized request {name!r} must have a request and output path")
    run_id = uuid.uuid4().hex
    pinned = {} if data_ids is None else dict(data_ids)
    observations = _plan_observations(
        requests, index, pinned, automatic_candidate_ids)
    graph = plan_dataset_graph(requests, observations, data_ids=pinned,
        candidate_data_ids=(None if automatic_candidate_ids is None else
                                tuple(sorted(automatic_candidate_ids))))
    if resume:
        for name in graph["order"]:
            if name not in outputs:
                continue
            saved = inspect_dataset(outputs[name], cfg=requests[name].template['configuration']['configuration'])
            if saved["state"] == "missing":
                continue
            bindings = requests[name].resolved_bindings(saved["spec"])
            if saved["state"] == "complete":
                try:
                    with index.open(saved["data_id"]):
                        bindings[name] = saved["data_id"]
                except KeyError:
                    pass
            for parent, identity in bindings.items():
                if parent in pinned and pinned[parent] != identity:
                    raise ValueError("resume parent dataset binding differs between requests or drafts")
                pinned[parent] = identity
        # Persisted scientific definitions fix the original actual parents.
        # The index must still prove each pinned parent complete and compatible.
        observations = _plan_observations(
            requests, index, pinned, automatic_candidate_ids)
        graph = plan_dataset_graph(requests, observations, data_ids=pinned,
            candidate_data_ids=(None if automatic_candidate_ids is None else
                                    tuple(sorted(automatic_candidate_ids))))
    completed = {}
    for name in graph["order"]:
        observations = _plan_observations(
            requests, index, pinned, automatic_candidate_ids)
        graph = plan_dataset_graph(requests, observations, data_ids=pinned,
            candidate_data_ids=(None if automatic_candidate_ids is None else
                                    tuple(sorted(automatic_candidate_ids))))
        stage = graph["stages"][name]
        if stage["action"] == "blocked":
            raise ValueError(f"dataset stage {name!r} is blocked: {stage['reasons']}")
        if stage["action"] == "reuse":
            identity = stage["data_id"]
            if resume and name in outputs:
                saved = inspect_dataset(outputs[name], cfg=requests[name].template['configuration']['configuration'])
                if saved["state"] == "complete":
                    if saved["data_id"] != identity:
                        raise ValueError(
                            f"completed resume output for {name!r} differs from its pinned actual ID")
            if name in materialize_reused_outputs:
                identity = index.materialize(identity, outputs[name])
        else:
            published_current = False
            fixed_id = requests[name].data_id or (
                {} if data_ids is None else data_ids).get(name)
            if fixed_id is not None:
                raise ValueError(f"cannot produce missing coverage for fixed actual dataset {name!r}")
            if name not in outputs or name not in provenance or name not in producers:
                raise ValueError(f"missing explicit output, provenance or producer for {name!r}")
            spec = stage["spec"]
            source_id = stage["data_id"]
            copied = stage["completed"] if source_id is not None else ()
            metadata = dict(provenance[name])
            if "coverage_reuse" in metadata:
                raise ValueError("coverage_reuse provenance is owned by dataset execution")
            observed = inspect_dataset(outputs[name], cfg=requests[name].template['configuration']['configuration']) if resume else None
            if observed is not None and observed["state"] != "missing":
                # A new registered candidate must not change an existing draft's
                # execution origin. Its verified draft is the recovery authority.
                saved = observed["provenance"].get("coverage_reuse")
                source_id, copied = None, ()
                if saved is not None:
                    if (not isinstance(saved, dict) or
                            set(saved) != {"data_id", "requested_positions"} or
                            not isinstance(saved["data_id"], str) or not saved["data_id"] or
                            not isinstance(saved["requested_positions"], list) or
                            not saved["requested_positions"] or
                            any(type(position) is not int for position in saved["requested_positions"])):
                        raise ValueError("invalid resume coverage origin")
                    with index.open(saved["data_id"]) as source_reader:
                        source_observation = source_reader.observation()
                    prior = plan_requirement(spec, (source_observation,),
                        subset_axes=requests[name].subset_axes, data_id=saved["data_id"])
                    if (prior["data_id"] != saved["data_id"] or
                            prior["completed"] != tuple(saved["requested_positions"])):
                        raise ValueError("resume coverage origin is unavailable or incompatible")
                    source_id, copied = prior["data_id"], prior["completed"]
            if copied:
                metadata["coverage_reuse"] = dict(data_id=source_id,
                                                  requested_positions=list(copied))
            if observed is not None:
                local = plan_stage(spec, metadata, observed)
                if local["action"] == "blocked":
                    raise ValueError(f"dataset stage {name!r} is blocked: {local['reasons']}")
                identity = local["data_id"] if local["action"] == "reuse" else None
            else:
                identity = None
            if identity is None:
                with ExitStack() as stack:
                    inputs = {}
                    for parent in stage["requires"]:
                        parent_stage = graph["stages"][parent]
                        inputs[parent] = stack.enter_context(open_view(
                            parent_stage["spec"], graph["bindings"][parent],
                            subset_axes=requests[parent].subset_axes))
                    attempt_context = dict(run_id=run_id,
                        producer_attempt_id=None, stage=name)
                    with DatasetWriter(outputs[name], spec, provenance=metadata,
                                        resume=resume,
                                        attempt_context=attempt_context) as writer:
                        if copied:
                            first = spec.axis_names[0]
                            document = spec.document
                            coordinates = spec.coordinates(first)
                            document["axes"][0]["coordinates"] = [coordinates[i] for i in copied]
                            if first in document["bases"]:
                                document["bases"][first]["indices"] = document["axes"][0]["coordinates"]
                            with open_view(DatasetSpec(document), source_id,
                                           subset_axes=requests[name].subset_axes) as source:
                                for position in copied:
                                    if position in writer.missing:
                                        writer.write(position, source.read(**{first: coordinates[position]}))
                        if writer.missing:
                            if (normalized_execution_parameters is not None and
                                    name in normalized_execution_parameters):
                                writer.bind_producer_attempt(
                                    uuid.uuid4().hex,
                                    execution_parameters=normalized_execution_parameters[name],
                                )
                            else:
                                writer.bind_producer_attempt(uuid.uuid4().hex)
                            producers[name](spec, inputs, writer)
                        identity = writer.publish()
                        published_current = True
            registered = index.register(outputs[name], cfg=spec.document['configuration']['configuration'])
            if registered != identity:
                raise ValueError("published dataset identity changed before registration")
            if automatic_candidate_ids is not None and published_current:
                automatic_candidate_ids.add(identity)
        pinned[name] = identity
        completed[name] = identity
    return completed
