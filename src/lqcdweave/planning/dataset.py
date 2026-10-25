"""Pure decisions over verified storage observations and explicit requirements."""

from collections.abc import Mapping, Sequence

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec, canonical_json, scientific_input_bindings


def plan_stage(spec, provenance, observed):
    """Choose read-only reuse or exact first-axis blocks to compute.

    ``observed`` comes from storage.inspect_dataset, which has already checked
    storage metadata and coverage. This function performs no file or numerical work.
    """
    state = observed["state"]
    if state not in ("missing", "partial", "complete"):
        raise ValueError("unknown dataset state")
    if state != "missing":
        if observed["spec"].definition_id != spec.definition_id:
            raise ValueError("resume definition or input dataset differs")
        if canonical_json(observed["provenance"]) != canonical_json(provenance):
            raise ValueError("stage execution provenance differs")
        execution = observed.get("execution")
        if (state == "partial" and isinstance(execution, Mapping) and
                execution.get("scientific_validation", {}).get("status") == "failed"):
            return dict(action="blocked", data_id=None, completed=(), missing=(),
                        reasons=["partial dataset has a failed scientific validation"])
    completed = tuple(observed["completed"])
    if any(type(i) is not int or i not in range(spec.shape[0]) for i in completed):
        raise ValueError("invalid completed block indices")
    if state == "complete":
        if completed != tuple(range(spec.shape[0])) or not observed["data_id"]:
            raise ValueError("invalid complete dataset observation")
        return dict(action="reuse", data_id=observed["data_id"],
                    completed=completed, missing=())
    missing = tuple(i for i in range(spec.shape[0]) if i not in completed)
    return dict(action="compute", data_id=None, completed=completed,
                missing=missing)


def _contains_none(value):
    if value is None:
        return True
    if isinstance(value, Mapping):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return False


def _unknown_identity_fields(document):
    unknown = []
    configuration = document["configuration"]
    for field in ("ensemble", "configuration"):
        if configuration[field] is None:
            unknown.append(f"configuration.{field}")
    for axis, basis in document["bases"].items():
        if basis["data_id"] is None:
            unknown.append(f"bases.{axis}.data_id")
    for field in ("definition", "conventions"):
        if _contains_none(document[field]):
            unknown.append(f"{field} contains an unknown null value")
    return tuple(unknown)


def _coordinate_keys(coordinates):
    return tuple(canonical_json(value) for value in coordinates)


def _difference_paths(left, right, path):
    # Scientific definitions can contain complete ordered dilution lists. Admit
    # identical subtrees in one canonical comparison, rather than serializing
    # every scalar separately. Keep JSON's type/number distinctions (including
    # bool versus int and negative zero) and precise paths for actual changes.
    if canonical_json(left) == canonical_json(right):
        return []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            paths.append(f"{path}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(_difference_paths(left_item, right_item, f"{path}[{index}]"))
        return paths
    return [path]


def _complete_observation(observation, label):
    """Return a reason when an observation cannot attest complete coverage."""
    if "provenance" not in observation:
        return None, f"{label}: complete observation has no provenance record"
    candidate_spec = observation.get("spec")
    if not isinstance(candidate_spec, DatasetSpec):
        return None, f"{label}: complete observation has no DatasetSpec"
    identity = observation.get("data_id")
    if not isinstance(identity, str) or not identity:
        return None, f"{label}: complete observation has no data_id"
    completed = observation.get("completed")
    if (not isinstance(completed, (list, tuple)) or
            any(type(index) is not int for index in completed) or
            tuple(completed) != tuple(range(candidate_spec.shape[0]))):
        return None, f"{label}: complete observation does not attest every first-axis block"
    return candidate_spec, None


def _identity_mismatches(requested, candidate):
    mismatches = []
    for field in ("schema", "kind", "configuration", "dtype", "definition",
                  "conventions", "parents"):
        mismatches.extend(_difference_paths(requested[field], candidate[field], field))

    requested_bases = requested["bases"]
    candidate_bases = candidate["bases"]
    if set(requested_bases) != set(candidate_bases):
        mismatches.append("bases axis names")
    else:
        for axis in requested_bases:
            if requested_bases[axis]["data_id"] != candidate_bases[axis]["data_id"]:
                mismatches.append(f"bases.{axis}.data_id")

    requested_names = tuple(axis["name"] for axis in requested["axes"])
    candidate_names = tuple(axis["name"] for axis in candidate["axes"])
    if requested_names != candidate_names:
        mismatches.append("axis_names/order")
    return mismatches


def plan_requirement(spec, observations, *, subset_axes=(), data_id=None):
    """Plan reuse or missing first-axis work from complete storage observations."""
    if not isinstance(spec, DatasetSpec):
        raise TypeError("spec must be a DatasetSpec")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a storage observation sequence")
    if isinstance(subset_axes, str):
        raise TypeError("subset_axes must be an iterable of axis names")
    try:
        subset_axes = tuple(subset_axes)
    except TypeError as exc:
        raise TypeError("subset_axes must be an iterable of axis names") from exc
    if any(not isinstance(name, str) or not name for name in subset_axes):
        raise TypeError("subset_axes entries must be nonempty axis names")
    if len(set(subset_axes)) != len(subset_axes):
        raise ValueError("subset_axes contains a duplicate axis")
    if data_id is not None and (not isinstance(data_id, str) or not data_id):
        raise ValueError("data_id must be a nonempty string or None")
    requested = spec.document
    axis_names = tuple(axis["name"] for axis in requested["axes"])
    unknown_axes = set(subset_axes) - set(axis_names)
    if unknown_axes:
        raise ValueError(f"unknown subset axis: {sorted(unknown_axes)[0]}")

    requested_positions = tuple(range(spec.shape[0]))
    selection = {
        axis["name"]: list(axis["coordinates"])
        for axis in requested["axes"][1:]
    }
    unknown_fields = _unknown_identity_fields(requested)
    if unknown_fields:
        return dict(
            action="blocked", data_id=None, completed=(),
            missing=requested_positions, selection=selection,
            reasons=["requested scientific identity is unknown: " + ", ".join(unknown_fields)],
        )

    reasons = []
    compatible = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation {index} must be a mapping")
        label = observation.get("data_id") or f"observation[{index}]"
        if data_id is not None and observation.get("data_id") != data_id:
            reasons.append(f"{label}: excluded by explicit data_id filter")
            continue
        state = observation.get("state")
        if state != "complete":
            reasons.append(f"{label}: state is {state!r}; only complete datasets are reusable")
            continue
        candidate_spec, invalid_reason = _complete_observation(observation, str(label))
        if invalid_reason is not None:
            reasons.append(invalid_reason)
            continue
        candidate = candidate_spec.document
        mismatches = _identity_mismatches(requested, candidate)
        if mismatches:
            reasons.append(f"{label}: scientific identity differs in {', '.join(mismatches)}")
            continue

        request_axes = {axis["name"]: axis["coordinates"] for axis in requested["axes"]}
        candidate_axes = {axis["name"]: axis["coordinates"] for axis in candidate["axes"]}
        first_axis = axis_names[0]
        request_first = request_axes[first_axis]
        candidate_first = candidate_axes[first_axis]
        if first_axis in subset_axes:
            candidate_first_keys = set(_coordinate_keys(candidate_first))
            covered = tuple(
                position for position, key in enumerate(_coordinate_keys(request_first))
                if key in candidate_first_keys
            )
            if not covered:
                reasons.append(f"{label}: allowed first-axis coverage has no requested coordinates")
                continue
        else:
            if _coordinate_keys(request_first) != _coordinate_keys(candidate_first):
                reasons.append(f"{label}: first-axis coordinates differ and {first_axis!r} is not subset-enabled")
                continue
            covered = requested_positions

        mismatch = None
        for axis_name in axis_names[1:]:
            request_coordinates = request_axes[axis_name]
            candidate_coordinates = candidate_axes[axis_name]
            if axis_name in subset_axes:
                candidate_keys = set(_coordinate_keys(candidate_coordinates))
                absent = [
                    value for value, key in zip(request_coordinates, _coordinate_keys(request_coordinates))
                    if key not in candidate_keys
                ]
                if absent:
                    mismatch = f"{axis_name!r} does not cover requested coordinates {absent!r}"
                    break
            elif _coordinate_keys(request_coordinates) != _coordinate_keys(candidate_coordinates):
                mismatch = f"{axis_name!r} coordinates differ and the axis is not subset-enabled"
                break
        if mismatch is not None:
            reasons.append(f"{label}: {mismatch}")
            continue

        compatible.append((len(covered), observation["data_id"], covered))
        reasons.append(f"{label}: compatible with {len(covered)} requested first-axis positions")

    if not compatible:
        reasons.append("no compatible complete dataset covers any requested first-axis coordinate")
        return dict(
            action="compute", data_id=None, completed=(),
            missing=requested_positions, selection=selection, reasons=reasons,
        )

    compatible.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    _, selected_id, completed = compatible[0]
    for coverage, candidate_id, _ in compatible[1:]:
        reasons.append(
            f"{candidate_id!r}: compatible but not selected; coverage {coverage} is below "
            f"the selected coverage or its data_id sorts later"
        )
    completed_set = set(completed)
    missing = tuple(position for position in requested_positions if position not in completed_set)
    reasons.append(f"selected {selected_id!r} by greatest coverage then data_id order")
    result = dict(
        action="reuse" if not missing else "compute",
        data_id=selected_id,
        completed=completed,
        missing=missing,
        selection=selection,
        reasons=reasons,
    )
    return result


def requested_axes(spec):
    """Return a detached ordered axis list for a validated DatasetSpec."""
    return spec.document["axes"]


def _scientific_input_roles(child_request, child_spec, parent_name, parent_id,
                            sibling_bindings):
    """Find an explicit semantic role for a graph dependency when present."""
    requested_roles = child_request.template.get("definition", {}).get("input_roles")
    resolved_roles = child_spec.document["definition"].get("input_roles")
    if not isinstance(requested_roles, Mapping) or not isinstance(resolved_roles, Mapping):
        return ()
    logical_reference = {"request": parent_name}
    candidates = [role for role, reference in requested_roles.items()
                  if reference == logical_reference]
    if not candidates:
        same_id_parents = [name for name, identity in sibling_bindings.items()
                           if identity == parent_id]
        candidates = [role for role, reference in requested_roles.items()
                      if reference == parent_id]
        if same_id_parents != [parent_name]:
            return ()
    return tuple(sorted(role for role in candidates
                        if resolved_roles.get(role) == parent_id))


def _blocked_dependency_stage(request, requires, blocked):
    template = request.template
    axes = template["axes"]
    selection = {
        axis["name"]: list(axis["coordinates"])
        for axis in axes[1:]
    }
    reasons = [
        f"required request {name!r} is not a complete reusable dataset: {reason}"
        for name, reason in blocked
    ]
    return dict(
        action="blocked", data_id=None, completed=(),
        missing=tuple(range(len(axes[0]["coordinates"]))),
        selection=selection, reasons=reasons,
        spec=None, requires=requires,
    )


def plan_dataset_graph(requests, observations, *, data_ids=None,
                        candidate_data_ids=None):
    """Plan a deterministic DAG of requests using complete datasets.

    A node becomes a scientific ``DatasetSpec`` only after every logical
    parent has a complete reusable binding. Partial candidate coverage can be
    reported for a node's own work, but never supplies a downstream identity.
    This function plans dependencies and reuse only; it performs no execution,
    publication, or cross-dataset assembly. ``candidate_data_ids`` limits
    observations available to unpinned requests; a fixed request ID is checked
    only for its own request and never widens another node's automatic scope.
    """
    if not isinstance(requests, Mapping):
        raise TypeError("requests must map node names to DatasetRequest instances")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a storage observation sequence")
    observations = tuple(observations)
    if data_ids is None:
        data_ids = {}
    elif not isinstance(data_ids, Mapping):
        raise TypeError("data_ids must map request names to actual dataset IDs")
    if candidate_data_ids is not None:
        if not isinstance(candidate_data_ids, tuple):
            raise TypeError("candidate_data_ids must be a tuple of actual dataset IDs")
        if (any(not isinstance(identity, str) or not identity
                for identity in candidate_data_ids) or
                len(set(candidate_data_ids)) != len(candidate_data_ids)):
            raise ValueError("candidate_data_ids must contain unique nonempty actual IDs")
        automatic_ids = set(candidate_data_ids)
    else:
        automatic_ids = None

    normalized = dict(requests)
    for name, request in normalized.items():
        if not isinstance(name, str) or not name:
            raise ValueError("request names must be nonempty strings")
        if not isinstance(request, DatasetRequest):
            raise TypeError(f"request {name!r} must be a DatasetRequest")
    if any(not isinstance(name, str) or not name for name in data_ids):
        raise ValueError("data_ids keys must be nonempty request names")
    unknown_id_nodes = set(data_ids) - set(normalized)
    if unknown_id_nodes:
        raise ValueError(f"data_ids names unknown request {sorted(unknown_id_nodes)[0]!r}")
    effective_ids = {}
    for name, request in normalized.items():
        has_supplied = name in data_ids
        supplied = data_ids[name] if has_supplied else None
        if has_supplied and (not isinstance(supplied, str) or not supplied):
            raise ValueError(f"fixed dataset ID for {name!r} must be a nonempty string")
        if request.data_id is not None and supplied is not None and request.data_id != supplied:
            raise ValueError(f"fixed dataset IDs conflict for request {name!r}")
        effective_ids[name] = supplied if supplied is not None else request.data_id

    dependencies = {}
    dependents = {name: [] for name in normalized}
    indegree = {}
    for name, request in normalized.items():
        requires = request.dependencies
        unknown = set(requires) - set(normalized)
        if unknown:
            raise ValueError(
                f"request {name!r} references unknown request {sorted(unknown)[0]!r}"
            )
        dependencies[name] = requires
        indegree[name] = len(requires)
        for dependency in requires:
            dependents[dependency].append(name)
    ready = sorted(name for name, count in indegree.items() if count == 0)
    order = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for child in sorted(dependents[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(normalized):
        cyclic = sorted(name for name, count in indegree.items() if count)
        raise ValueError("dataset request graph contains a cycle involving: " + ", ".join(cyclic))

    stages = {}
    bindings = {}
    for name in order:
        request = normalized[name]
        requires = dependencies[name]
        blocked = [
            (
                dependency,
                "action=" + stages[dependency]["action"]
                + ", missing=" + repr(stages[dependency]["missing"]),
            )
            for dependency in requires if dependency not in bindings
        ]
        if blocked:
            stages[name] = _blocked_dependency_stage(request, requires, blocked)
            continue

        parent_ids = {dependency: bindings[dependency] for dependency in requires}
        spec = request.resolve(parent_ids)
        science_roles = {}
        for dependency, identity in parent_ids.items():
            roles = _scientific_input_roles(
                request, spec, dependency, identity, parent_ids)
            science_roles[dependency] = roles
        fixed_id = effective_ids[name]
        stage_observations = observations
        if fixed_id is None and automatic_ids is not None:
            stage_observations = tuple(
                observation for observation in observations
                if (not isinstance(observation, Mapping) or
                    "data_id" not in observation or
                    observation.get("data_id") in automatic_ids)
            )
        result = plan_requirement(
            spec, stage_observations, subset_axes=request.subset_axes,
            data_id=fixed_id,
        )
        if fixed_id is not None and result["action"] != "blocked":
            if result["data_id"] != fixed_id:
                raise ValueError(
                    f"fixed dataset ID {fixed_id!r} for request {name!r} has no "
                    "fully compatible complete observation"
                )
        stage = dict(result)
        stage["spec"] = spec
        stage["requires"] = requires
        stage["scientific_roles"] = science_roles
        stages[name] = stage
        if stage["action"] == "reuse":
            bindings[name] = stage["data_id"]

    return dict(order=tuple(order), stages=stages, bindings=bindings)


def plan_dataset_retention(*, inventory, revision, candidate_records,
                            candidate_reasons, verified_receipts,
                            expected_project_id, pinned_actual_ids):
    """Pure retention decisions over an explicit, verified complete inventory.

    Location ownership, receipts and revisions are observed by the workflow;
    this service performs only parent protection and child-first ordering.
    Relocated from runtime.project_datasets._build_project_retention_plan.
    """
    candidate_keys = {(item["data_id"], item["location"])
                      for item in candidate_records}
    inventory_by_id = {}
    indexed_pairs = set()
    for item in inventory:
        spec = item["spec"]
        spec_document = spec.document if isinstance(spec, DatasetSpec) else spec
        record = dict(data_id=item["data_id"], location=item["location"],
                      spec=spec_document, source_origin=item["source_origin"],
                      provenance=item["provenance"])
        inventory_by_id.setdefault(record["data_id"], []).append(record)
        indexed_pairs.add((record["data_id"], record["location"]))
    if any((item["data_id"], item["location"]) not in indexed_pairs
           for item in candidate_records):
        raise ValueError("retention candidate is not an exact registered actual-ID location")

    pinned = set(pinned_actual_ids)
    receipt_ids = set()
    receipt_snapshot = []
    for receipt in verified_receipts:
        records = receipt["records"]
        receipt_ids.update(record["data_id"] for record in records)
        receipt_snapshot.append(receipt["receipt_path"])
    missing_pins = (pinned | receipt_ids) - set(inventory_by_id)
    if missing_pins:
        raise ValueError("retention inventory omits a pinned or receipt-referenced actual ID")

    protected_by = {}
    for identity in pinned:
        protected_by.setdefault(identity, set()).add("pinned_actual_id")
    for identity in receipt_ids:
        protected_by.setdefault(identity, set()).add("project_receipt")

    # First retain every identity which cannot be fully retired: noncandidate
    # locations, pins/receipts, and candidates which are unmanaged or already
    # retiring. Only then walk the parent closure, so a kept child protects its
    # managed parent even when both locations were explicitly proposed.
    candidate_by_identity = {}
    for candidate in candidate_records:
        candidate_by_identity.setdefault(candidate["data_id"], []).append(candidate)
    retained_ids = {
        identity for identity, locations in inventory_by_id.items()
        if identity in protected_by or any(
            (identity, item["location"]) not in candidate_keys for item in locations)
    }
    for identity, candidates_for_identity in candidate_by_identity.items():
        if any(candidate_reasons[(identity, item["location"])]
               for item in candidates_for_identity):
            retained_ids.add(identity)
    changed = True
    while changed:
        changed = False
        for identity in tuple(retained_ids):
            for item in inventory_by_id[identity]:
                refs, _ = scientific_input_bindings(DatasetSpec(item["spec"]))
                for parent_id in refs:
                    if parent_id in inventory_by_id and parent_id not in retained_ids:
                        retained_ids.add(parent_id)
                        protected_by.setdefault(parent_id, set()).add(
                            f"referenced_by_retained:{identity}")
                        changed = True
                    elif parent_id in inventory_by_id:
                        protected_by.setdefault(parent_id, set()).add(
                            f"referenced_by_retained:{identity}")

    decisions_by_key = {}
    deletable_ids = set()
    for candidate in candidate_records:
        identity, location = candidate["data_id"], candidate["location"]
        reasons = set(candidate_reasons[(identity, location)])
        # A noncandidate location (or another independently retained copy)
        # keeps the identity and its parents alive, but does not make this
        # exact eligible managed copy nonremovable. Pins, receipt nodes and
        # referenced parents, on the other hand, protect every location.
        reasons.update(reason for reason in protected_by.get(identity, ())
                       if reason == "pinned_actual_id" or
                       reason == "project_receipt" or
                       reason.startswith("referenced_by_retained:"))
        if not reasons:
            deletable_ids.add(identity)
        decisions_by_key[(identity, location)] = dict(
            data_id=identity, location=location,
            action="retain" if reasons else "quarantine_and_delete",
            reasons=sorted(reasons))

    # A parent location must never disappear while an indexed child which
    # references it remains. For the fully deletable subgraph, topologically
    # order child identities before parents; reject cycles rather than making
    # a partial, order-dependent decision.
    parents_by_child = {identity: set() for identity in deletable_ids}
    indegree = {identity: 0 for identity in deletable_ids}
    for child_id in deletable_ids:
        for item in inventory_by_id[child_id]:
            refs, _ = scientific_input_bindings(DatasetSpec(item["spec"]))
            for parent_id in refs:
                if parent_id in deletable_ids and parent_id not in parents_by_child[child_id]:
                    parents_by_child[child_id].add(parent_id)
                    indegree[parent_id] += 1
    original_identity_order = []
    for item in candidate_records:
        if item["data_id"] not in original_identity_order:
            original_identity_order.append(item["data_id"])
    ready = [identity for identity in original_identity_order
             if identity in deletable_ids and indegree[identity] == 0]
    ordered_delete_ids = []
    while ready:
        identity = ready.pop(0)
        ordered_delete_ids.append(identity)
        for parent_id in sorted(parents_by_child[identity],
                                key=original_identity_order.index):
            indegree[parent_id] -= 1
            if indegree[parent_id] == 0:
                ready.append(parent_id)
    if len(ordered_delete_ids) != len(deletable_ids):
        raise ValueError("retention parent graph contains a cycle; refusing deletion")
    delete_rank = {identity: position
                   for position, identity in enumerate(ordered_delete_ids)}
    ordered_candidates = sorted(enumerate(candidate_records), key=lambda pair: (
        0 if pair[1]["data_id"] in delete_rank else 1,
        delete_rank.get(pair[1]["data_id"], pair[0]), pair[0]))
    decisions = [decisions_by_key[(item["data_id"], item["location"])]
                 for _, item in ordered_candidates]

    inventory_snapshot = [dict(data_id=item["data_id"],
        location=item["location"], spec=(item["spec"].document
            if isinstance(item["spec"], DatasetSpec) else item["spec"]),
        source_origin=item["source_origin"],
        provenance=item["provenance"])
        for item in inventory]
    return dict(schema=1, publication_revision=revision,
        project_id=expected_project_id, receipts=receipt_snapshot,
        pinned_actual_ids=sorted(pinned), candidates=list(candidate_records),
        inventory=inventory_snapshot, decisions=decisions,
        complete_index_inventory=True, complete_receipt_inventory=True)
