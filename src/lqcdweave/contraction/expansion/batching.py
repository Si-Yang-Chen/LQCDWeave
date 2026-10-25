"""Bind dense native Diagram scans to full-task physical time slabs.

Each compatible compute batch keeps coordinate lanes on the Einstein equation,
so the backend contracts them together instead of wrapping scalar calls in a
Python loop. Physical slab reads are shared across all batches in the task.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import replace
import math

import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, BatchSpec, BatchTensorBinding, CoordinatePoint,
    CoordinateSpec, DiagramBindingProvider, DiagramInputs, TensorAxisMetadata,
    ContractionTerm, TensorSpec, ContractionProblem, SourceSpec,
    SourcePreparation, StorageBlock, TensorDescription, group_coordinate_batches,
    SparseBridgeSpec, TensorPreparation,
)
from lqcdweave.contraction.results.api import (
    build_output_recipe, estimate_output_retention_bytes,
)
from .tensor_description import stable_binding_value


class BatchDiagramBinding:
    """Compose a fixed-coordinate native adapter into compatible dense batches.

    It fails during metadata description for incompatible lane topology,
    sparse P2P inputs, or readers without dense slab APIs; no numerical getter
    runs before those checks.
    """

    def __init__(self, fixed_adapter, *, memory_limit=None, host_memory_limit=None,
                 reserved_host_bytes=0):
        if not callable(getattr(fixed_adapter, "describe", None)):
            raise TypeError("fixed_adapter must implement DiagramBindingProvider.describe")
        self.fixed_adapter = fixed_adapter
        self.memory_limit = memory_limit
        self.host_memory_limit = host_memory_limit
        self.reserved_host_bytes = reserved_host_bytes

    def describe(self, correlator, *, bindings, coordinates, batch):
        if not coordinates.scan_shape:
            return self.fixed_adapter.describe(
                correlator, bindings=bindings, coordinates=coordinates, batch=batch,
            )

        lanes = []
        for point in coordinates.points:
            lane_coordinates = CoordinateSpec(
                "bound", (), (), (), (CoordinatePoint(point.key, tuple(point.values)),), (),
            )
            lanes.append(self.fixed_adapter.describe(
                correlator, bindings=bindings, coordinates=lane_coordinates,
                batch=BatchSpec(),
            ))
        contract = _build_lane_contract(lanes)
        catalog = _collect_slab_catalog(lanes, contract)
        _collect_sparse_catalog(lanes, contract, catalog)
        # Preserve physical dependency descriptions across slab coalescing.
        # W uses these records for managed input production before X starts;
        # retain no lane graphs or numerical arrays here.
        catalog["preparation_inputs"] = tuple(
            (lane.read_requests, lane.loader.input_readers) for lane in lanes
            if lane.read_requests and hasattr(lane.loader, "input_readers")
            and any(getattr(reader, "requires_preparation", False) is True
                    for reader in lane.loader.input_readers.values()))
        grouping_batch = _effective_batch_spec(coordinates, batch)
        candidate_widths = (None,)
        if batch.size == "auto":
            recipe = build_output_recipe(
                correlator, contract["first"].diagram_terms,
                contract["first"].term_ids, coordinates,
                diagram_recipes=contract["first"].diagram_recipes,
            )
            assembly_bytes = estimate_output_retention_bytes(
                recipe, contract["first"].problem, (), coordinates,
            )
            candidate_widths = _auto_batch_widths(
                coordinates, grouping_batch, lanes, contract, catalog,
                assembly_bytes=assembly_bytes,
                memory_limit=self.memory_limit,
                host_memory_limit=self.host_memory_limit,
                reserved_host_bytes=self.reserved_host_bytes,
            )

        batch_candidates = {}
        for width in candidate_widths:
            candidate_batch = (grouping_batch if width is None else
                               BatchSpec(width, grouping_batch.axis))
            groups = group_coordinate_batches(coordinates, candidate_batch)
            task_loader = _BatchTaskLoader(catalog)
            task_batches = tuple(_combine_lanes(
                lanes, coordinates, batch, group, contract=contract,
                catalog=catalog, loader=task_loader,
            ) for group in groups)
            if batch.size == "auto":
                batch_candidates[width] = task_batches
            if not batch_candidates or width == candidate_widths[0]:
                selected_batches = task_batches
                selected_loader = task_loader

        task_batches = selected_batches
        task_loader = selected_loader
        first = task_batches[0]
        return DiagramInputs(
            first.problem, catalog["blocks"], first.preparations, task_loader,
            coordinates, batch, first.term_ids,
            diagram_terms=first.diagram_terms,
            diagram_recipes=first.diagram_recipes,
            tensor_requests=first.tensor_requests,
            batch_topology=first.batch_topology,
            task_batches=tuple(task_batches),
            batch_candidates=batch_candidates,
        )

def _leaf_bindings(lane, key):
    """Read the existing preparation tree; composite vertices stay opaque to P."""
    binding = lane.batch_bindings.get(key)
    if binding is None:
        binding = getattr(lane.loader, '_batch_bindings', {}).get(key)
    if binding is not None:
        return (binding,)
    prep = getattr(lane.loader, 'preparations', {}).get(key)
    if prep is not None and prep.recipe is not None and prep.recipe.kind == 'constant':
        return ()
    if prep is None or prep.recipe is None or prep.recipe.kind != 'linear_combination':
        raise NotImplementedError(
            'coordinate batching requires native slab bindings for preparation leaves')
    return tuple(binding for member in prep.recipe.inputs
                 for binding in _leaf_bindings(lane, member))


def _build_lane_contract(lanes):
    if not lanes or any(not isinstance(item, DiagramInputs) for item in lanes):
        raise TypeError("fixed-coordinate adapter must return DiagramInputs for every lane")
    first = lanes[0]
    if first.problem.backend != "numpy":
        raise NotImplementedError("native dense coordinate batching currently supports NumPy inputs only")
    term_shapes = tuple((term.subscripts, len(term.inputs)) for term in first.problem.terms)
    for lane_index, item in enumerate(lanes):
        if item.term_ids != first.term_ids:
            raise ValueError(f"coordinate lane {lane_index} has different term IDs")
        if item.batch_topology != first.batch_topology:
            raise ValueError(
                f"coordinate lane {lane_index} changes sector, point-pool, or coefficient topology"
            )
        if item.diagram_recipes != first.diagram_recipes or item.diagram_terms != first.diagram_terms:
            raise ValueError(f"coordinate lane {lane_index} changes its sampling/current output recipe")
        if tuple((term.subscripts, len(term.inputs)) for term in item.problem.terms) != term_shapes:
            raise ValueError(f"coordinate lane {lane_index} changes contraction equations")
        if item.problem.backend != first.problem.backend:
            raise ValueError("coordinate lanes use different contraction backends")

    # The complete lane-key signature handles coordinate-specific deduplication:
    # when two propagator occurrences coincide in one lane, they still retain
    # distinct roles if later lanes separate them.
    role_signatures, role_by_signature, term_roles = [], {}, []
    for term_index, term in enumerate(first.problem.terms):
        roles = []
        for operand_index in range(len(term.inputs)):
            signature = tuple(item.problem.terms[term_index].inputs[operand_index]
                              for item in lanes)
            role = role_by_signature.get(signature)
            if role is None:
                role = len(role_signatures)
                role_by_signature[signature] = role
                role_signatures.append(signature)
            roles.append(role)
        term_roles.append(tuple(roles))

    # Whole vertices hide representation leaves from the outer Term, but their
    # physical dependencies still need the same coordinate binding as legs.
    for join_index, join in enumerate(first.problem.baryon_joins):
        if any(len(item.problem.baryon_joins) != len(first.problem.baryon_joins) for item in lanes):
            raise ValueError('coordinate lanes change baryon vertex topology')
        keys = join.core.inputs if join.core is not None else (join.spin, join.elemental)
        for position in range(len(keys)):
            signature = tuple((item.problem.baryon_joins[join_index].core.inputs
                if item.problem.baryon_joins[join_index].core is not None else
                (item.problem.baryon_joins[join_index].spin, item.problem.baryon_joins[join_index].elemental))[position]
                for item in lanes)
            if signature not in role_by_signature:
                role_by_signature[signature] = len(role_signatures)
                role_signatures.append(signature)

    lane_specs = []
    for role, signature in enumerate(role_signatures):
        specs = tuple(item.problem.specs[signature[lane]] for lane, item in enumerate(lanes))
        if any(spec.shape != specs[0].shape or spec.dtype != specs[0].dtype
               for spec in specs[1:]):
            raise ValueError(f"coordinate lanes have incompatible TensorSpec for role {role}")
        lane_specs.append(specs[0])
        for lane_index, item in enumerate(lanes):
            key = signature[lane_index]
            for binding in _leaf_bindings(item, key):
                if (len(binding.physical_slabs) != 1 or len(binding.times) != 1
                        or len(binding.lane_materializers) != 1):
                    raise NotImplementedError(
                        'batch adapter requires one physical slab per preparation leaf')

    dynamic_roles = set()
    for role, signature in enumerate(role_signatures):
        constants = tuple(item.preparations[signature[index]].recipe
                          for index, item in enumerate(lanes))
        if constants[0] is not None and constants[0].kind == 'constant':
            if any(recipe != constants[0] for recipe in constants[1:]):
                raise ValueError('coordinate lanes change fixed spin coefficients')
        physical = tuple(
            tuple((binding.physical_slabs[0].identity, binding.times[0])
                  for binding in _leaf_bindings(item, signature[lane]))
            for lane, item in enumerate(lanes)
        )
        if len(set(physical)) > 1:
            dynamic_roles.add(role)
    lane_key_roles = tuple({signature[lane_index]: role
                            for role, signature in enumerate(role_signatures)}
                           for lane_index in range(len(lanes)))
    bridges_by_lane = []
    for lane_index, item in enumerate(lanes):
        by_scene = {}
        for bridge in item.problem.sparse_bridges:
            scene = bridge.scene
            if type(scene) is not int:
                raise NotImplementedError("coordinate sparse batching requires an explicit bridge scene index")
            by_scene.setdefault(scene, []).append(bridge)
        if any(len(items) > 2 for items in by_scene.values()):
            raise NotImplementedError(
                "coordinate sparse batching supports at most two P2P bridges per term"
            )
        grouped = {}
        for scene, scene_bridges in by_scene.items():
            for ordinal, bridge in enumerate(scene_bridges):
                grouped[(scene, ordinal)] = bridge
        bridges_by_lane.append(grouped)
    bridge_keys = tuple(sorted(bridges_by_lane[0]))
    if any(tuple(sorted(item)) != bridge_keys for item in bridges_by_lane[1:]):
        raise ValueError("coordinate lanes must expose the same ordered sparse bridge groups")
    sparse_bridges = []
    for bridge_key in bridge_keys:
        scene, ordinal = bridge_key
        lane_entries = []
        first_bridge = None
        role_signature = None
        for lane_index, item in enumerate(lanes):
            bridge = bridges_by_lane[lane_index][bridge_key]
            try:
                left_role = lane_key_roles[lane_index][bridge.left]
                right_role = lane_key_roles[lane_index][bridge.right]
            except KeyError as exc:
                raise ValueError("sparse bridge endpoints must be contraction operand roles") from exc
            current_signature = (
                left_role, right_role, bridge.used_np, bridge.used_np_right,
                tuple(bridge.left_axes), tuple(bridge.right_axes), bridge.same_point,
                bridge.transpose_bridge, bridge.part_count, bridge.reverse_operands,
            )
            if role_signature is None:
                role_signature = current_signature
                first_bridge = bridge
            elif current_signature != role_signature:
                raise ValueError("coordinate lanes change sparse bridge topology or endpoint extents")
            source = item.problem.sources[bridge.record]
            if (not isinstance(source.metadata, Mapping)
                    or source.metadata.get("family") != "p2p-record"):
                raise ValueError("unbatched lane sparse sources must retain the p2p-record family")
            endpoint_bindings = source.metadata.get("logical_endpoint_axis_bindings")
            if endpoint_bindings is None:
                # Older/custom SourceLoader fixtures can lack the optional
                # endpoint-view description.  Real DatasetElemental records
                # provide it; compare it across every coordinate lane so a
                # source dagger can never rotate the batch topology mid-axis.
                endpoint_axis_topology = None
            elif isinstance(endpoint_bindings, Mapping):
                endpoint_items = []
                for name in ("field", "bar"):
                    endpoint_data = endpoint_bindings.get(name)
                    if not isinstance(endpoint_data, Mapping):
                        raise ValueError(
                            "sparse lane metadata has incomplete field/bar raw-axis topology"
                        )
                    endpoint_items.append((name, endpoint_data.get("raw_axis")))
                endpoint_axis_topology = tuple(endpoint_items)
                if any(raw_axis is None for _, raw_axis in endpoint_axis_topology):
                    raise ValueError("sparse lane metadata has incomplete field/bar raw-axis topology")
            else:
                raise ValueError("sparse logical endpoint axis bindings must be a mapping")
            if lane_entries and endpoint_axis_topology != lane_entries[0][2]:
                raise ValueError(
                    "coordinate lanes change sparse field/bar axis topology or source-dagger orientation"
                )
            preparation = item.preparations[bridge.record]
            if len(preparation.blocks) != 1 or preparation.blocks[0] not in item.blocks:
                raise NotImplementedError("sparse lane materialization requires one declared source block")
            term = item.problem.terms[scene]
            pattern = term.subscripts
            if len(pattern.inputs) != len(term.inputs):
                raise ValueError("term labels do not match its ordered tensor operands")
            bridge_operand_pairs = tuple(
                (left_index, right_index)
                for left_index, left_key in enumerate(term.inputs)
                if left_key == bridge.left
                for right_index, right_key in enumerate(term.inputs)
                if right_key == bridge.right and right_index != left_index
            )
            if len(bridge_operand_pairs) != 1:
                raise NotImplementedError(
                    "coordinate sparse batching needs one unambiguous ordered operand pair per bridge"
                )
            left_index, right_index = bridge_operand_pairs[0]
            left_labels, right_labels = pattern.inputs[left_index], pattern.inputs[right_index]
            left_point = left_labels[bridge.left_axes[1] % len(left_labels)]
            right_point = right_labels[bridge.right_axes[1] % len(right_labels)]
            filter_points = bridge.same_point or left_point == right_point
            output_labels = pattern.output
            if left_point in output_labels or right_point in output_labels:
                raise NotImplementedError(
                    "coordinate sparse batching does not retain point labels in the result"
                )
            lane_entries.append((lane_index, bridge, endpoint_axis_topology))
        left_role, right_role = role_signature[:2]
        # Sparse records may differ even when endpoint propagators are the same
        # physical tensor. Carry explicit lane axes so the X kernel cannot form
        # an accidental left/right Cartesian product.
        dynamic_roles.update((left_role, right_role))
        sparse_bridges.append({
            "key": bridge_key,
            "scene": scene,
            "ordinal": ordinal,
            "roles": (left_role, right_role),
            "template": first_bridge,
            "filter_points": filter_points,
            "lanes": tuple((lane_index, bridge)
                           for lane_index, bridge, _ in lane_entries),
        })
    return {
        "first": first,
        "role_signatures": tuple(role_signatures),
        "term_roles": tuple(term_roles),
        "lane_specs": tuple(lane_specs),
        "dynamic_roles": frozenset(dynamic_roles),
        "sparse_bridges": tuple(sparse_bridges),
    }


def _collect_slab_catalog(lanes, contract):
    templates, times_by_identity, providers = {}, {}, {}
    role_signatures = contract["role_signatures"]
    for lane_index, item in enumerate(lanes):
        for signature in role_signatures:
            for binding in _leaf_bindings(item, signature[lane_index]):
                slab = binding.physical_slabs[0]
                time = binding.times[0]
                templates.setdefault(slab.identity, slab)
                times_by_identity.setdefault(slab.identity, set()).add(time)
                providers.setdefault(slab.identity, {})[id(slab)] = slab

    blocks, block_by_identity, metadata_by_block, positions = {}, {}, {}, {}
    read_workspace_by_block = {}
    for identity, slab in templates.items():
        selected_times = tuple(sorted(times_by_identity[identity]))
        metadata = slab.describe_slab(selected_times)
        if not isinstance(metadata, TensorAxisMetadata):
            raise TypeError("native describe_slab must return TensorAxisMetadata")
        if (metadata.sparse_layout != "dense" or not metadata.axes
                or metadata.axes[0] != "coordinate_batch"
                or metadata.shape[0] != len(selected_times)):
            raise ValueError("native slab metadata must describe a dense leading coordinate_batch axis")
        signature = _metadata_signature(metadata)
        read_workspace = 0
        for provider in providers[identity].values():
            candidate = provider.describe_slab(selected_times)
            if not isinstance(candidate, TensorAxisMetadata) or _metadata_signature(candidate) != signature:
                raise ValueError(
                    "shared physical slab identity has inconsistent shape/dtype/axes/selection metadata"
                )
            if provider.read_workspace_bytes is not None:
                workspace = provider.read_workspace_bytes(candidate)
                if type(workspace) is not int or workspace < 0:
                    raise ValueError("read_slab_workspace_bytes must return nonnegative integer bytes")
                read_workspace = max(read_workspace, workspace)
        block_key = f"slab:{len(blocks)}"
        byte_count = math.prod(metadata.shape) * metadata.dtype.itemsize
        blocks[block_key] = StorageBlock(
            dataset=f"native-slab:{identity!r}",
            selection=metadata.selection_identity,
            read_bytes=byte_count, host_bytes=byte_count, read_kind="dense",
            staging_host_bytes=read_workspace,
        )
        read_workspace_by_block[block_key] = read_workspace
        block_by_identity[identity] = block_key
        metadata_by_block[block_key] = metadata
        positions[identity] = {time: index for index, time in enumerate(selected_times)}
    return {
        "templates": templates,
        "times": {key: tuple(sorted(value)) for key, value in times_by_identity.items()},
        "blocks": blocks,
        "block_by_identity": block_by_identity,
        "metadata_by_block": metadata_by_block,
        "positions": positions,
        "read_workspace_by_block": read_workspace_by_block,
    }


def _collect_sparse_catalog(lanes, contract, catalog):
    """Add sparse source blocks once per explicit physical block identity."""
    blocks = dict(catalog["blocks"])
    sparse_blocks = {}
    sparse_by_scene = {}
    sparse_by_group = {}
    block_by_identity = {}
    for bridge_group in contract["sparse_bridges"]:
        scene = bridge_group["scene"]
        lane_entries = []
        for lane_index, bridge in bridge_group["lanes"]:
            lane = lanes[lane_index]
            source = lane.problem.sources[bridge.record]
            preparation = lane.preparations[bridge.record]
            original_block_key = preparation.blocks[0]
            original_block = lane.blocks[original_block_key]
            # Physical identity is the adapter-declared dataset plus selected
            # view.  Byte extents and read policy are descriptors to validate
            # for that identity, not a way to manufacture a second block.
            block_identity = (original_block.dataset, original_block.selection)
            block_key = block_by_identity.get(block_identity)
            if block_key is None:
                block_key = f"sparse:{len(sparse_blocks)}"
                if block_key in blocks:
                    raise ValueError("coordinate sparse block key collides with the task catalog")
                blocks[block_key] = original_block
                block_by_identity[block_identity] = block_key
            else:
                prior = sparse_blocks[block_key]
                if (prior["block"] != original_block
                        or _sparse_raw_source_descriptor(prior["source"])
                        != _sparse_raw_source_descriptor(source)):
                    raise ValueError(
                        "one physical sparse block identity has inconsistent block or raw record descriptors"
                    )
                # SourceLoader is the shared C protocol: read_block owns the
                # raw value for this execution, and materialize consumes only
                # the block mapping it is passed.  That contract lets distinct
                # lane loaders apply their own source preparation to one
                # descriptor-verified physical read.  Fail during description
                # when an adapter does not expose the required protocol.
            if (not callable(getattr(lane.loader, "read_block", None))
                    or not callable(getattr(lane.loader, "materialize", None))):
                raise TypeError("sparse coordinate lanes require the public SourceLoader protocol")
            entry = {
                "lane_index": lane_index,
                "bridge": bridge,
                "source_key": bridge.record,
                "source": source,
                "preparation": preparation,
                "original_block_key": original_block_key,
                "block_key": block_key,
                "loader": lane.loader,
                "block": original_block,
            }
            sparse_blocks.setdefault(block_key, entry)
            lane_entries.append(entry)
        group_key = bridge_group.get("key", (scene, bridge_group.get("ordinal", 0)))
        sparse_by_group[group_key] = tuple(lane_entries)
        # Preserve the original one-record lookup for existing single-bridge
        # callers/tests.  Two-bridge execution uses the explicit group key.
        sparse_by_scene.setdefault(scene, tuple(lane_entries))
    catalog["blocks"] = blocks
    catalog["sparse_blocks"] = sparse_blocks
    catalog["sparse_by_scene"] = sparse_by_scene
    catalog["sparse_by_group"] = sparse_by_group


def _sparse_raw_source_descriptor(source):
    """Return only raw-record identity fields, excluding lane provenance.

    The source metadata also carries endpoint times, pool IDs, and logical
    bindings. Those belong to each lane's preparation and must not prevent
    sharing an explicitly identical physical record block.
    """
    metadata = source.metadata
    raw_fields = ("family", "kind", "point_shape", "nnz", "value_dtype",
                  "index_dtype", "record_bytes", "backend")
    return (
        tuple(metadata.get(name) for name in raw_fields),
        source.region_bytes,
        source.staging_bytes,
        source.transfer_bytes,
        source.host_region_bytes,
        source.snapshot_bytes,
        source.read_bytes,
    )


def _auto_batch_widths(coordinates, batch, lanes, contract, catalog, *, assembly_bytes,
                       memory_limit, host_memory_limit, reserved_host_bytes):
    if memory_limit is None and host_memory_limit is None:
        return (1,)
    caps = [value for value in (memory_limit, host_memory_limit) if value is not None]
    budget = min(caps) - reserved_host_bytes
    sparse_block_keys = set(catalog.get("sparse_blocks", ()))
    raw_bytes = sum(block.host_bytes for key, block in catalog["blocks"].items()
                    if key not in sparse_block_keys)
    read_workspace = max(catalog["read_workspace_by_block"].values(), default=0)
    lane_snapshots = sum(math.prod(spec.shape) * spec.dtype.itemsize
                         for spec in contract["lane_specs"])
    workspace = 0
    for lane_index, lane in enumerate(lanes):
        for signature in contract["role_signatures"]:
            key = signature[lane_index]
            binding = lane.batch_bindings.get(key)
            workspace = max(workspace, binding.workspace_host_bytes if binding is not None
                            else lane.preparations[key].workspace_host_bytes)
    term_output_bytes = 0
    for term_index, term in enumerate(contract["first"].problem.terms):
        dtype = np.result_type(*(contract["lane_specs"][role].dtype
                                for role in contract["term_roles"][term_index]))
        labels = term.subscripts.inputs
        rhs = term.subscripts.output
        dimensions = {}
        for role, tensor_labels in zip(contract["term_roles"][term_index], labels, strict=True):
            for label, size in zip(tensor_labels, contract["lane_specs"][role].shape, strict=True):
                dimensions.setdefault(label, size)
        term_output_bytes += math.prod(dimensions[label] for label in rhs) * np.dtype(dtype).itemsize
    # TermValue snapshots and the assembled result may coexist during recipe
    # evaluation, so reserve both for the whole task, not only the current batch.
    output_bytes = max(assembly_bytes, 2 * len(coordinates.points) * term_output_bytes)
    all_groups = group_coordinate_batches(
        coordinates, BatchSpec("full", batch.axis),
    )
    max_group = max(group.actual_size for group in all_groups)
    candidates, candidate = {1, max_group}, 1
    while candidate < max_group:
        candidate *= 2
        candidates.add(min(candidate, max_group))
    feasible = []
    for width in sorted(candidates):
        groups = group_coordinate_batches(coordinates, BatchSpec(width, batch.axis))
        largest = max(group.actual_size for group in groups)
        sparse_peak = max(
            (_sparse_group_estimate(group, contract, catalog)
             for group in groups),
            default=0,
        )
        estimate = (raw_bytes + largest * lane_snapshots + workspace
                    + read_workspace + sparse_peak + output_bytes)
        if estimate <= budget:
            feasible.append(width)
    if not feasible:
        sparse_peak = max(
            (_sparse_group_estimate(group, contract, catalog)
             for group in group_coordinate_batches(coordinates, BatchSpec(1, batch.axis))),
            default=0,
        )
        raise MemoryError(
            f"automatic coordinate batching cannot fit its smallest candidate: "
            f"estimated={raw_bytes + lane_snapshots + output_bytes + workspace + read_workspace + sparse_peak} bytes, budget={budget}"
        )
    return tuple(feasible)


def _sparse_group_estimate(group, contract, catalog):
    """Conservative known P/P bytes for one candidate coordinate group."""
    peak = 0
    lane_indices = set(group.point_indices)
    scene_peaks = {}
    for bridge_group in contract["sparse_bridges"]:
        groups = catalog.get("sparse_by_group", {})
        group_key = bridge_group.get("key", (bridge_group["scene"], 0))
        group_entries = groups.get(group_key, catalog["sparse_by_scene"][bridge_group["scene"]])
        entries = [entry for entry in group_entries
                   if entry["lane_index"] in lane_indices]
        if not entries:
            continue
        bridge = bridge_group["template"]
        lane_count = len(entries)
        part_count = bridge.part_count
        lane_bounds = tuple(entry["bridge"].nnz_upper for entry in entries)
        selected_block_keys = tuple(dict.fromkeys(entry["block_key"] for entry in entries))
        raw = sum(catalog["blocks"][key].host_bytes for key in selected_block_keys)
        # The source materializers retain one transformed record per lane
        # until X has normalized the complete batch.  The C preparation
        # workspace is that declared copy extent (zero for borrowed raw data).
        materialized = sum(entry["preparation"].workspace_host_bytes for entry in entries)
        lane_raw = tuple(entry["source"].metadata for entry in entries)
        lane_value_bytes = tuple(
            np.dtype(raw_meta["value_dtype"]).itemsize
            if raw_meta["kind"] == "sparse" else np.dtype(np.complex128).itemsize
            for raw_meta in lane_raw
        )
        lane_rows = tuple(part_count * bound for bound in lane_bounds)
        left_spec = contract["lane_specs"][bridge_group["roles"][0]]
        right_spec = contract["lane_specs"][bridge_group["roles"][1]]
        left_axes = tuple(bridge.left_axes)
        right_axes = tuple(bridge.right_axes)
        left_external = math.prod(
            size for axis, size in enumerate(left_spec.shape)
            if axis not in {value % len(left_spec.shape) for value in left_axes}
        )
        right_external = math.prod(
            size for axis, size in enumerate(right_spec.shape)
            if axis not in {value % len(right_spec.shape) for value in right_axes}
        )
        spin_left = left_spec.shape[left_axes[0] % len(left_spec.shape)]
        spin_right = right_spec.shape[right_axes[0] % len(right_spec.shape)]
        color_left = left_spec.shape[left_axes[2] % len(left_spec.shape)]
        color_right = right_spec.shape[right_axes[2] % len(right_spec.shape)]
        normalized = sum(
            rows * (2 * np.dtype("int64").itemsize + color_left * color_right * value_bytes)
            + part_count * (spin_left * spin_right * 16 + 16)
            for rows, value_bytes in zip(lane_rows, lane_value_bytes, strict=True)
        )
        # The executor fills one row buffer across lanes and parts, flushing
        # only at 64 rows; a chunk can therefore contain the sum of every
        # lane/part bound, not just one part's largest bound.
        chunk = min(sum(lane_rows), 64)
        gathered = chunk * (
            left_external * spin_left * color_left * left_spec.dtype.itemsize
            + spin_right * color_right * right_external * right_spec.dtype.itemsize
        )
        # The executor stacks row values from different coordinate lanes in a
        # single bounded chunk.  NumPy promotes that pack across lane dtypes;
        # max(itemsize) is not sufficient for (for example) float64 together
        # with complex64, whose common stack dtype is complex128.
        stacked_value_itemsize = np.result_type(*(
            np.dtype(raw["value_dtype"]) if raw["kind"] == "sparse"
            else np.dtype(np.complex128)
            for raw in lane_raw
        )).itemsize if lane_raw else np.dtype(np.complex128).itemsize
        # Filtering is performed separately on each raw lane/part before row
        # packing, so its retained advanced-index bases keep that lane's dtype.
        filtered_value_itemsize = max(lane_value_bytes, default=16)
        # X packs three int64 row selectors, stacked gamma/coefficient/value
        # arrays, and (for diagonal bridge filtering) a boolean mask plus
        # advanced-index copies. Gamma/coefficient lane metadata is retained
        # even when every COO part is empty and is included in `normalized`.
        packed_chunk = chunk * (
            3 * np.dtype("int64").itemsize
            + spin_left * spin_right * np.dtype(np.complex128).itemsize
            + np.dtype(np.complex128).itemsize
            + color_left * color_right * stacked_value_itemsize
        )
        if bridge_group.get("filter_points", False) and chunk:
            # The diagonal mask is rebound per sparse chunk, so two masks can
            # overlap briefly. More importantly, each row value stored by X is
            # a view into the whole advanced-indexed chunk. At a flush, pending
            # views can retain one partly consumed base of up to F rows, up to
            # C-1 rows from the pending tail, and the chunk that triggered the
            # flush. Count those full bases (for both pair and value arrays),
            # not just one C-row chunk.
            max_filtered_part = min(64, max(lane_bounds, default=0))
            filtered_base_rows = (
                chunk + 2 * max_filtered_part - 2
                if max_filtered_part else 0
            )
            filtered_chunk = (
                filtered_base_rows * (
                    2 * np.dtype("int64").itemsize
                    + color_left * color_right * filtered_value_itemsize
                )
                + 2 * max_filtered_part
            )
        else:
            filtered_chunk = 0
        contributions = chunk * left_external * right_external * 16
        output = lane_count * left_external * right_external * 16
        # moveaxis is a view, but the subsequent reshape can copy a whole
        # endpoint when its selected axis layout is not contiguous.
        reshape_copies = lane_count * (
            math.prod(left_spec.shape) * left_spec.dtype.itemsize
            + math.prod(right_spec.shape) * right_spec.dtype.itemsize
        )
        normalization_workspace = max((
            int(raw_meta["nnz"]) * (
                3 + (2 * np.dtype("int64").itemsize
                     if np.dtype(raw_meta["index_dtype"]) != np.dtype("int64") else 0)
            ) for raw_meta in lane_raw
        ), default=0)
        preparation_peak = raw + materialized + normalized + normalization_workspace
        execution_peak = (raw + normalized + reshape_copies + gathered
                          + packed_chunk + filtered_chunk + contributions + output)
        group_peak = max(preparation_peak, execution_peak)
        scene_peaks[bridge_group["scene"]] = (
            scene_peaks.get(bridge_group["scene"], 0) + group_peak
        )
    peak = max(scene_peaks.values(), default=0)
    return peak


def _effective_batch_spec(coordinates, batch):
    axis = batch.axis
    if coordinates.mode == "pairs":
        axis = "pair"
    elif axis is None and len(coordinates.scan_axes) > 1:
        # Auto uses a deterministic finite candidate axis when the caller has
        # several scanned axes; other axes become independent outer groups.
        axis = coordinates.scan_axes[-1]
    return BatchSpec(batch.size, axis)


class _BatchTaskLoader:
    # All slabs are bound in the task catalog; reading one never consults the
    # active batch, whose mutable selection is used only by materialization.
    prefetch_across_batches = True

    def __init__(self, catalog):
        self.catalog = catalog
        self.preparations, self.specs = {}, {}
        self.raw_dtypes = tuple(metadata.dtype
                                for metadata in catalog.get("metadata_by_block", {}).values())
        self.preparation_inputs = catalog.get("preparation_inputs", ())
        self.slab_identity_by_block = {
            key: identity for identity, key in catalog["block_by_identity"].items()
        }
        self.materializations = {}
        self.active_batch = None
        self.cache = {}
        self.last_use = {}

    @contextmanager
    def activate(self, index):
        if index not in self.materializations:
            raise ValueError(f"unknown materialization batch {index}")
        previous = self.active_batch
        self.active_batch = index
        try:
            yield self
        finally:
            self.active_batch = previous

    def register(self, index, *, specs, lane_specs, role_entries, dynamic_roles,
                 sparse_records):
        self.materializations[index] = (dict(specs), tuple(lane_specs), dict(role_entries),
                                        frozenset(dynamic_roles), dict(sparse_records))

    def register_demands(self, index, block_keys):
        for key in block_keys:
            self.last_use[key] = max(index, self.last_use.get(key, -1))

    def _check_read_identity(self, key, block):
        try:
            expected = self.catalog["blocks"][key]
        except KeyError as exc:
            raise KeyError(f"unknown full-task slab block {key!r}") from exc
        # Retargeting changes allocation/transfer budgets, not the physical read.
        if (block.dataset != expected.dataset or block.selection != expected.selection
                or block.read_bytes != expected.read_bytes or block.read_kind != expected.read_kind):
            raise ValueError(f"physical slab block {key!r} differs from the full-task catalog")

    def read_block(self, key, block):
        self._check_read_identity(key, block)
        sparse_entry = self.catalog.get("sparse_blocks", {}).get(key)
        if sparse_entry is not None:
            if key not in self.cache:
                self.cache[key] = sparse_entry["loader"].read_block(
                    sparse_entry["original_block_key"], sparse_entry["block"],
                )
            return self.cache[key]
        if key not in self.cache:
            identity = self.slab_identity_by_block[key]
            times = self.catalog["times"][identity]
            value = self.catalog["templates"][identity].read_slab(times)
            metadata = self.catalog["metadata_by_block"][key]
            if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
                raise ValueError(f"slab reader returned invalid shape/dtype for {key}")
            self.cache[key] = value
        return self.cache[key]

    def can_prefetch_block(self, key, block):
        identity = self.slab_identity_by_block.get(key)
        return (identity is not None and
                self.catalog["templates"][identity].read_slab_in_background is not None)

    def prefetch_host_bytes(self, key, block):
        return 2 * block.host_bytes + block.staging_host_bytes

    def prefetch_block(self, key, block):
        self._check_read_identity(key, block)
        identity = self.slab_identity_by_block[key]
        read = self.catalog["templates"][identity].read_slab_in_background
        value = read(self.catalog["times"][identity])
        metadata = self.catalog["metadata_by_block"][key]
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError(f"slab reader returned invalid shape/dtype for {key}")
        return np.array(value, copy=True)

    def finish_prefetch(self, key, block, value):
        if key in self.cache:
            raise RuntimeError("prefetch attempts a second task slab read")
        self.cache[key] = value
        return value

    def close_prefetch(self, key, value):
        pass

    def close_block(self, key, value):
        # Direct replay closes at each batch boundary; planned replay closes
        # at the globally joined final use. Retain blocks only while a later
        # batch still needs them, matching the task-level storage schedule.
        sparse_entry = self.catalog.get("sparse_blocks", {}).get(key)
        if sparse_entry is not None:
            if (self.active_batch is None
                    or self.active_batch < self.last_use.get(key, self.active_batch)):
                return
            if key not in self.cache:
                return
            close = getattr(sparse_entry["loader"], "close_block", None)
            if callable(close):
                close(sparse_entry["original_block_key"], self.cache[key])
            self.cache.pop(key, None)
            return
        if self.active_batch is not None and self.active_batch >= self.last_use.get(key, self.active_batch):
            self.cache.pop(key, None)

    def materialize(self, key, source, raw_blocks):
        if self.active_batch is None:
            raise RuntimeError("batch materializer has no active task coordinate batch")
        specs, lane_specs, role_entries, dynamic_roles, sparse_records = self.materializations[self.active_batch]
        if key in sparse_records:
            record = sparse_records[key]
            lane_parts = []
            for entry in record["lanes"]:
                raw = raw_blocks[entry["block_key"]]
                value = entry["loader"].materialize(
                    entry["source_key"], entry["source"],
                    {entry["original_block_key"]: raw},
                )
                lane_parts.append(tuple(value))
            return {
                "layout": "coordinate_batch_coo",
                "point_keys": record["point_keys"],
                "lanes": tuple(lane_parts),
            }
        if key not in specs:
            raise KeyError(key)
        role = int(key.split(":", 1)[1])
        target = specs[key]
        rows = role_entries[role]
        if role not in dynamic_roles:
            block_key, lane_index, materializer = rows[0]
            value = materializer(raw_blocks[block_key][lane_index])
            if tuple(value.shape) != target.shape or np.dtype(value.dtype) != target.dtype:
                raise ValueError(f"lane materializer returned invalid shape/dtype for {key}")
            return value
        lane_spec = lane_specs[role]
        result = np.empty(target.shape, dtype=target.dtype)
        for coordinate_index, (block_key, lane_index, materializer) in enumerate(rows):
            value = materializer(raw_blocks[block_key][lane_index])
            if tuple(value.shape) != lane_spec.shape or np.dtype(value.dtype) != lane_spec.dtype:
                raise ValueError(f"lane materializer returned invalid shape/dtype for {key}")
            result[coordinate_index] = value
            # The staging result owns this lane now. Release its temporary
            # before creating the next lane's transformed value.
            del value
        return result

    def close_all(self):
        errors = []
        for key, entry in self.catalog.get("sparse_blocks", {}).items():
            if key not in self.cache:
                continue
            close = getattr(entry["loader"], "close_block", None)
            if callable(close):
                try:
                    close(entry["original_block_key"], self.cache[key])
                except Exception as exc:
                    errors.append(exc)
                else:
                    self.cache.pop(key, None)
        self.cache.clear()
        self.active_batch = None
        if errors:
            raise errors[0]


def _lift_vertex_lane(lane, key, prefix, catalog, closure, closure_specs):
    """Rebind one saved vertex tree to slab slices, without batching its leaves."""
    member = prefix + ':' + key
    if member in closure:
        return member
    prep = lane.loader.preparations[key]
    spec = lane.loader.specs[key]
    recipe = prep.recipe
    if recipe.kind == 'constant':
        lifted = prep
    elif recipe.kind == 'linear_combination':
        inputs = tuple(_lift_vertex_lane(lane, child, prefix, catalog,
                                        closure, closure_specs)
                       for child in recipe.inputs)
        blocks = tuple(dict.fromkeys(block for child in inputs
                                     for block in closure[child].blocks))
        lifted = replace(prep, blocks=blocks, recipe=replace(recipe, inputs=inputs))
    else:
        binding, = _leaf_bindings(lane, key)
        slab, time = binding.physical_slabs[0], binding.times[0]
        block = catalog['block_by_identity'][slab.identity]
        position = catalog['positions'][slab.identity][time]
        lifted = replace(prep, blocks=(block,),
                         recipe=replace(recipe, inputs=(block,), raw_index=position))
    closure[member], closure_specs[member] = lifted, spec
    return member


def _combine_vertex_lanes(lanes, keys, base_spec, tensor_key, is_dynamic,
                          coordinate_batch, catalog, closure, closure_specs):
    selected = range(len(lanes)) if is_dynamic else range(1)
    members = tuple(_lift_vertex_lane(lanes[index], keys[index],
                    f'batch:{coordinate_batch.batch_index}:{tensor_key}:lane:{index}',
                    catalog, closure, closure_specs) for index in selected)
    blocks = tuple(dict.fromkeys(block for member in members
                                 for block in closure[member].blocks))
    shape = ((coordinate_batch.actual_size, *base_spec.shape)
             if is_dynamic else base_spec.shape)
    description = base_spec.description
    if description is not None:
        description = replace(description,
            axes=(('coordinate_batch', *description.axes) if is_dynamic else description.axes),
            source_bindings=blocks,
            bindings={**dict(description.bindings),
                      'coordinate_keys': coordinate_batch.point_keys,
                      'coordinate_values': tuple(
                          tuple((str(axis), stable_binding_value(value)) for axis, value in point.values)
                          for point in coordinate_batch.points),
                      'lane_bindings': tuple(dict(lane.problem.specs[key].description.bindings)
                                             for lane, key in zip(lanes, keys, strict=True))})
    spec = TensorSpec(shape, base_spec.dtype, description=description)
    snapshot = math.prod(shape) * spec.dtype.itemsize
    raw_bytes = sum(catalog['blocks'][block].read_bytes for block in blocks)
    source = SourceSpec(region=f'logical:{tensor_key}', region_bytes=raw_bytes,
        host_region_bytes=raw_bytes, snapshot_bytes=snapshot,
        metadata={'family': 'dense-batch', 'blocks': blocks})
    first = closure[members[0]]
    # While stacking, one scalar whole vertex and its internal workspace coexist
    # with the final batched snapshot. Leaves never gain a spurious lane axis.
    workspace = max(closure[member].workspace_host_bytes for member in members)
    if is_dynamic:
        workspace += math.prod(base_spec.shape) * base_spec.dtype.itemsize
    prep = replace(first, blocks=blocks, workspace_host_bytes=workspace,
        flops=sum(closure[member].flops for member in members),
        traffic_bytes=sum(closure[member].traffic_bytes for member in members),
        kernel_calls=sum(closure[member].kernel_calls for member in members),
        recipe=(TensorPreparation('stack', members) if is_dynamic else first.recipe))
    return spec, source, prep


def _combine_lanes(all_lanes, coordinates, batch, coordinate_batch, *, contract,
                   catalog, loader):
    first = contract["first"]
    role_signatures = contract["role_signatures"]
    term_roles = contract["term_roles"]
    lane_specs = contract["lane_specs"]
    dynamic_roles = contract["dynamic_roles"]
    global_indices = coordinate_batch.point_indices
    lanes = tuple(all_lanes[index] for index in global_indices)
    # A scalar dense batch uses exactly the same tensor ranks and equations as
    # a one-pair planning template, regardless of how many pairs the task has.
    # Sparse lane kernels still require their explicit coordinate axis.
    if coordinate_batch.actual_size == 1 and not contract["sparse_bridges"]:
        dynamic_roles = frozenset()

    specs, sources, preparations, tensor_requests = {}, {}, {}, {}
    aggregate_bindings, role_entries = {}, {}
    closure, closure_specs = {}, {}
    sparse_bridges, sparse_records = [], {}
    for role, signature in enumerate(role_signatures):
        keys = tuple(signature[index] for index in global_indices)
        if keys[0] not in lanes[0].batch_bindings:
            tensor_key = f'tensor:{role}'
            spec, source, prep = _combine_vertex_lanes(
                lanes, keys, lane_specs[role], tensor_key, role in dynamic_roles,
                coordinate_batch, catalog, closure, closure_specs)
            specs[tensor_key], sources[tensor_key], preparations[tensor_key] = spec, source, prep
            tensor_requests[tensor_key] = tuple(lane.tensor_requests[key]
                for lane, key in zip(lanes, keys, strict=True))
            # Saved recipes execute this role; there is no opaque fallback
            # callback that would bypass the preparation ownership contract.
            continue
        rows = []
        for global_index, lane in zip(global_indices, lanes, strict=True):
            old_key = signature[global_index]
            binding = lane.batch_bindings[old_key]
            slab, time = binding.physical_slabs[0], binding.times[0]
            identity = slab.identity
            block_key = catalog["block_by_identity"][identity]
            rows.append((identity, time, binding.lane_materializers[0], binding,
                         block_key, catalog["positions"][identity][time]))

        base_spec = lane_specs[role]
        tensor_key = f"tensor:{role}"
        is_dynamic = role in dynamic_roles
        tensor_shape = ((coordinate_batch.actual_size, *base_spec.shape)
                        if is_dynamic else base_spec.shape)
        source_blocks = tuple(dict.fromkeys(row[4] for row in rows))
        lane_descriptions = tuple(lane.problem.specs[signature[global_index]].description
                                  for global_index, lane in zip(global_indices, lanes, strict=True))
        description = None
        if lane_descriptions and all(item is not None for item in lane_descriptions):
            if any(item.axes != lane_descriptions[0].axes or item.role != lane_descriptions[0].role
                   for item in lane_descriptions[1:]):
                raise ValueError("batched lane logical tensor axes or roles differ")
            data_identities = tuple(item.data_identity for item in lane_descriptions)
            selection_ids = tuple(item.selection_identity for item in lane_descriptions)
            common_data_identity = data_identities[0] if len(set(data_identities)) == 1 else None
            composite_selection = (None if any(value is None for value in selection_ids)
                                   else repr(selection_ids))
            batch_axis_name = "coordinate_batch" if is_dynamic else None
            logical_axes = ((batch_axis_name, *lane_descriptions[0].axes)
                            if is_dynamic else lane_descriptions[0].axes)
            description = TensorDescription(
                axes=logical_axes,
                data_identity=common_data_identity,
                selection_identity=composite_selection,
                role=lane_descriptions[0].role,
                source_bindings=source_blocks,
                transform=(
                    ("operation", "coordinate-lane-batch"),
                    ("batch_dependent", is_dynamic),
                    ("lane_transforms", tuple(item.transform for item in lane_descriptions)),
                ),
                semantics={
                    "lane_semantics": tuple(dict(item.semantics) for item in lane_descriptions),
                    "batch_axis": None if batch.axis is None else str(batch.axis),
                    "batch_dependent": is_dynamic,
                },
                bindings={
                    "coordinate_keys": tuple(point.key for point in coordinate_batch.points),
                    "coordinate_values": tuple(
                        tuple((str(axis), stable_binding_value(value)) for axis, value in point.values)
                        for point in coordinate_batch.points
                    ),
                    "lane_bindings": tuple(dict(item.bindings) for item in lane_descriptions),
                    "source_times": tuple(row[1] for row in rows),
                },
            )
        tensor_spec = TensorSpec(tensor_shape, base_spec.dtype, description=description)
        specs[tensor_key] = tensor_spec
        snapshot_bytes = math.prod(tensor_spec.shape) * tensor_spec.dtype.itemsize
        raw_region_bytes = sum(catalog["blocks"][key].read_bytes for key in source_blocks)
        sources[tensor_key] = SourceSpec(
            region=f"logical:{tensor_key}", region_bytes=raw_region_bytes,
            host_region_bytes=raw_region_bytes, snapshot_bytes=snapshot_bytes,
            metadata={"family": "dense-batch", "blocks": source_blocks},
        )
        single_lane_workspace = max(row[3].workspace_host_bytes for row in rows)
        workspace = (max(snapshot_bytes, single_lane_workspace)
                     if is_dynamic else single_lane_workspace)
        preparations[tensor_key] = SourcePreparation(
            source_blocks, workspace_host_bytes=workspace,
            # Batch staging changes the live-byte bound, not the existing
            # single-lane transformation cost (e.g. borrowed propagators stay 0).
            kernel_calls=1 if single_lane_workspace else 0,
            dtype=tensor_spec.dtype.name,
        )
        lane_preps = tuple(lane.preparations[signature[index]]
                           for index, lane in zip(global_indices, lanes, strict=True))
        if all(prep.recipe is not None and prep.recipe.kind in {'identity', 'spin_spatial'}
               and prep.recipe.raw_index is None for prep in lane_preps):
            members = []
            selected = range(len(rows)) if is_dynamic else range(1)
            for index in selected:
                row, prep = rows[index], lane_preps[index]
                member = f'batch:{coordinate_batch.batch_index}:{tensor_key}:lane:{index}'
                leaf = replace(prep, blocks=(row[4],),
                    recipe=replace(prep.recipe, inputs=(row[4],), raw_index=row[5]))
                closure[member], closure_specs[member] = leaf, base_spec
                members.append(member)
            recipe = (TensorPreparation('stack', tuple(members)) if is_dynamic
                      else closure[members[0]].recipe)
            preparations[tensor_key] = replace(preparations[tensor_key], recipe=recipe)
        role_entries[role] = tuple((row[4], row[5], row[2]) for row in rows)
        aggregate_bindings[tensor_key] = BatchTensorBinding(
            logical_role=("operand-role", role),
            physical_slabs=tuple(row[3].physical_slabs[0] for row in rows),
            times=tuple(row[1] for row in rows),
            lane_materializers=tuple(row[2] for row in rows),
            batch_dependent=is_dynamic,
            topology_key=first.batch_topology,
            workspace_host_bytes=workspace,
        )
        tensor_requests[tensor_key] = tuple(
            all_lanes[global_index].tensor_requests[signature[global_index]]
            for global_index in global_indices
        )

    for bridge_group in contract["sparse_bridges"]:
        scene = bridge_group["scene"]
        bridge_key = bridge_group["key"]
        by_lane = {entry["lane_index"]: entry
                   for entry in catalog["sparse_by_group"][bridge_key]}
        lane_entries = tuple(by_lane[index] for index in global_indices)
        if len(lane_entries) != coordinate_batch.actual_size:
            raise ValueError("sparse lane source count does not match CoordinateBatch")
        bridge_template = bridge_group["template"]
        ordinal = bridge_group["ordinal"]
        record_key = (f"sparse-record:{scene}" if ordinal == 0
                      else f"sparse-record:{scene}:{ordinal}")
        lane_bounds = tuple(entry["bridge"].nnz_upper for entry in lane_entries)
        part_count = bridge_template.part_count
        maximum_rows = part_count * sum(lane_bounds)
        unique_block_keys = tuple(dict.fromkeys(entry["block_key"] for entry in lane_entries))
        record_bytes = sum(catalog["blocks"][key].read_bytes for key in unique_block_keys)
        host_region_bytes = sum(catalog["blocks"][key].host_bytes
                                for key in unique_block_keys)
        left_role, right_role = bridge_group["roles"]
        left_spec = lane_specs[left_role]
        right_spec = lane_specs[right_role]
        left_spin = left_spec.shape[bridge_template.left_axes[0] % len(left_spec.shape)]
        right_spin = right_spec.shape[bridge_template.right_axes[0] % len(right_spec.shape)]
        left_color = left_spec.shape[bridge_template.left_axes[2] % len(left_spec.shape)]
        right_color = right_spec.shape[bridge_template.right_axes[2] % len(right_spec.shape)]
        # Normalized records own one int64 pair row and one color block per
        # selected COO entry. Each lane/part also retains its gamma and scalar
        # coefficient even when its COO record has no rows.
        part_metadata_bytes = left_spin * right_spin * 16 + 16
        lane_raw_metadata = tuple(entry["source"].metadata for entry in lane_entries)
        packed_snapshot_bytes = sum(
            part_count * bound * (
                2 * np.dtype("int64").itemsize
                + left_color * right_color
                * max(np.dtype(raw["value_dtype"]).itemsize,
                      16 if raw["kind"] == "identity" else 0)
            ) + part_count * part_metadata_bytes
            for bound, raw in zip(lane_bounds, lane_raw_metadata, strict=True)
        )
        # sparse_pairs creates a valid mask and may convert narrower raw
        # indices before producing the normalized snapshot. Those temporaries
        # peak while processing one raw part; the final arrays are counted in
        # snapshot_bytes above.
        normalization_workspace = max((
            int(raw["nnz"]) * (
                3 + (2 * np.dtype("int64").itemsize
                     if np.dtype(raw["index_dtype"]) != np.dtype("int64") else 0)
            ) for raw in lane_raw_metadata
        ), default=0)
        blocks_for_record = unique_block_keys
        lane_sources = tuple({
            "region": entry["source"].region,
            "metadata": entry["source"].metadata,
            "block": entry["block_key"],
            "source_key": entry["source_key"],
            "materialized_point_shape": entry["bridge"].point_shape,
        } for entry in lane_entries)
        preparation_dtype = np.result_type(
            np.dtype(np.complex128),
            *(np.dtype(raw["value_dtype"]) if raw["kind"] == "sparse"
              else np.dtype(np.complex128) for raw in lane_raw_metadata),
        )
        sources[record_key] = SourceSpec(
            region=f"sparse-batch:{scene}:{ordinal}",
            region_bytes=record_bytes,
            host_region_bytes=host_region_bytes,
            staging_bytes=normalization_workspace,
            snapshot_bytes=packed_snapshot_bytes,
            metadata={
                "family": "p2p-coordinate-batch",
                "bridge_ordinal": ordinal,
                "coordinate_keys": coordinate_batch.point_keys,
                "lane_nnz_upper": lane_bounds,
                "point_shape": bridge_template.point_shape,
                "part_count": part_count,
                "lane_sources": lane_sources,
            },
        )
        preparations[record_key] = SourcePreparation(
            blocks_for_record,
            workspace_host_bytes=sum(entry["preparation"].workspace_host_bytes
                                     for entry in lane_entries),
            kernel_calls=1,
            dtype=preparation_dtype.name,
        )
        left_role, right_role = bridge_group["roles"]
        left_axes = tuple(axis if axis < 0 else axis + 1
                          for axis in bridge_template.left_axes)
        right_axes = tuple(axis if axis < 0 else axis + 1
                           for axis in bridge_template.right_axes)
        sparse_bridges.append(replace(
            bridge_template,
            left=f"tensor:{left_role}",
            right=f"tensor:{right_role}",
            record=record_key,
            nnz_upper=sum(lane_bounds),
            left_axes=left_axes,
            right_axes=right_axes,
        ))
        sparse_records[record_key] = {
            "point_keys": coordinate_batch.point_keys,
            "lanes": lane_entries,
        }

    batch_axis = (max(
        (axis for term in first.problem.terms
         for operand_axes in term.subscripts.inputs for axis in operand_axes),
        default=-1,
    ) + 1 if dynamic_roles else None)
    terms, output_axes = [], []
    for term_index, term in enumerate(first.problem.terms):
        pattern = term.subscripts
        if len(pattern.inputs) != len(term.inputs):
            raise ValueError("fixed contraction equation does not match its tensor operands")
        roles = term_roles[term_index]
        has_batch = any(role in dynamic_roles for role in roles)
        input_axes = tuple(
            (batch_axis, *axes) if role in dynamic_roles else axes
            for role, axes in zip(roles, pattern.inputs, strict=True)
        )
        output_axes_for_term = ((batch_axis, *pattern.output)
                               if has_batch else pattern.output)
        terms.append(ContractionTerm(tuple(f"tensor:{role}" for role in roles),
                                     AxisPattern(input_axes, output_axes_for_term),
                                     term.coefficient))
        output_axes.append(0 if has_batch else None)

    from lqcdweave.contracts.correlator.cores import lift_baryon_join
    source_map = {signature[global_indices[0]]: f'tensor:{role}'
                  for role, signature in enumerate(role_signatures)}
    joins = tuple(lift_baryon_join(join, source_map,
        {f'tensor:{role}' for role in dynamic_roles}, specs, batch_axis, coordinate_batch.actual_size)
        for join in lanes[0].problem.baryon_joins)
    problem = ContractionProblem(
        specs, tuple(terms), sources, sparse_bridges=tuple(sparse_bridges), baryon_joins=joins,
        backend=first.problem.backend,
        single_read_regions=True,
        factorized_vertices=first.problem.factorized_vertices,
    )
    loader.register(coordinate_batch.batch_index, specs=specs,
                    lane_specs=lane_specs, role_entries=role_entries,
                    dynamic_roles=dynamic_roles, sparse_records=sparse_records)
    loader.register_demands(
        coordinate_batch.batch_index,
        {block for preparation in preparations.values() for block in preparation.blocks},
    )
    loader.preparations.update(closure)
    loader.specs.update(closure_specs)
    return DiagramInputs(
        problem, catalog["blocks"], preparations, loader, coordinates, batch,
        first.term_ids, diagram_recipes=first.diagram_recipes,
        diagram_terms=first.diagram_terms, tensor_requests=tensor_requests,
        coordinate_batch=coordinate_batch, batch_bindings=aggregate_bindings,
        batch_topology=first.batch_topology,
        batch_output_axes=tuple(output_axes),
    )


def _metadata_signature(metadata):
    pools_by_batch = metadata.point_pool_ids_by_batch
    return (
        tuple(metadata.axes), tuple(metadata.shape), np.dtype(metadata.dtype),
        metadata.data_identity, metadata.selection_identity,
        metadata.sparse_layout, metadata.point_shape, metadata.nnz_upper,
        tuple(sorted(dict(metadata.point_pool_ids).items())),
        None if pools_by_batch is None else tuple(
            tuple(sorted(dict(item).items())) for item in pools_by_batch
        ),
    )


__all__ = ["BatchDiagramBinding"]
