"""Instantiate one selected template over coordinates with global ownership."""
from array import array
from collections import defaultdict
from dataclasses import fields, replace
from functools import partial
from heapq import heapify, heappop, heappush
from itertools import islice
from math import prod
from operator import attrgetter, itemgetter
from time import perf_counter

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, AxisPattern, CfgInstance, CfgPlan, CfgTemplate, StorageEvent, FactorizedWideReduction,
    DenseGemmGroup, MemoryPoint, MesonWholeKernel, SourcePreparation, SourceReadRequest, TensorAxisMetadata, TensorPreparation, TensorCoreSpec,
    dense_batch_leaves,
)
from .canonical_operations import axis_key
from .sparse_workspace import prepare_coordinate_batch_workspace
from . import cfg_rounding
from lqcdweave.contracts.correlator.api import device_allocation_bytes


_MEMORY_POINT_VALUES = attrgetter(*(field.name for field in fields(MemoryPoint)
                                   if field.name != 'action_index'))


def _remap_memory_point(point, index, *, debug):
    """Remap a native frozen trace without rebuilding unchanged records."""
    if debug or type(point) is not MemoryPoint:
        return replace(point, action_index=index)
    return point if index == point.action_index else MemoryPoint(index, *_MEMORY_POINT_VALUES(point))


def _bind_batch_attributes(batches, *, debug):
    """Expand native immutable GEMM groups once within this cfg binding."""
    getters = tuple(attrgetter(name) for name in
                    ('action_indices', 'results', 'output_bytes', 'packing_bytes'))
    groups = tuple(batch for batch in batches if type(batch) is DenseGemmGroup)
    if debug or not groups:
        return getters

    def bind(getter):
        values = {id(batch): getter(batch) for batch in groups}

        def value(batch):
            fixed = values.get(id(batch))
            return getter(batch) if fixed is None else fixed
        return value

    return tuple(bind(getter) for getter in getters)


def _cfg_batch_candidates(batches, reused, workspace, backend,
                          indices_of=attrgetter('action_indices'),
                          packing_of=attrgetter('packing_bytes')):
    """Keep complete groups; partial reuse retains their original child batches."""
    side = 0 if backend == 'numpy' else 1
    candidates = []
    for batch in batches:
        if isinstance(batch, DenseGemmGroup) and any(reused[i] for i in indices_of(batch)):
            start = indices_of(batch)[0]
            values = list(workspace.get(start, (0, 0)))
            values[side] = max(0, values[side] - packing_of(batch))
            workspace[start] = tuple(values)
            for child in batch.batches:
                if not all(reused[i] for i in child.action_indices):
                    start = child.action_indices[0]
                    values = list(workspace.get(start, (0, 0)))
                    values[side] += child.packing_bytes
                    workspace[start] = tuple(values)
                    candidates.append(child)
        else:
            candidates.append(batch)
    return tuple(candidates)


def _cfg_dependency_batches(batches, reused, available, bindings, workspace, backend,
                            indices_of=attrgetter('action_indices'),
                            results_of=attrgetter('results'),
                            packing_of=attrgetter('packing_bytes')):
    """Keep saved kernels only where their actual inputs are already produced.

    Partial result reuse can remove an upstream Cartesian kernel, delaying its
    other results to their original Actions. A downstream independent GEMM
    group must then retain its children at their saved starts instead of
    reading those results at the group's earlier start. This is cfg binding,
    never numerical replay or a new contraction-path search.
    """
    pending = [(indices_of(batch)[0], serial, batch)
               for serial, batch in enumerate(batches)]
    heapify(pending)
    serial = len(pending)
    side = 0 if backend == 'numpy' else 1
    while pending:
        start, _, batch = heappop(pending)
        paid = all(not reused[index] for index in indices_of(batch))
        ready = paid and all(available.get(bindings[operand.tensor], -1) <= start
            for leaf in dense_batch_leaves((batch,))
            for operands in (leaf.left, leaf.right) for operand in operands)
        if paid and not ready and isinstance(batch, DenseGemmGroup):
            values = list(workspace.get(start, (0, 0)))
            values[side] = max(0, values[side] - packing_of(batch))
            workspace[start] = tuple(values)
            for child in batch.batches:
                child_start = child.action_indices[0]
                values = list(workspace.get(child_start, (0, 0)))
                values[side] += child.packing_bytes
                workspace[child_start] = tuple(values)
                heappush(pending, (child_start, serial, child))
                serial += 1
            continue
        if ready:
            for result in results_of(batch):
                available[bindings[result.tensor]] = start
        yield batch, ready


def _read_identity(request, roles=None):
    return (request.family,
            request.metadata.data_identity or (roles or {}).get(request.reader_role, request.reader_role),
            request.arguments, request.selection, request.conventions)


def _source_bound(source):
    if source.snapshot_bytes is not None:
        return source.snapshot_bytes
    return max(source.region_bytes, source.host_region_bytes)


def _tensor_bytes(spec):
    return prod(spec.shape) * spec.dtype.itemsize


def _resource_caps(plan, backend, execution_target, target_summary):
    """Resolve declared caps only; never inspect or initialize a local backend."""
    summary = {} if target_summary is None else dict(target_summary)
    target_backend = getattr(execution_target, 'backend', None)
    if target_backend is not None and target_backend != backend:
        raise ValueError('execution target backend differs from selected plan backend')
    target_memory = getattr(execution_target, 'memory', None)
    host_caps = []
    if backend == 'numpy':
        host_caps.append(plan.memory_limit)
    if plan.host_memory_limit is not None:
        host_caps.append(plan.host_memory_limit)
    if target_memory is not None and getattr(target_memory, 'host_available_bytes', None) is not None:
        host_caps.append(target_memory.host_available_bytes)
    for key in ('host_memory_limit', 'host_available_bytes'):
        value = summary.get(key)
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError(f'target summary {key} must be nonnegative integer bytes')
            host_caps.append(value)
    host_cap = min(host_caps) if host_caps else None

    if backend == 'cupy':
        device_caps = [plan.memory_limit]
        if target_memory is not None and getattr(target_memory, 'device_free_bytes', None) is not None:
            device_caps.append(target_memory.device_free_bytes)
        for key in ('device_memory_limit', 'device_free_bytes'):
            value = summary.get(key)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError(f'target summary {key} must be nonnegative integer bytes')
                device_caps.append(value)
        device_cap = min(device_caps)
    else:
        device_cap = None
    return host_cap, device_cap


def _native_dense_computation(action):
    """Native value semantics exclude occurrence-specific replay coordinates."""
    return (type(action) is Action and type(action.equation) is AxisPattern
            and action.kind == 'contract' and action.resource is None
            and action.used_np == 0 and not action.join_id and action.sparse_spin is None
            and (action.core is None and len(action.operands) in (1, 2)
                 or type(action.core) in (FactorizedWideReduction, MesonWholeKernel)))


def _native_baryon_join_computation(action):
    """Whole core or ordered sparse join plus ordered ports define the value.

    Term and factor positions only locate the proof occurrence: the ordered
    join reads operands, axes, sparse spin and kernel path, never the Term.
    """
    return (type(action) is Action and type(action.equation) is AxisPattern
            and action.kind == 'baryon_join' and action.resource is None
            and (action.core is None
                 or type(action.core) is TensorCoreSpec and action.core.kind == 'baryon'))


def _computation_identity(action, operands):
    # Preserve the established canonical identity for ordinary contractions;
    # special operations include every declared semantic parameter.
    if (action.kind == 'contract' and action.core is None and action.resource is None and not action.wide_accumulation
            and action.used_np == 0 and not action.join_id and action.sparse_spin is None):
        return action.kind, axis_key(operands, action.equation.inputs, action.equation.output)
    if _native_dense_computation(action):
        if type(action.core) is MesonWholeKernel:
            # The complete private equation defines the numerical value;
            # external Term coordinates only locate its proof occurrence.
            return ('meson-whole', operands, action.core.identity,
                    action.kernel_path, action.wide_accumulation)
        if action.core is None:
            return 'contract-wide', axis_key(operands, action.equation.inputs, action.equation.output)
        # term_index/positions identify the proof's occurrence, not its value.
        # Keep ordered operand roles, shape, output layout, precision and tile.
        core = replace(action.core, equation=action.core.equation.normalized())
        return ('factorized-wide', operands, action.equation.normalized(), core,
                action.kernel_path, action.wide_accumulation)
    core = getattr(action.core, 'identity', action.core)
    if _native_baryon_join_computation(action):
        return (
            action.kind, operands, action.equation.normalized(), action.kernel_path, core,
            action.left_axes, action.right_axes, action.same_point,
            action.transpose_bridge, action.used_np, action.used_np_right,
            action.sparse_spin, action.resource,
            action.wide_accumulation, action.kernel_description,
        )
    return (
        action.kind, operands, action.equation, action.kernel_path, core,
        action.left_axes, action.right_axes, action.same_point,
        action.transpose_bridge, action.used_np, action.used_np_right,
        action.sparse_spin, action.join_id, action.term_index, action.positions,
        action.resource, action.wide_accumulation,
    )


def _bind_computation_identity(action, axis_ids=None):
    """Normalize fixed unary/binary axes once; bind only actual input IDs later.

    Operand order can change between coordinates, including equal input IDs.
    All three binary cases preserve axis_key's canonical equality exactly.
    """
    ordinary = (action.kind == 'contract' and action.core is None and action.resource is None and not action.wide_accumulation
                and action.used_np == 0 and not action.join_id and action.sparse_spin is None)
    count = len(action.operands)
    if _native_dense_computation(action) and type(action.core) is MesonWholeKernel:
        core_identity = action.core.identity
        return lambda operands: ('meson-whole', operands, core_identity,
                                 action.kernel_path, action.wide_accumulation)
    if _native_dense_computation(action) and type(action.core) is FactorizedWideReduction:
        equation = action.equation.normalized()
        core = replace(action.core, equation=action.core.equation.normalized())
        return lambda operands: ('factorized-wide', operands, equation, core,
                                 action.kernel_path, action.wide_accumulation)
    if not ordinary or count not in (1, 2):
        return partial(_computation_identity, action)
    forward = axis_key(tuple(range(count)), action.equation.inputs, action.equation.output)[1:]
    if count == 1:
        if axis_ids is not None:
            forward_id = axis_ids.setdefault(forward, len(axis_ids))
            return lambda operands: ('contract-axes', operands, forward_id)
        return lambda operands: ('contract', (operands, *forward))
    reverse = axis_key((1, 0), action.equation.inputs, action.equation.output)[1:]
    equal = min(forward, reverse)
    if axis_ids is not None:
        forward_id = axis_ids.setdefault(forward, len(axis_ids))
        reverse_id = axis_ids.setdefault(reverse, len(axis_ids))
        equal_id = forward_id if equal == forward else reverse_id
        def identify(operands):
            left, right = operands
            if left < right:
                return 'contract-axes', operands, forward_id
            if left > right:
                return 'contract-axes', (right, left), reverse_id
            return 'contract-axes', operands, equal_id
        return identify
    def identify(operands):
        left, right = operands
        if left < right:
            return 'contract', (operands, *forward)
        if left > right:
            return 'contract', ((right, left), *reverse)
        return 'contract', (operands, *equal)
    return identify


def _bind_operands(keys):
    """Bind fixed operand names without allocating a generator per Action."""
    if len(keys) > 1:
        return itemgetter(*keys)
    if keys:
        key = keys[0]
        return lambda binding: (binding[key],)
    return lambda binding: ()


def _bind_computation_identities(actions, *, debug, output_layouts=None):
    """Share immutable ordinary axis programs within this template only."""
    axis_ids, programs, result = {}, {}, []
    layout_programs = {}
    output_layouts = {} if output_layouts is None else output_layouts
    for action in actions:
        if action.kind == 'load':
            result.append(None)
            continue
        if (not debug and type(action) is Action and type(action.equation) is AxisPattern
                and action.kind == 'contract' and action.core is None and action.resource is None and not action.wide_accumulation
                and action.used_np == 0 and not action.join_id and action.sparse_spin is None
                and len(action.operands) in (1, 2)):
            key = len(action.operands), action.equation
            bind = programs.get(key)
            if bind is None:
                bind = _bind_computation_identity(action, axis_ids)
                programs[key] = bind
        else:
            bind = _bind_computation_identity(action, axis_ids)
        layout = output_layouts.get(action.tensor)
        if layout is not None:
            # Equal logical values can have different physical owners. A
            # prepared batch side borrows its producer's exact C layout;
            # reusing an ordinary transpose view would lose that guarantee.
            key = bind, layout
            bound = layout_programs.get(key)
            if bound is None:
                bound = layout_programs[key] = partial(
                    _producer_layout_identity, bind=bind, layout=layout)
            bind = bound
        result.append(bind)
    return tuple(result)


def _producer_layout_identity(operands, *, bind, layout):
    return 'producer-layout', bind(operands), layout


def _bind_local_computation_reuses(actions, identity_binders, operand_binders,
                                   preparation_keys, *, debug):
    """Prove unchanged local outputs once, independently of Action.reused.

    Distinct preparation names are distinct formal inputs. Canonical native
    dense computations may equate their outputs; extensions get opaque values.
    Substituting actual IDs preserves these formal equalities even when two
    formal inputs happen to share an actual ID. Loads do not reset bindings.
    """
    if debug:
        return (False,) * len(actions)
    versions = {key: index for index, key in enumerate(preparation_keys)}
    next_version = len(versions)
    computations, result = {}, []
    for local, action in enumerate(actions):
        if action.kind == 'load':
            result.append(False)
            continue
        if _native_dense_computation(action) or _native_baryon_join_computation(action):
            identity = identity_binders[local](operand_binders[local](versions))
            version = computations.get(identity)
            if version is None:
                version = next_version
                next_version += 1
                computations[identity] = version
            result.append(versions.get(action.tensor) == version)
        else:
            # Do not infer equality from special/extension hooks or names.
            version = next_version
            next_version += 1
            result.append(False)
        versions[action.tensor] = version
    return tuple(result)


def _merge_read_aliases(previous, actual, *, debug):
    """Preserve ordered reader alternatives without copying unchanged records."""
    if debug or type(previous) is not SourceReadRequest or type(actual) is not SourceReadRequest:
        aliases = []
        for alias in (*previous.aliases, *actual.aliases, replace(actual, aliases=())):
            if alias != replace(previous, aliases=()) and alias not in aliases:
                aliases.append(alias)
        return replace(previous, aliases=tuple(aliases))
    previous_base = replace(previous, aliases=()) if previous.aliases else previous
    actual_base = replace(actual, aliases=()) if actual.aliases else actual
    aliases = []
    for alias in (*previous.aliases, *actual.aliases, actual_base):
        if alias != previous_base and alias not in aliases:
            aliases.append(alias)
    merged = tuple(aliases)
    return previous if merged == previous.aliases else replace(previous, aliases=merged)


def _preparation_order(preparations, requests):
    """Resolve fixed dependencies in the original first-discovery order.

    Stack members always name prepared tensors. Other recipes prefer a raw
    request when a name exists in both domains, just as coordinate binding did.
    This pass retains cycle/missing-input checks without repeating the DFS at
    every coordinate.
    """
    ordered, done, visiting = [], set(), set()

    def visit(key):
        if key in done:
            return
        if key in visiting:
            raise ValueError('cyclic saved preparation')
        if key not in preparations:
            raise ValueError(f'complete cfg has no preparation for input {key!r}')
        visiting.add(key)
        recipe = preparations[key].recipe
        if recipe.kind in {'stack', 'linear_combination'}:
            for member in recipe.inputs:
                visit(member)
        elif recipe.kind != 'constant':
            for member in recipe.inputs:
                if member not in requests:
                    visit(member)
        visiting.remove(key)
        done.add(key)
        ordered.append(key)

    for key in preparations:
        visit(key)
    return tuple(ordered)


def _bind_preparation(prep, *, debug):
    """Rebind generated IDs without revalidating accepted fixed metadata.

    Only P calls the returned binder, with the original input/block counts
    and nonempty IDs allocated by the current dependency pass. Coordinate
    aliasing can collapse distinct blocks, so that check remains necessary.
    Extension records and debug keep their constructors and custom hooks.
    """
    recipe = prep.recipe
    if debug or type(prep) is not SourcePreparation or type(recipe) is not TensorPreparation:
        return lambda inputs, blocks, constants=None: replace(prep, blocks=blocks,
            recipe=replace(recipe, inputs=inputs,
                constants=recipe.constants if constants is None else constants,
                coordinate_constants=recipe.coordinate_constants if constants is None else ()))
    recipe_fields, prep_fields = vars(recipe).copy(), vars(prep).copy()

    def bind(inputs, blocks, constants=None):
        if len(blocks) > 1 and len(set(blocks)) != len(blocks):
            raise ValueError('duplicate source block dependency')
        bound_recipe = object.__new__(TensorPreparation)
        values = {**recipe_fields, 'inputs': inputs}
        if constants is not None:
            values.update(constants=constants, coordinate_constants=())
        object.__setattr__(bound_recipe, '__dict__', values)
        bound = object.__new__(SourcePreparation)
        object.__setattr__(bound, '__dict__', {**prep_fields, 'blocks': blocks, 'recipe': bound_recipe})
        return bound

    return bind


def _bind_read_request(request, read_metadata, *, debug):
    """Bind accepted read structure once, keeping actual endpoint checks.

    Extension records and debug retain at()/constructor hooks. The standard
    path changes only fields owned by this coordinate pass; no numerical data
    or reader is captured by these template-local callables.
    """
    aliases = tuple(_bind_read_request(alias, read_metadata, debug=debug)
                    for alias in request.aliases)
    if debug or type(request) is not SourceReadRequest:
        def bind(values):
            actual = request.at(values)
            metadata = read_metadata.get(
                (actual.reader_role, actual.arguments, actual.selection), actual.metadata)
            return replace(actual, metadata=metadata,
                           aliases=tuple(alias(values) for alias in aliases))
        return bind

    fixed = vars(request).copy()
    arguments = request.arguments
    coordinates, tables = request.coordinate_arguments, request.coordinate_tables

    def create(arguments, bound_aliases):
        metadata = read_metadata.get(
            (request.reader_role, arguments, request.selection), request.metadata)
        # An external metadata override is new input, unlike accepted fields.
        if metadata is not request.metadata and (
                not isinstance(metadata, TensorAxisMetadata) or metadata.sparse_layout != 'dense'):
            raise TypeError('dense read request requires dense TensorAxisMetadata')
        bound = object.__new__(SourceReadRequest)
        object.__setattr__(bound, '__dict__',
                           {**fixed, 'arguments': arguments, 'metadata': metadata, 'aliases': bound_aliases})
        return bound

    if not coordinates and not tables and not aliases:
        metadata = read_metadata.get(
            (request.reader_role, arguments, request.selection), request.metadata)
        bound = request if metadata is request.metadata else create(arguments, ())
        return lambda values: bound

    def bind(values):
        actual = list(arguments)
        for index, role, offset, period in coordinates:
            if role in values:
                value = int(values[role]) + offset
                actual[index] = value if period is None else value % period
        for index, role, table in tables:
            if role in values:
                slot = values[role]
                if (isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
                        or not 0 <= int(slot) < len(table)):
                    raise ValueError('coordinate slot is outside saved lookup table')
                mapped = table[int(slot)]
                if mapped is None:
                    raise KeyError('physical endpoint is missing from saved time selection')
                actual[index] = mapped
        return create(tuple(actual), tuple(alias(values) for alias in aliases))

    return bind


# Bound the working set of canonical identity keys while binding coordinates.
# This is a planner metadata limit, independent of target host/device budgets.
_CFG_BINDING_WINDOW_SLOTS = 4_000_000


class _CfgResourceExceeded(MemoryError):
    pass


def iter_cfg_plan_windows(plan, output, points, requests, preparations, specs, *,
                          window_size=None, output_mode='dataset',
                          prefetch_host_bytes=None, read_metadata=None,
                          reader_aliases=(), execution_target=None, target_summary=None,
                          debug=False, prefetch_device_bytes=None):
    """Yield independently feasible windows without binding future coordinates.

    IDs and array ownership are local to each delivered CfgPlan. The caller
    keeps output accumulation across windows and publishes only after the
    entire coordinate iterator succeeds. This iterator performs no numerical
    IO, backend initialization or concurrency of its own.
    """
    limit = max(1, _CFG_BINDING_WINDOW_SLOTS // max(1, len(plan.state.actions)))
    if window_size is not None:
        if type(window_size) is not int or window_size < 1:
            raise ValueError('window_size must be a positive integer or None')
        limit = min(limit, window_size)
    iterator = iter(points)
    offset = index = 0
    binding = None
    while True:
        window = tuple(islice(iterator, limit))
        if not window:
            if not index:
                raise ValueError('cfg plan requires at least one coordinate')
            return
        started = perf_counter()
        first = binding is None
        if first:
            binding = _bind_cfg_template(plan, output, requests, preparations, specs,
                output_mode=output_mode, prefetch_host_bytes=prefetch_host_bytes,
                read_metadata=read_metadata, reader_aliases=reader_aliases,
                execution_target=execution_target, target_summary=target_summary,
                debug=debug, prefetch_device_bytes=prefetch_device_bytes)
        binding_seconds = perf_counter() - started if first else 0.0
        result = _build_cfg_plan(binding, window, len(plan.state.actions), started,
            template_binding_count=int(first), template_binding_seconds=binding_seconds)
        yield replace(result, statistics=dict(result.statistics,
            window_index=index, coordinate_offset=offset))
        offset += len(window)
        index += 1


def build_cfg_plan(plan, output, points, requests, preparations, specs, *,
                   output_mode='memory', prefetch_host_bytes=None, read_metadata=None,
                   reader_aliases=(), execution_target=None, target_summary=None,
                   debug=False, prefetch_device_bytes=None):
    """Bind the selected path with bounded, target-budgeted coordinate reuse.

    Windows delimit residency, not scientific coverage or the continuous IO
    queue. A failed envelope reduces reuse before saving; execution never
    changes the delivered arrangement or retries on another device.
    """
    points = tuple(points)
    if not points:
        raise ValueError('cfg plan requires at least one coordinate')
    started = perf_counter()
    binding = _bind_cfg_template(plan, output, requests, preparations, specs,
        output_mode=output_mode, prefetch_host_bytes=prefetch_host_bytes,
        read_metadata=read_metadata, reader_aliases=reader_aliases,
        execution_target=execution_target, target_summary=target_summary,
        debug=debug, prefetch_device_bytes=prefetch_device_bytes)
    return _build_cfg_plan(binding, points, len(plan.state.actions), started,
        template_binding_count=1, template_binding_seconds=perf_counter() - started)


def _build_cfg_plan(binding, points, action_count, started, *,
                    template_binding_count, template_binding_seconds):
    """Retry only coordinate residency; accepted fixed metadata stays bound."""
    window = min(len(points), max(1, _CFG_BINDING_WINDOW_SLOTS // max(1, action_count)))
    attempts = 0
    while True:
        attempts += 1
        try:
            result = _bind_cfg_residency_windows(binding, points, window)
        except _CfgResourceExceeded:
            if window == 1:
                raise
            window = max(1, window // 2)
        else:
            return replace(result, statistics=dict(result.statistics,
                reuse_window=window, residency_windows=(len(points)+window-1)//window,
                resource_attempts=attempts, lifecycle_seconds=perf_counter()-started,
                template_binding_count=template_binding_count,
                template_binding_seconds=template_binding_seconds))


def _bind_cfg_residency_windows(binding, points, window):
    """Finalize each saved residency interval before binding the next one.

    Compute ownership/size/workspace tables need not span the entire cfg.
    Only the delivered CfgPlan records accumulate; all coordinates and their
    already selected events still belong to one complete, archivable plan.
    """
    if len(points) <= window:
        return binding(points, reuse_window=window)
    instances, requests, blocks, preparations, specs = [], {}, {}, {}, {}
    sums = ('instances', 'read_blocks', 'compute_nodes', 'prepared_nodes',
            'compute_seconds', 'preparation_seconds', 'batch_owner_bytes',
            'binding_seconds', 'resource_seconds', 'binding_bytes', 'binding_node_count')
    maxima = ('peak_host_bytes', 'peak_device_bytes', 'peak_owned_tensor_bytes',
              'peak_numeric_tensor_host_bytes', 'peak_numeric_tensor_device_bytes',
              'peak_raw_input_bytes', 'peak_raw_host_bytes', 'peak_raw_device_bytes',
              'peak_operation_workspace_bytes', 'peak_preparation_workspace_host_bytes',
              'peak_preparation_workspace_device_bytes')
    stats = None
    first = None
    node_offset = action_offset = 0
    host_base = device_base = 0
    preparation_costs = array('d')
    for start in range(0, len(points), window):
        part = binding(points[start:start+window], reuse_window=window,
                       node_offset=node_offset, output_instances=len(points),
                       preparation_costs=preparation_costs)
        if first is None:
            first = part
            stats = dict(part.statistics)
        else:
            for key in sums:
                stats[key] += part.statistics[key]
            for key in maxima:
                stats[key] = max(stats[key], part.statistics[key])
            for key in ('prefetch_host_bytes', 'prefetch_device_bytes'):
                stats[key] = min(stats[key], part.statistics[key])
            if 'audit_seconds' in stats:
                stats['audit_seconds'] += part.statistics['audit_seconds']
        instances.extend(replace(instance, action_offset=instance.action_offset+action_offset)
                         if action_offset else instance for instance in part.instances)
        requests.update(part.requests)
        blocks.update(part.blocks)
        preparations.update(part.preparations)
        specs.update(part.specs)
        host_base = max(host_base, part.peak_host_bytes-part.statistics['constant_host_bytes'])
        device_base = max(device_base, part.peak_device_bytes
            -part.statistics['constant_device_bytes']-part.statistics['packed_value_cache_device_bytes'])
        node_offset += part.statistics['binding_node_count']
        action_offset += part.instances[-1].action_offset + len(
            part.templates[part.instances[-1].template].plan.state.actions) + 1
        # Do not retain the last window's temporary CfgPlan dictionaries while
        # constructing its successor. The final records above retain sharing.
        part = None
    # Preserve the original first-load order and Python's floating-point sum,
    # without retaining preparation records solely for a whole-cfg cost pass.
    stats['preparation_seconds'] = sum(preparation_costs)
    constant_payload = cfg_rounding.constant_bytes(preparations)
    stats['constant_host_bytes'] = constant_payload if first.backend == 'numpy' else 0
    stats['constant_device_bytes'] = (cfg_rounding.constant_bytes(preparations, device=True)
                                      if first.backend == 'cupy' else 0)
    stats['constant_device_payload_bytes'] = constant_payload if first.backend == 'cupy' else 0
    stats['peak_host_bytes'] = host_base + stats['constant_host_bytes']
    stats['peak_device_bytes'] = device_base + stats['constant_device_bytes']
    host_cap, device_cap = stats['host_memory_cap'], stats['device_memory_cap']
    for side, cap in (('host', host_cap), ('device', device_cap)):
        if cap is not None and stats[f'peak_{side}_bytes'] > cap:
            raise _CfgResourceExceeded(f'complete cfg resource envelope requires '
                f'{stats[f"peak_{side}_bytes"]} {side} bytes; cap is {cap}')
    cache = min(first.templates[0].plan.packed_value_cache_device_bytes,
                0 if device_cap is None else device_cap-stats['peak_device_bytes'])
    stats['packed_value_cache_device_bytes'] = cache
    stats['peak_device_bytes'] += cache
    for side, cap in (('host', host_cap), ('device', device_cap)):
        spare = 0 if cap is None else cap-stats[f'peak_{side}_bytes']
        stats[f'prefetch_{side}_bytes'] = min(spare, stats[f'prefetch_{side}_bytes'])
    result = replace(first, instances=tuple(instances), requests=requests, blocks=blocks,
        preparations=preparations, specs=specs,
        peak_host_bytes=stats['peak_host_bytes'], peak_device_bytes=stats['peak_device_bytes'],
        prefetch_host_bytes=stats['prefetch_host_bytes'],
        prefetch_device_bytes=stats['prefetch_device_bytes'], statistics=stats)
    from .cfg_preparation_carry import retain_cfg_preparations
    result = retain_cfg_preparations(result, preparation_costs)
    if 'audit_seconds' in stats:
        audit_start = perf_counter()
        audit_cfg_plan(result)
        result = replace(result, statistics=dict(result.statistics,
            audit_seconds=stats['audit_seconds']+perf_counter()-audit_start))
    return result


def _bind_cfg_template(plan, output, requests, preparations, specs, *,
                   output_mode='memory', prefetch_host_bytes=None, read_metadata=None,
                   reader_aliases=(), execution_target=None, target_summary=None,
                   debug=False, prefetch_device_bytes=None):
    """Bind the fixed part of one selected Action/SceneStoragePlan.

    The selected path and any verified dense batches are retained. Source
    callbacks are excluded: each input must have a complete pure preparation
    recipe and an explicit saved reader request for every physical block.
    """
    plan, _ = prepare_coordinate_batch_workspace(plan)
    plan = replace(plan, blocks=dict(plan.blocks), preparations=dict(plan.preparations),
                   preparation_costs=dict(plan.preparation_costs))
    requests, specs = dict(requests), dict(specs)
    reader_aliases = tuple(tuple(group) for group in reader_aliases)
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    if output_mode not in {'memory', 'dataset'}:
        raise ValueError('cfg output mode must be memory or dataset')
    backend = plan.state.problem.backend
    if backend not in {'numpy', 'cupy'}:
        raise ValueError('unsupported complete cfg backend')
    # E supplies the complete dependency table, while P's selected entries
    # carry placement on the execution target (including preparation workspace).
    preparations = dict(preparations)
    preparations.update({key: replace(prep, recipe=preparations[key].recipe)
        if prep.recipe is None and key in preparations else prep
        for key, prep in plan.preparations.items()})
    if backend == 'cupy':
        preparations = {key: replace(prep, workspace_host_bytes=0,
            workspace_device_bytes=prep.workspace_host_bytes + prep.workspace_device_bytes)
            if prep.recipe is not None else prep for key, prep in preparations.items()}
    if target_summary is not None and not hasattr(target_summary, 'items'):
        raise TypeError('target_summary must be a mapping or None')
    if target_summary is not None:
        target_summary = dict(target_summary)
    host_cap, device_cap = _resource_caps(plan, backend, execution_target, target_summary)
    if backend == 'numpy' and plan.reserved_device_bytes:
        raise ValueError('NumPy cfg plans cannot reserve device memory')

    # Recipe-less inputs are still usable by ordinary execution, but cannot be
    # made self-contained in a cfg archive.
    problem_sources = plan.state.problem.sources
    for key, prep in preparations.items():
        if prep.recipe is None:
            raise ValueError(f'complete cfg input {key!r} has no described preparation recipe')
        if prep.recipe.kind not in {'identity', 'spin_spatial', 'stack', 'constant', 'linear_combination'}:
            raise ValueError(f'complete cfg input {key!r} has an unsupported preparation recipe')
        if key not in specs and key not in plan.state.catalog and prep.recipe.kind != 'identity':
            raise ValueError(f'complete cfg input {key!r} needs a TensorSpec for its preparation recipe')
        if key not in specs and key not in plan.state.catalog and key not in problem_sources:
            raise ValueError(f'complete cfg input {key!r} has neither TensorSpec nor SourceSpec')

    read_metadata = {} if read_metadata is None else dict(read_metadata)
    read_bindings = []
    for key, request in requests.items():
        if key not in plan.blocks:
            raise ValueError(f'saved reader request {key!r} has no StorageBlock')
        read_bindings.append((key, _bind_read_request(request, read_metadata, debug=debug)))
    roles = {role: group[0] for group in reader_aliases for role in group}
    selected = tuple((i, action) for i, action in enumerate(plan.state.actions)
                     if action.kind != 'release')
    actions = tuple(action for _, action in selected)
    if any(action.kind not in {'load', 'contract', 'sparse_bridge', 'baryon_join'}
           for action in actions):
        raise ValueError('complete cfg contains an unsupported Action resource domain')
    positions = {old: new for new, (old, _) in enumerate(selected)}
    output_layouts = {
        side[0].tensor: side[0].permutation
        for batch in dense_batch_leaves(plan.kernel_batches)
        for side, prepared in zip((batch.left, batch.right), batch.prepared_sides)
        if prepared
    }
    identity_binders = _bind_computation_identities(
        actions, debug=debug, output_layouts=output_layouts)
    operand_binders = tuple(_bind_operands(action.operands) for action in actions)
    defined = set()
    output_last_uses, size_last_writes = {}, {}
    redefinitions = []
    for local, action in enumerate(actions):
        if action.kind == 'load' and action.tensor not in preparations:
            raise ValueError(f'load Action {action.tensor!r} has no preparation')
        if action.kind != 'load' and action.tensor not in plan.state.catalog:
            raise ValueError(f'Action output {action.tensor!r} has no TensorSpec')
        if action.kind != 'load' and any(key not in defined for key in action.operands):
            raise ValueError('cfg compute consumer precedes its producer')
        if not debug:
            output_last_uses[action.tensor] = local
            if action.kind != 'load':
                size_last_writes[action.tensor] = local
                if action.tensor in defined:
                    redefinitions.append(local)
        defined.add(action.tensor)
    local_reuses = _bind_local_computation_reuses(actions, identity_binders,
        operand_binders, preparations, debug=debug)
    fine_operand_uses = debug or any(not local_reuses[index] for index in redefinitions)
    binding_indices = (range(len(actions)) if fine_operand_uses else
                       tuple(index for index, reused in enumerate(local_reuses) if not reused))
    action_bytes = tuple(0 if action.kind == 'load' else _tensor_bytes(plan.state.catalog[action.tensor])
                         for action in actions)
    # Different names can bind one actual ID. Apply the last size writes in
    # their original order, including writes from locally repeated Actions.
    size_writes = tuple((key, action_bytes[local]) for key, local in
                        sorted(size_last_writes.items(), key=itemgetter(1)))
    # Requests always bind the same keys; only their physical values vary.
    # Check this frozen recipe structure once, not for every time pair.
    for key, prep in preparations.items():
        for raw in prep.blocks:
            if raw not in requests:
                raise ValueError(f'complete cfg block {raw!r} has no saved reader request')
        if prep.recipe.kind == 'identity' and len(prep.recipe.inputs) == 1:
            if prep.recipe.inputs[0] not in requests and prep.recipe.inputs[0] not in preparations:
                raise ValueError(f'complete cfg input {key!r} has an undescribed recipe input')
        elif prep.recipe.kind in {'stack', 'linear_combination'}:
            if any(member not in preparations for member in prep.recipe.inputs):
                raise ValueError(f'complete cfg stack {key!r} has an undescribed member')
        elif prep.recipe.kind == 'spin_spatial':
            if any(raw not in requests for raw in prep.recipe.inputs):
                raise ValueError(f'complete cfg input {key!r} has an undescribed raw input')
    remapped_batches = []
    for batch in plan.kernel_batches:
        try:
            if isinstance(batch, DenseGemmGroup):
                remapped = replace(batch, batches=tuple(replace(child,
                    action_indices=tuple(positions[index] for index in child.action_indices))
                    for child in batch.batches))
            else:
                remapped = replace(batch, action_indices=tuple(positions[index] for index in batch.action_indices))
        except KeyError as exc:
            raise ValueError('dense batch references a removed Action') from exc
        remapped_batches.append(remapped)
    # The terminal point describes output ownership after all Actions. It is
    # not an Action, but must survive removal of release Actions: its peak can
    # exceed every compute point (for example, rounded output snapshots).
    terminal_index = len(plan.state.actions)
    template_plan = replace(
        plan,
        state=replace(plan.state, actions=actions),
        memory=tuple(_remap_memory_point(point,
                         len(actions) if point.action_index == terminal_index else positions[point.action_index],
                         debug=debug)
                     for point in plan.memory
                     if point.action_index in positions or point.action_index == terminal_index),
        preparation_costs={positions[index]: cost for index, cost in plan.preparation_costs.items()
                           if index in positions},
        kernel_batches=tuple(remapped_batches),
        scalar_reduction_groups=tuple(replace(group,
            action_indices=tuple(positions[index] for index in group.action_indices))
            for group in plan.scalar_reduction_groups),
        cartesian_reduction_groups=tuple(replace(group,
            batch_action_indices=tuple(tuple(positions[index] for index in child)
                                       for child in group.batch_action_indices))
            for group in plan.cartesian_reduction_groups),
    )
    keys = tuple(dict.fromkeys((*specs, *plan.state.catalog, *preparations)))
    preparation_specs = {}
    for key in preparations:
        spec = specs.get(key, plan.state.catalog.get(key))
        if spec is not None:
            preparation_specs[key] = spec
    template = CfgTemplate(template_plan, keys, output, preparations, preparation_specs)

    def source_size(key):
        spec = specs.get(key, plan.state.catalog.get(key))
        if spec is not None:
            return _tensor_bytes(spec)
        source = problem_sources.get(key)
        if source is None:
            raise ValueError(f'complete cfg input {key!r} has no source size description')
        return _source_bound(source)

    borrowed_sources = set()
    definitions, definition_ids, source_bytes = {}, {}, {}
    preparation_binders, definition_binders = {}, {}
    dynamic_workspace_definitions = set()
    for key, prep in preparations.items():
        recipe = prep.recipe
        spec = specs.get(key, plan.state.catalog.get(key))
        source = problem_sources.get(key)
        spec_identity = ((spec.shape, spec.dtype.str) if spec is not None else
                         (None, source.snapshot_bytes, source.region_bytes,
                          source.host_region_bytes, repr(source.metadata)) if source is not None else None)
        identity = (
            tuple(getattr(prep, field.name) for field in fields(prep)
                  if field.name not in {'blocks', 'recipe'}),
            tuple(getattr(recipe, field.name) for field in fields(recipe) if field.name != 'inputs'),
            spec_identity,
        )
        definition_ids[key] = definitions.setdefault(identity, len(definitions))
        if debug or type(prep) is not SourcePreparation or type(recipe) is not TensorPreparation:
            dynamic_workspace_definitions.add(definition_ids[key])
        binding_key = definition_ids[key], type(prep), type(recipe)
        if binding_key not in definition_binders:
            definition_binders[binding_key] = _bind_preparation(prep, debug=debug)
        preparation_binders[key] = definition_binders[binding_key]
        source_bytes[key] = source_size(key)
        if spec is None:
            # An identity recipe over an untyped sparse record borrows the raw
            # block. Its SourceSpec bound is applied to that raw owner below.
            if recipe.kind != 'identity':
                raise ValueError(f'complete cfg input {key!r} has no TensorSpec')
            borrowed_sources.add(key)

    source_bindings = []
    for key in _preparation_order(preparations, requests):
        prep = preparations[key]
        recipe = prep.recipe
        # Accepted identity/spin recipes have exactly one input. Constants
        # have none, and stacks use only prepared members.
        raw_inputs = recipe.kind not in {'stack', 'linear_combination', 'constant'} and recipe.inputs[0] in requests
        source_bindings.append((key, definition_ids[key], raw_inputs,
            _bind_operands(recipe.inputs), _bind_operands(prep.blocks),
            preparation_binders[key], specs.get(key, plan.state.catalog.get(key)),
            0 if key in borrowed_sources else source_bytes[key], key in borrowed_sources,
            recipe.constants_at if recipe.coordinate_constants else None))

    # The fixed call contains descriptors only. Every call below starts with
    # empty ID/lifetime maps, so no previous window can keep numeric ownership.
    preparation_rounding = cfg_rounding.preparation_padding(preparations) if backend == 'cupy' else {}
    memory_h, memory_d, group_memory = {}, {}, {}
    for point in plan.memory:
        memory_h[point.action_index] = max(memory_h.get(point.action_index, 0), point.workspace_host_bytes)
        memory_d[point.action_index] = max(memory_d.get(point.action_index, 0), point.workspace_device_bytes)
        extra = point.scalar_reduction_device_bytes + point.cartesian_reduction_device_bytes
        if extra:
            group_memory[point.action_index] = max(group_memory.get(point.action_index, 0), extra)
    action_workspaces = []
    for original_index, action in selected:
        workspace_h = memory_h.get(original_index, 0)
        workspace_d = memory_d.get(original_index, 0)
        dynamic_preparation = False
        if action.kind == 'load':
            prep = preparations[action.tensor]
            # An equal definition may be shared with an extension that owns
            # the actual bound record, even when this Action names a standard
            # preparation. Preserve that record's constructor/resource hooks.
            dynamic_preparation = definition_ids[action.tensor] in dynamic_workspace_definitions
            if not dynamic_preparation:
                workspace_h = max(workspace_h, prep.workspace_host_bytes)
                workspace_d = max(workspace_d, prep.workspace_device_bytes)
        if backend == 'cupy' and not dynamic_preparation:
            workspace_d += (max(cfg_rounding.padding(workspace_d), preparation_rounding.get(action.tensor, 0)) if action.kind == 'load' else
                            cfg_rounding.action_workspace_padding(action, plan.state.catalog, workspace_d))
        action_workspaces.append((workspace_h, workspace_d, dynamic_preparation,
                                  preparation_rounding.get(action.tensor, 0)))
    return partial(_bind_cfg_coordinates, plan=plan, template=template,
        read_bindings=tuple(read_bindings), preparations=preparations,
        reader_aliases=reader_aliases, roles=roles, actions=actions,
        identity_binders=identity_binders, operand_binders=operand_binders,
        local_reuses=local_reuses,
        fine_operand_uses=fine_operand_uses,
        binding_indices=binding_indices, output_last_uses=tuple(output_last_uses.items()),
        size_writes=size_writes,
        action_bytes=action_bytes, remapped_batches=tuple(remapped_batches),
        batch_attributes=_bind_batch_attributes(remapped_batches, debug=debug),
        source_bindings=tuple(source_bindings), source_bytes=source_bytes,
        borrowed_sources=frozenset(borrowed_sources), action_workspaces=tuple(action_workspaces),
        physical_group_workspaces=tuple((local, group_memory[original])
            for local, (original, _) in enumerate(selected) if original in group_memory),
        backend=backend, host_cap=host_cap, device_cap=device_cap,
        output_mode=output_mode, prefetch_host_bytes=prefetch_host_bytes,
        prefetch_device_bytes=prefetch_device_bytes, execution_target=execution_target,
        target_summary=target_summary, debug=debug)


def _bind_cfg_coordinates(points, *, plan, template, read_bindings, preparations,
                          reader_aliases, roles, actions,
                          identity_binders, operand_binders, local_reuses, fine_operand_uses,
                          binding_indices, output_last_uses, size_writes,
                          action_bytes, remapped_batches, batch_attributes,
                          source_bindings, source_bytes, borrowed_sources, action_workspaces,
                          physical_group_workspaces,
                          backend, host_cap, device_cap, output_mode, prefetch_host_bytes,
                          prefetch_device_bytes, execution_target, target_summary, debug,
                          reuse_window, node_offset=0, output_instances=None,
                          preparation_costs=None):
    """Create independent actual IDs, lifetimes and resource envelopes."""
    start = perf_counter()
    keys = template.keys
    indices_of, results_of, output_of, packing_of = batch_attributes
    allocation_bytes = device_allocation_bytes if backend == 'cupy' else int
    snapshot_device = cfg_rounding.snapshot_device_bytes(plan.state) if backend == 'cupy' else 0
    packing_padding = ({id(batch): cfg_rounding.batch_packing_padding(batch, plan.state.catalog)
                        for batch in remapped_batches} if backend == 'cupy' else {})
    problem_sources = plan.state.problem.sources
    reads, raw_blocks, global_preps, global_specs = {}, {}, {}, {}
    raw_ids, prep_ids, compute_ids = {}, {}, {}
    instances, workspace_records = [], []
    batch_owners = {}
    last, raw_first, raw_last = {}, {}, {}
    produced_preparations = []
    sizes = {}
    next_node = node_offset
    compute_count = 0
    offset = 0

    def allocate():
        nonlocal next_node
        value = next_node
        next_node += 1
        return value

    def bind_reads(point):
        values = {str(role): value for role, value in point.values}
        bound = {}
        for key, bind_read in read_bindings:
            actual = bind_read(values)
            if any(_read_identity(alias, roles) != _read_identity(actual, roles)
                   for alias in actual.aliases):
                raise ValueError('coordinate changes the template input sharing relation')
            identity = _read_identity(actual, roles)
            if identity not in raw_ids:
                raw_id = allocate()
                raw_ids[identity] = raw_id
                reads[str(raw_id)] = actual
                raw_blocks[str(raw_id)] = replace(
                    plan.blocks[key],
                    dataset=actual.metadata.data_identity or plan.blocks[key].dataset,
                    selection=actual.metadata.selection_identity,
                )
            else:
                raw_id = raw_ids[identity]
                previous = reads[str(raw_id)]
                reads[str(raw_id)] = _merge_read_aliases(previous, actual, debug=debug)
            bound[key] = str(raw_id)
        return bound

    effective_blocks = dict(raw_blocks)
    borrowed_gids = set()
    for point_index, point in enumerate(points):
        if reuse_window is not None and point_index % reuse_window == 0:
            # Earlier windows retain their saved IDs/events, but cannot pin
            # arrays for a later use. Repeated physics gets a new lifetime.
            raw_ids.clear()
            prep_ids.clear()
            compute_ids.clear()
        raw_map = bind_reads(point)
        for raw_key in raw_map.values():
            effective_blocks.setdefault(raw_key, raw_blocks[raw_key])
        for key in borrowed_sources:
            prep = preparations[key]
            source = problem_sources[key]
            upper = source_bytes[key]
            for block_key in prep.blocks:
                raw_key = raw_map[block_key]
                block = effective_blocks[raw_key]
                if backend == 'numpy' or block.read_kind == 'p2p':
                    effective_blocks[raw_key] = replace(
                        block, host_bytes=max(block.host_bytes, source.host_region_bytes, upper))
                else:
                    effective_blocks[raw_key] = replace(
                        block, device_bytes=max(block.device_bytes, source.region_bytes, upper))
        node_map, source_names = {}, {}
        for (key, definition_id, raw_inputs, input_ids, block_ids,
             bind_preparation, spec, amount, borrowed, bind_constants) in source_bindings:
            recipe_inputs = input_ids(raw_map if raw_inputs else source_names)
            blocks = block_ids(raw_map)
            identity = definition_id, recipe_inputs, blocks
            constants = None
            if bind_constants is not None:
                constants = bind_constants(reads[recipe_inputs[0]].arguments)
                identity += (constants,)
            gid = prep_ids.get(identity)
            if gid is None:
                gid = allocate()
                prep_ids[identity] = gid
                name = str(gid)
                global_preps[name] = bind_preparation(recipe_inputs, blocks, constants)
                if spec is not None:
                    global_specs[name] = spec
            else:
                name = str(gid)
            node_map[key] = gid
            source_names[key] = name
            sizes[gid] = amount
            if borrowed:
                borrowed_gids.add(gid)

        reused, instance_ws, available = list(local_reuses), {}, {}
        for local in range(len(actions)) if fine_operand_uses else binding_indices:
            action = actions[local]
            location = offset + local
            if action.kind == 'load':
                gid = node_map[action.tensor]
            else:
                if fine_operand_uses or not local_reuses[local]:
                    try:
                        operands = operand_binders[local](node_map)
                    except KeyError as exc:
                        raise ValueError(f'Action {action.kind!r} consumes an undescribed value') from exc
                if local_reuses[local]:
                    gid = node_map[action.tensor]
                else:
                    identity = identity_binders[local](operands)
                    if _native_baryon_join_computation(action):
                        # P selected separate physical owners for these names.
                        # Equal actual input aliases can prove equal values,
                        # but merging their disjoint lifetimes can exceed P's
                        # accepted envelope. Retain its ownership partition.
                        identity = identity, action.tensor
                    gid = compute_ids.get(identity)
                    if gid is None:
                        gid = allocate()
                        compute_ids[identity] = gid
                        compute_count += 1
                    node_map[action.tensor] = gid
                if fine_operand_uses:
                    sizes[gid] = action_bytes[local]
            # Accepted template dependencies precede their consumers. Every
            # input has an earlier producer/reference in this coordinate.
            is_reuse = gid in last
            reused[local] = is_reuse
            if not is_reuse:
                available[gid] = local
                if action.kind != 'load':
                    # A reused occurrence reads only its existing result.
                    # Its unexecuted operands must not pin upstream owners.
                    for operand in operands:
                        last[operand] = location
                    if action.kind in {'sparse_bridge', 'baryon_join'}:
                        for operand_key in action.operands:
                            if operand_key in borrowed_sources:
                                raw_gid = node_map[operand_key]
                                for raw in global_preps[str(raw_gid)].blocks:
                                    raw_last[raw] = location
                workspace_h, workspace_d, dynamic_preparation, recipe_padding = action_workspaces[local]
                if action.kind == 'load':
                    prep = global_preps[str(gid)]
                    produced_preparations.append(prep)
                    if dynamic_preparation:
                        workspace_h = max(workspace_h, prep.workspace_host_bytes)
                        workspace_d = max(workspace_d, prep.workspace_device_bytes)
                        if backend == 'cupy':
                            workspace_d += max(cfg_rounding.padding(workspace_d), recipe_padding)
                    for raw in prep.blocks:
                        raw_first.setdefault(raw, location)
                        raw_last[raw] = location
                if workspace_h or workspace_d:
                    instance_ws[local] = (workspace_h, workspace_d)
            last[gid] = location

        if not fine_operand_uses:
            for key, amount in size_writes:
                sizes[node_map[key]] = amount
            # Each name denotes one unchanged value throughout this template.
            # Actual IDs may alias: merge their last uses with max, preserving
            # absolute locations and release ordering from the original loop.
            for key, local in output_last_uses:
                gid = node_map[key]
                location = offset + local
                if location > last[gid]:
                    last[gid] = location

        # Select only batches whose complete Cartesian result set is newly
        # produced in this instance. Partial reuse falls back to the existing
        # single-Action path for that batch only.
        instance_batches = []
        selected_batch_starts = {}
        candidates = _cfg_batch_candidates(remapped_batches, reused, instance_ws, backend,
                                          indices_of, packing_of)
        for batch, ready in _cfg_dependency_batches(candidates, reused, available,
                node_map, instance_ws, backend, indices_of, results_of, packing_of):
            member_ids = indices_of(batch)
            if ready:
                instance_batches.append(batch)
                if backend == 'cupy':
                    extra = packing_padding.get(id(batch))
                    if extra is None:
                        extra = packing_padding[id(batch)] = cfg_rounding.batch_packing_padding(batch, plan.state.catalog)
                    wh, wd = instance_ws.get(member_ids[0], (0, 0))
                    instance_ws[member_ids[0]] = wh, wd + extra
                start_local = member_ids[0]
                start_absolute = offset + start_local
                result_ids = tuple(node_map[result.tensor] for result in results_of(batch))
                if len(set(result_ids)) != len(result_ids):
                    raise ValueError('dense batch results share a global ID')
                # Stable names give newly selected owners disjoint global IDs.
                # Their common last use is consumed only after all coordinates.
                # Keep the original propagation for debug/opaque rewrites.
                if fine_operand_uses:
                    owner_last = max(last.get(gid, start_absolute) for gid in result_ids)
                    for gid in result_ids:
                        last[gid] = owner_last
                selected_batch_starts[start_absolute] = (batch, result_ids)
            elif any(not reused[index] for index in member_ids):
                # Partial reuse or delayed inputs retain the saved individual
                # Actions, so this batch's packing workspace is absent.
                batch_start = member_ids[0]
                workspace_h, workspace_d = instance_ws.get(batch_start, (0, 0))
                if backend == 'numpy':
                    workspace_h = max(0, workspace_h - packing_of(batch))
                else:
                    workspace_d = max(0, workspace_d - packing_of(batch))
                instance_ws[batch_start] = (workspace_h, workspace_d)

        # Early group outputs remain owned across intervening logical Actions,
        # including reused ones. Keep the saved envelope independently of the
        # per-Action scratch which reuse legitimately omits. Filtering an
        # incomplete group may conservatively leave this allowance unused.
        for local, amount in physical_group_workspaces:
            wh, wd = instance_ws.get(local, (0, 0))
            instance_ws[local] = wh, wd + amount

        output_location = offset + len(actions)
        for key in plan.state.results:
            if key not in node_map:
                raise ValueError(f'complete cfg output {key!r} has no global binding')
            last[node_map[key]] = output_location
        workspace_records.append(instance_ws)
        instances.append(CfgInstance(
            0,
            tuple((str(key), int(value)) for key, value in point.values),
            ArrayConstant.from_array(np.array([node_map.get(key, -1) for key in keys], dtype='<i8')),
            ArrayConstant.from_array(np.array(reused, dtype=bool)),
            (), (), offset, tuple(instance_batches),
        ))
        # Preserve batch owner facts for the one final accounting pass.
        batch_owners.update(selected_batch_starts)
        offset = output_location + 1

    # Shared result IDs may be reused in a later coordinate. Keep every view's
    # common owner through the furthest such use.
    for absolute, (batch, result_ids) in batch_owners.items():
        owner_last = max(last.get(gid, absolute) for gid in result_ids)
        for gid in result_ids:
            last[gid] = owner_last

    binding_seconds = perf_counter() - start
    events, releases = {}, {}
    for raw, location in raw_first.items():
        events.setdefault(location, []).append(StorageEvent(location, 'read', raw))
        events.setdefault(raw_last[raw], []).append(StorageEvent(raw_last[raw], 'release', raw))
    for gid, location in last.items():
        releases.setdefault(location, []).append(gid)

    output_size = plan.reserved_host_bytes
    first_binding = instances[0].node_ids.array()
    key_positions = {key: index for index, key in enumerate(keys)}
    output_ids = {int(first_binding[key_positions[key]]) for key in plan.state.results}
    output_size = max(output_size, sum(sizes[gid] for gid in output_ids))
    output_reserve = output_size * ((len(instances) if output_instances is None else output_instances)
                                   if output_mode == 'memory' else 1)

    constant_payload = cfg_rounding.constant_bytes(global_preps)
    constant_host = constant_payload if backend == 'numpy' else 0
    constant_device = cfg_rounding.constant_bytes(global_preps, device=True) if backend == 'cupy' else 0
    # The binding pass proves producer order while creating global IDs. Its
    # final-use table already fixes every lifetime. Account those intervals
    # directly; independent state replay belongs to audit_cfg_plan(debug=True).
    # Batch views charge their allocation once, until the common owner dies.
    release_bytes = np.zeros(next_node-node_offset, dtype=np.int64)
    for gid, amount in sizes.items():
        release_bytes[gid-node_offset] = allocation_bytes(amount)
    batch_results = np.zeros(next_node-node_offset, dtype=bool)
    for batch, result_ids in batch_owners.values():
        batch_results[np.asarray(result_ids)-node_offset] = True
        for gid in result_ids:
            release_bytes[gid-node_offset] = 0
        release_bytes[result_ids[0]-node_offset] = allocation_bytes(output_of(batch))
    instance_events = defaultdict(list)
    instance_releases = defaultdict(list)
    action_releases = defaultdict(list)
    span = len(actions) + 1
    for location, group in events.items():
        index, local = divmod(location, span)
        instance_events[index].extend(replace(event, action_index=local) for event in group)
    for location, gids in releases.items():
        index, local = divmod(location, span)
        if local == len(actions):
            instance_releases[index].extend(gids)
        else:
            action_releases[index].extend((local, gid) for gid in gids)
    resident = raw_host = raw_device = 0
    peak_host = peak_device = 0
    max_tensor_host = max_tensor_device = max_raw_host = max_raw_device = 0
    max_workspace_host = max_workspace_device = 0
    action_positions = np.array([key_positions[action.tensor] for action in actions], dtype=np.intp)
    action_seconds = np.array([action.seconds for action in actions])
    group_workspace = dict(physical_group_workspaces)
    compute_seconds = 0.0
    for index, instance in enumerate(instances):
        local_events = tuple(sorted(instance_events[index], key=lambda event:
                             (event.action_index, event.kind != 'read')))
        local_releases = tuple(instance_releases[index])
        gids = instance.node_ids.array()[action_positions]-node_offset
        reused = instance.reused.array()
        allocations = np.append(release_bytes[gids], np.int64(0))
        allocations[:-1][reused | batch_results[gids]] = 0
        single_cost = action_seconds.copy()
        single_cost[reused] = 0
        for batch in instance.kernel_batches:
            indices = indices_of(batch)
            allocations[indices[0]] += allocation_bytes(output_of(batch))
            single_cost[list(indices)] = 0
            compute_seconds += batch.seconds
        compute_seconds += float(single_cost.sum())
        release_table = np.array(sorted(action_releases[index]), dtype='<i8').reshape(-1, 2)
        freed = np.zeros(span, dtype=np.int64)
        if len(release_table):
            np.add.at(freed, release_table[:, 0], release_bytes[release_table[:, 1]-node_offset])
        # Charge each Action's output/workspace before releasing its last-use
        # operands; the following Action starts with those owners gone.
        numeric = allocations.cumsum() + resident - np.append(np.int64(0), freed[:-1].cumsum())
        before_numeric = numeric - allocations
        wh = np.zeros(span, dtype=np.int64)
        wd = np.zeros(span, dtype=np.int64)
        for local, (host, device) in workspace_records[index].items():
            wh[local], wd[local] = host, device
        # Output copies precede instance final releases, even when every
        # logical Action was reused. No Action workspace represents this slot.
        wd[-1] += snapshot_device
        # Even slots are after reads/before consumption; odd slots are after
        # raw releases. Multiple reads at one Action retain their staging peaks.
        dh = np.zeros(2 * span, dtype=np.int64)
        dd = np.zeros(2 * span, dtype=np.int64)
        for event in local_events:
            block = effective_blocks[event.block]
            local = event.action_index
            if event.kind == 'read':
                raw_host += block.host_bytes
                raw_device += cfg_rounding.raw_device_bytes(block)
                dh[2*local] += block.host_bytes
                dd[2*local] += cfg_rounding.raw_device_bytes(block)
                host_numeric = int(before_numeric[local]) if backend == 'numpy' else 0
                device_numeric = int(before_numeric[local]) if backend == 'cupy' else 0
                peak_host = max(peak_host, host_numeric + raw_host + block.staging_host_bytes)
                peak_device = max(peak_device, device_numeric + raw_device
                                  + cfg_rounding.staging_device_bytes(block) + group_workspace.get(local, 0))
                max_raw_host = max(max_raw_host, raw_host)
                max_raw_device = max(max_raw_device, raw_device)
            else:
                raw_host -= block.host_bytes
                raw_device -= cfg_rounding.raw_device_bytes(block)
                dh[2*local+1] -= block.host_bytes
                dd[2*local+1] -= cfg_rounding.raw_device_bytes(block)
        # Recover the incoming raw residency from the ending value and deltas.
        rh = (dh.cumsum() + raw_host - dh.sum())[::2]
        rd = (dd.cumsum() + raw_device - dd.sum())[::2]
        peak_numeric = int(numeric.max())
        if backend == 'numpy':
            max_tensor_host = max(max_tensor_host, peak_numeric)
            peak_host = max(peak_host, int((numeric + rh + wh).max()))
            peak_device = max(peak_device, int((rd + wd).max()))
        else:
            max_tensor_device = max(max_tensor_device, peak_numeric)
            peak_host = max(peak_host, int((rh + wh).max()))
            peak_device = max(peak_device, int((numeric + rd + wd).max()))
        max_workspace_host = max(max_workspace_host, int(wh.max()))
        max_workspace_device = max(max_workspace_device, int(wd.max()))
        resident = int(numeric[-1]) - int(release_bytes[np.asarray(local_releases, dtype=np.intp)-node_offset].sum())
        if resident < 0 or raw_host < 0 or raw_device < 0:
            raise ValueError('cfg schedule releases unowned storage')
        instances[index] = replace(instance, events=local_events, releases=local_releases,
            tensor_releases=ArrayConstant.from_array(release_table) if len(release_table) else None)
    if resident or raw_host or raw_device:
        raise ValueError('cfg schedule leaves storage live')
    peak_host += output_reserve + constant_host
    peak_device += plan.reserved_device_bytes + constant_device
    packed_value_cache = min(plan.packed_value_cache_device_bytes,
        0 if device_cap is None else max(0, device_cap - peak_device))
    peak_device += packed_value_cache
    if host_cap is not None and peak_host > host_cap:
        raise _CfgResourceExceeded(f'complete cfg resource envelope requires {peak_host} host bytes; cap is {host_cap}')
    if device_cap is not None and peak_device > device_cap:
        raise _CfgResourceExceeded(f'complete cfg resource envelope requires {peak_device} device bytes; cap is {device_cap}')
    host_spare = 0 if host_cap is None else host_cap - peak_host
    device_spare = 0 if device_cap is None else device_cap - peak_device
    prefetch_host = host_spare if prefetch_host_bytes is None else min(host_spare, prefetch_host_bytes)
    prefetch_device = (device_spare if prefetch_device_bytes is None else
                       min(device_spare, prefetch_device_bytes))
    if type(prefetch_host) is not int or prefetch_host < 0:
        raise ValueError('invalid host read-ahead capacity')
    if type(prefetch_device) is not int or prefetch_device < 0:
        raise ValueError('invalid device read-ahead capacity')
    if backend == 'numpy' and prefetch_device:
        raise ValueError('NumPy cfg plans cannot prefetch device storage')

    cost_values = (prep.seconds(plan.cost_profile) for prep in produced_preparations)
    if preparation_costs is None:
        preparation_seconds = sum(cost_values)
    else:
        cost_values = tuple(cost_values)
        preparation_costs.extend(cost_values)
        preparation_seconds = sum(cost_values)
    stats = dict(
        templates=1, instances=len(instances), read_blocks=len(reads),
        compute_nodes=compute_count, prepared_nodes=len(global_preps),
        binding_node_count=next_node-node_offset,
        compute_seconds=compute_seconds, preparation_seconds=preparation_seconds,
        constant_host_bytes=constant_host, constant_device_bytes=constant_device,
        constant_device_payload_bytes=constant_payload if backend == 'cupy' else 0,
        cfg_allocation_rounding_version=1 if backend == 'cupy' else 0,
        output_snapshot_device_bytes=snapshot_device,
        native_output_snapshot_device_bytes=(cfg_rounding.native_snapshot_device_bytes(plan.state)
                                             if backend == 'cupy' else 0),
        peak_host_bytes=peak_host, peak_device_bytes=peak_device,
        peak_owned_tensor_bytes=max_tensor_host if backend == 'numpy' else max_tensor_device,
        peak_numeric_tensor_host_bytes=max_tensor_host,
        peak_numeric_tensor_device_bytes=max_tensor_device,
        peak_raw_input_bytes=max_raw_host if backend == 'numpy' else max(max_raw_host, max_raw_device),
        peak_raw_host_bytes=max_raw_host, peak_raw_device_bytes=max_raw_device,
        peak_operation_workspace_bytes=max_workspace_host if backend == 'numpy' else max_workspace_device,
        peak_preparation_workspace_host_bytes=max_workspace_host,
        peak_preparation_workspace_device_bytes=max_workspace_device,
        output_host_bytes=output_reserve,
        batch_owner_bytes=sum(output_of(batch) for instance in instances for batch in instance.kernel_batches),
        reserved_device_bytes=plan.reserved_device_bytes,
        packed_value_cache_device_bytes=packed_value_cache,
        host_memory_cap=host_cap, device_memory_cap=device_cap,
        prefetch_host_bytes=prefetch_host, prefetch_device_bytes=prefetch_device,
        binding_seconds=binding_seconds, resource_seconds=perf_counter() - start - binding_seconds,
        binding_bytes=sum(instance.node_ids.array().nbytes + instance.reused.array().nbytes
                          for instance in instances),
        lifecycle_seconds=perf_counter() - start,
    )
    result = CfgPlan(
        (template,), tuple(instances), reads, effective_blocks, global_preps, global_specs,
        peak_host, prefetch_host, output_mode, output_reserve, stats,
        reader_aliases=reader_aliases, backend=backend, peak_device_bytes=peak_device,
        prefetch_device_bytes=prefetch_device, execution_target=execution_target,
        target_summary=target_summary,
    )
    if debug:
        audit_start = perf_counter()
        audit_cfg_plan(result)
        result = replace(result, statistics=dict(result.statistics,
                                                  audit_seconds=perf_counter() - audit_start))
    return result


def validate_cfg_allocation_envelope(plan):
    """Accept a legacy cfg only if its saved envelope covers current charges.

    This is a compatibility check, never replanning or an archive mutation.
    Native v1 plans were charged by the global owner builder; CPU plans do not
    acquire GPU request-granularity costs. Explicit audit remains available.
    """
    if plan.backend != 'cupy' or plan.statistics.get('cfg_allocation_rounding_version') == 1:
        return plan
    try:
        audit_cfg_plan(plan)
    except ValueError as error:
        if 'cfg envelope' not in str(error):
            raise
        raise MemoryError(
            'legacy cfg resource envelope does not cover allocation rounding; '
            'rebuild the resource plan from the existing selected contraction path '
            '(no contraction-path search is required)') from error
    return plan


def audit_cfg_plan(plan):
    """Independently validate saved lifetimes, special Actions, batches and budgets."""
    if plan.output_mode == 'dataset' and plan.output_fields:
        payload = sum(prod(field.shape) * np.dtype(field.dtype).itemsize for field in plan.output_fields)
        largest = max((prod(field.shape) * np.dtype(field.dtype).itemsize
                       for field in plan.output_fields), default=0)
        finite = max((prod(field.shape) for field in plan.output_fields
                      if np.dtype(field.dtype).kind in 'fc'), default=0)
        dt_payload = sum(prod(field.shape) * np.dtype(field.dtype).itemsize
                         for field in plan.output_fields if field.dt_offset >= 0)
        minimum = payload + prod(plan.output_storage_shape) * 8 + largest + finite
        minimum += (plan.physical_lt or 0) * (dt_payload + 8) + prod(plan.dt_storage_shape) * 8
        if plan.output_layout == 'dt_packed':
            n, slots = len(plan.dt_coordinates), plan.one_point_slots
            largest = max((prod(field.shape) * np.dtype(field.dtype).itemsize
                           * (n if field.dt_offset >= 0 else slots)
                           for field in plan.output_fields), default=0)
            minimum += (prod(plan.dt_packed_storage_shape) + prod(plan.one_point_storage_shape)) * 8 + largest
            minimum += sum(prod(field.shape) * np.dtype(field.dtype).itemsize * slots + slots
                           for field in plan.output_fields if field.dt_offset < 0)
        if plan.output_host_bytes < minimum:
            raise ValueError('cfg envelope omits output packing or reconstruction buffers')

    arrays, raw, opened, produced = {}, {}, set(), set()
    live = set()
    resident_host = resident_device = raw_host = raw_device = 0
    batch_owner_host = batch_owner_device = 0
    peak_host = peak_device = 0
    const_host = cfg_rounding.constant_bytes(plan.preparations) if plan.backend == 'numpy' else 0
    const_device = cfg_rounding.constant_bytes(plan.preparations, device=True) if plan.backend == 'cupy' else 0
    allocation_bytes = device_allocation_bytes if plan.backend == 'cupy' else int
    preparation_rounding = (cfg_rounding.preparation_padding(plan.preparations)
                            if plan.backend == 'cupy' else {})
    if plan.backend == 'cupy':
        for name, describe in (
                ('output_snapshot_device_bytes', cfg_rounding.snapshot_device_bytes),
                ('native_output_snapshot_device_bytes', cfg_rounding.native_snapshot_device_bytes)):
            required = max((describe(template.plan.state) for template in plan.templates), default=0)
            recorded = plan.statistics.get(name)
            if (recorded is None and plan.statistics.get('cfg_allocation_rounding_version') == 1
                    or recorded is not None and (type(recorded) is not int or recorded < required)):
                raise ValueError('cfg envelope omits output snapshot allocation metadata')
    owner_for_gid, owner_results, owners = {}, {}, {}
    borrowed = {key for key, prep in plan.preparations.items()
                if prep.recipe is not None and prep.recipe.kind == 'identity' and key not in plan.specs}

    def release_tensor(gid):
        nonlocal resident_host, resident_device, batch_owner_host, batch_owner_device
        if gid not in live:
            raise ValueError('cfg releases an absent tensor')
        live.remove(gid)
        owner_key = owner_for_gid.get(gid)
        if owner_key is None:
            amount = arrays.pop(gid, 0)
            if plan.backend == 'numpy':
                resident_host -= amount
            else:
                resident_device -= amount
        else:
            arrays.pop(gid, None)
            if all(value not in live for value in owners[owner_key][1]):
                amount = owners.pop(owner_key)[0]
                if plan.backend == 'numpy':
                    batch_owner_host -= amount
                else:
                    batch_owner_device -= amount
                for value in tuple(owner_for_gid):
                    if owner_for_gid[value] == owner_key:
                        owner_for_gid.pop(value, None)

    def local_source_size(template, local_key, gid):
        spec = template.plan.state.catalog.get(local_key) or plan.specs.get(str(gid))
        if spec is not None:
            return _tensor_bytes(spec)
        source = template.plan.state.problem.sources.get(local_key)
        if source is None:
            raise ValueError('cfg source has no TensorSpec or SourceSpec size bound')
        return _source_bound(source)

    for instance in plan.instances:
        template = plan.templates[instance.template]
        actions = template.plan.state.actions
        bindings = dict(zip(template.keys, map(int, instance.node_ids.array()), strict=True))
        reuse = instance.reused.array()
        before, after = defaultdict(list), defaultdict(list)
        for event in instance.events:
            if event.kind not in {'read', 'release'}:
                raise ValueError('unknown cfg resource event')
            if not 0 <= event.action_index <= len(actions):
                raise ValueError('cfg resource event has an invalid Action index')
            (before if event.kind == 'read' else after)[event.action_index].append(event)

        batches = {batch.action_indices[0]: batch for batch in instance.kernel_batches}
        if len(batches) != len(instance.kernel_batches):
            raise ValueError('cfg instance has overlapping batches')
        batch_ids = {}
        for batch in instance.kernel_batches:
            if any(index >= len(actions) or reuse[index] for index in batch.action_indices):
                raise ValueError('cfg instance batch member is missing or reused')
            results = tuple(bindings[result.tensor] for result in batch.results)
            if len(set(results)) != len(results):
                raise ValueError('cfg batch result IDs are not distinct')
            owner_key = (instance.action_offset, batch.action_indices[0])
            owners[owner_key] = (allocation_bytes(batch.output_bytes), results)
            for gid in results:
                if gid in owner_for_gid:
                    raise ValueError('one cfg value has multiple batch owners')
                owner_for_gid[gid] = owner_key
                batch_ids[gid] = owner_key

        workspace_host, workspace_device, group_memory = {}, {}, {}
        for point in template.plan.memory:
            workspace_host[point.action_index] = max(
                workspace_host.get(point.action_index, 0), point.workspace_host_bytes)
            workspace_device[point.action_index] = max(
                workspace_device.get(point.action_index, 0),
                point.workspace_device_bytes)
            group_memory[point.action_index] = max(group_memory.get(point.action_index, 0),
                point.scalar_reduction_device_bytes + point.cartesian_reduction_device_bytes)
        for index, action in enumerate(actions):
            if action.kind == 'load':
                prep = plan.preparations[str(bindings[action.tensor])]
                workspace_host[index] = max(workspace_host.get(index, 0), prep.workspace_host_bytes)
                workspace_device[index] = max(workspace_device.get(index, 0), prep.workspace_device_bytes)
        if plan.backend == 'cupy':
            for index, action in enumerate(actions):
                extra = (max(cfg_rounding.padding(workspace_device.get(index, 0)),
                             preparation_rounding.get(str(bindings[action.tensor]), 0))
                         if action.kind == 'load' else cfg_rounding.action_workspace_padding(
                             action, template.plan.state.catalog, workspace_device.get(index, 0)))
                workspace_device[index] = workspace_device.get(index, 0) + extra
        selected_starts = set(batches)
        candidate_ws = {index: (workspace_host.get(index, 0), workspace_device.get(index, 0))
                        for index in set(workspace_host) | set(workspace_device)}
        candidates = _cfg_batch_candidates(template.plan.kernel_batches, reuse, candidate_ws, plan.backend)
        available = {bindings[action.tensor]: index for index, action in enumerate(actions)
                     if not reuse[index]}
        candidates = tuple(_cfg_dependency_batches(candidates, reuse, available,
            bindings, candidate_ws, plan.backend))
        workspace_host = {index: values[0] for index, values in candidate_ws.items()}
        workspace_device = {index: values[1] for index, values in candidate_ws.items()}
        for candidate, ready in candidates:
            start_index = candidate.action_indices[0]
            members_reused = [bool(reuse[index]) for index in candidate.action_indices]
            if start_index not in selected_starts and not all(members_reused):
                if plan.backend == 'numpy':
                    workspace_host[start_index] = max(
                        0, workspace_host.get(start_index, 0) - candidate.packing_bytes)
                else:
                    workspace_device[start_index] = max(
                        0, workspace_device.get(start_index, 0) - candidate.packing_bytes)

        if plan.backend == 'cupy':
            for batch in instance.kernel_batches:
                index = batch.action_indices[0]
                workspace_device[index] = workspace_device.get(index, 0) + cfg_rounding.batch_packing_padding(
                    batch, template.plan.state.catalog)
        tensor_releases = defaultdict(list)
        if instance.tensor_releases is not None:
            for release_index, release_gid in instance.tensor_releases.array().tolist():
                tensor_releases[release_index].append(release_gid)
        for index in range(len(actions) + 1):
            for event in before.get(index, ()):
                if event.block in opened:
                    raise ValueError('cfg schedule repeats a physical read')
                if event.block not in plan.blocks:
                    raise ValueError('cfg event names an unknown physical block')
                opened.add(event.block)
                block = plan.blocks[event.block]
                raw[event.block] = (block.host_bytes, cfg_rounding.raw_device_bytes(block))
                raw_host += block.host_bytes
                raw_device += cfg_rounding.raw_device_bytes(block)
                peak_host = max(peak_host, resident_host + batch_owner_host + raw_host
                                + block.staging_host_bytes)
                peak_device = max(peak_device, resident_device + batch_owner_device + raw_device
                                  + cfg_rounding.staging_device_bytes(block) + group_memory.get(index, 0))
            if index < len(actions):
                batch = batches.get(index)
                if batch is not None:
                    if any(bindings[operand.tensor] not in live
                            for leaf in dense_batch_leaves((batch,))
                            for operands in (leaf.left, leaf.right) for operand in operands):
                        raise ValueError('cfg batch input precedes its producer')
                    results = tuple(bindings[result.tensor] for result in batch.results)
                    if any(gid in live for gid in results):
                        raise ValueError('cfg batch produces an already live value')
                    live.update(results)
                    produced.update(results)
                    arrays.update({gid: 0 for gid in results})
                    if plan.backend == 'numpy':
                        batch_owner_host += batch.output_bytes
                    else:
                        batch_owner_device += allocation_bytes(batch.output_bytes)

                action = actions[index]
                gid = bindings.get(action.tensor)
                if gid is None or gid < 0:
                    raise ValueError('cfg Action has no global value binding')
                if action.kind not in {'load', 'contract', 'sparse_bridge', 'baryon_join'}:
                    raise ValueError('cfg contains an unsupported Action')
                if action.kind != 'load' and not reuse[index]:
                    if any(bindings.get(key) not in live for key in action.operands):
                        raise ValueError('cfg compute consumer precedes its producer')
                if reuse[index]:
                    if gid not in live:
                        raise ValueError('cfg schedule reused an absent value')
                elif gid not in batch_ids:
                    if gid in produced:
                        raise ValueError('cfg schedule repeats a global computation')
                    produced.add(gid)
                    if action.kind == 'load':
                        prep = plan.preparations.get(str(gid))
                        if prep is None or prep.recipe is None:
                            raise ValueError('cfg load has no described preparation')
                        if any(block not in raw for block in prep.blocks):
                            raise ValueError('cfg preparation precedes a physical read')
                        amount = local_source_size(template, action.tensor, gid)
                        amount = 0 if str(gid) in borrowed else amount
                    else:
                        spec = template.plan.state.catalog.get(action.tensor)
                        if spec is None:
                            raise ValueError('cfg compute result has no TensorSpec')
                        amount = _tensor_bytes(spec)
                    live.add(gid)
                    arrays[gid] = amount
                    if plan.backend == 'numpy':
                        resident_host += amount
                    else:
                        resident_device += allocation_bytes(amount)
                        arrays[gid] = allocation_bytes(amount)
                wh, wd = ((0, 0) if reuse[index] else
                          (workspace_host.get(index, 0), workspace_device.get(index, 0)))
                if action.kind == 'load' and not reuse[index]:
                    prep = plan.preparations[str(gid)]
                    wh = max(wh, prep.workspace_host_bytes)
                    wd = max(wd, prep.workspace_device_bytes)
                wd += group_memory.get(index, 0)
                peak_host = max(peak_host, resident_host + batch_owner_host + raw_host + wh)
                peak_device = max(peak_device, resident_device + batch_owner_device + raw_device + wd)
            else:
                # All output snapshots coexist with their source owners until
                # download completes, before instance final releases.
                snapshot = (cfg_rounding.snapshot_device_bytes(template.plan.state)
                            if plan.backend == 'cupy' else 0)
                peak_device = max(peak_device, resident_device + batch_owner_device + raw_device + snapshot)
            for release_gid in tensor_releases.get(index, ()):
                release_tensor(release_gid)
            for event in after.get(index, ()):
                if event.block not in raw:
                    raise ValueError('cfg releases an absent input')
                host_bytes, device_bytes = raw.pop(event.block)
                raw_host -= host_bytes
                raw_device -= device_bytes
        if any(bindings.get(key) not in live for key in template.plan.state.results):
            raise ValueError('cfg output was released before consumption')
        for gid in instance.releases:
            release_tensor(gid)
    if live or arrays or raw or owners:
        raise ValueError('cfg schedule leaves storage live')

    required_host = peak_host + plan.output_host_bytes + const_host
    # The immutable cfg envelope saves the fixed reservation in statistics;
    # CfgPlan itself intentionally exposes only the final device peak.
    reserved_device = plan.statistics.get('reserved_device_bytes')
    if reserved_device is None:
        reserved_device = max((template.plan.reserved_device_bytes
                               for template in plan.templates), default=0)
    required_device = (peak_device + reserved_device + const_device
                       + plan.statistics.get('packed_value_cache_device_bytes', 0))
    host_cap = plan.statistics.get('host_memory_cap')
    device_cap = plan.statistics.get('device_memory_cap')
    if required_host > plan.peak_host_bytes:
        raise ValueError('cfg envelope omits host live storage')
    if required_device > plan.peak_device_bytes:
        raise ValueError('cfg envelope omits device live storage')
    if host_cap is not None and required_host > host_cap:
        raise ValueError('cfg envelope exceeds host memory cap')
    if device_cap is not None and required_device > device_cap:
        raise ValueError('cfg envelope exceeds device memory cap')
    if host_cap is not None and required_host + plan.prefetch_host_bytes > host_cap:
        raise ValueError('cfg host read-ahead exceeds remaining capacity')
    if device_cap is not None and required_device + plan.prefetch_device_bytes > device_cap:
        raise ValueError('cfg device read-ahead exceeds remaining capacity')
    return {
        'peak_host_bytes': required_host,
        'peak_device_bytes': required_device,
        'read_blocks': len(opened),
        'computed_values': len(produced),
        'batches': sum(len(instance.kernel_batches) for instance in plan.instances),
    }


def _output_storage_shape(words):
    columns=min(words,4096)
    return ((words+columns-1)//columns,columns) if columns else (0,0)


def configure_cfg_output_layout(plan,layout,coordinates,*,host_cap):
    """Pure offline plan-to-plan output layout and declared host-cap accounting.

    Copied from W.configure_packed_dt; no Calculation or resource discovery is
    used here. Execution consumes the frozen resulting packing/budget tables.
    """
    if layout=='pairs_and_dt':
        if coordinates is not None or plan.output_layout!='pairs_and_dt':
            raise ValueError('dt coordinates require packed output; layout cannot revert')
        return plan
    if layout!='dt_packed': raise ValueError('unknown output layout')
    if plan.output_mode!='dataset' or not plan.physical_lt or not plan.dt_words:
        raise ValueError('packed dt requires dataset mode, physical Lt and dt fields')
    coordinates=tuple(range(1,plan.physical_lt)) if coordinates is None else tuple(coordinates)
    fields=[]
    for f in plan.output_fields:
        role=f.time_role
        if f.dt_offset<0 and role is None:
            names={'vacuum_sink':plan.time_roles[0],'vacuum_source':plan.time_roles[1]}
            role=names.get(f.path[-1] if f.path else None)
            if role is None: raise ValueError(f'no time role for one-point field {f.path!r}')
        fields.append(replace(f,time_role=role))
    auxiliary=[f for f in fields if f.dt_offset<0]
    slots=1+max(int(value) for point in plan.coordinates for role,value in point if role in plan.time_roles) if auxiliary else 0
    dt_shape=_output_storage_shape(plan.dt_words*len(coordinates))
    aux_shape=_output_storage_shape(sum(f.words for f in auxiliary)*slots)
    largest=max((prod(f.shape)*np.dtype(f.dtype).itemsize*(len(coordinates) if f.dt_offset>=0 else slots)
                 for f in fields),default=0)
    # Packed buffers are int64: no isfinite boolean scratch can be needed.
    # Floating/complex field diagnostics retain the base per-field reserve.
    extra=(prod(dt_shape)+prod(aux_shape))*8+largest
    extra+=sum(prod(f.shape)*np.dtype(f.dtype).itemsize*slots+slots for f in auxiliary)
    delta=extra-plan.statistics.get('packed_dt_extra_host_bytes',0)
    peak=plan.peak_host_bytes+delta
    cap=host_cap
    if peak>cap: raise MemoryError(f'packed output requires {peak} host bytes; cap is {cap}')
    prefetch=min(plan.prefetch_host_bytes,cap-peak)
    plan=replace(plan,output_layout=layout,dt_coordinates=coordinates,output_fields=tuple(fields),
        dt_packed_storage_shape=dt_shape,one_point_storage_shape=aux_shape,one_point_slots=slots,
        peak_host_bytes=peak,output_host_bytes=plan.output_host_bytes+delta,prefetch_host_bytes=prefetch,
        statistics=dict(plan.statistics,packed_dt_extra_host_bytes=extra,peak_host_bytes=peak,
                        output_host_bytes=plan.output_host_bytes+delta,prefetch_host_bytes=prefetch))
    return plan


__all__ = ['build_cfg_plan', 'audit_cfg_plan', 'validate_cfg_allocation_envelope',
           'configure_cfg_output_layout', 'iter_cfg_plan_windows']
