"""Describe coordinate lanes from a selected scalar input contract.

This metadata pass binds the existing read/preparation recipes; it neither
reads arrays nor chooses contraction operations.  Global scalar preparation
IDs prove sharing.  A batch adds a retained leading axis only to sources whose
actual IDs differ, using the existing backend-independent stack preparation.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import prod
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import (
    AxisPattern, BatchSpec, CoordinateBatch, CoordinateSpec, DiagramInputs,
    SourcePreparation, TensorPreparation, canonical_axis_binding_identity,
)
from .cfg_plan import (
    _bind_read_request, _merge_read_aliases, _preparation_order, _read_identity,
)


def _frozen_key(value):
    """Allow the same list-valued reader conventions as complete cfg plans."""
    if isinstance(value, Mapping):
        return tuple((key, _frozen_key(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_key(item) for item in value)
    hash(value)
    return value


def _freeze_map(value):
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CoordinateInputBinding:
    problem: object
    points: tuple
    requests: Mapping
    blocks: Mapping
    preparations: Mapping
    specs: Mapping
    source_names: tuple
    reader_aliases: tuple = ()


@dataclass(frozen=True)
class CoordinateBatchInputs:
    inputs: DiagramInputs
    preparation_closure: Mapping
    specs: Mapping
    requests: Mapping
    dynamic_sources: frozenset
    lane_axis: int
    source_map: Mapping

    @property
    def problem(self):
        return self.inputs.problem

    @property
    def blocks(self):
        return self.inputs.blocks

    @property
    def preparations(self):
        return self.inputs.preparations

    @property
    def points(self):
        return self.inputs.coordinate_batch.points


def _compatible_read(actual, expected):
    """Ranks/extents and preparation semantics are fixed by the selected DAG."""
    if (actual.axes != expected.axes or actual.shape != expected.shape
            or actual.dtype != expected.dtype or actual.sparse_layout != 'dense'
            or actual.point_diagonal != expected.point_diagonal
            or actual.structure != expected.structure):
        raise ValueError('coordinate changes the selected input axes, shape, dtype or structure')


def _same_physical_metadata(left, right):
    _compatible_read(left, right)
    if (canonical_axis_binding_identity(left.axis_bindings)
            != canonical_axis_binding_identity(right.axis_bindings)
            or left.data_identity != right.data_identity
            or left.selection_identity != right.selection_identity):
        raise ValueError('coordinate aliases describe different physical inputs')


def bind_coordinate_inputs(template_scene_plan, requests, preparations, specs,
                           coordinate_points, *, read_metadata=None,
                           reader_aliases=(), debug=False):
    """Bind all actual scalar input identities without symbolic expansion/IO.

    ``source_names[i]`` maps each original logical source to its actual scalar
    preparation at point ``i``.  The other maps hold the deduplicated global
    read/preparation closure.  Only dense, described native recipes are admitted.
    """
    problem = template_scene_plan.state.problem
    virtual = {core.key for core in problem.meson_cores}
    if problem.sparse_bridges or set(problem.sources) | virtual != set(problem.specs):
        raise NotImplementedError('coordinate input lifting requires dense tensor sources')
    points = tuple(coordinate_points)
    if not points:
        raise ValueError('coordinate input binding requires at least one point')
    if any(len({str(role) for role, _ in point.values}) != len(point.values) for point in points):
        raise ValueError('coordinate input binding requires distinct encoded time roles')
    requests, specs = dict(requests), dict(specs)
    # P's chosen entries carry device workspace placement; E carries closure.
    preparations = dict(preparations)
    preparations.update({key: (replace(prep, recipe=preparations[key].recipe)
        if prep.recipe is None and key in preparations else prep)
        for key, prep in template_scene_plan.preparations.items()})
    for key in problem.sources:
        if key not in preparations:
            raise ValueError(f'coordinate input {key!r} has no preparation')
    for key, prep in preparations.items():
        if prep.recipe is None:
            raise ValueError(f'coordinate input {key!r} has an opaque preparation')
        if key not in specs:
            spec = problem.specs.get(key, template_scene_plan.state.catalog.get(key))
            if spec is None:
                raise ValueError(f'coordinate preparation {key!r} needs a TensorSpec')
            specs[key] = spec
        if any(raw not in requests for raw in prep.blocks):
            raise ValueError(f'coordinate preparation {key!r} has an undescribed raw block')
        recipe = prep.recipe
        if recipe.kind in {'identity', 'spin_spatial'} and recipe.inputs[0] not in requests:
            raise NotImplementedError('coordinate leaf recipes require described raw inputs')
    order = _preparation_order(preparations, requests)
    metadata = {} if read_metadata is None else dict(read_metadata)
    read_binders = {}
    for key, request in requests.items():
        if key not in template_scene_plan.blocks:
            raise ValueError(f'coordinate read {key!r} has no StorageBlock')
        if request.metadata.sparse_layout != 'dense' or template_scene_plan.blocks[key].read_kind != 'dense':
            raise NotImplementedError('coordinate input lifting requires dense reads')
        read_binders[key] = _bind_read_request(request, metadata, debug=debug)
    roles = {role: group[0] for group in reader_aliases for role in group}
    reserved = set(specs) | set(requests) | set(problem.sources)
    prefix = 'coordinate:'
    while any(key.startswith(prefix) for key in reserved):
        prefix = '_' + prefix
    raw_ids, prep_ids = {}, {}
    reads, blocks, global_preps, global_specs, source_names = {}, {}, {}, {}, []
    definitions = {}
    for key, prep in preparations.items():
        recipe = prep.recipe
        # The complete transform/precision/constants remain in this identity.
        definitions[key] = _frozen_key((
            tuple(getattr(prep, field.name) for field in fields(prep)
                  if field.name not in {'blocks', 'recipe'}),
            tuple(getattr(recipe, field.name) for field in fields(recipe) if field.name != 'inputs'),
            specs[key].shape, specs[key].dtype.str,
        ))
        if problem.backend == 'cupy':
            preparations[key] = replace(prep, workspace_host_bytes=0,
                workspace_device_bytes=prep.workspace_host_bytes + prep.workspace_device_bytes)
    for point in points:
        values = {str(role): value for role, value in point.values}
        raw_map = {}
        for key, bind_read in read_binders.items():
            actual = bind_read(values)
            _compatible_read(actual.metadata, requests[key].metadata)
            identity = _frozen_key(_read_identity(actual, roles))
            for alias in actual.aliases:
                if _frozen_key(_read_identity(alias, roles)) != identity:
                    raise ValueError('coordinate changes the template input sharing relation')
                _same_physical_metadata(alias.metadata, actual.metadata)
            name = raw_ids.get(identity)
            if name is None:
                name = f'{prefix}raw:{len(raw_ids)}'
                raw_ids[identity] = name
                reads[name] = actual
                block = template_scene_plan.blocks[key]
                blocks[name] = replace(block,
                    dataset=actual.metadata.data_identity or block.dataset,
                    # The storage owner ID includes the resolved request. A
                    # reader may use one selection header for several times.
                    selection=f'{actual.metadata.selection_identity}:{name}')
            else:
                _same_physical_metadata(reads[name].metadata, actual.metadata)
                reads[name] = _merge_read_aliases(reads[name], actual, debug=debug)
                previous, block = blocks[name], template_scene_plan.blocks[key]
                # Aliases share one owner; retain every declared byte bound.
                blocks[name] = replace(previous, **{field: max(getattr(previous, field), getattr(block, field))
                    for field in ('read_bytes', 'host_bytes', 'device_bytes', 'transfer_bytes',
                                  'staging_host_bytes', 'staging_device_bytes')})
            raw_map[key] = name
        bound = {}
        for key in order:
            prep, recipe = preparations[key], preparations[key].recipe
            members = tuple((bound[member] if recipe.kind in {'stack', 'linear_combination'} else raw_map[member])
                            for member in recipe.inputs)
            dependencies = tuple(dict.fromkeys(raw_map[raw] for raw in prep.blocks))
            constants = (recipe.constants_at(reads[raw_map[recipe.inputs[0]]].arguments)
                         if recipe.coordinate_constants else recipe.constants)
            identity = definitions[key], members, dependencies
            if recipe.coordinate_constants:
                identity += (constants,)
            name = prep_ids.get(identity)
            if name is None:
                name = f'{prefix}prepared:{len(prep_ids)}'
                prep_ids[identity] = name
                global_preps[name] = replace(prep, blocks=dependencies,
                    recipe=replace(recipe, inputs=members, constants=constants, coordinate_constants=()))
                global_specs[name] = specs[key]
            bound[key] = name
        source_names.append(_freeze_map({key: bound[key] for key in problem.sources}))
    return CoordinateInputBinding(problem, points, _freeze_map(reads), _freeze_map(blocks),
        _freeze_map(global_preps), _freeze_map(global_specs), tuple(source_names), tuple(reader_aliases))


def describe_coordinate_batch(binding, group_indices, *, lane_axis, key_prefix=''):
    """Create one pure batched problem and its existing stack recipe closure.

    Original source keys and term order remain unchanged for fixed-path lifting.
    Scalar members use global IDs, so a source-only input is read/prepared once.
    ``inputs.preparations`` contains only top-level logical sources; the full
    ``preparation_closure`` and ``specs`` must be passed to the restored loader.
    """
    indices = tuple(group_indices)
    if not isinstance(key_prefix, str):
        raise TypeError('coordinate source key prefix must be a string')
    if (not indices or any(type(index) is not int or not 0 <= index < len(binding.points) for index in indices)
            or tuple(sorted(set(indices))) != indices):
        raise ValueError('coordinate batch indices must be nonempty, unique and ordered')
    labels = {axis for term in binding.problem.terms
              for axes in (*term.subscripts.inputs, term.subscripts.output) for axis in axes}
    if type(lane_axis) is not int or lane_axis < 0 or lane_axis in labels:
        raise ValueError('coordinate lane axis must be a new nonnegative label')
    problem = binding.problem
    source_map = {key: key_prefix + key for key in problem.sources}
    if set(source_map.values()) & set(binding.preparations):
        raise ValueError('coordinate source keys collide with scalar closure IDs')
    dynamic, top_preps, top_specs, sources = set(), {}, {}, {}
    closure, closure_specs = {}, {}

    def include(name):
        if name in closure:
            return
        prep = binding.preparations[name]
        if prep.recipe.kind in {'stack', 'linear_combination'}:
            for member in prep.recipe.inputs:
                include(member)
        closure[name], closure_specs[name] = prep, binding.specs[name]

    lane_name = 'coordinate_batch'
    while any(lane_name in spec.axes for spec in problem.specs.values()):
        lane_name = '_' + lane_name
    for old_key, source in problem.sources.items():
        key = source_map[old_key]
        members = tuple(binding.source_names[index][old_key] for index in indices)
        spec = problem.specs[old_key]
        if all(member == members[0] for member in members):
            prep = binding.preparations[members[0]]
            if prep.recipe.kind in {'stack', 'linear_combination'}:
                for member in prep.recipe.inputs:
                    include(member)
            output_spec = spec
        else:
            dynamic.add(key)
            for member in members:
                include(member)
            parts = tuple(binding.preparations[member] for member in members)
            dependencies = tuple(dict.fromkeys(raw for part in parts for raw in part.blocks))
            output_spec = replace(spec, shape=(len(indices), *spec.shape),
                description=(None if spec.description is None else replace(spec.description,
                    axes=((lane_name, *spec.axes) if spec.axes else ()))))
            # Existing stack preparation can write members into destination
            # slices. The fallback may coexist with one materialized member.
            member_bytes = prod(spec.shape) * spec.dtype.itemsize
            workspace = max(part.workspace_host_bytes + part.workspace_device_bytes for part in parts) + member_bytes
            prep = SourcePreparation(dependencies,
                workspace_host_bytes=workspace if problem.backend == 'numpy' else 0,
                workspace_device_bytes=workspace if problem.backend == 'cupy' else 0,
                flops=sum(part.flops for part in parts),
                traffic_bytes=sum(part.traffic_bytes for part in parts) + 2 * len(indices) * member_bytes,
                kernel_calls=sum(part.kernel_calls for part in parts) + len(indices),
                dtype=spec.dtype.name,
                recipe=TensorPreparation('stack', members, compute_dtype=spec.dtype.name))
        top_preps[key], top_specs[key] = prep, output_spec
        raw_blocks = tuple(binding.blocks[raw] for raw in prep.blocks)
        region_members = members if key in dynamic else members[:1]
        sources[key] = replace(source, region=f'coordinate-source:{region_members!r}',
            region_bytes=sum(block.read_bytes for block in raw_blocks),
            host_region_bytes=sum(block.host_bytes for block in raw_blocks), staging_bytes=0,
            transfer_bytes=sum(block.transfer_bytes for block in raw_blocks),
            read_bytes=sum(block.read_bytes for block in raw_blocks),
            snapshot_bytes=prod(output_spec.shape) * output_spec.dtype.itemsize)
    closure.update(top_preps)
    closure_specs.update(top_specs)
    from .meson_coordinates import describe_meson_core_batch, lift_meson_kernel
    meson_cores = describe_meson_core_batch(problem, source_map, dynamic, top_specs,
        key_prefix=key_prefix, width=len(indices), lane_name=lane_name)
    closure_specs.update(top_specs)
    raw_keys = tuple(dict.fromkeys(raw for prep in closure.values() for raw in prep.blocks))
    blocks = _freeze_map({key: binding.blocks[key] for key in raw_keys})
    requests = _freeze_map({key: binding.requests[key] for key in raw_keys})
    terms, output_axes = [], []
    for term in problem.terms:
        term_inputs = tuple(source_map[key] for key in term.inputs)
        has_lane = any(key in dynamic for key in term_inputs)
        pattern = AxisPattern(tuple((lane_axis, *axes) if key in dynamic else axes
            for key, axes in zip(term_inputs, term.subscripts.inputs, strict=True)),
            (lane_axis, *term.subscripts.output) if has_lane else term.subscripts.output)
        terms.append(replace(term, inputs=term_inputs, subscripts=pattern,
            kernel=(None if term.kernel is None else lift_meson_kernel(
                term.kernel, source_map, dynamic, lane_axis))))
        output_axes.append(0 if has_lane else None)
    from lqcdweave.contracts.correlator.cores import lift_baryon_join
    joins = tuple(lift_baryon_join(join, source_map, dynamic, top_specs, lane_axis, len(indices))
                  for join in problem.baryon_joins)
    batched = replace(problem, specs=top_specs, sources=sources, terms=tuple(terms),
                      baryon_joins=joins, meson_cores=meson_cores)
    coordinates = CoordinateSpec('pairs', (), (), ('pair',), binding.points, (len(binding.points),))
    coordinate_batch = CoordinateBatch(0, 'pair', (), indices,
        tuple(binding.points[index] for index in indices), len(indices))
    inputs = DiagramInputs(batched, blocks, _freeze_map(top_preps), None, coordinates,
        BatchSpec(len(indices), 'pair'), tuple(f'term:{index}' for index in range(len(terms))),
        coordinate_batch=coordinate_batch, batch_output_axes=tuple(output_axes), read_requests=requests)
    return CoordinateBatchInputs(inputs, _freeze_map(closure), _freeze_map(closure_specs),
        requests, frozenset(dynamic), lane_axis, _freeze_map(source_map))


__all__ = ['CoordinateInputBinding', 'CoordinateBatchInputs',
           'bind_coordinate_inputs', 'describe_coordinate_batch']
