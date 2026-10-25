"""Budgeted dense preparation lifetimes across complete-cfg binding windows.

This pass edits the delivered P tables once. X needs neither a cache nor a
new operation: a retained owner is an ordinary reused load and a later release.
The scratch index contains input/preparation identities, never compute nodes.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from math import prod
from time import perf_counter

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, SourceReadRequest, TensorAxisMetadata,
    TensorPreparation, TensorSpec, device_allocation_bytes,
)


def _identity(value):
    """Lossless, type-aware identity of native immutable scientific metadata."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _identity(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_identity(item) for item in value)
    if is_dataclass(value):
        return type(value), tuple((f.name, _identity(getattr(value, f.name)))
                                  for f in fields(value))
    if isinstance(value, np.dtype):
        return np.dtype, value.str
    return type(value), value


def _preparation_groups(plan):
    """Intern full read/recipe identities, returning only native terminal owners."""
    roles = {role: group for group in plan.reader_aliases for role in group}
    metadata, specs, identities, tokens = {}, {}, {}, {}

    def read_identity(request):
        meta = metadata.get(id(request.metadata))
        if meta is None:
            meta = metadata[id(request.metadata)] = _identity(request.metadata)
        return (request.family, request.metadata.data_identity or
                roles.get(request.reader_role, request.reader_role),
                _identity(request.arguments), _identity(request.selection),
                _identity(request.conventions), meta)

    for key, request in plan.requests.items():
        if any(type(item) is not SourceReadRequest or type(item.metadata) is not TensorAxisMetadata
               for item in (request, *request.aliases)):
            tokens[key] = identities.setdefault(('opaque-read', key), len(identities))
            continue
        identity = ('read', read_identity(request),
                    frozenset(read_identity(alias) for alias in request.aliases))
        tokens[key] = identities.setdefault(identity, len(identities))
    members = {member for prep in plan.preparations.values()
               if prep.recipe is not None for member in prep.recipe.inputs}
    groups = defaultdict(list)
    # Native cfg preparation records are in dependency order. Opaque/extension
    # records cannot prove pure ownership or a fixed preparation, and propagate
    # distinct tokens to parents instead of being guessed compatible.
    for key, prep in plan.preparations.items():
        recipe = prep.recipe
        spec = plan.specs.get(key)
        native = (type(prep) is SourcePreparation and type(recipe) is TensorPreparation
                  and (spec is None or type(spec) is TensorSpec))
        if not native:
            tokens[key] = len(identities)
            identities[('opaque', key)] = tokens[key]
            continue
        # The immutable spec and recipe constants are shared across coordinates;
        # cache their semantic description instead of traversing basis metadata
        # again for every global owner.
        spec_identity = specs.get(id(spec))
        if spec_identity is None:
            spec_identity = specs[id(spec)] = _identity(spec)
        definition = (
            tuple(getattr(prep, f.name) for f in fields(prep)
                  if f.name not in {'blocks', 'recipe'}),
            tuple(getattr(recipe, f.name) for f in fields(recipe) if f.name != 'inputs'),
            spec_identity,
        )
        identity = ('preparation', definition, tuple(tokens[member] for member in recipe.inputs),
                    tuple(tokens[block] for block in prep.blocks))
        token = tokens[key] = identities.setdefault(identity, len(identities))
        if (key not in members and spec is not None and prep.blocks
                and recipe.kind in {'spin_spatial', 'stack'}):
            groups[token].append(int(key))
    return [values for values in groups.values() if len(values) > 1]


def retain_cfg_preparations(plan, preparation_costs):
    """Carry exact terminal dense owners when a conservative envelope admits it.

    Original global peaks plus the maximum added occupancy must fit each cap.
    Inclusive coordinate intervals overcharge partial-coordinate gaps, so this
    remains safe without rebuilding a per-Action memory trajectory. Ranking is
    role-independent: saved preparation plus read/transfer traffic cost per
    byte-coordinate. Original kernel groups and compute IDs remain fixed.
    """
    side = 'host' if plan.backend == 'numpy' else 'device'
    base = getattr(plan, f'peak_{side}_bytes')
    cap = plan.statistics[f'{side}_memory_cap']
    if len(plan.instances) < 2 or cap is None or cap <= base:
        return plan
    # Inspect only the small template preparation closure before considering
    # any full-cfg read/recipe index or scanning the compute Actions.
    eligible = False
    for template in plan.templates:
        preparations = template.preparation_closure or template.plan.preparations
        members = {member for prep in preparations.values()
                   if prep.recipe is not None for member in prep.recipe.inputs}
        if any(key not in members and type(prep) is SourcePreparation
               and type(prep.recipe) is TensorPreparation and prep.blocks
               and prep.recipe.kind in {'spin_spatial', 'stack'}
               for key, prep in preparations.items()):
            eligible = True
            break
    if not eligible:
        return plan
    if any(action.kind not in {'load', 'contract'} or action.resource is not None
           for template in plan.templates for action in template.plan.state.actions):
        return plan
    started = perf_counter()
    groups = _preparation_groups(plan)
    if not groups:
        return plan
    candidates = {gid for group in groups for gid in group}
    programs = []
    for template in plan.templates:
        positions = {key: index for index, key in enumerate(template.keys)}
        loads = [(index, positions[action.tensor])
                 for index, action in enumerate(template.plan.state.actions) if action.kind == 'load']
        programs.append((np.array([i for i, _ in loads], dtype=np.intp),
                         np.array([i for _, i in loads], dtype=np.intp)))
    producers, releases = {}, {}
    costs = iter(preparation_costs)
    for point, instance in enumerate(plan.instances):
        actions, keys = programs[instance.template]
        ids, reused = instance.node_ids.array()[keys], instance.reused.array()[actions]
        for index in np.flatnonzero(~reused):
            cost = next(costs)
            gid = int(ids[index])
            if gid in candidates:
                producers[gid] = (point, int(actions[index]), cost)
    candidates.intersection_update(producers)
    ordered = np.array(sorted(candidates), dtype='<i8')
    if not len(ordered):
        return plan
    for point, instance in enumerate(plan.instances):
        if instance.tensor_releases is not None:
            table = instance.tensor_releases.array()
            found = np.searchsorted(ordered, table[:, 1])
            mask = found < len(ordered)
            mask[mask] &= ordered[found[mask]] == table[mask, 1]
            for action, gid in table[mask]:
                releases[int(gid)] = (point, int(action))
        for gid in instance.releases:
            if gid in candidates:
                releases[gid] = (point, len(plan.templates[instance.template].plan.state.actions))
    edges = []
    allocation = device_allocation_bytes if plan.backend == 'cupy' else int
    for group in groups:
        owners = sorted((gid for gid in group if gid in candidates), key=producers.__getitem__)
        for previous, current in zip(owners, owners[1:]):
            begin, end = releases[previous][0], producers[current][0]
            # Distinct lifetimes must be strictly ordered. Within a binding
            # window native identity interning already provides all reuse.
            if releases[previous] >= producers[current][:2]:
                continue
            spec = plan.specs[str(current)]
            amount = allocation(prod(spec.shape)*spec.dtype.itemsize)
            if not amount:
                continue
            prep = plan.preparations[str(current)]
            profile = plan.templates[plan.instances[end].template].plan.cost_profile
            benefit = producers[current][2] + profile.snapshot_seconds(sum(
                plan.blocks[key].read_bytes + plan.blocks[key].transfer_bytes for key in prep.blocks))
            if benefit <= 0:
                continue
            edges.append((-(benefit / (amount*(end-begin+1))), current, previous, begin, end, amount))
    parent = {}

    def owner(gid):
        trail = []
        while gid in parent:
            trail.append(gid)
            gid = parent[gid]
        for old in trail:
            parent[old] = gid
        return gid

    occupancy = np.zeros(len(plan.instances), dtype=np.int64)
    spare = cap-base
    for _, current, previous, begin, end, amount in sorted(edges):
        interval = occupancy[begin:end+1]
        if int(interval.max()) + amount > spare:
            continue
        parent[owner(current)] = owner(previous)
        interval += amount
    if not parent:
        return plan
    merged = set(parent)
    replacements = {gid: owner(gid) for gid in merged}
    clusters = defaultdict(list)
    for gid, target in replacements.items():
        clusters[target].append(gid)
    moved_releases = {}
    remove = set(replacements)
    for target, others in clusters.items():
        remove.add(target)
        location = max(releases[gid] for gid in (target, *others))
        moved_releases.setdefault(location, []).append(target)
    moved_actions = defaultdict(list)
    for (point, action), gids in moved_releases.items():
        moved_actions[point].extend((action, gid) for gid in gids)
    old_ids = np.array(sorted(replacements), dtype='<i8')
    new_ids = np.array([replacements[int(gid)] for gid in old_ids], dtype='<i8')
    removed_ids = np.array(sorted(remove), dtype='<i8')
    instances, active_raw = [], set()
    kept_costs = []
    costs = iter(preparation_costs)
    skipped = 0
    skipped_seconds = 0.0
    for point, instance in enumerate(plan.instances):
        ids = instance.node_ids.array().copy()
        found = np.searchsorted(old_ids, ids)
        mask = found < len(old_ids)
        mask[mask] &= old_ids[found[mask]] == ids[mask]
        ids[mask] = new_ids[found[mask]]
        reused = instance.reused.array().copy()
        actions, keys = programs[instance.template]
        original_ids = instance.node_ids.array()[keys]
        for index in np.flatnonzero(~reused[actions]):
            cost = next(costs)
            if int(original_ids[index]) in replacements:
                reused[actions[index]] = True
                skipped += 1
                skipped_seconds += plan.templates[instance.template].plan.state.actions[actions[index]].seconds
            else:
                kept_costs.append(cost)
                active_raw.update(plan.preparations[str(int(original_ids[index]))].blocks)
        table = (instance.tensor_releases.array() if instance.tensor_releases is not None
                 else np.empty((0, 2), dtype='<i8'))
        table = table[~np.isin(table[:, 1], removed_ids)]
        last = len(plan.templates[instance.template].plan.state.actions)
        additions = [(action, gid) for action, gid in moved_actions[point] if action < last]
        if additions:
            table = np.concatenate((table, np.array(additions, dtype='<i8')))
            table = table[np.argsort(table[:, 0], kind='stable')]
        final = tuple(gid for gid in instance.releases if gid not in remove)
        final += tuple(moved_releases.get((point, last), ()))
        instances.append(replace(instance, node_ids=ArrayConstant.from_array(ids),
            reused=ArrayConstant.from_array(reused), releases=final,
            tensor_releases=ArrayConstant.from_array(table) if len(table) else None))
    before_reads = sum(event.kind == 'read' for instance in instances for event in instance.events)
    instances = tuple(replace(instance, events=tuple(event for event in instance.events
                            if event.block in active_raw)) for instance in instances)
    after_reads = sum(event.kind == 'read' for instance in instances for event in instance.events)
    extra = int(occupancy.max())
    stats = dict(plan.statistics, preparation_seconds=sum(kept_costs),
        compute_seconds=plan.statistics['compute_seconds']-skipped_seconds,
        cross_window_reused_preparations=skipped, cross_window_saved_reads=before_reads-after_reads,
        cross_window_carry_bytes=extra, cross_window_carry_seconds=perf_counter()-started,
        executed_read_blocks=after_reads)
    peak_key = f'peak_numeric_tensor_{side}_bytes'
    stats[peak_key] += extra
    stats['peak_owned_tensor_bytes'] += extra
    stats[f'peak_{side}_bytes'] += extra
    stats[f'prefetch_{side}_bytes'] = min(stats[f'prefetch_{side}_bytes'], spare-extra)
    return replace(plan, instances=instances, statistics=stats,
        **{f'peak_{side}_bytes': base+extra,
           f'prefetch_{side}_bytes': stats[f'prefetch_{side}_bytes']})
