"""Cfg owner/read peaks retain Cartesian group envelopes through reuse."""
from dataclasses import replace

import numpy as np
import pytest

from contracts.test_cartesian_reduction_group import fixture, annotated, describe
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionTerm, SourcePreparation, SourceSpec, SourceReadRequest,
    StorageBlock, TensorAxisMetadata, TensorPreparation, TensorSpec,
    dense_batch_memory, describe_coordinates,
    device_allocation_bytes,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.api import ContractionState, CostProfile, describe_dense_batch, plan_scene_storage
from lqcdweave.contraction.execution import cfg_runtime


def constant_preparations(specs, *, partial=False):
    result = {}
    for number, (key, spec) in enumerate(specs.items()):
        value = number+1
        if partial and key == 'a1:1':
            # One row aliases its predecessor. Only that child's second row
            # contractions are reused; the original complete group must fall back.
            value -= 1
        result[key] = SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.full(spec.shape, value, spec.dtype)),)))
    return result


def simple_cfg(*, partial=False, debug=False):
    original = fixture(2)
    plan = annotated(original, (describe(original),))
    specs = plan.state.problem.specs
    preparations = constant_preparations(specs, partial=partial)
    points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}])
    cfg = cfg_plan.build_cfg_plan(plan, None, points.points, {}, preparations, specs, debug=debug)
    return plan, cfg


def active_groups(cfg, instance):
    template = cfg.templates[instance.template]
    actions = tuple(replace(action, reused=bool(reused)) for action, reused in
                    zip(template.plan.state.actions, instance.reused.array(), strict=True))
    state = replace(template.plan.state, actions=actions)
    return cfg_runtime._active_cartesian_groups(state, instance.kernel_batches,
                                                template.plan.cartesian_reduction_groups)


@pytest.mark.parametrize('debug', (False, True))
@pytest.mark.parametrize('partial', (False, True))
def test_cfg_remap_partial_reuse_and_original_fallback_keep_conservative_budget(debug, partial):
    plan, cfg = simple_cfg(partial=partial, debug=debug)
    template = cfg.templates[0].plan
    kept = [i for i, a in enumerate(plan.state.actions) if a.kind != 'release']
    positions = {old: new for new, old in enumerate(kept)}
    positions[len(plan.state.actions)] = len(kept)  # Terminal output ownership survives.
    assert template.cartesian_reduction_groups[0].batch_action_indices == tuple(
        tuple(positions[i] for i in child) for child in plan.cartesian_reduction_groups[0].batch_action_indices)
    assert [p.cartesian_reduction_device_bytes for p in template.memory] == [
        p.cartesian_reduction_device_bytes for p in plan.memory if p.action_index in positions]
    audit = cfg_plan.audit_cfg_plan(cfg)
    assert audit['peak_device_bytes'] == cfg.peak_device_bytes
    assert np.all(cfg.instances[1].reused.array())
    assert active_groups(cfg, cfg.instances[1]) == ()
    if partial:
        assert active_groups(cfg, cfg.instances[0]) == ()
        assert len(cfg.instances[0].kernel_batches) == 1
    else:
        assert active_groups(cfg, cfg.instances[0]) == template.cartesian_reduction_groups
    # Even a fully reused coordinate retains the immutable template's charge;
    # dropping annotations must not silently drop this conservative envelope.
    filtered = replace(cfg, templates=(replace(cfg.templates[0], plan=replace(
        template, cartesian_reduction_groups=())),))
    assert cfg_plan.audit_cfg_plan(filtered)['peak_device_bytes'] == cfg.peak_device_bytes


def read_between_children(*, staging=100_000, partial=False):
    original = fixture(2)
    specs = dict(original.state.problem.specs)
    specs.update(probe=TensorSpec((1,), np.float32), q=TensorSpec((1,), np.float32))
    terms = (*original.state.problem.terms, ContractionTerm(('probe', 'q'), 'k,k->'))
    sources = dict(original.state.problem.sources)
    sources.update(probe=SourceSpec('probe-raw', 0, snapshot_bytes=4, read_bytes=0),
                   q=SourceSpec('q', 0, read_bytes=0))
    problem = replace(original.state.problem, specs=specs, terms=terms, sources=sources)
    profile = CostProfile()
    state = ContractionState.initial(problem)
    for key in (*original.state.problem.specs, 'q'):
        state = state.load(key, profile, 1 << 30)
    batches = []
    for child in range(2):
        if child == 1:
            state = state.load('probe', profile, 1 << 30)
            state = state.contract(8, (0, 1), profile, 1 << 30)
            state = state.release('probe').release('q')
        indices = []
        for term in range(4*child, 4*(child+1)):
            indices.append(len(state.actions))
            state = state.contract(term, (0, 1), profile, 1 << 30)
        batches.append(describe_dense_batch(state.actions, tuple(indices), state.catalog, profile, backend='cupy'))
        for key in (f'{side}{child}:{i}' for side in ('a', 'b') for i in range(2)):
            state = state.release(key)
    preparations = constant_preparations(specs, partial=partial)
    preparations['probe'] = SourcePreparation(('probe-raw',), recipe=TensorPreparation('identity', inputs=('probe-raw',)))
    block = StorageBlock('probe-data', 'probe-selection', 4, device_bytes=4, staging_device_bytes=staging)
    metadata = TensorAxisMetadata(('k',), (1,), np.float32, 'probe-data', 'probe-selection')
    request = SourceReadRequest('probe-reader', 'elemental', (0,), (), metadata)
    blocks = {'probe-raw': block}
    plan = plan_scene_storage(state, blocks, preparations, profile=profile,
                              memory_limit=1 << 30, host_memory_limit=1 << 30)
    # Optional cache admission shrinks with a tighter cap; isolate mandatory
    # owner/staging requirements for the one-byte boundary below.
    plan = replace(plan, kernel_batches=tuple(batches), packed_value_cache_device_bytes=0,
        memory=tuple(replace(p, packed_value_cache_device_bytes=0) for p in plan.memory))
    memory = dense_batch_memory(plan)
    plan = replace(plan, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory),
                   peak_host_bytes=max(p.host_bytes for p in memory))
    return plan, {'probe-raw': request}, preparations, specs


def build_read_cfg(plan, requests, preparations, specs, *, debug=False):
    points, _ = describe_coordinates()
    return cfg_plan.build_cfg_plan(plan, None, points.points, requests, preparations, specs, debug=debug)


@pytest.mark.parametrize('partial', (False, True))
def test_read_staging_peak_includes_the_independently_live_early_cube(partial):
    base, requests, preparations, specs = read_between_children(partial=partial)
    grouped = annotated(base, (describe(base),))
    ordinary = build_read_cfg(base, requests, preparations, specs, debug=True)
    actual = build_read_cfg(grouped, requests, preparations, specs, debug=True)
    template, = actual.templates
    instance, = actual.instances
    events = [event for event in instance.events if event.kind == 'read']
    assert len(events) == 1
    event, = events
    first, later = template.plan.cartesian_reduction_groups[0].batch_action_indices
    assert first[0] < event.action_index < later[0]
    assert template.plan.state.actions[event.action_index].tensor == 'probe'
    assert actual.blocks[event.block].staging_device_bytes == 100_000
    # Independent physical oracle at this read: the first cube remains a
    # final result; the second cube already exists only with grouping. All
    # partials and packed matrices from the grouped launch have been freed.
    # Raw staging dominates all other phases by over an order of magnitude.
    extra_owner = device_allocation_bytes(base.kernel_batches[1].output_bytes)
    assert extra_owner == 512  # One independent four-scalar cube, not four owners.
    assert actual.peak_device_bytes == ordinary.peak_device_bytes + extra_owner
    assert cfg_plan.audit_cfg_plan(actual)['peak_device_bytes'] == actual.peak_device_bytes
    # Enumerate the actual owners at the read independently of P's per-Action
    # arrays: first result cube, remaining second-child source vectors, q,
    # the new raw block, and constant uploads. Partial reuse merges one source.
    uploaded = {prep.recipe.constants[0] for prep in preparations.values() if prep.recipe.kind == 'constant'}
    constant_bytes = sum(device_allocation_bytes(int(np.prod(value.shape))*np.dtype(value.dtype).itemsize)
                         for value in uploaded)
    remaining_sources = (3 if partial else 4) * device_allocation_bytes(257 * np.dtype(np.float32).itemsize)
    read_lower_bound = (512 + remaining_sources + 512 + 512 + device_allocation_bytes(100_000)
                        + constant_bytes + extra_owner)
    # Current load scratch conservatively carries read staging through the
    # following independent source snapshot (512-byte request). Do not call that excess a leak,
    # or claim this fixture isolates only the raw-event branch.
    assert actual.peak_device_bytes == read_lower_bound + 512
    # With partial reuse native grouping falls back, but the stored upper bound
    # deliberately remains at least this same full-child owner allowance.
    assert bool(active_groups(actual, instance)) is (not partial)
    physical = {p.cartesian_reduction_device_bytes for p in template.plan.memory
                if p.action_index == event.action_index}
    assert physical == {extra_owner}
    # Audit accepts an overlarge saved cap. Removing the charge lowers its
    # independently reconstructed requirement, rather than forcing an error.
    damaged_memory = tuple(replace(p, cartesian_reduction_device_bytes=0)
                           if p.action_index == event.action_index else p for p in template.plan.memory)
    damaged = replace(actual, templates=(replace(template, plan=replace(template.plan, memory=damaged_memory)),))
    assert cfg_plan.audit_cfg_plan(damaged)['peak_device_bytes'] == ordinary.peak_device_bytes
    with pytest.raises(ValueError, match='omits device'):
        cfg_plan.audit_cfg_plan(replace(actual, peak_device_bytes=ordinary.peak_device_bytes))


def test_read_peak_cap_rejects_one_byte_below_the_full_early_owner_envelope():
    base, requests, preparations, specs = read_between_children()
    grouped = annotated(base, (describe(base),))
    full = build_read_cfg(grouped, requests, preparations, specs, debug=True)
    exact = replace(grouped, memory_limit=full.peak_device_bytes)
    assert build_read_cfg(exact, requests, preparations, specs).peak_device_bytes == full.peak_device_bytes
    with pytest.raises(MemoryError, match='device bytes'):
        build_read_cfg(replace(grouped, memory_limit=full.peak_device_bytes-1), requests, preparations, specs)
