"""Cartesian grouping keeps original budgets and independent cube lifetimes."""
from dataclasses import replace
import copy
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    DenseKernelBatch, DenseBatchOperand, DenseBatchResult, DenseGemmGroup,
    CartesianReductionGroup, cartesian_reduction_geometry, describe_cartesian_reduction_group,
    cartesian_reduction_group_memory, validate_cartesian_reduction_groups,
    cartesian_reduction_group_workspace, device_allocation_bytes,
    validate_executable_resources, dense_batch_memory, parse_axis_pattern,
    MemoryPoint,
)
from lqcdweave.contraction.planning.api import ContractionState, CostProfile, plan_scene_storage, describe_dense_batch
from lqcdweave.contraction.planning.allocation_rounding import account_allocation_rounding
from lqcdweave.storage.contraction_plans.api import (
    encode_plan_value, decode_plan_value, save_storage_schedule, load_storage_schedule, PlanArchiveError,
)


def fixture(count=4, *, upfront=True, dtype=np.float32, k=257, m=2, n=2):
    profile = CostProfile()
    specs = {f'{side}{b}:{i}': TensorSpec((k,), dtype) for b in range(count)
             for side, size in (('a', m), ('b', n)) for i in range(size)}
    pairs = [(f'a{b}:{i}', f'b{b}:{j}') for b in range(count) for i in range(m) for j in range(n)]
    problem = ContractionProblem(specs, tuple(ContractionTerm(pair, 'k,k->') for pair in pairs),
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    if upfront:
        for key in specs:
            state = state.load(key, profile, 1 << 30)
    batches = []
    item = np.dtype(dtype).itemsize
    for b in range(count):
        inputs = [key for key in specs if key.startswith((f'a{b}:', f'b{b}:'))]
        if not upfront:
            for key in inputs:
                state = state.load(key, profile, 1 << 30)
        indices = []
        for term in range(b*m*n, (b+1)*m*n):
            indices.append(len(state.actions))
            state = state.contract(term, (0, 1), profile, 1 << 30)
        batches.append(DenseKernelBatch(tuple(indices),
            tuple(DenseBatchOperand(f'a{b}:{i}', (0,)) for i in range(m)),
            tuple(DenseBatchOperand(f'b{b}:{j}', (0,)) for j in range(n)), 1, 1, 1, 0, 1,
            tuple(DenseBatchResult(state.actions[index].tensor, ((i, i+1), (j, j+1)), (0, 1))
                  for index, (i, j) in zip(indices, ((i, j) for i in range(m) for j in range(n)))),
            sum(state.actions[i].seconds for i in indices)/2, (m+n)*k*item, m*n*item,
            np.dtype(dtype).name, stacked_sides=(True, True)))
        for key in inputs:
            state = state.release(key)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
                              profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    plan = replace(plan, kernel_batches=tuple(batches))
    memory = dense_batch_memory(plan)
    return account_allocation_rounding(replace(plan, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory)))


def describe(plan, children=None):
    children = plan.kernel_batches if children is None else children
    return describe_cartesian_reduction_group(plan.kernel_batches,
        tuple(child.action_indices for child in children), plan.state.catalog,
        sum(child.seconds for child in children)/2)


def prepared_fixture(*, k=129, dtype=np.float32):
    """Two real producer-fed 1×3 batches, each borrowing one prepared side."""
    profile = CostProfile()
    specs = {**{f'x{b}': TensorSpec((k, 2), dtype) for b in range(2)},
             **{f'z{b}': TensorSpec((2,), dtype) for b in range(2)},
             **{f'b{b}:{i}': TensorSpec((k,), dtype) for b in range(2) for i in range(3)}}
    terms = tuple(ContractionTerm((f'x{b}', f'z{b}', f'b{b}:{i}'), 'ik,k,i->')
                  for b in range(2) for i in range(3))
    problem = ContractionProblem(specs, terms,
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 1 << 30)
    prepared = {}
    # Applying each shared producer updates all three dependent pending terms.
    for term in (0, 3):
        state = state.contract(term, (0, 1), profile, 1 << 30)
        prepared[state.actions[-1].tensor] = (0,)
    batches = []
    for b in range(2):
        indices = []
        for term in range(3*b, 3*(b+1)):
            indices.append(len(state.actions))
            state = state.contract(term, (0, 1), profile, 1 << 30)
        batch = describe_dense_batch(state.actions, tuple(indices), state.catalog, profile,
                                     backend='cupy', prepared_inputs=prepared)
        assert batch is not None and sum(batch.prepared_sides) == 1
        batches.append(batch)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
                             profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    plan = replace(plan, kernel_batches=tuple(batches))
    memory = dense_batch_memory(plan)
    return account_allocation_rounding(replace(plan, memory=memory,
                                               peak_device_bytes=max(p.device_bytes for p in memory)))


def annotated(plan, groups):
    plan = replace(plan, cartesian_reduction_groups=tuple(groups))
    memory = cartesian_reduction_group_memory(plan)
    host, device = max(p.host_bytes for p in memory), max(p.device_bytes for p in memory)
    return replace(plan, memory=memory, peak_host_bytes=host, peak_device_bytes=device,
        feasible=device <= plan.memory_limit and (plan.host_memory_limit is None or host <= plan.host_memory_limit))


def test_original_wide_geometry_and_exact_independent_owner_oracle():
    base = fixture(3)
    group = describe(base)
    geometry = cartesian_reduction_geometry(base.kernel_batches[0], base.state.catalog)
    assert (geometry.shapes, geometry.output_shape, geometry.allowance, geometry.blocks,
            geometry.partial_bytes) == (((2, 257), (2, 257)), (2, 2), 2060, 2, 64)
    assert group.workspace_bytes == 192
    plan = annotated(base, (group,))
    validate_executable_resources(plan)
    # Independent point oracle tracks cube owners, not logical scalar views.
    # Original scratch and packing are preserved even at displaced child starts.
    physical, original = set(), set()
    starts = {batch.action_indices[0]: b for b, batch in enumerate(base.kernel_batches)}
    expected = {}
    for i, action in enumerate(base.state.actions):
        if i == group.action_indices[0]:
            physical.update(range(3))
        if i in starts:
            original.add(starts[i])
        amount = sum(device_allocation_bytes(base.kernel_batches[b].output_bytes) for b in physical - original)
        if i == group.action_indices[0]:
            # One partial allocation, six separately packed sides. Subtract
            # only the first child's already budgeted packing payload.
            amount += 512 + 6 * 2560 - base.kernel_batches[0].packing_bytes
        expected[i] = amount
    for original_point, point in zip(base.memory, plan.memory, strict=True):
        assert point.cartesian_reduction_device_bytes == expected.get(point.action_index, 0)
        assert replace(point, cartesian_reduction_device_bytes=0) == original_point
        assert point.device_bytes == original_point.device_bytes + point.cartesian_reduction_device_bytes
    assert cartesian_reduction_group_memory(plan) == plan.memory
    assert cartesian_reduction_group_memory(plan, subtract=True) == base.memory
    assert plan.compute_seconds == pytest.approx(base.compute_seconds + group.seconds - sum(b.seconds for b in base.kernel_batches))


def test_reads_overlapping_groups_filtered_conservative_budget_and_reserves():
    base = fixture()
    points = list(base.memory)
    for batch in base.kernel_batches:
        position = next(i for i, p in enumerate(points) if p.action_index == batch.action_indices[0] and p.phase == 'contract')
        points.insert(position, replace(points[position], phase='read:probe'))
    base = replace(base, memory=tuple(points))
    a = describe(base, base.kernel_batches[::2])
    b = describe(base, base.kernel_batches[1::2])
    plan = annotated(base, (a, b))
    validate_executable_resources(plan)
    firsts = [batch.action_indices[0] for batch in base.kernel_batches]
    amounts = {(p.action_index, p.phase): p.cartesian_reduction_device_bytes for p in plan.memory}
    assert [amounts[i, 'read:probe'] for i in firsts] == [0, 512, 1024, 512]
    assert [amounts[i, 'contract'] for i in firsts] == [7152, 7664, 512, 0]
    exact = replace(plan, memory_limit=plan.peak_device_bytes)
    validate_executable_resources(exact)
    over = exact.with_reserves(reserved_device_bytes=1)
    assert not over.feasible
    assert [p.cartesian_reduction_device_bytes for p in over.memory] == [p.cartesian_reduction_device_bytes for p in plan.memory]
    validate_executable_resources(over)
    validate_executable_resources(replace(plan, cartesian_reduction_groups=(b,)))
    validate_executable_resources(replace(plan, cartesian_reduction_groups=()))
    with pytest.raises(ValueError, match='omits'):
        validate_cartesian_reduction_groups(replace(plan, memory=base.memory))
    with pytest.raises(ValueError, match='absent'):
        cartesian_reduction_group_memory(replace(plan, memory=base.memory), subtract=True)


@pytest.mark.parametrize('indices,workspace,seconds', [
    (((0, 1),), 8, 0), (tuple((2*i, 2*i+1) for i in range(9)), 8, 0),
    (((0,), (1, 2)), 8, 0), (((0, 1), (1, 2)), 8, 0), (((2, 3), (0, 1)), 8, 0),
    (((0, 0), (2, 3)), 8, 0), (((False, 1), (2, 3)), 8, 0), (((-1, 1), (2, 3)), 8, 0),
    (((0, 1), (2, 3)), 0, 0), (((0, 1), (2, 3)), True, 0),
    (((0, 1), (2, 3)), 8, -1), (((0, 1), (2, 3)), 8, float('nan')),
    (((0, 1), (2, 3)), 8, float('inf')), (((0, 1), (2, 3)), 8, True),
])
def test_record_rejects_ambiguous_members_and_cost(indices, workspace, seconds):
    with pytest.raises(ValueError):
        CartesianReductionGroup(indices, workspace, seconds)


def test_record_keeps_complete_child_identity_without_extra_interval_restriction():
    group = CartesianReductionGroup(((0, 3), (1, 2)), 8, 0)
    assert group.action_indices == (0, 1, 2, 3)
    assert group.batch_action_indices == ((0, 3), (1, 2))


@pytest.mark.parametrize('dtype', (np.float32, np.complex64))
def test_native_geometry_preserves_child_block_count(dtype):
    base = fixture(2, dtype=dtype, m=2, n=3)
    group = describe(base)
    geometry = cartesian_reduction_geometry(base.kernel_batches[0], base.state.catalog)
    item = np.dtype(dtype).itemsize
    expected = min(1024, 2, (515*item-6*item)//(6*2*item))
    assert geometry.blocks == expected
    assert group.workspace_bytes == 2*6*expected*2*item
    validate_executable_resources(annotated(base, (group,)))


def test_geometry_excludes_other_precision_unstacked_small_k_and_nonadmitted_grid():
    base = fixture(2)
    batch = base.kernel_batches[0]
    for changed in (replace(batch, stacked_sides=(False, True)), replace(batch, packing_dtype='float64')):
        with pytest.raises(ValueError): cartesian_reduction_geometry(changed, base.state.catalog)
    for dtype, k in ((np.float64, 257), (np.float32, 1)):
        bad = fixture(2, dtype=dtype, k=k)
        with pytest.raises(ValueError): describe(bad)
    # Large output versus tiny original allowance must not shrink blocks to 0
    # and silently switch accumulation strategy.
    bad = fixture(2, k=2, m=2, n=3)
    with pytest.raises(ValueError, match='wide block'):
        describe(bad)
    catalog = dict(base.state.catalog)
    catalog[batch.left[0].tensor] = TensorSpec((257,), np.complex64)
    with pytest.raises(ValueError, match='dtype'):
        cartesian_reduction_geometry(batch, catalog)
    catalog[batch.left[0].tensor] = TensorSpec((258,), np.float32)
    with pytest.raises(ValueError, match='geometry'):
        cartesian_reduction_geometry(batch, catalog)


def test_complete_top_level_only_no_gemm_children_overlap_or_reused_members():
    base = fixture(2)
    group = describe(base)
    plan = annotated(base, (group,))
    gemm = DenseGemmGroup(base.kernel_batches, (2, 257, 2), 0)
    with pytest.raises(ValueError, match='complete top-level'):
        describe_cartesian_reduction_group((gemm,), group.batch_action_indices, base.state.catalog, 0)
    with pytest.raises(ValueError, match='complete top-level'):
        describe_cartesian_reduction_group(base.kernel_batches,
            (group.batch_action_indices[0][:-1], group.batch_action_indices[1]), base.state.catalog, 0)
    with pytest.raises(ValueError, match='overlapping'):
        validate_cartesian_reduction_groups(replace(plan, cartesian_reduction_groups=(group, group)))
    with pytest.raises(ValueError, match='workspace'):
        validate_cartesian_reduction_groups(replace(plan, cartesian_reduction_groups=(replace(group, workspace_bytes=1),)))
    actions = list(base.state.actions)
    actions[group.action_indices[0]] = replace(actions[group.action_indices[0]], reused=True)
    with pytest.raises(ValueError, match='reused'):
        validate_cartesian_reduction_groups(replace(plan, state=replace(base.state, actions=tuple(actions))))
    from lqcdweave.contracts.correlator.api import ScalarReductionGroup
    with pytest.raises(ValueError, match='overlapping'):
        validate_cartesian_reduction_groups(replace(plan,
            scalar_reduction_groups=(ScalarReductionGroup(group.action_indices[:2], 8, 0),)))


def test_future_load_handoff_release_and_sparse_boundaries_are_not_guessed_ready():
    future = fixture(2, upfront=False)
    with pytest.raises(ValueError, match='resident'):
        validate_executable_resources(annotated(future, (describe(future),)))
    base = fixture(2)
    plan = annotated(base, (describe(base),))
    actions = list(base.state.actions)
    # Reused nonmembers are explicit current-owner handoffs; old operands need
    # not be live. The same marker in the future cannot make input ready now.
    actions[0] = Action('contract', actions[0].tensor, ('old_left', 'old_right'),
                        parse_axis_pattern('k,k->k'), reused=True)
    validate_executable_resources(replace(plan, state=replace(base.state, actions=tuple(actions))))
    first = base.kernel_batches[0].action_indices[0]
    boundary = base.kernel_batches[0].action_indices[-1] + 1
    for changed in (Action('sparse_bridge', 'unrelated'), Action('contract', 'unrelated', core=object())):
        actions = list(base.state.actions); actions[boundary] = changed
        with pytest.raises(ValueError, match='boundary'):
            validate_cartesian_reduction_groups(replace(plan, state=replace(base.state, actions=tuple(actions))))
    actions = list(base.state.actions)
    key = base.kernel_batches[1].left[0].tensor
    index = next(i for i, a in enumerate(actions) if a.kind == 'load' and a.tensor == key)
    actions[index] = Action('release', key)
    with pytest.raises(ValueError, match='resident'):
        validate_cartesian_reduction_groups(replace(plan, state=replace(base.state, actions=tuple(actions))))
    assert first > 0


def test_domains_memory_phases_bucket_values_and_peak_evidence():
    base = fixture(2); group = describe(base); plan = annotated(base, (group,))
    for changed in ({'explicit_resources': True}, {'retention_planned': True}, {'retained_before': ('a0:0',)}):
        with pytest.raises(NotImplementedError): validate_cartesian_reduction_groups(replace(plan, **changed))
    with pytest.raises(NotImplementedError):
        validate_cartesian_reduction_groups(replace(plan, state=replace(base.state,
            problem=replace(base.state.problem, backend='numpy'))))
    first = base.kernel_batches[1].action_indices[0]
    points = tuple(p for p in plan.memory if not (p.action_index == first and p.phase == 'contract'))
    with pytest.raises(ValueError, match='omits an original'):
        validate_cartesian_reduction_groups(replace(plan, memory=points))
    with pytest.raises(ValueError, match='peaks'):
        validate_cartesian_reduction_groups(replace(plan, peak_device_bytes=plan.peak_device_bytes+1))
    with pytest.raises(ValueError, match='feasibility'):
        validate_cartesian_reduction_groups(replace(plan, feasible=False))
    for amount in (-1, True, 0.5):
        with pytest.raises(ValueError, match='nonnegative integer'):
            validate_cartesian_reduction_groups(replace(plan, memory=(
                replace(plan.memory[0], cartesian_reduction_device_bytes=amount), *plan.memory[1:])))


def test_scene_codec_defaults_new_record_and_invalid_archive(tmp_path):
    base = fixture(2); plan = annotated(base, (describe(base),))
    assert decode_plan_value(encode_plan_value(plan)) == plan
    encoded = encode_plan_value(base)
    def remove_new(value):
        if isinstance(value, dict):
            value.pop('cartesian_reduction_groups', None)
            value.pop('cartesian_reduction_device_bytes', None)
            for item in value.values(): remove_new(item)
        elif isinstance(value, list):
            for item in value: remove_new(item)
    remove_new(encoded)
    assert decode_plan_value(encoded) == base
    invalid = copy.deepcopy(encode_plan_value(plan.cartesian_reduction_groups[0]))
    invalid['value']['workspace_bytes'] = -1
    with pytest.raises(PlanArchiveError): decode_plan_value(invalid)
    shapes = {key: list(spec.shape) for key, spec in base.state.problem.specs.items()}
    hardware = {'backend': 'cupy', 'memory': 1 << 30}
    path = tmp_path / 'cartesian-plan.json'
    save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
    restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
        expected=(base.state.problem,), blocks={}, preparations=base.preparations)
    assert restored[0].cartesian_reduction_groups == plan.cartesian_reduction_groups
    assert restored[0].memory == plan.memory
    validate_executable_resources(restored[0])
    with pytest.raises(ValueError, match='workspace'):
        save_storage_schedule(tmp_path/'invalid.json',
            (replace(plan, cartesian_reduction_groups=(replace(plan.cartesian_reduction_groups[0], workspace_bytes=1),)),),
            shapes=shapes, hardware=hardware)


@pytest.mark.parametrize('stream', (False, True))
def test_cfg_archive_preserves_remapped_children_and_exact_bucket(tmp_path, stream):
    from lqcdweave.contracts.correlator.api import ArrayConstant, TensorPreparation, describe_coordinates
    from lqcdweave.contraction.planning import cfg_plan
    from lqcdweave.storage.contraction_plans import cfg_stream
    from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan
    base = fixture(2)
    plan = annotated(base, (describe(base),))
    specs = plan.state.problem.specs
    preparations = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
        ArrayConstant.from_array(np.full(spec.shape, number+1, spec.dtype)),)))
        for number, (key, spec) in enumerate(specs.items())}
    coordinates, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}])
    cfg = cfg_plan.build_cfg_plan(plan, None, coordinates.points, {}, preparations, specs, debug=True)
    template = cfg.templates[0].plan
    kept = [i for i, action in enumerate(plan.state.actions) if action.kind != 'release']
    positions = {old: new for new, old in enumerate(kept)}
    assert template.cartesian_reduction_groups[0].batch_action_indices == tuple(
        tuple(positions[i] for i in child) for child in plan.cartesian_reduction_groups[0].batch_action_indices)
    # Output snapshots retain the terminal memory boundary after the last
    # Action, even when cfg removes release Actions from the template.
    positions[len(plan.state.actions)] = len(template.state.actions)
    assert plan.memory[-1].phase == 'output_copy'
    assert template.memory[-1] == replace(plan.memory[-1],
        action_index=len(template.state.actions))
    assert [p.cartesian_reduction_device_bytes for p in template.memory] == [
        p.cartesian_reduction_device_bytes for p in plan.memory if p.action_index in positions]
    path = tmp_path / ('cfg-stream.json.gz' if stream else 'cfg.json')
    with patch.object(cfg_stream, '_STREAM_ITEMS', 0 if stream else 10**9):
        save_cfg_plan(path, cfg, debug=True)
    with patch.object(cfg_plan, 'build_cfg_plan', side_effect=AssertionError('fixed archive cannot replan')):
        restored = load_cfg_plan(path)
    assert restored == cfg
    assert cfg_plan.audit_cfg_plan(restored)['peak_device_bytes'] == cfg.peak_device_bytes
    assert restored.templates[0].plan.cartesian_reduction_groups == template.cartesian_reduction_groups
    assert restored.templates[0].plan.memory == template.memory


def test_different_valid_children_cannot_share_one_group_geometry():
    base = fixture(2)
    changed = fixture(2, dtype=np.complex64)
    # Both complete children remain valid records. Their catalog dtypes must
    # agree with each other, rather than only their own packing dtype.
    catalog = dict(base.state.catalog)
    second = changed.kernel_batches[1]
    for key in (o.tensor for o in (*second.left, *second.right)):
        catalog[key] = changed.state.catalog[key]
    with pytest.raises(ValueError, match='different native'):
        describe_cartesian_reduction_group((base.kernel_batches[0], second),
            (base.kernel_batches[0].action_indices, second.action_indices), catalog, 0)


def test_input_dependent_on_a_future_nonmember_producer_is_rejected():
    base = fixture(2)
    plan = annotated(base, (describe(base),))
    key = base.kernel_batches[1].left[0].tensor
    actions = list(base.state.actions)
    source_index = next(i for i, a in enumerate(actions) if a.kind == 'load' and a.tensor == key)
    boundary = base.kernel_batches[0].action_indices[-1] + 1
    actions[source_index] = Action('release', key)
    # This is an explicit retained-owner handoff after the proposed group
    # start, so historical operands cannot establish early availability.
    actions[boundary] = Action('contract', key, ('old',), parse_axis_pattern('k->k'), reused=True)
    with pytest.raises(ValueError, match='resident'):
        validate_cartesian_reduction_groups(replace(plan, state=replace(base.state, actions=tuple(actions))))


def test_existing_gemm_physical_publication_supplies_later_child_inputs():
    base = fixture(2)
    catalog = dict(base.state.catalog)
    catalog.update({f'x{i}': TensorSpec((257, 2), np.float32) for i in range(4)})
    catalog.update({f'z{i}': TensorSpec((2,), np.float32) for i in range(2)})
    sources = [f'x{i}' for i in range(4)] + ['z0', 'z1'] + [f'b{b}:{i}' for b in range(2) for i in range(2)]
    actions = [Action('load', key) for key in sources]
    producers, children = [], []
    for b, original in enumerate(base.kernel_batches):
        indices = []
        for i in range(2):
            indices.append(len(actions))
            actions.append(Action('contract', f'a{b}:{i}', (f'x{2*b+i}', f'z{b}'), parse_axis_pattern('ik,k->i')))
        producers.append(DenseKernelBatch(tuple(indices),
            tuple(DenseBatchOperand(f'x{2*b+i}', (0, 1)) for i in range(2)),
            (DenseBatchOperand(f'z{b}', (0,)),), 1, 1, 1, 0, 1,
            tuple(DenseBatchResult(f'a{b}:{i}', ((257*i, 257*(i+1)), (0, 1)), (1,)) for i in range(2)),
            0, 4120, 2056, 'float32', stacked_sides=(False, True)))
        indices = tuple(range(len(actions), len(actions)+4))
        actions.extend(base.state.actions[i] for i in original.action_indices)
        children.append(replace(original, action_indices=indices))
    gemm = DenseGemmGroup(tuple(producers), (514, 2, 1), 0)
    logical, points = 0, []
    for index, action in enumerate(actions):
        spec = catalog[action.tensor]
        logical += int(np.prod(spec.shape)) * spec.dtype.itemsize
        points.append(MemoryPoint(index, action.kind, 0, 0, 0, logical, 0,
                                  100_000 if action.kind == 'contract' else 0))
    points.append(MemoryPoint(len(actions), 'outputs', 0, 0, 0, logical, 0, 0))
    base = replace(base, state=replace(base.state, actions=tuple(actions), catalog=catalog),
                   kernel_batches=(gemm, *children), memory=tuple(points))
    memory = dense_batch_memory(base)
    base = account_allocation_rounding(replace(base, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory)))
    plan = annotated(base, (describe(base, children),))
    validate_executable_resources(plan)
    # Retaining the same catalog names/shapes cannot replace physical timing:
    # without GEMM grouping, the second producer still lies after group start.
    with pytest.raises(ValueError, match='resident'):
        validate_cartesian_reduction_groups(replace(plan, kernel_batches=(*producers, *children)))


def test_early_cubes_do_not_merge_owners_or_end_at_another_child_release():
    base = fixture(2)
    actions = list(base.state.actions)
    first, second = base.kernel_batches
    # Use the existing four release positions between child births to release
    # each first cube view. The second cube must continue to be charged whole.
    positions = range(first.action_indices[-1]+1, second.action_indices[0])
    for index, result in zip(positions, first.results, strict=True):
        actions[index] = Action('release', result.tensor)
    base = replace(base, state=replace(base.state, actions=tuple(actions)))
    # Reconstruct base dense ownership after changing the release positions.
    old = fixture(2)
    memory = dense_batch_memory(old, subtract=True)
    live, logical = set(), {}
    for index, action in enumerate(actions):
        if action.kind in ('load', 'contract'):
            live.add(action.tensor)
        elif action.kind == 'release':
            live.discard(action.tensor)
        logical[index] = sum(int(np.prod(base.state.catalog[key].shape)) * base.state.catalog[key].dtype.itemsize
                             for key in live)
    logical[len(actions)] = logical[len(actions)-1]
    memory = tuple(replace(p, logical_device_bytes=logical[p.action_index]) for p in memory)
    base = replace(base, memory=memory)
    memory = dense_batch_memory(base)
    base = account_allocation_rounding(replace(base, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory)))
    plan = annotated(base, (describe(base),))
    validate_executable_resources(plan)
    for point in plan.memory:
        if first.action_indices[-1] < point.action_index < second.action_indices[0]:
            assert point.cartesian_reduction_device_bytes == device_allocation_bytes(second.output_bytes)
    at_second = next(p for p in plan.memory if p.action_index == second.action_indices[0])
    assert at_second.cartesian_reduction_device_bytes == 0


@pytest.mark.parametrize('dtype', (np.float32, np.complex64))
@pytest.mark.parametrize('k', (128, 129))
@pytest.mark.parametrize('prepared', (False, True))
def test_each_packing_side_and_owner_rounds_separately_with_prepared_sides_excluded(dtype, k, prepared):
    base = prepared_fixture(k=k, dtype=dtype) if prepared else fixture(2, m=2, n=3, k=k, dtype=dtype)
    group = describe(base)
    item = np.dtype(dtype).itemsize
    if prepared:
        # The single producer row is borrowed, so each child allocates only
        # its other 3×K packed side. No padding for the borrowed side is added.
        owner_payloads = [3*k*item, 3*k*item]
    else:
        owner_payloads = [2*k*item, 3*k*item, 2*k*item, 3*k*item]
    rounded = sum(((size+511)//512)*512 for size in owner_payloads)
    partial = ((group.workspace_bytes+511)//512)*512
    required = partial + rounded - base.kernel_batches[0].packing_bytes
    assert cartesian_reduction_group_workspace(base.kernel_batches, base.state.catalog, group.workspace_bytes) == required
    plan = annotated(base, (group,))
    validate_executable_resources(plan)
    first, second = (batch.action_indices[0] for batch in base.kernel_batches)
    start = next(p for p in plan.memory if p.action_index == first and p.phase == 'contract')
    assert start.cartesian_reduction_device_bytes == required + 512
    if k == 128:
        assert all(size % 512 == 0 for size in owner_payloads)
    else:
        assert all(size % 512 for size in owner_payloads)
    assert all(p.cartesian_reduction_device_bytes == 0 for p in plan.memory if p.action_index >= second)
    # Existing payload and allocator rounding remain byte-for-byte intact.
    assert all(replace(new, cartesian_reduction_device_bytes=0) == old
               for old, new in zip(base.memory, plan.memory, strict=True))


def test_early_full_cube_rounding_hands_off_to_original_owner_padding_at_birth():
    base = fixture(2, m=2, n=3, k=129)
    first, second = base.kernel_batches
    start = second.action_indices[0]
    plan = annotated(base, (describe(base),))
    before = max((p for p in plan.memory if p.action_index < start), key=lambda p: p.action_index)
    born = next(p for p in plan.memory if p.action_index == start)
    assert before.cartesian_reduction_device_bytes == 512
    assert born.cartesian_reduction_device_bytes == 0
    # At each original birth only the cube's padding is newly retained;
    # the six result views must not become six independent 512-byte owners.
    padding = device_allocation_bytes(second.output_bytes) - second.output_bytes
    no_child = replace(base, kernel_batches=(first,))
    without = account_allocation_rounding(no_child)
    after_child = next(p for p in base.memory if p.action_index > second.action_indices[-1])
    independent = next(p for p in without.memory if p.action_index == after_child.action_index and p.phase == after_child.phase)
    assert after_child.allocation_rounding_device_bytes - independent.allocation_rounding_device_bytes == padding - 6*(512-4)
