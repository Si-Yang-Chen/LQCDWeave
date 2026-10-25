"""Independent scalar annotations preserve algebra and all original memory."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    ScalarReductionGroup, scalar_reduction_geometry, describe_scalar_reduction_group,
    scalar_reduction_group_memory, validate_executable_resources, dense_batch_memory,
    DenseGemmGroup, DenseKernelBatch, DenseBatchOperand, DenseBatchResult, parse_axis_pattern,
)
from lqcdweave.contraction.planning.api import ContractionState, CostProfile, plan_scene_storage
from lqcdweave.storage.contraction_plans.api import (
    encode_plan_value, decode_plan_value, save_storage_schedule, load_storage_schedule, PlanArchiveError,
)


def fixture(count=4, *, upfront=True, dtype=np.float32):
    profile = CostProfile()
    specs = {f'{side}{i}': TensorSpec((2, 257), dtype) for i in range(count) for side in ('a', 'b')}
    terms = tuple(ContractionTerm((f'a{i}', f'b{i}'), 'ij,ij->') for i in range(count))
    problem = ContractionProblem(specs, terms, {k: SourceSpec(k, 0, read_bytes=0) for k in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    if upfront:
        for key in specs:
            state = state.load(key, profile, 1 << 30)
    indices = []
    for i in range(count):
        if not upfront:
            for key in (f'a{i}', f'b{i}'):
                state = state.load(key, profile, 1 << 30)
        indices.append(len(state.actions))
        state = state.contract(i, (0, 1), profile, 1 << 30)
        for key in (f'a{i}', f'b{i}'):
            state = state.release(key)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
                              profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    return plan, tuple(indices)


def annotated(plan, groups):
    plan = replace(plan, scalar_reduction_groups=tuple(groups))
    memory = scalar_reduction_group_memory(plan)
    host, device = max(p.host_bytes for p in memory), max(p.device_bytes for p in memory)
    return replace(plan, memory=memory, peak_host_bytes=host, peak_device_bytes=device,
                   feasible=device <= plan.memory_limit and
                   (plan.host_memory_limit is None or host <= plan.host_memory_limit))


def describe(plan, indices):
    return describe_scalar_reduction_group(plan.state.actions, indices, plan.state.catalog,
        sum(plan.state.actions[i].seconds for i in indices) / 2)


def test_geometry_and_independent_early_output_lifetimes_preserve_original_buckets():
    base, indices = fixture()
    group = describe(base, indices)
    geometry = scalar_reduction_geometry(base.state.actions[indices[0]], base.state.catalog)
    assert (geometry.size, geometry.blocks, geometry.allowance, geometry.partial_bytes) == (514, 3, 4116, 24)
    assert group.workspace_bytes == 96
    plan = annotated(base, (group,))
    validate_executable_resources(plan)
    # Independent reference: all four results exist from the physical start,
    # but the original logical trace already counts each at its own contract.
    born, logical = set(), set()
    expected = {}
    names = {base.state.actions[i].tensor for i in indices}
    for i, action in enumerate(base.state.actions):
        if i == indices[0]:
            born.update(names)
        if i in indices:
            logical.add(action.tensor)
        if action.kind == 'release':
            born.discard(action.tensor)
            logical.discard(action.tensor)
        # Independent allocator reference: each 4-byte scalar and the single
        # 96-byte partial request consume one 512-byte allocation unit.
        expected[i] = 512 * len(born - logical) + (512 if i == indices[0] else 0)
    for original, point in zip(base.memory, plan.memory, strict=True):
        required = expected.get(point.action_index, 0) if point.phase == 'contract' else 0
        assert point.scalar_reduction_device_bytes == required
        assert replace(point, scalar_reduction_device_bytes=0) == original
        assert point.device_bytes == original.device_bytes + required
    assert scalar_reduction_group_memory(plan) == plan.memory  # exact construction is idempotent
    assert scalar_reduction_group_memory(plan, subtract=True) == base.memory
    assert plan.compute_seconds == pytest.approx(base.compute_seconds + group.seconds - sum(base.state.actions[i].seconds for i in indices))


def test_read_phase_boundaries_conservative_filtered_groups_and_budget_reserves():
    base, indices = fixture(2)
    group = describe(base, indices)
    points = list(base.memory)
    for index in indices:
        position = next(i for i, p in enumerate(points) if p.action_index == index and p.phase == 'contract')
        points.insert(position, replace(points[position], phase='read:probe'))
    base = replace(base, memory=tuple(points))
    plan = annotated(base, (group,))
    first, last = indices
    observed = {(p.action_index, p.phase): p.scalar_reduction_device_bytes for p in plan.memory}
    assert observed[first, 'read:probe'] == 0
    assert observed[first, 'contract'] == 512 + 512
    assert observed[last, 'read:probe'] == 512
    assert observed[last, 'contract'] == 0
    exact = replace(plan, memory_limit=plan.peak_device_bytes)
    validate_executable_resources(exact)
    tighter = exact.with_reserves(reserved_device_bytes=1)
    assert not tighter.feasible
    assert [p.scalar_reduction_device_bytes for p in tighter.memory] == [p.scalar_reduction_device_bytes for p in exact.memory]
    validate_executable_resources(tighter)
    conservative = replace(plan, scalar_reduction_groups=())
    validate_executable_resources(conservative)
    assert all(p.scalar_reduction_device_bytes == 0 for p in scalar_reduction_group_memory(conservative))
    with pytest.raises(ValueError, match='omits'):
        validate_executable_resources(replace(plan, memory=base.memory))


@pytest.mark.parametrize('indices,workspace,seconds', [
    ((0,), 8, 0), (tuple(range(33)), 8, 0), ((1, 0), 8, 0), ((0, 0), 8, 0),
    ((False, 1), 8, 0), ((-1, 1), 8, 0), ((0, 1), 0, 0), ((0, 1), True, 0),
    ((0, 1), 8, -1), ((0, 1), 8, float('nan')), ((0, 1), 8, float('inf')), ((0, 1), 8, True),
])
def test_record_rejects_invalid_indices_budget_or_cost(indices, workspace, seconds):
    with pytest.raises(ValueError):
        ScalarReductionGroup(indices, workspace, seconds)


def test_invalid_members_geometry_reuse_lifetimes_overlap_and_domains():
    base, indices = fixture()
    group = describe(base, indices[:2])
    plan = annotated(base, (group,))
    mutations = [replace(base.state.actions[indices[0]], reused=True),
                 replace(base.state.actions[indices[0]], core=object()),
                 replace(base.state.actions[indices[0]], equation=parse_axis_pattern('ii,jj->')),
                 replace(base.state.actions[indices[0]], equation=parse_axis_pattern('ij,ij->i'))]
    for changed in mutations:
        actions = list(base.state.actions); actions[indices[0]] = changed
        with pytest.raises(ValueError):
            validate_executable_resources(replace(plan, state=replace(plan.state, actions=tuple(actions))))
    for key, spec in ((base.state.actions[indices[1]].operands[1], TensorSpec((2, 257), np.complex64)),
                      (base.state.actions[indices[1]].operands[1], TensorSpec((257, 2), np.float32))):
        catalog = dict(base.state.catalog); catalog[key] = spec
        with pytest.raises(ValueError):
            validate_executable_resources(replace(plan, state=replace(plan.state, catalog=catalog)))
    with pytest.raises(ValueError, match='multiple'):
        validate_executable_resources(replace(plan, scalar_reduction_groups=(group, group)))
    with pytest.raises(ValueError, match='workspace'):
        validate_executable_resources(replace(plan, scalar_reduction_groups=(replace(group, workspace_bytes=1),)))
    for changes in ({'explicit_resources': True}, {'retention_planned': True}, {'retained_before': ('a0',)}):
        with pytest.raises(NotImplementedError):
            validate_executable_resources(replace(plan, **changes))
    with pytest.raises(NotImplementedError):
        validate_executable_resources(replace(plan, state=replace(plan.state, problem=replace(plan.state.problem, backend='numpy'))))
    future, future_indices = fixture(2, upfront=False)
    with pytest.raises(ValueError, match='resident'):
        validate_executable_resources(annotated(future, (describe(future, future_indices),)))
    # The second member's source has been released before the group start.
    actions = list(base.state.actions)
    actions.insert(indices[0], Action('release', 'a1'))
    shifted = replace(base, state=replace(base.state, actions=tuple(actions)))
    shifted_indices = tuple(i + 1 for i in indices[:2])
    with pytest.raises(ValueError, match='resident'):
        validate_executable_resources(annotated(shifted, (describe(shifted, shifted_indices),)))


def test_terminal_requirement_and_sparse_core_boundaries():
    base, indices = fixture(2)
    plan = annotated(base, (describe(base, indices),))
    result = base.state.actions[indices[0]].tensor
    later = Action('contract', 'later', (result,), parse_axis_pattern('->'))
    with pytest.raises(ValueError, match='terminal'):
        validate_executable_resources(replace(plan, state=replace(base.state, actions=base.state.actions + (later,))))
    for changed in (Action('sparse_bridge', 'unrelated'), Action('contract', 'unrelated', core=object())):
        actions = list(base.state.actions); actions[indices[0] + 1] = changed
        with pytest.raises(ValueError, match='boundary'):
            validate_executable_resources(replace(plan, state=replace(base.state, actions=tuple(actions))))


def test_common_geometry_and_complete_memory_evidence_are_required():
    base, indices = fixture(2)
    plan = annotated(base, (describe(base, indices),))
    member = base.state.actions[indices[1]]
    # Each member remains individually legal, but the group must not mix
    # otherwise valid shapes or precisions in one raw launch.
    for shape, dtype in (((257, 2), np.float32), ((2, 257), np.complex64)):
        catalog = dict(base.state.catalog)
        for key in member.operands:
            catalog[key] = TensorSpec(shape, dtype)
        catalog[member.tensor] = TensorSpec((), dtype)
        scalar_reduction_geometry(member, catalog)
        with pytest.raises(ValueError, match='different normalized geometry'):
            validate_executable_resources(replace(plan, state=replace(base.state, catalog=catalog)))
    with pytest.raises(ValueError, match='unknown Action'):
        describe_scalar_reduction_group(base.state.actions, (indices[0], len(base.state.actions)), base.state.catalog, 0)
    missing = tuple(p for p in plan.memory if not (p.action_index == indices[1] and p.phase == 'contract'))
    with pytest.raises(ValueError, match='omits a member'):
        validate_executable_resources(replace(plan, memory=missing))
    with pytest.raises(ValueError, match='peaks'):
        validate_executable_resources(replace(plan, peak_device_bytes=plan.peak_device_bytes + 1))
    bad = (replace(plan.memory[0], scalar_reduction_device_bytes=-1), *plan.memory[1:])
    with pytest.raises(ValueError, match='nonnegative integer'):
        validate_executable_resources(replace(plan, memory=bad))
    complex_base, complex_indices = fixture(2, dtype=np.complex64)
    complex_group = describe(complex_base, complex_indices)
    assert complex_group.workspace_bytes == 2 * plan.scalar_reduction_groups[0].workspace_bytes
    validate_executable_resources(annotated(complex_base, (complex_group,)))


def test_interleaved_groups_sum_independent_owners_and_preserve_release_points():
    base, indices = fixture()
    first = describe(base, (indices[0], indices[2]))
    second = describe(base, (indices[1], indices[3]))
    plan = annotated(base, (first, second))
    validate_executable_resources(plan)
    by_action = {p.action_index: p.scalar_reduction_device_bytes for p in plan.memory if p.phase == 'contract'}
    assert [by_action[i] for i in indices] == [512 + 512, 512 + 2*512, 512, 0]
    reduced = replace(plan, scalar_reduction_groups=(second,))
    validate_executable_resources(reduced)  # A filtered group may retain full conservative charges.
    assert scalar_reduction_group_memory(reduced) != reduced.memory


def dense_producer_fixture():
    profile = CostProfile()
    specs = {**{f'a{i}': TensorSpec((3, 5), np.float32) for i in range(4)},
             'b0': TensorSpec((5, 2), np.float32), 'b1': TensorSpec((5, 2), np.float32),
             'c': TensorSpec((3, 2), np.float32)}
    terms = tuple(ContractionTerm((f'a{i}', f'b{i//2}', 'c'), 'ik,kj,ij->') if i % 2 == 0 else
                  ContractionTerm((f'a{i}', f'b{i//2}'), 'ik,kj->ij') for i in range(4))
    problem = ContractionProblem(specs, terms, {k: SourceSpec(k, 0, read_bytes=0) for k in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 1 << 30)
    children, scalar_indices = [], []
    for i in (0, 2):
        child = []
        for term in (i, i + 1):
            child.append(len(state.actions)); state = state.contract(term, (0, 1), profile, 1 << 30)
        children.append(tuple(child))
        scalar_indices.append(len(state.actions)); state = state.contract(i, (0, 1), profile, 1 << 30)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
                              profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    batches = tuple(DenseKernelBatch(child,
        tuple(DenseBatchOperand(f'a{i}', (0, 1)) for i in (2*g, 2*g+1)),
        (DenseBatchOperand(f'b{g}', (0, 1)),), 1, 1, 1, 0, 1,
        tuple(DenseBatchResult(state.actions[index].tensor, ((3*i, 3*i+3), (0, 2)))
              for i, index in enumerate(child)),
        sum(state.actions[index].seconds for index in child), 160, 48, 'float32')
        for g, child in enumerate(children))
    gemm = DenseGemmGroup(batches, (6, 5, 2), sum(b.seconds for b in batches))
    plan = replace(plan, kernel_batches=(gemm,))
    memory = dense_batch_memory(plan)
    return replace(plan, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory)), tuple(scalar_indices)


def test_existing_dense_gemm_outputs_are_available_at_physical_first_member():
    base, indices = dense_producer_fixture()
    plan = annotated(base, (describe(base, indices),))
    validate_executable_resources(plan)
    assert plan.kernel_batches == base.kernel_batches
    assert all(a.retained_device_bytes == b.retained_device_bytes and
               a.workspace_device_bytes == b.workspace_device_bytes for a, b in zip(base.memory, plan.memory))
    # Without GEMM co-submission, the second producer is still in the future.
    with pytest.raises(ValueError, match='resident'):
        validate_executable_resources(replace(plan, kernel_batches=()))
    with pytest.raises(ValueError, match='multiple'):
        validate_executable_resources(replace(plan, scalar_reduction_groups=(
            ScalarReductionGroup(base.kernel_batches[0].action_indices[:2], 16, 0),)))


def test_reused_nonmember_handoff_is_available_only_at_its_original_position():
    base, indices = dense_producer_fixture()
    plan = annotated(base, (describe(base, indices),))
    actions = list(base.state.actions)
    handoff = Action('contract', 'a0', ('historical_left', 'historical_right'),
                     parse_axis_pattern('ik,kj->ij'), reused=True)
    actions[0] = handoff
    validate_executable_resources(replace(plan, state=replace(base.state, actions=tuple(actions))))
    # Marking a paid dense member reused is still forbidden.
    member = base.kernel_batches[0].action_indices[0]
    actions[member] = replace(actions[member], reused=True)
    with pytest.raises(ValueError, match='paid'):
        validate_executable_resources(replace(plan, state=replace(base.state, actions=tuple(actions))))
    # A reused marker later in the stream cannot provide initial residency.
    base, indices = fixture(2, upfront=False)
    actions = list(base.state.actions)
    late = next(i for i, a in enumerate(actions) if a.tensor == 'a1' and a.kind == 'load')
    actions[late] = Action('contract', 'a1', ('historical_left', 'historical_right'),
                          parse_axis_pattern('ik,kj->ij'), reused=True)
    base = replace(base, state=replace(base.state, actions=tuple(actions)))
    with pytest.raises(ValueError, match='resident'):
        validate_executable_resources(annotated(base, (describe(base, indices),)))


@pytest.mark.parametrize('count', (2, 8, 32))
def test_record_codec_old_defaults_and_full_archive_roundtrip(tmp_path, count):
    base, indices = fixture(count)
    plan = annotated(base, (describe(base, indices),))
    assert decode_plan_value(encode_plan_value(plan)) == plan
    encoded = encode_plan_value(base)
    def remove_new_fields(value):
        if isinstance(value, dict):
            value.pop('scalar_reduction_groups', None)
            value.pop('scalar_reduction_device_bytes', None)
            for item in value.values(): remove_new_fields(item)
        elif isinstance(value, list):
            for item in value: remove_new_fields(item)
    remove_new_fields(encoded)
    assert decode_plan_value(encoded) == base
    record = encode_plan_value(plan.scalar_reduction_groups[0])
    invalid = copy.deepcopy(record); invalid['value']['workspace_bytes'] = -1
    with pytest.raises(PlanArchiveError): decode_plan_value(invalid)
    shapes = {key: list(spec.shape) for key, spec in base.state.problem.specs.items()}
    hardware = {'backend': 'cupy', 'memory': 1 << 30}
    path = tmp_path / 'scalar-plan.json'
    save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
    restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
        expected=(base.state.problem,), blocks={}, preparations=base.preparations)
    assert restored[0].scalar_reduction_groups == plan.scalar_reduction_groups
    assert restored[0].memory == plan.memory
    validate_executable_resources(restored[0])
    with pytest.raises(ValueError, match='workspace'):
        save_storage_schedule(tmp_path / 'invalid.json',
            (replace(plan, scalar_reduction_groups=(replace(plan.scalar_reduction_groups[0], workspace_bytes=1),)),),
            shapes=shapes, hardware=hardware)
