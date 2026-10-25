"""Lower already selected thin factor graphs to one explicit wide Action."""
from collections import Counter
from dataclasses import replace
import hashlib
import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, TensorSpec, FactorizedWideReduction,
)
from .state import ContractionState


def factorized_result_name(inputs, core):
    identity = ((inputs, core) if core.tile_elements is None else
                (inputs, replace(core, equation=core.equation.normalized()), core.tile_elements))
    if core.output_dtype is not None:
        identity = (*identity, ('output_dtype', core.output_dtype))
    if core.gram_strategy != 'reduction':
        identity = (*identity, ('gram_strategy',core.gram_strategy))
    return 'factorized-wide:' + hashlib.blake2b(repr(identity).encode(), digest_size=32).hexdigest()


def describe_factorized_action(state, scene, positions, large_operands, *, profile, output_order,
                               tile_elements=None, output_dtype='auto',gram_strategy='reduction'):
    """Independently prove algebra, precision boundary, cost and workspace."""
    if (state.problem.sparse_bridges or state.problem.baryon_joins
            or len(positions) not in ((4,) if tile_elements is None else (3,4))):
        raise ValueError('thin wide Actions require three/four ordinary dense factors')
    layout = list(state._contract_layout(scene, positions, allow_nary=True))
    selected, retained, dimensions, _, dtype = layout[:5]
    if set(output_order) != set(retained) or len(output_order) != len(retained):
        raise ValueError('thin wide output must preserve the selected retained axes')
    if dtype != np.dtype('complex64') or any(state.catalog[f.tensor].dtype != dtype for f in selected):
        raise ValueError('thin wide factors must all have the logical complex64 dtype')
    if any(f.tensor not in state.loaded for f in selected):
        raise ValueError('thin wide factors must be resident')
    equation = AxisPattern(tuple(f.labels for f in selected), tuple(output_order))
    if output_dtype == 'auto':
        # A terminal value is still internal: independent loops may cancel or
        # multiply later in R. Only the published field owns the final cast.
        output_dtype = ('complex128' if tile_elements is not None and len(positions) == 3
                        and len(positions) == len(state.pending[scene].factors) else None)
    core = FactorizedWideReduction(equation, tuple(state.catalog[f.tensor].shape for f in selected), large_operands,
                                   tile_elements=tile_elements, output_dtype=output_dtype,gram_strategy=gram_strategy)
    term = state.pending[scene]
    before = {label for f in term.factors for label in f.labels}
    after = set(output_order) | {label for i, f in enumerate(term.factors) if i not in positions for label in f.labels}
    crossing = tuple((trigger, dependent) for trigger, dependent in state.problem.terms[scene].sum_barriers
                     if before.intersection(trigger) and not after.intersection(trigger))
    if (not crossing or any(after.intersection(dependent) for _, dependent in crossing)
            or not set(core.reduced_axes) <= set().union(*(trigger for trigger, _ in crossing))):
        raise ValueError('thin wide Action must close an explicitly declared wide sum boundary')
    inputs = tuple(f.tensor for f in selected)
    key = factorized_result_name(inputs, core)
    reused = key in state.loaded
    if key in state.catalog and not reused:
        raise ValueError('released thin wide result cannot be recomputed')
    output = TensorSpec(core.output_shape, core.result_dtype)
    seconds = 0.0 if reused else profile.contract_seconds(core.flops, core.traffic_bytes,
        kind='contract', dtype='complex128' if gram_strategy=='gemm' else dtype,
        kernel_calls=core.kernel_calls)
    action = Action('contract', key, inputs, equation, seconds, reused,
        term_index=scene, positions=positions, core=core, wide_accumulation=True)
    layout[1], layout[3], layout[8] = tuple(output_order), output.shape, key
    return action, output, 0 if reused else core.workspace_bytes, tuple(layout)


def replay_factorized_action(state, action, profile, memory_limit=1 << 100):
    if type(action.core) is not FactorizedWideReduction:
        raise ValueError('not a thin wide Action')
    actual, output, workspace, layout = describe_factorized_action(state, action.term_index,
        action.positions, action.core.large_operands, profile=profile, output_order=action.equation.output,
        tile_elements=action.core.tile_elements, output_dtype=action.core.output_dtype,
        gram_strategy=action.core.gram_strategy)
    if actual != action:
        raise ValueError('thin wide Action differs from independent algebra/resource description')
    from .canonical_operations import apply_description
    return apply_description(state, actual, output, workspace, actual.equation.output,
                             actual.positions, memory_limit=memory_limit)


def _candidates(state):
    contracts = tuple(a for a in state.actions if a.kind == 'contract')
    producers = {a.tensor: a for a in contracts if not a.reused}
    consumers = Counter(key for a in contracts for key in a.operands)
    groups, consumed = {}, set()
    for action in contracts:
        if not action.wide_accumulation or action.core is not None or len(action.operands) != 2:
            continue
        triggers = {label for trigger, _ in state.problem.terms[action.term_index].sum_barriers for label in trigger}
        factors, large = [], []
        children = []
        for key, final_axes in zip(action.operands, action.equation.inputs, strict=True):
            child = producers.get(key)
            if (child is None or child.core is not None or child.wide_accumulation or child.reused
                    or len(child.operands) != 2 or consumers[key] != 1 or key in state.results
                    or child.term_index != action.term_index or key in consumed
                    or child.equation.output != final_axes):
                break
            big = [i for i, axes in enumerate(child.equation.inputs) if triggers.intersection(axes)]
            if len(big) != 1:
                break
            large.append((child.operands[big[0]], child.equation.inputs[big[0]]))
            factors.extend(zip(child.operands, child.equation.inputs, strict=True))
            children.append(child.tensor)
        else:
            if len(children) == 2 and len(set(children)) == 2:
                groups[action.tensor] = (tuple(factors), tuple(large), tuple(children))
                consumed.update(children)
    return groups


def _positions(term, expected):
    """Four occurrences at most; preserve repeated tensor/axis occurrences."""
    remaining = list(enumerate(term.factors))
    result = []
    for key, axes in expected:
        index, factor = next((i, f) for i, f in remaining if f.tensor == key and f.labels == axes)
        remaining.remove((index, factor));result.append(index)
    return tuple(sorted(result))


def lower_factorized_reductions(plan, *, profile, target_runtime):
    stats = dict(applied=False, cost_status='uncalibrated-wide-work-and-traffic')
    if (target_runtime != 'cuda' or plan.state.problem.backend != 'cupy'
            or not plan.feasible or plan.explicit_resources or plan.retention_planned
            or plan.retained_before or plan.retained_after or plan.retained_releases
            or plan.state.problem.sparse_bridges or plan.state.problem.baryon_joins
            or any(a.kind not in ('load', 'release', 'contract') or a.core is not None
                   for a in plan.state.actions)):
        return plan, dict(stats, reason='unsupported_domain')
    groups = _candidates(plan.state)
    if not groups:
        return plan, dict(stats, reason='no_factorized_wide_boundary')
    omitted = {key for _, _, keys in groups.values() for key in keys}
    state = ContractionState.initial(plan.state.problem)
    names = {key: key for key in state.problem.sources}
    try:
        from .canonical_operations import describe, apply_description
        for old in plan.state.actions:
            if old.kind != 'contract' or old.tensor in omitted:
                continue
            group = groups.get(old.tensor)
            expected = tuple((names[key], axes) for key, axes in
                (group[0] if group is not None else zip(old.operands, old.equation.inputs, strict=True)))
            for key, _ in expected:
                if key not in state.loaded:
                    state = state.load(key, profile, 1 << 100)
            scene = old.term_index
            positions = _positions(state.pending[scene], expected)
            if group is None:
                action, spec, workspace, output, positions = describe(state, scene, positions,
                    profile=profile, output_order=old.equation.output)
                state = apply_description(state, action, spec, workspace, output, positions,
                    memory_limit=1 << 100)
            else:
                large = tuple((names[key], axes) for key, axes in group[1])
                large_indices = tuple(i for i, position in enumerate(positions)
                    if (state.pending[scene].factors[position].tensor,
                        state.pending[scene].factors[position].labels) in large)
                action, _, _, _ = describe_factorized_action(state, scene, positions, large_indices,
                    profile=profile, output_order=old.equation.output)
                state = replay_factorized_action(state, action, profile)
            names[old.tensor] = action.tensor
            for key in sorted(state.loaded - state.live_inputs - set(state.results)):
                state = state.release(key)
        if not state.done:
            raise ValueError('factorized lowering did not finish the original Terms')
        from .storage import plan_scene_storage
        candidate = plan_scene_storage(state, plan.blocks, plan.preparations, profile=profile,
            memory_limit=plan.memory_limit, host_memory_limit=plan.host_memory_limit,
            reserved_host_bytes=plan.reserved_host_bytes, reserved_device_bytes=plan.reserved_device_bytes,
            disk_limit_bytes=plan.disk_limit_bytes)
    except (ValueError, KeyError, StopIteration, MemoryError) as error:
        return plan, dict(stats, reason='algebra_or_resource_ineligible', detail=str(error))
    if not candidate.feasible or candidate.seconds >= plan.seconds:
        return plan, dict(stats, reason='no_feasible_cost_improvement')
    return candidate, dict(stats, applied=True, groups=len(groups),
        partial_bytes=sum(a.core.partial_bytes for a in state.actions if type(a.core) is FactorizedWideReduction),
        peak_device_bytes=candidate.peak_device_bytes, peak_host_bytes=candidate.peak_host_bytes)
