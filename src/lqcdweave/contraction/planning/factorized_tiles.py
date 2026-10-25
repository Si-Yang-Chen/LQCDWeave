"""Budgeted tile selection on an already selected wide-reduction route."""
from dataclasses import replace
from functools import lru_cache
import numpy as np

from lqcdweave.contracts.correlator.api import FactorizedWideReduction
from lqcdweave.contracts.correlator.factorized_reduction import MAX_FACTORIZED_TILE_ELEMENTS
from .state import ContractionState
from .factorized_reduction import describe_factorized_action
from .canonical_operations import describe, apply_description


# A bounded planning policy, not an execution-side permission to borrow more.
# Fixed execution always respects the saved core and its charged workspace.
_PARTIAL_LIMIT_BYTES = 8 * 1024**2


def retile_factorized_reductions(plan, *, profile, target_runtime, max_tile_elements=MAX_FACTORIZED_TILE_ELEMENTS):
    """Keep selected algebra/sharing; independently describe and audit tile costs.

    Called only during new W preparation. It neither expands E nor edits an
    archived execution plan. Unsupported physical domains retain their plan.
    Invalid existing contracts remain errors rather than fallback signals.
    """
    if (target_runtime != 'cuda' or plan.state.problem.backend != 'cupy' or not plan.feasible
            or plan.explicit_resources or plan.retention_planned or plan.retained_before
            or plan.retained_after or plan.retained_releases or plan.kernel_batches
            or plan.state.problem.sparse_bridges or plan.state.problem.baryon_joins
            or any(a.kind not in ('load','release','contract')
                   or a.core is not None and type(a.core) is not FactorizedWideReduction
                   for a in plan.state.actions)):
        return plan,dict(applied=False,reason='unsupported_domain')
    if not any(type(a.core) is FactorizedWideReduction and a.core.tile_elements is not None
               and a.core.tile_width < min(a.core.epilogue_elements,max_tile_elements)
               for a in plan.state.actions):
        return plan,dict(applied=False,reason='no_wider_tile')

    @lru_cache(maxsize=2048)
    def choices(core):
        if core.gram_strategy != 'reduction':
            return ()
        widths=tuple(dict.fromkeys(min(core.epilogue_elements,w,max_tile_elements)
                     for w in (core.tile_width,8192,16384,MAX_FACTORIZED_TILE_ELEMENTS)))
        result=[]
        for width in widths:
            if width <= core.tile_width:
                continue
            candidate=replace(core,tile_elements=width)
            if candidate.partial_bytes > _PARTIAL_LIMIT_BYTES:
                continue
            seconds=profile.contract_seconds(candidate.flops,candidate.traffic_bytes,
                kind='contract',dtype=np.dtype(core.dtype),kernel_calls=candidate.kernel_calls)
            result.append((seconds,width,candidate.workspace_bytes))
        return tuple(sorted(result))

    state=ContractionState.initial(plan.state.problem).with_indexed_rows()
    names={key:key for key in state.problem.sources};selected={};actions=[];changed=0
    old_partial=new_partial=0
    for old in plan.state.actions:
        state=state.with_action_history(())
        if old.kind=='load':
            state=state.load(names[old.tensor],profile,plan.memory_limit)
        elif old.kind=='release':
            state=state.release(names[old.tensor])
        else:
            factors=tuple(state.pending[old.term_index].factors[i] for i in old.positions)
            if (tuple(f.tensor for f in factors)!=tuple(names[key] for key in old.operands)
                    or tuple(f.labels for f in factors)!=old.equation.inputs):
                raise ValueError('wide retile differs from selected operand roles')
            if type(old.core) is FactorizedWideReduction:
                core=old.core
                if (old.equation!=core.equation
                        or tuple(state.catalog[f.tensor].shape for f in factors)!=core.shapes
                        or plan.state.catalog[old.tensor].shape!=core.output_shape
                        or plan.state.catalog[old.tensor].dtype!=np.dtype(core.result_dtype)):
                    raise ValueError('wide retile received an invalid saved core contract')
                width=selected.get(old.tensor,core.tile_elements)
                if old.tensor not in selected:
                    if core.tile_elements is not None and not old.reused:
                        output_bytes=core.output_elements*np.dtype(core.result_dtype).itemsize
                        for seconds,candidate_width,workspace in choices(core):
                            if (seconds < old.seconds
                                    and state.resident_bytes+output_bytes+workspace <= plan.memory_limit):
                                width=candidate_width;changed+=1;break
                    selected[old.tensor]=width
                action,spec,workspace,_=describe_factorized_action(state,old.term_index,old.positions,
                    core.large_operands,profile=profile,output_order=old.equation.output,
                    tile_elements=width,output_dtype=core.output_dtype,gram_strategy=core.gram_strategy)
                output,positions=action.equation.output,action.positions
                old_partial=max(old_partial,core.partial_bytes)
                new_partial=max(new_partial,action.core.partial_bytes)
            else:
                action,spec,workspace,output,positions=describe(state,old.term_index,old.positions,
                    profile=profile,output_order=old.equation.output)
            if action.reused!=old.reused:
                raise ValueError('wide retile changed selected result sharing')
            state=apply_description(state,action,spec,workspace,output,positions,
                memory_limit=plan.memory_limit)
            names[old.tensor]=action.tensor
        actions.extend(state.actions)
    if not changed:
        return plan,dict(applied=False,reason='no_feasible_cost_improvement')
    state=state.with_action_history(actions).with_tuple_rows()
    if not state.done or tuple(state.results)!=tuple(names[key] for key in plan.state.results):
        raise ValueError('wide retile changed selected term completion')
    from .storage import plan_scene_storage
    candidate=plan_scene_storage(state,plan.blocks,plan.preparations,profile=profile,
        memory_limit=plan.memory_limit,host_memory_limit=plan.host_memory_limit,
        reserved_host_bytes=plan.reserved_host_bytes,reserved_device_bytes=plan.reserved_device_bytes,
        disk_limit_bytes=plan.disk_limit_bytes)
    if not candidate.feasible or candidate.seconds >= plan.seconds:
        return plan,dict(applied=False,reason='no_physical_cost_improvement')
    return candidate,dict(applied=True,actions=changed,old_max_partial_bytes=old_partial,
        max_partial_bytes=new_partial,peak_device_bytes=candidate.peak_device_bytes,
        peak_host_bytes=candidate.peak_host_bytes,cost_status='uncalibrated-same-route-tile-cost')
