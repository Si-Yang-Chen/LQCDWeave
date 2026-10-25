"""Resource-audited full wide Gram lowering of a selected path."""
from dataclasses import replace
from math import prod
import numpy as np

from lqcdweave.contracts.correlator.api import FactorizedWideReduction
from lqcdweave.contracts.correlator.factorized_geometry import gram_matrix_axes
from .state import ContractionState
from .factorized_reduction import describe_factorized_action
from .canonical_operations import describe, apply_description


def _amortized_matrix_gram(core):
    axes=gram_matrix_axes(core.equation,core.large_operands,core._geometry)
    if axes is None:
        return False
    dimensions=core.dimensions
    rows,columns=(prod(dimensions[axis] for axis in group) for group in axes)
    products=rows*columns*core.reduction_elements
    packed_bytes=16*(rows+columns)*core.reduction_elements
    # Fixed pack/GEMM submission is not amortized by the measured small-output
    # domain. Keep a conservative V100 work/output-parallelism fence in P;
    # Intensity is logical complex products per packed byte (at least 2).
    # These measured-domain heuristics are not a calibrated universal model;
    # the saved resource and provisional cost audits still apply independently.
    return (core.output_elements >= 128 and products >= 1 << 23
            and 2*packed_bytes <= products)


def _lower_plan(plan, profile):
    if (plan.state.problem.backend != 'cupy' or not plan.feasible or plan.explicit_resources
            or plan.retention_planned or plan.retained_before or plan.retained_after
            or plan.retained_releases or plan.kernel_batches or plan.state.problem.sparse_bridges
            or plan.state.problem.baryon_joins
            or any(a.kind not in ('load','release','contract')
                   or a.core is not None and type(a.core) is not FactorizedWideReduction
                   for a in plan.state.actions)):
        return plan,dict(applied=False,reason='unsupported_domain')
    eligible = {a.tensor for a in plan.state.actions
                if type(a.core) is FactorizedWideReduction and a.core.gram_strategy=='reduction'
                and a.core.tile_elements is not None and not getattr(a.core,'finish_support',())
                and _amortized_matrix_gram(a.core)}
    if not eligible:
        return plan,dict(applied=False,reason='no_amortized_matrix_gram')
    state=ContractionState.initial(plan.state.problem).with_indexed_rows()
    names={key:key for key in state.problem.sources}
    selected={};actions=[];changed=0
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
                raise ValueError('wide Gram lowering differs from the selected operand roles')
            if type(old.core) is FactorizedWideReduction:
                if (old.equation!=old.core.equation
                        or tuple(state.catalog[f.tensor].shape for f in factors)!=old.core.shapes
                        or plan.state.catalog[old.tensor].shape!=old.core.output_shape
                        or plan.state.catalog[old.tensor].dtype!=np.dtype(old.core.result_dtype)):
                    raise ValueError('wide Gram lowering received an invalid core contract')
                strategy=selected.get(old.tensor,old.core.gram_strategy)
                if old.tensor in eligible and old.tensor not in selected:
                    candidate=replace(old.core,gram_strategy='gemm')
                    output_bytes=candidate.output_elements*np.dtype(candidate.result_dtype).itemsize
                    predicted=profile.contract_seconds(candidate.flops,candidate.traffic_bytes,
                        kind='contract',dtype='complex128',kernel_calls=candidate.kernel_calls)
                    # This local check avoids an infeasible logical transition.
                    # Complete physical storage/rounding is audited below.
                    if (state.resident_bytes+output_bytes+candidate.workspace_bytes <= plan.memory_limit
                            and predicted < old.seconds):
                        strategy='gemm';changed+=1
                    selected[old.tensor]=strategy
                action,spec,workspace,_=describe_factorized_action(state,old.term_index,old.positions,
                    old.core.large_operands,profile=profile,output_order=old.equation.output,
                    tile_elements=old.core.tile_elements,output_dtype=old.core.output_dtype,
                    gram_strategy=strategy)
                output,positions=action.equation.output,action.positions
            else:
                action,spec,workspace,output,positions=describe(state,old.term_index,old.positions,
                    profile=profile,output_order=old.equation.output)
            if action.reused!=old.reused:
                raise ValueError('wide Gram lowering changed selected result sharing')
            state=apply_description(state,action,spec,workspace,output,positions,
                memory_limit=plan.memory_limit)
            names[old.tensor]=action.tensor
        actions.extend(state.actions)
    if not changed:
        return plan,dict(applied=False,reason='no_feasible_cost_improvement')
    state=state.with_action_history(actions).with_tuple_rows()
    if not state.done or tuple(state.results)!=tuple(names[key] for key in plan.state.results):
        raise ValueError('wide Gram lowering changed selected term completion')
    from .storage import plan_scene_storage
    candidate=plan_scene_storage(state,plan.blocks,plan.preparations,profile=profile,
        memory_limit=plan.memory_limit,host_memory_limit=plan.host_memory_limit,
        reserved_host_bytes=plan.reserved_host_bytes,reserved_device_bytes=plan.reserved_device_bytes,
        disk_limit_bytes=plan.disk_limit_bytes)
    if not candidate.feasible:
        return plan,dict(applied=False,reason='physical_storage_ineligible')
    return candidate,dict(applied=True,actions=changed,peak_device_bytes=candidate.peak_device_bytes,
        cost_status='uncalibrated-wide-gemm-pack-and-epilogue')


def lower_factorized_gemm_path(path, *, target_runtime):
    """New offline preparation only; archive execution never calls this pass."""
    if target_runtime!='cuda' or path.plan is None:
        return path
    if path.batch_paths:
        # Coordinate-lifted matrix geometry retains lanes and is deliberately
        # outside this first implementation. Do not change a saved batch DAG.
        return path
    selected,statistics=_lower_plan(path.plan,path.profile)
    if selected is path.plan:
        return replace(path,prediction=dict(path.prediction,factorized_gemm_lowering=statistics))
    from ..performance.api import predict_plan_cost
    prediction=dict(path.prediction)
    prediction.update(predict_plan_cost((selected,),path.profile,
        workload=prediction.get('workload',{}),provenance=prediction.get('provenance',{})))
    if 'predicted_seconds' in prediction:
        prediction['predicted_seconds']+=selected.seconds-path.plan.seconds
    prediction['factorized_gemm_lowering']=statistics
    search=path.search
    if search is not None:
        search=replace(search,plan=selected,statistics=dict(search.statistics,
            factorized_gemm_lowering=statistics))
    return replace(path,plan=selected,plan_identity=id(selected),prediction=prediction,search=search)
