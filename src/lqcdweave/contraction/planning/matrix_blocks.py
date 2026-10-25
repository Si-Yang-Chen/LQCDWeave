"""Compile selected rectangular meson blocks using shared binary Actions.

The K/H order follows meson_matrix.py's migrated fixed workflow. The component
maps stay tensors; neither operator cells nor component pairs are expanded.
"""
from dataclasses import replace
from lqcdweave.contracts.correlator.api import ContractionProblem

from .canonical_operations import describe, replay, logical_result_name
from .state import ContractionState
from .optimizers.macro_space import MacroSearchSpace
from .optimizers.storage_schedule import replay_fixed_dag


def plan_matrix_blocks(problem, kernels, blocks, preparations, *, profile,
                       memory_limit, host_memory_limit=None, reserved_host_bytes=0):
    physical = MacroSearchSpace(problem,blocks,preparations,profile=profile,
        memory_limit=memory_limit,host_memory_limit=host_memory_limit,
        reserved_host_bytes=reserved_host_bytes)
    final = ContractionState.initial(physical.problem)
    computed, actions, results = {}, [], []
    # Logical compilation does not allocate or read arrays. The physical replay
    # below prepares each input when needed and releases it after its final use.
    for scene,(term,core) in enumerate(zip(problem.terms,kernels,strict=True)):
        # Compile a single row at a time. Shared computation IDs/specs are
        # sufficient to reuse its K/H products; copying all pending rows and
        # the complete action prefix per binary operation serves no purpose.
        local = ContractionProblem({k:physical.problem.specs[k] for k in term.inputs},
            (term,),{k:physical.problem.sources[k] for k in term.inputs},
            backend=physical.problem.backend,single_read_regions=physical.problem.single_read_regions)
        state = ContractionState.initial(local)
        state = replace(state,loaded=frozenset(local.specs))

        def apply(operands):
            nonlocal state
            positions, taken = [], set()
            for key in operands:
                index = next(i for i,f in enumerate(state.pending[0].factors)
                             if f.tensor==key and i not in taken)
                taken.add(index)
                positions.append(index)
            positions = tuple(sorted(positions))
            key = logical_result_name(state,0,positions)
            if key in computed and key not in state.loaded:
                state = replace(state,catalog=state.catalog.updated(((key,computed[key]),)),
                                loaded=state.loaded|{key})
            action,_,_,_,_ = describe(state,0,positions,profile=physical.logical_profile)
            state = replay(state,action,physical.logical_profile)
            computed[action.tensor] = state.catalog[action.tensor]
            return action.tensor

        if not core.mode_block:
            # A request may mix projected and unprojected groups. Retain the
            # existing fixed template for the latter in the same Action IR.
            from .matrix_kernel import describe_matrix_action, replay_matrix_action
            action,_,_ = describe_matrix_action(state,0,core,profile=physical.logical_profile)
            if action.tensor in computed:
                state=replace(state,catalog=state.catalog.updated(((action.tensor,computed[action.tensor]),)),
                              loaded=state.loaded|{action.tensor})
                action,_,_=describe_matrix_action(state,0,core,profile=physical.logical_profile)
            state = replay_matrix_action(state,action,physical.logical_profile)
            computed[action.tensor] = state.catalog[action.tensor]
        elif core.kind=='connected':
            p,q,left,right,lw,rw = term.inputs
            k = apply((p,left))
            k = apply((k,lw))
            h = apply((q,right))
            h = apply((h,rw))
            apply((k,h))
        else:
            p,vertex,weight = term.inputs
            loop = apply((p,vertex))
            apply((loop,weight))
        if state.pending[0] is not None:
            # Complete a possible final layout permutation with the ordinary
            # one-factor Action; it is still independently costed and audited.
            apply((state.pending[0].factors[0].tensor,))
        if not state.done:
            raise RuntimeError('fixed rectangular matrix route did not complete every Term')
        actions.extend(replace(action,term_index=scene) for action in state.actions)
        results.append(state.results[0])
    final=replace(final,catalog=final.catalog.updated(computed.items()),
        pending=(None,)*len(results),results=tuple(results),actions=tuple(actions),
        loaded=frozenset(final.problem.sources)|frozenset(computed),
        seconds=sum(action.seconds for action in actions))
    plan = replay_fixed_dag(final,physical)
    if not plan.feasible:
        raise MemoryError('fixed rectangular matrix plan exceeds the physical storage budget before IO')
    return plan
