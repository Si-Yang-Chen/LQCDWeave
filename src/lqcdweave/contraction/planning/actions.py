"""Shared, data-free action preparation and selected operation replay."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from .state import Action, ContractionState, CostProfile


def _contract_positions(state: ContractionState, checkpoint=lambda: None):
    """Enumerate one representative per identical term and implicit context.

    Residency and cost are checked by the caller. This only removes duplicate
    ordinary/unary decisions; individual baryon join effects stay independent.
    """
    scene_joins = {}
    for join in state.problem.baryon_joins:
        checkpoint()
        if (join.scene, join.name) not in state.completed_joins:
            scene_joins.setdefault(join.scene, []).append(replace(join, scene=0))
    bridges = []
    for bridge in state.problem.sparse_bridges:
        checkpoint()
        bridges.append((bridge.scene, replace(bridge, scene=None)))
    seen = set()
    for index, term in enumerate(state.pending):
        if term is None:
            continue
        checkpoint()
        context = (term, tuple(scene_joins.get(index, ())),
                   tuple(bridge for scene, bridge in bridges if scene in (None, index)))
        if context in seen:
            continue
        seen.add(context)
        pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for pair in pairs:
            checkpoint()
            yield index, tuple(pair)


def _live_inputs(state: ContractionState) -> frozenset[str]:
    return state.live_inputs


def _find_live_inputs(state: ContractionState) -> set[str]:
    required = {factor.tensor for term in state.pending if term is not None for factor in term.factors}
    for index, term in enumerate(state.problem.terms):
        if term.kernel is not None and state.pending[index] is not None:
            required.update(term.kernel.inputs)
    for join in state.problem.baryon_joins:
        if state.pending[join.scene] is not None and (join.scene, join.name) not in state.completed_joins:
            required.update((join.spin, join.elemental))
            if join.core is not None:
                required.update(join.core.inputs)
    for bridge in state.problem.sparse_bridges:
        terms = state.pending if bridge.scene is None else (state.pending[bridge.scene],)
        endpoints = {bridge.left, bridge.right}
        if any(term is not None and endpoints.issubset(factor.tensor for factor in term.factors) for term in terms):
            required.add(bridge.record)
    return required


def _required_loads(state: ContractionState) -> set[str]:
    """Return currently live tensors, including source-backed bridge inputs."""
    return _live_inputs(state)


def _apply_selected(
    state: ContractionState,
    scene: int,
    chosen: Action,
    profile: CostProfile,
    memory_limit: int,
    *,
    layout_index=None,
    changes=None,
) -> ContractionState | None:
    from lqcdweave.contracts.correlator.api import MatrixKernelSpec, FactorizedWideReduction, MesonWholeKernel
    if type(chosen.core) is MesonWholeKernel:
        from .meson_core import replay_meson_whole_action
        candidate = replay_meson_whole_action(state, chosen, profile, memory_limit)
        if changes is not None:
            changes.append((scene, candidate.pending[scene], candidate.results[scene]))
        return candidate
    if type(chosen.core) is FactorizedWideReduction:
        from .factorized_reduction import replay_factorized_action
        if scene != chosen.term_index:
            raise ValueError('thin wide Action names a different Term')
        candidate = replay_factorized_action(state, chosen, profile, memory_limit)
        if changes is not None:
            changes.append((scene, candidate.pending[scene], candidate.results[scene]))
        return candidate
    if isinstance(chosen.core, MatrixKernelSpec):
        from .matrix_kernel import replay_matrix_action
        if scene != chosen.term_index:
            raise ValueError('matrix Action names a different Term')
        candidate = replay_matrix_action(state, chosen, profile, memory_limit)
        if changes is not None:
            changes.append((scene, candidate.pending[scene], candidate.results[scene]))
        return candidate
    from .canonical_operations import PREFIX, replay
    if chosen.tensor.startswith(PREFIX):
        candidate = replay(state, chosen, profile, memory_limit)
        if changes is not None:
            changes.append((scene, candidate.pending[scene], candidate.results[scene]))
        return candidate
    if chosen.kind == "baryon_join":
        join = next(
            (item for item in state.problem.baryon_joins if item.scene == scene and item.name == chosen.join_id),
            None,
        )
        if join is None:
            return None
        try:
            candidate = state.contract_join(
                join, profile, memory_limit, kernel_path_override=chosen.kernel_path,
            )
        except ValueError:
            return None
        if candidate.actions[-1].tensor != chosen.tensor:
            return None
        if changes is not None:
            changes.append((scene, candidate.pending[scene], candidate.results[scene]))
        return candidate

    term = state.pending[scene]
    if term is None:
        return state
    size = 1 if len(term.factors) == 1 else 2
    expected = chosen.operands[:size]
    for positions in ((0,),) if size == 1 else combinations(range(len(term.factors)), 2):
        if tuple(term.factors[i].tensor for i in positions) != expected:
            continue
        try:
            candidate_changes = []
            if layout_index is None:
                candidate = state.contract(scene, tuple(positions), profile, memory_limit)
            else:
                layout = layout_index.layout(scene, tuple(positions))
                candidate = state._contract_from_layout(
                    scene, tuple(positions), layout, profile, memory_limit,
                    layout_index=layout_index, changes=candidate_changes,
                )
        except ValueError:
            continue
        if candidate.actions[-1].tensor == chosen.tensor:
            if changes is not None:
                changes.extend(candidate_changes)
            return candidate
    return None
