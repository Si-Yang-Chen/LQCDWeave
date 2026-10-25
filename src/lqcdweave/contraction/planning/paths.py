"""Compressed legal contraction-path graph and lazy enumeration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import suppress
from functools import reduce
from itertools import combinations, islice, chain
import math
import time
from typing import Any, Mapping

from .state import Action, ContractionProblem, ContractionState, CostProfile
from .actions import _contract_positions
from .logical_index import FactorIndex, LayoutIndex, reuse_indexes
from lqcdweave.contracts.correlator.api import MatrixKernelSpec, FactorizedWideReduction, normalized_scene_relations


def rebind_scene_path(template, problem, source_map, *, profile, memory_limit,
                      host_memory_limit=None):
    """Reuse an operation order under an explicit, structure-preserving rename.

    No physical identity is inferred from matching shapes. The caller supplies
    the new source contracts. Dense shapes and algebra must match; sparse record
    bounds and source costs may change and are recomputed on every transition.
    The resulting compute path still needs physical-storage lifetime auditing.
    """
    old = template.problem
    names = dict(source_map)
    if old.meson_cores or problem.meson_cores:
        if len(old.meson_cores) != len(problem.meson_cores):
            raise ValueError('whole meson core topology differs')
        for before, after in zip(old.meson_cores, problem.meson_cores, strict=True):
            names.setdefault(before.key, after.key)
        for before, after in zip(old.meson_cores, problem.meson_cores, strict=True):
            if replace(before, key=names[before.key], inputs=tuple(names[key] for key in before.inputs)) != after:
                raise ValueError('whole meson core bindings differ')
    old_inputs = set(old.sources) | {core.key for core in old.meson_cores}
    new_inputs = set(problem.sources) | {core.key for core in problem.meson_cores}
    if not template.done:
        raise ValueError("path template must be complete")
    if (set(names) != old_inputs or set(names.values()) != new_inputs
            or len(set(names.values())) != len(names)):
        raise ValueError("source mapping must be a complete bijection")
    if old.backend != problem.backend:
        raise ValueError("path template backend differs")
    if {names[k]:v for k,v in old.specs.items()} != dict(problem.specs):
        raise ValueError("dense tensor specifications differ")
    terms = tuple(replace(t,inputs=tuple(names[k] for k in t.inputs),
        kernel=(None if t.kernel is None else replace(t.kernel,
            cores=tuple(names[k] for k in t.kernel.cores), inputs=tuple(names[k] for k in t.kernel.inputs))))
        for t in old.terms)
    if any(left.kernel != right.kernel for left, right in zip(terms, problem.terms, strict=True)):
        raise ValueError('whole meson local implementation differs')
    if normalized_scene_relations(terms)[0] != normalized_scene_relations(problem.terms)[0]:
        raise ValueError("term algebra or sharing differs")
    if len(old.sparse_bridges) != len(problem.sparse_bridges):
        raise ValueError("sparse bridge topology differs")
    for before,after in zip(old.sparse_bridges,problem.sparse_bridges):
        renamed = replace(before,left=names[before.left],right=names[before.right],
                          record=names[before.record],nnz_upper=after.nnz_upper)
        if renamed != after:
            raise ValueError("sparse bridge topology differs")
    def renamed_core(core):
        if core is None:
            return None
        return replace(core, components=tuple(replace(part, inputs=tuple(names[k] for k in part.inputs))
                                              for part in core.components))
    joins = tuple(replace(j,spin=names[j.spin],elemental=names[j.elemental], core=renamed_core(j.core))
                  for j in old.baryon_joins)
    if normalized_scene_relations(terms, joins)[1] != normalized_scene_relations(problem.terms, problem.baryon_joins)[1]:
        raise ValueError("baryon join topology differs")
    state = ContractionState.initial(problem).with_indexed_rows()
    actions = []
    for action in template.actions:
        if action.kind == "load":
            state = state.load(names[action.tensor],profile,memory_limit,host_limit_bytes=host_memory_limit)
        elif action.kind == "release":
            state = state.release(names[action.tensor])
        else:
            from .canonical_operations import PREFIX, describe, apply_description
            from lqcdweave.contracts.correlator.meson_core import MesonWholeKernel
            if type(action.core) is MesonWholeKernel:
                from .meson_core import describe_meson_whole_action, replay_meson_whole_action
                generated, _, _ = describe_meson_whole_action(state, action.term_index, profile=profile)
                state = replay_meson_whole_action(state, generated, profile, memory_limit)
            elif type(action.core) is FactorizedWideReduction:
                from .factorized_reduction import describe_factorized_action, replay_factorized_action
                selected = tuple(state.pending[action.term_index].factors[i] for i in action.positions)
                axes = {before: after for old_axes, factor in zip(action.equation.inputs, selected, strict=True)
                        for before, after in zip(old_axes, factor.labels, strict=True)}
                generated, _, _, _ = describe_factorized_action(state, action.term_index, action.positions,
                    action.core.large_operands, profile=profile,
                    output_order=tuple(axes[label] for label in action.equation.output),
                    tile_elements=action.core.tile_elements, output_dtype=action.core.output_dtype,
                    gram_strategy=action.core.gram_strategy)
                state = replay_factorized_action(state, generated, profile, memory_limit)
            elif isinstance(action.core, MatrixKernelSpec):
                from .matrix_kernel import describe_matrix_action, replay_matrix_action
                generated, _, _ = describe_matrix_action(state, action.term_index, action.core,
                                                         profile=profile)
                state = replay_matrix_action(state, generated, profile, memory_limit)
            elif action.tensor.startswith(PREFIX):
                join = (next(j for j in problem.baryon_joins
                             if j.scene == action.term_index and j.name == action.join_id)
                        if action.kind == "baryon_join" else None)
                output_order = None
                if join is None:
                    selected = tuple(state.pending[action.term_index].factors[i] for i in action.positions)
                    axes = {}
                    for before, factor in zip(action.equation.inputs, selected, strict=True):
                        for old_axis, new_axis in zip(before, factor.labels, strict=True):
                            if old_axis in axes and axes[old_axis] != new_axis:
                                raise ValueError("renaming changed axis alignment")
                            axes[old_axis] = new_axis
                    output_order = tuple(axes[axis] for axis in action.equation.output)
                generated, spec, workspace, output, positions = describe(state, action.term_index, action.positions,
                    join=join, profile=profile, kernel_path=action.kernel_path if join else None,
                    output_order=output_order)
                state = apply_description(state, generated, spec, workspace, output, positions,
                                          join=join, memory_limit=memory_limit)
            elif action.kind == "baryon_join":
                join = next(j for j in problem.baryon_joins
                            if j.scene==action.term_index and j.name==action.join_id)
                state = state.contract_join(join,profile,memory_limit,kernel_path_override=action.kernel_path)
            elif action.kind in ("contract","sparse_bridge"):
                state = state.contract(action.term_index,action.positions,profile,memory_limit)
            else:
                raise ValueError("unsupported template action")
            generated = state.actions[-1]
            if (generated.kind != action.kind or generated.equation.normalized() != action.equation.normalized()
                    or generated.operands != tuple(names[k] for k in action.operands)
                    or generated.join_id != action.join_id
                    or generated.left_axes != action.left_axes
                    or generated.right_axes != action.right_axes
                    or generated.same_point != action.same_point
                    or generated.transpose_bridge != action.transpose_bridge
                    or generated.used_np != action.used_np
                    or (generated.used_np_right if generated.used_np_right is not None else generated.used_np)
                    != (action.used_np_right if action.used_np_right is not None else action.used_np)):
                raise ValueError("renaming changed contraction semantics")
            if action.tensor in names and names[action.tensor] != generated.tensor:
                raise ValueError("renaming changed intermediate sharing")
            names[action.tensor] = generated.tensor
        if host_memory_limit is not None and state.peak_host_bytes > host_memory_limit:
            raise MemoryError("rebound path exceeds host memory limit")
        if len(state.actions) >= 256:
            actions.extend(state.actions)
            state = state.with_action_history(())
    if not state.done or state.results != tuple(names[k] for k in template.results):
        raise ValueError("rebound path changed outputs")
    actions.extend(state.actions)
    return state.with_action_history(actions).with_tuple_rows()


class PathLimitExceeded(ValueError):
    def __init__(self, count: int, limit: int):
        self.count, self.limit = count, limit
        super().__init__(f"{count} legal ordered paths exceed limit {limit}")


class StateGraphLimitExceeded(RuntimeError):
    def __init__(self, explored: int, limit: int):
        self.explored, self.limit = explored, limit
        super().__init__(f"state graph exceeded {limit} states before an exact path count was known")


def path_count(operand_count: int) -> int:
    """Count every ordered binary pair schedule, including outer products."""
    if isinstance(operand_count, bool) or operand_count < 1:
        raise ValueError("operand_count must be positive")
    return reduce(lambda value, n: value * math.comb(n, 2), range(2, operand_count + 1), 1)


def iter_paths(operand_count: int, *, max_paths: int = 100_000):
    """Yield complete opt_einsum-style pair paths without ranking them."""
    count = path_count(operand_count)
    if count > max_paths:
        raise PathLimitExceeded(count, max_paths)

    def walk(size: int, prefix: tuple[tuple[int, ...], ...]):
        if size == 1:
            yield prefix or ((0,),)
            return
        for pair in combinations(range(size), 2):
            yield from walk(size - 1, prefix + (pair,))

    yield from walk(operand_count, ())


from .optimizers.results import SearchResult


@dataclass(frozen=True)
class PathGraph:
    """Compressed graph of every legal contraction-only successor."""

    start: Any
    edges: Mapping[Any, tuple[tuple[Action, Any], ...]]
    suffix_counts: Mapping[Any, int]
    path_count: int


def _state_key(state: ContractionState):
    return state.search_key


def _contract_successors(state: ContractionState, profile: CostProfile, memory_limit: int, *, unique_terms=False):
    # Explicit path enumeration keeps occurrence counts. Physical exact search
    # can remove equal-cost parallel edges with identical future state.
    choices = _contract_positions(state) if unique_terms else (
        (index, tuple(pair)) for index, term in enumerate(state.pending) if term is not None
        for pair in (((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2))
    )
    reuse = reuse_indexes(state.pending)
    # Full-Term dedup can turn hundreds of occurrences into only three
    # choices. Such nodes need the occurrence map, not 64 layout contexts.
    prefix = list(islice(choices, 12)) if unique_terms else None
    if prefix is not None:
        choices = chain(prefix, choices)
    reuse_layout = reuse and (prefix is None or len(prefix) >= 12)
    layout_index, factor_index, scene_joins = None, None, None
    for term_index, pair in choices:
        term = state.pending[term_index]
        if all(term.factors[index].tensor in state.loaded for index in pair):
            with suppress(MemoryError, ValueError):
                if scene_joins is None:
                    scene_joins = {}
                    for join in state.problem.baryon_joins:
                        if (join.scene, join.name) not in state.completed_joins:
                            scene_joins.setdefault(join.scene, []).append(join)
                    if reuse_layout:
                        layout_index = LayoutIndex(state)
                    elif reuse:
                        factor_index = FactorIndex(state.pending)
                layout = (layout_index.layout(term_index, pair) if layout_index is not None else
                          state._contract_layout(term_index, pair, scene_joins=scene_joins))
                yield state._contract_from_layout(term_index, pair, layout, profile, memory_limit,
                                                  layout_index=layout_index, factor_index=factor_index)
    for join in state.problem.baryon_joins:
        if (join.scene, join.name) not in state.completed_joins:
            with suppress(MemoryError, ValueError):
                yield state.contract_join(join, profile, memory_limit)


def build_path_graph(
    problem: ContractionProblem,
    *,
    max_states: int = 1_000_000,
    profile: CostProfile | None = None,
    deadline_at: float | None = None,
) -> PathGraph:
    """Explore all contraction orders, with initial inputs symbolically resident."""
    initial = ContractionState.initial(problem)
    initial = replace(initial, loaded=frozenset(problem.sources))
    initial = replace(
        initial,
        peak_bytes=initial.resident_bytes,
        peak_host_bytes=initial.resident_bytes if problem.backend == "numpy" else 0,
        peak_device_bytes=initial.resident_bytes if problem.backend == "cupy" else 0,
    )
    edges: dict[Any, tuple[tuple[Action, Any], ...]] = {}
    counts: dict[Any, int] = {}
    active: set[Any] = set()
    profile = profile or CostProfile()

    def visit(state: ContractionState) -> int:
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError("contraction path graph deadline reached")
        key = _state_key(state)
        if key in counts:
            return counts[key]
        if key in active:
            raise RuntimeError("contraction-only state graph contains a cycle")
        if len(edges) >= max_states:
            raise StateGraphLimitExceeded(len(edges) + 1, max_states)
        active.add(key)
        successors = []
        total = 1 if state.done else 0
        if not state.done:
            for next_state in _contract_successors(state, profile, 2**63 - 1):
                target = _state_key(next_state)
                successors.append((next_state.actions[-1], target))
                total += visit(next_state)
        edges[key] = tuple(successors)
        counts[key] = total
        active.remove(key)
        return total

    count = visit(initial)
    return PathGraph(_state_key(initial), edges, counts, count)


def iter_action_paths(graph: PathGraph, *, max_paths: int = 100_000):
    """Lazily expand complete paths only when their exact count is acceptable."""
    if graph.path_count > max_paths:
        raise PathLimitExceeded(graph.path_count, max_paths)

    def walk(key, prefix):
        if graph.suffix_counts[key] == 0:
            return
        successors = graph.edges[key]
        if not successors:
            yield prefix
        for action, target in successors:
            if graph.suffix_counts[target]:
                yield from walk(target, prefix + (action,))

    yield from walk(graph.start, ())


__all__ = [
    "PathGraph",
    "PathLimitExceeded",
    "SearchResult",
    "StateGraphLimitExceeded",
    "build_path_graph",
    "iter_action_paths",
    "iter_paths",
    "path_count",
    "rebind_scene_path",
]
