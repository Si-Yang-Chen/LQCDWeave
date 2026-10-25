"""Heuristic partition from shared initial operation signatures."""
from dataclasses import replace
from copy import copy
from itertools import chain, combinations

from ..canonical_operations import logical_result_name, _result_name
from ..logical_index import BridgeIndex
from ..state import ContractionState
from .dense_patterns import compile_dense_choices
from lqcdweave.contracts.correlator.api import BaryonJoinSpec

_PARALLEL_CLASSIFICATION_MIN_TERMS = 4096


def _local_join(join, scene):
    # A component changes only the scene number. Its frozen spin/core records
    # were already checked when the parent problem was built; rebuilding them
    # here reparses every sparse coefficient for every component. Keep custom
    # subclasses' constructor hooks, and validate the new scope in the normal
    # component ContractionProblem constructor below.
    if join.scene == scene:
        return join
    if type(join) is not BaryonJoinSpec:
        return replace(join, scene=scene)
    result = copy(join)
    object.__setattr__(result, 'scene', scene)
    return result


def shared_operation_keys(problem, checkpoint=lambda: None, *, canonical=None):
    """Initial operation identities for each Term's heuristic partition.

    The dense fast path does not yet distinguish coupled wide boundaries or
    direct terminal cores; final candidate legality is the search's contract.

    `canonical` states whether the enclosing task is a fully dense problem.
    Parallel workers only see one component, so the parent passes its own
    answer to keep every chunk's keys comparable across processes.
    """
    state = None
    joins = {}
    for join in problem.baryon_joins:
        checkpoint()
        joins.setdefault(join.scene, []).append(join)
    bridges = BridgeIndex(problem.sparse_bridges)
    dense_keys, patterns = {}, {}
    # `_result_name(('contract', (inputs, *relation)))` hashes
    # (('contract', (inputs, *relation)), ()). Comparing that canonical tuple
    # partitions exactly the same classes, without one repr+blake2b per
    # candidate position. Only a task whose every key comes from this branch
    # may use the canonical tuple: a sparse/join task keeps the hashed name so
    # one dense scene can never stop matching an equivalent sparse one.
    dense_key = (not problem.baryon_joins and not problem.sparse_bridges
                 if canonical is None else canonical)
    for scene, logical in enumerate(problem.terms):
        checkpoint()
        scoped = bridges.scoped(scene)
        dense = not joins.get(scene) and not scoped
        if dense:
            row_key = logical.inputs, logical.subscripts
            keys = dense_keys.get(row_key)
            if keys is None:
                if logical.subscripts not in patterns:
                    patterns[logical.subscripts] = compile_dense_choices(logical.subscripts, checkpoint)
                compiled = []
                for positions, variants in patterns[logical.subscripts]:
                    checkpoint()
                    inputs = tuple(logical.inputs[i] for i in positions)
                    if len(inputs) == 1:
                        relation = variants[0][1][0]
                    else:
                        left, right = inputs
                        reverse = left > right
                        relation = variants[reverse][1][2 if left == right else int(reverse)]
                        if reverse:
                            inputs = (right, left)
                    compiled.append((("contract", inputs, *relation) if dense_key else
                                     _result_name(("contract", (inputs, *relation)))))
                keys = dense_keys[row_key] = tuple(compiled)
            for key in keys:
                checkpoint()
                yield key, scene
            continue
        if state is None:
            state = ContractionState.initial(problem)
        term = state.pending[scene]
        positions = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for selected in positions:
            checkpoint()
            try:
                layout = state._contract_layout(scene, selected, scene_joins=joins, active_bridges=scoped)
                key = logical_result_name(state, scene, selected, layout=layout)
            except ValueError:
                continue
            yield key, scene
        for join in joins.get(scene, ()):
            checkpoint()
            try:
                layout = state._join_layout(join, scene_joins=joins, scoped_bridges=scoped)
                key = logical_result_name(state, scene, join=join, layout=layout)
            except ValueError:
                continue
            yield key, scene


def shared_components(problem, checkpoint=lambda: None, *, component_workers=1, budget=None, statistics=None):
    """Match initial operations exactly; do not predict future intermediates.

    Operation identity includes input identities, full axis alignment, output
    order and sparse/core recipes. Shared inputs alone do not connect Terms.
    Matching operations connect classes transitively. This is a heuristic
    search partition, not a proof that future computations cannot be shared.
    Costs, IO, residency and memory limits do not participate in classification.
    """
    parent = list(range(len(problem.terms)))

    def root(scene):
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    def register(key, scene):
        representative = operations.setdefault(key, scene)
        left, right = root(representative), root(scene)
        if left != right:
            parent[max(left, right)] = min(left, right)

    operations = {}
    canonical = not problem.baryon_joins and not problem.sparse_bridges
    if (component_workers != 1 and len(problem.terms) >= _PARALLEL_CLASSIFICATION_MIN_TERMS
            and budget is not None):
        from .parallel_components import classified_chunks
        chunks = classified_chunks(problem, budget, component_workers,
                                    {} if statistics is None else statistics)
        try:
            for chunk in chunks:
                for key, scene in chunk:
                    register(key, scene)
        finally:
            chunks.close()
    else:
        for key, scene in shared_operation_keys(problem, checkpoint, canonical=canonical):
            register(key, scene)

    groups = {}
    for scene in range(len(parent)):
        checkpoint()
        groups.setdefault(root(scene), []).append(scene)
    return tuple(tuple(scenes) for scenes in groups.values())


def component_problem(problem, scenes, *, _scoped_joins=None):
    """Restrict logical inputs and remap scoped operations to local scene IDs.

    Physical blocks/preparations remain the caller's full-task responsibility.
    This helper neither schedules IO nor gives a component a storage budget.
    """
    scenes = tuple(scenes)
    if not scenes or len(set(scenes)) != len(scenes) or any(
            type(scene) is not int or not 0 <= scene < len(problem.terms) for scene in scenes):
        raise ValueError("component scenes must be distinct valid scene indices")
    remap = {scene: local for local, scene in enumerate(scenes)}
    terms = tuple(problem.terms[scene] for scene in scenes)
    tensors = set(chain.from_iterable(term.inputs for term in terms))
    # IntegerSpace already indexes joins by scene while constructing a whole
    # component. The optional subset saves a full scan for every local row.
    join_source = problem.baryon_joins if _scoped_joins is None else _scoped_joins
    joins = tuple(_local_join(join, remap[join.scene]) for join in join_source
                  if join.scene in remap)
    tensors.update(key for join in joins for key in (join.spin, join.elemental))
    tensors.update(key for join in joins if join.core is not None for key in join.core.inputs)
    bridges = tuple(replace(bridge, scene=None if bridge.scene is None else remap[bridge.scene])
                    for bridge in problem.sparse_bridges
                    if (bridge.scene in remap or bridge.scene is None) and any(
                        {bridge.left, bridge.right} <= set(term.inputs) for term in terms))
    sources = tensors | {bridge.record for bridge in bridges}
    return replace(problem, terms=terms, specs={key: problem.specs[key] for key in tensors},
                   sources={key: problem.sources[key] for key in sources},
                   sparse_bridges=bridges, baryon_joins=joins)
