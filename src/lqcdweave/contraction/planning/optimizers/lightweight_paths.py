"""Budgeted complete paths over one mutable integer workspace.

The first route needs only an undo stack. Historical states and alternate
connections are not retained; remaining time explores local alternatives.
"""
from dataclasses import dataclass
import math

from .budget import BudgetExhausted
from .integer_scoring import validate_score
from .protocols import SearchEvent


# Larger scenes take opt_einsum's greedy tree; DP cost grows exponentially.
PAIR_PLAN_DP_OPERANDS = 14


def _leaf_pairs(pattern, shapes):
    """Leaf pairs that one scene's own flop-optimal tree contracts first."""
    from opt_einsum import get_symbol
    from opt_einsum.paths import DynamicProgramming, greedy, linear_to_ssa
    count = len(pattern.inputs)
    if count < 2:
        return frozenset()
    sizes = {}
    for axes, shape in zip(pattern.inputs, shapes, strict=True):
        sizes.update(zip(axes, shape))
    symbols = {label: get_symbol(index) for index, label in enumerate(sizes)}
    inputs = [frozenset(symbols[label] for label in axes) for axes in pattern.inputs]
    output = frozenset(symbols[label] for label in pattern.output)
    size_dict = {symbols[label]: extent for label, extent in sizes.items()}
    solver = (DynamicProgramming(minimize="flops") if count <= PAIR_PLAN_DP_OPERANDS
              else greedy)
    path = linear_to_ssa(solver(inputs, output, size_dict))
    return frozenset(frozenset(pair) for pair in path if len(pair) == 2 and max(pair) < count)


class PairPlanGuide:
    """Seed weights from each scene's own contraction tree.

    Factorized vertices expose small coefficient and Phi operands. A scene's
    best pair order is a cheap per-scene einsum question; the seed applies the
    shared operation that follows the most such trees. It only orders the
    first route, which is admitted and priced like any other route.
    """

    def __init__(self, space):
        self.space, self.trees, self.weights = space, {}, {}

    def _pairs(self, term):
        registry = self.space.registry
        shapes = tuple(registry.data[data].spec.shape for data in term.factors)
        key = (term.pattern, shapes)
        pairs = self.trees.get(key)
        if pairs is None:
            pairs = self.trees[key] = _leaf_pairs(registry.patterns[term.pattern], shapes)
        return pairs

    def weight(self, identity):
        weight = self.weights.get(identity)
        if weight is None:
            space, weight = self.space, 0
            rows = space._row_weights
            for scene, occurrence in space.groups.get(identity, {}).items():
                if (not occurrence.cores and
                        frozenset(occurrence.positions) in self._pairs(space.terms[scene])):
                    weight += 1 if rows is None else rows[scene]
            self.weights[identity] = weight
        return weight

    def forget(self, scenes):
        rows = self.space.rows
        for scene in scenes:
            for identity, *_ in rows.get(scene, ()):
                self.weights.pop(identity, None)


@dataclass(slots=True)
class _Frame:
    undo: object = None
    choices: tuple | None = None
    offset: int = 0
    prefix_cost: float = 0.0
    deviated: bool = False


class LightweightPaths:
    """Find whole-problem routes without building a fingerprint graph.

    The caller admits each complete route against the fixed logical and
    physical contracts. Only admitted routes become incumbents.
    """

    def __init__(self, space, budget, scorer, *, root_width=4, branch_width=4,
                 include_ties=True, on_complete, guide=None):
        self.space, self.budget, self.scorer = space, budget, scorer
        self.guide = guide
        self.root_width, self.branch_width = root_width, branch_width
        self.include_ties, self.on_complete = include_ties, on_complete
        self.best_cost, self.first_solution_seconds = math.inf, None
        self.first_logical_seconds = self.first_apply_seconds = None
        self.best_plan = None
        self.first_route = None
        self.complete_routes = self.attempted = self.rejected = 0
        self.improvement_curve = []
        self.reason, self.limited = "completed", False

    def _choices(self, width, checkpoint, *, root=False):
        scorer = self.scorer
        begin = getattr(scorer, "begin", None)
        if begin is not None:
            begin()
        selected = self.space.rank_candidates(width, scorer, checkpoint,
                                               include_ties=self.include_ties and not root)
        if selected is not None:
            return tuple(operation for operation, _ in selected.choices)
        scored = []
        for ordinal, (operation, cost) in enumerate(self.space.search_candidates(checkpoint)):
            checkpoint()
            score = cost if scorer is None else scorer(self.space, operation, cost, checkpoint)
            validate_score(score)
            scored.append((score, ordinal, operation))
        scored.sort(key=lambda item: (item[0], item[1]))
        return tuple(operation for _, _, operation in scored[:width])

    def _admit(self, path, cost):
        self.complete_routes += 1
        if self.first_logical_seconds is None:
            self.first_logical_seconds = self.budget.clock() - self.budget.started
        if cost >= self.best_cost:
            return
        plan = self.on_complete(tuple(path), cost)
        if plan is None:
            self.rejected += 1
            return
        self.best_plan, self.best_cost = plan, cost
        elapsed = self.budget.clock() - self.budget.started
        if self.first_solution_seconds is None:
            self.first_solution_seconds = elapsed
            self.first_route = tuple(path)
        self.improvement_curve.append((elapsed, cost))

    def _seed(self, checkpoint):
        """Follow the guide once, then undo to the initial state."""
        space, guide, tokens, path, cost = self.space, self.guide, [], [], 0.0
        try:
            while not space.is_done():
                self.budget.check()
                best = None
                for operation, seconds in space.search_candidates(checkpoint):
                    key = (-guide.weight(operation.calculation), seconds, operation.calculation)
                    if best is None or key < best[0]:
                        best = key, operation
                if best is None:
                    return
                operation = best[1]
                scenes = tuple(space.groups[operation.calculation])
                guide.forget(scenes)
                if self.first_apply_seconds is None:
                    self.first_apply_seconds = self.budget.clock() - self.budget.started
                tokens.append(space.apply(operation, checkpoint))
                self.attempted += 1
                path.append(operation)
                cost += operation.seconds
                space._index(checkpoint)
                guide.forget(scenes)
            self._admit(path, cost)
        finally:
            for token in reversed(tokens):
                space.undo(token)
            guide.weights.clear()

    def run(self):
        space, budget = self.space, self.budget
        space.reset()
        frames, path, cost = [_Frame()], [], 0.0

        def checkpoint():
            budget.check(transition=True)

        try:
            if self.guide is not None:
                self._seed(checkpoint)
            while frames:
                frame = frames[-1]
                complete = space.is_done()
                if not complete:
                    budget.check()
                    if budget.memory_due():
                        budget.sample_memory((space, frames, path, self.best_plan))
                if complete:
                    self._admit(path, cost)
                    frame.offset = len(frame.choices or ())
                elif frame.choices is None:
                    budget.consume(SearchEvent("expand"))
                    frame.choices = self._choices(
                        self.root_width if len(frames) == 1 else self.branch_width,
                        checkpoint, root=len(frames) == 1,
                    )
                # Once a feasible route exists, explore only one changed
                # decision at a time. Its continuation remains greedy; the
                # finite neighborhood ends without consuming a long deadline.
                if (frame.offset >= len(frame.choices or ()) or
                        self.first_route is not None and frame.deviated and frame.offset >= 1):
                    frames.pop()
                    if frame.undo is not None:
                        space.undo(frame.undo)
                        path.pop()
                        cost = frame.prefix_cost
                    continue
                operation = frame.choices[frame.offset]
                frame.offset += 1
                if cost + operation.seconds >= self.best_cost:
                    continue
                checkpoint()
                prefix_cost = cost
                token = space.apply(operation, checkpoint)
                self.attempted += 1
                if self.first_apply_seconds is None:
                    self.first_apply_seconds = budget.clock() - budget.started
                path.append(operation)
                cost += operation.seconds
                changed = (self.first_route is not None and
                           (len(path) > len(self.first_route) or
                            operation != self.first_route[len(path) - 1]))
                frames.append(_Frame(undo=token, prefix_cost=prefix_cost,
                                     deviated=frame.deviated or changed))
        except BudgetExhausted as error:
            self.reason, self.limited = error.reason, True
        finally:
            space.release()
        return self


def solve_lightweight_storage(physical, budget, *, root_width, branch_width,
                              scorer, include_ties, debug, kernel_batching,
                              storage_branch_width):
    """Search one complete logical task; audit only improving complete routes."""
    from ..path_planning import PathSearchExhausted
    from .graph_session import finish_fixed_storage
    from .integer_scoring import IntegerScorer
    from .integer_space import IntegerRegistry, IntegerSpace
    from .results import StorageSearchResult

    problem = physical.problem
    registry = IntegerRegistry(problem, memory_limit=physical.allocation_limit)
    try:
        space = IntegerSpace(problem, profile=physical.logical_profile, registry=registry,
                             checkpoint=lambda: budget.check(transition=True), debug=debug,
                             track_fingerprint=False)
    except BudgetExhausted as error:
        statistics = dict(expanded_states=budget.expanded,
            evaluated_transitions=budget.transitions,
            elapsed_seconds=budget.clock() - budget.started,
            first_logical_complete_seconds=None,
            first_integer_operation_seconds=None,
            first_solution_seconds=None,
            complete_plans=0)
        exhausted = PathSearchExhausted(
            f"lightweight path setup reached {error.reason} before a complete route; "
            "infeasibility is unproved")
        exhausted.search_result = StorageSearchResult("UNSOLVED", None,
            budget.expanded, True, 0, 0, None, physical.rejected_memory,
            dict(optimizer="anytime", engine="lightweight_integer_paths",
                 proof_scope="none"), error.reason, None, statistics)
        raise exhausted from error
    logical_audit_seconds = physical_schedule_seconds = 0.0
    winning_schedule = {}

    def admit(operations, expected_cost):
        nonlocal logical_audit_seconds, physical_schedule_seconds
        begun = budget.clock()
        audit_space = IntegerSpace(problem, profile=physical.logical_profile,
                                   registry=registry, debug=debug, track_fingerprint=False)
        audited = audit_space.audit(operations, expected_cost)
        logical_audit_seconds += budget.clock() - begun
        snippets = audit_space.snippets
        workspaces = tuple(registry.calculations[operation.calculation].workspace
                           for operation in operations)
        statistics = dict(expanded_states=budget.expanded,
                          evaluated_transitions=budget.transitions)
        try:
            scheduled = budget.clock()
            plan = finish_fixed_storage(physical, audited, snippets, workspaces,
                stats=statistics, clock=budget.clock, debug=debug,
                kernel_batching=kernel_batching, storage_branch_width=storage_branch_width)
        except PathSearchExhausted as error:
            if (getattr(error, "search_result", None) is None or
                    error.search_result.stop_reason != "post_search_storage_limit"):
                raise
            return None
        finally:
            physical_schedule_seconds += budget.clock() - scheduled
        winning_schedule.clear()
        winning_schedule.update((key, statistics[key]) for key in (
            "storage_schedule", "dense_kernel_lowering", "execution_compute_seconds",
            "post_search_storage_seconds", "rejected_memory", "resource_branches")
            if key in statistics)
        return plan

    search = LightweightPaths(space, budget,
        None if scorer == "incremental_time" else IntegerScorer(scorer),
        root_width=root_width, branch_width=branch_width,
        include_ties=include_ties, on_complete=admit,
        guide=PairPlanGuide(space) if problem.factorized_vertices else None).run()
    statistics = dict(
        expanded_states=budget.expanded,
        evaluated_transitions=budget.transitions,
        complete_plans=search.complete_routes,
        rejected_physical_candidates=search.rejected,
        first_logical_complete_seconds=search.first_logical_seconds,
        first_integer_operation_seconds=search.first_apply_seconds,
        first_solution_seconds=search.first_solution_seconds,
        seeded_first_route=search.guide is not None,
        improvement_curve=tuple(search.improvement_curve),
        elapsed_seconds=budget.clock() - budget.started,
        final_audit_seconds=logical_audit_seconds + physical_schedule_seconds,
        logical_audit_seconds=logical_audit_seconds,
        physical_schedule_seconds=physical_schedule_seconds,
        sampled_peak_search_bytes=budget.sampled_peak_memory,
        memory_sampling=budget.memory_sampling_statistics,
        component_count=1,
        component_scenes=(tuple(range(len(problem.terms))),),
        attempted_operations=search.attempted,
        retained_states=0,
        retained_connections=0,
        registered_data=len(registry.data),
        registered_patterns=len(registry.patterns),
        registered_calculations=len(registry.calculations),
    )
    if search.best_plan is None:
        error = PathSearchExhausted(
            f"lightweight path search found no physically feasible complete path: "
            f"{search.reason}; infeasibility is unproved")
        error.search_result = StorageSearchResult("UNSOLVED", None, budget.expanded,
            search.limited, search.attempted, search.complete_routes,
            None, physical.rejected_memory,
            dict(optimizer="anytime", engine="lightweight_integer_paths",
                 proof_scope="none"), search.reason, None, statistics)
        raise error
    # finish_fixed_storage recorded the executable plan's own fixed-path
    # schedule; the search summary belongs to the winning complete task.
    statistics.update(winning_schedule)
    return StorageSearchResult("FEASIBLE", search.best_plan, budget.expanded,
        search.limited, search.attempted, search.complete_routes,
        search.first_solution_seconds, physical.rejected_memory,
        dict(optimizer="anytime", engine="lightweight_integer_paths",
             scorer=scorer, objective="compute_time_then_storage_peak",
             proof_scope="none", operation_domain="logical_complete_contractions",
             state_identity="one_live_integer_workspace_with_undo",
             component_schedule="whole_task_first_path_then_bounded_local_alternatives"),
        search.reason, None, statistics)
