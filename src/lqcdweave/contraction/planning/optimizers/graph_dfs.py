"""One-pass width-limited DFS with complete boundary ties on a fingerprint graph.

Historical records own IDs and numeric connections only. A state is expanded
once; cheaper arrivals propagate through retained connections without reopening
it. The final best-parent route is independently replayed before delivery.
"""
from array import array
from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
import math
import time

from .budget import (DEFAULT_MEMORY_SAMPLE_INTERVAL, DEFAULT_PLANNING_SECONDS,
                     SearchBudget, BudgetExhausted, integer)
from .choices import OperationPool, order_ties
from .integer_scoring import reverse_score, validate_score
from .protocols import SearchEvent
from .results import SearchResult


@dataclass(slots=True)
class GraphNode:
    cost: float
    parent: bytes | None = None
    children: list = field(default_factory=list)
    deferred: array = field(default_factory=lambda: array("Q"))
    goal: bool = False
    closed: bool = False
    completion: object = None


@dataclass(slots=True)
class GraphFrame:
    identity: bytes
    choices: list | None = None
    offset: int = 0
    undo: object = None
    shared_key: bytes | None = None
    shared_cost: float = 0.0


class FingerprintDFS:
    """Serial root branches, one live expression workspace and sparse undo.

    Domain methods: reset, fingerprint, is_done, candidates(checkpoint),
    apply(operation, checkpoint), undo(token), release, audit(operations,cost).
    Candidate increments are fixed and nonnegative in this session. Widths
    cap roots strictly; complete primary-score ties can extend nonroot quotas. Historical
    locally completed nodes are not expanded again. Shared-pruned nodes can
    be rechecked on a cheaper arrival without borrowing a foreign route.
    """
    # Floor of the adaptive retained-size walk schedule; see MemorySampling.
    memory_interval = DEFAULT_MEMORY_SAMPLE_INTERVAL

    def __init__(self, space, *, root_width=4, branch_width=4,
                 deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000, max_transitions=None,
                 search_memory_limit=None, clock=time.monotonic, debug=False, budget=None,
                 operations=None, operation_costs=None, connections=None,
                 scorer=None, profile_scoring=False, include_ties=True, root_choices=None,
                 shared_states=None, capture_bindings=False):
        if type(capture_bindings) is not bool:
            raise ValueError("capture_bindings must be bool")
        self.space = space
        # Optional search-time side table: (source fingerprint, calculation) ->
        # occurrence bindings still valid before the undo, plus the terminal
        # tail captured for the current best goal state. Both are temporary
        # delivery aids; the owner releases them once the final route is audited.
        self.bindings = {} if capture_bindings else None
        self.completion_bindings = None
        self.completion_owner = None
        self.root_width = integer(root_width, "root_width", 1)
        self.branch_width = integer(branch_width, "branch_width", 1)
        if type(include_ties) is not bool:
            raise ValueError("include_ties must be bool")
        self.include_ties = include_ties
        # Private worker entry: descriptors are rebound in its own registry.
        # The owner already ranked this root and paid its expansion cost.
        self.root_choices = root_choices
        self.shared_states = shared_states
        self.shared_pruned = set()
        self.shared_rechecks = 0
        if type(debug) is not bool:
            raise ValueError("debug must be bool")
        self.debug = debug
        if type(profile_scoring) is not bool:
            raise ValueError("profile_scoring must be bool")
        self.scorer, self.profile_scoring = scorer, profile_scoring
        self.scoring_evaluations, self.scoring_seconds = 0, 0.0
        self.budget = budget if budget is not None else SearchBudget(deadline_seconds=deadline_seconds, max_states=max_states,
            max_transitions=max_transitions, search_memory_limit=search_memory_limit, clock=clock,
            memory_interval=self.memory_interval)
        self.nodes = {}
        self.connections = {} if connections is None else connections
        self.operations = OperationPool() if operations is None else operations
        self.operation_costs = [] if operation_costs is None else operation_costs
        self.relax_pending = set()
        self.deferred_ready = set()
        self.frames = []
        self.best_id = self.best_plan = self.first_solution = None
        self.root_id = None
        self.first_solution_scoring_seconds = self.first_solution_scoring_evaluations = None
        self.best_cost = math.inf
        self.curve, self.branch_ticks = [], []
        self.complete_plans = self.duplicates = self.dropped = 0
        self.relaxations = self.propagated = self.materializations = self.attempted = 0
        self.reconstructions = self.reconstruction_steps = self.cost_pruned = 0
        self.actual_width_histogram = Counter()
        self.extra_tied_choices = 0
        self.started = self.finished = False
        self.memory_roots = lambda: (self,)
        self.memory_sampler = None
        self.on_solution = None
        self._selection = None
        self._ran = False

    def checkpoint(self):
        self.budget.check(transition=True)

    def _sample_memory(self, *, force=False):
        """Request a retained-size walk; the schedule decides whether it runs.

        A forced request always walks, so the first solution and the final
        accounting are measured exactly. A scheduled request is deferred until
        the adaptive interval elapses, which is what keeps the total walk cost
        linear in the number of steps instead of quadratic.
        """
        if not force and not self.budget.memory_due():
            return
        if self.memory_sampler is None:
            self.budget.sample_memory(self.memory_roots())
        else:
            self.memory_sampler(force=force)

    def _intern(self, operation, cost):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("graph increments must be finite and nonnegative")
        identity = self.operations.intern(operation)
        # The first appended cost is authoritative. Re-deriving it on every
        # interned choice is a debug-only consistency diagnostic.
        if identity == len(self.operation_costs):
            self.operation_costs.append(cost)
        elif self.debug and self.operation_costs[identity] != cost:
            raise ValueError("one replay operation must have a fixed incremental cost")
        return identity

    def rank_candidates(self, width, *, root=False):
        """Rank the live state's operations with this session's scoring policy.

        Shared by serial expansion and the parallel owner's root discovery;
        both account for expansion work, score evaluations and boundary ties.
        The session must be started and its space positioned at the state.
        """
        self.budget.check()
        keep_ties = self.include_ties and not root
        self.budget.consume(SearchEvent("expand"))
        heap, boundary_ties, count = [], [], 0
        begin = getattr(self.scorer, "begin", None)
        if begin is not None:
            started = time.perf_counter() if self.profile_scoring else None
            begin()
            if started is not None:
                self.scoring_seconds += time.perf_counter() - started
        rank = getattr(self.space, "rank_candidates", None)
        selected = None if rank is None else rank(width, self.scorer, self.checkpoint,
                                                 profile_scoring=self.profile_scoring,
                                                 include_ties=keep_ties)
        if selected is not None:
            self.scoring_evaluations += selected.evaluations
            self.scoring_seconds += selected.scoring_seconds
            self.dropped += max(0, selected.candidate_count - len(selected.choices))
            return self._ranked(selected.choices, width)
        candidates = getattr(self.space, "search_candidates", self.space.candidates)
        stream = iter(candidates(self.checkpoint))
        try:
            for ordinal, (operation, cost) in enumerate(stream):
                self.budget.check()
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("graph increments must be finite and nonnegative")
                count += 1
                started = time.perf_counter() if self.profile_scoring else None
                try:
                    score = cost if self.scorer is None else self.scorer(self.space, operation, cost, self.checkpoint)
                finally:
                    if started is not None:
                        self.scoring_seconds += time.perf_counter() - started
                self.scoring_evaluations += 1
                validate_score(score)
                item = (reverse_score(score), -ordinal, operation, cost)
                if len(heap) < width:
                    heapq.heappush(heap, item)
                elif score == reverse_score(heap[0][0]):
                    boundary_ties.append(item)
                elif score < reverse_score(heap[0][0]):
                    old_boundary = heap[0][0]
                    removed = heapq.heapreplace(heap, item)
                    if heap[0][0] == old_boundary:
                        boundary_ties.append(removed)
                    else:
                        boundary_ties.clear()
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        selected = order_ties(self.space,
            [(reverse_score(score), -ordinal, op, cost) for score, ordinal, op, cost in heap + boundary_ties],
            self.checkpoint)
        if not keep_ties:
            selected = selected[:width]
        self.dropped += max(0, count - len(selected))
        return self._ranked(((op, cost) for _, _, op, cost in selected), width)

    def _ranked(self, choices, width):
        selected = [self._intern(op, cost) for op, cost in choices]
        self.actual_width_histogram[len(selected)] += 1
        self.extra_tied_choices += max(0, len(selected) - width)
        return selected

    def _consider_goal(self, identity):
        node = self.nodes[identity]
        cost = node.cost + (0.0 if node.completion is None else node.completion.seconds)
        if cost < self.best_cost:
            self.best_id, self.best_cost = identity, cost
            elapsed = self.budget.clock() - self.budget.started
            self.curve.append((elapsed, cost))
            if self.first_solution is None:
                self.first_solution = elapsed
                self.first_solution_scoring_seconds = self.scoring_seconds
                self.first_solution_scoring_evaluations = self.scoring_evaluations
            if self.on_solution is not None:
                self.on_solution()
            if self.shared_states is not None:
                self.shared_states.mark_feasible()

    def _relax(self, identity):
        queue, pending = deque([identity]), {identity}
        self.relax_pending.add(identity)
        while queue:
            source = queue.popleft()
            pending.remove(source)
            node = self.nodes[source]
            self._ready(source)
            for target in node.children:
                operation = self.connections[source, target]
                increment = self.operation_costs[operation]
                self.checkpoint()
                self.propagated += 1
                child = self.nodes[target]
                cost = node.cost + increment
                if cost < child.cost:
                    child.cost, child.parent = cost, source
                    self.relaxations += 1
                    if child.goal or child.completion is not None:
                        self._consider_goal(target)
                    if target not in pending:
                        queue.append(target)
                        pending.add(target)
                        self.relax_pending.add(target)
            self.relax_pending.discard(source)

    def _connect(self, source, target, operation, goal):
        parent = self.nodes[source]
        previous = self.connections.get((source, target))
        if previous is None:
            parent.children.append(target)
            self.connections[source, target] = operation
        elif self.operation_costs[operation] < self.operation_costs[previous]:
            self.connections[source, target] = operation
        operation = self.connections[source, target]
        increment = self.operation_costs[operation]
        # Reachability is independent of the best-parent chain. Never remove
        # an old parent's connection when a cheaper parent is discovered.
        cost = parent.cost + increment
        node = self.nodes.get(target)
        fresh = node is None
        if fresh:
            node = self.nodes[target] = GraphNode(cost, source, goal=goal, closed=goal)
        else:
            self.duplicates += 1
            if self.debug and node.goal != goal:
                raise RuntimeError("fingerprint merged different terminal states")
            if cost < node.cost:
                node.cost, node.parent = cost, source
                self.relaxations += 1
                if node.goal or node.completion is not None:
                    self._consider_goal(target)
                self._relax(target)
        if goal:
            if fresh or not node.goal:
                self.complete_plans += 1
            node.goal = True
            self._consider_goal(target)
        if not goal and target in self.shared_pruned:
            # A cheaper arrival must not inherit an empty local future from
            # an earlier cross-process dominance hit. Recheck its bound.
            self.shared_pruned.remove(target)
            self.deferred_ready.discard(target)
            self.shared_rechecks += 1
            node.closed = False
            return True
        return fresh

    def route(self, identity):
        """Return node and operation IDs along the current best-parent chain."""
        nodes, operations, seen = [], [], set()
        while identity is not None:
            if identity in seen:
                raise RuntimeError("best-parent cycle: selected fingerprint route cannot be replayed")
            seen.add(identity)
            nodes.append(identity)
            node = self.nodes[identity]
            if node.parent is not None:
                operations.append(self.connections[node.parent, identity])
            identity = node.parent
        return list(reversed(nodes)), list(reversed(operations))

    def _ready(self, identity):
        node = self.nodes[identity]
        if identity in self.shared_pruned or any(
                node.cost + self.operation_costs[op] < self.best_cost for op in node.deferred):
            self.deferred_ready.add(identity)

    def _capture_completion(self, identity, completion):
        """Record the terminal tail's own bindings while its state is live."""
        space, captured = self.space, {}
        for calculation in completion.calculations:
            if calculation in space.groups:
                captured[calculation] = space.occurrence_bindings(
                    calculation, self.checkpoint)
        self.completion_bindings = captured
        self.completion_owner = identity

    def _expand_binding(self, rows):
        """Compacted row indices to physical scenes, sorted by scene, or None."""
        if not rows:
            return None
        members = self.space.scene_members
        if members is None:
            return tuple(sorted(rows))
        return tuple(sorted((physical, *row[1:]) for row in rows
                            for physical in members[row[0]]))

    def _route_bindings(self, nodes, route, total):
        """Search-time bindings aligned with the route operations, or all None.

        Occurrence bindings belong to the calculation, independent of the pooled
        operation ID that happens to carry the same recipe.
        """
        if self.bindings is None:
            return (None,) * total
        values = [self._expand_binding(
            self.bindings.get((nodes[i], self.operations[route[i]].calculation)))
            for i in range(len(route))]
        # An expanded terminal tail has no search-time apply; those entries stay
        # None and the audit falls back to independent matching per operation.
        return tuple(values) + (None,) * (total - len(values))

    def _selected(self):
        """Winning route operations plus one aligned binding slot per operation.

        Delivery asks for the operations through selected_operations() and for
        the bindings separately, so the route walk is memoized: both callers see
        one materialization of the same winner.
        """
        if self._selection is not None:
            return self._selection
        nodes, route = self.route(self.best_id)
        prefix = tuple(self.operations[i] for i in route)
        completion = self.nodes[self.best_id].completion
        if completion is None:
            return prefix, self._route_bindings(nodes, route, len(route))
        from .terminal_completion import expand_completion
        self.space._terminal_completions.expansions += 1
        expanded = expand_completion(completion, prefix, self.space.registry)
        values = list(self._route_bindings(nodes, route, len(route)))
        # The incumbent's own terminal tail, when this delivery owns it.
        tail = (self.completion_bindings
                if self.best_id == self.completion_owner else None)
        for operation in expanded[len(route):]:
            values.append(None if tail is None else
                          self._expand_binding(tail.get(operation.calculation)))
        self._selection = (expanded, tuple(values))
        return self._selection

    def selected_operations(self):
        """Materialize the winning tail once at the route delivery boundary."""
        return self._selected()[0]

    def selected_operations_and_bindings(self):
        """Winning operations and search-time bindings, aligned one-to-one.

        The operations are taken through selected_operations() so the delivery
        boundary stays observable to the fixed-work benchmark's phase timer.
        """
        operations = self.selected_operations()
        return operations, self._selected()[1]

    def _walk(self, stop_on_first_solution=False):
        branch = None
        memory_due = self.budget.memory_due
        while self.frames:
            self.budget.check()
            # The schedule amortizes each complete walk over the states it
            # accounted for; a request outside it costs nothing.
            if memory_due():
                self._sample_memory()
            frame = self.frames[-1]
            # A restored nonroot frame is still a nonroot: it must not use
            # the worker's fixed initial root recipe or the strict root quota.
            root = frame.identity == self.root_id
            if frame.choices is None:
                node = self.nodes[frame.identity]
                complete = getattr(self.space, 'terminal_completion', None)
                completion = (node.completion if node.completion is not None else
                              complete(self.checkpoint) if complete is not None and not root else None)
                if completion is not None:
                    if node.completion is None:
                        self.complete_plans += 1
                    node.completion = completion
                    frame.choices = []
                    # This state is live and about to become the incumbent, so
                    # its terminal families can hand over their own bindings.
                    if (self.bindings is not None
                            and node.cost + completion.seconds < self.best_cost):
                        self._capture_completion(frame.identity, completion)
                    self._consider_goal(frame.identity)
                    if stop_on_first_solution:
                        return
                else:
                    frame.choices = ([self._intern(op, cost) for op, cost in self.root_choices]
                                     if root and self.root_choices is not None else
                                     self.rank_candidates(self.root_width if root else self.branch_width, root=root))
            if self.shared_states is not None and frame.shared_key is None:
                frame.shared_key = self.shared_states.key(self.space,
                    tuple(self.operations[i] for i in frame.choices))
                frame.shared_cost = self.nodes[frame.identity].cost
                if self.shared_states.dominated(frame.shared_key, frame.shared_cost):
                    self.shared_pruned.add(frame.identity)
                    self.frames.pop()
                    if frame.undo is not None:
                        self.space.undo(frame.undo)
                    continue
            if not root and branch is not None:
                self.branch_ticks[branch] += 1
            if frame.offset >= len(frame.choices):
                self.frames.pop()
                self.nodes[frame.identity].closed = True
                if self.shared_states is not None:
                    self.shared_states.publish(frame.shared_key, frame.shared_cost)
                if frame.undo is not None:
                    self.space.undo(frame.undo)
                    if self.debug and self.space.fingerprint() != self.frames[-1].identity:
                        raise RuntimeError("DFS undo reached a different parent state")
                continue
            operation = frame.choices[frame.offset]
            frame.offset += 1
            if root:
                branch = len(self.branch_ticks)
                self.branch_ticks.append(1)
            node = self.nodes[frame.identity]
            if node.cost + self.operation_costs[operation] >= self.best_cost:
                if operation not in node.deferred:
                    node.deferred.append(operation)
                self.cost_pruned += 1
                continue
            self.checkpoint()
            if self.bindings is not None:
                # Capture while this frame's state is still live and before the
                # operation mutates it. The accessor refreshes the same index
                # apply() would have refreshed, so search work is unchanged.
                calculation = self.operations[operation].calculation
                self.bindings[frame.identity, calculation] = self.space.occurrence_bindings(
                    calculation, self.checkpoint)
            token = self.space.apply(self.operations[operation], self.checkpoint)
            self.materializations += 1
            self.attempted += 1
            try:
                target, goal = self.space.fingerprint(), self.space.is_done()
                fresh = self._connect(frame.identity, target, operation, goal)
            except BaseException:
                self.space.undo(token)
                raise
            if fresh and not goal:
                self.frames.append(GraphFrame(target, undo=token))
            else:
                # A duplicate's future is owned by its existing active frame
                # or historical connections. Only numeric labels are relaxed.
                self.space.undo(token)
            if goal and stop_on_first_solution:
                return
        self._sample_memory()

    def start(self):
        if self.started:
            return
        self.budget.check()
        self.space.reset()
        root = self.space.fingerprint()
        self.root_id = root
        goal = self.space.is_done()
        self.nodes[root] = GraphNode(0.0, goal=goal, closed=goal)
        self.started = True
        self._sample_memory()
        if goal:
            self.complete_plans = 1
            self._consider_goal(root)
        else:
            self.frames.append(GraphFrame(root))

    def advance(self, *, first_solution=False):
        """Pause at the first goal with the live workspace and undo chain intact."""
        self.start()
        if self.finished or (first_solution and self.best_id is not None):
            return
        self._walk(first_solution)
        if first_solution and self.best_id is not None:
            return
        while self.deferred_ready:
            self.budget.check()
            source = min(self.deferred_ready)
            self.deferred_ready.remove(source)
            node = self.nodes[source]
            external = source in self.shared_pruned
            choices = None if external else [op for op in node.deferred
                if node.cost + self.operation_costs[op] < self.best_cost]
            if not external and not choices:
                continue
            _, prefix = self.route(source)
            self.space.reset()
            self.reconstructions += 1
            for operation in prefix:
                self.checkpoint()
                self.space.apply(self.operations[operation], self.checkpoint)
                self.reconstruction_steps += 1
            if self.debug and self.space.fingerprint() != source:
                raise RuntimeError("deferred first exploration restored a different parent")
            if external:
                self.shared_pruned.remove(source)
                self.shared_rechecks += 1
            else:
                node.deferred = array("Q", (op for op in node.deferred if op not in set(choices)))
            self.frames.append(GraphFrame(source, choices))
            self._walk(first_solution)
            if first_solution and self.best_id is not None:
                return
        self.finished = True

    def finish(self, reason="completed", limited=False, *, audit=True):
        for frame in self.frames:
            self.nodes[frame.identity].closed = True
        self.frames.clear()
        self.space.release()
        begin = getattr(self.scorer, "begin", None)
        if begin is not None:
            begin()
        if self.relax_pending:
            self._settle_labels()
        elapsed = self.budget.clock() - self.budget.started
        audit_seconds = 0.0
        if audit and self.best_id is not None:
            started = self.budget.clock()
            self.best_plan = self.space.audit(self.selected_operations(), self.best_cost)
            audit_seconds = self.budget.clock() - started
        statistics = dict(expanded_states=self.budget.expanded,
            evaluated_transitions=self.budget.transitions, duplicates=self.duplicates,
            complete_plans=self.complete_plans, first_solution_seconds=self.first_solution,
            first_solution_compute_seconds=self.curve[0][1] if self.curve else None,
            first_solution_scoring_seconds=self.first_solution_scoring_seconds if self.profile_scoring else None,
            first_solution_scoring_evaluations=self.first_solution_scoring_evaluations,
            improvement_curve=tuple(self.curve), elapsed_seconds=elapsed,
            final_audit_seconds=audit_seconds, sampled_peak_search_bytes=self.budget.sampled_peak_memory,
            retained_states=len(self.nodes), retained_connections=len(self.connections),
            branch_ticks=tuple(self.branch_ticks), cost_relaxations=self.relaxations,
            propagated_connections=self.propagated, state_replays=self.reconstructions,
            deferred_reconstruction_steps=self.reconstruction_steps, cost_pruned=self.cost_pruned,
            historical_reexpansions=0,
            selected_materializations=self.materializations, pooled_operations=len(self.operations),
            attempted_operations=self.attempted, truncated_candidates=self.dropped,
            actual_width_histogram=dict(sorted(self.actual_width_histogram.items())),
            extra_tied_choices=self.extra_tied_choices,
            scoring_evaluations=self.scoring_evaluations,
            scoring_seconds=self.scoring_seconds if self.profile_scoring else None,
            logical_best_seconds=self.best_cost if self.best_id is not None else None)
        if self.shared_states is not None:
            statistics.update(self.shared_states.statistics())
            statistics['shared_state_rechecks'] = self.shared_rechecks
        terminal_stats = getattr(self.space, 'terminal_statistics', None)
        if terminal_stats is not None:
            statistics.update(terminal_stats())
        return SearchResult("FEASIBLE" if self.best_id is not None else "UNSOLVED", None,
            self.budget.expanded, limited, stop_reason=reason, lower_bound_seconds=0.0,
            statistics=statistics)

    def run(self):
        if self._ran:
            raise RuntimeError("a graph DFS session can only run once")
        self._ran = True
        reason, limited = "completed", False
        try:
            self.advance()
        except BudgetExhausted as exc:
            reason, limited = exc.reason, True
        except BaseException:
            self.space.release()
            self.frames.clear()
            raise
        return self.finish(reason, limited)

    def _settle_labels(self):
        queue, pending = deque(self.relax_pending), set(self.relax_pending)
        while queue:
            source = queue.popleft()
            pending.remove(source)
            node = self.nodes[source]
            self._ready(source)
            for target in node.children:
                operation = self.connections[source, target]
                increment = self.operation_costs[operation]
                child = self.nodes[target]
                cost = node.cost + increment
                if cost < child.cost:
                    child.cost, child.parent = cost, source
                    self.relaxations += 1
                    if child.goal:
                        self._consider_goal(target)
                    if target not in pending:
                        pending.add(target)
                        queue.append(target)
        self.relax_pending.clear()
