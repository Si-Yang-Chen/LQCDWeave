"""Bounded scheduling of a selected computation DAG with unchanged work.

Nodes are existing complete-step action fragments. Data dependencies and the
existing per-Term update order constrain their placement. The scheduler never
chooses a different contraction, prepares an input twice, or spills data.

Candidate selection intersects ready bits with each release target directly.
The previous whole-ready-set scan is retained for diagnostics through
_LEGACY_CHOICES and verify_schedule_equivalence.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cached_property, cmp_to_key
from heapq import heapify, heappop, heappush, nsmallest
import math
import time
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import Action, MemoryPoint, SceneStoragePlan, StorageEvent
from ..resources import audit_resource_trajectory
from ..sparse_workspace import prepare_coordinate_batch_workspace
from ..state import _nbytes
from .delivery import executable_delivery
from .budget import DEFAULT_PLANNING_SECONDS


@dataclass(frozen=True, slots=True)
class _Node:
    """One selected contraction, with fixed input and teardown dependencies."""
    actions: tuple
    parents: frozenset
    references: tuple
    workspace: int
    inputs: tuple
    touched: tuple
    raw_references: tuple


@dataclass(frozen=True, slots=True)
class _ReleaseGoal:
    domain: str
    key: str
    nbytes: int
    required: int
    computations: int


def _compare_goals(left, right):
    if (left.computations == 0) != (right.computations == 0):
        return -1 if left.computations == 0 else 1
    a = left.nbytes * (right.computations or 1)
    b = right.nbytes * (left.computations or 1)
    if a != b:
        return -1 if a > b else 1
    a, b = (left.domain, left.key), (right.domain, right.key)
    return (a > b) - (a < b)


_GOAL_KEY = cmp_to_key(_compare_goals)


# Candidate-selection implementations. The default walks a target requirement
# mask bit by bit; the legacy one rescans the whole ready set with a full-width
# integer per target. The equivalence check flips this switch to re-derive the
# same schedule with the pre-optimization rule and compare field by field.
_LEGACY_CHOICES = False


def _legacy_choices(frontier, width, checkpoint):
    """Candidate sequence produced by the pre-optimization ready-set scan."""
    seen = set()
    if len(seen) < width:
        limit, visited = width, set()
        while checkpoint():
            targets = frontier.release_goals(limit, checkpoint)
            for target in targets:
                identity = (target.domain, target.key)
                if identity in visited:
                    continue
                visited.add(identity)
                eligible = (i for i in frontier.ready if target.required & (1 << i))
                for index in nsmallest(width, eligible):
                    if index not in seen:
                        seen.add(index)
                        yield index, target
                        if len(seen) == width:
                            return
            if len(targets) < limit:
                break
            limit *= 2
    if len(seen) < width:
        for index in nsmallest(width, frontier.ready):
            if index not in seen:
                seen.add(index)
                yield index, None
                if len(seen) == width:
                    return


class _ReleaseRanking:
    """Cached priorities with watches on remaining prerequisite computations.

    Watches include indirect prerequisites, not just the direct consumers.
    They are registered when data becomes resident and retained across undo;
    inactive goal bits are masked out. Normal replay never creates this index.
    """
    def __init__(self, frontier):
        self.frontier = frontier
        self.ids, self.definitions, self.entries, self.returned = {}, [], {}, {}
        self.watches = [0] * len(frontier.dag.nodes)
        self.active, self.serial, self.heap = 0, 0, []
        self.activations = self.updates = self.reuses = self.registrations = self.compactions = 0
        # A diagnostic may first ask for a ranking after already advancing.
        # Include its past prerequisites so those steps can also be undone.
        for key in frontier.live - frontier.dag.outputs:
            self._sync(('tensor', key), include_done=True)
        for key in frontier.raw:
            self._sync(('block', key), include_done=True)

    def _store(self, identity, goal):
        self.serial += 1
        entry = (_GOAL_KEY(goal), self.serial, identity)
        self.entries[identity] = entry
        heappush(self.heap, entry)
        self.frontier.target_evaluations += 1

    def _sync(self, key, *, include_done=False):
        f, (domain, name) = self.frontier, key
        identity = self.ids.get(key)
        resident = ((name in f.live and name not in f.dag.outputs) if domain == 'tensor'
                    else name in f.raw)
        if not resident:
            if identity is not None and identity in self.entries:
                del self.entries[identity]
                self.returned.pop(identity, None)
                self.active &= ~(1 << identity)
            return
        if identity is not None and identity in self.entries:
            return  # membership changed elsewhere; reuse this priority
        if identity is None:
            if domain == 'tensor':
                size = f.dag.sizes.get(name, 0)
                mask = f.dag.release_masks.get(name, 0)
            else:
                block = f.dag.physical.blocks[name]
                size = block.host_bytes + block.device_bytes
                mask = f.dag.raw_release_masks.get(name, 0)
            if size <= 0 or not mask:
                return
            identity = len(self.definitions)
            self.ids[key] = identity
            self.definitions.append((domain, name, size, mask))
        domain, name, size, mask = self.definitions[identity]
        remaining = mask & ~f.done
        if not remaining:
            return
        computations = remaining & f.dag.computation_mask
        self._store(identity, _ReleaseGoal(domain, name, size, mask, computations.bit_count()))
        self.activations += 1
        bit = 1 << identity
        self.active |= bit
        pending = mask & f.dag.computation_mask if include_done else computations
        while pending:
            first = pending & -pending
            index = first.bit_length() - 1
            if not self.watches[index] & bit:
                self.watches[index] |= bit
                self.registrations += 1
            pending ^= first

    def update(self, index, tensors, delta):
        f, previous = self.frontier, self.active
        for key in tensors:
            self._sync(('tensor', key))
        raw = {block for key in tensors if key in f.dag.physical.preparations
               for block in f.dag.physical.preparations[key].blocks}
        for key in raw:
            self._sync(('block', key))
        # Newly activated goals were priced against the new frontier already.
        affected = self.watches[index] & previous & self.active
        while affected:
            first = affected & -affected
            identity = first.bit_length() - 1
            goal = self.entries[identity][0].obj
            self._store(identity, _ReleaseGoal(goal.domain, goal.key, goal.nbytes,
                goal.required, goal.computations + delta))
            self.updates += 1
            affected ^= first
        if len(self.heap) > 2 * len(self.entries) + 128:
            self.heap = list(self.entries.values())
            heapify(self.heap)
            self.compactions += 1

    def goals(self, width, checkpoint):
        selected, goals = [], []
        try:
            while self.heap and len(selected) < width and checkpoint():
                entry = heappop(self.heap)
                identity = entry[2]
                if self.entries.get(identity) is not entry:
                    continue
                selected.append(entry)
                if self.returned.get(identity) is entry:
                    self.reuses += 1
                self.returned[identity] = entry
                goal = entry[0].obj
                goals.append(_ReleaseGoal(goal.domain, goal.key, goal.nbytes,
                    goal.required & ~self.frontier.done, goal.computations))
        finally:
            for entry in selected:
                heappush(self.heap, entry)
        return goals


@dataclass(slots=True)
class _AdvanceUndo:
    index: int
    counters: tuple
    fresh: tuple
    opened: list
    released: list
    closed: list


def _validate_computation_fragment(actions):
    if not actions or any(action.tensor != actions[0].tensor for action in actions):
        raise ValueError("a DAG node must be one selected shared calculation")
    if any(action.kind in ("load", "release", "read_block", "close_block")
           or action.resource is not None for action in actions):
        raise ValueError("DAG input must contain only audited computations")


class _Dag:
    def __init__(self, state, snippets, workspaces, physical, *, audited_computations=False):
        self.state, self.physical = state, physical
        if len({(block.dataset, block.selection) for block in physical.blocks.values()}) != len(physical.blocks):
            raise ValueError('one physical selection must have one owner')
        self.outputs = frozenset(state.results)
        self.sizes = {key: (_nbytes(spec) if key not in state.problem.sources
                           else state._snapshot_bytes(key))
                      for key, spec in state.catalog.items()}
        self.nodes, self.children = [], []
        self.preparation_blocks = {key: (prep.blocks if len(prep.blocks) < 2 else tuple(sorted(prep.blocks)))
                                   for key, prep in physical.preparations.items()}
        self.raw_sources = defaultdict(set)
        for key, prep in physical.preparations.items():
            for raw in prep.blocks:
                self.raw_sources[raw].add(key)
        producers, term_updates = {}, {}
        if len(snippets) != len(workspaces):
            raise ValueError("one workspace bound is required per selected step")
        for index, ((begin, end), workspace) in enumerate(zip(snippets, workspaces)):
            actions = tuple(state.actions[begin:end])
            # The selected-path producer and batch emitter construct each
            # fragment from one accepted computation. Recheck foreign records
            # and diagnostics, rather than every occurrence of that recipe.
            if not audited_computations:
                _validate_computation_fragment(actions)
            parents, references = set(), Counter()
            inputs = None
            for action in actions:
                # Include descriptor operands even on a cache reuse. Their Term
                # substitutions must finish before those symbolic factors die.
                keys = set(action.operands)
                if action.reused:
                    keys.add(action.tensor)
                if inputs is None:
                    inputs = (action.tensor,) if action.reused else tuple(sorted(keys))
                references.update(keys)
                for key in keys:
                    if key == actions[0].tensor and not actions[0].reused:
                        continue  # later occurrences reuse this node's result
                    if key in producers:
                        parents.add(producers[key])
                    elif key not in state.problem.sources:
                        raise ValueError("selected DAG consumes data before its producer")
                previous = term_updates.get(action.term_index)
                if previous is not None and previous != index:
                    parents.add(previous)
                term_updates[action.term_index] = index
            parents.discard(index)
            first = actions[0]
            if not first.reused:
                if first.tensor in producers:
                    raise ValueError("selected DAG repeats a computation")
                producers[first.tensor] = index
            touched = set(references) | {first.tensor}
            raw_references = {}
            for key, count in references.items():
                for raw in self.preparation_blocks.get(key, ()):
                    raw_references[raw] = raw_references.get(raw, 0) + count
            self.nodes.append(_Node(actions, frozenset(parents), tuple(references.items()),
                                    0 if first.reused else workspace, inputs,
                                    tuple(sorted(touched)), tuple(sorted(raw_references.items()))))
            self.children.append([])
        for index, node in enumerate(self.nodes):
            for parent in node.parents:
                self.children[parent].append(index)
        self.references = Counter()
        for node in self.nodes:
            self.references.update(dict(node.references))
        # A raw owner's count is the sum of the same logical source uses.
        # Updating it alongside tensor counts avoids rescanning shared sources
        # on every speculative contraction and again after backtracking.
        self.raw_references = Counter()
        self.raw_borrowed_outputs = {}
        for raw, keys in self.raw_sources.items():
            self.raw_references[raw] = sum(self.references[key] for key in keys)
            retained = keys & self.outputs
            if retained:
                self.raw_borrowed_outputs[raw] = frozenset(
                    key for key in retained if self.sizes.get(key, 0) == 0)

    def _prepare_release_ranking(self):
        """Build transitive target metadata only for the existing selector."""
        ancestors, consumers, computations = [], defaultdict(int), 0
        for index, node in enumerate(self.nodes):
            required = 1 << index
            for parent in node.parents:
                required |= ancestors[parent]
            ancestors.append(required)
            if not node.actions[0].reused:
                computations |= 1 << index
            for key, _ in node.references:
                consumers[key] |= required
        self.release_masks = dict(consumers)
        self.raw_release_masks = {raw: _union_masks(consumers[key] for key in keys)
                                  for raw, keys in self.raw_sources.items()
                                  if not keys & self.outputs}
        self.computation_mask = computations

    @cached_property
    def release_masks(self):
        self._prepare_release_ranking()
        return self.__dict__['release_masks']

    @cached_property
    def raw_release_masks(self):
        self._prepare_release_ranking()
        return self.__dict__['raw_release_masks']

    @cached_property
    def computation_mask(self):
        self._prepare_release_ranking()
        return self.__dict__['computation_mask']


def _union_masks(masks):
    result = 0
    for mask in masks:
        result |= mask
    return result


def _first_indices(mask, width):
    """The same ascending alternatives as nsmallest, without scanning ready work."""
    for _ in range(width):
        if not mask:
            return
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class _Frontier:
    """Only the live frontier; speculative scheduling holds no expression tree."""
    def __init__(self, dag):
        self.dag = dag
        self.remaining = dag.references.copy()
        self.raw_remaining = dag.raw_references.copy()
        self.indegree = [len(node.parents) for node in dag.nodes]
        self.ready = {i for i, count in enumerate(self.indegree) if not count}
        self.ready_mask = sum(1 << i for i in self.ready)
        self.live, self.prepared, self.raw, self.opened = set(), set(), set(), set()
        self.host = self.device = self.peak_host = self.peak_device = 0
        self.raw_host = self.raw_device = 0
        self.record = None
        self.done, self.order = 0, []
        self.target_evaluations = 0
        self.ranking = None
        self._point()

    def release_goals(self, width=1, checkpoint=lambda: True):
        """Maximize resident bytes / remaining new computations to release.

        Borrowed views have zero snapshot bytes; their raw owner is scored
        once using all its consumers. Final outputs remain protected.
        """
        if self.ranking is None:
            self.ranking = _ReleaseRanking(self)
        return self.ranking.goals(width, checkpoint)

    def release_goal(self, checkpoint=lambda: True):
        goals = self.release_goals(checkpoint=checkpoint)
        return goals[0] if goals else None

    def choices(self, width, checkpoint):
        """Rank alternatives lazily, on return to this unchanged frontier."""
        if _LEGACY_CHOICES:
            return _legacy_choices(self, width, checkpoint)
        return self._ranked_choices(width, checkpoint)


    def _ranked_choices(self, width, checkpoint):
        """Requirement-bit walk; identical candidates in identical order."""
        seen = set()
        if len(seen) < width:
            # Several allocations can have the same consumers. They must not
            # occupy distinct operation slots and hide lower-ranked goals.
            limit, visited = width, set()
            while checkpoint():
                targets = self.release_goals(limit, checkpoint)
                for target in targets:
                    identity = (target.domain, target.key)
                    if identity in visited:
                        continue
                    visited.add(identity)
                    for index in _first_indices(self.ready_mask & target.required, width):
                        if index not in seen:
                            seen.add(index)
                            yield index, target
                            if len(seen) == width:
                                return
                if len(targets) < limit:
                    break
                limit *= 2
        if len(seen) < width:
            for index in _first_indices(self.ready_mask, width):
                if index not in seen:
                    seen.add(index)
                    yield index, None
                    if len(seen) == width:
                        return

    def _point(self, host_workspace=0, device_workspace=0, *, phase='resident', output=False):
        p = self.dag.physical
        self.peak_host = max(self.peak_host, self.host + host_workspace + p.reserved_host_bytes)
        self.peak_device = max(self.peak_device, self.device + device_workspace + p.reserved_device_bytes)
        if self.record is not None:
            self.record(phase, host_workspace, device_workspace, output)

    def _tensor(self, key, sign):
        size = self.dag.sizes.get(key, 0)
        if self.dag.physical.gpu:
            self.device += sign * size
        else:
            self.host += sign * size

    def advance(self, index, actions=None, *, eager_inputs=(), deferred_touched=None, release=True,
                reversible=False):
        if index not in self.ready:
            raise ValueError("DAG scheduling requires a ready node")
        node, p = self.dag.nodes[index], self.dag.physical
        first = node.actions[0]
        fresh = (tuple(sorted((set(node.inputs) | set(eager_inputs)) - self.live))
                 if eager_inputs else tuple(key for key in node.inputs if key not in self.live))
        undo = (_AdvanceUndo(index,
            (self.host, self.device, self.peak_host, self.peak_device, self.raw_host, self.raw_device),
            fresh, [], [], []) if reversible else None)
        emit = (lambda action: None) if actions is None else actions.append
        for key in fresh:
            if key not in p.preparations or key in self.prepared:
                raise ValueError("fixed DAG cannot reload or recompute a released input")
            prep = p.preparations[key]
            for raw in self.dag.preparation_blocks[key]:
                if raw in self.raw:
                    continue
                if raw in self.opened:
                    raise ValueError("fixed DAG cannot read an original block twice")
                block = p.blocks[raw]
                self.raw.add(raw)
                self.opened.add(raw)
                if undo is not None:
                    undo.opened.append(raw)
                self.host += block.host_bytes
                self.device += block.device_bytes
                self.raw_host += block.host_bytes
                self.raw_device += block.device_bytes
                emit(p.read_actions[raw])
                self._point(block.staging_host_bytes, block.staging_device_bytes, phase='read:' + raw)
            self.prepared.add(key)
            self.live.add(key)
            self._tensor(key, 1)
            emit(p.load_actions[key])
            self._point(prep.workspace_host_bytes, prep.workspace_device_bytes, phase='materialize:' + key)
        if not first.reused:
            self.live.add(first.tensor)
            self._tensor(first.tensor, 1)
        elif first.tensor not in self.live:
            raise ValueError("a reused DAG result is unavailable")
        for action in node.actions:
            emit(action)
            workspace = 0 if action.reused else node.workspace
            self._point(0 if p.gpu else workspace, workspace if p.gpu else 0, phase=action.kind)
        for key, count in node.references:
            self.remaining[key] -= count
        for raw, count in node.raw_references:
            self.raw_remaining[raw] -= count
        touched = node.touched
        raw_candidates = node.raw_references
        if deferred_touched is not None:
            deferred_touched.update(touched)
            touched = tuple(sorted(deferred_touched)) if release else ()
            raw_candidates = ((raw, 0) for raw in sorted({raw for key in touched
                for raw in self.dag.preparation_blocks.get(key, ())}))
        for key in touched:
            if key in self.live and key not in self.dag.outputs and self.remaining[key] == 0:
                emit(Action("release", key))
                self.live.remove(key)
                if undo is not None:
                    undo.released.append(key)
                self._tensor(key, -1)
                self._point(phase='release:' + key)
        for raw, _ in raw_candidates:
            if (raw in self.raw and self.raw_remaining[raw] == 0
                    and (raw not in self.dag.raw_borrowed_outputs
                         or self.dag.raw_borrowed_outputs[raw].isdisjoint(self.live))):
                emit(Action("close_block", raw))
                block = p.blocks[raw]
                self.host -= block.host_bytes
                self.device -= block.device_bytes
                self.raw_host -= block.host_bytes
                self.raw_device -= block.device_bytes
                self.raw.remove(raw)
                if undo is not None:
                    undo.closed.append(raw)
                self._point(phase='close:' + raw)
        self.ready.remove(index)
        self.ready_mask &= ~(1 << index)
        self.done |= 1 << index
        self.order.append(index)
        for child in self.dag.children[index]:
            self.indegree[child] -= 1
            if self.indegree[child] == 0:
                self.ready.add(child)
                self.ready_mask |= 1 << child
        if self.ranking is not None:
            self.ranking.update(index, set(touched) | set(fresh), -1)

        return undo

    def undo(self, change):
        """Restore only this step's mutations, without copying the DAG/frontier."""
        index = change.index
        node = self.dag.nodes[index]
        for child in self.dag.children[index]:
            if self.indegree[child] == 0:
                self.ready.remove(child)
                self.ready_mask &= ~(1 << child)
            self.indegree[child] += 1
        self.ready.add(index)
        self.ready_mask |= 1 << index
        self.done ^= 1 << index
        self.order.pop()
        for key, count in node.references:
            self.remaining[key] += count
        for raw, count in node.raw_references:
            self.raw_remaining[raw] += count
        self.live.update(change.released)
        self.live.difference_update(change.fresh)
        if not node.actions[0].reused:
            self.live.discard(node.actions[0].tensor)
        self.prepared.difference_update(change.fresh)
        self.raw.update(change.closed)
        self.raw.difference_update(change.opened)
        self.opened.difference_update(change.opened)
        (self.host, self.device, self.peak_host, self.peak_device,
         self.raw_host, self.raw_device) = change.counters
        if self.ranking is not None:
            touched = set(node.touched) | set(change.fresh)
            self.ranking.update(index, touched, 1)


    def finish(self):
        copies = sum(self.dag.sizes[key] for key in self.dag.state.results)
        self._point(0 if self.dag.physical.gpu else copies,
                    copies if self.dag.physical.gpu else 0, phase='output_copy', output=True)


class _ExecutableActions(list):
    """Emit the final ABI while the frontier accounts physical allocations."""
    def __init__(self, physical):
        super().__init__()
        self.physical = physical
        self.events, self.costs, self.open_blocks = [], {}, set()
        self.last_index = 0

    def append(self, action):
        # Used only for accepted computations. Search validates their costs;
        # physical load costs come from the fixed profile. Independent replay
        # retains per-action cost validation for diagnostics and foreign traces.
        if action.kind == 'read_block':
            self.last_index = len(self)
            self.events.append(StorageEvent(self.last_index, 'read', action.tensor))
            self.open_blocks.add(action.tensor)
        elif action.kind == 'close_block':
            self.last_index = max(0, len(self)-1)
            self.events.append(StorageEvent(self.last_index, 'release', action.tensor))
            self.open_blocks.remove(action.tensor)
        else:
            self.last_index = len(self)
            if action.kind == 'load':
                self.costs[self.last_index] = self.physical.preparation_costs[action.tensor]
            super().append(action)


def _plan(dag, order, *, audited_computations=False, debug=False, atomic_groups=()):
    frontier = _Frontier(dag)
    points = []
    p = dag.physical
    direct = audited_computations and not debug
    actions = _ExecutableActions(p) if direct else []
    if audited_computations:
        def record(phase, wh, wd, output):
            index = actions.last_index if direct else len(actions)-1
            points.append(MemoryPoint(len(actions) if output else index, phase,
                frontier.raw_host, frontier.raw_device,
                frontier.host-frontier.raw_host, frontier.device-frontier.raw_device,
                wh, wd, p.reserved_host_bytes, p.reserved_device_bytes))
        frontier.record = record
    atomic = {}
    for group in atomic_groups:
        touched = set()
        inputs = {key for index in group for action in dag.nodes[index].actions for key in action.operands}
        for position, index in enumerate(group):
            if index in atomic:
                raise ValueError('a computation cannot belong to two atomic groups')
            atomic[index] = (inputs if position == 0 else (), touched, position == len(group)-1)
    for index in order:
        eager, touched, release = atomic.get(index, ((), None, True))
        frontier.advance(index, actions, eager_inputs=eager, deferred_touched=touched, release=release)
    if frontier.opened != dag.physical.mandatory_blocks:
        raise ValueError("fixed DAG changed its mandatory original reads")
    if len(frontier.order) != len(dag.nodes) or not dag.outputs <= frontier.live:
        raise ValueError('fixed DAG does not complete all computations and outputs')
    frontier.finish()
    state = replace(dag.state, actions=tuple(actions), loaded=frozenset(frontier.live),
                    seconds=sum(action.seconds for action in actions),
                    peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
    if not audited_computations or debug:
        reference = audit_resource_trajectory(state, p.blocks, p.preparations, profile=p.profile,
            memory_limit=p.memory_limit, host_memory_limit=p.host_memory_limit,
            disk_limit_bytes=p.disk_limit_bytes, reserved_host_bytes=p.reserved_host_bytes,
            reserved_device_bytes=p.reserved_device_bytes)
    if not audited_computations:
        return executable_delivery(reference)
    if direct:
        costs, events = actions.costs, actions.events
        events.extend(StorageEvent(len(actions), 'release', key) for key in sorted(actions.open_blocks))
    else:
        costs, events = {}, []
        for index, action in enumerate(actions):
            if not math.isfinite(action.seconds) or action.seconds < 0:
                raise ValueError('all action costs must be finite and nonnegative')
            if action.kind == 'read_block':
                events.append(StorageEvent(index, 'read', action.tensor))
            elif action.kind == 'close_block':
                events.append(StorageEvent(index, 'release', action.tensor))
            elif action.kind == 'load':
                costs[index] = p.preparations[action.tensor].seconds(p.profile)
    feasible = (frontier.peak_host <= p.memory_limit if not p.gpu else
                frontier.peak_device <= p.memory_limit)
    feasible = feasible and (p.host_memory_limit is None or frontier.peak_host <= p.host_memory_limit)
    if not p.gpu and frontier.peak_device:
        raise ValueError('CPU trajectory contains device allocations')
    plan = SceneStoragePlan(state, MappingProxyType(dict(p.blocks)), MappingProxyType(dict(p.preparations)),
        tuple(events), tuple(points), sum(costs.values()), MappingProxyType(costs),
        frontier.peak_host, frontier.peak_device, feasible, p.memory_limit, p.host_memory_limit,
        p.reserved_host_bytes, p.reserved_device_bytes, p.profile,
        disk_limit_bytes=p.disk_limit_bytes, explicit_resources=not direct)
    from ..packed_values import admit_packed_values
    from ..allocation_rounding import account_allocation_rounding
    plan = admit_packed_values(account_allocation_rounding(prepare_coordinate_batch_workspace(plan)[0]))
    if debug and (plan.memory != reference.memory or plan.events != reference.events
                  or plan.preparation_costs != reference.preparation_costs
                  or plan.feasible != reference.feasible
                  or plan.peak_host_bytes != reference.peak_host_bytes
                  or plan.peak_device_bytes != reference.peak_device_bytes):
        raise RuntimeError('fixed DAG resource accounting differs from independent replay')
    return plan if direct else executable_delivery(plan)


def replay_fixed_dag(state, physical):
    """Apply the selected computation order and early final-use releases."""
    actions = tuple(action for action in state.actions if action.kind not in ("load", "release"))
    spans = []
    for index, action in enumerate(actions):
        if not spans or actions[spans[-1][0]].tensor != action.tensor:
            spans.append((index, index + 1))
        else:
            spans[-1] = (spans[-1][0], index + 1)
    state = replace(state, actions=actions)
    dag = _Dag(state, spans, (0,) * len(spans), physical)
    return _plan(dag, range(len(dag.nodes)))


def schedule_fixed_dag(state, snippets, workspaces, physical, *, baseline=None,
                       max_states=1024, branch_width=4, deadline_seconds=DEFAULT_PLANNING_SECONDS,
                       clock=time.monotonic, debug=False):
    """Bounded DFS of ready work, with incremental peak pruning and undo.

    Release benefit ranks the alternatives; both memory domains constrain
    every transition. A complete improvement tightens the incumbent bound.
    Only the selected order is emitted; independent replay is debug-only.
    Search until the first complete feasible order, then start the optimization
    budget. The original fallback does not start that budget. On exhaustion,
    return the best complete order without completing a speculative suffix.
    """
    start = clock()
    if type(max_states) is not int or max_states < 1:
        raise ValueError("DAG state budget must be a positive integer")
    if type(branch_width) is not int or branch_width < 1:
        raise ValueError("DAG branch width must be a positive integer")
    if (isinstance(deadline_seconds, bool) or not math.isfinite(deadline_seconds)
            or deadline_seconds < 0):
        raise ValueError("DAG scheduling time budget must be finite and nonnegative")
    dag = _Dag(state, snippets, workspaces, physical, audited_computations=not debug)
    original = tuple(range(len(dag.nodes)))
    baseline_eager = _plan(dag, original, audited_computations=True, debug=debug)
    best = baseline_eager if baseline is None else baseline
    expected = best.seconds
    def score(host, device):
        return (device, host) if physical.gpu else (host, device)
    def improves(host, device, feasible):
        return (feasible and (not best_feasible or
                (host <= best_host and device <= best_device
                 and score(host, device) < score(best_host, best_device))))
    best_host, best_device, best_feasible = best.peak_host_bytes, best.peak_device_bytes, best.feasible
    chosen_order = None
    if improves(baseline_eager.peak_host_bytes, baseline_eager.peak_device_bytes, baseline_eager.feasible):
        best = baseline_eager
        best_host, best_device, best_feasible = best.peak_host_bytes, best.peak_device_bytes, best.feasible
    search_started = clock()
    deadline = None
    first_solution_expanded = first_solution_seconds = optimization_started = None
    frontier, expanded = _Frontier(dag), 0
    selected_targets = completed_targets = longest = pruned = backtracks = complete = 0
    stop_reason = 'width_exhausted'

    def physical_fits():
        return ((frontier.peak_device if physical.gpu else frontier.peak_host) <= physical.memory_limit
                and (physical.host_memory_limit is None
                     or frontier.peak_host <= physical.host_memory_limit))

    def fits():
        if first_solution_expanded is None:
            # The first feasible route need not improve an already good
            # fallback. Do not prune it by the incumbent's objective yet.
            return physical_fits()
        host_cap = min(best_host, physical.host_memory_limit) if physical.host_memory_limit is not None else best_host
        if not physical.gpu:
            host_cap = min(host_cap, physical.memory_limit)
        device_cap = min(best_device, physical.memory_limit) if physical.gpu else best_device
        return (frontier.peak_host <= host_cap and frontier.peak_device <= device_cap
                and (not best_feasible or score(frontier.peak_host, frontier.peak_device)
                     < score(best_host, best_device)))

    def consider():
        nonlocal best_host, best_device, best_feasible, chosen_order, complete
        nonlocal first_solution_expanded, first_solution_seconds, optimization_started, deadline
        frontier.finish()
        if not physical_fits():
            return
        if first_solution_expanded is None:
            optimization_started = clock()
            first_solution_seconds = optimization_started - search_started
            first_solution_expanded = expanded
            deadline = optimization_started + deadline_seconds
        if improves(frontier.peak_host, frontier.peak_device, True):
            complete += 1
            best_host, best_device, best_feasible = frontier.peak_host, frontier.peak_device, True
            chosen_order = tuple(frontier.order)

    def within_budget():
        return (first_solution_expanded is None
                or (expanded - first_solution_expanded < max_states and clock() < deadline))

    # Each frame retains only alternatives and the parent step's local undo.
    # No whole-frontier snapshots, expression copies, or recursion-depth limit.
    frames = [[None, None]]
    while frames:
        if not within_budget():
            stop_reason = ('max_states' if expanded - first_solution_expanded >= max_states else 'deadline')
            break
        frame = frames[-1]
        # A newly improved bound can make an already visited prefix hopeless.
        # Discard it before scoring or trying any further ready operations.
        if not fits():
            pruned += 1
            frames.pop()
            if frame[1] is not None:
                frontier.undo(frame[1])
                backtracks += 1
            continue
        if frame[0] is None:
            if not frontier.ready:
                consider()
                if branch_width == 1 and not within_budget():
                    stop_reason = ('max_states' if expanded - first_solution_expanded >= max_states else 'deadline')
                frames.pop()
                if frame[1] is not None:
                    frontier.undo(frame[1])
                    backtracks += 1
                continue
            frame[0] = frontier.choices(branch_width, within_budget)
        if not within_budget():
            continue
        choice = next(frame[0], None)
        if not within_budget():
            continue
        if choice is None:
            frames.pop()
            if frame[1] is not None:
                frontier.undo(frame[1])
                backtracks += 1
            continue
        index, goal = choice
        if goal is not None:
            selected_targets += 1
            longest = max(longest, goal.computations)
        undo = frontier.advance(index, reversible=branch_width != 1)
        expanded += 1
        if not fits():
            pruned += 1
            if branch_width == 1:
                break  # Every earlier frame also has no alternative.
            frontier.undo(undo)
            continue
        if goal is not None and not goal.required & ~frontier.done:
            completed_targets += 1
        if branch_width == 1:
            # The selected branch is the only one at every depth. Retaining
            # exhausted generators and undo records cannot expose more work.
            frames[-1] = [None, None]
        else:
            frames.append([None, undo])
    search_finished = clock()
    audited = 1
    if chosen_order is not None:
        if chosen_order == original:
            # Same DAG, complete order and emission options as the trajectory
            # already built above. An external baseline may carry other fields.
            best = baseline_eager
        else:
            best = _plan(dag, chosen_order, audited_computations=True, debug=debug)
            audited += 1
        if debug:
            # Incremental search ranks payload/workspace only. Final delivery
            # adds request granularity and optional cache residency separately.
            payload_device = max((point.device_bytes - point.allocation_rounding_device_bytes
                - point.packed_value_cache_device_bytes for point in best.memory), default=0)
            if best.peak_host_bytes != best_host or payload_device != best_device:
                raise RuntimeError('incremental search peaks differ from emitted schedule')
        if not best.feasible:
            fallback = baseline_eager if baseline is None or not baseline.feasible else baseline
            if fallback.feasible:
                best = fallback
                chosen_order = None
    if not math.isclose(best.seconds, expected, rel_tol=1e-11, abs_tol=1e-14):
        raise RuntimeError("DAG scheduling changed the selected compute-model cost")
    ranking = frontier.ranking
    return best, dict(nodes=len(dag.nodes), explored_frontiers=expanded,
        audited_schedules=audited, changed_order=chosen_order is not None and chosen_order != original,
        baseline_peak_host_bytes=(baseline or baseline_eager).peak_host_bytes,
        baseline_peak_device_bytes=(baseline or baseline_eager).peak_device_bytes,
        peak_host_bytes=best.peak_host_bytes, peak_device_bytes=best.peak_device_bytes,
        seconds=clock()-start, computation_seconds=state.seconds,
        policy="peak_bounded_release_dfs", proof_scope="heuristic", branch_width=branch_width,
        target_selection="each_step_incremental",
        rank_activations=0 if ranking is None else ranking.activations,
        rank_updates=0 if ranking is None else ranking.updates,
        rank_reuses=0 if ranking is None else ranking.reuses,
        rank_watch_registrations=0 if ranking is None else ranking.registrations,
        rank_heap_compactions=0 if ranking is None else ranking.compactions,
        peak_pruned=pruned, backtracks=backtracks, complete_improvements=complete,
        suffix_steps=0, stop_reason=stop_reason, budget_scope="after_first_search_solution",
        first_solution_expanded=first_solution_expanded, first_solution_seconds=first_solution_seconds,
        optimization_expanded=0 if first_solution_expanded is None else expanded-first_solution_expanded,
        optimization_seconds=0.0 if optimization_started is None else search_finished-optimization_started,
        release_targets_selected=selected_targets, release_targets_completed=completed_targets,
        target_evaluations=frontier.target_evaluations, longest_consumption_steps=longest)


_PLAN_FIELDS = ("state", "events", "memory", "preparation_seconds", "preparation_costs",
                "peak_host_bytes", "peak_device_bytes", "feasible", "memory_limit",
                "host_memory_limit", "reserved_host_bytes", "reserved_device_bytes",
                "explicit_resources", "peak_disk_bytes", "disk_limit_bytes")


def compare_plans(candidate, reference):
    """Field-by-field difference between two delivered schedules.

    Returns a dict of field name -> (candidate, reference) for every field that
    differs. An empty dict is full semantic equality of the delivered plan.
    """
    if candidate is reference:
        return {}
    differences = {}
    for name in _PLAN_FIELDS:
        left, right = getattr(candidate, name, None), getattr(reference, name, None)
        if left != right:
            differences[name] = (left, right)
    if not differences:
        return differences
    # Locate the first differing action for a readable diagnostic.
    left_actions = getattr(getattr(candidate, "state", None), "actions", ())
    right_actions = getattr(getattr(reference, "state", None), "actions", ())
    for index, (a, b) in enumerate(zip(left_actions, right_actions)):
        if a != b:
            differences["first_action_index"] = (index, (a, b))
            break
    else:
        if len(left_actions) != len(right_actions):
            differences["action_count"] = (len(left_actions), len(right_actions))
    return differences


def verify_schedule_equivalence(state, snippets, workspaces, physical, **kwargs):
    """Re-derive one fixed-DAG schedule with both candidate rules and compare.

    Diagnostic entry point for the storage scheduler. It runs the shipped
    requirement-bit walk and the pre-optimization whole-ready-set scan on the
    same input and budget, then compares the two delivered plans field by field.
    The comparison never changes the plan the production path returns.
    """
    if _LEGACY_CHOICES:
        raise RuntimeError("equivalence check requires the default candidate rule")
    candidate, candidate_stats = schedule_fixed_dag(
        state, snippets, workspaces, physical, **kwargs)
    try:
        globals()["_LEGACY_CHOICES"] = True
        reference, reference_stats = schedule_fixed_dag(
            state, snippets, workspaces, physical, **kwargs)
    finally:
        globals()["_LEGACY_CHOICES"] = False
    differences = compare_plans(candidate, reference)
    return dict(
        equivalent=not differences,
        differences=differences,
        candidate_peak_host_bytes=candidate.peak_host_bytes,
        reference_peak_host_bytes=reference.peak_host_bytes,
        candidate_peak_device_bytes=candidate.peak_device_bytes,
        reference_peak_device_bytes=reference.peak_device_bytes,
        candidate_seconds=candidate.seconds,
        reference_seconds=reference.seconds,
        candidate_actions=len(candidate.state.actions),
        reference_actions=len(reference.state.actions),
        candidate_explored=candidate_stats.get("explored_frontiers"),
        reference_explored=reference_stats.get("explored_frontiers"),
        candidate_stop_reason=candidate_stats.get("stop_reason"),
        reference_stop_reason=reference_stats.get("stop_reason"),
    )
