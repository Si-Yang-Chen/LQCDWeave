"""Shared contraction effects and conservative complete-step resource bounds.

The exact primitive-action domain remains in storage_space. This domain has
one edge per complete contraction, or one pressure-triggered resource move.
Only edge snippets retain actions; cursors carry no path history.
"""
from dataclasses import dataclass, field, replace
from collections import OrderedDict
from functools import cached_property
from types import MappingProxyType
import math
from itertools import chain, combinations

from lqcdweave.contracts.correlator.api import (
    Action, Factor, PendingTerm, ResourceAllocation, ResourceSnapshot, ResourceTransition, MemoryPoint,
)
from ..actions import _apply_selected, _live_inputs
from ..state import ContractionState, _nbytes, _ImmutableKey
from ..contraction_index import ContractionKey
from ..logical_index import LayoutIndex, BridgeIndex, reuse_indexes, bridge_identity, join_identity
from ..paths import _state_key
from ..resources import resource_point, transition_resource, audit_resource_trajectory, _check
from .storage_space import StorageSearchSpace
from .candidate_catalog import CandidateCatalog, CatalogOverflow


class UnavailableInput(ValueError):
    """A discarded original block or intermediate cannot be regenerated."""


def _patch_tuple(values, changes):
    """Apply sorted sparse index changes with one output tuple allocation."""
    if not changes:
        return values

    def updated():
        change_index = 0
        for index, value in enumerate(values):
            if change_index < len(changes) and changes[change_index][0] < index:
                raise ValueError("tuple changes must be sorted and in range")
            if change_index < len(changes) and changes[change_index][0] == index:
                yield changes[change_index][1]
                change_index += 1
            else:
                yield value
        if change_index != len(changes):
            raise ValueError("tuple changes must be sorted and in range")

    return tuple(updated())


def _apply_tuple_delta(values, removed, added):
    """Build a resource tuple once, preserving survivors and their order."""
    if not removed and not added:
        return values
    removed = set(removed)
    return tuple(chain((value for value in values if value not in removed), added))


def _merge_sorted_items(values, updates):
    """Merge sorted key/value tuples, with updates replacing matching keys."""
    if not updates:
        return values

    def merged():
        left = right = 0
        while left < len(values) and right < len(updates):
            old_key, old_value = values[left]
            new_key, new_value = updates[right]
            if old_key < new_key:
                yield values[left]
                left += 1
            elif new_key < old_key:
                yield updates[right]
                right += 1
            else:
                yield updates[right]
                left += 1
                right += 1
        while left < len(values):
            yield values[left]
            left += 1
        while right < len(updates):
            yield updates[right]
            right += 1

    return tuple(merged())


@dataclass(frozen=True)
class MacroCursor:
    state: ContractionState
    resources: ResourceSnapshot = ResourceSnapshot()
    preparation_seconds: float = 0.0
    peak_host: int = 0
    peak_device: int = 0
    peak_disk: int = 0
    producers: tuple = ()

    @cached_property
    def search_key(self):
        return _ImmutableKey((_state_key(self.state), tuple(sorted(self.resources.allocations,
            key=lambda a: (a.domain, a.key))), tuple(sorted(self.resources.borrowed)),
            self.resources.opened_blocks, self.producers))

    @cached_property
    def owners(self):
        return MappingProxyType({(a.domain, a.key): a for a in self.resources.allocations})

    @cached_property
    def borrowed(self):
        return MappingProxyType(dict(self.resources.borrowed))

    @cached_property
    def useful_tensors(self):
        return _useful_tensors(self.state, self.producers)

    def _inherit_useful(self, successor):
        """Keep liveness only when a transition removes no useful data.

        With fixed expressions/producers, useful = required | (loaded &
        reachable). Removing loaded entries outside useful leaves that set
        unchanged. Resource locations, peak values and cost do not affect it.
        """
        useful = self.__dict__.get("useful_tensors")
        before, after = self.state, successor.state
        if (useful is not None and after.problem is before.problem
                and after.pending is before.pending and after.results is before.results
                and after.completed_joins is before.completed_joins
                and successor.producers is self.producers and after.loaded <= before.loaded
                and (before.loaded - after.loaded).isdisjoint(useful)):
            object.__setattr__(successor, "useful_tensors", useful)
        return successor


@dataclass(frozen=True)
class MacroTransition:
    identity: object
    state: MacroCursor
    actions: tuple[Action, ...]
    operation: Action
    pending_delta: object = field(default=None, compare=False, repr=False)
    results_delta: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class LogicalStep:
    """Temporary shared algebra effect and fixed operation workspace estimate."""
    state: ContractionState
    actions: tuple
    workspace_host: int
    workspace_device: int
    pending_delta: object = field(default=None, compare=False, repr=False)
    results_delta: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class LogicalCandidate:
    """Short-lived computation description; sharing is deferred until needed."""
    before: ContractionState
    index: object
    term_index: int
    positions: tuple
    layout: tuple
    description: tuple
    identity: tuple

    @property
    def actions(self):
        return (self.description[0],)


class BoundedSuccessors:
    """Report skipped descriptions so a partial ranking can be widened."""
    def __init__(self, space, cursor, cutoff, checkpoint):
        self.space, self.cursor, self.cutoff, self.checkpoint = space, cursor, cutoff, checkpoint
        self.omitted = 0

    def __iter__(self):
        try:
            yield from self.space._iter_successors(self.cursor, self.checkpoint, batch=self)
        finally:
            self.space._release_facts(self.cursor)


@dataclass(frozen=True, slots=True)
class LogicalEffect:
    """A bounded cached algebra delta, with no parent/successor reference."""
    spec: object
    pending: tuple
    results: tuple
    joins: frozenset
    actions: tuple
    workspace_host: int
    workspace_device: int
    peak_bytes: int


@dataclass(frozen=True, slots=True)
class StepDelta:
    """Parent-relative changes; never retains a parent or successor snapshot."""
    operation: Action
    actions: tuple
    catalog: tuple
    pending: tuple
    results: tuple
    joins: frozenset
    loaded_add: frozenset
    loaded_remove: frozenset
    allocations_add: tuple
    allocations_remove: tuple
    borrowed_add: tuple
    borrowed_remove: tuple
    opened: frozenset
    producers: tuple
    seconds: float
    preparation_seconds: float
    peak_host: int
    peak_device: int
    peak_disk: int


def operation_identity(action):
    # Scene/positions select a representative. The generated identity already
    # includes algebra, bridges and joins; effect sharing is verified below.
    return replace(action, term_index=-1, positions=())


def _useful_tensors(state, descriptors):
    """Pure per-cursor liveness, including reachable cached computations."""
    required = _live_inputs(state) | {k for k in state.results if k is not None}
    required = set(required)
    producers = dict(descriptors)
    implicit = required - {f.tensor for t in state.pending if t for f in t.factors}
    for term in state.pending:
        if term is None:
            continue
        factors = {f.tensor for f in term.factors} | implicit
        memo = {}
        def reachable(key):
            if key in factors:
                return True
            if key in memo:
                return memo[key]
            memo[key] = False
            if key in producers:
                memo[key] = all(reachable(k) for k in producers[key])
            return memo[key]
        required.update(k for k in state.loaded if k in producers and reachable(k))
    return frozenset(required)


class MacroSearchSpace(StorageSearchSpace):
    """One logical effect and aggregate resource check per computation."""
    def __init__(self, *args, disk_limit_bytes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if disk_limit_bytes is not None and (type(disk_limit_bytes) is not int or disk_limit_bytes <= 0):
            raise ValueError("disk_limit_bytes must be positive integer bytes or None")
        self.disk_limit_bytes = disk_limit_bytes
        self.unavailable = self.resource_branches = 0
        self.logical_effects = self.aggregate_checks = self.selected_delta_applications = 0
        self.workspace_estimates = {}
        self.descriptions = {}
        self.bound_recipes = {}
        self.candidate_catalog = None
        self.catalog_fallbacks = 0
        self.description_hits = self.omitted_previews = 0
        self.layout_index = None
        self.layout_index_active = False
        self.index_retargets = 0
        # Only the last pressure context is retained. Ordinary feasible nodes
        # stream effects and do not accumulate another candidate history.
        self.logical_cache = None
        self.logical_cache_hits = 0
        self.preparation_orders = OrderedDict()
        # Physical scheduling facts are derived on first use by the cached
        # properties below. Logical planning reads only `problem` and
        # `logical_profile`, so a run that stops before the physical storage
        # schedule never pays for the initial expression, bridge index, action
        # tables or preparation costs.

    @cached_property
    def _initial_expression(self):
        """The sole immutable initial expression; only physical planning reads it."""
        return ContractionState.initial(self.problem)

    @cached_property
    def bridge_index(self):
        return BridgeIndex(self.problem.sparse_bridges)

    @cached_property
    def mandatory_blocks(self):
        # Required raw blocks are tracked for lifecycle and capacity auditing.
        required = _live_inputs(self._initial_expression) & self.preparations.keys()
        return frozenset(dep for key in required for dep in self.preparations[key].blocks)

    @cached_property
    def read_actions(self):
        return {key: Action("read_block", key) for key in self.blocks}

    @cached_property
    def load_actions(self):
        initial = self._initial_expression
        return {key: Action("load", key, seconds=self.logical_profile.snapshot_seconds(
            initial._snapshot_bytes(key)))
            for key in self.preparations}

    @cached_property
    def preparation_costs(self):
        return {key: prep.seconds(self.profile) for key, prep in self.preparations.items()}

    def materialize_physical_facts(self):
        """Force every deferred physical scheduling fact for diagnostics.

        Logical planning never needs these tables; the pre-laziness behavior
        (and its timing) is reproduced by accessing all of them up front.
        """
        return (self.bridge_index, self.mandatory_blocks, self.read_actions,
                self.load_actions, self.preparation_costs)

    def initial_state(self):
        cursor = MacroCursor(ContractionState.initial(self.problem))
        return self._record(cursor, self._point_at(cursor.resources, "initial"))

    def key(self, cursor):
        return cursor.search_key

    def cost(self, cursor):
        return cursor.state.seconds + cursor.preparation_seconds

    def is_feasible(self, cursor):
        return ((cursor.peak_device if self.gpu else cursor.peak_host) <= self.memory_limit
            and (self.host_memory_limit is None or cursor.peak_host <= self.host_memory_limit)
            and (self.disk_limit_bytes is None or cursor.peak_disk <= self.disk_limit_bytes))

    def finish(self, cursor):
        return cursor if cursor.state.done and self.is_feasible(cursor) else None

    def _point_at(self, snapshot, phase, wh=0, wd=0):
        return resource_point(snapshot, 0, phase, workspace_host=wh, workspace_device=wd,
            reserved_host=self.reserved_host_bytes, reserved_device=self.reserved_device_bytes)

    def _maxima_point(self, maxima, phase, wh, wd):
        raw_host = raw_device = tensor_host = tensor_device = disk = 0
        for (domain, _), (host, device, temporary) in maxima.items():
            if domain == "block":
                raw_host += host
                raw_device += device
            else:
                tensor_host += host
                tensor_device += device
            disk += temporary
        return MemoryPoint(0, phase, raw_host, raw_device, tensor_host, tensor_device,
            wh, wd, self.reserved_host_bytes, self.reserved_device_bytes, temporary_disk_bytes=disk)

    def _ordered(self, keys):
        identity = frozenset(keys)
        ordered = self.preparation_orders.get(identity)
        if ordered is None:
            ordered = self.preparation_orders[identity] = tuple(sorted(identity))
            if len(self.preparation_orders) > 256:
                self.preparation_orders.popitem(last=False)
        else:
            self.preparation_orders.move_to_end(identity)
        return ordered

    def _record(self, cursor, point):
        try:
            _check(point, backend=self.problem.backend, memory_limit=self.memory_limit,
                   host_memory_limit=self.host_memory_limit, disk_limit_bytes=self.disk_limit_bytes)
        except MemoryError as exc:
            exc.point = point
            raise
        return replace(cursor, peak_host=max(cursor.peak_host, point.host_bytes),
            peak_device=max(cursor.peak_device, point.device_bytes),
            peak_disk=max(cursor.peak_disk, point.temporary_disk_bytes))

    @staticmethod
    def owner(cursor, domain, key):
        return cursor.owners.get((domain, key))

    def usable(self, cursor, domain, key):
        owner = self.owner(cursor, domain, key)
        if owner is None:
            deps = cursor.borrowed.get(key) if domain == "tensor" else None
            return deps is not None and all(self.usable(cursor, "block", dep) for dep in deps)
        if domain == "block":
            block = self.blocks[key]
            return (owner.host_bytes, owner.device_bytes, owner.disk_bytes) == (block.host_bytes, block.device_bytes, 0)
        return owner.disk_bytes == 0 and (owner.host_bytes == 0 if self.gpu else owner.device_bytes == 0)

    def useful(self, cursor):
        """Protect direct uses and cached computations reachable in any Term.

        Current factors are opaque: we never split an intermediate into old
        operands. Producer descriptors only conservatively track future reuse.
        """
        return cursor.useful_tensors

    def recovery(self, cursor):
        return {key for key, prep in self.preparations.items()
                if all(self.owner(cursor, "block", dep) is not None for dep in prep.blocks)}

    def required_blocks(self, cursor):
        # Raw data are useful while they can still recover a useful source.
        return {dep for key in self.useful(cursor) if key in self.preparations
                for dep in self.preparations[key].blocks}

    def _move(self, cursor, kind, domain, key):
        owner = self.owner(cursor, domain, key)
        source = next((tier for tier in ("device", "host", "disk")
                       if owner is not None and getattr(owner, tier + "_bytes")),
                      "device" if self.gpu else "host")
        nbytes = getattr(owner, source + "_bytes") if owner is not None else 0
        target = {"discard": "absent", "spill_host": "host", "spill_disk": "disk",
                  "restore_host": "host", "restore_device": "device"}[kind]
        return Action(kind, key,
                      resource=ResourceTransition(domain, source, target, nbytes))

    def apply(self, cursor, action):
        """Replay one declared atomic phase, without choosing memory policy."""
        state = replace(cursor.state, actions=(), peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
        snapshot = cursor.resources
        cost = 0.0
        producers = cursor.producers
        if action.kind in ("discard", "spill_host", "spill_disk", "restore_host", "restore_device"):
            required = self.useful(cursor)
            future = {("tensor", key) for key in required}
            future.update(("block", key) for key in self.required_blocks(cursor))
            recovery = {("tensor", key) for key in self.recovery(cursor)}
            snapshot, point = transition_resource(snapshot, action,
                future_needed=future, recoverable_sources=recovery, backend=self.problem.backend,
                memory_limit=self.memory_limit, host_memory_limit=self.host_memory_limit,
                disk_limit_bytes=self.disk_limit_bytes, reserved_host_bytes=self.reserved_host_bytes,
                reserved_device_bytes=self.reserved_device_bytes)
            if action.kind == "discard" and action.resource.domain == "tensor":
                state = replace(state, loaded=state.loaded - {action.tensor})
            state = replace(state, seconds=state.seconds + action.seconds)
        elif action.kind == "read_block":
            if action.tensor in snapshot.opened_blocks:
                raise UnavailableInput("original blocks may only be read once")
            block = self.blocks[action.tensor]
            if action.seconds != 0.0:
                raise ValueError("raw block reads have no predicted time cost")
            snapshot = replace(snapshot, allocations=snapshot.allocations +
                (ResourceAllocation("block", action.tensor, block.host_bytes, block.device_bytes),),
                opened_blocks=snapshot.opened_blocks | {action.tensor})
            point = self._point_at(snapshot, "read:" + action.tensor, block.staging_host_bytes, block.staging_device_bytes)
        elif action.kind == "close_block":
            owner = self.owner(cursor, "block", action.tensor)
            if (owner is None or action.tensor in self.required_blocks(cursor)
                    or any(action.tensor in deps for _, deps in snapshot.borrowed)):
                raise ValueError("cannot close a useful or borrowed block")
            snapshot = replace(snapshot, allocations=tuple(a for a in snapshot.allocations if a != owner))
            point = self._point_at(snapshot, "close:" + action.tensor)
        elif action.kind == "load":
            prep = self.preparations[action.tensor]
            if any(not self.usable(cursor, "block", dep) for dep in prep.blocks):
                raise UnavailableInput("materialization requires available raw owners")
            state = state.load(action.tensor, self.logical_profile, 1 << 100)
            cost = prep.seconds(self.profile)
            if action.tensor in state.catalog:
                size = state._snapshot_bytes(action.tensor)
                snapshot = replace(snapshot, allocations=snapshot.allocations +
                    (ResourceAllocation("tensor", action.tensor, 0 if self.gpu else size, size if self.gpu else 0),))
            else:
                snapshot = replace(snapshot, borrowed=snapshot.borrowed + ((action.tensor, prep.blocks),))
            point = self._point_at(snapshot, "materialize:" + action.tensor, prep.workspace_host_bytes, prep.workspace_device_bytes)
        elif action.kind == "release":
            if action.tensor in self.useful(cursor):
                raise ValueError("dead reclamation cannot release a useful tensor")
            state = state.release(action.tensor)
            snapshot = replace(snapshot, allocations=tuple(a for a in snapshot.allocations
                if (a.domain, a.key) != ("tensor", action.tensor)),
                borrowed=tuple(b for b in snapshot.borrowed if b[0] != action.tensor))
            point = self._point_at(snapshot, "release:" + action.tensor)
        else:
            if action.reused:
                if not self.usable(cursor, "tensor", action.tensor):
                    raise UnavailableInput("cached output requires restoration")
                symbolic = replace(state, loaded=state.loaded | set(action.operands))
                updated = _apply_selected(symbolic, action.term_index, action, self.logical_profile, 1 << 100)
                if updated is not None:
                    updated = replace(updated, loaded=state.loaded | {action.tensor})
            else:
                if any(not self.usable(cursor, "tensor", key) for key in action.operands):
                    raise UnavailableInput("compute operands require full preparation")
                updated = _apply_selected(state, action.term_index, action, self.logical_profile, 1 << 100)
            if updated is None:
                raise ValueError("invalid logical contraction")
            if updated.actions[-1] != action:
                raise ValueError("compute fields differ from the logical transition")
            if action.tensor not in state.loaded:
                size = _nbytes(updated.catalog[action.tensor])
                snapshot = replace(snapshot, allocations=snapshot.allocations +
                    (ResourceAllocation("tensor", action.tensor, 0 if self.gpu else size, size if self.gpu else 0),))
                producers = tuple(sorted(dict(producers, **{action.tensor: action.operands}).items()))
            workspace = 0 if action.reused else max(0, updated.peak_bytes - max(state.resident_bytes, updated.resident_bytes))
            point = self._point_at(snapshot, action.kind, 0 if self.gpu else workspace, workspace if self.gpu else 0)
            state = updated
        return cursor._inherit_useful(self._record(replace(cursor, state=replace(state, actions=()), resources=snapshot,
            preparation_seconds=cursor.preparation_seconds + cost, producers=producers), point))

    def _symbolic(self, cursor):
        return replace(cursor.state, loaded=cursor.state.loaded | _live_inputs(cursor.state),
                       actions=(), peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)

    @staticmethod
    def _join_identity(join, layout):
        # Compare the complete algebra, not just the generated fingerprint.
        return ("baryon_join", join_identity(join), layout[0], layout[2])

    def _join_index(self, state, checkpoint, scene_joins=None):
        """Resolve each active layout once, for enumeration and substitution."""
        entries, groups, layouts = [], {}, {}
        if scene_joins is None:
            scene_joins = {}
            for join in self.problem.baryon_joins:
                if (join.scene, join.name) not in state.completed_joins:
                    scene_joins.setdefault(join.scene, []).append(join)
        for join in self.problem.baryon_joins:
            if (join.scene, join.name) in state.completed_joins:
                continue
            checkpoint()
            scoped = self.bridge_index.scoped(join.scene)
            context = (state.pending[join.scene], join_identity(join),
                       tuple(join_identity(other) for other in scene_joins.get(join.scene, ())), scoped)
            try:
                layout = layouts.get(context)
                if layout is None:
                    layout = state._join_layout(join, scoped_bridges=scoped, scene_joins=scene_joins)
                    layouts[context] = layout
            except ValueError:
                continue
            identity = self._join_identity(join, layout)
            entry = (join, layout, identity)
            entries.append(entry)
            groups.setdefault(identity, []).append(entry)
        return entries, groups

    def _logical_step(self, state, action_state, checkpoint, workspace_key=None, join_group=None,
                      scene_joins=None, resident_bytes=None, changes=None):
        actions = list(action_state.actions)
        action = actions[0]
        workspace = 0 if action.reused else max(0,
            action_state.peak_bytes - (state.resident_bytes if resident_bytes is None else resident_bytes)
            - _nbytes(action_state.catalog[action.tensor]))
        if not action.reused and workspace_key is not None:
            self.workspace_estimates.setdefault(workspace_key, workspace)
        if action.kind == "baryon_join":
            if join_group is None:
                entries, groups = self._join_index(state, checkpoint)
                identity = next(identity for join, layout, identity in entries
                                if join.scene == action.term_index and join.name == action.join_id)
                join_group = groups[identity]
            if scene_joins is None:
                scene_joins = {}
                for join in self.problem.baryon_joins:
                    scene_joins.setdefault(join.scene, []).append(join)
            pending, results = list(action_state.pending), list(action_state.results)
            completed = set(action_state.completed_joins)
            templates = {}
            replacement = Factor(action.tensor, join_group[0][1][2])
            for join, layout, identity in join_group:
                marker = join.scene, join.name
                if marker in completed:
                    continue
                checkpoint()
                # An identical join has the same name/slots/inputs/output and
                # a distinct scene (duplicate markers have been skipped).
                # Updating one scene cannot invalidate another scene's layout.
                # Substitute the cached output for every matching join now;
                # no second kernel or resource simulation is needed later.
                selected, rest, output, key = layout
                if key != action.tensor or (workspace_key is not None and identity != workspace_key):
                    raise ValueError("shared join algebra differs from its representative")
                term = pending[join.scene]
                if term is None:
                    raise ValueError("shared join requires an unfinished scene")
                completed.add(marker)
                template = rest, term.output
                updated = templates.get(template)
                if updated is None:
                    updated = templates[template] = PendingTerm(rest + (replacement,), term.output)
                if (len(updated.factors) == 1 and updated.factors[0].labels == updated.output
                        and not any((other.scene, other.name) not in completed
                                    for other in scene_joins.get(join.scene, ()))):
                    pending[join.scene], results[join.scene] = None, key
                else:
                    pending[join.scene] = updated
                actions.append(replace(action, seconds=0.0, reused=True, term_index=join.scene))
            action_state = replace(action_state, pending=tuple(pending), results=tuple(results),
                                   completed_joins=frozenset(completed))
        self.logical_effects += 1
        indices = (sorted({a.term_index for a in actions}) if action.kind == "baryon_join" else
                   sorted({i for i, _, _ in changes}) if changes is not None else None)
        pending_delta = (None if indices is None else
                         tuple((i, action_state.pending[i]) for i in indices
                               if action_state.pending[i] != state.pending[i]))
        results_delta = (None if indices is None else tuple((i, action_state.results[i]) for i in indices
                         if action_state.results[i] != state.results[i]))
        return LogicalStep(replace(action_state, actions=()), tuple(actions),
                           0 if self.gpu else workspace, workspace if self.gpu else 0,
                           pending_delta, results_delta)

    def logical_candidates(self, state, checkpoint=lambda: None):
        """Enumerate algebra operations without physical preparation or IO."""
        return self._build_logical_candidates(state, checkpoint)

    def describe_logical(self, cursor, action, checkpoint=lambda: None):
        """Replay a selected algebra operation and its shared Term updates."""
        return self._describe(cursor, action, checkpoint)

    def shared_logical_step(self, state, action_state, checkpoint=lambda: None, *, changes=None):
        """Describe shared updates from an already constructed algebra result."""
        return self._logical_step(state, action_state, checkpoint, changes=changes)

    def logical_steps(self, cursor, checkpoint=lambda: None):
        for candidate in self._logical_candidates(cursor, checkpoint):
            yield self._materialize_logical(candidate, checkpoint)

    def _logical_candidates(self, cursor, checkpoint):
        state = self._symbolic(cursor)
        key = _state_key(state)
        if self.logical_cache is not None and self.logical_cache[0] == key:
            self.logical_cache_hits += 1
            for effect in self.logical_cache[1]:
                checkpoint()
                yield self._apply_effect(state, effect)
            return
        yield from self._build_logical_candidates(state, checkpoint)

    @staticmethod
    def _freeze_effect(before, step):
        return LogicalEffect(step.state.catalog[step.actions[0].tensor],
            step.pending_delta if step.pending_delta is not None else
                tuple((i, v) for i, v in enumerate(step.state.pending) if v != before.pending[i]),
            step.results_delta if step.results_delta is not None else
                tuple((i, v) for i, v in enumerate(step.state.results) if v != before.results[i]),
            step.state.completed_joins - before.completed_joins, step.actions,
            step.workspace_host, step.workspace_device, step.state.peak_bytes)

    def _apply_effect(self, before, effect):
        pending = _patch_tuple(before.pending, effect.pending)
        results = _patch_tuple(before.results, effect.results)
        operation = effect.actions[0]
        state = replace(before, catalog=before.catalog.updated(((operation.tensor, effect.spec),)),
            pending=pending, results=results, completed_joins=before.completed_joins | effect.joins,
            loaded=before.loaded | {operation.tensor}, seconds=before.seconds + sum(a.seconds for a in effect.actions),
            peak_bytes=effect.peak_bytes, peak_host_bytes=0 if self.gpu else effect.peak_bytes,
            peak_device_bytes=effect.peak_bytes if self.gpu else 0)
        return LogicalStep(state, effect.actions, effect.workspace_host, effect.workspace_device,
                           effect.pending, effect.results)

    def _build_logical_steps(self, state, checkpoint):
        for candidate in self._build_logical_candidates(state, checkpoint):
            yield self._materialize_logical(candidate, checkpoint)

    def _build_logical_candidates(self, state, checkpoint):
        # One expansion owns the mutable lookup. Siblings only retain deltas;
        # interleaved diagnostic generators get independent indexes. A budget
        # interruption discards the partially advanced index before reuse.
        if self.layout_index_active:
            yield from self._enumerate_logical_candidates(state, checkpoint)
            return
        self.layout_index_active = True
        try:
            index = self.layout_index
            if index is None or index.factors is None:
                index = LayoutIndex(state, checkpoint, reuse=reuse_indexes(state.pending), bridges=self.bridge_index)
            else:
                index.checkpoint = checkpoint
                if index.contractions is not None:
                    index.contractions.checkpoint = checkpoint
                index.retarget(state)
                self.index_retargets += 1
            yield from self._enumerate_logical_candidates(state, checkpoint, index)
            self.layout_index = index
        except BaseException:
            self.layout_index = None
            raise
        finally:
            self.layout_index_active = False

    def _enumerate_logical_candidates(self, state, checkpoint, index=None):
        seen = set()
        # Tiny nodes seldom reuse a layout: avoid building lookup tables that
        # cost more than their few ordinary contractions. This changes only
        # metadata lookup, never the candidate domain, order or scoring.
        reuse = reuse_indexes(state.pending)
        if index is None:
            index = LayoutIndex(state, checkpoint, reuse=reuse, bridges=self.bridge_index)
        resident_bytes = state.resident_bytes
        # Deduplicate computation identities, not complete Terms. Layouts
        # check all labels/bridges before a shared effect is built once.
        for term_index, pair, layout, identity in self._contraction_layouts(state, index, checkpoint):
            checkpoint()
            if identity in seen:
                continue
            try:
                cache_key = (identity, tuple(layout[2].items()), layout[3], layout[4],
                             tuple(state.catalog[f.tensor] for f in layout[0]), layout[8] in state.loaded)
                description = self.descriptions.get(cache_key)
                if description is None:
                    description = state._describe_contract(term_index, pair, layout, self.logical_profile,
                        workspace_bytes=self.workspace_estimates.get(identity))
                    if len(self.descriptions) < 2048:
                        self.descriptions[cache_key] = description
                else:
                    if layout[8] in state.catalog and layout[8] not in state.loaded:
                        continue
                    self.description_hits += 1
            except ValueError:
                continue
            seen.add(identity)
            yield LogicalCandidate(state, index, term_index, pair, layout, description, identity)
        entries, groups = self._join_index(state, checkpoint, index.scene_joins)
        for join, layout, identity in entries:
            if identity in seen:
                continue
            try:
                after = state._join_from_layout(join, layout, self.logical_profile, 1 << 100,
                    workspace_bytes=self.workspace_estimates.get(identity))
            except ValueError:
                continue
            seen.add(identity)
            yield self._logical_step(state, after, checkpoint, identity, groups[identity],
                                     index.scene_joins, resident_bytes)

    def _contraction_layouts(self, state, index, checkpoint):
        if self.candidate_catalog is None:
            self.candidate_catalog = CandidateCatalog(self.problem)
        try:
            layouts = self.candidate_catalog.layouts(state, index, checkpoint)
        except CatalogOverflow:
            # A cache limit changes lookup work, never the available domain.
            self.candidate_catalog = None
            self.catalog_fallbacks += 1
            yield from self._scan_contraction_layouts(state, index, checkpoint)
        else:
            yield from layouts

    @staticmethod
    def _scan_contraction_layouts(state, index, checkpoint):
        for i, term in enumerate(state.pending):
            if term is None:
                continue
            pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
            for pair in pairs:
                checkpoint()
                try:
                    layout = index.layout(i, pair)
                except ValueError:
                    continue
                yield i, pair, layout, ContractionKey(("contract", layout[0], layout[1],
                    bridge_identity(layout[5]), bridge_identity(layout[6])))

    def _materialize_logical(self, candidate, checkpoint):
        if not isinstance(candidate, LogicalCandidate):
            return candidate
        changes = []
        action, spec, workspace = candidate.description
        if (action.term_index, action.positions) != (candidate.term_index, candidate.positions):
            action = replace(action, term_index=candidate.term_index, positions=candidate.positions)
        after = candidate.before._contract_from_layout(candidate.term_index, candidate.positions,
            candidate.layout, self.logical_profile, 1 << 100, layout_index=candidate.index,
            changes=changes, description=(action, spec, workspace))
        return self._logical_step(candidate.before, after, checkpoint, candidate.identity,
            resident_bytes=candidate.before.resident_bytes, changes=changes)

    def candidates(self, cursor, checkpoint=lambda: None):
        """Diagnostic operation enumeration; search consumes evaluated steps."""
        for step in self._logical_candidates(cursor, checkpoint):
            action = step.actions[0]
            yield (replace(action, term_index=step.term_index, positions=step.positions)
                   if isinstance(step, LogicalCandidate) else action)

    def _describe(self, cursor, action, checkpoint):
        state = self._symbolic(cursor)
        changes = None
        if action.kind == "baryon_join":
            join = next(j for j in self.problem.baryon_joins
                        if j.scene == action.term_index and j.name == action.join_id)
            after = state._join_from_layout(join, state._join_layout(join), self.logical_profile, 1 << 100,
                                            kernel_path_override=action.kernel_path)
        else:
            changes = []
            after = state._contract_from_layout(action.term_index, action.positions,
                state._contract_layout(action.term_index, action.positions), self.logical_profile, 1 << 100, changes=changes)
        if after.actions[-1] != action:
            raise ValueError("operation differs from the current logical state")
        return self._logical_step(state, after, checkpoint, changes=changes)

    def _dead_actions(self, cursor, protected, pressure=None):
        tiers = set()
        if pressure is not None:
            host_cap = min(self.memory_limit if not self.gpu else math.inf,
                           self.host_memory_limit if self.host_memory_limit is not None else math.inf)
            if pressure.host_bytes > host_cap:
                tiers.add("host")
            if self.gpu and pressure.device_bytes > self.memory_limit:
                tiers.add("device")
            if self.disk_limit_bytes is not None and pressure.temporary_disk_bytes > self.disk_limit_bytes:
                tiers.add("disk")
        def contributes(domain, key):
            if pressure is None:
                return True
            owner = self.owner(cursor, domain, key)
            if owner is not None:
                return any(getattr(owner, tier + "_bytes") for tier in tiers)
            return any(contributes("block", dep) for dep in dict(cursor.resources.borrowed).get(key, ()))
        required = self.useful(cursor) | protected
        for key in sorted(cursor.state.loaded - required):
            if not contributes("tensor", key):
                continue
            if self.usable(cursor, "tensor", key):
                yield Action("release", key)
            else:
                yield self._move(cursor, "discard", "tensor", key)
        raw = self.required_blocks(cursor) | {dep for key in protected if key in self.preparations
                                            for dep in self.preparations[key].blocks}
        borrowed = {dep for _, deps in cursor.resources.borrowed for dep in deps}
        for owner in sorted(cursor.resources.allocations, key=lambda a: (a.domain, a.key)):
            if owner.domain == "block" and owner.key not in raw | borrowed and contributes("block", owner.key):
                yield Action("close_block", owner.key)

    def _prepare_step(self, cursor, step, checkpoint):
        """Build one fixed preparation recipe and an aggregate occupancy bound.

        Per-owner tier maxima cover overlap during migration. Adding their
        maxima and the largest workspace bounds every declared atomic phase,
        without constructing/checking a cursor for each read/load/compute.
        """
        operation = step.actions[0]
        allocations = dict(cursor.owners)
        high = {identity: (a.host_bytes, a.device_bytes, a.disk_bytes) for identity, a in allocations.items()}
        borrowed = dict(cursor.borrowed)
        opened = set(cursor.resources.opened_blocks)
        loaded = set(cursor.state.loaded)
        actions = []
        wh, wd = step.workspace_host, step.workspace_device
        prep_cost = 0.0

        def remember(allocation):
            identity = (allocation.domain, allocation.key)
            old = high.get(identity, (0, 0, 0))
            high[identity] = (max(old[0], allocation.host_bytes), max(old[1], allocation.device_bytes),
                              max(old[2], allocation.disk_bytes))

        def snapshot():
            return ResourceSnapshot(tuple(allocations.values()), tuple(borrowed.items()), frozenset(opened))

        def usable(domain, key):
            owner = allocations.get((domain, key))
            if owner is None:
                return (domain == "tensor" and key in borrowed and
                        all(usable("block", dep) for dep in borrowed[key]))
            if domain == "block":
                block = self.blocks[key]
                return (owner.host_bytes, owner.device_bytes, owner.disk_bytes) == (block.host_bytes, block.device_bytes, 0)
            return owner.disk_bytes == 0 and (owner.host_bytes == 0 if self.gpu else owner.device_bytes == 0)

        def restore(domain, key):
            nonlocal wh, wd
            while not usable(domain, key):
                checkpoint()
                owner = allocations.get((domain, key))
                if owner is None:
                    raise UnavailableInput("no preserved temporary copy")
                kind = "restore_host" if owner.disk_bytes else "restore_device"
                source = "disk" if owner.disk_bytes else "host"
                target = "host" if owner.disk_bytes else "device"
                size = getattr(owner, source + "_bytes")
                action = Action(kind, key, resource=ResourceTransition(domain, source, target, size))
                updated, point = transition_resource(snapshot(), action,
                    backend=self.problem.backend, memory_limit=1 << 100)
                identity = domain, key
                moved = next(a for a in updated.allocations if (a.domain, a.key) == identity)
                del allocations[identity]
                allocations[identity] = moved
                remember(moved)
                wh, wd = max(wh, point.workspace_host_bytes), max(wd, point.workspace_device_bytes)
                actions.append(action)

        protected = {operation.tensor} if operation.reused else set(operation.operands)
        for key in sorted(protected):
            checkpoint()
            if usable("tensor", key):
                continue
            if key in loaded:
                restore("tensor", key)
                continue
            prep = self.preparations.get(key)
            if prep is None:
                raise UnavailableInput("intermediate recomputation is unsupported")
            for dep in self._ordered(prep.blocks):
                checkpoint()
                if usable("block", dep):
                    continue
                if ("block", dep) in allocations:
                    restore("block", dep)
                else:
                    if dep in opened:
                        raise UnavailableInput("original blocks may only be read once")
                    block = self.blocks[dep]
                    allocation = ResourceAllocation("block", dep, block.host_bytes, block.device_bytes)
                    allocations[("block", dep)] = allocation
                    remember(allocation)
                    opened.add(dep)
                    actions.append(self.read_actions[dep])
                    wh, wd = max(wh, block.staging_host_bytes), max(wd, block.staging_device_bytes)
            actions.append(self.load_actions[key])
            prep_cost += self.preparation_costs[key]
            loaded.add(key)
            if key in cursor.state.catalog:
                size = cursor.state._snapshot_bytes(key)
                allocation = ResourceAllocation("tensor", key, 0 if self.gpu else size, size if self.gpu else 0)
                allocations[("tensor", key)] = allocation
                remember(allocation)
            else:
                borrowed[key] = prep.blocks
            wh, wd = max(wh, prep.workspace_host_bytes), max(wd, prep.workspace_device_bytes)
        if operation.tensor not in loaded:
            size = _nbytes(step.state.catalog[operation.tensor])
            allocation = ResourceAllocation("tensor", operation.tensor, 0 if self.gpu else size, size if self.gpu else 0)
            allocations[("tensor", operation.tensor)] = allocation
            remember(allocation)
        loaded.add(operation.tensor)
        actions.extend(step.actions)
        producers = dict(cursor.producers)
        if not operation.reused:
            producers[operation.tensor] = operation.operands
        state = replace(step.state, loaded=loaded, seconds=cursor.state.seconds + sum(a.seconds for a in actions),
                        actions=(), peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
        result = replace(cursor, state=state, resources=snapshot(), producers=tuple(sorted(producers.items())),
            preparation_seconds=cursor.preparation_seconds + prep_cost)
        bound = self._maxima_point(high, "complete_step_bound", wh, wd)
        return result, actions, bound

    def _terminal(self, cursor, actions, checkpoint):
        """Final output restoration/copy is a separate aggregate lifetime."""
        outputs = set(cursor.state.results)
        while True:
            checkpoint()
            trial = cursor
            snippet = []
            high = {(a.domain, a.key): a for a in cursor.resources.allocations}
            for key in sorted(outputs):
                while not self.usable(trial, "tensor", key):
                    checkpoint()
                    owner = self.owner(trial, "tensor", key)
                    if owner is None:
                        raise UnavailableInput("output has no preserved copy")
                    kind = "restore_host" if owner.disk_bytes else "restore_device"
                    action = self._move(trial, kind, "tensor", key)
                    snap, point = transition_resource(trial.resources, action,
                        backend=self.problem.backend, memory_limit=1 << 100)
                    for a in snap.allocations:
                        old = high.get((a.domain, a.key), a)
                        high[(a.domain, a.key)] = replace(a, host_bytes=max(old.host_bytes, a.host_bytes),
                            device_bytes=max(old.device_bytes, a.device_bytes), disk_bytes=max(old.disk_bytes, a.disk_bytes))
                    trial = replace(trial, resources=snap,
                                    state=replace(trial.state, seconds=trial.state.seconds + action.seconds))
                    snippet.append(action)
            copies = sum(_nbytes(cursor.state.catalog[k]) for k in cursor.state.results)
            point = self._point_at(ResourceSnapshot(tuple(high.values())), "output_bound",
                                  0 if self.gpu else copies, copies if self.gpu else 0)
            try:
                self.aggregate_checks += 1
                return self._record(trial, point), actions + snippet
            except MemoryError:
                dead = next(iter(self._dead_actions(cursor, outputs, point)), None)
                if dead is None:
                    raise
                cursor = self.apply(cursor, dead)
                actions.append(dead)

    def evaluate(self, cursor, step, checkpoint=lambda: None, *, retained=()):
        step = self._materialize_logical(step, checkpoint)
        operation = step.actions[0]
        protected = {operation.tensor} if operation.reused else set(operation.operands)
        protected.update(retained)
        prefix = []
        while True:
            result, actions, point = self._prepare_step(cursor, step, checkpoint)
            try:
                checkpoint()
                self.aggregate_checks += 1
                result = self._record(result, point)
                break
            except MemoryError:
                dead = next(iter(self._dead_actions(cursor, protected, point)), None)
                if dead is None:
                    raise
                checkpoint()
                cursor = self.apply(cursor, dead)
                prefix.append(dead)
        actions = prefix + actions
        if result.state.done:
            result, actions = self._terminal(result, actions, checkpoint)
        return MacroTransition(operation_identity(operation), result, tuple(actions), operation,
                               step.pending_delta, step.results_delta)

    def preview(self, cursor, action, checkpoint=lambda: None):
        """Diagnostic fixed-operation evaluation; normal search describes once."""
        return self.evaluate(cursor, self._describe(cursor, action, checkpoint), checkpoint)

    def successors(self, cursor, checkpoint=lambda: None):
        """Collect transitions for diagnostics; DFS consumes the iterator."""
        return tuple(self.iter_successors(cursor, checkpoint))

    def materialize(self, cursor, operation, checkpoint=lambda: None):
        """Recreate a selected successor using this state's fixed preparation."""
        try:
            if operation.kind in ("discard", "spill_host", "spill_disk"):
                checkpoint()
                return MacroTransition(operation, self.apply(cursor, operation), (operation,), operation)
            return self.preview(cursor, operation, checkpoint)
        finally:
            self._release_facts(cursor)

    def materialize_actions(self, cursor, actions, checkpoint=lambda: None, *, retained=()):
        """Prepare a canonical complete step delivered by the integer domain."""
        from ..canonical_operations import replay, describe
        state = self._symbolic(cursor)
        first = actions[0]
        join = (next(j for j in self.problem.baryon_joins
                     if j.scene == first.term_index and j.name == first.join_id)
                if first.kind == "baryon_join" else None)
        _, _, workspace, _, _ = describe(state, first.term_index, first.positions,
            join=join, profile=self.logical_profile, kernel_path=first.kernel_path if join else None,
            output_order=first.equation.output if join is None else None)
        for action in actions:
            checkpoint()
            state = replay(state, action, self.logical_profile)
        step = LogicalStep(replace(state, actions=()), tuple(actions),
                           0 if self.gpu else workspace, workspace if self.gpu else 0)
        try:
            return self.evaluate(cursor, step, checkpoint, retained=retained)
        finally:
            self._release_facts(cursor)

    @staticmethod
    def _release_facts(cursor):
        # Facts accelerate one expansion, but arrival records can retain a
        # cursor for the whole search. Keep complete keys, release batch facts.
        for name in ("owners", "borrowed", "useful_tensors"):
            cursor.__dict__.pop(name, None)
        cursor.state.__dict__.pop("live_inputs", None)

    def iter_successors(self, cursor, checkpoint=lambda: None):
        try:
            yield from self._iter_successors(cursor, checkpoint)
        finally:
            self._release_facts(cursor)

    def iter_bounded_successors(self, cursor, cutoff, checkpoint):
        """The engine's live top-k cost cutoff never affects legality."""
        return BoundedSuccessors(self, cursor, cutoff, checkpoint)

    def _increment_bound(self, cursor, step):
        """Exact fixed preparation cost, excluding nonnegative final recovery.

        Unknown recovery/cold-copy cases use full preview. Stable preparation
        order and the original arithmetic order make ties safe to truncate.
        Protected data/blocks cannot be reclaimed by the dead-cache policy.
        """
        operation = step.actions[0]
        keys = (operation.tensor,) if operation.reused else operation.operands
        records = self.bound_recipes.get(keys)
        if records is None:
            items = []
            for key in sorted(set(keys)):
                prep = self.preparations.get(key)
                recipe = ((tuple(self._ordered(prep.blocks)),
                           self.load_actions[key].seconds, self.preparation_costs[key]) if prep else None)
                items.append((key, recipe))
            records = tuple(items)
            if len(self.bound_recipes) < 2048:
                self.bound_recipes[keys] = records
        costs, opened = [], set()
        prep_cost = 0.0
        for key, recipe in records:
            if self.usable(cursor, "tensor", key):
                continue
            if key in cursor.state.loaded:
                return None
            if recipe is None:
                return None
            dependencies, load_fee, prep_fee = recipe
            for dep in dependencies:
                if dep in opened or self.usable(cursor, "block", dep):
                    continue
                if self.owner(cursor, "block", dep) is not None or dep in cursor.resources.opened_blocks:
                    return None
                opened.add(dep)
            costs.append(load_fee)
            prep_cost += prep_fee
        costs.extend(action.seconds for action in step.actions)
        price = cursor.state.seconds + sum(costs) + cursor.preparation_seconds + prep_cost
        return price - self.cost(cursor)

    def _iter_successors(self, cursor, checkpoint, *, batch=None):
        """Yield previews one at a time, retaining no full sibling snapshots."""
        count = 0
        effects = []
        symbolic = None
        feasible = memory = 0
        host_pressure = False
        host_cap = min(self.memory_limit if not self.gpu else math.inf,
                       self.host_memory_limit if self.host_memory_limit is not None else math.inf)
        for step in self._logical_candidates(cursor, checkpoint):
            count += 1
            checkpoint()
            # Only an already feasible candidate can establish a cutoff.
            # Before that point every contraction is fully checked, ensuring
            # useful-data adjustments are entered only after all fail memory.
            if batch is not None and feasible:
                bound = self._increment_bound(cursor, step)
                if bound is not None and math.isfinite(bound) and bound >= batch.cutoff():
                    batch.omitted += 1
                    self.omitted_previews += 1
                    continue
            step = self._materialize_logical(step, checkpoint)
            try:
                edge = self.evaluate(cursor, step, checkpoint)
            except MemoryError as exc:
                point = getattr(exc, "point", None)
                host_pressure |= point is not None and point.host_bytes > host_cap
                memory += 1
                self.rejected_memory += 1
                if not feasible:
                    if symbolic is None:
                        symbolic = self._symbolic(cursor)
                    effects.append(self._freeze_effect(symbolic, step))
            except UnavailableInput:
                self.unavailable += 1
            else:
                feasible += 1
                effects.clear()
                yield edge
                del edge
        if feasible:
            return
        if not count or memory != count:
            return
        self.logical_cache = _state_key(symbolic), tuple(effects)
        self.resource_branches += 1
        # Restoration is never an independent candidate. A successful move is
        # followed by another full contraction check on the next expansion.
        # Host spill is permitted only under host pressure, including the
        # space needed to receive a device spill. Spare host caches stay put.
        host_receivers = set()
        for owner in cursor.resources.allocations:
            if not owner.device_bytes:
                continue
            checkpoint()
            action = self._move(cursor, "spill_host", owner.domain, owner.key)
            try:
                self.apply(cursor, action)
                host_receivers.add((owner.domain, owner.key))
            except MemoryError as exc:
                point = getattr(exc, "point", None)
                host_pressure |= point is not None and point.host_bytes > host_cap
            except ValueError:
                pass
        for owner in sorted(cursor.resources.allocations, key=lambda a: (a.domain, a.key)):
            for kind in ("discard", "spill_host", "spill_disk"):
                if kind == "discard" and owner.host_bytes and not owner.device_bytes and not host_pressure:
                    continue
                if kind == "spill_host" and not owner.device_bytes:
                    continue
                if kind == "spill_disk" and (not owner.host_bytes or not host_pressure):
                    continue
                if not (owner.host_bytes or owner.device_bytes or owner.disk_bytes):
                    continue
                checkpoint()
                action = self._move(cursor, kind, owner.domain, owner.key)
                if kind == "spill_host" and (owner.domain, owner.key) not in host_receivers:
                    continue
                try:
                    result = self.apply(cursor, action)
                except (MemoryError, ValueError):
                    continue
                if self.key(result) != self.key(cursor):
                    yield MacroTransition(action, result, (action,), action)
                del result
        for key, _ in sorted(cursor.resources.borrowed):
            checkpoint()
            action = self._move(cursor, "discard", "tensor", key)
            try:
                result = self.apply(cursor, action)
            except (MemoryError, ValueError):
                continue
            yield MacroTransition(action, result, (action,), action)
            del result

    def freeze_transition(self, before, edge):
        """Keep only changes for retained candidates; release the preview state."""
        after = edge.state
        before_allocations = set(before.resources.allocations)
        after_allocations = set(after.resources.allocations)
        before_borrowed, after_borrowed = set(before.resources.borrowed), set(after.resources.borrowed)
        before_producers = set(before.producers)
        key = edge.operation.tensor
        return StepDelta(edge.operation, edge.actions,
            ((key, after.state.catalog[key]),) if key not in before.state.catalog and key in after.state.catalog else (),
            edge.pending_delta if edge.pending_delta is not None else
                tuple((i, v) for i, v in enumerate(after.state.pending) if v != before.state.pending[i]),
            edge.results_delta if edge.results_delta is not None else
                tuple((i, v) for i, v in enumerate(after.state.results) if v != before.state.results[i]),
            after.state.completed_joins - before.state.completed_joins,
            after.state.loaded - before.state.loaded, before.state.loaded - after.state.loaded,
            tuple(a for a in after.resources.allocations if a not in before_allocations),
            tuple(a for a in before.resources.allocations if a not in after_allocations),
            tuple(a for a in after.resources.borrowed if a not in before_borrowed),
            tuple(a for a in before.resources.borrowed if a not in after_borrowed),
            after.resources.opened_blocks - before.resources.opened_blocks,
            tuple(a for a in after.producers if a not in before_producers),
            after.state.seconds - before.state.seconds,
            after.preparation_seconds - before.preparation_seconds,
            after.peak_host, after.peak_device, after.peak_disk)

    def apply_delta(self, cursor, delta, checkpoint=lambda: None):
        checkpoint()
        self.selected_delta_applications += 1
        pending = _patch_tuple(cursor.state.pending, delta.pending)
        results = _patch_tuple(cursor.state.results, delta.results)
        state = replace(cursor.state, catalog=cursor.state.catalog.updated(delta.catalog) if delta.catalog else cursor.state.catalog,
            pending=pending, results=results, completed_joins=cursor.state.completed_joins | delta.joins,
            loaded=(cursor.state.loaded - delta.loaded_remove) | delta.loaded_add,
            seconds=cursor.state.seconds + delta.seconds, actions=())
        snapshot = ResourceSnapshot(
            _apply_tuple_delta(cursor.resources.allocations, delta.allocations_remove, delta.allocations_add),
            _apply_tuple_delta(cursor.resources.borrowed, delta.borrowed_remove, delta.borrowed_add),
            cursor.resources.opened_blocks | delta.opened)
        result = replace(cursor, state=state, resources=snapshot,
            preparation_seconds=cursor.preparation_seconds + delta.preparation_seconds,
            producers=_merge_sorted_items(cursor.producers, delta.producers),
            peak_host=max(cursor.peak_host, delta.peak_host), peak_device=max(cursor.peak_device, delta.peak_device),
            peak_disk=max(cursor.peak_disk, delta.peak_disk))
        return MacroTransition(operation_identity(delta.operation), result, delta.actions, delta.operation)

    def plan(self, cursor, actions, checkpoint=lambda: None):
        if not self.mandatory_blocks <= cursor.resources.opened_blocks:
            raise RuntimeError("complete path omitted a mandatory IO block")
        state = replace(cursor.state, actions=tuple(actions))
        plan = audit_resource_trajectory(state, self.blocks, self.preparations, profile=self.profile,
            memory_limit=self.memory_limit, host_memory_limit=self.host_memory_limit,
            disk_limit_bytes=self.disk_limit_bytes, reserved_host_bytes=self.reserved_host_bytes,
            reserved_device_bytes=self.reserved_device_bytes, checkpoint=checkpoint)
        if not plan.feasible or not math.isclose(plan.seconds, self.cost(cursor), rel_tol=1e-11, abs_tol=1e-14):
            raise RuntimeError("macro preview disagrees with independent trajectory audit")
        if any(actual > predicted for actual, predicted in zip(
                (plan.peak_host_bytes, plan.peak_device_bytes, plan.peak_disk_bytes),
                (cursor.peak_host, cursor.peak_device, cursor.peak_disk))):
            raise RuntimeError("aggregate storage bound failed independent trajectory audit")
        return plan
