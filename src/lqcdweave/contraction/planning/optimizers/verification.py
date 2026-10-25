"""Independent reference replays and index self-checks for a delivered route.

This module is the delivery-insurance facility for the integer graph search in
graph_session.  It is deliberately *not* part of the search: nothing here runs
unless an explicit verify=True / direct call asks for it, and the production
numbers are never touched.

It answers three questions about one fixed winning route:

1. Route replay.  A non-incremental reference walks the route from the head.
   At every step it rebuilds the complete candidate row of every live scene
   from the current complete expression (canonical_operations.describe)
   instead of consulting IntegerSpace's incremental rows / groups or the
   compiled dense hot tables.  It asserts the operation sequence, the
   per-step availability (reused) and cost, the resulting expression state
   (pending / loaded / catalog) and, at the end, the delivered Action table
   field by field.
2. Incremental index self-check.  At sampled states the incrementally
   maintained groups / _group_weights / group_ordinals / terminal_families
   are compared with a from-scratch rebuild of the same state.
   capture_index_snapshot / rebuild_index_tables / compare_index_tables are
   public so a future in-search hook can call them directly; verify_route
   already samples every index_every steps.
Numerical replay and array-oracle comparison belong to the test harness.
This planner module never imports or invokes the numerical executor.

Known limits (also reported in VerificationReport):

* Only the delivered winning route is re-walked.  Discarded branches, width
  truncation and ranking decisions are not re-derived.
* The reference index rebuild shares the live IntegerRegistry; it is
  documented as a debug-only entry point.
* solve_graph_storage(verify=True) covers the serial in-process path.  The
  parallel worker path (component_workers) is not covered; verify the serial
  route it produces instead.
* Numerical replay is exercised by tests with explicit arrays and is never run
  by solve_graph_storage itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math

from lqcdweave.contraction.planning.actions import _live_inputs
from lqcdweave.contraction.planning.canonical_operations import apply_description, describe
from lqcdweave.contraction.planning.state import ContractionState


class VerificationError(AssertionError):
    """An independent replay disagreed with the fast path."""


# Fields that make up the delivered Action identity.  resource and core are
# intentionally excluded: they are object references into the immutable
# problem, not recomputed values.
ACTION_FIELDS = ("kind", "tensor", "operands", "equation", "seconds", "reused",
                 "used_np", "used_np_right", "join_id", "left_axes", "right_axes",
                 "same_point", "transpose_bridge", "term_index", "positions",
                 "kernel_path")


def action_fields(action):
    """Return the comparable identity of one Action as a plain mapping."""
    equation = action.equation
    return dict(
        kind=action.kind,
        tensor=action.tensor,
        operands=tuple(action.operands),
        equation=None if equation is None else (tuple(tuple(item) for item in equation.inputs),
                                                tuple(equation.output)),
        seconds=action.seconds,
        reused=action.reused,
        used_np=action.used_np,
        used_np_right=action.used_np_right,
        join_id=action.join_id,
        left_axes=tuple(action.left_axes),
        right_axes=tuple(action.right_axes),
        same_point=action.same_point,
        transpose_bridge=action.transpose_bridge,
        term_index=action.term_index,
        positions=tuple(action.positions),
        kernel_path=tuple(tuple(step) for step in action.kernel_path),
    )


def compare_actions(expected, actual, *, label="route"):
    """Raise VerificationError on the first differing Action field."""
    expected, actual = tuple(expected), tuple(actual)
    if len(expected) != len(actual):
        raise VerificationError(
            f"{label}: reference Action count {len(expected)} != fast path {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left == right:
            continue
        left_fields, right_fields = action_fields(left), action_fields(right)
        differing = [name for name in ACTION_FIELDS if left_fields[name] != right_fields[name]]
        detail = ", ".join(f"{name}={left_fields[name]!r} != {right_fields[name]!r}"
                           for name in differing)
        raise VerificationError(f"{label}: Action {index} differs: {detail}")
    return True


@dataclass(frozen=True)
class ReferenceStep:
    """What the reference derived for one step of the route."""
    index: int
    calculation: int
    result: int
    tensor: str
    reused: bool
    seconds: float
    scenes: tuple
    actions: tuple
    generated: frozenset
    remaining: int


@dataclass(frozen=True)
class Candidate:
    """One from-scratch legal computation in a rebuilt scene row."""
    scene: int
    positions: tuple
    join: object
    action: object
    spec: object
    workspace: int
    output: tuple
    output_order: tuple = None


@dataclass(frozen=True)
class ReferenceRoute:
    problem: object
    steps: tuple
    state: ContractionState
    actions: tuple
    seconds: float


class ReferenceReplay:
    """Non-incremental complete-row replay of one fixed integer route.

    describe is the canonical definitional descriptor.  The incremental dense
    fast path binds the same facts through compiled hot tables, so comparing
    the two is a genuine cross-check of the optimized path.
    """

    def __init__(self, problem, profile, registry, *, checkpoint=None, debug=True):
        self.problem = problem
        self.profile = profile
        self.registry = registry
        self.checkpoint = (lambda: None) if checkpoint is None else checkpoint
        self.debug = debug
        self.rows_built = 0
        self.descriptions = 0

    def _checkpoint(self):
        self.checkpoint()

    def initial_state(self):
        state = ContractionState.initial(self.problem)
        return replace(state, loaded=frozenset(state.catalog) | _live_inputs(state))

    def rows(self, state):
        """Rebuild the complete candidate row of every live scene from scratch."""
        rows = {}
        for scene, term in enumerate(state.pending):
            if term is None:
                continue
            self._checkpoint()
            candidates = []
            count = len(term.factors)
            pairs = ((0,),) if count == 1 else combinations(range(count), 2)
            for positions in pairs:
                candidate = self._contract_candidate(state, scene, tuple(positions))
                if candidate is not None:
                    candidates.append(candidate)
            for join in state.problem.baryon_joins:
                if join.scene != scene or (scene, join.name) in state.completed_joins:
                    continue
                candidate = self._join_candidate(state, scene, join)
                if candidate is not None:
                    candidates.append(candidate)
            if candidates:
                rows[scene] = candidates
        self.rows_built += 1
        return rows

    def _contract_candidate(self, state, scene, positions, *, output_order=None):
        self._checkpoint()
        try:
            action, spec, workspace, output, _ = describe(
                state, scene, positions, profile=self.profile, output_order=output_order)
        except (ValueError, NotImplementedError):
            return None
        self.descriptions += 1
        return Candidate(scene, positions, None, action, spec, workspace, tuple(output),
                         None if output_order is None else tuple(output_order))

    def _join_candidate(self, state, scene, join):
        self._checkpoint()
        try:
            action, spec, workspace, output, _ = describe(state, scene, join=join, profile=self.profile)
        except (ValueError, NotImplementedError):
            return None
        self.descriptions += 1
        return Candidate(scene, tuple(action.positions), join, action, spec, workspace, tuple(output))

    def _candidate_positions(self, term, operands):
        """Cheap operand-name scan used when a full row rebuild is not sampled."""
        count = len(term.factors)
        if len(operands) == 1 or count == 1:
            if not operands:
                return ()
            return tuple((index,) for index, factor in enumerate(term.factors)
                         if factor.tensor == operands[0])
        # The delivered action's operand order is scene-specific: the recipe
        # records one order but a scene may bind the same computation in the
        # other order.  Match the operand multiset instead of the recipe order.
        wanted = sorted(operands[:2])
        return tuple((left, right) for left in range(count) for right in range(count)
                     if left != right
                     and sorted((term.factors[left].tensor, term.factors[right].tensor)) == wanted)

    def locate_targeted(self, state, operation):
        """Locate one operation without rebuilding unrelated candidate rows.

        Each candidate is still described independently through describe; only
        the enumeration of unrelated pairs is skipped.  Used between sampled
        full-row rebuilds so a long production route stays verifiable.
        """
        calculation = self.registry.calculations[operation.calculation]
        recipe = calculation.recipe
        target = self.registry.data[operation.result].name
        matches, seen = [], set()
        for scene, term in enumerate(state.pending):
            if term is None:
                continue
            if recipe.kind == "baryon_join":
                join = next((item for item in state.problem.baryon_joins
                             if item.scene == scene and item.name == recipe.join_id), None)
                if join is None or (scene, join.name) in state.completed_joins:
                    continue
                candidate = self._join_candidate(state, scene, join)
                if candidate is not None and candidate.action.tensor == target:
                    matches.append(candidate)
                continue
            for positions in self._candidate_positions(term, recipe.operands):
                candidate = self._contract_candidate(state, scene, positions)
                if candidate is None or candidate.action.tensor != target:
                    output_order = self._raw_output_order(state, scene, positions, recipe)
                    if output_order is None:
                        output_order = recipe.equation.output
                    candidate = self._contract_candidate(state, scene, positions,
                                                         output_order=output_order)
                    if candidate is None or candidate.action.tensor != target:
                        continue
                if not operation.reused and (candidate.action.seconds != calculation.seconds
                                             or candidate.workspace != calculation.workspace):
                    continue
                if candidate.scene in seen:
                    continue
                seen.add(candidate.scene)
                matches.append(candidate)
        return calculation, target, matches

    def _raw_output_order(self, state, scene, positions, recipe):
        """Translate a registered recipe's output axes back to raw scene labels.

        The compiled dense fast path registers recipes with the pattern's
        normalized axis ids, while a live expression uses the problem's raw
        labels.  Operand axes are positionally associated (recipe.operands[i]
        carries recipe.equation.inputs[i]), so the mapping is recovered from
        the factors actually selected in this scene.
        """
        equation = recipe.equation
        if equation is None:
            return None
        term = state.pending[scene]
        selected = [term.factors[index] for index in positions]
        if len(selected) != len(equation.inputs):
            return None
        mapping = {}
        for factor, axes in zip(selected, equation.inputs):
            if len(factor.labels) != len(axes):
                return None
            for raw, normalized in zip(factor.labels, axes):
                if mapping.setdefault(normalized, raw) != raw:
                    return None
        output = tuple(mapping.get(axis, axis) for axis in equation.output)
        if len(output) != len(set(output)):
            return None
        return output

    def locate(self, state, rows, operation):
        """Find the route operation's occurrences in the rebuilt rows.

        Matching is driven by the canonical tensor name; the recipe's operand
        multiset only gates the variant-output fallback.  The incremental index
        is never consulted, and one occurrence per scene is kept because that
        is what the index maintains.
        """
        calculation = self.registry.calculations[operation.calculation]
        recipe = calculation.recipe
        target = self.registry.data[operation.result].name
        matches, seen = [], set()
        for scene in sorted(rows):
            for candidate in rows[scene]:
                if candidate.join is not None:
                    if candidate.action.tensor == target:
                        matches.append(candidate)
                    continue
                positions = candidate.positions
                # A direct canonical name hit is the located occurrence.  The
                # canonical name encodes the retained axis relation, so an
                # operand order that differs from the recipe still matches.
                if candidate.action.tensor == target:
                    match = candidate
                else:
                    # Variant recipes registered by register_layouts are not
                    # part of the default row; re-describe the same operands
                    # with the recipe's output order.  Keep the cheap operand
                    # multiset filter before paying for another description.
                    names = tuple(state.pending[scene].factors[index].tensor for index in positions)
                    if sorted(names) != sorted(recipe.operands[:len(names)]):
                        continue
                    output_order = self._raw_output_order(state, scene, positions, recipe)
                    if output_order is None:
                        output_order = recipe.equation.output
                    match = self._contract_candidate(state, scene, positions,
                                                     output_order=output_order)
                    if match is None or match.action.tensor != target:
                        continue
                if not operation.reused and (match.action.seconds != calculation.seconds
                                             or match.workspace != calculation.workspace):
                    continue
                # The incremental index keeps exactly one occurrence per scene
                # (the first in row order), so the reference must consume the
                # same one instead of applying every identical occurrence.
                if match.scene in seen:
                    continue
                seen.add(match.scene)
                matches.append(match)
        return calculation, target, matches

    def advance(self, state, operation, index, *, full=True):
        """Apply one route operation to the reference and return its record.

        full=True rebuilds every live scene's complete candidate row and, as a
        cross-check, also runs the targeted scan and requires both to locate
        the same occurrences.  full=False uses only the targeted scan, which is
        what keeps long production routes affordable between sampled rebuilds.
        """
        calculation = self.registry.calculations[operation.calculation]
        target = self.registry.data[operation.result].name
        already = target in state.loaded
        if operation.reused != already:
            raise VerificationError(
                f"step {index}: route says reused={operation.reused} but the reference expression "
                f"has {target!r} {'resident' if already else 'absent'}")
        registered_reused_seconds = 0.0 if already else calculation.seconds
        if operation.seconds != registered_reused_seconds:
            raise VerificationError(
                f"step {index}: route cost {operation.seconds!r} != registered "
                f"{registered_reused_seconds!r} for calculation {operation.calculation}")
        if full:
            rows = self.rows(state)
            _, _, matches = self.locate(state, rows, operation)
            _, _, targeted = self.locate_targeted(state, operation)
            located = {(match.scene, match.positions) for match in matches}
            scanned = {(match.scene, match.positions) for match in targeted}
            if located != scanned:
                raise VerificationError(
                    f"step {index}: full-row and targeted occurrence scans disagree: "
                    f"full-only {sorted(located - scanned)[:4]}, "
                    f"targeted-only {sorted(scanned - located)[:4]}")
        else:
            _, _, matches = self.locate_targeted(state, operation)
        if not matches:
            raise VerificationError(
                f"step {index}: route operation {operation.calculation} -> {target!r} has no "
                f"from-scratch matching occurrence")
        before_actions, before_seconds = len(state.actions), state.seconds
        scenes, step_actions = [], []
        for candidate in sorted(matches, key=lambda item: item.scene):
            output_order = None if candidate.join is not None else candidate.output_order
            action, spec, workspace, output, positions = describe(
                state, candidate.scene, candidate.positions, join=candidate.join,
                profile=self.profile, output_order=output_order)
            if action.tensor != target:
                raise VerificationError(
                    f"step {index}: scene {candidate.scene} re-description produced "
                    f"{action.tensor!r} instead of {target!r}")
            state = apply_description(state, action, spec, workspace, output, positions,
                                      join=candidate.join)
            scenes.append(candidate.scene)
            step_actions.append(action)
        delta = state.seconds - before_seconds
        if not math.isclose(delta, operation.seconds, rel_tol=1e-11, abs_tol=1e-14):
            raise VerificationError(
                f"step {index}: reference charged {delta!r} seconds but the route says "
                f"{operation.seconds!r}")
        generated = frozenset(state.catalog) - frozenset(self.problem.specs)
        record = ReferenceStep(index, operation.calculation, operation.result, target,
                               already, operation.seconds, tuple(scenes), tuple(step_actions),
                               generated, sum(term is not None for term in state.pending))
        return state, record

    def run(self, operations):
        operations = tuple(operations)
        state = self.initial_state()
        steps = []
        for index, operation in enumerate(operations):
            self._checkpoint()
            state, record = self.advance(state, operation, index)
            steps.append(record)
        return ReferenceRoute(self.problem, tuple(steps), state, tuple(state.actions), state.seconds)


def reference_route(problem, profile, registry, operations, *, checkpoint=None, debug=True):
    """Convenience wrapper: run one full independent reference replay."""
    return ReferenceReplay(problem, profile, registry, checkpoint=checkpoint, debug=debug).run(operations)


# ---------------------------------------------------------------------------
# Incremental index self-check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexSnapshot:
    """One sampled search state plus the incrementally maintained index tables."""
    step: int
    terms: tuple
    available: frozenset
    catalog: dict
    recent_scenes: frozenset
    remaining: int
    groups: dict
    group_weights: dict
    group_ordinals: dict
    next_group_ordinal: int
    terminal_families: dict
    terminal_members: dict
    layout_variants: frozenset


@dataclass(frozen=True)
class IndexTables:
    """A from-scratch rebuild of the maintained index tables."""
    groups: dict
    group_weights: dict
    group_ordinals: dict
    next_group_ordinal: int
    terminal_families: dict
    terminal_members: dict
    layout_variants: frozenset


def capture_index_snapshot(space, step=0):
    """Flush and freeze the incremental index of one live IntegerSpace."""
    space._index(lambda: None)
    return IndexSnapshot(
        step=step,
        terms=tuple(space.terms),
        available=frozenset(space.available),
        catalog=dict(space.catalog),
        recent_scenes=space.recent_scenes,
        remaining=space.remaining,
        groups={identity: dict(occurrences) for identity, occurrences in space.groups.items()},
        group_weights=dict(space._group_weights),
        group_ordinals=dict(space.group_ordinals),
        next_group_ordinal=space.next_group_ordinal,
        terminal_families=dict(space.terminal_families),
        terminal_members={family: frozenset(members)
                          for family, members in space.terminal_members.items()},
        layout_variants=frozenset(space.layout_variants),
    )


def rebuild_index_tables(problem, profile, registry, snapshot, *, debug=False, checkpoint=None):
    """Rebuild groups and friends from scratch for a sampled state.

    A fresh workspace is seeded with the snapshot's logical state and then
    indexed in one full pass, so no incremental removal/recreation events are
    involved.  The shared registry is reused; the returned tables are only
    meaningful against that same registry.
    """
    from .integer_space import IntegerSpace
    checkpoint = (lambda: None) if checkpoint is None else checkpoint
    space = IntegerSpace(problem, profile=profile, registry=registry, debug=debug,
                         checkpoint=checkpoint)
    space.terms = list(snapshot.terms)
    space.available = set(snapshot.available)
    space.catalog = dict(snapshot.catalog)
    space.recent_scenes = snapshot.recent_scenes
    space.remaining = snapshot.remaining
    space.rows, space.groups = {}, {}
    space._group_weights = {}
    space.group_ordinals, space.next_group_ordinal = {}, 0
    space.terminal_families, space.terminal_members, space.terminal_heap = {}, {}, []
    space.layout_variants = set()
    if space._scalar_terminals is not None:
        space._scalar_terminals.reset()
    space.dirty = set(range(len(space.terms)))
    space._index(checkpoint)
    return IndexTables(
        groups={identity: dict(occurrences) for identity, occurrences in space.groups.items()},
        group_weights=dict(space._group_weights),
        group_ordinals=dict(space.group_ordinals),
        next_group_ordinal=space.next_group_ordinal,
        terminal_families=dict(space.terminal_families),
        terminal_members={family: frozenset(members)
                          for family, members in space.terminal_members.items()},
        layout_variants=frozenset(space.layout_variants),
    )


def _table_diff(name, expected, actual):
    if expected == actual:
        return []
    if isinstance(expected, dict):
        keys = set(expected) | set(actual)
        return [f"{name}[{key!r}]: {expected.get(key)!r} != {actual.get(key)!r}"
                for key in sorted(keys, key=repr) if expected.get(key) != actual.get(key)]
    return [f"{name}: {expected!r} != {actual!r}"]


def compare_index_tables(snapshot, tables, *, strict_ordinals=False):
    """Raise VerificationError when the incremental index drifted.

    groups, _group_weights, terminal_families, terminal_members and
    layout_variants must equal a from-scratch rebuild exactly.

    group_ordinals and next_group_ordinal are historical labels: the
    incremental maintenance deliberately keeps a surviving calculation's old
    ordinal and only assigns fresh ones to recreated groups, so a later state's
    absolute numbering is *not* expected to equal a from-scratch numbering even
    when the index is correct.  They are therefore checked for structural
    consistency (one unique non-negative ordinal per live group, behind the
    running counter).  Pass strict_ordinals=True only for a state whose index
    was built in a single pass, such as the initial state.
    """
    problems = []
    problems += _table_diff("groups", snapshot.groups, tables.groups)
    problems += _table_diff("_group_weights", snapshot.group_weights, tables.group_weights)
    problems += _table_diff("terminal_families", snapshot.terminal_families, tables.terminal_families)
    problems += _table_diff("terminal_members", snapshot.terminal_members, tables.terminal_members)
    problems += _table_diff("layout_variants", snapshot.layout_variants, tables.layout_variants)
    live = set(snapshot.groups)
    ordinals = snapshot.group_ordinals
    if set(ordinals) != live:
        problems.append(f"group_ordinals keys {sorted(set(ordinals) ^ live)} do not match live groups")
    values = [value for value in ordinals.values() if type(value) is int]
    if len(values) != len(ordinals):
        problems.append("group_ordinals contains a non-integer value")
    elif len(set(values)) != len(values):
        problems.append("group_ordinals contains a duplicate ordinal")
    if snapshot.next_group_ordinal < len(ordinals):
        problems.append(
            f"next_group_ordinal {snapshot.next_group_ordinal} < live group count {len(ordinals)}")
    if values and max(values) >= snapshot.next_group_ordinal:
        problems.append(
            f"ordinal {max(values)} is not behind next_group_ordinal {snapshot.next_group_ordinal}")
    if strict_ordinals:
        problems += _table_diff("group_ordinals", snapshot.group_ordinals, tables.group_ordinals)
        problems += _table_diff("next_group_ordinal", snapshot.next_group_ordinal,
                                tables.next_group_ordinal)
    if problems:
        raise VerificationError(
            f"index self-check failed at step {snapshot.step}: " + "; ".join(problems[:8]))
    return True


# ---------------------------------------------------------------------------
# Whole-route verification
# ---------------------------------------------------------------------------

@dataclass
class VerificationReport:
    """What was actually checked, and what was deliberately not."""
    kind: str
    steps: int = 0
    index_samples: int = 0
    numeric_checked: bool = False
    reference_actions: int = 0
    fast_actions: int = 0
    limitations: tuple = ()
    registry_growth: int = 0

    def summary(self):
        return dict(kind=self.kind, steps=self.steps, index_samples=self.index_samples,
                    numeric_checked=self.numeric_checked,
                    reference_actions=self.reference_actions, fast_actions=self.fast_actions,
                    registry_growth=self.registry_growth, limitations=list(self.limitations))


_LIMITATIONS = (
    "only the delivered winning route is re-walked; discarded branches, width "
    "truncation and ranking are not re-derived",
    "the reference index rebuild shares the live IntegerRegistry",
    "parallel component workers and multi-process paths are not covered",
    "numerical replay needs caller supplied source arrays",
)


def _compare_expression(index, state, space, problem):
    reference_catalog = frozenset(state.catalog)
    fast_catalog = frozenset(space.catalog)
    if reference_catalog != fast_catalog:
        extra = sorted(reference_catalog - fast_catalog)
        missing = sorted(fast_catalog - reference_catalog)
        raise VerificationError(
            f"step {index}: expression catalog differs (reference-only {extra[:4]}, "
            f"fast-only {missing[:4]})")

    def decode(scene):
        term = space.terms[scene]
        return None if term.done else space._decode(scene, raw=True)

    reference_pending = tuple(state.pending)
    if space.scene_members is None:
        fast_pending = tuple(decode(scene) for scene in range(len(space.terms)))
    else:
        pending = [None] * len(problem.terms)
        for representative, members in enumerate(space.scene_members):
            value = decode(representative)
            for scene in members:
                pending[scene] = value
        fast_pending = tuple(pending)
    if reference_pending != fast_pending:
        for scene, (left, right) in enumerate(zip(reference_pending, fast_pending)):
            if left != right:
                raise VerificationError(
                    f"step {index}: scene {scene} expression differs: reference {left!r} != fast {right!r}")
        raise VerificationError(f"step {index}: expression row count differs")

    reference_live = sum(term is not None for term in reference_pending)
    if space.scene_members is None:
        fast_live = sum(not term.done for term in space.terms)
    else:
        fast_live = sum(space._row_weights[scene_index]
                        for scene_index, term in enumerate(space.terms) if not term.done)
    if reference_live != fast_live:
        raise VerificationError(
            f"step {index}: live scene count {reference_live} != {fast_live}")


def verify_route(problem, profile, registry, operations, expected_cost, *,
                 fast_state=None, debug=False, index_every=0, full_every=1, checkpoint=None):
    """Verify one fixed winning route against independent replays.

    problem, profile, registry : the exact inputs the search used.
    operations : the route's IntegerOperation sequence.
    expected_cost : the accepted logical best cost.
    fast_state : the delivered Action state to compare the final table with.
    index_every : when positive, compare the incremental index tables with a
        from-scratch rebuild every index_every steps.
    full_every : rebuild the complete candidate row every full_every steps.
        Every step is still re-described from scratch, but between rebuilds the
        reference uses the targeted scan; the two are cross-checked whenever a
        full rebuild runs.  full_every=1 checks every step's whole row.
    """
    from .integer_space import IntegerSpace

    checkpoint = (lambda: None) if checkpoint is None else checkpoint
    operations = tuple(operations)
    report = VerificationReport(kind="route", steps=len(operations), limitations=_LIMITATIONS)
    reference = ReferenceReplay(problem, profile, registry, checkpoint=checkpoint, debug=debug)
    state = reference.initial_state()
    space = IntegerSpace(problem, profile=profile, registry=registry, debug=debug,
                         checkpoint=checkpoint)
    baseline_calculations = len(registry.calculations)
    for index, operation in enumerate(operations):
        checkpoint()
        full = full_every <= 1 or index % full_every == 0 or index == len(operations) - 1
        state, record = reference.advance(state, operation, index, full=full)
        space._index(checkpoint)
        if operation.calculation not in space.groups:
            raise VerificationError(
                f"step {index}: fast index has no group for calculation {operation.calculation}")
        fast_reused = operation.result in space.available
        if operation.reused != fast_reused:
            raise VerificationError(
                f"step {index}: route reused={operation.reused} but fast availability says {fast_reused}")
        space.apply(operation, checkpoint)
        _compare_expression(index, state, space, problem)
        if index_every and (index + 1) % index_every == 0:
            snapshot = capture_index_snapshot(space, index + 1)
            tables = rebuild_index_tables(problem, profile, registry, snapshot, debug=debug,
                                          checkpoint=checkpoint)
            compare_index_tables(snapshot, tables)
            report.index_samples += 1
    report.registry_growth = len(registry.calculations) - baseline_calculations
    if not math.isclose(state.seconds, expected_cost, rel_tol=1e-11, abs_tol=1e-14):
        raise VerificationError(
            f"reference route cost {state.seconds!r} != accepted best cost {expected_cost!r}")
    if not state.done:
        raise VerificationError("reference route did not finish the logical expression")
    report.reference_actions = len(state.actions)
    if fast_state is not None:
        compare_actions(state.actions, fast_state.actions)
        report.fast_actions = len(fast_state.actions)
        if not fast_state.done:
            raise VerificationError("fast path state is not done")
        if not math.isclose(fast_state.seconds, expected_cost, rel_tol=1e-11, abs_tol=1e-14):
            raise VerificationError(
                f"fast path cost {fast_state.seconds!r} != accepted best cost {expected_cost!r}")
    return report
