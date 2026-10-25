"""Integer Term workspace and incremental shared-computation occurrence index."""
from dataclasses import dataclass, replace
from copy import copy
import hashlib
import heapq
from itertools import combinations
import math
import numpy as np
from typing import NamedTuple

from lqcdweave.contracts.correlator.api import (
    AxisPattern, BaryonJoinSpec, CoreComponent, FactorizedWideReduction,
    TensorCoreSpec, TensorSpec, Factor, PendingTerm,
)
from ..actions import _live_inputs
from ..canonical_operations import apply_description, axis_key, computation_key, describe, describe_dense, predict_dense, replay
from .dense_patterns import compile_dense_choices
from ..state import ContractionState
from ..logical_index import BridgeIndex
from .shared_components import component_problem
from .choices import OperationPool


# Row slots are checked by their caller; a long row also bounds itself at this
# interval so a deadline is still noticed inside one oversized rebuild. Must be
# a power of two: the hot loop tests it with a subtract-one bitmask.
DENSE_ROW_CHECKPOINT_INTERVAL = 4096


@dataclass(frozen=True, slots=True)
class TermRecord:
    factors: tuple[int, ...]
    pattern: int
    done: bool = False


@dataclass(frozen=True, slots=True)
class DataDefinition:
    name: str
    spec: object
    core: object = None


@dataclass(frozen=True, slots=True)
class Calculation:
    result: int
    seconds: float
    workspace: int
    recipe: object


@dataclass(frozen=True, slots=True)
class IntegerOperation:
    calculation: int
    result: int
    reused: bool
    seconds: float


class Occurrence(NamedTuple):
    scene: int
    positions: tuple[int, ...]
    output: tuple[int, ...]
    join: int | None
    cores: tuple[int, ...]


@dataclass(slots=True)
class IntegerUndo:
    terms: tuple
    created: tuple
    digest: int | None
    recent_scenes: frozenset
    shared_digest: int | None = None


def _normalized_join(join, numbers):
    """Rename checked join labels without reparsing its immutable spin entries."""
    spin = tuple(numbers[label] for label in join.spin_labels)
    mode = tuple(numbers[label] for label in join.mode_labels)
    batch = None if join.batch_label is None else numbers[join.batch_label]
    if type(join) is not BaryonJoinSpec:
        return replace(join, spin_labels=spin, mode_labels=mode, batch_label=batch)
    if spin == join.spin_labels and mode == join.mode_labels and batch == join.batch_label:
        return join
    normalized = copy(join)
    object.__setattr__(normalized, "spin_labels", spin)
    object.__setattr__(normalized, "mode_labels", mode)
    object.__setattr__(normalized, "batch_label", batch)
    return normalized


class IntegerRegistry:
    """Session-shared definitions. Registration never grants cache availability."""
    def __init__(self, problem, *, memory_limit=None):
        self.memory_limit = memory_limit
        self.rejected_allocations = 0
        self.data, self.names, self.patterns, self.pattern_ids = [], {}, [], {}
        self.core_ids, self.calculations, self.calculation_ids = {}, [], {}
        self.has_alternative_calculations = False
        self.layout_bases = {}
        self.dense_pair_masks = {}
        self.dense_choices = {}
        self.dense_hot = {}
        self.dense_relation_ids = {}
        self.term_transitions = {}
        self.operations, self.operation_costs, self.connection_tables = OperationPool(), [], []
        # Definitions belong to the session, not to an expansion or DFS node.
        self.link_recipes, self.link_predictions, self.join_recipes = {}, {}, {}
        self.special_recipes = {}
        self.factorized_recipes, self.layout_recipes, self.normalizer_costs = {}, {}, {}
        self.data_names, self.data_shape_keys = [], []
        for key in sorted(set(problem.specs) | set(problem.sources)):
            self.register_data(key, problem.specs.get(key))

    def memory_constants_since(self, positions):
        """New immutable definitions from these session-owned append-only tables.

        The lists, name maps and calculation-ID lists remain mutable roots and
        must still be visited at every memory sample. Their records never change.
        """
        groups = self.data, self.patterns, self.calculations
        ends = tuple(len(group) for group in groups)
        roots = (group[index] for group, begin, end in zip(groups, positions, ends)
                 for index in range(begin, end))
        return roots, ends

    def register_data(self, name, spec=None, core=None):
        if name in self.names:
            identity = self.names[name]
            if self.data[identity].spec != spec or self.data[identity].core != core:
                raise ValueError("one data identity cannot change its definition")
            return identity
        identity = len(self.data)
        self.names[name] = identity
        self.data.append(DataDefinition(name, spec, core))
        self.data_names.append(name)
        self.data_shape_keys.append(None if spec is None else (spec.shape, spec.dtype))
        return identity

    def core(self, definition, view=()):
        key = definition.identity, view
        if key not in self.core_ids:
            name = "core:" + hashlib.blake2b(repr(key).encode(), digest_size=32).hexdigest()
            self.core_ids[key] = self.register_data(name, definition.spec, definition)
        return self.core_ids[key]

    def pattern(self, axes, output):
        key = (tuple(tuple(a) for a in axes), tuple(output))
        identity = self.pattern_ids.get(key)
        if identity is None:
            identity = self.pattern_ids[key] = len(self.patterns)
            self.patterns.append(AxisPattern(*key))
        return identity

    def calculation(self, action, spec, workspace):
        result = self.register_data(action.tensor, spec)
        # A result can have alternative recipes/cost predictions. Those are
        # separate calculation IDs but share the same logical data identity.
        if action.reused:
            for identity in self.calculation_ids.get(result, ()):
                return identity
        identity = len(self.calculations)
        for candidate in self.calculation_ids.get(result, ()):
            value = self.calculations[candidate]
            if value.seconds == action.seconds and value.workspace == workspace:
                return candidate
        if self.calculation_ids.get(result):
            self.has_alternative_calculations = True
        self.calculations.append(Calculation(result, action.seconds, workspace, action))
        self.calculation_ids.setdefault(result, []).append(identity)
        return identity

    def admissible_allocation(self, spec, workspace):
        # A single output and its declared workspace must fit regardless of
        # future residency/order. This cheap necessary bound is not a storage
        # schedule, and never assumes that inputs or raw blocks are free.
        allowed = (self.memory_limit is None or
                   math.prod(spec.shape)*spec.dtype.itemsize + workspace <= self.memory_limit)
        self.rejected_allocations += not allowed
        return allowed

    def dense_pair_mask(self, pattern_id, checkpoint):
        """Pairs that survive the existing no-outer-product rule, by layout.

        Only axis equality and retained axes enter this test. Data identities,
        sizes, costs and availability still use the ordinary description path.
        Sparse/core operations must not use this dense-only filter.
        """
        if pattern_id not in self.dense_pair_masks:
            pattern = self.patterns[pattern_id]
            axes = tuple(set(labels) for labels in pattern.inputs)
            mask = 0
            for ordinal, (left, right) in enumerate(combinations(range(len(axes)), 2)):
                checkpoint()
                needed = set(pattern.output)
                for position, labels in enumerate(axes):
                    if position not in (left, right):
                        needed.update(labels)
                if not (axes[left] - axes[right] and axes[right] - axes[left]
                        and (axes[left] | axes[right]) <= needed):
                    mask |= 1 << ordinal
            self.dense_pair_masks[pattern_id] = mask
        return self.dense_pair_masks[pattern_id]

    def dense_pattern_choices(self, pattern_id, checkpoint):
        """Compile legal pairs and all operand-order axis keys once per pattern."""
        cached = self.dense_choices.get(pattern_id)
        if cached is not None:
            return cached
        pattern = self.patterns[pattern_id]
        mask = self.dense_pair_mask(pattern_id, checkpoint) if len(pattern.inputs) > 1 else None
        result = compile_dense_choices(pattern, checkpoint, mask=mask)
        compiled = []
        for positions, variants in result:
            bound = []
            for output, keys, equation in variants:
                relations = []
                for key in keys:
                    identity = self.dense_relation_ids.get(key)
                    if identity is None:
                        identity = self.dense_relation_ids[key] = len(self.dense_relation_ids)
                    relations.append(identity)
                bound.append((output, keys, equation, tuple(relations)))
            compiled.append((positions, tuple(bound)))
        result = tuple(compiled)
        self.dense_choices[pattern_id] = result
        # Flattened hot-path table: operand positions plus both name-order
        # variants of (output, canonical relation). Row rebuilds then need no
        # variant selection, spec lookup or axis reconstruction to score a
        # candidate whose recipe is already registered.
        hot = []
        for ordinal, (positions, variants) in enumerate(result):
            entries = []
            for output, keys, equation, relations in variants:
                entries.append((output, *relations, equation, keys))
            if len(positions) == 1:
                hot.append((positions, positions[0], -1, ordinal, *entries[0]))
            else:
                hot.append((positions, positions[0], positions[1], ordinal,
                            *entries[0], *entries[1]))
        self.dense_hot[pattern_id] = tuple(hot)
        return result

    def term_transition(self, pattern_id, positions, cores, output):
        """Compile factor removal and resulting axes once, independently of data."""
        key = (pattern_id, positions, cores, output)
        result = self.term_transitions.get(key)
        if result is None:
            pattern = self.patterns[pattern_id]
            consumed = set(positions + cores)
            keep = tuple(i for i in range(len(pattern.inputs)) if i not in consumed)
            axes = tuple(pattern.inputs[i] for i in keep) + (output,)
            result = (keep, self.pattern(axes, pattern.output),
                      not keep and output == pattern.output)
            self.term_transitions[key] = result
        return result


class IntegerSpace:
    """One mutable integer array, sparse undo, and only changed-row reindexing."""
    def __init__(self, problem, *, profile, registry=None, checkpoint=lambda: None,
                 debug=False, memory_limit=None, track_fingerprint=True):
        if type(debug) is not bool:
            raise ValueError("debug must be bool")
        if type(track_fingerprint) is not bool:
            raise ValueError("track_fingerprint must be bool")
        self.debug = debug
        self._track_fingerprint = track_fingerprint
        self.shared_digest = None
        self.shared_term_hashes = []
        self._shared_data_parts = None
        self._terminal_completions = None
        self.problem, self.profile = problem, profile
        self.has_sum_barriers = any(term.sum_barriers for term in problem.terms)
        self.registry = registry if registry is not None else IntegerRegistry(problem, memory_limit=memory_limit)
        self.contexts, self.symbols, self.initial_terms = [], [], []
        self._barrier_cache = {}
        self._dense_contexts = {}
        self.join_cores, self.bridge_cores, self.join_axes, self.bridge_axes = [], [], [], []
        self.applied = self.enumerations = self.undo_count = self.audited_steps = 0
        self.row_rebuilds = self.logical_effects = 0
        self.layout_candidates = self.terminal_deferred = self.terminal_resolutions = 0
        self.incremental_rankings = self.rank_heap_compactions = 0
        self.row_descriptions = self.row_description_reuses = 0
        self._dense_recipes = self.registry.link_recipes
        self._dense_predictions = self.registry.link_predictions
        self.dense_recipe_hits = self.dense_recipe_builds = 0
        # Dense binding consumes integer rows, not single-row Problems, so a
        # full Term context is only materialized if a layout actually asks for
        # one. Equal initial expressions need no separate binding at all. The
        # static problem already carries every integer axis relation of the
        # immutable initial expression, so the per-operand Factor normalization
        # of ContractionState.initial is skipped and repeated expressions are
        # compiled once per class.
        dense_only = not debug and not problem.baryon_joins and not problem.sparse_bridges
        compiled = {} if dense_only else None
        joins_by_scene = None
        if problem.baryon_joins:
            joins_by_scene = {}
            for join in problem.baryon_joins:
                joins_by_scene.setdefault(join.scene, []).append(join)
        self._joins_by_scene = (None if joins_by_scene is None else
                                {scene: tuple(joins) for scene, joins in joins_by_scene.items()})
        for scene, logical in enumerate(problem.terms):
            checkpoint()
            labels, output = logical.subscripts.inputs, logical.subscripts.output
            if dense_only:
                key = (logical.inputs, logical.subscripts)
                entry = compiled.get(key)
                if entry is None:
                    symbols = tuple(dict.fromkeys(
                        tuple(c for axes in labels for c in axes) + output))
                    numbers = {c: i for i, c in enumerate(symbols)}
                    axes = tuple(tuple(numbers[c] for c in operand) for operand in labels)
                    entry = (symbols, TermRecord(
                        tuple(self.registry.names[name] for name, _ in
                              zip(logical.inputs, labels, strict=True)),
                        self.registry.pattern(axes, tuple(numbers[c] for c in output))))
                    compiled[key] = entry
                self.symbols.append(entry[0])
                self.contexts.append(None)
                self.initial_terms.append(entry[1])
                for values in (self.join_cores, self.bridge_cores, self.join_axes, self.bridge_axes):
                    values.append(())
                continue
            term = PendingTerm(
                tuple(Factor(name, axes) for name, axes in zip(logical.inputs, labels, strict=True)),
                output)
            symbols = tuple(dict.fromkeys(tuple(c for f in term.factors for c in f.labels) + term.output))
            self.symbols.append(symbols)
            numbers = {c: i for i, c in enumerate(symbols)}
            normalized = numbers
            subscripts = AxisPattern(tuple(tuple(normalized[c] for c in f.labels)
                                           for f in term.factors),
                                     tuple(normalized[c] for c in term.output))
            # The original problem stays immutable; labels in its static Term
            # are irrelevant to the temporary current-expression views below.
            # Scoped joins keep their raw labels for the core bindings below;
            # only the context view carries normalized labels.
            local = component_problem(problem, (scene,),
                _scoped_joins=None if joins_by_scene is None else joins_by_scene.get(scene, ()))
            joins = tuple(_normalized_join(j, normalized) for j in local.baryon_joins)
            self.contexts.append(replace(local, baryon_joins=joins,
                terms=(replace(local.terms[0], subscripts=subscripts,
                    sum_barriers=tuple((tuple(normalized[a] for a in trigger),
                                        tuple(normalized[a] for a in dependent))
                                       for trigger, dependent in logical.sum_barriers)),)))
            factors = [self.registry.names[f.tensor] for f in term.factors]
            axes = [tuple(numbers[c] for c in f.labels) for f in term.factors]
            core_joins, core_bridges = [], []
            for join in local.baryon_joins:
                spin, mode = problem.specs[join.spin], problem.specs[join.elemental]
                core = join.core or TensorCoreSpec("baryon", ("spin0", "spin1", "spin2",
                    "laplace_vector0", "laplace_vector1", "laplace_vector2"),
                    TensorSpec((*spin.shape, *mode.shape), np.result_type(spin.dtype, mode.dtype)),
                    (CoreComponent((join.spin, join.elemental), sparse_spin=join.sparse_spin),))
                identity = self.registry.core(core)
                core_joins.append(identity)
                factors.append(identity)
                axes.append(tuple(numbers[c] for c in join.core_labels))
            for bridge in local.sparse_bridges:
                left = next(f for f in term.factors if f.tensor == bridge.left)
                right = next(f for f in term.factors if f.tensor == bridge.right)
                la = tuple(axis % len(left.labels) for axis in bridge.left_axes)
                ra = tuple(axis % len(right.labels) for axis in bridge.right_axes)
                shape = tuple(problem.specs[left.tensor].shape[a] for a in la) + tuple(problem.specs[right.tensor].shape[a] for a in ra)
                core = TensorCoreSpec("p2p", ("left_spin", "left_point", "left_color",
                    "right_spin", "right_point", "right_color"), TensorSpec(shape, "complex128"),
                    (CoreComponent((bridge.record,)),))
                identity = self.registry.core(core, (bridge.transpose_bridge, bridge.reverse_operands))
                core_bridges.append(identity)
                factors.append(identity)
                axes.append(tuple(numbers[left.labels[a]] for a in la) + tuple(numbers[right.labels[a]] for a in ra))
            self.join_cores.append(tuple(core_joins))
            self.bridge_cores.append(tuple(core_bridges))
            self.join_axes.append(tuple(axes[len(term.factors):len(term.factors) + len(core_joins)]))
            self.bridge_axes.append(tuple(axes[len(term.factors) + len(core_joins):]))
            self.initial_terms.append(TermRecord(tuple(factors), self.registry.pattern(axes, tuple(numbers[c] for c in term.output))))
        self.initial_terms = tuple(self.initial_terms)
        self.contexts, self.symbols = tuple(self.contexts), tuple(self.symbols)
        self.join_cores, self.bridge_cores = tuple(self.join_cores), tuple(self.bridge_cores)
        self.join_axes, self.bridge_axes = tuple(self.join_axes), tuple(self.bridge_axes)
        from .compact_rows import compact_initial_rows
        self.scene_members = compact_initial_rows(self)
        self._row_weights = (None if self.scene_members is None else
                             tuple(len(group) for group in self.scene_members))
        self._scene_first = (None if self.scene_members is None else
                             tuple(group[0] for group in self.scene_members))
        self._scene_last = (None if self.scene_members is None else
                            tuple(group[-1] for group in self.scene_members))
        self.base_specs = dict(problem.specs)
        from .scalar_terminal import ScalarTerminalIndex
        self._scalar_terminals = (ScalarTerminalIndex() if not self.debug
            and not problem.baryon_joins and not problem.sparse_bridges
            and all(not self.registry.patterns[t.pattern].output for t in self.initial_terms)
            else None)
        self.reset()

    def memory_constants(self):
        """Owned expression definitions unchanged by apply, undo or reset."""
        return (self.problem, self.contexts, self.symbols, self.initial_terms,
                self.join_cores, self.bridge_cores, self.join_axes, self.bridge_axes,
                self.scene_members, self._row_weights, self._scene_first, self._scene_last)

    @staticmethod
    def _part(value):
        return int.from_bytes(hashlib.blake2b(repr(value).encode(), digest_size=32,
                                            person=b"ED.integer.DFS").digest(), "little")

    def reset(self):
        if self._scalar_terminals is not None:
            self._scalar_terminals.reset()
        from .shared_links import ScopeLabels, SharedLinks
        self.terms = ScopeLabels(self.initial_terms)
        self.links = SharedLinks(self)
        self.available = set()
        self.recent_scenes = frozenset()
        self.catalog = dict(self.base_specs)
        self._catalog_bytes = sum(math.prod(spec.shape) * spec.dtype.itemsize
                                  for spec in self.catalog.values())
        self._residency_offsets = {}
        self._prediction_catalog = self._prediction_loaded = None
        self.rows, self.groups = {}, {}
        self._group_weights = {}
        self.group_ordinals, self.next_group_ordinal = {}, 0
        self._ranking = None
        self.remaining = len(self.initial_terms)
        self.layout_variants, self.terminal_families = set(), {}
        self.terminal_members, self.terminal_heap = {}, []
        self.dirty = set(range(len(self.terms)))
        self.digest = 0 if self._track_fingerprint else None
        self.term_hashes = [] if self._track_fingerprint else None
        if self._track_fingerprint:
            for scene, term in enumerate(self.terms):
                value = self._part(("term", scene, term))
                self.term_hashes.append(value)
                self.digest ^= value
        if self.shared_digest is not None:
            self.enable_shared_fingerprint()

    def _shared_data(self, identity):
        value = self._shared_data_parts.get(identity)
        if value is None:
            value = self._part(("data", self.registry.data[identity].name))
            self._shared_data_parts[identity] = value
        return value

    def _shared_term(self, scene, term):
        pattern = self.registry.patterns[term.pattern]
        relation = axis_key(tuple(self._shared_data(i) for i in term.factors),
                            pattern.inputs, pattern.output)
        return self._part((scene, relation, term.done))

    def enable_shared_fingerprint(self):
        """Incremental identity independent of process-local registry numbers."""
        if not self._track_fingerprint:
            raise RuntimeError("fingerprint tracking is disabled for this workspace")
        if self._shared_data_parts is None:
            self._shared_data_parts = {}
        self.shared_digest = 0
        self.shared_term_hashes = []
        for scene, term in enumerate(self.terms):
            value = self._shared_term(scene, term)
            self.shared_term_hashes.append(value)
            self.shared_digest ^= value
        for identity in self.available:
            self.shared_digest ^= self._shared_data(identity)

    def shared_fingerprint(self):
        return self.shared_digest.to_bytes(32, "little")

    def release(self):
        self._dense_contexts.clear()
        if self.links is not None:
            self.link_statistics = self.links.statistics()
        self._dense_recipes, self._dense_predictions = {}, {}
        self.links = None
        if self._scalar_terminals is not None:
            self._scalar_terminals.clear()
        self.terms, self.catalog, self.rows, self.groups = [], {}, {}, {}
        self.term_hashes = []
        self.shared_term_hashes = []
        self.group_ordinals, self._ranking = {}, None
        self._group_weights = {}
        self.remaining = 0
        self._prediction_catalog = self._prediction_loaded = None
        self.available, self.dirty = set(), set()
        self.recent_scenes = frozenset()
        self.layout_variants, self.terminal_families = set(), {}
        self.terminal_members, self.terminal_heap = {}, []
        self.shared_digest = self._shared_data_parts = None
        if self._terminal_completions is not None:
            self._terminal_completions.clear()

    def fingerprint(self):
        if not self._track_fingerprint:
            raise RuntimeError("fingerprint tracking is disabled for this workspace")
        return self.digest.to_bytes(32, "little")

    def is_done(self):
        return self.remaining == 0

    def _touch_group(self, identity):
        if self._ranking is not None:
            self._ranking.dirty.add(identity)

    def _set_terminal(self, identity, family):
        if self._scalar_terminals is not None:
            self._scalar_terminals.set(identity, family, self)
        self.terminal_families[identity] = family
        if family not in self.terminal_members:
            self.terminal_members[family] = set()
            heapq.heappush(self.terminal_heap, family)
        self.terminal_members[family].add(identity)
        self._touch_group(identity)

    def _remove_terminal(self, identity):
        family = self.terminal_families.pop(identity, None)
        if family is not None:
            if self._scalar_terminals is not None:
                self._scalar_terminals.remove(identity)
            members = self.terminal_members[family]
            members.remove(identity)
            if not members:
                del self.terminal_members[family]
            self._touch_group(identity)

    def _terminal_family(self, checkpoint):
        while self.terminal_heap and self.terminal_heap[0] not in self.terminal_members:
            checkpoint()
            heapq.heappop(self.terminal_heap)
        return self.terminal_heap[0]

    def _add_group(self, identity, occurrences):
        self.groups[identity] = occurrences
        if self._row_weights is not None:
            self._group_weights[identity] = sum(self._row_weights[s] for s in occurrences)
        self.group_ordinals[identity] = self.next_group_ordinal
        self.next_group_ordinal += 1
        self._touch_group(identity)

    def _remove_group(self, identity):
        occurrences = self.groups.pop(identity)
        self._group_weights.pop(identity, None)
        del self.group_ordinals[identity]
        from .terminal_steps import forget_terminal_witness
        forget_terminal_witness(self, identity)
        self._touch_group(identity)
        return occurrences

    def tie_break(self, operation, checkpoint):
        scenes = self.recent_scenes.intersection(self.groups[operation.calculation])
        return len(scenes) if self._row_weights is None else sum(self._row_weights[s] for s in scenes)

    def occurrence_count(self, calculation):
        return (len(self.groups[calculation]) if self._row_weights is None else
                self._group_weights[calculation])

    def mask_count(self, mask):
        if self._row_weights is None:
            return mask.bit_count()
        count = 0
        while mask:
            bit = mask & -mask
            count += self._row_weights[bit.bit_length() - 1]
            mask ^= bit
        return count

    def rank_candidates(self, width, scorer, checkpoint, *, profile_scoring=False,
                        include_ties=True):
        from .integer_ranking import CandidateRanking, supports_incremental_ranking
        if not supports_incremental_ranking(scorer):
            return None
        self._index(checkpoint)
        name = "incremental_time" if scorer is None else scorer.name
        if self._ranking is None or self._ranking.name != name:
            self._ranking = CandidateRanking(self, name)
        return self._ranking.select(width, scorer, checkpoint, profile_scoring,
                                    include_ties=include_ties)

    def _decode(self, scene, *, raw=False):
        term = self.terms[scene]
        if term.done:
            return None
        pattern = self.registry.patterns[term.pattern]
        symbols = self.symbols[scene] if raw else tuple(range(len(self.symbols[scene])))
        factors = tuple(Factor(self.registry.data[data].name, tuple(symbols[a] for a in axes))
                        for data, axes in zip(term.factors, pattern.inputs, strict=True)
                        if self.registry.data[data].core is None)
        return PendingTerm(factors, tuple(symbols[a] for a in pattern.output))

    def _state(self, scene):
        term = self._decode(scene)
        local = self.contexts[scene]
        if local is None:
            local = self._dense_contexts.get(scene)
            if local is None:
                physical = scene if self.scene_members is None else self.scene_members[scene][0]
                local = component_problem(self.problem, (physical,))
                local = replace(local, terms=(replace(local.terms[0],
                    subscripts=self.registry.patterns[self.initial_terms[scene].pattern],
                    sum_barriers=self._sum_barriers(scene)),))
                self._dense_contexts[scene] = local
        completed = frozenset((0, j.name) for i, j in enumerate(local.baryon_joins)
                              if self._core_position(scene, i, join=True) is None)
        # All changed Terms at this node share the same immutable metadata.
        # Keep one snapshot, rather than copying the complete catalog per row.
        catalog = self.catalog if self._prediction_catalog is None else self._prediction_catalog
        loaded = frozenset(self.catalog) if self._prediction_loaded is None else self._prediction_loaded
        state = ContractionState(local, catalog, (term,), loaded, (None,), completed_joins=completed)
        self._prediction_catalog, self._prediction_loaded = state.catalog, loaded
        required = _live_inputs(state)
        if not required <= loaded:
            return replace(state, loaded=loaded | required)
        # Prediction loads the entire catalog. Its total grows only when a
        # result is published; each local source declaration changes that
        # common total by a fixed offset, not another whole-catalog scan.
        offset = self._residency_offsets.get(id(local))
        if offset is None:
            # Source-only sparse records need not be loaded in this prediction.
            # Match ContractionState residency: count only loaded sources.
            sources = {key: source for key, source in local.sources.items() if key in loaded}
            regions = {source.region: source.region_bytes for source in sources.values()}
            host_regions = {source.region: source.host_region_bytes for source in sources.values()}
            delta = sum(regions.values())
            for key, source in sources.items():
                size = math.prod(catalog[key].shape) * catalog[key].dtype.itemsize
                delta += (size if source.snapshot_bytes is None else source.snapshot_bytes) - size
            offset = delta, sum(host_regions.values())
            self._residency_offsets[id(local)] = offset
        total = self._catalog_bytes + offset[0]
        object.__setattr__(state, '_resident_totals',
                           (total, total if local.backend == 'numpy' else offset[1]))
        return state

    def _core_position(self, scene, index, *, join):
        core = (self.join_cores if join else self.bridge_cores)[scene][index]
        axes = (self.join_axes if join else self.bridge_axes)[scene][index]
        term = self.terms[scene]
        pattern = self.registry.patterns[term.pattern]
        return next((i for i, (data, labels) in enumerate(zip(term.factors, pattern.inputs, strict=True))
                     if data == core and labels == axes), None)

    def _publish_row(self, scene, row):
        for calculation, occurrence in row:
            if calculation not in self.groups:
                self._add_group(calculation, {})
            self.groups[calculation].setdefault(scene, occurrence)
            self._touch_group(calculation)
        self.rows[scene] = tuple(row)

    def _sum_barriers(self, scene):
        if not self.has_sum_barriers:
            return ()
        cached = self._barrier_cache.get(scene)
        if cached is not None:
            return cached
        numbers = {label: i for i, label in enumerate(self.symbols[scene])}
        result = tuple((tuple(numbers[a] for a in trigger), tuple(numbers[a] for a in dependent))
                       for trigger, dependent in self.problem.terms[scene].sum_barriers)
        self._barrier_cache[scene] = result
        return result

    def _dense_row(self, scene, checkpoint, *, plain_only=False, touching=None):
        term = self.terms[scene]
        registry = self.registry
        pattern = term.pattern
        factors = term.factors
        recipes = self._dense_recipes
        barriers = self._sum_barriers(scene)
        if barriers and registry.data[factors[0]].spec.dtype not in (np.dtype('float32'), np.dtype('complex64')):
            barriers = ()
        axes = registry.patterns[pattern].inputs if barriers else ()
        slots = registry.dense_hot.get(pattern)
        if slots is None:
            registry.dense_pattern_choices(pattern, checkpoint)
            slots = registry.dense_hot[pattern]
        names = registry.data_names
        mask = DENSE_ROW_CHECKPOINT_INTERVAL - 1
        row = []
        append = row.append
        for slot in slots:
            if touching is not None and touching not in slot[0]:
                continue
            if plain_only and (registry.data[factors[slot[1]]].core is not None or
                               (slot[2] >= 0 and registry.data[factors[slot[2]]].core is not None)):
                continue
            # A row is checked by its caller. Long rows bound themselves at a
            # coarse interval; the caller's per-row check keeps the deadline
            # latency below one row. The interval must stay a power of two.
            if slot[3] & mask == 0:
                checkpoint()
            first = factors[slot[1]]
            other = slot[2]
            # The canonical key stays a small tuple: allocating (small, small,
            # small) measured faster than integer encodings, which must
            # allocate the encoded value on every lookup.
            if other < 0:
                second = -1
                output, key, order = slot[4], (first, -1, slot[5]), 0
            else:
                second = factors[other]
                if names[first] > names[second]:
                    output, forward, reverse, same = slot[10], slot[11], slot[12], slot[13]
                    order = 1
                else:
                    output, forward, reverse, same = slot[4], slot[5], slot[6], slot[7]
                    order = 0
                if first < second:
                    key = (first, second, forward)
                elif first > second:
                    key = (second, first, reverse)
                else:
                    key = (first, second, same)
            wide = False
            if barriers:
                before = {a for operand in axes for a in operand}
                after = set(output) | {a for i, operand in enumerate(axes)
                                       if i not in slot[0] for a in operand}
                closed = tuple(dependent for trigger, dependent in barriers
                               if before.intersection(trigger) and not after.intersection(trigger))
                if any(after.intersection(dependent) for dependent in closed):
                    continue
                wide = bool(closed)
                if wide:
                    key = (*key, True)
            calculation = recipes.get(key)
            if calculation is None:
                calculation = self._dense_recipe(slot, order, first, second, key,
                                                 checkpoint, wide_accumulation=wide)
                if calculation is None:
                    continue
            else:
                self.dense_recipe_hits += 1
            append((calculation, Occurrence(scene, slot[0], output, None, ())))
        if not plain_only:
            row.extend(self._factorized_row(scene, checkpoint))
        return row

    def _factorized_row(self, scene, checkpoint):
        """Admit a declared four-factor wide boundary before binary expansion.

        Every geometry/cost is independently described by the existing core.
        This is a terminal integer candidate, not a channel-specific route.
        """
        term = self.terms[scene]
        if len(term.factors) not in (3,4) or len(self._sum_barriers(scene)) < 2:
            return []
        if self.registry.data[term.factors[0]].spec.dtype != np.dtype('complex64'):
            return []
        from ..factorized_reduction import describe_factorized_action
        state = self._state(scene)
        row = []
        positions = tuple(range(len(term.factors)))
        for large in combinations(positions, 2):
            checkpoint()
            key = (term.factors, term.pattern, large)
            if key in self.registry.factorized_recipes:
                row.append((self.registry.factorized_recipes[key],
                            Occurrence(scene, positions, state.pending[0].output, None, ())))
                continue
            try:
                action, spec, workspace, _ = describe_factorized_action(state, 0, positions, large,
                    profile=self.profile, output_order=state.pending[0].output, tile_elements=4096)
            except ValueError:
                continue
            if not self.registry.admissible_allocation(spec, workspace):
                continue
            calculation = self.registry.calculation(action, spec, workspace)
            self.registry.factorized_recipes[key] = calculation
            row.append((calculation, Occurrence(scene, positions, state.pending[0].output, None, ())))
        return row

    def _dense_recipe(self, slot, order, first, second, key, checkpoint, *, wide_accumulation=False):
        """Register one dense recipe for a compiled row slot (cold path).

        The hot loop already selected the operand order variant and built the
        cache key, so this path only prices, describes and registers it.
        """
        registry = self.registry
        definitions = registry.data
        recipes, predictions = self._dense_recipes, self._dense_predictions
        positions = slot[0]
        if second < 0:
            output, equation, keys = slot[4], slot[6], slot[7]
            selected = (definitions[first],)
        else:
            if order:
                output, equation, keys = slot[10], slot[14], slot[15]
            else:
                output, equation, keys = slot[4], slot[8], slot[9]
            selected = (definitions[first], definitions[second])
        specs = tuple(data.spec for data in selected)
        # Shape/dtype facts are already interned per registered data identity,
        # so the shared prediction key is assembled without rebuilding value
        # tuples that only repeat what the registry owns.
        shape_keys = registry.data_shape_keys
        if second < 0:
            cost_key = (equation, (shape_keys[first],))
        else:
            cost_key = (equation, (shape_keys[first], shape_keys[second]))
        if wide_accumulation:
            cost_key = (*cost_key, True)
        try:
            prediction = predictions.get(cost_key)
            if prediction is None:
                prediction = predict_dense(equation, specs, self.profile,
                                           wide_accumulation=wide_accumulation)
                predictions[cost_key] = prediction
            names = tuple(data.name for data in selected)
            ordered_names = names[::-1] if order else names
            name_relation = keys[2 if len(names) == 2 and names[0] == names[1] else order]
            action, spec, workspace = describe_dense(
                names, equation.inputs, output, specs, positions=positions,
                profile=self.profile, equation=equation, prediction=prediction,
                canonical_key=("contract", (ordered_names, *name_relation)),
                wide_accumulation=wide_accumulation)
        except ValueError:
            return None
        if not registry.admissible_allocation(spec, workspace):
            return None
        calculation = registry.calculation(action, spec, workspace)
        recipes[key] = calculation
        self.dense_recipe_builds += 1
        return calculation

    def _uses_links(self, scene):
        return not self.debug

    def _special_links(self, scene):
        context = self.contexts[scene]
        return bool(self._sum_barriers(scene) or context is not None and context.sparse_bridges)

    def _special_link_row(self, scene, checkpoint):
        """Keep bridge/barrier rules in the existing geometry service.

        Only geometry is rebound after a local boundary changes. The complete
        operation identity is classified before its first (and only) price.
        """
        from ..canonical_operations import logical_result_name
        term = self.terms[scene]
        state = self._state(scene)
        plain = tuple(p for p, node in enumerate(term.factors)
                      if self.registry.data[node].core is None)
        choices = [(p, None) for p in (((0,),) if len(plain) == 1
                                      else combinations(range(len(plain)), 2))]
        choices.extend(((), j) for j in range(len(self.join_cores[scene]))
                       if self._core_position(scene, j, join=True) is not None)
        row = []
        for positions, index in choices:
            checkpoint()
            join = None if index is None else state.problem.baryon_joins[index]
            try:
                key = logical_result_name(state, 0, positions, join=join)
                cached = self.registry.special_recipes.get(key)
                if cached is None:
                    action, spec, workspace, output, selected = describe(
                        state, 0, positions, join=join, profile=self.profile)
                    if not self.registry.admissible_allocation(spec, workspace):
                        continue
                    identity = self.registry.calculation(action, spec, workspace)
                    # Outputs use local labels: save an axis mapping, not the
                    # origin scope's labels or factor positions.
                    selected_axes = tuple(state.pending[0].factors[p].labels for p in selected)
                    symbols = tuple(dict.fromkeys(a for axes in selected_axes for a in axes))
                    cached = (identity, tuple(symbols.index(a) for a in output))
                    self.registry.special_recipes[key] = cached
                else:
                    selected = positions if join is None else tuple(
                        p for p, factor in enumerate(state.pending[0].factors)
                        if set(join.spin_labels + join.mode_labels).intersection(factor.labels))
                    symbols = tuple(dict.fromkeys(a for p in selected
                                                  for a in state.pending[0].factors[p].labels))
                identity, output_indices = cached
                output = tuple(symbols[p] for p in output_indices)
            except ValueError:
                continue
            action = self.registry.calculations[identity].recipe
            consumed = ()
            if index is not None:
                consumed = (self._core_position(scene, index, join=True),)
            elif action.kind == 'sparse_bridge':
                consumed = tuple(p for i, bridge in enumerate(state.problem.sparse_bridges)
                    if bridge.record in action.operands[2:]
                    and (p := self._core_position(scene, i, join=False)) is not None)
            row.append((identity, Occurrence(scene, tuple(plain[p] for p in selected),
                                            output, index, consumed)))
        row.extend(self._factorized_row(scene, checkpoint))
        return row

    def _link_row(self, scene, checkpoint, *, touching=None):
        """Enumerate endpoint/axis classes before invoking any cost service."""
        term = self.terms[scene]
        if self._special_links(scene):
            return self._special_link_row(scene, checkpoint)
        context = self.contexts[scene]
        if context is None or not context.baryon_joins:
            return self._dense_row(scene, checkpoint, touching=touching)
        from ..canonical_operations import output_axes
        registry = self.registry
        axes = registry.patterns[term.pattern].inputs
        plain = tuple(p for p, node in enumerate(term.factors)
                      if registry.data[node].core is None)
        row = self._dense_row(scene, checkpoint, plain_only=True, touching=touching)
        for index, join in enumerate(context.baryon_joins):
            core_position = self._core_position(scene, index, join=True)
            if core_position is None:
                continue
            local = set(join.spin_labels + join.mode_labels)
            selected = tuple(p for p in plain if local.intersection(axes[p]))
            if not selected or touching is not None and touching not in selected:
                continue
            if not local <= {axis for p in selected for axis in axes[p]}:
                continue
            needed = set(registry.patterns[term.pattern].output)
            for p, labels in enumerate(axes):
                if p not in selected and p != core_position:
                    needed.update(labels)
            retained = {axis for p in selected for axis in axes[p]
                        if axis not in local and axis in needed}
            names = (join.spin, join.elemental,
                     *(registry.data[term.factors[p]].name for p in selected))
            labels = (join.spin_labels,
                      join.elemental_labels(self.problem.specs[join.elemental]),
                      *(axes[p] for p in selected))
            output = output_axes(names, labels, retained, ordered=True)
            # Local axis names and term identities never enter the key.
            nodes = (term.factors[core_position], *(term.factors[p] for p in selected))
            relation = axis_key(nodes, (axes[core_position], *(axes[p] for p in selected)),
                                output, ordered=True)[1:]
            key = (nodes, relation)
            calculation = registry.join_recipes.get(key)
            if calculation is None:
                state = self._state(scene)
                action, spec, workspace, described_output, _ = describe(
                    state, 0, join=join, profile=self.profile)
                if not registry.admissible_allocation(spec, workspace):
                    continue
                calculation = registry.calculation(action, spec, workspace)
                registry.join_recipes[key] = calculation
                if self.debug and tuple(described_output) != output:
                    raise RuntimeError('shared join output differs from independent description')
            row.append((calculation, Occurrence(scene, selected, output, index, (core_position,))))
        return row

    def _rebuild(self, scene, checkpoint, row_templates=None):
        for calculation in {c for c, _ in self.rows.pop(scene, ())}:
            occurrences = self.groups[calculation]
            occurrences.pop(scene, None)
            self._touch_group(calculation)
            if not occurrences:
                self._remove_group(calculation)
        if self.terms[scene].done:
            if self._uses_links(scene):
                self.links.publish(scene, ())
            return
        checkpoint()
        self.row_rebuilds += 1
        term = self.terms[scene]
        context = self.contexts[scene]
        dense = context is None or (not context.baryon_joins and not context.sparse_bridges)
        # A core-bearing row can share its calculation descriptions with a
        # second row only when the full scoped core/bridge declarations match.
        # TermRecord alone does not encode a join's name or bridge scope.
        if dense:
            row_key = (term, self._sum_barriers(scene)) if self.has_sum_barriers else term
        else:
            row_key = (term, self._sum_barriers(scene), self.join_cores[scene],
                       self.join_axes[scene], self.bridge_cores[scene], self.bridge_axes[scene])
        if row_templates is not None and row_key in row_templates:
            row = []
            self.row_description_reuses += 1
            for calculation, occurrence in row_templates[row_key]:
                checkpoint()
                row.append((calculation, Occurrence(scene, occurrence.positions,
                                                    occurrence.output, occurrence.join,
                                                    occurrence.cores)))
            self._publish_row(scene, row)
            if self._uses_links(scene):
                self.links.publish(scene, row)
            return
        self.row_descriptions += 1
        if self._uses_links(scene):
            row = self.links.row(scene, checkpoint)
            self._publish_row(scene, row)
            self.links.publish(scene, row)
            if row_templates is not None:
                row_templates[row_key] = row
            return
        # Descriptions depend on fixed input definitions and axis relations,
        # not on the expression containing them. Residency is resolved later
        # from available; retaining a recipe never makes its result available.
        if dense and not self.debug:
            self._publish_row(scene, self._dense_row(scene, checkpoint))
            if row_templates is not None:
                row_templates[row_key] = self.rows[scene]
            return
        # Whole baryon joins remain on the canonical description path. Their
        # virtual core factors already encode the axes that a plain pair must
        # retain, so the compiled dense table can describe those pairs without
        # rebuilding a ContractionState for every candidate in every scene.
        compiled_plain = (not self.debug and context is not None and
                          bool(context.baryon_joins) and not context.sparse_bridges and
                          not self._sum_barriers(scene))
        row = self._dense_row(scene, checkpoint, plain_only=True) if compiled_plain else []
        plain = tuple(i for i, data in enumerate(term.factors) if self.registry.data[data].core is None)
        choices = ([] if compiled_plain else [(positions, None) for positions in
                   (((0,),) if len(plain) == 1 else combinations(range(len(plain)), 2))])
        choices += [((), j) for j in range(len(self.join_cores[scene])) if self._core_position(scene, j, join=True) is not None]
        dense_mask = (self.registry.dense_pair_mask(term.pattern, checkpoint)
                      if len(plain) > 1 and dense else None)
        state = self._state(scene) if choices else None
        for ordinal, (positions, join_index) in enumerate(choices):
            checkpoint()
            if dense_mask is not None and not (dense_mask & (1 << ordinal)):
                continue
            join = None if join_index is None else state.problem.baryon_joins[join_index]
            try:
                action, spec, workspace, output, selected = describe(state, 0, tuple(positions), join=join, profile=self.profile)
            except ValueError:
                continue
            if not self.registry.admissible_allocation(spec, workspace):
                continue
            calculation = self.registry.calculation(action, spec, workspace)
            consumed = ()
            if join_index is not None:
                consumed = (self._core_position(scene, join_index, join=True),)
            elif action.kind == "sparse_bridge":
                consumed = tuple(position for i, bridge in enumerate(state.problem.sparse_bridges)
                    if bridge.record in action.operands[2:]
                    and (position := self._core_position(scene, i, join=False)) is not None)
            occurrence = Occurrence(scene, tuple(plain[i] for i in selected),
                tuple(output), join_index, consumed)
            row.append((calculation, occurrence))
        if dense:
            row.extend(self._factorized_row(scene, checkpoint))
        self._publish_row(scene, row)
        if row_templates is not None:
            row_templates[row_key] = self.rows[scene]

    def _index(self, checkpoint):
        if not self.dirty:
            return
        if self._row_weights is not None and self.debug:
            from .compact_rows import reindex
            reindex(self, checkpoint)
            return
        from .terminal_steps import invalidate_terminal_scenes, remove_layouts, register_layouts
        invalidate_terminal_scenes(self, self.dirty)
        affected = set()
        for scene in self.dirty:
            checkpoint()
            affected.update(self.terminal_families.get(identity, identity)
                            for identity, _ in self.rows.get(scene, ()))
        remove_layouts(self, affected)
        # Availability/catalog/profile stay fixed throughout this indexing
        # pass. Equal dense TermRecords therefore have equal descriptions;
        # only the scene of each occurrence differs. Discard the memo after
        # this pass so a later apply/undo can never reuse stale availability.
        row_templates = {}
        for scene in sorted(self.dirty):
            self._rebuild(scene, checkpoint, row_templates)
            affected.update(identity for identity, _ in self.rows.get(scene, ()))
        self.dirty.clear()
        if self._row_weights is not None:
            for identity in affected:
                if identity in self.groups:
                    self._group_weights[identity] = sum(self._row_weights[s] for s in self.groups[identity])
        # A new row can join an existing family that the old row did not use.
        # Invalidate that family's consumer layouts before reclassifying it.
        remove_layouts(self, affected)
        register_layouts(self, affected, checkpoint)
        if len(self.terminal_heap) > max(64, 2 * len(self.terminal_members)):
            heap = []
            for family in self.terminal_members:
                checkpoint()
                heap.append(family)
            heapq.heapify(heap)
            self.terminal_heap = heap

    def search_candidates(self, checkpoint):
        from .terminal_steps import search_candidates
        return search_candidates(self, checkpoint)

    def terminal_completion(self, checkpoint):
        self._index(checkpoint)
        if not self.terminal_members or len(self.groups) != len(self.terminal_families):
            return None
        if self._terminal_completions is None:
            from .terminal_completion import TerminalCompletions
            self._terminal_completions = TerminalCompletions()
        return self._terminal_completions.complete(self, checkpoint)

    def terminal_statistics(self):
        return {} if self._terminal_completions is None else self._terminal_completions.statistics()

    def candidates(self, checkpoint):
        self.enumerations += 1
        self._index(checkpoint)
        for identity, occurrences in self.groups.items():
            checkpoint()
            calculation = self.registry.calculations[identity]
            reused = calculation.result in self.available
            seconds = 0.0 if reused else calculation.seconds
            yield IntegerOperation(identity, calculation.result, reused, seconds), seconds

    def apply(self, operation, checkpoint):
        self._index(checkpoint)
        # Re-deriving the operation's identity from the registry is a debug
        # diagnostic: every producer already owns the exact calculation it
        # ranks and no queued operation outlives an undo. Non-debug apply only
        # needs the occurrence group itself.
        if self.debug:
            if operation.calculation not in self.groups or operation.reused != (operation.result in self.available):
                raise ValueError("integer operation is unavailable in this state")
            calculation = self.registry.calculations[operation.calculation]
            if operation.result != calculation.result or operation.seconds != (0 if operation.reused else calculation.seconds):
                raise ValueError("integer operation changed its registered cost")
        changes = []
        for scene, occurrence in sorted(self.groups[operation.calculation].items()):
            checkpoint()
            before = self.terms[scene]
            keep, pattern, done = self.registry.term_transition(
                before.pattern, occurrence.positions, occurrence.cores, occurrence.output)
            factors = tuple(before.factors[i] for i in keep) + (operation.result,)
            after = TermRecord(factors, pattern, done)
            changes.append((scene, before, after))
        if not changes:
            raise ValueError("integer operation has no matching occurrences")
        undo = IntegerUndo(tuple((scene, before) for scene, before, _ in changes),
                           () if operation.reused else (operation.result,), self.digest,
                           self.recent_scenes, self.shared_digest)
        # All legal updates are prepared before publishing any mutable state.
        self._prediction_catalog = self._prediction_loaded = None
        for scene, before, after in changes:
            if self._uses_links(scene):
                self.links.remember(scene, before, after, self.groups[operation.calculation][scene])
            if self._track_fingerprint:
                previous = self.term_hashes[scene]
                if previous is None:
                    previous = self._part(("term", scene, before))
                value = self._part(("term", scene, after))
                self.digest ^= previous ^ value
                self.term_hashes[scene] = value
            if self.shared_digest is not None:
                previous = self.shared_term_hashes[scene]
                if previous is None:
                    previous = self._shared_term(scene, before)
                value = self._shared_term(scene, after)
                self.shared_digest ^= previous ^ value
                self.shared_term_hashes[scene] = value
            self.terms[scene] = after
            self.remaining += int(before.done) - int(after.done)
            self.dirty.add(scene)
        if not operation.reused:
            self.available.add(operation.result)
            for identity in self.registry.calculation_ids[operation.result]:
                self._touch_group(identity)
                if self._scalar_terminals is not None:
                    self._scalar_terminals.availability(identity, self)
            data = self.registry.data[operation.result]
            self.catalog[data.name] = data.spec
            self._catalog_bytes += math.prod(data.spec.shape) * data.spec.dtype.itemsize
            if self._track_fingerprint:
                self.digest ^= self._part(("available", operation.result))
            if self.shared_digest is not None:
                self.shared_digest ^= self._shared_data(operation.result)
        self.applied += 1
        self.logical_effects += 1
        self.recent_scenes = frozenset(scene for scene, _, _ in changes)
        return undo

    def undo(self, token):
        self._prediction_catalog = self._prediction_loaded = None
        for scene, term in token.terms:
            self.links.restore(scene, term)
            self.remaining += int(self.terms[scene].done) - int(term.done)
            self.terms[scene] = term
            if self._track_fingerprint:
                self.term_hashes[scene] = None
            if self.shared_digest is not None:
                self.shared_term_hashes[scene] = None
            self.dirty.add(scene)
        for data in token.created:
            self.available.remove(data)
            for identity in self.registry.calculation_ids[data]:
                self._touch_group(identity)
                if self._scalar_terminals is not None:
                    self._scalar_terminals.availability(identity, self)
            spec = self.registry.data[data].spec
            self._catalog_bytes -= math.prod(spec.shape) * spec.dtype.itemsize
            del self.catalog[self.registry.data[data].name]
        self.digest = token.digest
        self.shared_digest = token.shared_digest
        self.recent_scenes = token.recent_scenes
        self.undo_count += 1

    def _selected_description(self, state, operation, occurrence, scene_joins):
        """Bind a registered calculation to one selected full-axis occurrence.

        Search already fixed the calculation's cost, workspace, result and
        kernel choice. Only the live operand labels and scene-local join labels
        need to be restored here; independent pricing belongs to debug audit.
        """
        scene = occurrence.scene
        term = self.terms[scene]
        plain = tuple(i for i, data in enumerate(term.factors)
                      if self.registry.data[data].core is None)
        positions = tuple(plain.index(i) for i in occurrence.positions)
        selected = tuple(state.pending[scene].factors[i] for i in positions)
        output = tuple(self.symbols[scene][axis] for axis in occurrence.output)
        calculation = self.registry.calculations[operation.calculation]
        recipe = calculation.recipe
        join = None if occurrence.join is None else scene_joins[scene][occurrence.join]
        names = tuple(factor.tensor for factor in selected)
        if join is not None:
            if join.core is not None and recipe.core is not None:
                inputs = (join.core_labels, *(factor.labels for factor in selected))
                operands = join.core.inputs + names
            else:
                inputs = (join.spin_labels,
                          join.elemental_labels(state.catalog[join.elemental]),
                          *(factor.labels for factor in selected))
                operands = (join.spin, join.elemental, *names)
        else:
            inputs = tuple(factor.labels for factor in selected)
            operands = names + recipe.operands[len(names):] if recipe.kind == "sparse_bridge" else names
        equation = AxisPattern(inputs, output)
        reused = self.registry.data[operation.result].name in state.loaded
        action = replace(recipe, tensor=self.registry.data[operation.result].name,
                         operands=operands, equation=equation, term_index=scene,
                         positions=recipe.positions if join is not None else positions,
                         reused=reused,
                         seconds=0.0 if reused else calculation.seconds,
                         join_id=join.name if join is not None else recipe.join_id,
                         kernel_description=(join.kernel_description if join is not None
                                             else recipe.kernel_description))
        if type(recipe.core) is FactorizedWideReduction:
            action = replace(action, core=replace(recipe.core, equation=equation))
        return action, self.registry.data[operation.result].spec, \
            (0 if reused else calculation.workspace), output, positions, join

    def audit(self, operations, expected_cost):
        """Build executable Actions and check completion/cost for the fixed path.

        Full expression and collision diagnostics are optional. Keep the
        two-argument audit protocol used by generic DFS domains unchanged.
        """
        if (not self.debug and not self.problem.baryon_joins and not self.problem.sparse_bridges
                and not self.has_sum_barriers):
            from .selected_path import materialize_dense_path
            try:
                state, self.snippets = materialize_dense_path(
                    self.problem, operations, expected_cost, self.registry)
                self.audited_steps += len(operations)
                return state
            finally:
                self.release()
        self.reset()
        state = ContractionState.initial(self.problem)
        state = replace(state, loaded=frozenset(state.catalog) | _live_inputs(state))
        state = state.with_indexed_rows()
        self.snippets = []
        actions = []
        computations = {} if self.debug else None
        # Fixed-path replay repeatedly visits the same scenes. Keep the
        # immutable declaration scopes in problem order for all operations.
        scene_joins = self._joins_by_scene
        bridge_index = BridgeIndex(self.problem.sparse_bridges) if self.problem.sparse_bridges else None
        try:
            for operation in operations:
                begin = len(actions)
                # Only this operation's actions are needed during replay.
                # Keep the complete immutable history in one final tuple,
                # rather than copying its growing prefix for every occurrence.
                state = state.with_action_history(())
                self._index(lambda: None)
                occurrences = tuple(self.groups.get(operation.calculation, {}).values())
                if not occurrences:
                    raise RuntimeError("selected hash path has no independently matching operation")
                for occurrence in sorted(occurrences, key=lambda o: o.scene):
                    scene = occurrence.scene
                    if not self.debug:
                        action, spec, workspace, output, positions, join = self._selected_description(
                            state, operation, occurrence, scene_joins)
                    else:
                        term = self.terms[scene]
                        plain = tuple(i for i, data in enumerate(term.factors)
                                      if self.registry.data[data].core is None)
                        positions = tuple(plain.index(i) for i in occurrence.positions)
                        join = None if occurrence.join is None else scene_joins[scene][occurrence.join]
                        recipe = self.registry.calculations[operation.calculation].recipe
                        if type(recipe.core) is FactorizedWideReduction:
                            from ..factorized_reduction import describe_factorized_action
                            action, spec, workspace, _ = describe_factorized_action(state, scene, positions,
                                recipe.core.large_operands, profile=self.profile,
                                output_order=tuple(self.symbols[scene][axis] for axis in occurrence.output),
                                tile_elements=recipe.core.tile_elements)
                            output = action.equation.output
                        else:
                            action, spec, workspace, output, positions = describe(state, scene, positions,
                                join=join, profile=self.profile,
                                output_order=tuple(self.symbols[scene][axis] for axis in occurrence.output) if join is None else None,
                                scene_joins=scene_joins,
                                scoped_bridges=None if bridge_index is None else bridge_index.scoped(scene))
                    if action.tensor != self.registry.data[operation.result].name:
                        raise RuntimeError("selected integer identity differs from full-axis replay")
                    if self.debug:
                        definition = computation_key(action)
                        previous = computations.setdefault(action.tensor, definition)
                        if previous != definition:
                            raise RuntimeError("selected data hash merged different full computations")
                    if self.debug and type(action.core) is FactorizedWideReduction:
                        from ..factorized_reduction import replay_factorized_action
                        state = replay_factorized_action(state, action, self.profile)
                    else:
                        if self.debug:
                            state = replay(state, action, self.profile)
                        else:
                            # apply_description appends to the state's tuple.
                            # Keep it one action long; the delivered history is
                            # accumulated once below, outside the state copies.
                            state = apply_description(state.with_action_history(()), action,
                                spec, workspace, output, positions, join=join,
                                scene_joins=scene_joins)
                    if not self.debug:
                        actions.append(action)
                self.apply(operation, lambda: None)
                # The full Action replay above is required by execution and
                # resource scheduling. Rebuilding expressions solely to compare
                # the two internal representations is a debug diagnostic.
                for occurrence in occurrences:
                    scene = occurrence.scene
                    term = self.terms[scene]
                    if self.debug and state.pending[scene] != self._decode(scene, raw=True):
                        raise RuntimeError("selected integer path changed its full expression")
                    if term.done and state.results[scene] != self.registry.data[term.factors[0]].name:
                        raise RuntimeError("selected integer path changed its final result")
                self.audited_steps += 1
                if self.debug:
                    actions.extend(state.actions)
                self.snippets.append((begin, len(actions)))
            state = state.with_action_history(actions).with_tuple_rows()
            if self.debug:
                for scene, term in enumerate(self.terms):
                    if state.pending[scene] != self._decode(scene, raw=True):
                        raise RuntimeError("selected integer path changed its final expression")
                    if term.done and state.results[scene] != self.registry.data[term.factors[0]].name:
                        raise RuntimeError("selected integer path changed its final result")
            if not self.is_done() or not state.done or not math.isclose(state.seconds, expected_cost, rel_tol=1e-11, abs_tol=1e-14):
                raise RuntimeError("selected integer path failed final completion or cost audit")
            return state
        finally:
            self.release()

    def occurrence_bindings(self, calculation, checkpoint=lambda: None):
        """Read-only snapshot of this state's occurrence bindings for one calculation.

        Rows are (scene, positions, output, operands, inputs) in ascending
        scene order. Positions index the current term's factors, exactly as
        apply() consumes them; operands are the selected data names in position
        order; and output/inputs are renamed back to this scene's original axis
        labels so a caller replaying the same route sees the same relation
        without the private per-scene numbering. Refreshing the incremental
        index is the only side effect; no definition, availability or
        occurrence is created. An empty result means no snapshot, so callers
        fall back to independent matching.
        """
        self._index(checkpoint)
        occurrences = self.groups.get(calculation)
        if not occurrences:
            return ()
        symbols = self.symbols
        terms = self.terms
        definitions = self.registry.data
        patterns = self.registry.patterns
        rows = []
        for scene, occurrence in sorted(occurrences.items()):
            term = terms[scene]
            pattern = patterns[term.pattern]
            positions = occurrence.positions
            rows.append((scene, positions,
                         tuple(symbols[scene][axis] for axis in occurrence.output),
                         tuple(definitions[term.factors[i]].name for i in positions),
                         tuple(tuple(symbols[scene][axis] for axis in pattern.inputs[i])
                               for i in positions)))
        return tuple(rows)
