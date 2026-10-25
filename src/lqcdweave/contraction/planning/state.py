"""Immutable symbolic contraction state and numerical-free transitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping as MappingABC, Sequence
from functools import lru_cache, cached_property
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import opt_einsum

from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    Action,
    BaryonJoinSpec,
    ContractionProblem,
    Factor,
    PendingTerm,
    SourceSpec,
    SparseBridgeSpec,
    TensorSpec,
    DEFAULT_JOIN_INDEX_BLOCK_SIZE,
    ESTIMATED_JOIN_INDEX_BYTES_PER_ROW,
    JOIN_COST_STATUS,
    bridge_point_equalities_covered,
    bridge_intermediate_labels,
    cycle_external_labels,
    cycle_requires_generic_join,
    cycle_intermediate_labels,
    parse_term as _parse_term,
)
from ..performance.api import CostProfile


def _outer_product_axes(left, right, retained):
    """Detect ordinary and batched outer products; allow aligned/scalar products."""
    left, right = set(left), set(right)
    return bool(left - right and right - left and (left | right) <= set(retained))


@dataclass(frozen=True, slots=True)
class _ImmutableKey:
    """Search-local cached hash with full equality, never a scientific ID."""
    values: tuple
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash(self.values))

    def __hash__(self):
        return self._hash


class _Catalog(MappingABC):
    """Owned immutable base with a bounded overlay of recent entries."""
    __slots__ = ("_values", "_overlay", "_size")
    overlay_limit = 64

    def __init__(self, values):
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))
        object.__setattr__(self, "_overlay", MappingProxyType({}))
        object.__setattr__(self, "_size", len(self._values))

    def __setattr__(self, name, value):
        raise AttributeError("catalog snapshots are immutable")

    def __getitem__(self, key):
        if key in self._overlay:
            return self._overlay[key]
        return self._values[key]

    def __iter__(self):
        yield from self._values
        yield from (key for key in self._overlay if key not in self._values)

    def __len__(self):
        return self._size

    def changed_keys_since(self, previous):
        """Existing keys with changed specs; newly added keys affect new rows.

        A shared immutable base restricts comparisons to bounded overlays.
        Backtracking or compaction falls back to all previous keys.
        """
        if self is previous:
            return frozenset()
        names = (self._overlay.keys() | previous._overlay.keys()
                 if self._values is previous._values else previous.keys())
        return frozenset(key for key in names if key in previous and self.get(key) != previous[key])

    def updated(self, entries):
        overlay = dict(self._overlay)
        overlay.update(entries)
        size = len(self._values) + sum(key not in self._values for key in overlay)
        values = self._values
        if len(overlay) > self.overlay_limit:
            values = dict(self._values)
            values.update(overlay)
            values, overlay = MappingProxyType(values), {}
        result = object.__new__(type(self))
        object.__setattr__(result, "_values", values)
        object.__setattr__(result, "_overlay", MappingProxyType(overlay))
        object.__setattr__(result, "_size", size)
        return result


@dataclass(frozen=True, slots=True, eq=False)
class _RowTable(Sequence):
    """Immutable blocks for single-occurrence updates during fixed replay."""
    blocks: tuple[tuple, ...]
    size: int
    block_size = 64

    @classmethod
    def from_values(cls, values):
        values = tuple(values)
        return cls(tuple(values[i:i + cls.block_size] for i in range(0, len(values), cls.block_size)),
                   len(values))

    def __len__(self):
        return self.size

    def __iter__(self):
        for block in self.blocks:
            yield from block

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.blocks[index // self.block_size][index % self.block_size]

    def updated(self, index, value):
        block_index, offset = divmod(index, self.block_size)
        blocks = list(self.blocks)
        block = list(blocks[block_index])
        block[offset] = value
        blocks[block_index] = tuple(block)
        return type(self)(tuple(blocks), self.size)

    def __eq__(self, other):
        if isinstance(other, _RowTable):
            return self.size == other.size and self.blocks == other.blocks
        if isinstance(other, tuple):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self))


def _replace_row(values, index, value):
    if isinstance(values, _RowTable):
        return values.updated(index, value)
    updated = list(values)
    updated[index] = value
    return tuple(updated)






















def has_native_catalog(state):
    """Whether a native state owns the immutable catalog implementation.

    Execution uses this read-only type proof when admitting static audits to
    its owner-scoped cache. Catalog subclasses may expose mutable protocols.
    This does not inspect values, validate a plan, or perform planning.
    """
    return type(state) is ContractionState and type(state.catalog) is _Catalog


@dataclass(frozen=True)
class ContractionState:
    problem: ContractionProblem
    catalog: Mapping[str, TensorSpec]
    pending: tuple[PendingTerm | None, ...]
    loaded: frozenset[str]
    results: tuple[str | None, ...]
    completed_joins: frozenset[tuple[int, str]] = frozenset()
    actions: tuple[Action, ...] = ()
    seconds: float = 0.0
    peak_bytes: int = 0
    peak_host_bytes: int = 0
    peak_device_bytes: int = 0
    opened_regions: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "catalog", self.catalog if isinstance(self.catalog, _Catalog) else _Catalog(self.catalog))
        object.__setattr__(self, "pending", self.pending if isinstance(self.pending, _RowTable) else tuple(self.pending))
        object.__setattr__(self, "loaded", frozenset(self.loaded))
        object.__setattr__(self, "results", self.results if isinstance(self.results, _RowTable) else tuple(self.results))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "opened_regions", frozenset(self.opened_regions))
        object.__setattr__(self, "completed_joins", frozenset(self.completed_joins))

    @cached_property
    def search_key(self):
        generated = frozenset(key for key in self.catalog if key not in self.problem.specs)
        return _ImmutableKey((self.pending, self.loaded, self.results, self.completed_joins,
                              generated, self.opened_regions))

    @cached_property
    def live_inputs(self):
        from .actions import _find_live_inputs
        return frozenset(_find_live_inputs(self))

    @classmethod
    def initial(cls, problem: ContractionProblem):
        pending = []
        for term in problem.terms:
            labels, output = term.subscripts.inputs, term.subscripts.output
            pending.append(
                PendingTerm(
                    tuple(Factor(key, labels) for key, labels in zip(term.inputs, labels, strict=True)), output
                )
            )
        return cls(problem, dict(problem.specs), tuple(pending), frozenset(), tuple(None for _ in pending))

    @property
    def done(self) -> bool:
        return all(result is not None for result in self.results)

    @cached_property
    def _resident_totals(self):
        regions = {
            self.problem.sources[key].region: self.problem.sources[key].region_bytes
            for key in self.loaded
            if key in self.problem.sources
        }
        snapshots = sum(self._snapshot_bytes(key) for key in self.loaded if key in self.problem.sources)
        intermediates = sum(_nbytes(self.catalog[key]) for key in self.loaded if key not in self.problem.sources)
        total = sum(regions.values()) + snapshots + intermediates
        host = total if self.problem.backend == "numpy" else sum({
            self.problem.sources[key].region: self.problem.sources[key].host_region_bytes
            for key in self.loaded if key in self.problem.sources
        }.values())
        return total, host

    @property
    def resident_bytes(self) -> int:
        return self._resident_totals[0]

    def _snapshot_bytes(self, key: str) -> int:
        source = self.problem.sources[key]
        if source.snapshot_bytes is not None:
            return source.snapshot_bytes
        return _nbytes(self.catalog[key]) if key in self.catalog else 0

    @property
    def resident_host_bytes(self) -> int:
        return self._resident_totals[1]

    def _seed_residency(self, successor, delta=0, host_delta=0):
        """Carry exact integer totals only across known immutable transitions."""
        total, host = self._resident_totals
        object.__setattr__(successor, "_resident_totals", (
            total + delta, host + (delta if self.problem.backend == "numpy" else host_delta),
        ))
        return successor

    def _result_residency_delta(self, key, spec):
        if key in self.problem.sources:
            source = self.problem.sources[key]
            new = source.snapshot_bytes if source.snapshot_bytes is not None else _nbytes(spec)
            old = self._snapshot_bytes(key) if key in self.loaded else 0
        else:
            new = _nbytes(spec)
            old = _nbytes(self.catalog[key]) if key in self.loaded else 0
        return new - old

    def _without_peaks(self):
        return self._seed_residency(replace(self, peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0))

    def with_action_history(self, actions):
        """Replace construction history while preserving exact residency totals.

        Fixed-path builders may collect validated Actions in a local append
        buffer. This changes neither the expression nor accumulated costs;
        publish the complete history before returning an executable state.
        """
        return self._seed_residency(replace(self, actions=tuple(actions)))

    def with_indexed_rows(self):
        """Use persistent row blocks while constructing a fixed replay."""
        if len(self.pending) <= _RowTable.block_size:
            return self
        return self._seed_residency(replace(self, pending=_RowTable.from_values(self.pending),
                                             results=_RowTable.from_values(self.results)))

    def with_tuple_rows(self):
        """Publish the ordinary immutable row tuples after fixed replay."""
        return self._seed_residency(replace(self, pending=tuple(self.pending), results=tuple(self.results)))

    def load(self, key: str, profile: CostProfile, limit_bytes: int, host_limit_bytes: int | None = None):
        if key not in self.problem.sources or key in self.loaded:
            raise ValueError("load requires an absent original input")
        source = self.problem.sources[key]
        first = not any(
            self.problem.sources[item].region == source.region for item in self.loaded if item in self.problem.sources
        )
        if self.problem.single_read_regions and first and source.region in self.opened_regions:
            raise MemoryError("backing region cannot be read a second time")
        extra = (source.region_bytes if first else 0) + self._snapshot_bytes(key)
        transient = self.resident_bytes + extra
        peak = max(self.peak_bytes, transient + (source.staging_bytes if self.problem.backend == "numpy" else 0))
        if peak > limit_bytes:
            raise MemoryError("load exceeds declared memory cap")
        host_transient = (
            peak
            if self.problem.backend == "numpy"
            else self.resident_host_bytes + (source.host_region_bytes if first else 0) + source.staging_bytes
        )
        host_peak = max(self.peak_host_bytes, host_transient)
        device_peak = max(self.peak_device_bytes, peak if self.problem.backend == "cupy" else 0)
        if host_limit_bytes is not None and host_peak > host_limit_bytes:
            raise MemoryError("load exceeds declared host staging cap")
        seconds = profile.snapshot_seconds(self._snapshot_bytes(key))
        return self._seed_residency(replace(
            self,
            loaded=self.loaded | {key},
            seconds=self.seconds + seconds,
            peak_bytes=peak,
            peak_host_bytes=host_peak,
            peak_device_bytes=device_peak,
            actions=self.actions + (Action("load", key, seconds=seconds),),
            opened_regions=self.opened_regions | {source.region} if self.problem.single_read_regions else self.opened_regions,
        ), extra, source.host_region_bytes if first else 0)

    def release(self, key: str):
        if key not in self.loaded or key in self.results:
            raise ValueError("release requires a resident non-output tensor")
        if key not in self.problem.sources and key in self.live_inputs:
            raise ValueError("intermediate recomputation is unsupported")
        host_delta = 0
        if key in self.problem.sources:
            source = self.problem.sources[key]
            removed = self._snapshot_bytes(key)
            if (source.region_bytes or source.host_region_bytes) and not any(
                item != key and item in self.problem.sources
                and self.problem.sources[item].region == source.region for item in self.loaded
            ):
                removed += source.region_bytes
                host_delta = -source.host_region_bytes
        else:
            removed = _nbytes(self.catalog[key])
        return self._seed_residency(
            replace(self, loaded=self.loaded - {key}, actions=self.actions + (Action("release", key),)),
            -removed, host_delta,
        )

    def contract(
        self,
        term_index: int,
        pair: tuple[int, ...],
        profile: CostProfile,
        limit_bytes: int,
        *,
        workspace_bytes: int | None = None,
    ):
        layout = self._contract_layout(term_index, pair)
        return self._contract_from_layout(term_index, pair, layout, profile, limit_bytes,
                                          workspace_bytes=workspace_bytes)

    def _contract_layout(self, term_index, pair, *, scene_joins=None, active_bridges=None, allow_nary=False):
        """Resolve a computation identity before costing or building effects."""
        term = self.pending[term_index]
        if term is None:
            raise ValueError("scene result is already complete")
        # Build the active scene index once, including the other scenes whose
        # matching subexpressions will be replaced below.
        if scene_joins is None:
            scene_joins = {}
            for join in self.problem.baryon_joins:
                if (join.scene, join.name) not in self.completed_joins:
                    scene_joins.setdefault(join.scene, []).append(join)
        joins_pending = bool(scene_joins.get(term_index))
        if len(term.factors) == 1:
            if pair != (0,):
                raise ValueError("unary result requires path step (0,)")
            if joins_pending:
                raise ValueError("baryon joins must be applied before final unary reduction")
        elif (len(pair) != 2 and not (allow_nary and len(pair) in (3,4))
              or any(a >= b for a, b in zip(pair, pair[1:]))
              or not pair or pair[0] < 0 or pair[-1] >= len(term.factors)):
            raise ValueError("contract requires an ordered pair of factor positions")
        selected = tuple(term.factors[index] for index in pair)
        rest = tuple(f for index, f in enumerate(term.factors) if index not in pair)
        needed = set(term.output + tuple(c for f in rest for c in f.labels))
        for join in scene_joins.get(term_index, ()):
            needed.update(join.spin_labels + join.mode_labels)
        labels = tuple(dict.fromkeys(c for f in selected for c in f.labels if c in needed))
        if not rest and not joins_pending:
            labels = term.output
        dimensions: dict[int, int] = {}
        for f in selected:
            for label, size in zip(f.labels, self.catalog[f.tensor].shape, strict=True):
                dimensions.setdefault(label, size)
        shape = tuple(dimensions[c] for c in labels)
        dtype = np.result_type(*(self.catalog[f.tensor].dtype for f in selected))
        barriers = self.problem.terms[term_index].sum_barriers
        if barriers and dtype in (np.dtype('float32'), np.dtype('complex64')):
            before = {c for f in term.factors for c in f.labels}
            after = set(labels) | {c for f in rest for c in f.labels}
            if any(before.intersection(trigger) and not after.intersection(trigger)
                   and after.intersection(dependent) for trigger, dependent in barriers):
                raise ValueError('last spatial sum must retain the coupled spin reduction')
        active_bridges = active_bridges if active_bridges is not None else tuple(
            b
            for b in self.problem.sparse_bridges
            if b.scene in (None, term_index) and {b.left, b.right} <= {factor.tensor for factor in term.factors}
        )
        bridge_record = next(
            (
                b
                for b in active_bridges
                if len(selected) == 2 and (selected[0].tensor, selected[1].tensor) == (b.left, b.right)
            ),
            None,
        )
        bridge = _bridge_in_operand_order(bridge_record)
        protected = {name for b in active_bridges for name in (b.left, b.right)}
        if any(f.tensor in protected for f in selected) and bridge is None:
            raise ValueError("P2P propagation legs must use their sparse bridge first")
        cycle_candidates = (
            tuple(
                item
                for item in active_bridges
                if item is not bridge_record and {item.left, item.right} == {selected[0].tensor, selected[1].tensor}
            )
            if bridge is not None
            else ()
        )
        if len(cycle_candidates) > 1:
            raise ValueError("more than two P2P currents on one propagator pair are unsupported")
        cycle_record = cycle_candidates[0] if cycle_candidates else None
        cycle_bridge = _bridge_in_operand_order(cycle_record)
        join_operation = False
        if cycle_bridge is not None:
            _, external, _ = cycle_intermediate_labels(
                AxisPattern(tuple(f.labels for f in selected), ()), bridge, cycle_bridge,
                needed_labels=tuple(sorted(needed)),
            )
            labels = tuple(dict.fromkeys(label for label in external if label in needed))
            if not rest and not joins_pending:
                labels = term.output
            point_equation = AxisPattern(tuple(f.labels for f in selected), labels)
            join_operation = cycle_requires_generic_join(
                point_equation, bridge, cycle_bridge,
            )
            if self.problem.backend != "numpy" and join_operation:
                raise NotImplementedError(
                    "cross-current sparse point joins currently require the CPU bounded COO backend; "
                    "use optimize='none' only with a NumPy ContractionProblem"
                )
            shape = tuple(dimensions[label] for label in labels)
        elif bridge is not None:
            point_equation = AxisPattern(tuple(f.labels for f in selected), labels)
            join_operation = not bridge_point_equalities_covered(
                point_equation, bridge.left_axes, bridge.right_axes,
                same_point=bridge.same_point,
            )
            if self.problem.backend != "numpy" and join_operation:
                raise NotImplementedError(
                    "external-point COO bridges currently require the CPU bounded COO backend"
                )
        if bridge is not None:
            dtype = np.result_type(dtype, np.complex128)
        elif len(selected) == 2:
            # Shared retained axes can be batch axes: broadcasting two sets
            # of independent free axes is still an outer product. Aligned
            # elementwise multiplication and scalar scaling remain legal.
            if _outer_product_axes(selected[0].labels, selected[1].labels, labels):
                raise ValueError("outer products are not planning operations")
        signature = (
            tuple((f.tensor, f.labels) for f in selected),
            labels,
            (bridge.record, bridge.point_shape, bridge.left_axes, bridge.right_axes,
             bridge.same_point, bridge.transpose_bridge)
            if bridge
            else None,
            _bridge_identity(cycle_bridge),
        )
        if (barriers and dtype in (np.dtype('float32'), np.dtype('complex64'))
                and any(before.intersection(trigger) and not after.intersection(trigger)
                        for trigger, _ in barriers)):
            signature += ('wide_accumulation',)
        key = "node:" + hashlib.sha256(repr(signature).encode()).hexdigest()[:24]
        return (selected, labels, dimensions, shape, dtype, bridge, cycle_bridge, join_operation, key, scene_joins)

    def _describe_contract(self, term_index, pair, layout, profile,
                           *, workspace_bytes=None):
        """Describe cost/output/workspace without constructing shared effects."""
        (selected, labels, dimensions, shape, dtype, bridge, cycle_bridge, join_operation, key, scene_joins) = layout
        if any(f.tensor not in self.loaded for f in selected):
            raise ValueError("contract operands must be resident")
        if bridge is not None and bridge.record not in self.loaded:
            raise ValueError("P2P sparse records must be resident before bridging")
        if cycle_bridge is not None and cycle_bridge.record not in self.loaded:
            raise ValueError("both P2P sparse records must be resident before closing the cycle")
        equation = AxisPattern(tuple(f.labels for f in selected), labels)
        spec = TensorSpec(shape, dtype)
        already = key in self.loaded
        if key in self.catalog and not already:
            raise ValueError("released intermediate tensors cannot be recomputed")
        reduction = math.prod(dimensions[c] for c in set(tuple(c for f in selected for c in f.labels)) - set(labels))
        flops = math.prod(shape) * max(reduction, 1) * max(len(selected) - 1, 1)
        traffic = sum(_nbytes(self.catalog[f.tensor]) for f in selected) + _nbytes(spec)
        bridge_projection = False
        raw_bridge_bytes = 0
        join_workspace_bytes = 0
        kernel_calls = 1
        barriers = self.problem.terms[term_index].sum_barriers
        wide_accumulation = False
        if barriers and dtype in (np.dtype('float32'), np.dtype('complex64')):
            pending = self.pending[term_index]
            before = {a for factor in pending.factors for a in factor.labels}
            after = set(labels) | {a for i, factor in enumerate(pending.factors)
                                   if i not in pair for a in factor.labels}
            wide_accumulation = any(before.intersection(trigger) and not after.intersection(trigger)
                                    for trigger, _ in barriers)
        if wide_accumulation:
            kernel_calls = 2
        if bridge is not None:
            left_spec = self.catalog[bridge.left]
            right_spec = self.catalog[bridge.right]
            endpoint_work = (
                left_spec.shape[bridge.left_axes[0]]
                * right_spec.shape[bridge.right_axes[0]]
                * left_spec.shape[bridge.left_axes[2]]
                * right_spec.shape[bridge.right_axes[2]]
            )
            if cycle_bridge is None:
                if join_operation:
                    pair_bound = bridge.nnz_upper * bridge.part_count
                    flops = pair_bound * 2 * endpoint_work
                    kernel_calls = pair_bound
                    traffic += (
                        self.problem.sources[bridge.record].region_bytes
                        + pair_bound * _nbytes(spec)
                    )
                else:
                    _, raw_labels, output = bridge_intermediate_labels(
                        equation,
                        bridge.left_axes,
                        bridge.right_axes,
                        same_point=bridge.same_point,
                    )
                    raw_elements = math.prod(dimensions[label] for label in raw_labels)
                    bridge_projection = raw_labels != output
                    raw_bridge_bytes = raw_elements * dtype.itemsize
                    flops = max(bridge.nnz_upper, 1) * raw_elements * endpoint_work
                    if bridge_projection:
                        flops += raw_elements
                        traffic += 2 * raw_bridge_bytes
                    traffic += self.problem.sources[bridge.record].region_bytes
                    kernel_calls = bridge.part_count + int(bridge_projection)
            else:
                first_external, final_external, output = cycle_external_labels(equation, bridge, cycle_bridge)
                raw_elements = math.prod(dimensions[label] for label in first_external)
                final_elements = math.prod(dimensions[label] for label in final_external)
                bridge_projection = final_external != output
                raw_bridge_bytes = raw_elements * dtype.itemsize
                second_left = self.catalog[cycle_bridge.left]
                second_right = self.catalog[cycle_bridge.right]
                second_endpoint_work = (
                    second_left.shape[cycle_bridge.left_axes[0]]
                    * second_right.shape[cycle_bridge.right_axes[0]]
                    * second_left.shape[cycle_bridge.left_axes[2]]
                    * second_right.shape[cycle_bridge.right_axes[2]]
                )
                if join_operation:
                    pair_bound = (
                        bridge.nnz_upper * cycle_bridge.nnz_upper
                        * bridge.part_count * cycle_bridge.part_count
                    )
                    # Conservative all-pairs upper bound; no sparse_join cost
                    # fitting/production timings are available in this build.
                    flops = pair_bound * 2 * (endpoint_work + second_endpoint_work)
                    kernel_calls = pair_bound
                    join_workspace_bytes = (
                        min(cycle_bridge.nnz_upper, DEFAULT_JOIN_INDEX_BLOCK_SIZE)
                        * ESTIMATED_JOIN_INDEX_BYTES_PER_ROW
                    )
                    traffic += (
                        self.problem.sources[bridge.record].region_bytes
                        + self.problem.sources[cycle_bridge.record].region_bytes
                        + join_workspace_bytes
                        + pair_bound * _nbytes(spec)
                    )
                else:
                    flops = (
                        max(bridge.nnz_upper, 1) * raw_elements * endpoint_work
                        + max(cycle_bridge.nnz_upper, 1) * final_elements * second_endpoint_work
                        + (final_elements if bridge_projection else 0)
                    )
                    traffic += (
                        2 * raw_bridge_bytes
                        + self.problem.sources[bridge.record].region_bytes
                        + self.problem.sources[cycle_bridge.record].region_bytes
                    )
                    kernel_calls = bridge.part_count + cycle_bridge.part_count + int(bridge_projection)
                if not join_operation:
                    for factor in selected:
                        repeated = len(factor.labels) - len(set(factor.labels))
                        if repeated:
                            elements = math.prod(self.catalog[factor.tensor].shape)
                            flops += repeated * elements
                            traffic += repeated * 3 * _nbytes(self.catalog[factor.tensor])
                            kernel_calls += repeated * 2
        if workspace_bytes is None:
            workspace_bytes = sum(_nbytes(self.catalog[f.tensor]) for f in selected) + _nbytes(spec)
            if bridge is not None:
                workspace_bytes += max(self.problem.sources[bridge.record].region_bytes,
                                       self._snapshot_bytes(bridge.record))
                if cycle_bridge is not None:
                    workspace_bytes += max(self.problem.sources[cycle_bridge.record].region_bytes,
                                           self._snapshot_bytes(cycle_bridge.record))
                    if join_operation:
                        # Fixed chunk plus one output-sized einsum result and
                        # scalar-multiply temporary. Python overhead is an
                        # estimate, not a measured allocation upper bound.
                        workspace_bytes += join_workspace_bytes + 2 * _nbytes(spec)
                    else:
                        # Repeated input labels need rank-preserving diagonal masks
                        # before sparse elimination. Allow old/new copies and masks.
                        for factor in selected:
                            if len(set(factor.labels)) != len(factor.labels):
                                input_spec = self.catalog[factor.tensor]
                                workspace_bytes += 2 * _nbytes(input_spec) + math.prod(input_spec.shape)
                        workspace_bytes += raw_bridge_bytes
                        workspace_bytes += (
                            max(cycle_bridge.nnz_upper, 1)
                            * final_elements
                            * second_endpoint_work
                            * dtype.itemsize
                        )
                elif join_operation:
                    workspace_bytes += join_workspace_bytes + 2 * _nbytes(spec)
                elif bridge_projection:
                    workspace_bytes += raw_bridge_bytes
                if not join_operation:
                    for factor, axes in zip(selected, (bridge.left_axes, bridge.right_axes), strict=True):
                        input_spec = self.catalog[factor.tensor]
                        positions = {axis % len(input_spec.shape) for axis in axes}
                        external = math.prod(size for i, size in enumerate(input_spec.shape) if i not in positions)
                        spin = input_spec.shape[axes[0]]
                        color = input_spec.shape[axes[2]]
                        workspace_bytes += external * spin * bridge.nnz_upper * color * input_spec.dtype.itemsize
        seconds = (
            0.0
            if already
            else profile.contract_seconds(
                flops,
                traffic,
                kind="sparse_bridge" if bridge else "contract",
                dtype=spec.dtype,
                kernel_calls=kernel_calls,
                sparse_records=2 if cycle_bridge is not None and not join_operation else 1,
            )
        )
        return Action(
                    "sparse_bridge" if bridge else "contract",
                    key,
                    tuple(f.tensor for f in selected)
                    + ((bridge.record,) if bridge else ())
                    + ((cycle_bridge.record,) if cycle_bridge else ()),
                    equation,
                    seconds,
                    already,
                    bridge.used_np if bridge else 0,
                    left_axes=bridge.left_axes if bridge else (-3, -2, -1),
                    right_axes=bridge.right_axes if bridge else (0, 1, 2),
                    same_point=bridge.same_point if bridge else False,
                    transpose_bridge=bridge.transpose_bridge if bridge else False,
                    term_index=term_index,
                    positions=pair,
                    join_id="bounded_coo_join" if join_operation else "",
                    used_np_right=bridge.used_np_right if bridge else None,
                    wide_accumulation=wide_accumulation,
                ), spec, workspace_bytes

    def _contract_from_layout(self, term_index, pair, layout, profile, limit_bytes,
                              *, workspace_bytes=None, factor_index=None, layout_index=None, changes=None,
                              description=None):
        """Build one shared logical effect from an already resolved layout."""
        (selected, labels, dimensions, shape, dtype, bridge, cycle_bridge, join_operation, key, scene_joins) = layout
        if description is None:
            description = self._describe_contract(term_index, pair, layout, profile,
                                                  workspace_bytes=workspace_bytes)
        action, spec, workspace_bytes = description
        if description[0].tensor != key or spec != TensorSpec(shape, dtype):
            raise ValueError("computation description differs from its layout")
        if any(f.tensor not in self.loaded for f in selected):
            raise ValueError("contract operands must be resident")
        if any(b is not None and b.record not in self.loaded for b in (bridge, cycle_bridge)):
            raise ValueError("sparse records must be resident before bridging")
        if action.reused != (key in self.loaded) or (key in self.catalog and key not in self.loaded):
            raise ValueError("computation description belongs to a different residency")
        already, seconds = action.reused, action.seconds
        catalog = self.catalog.updated(((key, spec),))
        peak = max(self.peak_bytes, self.resident_bytes + (0 if already else _nbytes(spec) + workspace_bytes))
        if peak > limit_bytes:
            raise MemoryError("contract exceeds declared memory cap")
        pending = list(self.pending)
        results = list(self.results)
        barriers = self.problem.terms[term_index].sum_barriers
        if (layout_index is not None and layout_index.factors is not None
                and not self.problem._has_sum_barriers):
            if layout_index.state.pending is not self.pending:
                raise ValueError("layout index belongs to a different pending state")
            for index, (updated, result) in layout_index.substitutions(layout):
                pending[index] = updated
                if result is not None:
                    results[index] = result
                if changes is not None:
                    changes.append((index, updated, result))
            matches = ()
        elif factor_index is None:
            matches = ((index, _matching_positions(other.factors, selected))
                       for index, other in enumerate(self.pending) if other is not None)
        else:
            if factor_index.pending is not self.pending:
                raise ValueError("factor index belongs to a different pending state")
            matches = factor_index.matching_terms(selected)
        replacement = Factor(key, labels)
        for index, positions_iter in matches:
            other = self.pending[index]
            # Only an exactly matching ordered subexpression is shared.
            for positions in positions_iter:
                candidates = selected
                other_bridge_record = next(
                    (
                        item
                        for item in self.problem.sparse_bridges
                        if item.scene in (None, index)
                        and len(candidates) == 2
                        and (candidates[0].tensor, candidates[1].tensor) == (item.left, item.right)
                    ),
                    None,
                )
                other_bridge = _bridge_in_operand_order(other_bridge_record)
                if _bridge_identity(other_bridge) != _bridge_identity(bridge):
                    continue
                other_cycle_record = next(
                    (
                        item
                        for item in self.problem.sparse_bridges
                        if item is not other_bridge_record
                        and item.scene in (None, index)
                        and len(candidates) == 2
                        and {item.left, item.right} == {candidates[0].tensor, candidates[1].tensor}
                    ),
                    None,
                )
                other_cycle = _bridge_in_operand_order(other_cycle_record)
                if _bridge_identity(other_cycle) != _bridge_identity(cycle_bridge):
                    continue
                remaining = tuple(f for position, f in enumerate(other.factors) if position not in positions)
                other_needed = set(other.output + tuple(c for f in remaining for c in f.labels))
                for join in scene_joins.get(index, ()):
                    other_needed.update(join.spin_labels + join.mode_labels)
                if other_cycle is not None:
                    _, other_external, _ = cycle_intermediate_labels(
                        AxisPattern(tuple(f.labels for f in candidates), ()), other_bridge, other_cycle,
                        needed_labels=tuple(sorted(other_needed)),
                    )
                    other_labels = tuple(dict.fromkeys(c for c in other_external if c in other_needed))
                else:
                    other_labels = tuple(dict.fromkeys(c for f in candidates for c in f.labels if c in other_needed))
                other_joins_pending = bool(scene_joins.get(index))
                if not remaining and not other_joins_pending:
                    other_labels = other.output
                if other_labels != labels:
                    continue
                if self.problem.terms[index].sum_barriers or barriers:
                    try:
                        candidate_layout = self._contract_layout(index, tuple(positions), scene_joins=scene_joins)
                        candidate_action = self._describe_contract(index, tuple(positions), candidate_layout,
                            profile, workspace_bytes=workspace_bytes)[0]
                    except ValueError:
                        continue
                    if candidate_action.wide_accumulation != action.wide_accumulation:
                        continue
                updated = PendingTerm(remaining + (replacement,), other.output)
                joins_left = other_joins_pending
                if len(updated.factors) == 1 and updated.factors[0].labels == updated.output and not joins_left:
                    pending[index] = None
                    results[index] = key
                else:
                    pending[index] = updated
                if changes is not None:
                    changes.append((index, pending[index], key if pending[index] is None else None))
                break
        return self._seed_residency(replace(
            self,
            catalog=catalog,
            pending=tuple(pending),
            results=tuple(results),
            loaded=self.loaded | {key},
            seconds=self.seconds + seconds,
            peak_bytes=peak,
            peak_host_bytes=max(self.peak_host_bytes, peak if self.problem.backend == "numpy" else 0),
            peak_device_bytes=max(self.peak_device_bytes, peak if self.problem.backend == "cupy" else 0),
            actions=self.actions
            + (
                action,
            ),
        ), self._result_residency_delta(key, spec))

    def _join_layout(self, join: BaryonJoinSpec, *, scoped_bridges=None, scene_joins=None):
        """Resolve algebra and identity without residency, costing or transition."""
        if (join.scene, join.name) in self.completed_joins:
            raise ValueError("baryon join has already been applied")
        term = self.pending[join.scene]
        if term is None:
            raise ValueError("baryon join requires an unfinished scene")
        local = set(join.spin_labels + join.mode_labels)
        positions = tuple(i for i, factor in enumerate(term.factors) if local.intersection(factor.labels))
        selected = tuple(term.factors[i] for i in positions)
        if not selected:
            raise ValueError("baryon join requires quark legs")
        protected = {
            name for bridge in (self.problem.sparse_bridges if scoped_bridges is None else scoped_bridges)
            if bridge.scene in (None, join.scene)
            and {bridge.left, bridge.right} <= {factor.tensor for factor in term.factors}
            for name in (bridge.left, bridge.right)
        }
        if any(f.tensor in protected for f in selected):
            raise ValueError("P2P propagation legs must be bridged before baryon joining")
        if not local <= set(tuple(c for f in selected for c in f.labels)):
            raise ValueError("baryon join has an unbound quark slot")
        rest = tuple(f for i, f in enumerate(term.factors) if i not in positions)
        needed = set(term.output + tuple(c for f in rest for c in f.labels))
        for other in (self.problem.baryon_joins if scene_joins is None else scene_joins.get(join.scene, ())):
            if other.scene == join.scene and other != join and (other.scene, other.name) not in self.completed_joins:
                needed.update(other.spin_labels + other.mode_labels)
        output = tuple(dict.fromkeys(c for f in selected for c in f.labels if c not in local and c in needed))
        signature = (
            "baryon", join.name, join.spin, join.elemental, join.spin_labels,
            join.mode_labels, join.sparse_spin,
            tuple((f.tensor, f.labels) for f in selected), output,
        )
        key = "node:" + hashlib.sha256(repr(signature).encode()).hexdigest()[:24]
        return selected, rest, output, key

    def join_output_key(self, join: BaryonJoinSpec):
        """Compare shared join identities; callers must still validate the step."""
        return self._join_layout(join)[-1]

    def contract_join(
        self,
        join: BaryonJoinSpec,
        profile: CostProfile,
        limit_bytes: int,
        *,
        kernel_path_override: tuple[tuple[int, ...], ...] | None = None,
        search_kernel_path: bool = True,
    ):
        """Eliminate one baryon endpoint; the action position remains searchable."""
        if join.core is not None and (len(join.core.components) != 1 or join.core.components[0].coefficient != 1 or join.batch_label is not None):
            from .canonical_operations import describe, replay
            action, _, _, _, _ = describe(self, join.scene, join=join, profile=profile,
                                          kernel_path=kernel_path_override)
            return replay(self, action, profile, limit_bytes)
        return self._join_from_layout(join, self._join_layout(join), profile, limit_bytes,
            kernel_path_override=kernel_path_override, search_kernel_path=search_kernel_path)

    def _join_from_layout(self, join, layout, profile, limit_bytes, *,
                          kernel_path_override=None, search_kernel_path=True, workspace_bytes=None,
                          scene_joins=None):
        """Cost and build a join effect without resolving its layout again."""
        marker = (join.scene, join.name)
        if marker in self.completed_joins:
            raise ValueError("baryon join has already been applied")
        term = self.pending[join.scene]
        if term is None or join.spin not in self.loaded or join.elemental not in self.loaded:
            raise ValueError("baryon join data must be resident in an unfinished scene")
        selected, rest, output, key = layout
        if any(f.tensor not in self.loaded for f in selected):
            raise ValueError("baryon legs are not resident")
        dimensions = {}
        for factor in selected:
            for label, size in zip(factor.labels, self.catalog[factor.tensor].shape, strict=True):
                if label in dimensions and dimensions[label] != size:
                    raise ValueError("baryon leg dimensions disagree")
                dimensions[label] = size
        for labels, source in ((join.spin_labels, join.spin), (join.elemental_labels(self.catalog[join.elemental]), join.elemental)):
            spec = self.catalog[source]
            if len(spec.shape) != len(labels):
                raise ValueError("baryon input rank differs from its declared slots")
            for label, size in zip(labels, spec.shape, strict=True):
                if label in dimensions and dimensions[label] != size:
                    raise ValueError("baryon slot dimensions disagree")
                dimensions[label] = size
        spec = TensorSpec(
            tuple(dimensions[c] for c in output),
            np.result_type(
                self.catalog[join.spin].dtype,
                self.catalog[join.elemental].dtype,
                *(self.catalog[f.tensor].dtype for f in selected),
            ),
        )
        if key in self.catalog and key not in self.loaded:
            raise ValueError("released baryon intermediate cannot be recomputed")
        reused = key in self.loaded
        peak = max(self.peak_bytes, self.resident_bytes + (0 if reused else _nbytes(spec)))
        if peak > limit_bytes:
            raise MemoryError("baryon join exceeds declared memory cap")
        flops = math.prod(dimensions.values()) * max(len(selected), 1)
        traffic = _nbytes(spec) + sum(
            _nbytes(self.catalog[k]) for k in (join.spin, join.elemental, *(f.tensor for f in selected))
        )
        seconds = 0.0 if reused else profile.contract_seconds(flops, traffic, kind="baryon_join", dtype=spec.dtype)
        completed = self.completed_joins | {marker}
        updated = PendingTerm(rest + (Factor(key, output),), term.output)
        pending, results = self.pending, self.results
        if (
            len(updated.factors) == 1
            and updated.factors[0].labels == updated.output
            and not any(
                other.scene == join.scene and (other.scene, other.name) not in completed
                for other in (self.problem.baryon_joins if scene_joins is None else scene_joins.get(join.scene, ()))
            )
        ):
            pending = _replace_row(pending, join.scene, None)
            results = _replace_row(results, join.scene, key)
        else:
            pending = _replace_row(pending, join.scene, updated)
        catalog = self.catalog.updated(((key, spec),))
        equation = AxisPattern((join.spin_labels, join.elemental_labels(self.catalog[join.elemental]), *(f.labels for f in selected)), output)
        kernel_specs = (
            self.catalog[join.spin],
            self.catalog[join.elemental],
            *(self.catalog[f.tensor] for f in selected),
        )
        shapes = tuple(item.shape for item in kernel_specs)
        if join.kernel_description is not None:
            from lqcdweave.contraction.performance.api import estimate_vertex_kernel
            join_flops, traffic, estimated_workspace = estimate_vertex_kernel(
                join.kernel_description, equation, kernel_specs)
            kernel_path = ()
        elif not search_kernel_path:
            if kernel_path_override is not None:
                raise ValueError("a fixed direct join cannot override its kernel path")
            kernel_path, join_flops, largest = _direct_join_path(equation, shapes)
        elif kernel_path_override is None:
            kernel_path, join_flops, largest = _prepared_join_path(equation, shapes)
        else:
            kernel_path = tuple(tuple(step) for step in kernel_path_override)
            join_flops, largest = _join_path_cost(equation, shapes, kernel_path)
        workspace = 0 if reused else workspace_bytes if workspace_bytes is not None else (
            estimated_workspace if join.kernel_description is not None else max(
            largest * spec.dtype.itemsize + sum(_nbytes(item) for item in kernel_specs),
            _join_path_workspace(equation, shapes, kernel_path) * spec.dtype.itemsize,
        ))
        if not reused:
            peak = max(peak, self.resident_bytes + _nbytes(spec) + workspace)
        if peak > limit_bytes:
            raise MemoryError("baryon join workspace exceeds declared memory cap")
        seconds = 0.0 if reused else profile.contract_seconds(join_flops, traffic, kind="baryon_join", dtype=spec.dtype)
        action = Action(
            "baryon_join",
            key,
            (join.spin, join.elemental, *(f.tensor for f in selected)),
            equation,
            seconds,
            reused,
            join_id=join.name,
            term_index=join.scene,
            kernel_path=kernel_path,
            sparse_spin=join.sparse_spin,
            kernel_description=join.kernel_description,
        )
        return self._seed_residency(replace(
            self,
            catalog=catalog,
            pending=pending,
            results=results,
            completed_joins=completed,
            loaded=self.loaded | {key},
            seconds=self.seconds + seconds,
            peak_bytes=peak,
            peak_host_bytes=max(self.peak_host_bytes, peak if self.problem.backend == "numpy" else 0),
            peak_device_bytes=max(self.peak_device_bytes, peak if self.problem.backend == "cupy" else 0),
            actions=self.actions + (action,),
        ), self._result_residency_delta(key, spec))


def _matching_positions(factors, selected):
    """Find ordered occurrences without constructing unrelated combinations.

    Positions follow the same lexicographic order as combinations. A repeated
    factor keeps every occurrence; its labels are part of identity.
    """
    if len(selected) not in (1, 2):
        raise ValueError("matching positions requires a unary or binary contraction")
    first = [i for i, factor in enumerate(factors) if factor == selected[0]]
    if len(selected) == 1:
        for i in first:
            yield (i,)
    elif first:
        second = first if selected[0] == selected[1] else [
            i for i, factor in enumerate(factors) if factor == selected[1]
        ]
        for i in first:
            for j in second:
                if i < j:
                    yield (i, j)


def _nbytes(spec: TensorSpec) -> int:
    return math.prod(spec.shape) * spec.dtype.itemsize


def _direct_join_path(equation, shapes):
    """Describe a fixed spin/leg, elemental, remaining-leg order without search."""
    count = len(shapes)
    path = [(0, 2)]
    count -= 1
    while count > 1:
        path.append((0, count - 1))
        count -= 1
    path = tuple(path)
    return (path, *_join_path_cost(equation, shapes, path))


def _join_path_cost(equation, shapes, path):
    """Validate and cost an explicit elimination order without choosing a path."""
    labels, output = equation.inputs, equation.output
    dimensions = {
        label: size for term, shape in zip(labels, shapes, strict=True)
        for label, size in zip(term, shape, strict=True)
    }
    pending = [set(term) for term in labels]
    flops, largest = 0, 0
    for step in path:
        if (not step or len(set(step)) != len(step)
                or any(type(index) is not int or index < 0 or index >= len(pending)
                       for index in step)):
            raise ValueError("archived baryon kernel path is invalid")
        selected = set().union(*(pending[index] for index in step))
        remaining = [term for index, term in enumerate(pending) if index not in step]
        needed = set(output).union(*remaining)
        retained = selected & needed
        factor = max(1, len(step) - 1) + bool(selected - retained)
        flops += factor * math.prod(dimensions[label] for label in selected)
        largest = max(largest, math.prod(dimensions[label] for label in retained))
        pending = remaining + [retained]
    if len(pending) != 1 or pending[0] != set(output):
        raise ValueError("archived baryon kernel path is incomplete")
    return flops, largest


@lru_cache(maxsize=256)
def _join_path_workspace(equation, shapes, path):
    """Reserve live intermediates, worst-case operand packing and final copy.

    Original inputs are already resident; their possible contiguous packing
    copies belong to workspace. Generated operands may coexist with those
    copies and the next output until a fixed-path step finishes.
    """
    labels, output = equation.inputs, equation.output
    dimensions = {label: size for term, shape in zip(labels, shapes, strict=True)
                  for label, size in zip(term, shape, strict=True)}
    final_size = math.prod(dimensions[label] for label in output)
    pending = [(set(term), math.prod(shape), False)
               for term, shape in zip(labels, shapes, strict=True)]
    peak, largest = 2 * final_size, 0
    for step in path:
        selected = set().union(*(pending[index][0] for index in step))
        remaining = [item for index, item in enumerate(pending) if index not in step]
        needed = set(output).union(*(item[0] for item in remaining))
        retained = selected & needed
        size = math.prod(dimensions[label] for label in retained)
        live_generated = sum(item[1] for item in pending if item[2])
        operand_packs = sum(pending[index][1] for index in step)
        peak = max(peak, live_generated + operand_packs + size)
        largest = max(largest, size)
        pending = remaining + [(retained, size, True)]
    return max(largest + sum(math.prod(shape) for shape in shapes), peak - final_size)


@lru_cache(maxsize=256)
def _prepared_join_path(equation: AxisPattern, shapes: tuple[tuple[int, ...], ...]):
    dimensions = equation.dimensions(shapes)
    path = opt_einsum.paths.greedy([frozenset(axes) for axes in equation.inputs],
                                 frozenset(equation.output), dimensions, None)
    path = tuple(tuple(step) for step in path)
    return path, *_join_path_cost(equation, shapes, path)


def _bridge_identity(item: SparseBridgeSpec | None):
    if item is None:
        return None
    return (
        item.left,
        item.right,
        item.record,
        item.used_np,
        item.used_np_right,
        item.left_axes,
        item.right_axes,
        item.same_point,
        item.transpose_bridge,
        item.reverse_operands,
    )


def _bridge_in_operand_order(item: SparseBridgeSpec | None):
    """Orient a sparse bridge's endpoint axes to the term's factor order."""
    if item is None or not item.reverse_operands:
        return item
    return replace(
        item,
        left_axes=item.right_axes,
        right_axes=item.left_axes,
        transpose_bridge=not item.transpose_bridge,
        reverse_operands=False,
    )
