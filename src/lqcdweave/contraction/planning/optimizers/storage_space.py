"""Complete physical storage state for exact contraction planning.

The incremental memory accounting was adapted from storage_anytime.advance.
Exact strategies share logical eviction/rematerialization and explicit
choices to retain or close raw blocks. Closing a block forbids a
second read, matching the existing SceneStoragePlan execution contract.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, replace
import math
import time

from ..actions import _required_loads
from ..paths import _contract_successors, _state_key
from ..state import ContractionState, _nbytes
from ..storage import logical_cost_profile, logical_source_problem, plan_scene_storage
from ...performance.api import CostProfile
from .results import CandidateMemoryExceeded
from .budget import DEFAULT_PLANNING_SECONDS


@dataclass(frozen=True)
class _StorageState:
    state: ContractionState
    resident: frozenset = frozenset()
    opened: frozenset = frozenset()
    extra_seconds: float = 0.0
    peak_host: int = 0
    peak_device: int = 0


class StorageSearchSpace:
    """One task, one fixed operation model, minimum compute-model cost.

    Source materialization is an atomic move with just-in-time block reads in
    the same stable order as plan_scene_storage. Raw blocks can be retained for
    later rematerialization or closed after borrowed references are released.
    The space contains no arrays. Adding operations changes this domain adapter,
    not the exact search engines; its key/layer must reflect new future behavior.
    """
    def __init__(self, problem, blocks, preparations, *, memory_limit, profile=None,
                 host_memory_limit=None, reserved_host_bytes=0, reserved_device_bytes=0):
        self.profile = CostProfile() if profile is None else profile
        if type(memory_limit) is not int or memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        if host_memory_limit is not None and (type(host_memory_limit) is not int or host_memory_limit <= 0):
            raise ValueError("host_memory_limit must be positive")
        self.problem = logical_source_problem(problem)
        self.logical_profile = logical_cost_profile(self.profile)
        self.blocks, self.preparations = dict(blocks), dict(preparations)
        self.memory_limit, self.host_memory_limit = memory_limit, host_memory_limit
        self.reserved_host_bytes, self.reserved_device_bytes = reserved_host_bytes, reserved_device_bytes
        self.gpu = problem.backend == "cupy"
        self.rejected_memory = 0
        if any(type(value) is not int or value < 0 for value in (reserved_host_bytes, reserved_device_bytes)):
            raise ValueError("reserved memory must be nonnegative integer bytes")
        if set(self.preparations) != set(problem.sources):
            raise ValueError("preparations must cover exactly the logical sources")
        if any(set(prep.blocks) - self.blocks.keys() for prep in self.preparations.values()):
            raise ValueError("unknown physical block dependency")
        if len({(b.dataset, b.selection) for b in self.blocks.values()}) != len(self.blocks):
            raise ValueError("one physical selection must have one block key")
        if not self.gpu and (reserved_device_bytes or
            any(b.device_bytes or b.staging_device_bytes for b in self.blocks.values()) or
            any(p.workspace_device_bytes for p in self.preparations.values())):
            raise ValueError("CPU plans cannot declare device storage")

    @property
    def allocation_limit(self):
        """Necessary per-operation cap; full residency is audited separately."""
        limit = self.memory_limit-(self.reserved_device_bytes if self.gpu else self.reserved_host_bytes)
        if not self.gpu and self.host_memory_limit is not None:
            limit = min(limit, self.host_memory_limit-self.reserved_host_bytes)
        return limit

    def _point(self, cursor, *, host_work=0, device_work=0, operation_peak=False):
        state = cursor.state
        host = (state.peak_host_bytes if operation_peak else state.resident_host_bytes)
        device = (state.peak_device_bytes if operation_peak else state.resident_bytes if self.gpu else 0)
        host += sum(self.blocks[k].host_bytes for k in cursor.resident) + self.reserved_host_bytes + host_work
        device += sum(self.blocks[k].device_bytes for k in cursor.resident) + self.reserved_device_bytes + device_work
        return replace(cursor, peak_host=max(cursor.peak_host, host), peak_device=max(cursor.peak_device, device))

    def initial_state(self):
        return self._point(_StorageState(ContractionState.initial(self.problem)))

    def key(self, cursor):
        return _state_key(cursor.state), cursor.resident, cursor.opened

    def cost(self, cursor):
        return cursor.state.seconds + cursor.extra_seconds

    def is_feasible(self, cursor):
        return (cursor.peak_device if self.gpu else cursor.peak_host) <= self.memory_limit and (
            self.host_memory_limit is None or cursor.peak_host <= self.host_memory_limit)

    def layer(self, cursor):
        state = cursor.state
        factors = sum(len(term.factors) for term in state.pending if term is not None)
        joins = sum(1 for join in self.problem.baryon_joins if state.pending[join.scene] is not None
                    and (join.scene, join.name) not in state.completed_joins)
        return factors + joins

    def _load(self, cursor, tensor):
        prep = self.preparations[tensor]
        for key in sorted(set(prep.blocks) - cursor.resident):
            if key in cursor.opened:
                return None
            block = self.blocks[key]
            cursor = replace(cursor, resident=cursor.resident | {key}, opened=cursor.opened | {key})
            cursor = self._point(cursor, host_work=block.staging_host_bytes,
                                 device_work=block.staging_device_bytes)
            if not self.is_feasible(cursor):
                self.rejected_memory += 1
                return None
        state = cursor.state.load(tensor, self.logical_profile, self.memory_limit)
        cursor = replace(cursor, state=state, extra_seconds=cursor.extra_seconds + prep.seconds(self.profile))
        return self._point(cursor, host_work=prep.workspace_host_bytes, device_work=prep.workspace_device_bytes)

    def successors(self, cursor):
        # Physical peaks are cumulative; the logical peak below describes only
        # this step so it can be combined with the CURRENT raw-block residency.
        before = replace(cursor.state, peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
        base = replace(cursor, state=before)
        for tensor in sorted((_required_loads(before) & before.problem.sources.keys()) - before.loaded):
            with suppress(MemoryError):
                candidate = self._load(base, tensor)
                if candidate is not None:
                    yield candidate
        for state in _contract_successors(before, self.logical_profile, self.memory_limit, unique_terms=True):
            yield self._point(replace(base, state=state), operation_peak=True)
        for tensor in sorted(before.loaded - set(before.results)):
            with suppress(ValueError):
                yield replace(base, state=before.release(tensor))
        borrowed = {block for source in before.loaded - before.catalog.keys()
                    for block in self.preparations[source].blocks}
        for block in sorted(cursor.resident - borrowed):
            yield replace(base, resident=cursor.resident - {block})

    def finish(self, cursor):
        if not cursor.state.done:
            return None
        copies = sum(_nbytes(cursor.state.catalog[k]) for k in cursor.state.results)
        return self._point(cursor, host_work=0 if self.gpu else copies,
                           device_work=copies if self.gpu else 0)


    def close_block(self, cursor, block):
        borrowed = {k for s in cursor.state.loaded - cursor.state.catalog.keys()
                    for k in self.preparations[s].blocks}
        if block not in cursor.resident or block in borrowed:
            raise ValueError("cannot close an absent or borrowed raw block")
        return replace(cursor, resident=cursor.resident - {block})

    def plan(self, cursor):
        if cursor.state.problem != self.problem:
            raise ValueError("candidate belongs to a different physical planning problem")
        plan = plan_scene_storage(
            cursor.state, self.blocks, self.preparations, profile=self.profile,
            memory_limit=self.memory_limit, host_memory_limit=self.host_memory_limit,
            reserved_host_bytes=self.reserved_host_bytes, reserved_device_bytes=self.reserved_device_bytes)
        if not plan.feasible:
            raise CandidateMemoryExceeded("selected storage path exceeds its final workspace budget")
        if not math.isclose(plan.seconds, self.cost(cursor), rel_tol=1e-11, abs_tol=1e-14):
            raise RuntimeError("exact storage search disagrees with independent plan audit")
        return plan


def solve_storage_exact(problem, blocks, preparations, *, memory_limit, profile=None,
                        host_memory_limit=None, reserved_host_bytes=0, reserved_device_bytes=0,
                        optimizer="exact", max_states=1_000_000, deadline_seconds=DEFAULT_PLANNING_SECONDS,
                        clock=time.monotonic):
    """Both exact algorithms minimize SceneStoragePlan.seconds under both caps."""
    from .session import solve_storage
    if optimizer not in ("exact", "dynamic_programming"):
        raise ValueError("storage exact optimizer must be exact or dynamic_programming")
    return solve_storage(problem, blocks, preparations, memory_limit=memory_limit, profile=profile,
                         host_memory_limit=host_memory_limit, reserved_host_bytes=reserved_host_bytes,
                         reserved_device_bytes=reserved_device_bytes, optimizer=optimizer,
                         max_states=max_states, deadline_seconds=deadline_seconds, clock=clock)


__all__ = ["StorageSearchSpace", "solve_storage_exact"]
