"""Bounded, process-shared dominance records for completed search work.

No live claims, futures, numerical arrays or routes enter this table. Eviction
only loses an optimization. A hit requires a completed traversal at no larger
prefix cost and an already retained feasible route in the same component.
"""
import hashlib
import os
import struct

from ..canonical_operations import computation_key


_ENTRY = struct.Struct('<32sdq')
_WAYS = 4


class SharedStates:
    def __init__(self, context, components, *, slots=262144):
        buckets = max(1, slots // _WAYS)
        self.buckets = 1 << (buckets.bit_length() - 1)
        self.data = context.RawArray('B', self.buckets * _WAYS * _ENTRY.size)
        self.locks = tuple(context.Lock() for _ in range(min(64, self.buckets)))
        self.feasible = context.Array('b', components)
        self.nbytes = len(self.data) + components + len(self.locks) * 256

    def client(self, component):
        return SharedStateClient(self, component)


class SharedStateClient:
    def __init__(self, table, component):
        self.table, self.component = table, component
        self.recipes = {}
        self.lookups = self.hits = self.published = self.evictions = 0
        self.cross_process_hits = 0
        self._feasible = False

    def mark_feasible(self):
        with self.table.feasible.get_lock():
            self.table.feasible[self.component] = 1
        self._feasible = True

    def key(self, space, operations):
        recipes = []
        for op in operations:
            identity = (op.calculation, op.reused)
            value = self.recipes.get(identity)
            if value is None:
                calc = space.registry.calculations[op.calculation]
                action = calc.recipe
                value = hashlib.blake2b(repr((computation_key(action), action.tensor,
                    calc.seconds, calc.workspace, action.kernel_path, op.reused)).encode(),
                    digest_size=32).digest()
                self.recipes[identity] = value
            recipes.append(value)
        # Boundary ties are all retained in shared workers; their order does
        # not change the completed set. Keep the actual selected recipes in
        # the identity, including fixed-root or deferred-only restrictions.
        return hashlib.blake2b(self.component.to_bytes(8, 'little') +
            space.shared_fingerprint() + repr(tuple(sorted(space.recent_scenes))).encode() +
            b''.join(sorted(recipes)), digest_size=32).digest()

    def _bucket(self, key):
        return int.from_bytes(key[:8], 'little') & (self.table.buckets - 1)

    def dominated(self, key, cost):
        self.lookups += 1
        if not self._feasible:
            self._feasible = bool(self.table.feasible[self.component])
        if not self._feasible:
            return False
        bucket = self._bucket(key)
        with self.table.locks[bucket % len(self.table.locks)]:
            for slot in range(bucket * _WAYS, (bucket + 1) * _WAYS):
                saved, bound, owner = _ENTRY.unpack_from(self.table.data, slot * _ENTRY.size)
                if owner and saved == key and bound <= cost:
                    self.hits += 1
                    self.cross_process_hits += owner != os.getpid()
                    return True
        return False

    def publish(self, key, cost):
        if not self._feasible:
            self._feasible = bool(self.table.feasible[self.component])
            if not self._feasible:
                return
        bucket = self._bucket(key)
        with self.table.locks[bucket % len(self.table.locks)]:
            empty = None
            for slot in range(bucket * _WAYS, (bucket + 1) * _WAYS):
                saved, bound, valid = _ENTRY.unpack_from(self.table.data, slot * _ENTRY.size)
                if valid and saved == key:
                    if bound <= cost:
                        return
                    empty = slot
                    break
                if not valid:
                    empty = slot
            if empty is None:
                # Fixed-way deterministic replacement, never a global lock.
                empty = bucket * _WAYS + (int.from_bytes(key[8:12], 'little') % _WAYS)
                self.evictions += 1
            _ENTRY.pack_into(self.table.data, empty * _ENTRY.size, key, cost, os.getpid())
            self.published += 1

    def statistics(self):
        return dict(shared_state_lookups=self.lookups, shared_state_hits=self.hits,
                    shared_state_cross_process_hits=self.cross_process_hits,
                    shared_state_publications=self.published, shared_state_evictions=self.evictions)
