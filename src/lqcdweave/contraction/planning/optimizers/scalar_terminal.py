"""Incremental cache identity for complete independent scalar tails.

Only terminal computations and their own cache availability affect this key.
Completed rows and unrelated cached intermediates cannot change their cost.

The published tail is one ordered fold: ascending block keys, ascending
calculation identities inside a block, and the exact running sum the reference
has after that block. A state therefore keeps every block it did not change
and re-walks only the blocks after the first change, which reproduces the
reference's left-to-right additions bit for bit.
"""
import math
from bisect import bisect_left


class ScalarTerminalIndex:
    # Re-openable representation check: rebuilding the carried block overrides
    # from the retained base on every call proves the modification tracker.
    verify_overrides = False

    def __init__(self):
        self.parts = {}
        # Reuse accounting is reported for the whole index lifetime, so reset()
        # (a fresh search on the same workspace) does not clear it.
        self.fold_steps = self.fold_reuses = self.block_rebuilds = 0
        self.reset()

    def reset(self):
        self.entries = {}
        self.digest = self.coverage = 0
        self.buckets, self.dirty_buckets = {}, set()
        self.summaries = {}
        # Ordered fold state, one slot per retained block: the ascending block
        # key, the identity tuple that block publishes, and the exact running
        # sum after it. Every slot before `restart` is proven unchanged since
        # it was folded, so only slots from there on have to be re-walked.
        self.order, self.chunks, self.running = [], [], []
        self.size = 0
        self.seconds = 0.0
        self.restart = 0
        # Carried override list: the ascending (position, block) pairs in which
        # `chunks` differs from the retained sequence `diff_base`. Sharing a
        # tail only needs those positions, so an unchanged block never has to
        # be compared again.
        self.diffs, self.diff_base = None, None

    def clear(self):
        self.parts.clear()
        self.reset()

    def _part(self, identity, family, space):
        result = space.registry.calculations[identity].result
        key = identity, family, result in space.available
        value = self.parts.get(key)
        if value is None:
            value = space._part(('scalar-terminal', *key))
            self.parts[key] = value
        return value

    def set(self, identity, family, space):
        self.remove(identity)
        part = self._part(identity, family, space)
        count = len(space.groups[identity])
        self.entries[identity] = family, part, count
        self.digest ^= part
        self.coverage += count
        self.buckets.setdefault(identity // 64, set()).add(identity)
        self.dirty_buckets.add(identity // 64)

    def remove(self, identity):
        old = self.entries.pop(identity, None)
        if old is not None:
            self.digest ^= old[1]
            self.coverage -= old[2]
            bucket = identity // 64
            self.buckets[bucket].remove(identity)
            if not self.buckets[bucket]:
                del self.buckets[bucket]
            self.dirty_buckets.add(bucket)

    def availability(self, identity, space):
        old = self.entries.get(identity)
        if old is not None:
            part = self._part(identity, old[0], space)
            self.digest ^= old[1] ^ part
            self.entries[identity] = old[0], part, old[2]
            self.dirty_buckets.add(identity // 64)

    def _blocks(self, space, checkpoint):
        """Rebuild changed blocks in place and mark the first stale fold slot.

        Each block keeps the reference order: it is rebuilt, inserted or
        dropped exactly where its own ascending key places it. Only the slot a
        changed or moved block occupies can invalidate the retained prefix, so
        the first such slot is remembered for the fold.
        """
        order, chunks, running = self.order, self.chunks, self.running
        for bucket in sorted(self.dirty_buckets):
            checkpoint()
            identities = self.buckets.get(bucket)
            index = bisect_left(order, bucket)
            if identities is None:
                self.summaries.pop(bucket, None)
                if index < len(order) and order[index] == bucket:
                    del order[index]
                    del chunks[index]
                    del running[index]
                    # Positions after a departed block move: the carried
                    # overrides describe the old indexing and are dropped.
                    self.diffs = None
                    if index < self.restart:
                        self.restart = index
            else:
                values = tuple(sorted(identities))
                costs = tuple(0.0 if space.registry.calculations[i].result in space.available
                              else space.registry.calculations[i].seconds for i in values)
                self.summaries[bucket] = (values, costs, None, None)
                if index < len(order) and order[index] == bucket:
                    self.size += len(values) - len(chunks[index])
                    chunks[index] = values
                    self._carry(index, values)
                else:
                    order.insert(index, bucket)
                    chunks.insert(index, values)
                    running.insert(index, 0.0)
                    # A new block shifts every later position, so the carried
                    # overrides are dropped and rebuilt on the next request.
                    self.diffs = None
                    self.size += len(values)
                if index < self.restart:
                    self.restart = index
            self.block_rebuilds += 1
            self.dirty_buckets.remove(bucket)

    def _carry(self, index, chunk):
        """Fold one rebuilt block into the carried override list.

        The list stays ascending and holds exactly the positions whose block no
        longer is the retained base's block, so a rebuilt block that reproduces
        its base entry leaves nothing behind.
        """
        diffs = self.diffs
        if diffs is None:
            return
        base = self.diff_base
        changed = chunk is not base[index]
        for slot, (position, _) in enumerate(diffs):
            if position == index:
                if changed:
                    diffs[slot] = (index, chunk)
                else:
                    del diffs[slot]
                return
            if position > index:
                if changed:
                    diffs.insert(slot, (index, chunk))
                return
        if changed:
            diffs.append((index, chunk))

    def overrides(self, base):
        """Ascending block overrides against a retained base, or None to rescan.

        Sharing a tail sequence only needs the positions where it differs from
        the retained base. Those positions are carried across calls instead of
        re-walking the whole sequence, and any structural change drops the
        carried list so the next request rebuilds it against the base it is
        asked about.
        """
        chunks, diffs = self.chunks, self.diffs
        if base is None or len(base) != len(chunks):
            self.diffs = None
            return None
        if diffs is None or self.diff_base is not base:
            self.diff_base = base
            self.diffs = diffs = [(position, chunk) for position, chunk in enumerate(chunks)
                                  if chunk is not base[position]]
        elif self.verify_overrides:
            walked = [(position, chunk) for position, chunk in enumerate(chunks)
                      if chunk is not base[position]]
            if walked != diffs:
                raise RuntimeError('carried terminal block overrides diverged from the retained base')
        return diffs

    def completion(self, space, checkpoint):
        """Reuse the unchanged ordered prefix of the previous summary.

        key() proves disjoint, single-recipe scalar families with full coverage.
        Their family IDs equal calculation IDs; no output normalizers exist.
        Each block retains the exact left-to-right additions of the reference,
        including its incoming sum, so a changed prefix never reuses an
        incompatible rounded sum. Published completions own immutable tuples.
        """
        from .terminal_completion import CalculationChunks, TerminalCompletion

        self._blocks(space, checkpoint)
        order, running = self.order, self.running
        # The restart point is committed before the fold: a budget stop inside
        # the walk must leave the next call resuming at or before this slot.
        start = self.restart
        end = len(order)
        if start < end:
            seconds = running[start - 1] if start else 0.0
            for index in range(start, end):
                checkpoint()
                bucket = order[index]
                values, costs, incoming, outgoing = self.summaries[bucket]
                if incoming == seconds:
                    seconds = outgoing
                else:
                    incoming = seconds
                    for cost in costs:
                        if not math.isfinite(cost) or cost < 0:
                            raise ValueError('terminal increments must be finite and nonnegative')
                        seconds += cost
                    self.summaries[bucket] = values, costs, incoming, seconds
                running[index] = seconds
            self.seconds = seconds
        elif start:
            # Only trailing blocks were dropped; the surviving prefix is the
            # whole ordered sum.
            self.seconds = running[start - 1]
        else:
            self.seconds = 0.0
        self.restart = end
        self.fold_steps += end - start
        self.fold_reuses += start
        return TerminalCompletion(self.seconds, CalculationChunks(tuple(self.chunks), self.size))

    def statistics(self):
        steps, reuses = self.fold_steps, self.fold_reuses
        return dict(scalar_terminal_fold_steps=steps, scalar_terminal_prefix_reuses=reuses,
                    scalar_terminal_block_rebuilds=self.block_rebuilds,
                    scalar_terminal_prefix_reuse_rate=round(reuses / (steps + reuses), 6)
                    if steps + reuses else None)

    def key(self, space):
        # Full coverage excludes active rows having no legal candidate. Pure
        # scalar terminal rows consume every factor, so they have one pair or
        # one unary candidate. Excluding alternate recipes and family variants
        # then guarantees disjoint consumers and independent result identities.
        if (space.dirty or space.registry.has_alternative_calculations
                or self.coverage != space.remaining
                or len(space.groups) != len(space.terminal_families)
                or len(space.terminal_members) != len(space.terminal_families)):
            return None
        return 'scalar', self.digest
