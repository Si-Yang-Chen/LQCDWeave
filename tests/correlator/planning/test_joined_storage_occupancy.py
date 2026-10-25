"""Independent lifetime intervals and bounded work for joined raw occupancy."""
from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import StorageBlock
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, join_storage_batches, plan_scene_storage,
)
from lqcdweave.contraction.execution.api import replay_storage_batches
from test_scene_storage_batches import batch, Loader


def sized_batch(key, host_bytes, device_bytes=0, *, backend='numpy'):
    original = batch(key)
    profile = CostProfile()
    problem = replace(original.state.problem, backend=backend)
    state = ContractionState.initial(problem)
    state = state.load(key, profile, 10**7).contract(0, (0,), profile, 10**7).release(key)
    blocks = {key: replace(original.blocks[key], host_bytes=host_bytes,
        device_bytes=device_bytes, staging_host_bytes=17,
        staging_device_bytes=19 if backend == 'cupy' else 0)}
    return plan_scene_storage(state, blocks, original.preparations, profile=profile,
        memory_limit=10**7, host_memory_limit=10**7)


class JoinedStorageOccupancyTests(unittest.TestCase):
    def test_every_point_matches_independent_closed_lifetime_intervals(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                sizes = dict(a=(53, 33), b=(79, 67), c=(101, 101), d=(149, 163))
                templates = {key: sized_batch(key, host, device if backend == 'cupy' else 0,
                    backend=backend) for key, (host, device) in sizes.items()}
                plans = tuple(templates[key] for key in ('a', 'c', 'b', 'a', 'd', 'b', 'c', 'd'))
                blocks = {key: block for plan in plans for key, block in plan.blocks.items()}
                first, last = {}, {}
                for index, plan in enumerate(plans):
                    for offset, point in enumerate(plan.memory):
                        if point.phase.startswith('read:'):
                            first.setdefault(point.phase[5:], (index, point.action_index, offset))
                    for event in plan.events:
                        if event.kind == 'release':
                            last[event.block] = (index, event.action_index, len(plan.memory))
                joined = join_storage_batches(plans)
                for index, (original, current) in enumerate(zip(plans, joined, strict=True)):
                    positions = [(offset, point) for offset, point in enumerate(original.memory)
                        if not point.phase.startswith('read:') or
                        first[point.phase[5:]] == (index, point.action_index, offset)]
                    self.assertIs(current.state, original.state)
                    self.assertEqual(current.seconds, original.seconds)
                    for (offset, before), actual in zip(positions, current.memory, strict=True):
                        position = (index, before.action_index, offset)
                        active = [key for key in blocks if first[key] <= position <= last[key]]
                        self.assertEqual((actual.action_index, actual.phase),
                                         (before.action_index, before.phase))
                        self.assertEqual(actual.raw_host_bytes,
                                         sum(blocks[key].host_bytes for key in active))
                        self.assertEqual(actual.raw_device_bytes,
                                         sum(blocks[key].device_bytes for key in active))
                    self.assertTrue(current.feasible)
                events = [(event.kind, event.block) for plan in joined for event in plan.events]
                self.assertEqual(Counter(events), Counter(
                    (kind, key) for key in blocks for kind in ('read', 'release')))

    def test_raw_byte_lookups_do_not_scale_with_points_times_live_blocks(self):
        keys = tuple(f'raw-{index}' for index in range(64))
        plans = tuple(batch(key) for key in keys) * 2
        lookups = Counter()

        def counted_getattribute(block, name):
            if name in ('host_bytes', 'device_bytes'):
                lookups[name] += 1
            return object.__getattribute__(block, name)

        with patch.object(StorageBlock, '__getattribute__', counted_getattribute):
            joined = join_storage_batches(plans)
        # Includes ordinary descriptor validation. The bound allows that fixed
        # per-batch work but rejects a registry scan at every memory point.
        self.assertLess(sum(lookups.values()), 50 * len(plans))
        loader, values = Loader(), []
        replay_storage_batches(joined, loader, lambda _index, result: values.append(result[0]))
        self.assertEqual(values, [6.] * len(plans))
        self.assertEqual(loader.reads, list(keys))
        self.assertEqual(loader.closes, list(keys))
