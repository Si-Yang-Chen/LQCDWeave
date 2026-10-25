"""Independent owner-sized allocation checks require no CUDA installation."""
from dataclasses import replace
import sys
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionProblem, ContractionTerm, DenseBatchOperand,
    DenseBatchResult, DenseKernelBatch, SourcePreparation, SourceSpec,
    StorageBlock, TensorPreparation, TensorSpec, device_allocation_bytes,
    dense_batch_memory,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, join_storage_batches, plan_scene_storage,
)
from lqcdweave.contraction.planning.allocation_rounding import account_allocation_rounding
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


def fixture(*, backend='cupy', duplicates=False, two_outputs=False, cap=3072, raw_prefix=''):
    shapes = {'a': (2, 3), 'b': (3, 2)}
    if two_outputs:
        shapes['a1'] = (2, 3)
    specs = {key: TensorSpec(shape, 'complex64') for key, shape in shapes.items()}
    pattern = AxisPattern(((0, 1), (1, 2)), (0, 2))
    terms = [ContractionTerm(('a', 'b'), pattern)]
    if duplicates or two_outputs:
        terms.append(ContractionTerm(('a1' if two_outputs else 'a', 'b'), pattern))
    problem = ContractionProblem(specs, tuple(terms),
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=backend)
    profile = CostProfile()
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 10**8)
    for index in range(len(terms)):
        if state.pending[index] is not None:
            state = state.contract(index, (0, 1), profile, 10**8)
    blocks = {raw_prefix+key: StorageBlock('fixture', raw_prefix+key, spec.shape[0]*spec.shape[1]*8,
        host_bytes=spec.shape[0]*spec.shape[1]*8,
        device_bytes=spec.shape[0]*spec.shape[1]*8 if backend == 'cupy' else 0)
        for key, spec in specs.items()}
    preps = {key: SourcePreparation((raw_prefix+key,),
             recipe=TensorPreparation('identity', (raw_prefix+key,)))
             for key in specs}
    plan = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=cap)
    # Tests inspect allocator rounding rather than the optional packing cache.
    memory = tuple(replace(point, packed_value_cache_device_bytes=0) for point in plan.memory)
    return replace(plan, memory=memory, packed_value_cache_device_bytes=0,
                   peak_device_bytes=max(point.device_bytes for point in memory))


class AllocationRoundingTests(unittest.TestCase):
    def test_one_owner_quantum_and_invalid_bytes(self):
        self.assertEqual([device_allocation_bytes(size) for size in (0, 1, 128, 512, 513)],
                         [0, 512, 512, 512, 1024])
        for size in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                device_allocation_bytes(size)

    def test_independent_allocator_reference_for_dense_workspace(self):
        before = 'cupy' in sys.modules
        plan = fixture()
        point = next(point for point in plan.memory if point.phase == 'contract')
        # At this point raw reads are closed. Three live tensors and three
        # permitted copy slots are six separate allocations, not one sum.
        requests = (48, 48, 32, 48, 48, 32)
        actual = sum(((size-1)//512+1)*512 for size in requests)
        self.assertEqual(actual, 3072)
        self.assertEqual(point.device_bytes, actual)
        self.assertTrue(plan.feasible)
        self.assertEqual('cupy' in sys.modules, before)

    def test_reject_cap_one_byte_below_rounded_peak(self):
        plan = fixture(cap=3071)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.peak_device_bytes, 3072)

    def test_cpu_actions_costs_and_values_are_unchanged(self):
        cpu = fixture(backend='numpy', cap=10**8)
        gpu = fixture()
        self.assertEqual(cpu.state.actions, gpu.state.actions)
        self.assertEqual(cpu.seconds, gpu.seconds)
        self.assertTrue(all(point.allocation_rounding_device_bytes == 0 for point in cpu.memory))
        self.assertIs(account_allocation_rounding(cpu), cpu)
        arrays = {'a': np.arange(6, dtype=np.complex64).reshape(2, 3),
                  'b': np.arange(6, dtype=np.complex64).reshape(3, 2)}
        result = np.einsum('ij,jk->ik', arrays['a'], arrays['b'])
        np.testing.assert_array_equal(result, arrays['a'] @ arrays['b'])

    def test_duplicate_terms_share_result_and_snapshot_owners(self):
        once, twice = fixture(), fixture(duplicates=True)
        self.assertEqual(len(set(twice.state.results)), 1)
        self.assertEqual(once.memory[-1].allocation_rounding_device_bytes,
                         twice.memory[-1].allocation_rounding_device_bytes)

    def test_packed_kernel_output_counts_one_owner_for_two_views(self):
        plain = fixture(two_outputs=True, cap=10**8)
        paid = tuple(index for index, action in enumerate(plain.state.actions)
                     if action.kind == 'contract' and not action.reused)
        keys = plain.state.results
        batch = DenseKernelBatch(paid,
            (DenseBatchOperand('a', (0, 1)), DenseBatchOperand('a1', (0, 1))),
            (DenseBatchOperand('b', (0, 1)),), 1, 1, 1, 0, 1,
            (DenseBatchResult(keys[0], ((0, 2), (0, 2))),
             DenseBatchResult(keys[1], ((2, 4), (0, 2)))),
            0., 96, 64, 'complex64')
        packed = replace(plain, kernel_batches=(batch,))
        packed = replace(packed, memory=dense_batch_memory(packed))
        packed = account_allocation_rounding(packed)
        # Two 32-byte outputs round to 1024 independently. One 64-byte owner
        # rounds to 512. Snapshot groups are identical in both cases.
        self.assertEqual(plain.memory[-1].allocation_rounding_device_bytes
                         - packed.memory[-1].allocation_rounding_device_bytes, 512)

    def test_cross_batch_raw_owner_remains_charged_until_final_use(self):
        first = fixture(cap=10**8)
        joined = join_storage_batches((first, first))
        start = joined[1].memory[0]
        original = next(point for point in first.memory if point.phase == start.phase)
        self.assertEqual(start.allocation_rounding_device_bytes
                         - original.allocation_rounding_device_bytes, 512-48)

    def test_idempotence_and_archive_round_trip(self):
        plan = fixture()
        again = account_allocation_rounding(plan)
        self.assertEqual(again, plan)
        point = next(point for point in plan.memory if point.phase == 'contract')
        self.assertEqual(decode_plan_value(encode_plan_value(point)), point)

    def test_shared_raw_can_skip_an_intermediate_batch(self):
        first = fixture(cap=10**8)
        middle = fixture(cap=10**8, raw_prefix='middle:')
        joined = join_storage_batches((first, middle, first))
        start = joined[1].memory[0]
        self.assertEqual(start.allocation_rounding_device_bytes
                         - middle.memory[0].allocation_rounding_device_bytes, 2*(512-48))


if __name__ == '__main__':
    unittest.main()
