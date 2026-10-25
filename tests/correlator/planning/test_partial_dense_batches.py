"""Incomplete input-pair graphs must compute exactly their requested edges."""
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, ContractionProblem, ContractionTerm, DenseBatchOperand,
    SourcePreparation, SourceSpec, TensorSpec, validate_executable_resources,
)
from lqcdweave.contraction.execution.operations.dense_batch import contract_dense_batch
from lqcdweave.contraction.planning import dense_batches
from lqcdweave.contraction.planning.api import (
    CostProfile, lower_dense_batches, plan_scene_storage, solve_storage,
)
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value


PAIRS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3),
         (2, 0), (2, 2), (2, 3), (3, 4))


def scalar_state(backend, left_dtype=np.complex128, right_dtype=np.complex128):
    catalog = {**{f'a{i}': TensorSpec((2, 3, 4), left_dtype) for i in range(4)},
               **{f'b{j}': TensorSpec((4, 2, 3), right_dtype) for j in range(5)}}
    actions = []
    for i, j in PAIRS:
        key = f'c{i}{j}'
        catalog[key] = TensorSpec((), np.result_type(left_dtype, right_dtype))
        actions.append(Action('contract', key, (f'a{i}', f'b{j}'),
            AxisPattern(((0, 1, 2), (2, 0, 1)), ()), seconds=1.0))
    return SimpleNamespace(actions=tuple(actions), catalog=catalog,
        results=tuple(action.tensor for action in actions), problem=SimpleNamespace(backend=backend))


class PartialDenseBatchTests(unittest.TestCase):
    def test_sparse_pair_graph_matches_independent_complex_sum_without_missing_outputs(self):
        rng = np.random.default_rng(719)
        for backend in ('numpy', 'cupy'):
            for right_dtype in (np.complex64, np.complex128):
                with self.subTest(backend=backend, right_dtype=right_dtype):
                    state = scalar_state(backend, left_dtype=right_dtype, right_dtype=right_dtype)
                    arrays = {key: (rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape))
                              .astype(spec.dtype)[::-1]
                              for key, spec in state.catalog.items() if spec.shape}
                    saved = {key: value.copy() for key, value in arrays.items()}
                    expected = {f'c{i}{j}': sum(
                        arrays[f'a{i}'][x, y, z] * arrays[f'b{j}'][z, x, y]
                        for x in range(2) for y in range(3) for z in range(4)) for i, j in PAIRS}
                    batches = dense_batches.find_dense_batches(state, CostProfile())
                    self.assertTrue(batches)
                    self.assertEqual(batches, dense_batches.find_dense_batches(state, CostProfile()))
                    covered = set()
                    for batch in batches:
                        self.assertFalse(covered.intersection(batch.action_indices))
                        covered.update(batch.action_indices)
                        self.assertEqual(len(batch.action_indices), len(batch.left)*len(batch.right))
                        # Execute both saved layouts on CPU; this does not load CuPy.
                        contract_dense_batch(decode_plan_value(encode_plan_value(batch)), arrays, backend=np)
                    for index, action in enumerate(state.actions):
                        if index not in covered:
                            arrays[action.tensor] = expected[action.tensor]
                    self.assertEqual(set(arrays)-set(saved), set(expected))
                    tolerance = 5e-6 if right_dtype == np.complex64 else 2e-12
                    for key, value in expected.items():
                        np.testing.assert_allclose(arrays[key], value, rtol=tolerance, atol=tolerance)
                    for key, value in saved.items():
                        np.testing.assert_array_equal(arrays[key], value)
                    self.assertLess(len(batches)+len(state.actions)-len(covered), len(state.actions))
                    expensive = replace(CostProfile(), traffic_seconds_per_byte=1.0)
                    self.assertEqual(dense_batches.find_dense_batches(state, expensive), ())

    def test_partition_preserves_permutations_duplicates_self_edges_and_random_missing_pairs(self):
        rng = random.Random(129)
        for trial in range(60):
            vertices = [DenseBatchOperand(f'v{i}', permutation)
                        for i in range(9) for permutation in ((0, 1), (1, 0))]
            edges = [(left, right) for left in vertices[:10] for right in vertices[8:]
                     if rng.random() < .35]
            if edges:
                edges.append(edges[0])
            entries = [(index, None, ((left, right),)) for index, (left, right) in enumerate(edges)]
            groups = dense_batches._partial_grid_action_groups(entries)
            self.assertEqual(groups, dense_batches._partial_grid_action_groups(entries))
            covered = set()
            for group in groups:
                self.assertEqual(tuple(sorted(group)), group)
                self.assertFalse(covered.intersection(group))
                covered.update(group)
                selected = [edges[index] for index in group]
                lefts, rights = {a for a, _ in selected}, {b for _, b in selected}
                self.assertEqual(len(group), len(lefts)*len(rights))
                self.assertEqual(set(selected), {(a, b) for a in lefts for b in rights})
                self.assertFalse({a.tensor for a in lefts} & {b.tensor for b in rights})
                self.assertTrue(all(edges.count(edge) == 1 for edge in selected))

    def test_partial_lowering_keeps_debug_reference_and_physical_budget_fallback(self):
        specs = {f'{side}{i}': TensorSpec((4, 5, 4, 5), np.complex128)
                 for side in ('a', 'b') for i in range(6)}
        terms = tuple(ContractionTerm((f'a{i}', f'b{j}'), 'ijkl,ijkl->')
                      for i in range(6) for j in range(6) if i != j)
        profile = CostProfile()
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                problem = ContractionProblem(specs, terms,
                    {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=backend)
                result = solve_storage(problem, {}, {key: SourcePreparation(()) for key in specs},
                    optimizer='anytime', kernel_batching=True, memory_limit=10**8, deadline_seconds=.1,
                    max_states=1000, root_width=1, branch_width=1, profile=profile, component_workers=1)
                plan = result.plan
                self.assertTrue(plan.kernel_batches)
                validate_executable_resources(plan)
                plain = plan_scene_storage(plan.state, {}, plan.preparations, profile=profile,
                                           memory_limit=10**8)
                audited, report = lower_dense_batches(plain, profile=profile, debug=True)
                self.assertTrue(report['applied'])
                self.assertEqual(audited.state.results, plain.state.results)
                self.assertLess(audited.seconds, plain.seconds)
                peak = plain.peak_device_bytes if backend == 'cupy' else plain.peak_host_bytes
                limited = replace(plain, memory_limit=peak)
                unchanged, report = lower_dense_batches(limited, profile=profile)
                self.assertIs(unchanged, limited)
                self.assertEqual(report['reason'], 'storage_limit')
                validate_executable_resources(unchanged)


if __name__ == '__main__':
    unittest.main()
