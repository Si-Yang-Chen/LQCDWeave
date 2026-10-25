"""Local Cartesian and independent GEMM coverage around a real baryon kernel."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from correlator.planning.test_gemm_groups import fixture as dense_fixture
from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem, ENTRIES
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SparseBridgeSpec,
    SourceSpec, SourcePreparation, DenseGemmGroup, PlannedContraction,
    validate_executable_resources,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, lower_dense_batches, lower_independent_dense_batches,
    join_storage_batches, plan_scene_storage,
)
from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
from lqcdweave.contraction.performance.api import predict_plan_cost
from lqcdweave.contraction.execution.scene_runtime import bind_scene_execution
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule


def mixed_fixture(backend='numpy', *, dense_scale=1, reduction_scale=1, special='baryon'):
    dense, profile = dense_fixture(count=6)
    if special == 'baryon':
        sparse = sparse_spin_problem()
    else:
        sparse = ContractionProblem(
            {'L': TensorSpec((2, 4, 2, 3), 'complex128'), 'R': TensorSpec((4, 3, 3, 5), 'complex128')},
            (ContractionTerm(('L', 'R'), 'iapc,bqdj->ij'),),
            {key: SourceSpec(key, 0, read_bytes=0) for key in ('L', 'R', 'J')},
            sparse_bridges=(SparseBridgeSpec('L', 'R', 'J', 2, 2,
                left_axes=(1, 2, 3), right_axes=(0, 1, 2), used_np_right=3),))
    specs = dict({key: replace(spec, shape=tuple(n*dense_scale*(reduction_scale
                  if (axis == 1 if key.startswith('a') else axis == 0) else 1)
                  for axis, n in enumerate(spec.shape)))
                  for key, spec in dense.state.problem.specs.items()}, **sparse.specs)
    joins = tuple(replace(join, scene=len(dense.state.problem.terms)) for join in sparse.baryon_joins)
    bridges = tuple(replace(bridge, scene=len(dense.state.problem.terms)) for bridge in sparse.sparse_bridges)
    sources = {key: SourceSpec(key, 0, read_bytes=0) for key in dict.fromkeys((*specs, *sparse.sources))}
    problem = ContractionProblem(specs, dense.state.problem.terms+sparse.terms,
        sources, backend=backend, baryon_joins=joins, sparse_bridges=bridges)
    state = ContractionState.initial(problem)
    for key in sources:
        state = state.load(key, profile, 1 << 30)
    for group in range(6):
        for term in range(4*group, 4*group+4):
            state = state.contract(term, (0, 1), profile, 1 << 30)
        if group == 2:
            state = (state.contract_join(joins[0], profile, 1 << 30, search_kernel_path=False)
                     if joins else state.contract(len(dense.state.problem.terms), (0, 1), profile, 1 << 30))
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in sources},
        profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    return plan, profile


def arrays_and_reference(plan):
    rng = np.random.default_rng(159)
    arrays = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
              for key, spec in plan.state.problem.specs.items()}
    expected = [arrays[t.inputs[0]] @ arrays[t.inputs[1]] for t in plan.state.problem.terms[:-1]]
    if plan.state.problem.baryon_joins:
        arrays['G'].fill(0)
        for i, j, k, coefficient in ENTRIES:
            arrays['G'][i, j, k] = coefficient
        expected.append(np.einsum('ijk,abc,ilad,jmbe,kncf->ldmenf',
            *(arrays[key] for key in ('G', 'E', 'P', 'Q', 'R')), optimize=False))
    else:
        gamma = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        values = rng.normal(size=(2, 3, 3))+1j*rng.normal(size=(2, 3, 3))
        record = dict(type='sparse', indices=np.array(((0, 2), (1, 0))), values=values)
        arrays['J'] = ((gamma, .3+.7j, record),)
        dense = np.zeros((2, 3, 3, 3), complex)
        dense[0, 2], dense[1, 0] = values
        expected.append((.3+.7j)*np.einsum('iapc,ab,pqcd,bqdj->ij', arrays['L'], gamma, dense, arrays['R']))
    return arrays, expected


def execute_with_numpy_reference(plan, arrays):
    # NumPy independently evaluates the saved CUDA group description; this is
    # a structural/numerical reference, not a real GPU performance test.
    class Execution:
        def begin_action(self, *args): pass
        def end_action(self, *args): pass
        def source(self, key): return plan.state.problem.sources[key]
        def materialize(self, key, source): return arrays[key]
        def materializes_owned(self, key): return False

    bound = bind_scene_execution(plan.state, plan.kernel_batches, np, events=())
    return bound.program.run(Execution(), {})


class MixedDenseBatchTests(unittest.TestCase):
    def test_local_regions_preserve_special_actions_and_full_numerical_results(self):
        for backend, special in (('numpy', 'baryon'), ('cupy', 'baryon'), ('numpy', 'sparse'), ('cupy', 'sparse')):
            with self.subTest(backend=backend, special=special):
                plain, profile = mixed_fixture(backend, special=special)
                lowered, stats = lower_dense_batches(plain, profile=profile, debug=True)
                self.assertTrue(stats['applied'])
                self.assertTrue(stats['local_regions'])
                self.assertIs(lowered.state, plain.state)
                self.assertEqual(lowered.events, plain.events)
                self.assertEqual(len(lowered.kernel_batches), 6)
                self.assertLess(lowered.seconds, plain.seconds)
                validate_executable_resources(lowered)
                arrays, expected = arrays_and_reference(plain)
                for actual, reference in zip(execute_with_numpy_reference(lowered, arrays), expected, strict=True):
                    np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-12)

    def test_independent_groups_do_not_cross_special_barrier_and_archive_does_not_regroup(self):
        plain, profile = mixed_fixture('cupy')
        lowered, _ = lower_dense_batches(plain, profile=profile)
        grouped, stats = lower_independent_dense_batches(lowered, profile=profile, target_runtime='cuda')
        self.assertTrue(stats['applied'])
        self.assertIs(grouped.state, plain.state)
        self.assertEqual(len(grouped.kernel_batches), 2)
        barrier = next(i for i, a in enumerate(plain.state.actions) if a.kind == 'baryon_join')
        for group in grouped.kernel_batches:
            self.assertIsInstance(group, DenseGemmGroup)
            self.assertEqual(len(group.batches), 3)
            self.assertTrue(max(group.action_indices) < barrier or min(group.action_indices) > barrier)
        arrays, expected = arrays_and_reference(plain)
        for actual, reference in zip(execute_with_numpy_reference(grouped, arrays), expected, strict=True):
            np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-12)
        shapes = {key: list(spec.shape) for key, spec in plain.state.problem.specs.items()}
        hardware = dict(backend='cupy', memory=1 << 30)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.json'
            save_storage_schedule(path, (grouped,), shapes=shapes, hardware=hardware)
            with patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                       side_effect=AssertionError('archive must not regroup')):
                restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
                    expected=(plain.state.problem,), blocks={}, preparations=plain.preparations)
        self.assertEqual(restored[0].kernel_batches, grouped.kernel_batches)
        self.assertEqual(restored[0].state.actions, grouped.state.actions)

    def test_local_packing_respects_existing_memory_limit(self):
        plain, profile = mixed_fixture(dense_scale=8, reduction_scale=4)
        lowered, _ = lower_dense_batches(plain, profile=profile)
        self.assertGreater(lowered.peak_host_bytes, plain.peak_host_bytes)
        limited = replace(plain, memory_limit=plain.peak_host_bytes)
        unchanged, stats = lower_dense_batches(limited, profile=profile)
        self.assertIs(unchanged, limited)
        self.assertEqual(stats['reason'], 'storage_limit')

    def test_task_batch_wrapper_lowers_children_and_updates_joint_prediction(self):
        original, profile = dense_fixture()
        plans = join_storage_batches((original, original))
        paths = tuple(PlannedContraction(None, plan, None,
            predict_plan_cost((plan,), profile, workload={}, provenance={}), 0,
            profile=profile, plan_identity=id(plan)) for plan in plans)
        prediction = predict_plan_cost(plans, profile, workload={}, provenance={})
        parent = replace(paths[0], prediction=prediction, batch_paths=paths, batch_shape_keys=('a', 'b'))
        selected = lower_independent_gemm_path(parent, target_runtime='cuda')
        self.assertIs(selected.plan, selected.batch_paths[0].plan)
        self.assertEqual(selected.plan_identity, id(selected.plan))
        self.assertEqual(selected.batch_shape_keys, parent.batch_shape_keys)
        for child in selected.batch_paths:
            self.assertIsInstance(child.plan.kernel_batches[0], DenseGemmGroup)
            validate_executable_resources(child.plan)
        self.assertAlmostEqual(selected.prediction['total_seconds'],
                               sum(child.plan.seconds for child in selected.batch_paths))


if __name__ == '__main__':
    unittest.main()
