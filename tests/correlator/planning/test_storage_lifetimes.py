"""Whole kernels consume fixed allocation lifetimes without symbolic replay."""
from dataclasses import replace
from unittest.mock import patch
import pytest

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, CoreComponent,
    SourcePreparation, SourceSpec, TensorCoreSpec, TensorSpec,
)
from lqcdweave.contraction.planning.api import CostProfile, plan_scene_storage
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.storage import logical_source_problem
from lqcdweave.contraction.planning import storage, canonical_operations
from lqcdweave.contraction.planning.meson_core import plan_meson_whole
from correlator.execution.test_meson_whole_core import case
from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem


PROFILE = CostProfile()


def compare(plan):
    kwargs = dict(profile=PROFILE, memory_limit=10**9, host_memory_limit=10**9,
                  reserved_host_bytes=13,
                  reserved_device_bytes=17 if plan.state.problem.backend == 'cupy' else 0)
    expected = plan_scene_storage(plan.state, plan.blocks, plan.preparations, debug=True, **kwargs)
    with patch.object(storage, '_apply_selected', side_effect=AssertionError('symbolic replay')), \
         patch.object(CostProfile, 'contract_seconds', side_effect=AssertionError('repeated price')):
        actual = plan_scene_storage(plan.state, plan.blocks, plan.preparations, **kwargs)
    assert actual.state is plan.state
    assert actual.memory == expected.memory
    assert actual.events == expected.events
    assert actual.preparation_costs == expected.preparation_costs
    assert (actual.peak_host_bytes, actual.peak_device_bytes) == (expected.peak_host_bytes, expected.peak_device_bytes)


@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
@pytest.mark.parametrize('kind', ('connected', 'loop'))
@pytest.mark.parametrize('batched', (False, True))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
def test_whole_meson_shared_results_and_coordinate_lanes(backend, kind, batched, dtype):
    problem, _ = case(kind, dtype, batched)
    problem = replace(problem, backend=backend, terms=problem.terms * 3)
    preparations = {key: SourcePreparation(()) for key in problem.sources}
    plan = plan_meson_whole(problem, {}, preparations, profile=PROFILE, memory_limit=10**9)
    compare(plan)


@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
@pytest.mark.parametrize('core', (False, True))
def test_baryon_sparse_spin_and_multicomponent_core(backend, core):
    if core:
        shapes = {'S1': (4, 4, 4), 'S2': (4, 4, 4), 'E1': (2, 2, 2),
                  'E2': (2, 2, 2), 'P': (4, 2), 'Q': (4, 2), 'R': (4, 2)}
        specs = {key: TensorSpec(shape, 'complex128') for key, shape in shapes.items()}
        vertex = TensorCoreSpec('baryon', ('spin0', 'spin1', 'spin2', 'laplace_vector0',
            'laplace_vector1', 'laplace_vector2'), TensorSpec((4, 4, 4, 2, 2, 2), 'complex128'),
            (CoreComponent(('S1', 'E1'), coefficient=0.5j), CoreComponent(('S2', 'E2'), coefficient=-2)))
        problem = ContractionProblem(specs, (ContractionTerm(('P', 'Q', 'R'), 'ax,by,cz->'),),
            {key: SourceSpec(key, 0, read_bytes=0) for key in specs},
            baryon_joins=(BaryonJoinSpec(0, 'sink', 'S1', 'E1', 'abc', 'xyz', core=vertex),))
    else:
        problem = logical_source_problem(sparse_spin_problem())
    problem = replace(problem, backend=backend)
    state = ContractionState.initial(problem)
    for key in problem.sources:
        state = state.load(key, PROFILE, 10**9)
    join = problem.baryon_joins[0]
    if core:
        action, spec, workspace, output, positions = canonical_operations.describe(
            state, 0, join=join, profile=PROFILE)
        state = canonical_operations.apply_description(state, action, spec, workspace, output, positions, join=join)
    else:
        state = state.contract_join(join, PROFILE, 10**9, search_kernel_path=False)
    for key in tuple(state.loaded):
        if key not in state.results:
            state = state.release(key)
    preparations = {key: SourcePreparation(()) for key in problem.sources}
    plan = plan_scene_storage(state, {}, preparations, profile=PROFILE, memory_limit=10**9)
    compare(plan)
