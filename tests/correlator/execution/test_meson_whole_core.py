"""Closed meson kernels preserve physics without a private-factor P search."""
from dataclasses import replace
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, AxisPattern, ContractionProblem, ContractionTerm,
    MesonWholeCore, MesonWholeKernel, SourcePreparation, SourceSpec,
    TensorPreparation, TensorSpec, VertexKernelDescription,
)
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.contraction.planning.meson_core import plan_meson_whole
from lqcdweave.contraction.performance.api import CostProfile
from correlator.execution.test_mixed_structure_vertex import array_backend


def case(kind, dtype, batched, *, spatial_components=2):
    rng = np.random.default_rng(82415)
    ne = 2
    if kind == 'connected':
        shapes = ((4, 4, ne, ne), (4, 4, ne, ne),
                  (spatial_components, ne, ne), (spatial_components, ne, ne),
                  (2, spatial_components, 4, 4, 2), (3, spatial_components, 4, 4, 2))
        private = AxisPattern(((0, 1, 2, 3), (4, 5, 6, 7), (8, 3, 6), (9, 7, 2),
                               (10, 8, 1, 4, 12), (11, 9, 5, 0, 13)), (10, 11))
        outer = AxisPattern(((0, 1, 2, 3), (4, 5, 6, 7), (10, 1, 4, 3, 6),
                             (11, 5, 0, 7, 2)), (10, 11))
        flags = (True, True, True, False, False, False)
        core_shapes = ((2, 4, 4, ne, ne), (3, 4, 4, ne, ne))
    else:
        shapes = ((4, 4, ne, ne), (spatial_components, ne, ne),
                  (3, spatial_components, 4, 4, 2))
        private = AxisPattern(((0, 1, 2, 3), (4, 3, 2), (5, 4, 1, 0, 6)), (5,))
        outer = AxisPattern(((0, 1, 2, 3), (5, 1, 0, 3, 2)), (5,))
        flags = (True, True, False)
        core_shapes = ((3, 4, 4, ne, ne),)
    if batched:
        shapes = tuple((2, *shape) if flag else shape for shape, flag in zip(shapes, flags))
        private = AxisPattern(tuple((14, *axes) if flag else axes
            for axes, flag in zip(private.inputs, flags)), (14, *private.output))
        outer_flags = (True, True, True, False) if kind == 'connected' else (True, True)
        outer = AxisPattern(tuple((14, *axes) if flag else axes
            for axes, flag in zip(outer.inputs, outer_flags)), (14, *outer.output))
        core_shapes = ((2, *core_shapes[0]), *core_shapes[1:])
    values = tuple((.2 * (rng.normal(size=shape) + 1j*rng.normal(size=shape))).astype(dtype)
                   for shape in shapes)
    inputs = tuple(f'input:{i}' for i in range(len(values)))
    count = 2 if kind == 'connected' else 1
    core_keys = tuple(f'whole:{i}' for i in range(count))
    kernel = MesonWholeKernel(kind, core_keys, inputs, private, 'unit-spin', 14 if batched else None,
                             VertexKernelDescription('meson', 'whole-matrix-v3'))
    specs = {key: TensorSpec(value.shape, dtype) for key, value in zip(inputs, values)}
    specs.update({key: TensorSpec(shape, dtype) for key, shape in zip(core_keys, core_shapes)})
    cores = tuple(MesonWholeCore(key, (inputs[count+i], inputs[2*count+i]),
                  tuple((i, row) for row in range(core_shapes[i][-5])), 'unit-spin')
                  for i, key in enumerate(core_keys))
    term = ContractionTerm((*inputs[:count], *core_keys), outer, kernel=kernel)
    sources = {key: SourceSpec(key, 0, snapshot_bytes=value.nbytes, read_bytes=0)
               for key, value in zip(inputs, values)}
    problem = ContractionProblem(specs, (term,), sources, meson_cores=cores)
    return problem, values


@pytest.mark.parametrize('kind', ('connected', 'loop'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('batched', (False, True))
def test_whole_kernel_matches_independent_full_index_equation(kind, dtype, batched):
    problem, values = case(kind, dtype, batched)
    kernel = problem.terms[0].kernel
    axes = kernel.private_equation
    dimensions = {label: size for labels, value in zip(axes.inputs, values)
                  for label, size in zip(labels, value.shape)}
    result_dtype = 'complex128' if kind == 'loop' and dtype == 'complex64' else dtype
    output = TensorSpec(tuple(dimensions[a] for a in axes.output), result_dtype)
    action = Action('contract', 'result', kernel.inputs, problem.terms[0].subscripts, core=kernel)
    backend = array_backend()
    function = bind_dense_action(action, dict(problem.specs) | {'result': output}, backend)
    actual, _ = function(tuple(backend.asarray(value) for value in values))
    if backend is not np:
        actual = backend.asnumpy(actual)
    arguments = tuple(item for value, labels in zip(values, axes.inputs)
                      for item in (value.astype('complex128'), labels)) + (axes.output,)
    expected = np.einsum(*arguments, optimize=False)
    tolerance = 5e-5 if dtype == 'complex64' else 5e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_outer_planner_consumes_whole_vertices_without_searching_private_factors():
    problem, values = case('connected', 'complex128', False)
    preparations = {key: SourcePreparation((), dtype='complex128',
        recipe=TensorPreparation('constant', constants=(ArrayConstant.from_array(value),)))
        for key, value in zip(problem.terms[0].kernel.inputs, values)}
    with patch('lqcdweave.contraction.planning.path_planning.solve_storage',
               side_effect=AssertionError('private-factor P search')):
        plan = plan_meson_whole(problem, {}, preparations, profile=CostProfile(), memory_limit=1 << 28)
    term = problem.terms[0]
    assert set(term.inputs).intersection(problem.sources) == set(term.kernel.inputs[:2])
    assert set(term.inputs).difference(problem.sources) == {core.key for core in problem.meson_cores}
    contracts = [action for action in plan.state.actions if action.kind == 'contract']
    assert len(contracts) == 1 and type(contracts[0].core) is MesonWholeKernel
    assert contracts[0].kernel_path == ()
    # Calibration may consume structural estimates, but must not present
    # fixed stage counts as measured CUDA launches.
    from lqcdweave.contraction.execution.scene_runtime import _model_features
    features = _model_features(plan.state, contracts[0])
    assert features['kernel_calls_kind'] == 'estimate'
    assert features['operation_domain'] == 'whole_meson' and features['flops'] > 0


def test_large_whole_plan_matches_original_immutable_construction():
    from lqcdweave.contraction.planning.state import ContractionState
    from lqcdweave.contraction.planning.storage import logical_source_problem, logical_cost_profile
    from lqcdweave.contraction.planning.meson_core import describe_meson_whole_action, replay_meson_whole_action
    problem, _ = case('connected', 'complex128', False)
    problem = replace(problem, terms=problem.terms * 130)
    profile = CostProfile()
    model = logical_cost_profile(profile)
    reference = ContractionState.initial(logical_source_problem(problem))
    remaining = Counter(key for term in problem.terms for key in set(term.kernel.inputs))
    for scene, term in enumerate(problem.terms):
        for key in dict.fromkeys(term.kernel.inputs):
            if key not in reference.loaded:
                reference = reference.load(key, model, 1 << 100)
        action, _, _ = describe_meson_whole_action(reference, scene, profile=model)
        reference = replay_meson_whole_action(reference, action, model)
        for key in dict.fromkeys(term.kernel.inputs):
            remaining[key] -= 1
            if not remaining[key]:
                reference = reference.release(key)
    with patch('lqcdweave.contraction.planning.storage.plan_scene_storage',
               side_effect=lambda state, *args, **kwargs: SimpleNamespace(state=state, feasible=True)):
        result = plan_meson_whole(problem, {}, {}, profile=profile, memory_limit=1 << 28)
    for field in ('actions', 'pending', 'results', 'catalog', 'loaded', 'seconds',
                  'peak_bytes', 'peak_host_bytes', 'peak_device_bytes'):
        assert getattr(result.state, field) == getattr(reference, field)


@pytest.mark.parametrize('implementation', ('whole-matrix-v1', 'whole-matrix-v2'))
def test_saved_implementation_retains_its_phase_closure(implementation):
    problem, values = case('connected', 'complex128', True)
    kernel = replace(problem.terms[0].kernel,
                     kernel_description=VertexKernelDescription('meson', implementation))
    result = TensorSpec((2, 2, 3), 'complex128')
    action = Action('contract', 'result', kernel.inputs, problem.terms[0].subscripts, core=kernel)
    bound = []
    def capture(local, catalog, backend, *args, **kwargs):
        bound.append(local)
        return bind_dense_action(local, catalog, backend, *args, **kwargs)
    backend = array_backend()
    with patch('lqcdweave.contraction.execution.operations.contract.bind_dense_action', capture):
        function = bind_dense_action(action, dict(problem.specs) | {'result': result}, backend)
    actual, _ = function(tuple(backend.asarray(value) for value in values))
    if backend is not np:
        actual = backend.asnumpy(actual)
    axes = kernel.private_equation
    arguments = tuple(item for value, labels in zip(values, axes.inputs)
                      for item in (value, labels)) + (axes.output,)
    expected = np.einsum(*arguments, optimize=False)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-12)
    # The saved v1 finish consumes two mode-sized H arrays, rather than the
    # new v2 finish's small mode-closed result. Old archives retain that order.
    assert len(bound) == 5
    if implementation == 'whole-matrix-v1':
        assert all({2, 6} <= set(labels) for labels in bound[-1].equation.inputs)
    else:
        assert all(not {2, 6}.intersection(labels) for labels in bound[-1].equation.inputs)


@pytest.mark.parametrize('batched', (False, True))
def test_single_spatial_complex128_closes_modes_before_phase(batched):
    problem, values = case('connected', 'complex128', batched, spatial_components=1)
    kernel = problem.terms[0].kernel
    shape = (2, 2, 3) if batched else (2, 3)
    action = Action('contract', 'result', kernel.inputs, problem.terms[0].subscripts, core=kernel)
    bound = []
    def capture(local, catalog, backend, *args, **kwargs):
        bound.append(local)
        return bind_dense_action(local, catalog, backend, *args, **kwargs)
    backend = array_backend()
    with patch('lqcdweave.contraction.execution.operations.contract.bind_dense_action', capture):
        function = bind_dense_action(action, dict(problem.specs) | {'result': TensorSpec(shape, 'complex128')}, backend)
    actual, _ = function(tuple(backend.asarray(value) for value in values))
    if backend is not np:
        actual = backend.asnumpy(actual)
    equation = kernel.private_equation
    arguments = tuple(item for value, labels in zip(values, equation.inputs)
                      for item in (value, labels)) + (equation.output,)
    np.testing.assert_allclose(actual, np.einsum(*arguments, optimize=False), rtol=5e-12, atol=5e-12)
    assert len(bound) == 5 and not {2, 6}.intersection(bound[2].equation.output)

