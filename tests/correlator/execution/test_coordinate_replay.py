"""A fixed scalar selection executes forty sinks through the public run path."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from correlator.planning.test_coordinate_inputs import fixture
from lqcdweave.contracts.correlator.api import (
    BatchSpec, DiagramBindings, DiagramInputs, ExpandedCorrelator, OutputNode,
    OutputRecipe, PlannedContraction, describe_coordinates,
)
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.planning.api import CostProfile, solve_storage
from lqcdweave.contraction.calculation import Calculation


def selected():
    raw, requests, preps, specs, _, _, readers, _ = fixture()
    # Toy local propagators still carry both absolute endpoints.
    for key, role in (('v', 'sink'), ('s', 'source')):
        readers[role].arrays = {(args[0], args[0]): value
                               for args, value in readers[role].arrays.items()}
        request = requests['raw-' + key]
        arguments = (request.arguments[0], request.arguments[0])
        coordinate_role = 'sink' if key == 'v' else 'source'
        requests['raw-' + key] = replace(request, family='propagator', arguments=arguments,
            metadata=readers[role].describe_block(*arguments),
            coordinate_arguments=((0, coordinate_role, 0, None), (1, coordinate_role, 0, None)))
    profile = CostProfile()
    plan = solve_storage(raw.state.problem, raw.blocks, preps, optimizer='anytime',
        memory_limit=1 << 20, deadline_seconds=1, max_states=100,
        root_width=1, branch_width=1, component_workers=1, kernel_batching=False).plan
    coordinates, batch = describe_coordinates(time_axes={'source': 3, 'sink': 0})
    ids = ('loop', 'static')
    recipe = OutputRecipe(OutputNode('list', children=tuple(
        OutputNode('diagram_sum', ((key, 1),)) for key in ids)), ids, coordinates, 'complex64')
    loader = restore_input_loader(requests, readers, preps, specs)
    inputs = DiagramInputs(plan.state.problem, plan.blocks, plan.preparations, loader,
        coordinates, batch, ids, read_requests=requests)
    expanded = ExpandedCorrelator(None, inputs, recipe, DiagramBindings(), coordinates, batch)
    path = PlannedContraction(expanded, plan, None, {}, 0., 'planned',
        plan.memory_limit, plan.host_memory_limit, id(plan), profile)
    return Calculation(expanded, path, options={'kernel_batching': True}), readers


def reference(readers, sinks):
    a, v, r, s = (readers[key].arrays for key in ('forward', 'sink', 'reverse', 'source'))
    return np.array([np.einsum('ij,jk,kl,li->', a[sink, 3], v[sink, sink], r[3, sink], s[3, 3])
                     for sink in sinks])


@pytest.mark.parametrize('batch, sizes', [('auto', (40,)), (7, (7, 7, 7, 7, 7, 5)), (1, (1,) * 40)])
def test_public_run_fixed_path_forty_sinks_and_repeat(batch, sizes):
    calculation, readers = selected()
    with patch('lqcdweave.contraction.planning.path_planning.find_contraction_path',
               side_effect=AssertionError('fixed replay searched a contraction path')):
        result = calculation.run(time_axes={'source': 3, 'sink': range(40)},
                                 batch={'axis': 'sink', 'size': batch})
        np.testing.assert_allclose(result.value[0], reference(readers, range(40)), rtol=4e-6, atol=2e-5)
        assert result.prediction['coordinate_batching']['batch_sizes'] == sizes
        assert readers['source'].reads == 1
        assert calculation.value is None
        # Execution table reuse must not retain numeric inputs from the last run.
        readers['forward'].arrays = {key: value * np.complex64(1.03)
                                     for key, value in readers['forward'].arrays.items()}
        repeated = result.run()
    np.testing.assert_allclose(repeated.value[0], reference(readers, range(40)), rtol=4e-6, atol=2e-5)
    assert readers['source'].reads == 2
    assert repeated.path is result.path
    assert isinstance(repeated.report, dict)


@pytest.mark.parametrize("batch", [3, "full", "auto"])
def test_repeated_coordinates_static_batch_and_order(batch):
    calculation, readers = selected()
    sinks = (6, 2, 6, 1, 0, 1, 1)
    result = calculation.run(time_pairs=[{'source': 3, 'sink': sink} for sink in sinks], batch=batch)
    np.testing.assert_allclose(result.value[0], reference(readers, sinks), rtol=4e-6, atol=2e-5)
    assert result.prediction['coordinate_batching']['batch_sizes'] == ((3, 3, 1) if batch == 3 else (7,))
    assert len(result.value[1]) == len(sinks)
    assert all(value == result.value[1][0] for value in result.value[1])


def test_explicit_too_wide_batch_rejects_before_reading():
    from lqcdweave.contraction.planning.api import plan_coordinate_batches
    from lqcdweave.contraction.expansion.api import describe_coordinate_reads
    calculation, readers = selected()
    points, batch = describe_coordinates(time_axes={'source': 3, 'sink': range(40)}, batch='full')
    inputs = calculation.inputs
    with pytest.raises(MemoryError):
        plan_coordinate_batches(calculation.plan, calculation.expanded.output_recipe,
            points, inputs.read_requests, inputs.loader.preparations, inputs.loader.specs,
            batch=batch, read_metadata=describe_coordinate_reads(inputs, points.points), memory_limit=1000)
    assert sum(reader.reads for reader in readers.values()) == 0


def test_public_hadron_components_source_dagger_and_selected_scalar():
    from test_structured_correlator import fixture as hadron_fixture, task
    from lqcdweave.contraction.correlator import calculate
    bind, _, propagator = hadron_fixture()
    correlator = task(flavor='I0')
    options = dict(vertex_map=bind, propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
        configuration='cfg', usedNe=3, optimize='none', backend=np, memory_limit=1 << 26)
    original = calculate(correlator, time_axes={0: 1, 1: 0}, execute=False, **options)
    actual = original.run(time_axes={0: (1, 0), 1: 0}, batch={'axis': 0, 'size': 'auto'})
    expected = [calculate(correlator, time_axes={0: sink, 1: 0}, **options).value for sink in (1, 0)]
    np.testing.assert_allclose(actual.value, np.stack(expected, axis=-1), rtol=3e-12, atol=3e-12)
    scalar = original.run(time_axes={0: 0, 1: 0}, batch='auto')
    np.testing.assert_allclose(scalar.value, expected[1], rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(scalar.run().value, expected[1], rtol=3e-12, atol=3e-12)


def test_auto_reduces_width_to_budget_and_measured_debug_replay():
    calculation, readers = selected()
    plan = replace(calculation.plan, memory_limit=4000, host_memory_limit=4000)
    path = replace(calculation.path, plan=plan, plan_identity=id(plan), memory_limit=4000,
                   host_memory_limit=4000)
    calculation = replace(calculation, path=path)
    result = calculation.run(time_axes={'source': 3, 'sink': range(40)}, batch='auto')
    assert 1 < result.prediction['coordinate_batching']['selected_width'] < 40
    assert result.prediction['coordinate_batching']['peak_host_bytes'] <= 4000
    measured = result.run(debug=True, measure=True)
    np.testing.assert_allclose(measured.value[0], reference(readers, range(40)), rtol=4e-6, atol=2e-5)
    assert measured.report['coordinate_batching'] == result.report['coordinate_batching']
