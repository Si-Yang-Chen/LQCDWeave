"""Large raw results are consumed per joint batch, with only final fields kept."""
from dataclasses import replace
from unittest.mock import patch
import weakref

import numpy as np
import pytest

from correlator.planning.test_coordinate_inputs import Reader
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, TensorSpec, StorageBlock,
    SourcePreparation, TensorPreparation, SourceReadRequest, DiagramInputs,
    DiagramBindings, ExpandedCorrelator, OutputNode, OutputRecipe, PlannedContraction,
    describe_coordinates,
)
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.planning.api import (
    CostProfile, ContractionState, logical_source_problem, logical_cost_profile, plan_scene_storage,
)
from lqcdweave.contraction.results.api import estimate_output_retention_bytes
from lqcdweave.contraction.calculation import Calculation
from lqcdweave.contraction import correlator_batch_output as output


def selected(points=400, cap=2 << 20):
    rng = np.random.default_rng(235426)
    arrays = {(i, 0): (rng.normal(size=(128, 64)) + 1j * rng.normal(size=(128, 64))).astype('complex64')
              for i in range(points)}
    reader = Reader(arrays, 'grid-input')
    metadata = reader.describe_block(0, 0)
    request = SourceReadRequest('grid', 'propagator', (0, 0), (), metadata,
        coordinate_arguments=((0, 'sink', 0, None), (1, 'source', 0, None)))
    problem = ContractionProblem({'a': TensorSpec(metadata.shape, 'complex64')},
        (ContractionTerm(('a',), 'ij->ij'), ContractionTerm(('a',), 'ij->')),
        {'a': SourceSpec('a', metadata.shape[0]*metadata.shape[1]*8)})
    block = StorageBlock('grid-input', '0,0', arrays[0, 0].nbytes, host_bytes=arrays[0, 0].nbytes)
    preps = {'a': SourcePreparation(('raw',), dtype='complex64',
        recipe=TensorPreparation('identity', ('raw',), compute_dtype='complex64'))}
    coordinates, batch = describe_coordinates(time_axes={'source': 0, 'sink': 0})
    ids = ('grid', 'sum')
    leaf = lambda name: OutputNode('diagram_sum', ((name, 1),))
    recipe = OutputRecipe(OutputNode('dict', keys=('selected', 'sum'), children=(
        OutputNode('select', (0, 0), children=(leaf('grid'),)), leaf('sum'))),
        ids, coordinates, 'complex64')
    reserve = estimate_output_retention_bytes(recipe, problem, (), coordinates)
    profile = CostProfile()
    model = logical_cost_profile(profile)
    state = ContractionState.initial(logical_source_problem(problem)).load('a', model, cap)
    state = state.contract(0, (0,), model, cap).contract(1, (0,), model, cap)
    plan = plan_scene_storage(state, {'raw': block}, preps, profile=profile,
                              memory_limit=cap, host_memory_limit=cap, reserved_host_bytes=reserve)
    loader = restore_input_loader({'raw': request}, {'grid': reader}, preps, problem.specs)
    inputs = DiagramInputs(problem, {'raw': block}, preps, loader, coordinates, batch, ids,
                           read_requests={'raw': request}, assembly_host_bytes=reserve)
    expanded = ExpandedCorrelator(None, inputs, recipe, DiagramBindings(), coordinates, batch)
    path = PlannedContraction(expanded, plan, None, {}, 0., 'planned', cap, cap,
                              id(plan), profile, reserve, 0)
    return Calculation(expanded, path, options={'kernel_batching': True}), reader


def test_four_hundred_points_joint_output_budget_and_raw_owner_release():
    calculation, reader = selected()
    original = output.execute_contraction_path
    references = []
    visits = []

    def execute(*args, **kwargs):
        consume = kwargs['batch_consumer']

        def checked(index, raw, axes):
            assert all(reference() is None for reference in references)
            references[:] = [weakref.ref(value) for value in raw
                             if isinstance(value, np.ndarray) and value.nbytes > 1024]
            visits.append(index)
            consume(index, raw, axes)
        return original(*args, **dict(kwargs, batch_consumer=checked))

    with patch.object(output, 'execute_contraction_path', side_effect=execute), patch(
        'lqcdweave.contraction.correlator.find_contraction_path',
        side_effect=AssertionError('fixed replay searched')), patch(
        'lqcdweave.contraction.correlator.expand_correlator',
        side_effect=AssertionError('fixed replay expanded')):
        result = calculation.run(time_axes={'source': 0, 'sink': range(400)}, batch='auto')
        assert len(visits) > 1
        assert all(reference() is None for reference in references)
        assert result.prediction['coordinate_batching']['peak_host_bytes'] <= 2 << 20
        assert result.report['coordinate_points'] == 400
        np.testing.assert_allclose(result.value['selected'], [reader.arrays[i, 0][0, 0] for i in range(400)])
        np.testing.assert_allclose(result.value['sum'], [reader.arrays[i, 0].sum() for i in range(400)],
                                   rtol=5e-5, atol=2e-4)
        saved = {key: value.copy() for key, value in result.value.items()}
        reader.arrays = {key: value * 2 for key, value in reader.arrays.items()}
        repeated = result.run()
        assert repeated.path is result.path
        for key in saved:
            np.testing.assert_allclose(repeated.value[key], saved[key]*2, rtol=5e-5, atol=2e-4)
            np.testing.assert_array_equal(result.value[key], saved[key])

    # Raw diagnostic reports really do need the scan's complete raw payload.
    # Reject that optional mode before reading rather than under-account it.
    reads = reader.reads
    with pytest.raises(MemoryError, match='raw measurement reports'):
        result.run(measure=True)
    assert reader.reads == reads


def test_raw_report_mode_reproduces_the_old_budget_rejection_before_reads():
    calculation, reader = selected()
    with pytest.raises(MemoryError):
        calculation.run(time_axes={'source': 0, 'sink': range(400)}, batch='auto', measure=True)
    assert reader.reads == 0


def test_coordinate_group_order_preserves_two_scan_axes():
    calculation, reader = selected(points=7)
    reader.arrays.update({(sink, 1): value * 3
                         for (sink, source), value in tuple(reader.arrays.items())})
    result = calculation.run(time_axes={'source': (0, 1), 'sink': range(7)},
                             batch={'axis': 'source', 'size': 2})
    expected = [[reader.arrays[sink, source][0, 0] for sink in range(7)] for source in (0, 1)]
    np.testing.assert_allclose(result.value['selected'], expected)


def test_heterogeneous_object_container_matches_existing_assembler():
    from lqcdweave.contraction.execution.api import execute_contraction_path
    from lqcdweave.contraction.results.api import assemble_correlator
    calculation, _reader = selected(points=5)
    root = OutputNode('array', shape=(2,), children=(
        calculation.expanded.output_recipe.root.children[0],
        OutputNode('constant_array', np.array((1., 2., 3.)), shape=(3,)),
    ))
    expanded = replace(calculation.expanded,
                       output_recipe=replace(calculation.expanded.output_recipe, root=root))
    calculation = replace(calculation, expanded=expanded,
                          path=replace(calculation.path, expanded=expanded))
    result = calculation.run(time_axes={'source': 0, 'sink': range(5)}, batch=2)
    raw = execute_contraction_path(result.expanded, result.path)
    expected = assemble_correlator(result.expanded, raw.term_values)
    assert result.value.shape == expected.shape == (2,)
    for actual, reference in zip(result.value, expected):
        np.testing.assert_allclose(actual, reference)


def test_saved_coordinate_selection_replays_without_raw_term_retention(tmp_path):
    from lqcdweave.contraction.correlator import save_calculation_plan, load_calculation_plan
    calculation, _reader = selected(points=7)
    result = calculation.run(time_axes={'source': 0, 'sink': range(7)}, batch=2)
    archive = tmp_path/'coordinate-plan.json.gz'
    hardware = {'backend': 'numpy', 'memory': 2 << 20}
    save_calculation_plan(archive, result, hardware=hardware)
    restored = load_calculation_plan(archive, expected=result.expanded, hardware=hardware)
    with patch('lqcdweave.contraction.correlator.find_contraction_path',
               side_effect=AssertionError('archive searched')), patch(
               'lqcdweave.contraction.correlator.expand_correlator',
               side_effect=AssertionError('archive expanded')):
        repeated = restored.run()
    assert repeated.report['output_retention'] == 'assembled-coordinates'
    for key in result.value:
        np.testing.assert_array_equal(repeated.value[key], result.value[key])
