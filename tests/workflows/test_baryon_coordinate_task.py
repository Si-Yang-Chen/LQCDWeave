"""Baryon coordinate lanes retain ordered quark slots and current inputs."""
import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.correlator import ContractionTask, calculate
from lqcdweave.operators.correlator.api import Diagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries
from workflows.test_contraction_task_plan import no_planning
from lqcdweave.storage.correlator_results import load_correlator_output


def fixture(slots=(2, 0, 1), multiple=False):
    case = _case(slots)
    times = 4
    vertices = {}
    readers = {}
    for name in ('source', 'sink'):
        original = case['vertices'][name]
        spatial = case[name + '_spatial']
        values = np.stack([spatial * (1 + .2*t + .1j*t) for t in range(times)])
        reader = ArrayBaryonElemental({'vvv': values[None, None]}, original.reader.components,
            cfg='cfg-a', data_identity='coordinate-' + name)
        readers[name] = reader
        vertices[name] = Baryon(reader, original.operator, source=name == 'source')
    extras = []
    if multiple:
        first = vertices['sink'].operator.terms[0]
        extra = OperatorTerm.from_baryon(first.component, _spin_entries(8),
            momentum=first.momentum, coefficient=-.2 + .3j)
        vertices['sink'].operator = Operator.from_terms('sum-sink', (first, extra))
        extras.append(dict(case, sink_spin=extra.spin.dense(), sink_coefficient=extra.coefficient))
    propagators = {}
    for q in range(3):
        original = case['propagators'][f'p{q}'].sectors['vsv'][0, 1]
        values = np.empty((times, times, *original.shape), dtype=original.dtype)
        for source in range(times):
            for sink in range(times):
                values[source, sink] = original * (1 + .03*source + .07j*sink)
        propagators[f'p{q}'] = ArrayPropagator({'vsv': values}, time_convention='absolute',
            cfg='cfg-a', data_identity=f'coordinate-p{q}')
    old = case['diagram']
    diagram = Diagram(old.diagram, ['source', 'sink'], old.vertex_list, old.propagator_list)
    points = [{'source': 0, 'sink': t} for t in (1, 2, 3)]
    base = _independent_oracle(case) + sum(_independent_oracle(item) for item in extras)
    expected = np.array([base * (1 + .2*t + .1j*t) * (1 + .07j*t)**3 for t in (1, 2, 3)])
    options = dict(vertex_map=vertices, propagator_map=propagators, configuration='cfg-a',
        time_pairs=points, batch=2, optimize='none', memory_limit=1 << 28, host_memory_limit=1 << 28)
    return diagram, options, readers, expected


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 'full', 'auto'])
def test_initial_and_selected_baryon_batches_match_independent_slots(batch, multiple, tmp_path):
    request, options, readers, expected = fixture(multiple=multiple)
    options['batch'] = batch
    direct = calculate(request, **options)
    np.testing.assert_allclose(direct.value, expected, rtol=5e-12, atol=1e-10)
    with ContractionTask(request, **options) as task:
        task.plan()
        if batch == 2:
            assert task._selected.plan.statistics['coordinate_batch_sizes'] == (2, 1)
        archive = tmp_path / 'baryon.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value, expected, rtol=5e-12, atol=1e-10)
            with ContractionTask.load_plan(archive, configuration='cfg-a', vertex_map=options['vertex_map'],
                    propagator_map=options['propagator_map']) as loaded:
                np.testing.assert_allclose(loaded.run().value, expected, rtol=5e-12, atol=1e-10)
                readers['sink'].sectors['vvv'] *= .5
                np.testing.assert_allclose(loaded.run().value, .5*expected, rtol=5e-12, atol=1e-10)


def test_real_gpu_baryon_coordinate_tail_and_archive(tmp_path):
    cp = pytest.importorskip('cupy')
    assert float(cp.asnumpy(cp.arange(3).sum())) == 3
    request, options, _, expected = fixture(multiple=True)
    options['backend'] = 'cupy'
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path / 'gpu-baryon.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value, expected, rtol=5e-12, atol=1e-10)
            with ContractionTask.load_plan(archive, configuration='cfg-a', vertex_map=options['vertex_map'],
                    propagator_map=options['propagator_map']) as loaded:
                np.testing.assert_allclose(loaded.run().value, expected, rtol=5e-12, atol=1e-10)


@pytest.mark.parametrize('backend', ['numpy', pytest.param('cupy', id='gpu')])
def test_default_baryon_plan_dataset_and_current_input_replay(backend, tmp_path):
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        assert float(cp.asnumpy(cp.arange(3).sum())) == 3
    request, options, readers, expected = fixture(multiple=True)
    options.update(backend=backend, optimize='anytime', output=tmp_path/'dataset')
    with ContractionTask(request, **options) as task:
        task.plan()
        assert task.plan_report['prediction']['coordinate_batch_sizes'] == (2, 1)
        archive = tmp_path/'dataset-plan.json.gz'
        with no_planning():
            task.save_plan(archive)
            assert task.run().report['status'] == 'completed'
            readers['sink'].sectors['vvv'] *= .5
            with ContractionTask.load_plan(archive, configuration='cfg-a',
                    vertex_map=options['vertex_map'], propagator_map=options['propagator_map'],
                    output=tmp_path/'again') as restored:
                assert restored.run().report['status'] == 'completed'
    for directory, weight in (('dataset', 1), ('again', .5)):
        actual = load_correlator_output(tmp_path/directory, configuration='cfg-a')
        np.testing.assert_allclose(actual, weight*expected, rtol=5e-12, atol=1e-10)
