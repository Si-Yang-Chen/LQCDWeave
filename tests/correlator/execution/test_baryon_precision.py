"""The public baryon task keeps its core, spin and elemental precision aligned."""

import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.correlator import ContractionTask, calculate
from lqcdweave.operators.correlator.api import Diagram
from lqcdweave.contraction.expansion.api import normalize_contraction_precision
from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries
from workflows.test_contraction_task_plan import no_planning


def _precision_case(dtype):
    case = _case((2, 0, 1))
    if np.dtype(dtype) == np.dtype('complex64'):
        for name in ('source', 'sink'):
            old = case['vertices'][name]
            reader = ArrayBaryonElemental(
                {'vvv': old.reader.sectors['vvv'].astype(dtype)},
                old.reader.components, cfg='cfg-a')
            case['vertices'][name] = Baryon(reader, old.operator, source=name == 'source')
            case[name + '_spatial'] = case[name + '_spatial'].astype(dtype)
        for key, old in tuple(case['propagators'].items()):
            case['propagators'][key] = ArrayPropagator(
                {'vsv': old.sectors['vsv'].astype(dtype)},
                time_convention='absolute', cfg='cfg-a', data_identity=f'c64-{key}')
        case['arrays'] = tuple(value.astype(dtype) for value in case['arrays'])
    options = dict(configuration='cfg-a', vertex_map=case['vertices'],
                   propagator_map=case['propagators'], optimize='none',
                   memory_limit=1 << 28, host_memory_limit=1 << 28)
    return case, options


def _coordinate_case():
    case, options = _precision_case('complex64')
    diagram = case['diagram']
    case['diagram'] = Diagram(diagram.diagram, ['source', 'sink'],
                              diagram.vertex_list, diagram.propagator_list)
    times = (1, 2, 0)
    for name in ('source', 'sink'):
        old = case['vertices'][name]
        spatial = case[name + '_spatial']
        values = np.stack([spatial * (1 if name == 'source' else
                                      1 + .15j * (time - 1))
                           for time in range(3)]).astype('complex64')
        reader = ArrayBaryonElemental({'vvv': values[None, None]},
                                      old.reader.components, cfg='cfg-a')
        case['vertices'][name] = Baryon(reader, old.operator, source=name == 'source')
    for key, old in tuple(case['propagators'].items()):
        base = old.sectors['vsv'][0, 1]
        data = np.empty((3, 3, *base.shape), dtype='complex64')
        for source in range(3):
            for sink in range(3):
                data[source, sink] = base * (1 + .04 * source + .07j * (sink - 1))
        case['propagators'][key] = ArrayPropagator(
            {'vsv': data}, time_convention='absolute', cfg='cfg-a',
            data_identity=f'coordinate-{key}')
    expected = np.asarray([_independent_oracle(dict(
        case, sink_spatial=case['sink_spatial'] * (1 + .15j * (sink - 1)),
        arrays=tuple(value * (1 + .07j * (sink - 1)) for value in case['arrays'])))
        for sink in times])
    options.update(time_pairs=[{'source': 0, 'sink': sink} for sink in times], batch=2)
    return case, options, expected


def _host(value):
    if type(value).__module__.startswith('cupy'):
        import cupy as cp
        return cp.asnumpy(value)
    return np.asarray(value)


def _assert_precision_graph(expanded, dtype):
    dtype = np.dtype(dtype)
    assert expanded.output_recipe.dtype == dtype.name
    assert expanded.output_recipe.intermediate_dtype == (
        'complex128' if dtype == np.dtype('complex64') else None)
    assert expanded.problem.baryon_joins
    assert normalize_contraction_precision(expanded) is expanded
    for join in expanded.problem.baryon_joins:
        assert join.core is not None and join.core.spec.dtype == dtype
        for component in join.core.components:
            assert component.coefficient == complex(dtype.type(component.coefficient))
            spin, elemental = component.inputs
            assert expanded.problem.specs[spin].dtype == dtype
            assert expanded.problem.specs[elemental].dtype == dtype
            spin_preparation = expanded.inputs.preparations[spin]
            elemental_preparation = expanded.inputs.preparations[elemental]
            assert spin_preparation.recipe.compute_dtype == dtype.name
            assert elemental_preparation.recipe.compute_dtype == dtype.name
            actual = spin_preparation.recipe.constants[0].array()
            assert actual.dtype == dtype
            sparse = np.zeros((4, 4, 4), dtype=dtype)
            for i, j, k, weight in component.sparse_spin:
                sparse[i, j, k] = weight
            np.testing.assert_array_equal(actual, sparse)


@pytest.mark.parametrize('input_dtype,dtype', (
    ('complex64', 'complex64'), ('complex128', 'complex128'),
    ('complex128', 'complex64'), ('complex64', 'complex128')))
def test_calculate_baryon_precision_matches_independent_oracle(input_dtype, dtype):
    case, options = _precision_case(input_dtype)
    options['dtype'] = dtype
    expected = _independent_oracle(case)
    with calculate(case['diagram'], execute=False, baryon_vertices='joined', **options) as selected:
        _assert_precision_graph(selected.expanded, dtype)
        actual = _host(selected.run().value)
    assert actual.dtype == np.dtype(dtype)
    rtol, atol = ((3e-6, 2e-5) if dtype == 'complex64' else (5e-12, 1e-10))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
def test_declared_precision_rounds_multicomponent_vertex_coefficients(backend):
    from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        assert int(cp.arange(3).sum().get()) == 3
    case, options = _precision_case('complex128')
    for name, seed, weight in (('source', 9, .123456789+.234567891j),
                               ('sink', 11, -.345678912+.456789123j)):
        vertex = case['vertices'][name]
        first, = vertex.operator.terms
        extra = OperatorTerm.from_baryon(first.component, _spin_entries(seed),
                                          momentum=first.momentum, coefficient=weight)
        vertex.operator = Operator.from_terms(name+'-sum', (first, extra))
    expected = sum(_independent_oracle(dict(case,
        source_spin=source.spin.dense(), sink_spin=sink.spin.dense(),
        source_coefficient=source.coefficient, sink_coefficient=sink.coefficient))
        for source in case['vertices']['source'].operator.terms
        for sink in case['vertices']['sink'].operator.terms)
    with calculate(case['diagram'], execute=False, backend=backend, baryon_vertices='joined',
                   dtype='complex64', **options) as selected:
        _assert_precision_graph(selected.expanded, 'complex64')
        assert any(len(join.core.components) == 2 for join in selected.expanded.problem.baryon_joins)
        actual = _host(selected.run().value)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-5)


@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
def test_complex64_baryon_task_archive_replays_on_current_inputs(backend, tmp_path):
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        assert int(cp.arange(3).sum().get()) == 3
    case, options, expected = _coordinate_case()
    options.update(backend=backend, dtype='complex64')
    with ContractionTask(case['diagram'], **options) as task:
        task.plan()
        plan = task._selected.plan
        assert tuple(plan.statistics['coordinate_batch_sizes']) == (2, 1)
        assert all(spec.dtype == np.dtype('complex64') for spec in plan.specs.values())
        assert all(join.core is None or join.core.spec.dtype == np.dtype('complex64')
                   for template in plan.templates
                   for join in template.plan.state.problem.baryon_joins)
        archive = tmp_path / 'baryon-c64.json.gz'
        with no_planning():
            task.save_plan(archive)
            first = _host(task.run().value)
            with ContractionTask.load_plan(archive, configuration='cfg-a',
                    vertex_map=case['vertices'], propagator_map=case['propagators']) as restored:
                replay = _host(restored.run().value)
                case['propagators']['p0'].sectors['vsv'] *= np.complex64(.5)
                changed = _host(restored.run().value)
    assert first.dtype == replay.dtype == np.dtype('complex64')
    np.testing.assert_allclose(first, expected, rtol=3e-6, atol=2e-5)
    np.testing.assert_allclose(replay, expected, rtol=3e-6, atol=2e-5)
    np.testing.assert_allclose(changed, .5 * expected, rtol=3e-6, atol=2e-5)
