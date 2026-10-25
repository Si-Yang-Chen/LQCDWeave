"""Complete saved tasks keep mixed support internal to each vertex."""
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import ContractionTask
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, HadronIrrep, Tag, gen_correlator, iter_structured_wick_entries,
)
from correlator.execution.test_mixed_structure_vertex import fixture, explicit_vertex, array_backend
from workflows.test_contraction_task_plan import no_planning


def declarations(dtype=np.complex128, *, batch=2, ne=3, family='mixed', profile=False):
    vertex, raw, terms, _ = fixture(dtype=dtype, ne=ne, profile=profile)
    if family == 'dense':
        vertex.reader.elemental_structures = {}
    if family == 'diagonal':
        from lqcdweave.operators.single_hadron.api import Operator
        vertex.operator = Operator.from_terms('diagonal', (terms[0], terms[2]))
    rng = np.random.default_rng(73155)
    data = (rng.normal(size=(3, 3, 4, 4, ne, ne))
            + 1j * rng.normal(size=(3, 3, 4, 4, ne, ne))).astype(np.result_type(dtype, np.complex64))
    reader = ArrayPropagator({'vsv': data}, time_convention='absolute', cfg='cfg', data_identity='mixed/P')
    pion = Hadron(HadronIrrep('mixed', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0], Flavor('ud'))
    request = gen_correlator([[pion], [pion]], [0, 1], [False, True], structured=True)
    def bind(row):
        return Meson(vertex.reader, vertex.operator, row.dagger, phase_convention='anchor')
    options = dict(vertex_map=bind, propagator_map={'S^q': reader, r'S^q_\mathrm{local}': reader},
        configuration='cfg', usedNe=ne, time_pairs=[{0:t, 1:1} for t in range(3)], batch=batch,
        optimize='none', backend='cupy' if array_backend() is not np else 'numpy',
        memory_limit=1 << 29, host_memory_limit=1 << 29)
    return request, options, vertex, raw, data


def independent_task_value(request, vertex, raw, peram):
    entries = tuple(iter_structured_wick_entries(request))
    assert len(entries) == 1 and len(entries[0][1]) == 1
    coefficient, diagrams = entries[0][1][0]
    assert len(diagrams) == 1
    def matrix(value):
        return value.transpose(0, 2, 1, 3).reshape(value.shape[0] * value.shape[2], -1)
    source = matrix(explicit_vertex(raw, vertex.operator.terms, 1, source=True))
    values = []
    for t in range(3):
        sink = matrix(explicit_vertex(raw, vertex.operator.terms, t))
        values.append(complex(coefficient) * np.trace(source @ matrix(peram[t, 1]) @ sink @ matrix(peram[1, t])))
    return np.asarray(values).reshape((1, 1, 3))


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.complex64, np.complex128))
@pytest.mark.parametrize('batch', (1, 2, 'full', 'auto'))
def test_mixed_task_complete_archive_tail_and_changed_inputs_without_e_or_p(tmp_path, dtype, batch):
    from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
    request, options, vertex, raw, peram = declarations(dtype, batch=batch)
    tolerance = 5e-5 if np.dtype(dtype).itemsize <= (8 if np.dtype(dtype).kind == 'c' else 4) else 6e-12
    captured = []
    original = LegacyDiagramBinding.describe
    def capture(binding, *args, **kwargs):
        inputs = original(binding, *args, **kwargs)
        captured.append(inputs.problem)
        return inputs
    with ContractionTask(request, **options) as task:
        with patch.object(vertex.reader, 'block', side_effect=AssertionError('planning read numerical input')), \
                patch.object(LegacyDiagramBinding, 'describe', capture):
            task.plan()
        # Inspect E's original complete vertices, rather than residual Terms
        # left in a final P state or separate coordinate output slots.
        assert captured
        for problem in captured:
            keys = {key for term in problem.terms for key in term.inputs}
            elementals = [problem.specs[key].description for key in keys
                          if problem.specs[key].description is not None
                          and problem.specs[key].description.role == 'elemental']
            assert len(elementals) == 2
            assert all(desc.semantics['structure'] == 'general'
                       and desc.semantics['coefficient_owner'] == 'vertex-preparation'
                       for desc in elementals)
        archive = tmp_path / 'mixed-task.json.gz'
        task.save_plan(archive)
        with no_planning():
            for repeat in range(2):
                with ContractionTask.load_plan(archive, configuration='cfg', vertex_map=vertex.reader,
                        propagator_map=options['propagator_map']) as loaded:
                    actual = loaded.run().value
                expected = independent_task_value(request, vertex, raw, peram)
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                if repeat == 0:
                    raw *= .7
                    peram *= 1.1


def test_profiled_complete_mixed_task_preserves_source_adjoint(tmp_path):
    request, options, vertex, raw, peram = declarations(profile=True)
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path / 'profiled.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive, configuration='cfg',
            vertex_map=vertex.reader, propagator_map=options['propagator_map']) as task:
        actual = task.run().value
    np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                               rtol=8e-12, atol=8e-12)


def test_profiled_default_whole_matrix_archive_preserves_coordinate_rows(tmp_path):
    request, options, vertex, raw, peram = declarations(profile=True)
    options.pop('optimize')
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path / 'profiled-whole-matrix.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive, configuration='cfg',
            vertex_map=vertex.reader, propagator_map=options['propagator_map']) as task:
        actual = task.run().value
    np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                               rtol=8e-12, atol=8e-12)


def test_profiled_diagonal_whole_archive_keeps_single_space_gram_and_tail(tmp_path):
    request, options, vertex, raw, peram = declarations(profile=True, family='diagonal', ne=2, batch=2)
    options.pop('optimize')
    with ContractionTask(request, **options) as task:
        task.plan()
        for template in task._selected.plan.templates:
            state = template.plan.state
            kernels = [action.core for action in state.actions if action.kind == 'contract']
            assert kernels and all(kernel.kernel_description.implementation == 'whole-matrix-v3'
                                   for kernel in kernels)
            assert all(state.catalog[key].shape[-3] == 1
                       for kernel in kernels for key in kernel.inputs[2:4])
        archive = tmp_path / 'profiled-single-space.json.gz'
        task.save_plan(archive)
    with no_planning():
        for repeat in range(2):
            with ContractionTask.load_plan(archive, configuration='cfg',
                    vertex_map=vertex.reader, propagator_map=options['propagator_map']) as loaded:
                actual = loaded.run().value
            np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                                       rtol=8e-12, atol=8e-12)
            if repeat == 0:
                raw *= .7
                peram *= 1.1


@pytest.mark.parametrize('optimizer', ('default', 'meson_matrix'))
def test_mixed_common_optimizer_complete_archive_matches_independent_trace(tmp_path, optimizer):
    from lqcdweave.contraction.expansion import matrix_inputs
    request, options, vertex, raw, peram = declarations(ne=2, batch=2)
    if optimizer == 'default':
        options.pop('optimize')
    else:
        options['optimize'] = optimizer
    captured = []
    original = matrix_inputs._bind_matrix_base
    def capture(*args, **kwargs):
        expanded = original(*args, **kwargs)
        captured.append(expanded.problem)
        return expanded
    with ContractionTask(request, **options) as task:
        with patch.object(matrix_inputs, '_bind_matrix_base', capture), \
                patch('lqcdweave.contraction.planning.path_planning.solve_storage',
                      side_effect=AssertionError('P searched private meson factors')):
            task.plan()
        assert captured and all(len(problem.terms) == 1 for problem in captured)
        for problem in captured:
            assert len(problem.meson_cores) == 2
            referenced = set(problem.terms[0].inputs)
            assert {core.key for core in problem.meson_cores} <= referenced
            assert not any(set(core.inputs) & referenced for core in problem.meson_cores)
        archive = tmp_path / f'mixed-{optimizer}.json.gz'
        task.save_plan(archive)
    with no_planning():
        for repeat in range(2):
            with ContractionTask.load_plan(archive, configuration='cfg',
                    vertex_map=vertex.reader, propagator_map=options['propagator_map']) as loaded:
                actual = loaded.run().value
            np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                                       rtol=8e-12, atol=8e-12)
            if repeat == 0:
                raw *= .7
                peram *= 1.1


@pytest.mark.parametrize('dtype', (np.complex64, np.complex128))
def test_scalar_mode_default_whole_archive_keeps_current_input_precision(tmp_path, dtype):
    # The old unit-spin/FWR route excludes a reduction of length one. Keep
    # that numerical domain in the local dense fallback without expanding
    # component choices into the outer plan.
    request, options, vertex, raw, peram = declarations(dtype, ne=1, batch=2)
    options.pop('optimize')
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path / 'scalar-whole.json.gz'
        task.save_plan(archive)
    tolerance = 5e-5 if dtype == np.complex64 else 8e-12
    with no_planning():
        for repeat in range(2):
            with ContractionTask.load_plan(archive, configuration='cfg',
                    vertex_map=vertex.reader, propagator_map=options['propagator_map']) as loaded:
                actual = loaded.run().value
            np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                                       rtol=tolerance, atol=tolerance)
            if repeat == 0:
                raw *= .7
                peram *= 1.1


@pytest.mark.parametrize('input_dtype,declared_dtype',
                         ((np.complex128, 'complex64'), (np.complex64, 'complex128')))
def test_declared_whole_meson_precision_survives_complete_archive(tmp_path, input_dtype, declared_dtype):
    request, options, vertex, raw, peram = declarations(input_dtype, ne=3, batch=2)
    options.pop('optimize')
    with ContractionTask(request, dtype=declared_dtype, **options) as task:
        with patch.object(vertex.reader, 'block', side_effect=AssertionError('planning read numerical input')):
            task.plan()
        archive = tmp_path / 'declared-whole.json.gz'
        task.save_plan(archive)
    tolerance = 5e-5 if declared_dtype == 'complex64' else 8e-12
    with no_planning():
        for repeat in range(2):
            with ContractionTask.load_plan(archive, configuration='cfg',
                    vertex_map=vertex.reader, propagator_map=options['propagator_map']) as loaded:
                actual = loaded.run().value
            assert actual.dtype == np.dtype(declared_dtype)
            expected = independent_task_value(request, vertex, raw.astype(declared_dtype),
                                              peram.astype(declared_dtype))
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            if repeat == 0:
                raw *= .7
                peram *= 1.1


def test_default_disconnected_whole_matrix_archive_keeps_wide_loops(tmp_path):
    from lqcdweave.contraction.correlator import calculate

    _, options, vertex, _, _ = declarations(np.complex64, ne=2, batch=2)
    particle = Hadron(HadronIrrep('mixed', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0], Flavor('uu'))
    request = gen_correlator([[particle], [particle]], [0, 1], [False, True], structured=True)
    # Use the established symbolic nonmatrix contraction for the end-to-end
    # Wick/output comparison. The fixed loop kernel has a separate literal
    # index reference in test_meson_whole_core.py.
    symbolic = gen_correlator([[particle], [particle]], [0, 1], [False, True], structured=False)
    reference_options = dict(options, backend='numpy')
    expected = calculate(symbolic, **reference_options).value
    options.pop('optimize')
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path / 'disconnected-whole.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive, configuration='cfg',
            vertex_map=vertex.reader, propagator_map=options['propagator_map']) as loaded:
        actual = loaded.run().value
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize('source', (False, True))
def test_baryon_mixed_meson_complete_archive_uses_one_supported_vertex(tmp_path, source):
    from lqcdweave.adapters.elemental import ArrayElemental
    from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
    from test_baryon_diagram_frontend import _mixed_case, _hand_current_chain, _independent_oracle

    case = _mixed_case((2, 0, 1))
    vertex, raw, terms, _ = fixture(ne=2, times=5)
    reader = ArrayElemental({'v2v': raw}, vertex.reader.components, phase_convention='anchor',
        cfg='cfg-mixed', data_identity='mixed-current/E',
        elemental_structures={('v2v', terms[0].component, 0): 'diagonal'})
    case['vertices']['current'] = Meson(reader, vertex.operator, source, phase_convention='anchor')
    # Raw ArrayPropagator storage is (source_time, sink_time), unlike block().
    effective = _hand_current_chain(case['arrays']['in'][0, 1],
        explicit_vertex(raw, terms, 1, source=source), case['arrays']['out'][1, 4])
    expected = _independent_oracle({**case, 'arrays':
        (case['arrays']['direct0'][0, 4], case['arrays']['direct1'][0, 4], effective)})
    options = dict(vertex_map=case['vertices'], propagator_map=case['propagators'],
        configuration='cfg-mixed', usedNe=2, optimize='none',
        backend='cupy' if array_backend() is not np else 'numpy',
        diagram_adapter=BaryonDiagramBinding(sector_scope='dense-v'),
        memory_limit=1 << 29, host_memory_limit=1 << 29)
    with ContractionTask(case['diagram'], **options) as task:
        task.plan()
        archive = tmp_path / 'baryon-mixed.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive, configuration='cfg-mixed',
            vertex_map=case['vertices'], propagator_map=case['propagators']) as loaded:
        actual = loaded.run().value
    np.testing.assert_allclose(np.asarray(actual).reshape(()), expected, rtol=7e-12, atol=7e-12)


@pytest.mark.parametrize('structure', ('diagonal', 'point_diagonal'))
def test_dense_point_mixed_operator_complete_archive_matches_full_index_trace(tmp_path, structure):
    from dataclasses import replace
    from test_diagonal_current_frontend import (
        DiagonalCurrentFrontendTests, _matrix_from_sector_blocks, _matrix_from_propagator_blocks,
    )
    from lqcdweave.operators.single_hadron.api import Operator, gamma

    points = 4  # four general P endpoints have a nonzero inclusion probability
    context = DiagonalCurrentFrontendTests()._fixture(seed=5215, current_count=2,
        mixed=True, population=points)
    base = context['operators'][0].terms[0]
    other = context['operators'][1].terms[0]
    terms = (replace(base, coefficient=.7+.2j), replace(other, coefficient=-.3+.1j),
             replace(base, gamma=0, coefficient=1e-16j))
    operator = Operator.from_terms('mixed-points', terms)
    reader = context['reader']
    reader.elemental_structures[('p2p', base.component, 0)] = structure
    reader._point_diagonal_component_set = frozenset()
    vertices = {name: Meson(reader, operator, source=True, phase_convention='anchor')
                for name in ('current-0', 'current-1')}
    # Freeze a literal full (V+P) matrix oracle, then poison every declaration's
    # off-support entry in the production reader. No structure-aware E/X is used.
    sectors = {key: value.copy() for key, value in reader.sectors.items()}
    ci = context['component_indices'][base.component]
    for p, c, q, d in np.ndindex(points, 3, points, 3):
        if p != q or (structure == 'diagonal' and c != d):
            sectors['p2p'][ci, 0, :, p, c, q, d] = 0
            reader.sectors['p2p'][ci, 0, :, p, c, q, d] = 70 + 90j
    def current(time):
        return sum(term.coefficient.conjugate() * np.kron(
            gamma(8) @ gamma(term.gamma).conj().T @ gamma(8),
            _matrix_from_sector_blocks(sectors, context['component_indices'][term.component],
                                      time, points).conj().T) for term in terms)
    forward = _matrix_from_propagator_blocks(context['full_propagators'], 0, 2, points)
    backward = _matrix_from_propagator_blocks(context['full_propagators'], 2, 0, points)
    expected = np.trace(forward @ current(0) @ backward @ current(2))
    with ContractionTask(context['diagram'], vertex_map=vertices,
            propagator_map=context['prop_map'], configuration='cfg-diagonal', usedNe=1,
            usedNp=points, optimize='none', backend='cupy' if array_backend() is not np else 'numpy',
            memory_limit=1 << 29, host_memory_limit=1 << 29) as task:
        task.plan()
        archive = tmp_path / f'points-{structure}.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive, configuration='cfg-diagonal',
            vertex_map=vertices, propagator_map=context['prop_map']) as loaded:
        actual = loaded.run().value
    np.testing.assert_allclose(np.asarray(actual).reshape(()), expected, rtol=8e-12, atol=8e-10)


@pytest.mark.parametrize('all_zero', (False, True))
@pytest.mark.parametrize('optimizer', ('none', 'default'))
def test_exact_zero_components_never_read_but_tiny_nonzero_survives(all_zero, optimizer):
    from dataclasses import replace
    from lqcdweave.contraction.correlator import calculate
    from lqcdweave.operators.single_hadron.api import Operator
    request, options, vertex, raw, peram = declarations(ne=2, batch=1)
    if optimizer == 'default':
        options.pop('optimize')
    terms = tuple(replace(term, coefficient=0) if all_zero or i == 1 else term
                  for i, term in enumerate(vertex.operator.terms))
    vertex.operator = Operator.from_terms('zero-test', terms)
    original = vertex.reader.block
    def read(component, *args, **kwargs):
        assert not all_zero and component.key == 0, 'exact zero requested a raw read'
        return original(component, *args, **kwargs)
    with patch.object(vertex.reader, 'block', side_effect=read):
        actual = calculate(request, **options).value
    if all_zero:
        np.testing.assert_array_equal(actual, np.zeros((1, 1, 3)))
    else:
        assert terms[2].coefficient == 1e-16j
        np.testing.assert_allclose(actual, independent_task_value(request, vertex, raw, peram),
                                   rtol=8e-12, atol=8e-12)
