"""Whole vertices: actual native planning, numerical replay and archive ABI."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.execution.operations.baryon_kernel import bind_baryon_kernel
from lqcdweave.contraction.expansion.api import expand_correlator, normalize_contraction_precision
from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
from lqcdweave.contraction.kernel_descriptions import attach_vertex_kernel_descriptions
from lqcdweave.contraction.performance.api import estimate_vertex_kernel
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contracts.correlator.api import (
    AxisPattern, DiagramBindings, TensorSpec, VertexKernelDescription, parse_axis_pattern,
)
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.storage.contraction_plans.api import (
    decode_plan_value, encode_plan_value, load_storage_schedule, save_storage_schedule,
)
from test_baryon_diagram_frontend import _case, _independent_oracle


def meson_case(dtype=np.complex128, *, trace=False):
    rng = np.random.default_rng(19442)
    components = tuple(ElementalComponent(i) for i in (0, 1))
    raw = rng.normal(size=(2, 1, 2, 2, 2))
    peram = rng.normal(size=(2, 2, 4, 4, 2, 2))
    if np.dtype(dtype).kind == 'c':
        raw = raw + 1j * rng.normal(size=raw.shape)
        peram = peram + 1j * rng.normal(size=peram.shape)
    raw, peram = raw.astype(dtype), peram.astype(np.result_type(dtype, np.complex64))
    reader = ArrayElemental({'v2v': raw}, components, phase_convention='anchor',
                            cfg='whole-cfg', data_identity='whole-cfg/E')
    prop = ArrayPropagator({'vsv': peram}, time_convention='absolute',
                           cfg='whole-cfg', data_identity='whole-cfg/P')
    terms = (OperatorTerm(0, components[0], 0, 1+.2j),
             OperatorTerm(1, components[1], 0, -.3+.7j),
             OperatorTerm(2, components[0], 0, 1e-16j))
    operator = Operator.from_terms('whole', terms)
    q = Propagator('u', Tag(0, 0), Tag(1, 0))
    if trace:
        diagram = Diagram(QuarkDiagram([[1]]), [0], ['sink'], [None, q])
    else:
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['sink', 'source'], [None, q])
    vertices = {'sink': Meson(reader, operator, phase_convention='anchor'),
                'source': Meson(reader, operator, True, phase_convention='anchor')}
    expanded = expand_correlator(diagram, adapter=LegacyDiagramBinding(),
        bindings=DiagramBindings({q: prop}, vertices, 'whole-cfg', usedNe=2),
        time_axes=None, time_pairs=None, batch='auto')
    return expanded, raw, peram, terms


def meson_reference(raw, peram, terms):
    # Explicit scalar products retain original ordered spin/spatial pairings.
    sink = np.zeros((4, 4, 2, 2), np.complex128)
    source = np.zeros_like(sink)
    for term in terms:
        index = term.component.key
        spin = gamma(term.gamma)
        sink += term.coefficient * np.einsum('ab,ij->abij', spin, raw[index, 0, 0])
        source += term.coefficient.conjugate() * np.einsum('ab,ij->abij',
            gamma(8) @ spin.conj().T @ gamma(8), raw[index, 0, 1].conj().T)
    return np.einsum('BAON,DCQP,ADNQ,CBPO->', peram[0, 1], peram[1, 0], sink, source,
                     optimize=False)


def test_complete_meson_vertex_trace_uses_one_actual_propagator():
    expanded, raw, peram, terms = meson_case(trace=True)
    plan = plan_direct_execution(expanded, memory_limit=1 << 26).plan
    full = sum(term.coefficient * np.einsum('ab,ij->abij', gamma(term.gamma),
               raw[term.component.key, 0, 0]) for term in terms)
    expected = np.einsum('baji,abij->', peram[0, 0], full, optimize=False)
    values = replay_scene_storage(plan, expanded.inputs.loader, io_overlap=False)
    assert len(expanded.terms[0].inputs) == 2
    np.testing.assert_allclose(values[0], expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.complex64, np.complex128))
def test_native_complete_sum_archive_replay_uses_current_data(tmp_path, dtype):
    expanded, raw, peram, terms = meson_case(dtype)
    assert len(expanded.terms) == 1  # 3 x 3 components remain inside two sources.
    assert sum(prep.recipe.kind == 'linear_combination' for prep in expanded.inputs.preparations.values()) == 2
    expanded = normalize_contraction_precision(expanded)
    plan = plan_direct_execution(expanded, memory_limit=1 << 26).plan
    shapes = {key: list(spec.shape) for key, spec in plan.state.problem.specs.items()}
    hardware = {'backend': 'numpy'}
    path = tmp_path / 'whole.json.gz'
    save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
    restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
        expected=(plan.state.problem,), blocks=plan.blocks, preparations=plan.preparations)
    tolerance = 2e-5 if np.dtype(dtype).itemsize <= 8 and np.dtype(dtype) not in (np.float64,) else 3e-12
    for repeat in range(2):
        if repeat:
            raw *= .7
            peram *= 1.1
        with patch('opt_einsum.paths.greedy', side_effect=AssertionError('archive searched')):
            values = replay_scene_storage(restored[0], expanded.inputs.loader, io_overlap=False)
        np.testing.assert_allclose(values[0], meson_reference(raw, peram, terms),
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('packed', (False, True))
def test_new_baryon_plans_have_neutral_description_and_no_global_internal_path(tmp_path, packed):
    case = _case((2, 0, 1))
    from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
    expanded = expand_correlator(case['diagram'], adapter=BaryonDiagramBinding(sector_scope='dense-v', vertices='joined'),
        bindings=DiagramBindings(case['propagators'], case['vertices'], 'cfg-a'),
        time_axes=None, time_pairs=None, batch='auto')
    expanded = attach_vertex_kernel_descriptions(expanded)
    with (patch('lqcdweave.contraction.planning.state._prepared_join_path', side_effect=AssertionError('P searched')),
          patch('lqcdweave.contraction.planning.state._direct_join_path', side_effect=AssertionError('P lowered local program')),
          patch('lqcdweave.contraction.planning.state._join_path_cost', side_effect=AssertionError('P cost local path'))):
        plan = plan_direct_execution(expanded, memory_limit=1 << 26).plan
    actions = [action for action in plan.state.actions if action.kind == 'baryon_join']
    assert actions and all(not action.kernel_path and action.kernel_description is not None for action in actions)
    if packed:
        plan = decode_plan_value(encode_plan_value(plan))
    values = replay_scene_storage(plan, expanded.inputs.loader, io_overlap=False, measure=packed)
    values = values.values if packed else values
    np.testing.assert_allclose(values[0], _independent_oracle(case), rtol=3e-12, atol=1e-10)


@pytest.mark.parametrize('inputs', ((('a','b','c','i','j','k'),),
    (('a','b','i','j','x'), ('c','k','x'))))
def test_baryon_general_topology_one_input_or_shared_legs_without_cost_replay(inputs):
    descriptor = VertexKernelDescription('baryon', 'ordered-join-v1')
    axes = (('a','b','c'), ('i','j','k'), *inputs)
    equation = parse_axis_pattern(','.join(''.join(labels) for labels in axes) + '->')
    dimensions = dict.fromkeys('abcijkx', 2)
    dimensions.update(dict.fromkeys('abc', 4))
    rng = np.random.default_rng(119)
    arrays = tuple(rng.normal(size=tuple(dimensions[label] for label in labels)).astype(np.complex128)
                   for labels in axes)
    specs = tuple(TensorSpec(array.shape, array.dtype) for array in arrays)
    assert estimate_vertex_kernel(descriptor, equation, specs)[2] > 0
    with patch('lqcdweave.contraction.performance.api.estimate_vertex_kernel',
               side_effect=AssertionError('execution must not reprice the kernel')):
        binding = bind_baryon_kernel(equation, tuple(v.shape for v in arrays),
            tuple(str(v.dtype) for v in arrays), (), None, False, True, descriptor)
    prepared = binding.prepare(*arrays[:2], np)
    try:
        actual, _ = prepared.compute(arrays[2:])
    finally:
        prepared.close()
    from lqcdweave.contraction.execution.indices import einsum_equation
    expected = np.einsum(einsum_equation(equation), *arrays, optimize=False)
    np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)
