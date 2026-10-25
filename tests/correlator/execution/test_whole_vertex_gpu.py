"""Optional real CUDA acceptance; CPU protocol doubles do not satisfy these tests."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import cupy as cp
except ImportError as error:
    from unittest import SkipTest
    raise SkipTest('CuPy is required for real GPU replay tests') from error

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.execution.api import execute_contraction_path
from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.expansion.api import expand_correlator, normalize_contraction_precision
from lqcdweave.contraction.expansion.batching import BatchDiagramBinding
from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
from lqcdweave.contraction.planning.api import plan_direct_execution, retarget_expanded
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.storage.contraction_plans.api import save_contraction_plan, load_contraction_plan
from correlator.execution.test_whole_vertex_integration import meson_case, meson_reference


def _saved(path, tmp_path):
    # Complete path archives retain each batch shape, including a short tail.
    children = tuple(replace(child, execution_mode='planned') for child in path.batch_paths)
    selected = replace(path, execution_mode='planned', batch_paths=children)
    hardware = {'backend': 'cupy'}
    destination = tmp_path / 'real-gpu.json.gz'
    save_contraction_plan(destination, selected, hardware=hardware)
    return load_contraction_plan(destination, expected=path.expanded, hardware=hardware)


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.complex64, np.complex128))
def test_real_native_whole_meson_saved_replay_and_current_data(tmp_path, dtype):
    cp.cuda.runtime.getDeviceProperties(0)
    expanded, raw, peram, terms = meson_case(dtype)
    target = retarget_expanded(normalize_contraction_precision(expanded), 'cupy')
    path = _saved(plan_direct_execution(target, memory_limit=1 << 27, host_memory_limit=1 << 27), tmp_path)
    tolerance = 3e-5 if np.dtype(dtype) in (np.dtype('float32'), np.dtype('complex64')) else 4e-12
    for repeat in range(2):
        if repeat:
            raw *= .7
            peram *= 1.1
        result = execute_contraction_path(target, path, backend=cp, io_overlap=True)
        actual = assemble_correlator(target, result.term_values)
        np.testing.assert_allclose(actual, meson_reference(raw, peram, terms),
                                   rtol=tolerance, atol=tolerance)


def test_real_whole_meson_multi_raw_batch_tail_archive_and_changed_data(tmp_path):
    rng = np.random.default_rng(261022)
    def random(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    raw, peram = random((2, 1, 3, 2, 2)), random((3, 3, 4, 4, 2, 2))
    components = tuple(ElementalComponent(i) for i in (0, 1))
    elemental = ArrayElemental({'v2v': raw}, components, phase_convention='anchor',
                              data_identity='real-gpu-whole/E')
    propagator = ArrayPropagator({'vsv': peram}, time_convention='absolute',
                                data_identity='real-gpu-whole/P')
    terms = (OperatorTerm(0, components[0], 0, 1+.2j),
             OperatorTerm(1, components[1], 0, -.3+.7j),
             OperatorTerm(2, components[0], 0, 1e-16j))
    operator = Operator.from_terms('whole-batch', terms)
    q = Propagator('u', Tag(0, 0), Tag(1, 1))
    graph = Diagram(QuarkDiagram([[0, 1], [1, 0]]), ['sink', 'source'], ['a', 'b'], [None, q])
    bindings = DiagramBindings({q: propagator}, {
        'a': Meson(elemental, operator, phase_convention='anchor'),
        'b': Meson(elemental, operator, True, phase_convention='anchor')}, usedNe=2)
    expanded = expand_correlator(graph, adapter=BatchDiagramBinding(LegacyDiagramBinding()), bindings=bindings,
        time_axes={'sink': range(3), 'source': 1}, batch=2)
    target = retarget_expanded(normalize_contraction_precision(expanded), 'cupy')
    assert [item.coordinate_batch.actual_size for item in target.inputs.task_batches] == [2, 1]
    path = _saved(plan_direct_execution(target, memory_limit=1 << 27, host_memory_limit=1 << 27), tmp_path)
    for repeat in range(2):
        if repeat:
            raw *= .7
            peram *= 1.1
        vertices = []
        for t in range(3):
            vertices.append(sum(term.coefficient * np.einsum('ab,ij->abij', gamma(term.gamma),
                            raw[term.component.key, 0, t]) for term in terms))
        source = sum(term.coefficient.conjugate() * np.einsum('ab,ij->abij',
            gamma(8) @ gamma(term.gamma).conj().T @ gamma(8), raw[term.component.key, 0, 1].conj().T)
                     for term in terms)
        expected = np.asarray([np.einsum('BAON,DCQP,ADNQ,CBPO->', peram[t, 1], peram[1, t],
                                      vertices[t], source, optimize=False) for t in range(3)])
        result = execute_contraction_path(target, path, backend=cp, io_overlap=True)
        np.testing.assert_allclose(assemble_correlator(target, result.term_values), expected,
                                   rtol=5e-12, atol=5e-12)


def test_real_upload_and_consumer_completion_keep_current_owners():
    props = cp.cuda.runtime.getDeviceProperties(0)
    delay = cp.RawKernel('extern "C" __global__ void delay(unsigned long long ticks) {'
        'unsigned long long start=clock64(); while(clock64()-start<ticks) {} }', 'delay')
    # Compile/warm before the queued dependency test, without timing assertions.
    delay((1,), (1,), (np.uint64(1),))
    cp.cuda.get_current_stream().synchronize()
    host = np.arange(1 << 15, dtype=np.float64)
    with device_execution(cp, device_memory_limit=1 << 26) as device:
        warm = device.upload(host)
        device.complete_upload(warm, wait=True)
        with device.upload_stream:
            delay((1,), (1,), (np.uint64(props['clockRate'] * 100),))
        ticket = device.upload(host, background=True)
        assert ticket.event is not None and not device.complete_upload(ticket)
        assert ticket.pinned is not None
        value = device.ready(ticket)
        assert not ticket.complete and ticket.pinned is not None
        device.complete_upload(ticket, wait=True)
        assert ticket.complete and ticket.pinned is None
        np.testing.assert_array_equal(cp.asnumpy(value), host)
        retired = []
        queue = SimpleNamespace(device_capacity=1 << 26, _stopping=False,
                                retire_device_input=lambda owner, item: retired.append(item) or True)
        delay((1,), (1,), (np.uint64(props['clockRate'] * 100),))
        device.release_input(value, queue)
        assert len(retired) == 1 and retired[0].value is value
        assert not device.complete_input_retirement(retired[0])
        device.complete_input_retirement(retired[0], wait=True)
        assert retired[0].value is None
