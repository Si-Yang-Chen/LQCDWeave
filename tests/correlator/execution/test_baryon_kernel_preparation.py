"""Actual baryon operation preparation, cleanup and fixed archive replay."""
import gc
from unittest.mock import patch
import weakref

import numpy as np
import pytest

from lqcdweave.contraction.execution.operations.baryon_join import contract_baryon_join
from lqcdweave.contraction.execution.operations import baryon_kernel
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contraction.results.api import assemble_correlator, TermValue
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule
from correlator.execution import test_baryon_sparse_spin as _sparse_fixture
from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries


def timed(phase, fn, **kwargs):
    return fn()


@pytest.mark.parametrize('no_search', (False, True))
def test_actual_operation_prepares_once_and_compute_does_not_pack_or_select(no_search):
    fixture = _sparse_fixture.SparseBaryonSpinTests()
    spin = fixture.spin()
    arrays = fixture.arrays(spin)
    expected = fixture.reference(arrays)
    action = fixture.action(spin.entries)
    prepared_refs, events = [], []
    original_prepare = baryon_kernel.BaryonKernelBinding.prepare

    def prepare(binding, *args):
        value = original_prepare(binding, *args)
        prepared_refs.append(weakref.ref(value))
        events.append(('prepare', binding.implementation))
        return value

    def record(phase, fn, **kwargs):
        if phase == 'baryon_join_kernel':
            # Neither compatibility dispatch nor Phi packing may be deferred
            # into compute after preparation has produced the current owner.
            with (patch.object(baryon_kernel, 'sparse_spin_recipe', side_effect=AssertionError('selected in compute')),
                  patch.object(baryon_kernel, 'prepare_sparse_spin_elemental', side_effect=AssertionError('packed in compute'))):
                value = fn()
            events.append(('compute', phase))
            return value
        return fn()

    with patch.object(baryon_kernel.BaryonKernelBinding, 'prepare', prepare):
        contract_baryon_join(action, arrays, np, timed=record, no_search=no_search)
    np.testing.assert_allclose(arrays['result'], expected, rtol=3e-13, atol=3e-13)
    assert events == [('prepare', 'saved-sparse-spin'), ('compute', 'baryon_join_kernel')]
    gc.collect()
    assert all(ref() is None for ref in prepared_refs)


def test_failed_actual_compute_clears_current_preparation_owners():
    fixture = _sparse_fixture.SparseBaryonSpinTests()
    spin = fixture.spin()
    arrays = fixture.arrays(spin)
    references = tuple(weakref.ref(value) for value in arrays.values())
    closed = []
    original_close = baryon_kernel.PreparedBaryonKernel.close

    def close(prepared):
        original_close(prepared)
        closed.append((prepared.closed, prepared.spin, prepared.spatial, prepared.packed))

    with (patch.object(baryon_kernel.PreparedBaryonKernel, 'compute', side_effect=RuntimeError('device submission failed')),
          patch.object(baryon_kernel.PreparedBaryonKernel, 'close', close)):
        with pytest.raises(RuntimeError, match='submission failed'):
            contract_baryon_join(fixture.action(spin.entries), arrays, np, timed=timed)
    assert closed == [(True, None, None, None)]
    assert 'result' not in arrays
    del arrays
    gc.collect()
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize('measure', (False, True))
@pytest.mark.parametrize('multiple_components', (False, True))
@pytest.mark.parametrize('source_slots', ((0, 1, 2), (2, 0, 1)))
def test_native_fixed_storage_archive_replays_current_values_without_path_search(
        tmp_path, measure, multiple_components, source_slots):
    case = _case(source_slots)
    extra_case = None
    if multiple_components:
        sink = case['vertices']['sink']
        first = sink.operator.terms[0]
        extra = OperatorTerm.from_baryon(first.component, _spin_entries(8),
                                        momentum=first.momentum, coefficient=-.2+.3j)
        sink.operator = Operator.from_terms('sum-sink', (first, extra))
        extra_case = dict(case, sink_spin=extra.spin.dense(), sink_coefficient=extra.coefficient)
    expanded = expand_correlator(case['diagram'], adapter=BaryonDiagramBinding(sector_scope='dense-v', vertices='joined'),
        bindings=DiagramBindings(case['propagators'], case['vertices'], 'cfg-a'),
        time_axes=None, time_pairs=None, batch='auto')
    planned = plan_direct_execution(expanded, memory_limit=1 << 26).plan
    original_actions = tuple((a.kernel_path, a.sparse_spin, a.core)
                             for a in planned.state.actions if a.kind == 'baryon_join')
    assert len(original_actions) == 2
    path = tmp_path/'fixed-baryon-storage.json.gz'
    shapes = {key: list(spec.shape) for key, spec in planned.state.problem.specs.items()}
    hardware = {'backend': 'numpy', 'memory': planned.memory_limit}
    save_storage_schedule(path, (planned,), shapes=shapes, hardware=hardware)
    with (patch('opt_einsum.paths.get_path_fn', side_effect=AssertionError('replay searched')),
          patch('opt_einsum.paths.greedy', side_effect=AssertionError('replay chose path')),
          patch('lqcdweave.contraction.planning.state._prepared_join_path', side_effect=AssertionError('P reselected'))):
        restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
            expected=(planned.state.problem,), blocks=planned.blocks, preparations=planned.preparations)
        restored = restored[0]
        assert tuple((a.kernel_path, a.sparse_spin, a.core)
                     for a in restored.state.actions if a.kind == 'baryon_join') == original_actions
        previous = None
        for repeat in range(2):
            if repeat:
                # New current numeric payloads under this fixed synthetic data
                # binding. This is not a claim of native opaque cfg rebinding.
                for array in case['arrays']:
                    array *= 1.1-.2j
            expected = _independent_oracle(case)
            if extra_case is not None:
                expected += _independent_oracle(extra_case)
            result = replay_scene_storage(restored, expanded.inputs.loader, measure=measure, io_overlap=False)
            values = result.values if measure else result
            actual = assemble_correlator(expanded, tuple(TermValue(term,
                expanded.coordinates.points[0].key, value)
                for term, value in zip(expanded.inputs.term_ids, values, strict=True)))
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-10)
            if previous is not None:
                assert not np.allclose(previous, actual)
            previous = actual
            if measure:
                phases = [p.name for timing in result.timings for p in timing.phases]
                assert phases.count('baryon_prepare') == phases.count('baryon_join_kernel')
                assert phases.count('baryon_prepare') >= 2
