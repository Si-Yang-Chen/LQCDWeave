"""Managed baryon production survives complete-plan save/load and fixed replay."""
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.correlator import ContractionTask, Elemental
from lqcdweave.operators.single_hadron.api import (
    Operator, OperatorTerm, SparseSpinTensor, baryon_preset,
)
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.storage.dataset import DatasetReader
from test_baryon_elemental import determinant_reference
from test_baryon_quark_exchange import request as symbolic_request
from test_production_dataset import _ProductionDatasetFixture
from workflows.test_contraction_task_plan import no_planning


@pytest.mark.parametrize('backend', ['numpy', pytest.param('cupy', id='gpu')])
def test_complete_archive_first_replay_prepares_all_times_then_reuses(backend):
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        assert float(cp.asnumpy(cp.arange(3).sum())) == 3
    fixture = _ProductionDatasetFixture()
    fixture.setUp()
    try:
        gauge = DatasetReader(fixture.root/'gauge', cfg='cfg')
        basis = DatasetReader(fixture.root/'basis', cfg='cfg')
        fixture.addCleanup(gauge.close)
        fixture.addCleanup(basis.close)
        table = baryon_preset().select(('local',))

        def reader():
            result = Elemental(gauge, basis, table, {0: (0, 0, 0)},
                               directory=fixture.root/'managed')
            fixture.addCleanup(result.close)
            return result

        original = reader()
        weights = {'sink': .7+.2j, 'source': -.3+.4j}
        operators = {name: Operator.from_terms(name, (OperatorTerm.from_baryon(
            original.components[0], SparseSpinTensor(((0, 0, 0, 1),)),
            coefficient=weight),)) for name, weight in weights.items()}

        def bind(row):
            return Baryon(original, operators[row.hadron_name], source=row.dagger)

        rng = np.random.default_rng(7162)
        data = (rng.normal(size=(3, 3, 4, 4, 3, 3))
                + 1j*rng.normal(size=(3, 3, 4, 4, 3, 3)))/3
        prop = ArrayPropagator({'vsv': data}, time_convention='absolute', cfg='cfg',
            basis_endpoints={('vsv', side): (fixture.bid, (4, 1, 9))
                             for side in ('sink', 'source')})
        points = ({'sink': 0, 'source': 2}, {'sink': 2, 'source': 1},
                  {'sink': 1, 'source': 1})
        spatial = [sum(complex(*term['coefficient'])*determinant_reference(
            fixture.gauge, (fixture.vectors,)*3, term['paths'], time, (0, 0, 0))
            for term in table[0]['terms']) for time in range(3)]
        expected = []
        for point in points:
            sink, source = point['sink'], point['source']
            effective = np.einsum('b,bmn->mn', gamma(8)[:, 0],
                                  data[source, sink, 0], optimize=False)
            expected.append(36*np.linalg.det(effective)*spatial[sink][0, 1, 2]
                *spatial[source][0, 1, 2].conjugate()
                *weights['sink']*weights['source'].conjugate())
        expected = np.asarray(expected)
        archive = fixture.root/'complete.json.gz'
        with ContractionTask(symbolic_request('antisymmetric', structured=True),
                vertex_map=bind, propagator_map={'S^q': prop}, configuration='cfg',
                dtype='complex128', backend=backend, time_pairs=points, batch=2,
                optimize='none', memory_limit=1 << 28, host_memory_limit=1 << 28) as task:
            with patch.object(gauge, 'read', side_effect=AssertionError('planning read')):
                task.plan()
                task.save_plan(archive)
            assert task._selected.plan.statistics['coordinate_batch_sizes'] == (2, 1)
            assert not (fixture.root/'managed').exists()

        current = reader()
        with no_planning(), ContractionTask.load_plan(archive, configuration='cfg',
                vertex_map=current, propagator_map={'S^q': prop}) as loaded:
            with patch.object(current, '_produce', wraps=current._produce) as produce:
                actual = loaded.run().value
            assert sorted(call.args[3] for call in produce.call_args_list) == [0, 1, 2]
            assert all(call.args[4].__name__ == backend for call in produce.call_args_list)
            np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected,
                                       atol=3e-10, rtol=3e-12)
            data *= .5
            with patch.object(current, '_produce', side_effect=AssertionError('reproduction')):
                second = loaded.run().value
            np.testing.assert_allclose(np.asarray(second).reshape(-1), expected/8,
                                       atol=3e-10, rtol=3e-12)
    finally:
        fixture.doCleanups()
