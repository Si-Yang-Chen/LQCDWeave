"""B's neutral preparations retain source adjoints, precision and owned values."""
import numpy as np
import pytest

from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings, SourcePreparation, TensorSpec
from test_baryon_diagram_frontend import _case


@pytest.mark.parametrize('source', (False, True))
def test_neutral_baryon_preparation_matches_b_definition_and_owns_current_input(source):
    case = _case((2, 0, 1))
    vertex = case['vertices']['source' if source else 'sink']
    description = vertex.describe_factors(0)[0]
    raw = vertex.read_factor_block(description)
    expected = vertex.materialize_unweighted_elemental(description, raw)
    recipe = vertex.describe_elemental_preparation(description, 'raw')
    spin = vertex.describe_spin_preparation(description)
    specs = {'phi': TensorSpec(raw.shape, np.complex128),
             'spin': TensorSpec((4, 4, 4), np.complex128)}
    preps = {'phi': SourcePreparation(('raw',), recipe=recipe),
             'spin': SourcePreparation((), recipe=spin)}
    loader = bind_prepared_loader(None, preps, specs, backend=np)
    actual = loader.materialize('phi', None, {'raw': raw})
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype('complex128') and not np.shares_memory(actual, raw)
    assert loader.materializes_owned('phi')
    np.testing.assert_array_equal(loader.materialize('spin', None, {}),
                                  vertex.materialize_spin_factor(description))
    raw *= .7
    np.testing.assert_array_equal(actual, expected)


def test_actual_baryon_expansion_delivers_only_neutral_spin_phi_preparations():
    case = _case((2, 0, 1))
    expanded = expand_correlator(case['diagram'], adapter=BaryonDiagramBinding(sector_scope='dense-v'),
        bindings=DiagramBindings(case['propagators'], case['vertices'], 'cfg-a'),
        time_axes=None, time_pairs=None, batch='auto')
    for join in expanded.problem.baryon_joins:
        for part in join.core.components:
            spin, phi = (expanded.inputs.preparations[key].recipe for key in part.inputs)
            assert spin.kind == 'constant' and phi.kind == 'identity'
            assert phi.compute_dtype == 'complex128'
