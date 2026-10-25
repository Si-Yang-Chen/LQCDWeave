"""Reuse static baryon spin descriptions across Wick graphs in one expansion."""

from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contraction.results.api import TermValue, assemble_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.correlator.api import Diagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.baryon import SparseSpinTensor

from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries


def test_repeated_graphs_describe_each_source_spin_once_and_keep_components_distinct():
    case = _case()
    source, sink = case['vertices']['source'], case['vertices']['sink']
    source_first, = source.operator.terms
    sink_first, = sink.operator.terms
    source_extra = OperatorTerm.from_baryon(
        source_first.component, _spin_entries(9), momentum=source_first.momentum,
        coefficient=0.2 + 0.6j)
    sink_extra = OperatorTerm.from_baryon(
        sink_first.component, _spin_entries(11), momentum=sink_first.momentum,
        coefficient=-0.7 + 0.1j)
    source.operator = Operator.from_terms('source-sum', (source_first, source_extra))
    sink.operator = Operator.from_terms('sink-sum', (sink_first, sink_extra))

    first = case['diagram']
    second = Diagram(first.diagram, first.time_list,
                     ['source-copy', 'sink-copy'], first.propagator_list)
    vertices = dict(case['vertices'], **{'source-copy': source, 'sink-copy': sink})
    calls = []
    described_propagators = []
    original_source = SparseSpinTensor.source
    original_describe = ArrayPropagator.describe_block

    def count_source(spin):
        calls.append(spin)
        return original_source(spin)

    def count_propagator(reader, sink_time, source_time, sector='vsv', **selection):
        described_propagators.append((reader, sink_time, source_time, sector, selection))
        return original_describe(reader, sink_time, source_time, sector, **selection)

    with (patch.object(SparseSpinTensor, 'source', count_source),
          patch.object(ArrayPropagator, 'describe_block', count_propagator)):
        expanded = expand_correlator(
            first + second, adapter=BaryonDiagramBinding(sector_scope='dense-v'),
            bindings=DiagramBindings(case['propagators'], vertices, 'cfg-a'),
            time_axes=None, time_pairs=None, batch='auto')
    assert len(calls) == 2
    assert set(calls) == {source_first.spin, source_extra.spin}
    assert len(described_propagators) == 3
    assert {id(call[0]) for call in described_propagators} == {
        id(reader) for reader in case['propagators'].values()}

    planned = plan_direct_execution(expanded, memory_limit=1 << 26).plan
    values = replay_scene_storage(planned, expanded.inputs.loader, io_overlap=False)
    actual = assemble_correlator(expanded, tuple(
        TermValue(term, expanded.coordinates.points[0].key, value)
        for term, value in zip(expanded.inputs.term_ids, values, strict=True)))
    expected = 0j
    for source_term in source.operator.terms:
        for sink_term in sink.operator.terms:
            component = dict(case, source_spin=source_term.spin.dense(),
                             sink_spin=sink_term.spin.dense(),
                             source_coefficient=source_term.coefficient,
                             sink_coefficient=sink_term.coefficient)
            expected += _independent_oracle(component)
    np.testing.assert_allclose(actual, 2 * expected, rtol=3e-12, atol=1e-10)
