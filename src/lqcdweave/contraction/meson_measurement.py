"""Read-only Meson planning and shared row writers."""


import numpy as np

from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.contraction.meson import connected_meson_pair
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.planning.meson import validate_meson_inputs, meson_measurement_document


def _operator_snapshot(operator, reader):
    """Resolve positional operator references against explicit stored coordinates."""
    terms, definition = [], []
    for term in operator.terms:
        if (term.profile is not None or term.anchor_time_offset != 0 or
                term.component.time_offsets != (0, 0) or
                not isinstance(term.component.key, GaugePath)):
            raise ValueError("workflow supports equal-time raw paths without spectral profiles")
        if term.component not in reader.components or not 0 <= term.momentum < len(reader.momenta):
            raise ValueError("operator needs missing elemental coverage")
        terms.append(OperatorTerm(term.gamma, term.component, term.momentum, term.coefficient))
        definition.append(dict(gamma=term.gamma, path=list(term.component.key.steps),
                               momentum=reader.momenta[term.momentum],
                               coefficient=[term.coefficient.real, term.coefficient.imag]))
    if not terms:
        raise ValueError("empty operator")
    return Operator.from_terms(operator.name, terms), definition


def prepare_meson_stage(datasets, sink_operator, source_operator):
    """Validate registered inputs and build the shared plan/run contract."""
    fwd, rev, sink_data, source_data = datasets
    times = validate_meson_inputs(*(a.spec for a in datasets))
    sink_reader, source_reader = DatasetElemental(sink_data), DatasetElemental(source_data)
    sink_op, sink_def = _operator_snapshot(sink_operator, sink_reader)
    source_op, source_def = _operator_snapshot(source_operator, source_reader)
    sink = Meson(sink_reader, sink_op, phase_convention=sink_reader.phase_convention)
    source = Meson(source_reader, source_op, source=True,
                   phase_convention=source_reader.phase_convention)
    spec = DatasetSpec(meson_measurement_document(fwd.spec.document["configuration"], times,
        [a.spec.document for a in datasets], [a.data_id for a in datasets], sink_def, source_def))
    return spec, sink, source, times


def write_meson_rows(datasets, sink_operator, source_operator, writer):
    """Fill a graph-owned writer with the same checked full-time numerical rows.

    Inputs and writer are borrowed; the caller owns publication and their
    lifetime. Only writer.missing rows are computed and saved synchronously.
    """
    spec, sink, source, times = prepare_meson_stage(datasets, sink_operator, source_operator)
    if writer.spec.definition_id != spec.definition_id:
        raise ValueError("meson writer definition differs from its actual input views")
    _write_meson_rows(datasets, sink, source, times, writer)


def _write_meson_rows(datasets, sink, source, times, writer):
    fwd, rev, _, _ = datasets
    for index in writer.missing:
        s = times[index]
        source_factors = tuple(source.factors(s))
        row = np.empty(len(times), dtype=np.complex128)
        for t in times:
            row[t] = connected_meson_pair(
                sink.factors(t), source_factors,
                fwd.read(sink_time=t, source_time=s),
                rev.read(sink_time=s, source_time=t))
        writer.write(index, row)
