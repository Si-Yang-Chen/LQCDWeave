from __future__ import annotations

import pytest

from lqcdweave.contracts.correlator.api import (
    BaryonLine as ContractLine,
    BaryonTerm as ContractTerm,
    BaryonTopology as ContractTopology,
    BaryonVertex as ContractVertex,
)
from lqcdweave.contraction import BaryonTerm as RootTerm
from lqcdweave.contraction import BaryonVertex as RootVertex



def test_baryon_term_records_have_one_public_class_identity() -> None:
    assert ContractTerm is RootTerm
    assert ContractVertex is RootVertex


def test_topology_record_preserves_line_order_and_source_slots() -> None:
    lines = (
        ContractLine(11, 2, 0),
        ContractLine(12, 0, 1),
        ContractLine(13, 1, 2),
    )

    topology = ContractTopology(0, 1, lines)

    assert type(topology) is ContractTopology
    assert all(type(line) is ContractLine for line in topology.lines)
    assert tuple(line.propagator for line in topology.lines) == (11, 12, 13)
    assert tuple(line.source_slot for line in topology.lines) == (2, 0, 1)


@pytest.mark.parametrize(
    "line_kwargs",
    [
        {"propagator": True, "source_slot": 0, "sink_slot": 0},
        {"propagator": 1, "source_slot": 3, "sink_slot": 0},
        {"propagator": 1, "source_slot": 0, "sink_slot": -1},
    ],
)
def test_line_record_rejects_invalid_ids_and_slots(line_kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ContractLine(**line_kwargs)


def test_topology_requires_sink_order_and_source_bijection() -> None:
    lines = (
        ContractLine(1, 0, 0),
        ContractLine(2, 0, 1),
        ContractLine(3, 2, 2),
    )

    with pytest.raises(ValueError, match="source slots"):
        ContractTopology(0, 1, lines)


def test_baryon_term_record_keeps_coefficient_and_source_slots() -> None:
    term = ContractTerm(
        ContractVertex("source-spin", "source-elemental"),
        ContractVertex("sink-spin", "sink-elemental"),
        ("p0", "p1", "p2"),
        source_slots=(2, 0, 1),
        coefficient=1.25 + 0.5j,
    )

    assert term.source_slots == (2, 0, 1)
    assert term.coefficient == 1.25 + 0.5j

