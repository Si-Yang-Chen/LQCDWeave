"""Test-owned full-time two-point composition; not an installed API."""

from functools import partial

from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron import Operator, OperatorTerm


def prepare_two_point(elemental, perambulator, *, Lt, Ne=None):
    """Define the full-time task once, then call compute(configuration=cfg)."""
    operator = Operator.from_terms("local", [OperatorTerm(15, ElementalComponent(0), 0)])
    source = Meson(elemental, operator, True)
    sink = Meson(elemental, operator)
    # The minus sign is the explicit connected Wick-loop coefficient.
    diagram = -Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                       ["source", "sink"], [None, "q"])
    return partial(
        calculate, diagram, propagator_map={"q": perambulator},
        vertex_map={"source": source, "sink": sink},
        usedNe=Ne, time_axes={0: range(Lt), 1: range(Lt)},
        optimize="none",
    )


def two_point(elemental, perambulator, cfg, Ne=None):
    """Single-cfg convenience; component/momentum row zero has declared meaning."""
    compute = prepare_two_point(elemental, perambulator, Lt=perambulator.load(cfg).Lt, Ne=Ne)
    return compute(configuration=cfg).value
