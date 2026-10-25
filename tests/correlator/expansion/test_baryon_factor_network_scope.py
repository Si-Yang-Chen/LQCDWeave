"""Every baryon request opens its vertices unless ``"joined"`` is declared.

Each case changes one property of a plain row request (its control): a Meson
current or a glueball beside the rows, quark-exchange multiplet labels, mode
blocks, a second Phi or a second leg binding in one row, a single row, a time
scan. Every case runs all five ``baryon_vertices`` lowerings; values are
checked against references that do not depend on the row packing and against
each other. A Meson stands in the current slot until the current module is
settled. Timings of the same cases: ``benchmarks/baryon_vertex_scope.py``.
"""
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Callable
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.contraction.binding.api import Baryon, Glueball
from lqcdweave.contraction.correlator import calculate
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, HadronIrrep, QuarkExchange, Tag, gen_correlator, split_baryon_eigenmodes,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, baryon_spin_operators
from lqcdweave.operators.single_hadron.baryon import BaryonOperator, BaryonOperatorTerm

import test_baryon_mixed_exchange as exchange
import test_baryon_quark_exchange as uuu
from correlator.test_glueball import glue_input, particle
from test_baryon_diagram_frontend import _independent_oracle, _spin_entries
from correlator.expansion.test_baryon_factor_network import _expand, _row_case, _scan_case

LOWERINGS = ("auto", "pair", "nonzero", "dense", "joined")


@dataclass
class ScopeCase:
    request: object
    options: dict          # calculate/ContractionTask declarations, without baryon_vertices
    expected: np.ndarray   # packing-free reference of reduce(value)
    reduce: Callable = field(default=np.asarray)
    readers: tuple = ()    # elemental readers behind a callable vertex_map


def _uuu_rows(*, current=False, glueball=False, symmetry="antisymmetric"):
    """G1 uuu rows at both ends, optionally with a Meson between them or a glueball in each row."""
    bind_fixture, propagator, readers, _ = uuu.fixture()
    glue = glue_input([2 + 3j, 5 - 7j, -1 + 4j], cfg="exchange-cfg")
    g = particle("G", "", 1)
    recipe = {r.name: r for r in baryon_spin_operators()}["spin_G1g_S_copy0"]
    component = BaryonComponent.paths(((), (), ()))
    vertices, sides = {}, {}
    for side, source in (("sink", False), ("source", True)):
        sides[side] = []
        for r, spin in enumerate(recipe.rows):
            name = f"{side}_r{r}"
            row = HadronIrrep(name, (0, 0, 0), "G1", 1, Tag(0, 0))[r]
            hadron = Hadron(row, Flavor("uuu"), quark_symmetry=symmetry)
            sides[side].append(Hadron(hadron.irrep_row * g.irrep_row, hadron.flavor_structure * g.flavor_structure)
                               if glueball else hadron)
            vertices[name] = Baryon(readers[side], Operator.from_terms(name, (
                OperatorTerm.from_baryon(component, spin, coefficient=1.0),)), source=source,
                phase_convention="anchor")
    axes, times, daggers = [sides["sink"], sides["source"]], ["sink", "source"], [False, True]
    if current:
        row = HadronIrrep("current", (0, 0, 0), "A_1", 1, Tag(0, 0))[0]
        axes.insert(1, [Hadron(row, Flavor("uu"))])
        times.insert(1, "current")
        daggers.insert(1, False)
    request = np.asarray(gen_correlator(axes, times, daggers, degenerate=True), dtype=object)

    def bind(row):  # the fixture binds the Meson; rows bind to their own operators
        if row.hadron_name == "G":
            return Glueball(glue, "G", source=row.dagger)
        return vertices[row.hadron_name] if row.hadron_name in vertices else bind_fixture(row)
    options = dict(vertex_map=bind, propagator_map={"S^q": propagator, "S^q_\\mathrm{local}": propagator},
                   configuration="exchange-cfg", time_axes={"sink": 0, "source": 2, "current": 1})
    return request, options, tuple(readers.values())


def _per_cell(request, options):
    """Each cell alone under the closed join: a packing-free reference."""
    return np.array([calculate(cell, **options, baryon_vertices="joined").value.item()
                     for cell in request.flat]).reshape(request.shape)


def uuu_case(*, current=False, glueball=False, symmetry="antisymmetric"):
    request, options, readers = _uuu_rows(current=current, glueball=glueball, symmetry=symmetry)
    return ScopeCase(request, options, _per_cell(request, options), readers=readers)


def mode_blocked(case, partitions=(0, 1, 3)):
    """Split a pure-baryon case into mode blocks; their sum is the unsplit matrix."""
    blocks = split_baryon_eigenmodes(case.request, partitions=partitions)
    ndim = np.ndim(case.request)
    return ScopeCase(blocks, dict(case.options, usedNe=partitions[-1]), case.expected,
                     reduce=lambda value: np.asarray(value).sum(axis=tuple(range(ndim, np.ndim(value)))))


def exchange_case(*, labelled):
    spins = np.zeros((4, 4, 4, 4), complex)
    spins[0, 0, 0, 0] = 1
    for p in permutations(range(3)):
        spins[(1, *p)] = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) / np.sqrt(6)
    spins[2:] = exchange.mixed_basis(4)
    axes, representation, bind, propagator, readers, expected = exchange.make_case(
        coefficient_basis=(spins, QuarkExchange("all")))
    # One mode state: every row's Phi is equal, so one reader per end lets rows pack.
    shared = {}

    def shared_bind(row):
        key = row.hadron_name, row.dagger
        if key not in shared:
            vertex = bind(row)
            end = "source" if row.hadron_name.startswith("source") else "sink"
            shared[key] = Baryon(readers[end + "0"], vertex.operator, source=vertex.dagger,
                                 phase_convention="anchor")
        return shared[key]
    request = np.asarray(exchange.request(axes, representation if labelled else None), dtype=object)
    options = dict(vertex_map=shared_bind, propagator_map={"S^q": propagator},
                   configuration="mixed-cfg", time_axes={"sink": 1, "source": 0})
    return ScopeCase(request, options, expected, readers=(readers["sink0"], readers["source0"]))


def phi_case(*, second="same"):
    """`_row_case` sinks on a two-component reader; snk1 has two spin parts.

    Its second part reads the same Phi with the same leg binding (``"same"``),
    the other Phi (``"other-phi"``), or the same Phi under another leg binding
    (``"other-core"``, different operator flavors).
    """
    case, vertices, correlator, _ = _row_case()
    first = case["vertices"]["sink"].operator.terms[0].component
    other = BaryonComponent.paths(((0,), (), ()), flavors=first.flavors)
    phis = {first: case["sink_spatial"], other: np.conj(case["sink_spatial"])}
    reader = ArrayBaryonElemental(
        {"vvv": np.repeat(np.stack(tuple(phis.values()))[:, None, None], 2, axis=2)},
        (first, other), cfg="cfg-a", data_identity="two-component-sink")
    component = other if second == "other-phi" else first
    flavors = ("u", "d", "s") if second == "other-core" else first.flavors
    sinks = {"snk0": ((_spin_entries(5), -0.3 + 0.8j, first, first.flavors),),
             "snk1": ((_spin_entries(9), 0.6 - 0.1j, first, first.flavors),
                      (_spin_entries(11), -0.4 + 0.25j, component, flavors))}
    for name, parts in sinks.items():
        vertices[name] = Baryon(reader, BaryonOperator.from_terms(name, tuple(
            BaryonOperatorTerm(entries, component, coefficient, flavors, 0)
            for entries, coefficient, component, flavors in parts)))
    sources = {name: vertices[name].operator.terms[0] for name in ("src0", "src1")}
    expected = np.array([[sum(_independent_oracle(dict(
        case, source_spin=sources[source].spin.dense(), source_coefficient=sources[source].coefficient,
        sink_spin=entries.dense(), sink_coefficient=coefficient, sink_spatial=phis[component]))
        for entries, coefficient, component, _ in sinks[sink]) for sink in sinks] for source in sources])
    options = dict(vertex_map=vertices, propagator_map=case["propagators"], configuration="cfg-a")
    return ScopeCase(correlator, options, expected)


def single_row_case():
    case, vertices, correlator, expected = _row_case()
    options = dict(vertex_map=vertices, propagator_map=case["propagators"], configuration="cfg-a")
    return ScopeCase(correlator[:1, :1], options, expected[:1, :1])


def scan_case():
    request, options, expected = _scan_case()
    return ScopeCase(request, options, expected)


SCOPE_CASES = {
    "rows-control": uuu_case,
    "rows-meson-current": lambda: uuu_case(current=True),
    "rows-glueball": lambda: uuu_case(glueball=True, symmetry=None),
    # Exchange labels require equal mode selections on exchanged legs, so
    # blocks split the full Wick matrix; its unsplit form is the control.
    "rows-full-wick-control": lambda: uuu_case(symmetry=None),
    "rows-mode-blocked": lambda: mode_blocked(uuu_case(symmetry=None)),
    "exchange-control": lambda: exchange_case(labelled=False),
    "exchange-multiplet": lambda: exchange_case(labelled=True),
    "phi-control": lambda: phi_case(),
    "phi-two-in-one-row": lambda: phi_case(second="other-phi"),
    "core-two-in-one-row": lambda: phi_case(second="other-core"),
    "single-row": single_row_case,
    "time-scan-control": scan_case,
}
# Minutes of planning on CSNS CPU (1536 blocked terms; 96 joined exchange
# terms): run by tests/temporary/tmp_baryon_factor_network_scope.py.
TEMPORARY_CASES = ("rows-mode-blocked", "exchange-control")


class BaryonFactorNetworkScopeTests(unittest.TestCase):
    def check_cases(self, names):
        for name in names:
            case = SCOPE_CASES[name]()
            values = {}
            for lowering in LOWERINGS:
                with self.subTest(case=name, baryon_vertices=lowering):
                    result = calculate(case.request, **case.options, baryon_vertices=lowering)
                    self.assertEqual(result.expanded.problem.factorized_vertices, lowering != "joined")
                    values[lowering] = case.reduce(result.value)
                    np.testing.assert_allclose(values[lowering], case.expected, rtol=3e-12, atol=1e-10)
            for a, b in combinations(values, 2):
                with self.subTest(case=name, pair=(a, b)):
                    np.testing.assert_allclose(values[a], values[b], rtol=3e-12, atol=1e-10)

    def test_every_case_under_every_lowering(self):
        self.check_cases(name for name in SCOPE_CASES if name not in TEMPORARY_CASES)

    def test_explicit_adapter_lowers_like_the_frontend_switch(self):
        case = phi_case(second="other-phi")
        for lowering in LOWERINGS:
            with self.subTest(vertices=lowering):
                result = calculate(case.request, **case.options,
                                   diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v", vertices=lowering))
                self.assertEqual(result.expanded.problem.factorized_vertices, lowering != "joined")
                np.testing.assert_allclose(np.asarray(result.value), case.expected, rtol=3e-12, atol=1e-10)

    def test_dagger_against_graph_orientation_is_rejected_by_every_lowering(self):
        # Not a scope rule: only a hand-written vertex_map can flip `source`.
        case, vertices, correlator, _ = _row_case()
        vertices["src0"] = Baryon(vertices["src0"].reader, vertices["src0"].operator, source=False)
        for lowering in LOWERINGS:
            with self.subTest(vertices=lowering), \
                    patch.object(vertices["src0"].reader, "block", side_effect=AssertionError("payload read")):
                with self.assertRaisesRegex(ValueError, "dagger state disagrees"):
                    _expand(case, vertices, correlator, vertices=lowering)


if __name__ == "__main__":
    unittest.main()
