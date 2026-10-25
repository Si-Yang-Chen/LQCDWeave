"""Literal component sums and exact finite-subset averages through calculate."""

from itertools import combinations, product
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.dilution import DilutionGroup, DilutionSource
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calculate


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def inclusion(n, m, labels):
    """Literal inverse probability of retaining this many distinct columns."""
    distinct = len(set(labels))
    numerator = denominator = 1
    for i in range(distinct):
        numerator *= n - i
        denominator *= m - i
    if denominator == 0:
        raise ValueError("the saved sample cannot identify this component pattern")
    return numerator / denominator


def source(n, m, identity="actual-D-component-sample"):
    group = DilutionGroup(tuple(range(n)), tuple((0, i) for i in range(m)))
    return DilutionSource(identity, "actual-ordered-parent", (group,), ((0, "actual-Q-column-pool"),))


def single_time_prop(spin, spatial, sector, endpoints, identity):
    value = np.einsum("ab,ij->abij", spin, spatial).reshape(1, 1, 4, 4, *spatial.shape)
    return ArrayPropagator({sector: value}, time_convention="absolute", cfg="component-cfg",
        data_identity=identity, basis_endpoints={(sector, side): endpoint
            for side, endpoint in zip(("sink", "source"), endpoints)})


class DilutionComponentReferenceTests(unittest.TestCase):
    def calculate(self, context):
        return calculate(context["diagram"], propagator_map=context["propagators"],
            vertex_map=context["vertices"], configuration="component-cfg", optimize="none",
            elemental_weighted=context["weighted"])

    def three_loops(self, n=5, m=3, weighted=False):
        rng = np.random.default_rng(1681)
        q, _ = np.linalg.qr(random_complex(rng, (n, n)))
        d, v = q[:, :m], np.eye(n, dtype=complex)[:, :2]
        provenance = source(n, m)
        components = tuple(ElementalComponent(("single-D-loop", i)) for i in range(3))
        gamma_indices = (3, 8, 14)
        coefficients = (0.21 + 0.73j, -0.62 + 0.17j, 0.93 - 0.28j)
        a = tuple(random_complex(rng, (n, n)) for _ in range(3))
        t = tuple(random_complex(rng, (n, n)) for _ in range(3))
        spins = tuple(random_complex(rng, (4, 4)) for _ in range(3))
        values = np.zeros((3, 1, 1, m, 2), complex)
        for i in range(3):
            values[i, 0, 0] = d.conj().T @ a[i] @ v * (n / m if weighted else 1)
        reader = ArrayElemental({"d2v": values}, components, phase_convention="anchor", cfg="component-cfg",
            data_identity="three-current-components", elemental_weighted=weighted,
            dilution_endpoints={("d2v", "bar"): provenance}, basis_endpoints={
                ("d2v", "bar"): (provenance.data_id, tuple(range(m))),
                ("d2v", "field"): ("actual-V-two", (0, 1))})
        vertices, symbols, propagators = {}, [], {}
        for i in range(3):
            operator = Operator.from_terms(f"loop-{i}", (OperatorTerm(gamma_indices[i], components[i], 0, coefficients[i]),))
            vertices[f"loop-{i}"] = Meson(reader, operator, phase_convention="anchor", requested_sectors=("d2v",))
            symbol = Propagator(f"flavor-{i}", Tag(i, 0), Tag(i, 0))
            symbols.append(symbol)
            propagators[symbol] = single_time_prop(spins[i], v.conj().T @ t[i] @ d, "vsd",
                (("actual-V-two", (0, 1)), (provenance.data_id, tuple(range(m)))), f"actual-loop-{i}")
        adjacency = np.diag((1, 2, 3)).tolist()
        diagram = Diagram(QuarkDiagram(adjacency), [0, 0, 0], list(vertices), [None, *symbols])
        # Each loop preserves its own spin trace even when spatial labels coincide.
        prefactor = np.prod([coefficients[i] * np.trace(gamma(gamma_indices[i]) @ spins[i]) for i in range(3)])
        diagonals = tuple(np.diag(d.conj().T @ a[i] @ v @ v.conj().T @ t[i] @ d) for i in range(3))
        return dict(diagram=diagram, vertices=vertices, propagators=propagators, reader=reader,
            weighted=weighted, n=n, m=m, diagonals=diagonals, prefactor=prefactor)

    def test_three_disconnected_currents_match_all_component_equalities(self):
        results = []
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.three_loops(weighted=weighted)
                diagonal = context["diagonals"]
                expected = sum(inclusion(5, 3, (a, b, c)) * diagonal[0][a] * diagonal[1][b] * diagonal[2][c]
                    for a, b, c in product(range(3), repeat=3)) * context["prefactor"]
                # Explicitly nonzero two-equal/one-independent assignments exercise
                # a spatially joined pair of fermion loops plus a separate loop.
                two_equal = sum(inclusion(5, 3, (a, a, c)) * diagonal[0][a] * diagonal[1][a] * diagonal[2][c]
                    for a, c in product(range(3), repeat=2) if a != c)
                self.assertGreater(abs(two_equal), 1e-5)
                value = self.calculate(context).value
                results.append(value)
                np.testing.assert_allclose(value, expected, rtol=5e-12, atol=3e-9)
        np.testing.assert_allclose(results[0], results[1], rtol=5e-12, atol=3e-9)

    def test_rank_two_cannot_estimate_three_unconstrained_same_pool_components(self):
        context = self.three_loops(n=5, m=2)
        with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")):
            with self.assertRaisesRegex(ValueError, "support|sample|points|rank|distinct"):
                self.calculate(context)

    def two_dd_currents(self, weighted=False, *, n=6, m=4, selected=None, reorder=False):
        rng = np.random.default_rng(2791)
        q, _ = np.linalg.qr(random_complex(rng, (n, n)))
        columns = tuple(range(m)) if selected is None else tuple(selected)
        if reorder:
            columns = columns[::-1]
        d = q[:, columns]
        provenance = source(n, m, identity="actual-four-leg-columns")
        a, b, r, t = (random_complex(rng, (n, n)) for _ in range(4))
        spin_r, spin_t = random_complex(rng, (4, 4)), random_complex(rng, (4, 4))
        components = (ElementalComponent(("DD-current", 0)), ElementalComponent(("DD-current", 1)))
        coefficients = (0.29 + 0.61j, -0.13 + 0.79j)
        e0, e1 = d.conj().T @ a @ d, d.conj().T @ b @ d
        pair_weights = np.asarray([[inclusion(n, m, (i, j)) for j in range(m)] for i in range(m)])
        values = np.stack((e0, e1)).reshape(2, 1, 1, m, m)
        if weighted:
            values *= pair_weights
        # Storage indices are explicit and can be permuted independently of
        # the array positions. A complete group must remain selected.
        stored_indices = tuple(range(m))[::-1] if reorder else tuple(range(m))
        reader = ArrayElemental({"d2d": values}, components, phase_convention="anchor", cfg="component-cfg",
            data_identity="two-DD-currents", elemental_weighted=weighted,
            dilution_endpoints={("d2d", side): provenance for side in ("bar", "field")},
            basis_endpoints={("d2d", side): (provenance.data_id, stored_indices) for side in ("bar", "field")})
        vertices = {f"current-{i}": Meson(reader, Operator.from_terms(f"DD-{i}",
            (OperatorTerm((3, 14)[i], components[i], 0, coefficients[i]),)),
            phase_convention="anchor", requested_sectors=("d2d",)) for i in range(2)}
        outgoing, incoming = Propagator("u", Tag(0, 0), Tag(1, 0)), Propagator("d", Tag(1, 0), Tag(0, 0))
        endpoints = ((provenance.data_id, stored_indices),) * 2
        propagators = {
            incoming: single_time_prop(spin_r, d.conj().T @ r @ d, "dsd", endpoints, "actual-DD-incoming"),
            outgoing: single_time_prop(spin_t, d.conj().T @ t @ d, "dsd", endpoints, "actual-DD-outgoing")}
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 0], list(vertices), [None, outgoing, incoming])
        prefactor = coefficients[0] * coefficients[1] * np.trace(gamma(3) @ spin_r @ gamma(14) @ spin_t)
        return dict(diagram=diagram, vertices=vertices, propagators=propagators, reader=reader,
            weighted=weighted, matrices=(e0, d.conj().T @ r @ d, e1, d.conj().T @ t @ d),
            parent_matrices=(a, r, b, t), prefactor=prefactor)

    def test_four_same_pool_legs_remove_both_local_dd_preweights(self):
        values = []
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.two_dd_currents(weighted)
                a, r, b, t = context["matrices"]
                expected = sum(inclusion(6, 4, (i, j, k, l)) * a[i, j] * r[j, k] * b[k, l] * t[l, i]
                    for i, j, k, l in product(range(4), repeat=4)) * context["prefactor"]
                actual = self.calculate(context).value
                values.append(actual)
                np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=2e-9)
        np.testing.assert_allclose(values[0], values[1], rtol=5e-12, atol=2e-9)

    def test_reordered_complete_d_axes_keep_actual_component_sums(self):
        natural = self.two_dd_currents()
        reordered = self.two_dd_currents(reorder=True)
        np.testing.assert_allclose(self.calculate(natural).value, self.calculate(reordered).value,
                                   rtol=5e-12, atol=2e-9)

    def test_finite_complete_qr_subsets_average_to_parent_trace_exactly(self):
        # Only two D occurrences can be unrestricted at rank 2. Use one DD
        # vertex and an exact full-parent V vertex, not two four-leg DD currents.
        from test_dilution_correlator import DilutionCorrelatorTests
        reference = DilutionCorrelatorTests()
        samples = []
        expected = None
        rng = np.random.default_rng(482)
        complete_q, _ = np.linalg.qr(random_complex(rng, (4, 4)))
        for columns in combinations(range(4), 2):
            for weighted in (False, True):
                with self.subTest(columns=columns, weighted=weighted):
                    context = reference.fixture(weighted=weighted, v_count=4)
                    d = complete_q[:, columns]
                    context["reader"].sectors["d2d"][0, 0, 0] = d.conj().T @ context["a"] @ d
                    if weighted:
                        mask = np.asarray([[inclusion(4, 2, (i, j)) for j in range(2)] for i in range(2)])
                        context["reader"].sectors["d2d"][0, 0, 0] *= mask
                    context["propagator"].sectors["dsv"][1, 0] = np.einsum("ab,ij->abij", context["spin_r"], d.conj().T @ context["r"] @ context["v"])
                    context["propagator"].sectors["vsd"][0, 1] = np.einsum("ab,ij->abij", context["spin_t"], context["v"].conj().T @ context["t"] @ d)
                    value = reference.evaluate(context).value
                    if not weighted:
                        samples.append(value)
                    else:
                        np.testing.assert_allclose(value, samples[-1], rtol=5e-12, atol=2e-9)
                    expected = np.trace(context["a"] @ context["c"]) * context["prefactor"]
        np.testing.assert_allclose(np.mean(samples), expected, rtol=5e-12, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
