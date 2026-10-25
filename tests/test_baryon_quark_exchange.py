"""Reduced Wick graphs versus full enumeration and an independent determinant."""

from dataclasses import replace
from itertools import permutations
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, HadronIrrep, Tag, gen_correlator, split_baryon_eigenmodes,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SparseSpinTensor, baryon_spin_operators
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.correlator import ContractionTask, calculate
from lqcdweave.contracts.correlator.api import CfgPlan


def fixture():
    rng = np.random.default_rng(51072)
    random = lambda shape: (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / 3
    component = BaryonComponent.paths(((), (), ()))
    epsilon = np.zeros((3, 3, 3), complex)
    for permutation in permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j] for i in range(3) for j in range(i + 1, 3))
        epsilon[permutation] = (-1) ** inversions
    spatial_coefficients = {"sink": 0.8 + 0.5j, "source": 1.1 - 0.2j}
    operator_coefficients = {"sink": 0.7 + 0.2j, "source": -0.3 + 0.4j}
    readers, operators = {}, {}
    for name in ("sink", "source"):
        readers[name] = ArrayBaryonElemental(
            {"vvv": np.repeat((epsilon * spatial_coefficients[name])[None, None, None], 3, axis=2)},
            (component,), cfg="exchange-cfg", data_identity=f"exchange-{name}",
        )
        operators[name] = Operator.from_terms(name, (OperatorTerm.from_baryon(
            component, SparseSpinTensor(((0, 0, 0, 1),)),
            coefficient=operator_coefficients[name],
        ),))
    meson_component = ElementalComponent(0)
    readers["current"] = ArrayElemental(
        {"v2v": random((1, 1, 3, 3, 3))}, (meson_component,), cfg="exchange-cfg", phase_convention="anchor",
    )
    operators["current"] = Operator.from_terms("current", (OperatorTerm(0, meson_component, 0, .6 - .2j),))
    data = random((3, 3, 4, 4, 3, 3))
    propagator = ArrayPropagator({"vsv": data}, cfg="exchange-cfg", time_convention="absolute")
    bindings = {}

    def bind(row):
        key = row.hadron_name, row.dagger
        if key not in bindings:
            cls = Meson if row.hadron_name == "current" else Baryon
            bindings[key] = cls(readers[row.hadron_name], operators[row.hadron_name],
                                source=row.dagger, phase_convention="anchor")
        return bindings[key]

    # B = sum epsilon_ijk q_i q_j q_k = 6 q_0 q_1 q_2. Its adjoint
    # supplies the gamma4 spin vector on each leg. No E indices or Wick
    # coefficients are used by this independent 36*det(S) reference.
    effective = np.einsum("b,bmn->mn", gamma(8)[:, 0], data[2, 0, 0], optimize=False)
    expected = (36 * np.linalg.det(effective)
                * spatial_coefficients["sink"] * spatial_coefficients["source"].conjugate()
                * operator_coefficients["sink"] * operator_coefficients["source"].conjugate())
    return bind, propagator, readers, expected


def request(symmetry=None, *, structured=False, flavor="uuu", mixed=False):
    def baryon(name):
        row = HadronIrrep(name, (0, 0, 0), "G1", 1, Tag(0, 0))[0]
        return Hadron(row, Flavor(flavor), quark_symmetry=symmetry)
    axes = [[baryon("sink")], [baryon("source")]]
    times, daggers = ["sink", "source"], [False, True]
    if mixed:
        row = HadronIrrep("current", (0, 0, 0), "A_1", 1, Tag(0, 0))[0]
        axes.insert(1, [Hadron(row, Flavor("uu"))])
        times.insert(1, "current")
        daggers.insert(1, False)
    return gen_correlator(axes, times, daggers, structured=structured)


def row_matrix(recipe, irrep):
    """All rows of one uuu spin recipe at both ends; rows share their joins."""
    _, _, readers, _ = fixture()
    component = BaryonComponent.paths(((), (), ()))
    vertices, sides = {}, {}
    for side, source in (("sink", False), ("source", True)):
        hadrons = []
        for r, spin in enumerate(recipe.rows):
            name = f"{side}_r{r}"
            row = HadronIrrep(name, (0, 0, 0), irrep, 1, Tag(0, 0))[r]
            hadrons.append(Hadron(row, Flavor("uuu"), quark_symmetry="antisymmetric"))
            vertices[name] = Baryon(readers[side], Operator.from_terms(name, (
                OperatorTerm.from_baryon(component, spin, coefficient=1.0),)), source=source,
                phase_convention="anchor")
        sides[side] = hadrons
    matrix = gen_correlator([sides["sink"], sides["source"]], ["sink", "source"], [False, True], degenerate=True)
    rows = len(recipe.rows)
    value = np.empty((rows, rows), dtype=object)
    for a in range(rows):
        for b in range(rows):
            value[a, b] = matrix[a, b]
    return value, vertices


def run(value, bind, propagator, **options):
    return calculate(value, vertex_map=bind,
                     propagator_map={"S^q": propagator, "S^q_\\mathrm{local}": propagator},
                     configuration="exchange-cfg", time_axes={"sink": 0, "source": 2, "current": 1},
                     optimize="none", **options)


class BaryonQuarkExchangeTests(unittest.TestCase):
    def test_full_and_reduced_symbolic_and_structured_match_independent_determinant(self):
        bind, propagator, _, expected = fixture()
        for structured in (False, True):
            for symmetry in (None, "antisymmetric", {(1, 0, 2): -1}):
                with self.subTest(structured=structured, symmetry=symmetry):
                    result = run(request(symmetry, structured=structured), bind, propagator)
                    np.testing.assert_allclose(result.value.item(), expected, rtol=2e-12, atol=2e-12)
                    self.assertEqual(len(result.expanded.problem.terms),
                                     6 if symmetry is None else 1 if isinstance(symmetry, str) else 2)

    def test_every_vertex_lowering_replays_repeated_wick_joins_in_complete_cfg_plan(self):
        # Full uuu Wick terms repeat one source/sink join over equal legs; the
        # complete cfg plan must keep one owner per repeated canonical value.
        spins = {recipe.name: recipe for recipe in baryon_spin_operators()}
        cases = [(request(symmetry), fixture()[0]) for symmetry in (None, "antisymmetric")]
        cases.append(row_matrix(spins["spin_G1g_S_copy0"], "G1"))
        for value, bind in cases:
            _, propagator, _, _ = fixture()
            propagators = {"S^q": propagator, "S^q_\\mathrm{local}": propagator}
            expected = np.stack([np.asarray(calculate(
                value, vertex_map=bind, propagator_map=propagators, configuration="exchange-cfg",
                time_axes={"sink": t, "source": 2}, optimize="none").value) for t in (0, 1)], axis=-1)
            for lowering in ("auto", "pair", "nonzero", "joined"):
                with self.subTest(shape=np.shape(value), baryon_vertices=lowering):
                    with ContractionTask(value, configuration="exchange-cfg", vertex_map=bind,
                                         propagator_map=propagators, time_axes={"sink": (0, 1), "source": (2,)},
                                         batch="auto", storage="bounded", optimize="anytime",
                                         baryon_vertices=lowering) as task:
                        task.plan()
                        self.assertIsInstance(task._selected.plan, CfgPlan)
                        actual = np.asarray(task.run().value)
                    np.testing.assert_allclose(actual.reshape(expected.shape), expected, rtol=2e-12, atol=2e-12)

    def test_two_identical_flavors_preserve_complex_weight_and_source_adjoint(self):
        bind, propagator, _, _ = fixture()
        plain = run(request(flavor="uud"), bind, propagator)
        for structured in (False, True):
            reduced = run(request({(1, 0, 2): -1}, structured=structured, flavor="uud"), bind, propagator)
            np.testing.assert_allclose(reduced.value, plain.value, rtol=2e-12, atol=2e-12)
            self.assertEqual(len(plain.expanded.problem.terms), 2)
            self.assertEqual(len(reduced.expanded.problem.terms), 1)

    def test_mixed_current_keeps_connected_and_disconnected_contributions(self):
        bind, propagator, _, _ = fixture()
        plain = run(request(mixed=True), bind, propagator)
        reduced = run(request("antisymmetric", mixed=True), bind, propagator)
        np.testing.assert_allclose(reduced.value, plain.value, rtol=3e-12, atol=3e-12)
        self.assertLess(len(reduced.expanded.problem.terms), len(plain.expanded.problem.terms))

    def test_incompatible_extent_or_actual_basis_rejects_before_payload_reads(self):
        for different_basis in (False, True):
            with self.subTest(different_basis=different_basis):
                bind, propagator, readers, _ = fixture()
                original = readers["sink"].describe_block

                def describe(*args, **kwargs):
                    metadata = original(*args, **kwargs)
                    return replace(metadata, axis_bindings={
                        axis: ("different" if i == 1 else "same", (0, 1, 2))
                        for i, axis in enumerate(metadata.axes)
                    })

                if different_basis:
                    readers["sink"].describe_block = describe
                with patch.object(readers["sink"], "block", side_effect=AssertionError("elemental payload read")), \
                     patch.object(propagator, "block", side_effect=AssertionError("propagator payload read")):
                    with self.assertRaisesRegex(ValueError, "quark exchange symmetry requires"):
                        run(request("antisymmetric"), bind, propagator, execute=False,
                            usedNe=None if different_basis else (3, 2, 3))

    def test_unequal_mode_selections_reject_even_with_equal_extents(self):
        bind, propagator, readers, _ = fixture()
        blocks = split_baryon_eigenmodes(request("antisymmetric"), partitions=(0, 1, 2, 3))
        unequal = blocks[0, 0, 0, 1, 0, 0, 0, 0]
        with patch.object(readers["sink"], "block", side_effect=AssertionError("elemental payload read")):
            with self.assertRaisesRegex(ValueError, "ordered mode selection"):
                run(unequal, bind, propagator, execute=False, usedNe=3)


if __name__ == "__main__":
    unittest.main()
