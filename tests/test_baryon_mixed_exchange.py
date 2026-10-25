"""Matrix exchange reduction versus full Wick sums and fermionic determinants."""

from itertools import permutations, product
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, QuarkExchange, Tag, gen_correlator,
    split_baryon_eigenmodes,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SparseSpinTensor
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.correlator import calculate


def mixed_basis(size):
    result = np.zeros((2, size, size, size), complex)
    for word, ms, ma in (((0, 0, 1), 2, 0), ((0, 1, 0), -1, 1), ((1, 0, 0), -1, -1)):
        result[(0, *word)] = ms / np.sqrt(6)
        result[(1, *word)] = ma / np.sqrt(2)
    return result


def fermionic_reference(left_tensors, right_tensors, flavors, kernels):
    """Canonical Fock coefficients of ordered flavor fields, without Wick graphs."""
    def amplitudes(tensors):
        result = []
        for tensor in tensors:
            values = {}
            for indices in np.ndindex(tensor.shape):
                coefficient = tensor[indices]
                if coefficient == 0:
                    continue
                fields = tuple(zip(flavors, indices))
                if len(set(fields)) != 3:
                    continue
                sign = (-1)**sum(fields[i] > fields[j] for i in range(3) for j in range(i+1, 3))
                ordered = tuple(sorted(fields))
                values[ordered] = values.get(ordered, 0) + sign*coefficient
            result.append(values)
        return result
    left, right = amplitudes(left_tensors), amplitudes(right_tensors)
    basis = sorted(set().union(*(values.keys() for values in (*left, *right))))
    l = np.array([[values.get(key, 0) for key in basis] for values in left], complex)
    r = np.array([[values.get(key, 0) for key in basis] for values in right], complex)
    minors = np.array([[np.linalg.det(np.array([
        [kernels[f][i, j] if f == g else 0 for g, j in source] for f, i in sink]))
        for source in basis] for sink in basis], complex).reshape(len(basis), len(basis))
    return l @ minors @ r.conj().T


def make_case(*, coupled=True, spin_change=None, flavor="uuu", coefficient_basis=None):
    """Factorized synthetic full coefficients; no Wick/E equation in oracle."""
    rng = np.random.default_rng(615031)
    random = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/3
    modes = mixed_basis(2)
    spins = mixed_basis(4) if coupled else np.eye(1, 64).reshape(1, 4, 4, 4)
    states = tuple(product(range(len(spins)), range(len(modes))))
    exchange = QuarkExchange().tensor_product(QuarkExchange()) if coupled else QuarkExchange()
    if coefficient_basis is not None:
        spins, exchange = coefficient_basis
        modes = np.zeros((1, 2, 2, 2), complex)
        modes[0, 0, 0, 0] = 1
        states = tuple(product(range(len(spins)), range(len(modes))))
    if spin_change is not None:
        change = np.asarray(spin_change, dtype=complex)
        spins = np.einsum('ba,bijk->aijk', change, spins)
        changed = QuarkExchange({p: spin_change.inv()*m*spin_change for p, m in QuarkExchange().elements})
        exchange = changed.tensor_product(QuarkExchange())
    component = BaryonComponent.paths(((), (), ()))
    readers, operators, full_tensors = {}, {}, {}
    axes = []
    for endpoint, weight in (("sink", .8+.3j), ("source", -.4+.7j)):
        axis, tensors = [], []
        for i, (a, b) in enumerate(states):
            name = f"{endpoint}{i}"
            row = HadronIrrep(name, (0, 0, 0), "G1", 1, Tag(0, 0))[0]
            axis.append(Hadron(row, Flavor(flavor)))
            readers[name] = ArrayBaryonElemental(
                {"vvv": np.repeat(modes[b][None, None, None], 2, axis=2)},
                (component,), cfg="mixed-cfg", data_identity=name)
            entries = tuple((*index, spins[(a, *index)]) for index in np.ndindex(4, 4, 4)
                            if spins[(a, *index)] != 0)
            operators[name] = Operator.from_terms(name, (OperatorTerm.from_baryon(
                component, SparseSpinTensor(entries), coefficient=weight),))
            tensors.append(weight*np.einsum("abc,ijk->aibjck", spins[a], modes[b]).reshape(8, 8, 8))
        full_tensors[endpoint] = tensors
        axes.append(axis)
    data = random((2, 2, 4, 4, 2, 2))
    propagator = ArrayPropagator({"vsv": data}, cfg="mixed-cfg", time_convention="absolute")
    bindings = {}

    def bind(row):
        key = row.hadron_name, row.dagger
        if key not in bindings:
            bindings[key] = Baryon(readers[row.hadron_name], operators[row.hadron_name],
                                   source=row.dagger, phase_convention="anchor")
        return bindings[key]

    # Canonical Fock amplitudes and determinants of the one-particle kernel.
    # gamma4 comes from the physical source adjoint, independent of E layout.
    kernel = np.einsum("abij,bc->aicj", data[0, 1], gamma(8), optimize=False).reshape(8, 8)
    expected = fermionic_reference(full_tensors['sink'], full_tensors['source'], flavor,
                                   {f: kernel for f in set(flavor)})
    return axes, exchange, bind, propagator, readers, expected


def request(axes, exchange=None, *, structured=False, degenerate=True):
    choices = [list(axis) if exchange is None else list(Hadron.multiplet(axis, quark_symmetry=exchange))
               for axis in axes]
    return gen_correlator(choices, ["sink", "source"], [False, True], structured=structured,
                          degenerate=degenerate)


def run(value, bind, propagator, *, propagators=None, **options):
    return calculate(value, vertex_map=bind, propagator_map=propagators or {"S^q": propagator},
                     time_axes={"sink": 1, "source": 0}, configuration="mixed-cfg",
                     optimize="none", **options)


class BaryonMixedExchangeTests(unittest.TestCase):
    def test_all_s3_components_match_independent_fock_reference_and_scalar_endpoint(self):
        spins = np.zeros((4, 4, 4, 4), complex)
        spins[0, 0, 0, 0] = 1
        for p in permutations(range(3)):
            sign = (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
            spins[(1, *p)] = sign/np.sqrt(6)
        spins[2:] = mixed_basis(4)
        axes, rep, bind, propagator, _, expected = make_case(coefficient_basis=(spins, QuarkExchange('all')))
        self.assertGreater(abs(expected[1, 1]), 1e-6)
        full = run(request(axes), bind, propagator)
        np.testing.assert_allclose(full.value, expected, atol=2e-12, rtol=2e-12)
        for structured in (False, True):
            value = run(request(axes, rep, structured=structured), bind, propagator)
            np.testing.assert_allclose(value.value, expected, atol=2e-12, rtol=2e-12)
            self.assertEqual(len(value.expanded.problem.terms), 1)
            sink = Hadron.multiplet(axes[0], quark_symmetry=rep)
            source = Hadron(axes[1][1].irrep_row, axes[1][1].flavor_structure,
                            quark_symmetry='antisymmetric')
            joint = gen_correlator([list(sink), [source]], ['sink', 'source'], [False, True],
                                  structured=structured)
            value = run(joint, bind, propagator)
            np.testing.assert_allclose(value.value, expected[:, 1:2], atol=2e-12, rtol=2e-12)
            self.assertEqual(len(value.expanded.problem.terms), 1)

    def test_pair_s_a_for_all_slot_choices_match_non_degenerate_fock_reference(self):
        for flavor, slots in (('uud', (0, 1)), ('udu', (0, 2)), ('duu', (1, 2))):
            spins = np.zeros((2, 4, 4, 4), complex)
            first = (0, 1, 2)
            second = list(first)
            second[slots[0]], second[slots[1]] = second[slots[1]], second[slots[0]]
            spins[(0, *first)] = spins[(0, *second)] = 1/np.sqrt(2)
            spins[(1, *first)], spins[(1, *second)] = 1/np.sqrt(2), -1/np.sqrt(2)
            rep = QuarkExchange.pair(slots=slots)
            axes, _, bind, propagator, _, expected = make_case(flavor=flavor, coefficient_basis=(spins, rep))
            other = ArrayPropagator({'vsv': (.6+.2j)*propagator.sectors['vsv']},
                                    cfg='mixed-cfg', time_convention='absolute')
            props = {'S^u': propagator, 'S^d': other}
            self.assertGreater(np.max(abs(expected)), 1e-6)
            for representation, structured in ((None, False), (rep, False), (rep, True)):
                with self.subTest(flavor=flavor, labelled=representation is not None, structured=structured):
                    value = run(request(axes, representation, structured=structured, degenerate=False),
                                bind, propagator, propagators=props)
                    np.testing.assert_allclose(value.value, expected*(.6+.2j), atol=2e-12, rtol=2e-12)
                    if representation is not None:
                        self.assertEqual(len(value.expanded.problem.terms), 1)

    def test_nonzero_coupled_mixed_factors_match_full_wick_and_independent_determinants(self):
        axes, exchange, bind, propagator, _, expected = make_case()
        self.assertGreater(np.max(abs(expected)), 1e-5)
        full = run(request(axes), bind, propagator)
        np.testing.assert_allclose(full.value, expected, atol=2e-12, rtol=2e-12)
        for structured in (False, True):
            with self.subTest(structured=structured):
                reduced = run(request(axes, exchange, structured=structured), bind, propagator)
                np.testing.assert_allclose(reduced.value, full.value, atol=2e-12, rtol=2e-12)
                self.assertLess(len(reduced.expanded.problem.terms), len(full.expanded.problem.terms))

    def test_pure_ms_ma_complete_coefficients_cancel_against_full_numeric_enumeration(self):
        axes, exchange, bind, propagator, _, expected = make_case(coupled=False)
        full = run(request(axes), bind, propagator)
        np.testing.assert_allclose(full.value, 0, atol=2e-12)
        np.testing.assert_allclose(expected, 0, atol=2e-12)
        for structured in (False, True):
            result = run(request(axes, exchange, structured=structured), bind, propagator)
            np.testing.assert_array_equal(result.value, np.zeros_like(full.value))
            self.assertIsNone(result.expanded.problem)

    def test_incompatible_mode_extents_reject_before_payload_reads_for_both_entries(self):
        axes, exchange, bind, propagator, readers, _ = make_case()
        for structured in (False, True):
            with self.subTest(structured=structured), patch.object(
                    propagator, 'block', side_effect=AssertionError('numeric payload read')):
                with self.assertRaisesRegex(ValueError, 'quark exchange symmetry requires'):
                    run(request(axes, exchange, structured=structured), bind, propagator,
                        execute=False, usedNe=(2, 1, 2))

    def test_partial_actual_flavor_subgroup_with_different_propagators_matches_full_enumeration(self):
        for flavor in ('uud', 'udu', 'duu'):
            axes, exchange, bind, propagator, _, expected = make_case(flavor=flavor)
            other = ArrayPropagator({'vsv': (.6+.2j)*propagator.sectors['vsv']},
                                    cfg='mixed-cfg', time_convention='absolute')
            propagators = {'S^u': propagator, 'S^d': other}
            full = run(request(axes, degenerate=False), bind, propagator, propagators=propagators)
            self.assertGreater(np.max(abs(full.value)), 1e-6)
            np.testing.assert_allclose(full.value, expected*(.6+.2j), atol=3e-12, rtol=3e-12)
            for structured in (False, True):
                with self.subTest(flavor=flavor, structured=structured):
                    reduced = run(request(axes, exchange, structured=structured, degenerate=False),
                                  bind, propagator, propagators=propagators)
                    np.testing.assert_allclose(reduced.value, full.value, atol=3e-12, rtol=3e-12)
                    self.assertLess(len(reduced.expanded.problem.terms), len(full.expanded.problem.terms))

    def test_three_distinct_flavors_match_independent_fock_reference(self):
        axes, exchange, bind, propagator, _, expected = make_case(flavor='uds')
        propagators = {'S^'+f: ArrayPropagator({'vsv': scale*propagator.sectors['vsv']},
                        cfg='mixed-cfg', time_convention='absolute')
                       for f, scale in (('u', 1), ('d', .6+.2j), ('s', -.3+.7j))}
        for structured in (False, True):
            result = run(request(axes, exchange, structured=structured, degenerate=False),
                         bind, propagator, propagators=propagators)
            np.testing.assert_allclose(result.value, expected*(.6+.2j)*(-.3+.7j), atol=3e-12, rtol=3e-12)

    def test_partner_basis_id_conflict_rejects_before_any_numeric_reads(self):
        axes, exchange, bind, propagator, readers, _ = make_case()
        for structured in (False, True):
            with self.subTest(structured=structured), ExitStack() as stack:
                for name, reader in readers.items():
                    original = reader.describe_block
                    def describe(*args, _original=original, _name=name, **kwargs):
                        metadata = _original(*args, **kwargs)
                        return replace(metadata, axis_bindings={axis: ('basis-'+_name, (0, 1))
                                                                 for axis in metadata.axes})
                    stack.enter_context(patch.object(reader, 'describe_block', side_effect=describe))
                    stack.enter_context(patch.object(reader, 'block', side_effect=AssertionError('payload read')))
                with self.assertRaisesRegex(ValueError, 'matrix quark exchange components require'):
                    run(request(axes, exchange, structured=structured), bind, propagator, execute=False)

    def test_equal_size_but_different_ordered_mode_blocks_reject(self):
        axes, exchange, bind, propagator, _, _ = make_case()
        primitive = next(iter(request(axes, exchange)[1, 1].atoms(Diagram)))
        blocks = split_baryon_eigenmodes(primitive, partitions=(0, 1, 2))
        chosen = blocks[0, 1, 0, 0, 0, 0]
        with patch.object(propagator, 'block', side_effect=AssertionError('payload read')):
            with self.assertRaisesRegex(ValueError, 'ordered mode selection'):
                run(chosen, bind, propagator, execute=False, usedNe=2)


if __name__ == '__main__':
    unittest.main()
