"""Compact requests versus the existing symbolic CPU numerical chain."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson, structured_vertex_components
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, ModeBlockedDiagram, Tag,
    gen_correlator, iter_structured_wick_entries, remove_disconneted_diagram,
    split_meson_mode_pairs,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.api import SparseSpinTensor
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation


def fixture():
    rng = np.random.default_rng(771)
    random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    component = ElementalComponent(0)
    elemental = ArrayElemental({'v2v': random((1, 1, 2, 3, 3))}, (component,),
                              phase_convention='anchor', cfg='cfg', data_identity='structured-elemental')
    propagator = ArrayPropagator({'vsv': random((2, 2, 4, 4, 3, 3))},
                                time_convention='absolute', cfg='cfg', data_identity='structured-propagator')
    operators = {
        'z': Operator.from_terms('z', (OperatorTerm(5, component, 0, .3 + .7j),
                                      OperatorTerm(1, component, 0, -.4j))),
        'a': Operator.from_terms('a', (OperatorTerm(5, component, 0, 1 - .2j),)),
        'copy': Operator.from_terms('copy', (OperatorTerm(5, component, 0, 1 - .2j),)),
    }
    bindings = {}

    def bind(row):
        key = (row.hadron_name, row.dagger)
        if key not in bindings:
            bindings[key] = Meson(elemental, operators[row.hadron_name], row.dagger,
                                  phase_convention='anchor')
        return bindings[key]

    return bind, elemental, propagator


def task(*, structured=False, flavor='charged', names=('z', 'a')):
    flavor = (Flavor('ud') if flavor == 'charged' else
              (Flavor('uu') + Flavor('dd')) / sp.sqrt(2) if flavor == 'I0' else
              (Flavor('uu') - Flavor('dd')) / sp.sqrt(2))
    axis = [Hadron(HadronIrrep(name, (0, 0, 0), 'A_1', -1, Tag(0, 0))[0], flavor)
            for name in names]
    return gen_correlator([axis.copy(), axis.copy()], [0, 1], [False, True],
                          degenerate=True, structured=structured)


class StructuredCorrelatorTests(unittest.TestCase):
    def run_task(self, value, bind, propagator, **options):
        return calculate(value, vertex_map=bind, propagator_map={'S^q': propagator, 'S^q_\\mathrm{local}': propagator},
                         configuration='cfg', usedNe=3, optimize='none', **options)

    def test_wick_entries_match_symbolic_including_position_weights(self):
        for flavor in ('charged', 'I0', 'I1'):
            expected = task(flavor=flavor)
            compact = task(structured=True, flavor=flavor)
            for index, entries in iter_structured_wick_entries(compact):
                actual = sum(coefficient * sp.Mul(*diagrams) for coefficient, diagrams in entries)
                self.assertEqual(sp.expand(actual - expected[index]), 0)

        a = HadronIrrep('a', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        z = HadronIrrep('z', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        axis = [Hadron((1 + sp.I) * a + sp.sqrt(2) * z, Flavor('ud'))]
        expected = gen_correlator([axis.copy(), axis.copy()])
        compact = gen_correlator([axis.copy(), axis.copy()], structured=True)
        for index, entries in iter_structured_wick_entries(compact):
            actual = sum(coefficient * sp.Mul(*diagrams) for coefficient, diagrams in entries)
            difference = sp.expand(actual - expected[index])
            for coefficient in sp.Poly(difference, *difference.atoms(Diagram)).coeffs():
                self.assertLess(abs(complex(coefficient)), 1e-14)

    def test_baryon_complex_core_stays_native_and_matches_symbolic(self):
        rng = np.random.default_rng(181)
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        component = BaryonComponent.paths()
        reader = ArrayBaryonElemental({'vvv': random((1, 1, 2, 1, 1, 1))}, (component,), cfg='cfg')
        operator = Operator.from_terms('baryon', (
            OperatorTerm.from_baryon(component, SparseSpinTensor(((0, 1, 2, 1),)), coefficient=.4 + .3j),
            OperatorTerm.from_baryon(component, SparseSpinTensor(((3, 2, 1, .5j),)), coefficient=-.2j),
        ))
        vertices = {False: Baryon(reader, operator), True: Baryon(reader, operator, True)}
        propagator = ArrayPropagator({'vsv': random((2, 2, 4, 4, 1, 1))},
                                    time_convention='absolute', cfg='cfg')
        row = HadronIrrep('N', (0, 0, 0), 'G_1', 1, Tag(0, 0))[0]
        axis = [Hadron(row, Flavor('uds'))]
        options = dict(vertex_map=lambda row: vertices[row.dagger],
                       propagator_map={'S^q': propagator, 'S^s': propagator},
                       configuration='cfg', usedNe=1, optimize='none', baryon_vertices='joined')
        symbolic = gen_correlator([axis.copy(), axis.copy()])
        compact = gen_correlator([axis.copy(), axis.copy()], structured=True)
        expected = calculate(symbolic, **options).value
        with patch.object(reader, 'block', side_effect=AssertionError('baryon metadata read')):
            planned = calculate(compact, execute=False, **options)
        self.assertTrue(planned.expanded.problem.baryon_joins)
        self.assertTrue(all(len(spec.shape) < 6 for spec in planned.expanded.problem.specs.values()))
        np.testing.assert_allclose(planned.run().value, expected, rtol=3e-12, atol=3e-12)

    def test_complex_components_and_independent_loops_match_symbolic(self):
        bind, _, propagator = fixture()
        for flavor in ('charged', 'I0', 'I1'):
            with self.subTest(flavor=flavor):
                expected = self.run_task(task(flavor=flavor), bind, propagator).value
                actual = self.run_task(task(structured=True, flavor=flavor), bind, propagator).value
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_projection_has_no_selected_symbols_or_numeric_reads(self):
        bind, elemental, propagator = fixture()
        compact = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=[0, 1, 3])
        expected_task = split_meson_mode_pairs(task(flavor='I0'), partitions=[0, 1, 3])
        with patch.object(elemental, 'block', side_effect=AssertionError('metadata elemental read')), \
             patch.object(propagator, 'block', side_effect=AssertionError('metadata propagator read')), \
             patch.object(ModeBlockedDiagram, '_from_partition', side_effect=AssertionError('selected symbol')):
            planned = self.run_task(compact, bind, propagator, execute=False)
        actual = planned.run().value
        expected = self.run_task(expected_task, bind, propagator).value
        self.assertEqual(actual.shape, (2, 2, 3, 3))
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_repeated_full_request_reuses_tree_and_owns_numeric_outputs(self):
        bind, _, propagator = fixture()
        full = task(structured=True, flavor='I0', names=('a',))
        connected = remove_disconneted_diagram(full, ['S^q_\\mathrm{local}'])
        projected = split_meson_mode_pairs(
            {'full': full, 'again': full, 'difference': full - connected}, partitions=[0, 1, 3])
        self.assertIs(projected['full'], projected['again'])
        self.assertIs(projected['full'], projected['difference'].children[0])
        planned = self.run_task(projected, bind, propagator, execute=False)
        root = planned.expanded.output_recipe.root
        self.assertIs(root.children[0], root.children[1])
        self.assertIs(root.children[0], root.children[2].children[0])
        values = planned.run().value
        np.testing.assert_allclose(values['full'], values['again'], rtol=1e-13)
        self.assertFalse(np.shares_memory(values['full'], values['again']))

    def test_duplicate_operator_rows_share_basis_terms(self):
        bind, _, propagator = fixture()
        single = self.run_task(task(structured=True, names=('a',)), bind, propagator, execute=False)
        duplicate = self.run_task(task(structured=True, names=('a', 'copy')), bind, propagator, execute=False)
        self.assertEqual(len(single.expanded.problem.terms), len(duplicate.expanded.problem.terms))
        result = duplicate.run().value
        np.testing.assert_allclose(result, np.full((2, 2), result[0, 0]), rtol=1e-13)

    def test_component_identity_keeps_anchor_time_and_dagger(self):
        _, elemental, _ = fixture()
        component = ElementalComponent(0)
        registry = {}
        vertices = []
        for offset, dagger in ((0, False), (1, False), (0, True), (0, False)):
            operator = Operator.from_terms('offset', (
                OperatorTerm(5, component, 0, .4 + .3j, anchor_time_offset=offset),))
            vertices.append(structured_vertex_components(
                Meson(elemental, operator, dagger, phase_convention='anchor'), registry=registry)[0])
        self.assertIs(vertices[0][0], vertices[3][0])
        self.assertIsNot(vertices[0][0], vertices[1][0])
        self.assertIsNot(vertices[0][0], vertices[2][0])
        self.assertEqual(vertices[2][1], .4 - .3j)
        limited = Meson(elemental, operator, False, phase_convention='anchor')
        limited.usedNe = 2
        limited_component, = structured_vertex_components(limited, registry=registry)
        self.assertIsNot(limited_component[0], vertices[0][0])

    def test_single_insertion_and_all_zero_request(self):
        bind, _, propagator = fixture()
        row = HadronIrrep('a', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        for flavor in (Flavor('uu'), Flavor('ud')):
            axis = [Hadron(row, flavor)]
            values = []
            for structured in (False, True):
                request = gen_correlator([axis.copy()], structured=structured)
                values.append(self.run_task(request, bind, propagator).value)
            np.testing.assert_allclose(values[1], values[0], rtol=1e-13)

    def test_nested_fields_filter_and_scan_keep_public_shape(self):
        bind, _, propagator = fixture()
        results = []
        for structured in (False, True):
            full = task(structured=structured, flavor='I0', names=('a',))
            connected = remove_disconneted_diagram(full, ['S^q_\\mathrm{local}'])
            # Use an absent propagator to exercise deferred filtering without removing all graphs.
            retained = remove_disconneted_diagram(full, ['S^s'])
            fields = {'full': full, 'retained': retained, 'connected': connected, 'scaled': 2j * full}
            results.append(self.run_task(fields, bind, propagator, time_pairs=[{0: 0, 1: 1}, {0: 1, 1: 0}], batch=1).value)
        for key in results[0]:
            np.testing.assert_allclose(results[1][key], results[0][key], rtol=3e-12, atol=3e-12)
        self.assertEqual(results[1]['full'].shape, (1, 1, 2))

    def test_archive_roundtrip_with_current_request(self):
        bind, elemental, propagator = fixture()
        request = split_meson_mode_pairs(task(structured=True, names=('a',)), partitions=[0, 1, 3])
        hardware = TargetHardwareMemory(host_available_bytes=100000000)
        options = dict(execute=False, target_hardware=hardware,
                       profile=CostProfile(hardware_fingerprint='structured-cpu'),
                       memory_limit=100000000, host_memory_limit=100000000)
        selected = self.run_task(request, bind, propagator,time_axes={0:1,1:0},**options)
        coordinates, _ = describe_coordinates(time_pairs=({0:1,1:0},{0:0,1:1}))
        planned = complete_calculation(selected,coordinates=coordinates,
            readers=selected.inputs.loader.input_readers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json.gz'
            save_calculation_plan(path, planned)
            restored = calculate(None,plan_archive=path,configuration='cfg',
                propagator_map={'S^q':propagator,'S^q_\\mathrm{local}':propagator},
                vertex_map=elemental,memory_limit=100000000,
                host_memory_limit=100000000,execute=False)
            np.testing.assert_allclose(restored.run().value, planned.run().value, rtol=1e-13)

    def test_cycles_and_invalid_requests_fail_explicitly(self):
        bind, _, propagator = fixture()
        values = [task(structured=True)]
        values.append(values)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            self.run_task(values, bind, propagator, execute=False)
        with self.assertRaises(TypeError):
            gen_correlator([], structured=1)
        with self.assertRaises(NotImplementedError):
            gen_correlator([], structured=True)


if __name__ == '__main__':
    unittest.main()
