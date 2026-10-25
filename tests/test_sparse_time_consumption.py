"""Full physical arrays -> ordered compact slots -> native calculation -> dt."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave import preset
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson, validate_slot_compatibility
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calculate, save_correlator, load_correlator


class SparseTimeConsumptionTests(unittest.TestCase):
    def setUp(self):
        self.selection = TimeSelectionRecord(8, (6, 0, 1, 2))
        rng = np.random.default_rng(720)
        self.e = rng.normal(size=(1, 1, 8, 2, 2)) + 1j*rng.normal(size=(1, 1, 8, 2, 2))
        self.p = rng.normal(size=(8, 8, 4, 4, 2, 2)) + 1j*rng.normal(size=(8, 8, 4, 4, 2, 2))
        self.component = ElementalComponent(0)
        self.operator = Operator.from_terms('local', [OperatorTerm(1, self.component, 0)])
        self.diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['source', 'sink'], [None, 'q'])

    def calculation(self, elemental, propagator, times, **options):
        return calculate(self.diagram, propagator_map={'q': propagator},
            vertex_map={'source': Meson(elemental, self.operator, True), 'sink': Meson(elemental, self.operator)},
            configuration='cfg', usedNe=2, time_axes={0: times, 1: times}, optimize='none',
            memory_limit=32*2**20, **options)

    def test_sparse_preset_native_matches_full_selected_coordinates_and_roundtrips(self):
        full_e = ArrayElemental({'v2v': self.e}, [self.component], phase_convention='midpoint', cfg='cfg')
        full_p = ArrayPropagator({'vsv': self.p}, time_convention='absolute', cfg='cfg')
        expected = self.calculation(full_e, full_p, self.selection.times).value
        with TemporaryDirectory() as directory:
            root = Path(directory)
            compact_e = self.e[:, :, self.selection.times]
            compact_p = self.p[list(self.selection.times)][:, list(self.selection.times)]
            np.save(root/'e_cfg.npy', compact_e)
            np.save(root/'p_cfg.npy', compact_p)
            e = preset.ElementalNpy(str(root/'e_'), selection=self.selection)
            p = preset.PerambulatorNpy(str(root/'p_'), time_convention='absolute', selection=self.selection)
            with e.load('cfg') as compact_elemental, p.load('cfg') as compact_propagator:
                actual = self.calculation(compact_elemental, compact_propagator, range(self.selection.Nt))
                np.testing.assert_allclose(actual.value, expected, rtol=1e-12, atol=1e-10)
                self.assertEqual(compact_elemental.Lt, 4)
                self.assertEqual(compact_propagator.Lt, 4)
                save_correlator(root/'result', actual, coordinate_domain='slot', selection=self.selection,
                                source_axis=0, sink_axis=1, boundary='periodic', average_tsrc=True)
                saved = load_correlator(root/'result')
                oracle = []
                for dt in range(8):
                    pairs = [expected[i, j] for i, s in enumerate(self.selection.times)
                             for j, t in enumerate(self.selection.times) if (t-s)%8 == dt]
                    oracle.append(sum(pairs)/len(pairs))
                np.testing.assert_allclose(saved.value, oracle, rtol=1e-12, atol=1e-10)

    def test_binding_rejects_order_mismatch_mixed_domain_and_asymmetric(self):
        e = ArrayElemental({'v2v': self.e[:, :, self.selection.times]}, [self.component],
                          phase_convention='midpoint', selection=self.selection, cfg='cfg')
        compact = self.p[list(self.selection.times)][:, list(self.selection.times)]
        other = TimeSelectionRecord(8, (0, 6, 1, 2))
        for source, sink in ((other, other), (self.selection, other), (None, None)):
            p = ArrayPropagator({'vsv': compact}, time_convention='absolute', cfg='cfg',
                                source_selection=source, sink_selection=sink)
            with self.subTest(source=source), patch.object(p, 'block', side_effect=AssertionError('numeric read')):
                with self.assertRaisesRegex(ValueError, 'slot|ordered'):
                    self.calculation(e, p, range(4), execute=False)

    def test_metadata_identity_and_saved_binding_reject_reordered_same_shape(self):
        from lqcdweave.contracts.correlator.api import SourceReadRequest
        from lqcdweave.contraction.expansion.saved_inputs import SavedPhysicalLoader
        e = ArrayElemental({'v2v': self.e[:, :, self.selection.times]}, [self.component],
                          phase_convention='midpoint', selection=self.selection)
        reordered = ArrayElemental({'v2v': self.e[:, :, self.selection.times]}, [self.component],
                          phase_convention='midpoint', selection=TimeSelectionRecord(8, (0, 6, 1, 2)))
        metadata = e.describe_block(self.component, 0, 0, 'v2v')
        self.assertNotEqual(metadata.selection_identity, reordered.describe_block(self.component, 0, 0, 'v2v').selection_identity)
        request = SourceReadRequest('e', 'elemental', (self.component, 0, 0, 'v2v'), (), metadata,
                                   conventions=(('time_selection_identity', e.time_selection_identity),))
        with patch.object(reordered, 'block', side_effect=AssertionError('numeric read')):
            with self.assertRaisesRegex(ValueError, 'conventions'):
                SavedPhysicalLoader({'block': request}, {'e': reordered})

    def test_eigenvector_preset_is_explicit_physical_producer_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(4*2*3).reshape(4, 2, 1, 1, 1, 3).astype(complex)
            np.save(root/'v_cfg.npy', values)
            handle = preset.EigenvectorNpy(str(root/'v_'), lattice_size=(1, 1, 1, 8), selection=self.selection)
            array = handle.load('cfg')
            try:
                np.testing.assert_array_equal(handle.read_modes('cfg', 6, 0, 2), values[0])
                np.testing.assert_array_equal(handle.read_modes('cfg', 0, 0, 2), values[1])
                with self.assertRaises((KeyError, ValueError)):
                    handle.read_modes('cfg', 7, 0, 2)
            finally:
                array._mmap.close()


class TemporalSlotConsumptionTests(unittest.TestCase):
    calculation = SparseTimeConsumptionTests.calculation

    def setUp(self):
        SparseTimeConsumptionTests.setUp(self)
        self.selection = TimeSelectionRecord(8, (7, 0, 2))
        self.endpoints = TimeSelectionRecord(8, (7, 0, 2, 1, 3))
        self.component = ElementalComponent('temporal', (0, 1))
        self.operator = Operator.from_terms('temporal', [OperatorTerm(1, self.component, 0)])

    def readers(self, endpoint_selection=None):
        endpoint_selection = self.endpoints if endpoint_selection is None else endpoint_selection
        e = ArrayElemental({'v2v': self.e[:, :, self.selection.times]}, [self.component],
                          phase_convention='midpoint', cfg='cfg', selection=self.selection)
        p = ArrayPropagator({'vsv': self.p[list(endpoint_selection.times)][:, list(endpoint_selection.times)]},
                            time_convention='absolute', cfg='cfg', source_selection=endpoint_selection,
                            sink_selection=endpoint_selection)
        return e, p

    def test_physical_offsets_use_union_slots_and_match_full_selected_reference(self):
        full_e = ArrayElemental({'v2v': self.e}, [self.component], phase_convention='midpoint', cfg='cfg')
        full_p = ArrayPropagator({'vsv': self.p}, time_convention='absolute', cfg='cfg')
        expected = self.calculation(full_e, full_p, self.selection.times).value
        e, p = self.readers()
        actual = self.calculation(e, p, range(self.selection.Nt)).value
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        self.assertEqual(e.Nt, 3)
        self.assertEqual(p.Nt, 5)

    def test_missing_endpoint_fails_before_any_read_and_eager_path_is_explicit(self):
        e, p = self.readers(self.selection)
        with patch.object(p, 'block', side_effect=AssertionError('numeric read')):
            with self.assertRaisesRegex(KeyError, 'endpoint'):
                self.calculation(e, p, range(self.selection.Nt), execute=False)
        vertex = Meson(e, self.operator)
        with self.assertRaisesRegex(NotImplementedError, 'physical endpoint slot map'):
            list(vertex.factors(0))

    def test_complete_cfg_archive_retains_nonaffine_tables_and_checks_selection(self):
        from lqcdweave.contraction.correlator import save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation, load_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        e, p = self.readers()
        selected = calculate(self.diagram, propagator_map={'q': p},
            vertex_map={'source': Meson(e, self.operator, True), 'sink': Meson(e, self.operator)},
            configuration='cfg', usedNe=2, time_axes={0: 0, 1: 1}, optimize='none',
            memory_limit=32*2**20, execute=False)
        coordinates, _ = describe_coordinates(time_axes={0: range(3), 1: range(3)})
        readers = selected.inputs.loader.input_readers
        complete = complete_calculation(selected, coordinates=coordinates, readers=readers)
        expected = self.calculation(e, p, range(3)).value
        self.assertTrue(any(request.coordinate_tables for request in complete.plan.requests.values()))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json.gz'
            save_calculation_plan(path, complete)
            with patch('lqcdweave.contraction.expansion.legacy_diagram.LegacyDiagramBinding.describe',
                       side_effect=AssertionError('re-expanded archive')):
                actual = load_calculation(path, configuration='cfg', readers=readers).run().value
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
            wrong_record = TimeSelectionRecord(8, tuple(reversed(self.endpoints.times)))
            _, wrong = self.readers(wrong_record)
            rebound = {role: (wrong if role.startswith('propagator:') else reader)
                       for role, reader in readers.items()}
            with patch.object(wrong, 'block', side_effect=AssertionError('numeric read')):
                with self.assertRaisesRegex(ValueError, 'conventions'):
                    load_calculation(path, configuration='cfg', readers=rebound).run()


class AsymmetricSlotBindingsTests(unittest.TestCase):
    def test_same_time_meson_source_sink_records_are_checked_per_edge(self):
        rng = np.random.default_rng(19)
        e = rng.normal(size=(1, 1, 6, 1, 1)).astype(complex)
        p = rng.normal(size=(6, 6, 4, 4, 1, 1)).astype(complex)
        component = ElementalComponent(0)
        operator = Operator.from_terms('local', [OperatorTerm(1, component, 0)])
        source, sink = TimeSelectionRecord(6, (5, 1)), TimeSelectionRecord(6, (2, 0, 4))
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ['source', 'sink'], [None, 'forward', 'reverse'])
        full_e = ArrayElemental({'v2v': e}, [component], phase_convention='midpoint', cfg='cfg')
        full_p = ArrayPropagator({'vsv': p}, time_convention='absolute', cfg='cfg')
        def run(vertices, propagators, times):
            return calculate(diagram, vertex_map=vertices, propagator_map=propagators, configuration='cfg',
                             time_axes=times, usedNe=1, optimize='none', memory_limit=32*2**20).value
        expected = run({'source': Meson(full_e, operator, True), 'sink': Meson(full_e, operator)},
                       {'forward': full_p, 'reverse': full_p}, {0: source.times, 1: sink.times})
        readers = [ArrayElemental({'v2v': e[:, :, record.times]}, [component],
                                  phase_convention='midpoint', cfg='cfg', selection=record)
                   for record in (source, sink)]
        forward = ArrayPropagator({'vsv': p[list(source.times)][:, list(sink.times)]}, time_convention='absolute',
                                   cfg='cfg', source_selection=source, sink_selection=sink)
        reverse = ArrayPropagator({'vsv': p[list(sink.times)][:, list(source.times)]}, time_convention='absolute',
                                   cfg='cfg', source_selection=sink, sink_selection=source)
        actual = run({'source': Meson(readers[0], operator, True), 'sink': Meson(readers[1], operator)},
                     {'forward': forward, 'reverse': reverse}, {0: range(source.Nt), 1: range(sink.Nt)})
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_same_time_baryon_three_edges_accept_distinct_source_sink_records(self):
        import test_baryon_diagram_frontend as fixture
        from lqcdweave.adapters.elemental import ArrayBaryonElemental
        from lqcdweave.contraction.binding.api import Baryon
        case = fixture._case()
        source, sink = TimeSelectionRecord(2, (1,)), TimeSelectionRecord(2, (1, 0))
        def run(vertices, propagators, source_time, sink_time):
            return calculate(case['diagram'], vertex_map=vertices, propagator_map=propagators,
                             configuration='cfg-a', time_axes={0: source_time, 1: sink_time},
                             optimize='none', memory_limit=32*2**20).value
        compact_vertices = {}
        for role, record in (('source', source), ('sink', sink)):
            vertex = case['vertices'][role]
            reader = ArrayBaryonElemental({'vvv': vertex.reader.sectors['vvv'][:, :, record.times]},
                vertex.reader.components, cfg='cfg-a', selection=record)
            compact_vertices[role] = Baryon(reader, vertex.operator, source=role=='source')
        compact_props = {}
        for role, reader in case['propagators'].items():
            compact_props[role] = ArrayPropagator(
                {'vsv': reader.sectors['vsv'][list(source.times)][:, list(sink.times)]},
                time_convention='absolute', cfg='cfg-a', source_selection=source, sink_selection=sink)
        for slot, physical in enumerate(sink.times):
            expected = run(case['vertices'], case['propagators'], 1, physical)
            actual = run(compact_vertices, compact_props, 0, slot)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-8)
        # Wrong ordered sink map on one of the three lines is rejected at binding.
        original = compact_props['p1']
        compact_props['p1'] = ArrayPropagator({'vsv': original.sectors['vsv']}, time_convention='absolute',
            cfg='cfg-a', source_selection=source, sink_selection=TimeSelectionRecord(2, (0, 1)))
        with self.assertRaisesRegex(ValueError, 'ordered time selections'):
            run(compact_vertices, compact_props, 0, 0)


class MixedBaryonTemporalSlotTests(unittest.TestCase):
    def _case(self, include_normal):
        import test_baryon_diagram_frontend as fixture
        from lqcdweave.adapters.elemental import ArrayBaryonElemental
        from lqcdweave.contraction.binding.api import Baryon
        case = fixture._mixed_case((2, 0, 1), include_normal=include_normal)
        physical = dict(source=3, current=4, sink=1, normal=2)
        records = dict(source=TimeSelectionRecord(5, (3, 1)), sink=TimeSelectionRecord(5, (1, 4)),
                       current=TimeSelectionRecord(5, (4, 0, 2)), normal=TimeSelectionRecord(5, (2, 0)))
        vertices = {}
        for role, vertex in case['vertices'].items():
            record = records[role]
            sectors = {sector: array[:, :, record.times] for sector, array in vertex.reader.sectors.items()}
            cls = ArrayBaryonElemental if isinstance(vertex, Baryon) else ArrayElemental
            reader = cls(sectors, vertex.reader.components, phase_convention=vertex.phase_convention,
                         cfg='cfg-mixed', selection=record)
            vertices[role] = vertex.with_reader(reader, configuration='cfg-mixed')
        current_endpoints = TimeSelectionRecord(5, (0, 4))
        routes = {'p-in': (records['source'], records['normal'] if include_normal else current_endpoints),
                  'p-out': (current_endpoints, records['sink']),
                  'p-direct0': (records['source'], records['sink']),
                  'p-direct1': (TimeSelectionRecord(5, (1, 3)), TimeSelectionRecord(5, (4, 1)))}
        if include_normal:
            routes['p-mid'] = (records['normal'], TimeSelectionRecord(5, (4, 0)))
        props = {}
        for role, reader in case['propagators'].items():
            source, sink = routes[role]
            props[role] = ArrayPropagator({'vsv': reader.sectors['vsv'][list(source.times)][:, list(sink.times)]},
                time_convention='absolute', cfg='cfg-mixed', source_selection=source, sink_selection=sink)
        return fixture, case, vertices, props, physical, records

    @staticmethod
    def _calculate(case, vertices, props, times, *, execute=True):
        axes = {label: times[role] for label, role in zip(case['diagram'].time_list, case['diagram'].vertex_list)}
        return calculate(case['diagram'], vertex_map=vertices, propagator_map=props, configuration='cfg-mixed',
                         usedNe=2, time_axes=axes, optimize='none', memory_limit=64*2**20, execute=execute)

    @staticmethod
    def _oracle(fixture, case, times, include_normal):
        source_time, sink_time, current_time = times['source'], times['sink'], times['current']
        current = case['vertices']['current']
        effective = np.zeros((4, 4, 2, 2), complex)
        descriptors = current.describe_factors(current_time, 'v2v', usedNe=2)
        for descriptor in descriptors:
            if include_normal:
                normal = case['vertices']['normal']
                normal_descriptor, = normal.describe_factors(times['normal'], 'v2v', usedNe=2)
                first = case['propagators']['p-in'].block(times['normal'], source_time, usedNe=2)
                middle = case['propagators']['p-mid'].block(descriptor.bar_time, times['normal'], usedNe=2)
                first = fixture._hand_current_chain(first, fixture._hand_current_factor(normal.reader, normal_descriptor), middle)
                factor = normal_descriptor.operator_term.coefficient
            else:
                first = case['propagators']['p-in'].block(descriptor.bar_time, source_time, usedNe=2)
                factor = 1
            last = case['propagators']['p-out'].block(sink_time, descriptor.field_time, usedNe=2)
            effective += factor * descriptor.operator_term.coefficient * fixture._hand_current_chain(
                first, fixture._hand_current_factor(current.reader, descriptor), last)
        return fixture._independent_oracle(dict(case, arrays=(
            case['propagators']['p-direct0'].block(sink_time, source_time, usedNe=2),
            case['propagators']['p-direct1'].block(sink_time, source_time, usedNe=2), effective)))

    def test_wrapped_temporal_current_with_baryons_matches_hand_chain_and_fulltime(self):
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        for include_normal in (False, True):
            with self.subTest(include_normal=include_normal):
                fixture, case, vertices, props, physical, records = self._case(include_normal)
                expected = self._oracle(fixture, case, physical, include_normal)
                full = self._calculate(case, case['vertices'], case['propagators'], physical).value
                slots = {role: records[role].slot(physical[role]) for role in vertices}
                planned = self._calculate(case, vertices, props, slots, execute=False)
                actual = planned.run().value
                np.testing.assert_allclose(full, expected, rtol=3e-12, atol=5e-7)
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=5e-7)
                complete = complete_calculation(planned, readers=planned.inputs.loader.input_readers)
                np.testing.assert_allclose(np.asarray(complete.run().value).reshape(-1),
                    np.asarray(expected).reshape(-1), rtol=3e-12, atol=5e-7)

    def test_missing_connected_current_endpoint_rejected_before_reads(self):
        from contextlib import ExitStack
        _fixture, case, vertices, props, physical, records = self._case(False)
        source, sink = TimeSelectionRecord(5, (4,)), records['sink']
        full = case['propagators']['p-out'].sectors['vsv']
        props['p-out'] = ArrayPropagator({'vsv': full[list(source.times)][:, list(sink.times)]},
            time_convention='absolute', cfg='cfg-mixed', source_selection=source, sink_selection=sink)
        slots = {role: records[role].slot(physical[role]) for role in vertices}
        with ExitStack() as stack:
            for reader in [*(vertex.reader for vertex in vertices.values()), *props.values()]:
                stack.enter_context(patch.object(reader, 'block', side_effect=AssertionError('numeric read')))
            with self.assertRaises(KeyError):
                self._calculate(case, vertices, props, slots, execute=False)


class UnusedSlotReaderTests(unittest.TestCase):
    def test_unused_reader_and_unreferenced_declared_symbol_do_not_set_time_domain(self):
        record = TimeSelectionRecord(8, (7, 2))
        component = ElementalComponent(0)
        e = ArrayElemental({'v2v': np.ones((1, 1, 2, 1, 1), complex)}, [component],
                          phase_convention='midpoint', selection=record, cfg='cfg')
        p = ArrayPropagator({'vsv': np.ones((2, 2, 4, 4, 1, 1), complex)}, time_convention='absolute',
                            source_selection=record, sink_selection=record, cfg='cfg')
        unused = ArrayPropagator({'vsv': np.ones((3, 3, 4, 4, 1, 1), complex)},
                                time_convention='absolute', cfg='unused-cfg')
        operator = Operator.from_terms('local', [OperatorTerm(1, component, 0)])
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['source', 'sink'], [None, 'q', 'unused'])
        options = dict(vertex_map={'source': Meson(e, operator, True), 'sink': Meson(e, operator)},
                       configuration='cfg', time_axes={0: 0, 1: 1}, usedNe=1,
                       optimize='none', memory_limit=32*2**20)
        expected = calculate(diagram, propagator_map={'q': p}, **options).value
        with patch.object(unused, 'describe_block', side_effect=AssertionError('unused reader described')):
            actual = calculate(diagram, propagator_map={'q': p, 'unused': unused}, **options).value
        np.testing.assert_array_equal(actual, expected)
