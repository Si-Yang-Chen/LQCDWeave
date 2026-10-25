"""Ordered time views read the original physical pairs, with no eager grid."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator, select_propagator_times
from lqcdweave.contracts.time_selection import TimeSelectionRecord


class SelectedPropagatorTimesTests(unittest.TestCase):
    def test_full_relative_mapping_uses_physical_modulo_not_selected_slot_difference(self):
        record = TimeSelectionRecord(128, (126, 0, 7, 127))
        shape = (128, 128, 4, 4, 2, 3)
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(np.complex64)
        parent = ArrayPropagator({'vsv': raw}, time_convention='relative', cfg='a', data_identity='full-a')
        view = select_propagator_times(parent, record)
        self.assertEqual((view.Lt, view.Nt, view.physical_Lt), (4, 4, 128))
        self.assertEqual(view.source_selection, record)
        self.assertEqual(view.sink_selection, record)
        self.assertEqual(view.time_convention, 'absolute')
        self.assertEqual(view.data_identity, parent.data_identity)
        # Independent equivalent of the previous eager relative->absolute conversion.
        eager = np.stack([np.stack([raw[source, (sink-source) % 128]
            for sink in record.times]) for source in record.times])
        for source in range(4):
            for sink in range(4):
                np.testing.assert_array_equal(view.block(sink, source), eager[source, sink])
                self.assertTrue(np.shares_memory(view.block(sink, source), raw))
        np.testing.assert_array_equal(view.read_slab((3, 0, 3, 1), 2), eager[2, [3, 0, 3, 1]])
        np.testing.assert_array_equal(view.read_times((3, 1), (2, 0)), eager[[2, 0]][:, [3, 1]])
        np.testing.assert_array_equal(view.read_slab_in_background((0, 2), 3), eager[3, [0, 2]])
        self.assertNotEqual(float(eager[0, 1, 0, 0, 0, 0].real),
                            float(raw[0, 1, 0, 0, 0, 0].real))
        metadata = view.describe_block(1, 0)
        self.assertEqual(metadata.dtype, raw.dtype)
        self.assertEqual(metadata.data_identity, 'full-a')
        self.assertIn("128, 'relative'", metadata.selection_identity)

    def test_compact_parent_uses_its_own_declared_slots_and_nested_views_compose(self):
        parent_record = TimeSelectionRecord(128, (127, 5, 0, 17))
        record = TimeSelectionRecord(128, (0, 127, 17))
        values = np.arange(4*4*16).reshape(4, 4, 4, 4, 1, 1).astype(complex)
        for layout in ('absolute', 'relative'):
            with self.subTest(layout=layout):
                parent = ArrayPropagator({'vsv': values}, time_convention=layout,
                    source_selection=parent_record, sink_selection=parent_record)
                view = select_propagator_times(parent, record)
                nested = select_propagator_times(view, TimeSelectionRecord(128, (17, 0)))
                # physical source=127 -> parent slot 0, physical sink=0 -> parent slot 2.
                np.testing.assert_array_equal(view.block(0, 1), values[0, 2])
                parent_sink, parent_source = 2, 3
                time = parent_sink if layout == 'absolute' else (parent_sink-parent_source) % 4
                np.testing.assert_array_equal(nested.block(1, 0), values[parent_source, time])
                self.assertEqual(view.time_selection_identity,
                    tuple((axis, 128, record.times) for axis in ('sink_time', 'source_time')))

    def test_only_requested_pairs_are_read_instead_of_materializing_the_selected_grid(self):
        record = TimeSelectionRecord(128, tuple((78+16*b+o) % 128
            for b in range(8) for o in (0, 1, 4, 9, 11)))
        values = np.ones((128, 128, 4, 4, 1, 1), np.complex64)
        parent = ArrayPropagator({'vsv': values}, time_convention='relative')
        with patch.object(parent, 'block', wraps=parent.block) as reads:
            view = select_propagator_times(parent, record)
            for source in range(5):
                for sink in range(5):
                    view.describe_block(sink, source)
            self.assertEqual(reads.call_count, 0)
            for source in range(5):
                for sink in range(5):
                    np.testing.assert_array_equal(view.block(sink, source), 1)
            self.assertEqual(reads.call_count, 25)
            self.assertEqual(reads.call_args_list[0].args[:2], (record.times[0], record.times[0]))
        self.assertEqual(len(tuple(view.iter_blocks((0, 1), (2, 3)))), 4)

    def test_metadata_is_lazy_and_preserves_endpoint_labels_and_exact_mode_selection(self):
        record = TimeSelectionRecord(128, (127, 0, 4))
        class HeaderOnly:
            shape, dtype = (128, 128, 4, 4, 5, 3), np.dtype('>c16')
            def __getitem__(self, key):
                raise AssertionError('numeric read during view construction or description')
        endpoints = {('vsv', 'sink'): ('sink-basis', (9, 8, 7, 6, 5)),
                     ('vsv', 'source'): ('source-basis', (3, 2, 1))}
        parent = ArrayPropagator({'vsv': HeaderOnly()}, time_convention='relative', basis_endpoints=endpoints)
        with patch.object(parent, 'block', side_effect=AssertionError('numeric read')):
            view = select_propagator_times(parent, record)
            metadata = view.describe_block(1, 0,
                mode_indices={'sink_mode': (4, 1), 'source_mode': (2, 0)})
            self.assertEqual(metadata.shape, (4, 4, 2, 2))
            self.assertEqual(metadata.dtype, np.dtype('>c16'))
            self.assertEqual(metadata.axis_bindings['sink_mode'], ('sink-basis', (5, 8)))
            self.assertEqual(metadata.axis_bindings['source_mode'], ('source-basis', (1, 3)))
            self.assertEqual(view.describe_slab((2, 0, 2), 1, usedNe=2).shape, (3, 4, 4, 2, 2))
        self.assertFalse(view.background_read_safe)

    def test_pair_coverage_missing_endpoints_and_invalid_coordinates_are_not_fabricated(self):
        record = TimeSelectionRecord(8, (7, 0, 3))
        parent = ArrayPropagator({'vsv': {(0, 2): np.ones((4, 4, 1, 1), complex)}},
            time_convention='absolute', source_selection=record, sink_selection=record)
        view = select_propagator_times(parent, TimeSelectionRecord(8, (3, 7)))
        np.testing.assert_array_equal(view.block(1, 0), 1)
        with self.assertRaises(KeyError):
            view.describe_block(0, 0)
        with self.assertRaises(KeyError):
            select_propagator_times(parent, TimeSelectionRecord(8, (1, 7)))
        with self.assertRaisesRegex(ValueError, 'physical Lt'):
            select_propagator_times(parent, TimeSelectionRecord(9, (3, 7)))
        for sink, source in ((True, 0), (-1, 0), (0, 2)):
            with self.subTest(pair=(sink, source)), self.assertRaises((TypeError, ValueError)):
                view.describe_block(sink, source)

    def test_npy_view_borrows_parent_mapping_and_view_close_keeps_parent_open(self):
        values = np.arange(8*8*16).reshape(8, 8, 4, 4, 1, 1).astype(complex)
        record = TimeSelectionRecord(8, (7, 0, 3))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'original.npy'
            np.save(path, values)
            with ArrayPropagator.from_npy({'vsv': path}, time_convention='relative') as parent:
                with select_propagator_times(parent, record) as view:
                    self.assertFalse(view.host_resident)
                    np.testing.assert_array_equal(view.block(1, 0), values[7, 1])
                    self.assertEqual(len(list(Path(directory).iterdir())), 1)
                with self.assertRaisesRegex(ValueError, 'view is closed'):
                    view.block(0, 0)
                np.testing.assert_array_equal(parent.block(0, 7), values[7, 1])

    def test_all_nine_sectors_keep_point_mode_axes_and_slabs(self):
        record = TimeSelectionRecord(8, (7, 0, 3))
        for left in ('v', 'd', 'p'):
            for right in ('v', 'd', 'p'):
                sector = left+'s'+right
                tail = ((2, 3) if left == 'p' else (2,))+((2, 3) if right == 'p' else (2,))
                values = np.arange(np.prod((8, 8, 4, 4, *tail))).reshape(8, 8, 4, 4, *tail).astype(complex)
                parent = ArrayPropagator({sector: values}, time_convention='relative')
                view = select_propagator_times(parent, record)
                options = dict(usedNp=1) if 'p' in sector else dict(usedNe=1)
                with self.subTest(sector=sector):
                    np.testing.assert_array_equal(view.block(1, 0, sector, **options),
                        parent.block(0, 7, sector, **options))
                    np.testing.assert_array_equal(view.read_slab((2, 0, 2), 1, sector, **options),
                        parent.read_slab((3, 7, 3), 0, sector, **options))
                    self.assertEqual(view.describe_slab((2, 0), 1, sector, **options).shape,
                                     view.read_slab((2, 0), 1, sector, **options).shape)

    def test_calculate_complete_archive_replays_current_cfg_and_accepts_eager_template(self):
        from test_structured_correlator import task
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
        from lqcdweave.contraction.correlator import calculate, save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        record = TimeSelectionRecord(8, (7, 0, 3))
        rng = np.random.default_rng(872)
        random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
        component = ElementalComponent(0)
        operator = Operator.from_terms('a', (OperatorTerm(5, component, 0, 1-.2j),))
        data = {}
        for cfg in ('a', 'b'):
            values = random((8, 8, 4, 4, 2, 2))
            elemental = ArrayElemental({'v2v': random((1, 1, 3, 2, 2))}, (component,),
                phase_convention='anchor', cfg=cfg, data_identity='elemental-'+cfg, selection=record)
            parent = ArrayPropagator({'vsv': values}, time_convention='relative',
                cfg=cfg, data_identity='propagator-'+cfg)
            eager = np.stack([np.stack([values[source, (sink-source) % 8]
                for sink in record.times]) for source in record.times])
            old = ArrayPropagator({'vsv': eager}, time_convention='absolute', cfg=cfg,
                data_identity=parent.data_identity, source_selection=record, sink_selection=record)
            data[cfg] = elemental, parent, select_propagator_times(parent, record), old
        def mapping(prop):
            return {'S^q': prop, r'S^q_\mathrm{local}': prop}
        def binding(elemental):
            return lambda row: Meson(elemental, operator, row.dagger, phase_convention='anchor')
        pairs = [{0: 1, 1: 0}, {0: 2, 1: 0}, {0: 0, 1: 2}]
        request = task(structured=True, flavor='I0', names=('a',))
        expected = {cfg: calculate(request, vertex_map=binding(e), propagator_map=mapping(old),
            configuration=cfg, time_pairs=pairs, optimize='none', memory_limit=1<<28).value
            for cfg, (e, _, _, old) in data.items()}
        coordinates, _ = describe_coordinates(time_pairs=pairs)
        for old_template in (False, True):
            with self.subTest(eager_template=old_template), TemporaryDirectory() as directory:
                e, _, prop, old = data['a']
                selected = calculate(request, vertex_map=binding(e),
                    propagator_map=mapping(old if old_template else prop), configuration='a',
                    time_axes={0:1, 1:0}, optimize='none', memory_limit=1<<28, execute=False)
                complete = complete_calculation(selected, coordinates=coordinates)
                if not old_template:
                    self.assertTrue(all('propagator_time_view' in dict(r.conventions)
                        for r in complete.plan.requests.values() if r.family == 'propagator'))
                archive = Path(directory)/'complete.plan.gz'
                save_calculation_plan(archive, complete)
                @contextmanager
                def factory(cfg):
                    elemental, _, view, _ = data[cfg]
                    yield {r.reader_role: view if r.family == 'propagator' else elemental
                        for primary in complete.plan.requests.values() for r in (primary, *primary.aliases)}
                with ExitStack() as blocked:
                    for name in ('lqcdweave.contraction.correlator.expand_correlator',
                                 'lqcdweave.contraction.correlator.find_contraction_path',
                                 'lqcdweave.contraction.planning.cfg_plan._bind_cfg_coordinates'):
                        blocked.enter_context(patch(name, side_effect=AssertionError(name)))
                    actual = calculate(plan_archive=archive, configuration=iter(('a', 'b')), data=factory)
                    outputs = dict(actual.value)
                self.assertEqual(tuple(outputs), ('a', 'b'))
                for cfg in expected:
                    np.testing.assert_allclose(outputs[cfg], expected[cfg], rtol=3e-12, atol=3e-12)
                # Wrong actual ordered endpoints are rejected before numerical IO.
                changed = select_propagator_times(data['a'][1], TimeSelectionRecord(8, (0, 7, 3)))
                with patch.object(data['a'][1], 'block', side_effect=AssertionError('read before validation')):
                    with self.assertRaisesRegex(ValueError, 'conventions'):
                        calculate(plan_archive=archive, configuration='a', vertex_map=e,
                                  propagator_map=mapping(changed))
                if not old_template:
                    # Same selected slots and source ID cannot conceal a changed
                    # interpretation of the original parent's second axis.
                    parent = data['a'][1]
                    wrong_parent = ArrayPropagator(parent.sectors, time_convention='absolute',
                        cfg='a', data_identity=parent.data_identity)
                    changed = select_propagator_times(wrong_parent, record)
                    with patch.object(wrong_parent, 'block', side_effect=AssertionError('read before validation')):
                        with self.assertRaisesRegex(ValueError, 'conventions'):
                            calculate(plan_archive=archive, configuration='a', vertex_map=e,
                                      propagator_map=mapping(changed))


if __name__ == '__main__':
    unittest.main()
