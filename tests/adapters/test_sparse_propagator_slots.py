"""Compact consumer axes are slots, with explicit physical metadata."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.time_selection import TimeSelectionRecord


class SparsePropagatorSlotTests(unittest.TestCase):
    def setUp(self):
        self.source = TimeSelectionRecord(8,(7,1,4))
        self.sink = TimeSelectionRecord(8,(6,0))
        shapes = {'vsv':(3,2,4,4,2,1), 'psv':(3,2,4,4,2,3,1),
                  'vsp':(3,2,4,4,2,1,3), 'psp':(3,2,4,4,2,3,1,3)}
        shapes = {a+'s'+b: shapes[(a+'s'+b).replace('d', 'v')]
                  for a in ('v', 'd', 'p') for b in ('v', 'd', 'p')}
        self.values = {name:np.arange(np.prod(shape)).reshape(shape).astype(complex)
                       for name,shape in shapes.items()}
        self.options = dict(time_convention='absolute', source_selection=self.source,
                            sink_selection=self.sink)

    def test_explicit_pair_blocks_keep_ordered_selection_and_reject_holes(self):
        pairs = ((0, 2), (1, 2), (1, 0))
        sectors = {k: {pair: v[pair[1], pair[0]] for pair in pairs}
                   for k, v in self.values.items()}
        reader = ArrayPropagator(sectors, **self.options)
        self.assertTrue(reader.background_read_safe)
        for product, values in self.values.items():
            np.testing.assert_array_equal(reader.block(1, 2, product), values[2, 1])
            selected = dict(usedNe=1) if 'p' not in product else dict(usedNp=1)
            expected = ArrayPropagator({product: values}, **self.options)
            np.testing.assert_array_equal(reader.read_slab((1, 0, 1), 2, product, **selected),
                expected.read_slab((1, 0, 1), 2, product, **selected))
            self.assertEqual(reader.describe_slab((1, 0, 1), 2, product, **selected).shape,
                             reader.read_slab((1, 0, 1), 2, product, **selected).shape)
            for operation in (reader.block, reader.describe_block):
                with self.assertRaises(KeyError): operation(0, 0, product)
            for operation in (reader.read_slab, reader.describe_slab):
                with self.assertRaises(KeyError): operation((1, 0), 0, product)
        # Pair coverage is frozen; values continue to borrow the input arrays.
        sectors['vsv'].clear()
        np.testing.assert_array_equal(reader.block(0, 2), self.values['vsv'][2, 0])

    def test_pair_headers_never_materialize_values(self):
        class HeaderOnly:
            shape, dtype = (4, 4, 2, 1), np.dtype('complex128')
            def __array__(self, *args, **kwargs):
                raise AssertionError('numeric read during metadata planning')
            def __getitem__(self, key):
                raise AssertionError('numeric read during metadata planning')
        reader = ArrayPropagator({'vsv': {(1, 2): HeaderOnly()}}, **self.options)
        self.assertFalse(reader.background_read_safe)
        self.assertEqual(reader.describe_block(1, 2).shape, (4, 4, 2, 1))
        self.assertEqual(reader.describe_slab((1, 1), 2).shape, (2, 4, 4, 2, 1))
        for options in (dict(time_convention='absolute'),
                        dict(self.options, time_convention='relative')):
            with self.assertRaisesRegex(ValueError, 'explicit|identical'):
                ArrayPropagator({'vsv': {(1, 2): HeaderOnly()}}, **options)
        with self.assertRaisesRegex(ValueError, 'within selections'):
            ArrayPropagator({'vsv': {(2, 0): HeaderOnly()}}, **self.options)

    def test_all_products_keep_asymmetric_slots_and_physical_identity(self):
        with ArrayPropagator(self.values,**self.options) as reader:
            self.assertEqual((reader.Lt,reader.Nt,reader.physical_Lt),(3,3,8))
            self.assertIsNone(reader.time_selection)
            self.assertEqual(reader.time_selections['source_time'],self.source)
            self.assertEqual(reader.time_selections['sink_time'],self.sink)
            for product,values in self.values.items():
                np.testing.assert_array_equal(reader.block(1,2,product),values[2,1])
                np.testing.assert_array_equal(reader.read_slab([1,0,1],2,product),values[2,[1,0,1]])
                np.testing.assert_array_equal(reader.read_times([1,0],[2,0],product),values[[2,0]][:,[1,0]])
                with self.assertRaises(KeyError): reader.block(2,0,product)
                with self.assertRaises(KeyError): reader.describe_block(0,3,product)
                with self.assertRaises(ValueError): reader.read_slab([2],0,product)
            changed = ArrayPropagator(self.values,**dict(self.options,
                        source_selection=TimeSelectionRecord(8,(1,7,4))))
            self.assertNotEqual(reader.time_selection_identity,changed.time_selection_identity)
            self.assertNotEqual(reader.describe_block(0,0).selection_identity,
                                changed.describe_block(0,0).selection_identity)
            self.assertNotEqual(reader.describe_slab([0],0).selection_identity,
                                changed.describe_slab([0],0).selection_identity)

    def test_compact_npy_binary_and_source_files_share_slot_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            npy,binary,slices = {},{},{}
            for product,values in self.values.items():
                npy[product] = root/f'{product}.npy'; np.save(npy[product],values)
                binary[product] = root/f'{product}.bin'; values.tofile(binary[product])
                slices[product] = []
                for slot,row in enumerate(values):
                    path = root/f'{product}-{slot}.npy'; np.save(path,row); slices[product].append(path)
            readers = [ArrayPropagator.from_npy(npy,**self.options),
                ArrayPropagator.from_binary(binary,shapes={k:v.shape for k,v in self.values.items()},
                    dtypes={k:v.dtype for k,v in self.values.items()},**self.options),
                ArrayPropagator.from_npy_timeslices(slices,source_times=(2,0),**self.options)]
            for reader in readers:
                with reader:
                    for product,values in self.values.items():
                        np.testing.assert_array_equal(reader.block(0,2,product),values[2,0])
                        self.assertEqual(reader.physical_Lt,8)

    def test_relative_compact_layout_is_explicit_cyclic_slot_difference(self):
        values = np.arange(3*3*16).reshape(3,3,4,4,1,1).astype(complex)
        record = TimeSelectionRecord(8,(7,0,3))
        reader = ArrayPropagator({'vsv':values},time_convention='relative',
                                source_selection=record,sink_selection=record)
        self.assertEqual(reader.time_selection,record)
        np.testing.assert_array_equal(reader.block(0,2),values[2,1])
        np.testing.assert_array_equal(reader.read_slab((2,0),1),values[1,[1,2]])
        with self.assertRaisesRegex(ValueError,'identical'):
            ArrayPropagator(self.values,**dict(self.options,time_convention='relative'))
        with self.assertRaisesRegex(ValueError,'explicit'):
            ArrayPropagator(self.values,time_convention='absolute')
        with self.assertRaisesRegex(ValueError,'shape'):
            ArrayPropagator(self.values,**dict(self.options,sink_selection=record))
        full = ArrayPropagator({'vsv':values},time_convention='absolute')
        self.assertIsNone(full.time_selection_identity)
