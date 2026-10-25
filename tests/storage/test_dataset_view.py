"""Storage-owned selection mechanics after explicit scientific admission."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetReader, DatasetView, DatasetWriter


def spec(modes=(4, 1, 7)):
    return DatasetSpec(dict(schema=1, kind='view-fixture',
        configuration=dict(ensemble='test', configuration='cfg', lattice=[1, 1, 1, 2]),
        axes=[dict(name='time', coordinates=[1, 0]), dict(name='mode', coordinates=list(modes))],
        dtype='complex128', bases={}, definition={}, conventions={}, parents=[]))


class DatasetViewTests(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path=Path(tmp.name)/'data'
        with DatasetWriter(path,spec(),provenance={}) as writer:
            for i in range(2): writer.write(i,np.array([4+10*i,1+10*i,7+10*i],dtype=complex))
            writer.publish()
        self.reader=DatasetReader(path); self.addCleanup(self.reader.close)

    def test_direct_storage_view_reorders_without_planning_or_extra_registration(self):
        with patch('lqcdweave.planning.dataset.plan_requirement',side_effect=AssertionError('storage planned')):
            view=DatasetView(self.reader,spec((7,4)))
            values=view.read(time=1)
        np.testing.assert_array_equal(values,[7,4])
        self.assertFalse(values.flags.writeable)
        self.assertFalse(np.shares_memory(values,self.reader.read(time=1)))
        self.assertEqual(view.data_id,self.reader.data_id)
        self.assertEqual(view.provenance,self.reader.provenance)
        self.assertEqual([x for x,_ in view.iter_read([0,1])],[0,1])
        for selection in ([7,7],[1],[]):
            with self.assertRaises(KeyError): view.read(time=1,select={'mode':selection})
        with self.assertRaises(KeyError): view.read(time=2)
        with self.assertRaises(ValueError): view.read()
        view.close()
        with self.assertRaisesRegex(ValueError,'closed'): view.read(time=1)

    def test_contiguous_requested_range_borrows_readonly_basic_slice(self):
        view=DatasetView(self.reader,spec((4,1)))
        value=view.read(time=1)
        self.assertTrue(np.shares_memory(value,self.reader.read(time=1)))
        self.assertFalse(value.flags.writeable)
        np.testing.assert_array_equal(value,[4,1])
