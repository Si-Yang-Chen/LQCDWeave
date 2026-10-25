"""Module tests split from test_qdp_production_blocks.py; shared numerical fixtures stay unchanged."""

from test_qdp_production_blocks import (
    Path,
    QDPTimesliceField,
    _QDPProductionBlockFixture,
    np,
    patch,
)


class QDPProductionBlockTests(_QDPProductionBlockFixture):
    def test_slices_read_only_requested_records(self):
        for key in ((1,slice(0,1)), (slice(None,None,-1),slice(2,0,-1)),
                    (slice(0,2),slice(1,3)), (1,slice(0,0)),
                    (slice(0,0),slice(None)), (1,slice(0,2),0,0)):
            with self.subTest(key=key):
                with patch.object(self.reader, "_record", wraps=self.reader._record) as read:
                    actual = self.reader[key]
                np.testing.assert_array_equal(actual, self.vectors[key])
                ts = [key[0]] if type(key[0]) is int else range(*key[0].indices(self.sizes[3]))
                ns = range(*key[1].indices(3))
                self.assertEqual([c.args for c in read.call_args_list], [(t,n) for t in ts for n in ns])
        with self.assertRaises(ValueError):
            self.reader[0, ::0]
        with self.assertRaises(IndexError):
            self.reader[self.sizes[3], :1]

    def test_mode_block_handle_reads_only_selected_qdp_records(self):
        path = Path(self.temp.name) / "basis.mod"
        handle = QDPTimesliceField(str(path.parent)+"/", path.name,
            role="eigenvector", count=3, dtype=">c16")
        handle.read_modes("",0,0,1)  # Open the record index once.
        with patch.object(handle._stream_array,"_record",
                          wraps=handle._stream_array._record) as read:
            block = handle.read_modes("",1,1,3)
        np.testing.assert_array_equal(block,self.vectors[1,1:3])
        self.assertEqual([call.args for call in read.call_args_list],[(1,1),(1,2)])
