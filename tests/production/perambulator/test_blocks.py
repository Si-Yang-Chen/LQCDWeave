"""Callback delivery against independently embedded full Dirac inverses."""

import unittest

import numpy as np

from production.perambulator import test_dilution as fixtures
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.perambulator.api import PointBasis


class BlockProductionTests(unittest.TestCase):
    def setUp(self):
        self.reference = fixtures.DilutionTests()
        self.reference.setUp()

    def test_all_nine_products_have_exact_callback_coverage(self):
        f = self.reference
        for streamed, dtype in ((False,'complex128'),(True,'complex64')):
            for source_kind in ('V','D'):
                for sink_kind in ('V','D'):
                    with self.subTest(streamed=streamed,dtype=dtype,source=source_kind,sink=sink_kind):
                        gen=f.generator(source_kind,sink_kind,contract_prec=dtype,
                            stream_eigenvectors=streamed,sink_ne_block_size=2,
                            sink_time_block_size=1,
                            source_selection=TimeSelectionRecord(3,(2,0)),
                            sink_selection=TimeSelectionRecord(3,(2,0)))
                        gen.load('cfg')
                        for source_basis, products in (
                            (f.src_basis,(sink_kind+'S'+source_kind,'PS'+source_kind)),
                            (f.point_src_basis,(sink_kind+'SP','PSP')),
                        ):
                            for time in (2,0):
                                expected={p:np.stack([f.oracle(p,t,time) for t in (2,0)])
                                          for p in products}
                                actual={p:np.empty(value.shape,dtype=dtype)
                                        for p,value in expected.items()}
                                coverage={p:np.zeros(value.shape,dtype=np.uint8)
                                          for p,value in expected.items()}

                                def write(sink_index,selection,value):
                                    product=products[sink_index]
                                    self.assertEqual(value.dtype,np.dtype(dtype))
                                    self.assertEqual(value.shape,actual[product][selection].shape)
                                    self.assertLessEqual(value.shape[0],gen.sink_time_block_size)
                                    actual[product][selection]=value
                                    coverage[product][selection]+=1

                                results=gen.calc(time,src=source_basis,sink=f.sink_bases,write=write)
                                self.assertTrue(all(result.values is None for result in results))
                                for product in products:
                                    np.testing.assert_array_equal(coverage[product],1)
                                    tolerance=2e-6 if dtype=='complex64' else 1e-12
                                    atol=3e-7 if dtype=='complex64' else 3e-14
                                    np.testing.assert_allclose(actual[product],expected[product],rtol=tolerance,atol=atol)
                        self.assertEqual(gen.backend.rhs_count,2*(4*f.data[source_kind].shape[1]+24))

    def test_callback_failure_does_not_publish_metrics_or_results(self):
        f=self.reference
        gen=f.generator('V','D',source_selection=TimeSelectionRecord(3,(2,0)))
        gen.load('cfg')
        calls=[]
        def fail(sink_index,selection,value):
            calls.append(sink_index)
            raise OSError('writer failed')
        with self.assertRaisesRegex(OSError,'writer failed'):
            gen.calc(0,src=f.src_basis,sink=f.sink_bases,write=fail)
        self.assertTrue(calls)
        self.assertEqual(gen.last_metrics,{})

    def test_callback_delivery_does_not_change_an_earlier_owned_result(self):
        f=self.reference
        gen=f.generator('V','V')
        gen.load('cfg')
        dense=gen.calc(0,src=f.src_basis,sink=[f.sink_v_basis])[0]
        saved=dense.values.copy()
        received=[]
        gen.calc(2,src=f.src_basis,sink=[f.sink_v_basis],
                 write=lambda sink_index,selection,value:received.append(value.copy()))
        np.testing.assert_array_equal(dense.values,saved)
        self.assertTrue(received)
        self.assertTrue(dense.values.flags.owndata)

    def test_invalid_callback_requests_fail_before_solving(self):
        f=self.reference
        gen=f.generator('D','D',source_selection=TimeSelectionRecord(3,(2,0)))
        gen.load('cfg')
        with self.assertRaises(TypeError):
            gen.calc(0,src=f.src_basis,sink=[])
        with self.assertRaises(TypeError):
            gen.calc(0,src=f.src_basis,sink=f.sink_bases,write=object())
        with self.assertRaises(ValueError):
            gen.calc(0,src=f.src_basis,sink=[None],write=lambda *args:None)
        with self.assertRaises(KeyError):
            gen.calc(1,src=f.src_basis,sink=f.sink_bases)
        self.assertEqual(gen.backend.rhs_count,0)


if __name__=='__main__':
    unittest.main()
