"""Immutable metadata reuse preserves validation, records and logical owners."""
from dataclasses import fields, replace
import pickle
import unittest

import numpy as np

from lqcdweave.contracts.correlator.factorized_reduction import FactorizedWideReduction
from lqcdweave.contracts.correlator.factorized_geometry import _cached_geometry
from lqcdweave.contracts.correlator.indices import parse_axis_pattern
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


def core(*, output=None, tile=4, dtype=None):
    equation=parse_axis_pattern('ak,bk,cax,dby->cd')
    if output is not None:equation=replace(equation,output=output)
    return FactorizedWideReduction(equation,((3,7),(5,7),(2,3,3),(2,5,2)),(0,1),
                                   tile_elements=tile,output_dtype=dtype)


class FactorizedGeometryTests(unittest.TestCase):
    def test_shared_metadata_never_merges_logical_core_owners(self):
        a=core();b=replace(a)
        self.assertIsNot(a,b)
        self.assertEqual(a,b);self.assertEqual(hash(a),hash(b))
        self.assertIs(a._geometry,b._geometry)
        dims=a.dimensions;dims.clear()
        self.assertTrue(a.dimensions);self.assertTrue(b.dimensions)
        self.assertEqual(a.gram_elements,15)
        self.assertEqual(a.epilogue_elements,15)
        self.assertEqual(a.reduction_elements,7)
        self.assertEqual(a.small_reduction_elements,(3,2))
        self.assertEqual(a.output_shape,(2,2))
        self.assertEqual((a.tile_width,a.tiles,a.partial_elements,a.partial_bytes),(4,4,4,64))
        self.assertEqual(a.accumulator_bytes,64)
        self.assertEqual(a.kernel_calls,8)

    def test_full_order_tile_and_precision_keys_keep_distinct_geometry(self):
        a=core();changed=[core(output=a.equation.output[::-1]),core(tile=3),core(dtype='complex128')]
        for b in changed:
            self.assertNotEqual(a,b);self.assertIsNot(a._geometry,b._geometry)
        self.assertEqual(changed[-1].result_dtype,'complex128')
        self.assertEqual(changed[-1].traffic_bytes-a.traffic_bytes,a.output_elements*8)
        self.assertEqual([f.name for f in fields(a)],
            ['equation','shapes','large_operands','dtype','tile_elements','output_dtype','gram_strategy'])
        self.assertEqual(_cached_geometry.cache_info().maxsize,2048)

    def test_codec_pickle_and_legacy_pickle_keep_original_precision(self):
        for original in [core(),core(dtype='complex128')]:
            restored=decode_plan_value(encode_plan_value(original))
            self.assertEqual(original,restored)
            self.assertIsNot(original,restored)
            self.assertEqual(restored.result_dtype,original.result_dtype)
            self.assertEqual(pickle.loads(pickle.dumps(original)),original)
            legacy=replace(original);object.__delattr__(legacy,'_geometry')
            restored=pickle.loads(pickle.dumps(legacy))
            self.assertEqual(restored.workspace_bytes,original.workspace_bytes)
            self.assertEqual(restored.traffic_bytes,original.traffic_bytes)

    def test_cache_hits_do_not_hide_invalid_dimensions_or_domain(self):
        valid=core()
        shapes=list(valid.shapes)
        for bad in [True,np.int64(3),0,-1,[],3.0]:
            changed=list(shapes);changed[0]=(bad,7)
            with self.subTest(size=repr(bad)),self.assertRaises(ValueError):
                replace(valid,shapes=changed)
        for changes in [dict(tile_elements=True),dict(tile_elements=0),dict(tile_elements=65537),
                        dict(large_operands=(1,0)),dict(dtype='complex128'),
                        dict(output_dtype='float64'),dict(shapes=((3,7),(5,8),(2,3,3),(2,5,2)))]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(valid,**changes)


if __name__=='__main__':unittest.main()
