"""Warm layout reuse stays bounded and never keeps current numerical owners."""
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution import reduction_kernel as raw
from correlator.execution.test_raw_precise_reduction import _SourceBackend


def operands(step):
    left=np.arange(17*step,dtype=np.float32)[::step]
    right=np.linspace(-1,1,17,dtype=np.float32)[::-1]
    return left,right


class ReductionLayoutReuseTests(unittest.TestCase):
    def tearDown(self):
        raw._kernel.cache_clear()

    def test_saved_pitches_reuse_bindings_with_fresh_values_and_no_input_owners(self):
        backend=_SourceBackend()
        run=raw.bind_raw_reduction(parse_axis_pattern('i,i->'),((17,),)*2,
            np.dtype('float32'),backend,1024)
        refs=[]
        with patch.object(raw,'reduction_sources',wraps=raw.reduction_sources) as sources:
            for scale in (1,2):
                for step in range(1,41):
                    left,right=operands(step)
                    left*=scale
                    refs.append(weakref.ref(left.base))
                    reference=np.einsum('i,i->',left,right,dtype=np.float64).astype(np.float32)
                    result=run(left,right)
                    np.testing.assert_array_equal(result,reference)
            self.assertEqual(sources.call_count,40)
        del left,right,result
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertLessEqual(run.workspace_bytes,run.workspace_allowance)
        with self.assertRaisesRegex(ValueError,'bound geometry'):
            run(np.ones(18,np.float32),np.ones(18,np.float32))

    def test_least_recent_layout_is_evicted_at_the_fixed_bound(self):
        backend=_SourceBackend()
        run=raw.bind_raw_reduction(parse_axis_pattern('i,i->'),((17,),)*2,
            np.dtype('float32'),backend,1024)
        with patch.object(raw,'reduction_sources',wraps=raw.reduction_sources) as sources:
            for step in range(1,raw._MAX_LAYOUTS+2):
                run(*operands(step))
            self.assertEqual(sources.call_count,raw._MAX_LAYOUTS+1)
            run(*operands(raw._MAX_LAYOUTS+1))
            self.assertEqual(sources.call_count,raw._MAX_LAYOUTS+1)
            run(*operands(1))
            self.assertEqual(sources.call_count,raw._MAX_LAYOUTS+2)


if __name__=='__main__':
    unittest.main()
