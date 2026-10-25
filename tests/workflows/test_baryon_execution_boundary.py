"""Compile convenience belongs to W; compiled X execution cannot plan."""
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.correlator.api import BaryonTerm, BaryonVertex, TensorSpec
from lqcdweave.contraction.planning.api import ContractionPlanCache, compile_baryon_terms
from lqcdweave.contraction.execution.api import execute_plan
from lqcdweave.contraction.contraction_session import execute_baryon_terms


class BaryonExecutionBoundaryTests(unittest.TestCase):
    def test_cache_and_pure_execution_preserve_exact_scalar_result(self):
        keys=('s','e','r','f','p0','p1','p2')
        terms=(BaryonTerm(BaryonVertex('s','e'),BaryonVertex('r','f'),keys[4:]),)
        bindings={k:np.full((1,)*(3 if i<4 else 4),i+1+1j,dtype=complex) for i,k in enumerate(keys)}
        specs={k:TensorSpec(v.shape,v.dtype) for k,v in bindings.items()}
        plan=compile_baryon_terms(specs,terms)
        cache=ContractionPlanCache()
        first=execute_baryon_terms(bindings,terms,plan_cache=cache)
        second=execute_baryon_terms(bindings,terms,plan_cache=cache)
        expected=1+0j
        for i in range(1,8): expected *= i+1j
        np.testing.assert_array_equal(first.values[0],expected)
        np.testing.assert_array_equal(first.values[0],second.values[0])
        with patch('lqcdweave.contraction.planning.api.compile_plan',side_effect=AssertionError('X compiled')), \
             patch('lqcdweave.contraction.planning.api.compile_baryon_terms_cached',side_effect=AssertionError('X compiled')):
            actual=execute_plan(plan,bindings)
        np.testing.assert_array_equal(first.values[0],actual.values[0])
        self.assertFalse(np.shares_memory(actual.values[0],bindings['s']))
