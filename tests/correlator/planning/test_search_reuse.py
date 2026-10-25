"""Reuse must preserve arrival reopening, resource checks and interrupted work."""
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture, ContractionTerm


class SearchReuseTests(unittest.TestCase):
    def test_pressure_cache_reuses_only_algebra_and_rechecks_physical_steps(self):
        terms=(ContractionTerm(('A','B','C'),'i,i,i->'),
               ContractionTerm(('A','C'),'i,i->i'), ContractionTerm(('B','C'),'i,i->i'))
        space=fixture(terms,cap=264)
        first=space.successors(space.initial_state())[0]
        adjustments=space.successors(first.state)
        self.assertTrue(space.logical_cache)
        adjusted=adjustments[0].state
        cached=tuple(space.logical_steps(adjusted))
        self.assertGreater(space.logical_cache_hits,0)
        space.logical_cache=None
        direct=tuple(space.logical_steps(adjusted))
        self.assertEqual(cached,direct)
        # Cache the same algebra again; resource checks must still run.
        space.successors(first.state)
        with patch.object(space,'_build_logical_steps',side_effect=AssertionError('rebuilt algebra')):
            before=space.aggregate_checks
            actual=space.successors(adjusted)
            self.assertGreater(space.aggregate_checks,before)
        space.logical_cache=None
        self.assertEqual(actual,space.successors(adjusted))

    def test_feasible_or_interrupted_expansion_does_not_publish_pressure_cache(self):
        space=fixture()
        self.assertTrue(space.successors(space.initial_state()))
        self.assertIsNone(space.logical_cache)
        calls=[]
        def stop():
            calls.append(1)
            if len(calls)==4: raise RuntimeError('stop')
        with self.assertRaises(RuntimeError): space.successors(space.initial_state(),stop)
        self.assertIsNone(space.logical_cache)


if __name__=='__main__': unittest.main()
