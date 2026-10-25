"""Independent checks of compact metadata, state identity and resource bounds."""
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture, ContractionTerm
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contracts.correlator.api import Factor, ResourceAllocation, ResourceSnapshot
from lqcdweave.contraction.planning.state import _ImmutableKey
from lqcdweave.contraction.planning.paths import _state_key
from lqcdweave.contraction.planning.resources import resource_point
from lqcdweave.contraction.planning.optimizers.budget import retained_size


class ResourceCompactionTests(unittest.TestCase):
    def test_catalog_snapshots_are_owned_immutable_and_shared_by_replacements(self):
        state=fixture().initial_state().state
        mutable=dict(state.catalog)
        external=replace(state,catalog=MappingProxyType(mutable))
        mutable.clear()
        self.assertEqual(dict(external.catalog),dict(state.catalog))
        sibling=replace(state,seconds=1)
        self.assertIs(sibling.catalog,state.catalog)
        with self.assertRaises(TypeError): sibling.catalog['A']=state.catalog['A']
        with self.assertRaises(AttributeError): sibling.catalog._values={}
        loaded=state.load('A',fixture().logical_profile,10000)
        self.assertIs(loaded.catalog,state.catalog)

    def test_cached_keys_do_not_rehash_factors_and_replacements_get_new_keys(self):
        space=fixture()
        cursor=space.initial_state()
        key=space.key(cursor)
        state_key=_state_key(cursor.state)
        with patch.object(Factor,'__hash__',side_effect=AssertionError('rehashed Factor')):
            self.assertIs(space.key(cursor),key)
            self.assertEqual(hash(space.key(cursor)),hash(key))
            self.assertIs(_state_key(cursor.state),state_key)
        changed=replace(cursor,state=replace(cursor.state,loaded={'A'}))
        self.assertNotEqual(space.key(changed),key)
        equivalent=replace(cursor,peak_host=1)
        self.assertEqual(space.key(equivalent),key)

    def test_key_hash_collisions_keep_full_equality(self):
        with patch.object(_ImmutableKey,'__hash__',lambda key:7):
            a,b=_ImmutableKey(('a',)),_ImmutableKey(('b',))
            self.assertEqual({a:1,b:2}, {a:1,b:2})
            self.assertEqual(len({a,b}),2)

    def test_memory_sampling_counts_cached_properties_and_shared_values_once(self):
        @dataclass(frozen=True)
        class Item:
            value:int=1
            @cached_property
            def payload(self): return bytes(100000)
        item=Item()
        before=retained_size([item])
        item.payload
        after=retained_size([item])
        self.assertGreater(after-before,100000)
        self.assertEqual(retained_size([item,item]),after)

    def test_scalar_bound_matches_independent_resource_point_for_three_tiers(self):
        space=fixture(backend='cupy')
        owners=(ResourceAllocation('block','raw',40,30,20),ResourceAllocation('tensor','A',10,50,60))
        maxima={(a.domain,a.key):(a.host_bytes,a.device_bytes,a.disk_bytes) for a in owners}
        actual=space._maxima_point(maxima,'bound',7,11)
        expected=resource_point(ResourceSnapshot(owners),0,'bound',workspace_host=7,workspace_device=11)
        self.assertEqual(actual,expected)

    def test_batch_facts_are_released_on_completion_and_interruption(self):
        space=fixture()
        cursor=space.initial_state()
        key=space.key(cursor)
        space.successors(cursor)
        for name in ('owners','borrowed','useful_tensors'):
            self.assertNotIn(name,cursor.__dict__)
        self.assertNotIn('live_inputs',cursor.state.__dict__)
        self.assertIs(space.key(cursor),key)
        calls=[]
        def stop():
            calls.append(1)
            if len(calls)==5: raise RuntimeError('stop')
        with self.assertRaises(RuntimeError): space.successors(cursor,stop)
        self.assertNotIn('owners',cursor.__dict__)
        self.assertNotIn('live_inputs',cursor.state.__dict__)

    def test_emitted_deltas_match_independent_full_state_diff_and_replay(self):
        ordinary=fixture((ContractionTerm(('A','B','C'),'i,i,i->'),)*8)
        for space in (ordinary,join_fixture()):
            before=space.initial_state()
            for step in space.logical_steps(before):
                edge=space.evaluate(before,step)
                delta=space.freeze_transition(before,edge)
                expected_pending=tuple((i,v) for i,v in enumerate(edge.state.state.pending)
                                       if v!=before.state.pending[i])
                expected_results=tuple((i,v) for i,v in enumerate(edge.state.state.results)
                                       if v!=before.state.results[i])
                self.assertEqual(delta.pending,expected_pending)
                self.assertEqual(delta.results,expected_results)
                self.assertEqual(space.apply_delta(before,delta).state,edge.state)


if __name__=='__main__': unittest.main()
