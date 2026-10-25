import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.scene_storage import NpyPrefixRequest, merge_npy_prefixes, NpySceneBlockLoader
from lqcdweave.contraction import (ContractionProblem, ContractionTerm, TensorSpec,
    SourceSpec, SourcePreparation, solve_storage, replay_scene_storage)


class PrefixStorageTests(unittest.TestCase):
    def test_io_observations_do_not_feed_the_compute_cost_profile(self):
        from lqcdweave.contraction import CostProfile
        from lqcdweave.contracts.measurement import ReplayReport,ActionTiming,FileIOEvent
        from lqcdweave.contraction.performance.api import update_cost_profile
        report=ReplayReport((),(
            ActionTiming('contract',1.,2.,0,'kernel',
                         model_features={'kernel_calls':1,'flops':0,'traffic_bytes':0,'dtype':'complex128'}),
            ActionTiming('load',1.,1.,1,'source'),
        ),None,None,io_events=(
            FileIOEvent(0,'a','a.npy','npy_whole_read',100,3.),
            FileIOEvent(1,'b','b.npy','npy_whole_read',200,5.),
            FileIOEvent(0,'a','a.npy','npy_slice_transfer',10,.1),))
        fitted=update_cost_profile(CostProfile(),report,verified=True,hardware_fingerprint='test-device',alpha=1)
        self.assertAlmostEqual(fitted.kernel_call_seconds_complex128, 2.)
        self.assertFalse(hasattr(fitted, 'read_call_seconds'))
        self.assertEqual(len(report.io_events), 3)
        io_only=ReplayReport((),(),None,None,io_events=report.io_events)
        with self.assertRaisesRegex(ValueError, 'no measured snapshot or contraction'):
            update_cost_profile(CostProfile(),io_only,verified=True,
                                hardware_fingerprint='test-device',alpha=1)

    def test_dagger_matches_explicit_spin_matrix_reference(self):
        from lqcdweave.contraction.propagator_views import psv_to_vsp_dagger
        rng=np.random.default_rng(19)
        a=rng.normal(size=(2,4,4,3,3,2))+1j*rng.normal(size=(2,4,4,3,3,2))
        g=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],dtype=complex)
        expected=np.empty((2,4,4,2,3,3),dtype=complex)
        for t in range(2):
            for p in range(3):
                for c in range(3):
                    for n in range(2):
                        expected[t,:,:,n,p,c]=(g@a[t,:,:,p,c,n].conj()@g).T
        np.testing.assert_allclose(psv_to_vsp_dagger(a,g),expected)
        with self.assertRaises(ValueError):
            psv_to_vsp_dagger(a[0],g)

    def test_pool_cap_restored_after_failure_and_baseline_rejected(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from lqcdweave.adapters.scene_device_budget import scene_device_budget
        pool=Mock();pool.get_limit.return_value=0;pool.total_bytes.return_value=20;pool.used_bytes.return_value=10
        backend=SimpleNamespace(__name__='cupy',get_default_memory_pool=lambda:pool,
            cuda=SimpleNamespace(get_current_stream=lambda:Mock(),runtime=SimpleNamespace(memGetInfo=lambda:(900,1000))))
        plan=SimpleNamespace(state=SimpleNamespace(problem=SimpleNamespace(backend='cupy')),
                             feasible=True,reserved_device_bytes=100,memory_limit=500)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with scene_device_budget(plan,backend=backend):
                raise RuntimeError('stop')
        self.assertEqual(pool.set_limit.call_args_list[-1].kwargs,{'size':0})
        plan.reserved_device_bytes=10
        with self.assertRaises(MemoryError):
            with scene_device_budget(plan,backend=backend):
                self.fail('must reject before execution')

    def test_joint_maximum_single_actual_read_and_last_demand(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'psv.npy'
            raw = np.arange(2*4*4*5*3*6, dtype=np.float64).reshape(2,4,4,5,3,6)
            np.save(path, raw)
            shapes = {'psv': (2,4,4,3,3,2), 'vsp': (2,4,4,2,3,4)}
            requests = {k:NpyPrefixRequest('verified-psv', 'ordered-basis', path, raw.shape, s, 'float64')
                        for k,s in shapes.items()}
            blocks, bindings, aliases = merge_npy_prefixes(requests)
            self.assertEqual(bindings['verified-psv'].shape, (2,4,4,3,3,4))
            self.assertEqual(blocks['verified-psv'].read_bytes, raw.nbytes)
            problem = ContractionProblem({k:TensorSpec(s, 'float64') for k,s in shapes.items()},
                       tuple(ContractionTerm((k,), 'abcdef->') for k in shapes),
                       {k:SourceSpec(k, int(np.prod(s))*8) for k,s in shapes.items()})
            search = solve_storage(problem, blocks,
                       {k:SourcePreparation((aliases[k],)) for k in shapes}, memory_limit=1000000)
            loader = NpySceneBlockLoader(bindings, {k:(lambda m,b,s=s: b['verified-psv'][tuple(slice(n) for n in s)])
                                                   for k,s in shapes.items()})
            with patch('lqcdweave.adapters.scene_storage.np.load', wraps=np.load) as read:
                report = replay_scene_storage(search.plan, loader, measure=True)
            self.assertEqual(read.call_count, 1)
            self.assertIsNone(read.call_args.kwargs['mmap_mode'])
            np.testing.assert_allclose(report.values, [raw[tuple(slice(n) for n in s)].sum() for s in shapes.values()])
            self.assertEqual(len(report.io_events), 1)
            self.assertEqual(report.io_events[0].bytes, raw.nbytes)
            self.assertEqual(report.io_events[0].operation, 'npy_whole_read')
            self.assertIsNone(loader._io_events)
            events = search.plan.events
            self.assertEqual([e.kind for e in events], ['read', 'release'])
            loads = [i for i,a in enumerate(search.plan.state.actions) if a.kind == 'load']
            self.assertGreaterEqual(events[-1].action_index, max(loads))
            self.assertLessEqual(events[-1].action_index, len(search.plan.state.actions))

    def test_bad_identity_or_basis_is_not_merged(self):
        a=NpyPrefixRequest('a', 'basis1', Path('same.npy'), (4,5), (2,3), 'float64')
        with self.assertRaisesRegex(ValueError, 'identical basis'):
            merge_npy_prefixes({'a':a, 'b':dataclasses.replace(a,basis_identity='basis2')})
        with self.assertRaisesRegex(ValueError, 'conflicting dataset'):
            merge_npy_prefixes({'a':a, 'b':dataclasses.replace(a,dataset='b')})

    def test_staging_cap_and_header_rejected_before_payload_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.npy'; np.save(path,np.ones((10,20)))
            blocks,bindings,_=merge_npy_prefixes({'a':NpyPrefixRequest('a','b',path,(10,20),(2,2),'float64')})
            loader=NpySceneBlockLoader(bindings,{})
            with patch('lqcdweave.adapters.scene_storage.np.load', side_effect=AssertionError):
                with self.assertRaisesRegex(ValueError,'whole-file staging'):
                    loader.read_block('a',dataclasses.replace(blocks['a'],staging_host_bytes=0))
                np.save(path,np.ones((20,20)))
                with self.assertRaisesRegex(ValueError,'file header'):
                    loader.read_block('a',blocks['a'])

    def test_runtime_reservation_participates_in_search_and_audit(self):
        from test_scene_storage_search import StorageSearchTests
        p,blocks,preps=StorageSearchTests().case('cupy')
        feasible=solve_storage(p,blocks,preps,memory_limit=8192,host_memory_limit=1000,
                    reserved_device_bytes=50,reserved_host_bytes=60,optimizer='exact')
        self.assertIsNotNone(feasible.plan)
        self.assertTrue(all(m.device_bytes<=8192 and m.reserved_device_bytes==50 for m in feasible.plan.memory))
        self.assertGreaterEqual(feasible.plan.peak_host_bytes,60)
        # 200 bytes passes the logical payload estimate but fails the final
        # rounded physical audit. Reject that candidate without aborting search.
        for optimizer in ('exact', 'dynamic_programming'):
            for cap in (100, 200):
                infeasible=solve_storage(p,blocks,preps,memory_limit=cap,
                            reserved_device_bytes=50,optimizer=optimizer)
                self.assertEqual(infeasible.status, 'INFEASIBLE' if cap == 100 else 'UNSOLVED')
                if cap == 200:
                    self.assertEqual(infeasible.stop_reason, 'physical_admission')
                self.assertIsNone(infeasible.plan)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                solve_storage(p,blocks,preps,memory_limit=200,reserved_device_bytes=value)


if __name__ == '__main__':
    unittest.main()
