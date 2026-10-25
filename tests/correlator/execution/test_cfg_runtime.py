"""Persistent read queue and shared cfg ownership on the common CPU kernels."""
from contextlib import contextmanager
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch
from time import sleep
import numpy as np
from correlator.planning.test_cfg_plan import selected_template
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.execution.cfg_runtime import _LocalLoader
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.execution.preparation import PreparedSourceLoader,bind_prepared_loader
from lqcdweave.contraction.planning.api import build_cfg_plan
from lqcdweave.contraction.execution.api import execute_cfg_plan,replay_scene_storage


class NodeIdBindingTests(unittest.TestCase):
    def test_native_signed_ids_keep_python_integer_values_and_key_order(self):
        from lqcdweave.contracts.correlator.api import ArrayConstant
        from lqcdweave.contraction.execution.cfg_runtime import _bind_node_ids
        ids=np.array([-2**63,0,2**53+1,2**63-1],dtype='<i8')
        record=ArrayConstant.from_array(ids)
        keys=('minimum','zero','large','maximum')
        actual=_bind_node_ids(keys,record)
        self.assertEqual(list(actual.items()),list(zip(keys,map(int,ids))))
        self.assertTrue(all(type(value) is int for value in actual.values()))
        np.testing.assert_array_equal(record.array(),ids)

    def test_debug_and_external_records_keep_explicit_integer_conversion(self):
        from types import SimpleNamespace
        from lqcdweave.contracts.correlator.api import ArrayConstant
        from lqcdweave.contraction.execution.cfg_runtime import _bind_node_ids
        from lqcdweave.contraction.execution.diagnostics import execution_debug
        class InspectableArray(np.ndarray):
            def tolist(self):
                raise AssertionError('bulk conversion bypassed explicit debug conversion')
        record=ArrayConstant.from_array(np.array([3,7],dtype='<i8'))
        values=record.array().view(InspectableArray)
        with patch.object(ArrayConstant,'array',return_value=values),execution_debug(True):
            self.assertEqual(_bind_node_ids(('a','b'),record),{'a':3,'b':7})
        external=SimpleNamespace(array=lambda:np.array(['3','7'],dtype=object))
        self.assertEqual(_bind_node_ids(('a','b'),external),{'a':3,'b':7})
        external=SimpleNamespace(array=lambda:np.array(['3','invalid'],dtype=object))
        with self.assertRaises(ValueError):
            _bind_node_ids(('a','b'),external)


class SafeReader:
    background_read_safe=True
    def __init__(self,reader,cfg,scale,records):
        self.reader,self.cfg,self.scale,self.records=reader,cfg,scale,records
    def __getattr__(self,key): return getattr(self.reader,key)
    def describe_block(self,*args,**kwargs):
        metadata=self.reader.describe_block(*args,**kwargs)
        return replace(metadata,data_identity=metadata.data_identity+':'+self.cfg)
    def block(self,*args,**kwargs):
        self.records.append((self.cfg,threading.current_thread().name))
        return self.reader.block(*args,**kwargs)*self.scale
    def read_slab_in_background(self,*args,**kwargs):
        if isinstance(args[0],tuple):
            return np.stack([self.block(t,*args[1:],**kwargs) for t in args[0]])
        return np.stack([self.block(*args[:2],t,*args[3:],**kwargs) for t in args[2]])


class CfgReplayTests(unittest.TestCase):
    def setup_plan(self,*,prefetch=True):
        expanded,template,elemental,propagator=selected_template()
        points,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1}])
        requests={k:replace(v,prefetch_host_bytes=template.blocks[k].host_bytes+
                           template.blocks[k].staging_host_bytes if prefetch else None)
                  for k,v in expanded.inputs.read_requests.items()}
        loader=expanded.inputs.loader
        plan=build_cfg_plan(template,expanded.output_recipe,points.points,requests,
            loader.preparations,loader.specs)
        return plan,expanded,template,elemental,propagator

    def run_plan(self,*,capacity=None,fail=False,prefetch=True,debug=False):
        plan,expanded,template,elemental,propagator=self.setup_plan(prefetch=prefetch)
        records=[]; claims=[]; closed=[]; outputs={}
        @contextmanager
        def factory(cfg):
            readers={r.reader_role:SafeReader(propagator if r.family=='propagator' else elemental,
                        cfg,1 if cfg=='a' else 2,records) for r in plan.requests.values()}
            for r in plan.requests.values():
                for alias in r.aliases:
                    readers[alias.reader_role]=SafeReader(
                        propagator if alias.family=='propagator' else elemental,cfg,
                        1 if cfg=='a' else 2,records)
            loader=restore_input_loader(plan.requests,readers,plan.preparations,plan.specs,configuration=cfg)
            try: yield loader
            finally: closed.append(cfg)
        def configurations():
            for cfg in ('a','b'):
                claims.append((cfg,threading.current_thread().name))
                yield cfg
        def consume(cfg,index,values):
            if fail: raise RuntimeError('consumer failed')
            if index is not None: outputs[cfg,index]=values
        with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor',
                   wraps=__import__('concurrent.futures').futures.ThreadPoolExecutor) as pool:
            result=execute_cfg_plan(plan,configurations(),factory,consume,
                                    prefetch_host_bytes=capacity,debug=debug)
        return result,outputs,records,claims,closed,pool.call_count

    def test_multiple_cfgs_share_one_queue_but_no_arrays(self):
        result,outputs,records,claims,closed,pools=self.run_plan()
        self.assertEqual(result['completed'],('a','b'))
        self.assertEqual(pools,1)
        self.assertEqual([c for c,_ in claims],['a','b'])
        self.assertTrue(all(t=='MainThread' for _,t in claims))
        self.assertEqual(closed,['a','b'])
        self.assertTrue(any(t.startswith('contraction-io') for _,t in records))
        for index in (0,1):
            # Scaling both propagators and both vertices multiplies connected
            # values by 16; the two separate loops each scale by 4.
            for term,(a,b) in enumerate(zip(outputs['a',index],outputs['b',index],strict=True)):
                np.testing.assert_allclose(b,a*(16 if term==0 else 4),rtol=3e-12,atol=3e-12)

    def test_shared_resident_reuse_needs_no_cache_restore_and_matches_debug(self):
        from lqcdweave.contraction.execution import bound_program
        from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
        plan,*_=self.setup_plan()
        self.assertTrue(any(np.any(instance.reused.array()) for instance in plan.instances))
        with (patch.object(bound_program,'_reuse',side_effect=AssertionError('redundant reuse check')),
              patch.object(_BlockExecution,'has_retained',side_effect=AssertionError('cache restore'))):
            result,actual,*_=self.run_plan()
        checked_result,checked,*_=self.run_plan(debug=True)
        self.assertEqual(result['completed'],('a','b'))
        self.assertEqual(result['completed'],checked_result['completed'])
        self.assertEqual(actual.keys(),checked.keys())
        for key in actual:
            for value,reference in zip(actual[key],checked[key],strict=True):
                np.testing.assert_array_equal(value,reference)

    def test_report_counts_materialization_calls_and_host_wall_time(self):
        local_calls=[]
        class DelayedDownload:
            backend=np
            statistics={}
            def __init__(self): self.download_calls=0
            def bind_loader(self,loader,preparations,specs):
                return bind_prepared_loader(loader,preparations,specs,backend=self.backend)
            def download(self,values):
                self.download_calls+=1
                sleep(0.002)
                return tuple(values)
        device=DelayedDownload()
        original_local=_LocalLoader.materialize
        original_prepared=PreparedSourceLoader.materialize
        def count_local(loader,key,source,blocks):
            local_calls.append(key)
            return original_local(loader,key,source,blocks)
        def delayed_prepared(loader,key,source,blocks):
            sleep(0.002)
            return original_prepared(loader,key,source,blocks)
        with (patch.object(_LocalLoader,'materialize',count_local),
              patch.object(PreparedSourceLoader,'materialize',delayed_prepared),
              patch('lqcdweave.contraction.execution.cfg_runtime.active_device_execution',
                    return_value=device)):
            result,*_=self.run_plan(capacity=0,prefetch=False)
        self.assertEqual(result['completed'],('a','b'))
        self.assertEqual(result['source_materialization_calls'],len(local_calls))
        self.assertGreater(result['source_materialization_calls'],0)
        self.assertGreaterEqual(result['source_materialization_call_seconds'],
                                len(local_calls)*0.0015)
        self.assertEqual(device.download_calls,4)
        self.assertGreaterEqual(result['output_download_seconds'],
                                device.download_calls*0.0015)

    def test_zero_capacity_keeps_lazy_claims_and_synchronous_reads(self):
        result,outputs,records,claims,closed,pools=self.run_plan(capacity=0)
        self.assertEqual(pools,0)
        self.assertTrue(all(t=='MainThread' for _,t in records))
        self.assertEqual(result['prefetch_capacity_bytes'],0)
        self.assertEqual(result['completed'],('a','b'))

    def test_consumer_failure_joins_queue_and_closes_claimed_sources(self):
        with self.assertRaisesRegex(RuntimeError,'consumer failed'):
            self.run_plan(fail=True)

    def test_worker_memory_rejection_precedes_claim(self):
        plan,*_=self.setup_plan()
        def unavailable():
            raise AssertionError('cfg claimed')
            yield
        with self.assertRaisesRegex(MemoryError,'saved cfg envelope'):
            execute_cfg_plan(plan,unavailable(),None,None,host_memory_limit=plan.peak_host_bytes-1)

    def test_cross_cfg_read_is_running_during_current_cfg_kernel(self):
        from lqcdweave.contraction.execution.operations import contract_dense
        future_read=threading.Event(); release=threading.Event(); observed=[]
        original=SafeReader.block
        def blocked_read(reader,*args,**kwargs):
            if reader.cfg=='b' and threading.current_thread().name.startswith('contraction-io'):
                future_read.set()
                if not release.wait(10): raise RuntimeError('kernel never released next cfg read')
            return original(reader,*args,**kwargs)
        def kernel(*args,**kwargs):
            observed.append(future_read.wait(10))
            release.set()
            return contract_dense(*args,**kwargs)
        with (patch.object(SafeReader,'block',blocked_read),
              patch('lqcdweave.contraction.execution.scene_runtime.contract_dense',kernel)):
            self.run_plan()
        self.assertTrue(observed[0])

    def test_dispatch_consumer_keeps_shared_queue_after_execution(self):
        from pathlib import Path
        import tempfile
        from lqcdweave.runtime.dispatch import Dispatch
        from runtime.test_dispatch import SingleRankComm
        with tempfile.TemporaryDirectory() as directory:
            tasks=Path(directory)/'tasks.txt'; tasks.write_text('a\nb\nc\n')
            first=Dispatch(str(tasks),'same',comm=SingleRankComm())
            second=Dispatch(str(tasks),'same',comm=SingleRankComm())
            # Both consumers claim from the existing shared queue. X accepts
            # either one as an iterator; no extra scheduler is involved.
            self.assertEqual(next(iter(first)),'a')
            self.assertEqual(next(iter(second)),'b')
            self.assertEqual(tuple(first),('c',))
            self.assertTrue(Path(first.tmp).exists())
            self.assertEqual(tasks.read_text(),'a\nb\nc\n')


if __name__=='__main__': unittest.main()
