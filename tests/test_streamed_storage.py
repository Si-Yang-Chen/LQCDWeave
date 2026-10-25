import unittest
from dataclasses import replace
from test_scene_storage_batches import batch,Loader
from lqcdweave.contraction.planning.api import (
    join_storage_batches, retain_storage_tensors, iter_join_storage_batches, iter_retain_storage_tensors,
)
from lqcdweave.contraction.execution.api import replay_storage_batches


class StreamedStorageTests(unittest.TestCase):
    def test_stream_matches_complete_schedule_without_eager_planning(self):
        originals=(batch('a'),batch('b'),batch('a'))
        demands=({'a'},{'b'},{'a'})
        expected=retain_storage_tensors(join_storage_batches(originals))
        actual=tuple(iter_retain_storage_tensors(iter_join_storage_batches(iter(originals),demands)))
        for a,b in zip(expected,actual):
            self.assertEqual(a,b)
        loader=Loader();values=[]
        def source():
            for index,p in enumerate(originals):
                self.assertEqual(len(values),index,'must execute before planning next batch')
                yield p
        plans=iter_retain_storage_tensors(iter_join_storage_batches(source(),demands))
        replay_storage_batches(plans,loader,lambda i,r:values.append(float(r[0])),prejoined=True,streaming=True)
        self.assertEqual(values,[6.,6.,6.])
        self.assertEqual(loader.reads,['a','b'])
        self.assertEqual(set(loader.closes),{'a','b'})

    def test_later_memory_failure_closes_inputs_without_reading_failed_batch(self):
        loader=Loader();values=[]
        plans=(batch('a'),replace(batch('b'),memory_limit=1),batch('a'))
        stream=iter_retain_storage_tensors(iter_join_storage_batches(plans,({'a'},{'b'},{'a'})))
        with self.assertRaises(MemoryError):
            replay_storage_batches(stream,loader,lambda i,r:values.append(float(r[0])),prejoined=True,streaming=True)
        self.assertEqual(values,[6.]);self.assertEqual(loader.reads,['a'])
        self.assertEqual(loader.closes,['a'])

    def test_demand_mismatch_rejected_before_read(self):
        loader=Loader()
        with self.assertRaises(ValueError):
            replay_storage_batches(iter_join_storage_batches([batch('a')],[{'b'}]),loader,
                                   lambda *args:None,prejoined=True,streaming=True)
        self.assertFalse(loader.reads)

    def test_consumer_exception_cleans_up_retained_inputs(self):
        loader=Loader()
        stream=iter_retain_storage_tensors(iter_join_storage_batches([batch('a'),batch('a')],[{'a'},{'a'}]))
        def fail(*args): raise RuntimeError('write failed')
        with self.assertRaises(RuntimeError):
            replay_storage_batches(stream,loader,fail,prejoined=True,streaming=True)
        self.assertEqual(loader.reads,loader.closes)

    def test_runtime_reserve_refresh_evicts_cache_before_next_execution(self):
        originals=(batch('a',256),batch('b',256),batch('a',256))
        joined=iter_join_storage_batches(originals,({'a'},{'b'},{'a'}))
        refreshed=[];executed=[];loader=Loader()
        def prepare(p):
            self.assertEqual(len(executed),len(refreshed))
            if len(refreshed)==1:
                p=p.with_reserves(reserved_host_bytes=256-p.peak_host_bytes)
            refreshed.append(p)
            return p
        from contextlib import contextmanager
        @contextmanager
        def context(p):
            if len(executed)==1:
                self.assertIn('a',p.retained_releases)
                self.assertFalse(p.retained_before)
                self.assertLessEqual(p.peak_host_bytes,256)
            yield
        replay_storage_batches(iter_retain_storage_tensors(joined,prepare=prepare),loader,
            lambda i,r:executed.append(float(r[0])),prejoined=True,streaming=True,batch_context=context)
        self.assertEqual(executed,[6.,6.,6.]);self.assertEqual(loader.reads,['a','b'])


if __name__=='__main__': unittest.main()
