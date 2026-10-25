import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.adapters.scene_storage import NpyBlockBinding, NpyPackedBinding, NpySceneBlockLoader
from lqcdweave.contraction import (StorageBlock, SourcePreparation, SourceSpec, TensorSpec,
    ContractionTerm, ContractionProblem, solve_storage, replay_scene_storage)


class PackedStorageTests(unittest.TestCase):
    def binding(self, path, array):
        selections = ((0,slice(None),slice(3)), (3,slice(None,None,-1),slice(1,5,2)))
        return NpyPackedBinding(tuple(NpyBlockBinding(path,key,array[key].shape,str(array.dtype),
            'whole_file',array.shape) for key in selections))

    def test_shared_file_one_read_correct_views_and_last_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'elemental.npy'
            array=np.arange(4*5*6,dtype=np.float64).reshape(4,5,6)
            np.save(path,array)
            binding=self.binding(path,array)
            size=binding.shape[0]*array.itemsize
            block=StorageBlock('verified-file','explicit disjoint components',array.nbytes,
                               host_bytes=size,staging_host_bytes=array.nbytes)
            specs={str(i):TensorSpec(s.shape,'float64') for i,s in enumerate(binding.segments)}
            problem=ContractionProblem(specs,tuple(ContractionTerm((k,), 'ab->') for k in specs),
                {k:SourceSpec(k,int(np.prod(s.shape))*8) for k,s in specs.items()})
            preps={k:SourcePreparation(('raw',)) for k in specs}
            search=solve_storage(problem,{'raw':block},preps,memory_limit=10000,
                host_memory_limit=10000,deadline_seconds=2)
            self.assertIsNotNone(search.plan)
            mats={k:(lambda metadata,blocks,i=int(k):binding.segment_view(blocks['raw'],i)) for k in specs}
            loader=NpySceneBlockLoader({'raw':binding},mats)
            with patch('lqcdweave.adapters.scene_storage.np.load',wraps=np.load) as read:
                report=replay_scene_storage(search.plan,loader,measure=True)
            self.assertEqual(read.call_count,1)
            np.testing.assert_array_equal(report.values,[array[s.selection].sum() for s in binding.segments])
            self.assertEqual(len(report.io_events),1)
            self.assertEqual(report.io_events[0].bytes,array.nbytes)
            events=search.plan.events
            self.assertEqual([e.kind for e in events],['read','release'])
            loads=[i for i,a in enumerate(search.plan.state.actions) if a.kind=='load']
            self.assertGreaterEqual(events[-1].action_index, max(loads))
            self.assertLessEqual(events[-1].action_index, len(search.plan.state.actions))

    def test_only_packed_payload_transfers_once_and_staging_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'elemental.npy'
            array=np.arange(120,dtype=np.float64).reshape(4,5,6)
            np.save(path,array)
            binding=self.binding(path,array)
            size=binding.shape[0]*8
            backend=SimpleNamespace(__name__='cupy',asarray=Mock(side_effect=lambda a:a.copy()),
                cuda=SimpleNamespace(get_current_stream=lambda:Mock()))
            loader=NpySceneBlockLoader({'raw':binding},{},backend=backend)
            block=StorageBlock('file','selection',array.nbytes,device_bytes=size,
                transfer_bytes=size,staging_host_bytes=array.nbytes+size)
            with self.assertRaisesRegex(ValueError,'staging'):
                loader.read_block('raw',dataclasses.replace(block,staging_host_bytes=size))
            backend.asarray.assert_not_called()
            loader.set_measurement(True)
            packed=loader.read_block('raw',block)
            backend.asarray.assert_called_once()
            self.assertEqual(packed.nbytes,size)
            for i,segment in enumerate(binding.segments):
                view=binding.segment_view(packed,i)
                self.assertTrue(np.shares_memory(view,packed))
                np.testing.assert_array_equal(view,array[segment.selection])
            self.assertEqual([e['bytes'] for e in loader.drain_io_events()],[array.nbytes,size])

    def test_inconsistent_segment_metadata_rejected(self):
        array=np.zeros((4,5,6))
        binding=self.binding(Path('a.npy'),array)
        a,b=binding.segments
        for bad in (dataclasses.replace(b,path=Path('b.npy')),
                    dataclasses.replace(b,shape=(99,)),
                    dataclasses.replace(b,selection=(9,slice(None),slice(None)))):
            with self.assertRaises(ValueError):
                NpyPackedBinding((a,bad))
        with self.assertRaises(ValueError):
            NpyPackedBinding(())


if __name__=='__main__': unittest.main()
