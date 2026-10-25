import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import numpy as np
from test_scene_storage_batches import batch,Loader
from lqcdweave.contraction.planning.api import join_storage_batches
from lqcdweave.contraction.execution.api import replay_storage_batches
from lqcdweave.storage.contraction_plans.api import save_storage_schedule,load_storage_schedule
from lqcdweave.storage.contraction_plans.api import encode_plan_value


class CurrentDatasetLoader(Loader):
    def __init__(self):
        super().__init__();self.datasets=[]
    def read_block(self,key,block):
        self.reads.append(key);self.datasets.append(block.dataset)
        offset=10 if block.dataset.startswith('new-') else 0
        return np.arange(4,dtype=float)+offset


def legacy_plan_value(value):
    if isinstance(value,dict) and set(value)=={'type','value'}:
        kind,payload=value['type'],value['value']
        if kind=='numpy.dtype':kind='dtype'
        elif kind=='python.complex':kind='complex'
        elif kind.startswith('python.'):kind=kind.split('.',1)[1]
        elif kind.startswith('lqcdweave.'):kind=kind.rsplit('.',1)[1]
        if isinstance(payload,dict):
            payload={key:legacy_plan_value(item) for key,item in payload.items()}
            if kind=='TensorSpec' and payload.get('description') is None:
                payload.pop('description')
        elif isinstance(payload,list):
            payload=[legacy_plan_value(item) for item in payload]
        return {'type':kind,'value':payload}
    if isinstance(value,dict):return {key:legacy_plan_value(item) for key,item in value.items()}
    if isinstance(value,list):return [legacy_plan_value(item) for item in value]
    return value




class ScheduleArchiveTests(unittest.TestCase):
    def test_restore_without_search_or_join_and_rebind_realization(self):
        plans=join_storage_batches((batch('a'),batch('b'),batch('a')))
        shapes={'a':[4],'b':[4]};hardware={'backend':'numpy','memory':10000}
        expected=tuple(p.state.problem for p in plans)
        preparations={k:v for p in plans for k,v in p.preparations.items()}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json.gz'
            save_storage_schedule(path,plans,shapes=shapes,hardware=hardware,metadata={'output_order':[0,1,2]})
            blocks={k:replace(v,dataset='new-'+k) for p in plans for k,v in p.blocks.items()}
            with patch('lqcdweave.contraction.planning.api.join_storage_batches',side_effect=AssertionError('must not rejoin')):
                restored,metadata=load_storage_schedule(path,shapes=shapes,hardware=hardware,
                    expected=expected,blocks=blocks,preparations=preparations)
                loader=CurrentDatasetLoader();values=[]
                replay_storage_batches(restored,loader,lambda i,r:values.append(r[0]),prejoined=True)
            self.assertEqual(values,[46.,46.,46.]);self.assertEqual(loader.reads,['a','b'])
            self.assertEqual(loader.datasets,['new-a','new-b'])
            self.assertEqual(metadata,{'output_order':[0,1,2]})
            self.assertEqual(restored[0].state.actions,plans[0].state.actions)
            self.assertEqual(restored[0].memory,plans[0].memory)
            self.assertEqual(restored[0].blocks['a'].dataset,'new-a')
            with self.assertRaisesRegex(ValueError,'shapes'):
                load_storage_schedule(path,shapes={'a':[5],'b':[4]},hardware=hardware,
                    expected=expected,blocks=blocks,preparations=preparations)
            with self.assertRaisesRegex(ValueError,'hardware'):
                load_storage_schedule(path,shapes=shapes,hardware={'backend':'other'},
                    expected=expected,blocks=blocks,preparations=preparations)
            changed_source=replace(expected[0].sources['a'],metadata={'data_identity':'other-config'})
            changed_problem=replace(expected[0],sources={'a':changed_source})
            with self.assertRaisesRegex(ValueError,'semantic identity'):
                load_storage_schedule(path,shapes=shapes,hardware=hardware,
                    expected=(changed_problem,expected[1],expected[2]),
                    blocks=blocks,preparations=preparations)
            raw=path.read_bytes();path.write_bytes(raw[:-15])
            with self.assertRaises((ValueError,EOFError,OSError)):
                load_storage_schedule(path,shapes=shapes,hardware=hardware,
                    expected=expected,blocks=blocks,preparations=preparations)

if __name__=='__main__':unittest.main()
