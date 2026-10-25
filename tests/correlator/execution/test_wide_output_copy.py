"""Opt-in CUDA oracle for wider pointer-only output snapshot chunks."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.operations import output_kernel
from lqcdweave.contraction.execution.operations.output import bind_output_layout
from lqcdweave.contraction.execution.operations.result_views import (
    _ResultOwner, _DeferredResultView, bind_result_views,
)
from lqcdweave.contraction.execution.raw_arguments import bind_pointer_arguments


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1',
                     'explicit CUDA gate; CPU suite does not initialize CUDA')
class WideOutputCopyCuPyTests(unittest.TestCase):
    def test_current_arrays_and_deferred_sources_keep_bytes_offsets_and_snapshots(self):
        import cupy as cp

        rows=[]
        for lanes,shape,deferred in ((129,(),False),(259,(2,3),False),(259,(2,3),True)):
            for dtype in (np.float32,np.complex64):
                records=[]
                def counted(*args,**kwargs):
                    call=bind_pointer_arguments(*args,**kwargs)
                    self.assertIsNotNone(call,'real CUDA native pointer ABI must execute')
                    def submit(values,destination):
                        call(values,destination)
                        words=call._local.storage[0]
                        for position,index in call.positions:
                            self.assertEqual(words[position],output_kernel._array_pointer(values[index]))
                        records.append((len(call.positions),int(words[-2]),len(words)*8))
                    return submit

                keys=tuple('x'+str(i) for i in range(lanes))
                catalog={key:TensorSpec(shape,dtype) for key in keys}
                with patch.object(output_kernel,'bind_pointer_arguments',side_effect=counted):
                    layout=bind_output_layout(keys+keys[:2],catalog,backend=cp,validate=False)
                    first=None;first_bytes=None
                    for phase in (0,1):
                        host=(np.arange(lanes*max(1,int(np.prod(shape))),dtype=np.float32)
                              .reshape((lanes,*shape))+phase/8)
                        if np.dtype(dtype).kind=='c':host=host+0.5j*host
                        host=host.astype(dtype)
                        base=cp.asarray(host)
                        arrays={};expected={}
                        for i,key in enumerate(keys):
                            value=base[i:i+1].reshape(shape)
                            reference=host[i]
                            if not deferred and shape:
                                if i%3==1:value=value[::-1,::-1];reference=reference[::-1,::-1]
                                elif i%3==2:
                                    value=cp.broadcast_to(value[:1,:1],shape)
                                    reference=np.broadcast_to(reference[:1,:1],shape)
                            arrays[key]=value;expected[key]=np.asarray(reference)
                        if deferred:
                            owner=_ResultOwner(base)
                            views=bind_result_views(base.shape,base.dtype,
                                tuple((key,(i,slice(None),slice(None)),()) for i,key in enumerate(keys)))
                            binding={key:i+101 for i,key in enumerate(keys)}
                            arrays=_NodeView(binding,{binding[key]:_DeferredResultView(owner,view)
                                                     for key,view in views})
                        records.clear()
                        snapshot=layout.copy(arrays,cp)
                        cp.cuda.get_current_stream().synchronize()
                        expected_chunks=[min(128,lanes-start) for start in range(0,lanes,128)]
                        self.assertEqual([row[0] for row in records],expected_chunks)
                        self.assertEqual([row[1] for row in records],list(range(0,lanes,128)))
                        self.assertTrue(all(row[2]<=3072 for row in records))
                        self.assertEqual(len(snapshot.owners),1)
                        self.assertEqual(snapshot.owners[0].nbytes,host.nbytes)
                        for value,key in zip(snapshot,keys+keys[:2],strict=True):
                            self.assertEqual(cp.asnumpy(value).tobytes(),expected[key].tobytes())
                        if first is None:
                            first=snapshot
                            first_bytes=cp.asnumpy(first.owners[0]).tobytes()
                        else:self.assertEqual(cp.asnumpy(first.owners[0]).tobytes(),first_bytes)
                        rows.append(dict(lanes=lanes,shape=shape,dtype=np.dtype(dtype).name,
                                         deferred=deferred,phase=phase,submissions=len(records),bitwise_equal=True))
        import json
        print(json.dumps(dict(wide_output_copy_cuda=rows)),flush=True)


if __name__=='__main__':unittest.main()
