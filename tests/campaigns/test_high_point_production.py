"""Direct H-source sharing and saved gamma5 orientation versus a full inverse."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
import numpy as np

from production.perambulator import test_dilution as reference
from production.perambulator.test_dilution import Handle, CountingBackend
from lqcdweave.contracts.dilution import DilutionSource
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from lqcdweave.production.perambulator.api import (
    HighmodeBasis, ModePartition, PerambulatorGenerator, PointBasis, VectorBasis,
    PropagatorEndpoint, PropagatorReader,
)
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from tools.campaigns.high_point_production import save_source, specifications


class HighPointProductionTests(unittest.TestCase):
    def test_shared_H_sources_and_both_saved_orientations_match_independent_inverse(self):
        f=reference.DilutionTests();f.setUp();rng=np.random.default_rng(481)
        gamma5=np.diag([1,1,-1,-1])
        gamma_full=np.kron(np.eye(6),np.kron(gamma5,np.eye(3)))
        a=rng.normal(size=(72,72))+1j*rng.normal(size=(72,72))
        f.matrix=gamma_full@(10*np.eye(72)+.01*(a+a.conj().T));f.inverse=np.linalg.inv(f.matrix)
        parent=np.stack([np.linalg.qr(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))[0].T
                         for _ in range(3)]).reshape(3,6,1,1,2,3)
        ph=Handle(parent,Ne=6,basis_id='actual-parent',mode_labels=tuple(range(6)),
                  basis_orthonormal=True,basis_orthogonal=True)
        noise=NoisevectorGenerator(ph,((3,),(2,)),mode_offset=2).load('cfg')
        f.v=parent[:,:2];f.d=noise.calc(seed=348);f.data.update(V=f.v,D=f.d)
        ds=DilutionSource.from_description(noise.describe(),data_id='actual-D')
        v=Handle(f.v,Ne=2,basis_id='actual-low',mode_labels=(0,1),
                 endpoint=PropagatorEndpoint('V','actual-low',(0,1)))
        d=Handle(f.d,Ne=2,basis_id=ds.data_id,mode_labels=(0,1),dilution_source=ds)
        points=Handle(f.points,Np=2,basis_id='actual-points',point_labels=(0,1))
        high=HighmodeBasis(ph,ModePartition(ph.basis_id,(0,1),((2,3,4),)))
        ph.load=lambda key: (_ for _ in ()).throw(AssertionError('eager high parent'))
        gen=PerambulatorGenerator(f.lattice,Handle(f.matrix,gauge_id='actual-gauge'),0,1e-11,200,
            backend=CountingBackend(),MRHS=False,rhs_batch_size=1,stream_eigenvectors=True,
            sink_ne_block_size=1,sink_time_block_size=2,
            gamma5=gamma5,gamma5_hermitian=True)
        source_basis=PointBasis(points,count=2,highmode_basis=high)
        v_basis,d_basis=VectorBasis(v,count=2),VectorBasis(d,count=2)
        gen.load('cfg')
        specs=specifications(gen,f.configuration,source_basis,[v_basis,d_basis],{'mass':0})
        self.assertEqual(set(specs),{'PSV','PSD'})
        self.assertTrue(all(spec.shape==(9,4,4,2,3,2) for spec in specs.values()))

        def oracle(product,h,t):
            point=f.embedding('P',f.points,h);u=f.embedding('V',parent[:,:5],h)
            sink=(np.eye(72)-u@u.conj().T)@point
            source=f.embedding(product[2],f.data[product[2]],t)
            value=sink.conj().T@f.inverse@source
            return value.reshape(4,6,4,2).transpose(0,2,1,3).reshape(4,4,2,3,2)

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with ExitStack() as stack:
                writers={}
                for product,spec in specs.items():
                    writer=stack.enter_context(DatasetWriter(root/product,spec,provenance={}))
                    writers[product]=writer
                for h_time in gen.source_times:
                    results=gen.calc(h_time,src=source_basis,sink=[v_basis,d_basis])
                    save_source(writers,specs,h_time,results)
                    del results
                for writer in writers.values():
                    writer.publish()
            with ExitStack() as stack:
                saved={product:stack.enter_context(DatasetReader(root/product,cfg='cfg'))
                       for product in specs}
                consumer=PropagatorReader(tuple(saved.values()))
                for product,spec in specs.items():
                    definition=spec.document['definition']
                    sink=PropagatorEndpoint.from_document(definition['sink_endpoint'])
                    source=PropagatorEndpoint.from_document(definition['source_endpoint'])
                    self.assertEqual(sink.metadata['high_mode_projection'],high.partition.document)
                    if product=='PSD':
                        self.assertEqual(source.metadata['dilution_source'],ds.document)
                    self.assertEqual(len(consumer.coverage(product,sink=sink,source=source)),9)
                    for h_time in range(3):
                        for vector_time in range(3):
                            actual=consumer.read(product,h_time,vector_time,sink=sink,source=source)
                            np.testing.assert_allclose(actual.values,oracle(product,h_time,vector_time),atol=2e-15)
            self.assertEqual(gen.backend.rhs_count,3*2*12)


if __name__=='__main__':unittest.main()
