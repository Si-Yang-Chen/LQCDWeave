"""Archival acceptance uses independent inverse matrices and real Dataset IO."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.propagator import PropagatorPairReader
from lqcdweave.production.perambulator import PropagatorReader
from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.archive_perambulator_triangle import (
    cleanup, file_record, reserve_file, run_configuration,
)


class ArchiveTriangleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(source_root=str((self.root/'old').resolve()),
            output_root=str((self.root/'archive').resolve()), migration_id='test-archive',
            ensemble='independent-inverse', gamma5_diagonal=[1,1,-1,-1],
            spin_basis='test-spin', rtol=1e-10, atol=1e-12, configurations={'10000': {}})
        self.values = {}
        rng = np.random.default_rng(791)
        for product, n in [('VSV',2), ('PSP',3)]:
            size = 4*4*n
            random = rng.normal(size=(size,size)) + 1j*rng.normal(size=(size,size))
            hermitian = random + random.conj().T + 4*size*np.eye(size)
            gamma = np.diag(np.tile(np.repeat([1,1,-1,-1], n), 4))
            # D = gamma5 H, H Hermitian => S satisfies gamma5 hermiticity.
            full = np.linalg.inv(gamma @ hermitian)
            blocks = full.reshape(4,4,n,4,4,n).transpose(0,3,1,4,2,5)
            if product == 'PSP':
                blocks = blocks.reshape(4,4,4,4,1,3,1,3)
            self.values[product] = blocks
            directory = Path(self.config['source_root'])/product
            directory.mkdir(parents=True)
            sources = []
            for s in range(4):
                for rank in range(2):
                    path = directory/f'10000.t{s:03d}.rank{rank:04d}.npy'
                    np.save(path, blocks[rank*2:(rank+1)*2,s])
                    sources.append(file_record(path))
            manifest = dict(global_lattice=[1,1,1,4], local_lattice=[1,1,1,2],
                grid_size=[1,1,1,2], source_times=list(range(4)),
                time_convention='absolute-global-source-and-sink', Ne=2,
                np_snk=1, np_src=1, product=product)
            self.config['configurations']['10000'][product] = dict(manifest=manifest,
                metadata=dict(inputs=dict(gauge={'fixture':'actual-gauge'},
                                          vectors={'fixture':'actual-vectors'})), sources=sources)
        self.psv = Path(self.config['source_root'])/'PSV'/'keep.npy'
        self.psv.parent.mkdir()
        np.save(self.psv, np.array([71.]))
        self.reserve = patch('tools.campaigns.archive_perambulator_triangle.reserve_file')
        self.reserve.start()
        self.addCleanup(self.reserve.stop)

    def test_complete_readback_then_delete_and_native_correlator_reader(self):
        receipt = run_configuration(self.config, '10000', delete=True)
        self.assertEqual(receipt['status'], 'deleted')
        self.assertTrue(self.psv.exists())
        for product in ('VSV', 'PSP'):
            entry = self.config['configurations']['10000'][product]
            self.assertTrue(all(not Path(r['path']).exists() for r in entry['sources']))
            output = Path(self.config['output_root'])/product
            with DatasetReader(output, cfg='10000') as dataset:
                self.assertEqual(dataset.spec.shape[0], 10)
                reader = PropagatorPairReader(PropagatorReader([dataset]))
                for t in range(4):
                    for s in range(4):
                        self.assertEqual(reader.describe_block(t,s,product.lower()).shape,
                                         self.values[product][t,s].shape)
                        np.testing.assert_allclose(reader.block(t,s,product.lower()),
                                                   self.values[product][t,s], atol=1e-14)
            self.assertEqual(receipt['products'][product]['exact_pairs'],10)
            self.assertEqual(receipt['products'][product]['recovered_pairs'],6)
        # Re-entry after deletion reads the validated archive and receipt only.
        self.assertEqual(run_configuration(self.config, '10000', delete=True)['status'], 'deleted')
        with DatasetReader(Path(self.config['output_root'])/'VSV',cfg='10000') as vsv, \
             DatasetReader(Path(self.config['output_root'])/'PSP',cfg='10000') as psp:
            # Different product/solver manifests still share one physical action.
            combined=PropagatorPairReader(PropagatorReader([vsv,psp]))
            np.testing.assert_allclose(combined.block(0,3,'vsv'),self.values['VSV'][0,3])
            np.testing.assert_allclose(combined.block(0,3,'psp'),self.values['PSP'][0,3])

    def test_bad_reverse_pair_preserves_both_original_products(self):
        entry = self.config['configurations']['10000']['PSP']
        # Source 1, sink 0 belongs to the omitted half, leaving the saved half intact.
        path = Path(entry['sources'][2]['path'])
        data = np.load(path)
        data[0].flat[0] += 0.01j
        np.save(path,data)
        entry['sources'][2] = file_record(path)
        with self.assertRaisesRegex(ValueError,'gamma5 recovery failed'):
            run_configuration(self.config,'10000',delete=True)
        self.assertTrue(all(Path(r['path']).exists() for e in
            self.config['configurations']['10000'].values() for r in e['sources']))
        self.assertFalse((Path(self.config['output_root'])/'receipts/10000.json').exists())

    def test_changed_archive_prevents_cleanup(self):
        receipt = run_configuration(self.config,'10000')
        payload = Path(receipt['products']['VSV']['payload'])
        array = np.load(payload,mmap_mode='r+')
        array.flat[0] += 1.
        array.flush(); array._mmap.close()
        with self.assertRaisesRegex(ValueError,'archive changed'):
            cleanup(self.config,'10000',receipt,Path(self.config['output_root'])/'receipts/10000.json')
        self.assertTrue(all(Path(r['path']).exists() for e in
            self.config['configurations']['10000'].values() for r in e['sources']))

    def test_resume_after_interrupted_cleanup(self):
        receipt = run_configuration(self.config,'10000')
        first = Path(self.config['configurations']['10000']['VSV']['sources'][0]['path'])
        first.unlink()  # Simulate interruption after the first successful unlink.
        result = run_configuration(self.config,'10000',delete=True)
        self.assertEqual(result['status'],'deleted')
        self.assertTrue(self.psv.exists())

    def test_stale_source_stat_does_not_block_verified_migration(self):
        # A login-node snapshot can have a different st_dev on the compute node.
        # Migration uses paths and numerical read-back, not historical stat data.
        for entry in self.config['configurations']['10000'].values():
            for record in entry['sources']:
                record.update(device=-1, inode=-1, mtime_ns=-1, size=-1)
        receipt = run_configuration(self.config,'10000',delete=True)
        self.assertEqual(receipt['status'],'deleted')
        self.assertTrue(self.psv.exists())

    @unittest.skipUnless(hasattr(os, 'posix_fallocate'), 'POSIX filesystem required')
    def test_posix_reservation_preserves_existing_payload(self):
        path=self.root/'reserve.bin'
        content=os.urandom(8192)
        path.write_bytes(content)
        reserve_file(path)
        self.assertEqual(path.read_bytes(),content)
        self.assertGreaterEqual(path.stat().st_blocks*512,len(content))


if __name__ == '__main__':
    unittest.main()
