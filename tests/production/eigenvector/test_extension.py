"""Preserved-prefix extension against an independent site-index Laplacian."""
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from tests.fixtures.laplace_reference import GaugeHandle, random_gauge, dense_laplacian


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.sizes = (2, 2, 2, 2)
        self.gauge = random_gauge(self.sizes)
        self.matrices = [dense_laplacian(self.gauge[t, ..., :3, :, :].transpose(3, 0, 1, 2, 4, 5))
                         for t in range(2)]
        self.eigenpairs = [np.linalg.eigh(a) for a in self.matrices]
        self.prefix = np.stack([v[:, :3].T.reshape(3, 2, 2, 2, 3)
                               for _, v in self.eigenpairs]) * np.exp(.37j)

    def make(self, data=None, total=7, **options):
        data = self.prefix if data is None else data
        reader = SimpleNamespace(load=lambda key: data, basis_id='saved-basis', mode_labels=(0, 1, 2))
        gen = LaplaceEigenvectorGenerator(self.sizes, GaugeHandle(self.gauge), total, 1e-11,
            backend='numpy', existing_basis=reader, **options).load('cfg')
        return gen

    def test_prefix_is_unchanged_and_full_spectrum_matches_dense(self):
        before = self.prefix.copy()
        gen = self.make()
        for t, vectors, values in gen.iter_slices():
            np.testing.assert_array_equal(vectors[:3], before[t])
            np.testing.assert_allclose(values, self.eigenpairs[t][0][:7], atol=2e-12)
            columns = vectors.reshape(7, -1).T
            np.testing.assert_allclose(columns.conj().T @ columns, np.eye(7), atol=2e-12)
            np.testing.assert_allclose(self.matrices[t] @ columns, columns * values, atol=2e-10)
            record = gen.describe()['extension']
            self.assertEqual((record['source_basis_id'], record['prefix_count'], record['added_count']),
                             ('saved-basis', 3, 4))
            self.assertEqual(record['physical_time'], t)
        np.testing.assert_array_equal(self.prefix, before)
        gen.load('next')
        self.assertIsNone(gen.describe()['extension'])

    def test_rejects_wrong_norm_residual_order_and_missing_low_mode(self):
        bad_norm = self.prefix.copy(); bad_norm[:, 0] *= 2
        bad_residual = self.prefix.copy()
        bad_residual[:, :2] = np.einsum('ab,tbzyxc->tazyxc',
            np.array([[1, 1], [-1, 1]]) / np.sqrt(2), bad_residual[:, :2])
        bad_order = self.prefix[:, ::-1].copy()
        hole = np.stack([v[:, [0, 2, 3]].T.reshape(3, 2, 2, 2, 3) for _, v in self.eigenpairs])
        for data, error in ((bad_norm, 'orthonormal'), (bad_residual, 'residual'),
                            (bad_order, 'spectral order'), (hole, 'low-spectrum prefix')):
            with self.subTest(error=error):
                gen = self.make(data)
                with self.assertRaisesRegex(ValueError, error): gen.calc(0)
                self.assertIsNone(gen.describe()['extension'])
        gen = self.make()
        gen.stout_smear(1, .1)
        with self.assertRaisesRegex(ValueError, 'residual'): gen.calc(0)

    def test_no_growth_still_checks_the_low_spectrum(self):
        gen = self.make(total=3)
        vectors, values = gen.calc(0)
        np.testing.assert_array_equal(vectors, self.prefix[0])
        np.testing.assert_allclose(values, self.eigenpairs[0][0][:3], atol=1e-12)
        high = np.stack([v[:, 1:4].T.reshape(3, 2, 2, 2, 3) for _, v in self.eigenpairs])
        with self.assertRaisesRegex(ValueError, 'low-spectrum prefix'):
            self.make(high, total=3).calc(0)

    def test_compact_borrowed_reader_uses_physical_times(self):
        gen = self.make(selection=TimeSelectionRecord(2, (1, 0)))
        calls = []
        def read(key, t, start, stop):
            calls.append((key, t, start, stop))
            return self.prefix[t, start:stop]
        gen.existing_basis = SimpleNamespace(read_modes=read, Ne=3, basis_id='streamed',
            time_selection=TimeSelectionRecord(2, (1, 0)))
        gen.load('cfg')
        rows = list(gen.iter_slices())
        self.assertEqual(calls, [('cfg', 1, 0, 3), ('cfg', 0, 0, 3)])
        for t, vectors, _ in rows:
            np.testing.assert_array_equal(vectors[:3], self.prefix[t])
        fallback = self.make(self.prefix[::-1])
        fallback.existing_basis.time_selection = TimeSelectionRecord(2, (1, 0))
        np.testing.assert_array_equal(fallback.calc(1)[0][:3], self.prefix[1])

    def test_degenerate_prefix_and_complex_phase_remain_exact(self):
        self.gauge = np.broadcast_to(np.eye(3), (2, 2, 2, 2, 4, 3, 3)).astype(complex).copy()
        a = dense_laplacian(self.gauge[0, ..., :3, :, :].transpose(3, 0, 1, 2, 4, 5))
        values, columns = np.linalg.eigh(a)
        rotation, _ = np.linalg.qr(np.array([[1, 1j, 2], [2j, 1, 1], [1, 2, 3j]], complex))
        old = (columns[:, :3] @ rotation).T.reshape(3, 2, 2, 2, 3)
        self.prefix = np.stack([old, old])
        vectors, actual = self.make(total=5).calc(0)
        np.testing.assert_array_equal(vectors[:3], old)
        np.testing.assert_allclose(actual, values[:5], atol=1e-11)

    def test_custom_backend_requires_and_validates_extension_capability(self):
        from fixtures.laplace_backend import DenseLaplaceBackend
        backend = DenseLaplaceBackend()
        gen = self.make()
        gen.backend = backend
        gen.load('cfg')
        with self.assertRaisesRegex(NotImplementedError, 'solve_complement'): gen.calc(0)
        seen = []
        def solve(links, sizes, t, count, tol, prefix):
            seen.append((t, count, prefix.copy()))
            vals, vecs = self.eigenpairs[t]
            start = len(prefix)
            return vecs[:, start:start+count].T.reshape(count, 2, 2, 2, 3), vals[start:start+count]
        backend.solve_complement = solve
        vectors, values = gen.calc(0)
        self.assertEqual(seen[0][:2], (0, 4))
        np.testing.assert_array_equal(vectors[:3], self.prefix[0])
        backend.solve_complement = lambda *args: (np.zeros((4, 2, 2, 2, 3), complex), np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'independent'): gen.calc(0)
        self.assertIsNone(gen.describe()['extension'])

    def test_registered_compact_prefix_roundtrip_retains_source_identity(self):
        from lqcdweave.contracts.dataset import DatasetSpec
        from lqcdweave.planning.laplace import laplace_basis_document
        from lqcdweave.storage.dataset import DatasetWriter, DatasetReader
        from lqcdweave.adapters.production_dataset import DatasetFieldHandle
        cfg = dict(ensemble='extension-test', configuration='cfg', lattice=list(self.sizes))
        selection = TimeSelectionRecord(2, (1, 0))
        doc = laplace_basis_document(cfg, selection.times, 'actual-gauge', 3, 1e-11,
                                    selection=selection)
        doc['conventions']['phase'] = 'saved-complex-phase'
        with TemporaryDirectory() as directory:
            old_path, new_path = Path(directory)/'old', Path(directory)/'new'
            with DatasetWriter(old_path, DatasetSpec(doc), provenance={}) as writer:
                for slot, time in enumerate(selection.times): writer.write(slot, self.prefix[time])
                writer.publish()
            with DatasetReader(old_path, cfg='cfg') as reader:
                handle = DatasetFieldHandle(reader)
                gen = LaplaceEigenvectorGenerator(self.sizes, GaugeHandle(self.gauge), 7, 1e-11,
                    backend='numpy', existing_basis=handle, selection=selection).load('cfg')
                rows, records = [], []
                for time, vectors, values in gen.iter_slices():
                    rows.append(vectors)
                    records.append(gen.describe()['extension'])
                    np.testing.assert_array_equal(vectors[:3], self.prefix[time])
                    np.testing.assert_allclose(values, self.eigenpairs[time][0][:7], atol=2e-12)
                self.assertTrue(all(r['source_basis_id'] == handle.basis_id for r in records))
                out = laplace_basis_document(cfg, selection.times, 'actual-gauge', 7, 1e-11,
                                            selection=selection)
                out['definition']['preserved_prefix'] = dict(data_id=handle.basis_id, count=3)
                out['parents'].append(handle.basis_id)
                out['conventions']['phase'] = 'preserved-prefix/new-first-component-real'
                with DatasetWriter(new_path, DatasetSpec(out), provenance={'extension': records}) as writer:
                    for slot, row in enumerate(rows): writer.write(slot, row)
                    writer.publish()
                with DatasetReader(new_path, cfg='cfg') as saved:
                    self.assertNotEqual(saved.data_id, reader.data_id)
                    for time in selection.times:
                        np.testing.assert_array_equal(saved.read(time=selection.slot(time))[:3], self.prefix[time])


if __name__ == '__main__': unittest.main()
