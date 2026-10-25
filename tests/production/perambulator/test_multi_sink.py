"""Independent inverse checks for typed multiple sinks and reusable full solves."""

import unittest
import numpy as np

from production.perambulator import test_dilution as fixtures
from production.perambulator.test_dilution import Handle, CountingBackend
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, VectorBasis, PointBasis, PropagatorEndpoint,
)


class MultiSinkTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.DilutionTests()
        self.f.setUp()
        self.bases = {
            kind: (PointBasis(Handle(data, Np=data.shape[1], endpoint=self.f.endpoints[kind]))
                   if kind == 'P' else VectorBasis(Handle(data, Ne=data.shape[1], endpoint=self.f.endpoints[kind])))
            for kind, data in self.f.data.items()
        }

    def generator(self, **options):
        backend = CountingBackend()
        options = dict(MRHS=True, rhs_batch_size=5) | options
        gen = PerambulatorGenerator(self.f.lattice, Handle(self.f.matrix), 0, 1e-11, 200,
            backend=backend, gamma5=self.f.gamma, gamma5_hermitian=True,
            **options)
        gen.load('cfg')
        return gen

    def test_all_nine_products_multisink_share_rhs_and_reverse_physical_pairs(self):
        for streamed, mrhs in ((False, False), (False, True), (True, False), (True, True)):
            gen = self.generator(stream_eigenvectors=streamed, MRHS=mrhs, sink_ne_block_size=1,
                source_selection=TimeSelectionRecord(3, (2, 0, 1)),
                sink_selection=TimeSelectionRecord(3, (1, 2, 0)))
            for source_kind, source in self.bases.items():
                before = gen.backend.rhs_count
                results = gen.calc(2, src=source, sink=list(self.bases.values()))
                for sink_kind, result in zip(self.bases, results):
                    product = sink_kind + 'S' + source_kind
                    expected = np.stack([self.f.oracle(product, t, 2) for t in (1, 2, 0)])
                    np.testing.assert_allclose(result.values, expected, atol=3e-14, rtol=1e-12)
                    self.assertEqual(result.source, self.f.endpoints[source_kind])
                    self.assertEqual(result.sink, self.f.endpoints[sink_kind])
                    reverse = result.reverse(1)
                    np.testing.assert_allclose(reverse.values,
                        self.f.oracle(source_kind+'S'+sink_kind, 2, 1), atol=3e-14, rtol=1e-12)
                    self.assertEqual((reverse.sink_time, reverse.source_time), (2, 1))
                    self.assertEqual((reverse.sink, reverse.source), (result.source, result.sink))
                count = len(self.f.endpoints[source_kind].indices)
                self.assertEqual(gen.backend.rhs_count-before, count*(12 if source_kind == 'P' else 4))

    def test_distinct_same_kind_sinks_and_raw_reprojection(self):
        gen = self.generator(stream_eigenvectors=True, sink_ne_block_size=1)
        other = self.f.v[:, :1]*np.exp(.7j)
        endpoint = PropagatorEndpoint('V', 'different-actual-v', [17])
        other_basis = VectorBasis(Handle(other, Ne=1, endpoint=endpoint))
        for source_kind, src in self.bases.items():
            results = gen.calc(1, src=src, sink=[self.bases['V'], other_basis, self.bases['D'], None])
            self.assertEqual(results[1].sink, endpoint)
            expected = np.stack([self.f.oracle('VS'+source_kind, t, 1, sink_data=other) for t in range(3)])
            np.testing.assert_allclose(results[1].values, expected, atol=3e-14, rtol=1e-12)
            raw = results[-1]
            before = gen.backend.rhs_count
            projected = raw.project(sink=[self.bases['V'], other_basis, self.bases['D']])
            self.assertEqual(gen.backend.rhs_count, before)
            for a, b in zip(results, projected):
                np.testing.assert_allclose(a.values, b.values, atol=3e-14, rtol=1e-12)
            # Direct full-space columns check raw source/spin/color ordering.
            columns = self.f.inverse @ self.f.embedding(source_kind, self.f.data[source_kind], 1)
            n = len(raw.source.indices)
            for i in range(n):
                for spin in range(4):
                    if source_kind == 'P':
                        for color in range(3):
                            np.testing.assert_allclose(raw.values[i, spin, color].ravel(),
                                columns[:, spin*n*3+i*3+color], atol=3e-14, rtol=1e-12)
                    else:
                        np.testing.assert_allclose(raw.values[i, spin].ravel(),
                            columns[:, spin*n+i], atol=3e-14, rtol=1e-12)

    def test_owned_results_survive_next_calc_and_stream_delivery_uses_sink_index(self):
        gen = self.generator()
        src, sinks = self.bases['V'], [self.bases['V'], self.bases['V'], self.bases['P']]
        first = gen.calc(0, src=src, sink=sinks)
        saved = [result.values.copy() for result in first]
        gen.calc(1, src=src, sink=sinks)
        for result, expected in zip(first, saved):
            np.testing.assert_array_equal(result.values, expected)
        output = [np.empty_like(value) for value in saved]
        def write(index, selection, value):
            output[index][selection] = value
        streamed = gen.calc(0, src=src, sink=sinks, write=write)
        for value, expected, result in zip(output, saved, streamed):
            np.testing.assert_allclose(value, expected, atol=3e-14, rtol=1e-12)
            self.assertIsNone(result.values)
        def fail(*args):
            raise RuntimeError('writer failed')
        with self.assertRaisesRegex(RuntimeError, 'writer failed'):
            gen.calc(0, src=src, sink=sinks, write=fail)
        self.assertEqual(gen.last_metrics, {})

    def test_no_strings_unknown_gamma5_and_stale_raw_projection(self):
        gen = self.generator()
        with self.assertRaises(TypeError):
            gen.calc(0, src='vec', sink=[self.bases['V']])
        with self.assertRaises(TypeError):
            gen.calc(0, src=self.bases['V'], sink='vec')
        bad_point = PointBasis(Handle(self.f.points, Np=2), endpoint=self.f.endpoints['V'])
        with self.assertRaisesRegex(ValueError, 'basis type differs'):
            gen.calc(0, src=bad_point, sink=[self.bases['V']])
        raw = gen.calc(0, src=self.bases['V'], sink=None)[0]
        gen.load('other')
        with self.assertRaisesRegex(RuntimeError, 'changed the configuration'):
            raw.project(sink=[self.bases['V']])
        unknown = PerambulatorGenerator(self.f.lattice, Handle(self.f.matrix), 0, 1e-11, 200,
                                        backend=CountingBackend())
        unknown.load('cfg')
        result = unknown.calc(0, src=self.bases['V'], sink=[self.bases['V']])[0]
        with self.assertRaisesRegex(ValueError, 'declare gamma5 hermiticity'):
            result.reverse(1)

    def test_registered_d_without_pool_provenance_stays_d(self):
        gen = self.generator()
        handle = Handle(self.f.d, Ne=3, basis_id='imported-d', vector_kind='D')
        result = gen.calc(0, src=VectorBasis(handle), sink=[self.bases['V']])[0]
        self.assertEqual(result.product, 'VSD')
        self.assertEqual(result.source.data_id, 'imported-d')
        self.assertEqual(result.source.metadata, {})
        np.testing.assert_allclose(result.values,
            np.stack([self.f.oracle('VSD', t, 0) for t in range(3)]), atol=3e-14, rtol=1e-12)
        bad = VectorBasis(handle, endpoint=PropagatorEndpoint('V', 'wrong', range(3)))
        with self.assertRaisesRegex(ValueError, 'input vector representation'):
            gen.calc(0, src=bad, sink=[self.bases['V']])


if __name__ == '__main__':
    unittest.main()
