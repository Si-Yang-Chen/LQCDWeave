"""Explicit basis inputs, ordered sinks, and measurement synchronization."""

import unittest

import numpy as np
from production.perambulator import test_propagator as fixtures
from production.perambulator.test_propagator import Handle
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PropagatorTests()
        self.fixture.setUp()

    def test_only_explicit_measurement_synchronizes_production_stages(self):
        for measured in (False, True):
            gen = self.fixture.generator(measure=measured, MRHS=True)
            syncs = []
            gen.backend.synchronize = lambda: syncs.append('sync')
            backend = gen.backend
            gen = self.fixture.generator(backend=backend, measure=measured, MRHS=True)
            gen.load('cfg')
            results = gen.calc(1,src=self.fixture.src_basis,sink=self.fixture.sink_bases)
            for result in results:
                np.testing.assert_allclose(result.values,
                    self.fixture.reference(result.product,1).reshape(result.values.shape),atol=2e-14)
            self.assertEqual(gen.last_metrics['timing_mode'],
                             'synchronized' if measured else 'submission')
            self.assertEqual(bool(syncs), measured)

    def test_measure_rejects_nonboolean_before_allocations(self):
        with self.assertRaisesRegex(ValueError, 'measure must be boolean'):
            self.fixture.generator(measure='yes')

    def test_legacy_backend_retains_its_stage_dependency_hook(self):
        backend = fixtures.CountingBackend()
        backend.stream_ordered_solve = False
        syncs = []
        backend.synchronize = lambda: syncs.append('sync')
        gen = self.fixture.generator(backend=backend, MRHS=True)
        gen.load('cfg')
        actual = gen.calc(1,src=self.fixture.src_basis,sink=[self.fixture.sink_v_basis])[0]
        self.assertTrue(syncs)
        np.testing.assert_allclose(actual.values,self.fixture.reference('VSV',1),atol=2e-14)

    def test_point_validation_supports_unique_without_axis_and_narrow_coordinates(self):
        class WithoutUniqueAxis:
            def __getattr__(self, name):
                return getattr(np, name)

            def unique(self, values, **kwargs):
                if "axis" in kwargs:
                    raise NotImplementedError("unique(axis=...) is unavailable")
                return np.unique(values)

        gen = self.fixture.generator()
        gen.xp = WithoutUniqueAxis()
        gen.latt_size = (17, 16, 1, 2)
        # Spatial points are stored time-first: (time, point, xyz).
        coordinates = np.asarray([
            [[0, 0, 0], [16, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [16, 0, 0], [0, 1, 0]],
        ], dtype=np.int8)
        loaded = gen._read_points(Handle(coordinates), 3, 'cfg', gen.source_selection)
        np.testing.assert_array_equal(loaded, coordinates)
        duplicates = coordinates.copy()
        duplicates[:, 2] = duplicates[:, 0]
        with self.assertRaisesRegex(ValueError, "points must be distinct"):
            gen._read_points(Handle(duplicates), 3, 'cfg', gen.source_selection)

    def test_sink_list_order_is_preserved_and_each_sink_gets_its_actual_product(self):
        f = self.fixture
        gen = f.generator()
        gen.load('cfg')
        results = gen.calc(1,src=f.src_basis,sink=[f.sink_p_basis,f.sink_v_basis])
        self.assertEqual([result.product for result in results],['PSV','VSV'])
        for result in results:
            np.testing.assert_allclose(result.values,
                f.reference(result.product,1).reshape(result.values.shape),atol=2e-14)

    def test_independent_source_and_sink_bases_bind_result_endpoints(self):
        f = self.fixture
        source = Handle(f.vsrc, Ne=2, basis_id='source')
        sink = Handle(f.vsink, Ne=3, basis_id='different-sink')
        gen = f.generator(src=VectorBasis(source),vector_sink=VectorBasis(sink))
        gen.load('cfg')
        result = gen.calc(1,src=f.src_basis,sink=[f.sink_v_basis])[0]
        self.assertEqual(result.source.data_id,'source')
        self.assertEqual(result.sink.data_id,'different-sink')
        np.testing.assert_allclose(result.values,f.reference('VSV',1),atol=2e-14)

    def test_point_source_needs_only_the_sink_basis_actually_requested(self):
        f = self.fixture
        source = PointBasis(Handle(f.psrc,Np=1,basis_id='point-source'))
        gen = f.generator(src=source)
        gen.load('cfg')
        result = gen.calc(1,src=f.src_basis,sink=[f.sink_v_basis])[0]
        self.assertEqual(result.product,'VSP')
        np.testing.assert_allclose(result.values,
            f.reference('VSP',1).reshape(result.values.shape),atol=2e-14)

    def test_basis_loading_is_deferred_until_calc_receives_the_basis(self):
        f = self.fixture
        source = Handle(f.vsrc,Ne=2,basis_id='source')
        sink = Handle(f.vsink,Ne=3,basis_id='sink')
        gen = f.generator(src=VectorBasis(source),vector_sink=VectorBasis(sink))
        gen.load('cfg')
        self.assertEqual(source.keys,[])
        self.assertEqual(sink.keys,[])
        gen.calc(1,src=f.src_basis,sink=[f.sink_v_basis])
        self.assertEqual(source.keys,['cfg'])
        self.assertEqual(sink.keys,['cfg'])

    def test_calc_requires_explicit_source_and_sink_list(self):
        gen = self.fixture.generator()
        gen.load('cfg')
        with self.assertRaises(TypeError):
            gen.calc(1)
        with self.assertRaises(TypeError):
            gen.calc(1,src=self.fixture.src_basis)
        self.assertEqual(gen.backend.calls,0)


if __name__ == '__main__':
    unittest.main()
