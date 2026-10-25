"""Module tests split from test_laplace_backend.py; shared numerical fixtures stay unchanged."""

from fixtures.laplace_backend import (
    DenseLaplaceBackend,
    LaplaceEigenvectorGenerator,
    Mock,
    _LaplaceBackendFixture,
    dense_laplacian,
    np,
)


class LaplaceBackendTests(_LaplaceBackendFixture):
    def test_generic_backend_and_transform_invalidation(self):
        backend = DenseLaplaceBackend()
        gen = LaplaceEigenvectorGenerator(self.sizes, self.handle, 2, 1e-10, backend=backend).load("cfg")
        self.assertEqual(backend.loads, 1)
        self.assertEqual(self.handle.calls, 1)
        initial = backend.invalidations
        gen.project_SU3()
        self.assertEqual(backend.invalidations, initial+1)
        for t in range(2):
            vectors, values = gen.calc(t)
            matrix = dense_laplacian(gen._links[:3, t])
            np.testing.assert_allclose(matrix @ vectors.reshape(2, -1).T,
                                       vectors.reshape(2, -1).T * values, atol=1e-12)
        backend.fail = True
        with self.assertRaisesRegex(RuntimeError, "failed"):
            gen.stout_smear(1, .1)
        self.assertFalse(gen.describe()["loaded"])
        self.assertEqual(backend.invalidations, initial+2)
        gen.load("next")
        self.assertEqual(gen.describe()["transforms"], [])

    def test_bad_backend_load_invalidates_previous_configuration(self):
        backend = DenseLaplaceBackend()
        gen = LaplaceEigenvectorGenerator(self.sizes, self.handle, 2, 1e-10, backend=backend).load("cfg")
        backend.load = Mock(return_value=np.zeros((1,), complex))
        before = backend.invalidations
        with self.assertRaisesRegex(ValueError, "layout"):
            gen.load("bad")
        self.assertFalse(gen.describe()["loaded"])
        self.assertIsNone(gen.describe()["key"])
        self.assertEqual(backend.invalidations, before+2)
