"""Raw dilution normalization, legacy draws and independent module state."""

import unittest

import numpy as np

from lqcdweave.production.noisevector.api import NoisevectorGenerator


class BasisHandle:
    def __init__(self, values):
        self.values = values
        self.basis_id = {"actual_id": "test-basis"}

    def load(self, key):
        if key == "fail":
            raise OSError("input unavailable")
        return self.values


def basis(nt=2, ne=9):
    rng = np.random.RandomState(71)
    values = []
    for _ in range(nt):
        q, _ = np.linalg.qr(rng.randn(12, ne) + 1j * rng.randn(12, ne))
        values.append(q.T.reshape(ne, 1, 1, 4, 3))
    return np.array(values)


def legacy_core(values, counts, used, seed):
    """Frozen original draw/transfer order, using matrix products as oracle.

    Source: localized_blending/source/lattice/generator/noisevector.py,
    highmode=0, full_noisev=False. Local RandomState matches np.random.seed.
    """
    rng = np.random.RandomState(seed)
    nt = values.shape[0]
    result = []
    offset = 0
    for n, m in zip(counts, used):
        parent = values[:, offset:offset + n].reshape(nt, n, -1)
        if m == n:
            mixed = parent.copy()
        elif m == 1:
            transfer = np.exp(1j * rng.uniform(0, 2 * np.pi, (nt, n)))
            mixed = np.array([transfer[t:t + 1] @ parent[t] for t in range(nt)])
        else:
            transfer = []
            for _ in range(nt):
                random = rng.randn(n, n) + 1j * rng.randn(n, n)
                q, _ = np.linalg.qr(random)
                transfer.append(q[:, :m])
            mixed = np.array([transfer[t].T @ parent[t] for t in range(nt)])
        result.append(mixed.reshape(nt, m, *values.shape[2:]))
        offset += n
    return np.concatenate(result, axis=1)


class NoisevectorGeneratorTests(unittest.TestCase):
    def test_time_selection_matches_rows_and_failed_reload_invalidates_state(self):
        from lqcdweave.contracts.time_selection import TimeSelection, TimeSelectionRecord
        values = basis(nt=2, ne=3)
        source = BasisHandle(values)
        source.basis_orthogonal = True
        generator = NoisevectorGenerator(source, ((3,), (2,)))
        for selection in (TimeSelectionRecord(8, (6, 1)), TimeSelection(8, 8, (1, 6))):
            with self.subTest(selection=selection):
                source.time_selection = selection
                result = generator.load('cfg').calc(seed=31)
                np.testing.assert_allclose(result, legacy_core(values, (3,), (2,), 31), atol=2e-15)
                description = generator.describe()
                self.assertEqual(description['time_selection'], selection)
                self.assertEqual(tuple(time for time, _ in description['time_pools']), selection.times)
                self.assertEqual(len(description['time_pools']), result.shape[0])
                saved = result.copy()
                for times in ((6,), (6, 1, 3)):
                    source.time_selection = TimeSelectionRecord(8, times)
                    with self.assertRaisesRegex(ValueError, 'time extent differs'):
                        generator.load('bad')
                    description = generator.describe()
                    self.assertFalse(description['complete'])
                    self.assertFalse(description['basis_orthogonal'])
                    self.assertIsNone(description['parent_basis_id'])
                    self.assertIsNone(description['time_selection'])
                    self.assertEqual(description['time_pools'], ())
                    with self.assertRaisesRegex(RuntimeError, 'load a configuration'):
                        generator.calc(seed=31)
                    np.testing.assert_array_equal(result, saved)

    def test_time_selection_uses_the_shared_record_contract(self):
        from types import SimpleNamespace
        source = BasisHandle(basis(nt=2, ne=3))
        source.time_selection = SimpleNamespace(Nt=2, times=(6, 1))
        generator = NoisevectorGenerator(source, ((3,), (2,)))
        with self.assertRaisesRegex(TypeError, 'TimeSelection or TimeSelectionRecord'):
            generator.load('cfg')
        with self.assertRaisesRegex(RuntimeError, 'load a configuration'):
            generator.calc(seed=31)

    def test_actual_parent_labels_are_distinct_from_storage_positions(self):
        values = basis(ne=5)
        source = BasisHandle(values)
        source.mode_labels = (8, 3, 11, 2, 7)
        generator = NoisevectorGenerator(source, ((3,), (2,)), mode_groups=((4, 1, 2),)).load('cfg')
        result = generator.calc(seed=812)
        self.assertEqual(generator.describe()['mode_groups'], ((7, 3, 11),))
        np.testing.assert_allclose(result, legacy_core(values[:, [4, 1, 2]], (3,), (2,), 812), atol=1e-14)
        source.mode_labels = (8, 8, 11, 2, 7)
        with self.assertRaisesRegex(ValueError, 'parent mode labels'):
            generator.load('cfg')
        self.assertFalse(generator.describe()['complete'])

    def test_legacy_three_branches_and_public_identity(self):
        from lqcdweave.production import NoisevectorGenerator as Public
        from lqcdweave.production.noisevector import NoisevectorGenerator as Package
        self.assertIs(Public, NoisevectorGenerator)
        self.assertIs(Package, NoisevectorGenerator)
        values = basis()
        generator = NoisevectorGenerator(BasisHandle(values), ([2, 3, 4], [2, 1, 2]), mixing="legacy").load("cfg")
        expected = legacy_core(values, [2, 3, 4], [2, 1, 2], 19)
        actual = generator.calc(19)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
        self.assertEqual(actual.shape, (2, 5, 1, 1, 4, 3))
        self.assertEqual(actual.dtype, np.complex128)
        description = generator.describe()
        self.assertEqual(description["branches"], ("exact", "u1", "qr"))
        self.assertEqual(description["covariance_factors"], (1., 1., .5))
        self.assertEqual(description["vector_labels"], ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)))

    def test_noncontiguous_groups_and_medium_offset(self):
        values = basis()
        selected = values[:, [7, 3, 5, 0, 8]]
        generator = NoisevectorGenerator(BasisHandle(values), ([2, 3], [2, 1]),
                                         mode_groups=((7, 3), (5, 0, 8)), mixing="legacy").load("cfg")
        np.testing.assert_allclose(generator.calc(8), legacy_core(selected, [2, 3], [2, 1], 8))
        generator = NoisevectorGenerator(BasisHandle(values), ([3, 4], [1, 2]),
                                         mode_offset=2, mixing="legacy").load("cfg")
        np.testing.assert_allclose(generator.calc(8), legacy_core(values[:, 2:], [3, 4], [1, 2], 8))

    def test_raw_gram_and_parent_support(self):
        values = basis()
        generator = NoisevectorGenerator(BasisHandle(values), ([2, 3, 4], [2, 1, 2]), mixing="legacy").load("cfg")
        actual = generator.calc(20).reshape(2, 5, -1)
        for t in range(2):
            np.testing.assert_allclose(actual[t].conj() @ actual[t].T,
                                       np.diag([1, 1, 3, 1, 1]), atol=2e-14)
            coefficients = values[t].reshape(9, -1).conj() @ actual[t].T
            np.testing.assert_allclose(coefficients[:2, 2:], 0, atol=2e-14)
            np.testing.assert_allclose(coefficients[2:5, :2], 0, atol=2e-14)
            np.testing.assert_allclose(coefficients[5:, :3], 0, atol=2e-14)

    def test_expected_phase_and_qr_covariance(self):
        # Projectors are independent of QR column phase conventions.
        values = np.eye(6, dtype=np.complex128).reshape(1, 6, 1, 1, 2, 3)
        for mixing, expected in (("qr", [.5] * 6), ("legacy", [1, 1, .5, .5, .5, .5])):
            generator = NoisevectorGenerator(BasisHandle(values), ([2, 4], [1, 2]), mixing=mixing).load("cfg")
            covariance = np.zeros((6, 6), dtype=np.complex128)
            for seed in range(2048):
                d = generator.calc(seed).reshape(3, 6)
                covariance += d.T @ d.conj()
            covariance /= 2048
            with self.subTest(mixing=mixing):
                np.testing.assert_allclose(covariance, np.diag(expected), atol=.035)

    def test_rng_isolation_output_ownership_and_failed_load(self):
        values = basis()
        source = BasisHandle(values.copy())
        generator = NoisevectorGenerator(source, ([3], 1), mixing="legacy").load("cfg")
        source.values[:] = 0
        state = np.random.get_state()
        first = generator.calc(17)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_allclose(first, legacy_core(values, [3], [1], 17))
        saved = first.copy()
        second = generator.calc(18)
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(first, saved)
        description = generator.describe()
        description["parent_basis_id"]["actual_id"] = "changed"
        self.assertEqual(generator.describe()["parent_basis_id"]["actual_id"], "test-basis")
        with self.assertRaises(OSError):
            generator.load("fail")
        with self.assertRaises(RuntimeError):
            generator.calc(17)
        self.assertFalse(generator.describe()["complete"])

    def test_failures_do_not_claim_completed_realization(self):
        generator = NoisevectorGenerator(BasisHandle(basis()), ([3], 1))
        with self.assertRaises(RuntimeError):
            generator.calc(1)
        generator.load("cfg").calc(1)
        with self.assertRaises(ValueError):
            generator.calc(-1)
        self.assertFalse(generator.describe()["complete"])
        self.assertIsNone(generator.describe()["seed"])

    def test_default_single_vector_uses_qr_normalization(self):
        values = basis()
        generator = NoisevectorGenerator(BasisHandle(values), ([3], 1)).load("cfg")
        d = generator.calc(7).reshape(2, 1, -1)
        for t in range(2):
            np.testing.assert_allclose(d[t].conj() @ d[t].T, np.eye(1), atol=2e-14)
        self.assertEqual(generator.describe()["branches"], ("qr",))
        self.assertEqual(generator.describe()["covariance_factors"], (1 / 3,))

    def test_invalid_groups_counts_and_payload(self):
        source = BasisHandle(basis())
        for dilution in (([2], [3]), ([0], [1]), ([2, 3], [1]), ([], []), ([2], True)):
            with self.subTest(dilution=dilution), self.assertRaises(ValueError):
                NoisevectorGenerator(source, dilution)
        for groups in (((0, 0),), ((0,),), ((-1, 0),)):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                NoisevectorGenerator(source, ([2], [1]), mode_groups=groups)
        with self.assertRaises(ValueError):
            NoisevectorGenerator(source, ([2], 1), mode_groups=((0, 1),), mode_offset=1)
        with self.assertRaises(ValueError):
            NoisevectorGenerator(source, ([2], 1), mixing="unknown")
        with self.assertRaises(ValueError):
            NoisevectorGenerator(source, ([10], 1)).load("cfg")
        for payload in (np.zeros((2, 3)), np.zeros((1, 2, 1, 1, 1, 3))):
            with self.subTest(shape=payload.shape), self.assertRaises(ValueError):
                NoisevectorGenerator(BasisHandle(payload), ([2], 1)).load("cfg")
