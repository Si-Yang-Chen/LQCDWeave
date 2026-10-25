"""Public table input checked against independent site matrices."""

import inspect
import itertools
import unittest
from types import SimpleNamespace
import numpy as np
from lqcdweave.production.elemental import (
    ElementalGenerator,
    BaryonElementalGenerator,
)
from lqcdweave.contracts.elemental import elemental_operator_rows, SparsePointBlock
from production.elemental.test_elemental import fixture, reference_matrix, point_basis
from test_baryon_elemental import determinant_reference


class TableFrontendTests(unittest.TestCase):
    def setUp(self):
        self.sizes, self.u, self.v, self.p = fixture()
        self.moms = ((0, 0, 0), (1, -1, 0))
        self.momentum_dict = {23: "0 0 0", -5: (1, -1, 0)}

    def test_sparse_complex_combination_with_real_only_scatter_backend(self):
        calls = []
        class RealScatter:
            @staticmethod
            def at(out, indices, values):
                self.assertEqual(out.dtype.kind, 'f')
                self.assertEqual(values.dtype.kind, 'f')
                calls.append(len(values))
                np.add.at(out, indices, values)
        class Backend:
            add = RealScatter
            def __getattr__(self, name):
                return getattr(np, name)
        first = np.arange(27).reshape(3, 3, 3)*(1+2j)
        second = (1+np.arange(18).reshape(2, 3, 3))*(-.3+.8j)
        raw = {'p2p': [[SparsePointBlock(np.array([0, 1, 0]), np.array([1, 0, 0]), first, (2, 2))],
                       [SparsePointBlock(np.array([0, 1]), np.array([1, 0]), second, (2, 2))]]}
        for xp in (np, Backend()):
            generator = ElementalGenerator.__new__(ElementalGenerator)
            generator._engine = SimpleNamespace(xp=xp, paths=('a', 'b'), num_momentum=1)
            generator.rows = (('sum', (('a', .7+.2j), ('b', -.4+.5j))),)
            result = generator._combine(raw)['p2p'][0][0].to_dense()
            expected = np.zeros((2, 3, 2, 3), complex)
            expected[0, :, 1, :] = (.7+.2j)*first[0]+(-.4+.5j)*second[0]
            expected[1, :, 0, :] = (.7+.2j)*first[1]+(-.4+.5j)*second[1]
            expected[0, :, 0, :] = (.7+.2j)*first[2]
            np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(calls, [5, 5])

    def test_complex_rows_all_four_sectors(self):
        terms = [((), 0.3 + 0.7j), ((0,), -0.2j), ((3,), 0.4)]
        table = [dict(component="sum", path=p, coefficient=c) for p, c in terms]
        table.append(dict(component="local", path=(), coefficient=1))
        g = ElementalGenerator(
            self.u,
            self.v,
            self.momentum_dict,
            table,
            points=self.p,
            blocks=("v2v", "v2p", "p2v", "p2p"),
            Ne=2,
            ne_block_size=1,
        )
        out = g.calc_path_blocks(1)
        bases = {
            "v": self.v[1, :2].reshape(2, -1).T,
            "p": point_basis(self.p[1], self.sizes),
        }
        for mi, mom in enumerate(self.moms):
            matrix = sum(c * reference_matrix(self.u, p, 1, mom) for p, c in terms)
            for sector, values in out.items():
                expected = bases[sector[0]].conj().T @ matrix @ bases[sector[-1]]
                actual = values[0][mi]
                if sector == "p2p":
                    actual = actual.to_dense()
                np.testing.assert_allclose(
                    actual.reshape(expected.shape), expected, atol=1e-12
                )
        self.assertEqual(g.components, ("sum", "local"))

    def test_ordered_derivative_combinations(self):
        terms = [((0, 1), 0.2 + 0.5j), ((1, 0), -0.3), ((), 0.7)]
        table = [dict(component="d", path=p, coefficient=c) for p, c in terms]
        out = ElementalGenerator(self.u, self.v, self.momentum_dict, table).calc_derivative(1)[
            "v2v"
        ]
        v = self.v[1].reshape(3, -1).T
        ds = [
            reference_matrix(self.u, (d,), 1) - reference_matrix(self.u, (d + 3,), 1)
            for d in range(3)
        ]
        for mi, mom in enumerate(self.moms):
            f = reference_matrix(self.u, (), 1, mom)
            expected = np.zeros((3, 3), complex)
            for directions, c in terms:
                for assignments in itertools.product((0, 1), repeat=len(directions)):
                    left, right = v.copy(), v.copy()
                    for d in [d for d, s in zip(directions, assignments) if not s][
                        ::-1
                    ]:
                        left = ds[d] @ left
                    for d in [d for d, s in zip(directions, assignments) if s]:
                        right = ds[d] @ right
                    expected += c * (-1) ** sum(assignments) * left.conj().T @ f @ right
            np.testing.assert_allclose(out[0, mi], expected, atol=2e-12)

    def test_baryon_shared_and_independent_bases(self):
        terms = [(((), (), ()), 0.5j), (((0,), (1,), ()), 0.7 - 0.2j)]
        table = [dict(component="b", path=p, coefficient=c) for p, c in terms]
        for vectors, ne in [
            (self.v, None),
            ((self.v, self.v * 0.7, self.v + 0.2), (1, 2, 3)),
        ]:
            g = BaryonElementalGenerator(
                self.u, vectors, self.momentum_dict, table, Ne=ne, ne_block_size=1
            )
            out = g.calc_path_blocks(1)["vvv"]
            bases = (vectors,) * 3 if hasattr(vectors, "shape") else vectors
            if ne is not None:
                bases = tuple(v[:, :n] for v, n in zip(bases, ne))
            for mi, mom in enumerate(self.moms):
                expected = sum(
                    c * determinant_reference(self.u, bases, p, 1, mom)
                    for p, c in terms
                )
                np.testing.assert_allclose(out[0, mi], expected, atol=2e-12)

    def test_catalog_format_and_invalid_coefficients(self):
        table = [{"id": "x", "terms": [{"path": [], "coefficient": [0, 1]}]}]
        out = ElementalGenerator(self.u, self.v, ((0, 0, 0),), table).calc_path_blocks(
            0
        )["v2v"]
        v = self.v[0].reshape(3, -1).T
        np.testing.assert_allclose(out[0, 0], 1j * v.conj().T @ v, atol=1e-12)
        for table in ([], [{"component": "x", "path": [], "coefficient": np.nan}]):
            with self.assertRaises(ValueError):
                elemental_operator_rows(table)

    def test_mapping_order_matches_list_for_meson_and_baryon(self):
        momenta = ((1, -1, 0), (0, 0, 0))
        momentum_dict = {
            27: (np.int64(1), np.int64(-1), np.int64(0)),
            -4: "0 0 0",
        }
        meson_table = [
            dict(component="sum", path=(), coefficient=0.3 + 0.7j),
            dict(component="sum", path=(3,), coefficient=-0.2j),
        ]
        from_list = ElementalGenerator(self.u, self.v, momenta, meson_table)
        from_mapping = ElementalGenerator(
            self.u, self.v, operator_table=meson_table, momentum_dict=momentum_dict
        )
        self.assertEqual(from_mapping.momenta, momenta)
        list_values = from_list.calc_path_blocks(1)["v2v"]
        mapping_values = from_mapping.calc_path_blocks(1)["v2v"]
        vectors = self.v[1].reshape(3, -1).T
        for mi, momentum in enumerate(momenta):
            matrix = sum(
                row["coefficient"]
                * reference_matrix(self.u, row["path"], 1, momentum)
                for row in meson_table
            )
            expected = vectors.conj().T @ matrix @ vectors
            np.testing.assert_allclose(list_values[0, mi], expected, atol=2e-12)
            np.testing.assert_allclose(mapping_values[0, mi], expected, atol=2e-12)

        baryon_table = [
            dict(component="b", path=((), (), ()), coefficient=0.3 + 0.2j),
            dict(component="b", path=((0,), (), (1,)), coefficient=-0.1j),
        ]
        baryon_list = BaryonElementalGenerator(
            self.u, self.v, momenta, baryon_table, ne_block_size=1
        )
        baryon_mapping = BaryonElementalGenerator(
            self.u,
            self.v,
            operator_table=baryon_table,
            momentum_dict=momentum_dict,
            ne_block_size=1,
        )
        self.assertEqual(baryon_mapping.momenta, momenta)
        list_values = baryon_list.calc_path_blocks(1)["vvv"]
        mapping_values = baryon_mapping.calc_path_blocks(1)["vvv"]
        terms = [((), (), ()), ((0,), (), (1,))]
        coefficients = (0.3 + 0.2j, -0.1j)
        for mi, momentum in enumerate(momenta):
            expected = sum(
                coefficient
                * determinant_reference(
                    self.u, (self.v,) * 3, paths, 1, momentum
                )
                for paths, coefficient in zip(terms, coefficients)
            )
            np.testing.assert_allclose(list_values[0, mi], expected, atol=2e-12)
            np.testing.assert_allclose(mapping_values[0, mi], expected, atol=2e-12)

    def test_momentum_mapping_validation_is_shared_with_insertion_rows(self):
        from lqcdweave.operators.single_hadron.api import (
            DerivativeName,
            GammaName,
            Insertion,
            ProjectionName,
        )

        table = [dict(component="x", path=(), coefficient=1)]
        baryon_table = [
            dict(component="b", path=((), (), ()), coefficient=1)
        ]
        for cls, vectors, operator_table in (
            (ElementalGenerator, self.v, table),
            (BaryonElementalGenerator, self.v, baryon_table),
        ):
            with self.subTest(generator=cls.__name__, case="missing"):
                with self.assertRaises(ValueError):
                    cls(self.u, vectors, operator_table=operator_table)
            with self.subTest(generator=cls.__name__, case="conflict"):
                with self.assertRaises(ValueError):
                    cls(
                        self.u,
                        vectors,
                        self.moms,
                        operator_table,
                        momentum_dict=self.momentum_dict,
                    )

        bad_mappings = (
            {},
            {7: (0, 0, 0), -2: "0 0 0"},
            {4: (True, 0, 0)},
            {4: (0.5, 0, 0)},
            {4: (0, 0)},
            {4: "0 x 1"},
        )
        for mapping in bad_mappings:
            with self.subTest(mapping=mapping, target="production"):
                with self.assertRaises(ValueError):
                    ElementalGenerator(
                        self.u,
                        self.v,
                        operator_table=table,
                        momentum_dict=mapping,
                    )
            with self.subTest(mapping=mapping, target="insertion row"):
                insertion = Insertion(
                    GammaName.PI,
                    DerivativeName.IDEN,
                    ProjectionName.A1,
                    mapping,
                )
                with self.assertRaises(ValueError):
                    insertion[0](0, 0, 0)

    def test_constructor_callback_and_time_only_methods(self):
        table = [dict(component="x", path=(), coefficient=0.7j)]
        saved = []
        g = ElementalGenerator(
            self.u,
            self.v,
            self.momentum_dict,
            table,
            write=lambda *args: saved.append(args[-1].copy()),
        )
        self.assertIsNone(g.calc_path_blocks(0))
        self.assertEqual(len(saved), 2)
        v = self.v[0].reshape(3, -1).T
        np.testing.assert_allclose(saved[0], 0.7j * v.conj().T @ v, atol=1e-12)
        for cls, methods in [
            (ElementalGenerator, ("calc_path_blocks", "calc_derivative")),
            (BaryonElementalGenerator, ("calc_path_blocks",)),
        ]:
            for method in methods:
                self.assertEqual(
                    tuple(inspect.signature(getattr(cls, method)).parameters),
                    ("self", "t"),
                )

    def test_ne_inherits_and_invalid_selection_rejected(self):
        table = [dict(component="x", path=(), coefficient=1)]
        g = ElementalGenerator(self.u, self.v, self.momentum_dict, table)
        self.assertEqual(g.calc_path_blocks(0)["v2v"].shape[-2:], (3, 3))
        with self.assertRaises(ValueError):
            ElementalGenerator(self.u, self.v, self.momentum_dict, table, Ne=4)
        with self.assertRaises(ValueError):
            ElementalGenerator(self.u, self.v, self.momentum_dict, table, blocks=("unknown",))
        with self.assertRaises(ValueError):
            ElementalGenerator(
                self.u,
                self.v,
                self.momentum_dict,
                [dict(component="x", path=(3,), coefficient=1)],
            ).calc_derivative(0)

    def test_temporal_path_uses_real_endpoint_and_owned_gauge(self):
        table = [dict(component="forward", path=(6,), coefficient=1)]
        u, v = self.u.copy(), self.v.copy()
        g = ElementalGenerator(u, v, self.momentum_dict, table)
        u.fill(0)
        result = g.calc_path_blocks(1)["v2v"]
        for mi, mom in enumerate(self.moms):
            expected = (
                self.v[1].reshape(3, -1).conj()
                @ reference_matrix(self.u, (6,), 1, mom)
                @ self.v[2].reshape(3, -1).T
            )
            np.testing.assert_allclose(result[0, mi], expected, atol=1e-12)
        with self.assertRaises(ValueError):
            g.calc_path_blocks(self.sizes[3])

    def test_temporal_weighted_point_sectors_match_independent_site_loops(self):
        terms_by_component = (
            (
                "time-shift",
                (((6, 0), 0.3 + 0.7j), ((7, 3), -0.2 + 0.4j)),
            ),
            (
                "time-excursion",
                (((6, 7, 1), -0.15 + 0.35j), ((7, 6, 4), 0.25 - 0.1j)),
            ),
        )
        table = [
            dict(component=component, path=path, coefficient=coefficient)
            for component, terms in terms_by_component
            for path, coefficient in terms
        ]
        points_by_time = self.p
        g = ElementalGenerator(
            self.u,
            self.v,
            self.momentum_dict,
            table,
            points=points_by_time,
            blocks=("v2p", "p2v", "p2p"),
            Ne=2,
            phase_convention="midpoint",
        )

        lx, ly, lz, lt = self.sizes
        spatial_size = np.asarray((lx, ly, lz))
        step_axes = (0, 1, 2, 0, 1, 2, 3, 3)
        step_signs = (1, 1, 1, -1, -1, -1, 1, -1)

        for t in range(lt):
            actual = g.calc_path_blocks(t)
            left_modes = self.v[t, :2].reshape(2, -1).T
            left_points = point_basis(points_by_time[t], self.sizes)
            for ci, (component, terms) in enumerate(terms_by_component):
                self.assertEqual(g.components[ci], component)
                expected_v2p = [np.zeros((2, len(points_by_time[t]) * 3), complex)
                                for _ in self.moms]
                expected_p2v = [np.zeros((len(points_by_time[t]) * 3, 2), complex)
                                for _ in self.moms]
                expected_p2p = [np.zeros((len(points_by_time[t]) * 3,
                                          len(points_by_time[t]) * 3), complex)
                                for _ in self.moms]
                expected_support = set()

                for path, coefficient in terms:
                    displacement = np.zeros(4, dtype=int)
                    for step in path:
                        displacement[step_axes[step]] += step_signs[step]
                    endpoint = (t + int(displacement[3])) % lt
                    right_modes = self.v[endpoint, :2].reshape(2, -1).T
                    right_points = point_basis(points_by_time[endpoint], self.sizes)
                    for mi, momentum in enumerate(self.moms):
                        matrix = reference_matrix(
                            self.u, path, t, momentum, phase="midpoint"
                        )
                        expected_v2p[mi] += coefficient * (
                            left_modes.conj().T @ matrix @ right_points
                        )
                        expected_p2v[mi] += coefficient * (
                            left_points.conj().T @ matrix @ right_modes
                        )
                        expected_p2p[mi] += coefficient * (
                            left_points.conj().T @ matrix @ right_points
                        )

                    for left_index, coordinate in enumerate(points_by_time[t]):
                        target = (coordinate + displacement[:3]) % spatial_size
                        for right_index, right_coordinate in enumerate(
                                points_by_time[endpoint]):
                            if np.array_equal(target, right_coordinate):
                                expected_support.add((left_index, right_index))

                for mi in range(len(self.moms)):
                    np.testing.assert_allclose(
                        actual["v2p"][ci, mi],
                        expected_v2p[mi].reshape(2, len(points_by_time[t]), 3),
                        atol=2e-12,
                    )
                    np.testing.assert_allclose(
                        actual["p2v"][ci, mi],
                        expected_p2v[mi].reshape(len(points_by_time[t]), 3, 2),
                        atol=2e-12,
                    )
                    sparse = actual["p2p"][ci][mi]
                    self.assertEqual(
                        list(zip(sparse.rows.tolist(), sparse.columns.tolist())),
                        sorted(expected_support),
                    )
                    np.testing.assert_allclose(
                        sparse.to_dense().reshape(expected_p2p[mi].shape),
                        expected_p2p[mi],
                        atol=2e-12,
                    )

    def test_baryon_constructor_callback(self):
        table = [dict(component="b", path=((), (), ()), coefficient=0.3j)]
        output = np.zeros((1, 2, 3, 3, 3), complex)

        def write(ri, mi, sector, slices, value):
            self.assertEqual(sector, "vvv")
            output[(ri, mi, *slices)] = value

        g = BaryonElementalGenerator(
            self.u, self.v, self.momentum_dict, table, write=write, ne_block_size=1
        )
        self.assertIsNone(g.calc_path_blocks(1))
        for mi, mom in enumerate(self.moms):
            expected = 0.3j * determinant_reference(
                self.u, (self.v,) * 3, ((), (), ()), 1, mom
            )
            np.testing.assert_allclose(output[0, mi], expected, atol=1e-12)
