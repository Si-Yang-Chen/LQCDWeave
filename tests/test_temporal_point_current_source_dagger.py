from contextlib import ExitStack
import unittest

import numpy as np

from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.operators.single_hadron.api import gamma
from lqcdweave.operators.single_hadron.operator import OperatorPath
import test_directed_current_point_diagram as point_current_fixture


class TemporalPointCurrentSourceDaggerTests(unittest.TestCase):
    """Exercise B source-adjoint views over registered temporal point products."""

    def setUp(self):
        fixture = point_current_fixture.RegisteredTemporalPointCurrentTests(
            "test_open_temporal_pp_production_keeps_invalid_endpoint_support_empty"
        )
        fixture.setUp()
        self.fixture = fixture
        self.addCleanup(fixture.doCleanups)

        produced = fixture._produce_point_current(name="source-dagger-current")
        self._reader_stack = ExitStack()
        self.addCleanup(self._reader_stack.close)
        readers = {
            sector: self._reader_stack.enter_context(
                fixture.index.open(produced["datasets"][sector])
            )
            for sector in ("v2p", "p2v", "p2p")
        }
        left_points = self._reader_stack.enter_context(fixture.index.open(fixture.left_id))
        right_points = self._reader_stack.enter_context(fixture.index.open(fixture.right_id))
        self.elemental = DatasetElemental(
            readers, point_dataset=left_points, point_right_dataset=right_points,
        )

    def _meson(self, step, momentum, coefficient):
        operator = OperatorPath(
            "single-temporal-path",
            ((5, GaugePath((step,)), momentum, coefficient),),
        )
        return Meson(
            self.elemental, operator, source=True, phase_convention="midpoint",
        )

    def _raw_dense_reference(self, sector, component, momentum_index, anchor):
        fixture = self.fixture
        delta_t = component.key.displacement[3]
        endpoint = (anchor + delta_t) % fixture.lattice[3]
        momentum = fixture.momentum_list[momentum_index]
        if sector == "p2v":
            result = np.zeros((len(fixture.left_labels), 3, 2), dtype=np.complex128)
            for point, (x, _y, _z) in enumerate(fixture.left_coordinates[anchor]):
                x = int(x)
                for mode in range(2):
                    result[point, :, mode] = (
                        fixture._phase(momentum, x)
                        * fixture._vector_link(anchor, component.key.steps[0], x)
                        @ fixture.mode_values[endpoint][mode, 0, 0, x]
                    )
            return result
        if sector == "v2p":
            result = np.zeros((2, len(fixture.right_labels), 3), dtype=np.complex128)
            for mode in range(2):
                for point, (x, _y, _z) in enumerate(fixture.right_coordinates[endpoint]):
                    x = int(x)
                    result[mode, point] = (
                        fixture._phase(momentum, x)
                        * fixture.mode_values[anchor][mode, 0, 0, x].conj()
                        @ fixture._vector_link(anchor, component.key.steps[0], x)
                    )
            return result
        raise AssertionError(f"unexpected dense raw sector {sector!r}")

    def _raw_sparse_reference(self, component, momentum_index, anchor):
        fixture = self.fixture
        step = component.key.steps[0]
        endpoint = (anchor + component.key.displacement[3]) % fixture.lattice[3]
        momentum = fixture.momentum_list[momentum_index]
        left = fixture.left_coordinates[anchor]
        right = fixture.right_coordinates[endpoint]
        right_lookup = {tuple(coordinate): index for index, coordinate in enumerate(right)}
        extent = np.asarray(fixture.lattice[:3], dtype=np.int64)
        displacement = np.asarray(component.key.displacement[:3], dtype=np.int64)
        indices, values = [], []
        for row, coordinate in enumerate(left):
            target = tuple((coordinate + displacement) % extent)
            column = right_lookup.get(target)
            if column is None:
                continue
            x = int(coordinate[0])
            indices.append((row, column))
            values.append(
                fixture._phase(momentum, x) * fixture._vector_link(anchor, step, x)
            )
        return (
            np.asarray(indices, dtype=np.int64).reshape((-1, 2)),
            np.asarray(values, dtype=np.complex128).reshape((-1, 3, 3)),
        )

    def test_registered_vp_pv_pp_source_dagger_matches_independent_site_loops(self):
        fixture = self.fixture
        coefficient = 0.375 - 1.25j
        g5 = gamma(8)

        dense_cases = (
            # Each case reads the opposite registered sector, and exercises a
            # periodic endpoint wrap in its logical bar/field metadata.
            ("v2p", "p2v", 6, 2, 1, "point_left", fixture.left_id,
             fixture.left_labels, "mode_right", (2, 0, 1)),
            ("p2v", "v2p", 7, 0, 0, "point_right", fixture.right_id,
             fixture.right_labels, "mode_left", (1, 2, 0)),
        )
        for (logical_sector, raw_sector, step, time, momentum, raw_point_axis,
             point_id, labels, raw_mode_axis, permutation) in dense_cases:
            with self.subTest(sector=logical_sector, step=step, time=time):
                meson = self._meson(step, momentum, coefficient)
                description, = meson.describe_factors(
                    time, logical_sector, usedNe=2, usedNp=None,
                )
                self.assertEqual(description.logical_sector, logical_sector)
                self.assertEqual(description.raw_sector, raw_sector)
                self.assertTrue(description.dagger)
                self.assertEqual(description.anchor_time, time)
                self.assertEqual(description.operator_term.coefficient, coefficient)
                self.assertEqual(description.tensor.axis_bindings[raw_point_axis],
                                 (point_id, tuple(labels)))
                self.assertEqual(description.tensor.axis_bindings[raw_mode_axis],
                                 (fixture.basis_id, (17, 5)))

                if logical_sector == "v2p":
                    self.assertEqual(description.bar_time_unwrapped, 3)
                    self.assertEqual(description.field_time_unwrapped, 2)
                    self.assertEqual((description.bar_time, description.field_time), (0, 2))
                    self.assertEqual(
                        description.endpoint_offsets,
                        ((0, 0, 0, 1), (0, 0, 0, 0)),
                    )
                else:
                    self.assertEqual(description.bar_time_unwrapped, -1)
                    self.assertEqual(description.field_time_unwrapped, 0)
                    self.assertEqual((description.bar_time, description.field_time), (2, 0))
                    self.assertEqual(
                        description.endpoint_offsets,
                        ((0, 0, 0, -1), (0, 0, 0, 0)),
                    )

                raw = meson.read_factor_block(description)
                component = description.operator_term.component
                expected_raw = self._raw_dense_reference(
                    raw_sector, component, momentum, time,
                )
                np.testing.assert_allclose(raw, expected_raw, rtol=3e-13, atol=3e-13)
                expected_spatial = expected_raw.conj().transpose(permutation)
                weighted = next(meson.factors(time, logical_sector))
                self.assertEqual(weighted.coefficient, coefficient.conjugate())
                self.assertEqual(weighted.endpoint_times,
                                 (description.bar_time, description.field_time))
                np.testing.assert_allclose(
                    weighted.spatial, expected_spatial, rtol=3e-13, atol=3e-13,
                )

                unweighted = meson.materialize_unweighted_factor(description, raw)
                spin = g5 @ gamma(description.operator_term.gamma).conj().T @ g5
                expected_unweighted = np.einsum(
                    "ab,...->ab...", spin, expected_spatial, optimize=False,
                )
                np.testing.assert_allclose(
                    unweighted, expected_unweighted, rtol=3e-13, atol=3e-13,
                )
                np.testing.assert_allclose(
                    weighted.dense(), coefficient.conjugate() * expected_unweighted,
                    rtol=3e-13, atol=3e-13,
                )

        step, time, momentum = 7, 1, 1
        meson = self._meson(step, momentum, coefficient)
        description, = meson.describe_factors(time, "p2p", usedNe=2, usedNp=None)
        self.assertEqual(description.logical_sector, "p2p")
        self.assertEqual(description.raw_sector, "p2p")
        self.assertTrue(description.dagger)
        self.assertEqual(description.anchor_time, time)
        self.assertEqual((description.bar_time, description.field_time), (0, 1))
        self.assertEqual(description.bar_time_unwrapped, 0)
        self.assertEqual(description.field_time_unwrapped, 1)
        self.assertEqual(
            description.endpoint_offsets,
            ((0, 0, 0, -1), (0, 0, 0, 0)),
        )
        raw_bindings = description.tensor.axis_bindings
        self.assertEqual(raw_bindings["point_left"],
                         (fixture.left_id, tuple(fixture.left_labels)))
        self.assertEqual(raw_bindings["point_right"],
                         (fixture.right_id, tuple(fixture.right_labels)))
        self.assertEqual(description.tensor.point_shape,
                         (len(fixture.left_labels), len(fixture.right_labels)))

        raw = meson.read_factor_block(description)
        expected_indices, expected_values = self._raw_sparse_reference(
            description.operator_term.component, momentum, time,
        )
        self.assertGreater(len(expected_indices), 0)
        np.testing.assert_array_equal(raw["indices"], expected_indices)
        np.testing.assert_allclose(raw["values"], expected_values, rtol=3e-13, atol=3e-13)

        bridge, = meson.materialize_sparse_bridge_part(description, raw)
        spin, sparse_weight, source_record = bridge
        np.testing.assert_allclose(
            spin, g5 @ gamma(description.operator_term.gamma).conj().T @ g5,
            rtol=0, atol=0,
        )
        self.assertEqual(sparse_weight, 1.0)
        self.assertEqual(description.operator_term.coefficient, coefficient)
        np.testing.assert_array_equal(
            source_record["indices"], expected_indices[:, ::-1],
        )
        source_values = expected_values.conj().swapaxes(-1, -2)
        np.testing.assert_allclose(
            source_record["values"], source_values, rtol=3e-13, atol=3e-13,
        )


if __name__ == "__main__":
    unittest.main()
