"""Rectangular P/P datasets retain both actual point endpoint bindings."""

from contextlib import ExitStack
import unittest

import numpy as np

import test_elemental_point_production as production_tests
from production.elemental.test_elemental import point_basis, reference_matrix
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion.physical_inputs import (
    _compare_propagator_endpoint,
)
from lqcdweave.contraction.expansion.sampling_binding import (
    describe_point_pool_metadata,
    expand_explicit_point_pools,
)
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm


class IndependentPointDatasetBindingTests(unittest.TestCase):
    def setUp(self):
        helper = production_tests.PointElementalProductionTests()
        helper.setUp()
        self.helper = helper
        self.addCleanup(helper.doCleanups)
        self._opened = ExitStack()
        self.addCleanup(self._opened.close)

    def _open(self, data_id):
        return self._opened.enter_context(self.helper.index.open(data_id))

    def _publish_right_points(self):
        helper = self.helper
        sizes = np.asarray(helper.sizes[:3], dtype=np.int64)
        # Select the +x destinations of left slots 0 and 2. The right table has
        # a different actual identity, labels, size, and point ordering.
        values = []
        for time in range(helper.sizes[3]):
            left = helper.points[time]
            right = (left[[0, 2]] + np.asarray((1, 0, 0))) % sizes
            values.append(right)
        return helper.save_points(
            "independent-right-points", np.asarray(values, dtype=np.int64), (71, 4),
        )

    def _publish_rectangular_p2p(self, right_points_id):
        helper = self.helper
        left_labels = list(helper.labels)
        right_labels = [71, 4]
        time_count, left_count, right_count = helper.sizes[3], len(left_labels), len(right_labels)
        stored = np.zeros((time_count, 1, 1, left_count, 3, 3), dtype=np.complex128)
        displacement = np.asarray((1, 0, 0), dtype=np.int64)
        extent = np.asarray(helper.sizes[:3], dtype=np.int64)

        for time in range(time_count):
            left = helper.points[time]
            right = (left[[0, 2]] + displacement) % extent
            matrix = reference_matrix(
                helper.links, (0,), time, helper.momenta[0], phase="anchor",
            )
            left_projector = point_basis(left, helper.sizes)
            right_projector = point_basis(right, helper.sizes)
            projected = left_projector.conj().T @ matrix @ right_projector
            projected = projected.reshape(left_count, 3, right_count, 3)
            right_lookup = {tuple(xyz): index for index, xyz in enumerate(right)}
            for row, xyz in enumerate(left):
                target = tuple((xyz + displacement) % extent)
                column = right_lookup.get(target)
                if column is not None:
                    stored[time, 0, 0, row] = projected[row, :, column, :]

        document = dict(
            schema=1,
            kind="elemental_p2p_row_sparse",
            configuration=helper.cfg,
            axes=[
                dict(name="time", coordinates=list(range(time_count))),
                dict(name="component", coordinates=[dict(path=[0])]),
                dict(name="momentum", coordinates=[list(helper.momenta[0])]),
                dict(name="point_left", coordinates=left_labels),
                dict(name="color_left", coordinates=[0, 1, 2]),
                dict(name="color_right", coordinates=[0, 1, 2]),
            ],
            dtype="complex128",
            bases=dict(point_left=dict(data_id=helper.points_id, indices=left_labels)),
            definition=dict(
                gauge_id=helper.gauge_id,
                path_boundary="periodic",
                point_right_id=right_points_id,
                point_right_selection="explicit_ordered_labels",
                point_right_labels=right_labels,
            ),
            conventions=dict(
                normalization="unscaled",
                phase="anchor",
                point_pair_layout="left-point-row-slots-v1",
            ),
            parents=[helper.gauge_id, helper.points_id, right_points_id],
        )
        return helper.save(
            "independent-rectangular-p2p", DatasetSpec(document), stored,
        )

    def _make_adapter(self):
        right_id = self._publish_right_points()
        p2p_id = self._publish_rectangular_p2p(right_id)
        left_reader = self._open(self.helper.points_id)
        right_reader = self._open(right_id)
        p2p_reader = self._open(p2p_id)
        adapter = DatasetElemental(
            {"p2p": p2p_reader},
            point_dataset=left_reader,
            point_right_dataset=right_reader,
        )
        return adapter, left_reader, right_reader

    def test_rectangular_p2p_matches_independent_projector_oracle(self):
        adapter, left_reader, right_reader = self._make_adapter()
        component = adapter.components[0]
        labels_left = tuple(left_reader.spec.coordinates("point"))
        labels_right = tuple(right_reader.spec.coordinates("point"))
        self.assertNotEqual(left_reader.data_id, right_reader.data_id)
        self.assertNotEqual(labels_left, labels_right)

        for time in range(adapter.Lt):
            with self.subTest(time=time):
                left = self.helper.points[time]
                right = np.stack((
                    (left[0] + (1, 0, 0)) % np.asarray(self.helper.sizes[:3]),
                    (left[2] + (1, 0, 0)) % np.asarray(self.helper.sizes[:3]),
                ))
                full_site = reference_matrix(
                    self.helper.links, (0,), time, self.helper.momenta[0], phase="anchor",
                )
                expected = point_basis(left, self.helper.sizes).conj().T @ full_site
                expected = expected @ point_basis(right, self.helper.sizes)
                expected = expected.reshape(3, 3, 2, 3)
                right_lookup = {tuple(xyz): index for index, xyz in enumerate(right)}
                expected_pairs = []
                for row, xyz in enumerate(left):
                    column = right_lookup.get(tuple((xyz + (1, 0, 0)) % self.helper.sizes[:3]))
                    if column is not None:
                        expected_pairs.append((row, column))

                metadata = adapter.describe_block(component, 0, time, "p2p")
                self.assertEqual(metadata.point_shape, (3, 2))
                self.assertEqual(metadata.shape, (3, 3, 2, 3))
                self.assertEqual(dict(metadata.axis_bindings), {
                    "point_left": (left_reader.data_id, labels_left),
                    "point_right": (right_reader.data_id, labels_right),
                })
                sparse = adapter.block(component, 0, time, "p2p")
                np.testing.assert_array_equal(
                    sparse["indices"], np.asarray(expected_pairs, dtype=np.int64),
                )
                np.testing.assert_allclose(
                    sparse["values"],
                    np.asarray([expected[row, :, column, :]
                                for row, column in expected_pairs]),
                    atol=2e-12, rtol=2e-12,
                )

    def test_each_propagator_point_endpoint_mismatch_is_rejected_by_metadata_preflight(self):
        adapter, left_reader, right_reader = self._make_adapter()
        component = adapter.components[0]
        operator = Operator.from_terms("rectangular-current", (
            OperatorTerm(0, component, 0, 1.0),
        ))
        factor = Meson(adapter, operator, phase_convention="anchor").describe_factors(
            0, "p2p",
        )[0]
        left_binding = (left_reader.data_id,
                        tuple(self.helper.labels))
        right_binding = (right_reader.data_id, (71, 4))

        def propagator_metadata(source_binding, sink_binding):
            return TensorAxisMetadata(
                ("sink_spin", "source_spin", "sink_point", "sink_color",
                 "source_point", "source_color"),
                (4, 4, 2, 3, 3, 3), np.dtype("complex128"), "actual-psp",
                "endpoint-check",
                axis_bindings={
                    "sink_point": sink_binding,
                    "source_point": source_binding,
                },
            )

        correct = propagator_metadata(left_binding, right_binding)
        self.assertEqual(
            _compare_propagator_endpoint(correct, "psp", "source", factor, "field")["status"],
            "verified",
        )
        self.assertEqual(
            _compare_propagator_endpoint(correct, "psp", "sink", factor, "bar")["status"],
            "verified",
        )
        for endpoint, vertex_endpoint, source_binding, sink_binding in (
            ("source", "field", ("wrong-left-id", left_binding[1]), right_binding),
            ("sink", "bar", left_binding, ("wrong-right-id", right_binding[1])),
        ):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "disagree"):
                _compare_propagator_endpoint(
                    propagator_metadata(source_binding, sink_binding),
                    "psp", endpoint, factor, vertex_endpoint,
                )

    def test_independent_pool_extents_follow_each_logical_endpoint_after_dagger(self):
        adapter, _left_reader, _right_reader = self._make_adapter()
        component = adapter.components[0]
        left_pool = SamplingPoolSpec("left-sample", 12, 3)
        right_pool = SamplingPoolSpec("right-sample", 12, 2)
        pool_ids = {
            (component, 0, 0, "field"): left_pool.group_id,
            (component, 0, 0, "bar"): right_pool.group_id,
        }
        adapter.sampling_pool_ids.update(pool_ids)
        adapter.sampling_groups.update({left_pool.group_id: left_pool,
                                        right_pool.group_id: right_pool})

        for source in (False, True):
            meson = Meson(adapter, Operator.from_terms("rectangular-current", (
                OperatorTerm(0, component, 0, 1.0),
            )), source=source, phase_convention="anchor")
            descriptor = meson.describe_factors(0, "p2p")[0]
            point_legs, pools = describe_point_pool_metadata(
                ((descriptor,),), ({"left": "p", "right": "p"},),
            )
            self.assertEqual(set(pools), {left_pool.group_id, right_pool.group_id})
            if source:
                self.assertEqual(point_legs[(0, "field")], right_pool.group_id)
                self.assertEqual(point_legs[(0, "bar")], left_pool.group_id)
            else:
                self.assertEqual(point_legs[(0, "field")], left_pool.group_id)
                self.assertEqual(point_legs[(0, "bar")], right_pool.group_id)
            scenes = expand_explicit_point_pools(
                point_legs, pools, usedNp=None, groups_independent=True,
            )
            self.assertEqual(len(scenes), 1)

    def test_legacy_same_pool_definition_and_constructor_remain_unchanged(self):
        result = self.helper.run_products(
            "same-pool-regression", sectors=("p2p",), basis_id=None,
            usedNp=len(self.helper.labels), phase_convention="anchor",
        )
        p2p = self._open(result["datasets"]["p2p"])
        points = self._open(self.helper.points_id)
        adapter = DatasetElemental({"p2p": p2p}, point_dataset=points)
        definition = p2p.spec.document["definition"]
        self.assertEqual(definition["point_right_id"], self.helper.points_id)
        self.assertEqual(definition["point_right_selection"],
                         "same_ordered_prefix_as_point_left")
        self.assertNotIn("point_right_labels", definition)
        metadata = adapter.describe_block(adapter.components[1], 0, 0, "p2p")
        self.assertEqual(metadata.point_shape, (3, 3))
        self.assertEqual(metadata.axis_bindings["point_left"],
                         metadata.axis_bindings["point_right"])


if __name__ == "__main__":
    unittest.main()
