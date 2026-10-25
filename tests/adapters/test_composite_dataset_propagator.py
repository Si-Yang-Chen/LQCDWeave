"""Module tests split from test_point_propagator_dataset.py; shared numerical fixtures stay unchanged."""

from test_point_propagator_dataset import (
    DatasetPropagator,
    DatasetSpec,
    _CompositeDatasetPropagatorFixture,
    _publish,
    np,
)


class CompositeDatasetPropagatorTests(_CompositeDatasetPropagatorFixture):
    def test_all_public_sectors_delegate_time_and_selection_per_dataset(self):
        f = self.fixture
        basis_id = f.basis()
        points_id, _ = f.points("composite-propagator-points", labels=(9, 3))
        result = f.run_propagators(
            "composite-propagator-all", basis_id=basis_id,
            points_id=points_id, products=("VSV", "PSV", "VSP", "PSP"),
            Np=2,
        )
        readers = self._open_products(result)
        propagator = DatasetPropagator(readers)
        second = DatasetPropagator(readers)
        self.assertEqual(propagator.data_identity, second.data_identity)
        self.assertEqual(propagator.Lt, 3)

        for sector in ("vsv", "psv", "vsp", "psp"):
            reader = readers[sector]
            expected = reader.read(source_time=1, sink_time=2)
            actual = propagator.block(2, 1, sector)
            np.testing.assert_array_equal(actual, expected)
            metadata = propagator.describe_block(2, 1, sector)
            self.assertEqual(metadata.shape, actual.shape)
            self.assertEqual(metadata.data_identity,
                             second.describe_block(2, 1, sector).data_identity)
            expected_bindings = {
                axis: (reader.spec.document["bases"][axis]["data_id"],
                       tuple(reader.spec.coordinates(axis)))
                for axis in reader.spec.axis_names
                if axis.endswith("_mode") or axis.endswith("_point")
            }
            self.assertEqual(dict(metadata.axis_bindings), expected_bindings)
            slab = propagator.read_slab((2, 0), 1, sector)
            np.testing.assert_array_equal(slab, np.stack((
                propagator.block(2, 1, sector), propagator.block(0, 1, sector),
            )))
            slab_metadata = propagator.describe_slab((2, 0), 1, sector)
            self.assertEqual(dict(slab_metadata.axis_bindings), expected_bindings)
            self.assertNotIn("coordinate_batch", slab_metadata.axis_bindings)

        selected_axes = {
            "vsv": ({"sink_mode": (1, 0), "source_mode": (0, 1)}, {}),
            "psv": ({"source_mode": (1, 0)}, {"sink_point": (1, 0)}),
            "vsp": ({"sink_mode": (1, 0)}, {"source_point": (1, 0)}),
            "psp": ({}, {"sink_point": (1, 0), "source_point": (0, 1)}),
        }
        for sector, (mode_indices, point_indices) in selected_axes.items():
            reader = readers[sector]
            metadata = propagator.describe_block(
                2, 1, sector, mode_indices=mode_indices or None,
                point_indices=point_indices or None,
            )
            expected = {}
            for axis in reader.spec.axis_names:
                if not (axis.endswith("_mode") or axis.endswith("_point")):
                    continue
                positions = (mode_indices.get(axis) if axis.endswith("_mode")
                             else point_indices.get(axis))
                coordinates = tuple(reader.spec.coordinates(axis))
                labels = coordinates if positions is None else tuple(
                    coordinates[index] for index in positions
                )
                expected[axis] = (
                    reader.spec.document["bases"][axis]["data_id"], labels,
                )
            self.assertEqual(dict(metadata.axis_bindings), expected)
            slab_metadata = propagator.describe_slab(
                (2, 0), 1, sector, mode_indices=mode_indices or None,
                point_indices=point_indices or None,
            )
            self.assertEqual(dict(slab_metadata.axis_bindings), expected)

        self.assertEqual(propagator.block(2, 1, "vsv", usedNe=1).shape[-2:], (1, 1))
        self.assertEqual(propagator.block(2, 1, "psv", usedNe=1, usedNp=1).shape[-3:], (1, 3, 1))
        with self.assertRaisesRegex(KeyError, "no 'vvv' sector"):
            propagator.block(0, 0, "vvv")

    def test_role_validation_allows_distinct_point_pools_but_rejects_same_role_mismatch(self):
        f = self.fixture
        basis_id = f.basis()
        sink_id, _ = f.points("composite-sink-points", labels=(9, 3))
        source_id, _ = f.points("composite-source-points", xs=(1, 0, 1), labels=(4,))
        shared_roles = f.run_propagators(
            "composite-distinct-endpoints", basis_id=basis_id,
            points_snk_id=sink_id, points_src_id=source_id,
            products=("VSV", "PSV", "VSP", "PSP"), Ne=2, Np_snk=2, Np_src=1,
        )
        readers = self._open_products(shared_roles)
        propagator = DatasetPropagator({sector: readers[sector]
            for sector in ("vsv", "psv", "vsp", "psp")})
        self.assertEqual(propagator.point_bindings["psv"]["sink_point"][0], sink_id)
        self.assertEqual(propagator.point_bindings["vsp"]["source_point"][0], source_id)
        self.assertEqual(propagator.point_bindings["psp"]["source_point"][0], source_id)
        self.assertNotEqual(sink_id, source_id)
        # VSP uses the reverse PSV projected on its source point set. It is
        # distinct from the forward PSV projected on the sink point set.
        self.assertNotEqual(readers["vsp"].spec.document["parents"],
                            [readers["psv"].data_id])
        self.assertEqual(propagator.block(1, 0, "vsp").shape[-3:], (2, 1, 3))
        self.assertEqual(propagator.block(1, 0, "psp").shape[-4:], (2, 3, 1, 3))

        other_sink_id, _ = f.points("composite-wrong-sink-points", xs=(1, 0, 1), labels=(6,))
        wrong_role = f.run_propagators(
            "composite-wrong-role", points_snk_id=other_sink_id,
            points_src_id=source_id, products=("PSP",), Np_snk=1, Np_src=1,
        )
        wrong_readers = self._open_products(wrong_role)
        with self.assertRaisesRegex(ValueError, "sink_point endpoints have different actual bases"):
            DatasetPropagator({
                "psv": readers["psv"], "psp": wrong_readers["psp"],
            })

    def test_matching_endpoints_reject_a_different_derived_psv_parent(self):
        f = self.fixture
        basis_id = f.basis()
        points_id, _ = f.points("composite-parent-points")
        result = f.run_propagators(
            "composite-parent", basis_id=basis_id, points_id=points_id,
            products=("PSV", "VSP"),
        )
        readers = self._open_products(result)
        document = readers["vsp"].spec.document
        document["parents"] = ["different-actual-psv"]
        spec = DatasetSpec(document)
        values = np.stack([readers["vsp"].read(source_time=time)
                           for time in spec.coordinates("source_time")])
        wrong = _publish(f.root, "wrong-derived-parent", spec, values)
        self.addCleanup(wrong.close)
        with self.assertRaisesRegex(ValueError, "derived VSP must name its actual PSV parent"):
            DatasetPropagator({"psv": readers["psv"], "vsp": wrong})

    def test_sector_specific_time_views_keep_their_own_coverage(self):
        from lqcdweave.runtime.datasets import DatasetView

        f = self.fixture
        basis_id = f.basis()
        points_id, _ = f.points("composite-coverage-points")
        result = f.run_propagators(
            "composite-coverage", basis_id=basis_id,
            points_id=points_id, products=("VSV", "PSV"),
        )
        readers = self._open_products(result)
        vsv = readers["vsv"]
        document = vsv.spec.document
        for axis in document["axes"]:
            if axis["name"] == "source_time":
                axis["coordinates"] = [0, 1]
            elif axis["name"] == "sink_time":
                axis["coordinates"] = [2, 0]
        view = DatasetView(vsv, DatasetSpec(document),
                           subset_axes=("source_time", "sink_time"))
        propagator = DatasetPropagator({"vsv": view, "psv": readers["psv"]})
        self.assertEqual(propagator.block(2, 1, "vsv").shape[0:2], (4, 4))
        self.assertEqual(propagator.block(2, 1, "psv").shape[0:2], (4, 4))
        with self.assertRaisesRegex(KeyError, "missing sink_time coordinate 1"):
            propagator.block(1, 1, "vsv")
        self.assertEqual(propagator.block(1, 1, "psv").shape[0:2], (4, 4))
