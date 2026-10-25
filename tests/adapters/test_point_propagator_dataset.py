"""Module tests split from test_point_propagator_dataset.py; shared numerical fixtures stay unchanged."""

from test_point_propagator_dataset import (
    DatasetPropagator,
    DatasetSpec,
    _PointPropagatorDatasetFixture,
    _TAILS,
    _axis,
    _direct_block,
    _file_values,
    _logical_shape,
    _point_spec,
    _publish,
    _random,
    np,
    patch,
)


class PointPropagatorDatasetTests(_PointPropagatorDatasetFixture):
    def test_point_sector_axes_labels_and_side_specific_prefixes(self):
        cases = (
            ("psv", False, ("source_time", "sink_time", "sink_point", "sink_spin",
                             "sink_color", "source_mode", "source_spin"),
             {"usedNp": 1, "usedNe": 1}),
            ("vsp", True, ("relative_sink_time", "sink_mode", "source_point", "source_color",
                            "source_time", "source_spin", "sink_spin"),
             {"usedNp": 1, "usedNe": 2}),
            ("psp", False, ("sink_time", "source_time", "source_point", "source_color",
                             "sink_spin", "sink_point", "source_spin", "sink_color"),
             {"usedNp": (2, 1)}),
        )
        for kind, relative, order, selection in cases:
            with self.subTest(kind=kind):
                spec, _logical, values, propagator, reader = self.publish_sector(
                    kind, relative=relative, order=order, seed=22 + len(self.readers))
                sink, source = 2, 1
                with patch.object(reader, "read", side_effect=AssertionError("describe performed I/O")):
                    metadata = propagator.describe_block(sink, source, kind, **selection)
                expected = _direct_block(spec, values, sink, source, **selection)
                self.assertEqual(metadata.axes, _TAILS[kind])
                self.assertEqual(metadata.shape, expected.shape)
                np.testing.assert_array_equal(
                    propagator.block(sink, source, kind, **selection), expected)
                expected_points = {
                    name: (spec.document["bases"][name]["data_id"],
                           tuple(spec.coordinates(name)))
                    for name in ("sink_point", "source_point") if name in spec.axis_names
                }
                self.assertEqual(propagator.point_bindings, expected_points)
                for identity, labels in expected_points.values():
                    self.assertTrue(identity.startswith("actual-"))
                    self.assertTrue(all(type(label) is int for label in labels))
                self.assertIn(reader.data_id, metadata.selection_identity)
                for identity, _labels in expected_points.values():
                    self.assertIn(identity, metadata.selection_identity)

        spec, _logical, values, propagator, _reader = self.publish_sector(
            "psp", order=("source_time", "sink_time", "sink_spin", "source_point",
                          "source_color", "sink_point", "sink_color", "source_spin"),
            seed=44)
        exact = {"sink_point": (1, 0), "source_point": (1,)}
        np.testing.assert_array_equal(
            propagator.block(1, 2, "psp", point_indices=exact),
            _direct_block(spec, values, 1, 2, point_indices=exact))

    def test_time_selection_reads_physical_pairs_on_demand(self):
        spec, _logical, _values, propagator, reader = self.publish_sector(
            "psv", relative=True, order=("source_time", "relative_sink_time", "sink_spin",
                                         "source_spin", "sink_point", "sink_color", "source_mode"),
            seed=73)
        requested = tuple((sink, source)
                          for source in range(propagator.Lt)
                          for sink in range(propagator.Lt))
        read = reader.read
        requested_coordinates = []

        def track(**kwargs):
            requested_coordinates.append((kwargs["source_time"], kwargs[propagator.sink_time_axis]))
            return read(**kwargs)

        with patch.object(reader, "read", side_effect=track):
            observed = tuple(propagator.iter_blocks(range(propagator.Lt), range(propagator.Lt), "psv"))
        self.assertEqual(tuple((sink, source) for sink, source, _block in observed), requested)
        self.assertEqual(len(requested_coordinates), propagator.Lt ** 2)
        expected_stored = tuple(
            (source, (sink - source) % propagator.Lt) for sink, source in requested
        )
        self.assertEqual(tuple(requested_coordinates), expected_stored)
        slab = propagator.read_slab((2, 0), 1, "psv", usedNe=1, usedNp=1)
        self.assertEqual(slab.shape, (2, 4, 4, 1, 3, 1))
        np.testing.assert_array_equal(
            slab,
            np.stack([propagator.block(sink, 1, "psv", usedNe=1, usedNp=1)
                      for sink in (2, 0)]),
        )
        self.assertEqual(spec.document["conventions"]["time_convention"], "relative")

        explicit_spec = _point_spec("psv", lt=3, time_convention="relative")
        explicit_values = _file_values(explicit_spec, _random(_logical_shape(explicit_spec), 74))
        explicit_reader = _publish(self.temp.name, "explicit-relative", explicit_spec, explicit_values)
        self.readers.append(explicit_reader)
        self.addCleanup(explicit_reader.close)
        explicit = DatasetPropagator(explicit_reader)
        self.assertEqual(explicit.sink_time_axis, "sink_time")
        np.testing.assert_array_equal(
            explicit.block(2, 1, "psv"),
            _direct_block(explicit_spec, explicit_values, 2, 1),
        )

    def test_borrowed_prefix_and_owned_gather_and_damaged_block(self):
        spec, _logical, _values, propagator, reader = self.publish_sector(
            "psv", lt=2, order=("source_time", "sink_time", "sink_spin", "source_spin",
                                 "sink_point", "sink_color", "source_mode"), seed=89)
        borrowed = propagator.block(1, 0, "psv", usedNe=1, usedNp=1)
        self.assertFalse(borrowed.flags.writeable)
        self.assertFalse(borrowed.flags.owndata)
        owned = propagator.block(1, 0, "psv", usedNe=1,
                                 point_indices={"sink_point": (1, 0)})
        self.assertTrue(owned.flags.owndata)
        self.assertTrue(np.isfinite(owned).all())
        expected_shape = (4, 4, 2, 3, 1)
        with patch.object(reader, "read", return_value=np.zeros(expected_shape, dtype=np.float64)):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                propagator.block(1, 0, "psv")

        wrong = spec.document
        wrong["bases"]["sink_point"]["data_id"] = None
        invalid = DatasetSpec(wrong)
        with self.assertRaisesRegex(ValueError, "known actual basis"):
            DatasetPropagator(type("Header", (), {"spec": invalid, "data_id": "header-only"})())

    def test_point_sectors_reject_unknown_conventions_and_keep_vsv_contract(self):
        for field, value in (("spin_basis", "unknown-spin"),
                             ("temporal_boundary", None),
                             ("normalization", "unknown-normalization")):
            doc = _point_spec("psv", lt=2).document
            doc["conventions"][field] = value
            spec = DatasetSpec(doc)
            with self.subTest(field=field), self.assertRaises(ValueError):
                DatasetPropagator(type("Header", (), {"spec": spec, "data_id": "header"})())

        bad_color = _point_spec("psv", lt=2).document
        bad_color_axis = next(axis for axis in bad_color["axes"] if axis["name"] == "sink_color")
        bad_color_axis["coordinates"] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, "color axes"):
            DatasetPropagator(type("Header", (), {
                "spec": DatasetSpec(bad_color), "data_id": "bad-color"
            })())

        spec = _point_spec("psv", lt=2).document
        spec["kind"] = "vsv"
        spec["axes"] = [
            _axis("source_time", (0, 1)), _axis("sink_time", (1, 0)),
            _axis("sink_spin", (0, 1, 2, 3)), _axis("source_spin", (0, 1, 2, 3)),
            _axis("sink_mode", (7, 2)), _axis("source_mode", (9, 4)),
        ]
        spec["bases"] = {
            "sink_mode": {"data_id": "actual-mode-basis", "indices": [7, 2]},
            "source_mode": {"data_id": "actual-mode-basis", "indices": [9, 4]},
        }
        spec = DatasetSpec(spec)
        self.assertEqual(DatasetPropagator(type("Header", (), {
            "spec": spec, "data_id": "legacy-vsv"
        })()).basis_bindings["sink_mode"], ("actual-mode-basis", (7, 2)))
