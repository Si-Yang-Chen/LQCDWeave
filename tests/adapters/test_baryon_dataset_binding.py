"""Module tests split from test_baryon_dataset_binding.py; shared numerical fixtures stay unchanged."""

from test_baryon_dataset_binding import (
    DatasetBaryonElemental,
    DatasetSpec,
    DatasetView,
    GaugePath,
    _BaryonDatasetBindingFixture,
    _baryon_spec,
    _publish,
    np,
    patch,
    validate_diagram_input_specs,
)


class BaryonDatasetBindingTests(_BaryonDatasetBindingFixture):
    def test_raw_projected_and_distance_tags_decode_without_flavor_or_axis_reordering(self):
        for product, expected in (
            ("raw_paths", ((GaugePath((0,)), GaugePath((3,)), GaugePath(())),)),
            ("projected", (("projected", 0), ("projected", 1))),
            ("distance", (("distance", (0, 1, 2)), ("distance", (2, 0, 1)))),
        ):
            with self.subTest(product=product):
                _spec, _values, data_id = self._publish_baryon(product)
                reader = self.index.open(data_id)
                self.addCleanup(reader.close)
                with patch.object(reader, "read", side_effect=AssertionError("description read values")):
                    elemental = DatasetBaryonElemental(reader)
                    metadata = elemental.describe_block(elemental.components[0], 0, 1)
                self.assertEqual(tuple(component.key for component in elemental.components), expected)
                self.assertTrue(all(component.flavors == ("q", "q", "q")
                                    for component in elemental.components))
                self.assertEqual(elemental.datasets, {"vvv": reader})
                self.assertEqual(metadata.axes, ("mode0", "mode1", "mode2"))
                self.assertEqual(metadata.data_identity, data_id)
                self.assertEqual(metadata.axis_bindings["mode0"], (self.basis_id, (7, 2)))
                self.assertEqual(metadata.axis_bindings["mode1"], (self.basis_id, (7, 2, 9)))
                self.assertEqual(metadata.axis_bindings["mode2"], (self.basis_id, (7,)))

    def test_leg_specific_prefixes_and_borrowed_dataset_view_are_preserved(self):
        spec, values, data_id = self._publish_baryon()
        reader = self.index.open(data_id)
        self.addCleanup(reader.close)
        elemental = DatasetBaryonElemental(reader)
        component = elemental.components[0]
        metadata = elemental.describe_block(component, 0, 0, usedNe=(1, 2, 1))
        self.assertEqual(metadata.shape, (1, 2, 1))
        self.assertEqual(metadata.axis_bindings["mode0"], (self.basis_id, (7,)))
        self.assertEqual(metadata.axis_bindings["mode1"], (self.basis_id, (7, 2)))
        self.assertEqual(metadata.axis_bindings["mode2"], (self.basis_id, (7,)))
        np.testing.assert_array_equal(
            elemental.block(component, 0, 0, usedNe=(1, 2, 1)),
            values[0, 0, 0, :1, :2, :1],
        )

        document = spec.document
        requested_labels = ((7,), (7, 2), (7,))
        for index, labels in enumerate(requested_labels):
            axis = f"mode{index}"
            for entry in document["axes"]:
                if entry["name"] == axis:
                    entry["coordinates"] = list(labels)
            document["bases"][axis]["indices"] = list(labels)
        requested = DatasetSpec(document)
        view = DatasetView(reader, requested,
                            subset_axes=("mode0", "mode1", "mode2"))
        viewed = DatasetBaryonElemental(view)
        np.testing.assert_array_equal(
            viewed.block(viewed.components[0], 0, 1),
            values[1, 0, 0, :1, :2, :1],
        )
        self.assertFalse(reader._closed)

    def test_unknown_or_inconsistent_component_encoding_is_rejected(self):
        spec = _baryon_spec(self.gauge, self.gauge_id, self.basis,
                            self.basis_id, product="raw_paths").document
        spec["axes"][1]["coordinates"][0]["encoding"] = "projected_row_id_v1"
        invalid = DatasetSpec(spec)
        path = self.root / "invalid-baryon"
        values = np.zeros(invalid.shape, dtype=np.complex128)
        data_id = _publish(self.root, self.index, "invalid-baryon", invalid, values)
        with self.index.open(data_id) as reader:
            with self.assertRaisesRegex(ValueError, "raw baryon components"):
                DatasetBaryonElemental(reader)

    def test_nested_field_views_preserve_producer_spatial_and_color_layout(self):
        mutations = (
            ("basis_leg0", lambda view: view["axes"].pop(2), "canonical field axes"),
            ("basis_leg1", lambda view: view["axes"][-1].update(coordinates=[0, 1]),
             "canonical spatial/color coordinates"),
            ("gauge", lambda view: view["axes"][-1].update(coordinates=[0, 1]),
             "canonical spatial/color coordinates"),
        )
        for role, mutate, message in mutations:
            with self.subTest(role=role, message=message):
                document = _baryon_spec(
                    self.gauge, self.gauge_id, self.basis, self.basis_id,
                ).document
                mutate(document["definition"]["input_views"][role])
                invalid = DatasetSpec(document)
                data_id = _publish(
                    self.root, self.index, f"invalid-view-{role}", invalid,
                    np.zeros(invalid.shape, dtype=np.complex128),
                )
                with self.index.open(data_id) as reader:
                    with self.assertRaisesRegex(ValueError, message):
                        DatasetBaryonElemental(reader)
                with self.assertRaises(ValueError):
                    validate_diagram_input_specs({"baryon": invalid})
