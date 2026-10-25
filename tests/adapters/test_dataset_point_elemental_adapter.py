"""Module tests split from test_dataset_point_elemental_adapter.py; shared numerical fixtures stay unchanged."""

from test_dataset_point_elemental_adapter import (
    DatasetElemental,
    DatasetSpec,
    DatasetView,
    SamplingPoolSpec,
    _DatasetPointElementalAdapterFixture,
    np,
)


class DatasetPointElementalAdapterTests(_DatasetPointElementalAdapterFixture):
    def test_prefix_gather_ownership_and_selected_sparse_record(self):
        component, time, momentum = self.reader.components[1], 1, 0
        reader = self.readers["v2p"]
        borrowed = self.reader.block(component, momentum, time, "v2p")
        full_raw = reader.read(
            time=time,
            component=reader.spec.coordinates("component")[1],
            momentum=reader.spec.coordinates("momentum")[momentum],
        )
        self.assertTrue(np.shares_memory(borrowed, full_raw))

        gathered = self.reader.block(
            component, momentum, time, "v2p",
            mode_indices={"mode_left": (2, 0)},
            point_indices={"point_right": (2, 0)},
        )
        gathered_raw = reader.read(time=time, select={
            "component": (reader.spec.coordinates("component")[1],),
            "momentum": (reader.spec.coordinates("momentum")[momentum],),
            "mode_left": (reader.spec.coordinates("mode_left")[2],
                          reader.spec.coordinates("mode_left")[0]),
            "point_right": (reader.spec.coordinates("point_right")[2],
                            reader.spec.coordinates("point_right")[0]),
        })[0, 0]
        np.testing.assert_array_equal(gathered, gathered_raw)
        self.assertFalse(np.shares_memory(gathered, full_raw))

        slab = self.reader.read_slab(component, momentum, (2, 0), "v2p", usedNe=2, usedNp=2)
        expected_slab = np.stack([
            self.reader.block(component, momentum, t, "v2p", usedNe=2, usedNp=2)
            for t in (2, 0)
        ])
        np.testing.assert_array_equal(slab, expected_slab)
        self.assertFalse(np.shares_memory(slab, expected_slab))

        point_reader = self.readers["p2p"]
        selected_metadata = self.reader.describe_block(
            component, momentum, time, "p2p", point_indices={
                "point_left": (2, 0), "point_right": (1, 0),
            })
        selected = self.reader.block(
            component, momentum, time, "p2p", point_indices={
                "point_left": (2, 0), "point_right": (1, 0),
            })
        expected_selected = self._expected_point_sparse_selection(
            component, momentum, time, (2, 0), (1, 0),
        )
        np.testing.assert_array_equal(
            selected["indices"],
            np.asarray([pair for pair, _ in expected_selected], dtype=np.int64).reshape(-1, 2),
        )
        np.testing.assert_allclose(
            selected["values"], np.asarray([value for _, value in expected_selected]),
            atol=2e-12, rtol=2e-12,
        )
        sparse_record = self.reader.describe_selected_sparse_record(
            component, momentum, time, usedNp=(2, 2),
        )
        selected_prefix = self.reader.block(
            component, momentum, time, "p2p", usedNp=(2, 2),
        )
        prefix_expected = self._expected_point_sparse_selection(
            component, momentum, time, (0, 1), (0, 1),
        )
        self.assertEqual(sparse_record.point_shape, (2, 2))
        self.assertEqual(sparse_record.record_bytes,
                         sparse_record.nnz * (2 * np.dtype("int64").itemsize
                                              + 9 * np.dtype("complex128").itemsize))
        self.assertEqual(sparse_record.nnz, len(prefix_expected))
        self.assertEqual(sparse_record.nnz, len(selected_prefix["indices"]))
        self.assertEqual(sparse_record.selection_identity,
                         self.reader.describe_block(
                             component, momentum, time, "p2p", usedNp=(2, 2),
                         ).selection_identity)
        np.testing.assert_array_equal(
            selected_prefix["indices"],
            np.asarray([pair for pair, _ in prefix_expected], dtype=np.int64).reshape(-1, 2),
        )
        np.testing.assert_allclose(
            selected_prefix["values"], np.asarray([value for _, value in prefix_expected]),
            atol=2e-12, rtol=2e-12,
        )
        self.assertEqual(selected_metadata.point_shape, (2, 2))
        self.assertEqual(selected["indices"].shape, (selected_metadata.nnz_upper, 2))
        self.assertTrue(point_reader.data_id)

    def test_axis_bindings_preserve_actual_ids_and_requested_label_order(self):
        component, time, momentum = self.reader.components[1], 1, 0
        v2p_spec = self.readers["v2p"].spec
        mode_labels = tuple(v2p_spec.coordinates("mode_left"))
        point_labels = tuple(v2p_spec.coordinates("point_right"))
        selection = {"mode_left": (2, 0), "point_right": (2, 0)}
        v2p = self.reader.describe_block(
            component, momentum, time, "v2p",
            mode_indices={"mode_left": selection["mode_left"]},
            point_indices={"point_right": selection["point_right"]},
        )
        self.assertEqual(dict(v2p.axis_bindings), {
            "mode_left": (v2p_spec.document["bases"]["mode_left"]["data_id"],
                          (mode_labels[2], mode_labels[0])),
            "point_right": (v2p_spec.document["bases"]["point_right"]["data_id"],
                            (point_labels[2], point_labels[0])),
        })
        slab = self.reader.describe_slab(
            component, momentum, (2, 0), "v2p",
            mode_indices={"mode_left": selection["mode_left"]},
            point_indices={"point_right": selection["point_right"]},
        )
        self.assertEqual(dict(slab.axis_bindings), dict(v2p.axis_bindings))
        self.assertNotIn("coordinate_batch", slab.axis_bindings)

        p2v_spec = self.readers["p2v"].spec
        p2v_mode_labels = tuple(p2v_spec.coordinates("mode_right"))
        p2v_point_labels = tuple(p2v_spec.coordinates("point_left"))
        p2v_selection = {"mode_right": (1, 0), "point_left": (1, 0)}
        p2v = self.reader.describe_block(
            component, momentum, time, "p2v",
            mode_indices={"mode_right": p2v_selection["mode_right"]},
            point_indices={"point_left": p2v_selection["point_left"]},
        )
        self.assertEqual(dict(p2v.axis_bindings), {
            "mode_right": (p2v_spec.document["bases"]["mode_right"]["data_id"],
                           (p2v_mode_labels[1], p2v_mode_labels[0])),
            "point_left": (p2v_spec.document["bases"]["point_left"]["data_id"],
                            (p2v_point_labels[1], p2v_point_labels[0])),
        })
        p2v_slab = self.reader.describe_slab(
            component, momentum, (2, 0), "p2v",
            mode_indices={"mode_right": p2v_selection["mode_right"]},
            point_indices={"point_left": p2v_selection["point_left"]},
        )
        self.assertEqual(dict(p2v_slab.axis_bindings), dict(p2v.axis_bindings))

        p2p_spec = self.readers["p2p"].spec
        p2p_labels = tuple(p2p_spec.coordinates("point_left"))
        p2p = self.reader.describe_block(
            component, momentum, time, "p2p", point_indices={
                "point_left": (2, 0), "point_right": (1, 0),
            },
        )
        point_id = p2p_spec.document["definition"]["point_right_id"]
        self.assertEqual(dict(p2p.axis_bindings), {
            "point_left": (point_id, (p2p_labels[2], p2p_labels[0])),
            "point_right": (point_id, (p2p_labels[1], p2p_labels[0])),
        })
        self.assertEqual(set(p2p.axis_bindings), {"point_left", "point_right"})

    def test_explicit_sampling_identity_and_unregistered_group_rejection(self):
        component = self.reader.components[0]
        pool_id = "caller-declared-pool"
        group = SamplingPoolSpec(pool_id, spatial_sites=12, sampled_points=3,
                                 sampling_design="explicit-test-binding")
        bound = DatasetElemental(
            self.readers, point_dataset=self.points,
            sampling_pool_ids={(component, 0, 2, "bar"): pool_id},
            sampling_groups={pool_id: group},
        )
        self.assertEqual(bound.point_pool_id(component, 0, 2, "bar"), pool_id)
        self.assertIs(bound.sampling_group(pool_id), group)
        with self.assertRaisesRegex(ValueError, "no registered SamplingPoolSpec"):
            DatasetElemental(
                self.readers, point_dataset=self.points,
                sampling_pool_ids={(component, 0, 2, "bar"): "undeclared"},
            )

    def test_p2p_keeps_zero_valued_coordinate_support_and_rejects_wrong_identity(self):
        helper = self.helper
        source = self.readers["p2p"]
        zeros = np.zeros(source.spec.shape, dtype=np.complex128)
        zero_id = helper.save("zero-p2p", source.spec, zeros)
        with helper.index.open(zero_id) as zero_reader:
            zero_adapter = DatasetElemental({"p2p": zero_reader}, point_dataset=self.points)
            component = zero_adapter.components[0]
            zero_sparse = zero_adapter.block(component, 0, 0, "p2p")
            self.assertEqual(len(zero_sparse["indices"]), 3)
            np.testing.assert_array_equal(zero_sparse["values"], np.zeros((3, 3, 3), complex))
            self.assertEqual(zero_adapter.describe_sparse_record(component, 0, 0).nnz, 3)

        wrong_points = helper.points.copy()
        wrong_points[:, :, 0] = (wrong_points[:, :, 0] + 1) % helper.sizes[0]
        wrong_id = helper.save_points("wrong-p2p-points", wrong_points, helper.labels)
        with helper.index.open(wrong_id) as wrong_reader:
            with self.assertRaisesRegex(ValueError, "actual registered point dataset"):
                DatasetElemental({"p2p": source}, point_dataset=wrong_reader)

    def test_partial_point_view_and_time_coverage_are_checked_by_actual_coordinates(self):
        point_document = self.points.spec.document
        point_labels = point_document["axes"][1]["coordinates"][:2]
        point_document["axes"][1]["coordinates"] = point_labels
        point_view = DatasetView(self.points, DatasetSpec(point_document), subset_axes=("point",))
        prefix_adapter = DatasetElemental({"p2p": self.readers["p2p"]},
                                          point_dataset=point_view)
        component = prefix_adapter.components[0]
        selected = prefix_adapter.block(component, 0, 1, "p2p", usedNp=2)
        self.assertEqual(selected["indices"].shape[1], 2)

        truncated_document = self.points.spec.document
        truncated_document["axes"][0]["coordinates"] = [0, 1]
        truncated_points = self.helper.save(
            "truncated-points", DatasetSpec(truncated_document), self.helper.points[:2],
        )
        p2p_document = self.readers["p2p"].spec.document
        p2p_document["bases"]["point_left"]["data_id"] = truncated_points
        p2p_document["definition"]["point_right_id"] = truncated_points
        p2p_document["parents"] = [p2p_document["definition"]["gauge_id"], truncated_points]
        truncated_spec = DatasetSpec(p2p_document)
        truncated_id = self.helper.save(
            "truncated-p2p-parent", truncated_spec,
            np.zeros(truncated_spec.shape, dtype=np.complex128),
        )
        with self.helper.index.open(truncated_points) as short_points, \
             self.helper.index.open(truncated_id) as short_p2p:
            with self.assertRaisesRegex(ValueError, "full-time ordered XYZ"):
                DatasetElemental({"p2p": short_p2p}, point_dataset=short_points)
