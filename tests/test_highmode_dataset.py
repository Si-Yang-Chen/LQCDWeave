"""High-mode metadata, validation, and read-only planning contracts."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.highmode import (
    HIGHMODE_INPUTS, highmode_dataset_document, select_highmode_input_views,
    validate_highmode_spec,
)
from lqcdweave.planning.overlap import overlap_dataset_document
from lqcdweave.planning.production import propagator_dataset_document


class HighmodePlanningTests(unittest.TestCase):
    def setUp(self):
        cfg = dict(ensemble="e", configuration="c", lattice=[2, 1, 1, 2])
        def axis(name, coordinates):
            return dict(name=name, coordinates=list(coordinates))
        self.basis = DatasetSpec(dict(schema=1, kind="explicit_basis", configuration=cfg,
            axes=[axis("time", range(2)), axis("mode", [8, 3]), axis("z", [0]),
                  axis("y", [0]), axis("x", range(2)), axis("color", range(3))],
            dtype="complex64", bases={}, parents=["gauge"], definition={},
            conventions=dict(normalization="orthonormal")))
        self.points = DatasetSpec(dict(schema=1, kind="spatial_points", configuration=cfg,
            axes=[axis("time", range(2)), axis("point", [9]), axis("xyz", ["x", "y", "z"])],
            dtype="int64", bases={}, parents=[], definition={},
            conventions=dict(representation="coordinate_delta", normalization="unit_delta")))
        _, _, overlap, _ = overlap_dataset_document(self.basis, self.points,
            basis_id="basis", points_id="points")
        common = dict(gauge_id="gauge", basis_id="basis", modes=[8, 3],
            dirac=dict(action="wilson", mass=0.4), solver_tolerance=1e-10,
            temporal_boundary="antiperiodic")
        self.inputs = dict(
            vsv=DatasetSpec(propagator_dataset_document(cfg, range(2), product="VSV", **common)),
            psv=DatasetSpec(propagator_dataset_document(cfg, range(2), product="PSV",
                points_snk_id="points", point_snk_labels=[9], **common)),
            overlap_sink=DatasetSpec(overlap))
        self.ids = {role: "actual-" + role for role in self.inputs}

    def build(self):
        return highmode_dataset_document("PSV", self.inputs, self.ids,
            basis_specs={"basis": self.basis}, point_specs={"points": self.points})

    def _basis_view(self, modes):
        document = self.basis.document
        for axis in document["axes"]:
            if axis["name"] == "mode":
                axis["coordinates"] = list(modes)
        return DatasetSpec(document)

    def _point_view(self, labels):
        document = self.points.document
        for axis in document["axes"]:
            if axis["name"] == "point":
                axis["coordinates"] = list(labels)
        return DatasetSpec(document)

    def _raw_endpoint_specs(self, *, sink_basis_id="basis-sink",
                            source_basis_id="basis-source", point_id="points",
                            sink_modes=(8, 3), source_modes=(8, 3),
                            point_labels=(9, 6)):
        from lqcdweave.planning.production import propagator_dataset_document

        common = dict(gauge_id="gauge", dirac=dict(action="wilson", mass=0.4),
            solver_tolerance=1e-10, temporal_boundary="antiperiodic")
        configuration = self.basis.document["configuration"]
        times = tuple(range(configuration["lattice"][3]))
        return {
            "vsv": DatasetSpec(propagator_dataset_document(
                configuration, times, product="VSV", **common,
                basis_src_id=source_basis_id, modes_src=source_modes,
                basis_snk_id=sink_basis_id, modes_snk=sink_modes)),
            "psv": DatasetSpec(propagator_dataset_document(
                configuration, times, product="PSV", **common,
                basis_src_id=source_basis_id, modes_src=source_modes,
                points_snk_id=point_id, point_snk_labels=point_labels)),
            "vsp": DatasetSpec(propagator_dataset_document(
                configuration, times, product="VSP", **common,
                basis_snk_id=sink_basis_id, modes_snk=sink_modes,
                points_src_id=point_id, point_src_labels=point_labels)),
            "psp": DatasetSpec(propagator_dataset_document(
                configuration, times, product="PSP", **common,
                points_snk_id=point_id, point_snk_labels=point_labels,
                points_src_id=point_id, point_src_labels=point_labels)),
        }

    def _projected_reader(self, product, tag, *, sink_basis_id="basis-sink",
                          source_basis_id="basis-source", point_id="points",
                          sink_modes=(8, 3), source_modes=(8, 3),
                          point_labels=(9, 6), overlap_sink_points=(9, 6),
                          overlap_source_points=(9, 6)):
        raw = self._raw_endpoint_specs(
            sink_basis_id=sink_basis_id, source_basis_id=source_basis_id,
            point_id=point_id, sink_modes=(8, 3), source_modes=(8, 3),
            point_labels=point_labels,
        )
        inputs = {role: raw[role] for role in HIGHMODE_INPUTS[product]
                  if role in raw}
        basis_specs, point_specs = {}, {}
        for side, basis_id, modes, labels in (
                ("sink", sink_basis_id, sink_modes, overlap_sink_points),
                ("source", source_basis_id, source_modes, overlap_source_points)):
            role = "overlap_" + side
            if role not in HIGHMODE_INPUTS[product]:
                continue
            basis_view = self._basis_view(modes)
            point_view = self._point_view(labels)
            _, _, overlap, _ = overlap_dataset_document(
                basis_view, point_view, basis_id=basis_id, points_id=point_id)
            inputs[role] = DatasetSpec(overlap)
            basis_specs[(role, basis_id)] = basis_view
            point_specs[(role, point_id)] = point_view

        selected = select_highmode_input_views(product, inputs)
        input_ids = {role: f"{tag}-{role}" for role in selected}
        document = highmode_dataset_document(
            product, selected, input_ids,
            basis_specs=basis_specs, point_specs=point_specs)
        return SimpleNamespace(
            spec=DatasetSpec(document), data_id=f"actual-{tag}")

    def test_sparse_ordered_endpoint_contracts_survive_derivation(self):
        from lqcdweave.contracts.time_selection import (
            TimeSelectionRecord, bind_time_selections, dataset_time_selection)
        selection = TimeSelectionRecord(2, (1,))
        def compact(spec, names):
            document = spec.document
            for axis in document["axes"]:
                if axis["name"] in names:
                    axis["coordinates"] = list(selection.times)
            return DatasetSpec(bind_time_selections(document, {name: selection for name in names}))
        self.basis = compact(self.basis, ("time",))
        self.points = compact(self.points, ("time",))
        _, _, overlap, times = overlap_dataset_document(self.basis, self.points,
            basis_id="basis", points_id="points", selection=selection)
        self.inputs["overlap_sink"] = DatasetSpec(overlap)
        for role in ("vsv", "psv"):
            self.inputs[role] = compact(self.inputs[role], ("source_time", "sink_time"))
        output = DatasetSpec(self.build())
        self.assertEqual(times, (1,))
        self.assertEqual(output.shape[:2], (1, 1))
        self.assertEqual(dataset_time_selection(output, "source_time"), selection)
        validate_highmode_spec(output)
        changed = self.inputs["psv"].document
        changed["conventions"]["time_coordinates"]["selections"]["sink_time"]["times"] = [0]
        self.inputs["psv"] = DatasetSpec(changed)
        with self.assertRaisesRegex(ValueError, "time selections"):
            self.build()

    def test_records_actual_roles_projector_and_dtype(self):
        output = self.build()
        self.assertEqual(output["dtype"], "complex128")
        self.assertEqual(output["parents"], sorted(self.ids.values()))
        projection = output["definition"]["high_mode_projection"]
        self.assertEqual(output["definition"]["input_roles"], self.ids)
        self.assertEqual(projection["sides"], ["sink"])
        self.assertEqual(output["bases"], self.inputs["psv"].document["bases"])

    def test_rejects_same_shape_wrong_identity_and_nonorthonormal_basis(self):
        doc = self.inputs["psv"].document
        doc["bases"]["sink_point"]["data_id"] = "other-points"
        self.inputs["psv"] = DatasetSpec(doc)
        with self.assertRaisesRegex(ValueError, "actual endpoint"):
            self.build()
        self.setUp()
        doc = self.basis.document
        doc["conventions"]["normalization"] = "unit_norm_only"
        self.basis = DatasetSpec(doc)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.build()

    def test_composite_rejects_raw_and_highmode_point_endpoints(self):
        from lqcdweave.adapters.meson_dataset import DatasetPropagator

        raw = self._raw_endpoint_specs()
        cases = (("sink", "psv"), ("source", "vsp"))
        for side, sector in cases:
            with self.subTest(side=side):
                raw_reader = SimpleNamespace(
                    spec=raw[sector], data_id=f"actual-raw-{sector}")
                projected_psp = self._projected_reader("psp", f"mixed-{side}")
                with self.assertRaisesRegex(
                        ValueError, "different raw or high-mode projections"):
                    DatasetPropagator({sector: raw_reader, "psp": projected_psp})

    def test_composite_rejects_different_highmode_projectors(self):
        from lqcdweave.adapters.meson_dataset import DatasetPropagator

        cases = (
            ("sink actual basis ID", "psv",
             dict(sink_basis_id="other-sink")),
            ("sink complete mode set", "psv", dict(sink_modes=(8,))),
            ("source actual basis ID", "vsp",
             dict(source_basis_id="other-source")),
            ("source complete mode set", "vsp",
             dict(source_modes=(8,))),
        )
        for label, peer_sector, psp_options in cases:
            with self.subTest(difference=label):
                peer = self._projected_reader(peer_sector, f"same-{peer_sector}")
                psp = self._projected_reader("psp", "different-psp", **psp_options)
                with self.assertRaisesRegex(ValueError,
                        "different raw or high-mode projections"):
                    DatasetPropagator({peer_sector: peer, "psp": psp})

    def test_composite_accepts_same_q_across_overlap_views_and_output_subset(self):
        from lqcdweave.adapters.meson_dataset import DatasetPropagator

        psv = self._projected_reader("psv", "view-psv",
            sink_modes=(8, 3), overlap_sink_points=(9,))
        psp = self._projected_reader("psp", "view-psp",
            sink_modes=(3, 8), overlap_sink_points=(9, 6))
        self.assertNotEqual(
            psv.spec.document["definition"]["input_roles"]["overlap_sink"],
            psp.spec.document["definition"]["input_roles"]["overlap_sink"])
        psv_overlap = DatasetSpec(
            psv.spec.document["definition"]["input_views"]["overlap_sink"])
        psp_overlap = DatasetSpec(
            psp.spec.document["definition"]["input_views"]["overlap_sink"])
        self.assertEqual(psv_overlap.coordinates("point"), [9])
        self.assertEqual(psp_overlap.coordinates("point"), [9, 6])
        DatasetPropagator({"psv": psv, "psp": psp})

        vsp = self._projected_reader("vsp", "source-view-vsp",
            source_modes=(8, 3), overlap_source_points=(9,))
        psp_source = self._projected_reader("psp", "source-view-psp",
            source_modes=(3, 8), overlap_source_points=(9, 6))
        DatasetPropagator({"vsp": vsp, "psp": psp_source})

        output_view = psp.spec.document
        output_view["bases"]["sink_point"]["indices"] = [6]
        for axis in output_view["axes"]:
            if axis["name"] == "sink_point":
                axis["coordinates"] = [6]
        single_sector_view = SimpleNamespace(
            spec=DatasetSpec(output_view), data_id="actual-psp-output-view")
        DatasetPropagator(single_sector_view)

    def test_output_view_preserves_projector_and_rejects_tampering(self):
        doc = self.build()
        for axis in doc["axes"]:
            if axis["name"] == "source_mode":
                axis["coordinates"] = [3]
        doc["bases"]["source_mode"]["indices"] = [3]
        selected = DatasetSpec(doc)
        inputs, _ = validate_highmode_spec(selected, allow_output_view=True)
        self.assertEqual(inputs["overlap_sink"].document,
                         self.inputs["overlap_sink"].document)
        with self.assertRaisesRegex(ValueError, "declared derivation"):
            validate_highmode_spec(selected)
        for field, value in (("data_id", "other-basis"), ("indices", [8])):
            broken = selected.document
            broken["bases"]["source_mode"][field] = value
            with self.assertRaises(ValueError):
                validate_highmode_spec(DatasetSpec(broken), allow_output_view=True)
        missing = selected.document
        missing["axes"][-1]["coordinates"] = [99]
        missing["bases"]["source_mode"]["indices"] = [99]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_highmode_spec(DatasetSpec(missing), allow_output_view=True)

    def test_rejects_changed_solver_and_order(self):
        doc = self.inputs["psv"].document
        doc["definition"]["solver_tolerance"] = 1e-5
        self.inputs["psv"] = DatasetSpec(doc)
        with self.assertRaisesRegex(ValueError, "Dirac"):
            self.build()
        self.setUp()
        doc = self.inputs["psv"].document
        doc["axes"][-1]["coordinates"] = [3, 8]
        doc["bases"]["source_mode"]["indices"] = [3, 8]
        self.inputs["psv"] = DatasetSpec(doc)
        with self.assertRaisesRegex(ValueError, "actual endpoint"):
            self.build()

    def test_selects_ordered_raw_views_without_changing_projector(self):
        original = self.inputs["overlap_sink"].document
        doc = self.inputs["vsv"].document
        doc["axes"][-2]["coordinates"] = [99, 3, 8]
        doc["bases"]["sink_mode"]["indices"] = [99, 3, 8]
        self.inputs["vsv"] = DatasetSpec(doc)
        selected = select_highmode_input_views("PSV", self.inputs)
        self.assertEqual(selected["vsv"].coordinates("sink_mode"), [8, 3])
        self.assertEqual(self.inputs["vsv"].coordinates("sink_mode"), [99, 3, 8])
        self.assertEqual(selected["overlap_sink"].document, original)
        output = highmode_dataset_document("PSV", selected, self.ids,
            basis_specs={"basis": self.basis}, point_specs={"points": self.points})
        recorded = output["definition"]["input_views"]
        self.assertEqual(recorded["vsv"], selected["vsv"].document)
        doc["bases"]["sink_mode"]["data_id"] = "other-basis"
        self.inputs["vsv"] = DatasetSpec(doc)
        with self.assertRaisesRegex(ValueError, "actual basis identity"):
            select_highmode_input_views("PSV", self.inputs)

    def test_registered_planner_previews_missing_outputs(self):
        from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
        from lqcdweave.production.perambulator.highmode_planning import plan_indexed_highmode_propagators
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = DatasetIndex(root / "index")

            def publish(name, spec, value):
                path = root / name
                with DatasetWriter(path, spec, provenance={"test": "highmode"}) as writer:
                    for row in range(spec.shape[0]):
                        writer.write(row, np.full(spec.shape[1:], value,
                                                  dtype=spec.document["dtype"]))
                    identity = writer.publish()
                index.register(path)
                return identity

            gauge_spec = DatasetSpec(dict(schema=1, kind="gauge",
                configuration=self.basis.document["configuration"],
                axes=[dict(name="time", coordinates=[0, 1])], dtype="complex128",
                bases={}, parents=[], definition={}, conventions={}))
            gid = publish("gauge", gauge_spec, 0)
            bid = publish("basis", self.basis, 0)
            pid = publish("points", self.points, 0)
            _, _, overlap, _ = overlap_dataset_document(self.basis, self.points,
                basis_id=bid, points_id=pid)
            oid = publish("overlap", DatasetSpec(overlap), 0)
            raw = {}
            for role in ("vsv", "psv"):
                doc = self.inputs[role].document
                for binding in doc["bases"].values():
                    binding["data_id"] = bid if binding["data_id"] == "basis" else pid
                doc["definition"]["gauge_id"] = gid
                doc["parents"] = [bid if parent == "basis" else
                                  pid if parent == "points" else gid
                                  for parent in doc["parents"]]
                raw[role] = publish(role, DatasetSpec(doc), 2 if role == "psv" else 1)

            planned = plan_indexed_highmode_propagators(index,
                vsv_id=raw["vsv"], psv_id=raw["psv"], overlap_sink_id=oid)
            self.assertEqual(planned["graph"]["stages"]["psv"]["action"], "compute")
            self.assertEqual(planned["output_specs"]["psv"].document["kind"], "psv")
            self.assertFalse((root / "out").exists())
