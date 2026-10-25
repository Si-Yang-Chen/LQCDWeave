"""Actual point-set invalidation, pure point sources, and independent inverses."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from test_wilson_pipeline import random_gauge, stencil
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.adapters.production_dataset import DatasetPointHandle
from lqcdweave.planning.production import vsp_dataset_document
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter, inspect_dataset
from lqcdweave.production.wilson_reference import plan_indexed_wilson_propagators
from fixtures.c_group_production import publish_perambulators


class IndexedPointProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.links = random_gauge((2, 1, 1, 3), seed=94)
        self.configuration = dict(ensemble="test", configuration="cfg", lattice=[2, 1, 1, 3])
        axes = [dict(name=name, coordinates=list(range(size))) for name, size in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"), self.links.shape)]
        spec = DatasetSpec(dict(schema=1, kind="gauge", configuration=self.configuration, axes=axes,
            dtype="complex128", bases={}, parents=[], definition={}, conventions=dict(boundary="periodic")))
        self.gauge_id = self.publish("gauge", spec, {time: self.links[time] for time in range(3)})

    def publish(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance=dict(test="independent")) as writer:
            for position, time in enumerate(spec.coordinates("time")):
                writer.write(position, np.asarray(rows[time], dtype=spec.document["dtype"]))
            identity = writer.publish()
        self.index.register(directory)
        return identity

    def points(self, name, xs=(0, 1, 0), *, labels=(9,), conventions=None, configuration=None,
               kind="spatial_points"):
        # Storage is reordered. The point handle must load physical time, not
        # stored row position; labels never encode the actual XYZ coordinates.
        spec = DatasetSpec(dict(schema=1, kind=kind,
            configuration=self.configuration if configuration is None else configuration,
            axes=[dict(name="time", coordinates=[2, 0, 1]),
                  dict(name="point", coordinates=list(labels)),
                  dict(name="xyz", coordinates=["x", "y", "z"])], dtype="int64", bases={}, parents=[],
            definition=dict(selection="explicit"), conventions=conventions or dict(
                representation="coordinate_delta", normalization="unit_delta")))
        rows = {time: [[x, 0, 0] for x in ([xs[time]] if len(labels) == 1 else [xs[time], 1-xs[time]])]
                for time in range(3)}
        return self.publish(name, spec, rows), rows

    def basis(self, name="basis", modes=(8, 3)):
        values = np.zeros((len(modes), 1, 1, 2, 3), dtype=np.complex128)
        values[0, 0, 0, 0, 0] = .8 + .2j
        values[0, 0, 0, 1, 2] = .1 + .3j
        values[1, 0, 0, 1, 1] = .5 + .7j
        values[1, 0, 0, 0, 2] = .2 - .1j
        for mode in range(len(modes)):
            values[mode] /= np.linalg.norm(values[mode])
        spec = DatasetSpec(dict(schema=1, kind="explicit_basis", configuration=self.configuration,
            axes=[dict(name="time", coordinates=[0, 1, 2]),
                  dict(name="mode", coordinates=list(modes)),
                  dict(name="z", coordinates=[0]), dict(name="y", coordinates=[0]),
                  dict(name="x", coordinates=[0, 1]), dict(name="color", coordinates=[0, 1, 2])],
            dtype="complex128", bases={}, parents=[self.gauge_id],
            definition=dict(realization="asymmetric_complex_test"),
            conventions=dict(normalization="orthonormal")))
        return self.publish(name, spec, {time: values for time in range(3)})

    def run_propagators(self, output, **options):
        """Publish fresh test products through public plans and row writers."""
        identities = publish_perambulators(self.index, self.root, self.gauge_id,
            output_name=output, **options)
        return {"datasets": identities}

    def reference(self, ids, rows=None, *, rows_snk=None, rows_src=None,
                  point_snk_id=None, point_src_id=None, psv_parent_id=None,
                  ne=2, np_count=1, np_snk=None, np_src=None,
                  mass=.4, boundary=-1):
        shape = (3, 1, 1, 2, 4, 3)
        dimension = int(np.prod(shape))
        delta = np.eye(dimension)
        matrix = np.column_stack([stencil(self.links, delta[:, i].reshape(shape), mass, boundary).ravel()
                                  for i in range(dimension)])
        inverse = np.linalg.inv(matrix)
        if rows is not None:
            rows_snk = rows if rows_snk is None else rows_snk
            rows_src = rows if rows_src is None else rows_src
        rows_snk = rows_snk if rows_snk is not None else rows_src
        rows_src = rows_src if rows_src is not None else rows_snk
        self.assertIsNotNone(rows_snk)
        self.assertIsNotNone(rows_src)
        np_snk = np_count if np_snk is None else np_snk
        np_src = np_count if np_src is None else np_src

        def point_sources(point_rows, count):
            result = []
            for time in range(3):
                point = np.zeros((dimension, 4*count*3), complex)
                for spin in range(4):
                    for position, xyz in enumerate(point_rows[time][:count]):
                        for color in range(3):
                            x, y, z = xyz
                            point[np.ravel_multi_index((time, z, y, x, spin, color), shape),
                                  (spin*count+position)*3+color] = 1
                result.append(point)
            return result

        sink_points = point_sources(rows_snk, np_snk)
        source_points = point_sources(rows_src, np_src)
        vectors = []
        if "basis" in ids:
            with self.index.open(ids["basis"]) as basis:
                for time in range(3):
                    vector = np.zeros((dimension, 4*ne), complex)
                    data = basis.read(time=time)[:ne]
                    for spin in range(4):
                        for mode in range(ne):
                            field = np.zeros(shape, complex)
                            field[time, ..., spin, :] = data[mode]
                            vector[:, spin*ne+mode] = field.ravel()
                    vectors.append(vector)
        for product in ("psv", "psp", "vsp"):
            if product not in ids:
                continue
            with self.index.open(ids[product]) as result:
                self.assertEqual(result.spec.coordinates("source_time"), [0, 1, 2])
                self.assertEqual(result.spec.coordinates("sink_time"), [0, 1, 2])
                if product == "vsp":
                    self.assertEqual(result.spec.document["bases"]["sink_mode"]["data_id"], ids["basis"])
                    expected_source_id = point_src_id or ids.get("points_src", ids.get("points"))
                    self.assertEqual(result.spec.document["bases"]["source_point"]["data_id"],
                                     expected_source_id)
                elif product == "psv":
                    expected_sink_id = point_snk_id or ids.get("points_snk", ids.get("points"))
                    self.assertEqual(result.spec.document["bases"]["sink_point"]["data_id"],
                                     expected_sink_id)
                else:
                    expected_sink_id = point_snk_id or ids.get("points_snk", ids.get("points"))
                    expected_source_id = point_src_id or ids.get("points_src", ids.get("points"))
                    self.assertEqual(result.spec.document["bases"]["sink_point"]["data_id"],
                                     expected_sink_id)
                    self.assertEqual(result.spec.document["bases"]["source_point"]["data_id"],
                                     expected_source_id)
                for source in range(3):
                    for sink in range(3):
                        if product == "psv":
                            value = sink_points[sink].conj().T @ inverse @ vectors[source]
                            expected = value.reshape(4, np_snk, 3, 4, ne).transpose(0, 3, 1, 2, 4)
                        elif product == "vsp":
                            value = vectors[sink].conj().T @ inverse @ source_points[source]
                            expected = value.reshape(4, ne, 4, np_src, 3).transpose(0, 2, 1, 3, 4)
                        else:
                            value = sink_points[sink].conj().T @ inverse @ source_points[source]
                            expected = value.reshape(4, np_snk, 3, 4, np_src, 3).transpose(0, 3, 1, 2, 4, 5)
                        if product == "vsp":
                            parent_id = psv_parent_id or ids.get("psv_vsp", ids["psv"])
                            self.assertEqual(result.spec.document["parents"], [parent_id])
                        np.testing.assert_allclose(result.read(source_time=source, sink_time=sink),
                                                   expected, atol=1e-11)

    def test_pure_psp_has_no_laplace_dependency_and_matches_independent_inverse(self):
        point_id, rows = self.points("points")
        before = self.index.path.read_bytes()
        with patch.object(LaplaceEigenvectorGenerator, "calc", side_effect=AssertionError("Laplace solve")):
            graph = plan_indexed_wilson_propagators(self.index, self.gauge_id,
                points_id=point_id, products=("PSP",))
        self.assertEqual(set(graph["stages"]), {"gauge", "points", "psp"})
        self.assertEqual(graph["stages"]["psp"]["action"], "compute")
        self.assertEqual(self.index.path.read_bytes(), before)
        with patch.object(LaplaceEigenvectorGenerator, "calc", side_effect=AssertionError("Laplace solve")):
            result = self.run_propagators("pure", points_id=point_id, products=("PSP",))
        self.assertNotIn("basis", result["datasets"])
        with self.index.open(result["datasets"]["psp"]) as psp:
            self.assertEqual(psp.spec.document["parents"], [self.gauge_id, point_id])
        self.reference(result["datasets"], rows)

    def test_gamma5_derived_vsp_matches_independent_inverse_for_both_boundaries(self):
        basis_id = self.basis()
        point_id, rows = self.points("vsp-points", xs=(0, 1, 0))
        preview = plan_indexed_wilson_propagators(self.index, self.gauge_id,
            basis_id=basis_id, points_id=point_id, products=("VSP",))
        self.assertEqual(set(preview["stages"]), {"gauge", "basis", "points", "psv", "vsp"})
        self.assertEqual(preview["stages"]["psv"]["action"], "compute")
        self.assertEqual(preview["stages"]["vsp"]["requires"], ("psv",))
        self.assertEqual(preview["stages"]["vsp"]["action"], "blocked")

        for boundary in (-1, 1):
            result = self.run_propagators(f"vsp-boundary-{boundary}",
                basis_id=basis_id, points_id=point_id, products=("PSV", "VSP"),
                t_boundary=boundary)
            self.assertEqual(set(result["datasets"]) & {"psv", "vsp"}, {"psv", "vsp"})
            self.reference(result["datasets"], rows, boundary=boundary)

    def test_distinct_ordered_point_pools_flow_through_psv_vsp_and_psp(self):
        basis_id = self.basis()
        sink_id, sink_rows = self.points("distinct-sink-points", xs=(0, 1, 0), labels=(9, 3))
        source_id, source_rows = self.points("distinct-source-points", xs=(1, 0, 1), labels=(4,))
        options = dict(basis_id=basis_id, points_snk_id=sink_id, points_src_id=source_id,
                       products=("VSV", "PSV", "VSP", "PSP"), Ne=2,
                       Np_snk=2, Np_src=1)
        preview = plan_indexed_wilson_propagators(self.index, self.gauge_id, **options)
        self.assertEqual(set(preview["stages"]), {
            "gauge", "basis", "points_snk", "points_src", "vsv", "psv", "psv_vsp", "vsp", "psp"})
        self.assertEqual(preview["stages"]["psv_vsp"]["action"], "compute")
        self.assertEqual(preview["stages"]["vsp"]["requires"], ("psv_vsp",))
        self.assertEqual(preview["stages"]["psp"]["action"], "compute")

        result = self.run_propagators("distinct-points", **options)
        ids = result["datasets"]
        with self.index.open(ids["psv"]) as psv:
            self.assertEqual(psv.spec.document["bases"]["sink_point"]["data_id"], sink_id)
            self.assertEqual(psv.spec.document["bases"]["sink_point"]["indices"], [9, 3])
            self.assertEqual(psv.spec.document["parents"], [basis_id, self.gauge_id, sink_id])
        with self.index.open(ids["psv_for_vsp"]) as reverse_psv:
            self.assertEqual(reverse_psv.spec.document["bases"]["sink_point"]["data_id"], source_id)
            self.assertEqual(reverse_psv.spec.document["bases"]["sink_point"]["indices"], [4])
        with self.index.open(ids["vsp"]) as vsp:
            self.assertEqual(vsp.spec.document["parents"], [ids["psv_for_vsp"]])
            self.assertEqual(vsp.spec.document["bases"]["source_point"]["data_id"], source_id)
            self.assertEqual(vsp.spec.document["axes"][5]["coordinates"], [4])
        with self.index.open(ids["psp"]) as psp:
            self.assertEqual(psp.spec.document["parents"], [self.gauge_id, sink_id, source_id])
            self.assertEqual(psp.spec.document["bases"]["sink_point"]["data_id"], sink_id)
            self.assertEqual(psp.spec.document["bases"]["source_point"]["data_id"], source_id)
            self.assertEqual(psp.spec.coordinates("sink_point"), [9, 3])
            self.assertEqual(psp.spec.coordinates("source_point"), [4])
        self.reference(ids, rows_snk=sink_rows, rows_src=source_rows,
            point_snk_id=sink_id, point_src_id=source_id, psv_parent_id=ids["psv_for_vsp"],
            ne=2, np_snk=2, np_src=1)






    def test_vsp_descriptor_rejects_unknown_science_and_noncanonical_coordinates(self):
        basis_id = self.basis()
        point_id, _ = self.points("vsp-invalid-descriptor")
        graph = plan_indexed_wilson_propagators(self.index, self.gauge_id,
            basis_id=basis_id, points_id=point_id, products=("PSV",))
        source = graph["stages"]["psv"]["spec"].document
        cases = []
        for path, value in (
                (("definition", "dirac", "action"), "clover"),
                (("definition", "dirac", "r"), True),
                (("definition", "dirac", "lattice_spacing"), True),
                (("definition", "dirac", "clover"), True),
                (("conventions", "normalization"), "scaled"),
                (("conventions", "spin_basis"), "unknown"),
                (("conventions", "temporal_boundary"), "open"),
                (("axes", 0, "coordinates", 0), True),
                (("axes", 2, "coordinates", 0), 0.0),
                (("configuration", "ensemble"), None),
                (("dtype",), "complex64"),
        ):
            mutated = DatasetSpec(source).document
            target = mutated
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            cases.append(mutated)
        for document in cases:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    vsp_dataset_document(document, psv_id="psv-parent",
                        gauge_id=self.gauge_id, basis_id=basis_id, points_id=point_id)


    def test_point_prefix_view_preserves_actual_identity_and_coordinates(self):
        point_id, rows = self.points("points", labels=(9, 3))
        full = self.run_propagators("full", points_id=point_id,
                                    products=("PSP",))["datasets"]
        with self.assertRaisesRegex(ValueError, "points_id with Np"):
            plan_indexed_wilson_propagators(self.index, self.gauge_id,
                points_id=point_id, products=("PSP",), Np_snk=2, Np_src=1)
        preview = plan_indexed_wilson_propagators(self.index, self.gauge_id,
            points_snk_id=point_id, points_src_id=point_id, products=("PSP",),
            Np_snk=2, Np_src=1)
        product_spec = preview["stages"]["psp"]["spec"]
        self.assertEqual(product_spec.document["bases"]["sink_point"]["indices"], [9, 3])
        self.assertEqual(product_spec.document["bases"]["source_point"]["indices"], [9])
        self.assertEqual(product_spec.document["bases"]["sink_point"]["data_id"], point_id)
        self.assertEqual(product_spec.document["bases"]["source_point"]["data_id"], point_id)
        self.assertEqual(product_spec.document["parents"], [self.gauge_id, point_id])

        smaller = self.run_propagators("smaller", points_snk_id=point_id,
            points_src_id=point_id, products=("PSP",), Np_snk=2, Np_src=1)["datasets"]
        self.assertNotEqual(smaller["psp"], full["psp"])
        with self.index.open(smaller["psp"]) as prefix:
            self.assertEqual(prefix.spec.coordinates("sink_point"), [9, 3])
            self.assertEqual(prefix.spec.coordinates("source_point"), [9])
        self.reference(full, rows, np_snk=2, np_src=2)
        self.reference(smaller, rows, point_snk_id=point_id, point_src_id=point_id,
                       np_snk=2, np_src=1)


    def test_unknown_point_conventions_and_configuration_mismatch_rejected_by_plan(self):
        cases = (dict(conventions=dict(representation="coordinate_delta", normalization="unknown")),
                 dict(configuration=dict(self.configuration, configuration="other-cfg")))
        for position, options in enumerate(cases):
            point_id, _ = self.points(f"invalid-{position}", **options)
            with self.assertRaises(ValueError):
                plan_indexed_wilson_propagators(self.index, self.gauge_id,
                    points_id=point_id, products=("PSP",))

    def test_invalid_actual_coordinates_reject_during_public_production(self):
        point_id, _ = self.points("out-of-range", xs=(0, 2, 0))
        with self.assertRaisesRegex(ValueError, "outside the lattice"):
            self.run_propagators("invalid-points", points_id=point_id, products=("PSP",))
        output = self.root / "invalid-points" / "psp"
        self.assertNotEqual(inspect_dataset(output, cfg="cfg")["state"], "complete")
        self.assertFalse((output / "cfg.npy").exists())
        self.assertFalse((output / "cfg.h5").exists())

    def test_direct_point_adapter_rejects_same_layout_with_wrong_physical_representation(self):
        cases = (dict(kind="coordinate_observations"),
                 dict(conventions=dict(representation="weighted_points", normalization="unit_delta")),
                 dict(conventions=dict(representation="coordinate_delta", normalization="unknown")),
                 dict(labels=(-1,)))
        for position, options in enumerate(cases):
            with self.subTest(options=options):
                point_id, _ = self.points(f"wrong-representation-{position}", **options)
                with self.index.open(point_id) as dataset:
                    with self.assertRaisesRegex(ValueError, "spatial_points"):
                        DatasetPointHandle(dataset)
