"""Public baryon calculation persisted by a small caller-owned writer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from test_baryon_elemental import Handle, determinant_reference
from production.elemental.test_elemental import fixture, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator
from lqcdweave.production.elemental.baryon_planning import plan_indexed_baryon_elementals



class BaryonDatasetProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, self.links, source_vectors, _ = fixture()
        self.configuration = dict(
            ensemble="baryon-test", configuration="cfg-baryon",
            lattice=[self.sizes[0], self.sizes[1], self.sizes[2], self.sizes[3]],
        )
        self.times = tuple(range(self.sizes[3]))
        self.vectors_a = source_vectors.copy()
        self.vectors_b = source_vectors * (.7 + .2j) + (.13 - .17j)
        self.gauge_id = self._publish_gauge()
        self.basis_a_id = self._publish_basis("basis-a", (10, 3, 8), self.vectors_a)
        self.basis_b_id = self._publish_basis("basis-b", (7, 4, 9), self.vectors_b)
        self.basis_ids = (self.basis_a_id, self.basis_b_id, self.basis_b_id)
        self.counts = (2, 3, 1)
        self.vectors = (
            self.vectors_a[:, :2],
            self.vectors_b[:, :3],
            self.vectors_b[:, :1],
        )
        self.momenta = ((0, 0, 0), (1, -1, 1))
        self.raw_components = (
            ((), (), ()),
            ((0,), (4,), (2, 0)),
            ((1, 0), (0, 1), (5,)),
        )

    def _publish_gauge(self):
        axes = [
            dict(name=name, coordinates=list(values))
            for name, values in zip(
                ("time", "z", "y", "x", "direction", "color_row", "color_column"),
                (self.times[::-1], range(self.sizes[2]), range(self.sizes[1]),
                 range(self.sizes[0]), range(4), range(3), range(3)),
            )
        ]
        spec = DatasetSpec(dict(
            schema=1, kind="gauge", configuration=self.configuration, axes=axes,
            dtype="complex128", bases={}, definition={}, conventions={"boundary": "periodic"},
            parents=[],
        ))
        return self._publish("gauge", spec, {time: self.links[time] for time in self.times})

    def _publish_basis(self, name, labels, vectors):
        axes = [
            dict(name="time", coordinates=list(self.times[::-1])),
            dict(name="mode", coordinates=list(labels)),
            dict(name="z", coordinates=list(range(self.sizes[2]))),
            dict(name="y", coordinates=list(range(self.sizes[1]))),
            dict(name="x", coordinates=list(range(self.sizes[0]))),
            dict(name="color", coordinates=[0, 1, 2]),
        ]
        spec = DatasetSpec(dict(
            schema=1, kind="explicit_basis", configuration=self.configuration,
            axes=axes, dtype="complex128", bases={},
            definition={"realization": name},
            conventions={"normalization": "orthonormal"}, parents=[self.gauge_id],
        ))
        return self._publish(name, spec, {time: vectors[time] for time in self.times})

    def _publish(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance={"fixture": "independent"}) as writer:
            for position, time in enumerate(spec.coordinates("time")):
                writer.write(position, np.asarray(rows[time], dtype=spec.document["dtype"]))
            identity = writer.publish()
        self.assertEqual(self.index.register(directory), identity)
        return identity

    def _options(self, *, product="raw_paths", components=None):
        return dict(
            product=product,
            components=self.raw_components if components is None and product == "raw_paths" else components,
            momentum_list=self.momenta,
            usedNe=self.counts,
            phase_convention="anchor",
            path_boundary="periodic",
        )

    def _run(self, name, **options):
        """Compose the public calculator with a plain DatasetWriter for this test."""
        product = options["product"]
        preview = plan_indexed_baryon_elementals(
            self.index, self.gauge_id, self.basis_ids, **options)
        spec = preview["dataset_spec"]
        counts = tuple(len(spec.coordinates(f"mode{leg}")) for leg in range(3))
        selected = tuple(values[:, :count] for values, count in zip(self.vectors, counts))
        generator = BaryonElementalGenerator(
            self.sizes, Handle(self.links), tuple(Handle(value) for value in selected),
            momentum_list=options["momentum_list"], usedNe=counts,
            paths=self.raw_components, phase_convention=options["phase_convention"],
            path_boundary=options["path_boundary"],
        ).load(self.configuration["configuration"])
        output_root = self.root / name
        output_root.mkdir(parents=True, exist_ok=True)
        directory = output_root / product
        with DatasetWriter(directory, spec, provenance={"fixture": "independent"}) as writer:
            for position in writer.missing:
                time = spec.coordinates("time")[position]
                if product == "raw_paths":
                    row = generator.calc_path_blocks(time)["vvv"]
                elif product == "projected":
                    row = generator.calc_path_blocks(time, rows=options["components"])["vvv"]
                else:
                    row = generator.calc_path_blocks(
                        time, distances=options["components"])["vvv"]
                writer.write(position, np.array(row, copy=True))
            identity = writer.publish()
        self.assertEqual(self.index.register(directory), identity)
        return {"datasets": {product: identity}}

    def _expected(self, vectors, paths, time, momentum, phase=True):
        return determinant_reference(self.links, vectors, paths, time, momentum, phase=phase)

    def _read_rows(self, data_id):
        rows = {}
        with self.index.open(data_id) as reader:
            for time in reader.spec.coordinates("time"):
                rows[time] = reader.read(time=time).copy()
        return rows

    def test_versioned_catalog_survives_production_registration_and_binding(self):
        from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
        from lqcdweave.operators.single_hadron.api import baryon_preset
        from lqcdweave.production.elemental.api import BaryonElementalGenerator as Frontend
        from lqcdweave.planning.baryon import estimate_baryon_vvv_memory
        from lqcdweave.contracts.elemental_catalog import catalog_definition
        table = baryon_preset().select(('s3/0,1,2/mixed0/MA','local','distance/1,0,0'))
        options = self._options(product='projected',components=table)
        preview = plan_indexed_baryon_elementals(self.index,self.gauge_id,self.basis_ids,**options)
        spec = preview['dataset_spec']
        self.assertEqual(spec.document['definition']['catalog'],catalog_definition(table))
        producer = Frontend(self.links,self.vectors,self.momenta,table)
        directory = self.root/'catalog'
        with DatasetWriter(directory,spec,provenance={'test':'catalog'}) as writer:
            for slot,time in enumerate(self.times):
                writer.write(slot,producer.calc_path_blocks(time)['vvv'].copy())
            data_id = writer.publish()
        self.index.register(directory)
        with self.index.open(self.gauge_id) as gauge, self.index.open(self.basis_a_id) as a, \
                self.index.open(self.basis_b_id) as b, self.index.open(data_id) as dataset:
            estimate_baryon_vvv_memory({self.gauge_id:gauge.spec,self.basis_a_id:a.spec,
                self.basis_b_id:b.spec},spec,gauge_id=self.gauge_id,basis_ids=self.basis_ids)
            reader = DatasetBaryonElemental(dataset)
            self.assertEqual(reader.components,tuple(row['component'] for row in table))
            for row in table:
                for time in self.times:
                    for momentum,p in enumerate(self.momenta):
                        expected = sum(complex(*term['coefficient'])*self._expected(
                            self.vectors,term['paths'],time,p) for term in row['terms'])
                        np.testing.assert_allclose(reader.block(row['component'],momentum,time),
                            expected,atol=3e-12,rtol=3e-12)

    def test_raw_projected_and_distance_products_match_independent_determinants(self):
        with self.subTest(product="raw_paths"):
            raw_options = self._options()
            preview = plan_indexed_baryon_elementals(
                self.index, self.gauge_id, self.basis_ids, **raw_options)
            self.assertEqual(preview["product"], "raw_paths")
            self.assertEqual(preview["times"], self.times)
            self.assertIsInstance(preview["requests"]["raw_paths"].template, dict)
            self.assertIsInstance(preview["dataset_spec"], DatasetSpec)
            self.assertEqual(preview["dataset_spec"].coordinates("time"), list(self.times))
            self.assertIs(preview["dataset_spec"],
                          preview["graph"]["stages"]["raw_paths"]["spec"])
            raw_result = self._run("raw-paths", **raw_options)
            raw_id = raw_result["datasets"]["raw_paths"]
            raw_rows = self._read_rows(raw_id)
            for time in self.times:
                for component, paths in enumerate(self.raw_components):
                    for momentum, p in enumerate(self.momenta):
                        expected = self._expected(self.vectors, paths, time, p)
                        np.testing.assert_allclose(raw_rows[time][component, momentum],
                                                   expected, atol=3e-12, rtol=3e-12)
            with self.index.open(raw_id) as reader:
                document = reader.spec.document
                self.assertEqual(document["kind"], "elemental_baryon_vvv")
                self.assertEqual(tuple(axis["name"] for axis in document["axes"]),
                                 ("time", "component", "momentum", "mode0", "mode1", "mode2"))
                self.assertEqual(reader.spec.coordinates("time"), list(self.times))
                self.assertEqual(document["dtype"], "complex128")
                self.assertEqual(document["bases"]["mode0"],
                    {"data_id": self.basis_a_id, "indices": [10, 3]})
                self.assertEqual(document["bases"]["mode1"],
                    {"data_id": self.basis_b_id, "indices": [7, 4, 9]})
                self.assertEqual(document["bases"]["mode2"],
                    {"data_id": self.basis_b_id, "indices": [7]})
                self.assertEqual(document["parents"],
                                 [self.gauge_id, self.basis_a_id, self.basis_b_id])
                definition = document["definition"]
                self.assertEqual(definition["product"], "raw_paths")
                self.assertEqual(definition["component_encoding"], "ordered_path_triples_v1")
                self.assertEqual(definition["input_roles"], {
                    "gauge": self.gauge_id,
                    "basis_leg0": self.basis_a_id,
                    "basis_leg1": self.basis_b_id,
                    "basis_leg2": self.basis_b_id,
                })
                self.assertEqual(document["conventions"], {
                    "normalization": "unscaled", "phase": "anchor",
                    "path_boundary": "periodic", "color_contraction": "epsilon",
                    "vector_conjugation": "none",
                })

        projected_rows = [
            {"id": 0, "terms": [
                {"paths": self.raw_components[0], "coefficient": [.2, .7]},
                {"paths": self.raw_components[1], "coefficient": [-.4, .1]},
            ]},
            {"id": 1, "terms": [
                {"paths": self.raw_components[2], "coefficient": [1., 0.]},
                {"paths": self.raw_components[2], "coefficient": [-1., 0.]},
            ]},
        ]
        with self.subTest(product="projected"):
            result = self._run("projected", **self._options(
                product="projected", components=projected_rows))
            projected_id = result["datasets"]["projected"]
            rows = self._read_rows(projected_id)
            for time in self.times:
                for momentum, p in enumerate(self.momenta):
                    first = (.2 + .7j) * self._expected(
                        self.vectors, self.raw_components[0], time, p)
                    first += (-.4 + .1j) * self._expected(
                        self.vectors, self.raw_components[1], time, p)
                    np.testing.assert_allclose(rows[time][0, momentum], first,
                                               atol=3e-12, rtol=3e-12)
                    np.testing.assert_array_equal(rows[time][1, momentum],
                                                 np.zeros_like(rows[time][1, momentum]))
            with self.index.open(projected_id) as reader:
                definition = reader.spec.document["definition"]
                self.assertEqual(definition["product"], "projected")
                self.assertEqual(definition["component_encoding"], "projected_row_ids_v1")
                self.assertEqual(definition["projected_rows"], [
                    {"id": 0, "terms": [
                        {"paths": [[], [], []], "coefficient": [.2, .7]},
                        {"paths": [[0], [4], [2, 0]], "coefficient": [-.4, .1]},
                    ]},
                    {"id": 1, "terms": [
                        {"paths": [[1, 0], [0, 1], [5]], "coefficient": [1., 0.]},
                        {"paths": [[1, 0], [0, 1], [5]], "coefficient": [-1., 0.]},
                    ]},
                ])
                self.assertEqual(reader.spec.coordinates("component"), [
                    {"encoding": "projected_row_id_v1", "id": 0},
                    {"encoding": "projected_row_id_v1", "id": 1},
                ])

        distances = ((0, 0, 0), (1, 0, 2))
        with self.subTest(product="distance"):
            result = self._run("distance", **self._options(
                product="distance", components=distances))
            distance_id = result["datasets"]["distance"]
            rows = self._read_rows(distance_id)
            for time in self.times:
                for component, triple in enumerate(distances):
                    transformed = []
                    for vector, distance in zip(self.vectors, triple):
                        field = vector.copy()
                        if distance:
                            for t in self.times:
                                matrices = [reference_matrix(
                                    self.links, (direction,) * distance, t,
                                    phase="none", boundary="periodic")
                                    for direction in range(6)]
                                average = sum(matrices) / 6
                                field[t] = (average @ vector[t].reshape(vector.shape[1], -1).T
                                            ).T.reshape(vector[t].shape)
                        transformed.append(field)
                    for momentum, p in enumerate(self.momenta):
                        expected = self._expected(tuple(transformed),
                            ((), (), ()), time, p)
                        np.testing.assert_allclose(rows[time][component, momentum], expected,
                                                   atol=4e-12, rtol=4e-12)
            with self.index.open(distance_id) as reader:
                definition = reader.spec.document["definition"]
                self.assertEqual(definition["product"], "distance")
                self.assertEqual(definition["component_encoding"], "distance_triples_v1")
                self.assertEqual(definition["distance_rule"],
                    "independent_six_straight_paths_per_nonzero_leg")
                self.assertNotIn("projected_rows", definition)


def _with_nan(tile):
    copied = np.array(tile, copy=True)
    copied.flat[0] = complex(np.nan, 0.)
    return copied


if __name__ == "__main__":
    unittest.main()
