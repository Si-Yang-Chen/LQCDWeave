"""Support declarations compared with actual V/D spatial Gram operators."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.planning.production import elemental_dataset_document
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from production.elemental.test_elemental import fixture


class _Basis:
    basis_orthogonal = True
    basis_id = "actual-parent-basis"

    def __init__(self, values):
        self.values = values
        self.Ne = values.shape[1]

    def load(self, key):
        return self.values


class ElementalStructureProductionTests(unittest.TestCase):
    def setUp(self):
        self.sizes, self.gauge, values, self.points = fixture()
        rng = np.random.default_rng(841)
        nt, _ne, *tail = values.shape
        vectors = []
        for _time in range(nt):
            q, _ = np.linalg.qr(rng.normal(size=(np.prod(tail), 5))
                                + 1j * rng.normal(size=(np.prod(tail), 5)))
            vectors.append(q.T.reshape(5, *tail))
        self.basis = _Basis(np.stack(vectors))
        self.local = ElementalComponent.path(())
        self.closed = ElementalComponent.path((0, 1, 3, 4))
        self.table = [dict(component=self.local, path=(), coefficient=0.4 + 0.7j),
                      dict(component=self.closed, path=(0, 1, 3, 4), coefficient=1)]

    def generator(self, basis):
        return ElementalGenerator(self.gauge, basis, ((0, 0, 0), (1, 0, 0)),
                                  self.table, lattice_size=self.sizes)

    def test_same_actual_orthogonal_basis_local_zero_momentum(self):
        producer = self.generator(self.basis).load(None)
        self.assertEqual(producer.elemental_structures,
                         {("v2v", self.local, 0): "diagonal"})
        for time in range(self.sizes[3]):
            value = producer.calc_path_blocks(time)["v2v"][0, 0]
            flat = self.basis.values[time].reshape(5, -1)
            expected = (0.4 + 0.7j) * flat.conj() @ flat.T
            np.testing.assert_allclose(value, expected, atol=2e-14)
            np.testing.assert_allclose(value, (0.4 + 0.7j) * np.eye(5), atol=2e-14)

    def test_unlabelled_arrays_are_general_even_if_numerically_orthogonal(self):
        producer = self.generator(self.basis.values)
        self.assertEqual(producer.elemental_structures, {})

    def test_local_production_fast_paths_match_dense_gram_reference(self):
        for blocks in (("v2v",), ("v2v", "p2p")):
            with self.subTest(blocks=blocks):
                producer = ElementalGenerator(self.gauge, self.basis, ((0, 0, 0),),
                    self.table[:1], lattice_size=self.sizes, blocks=blocks,
                    points=self.points, ne_block_size=2).load(None)
                for time in (0, 2, 0):
                    value = producer.calc_path_blocks(time)["v2v"][0, 0]
                    flat = self.basis.values[time].reshape(5, -1)
                    np.testing.assert_allclose(value, (0.4 + 0.7j) * flat.conj() @ flat.T,
                                               atol=2e-14)

    def test_qr_and_legacy_d_norms_preserve_actual_diagonal_coefficients(self):
        for mixing, expected_norms in (("qr", (1, 1, 1)), ("legacy", (2, 1, 1))):
            with self.subTest(mixing=mixing):
                noise = NoisevectorGenerator(self.basis, ((2, 3), (1, 2)), mixing=mixing).load(None)
                vectors = noise.calc(seed=31)
                self.assertTrue(noise.describe()["basis_orthogonal"])
                source = _Basis(vectors)
                producer = self.generator(source).load(None)
                self.assertEqual(producer.elemental_structures[("v2v", self.local, 0)], "diagonal")
                result = producer.calc_path_blocks(1)["v2v"][0, 0]
                flat = vectors[1].reshape(3, -1)
                np.testing.assert_allclose(result, (0.4 + 0.7j) * flat.conj() @ flat.T, atol=2e-14)
                np.testing.assert_allclose(result, (0.4 + 0.7j) * np.diag(expected_norms), atol=2e-14)

    def test_point_local_full_diagonal_closed_loop_only_point_diagonal(self):
        producer = ElementalGenerator(self.gauge, self.basis.values,
            ((0, 0, 0), (1, 0, 0)), self.table, blocks=("p2p",),
            points=self.points)
        for momentum in range(2):
            self.assertEqual(producer.elemental_structures["p2p", self.local, momentum], "diagonal")
            self.assertEqual(producer.elemental_structures["p2p", self.closed, momentum], "point_diagonal")
        blocks = producer.calc_path_blocks(0)["p2p"]
        for momentum in range(2):
            local = blocks[0][momentum]
            self.assertEqual(local.copy().structure, "diagonal")
            value = local.to_dense().reshape(local.point_shape[0] * 3, -1)
            np.testing.assert_allclose(value, np.diag(np.diag(value)), atol=2e-14)
            self.assertEqual(blocks[1][momentum].structure, "point_diagonal")

    def test_registered_vv_proof_round_trip_and_unlabelled_legacy_reader(self):
        configuration = dict(ensemble="synthetic", configuration="cfg",
                             lattice=list(self.sizes))

        def field_spec(kind, axes, conventions):
            return DatasetSpec(dict(schema=1, kind=kind, configuration=configuration,
                axes=[dict(name=name, coordinates=list(labels)) for name, labels in axes],
                dtype="complex128", bases={}, parents=[], definition={}, conventions=conventions))

        lx, ly, lz, lt = self.sizes
        gauge = field_spec("gauge", (("time", range(lt)), ("z", range(lz)),
            ("y", range(ly)), ("x", range(lx)), ("direction", range(4)),
            ("color_row", range(3)), ("color_column", range(3))), {})
        basis = field_spec("explicit_basis", (("time", range(lt)), ("mode", range(5)),
            ("z", range(lz)), ("y", range(ly)), ("x", range(lx)), ("color", range(3))),
            dict(normalization="orthonormal"))
        momenta = ((0, 0, 0), (1, 0, 0))
        document = elemental_dataset_document(gauge, basis,
            gauge_id="actual-gauge", basis_id=self.basis.basis_id,
            components=[dict(path=[])], momenta=momenta, modes=list(range(5)),
            phase_convention="anchor")
        self.assertEqual(document["conventions"]["elemental_structures"],
                         [dict(component=0, momentum=0, structure="diagonal")])
        producer = ElementalGenerator(self.gauge, self.basis, momenta,
            [dict(component=self.local, path=(), coefficient=1)],
            phase_convention="anchor", lattice_size=self.sizes).load("cfg")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with DatasetWriter(root, DatasetSpec(document), provenance={"test": "VV structure round trip"}) as writer:
                for time in range(lt):
                    writer.write(time, producer.calc_path_blocks(time)["v2v"])
                writer.publish()
            with DatasetReader(root, cfg="cfg") as stored:
                reader = DatasetElemental(stored)
                component = reader.components[0]
                self.assertEqual(reader.describe_block(component, 0, 0, "v2v").structure, "diagonal")
                self.assertEqual(reader.describe_block(component, 1, 0, "v2v").structure, "general")
                np.testing.assert_allclose(reader.block(component, 0, 0, "v2v"), np.eye(5), atol=2e-14)
                reordered = reader.describe_block(component, 0, 0, "v2v", mode_indices={
                    "sink_mode": (2, 0), "source_mode": (0, 2)})
                self.assertEqual(reordered.structure, "general")
                legacy_document = stored.spec.document
                legacy_document["conventions"].pop("elemental_structures")
                legacy = DatasetElemental(SimpleNamespace(spec=DatasetSpec(legacy_document),
                    data_id=stored.data_id, read=stored.read))
                self.assertEqual(legacy.describe_block(component, 0, 0, "v2v").structure, "general")
                np.testing.assert_array_equal(legacy.block(component, 0, 0, "v2v"),
                                              reader.block(component, 0, 0, "v2v"))


if __name__ == "__main__":
    unittest.main()
