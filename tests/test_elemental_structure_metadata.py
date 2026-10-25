"""Producer-declared elemental support structures through reader adapters."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.adapters.sparse_elemental import SparseElemental
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import (
    BaryonComponent, ElementalComponent, SparsePointBlock,
    SparsePointRecordMetadata,
    normalize_elemental_structure,
)
from lqcdweave.runtime.datasets import DatasetView
from test_dataset_point_elemental_adapter import (
    _DatasetPointElementalAdapterFixture,
)


class ElementalStructureContractTests(unittest.TestCase):
    def test_normalizer_and_tensor_metadata_keep_legacy_boolean_compatible(self):
        self.assertEqual(normalize_elemental_structure("general"), "general")
        self.assertEqual(normalize_elemental_structure("general", True), "point_diagonal")
        self.assertEqual(normalize_elemental_structure("diagonal", True), "diagonal")
        for value in ("unknown", None, 1):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                normalize_elemental_structure(value)
        with self.assertRaises(TypeError):
            normalize_elemental_structure("general", 1)

        vector_diagonal = TensorAxisMetadata(
            ("mode_left", "mode_right"), (4, 4), np.dtype("complex128"),
            "vv", "vv-selection", structure="diagonal",
        )
        self.assertEqual(vector_diagonal.structure, "diagonal")
        self.assertFalse(vector_diagonal.point_diagonal)
        point_diagonal = TensorAxisMetadata(
            ("point_left", "color_left", "point_right", "color_right"),
            (3, 3, 3, 3), np.dtype("complex128"), "pp", "pp-selection",
            point_diagonal=True,
        )
        self.assertEqual(point_diagonal.structure, "point_diagonal")
        self.assertTrue(point_diagonal.point_diagonal)
        fully_diagonal = replace(point_diagonal, structure="diagonal")
        self.assertEqual(fully_diagonal.structure, "diagonal")
        self.assertTrue(fully_diagonal.point_diagonal)

    def test_tensor_structure_requires_the_complete_left_right_basis_pair(self):
        base = dict(dtype=np.dtype("complex128"), data_identity="data",
                    selection_identity="selection")
        with self.assertRaisesRegex(ValueError, "mode axes or equal point and color"):
            TensorAxisMetadata(("mode_left", "mode_right"), (2, 3),
                               structure="diagonal", **base)
        with self.assertRaisesRegex(ValueError, "mode axes or equal point and color"):
            TensorAxisMetadata(("point_left", "point_right"), (2, 2),
                               structure="diagonal", **base)
        with self.assertRaisesRegex(ValueError, "color axes"):
            TensorAxisMetadata(
                ("point_left", "color_left", "point_right", "color_right"),
                (2, 3, 2, 2), structure="diagonal", **base,
            )
        with self.assertRaisesRegex(ValueError, "equal left/right point"):
            TensorAxisMetadata(("point_left", "point_right"), (2, 3),
                               point_diagonal=True, **base)

    def test_sparse_records_and_copies_preserve_only_explicit_structure(self):
        common = dict(
            nnz=2, value_dtype=np.dtype("complex128"),
            index_dtype=np.dtype("int64"), record_bytes=0,
            data_identity="record", selection_identity="record-selection",
        )
        identity = SparsePointRecordMetadata(
            "identity", (2, 2), nnz=2, value_dtype=np.dtype("complex128"),
            index_dtype=np.dtype("int64"), record_bytes=0,
            data_identity="record", selection_identity="identity-selection",
        )
        self.assertEqual(identity.structure, "general")
        self.assertFalse(identity.point_diagonal)
        declared_identity = replace(identity, structure="diagonal")
        self.assertEqual(declared_identity.structure, "diagonal")
        self.assertTrue(declared_identity.point_diagonal)
        self.assertEqual(
            SparsePointRecordMetadata(
                "identity", (2, 3), nnz=2, value_dtype=np.dtype("complex128"),
                index_dtype=np.dtype("int64"), record_bytes=0,
                data_identity="record", selection_identity="rectangular-identity",
            ).structure,
            "general",
        )
        sparse = SparsePointRecordMetadata(
            "sparse", (2, 2), nnz=1, record_bytes=2 * 8 + 9 * 16,
            structure="point_diagonal",
            **{k: v for k, v in common.items() if k not in {"nnz", "record_bytes"}},
        )
        self.assertTrue(sparse.point_diagonal)

        block = SparsePointBlock(
            np.asarray([0]), np.asarray([0]), np.ones((1, 3, 3), complex),
            (2, 2), structure="diagonal",
        )
        copied = block.copy()
        self.assertEqual(copied.structure, "diagonal")
        self.assertTrue(copied.point_diagonal)


class ArrayElementalStructureTests(unittest.TestCase):
    def setUp(self):
        self.component = ElementalComponent(0)
        # Deliberately include off-diagonal values: structure comes from the
        # producer declaration and is never inferred by scanning payloads.
        rng = np.random.default_rng(19)
        self.vv = rng.normal(size=(1, 2, 1, 3, 3)).astype(np.complex128)
        self.pp = rng.normal(size=(1, 2, 1, 2, 3, 2, 3)).astype(np.complex128)

    def reader(self, *, declarations=None, legacy=()):
        return ArrayElemental(
            {"v2v": self.vv, "p2p": self.pp}, (self.component,),
            phase_convention="anchor", momentum_list=((0, 0, 0), (1, 0, 0)),
            elemental_structures=declarations, point_diagonal_components=legacy,
        )

    def test_component_momentum_declarations_and_selection_downgrade(self):
        reader = self.reader(declarations={
            ("v2v", self.component, 1): "diagonal",
            ("p2p", self.component, 0): "diagonal",
            ("p2p", self.component, 1): "point_diagonal",
        })
        vv = reader.describe_block(self.component, 1, 0, "v2v")
        self.assertEqual(vv.structure, "diagonal")
        self.assertFalse(vv.point_diagonal)
        pp_full = reader.describe_block(self.component, 0, 0, "p2p")
        pp_point = reader.describe_block(self.component, 1, 0, "p2p")
        self.assertEqual(pp_full.structure, "diagonal")
        self.assertEqual(pp_point.structure, "point_diagonal")
        self.assertTrue(pp_full.point_diagonal)
        self.assertTrue(pp_point.point_diagonal)

        vv_reordered = reader.describe_block(
            self.component, 1, 0, "v2v",
            mode_indices={"mode_left": (0, 1), "mode_right": (1, 0)},
        )
        pp_reordered = reader.describe_block(
            self.component, 0, 0, "p2p",
            point_indices={"point_left": (0, 1), "point_right": (1, 0)},
        )
        self.assertEqual(vv_reordered.structure, "general")
        self.assertFalse(vv_reordered.point_diagonal)
        self.assertEqual(pp_reordered.structure, "general")
        self.assertFalse(pp_reordered.point_diagonal)
        self.assertNotEqual(vv.selection_identity, vv_reordered.selection_identity)
        self.assertNotEqual(pp_full.selection_identity, pp_reordered.selection_identity)

        slab = reader.describe_slab(
            self.component, 0, (0,), "p2p",
            point_indices={"point_left": (1, 0), "point_right": (1, 0)},
        )
        self.assertEqual(slab.structure, "diagonal")
        reader.close()

    def test_declarations_validate_sector_component_momentum_and_shape(self):
        invalid = (
            ({("v2p", self.component, 0): "diagonal"}, ValueError),
            ({("p2p", ElementalComponent(1), 0): "diagonal"}, ValueError),
            ({("p2p", self.component, 2): "diagonal"}, ValueError),
            ({("v2v", self.component, 0): "point_diagonal"}, ValueError),
        )
        for declarations, error in invalid:
            with self.subTest(declarations=declarations), self.assertRaises(error):
                self.reader(declarations=declarations)
        with self.assertRaisesRegex(ValueError, "absent sector"):
            ArrayElemental(
                {"v2v": self.vv}, (self.component,), phase_convention="anchor",
                elemental_structures={("p2p", self.component, 0): "diagonal"},
            )

    def test_distinct_point_pools_downgrade_a_diagonal_declaration(self):
        reader = ArrayElemental(
            {"p2p": self.pp[:, :1]}, (self.component,), phase_convention="anchor",
            sampling_pool_ids={
                (self.component, 0, 0, "bar"): "left-pool",
                (self.component, 0, 0, "field"): "right-pool",
            },
            elemental_structures={("p2p", self.component, 0): "diagonal"},
        )
        metadata = reader.describe_block(self.component, 0, 0, "p2p")
        self.assertEqual(metadata.structure, "general")
        self.assertFalse(metadata.point_diagonal)
        reader.close()

    def test_npy_and_binary_factories_forward_structure_declarations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vv = self.vv[:, :1]
            npy_path = root / "vv.npy"
            np.save(npy_path, vv)
            npy_reader = ArrayElemental.from_npy(
                {"v2v": npy_path}, (self.component,),
                axis_order=("component", "momentum", "time"),
                phase_convention="anchor",
                elemental_structures={("v2v", self.component, 0): "diagonal"},
            )
            self.assertEqual(
                npy_reader.describe_block(self.component, 0, 0, "v2v").structure,
                "diagonal",
            )
            npy_reader.close()

            binary_path = root / "vv.bin"
            vv.tofile(binary_path)
            binary_reader = ArrayElemental.from_binary(
                {"v2v": binary_path}, (self.component,),
                shapes={"v2v": vv.shape}, dtypes={"v2v": vv.dtype},
                axis_order=("component", "momentum", "time"),
                phase_convention="anchor",
                elemental_structures={("v2v", self.component, 0): "diagonal"},
            )
            self.assertEqual(
                binary_reader.describe_block(self.component, 0, 0, "v2v").structure,
                "diagonal",
            )
            binary_reader.close()

    def test_three_mode_baryon_accepts_general_and_rejects_non_general_structure(self):
        component = BaryonComponent("baryon")
        sectors = {"vvv": np.zeros((1, 1, 1, 2, 2, 2), dtype=np.complex128)}
        reader = ArrayBaryonElemental(
            sectors, (component,), elemental_structures={
                ("vvv", component, 0): "general",
            },
        )
        self.assertEqual(reader.elemental_structure(component, 0, "vvv"), "general")
        reader.close()
        with self.assertRaisesRegex(ValueError, "three-mode baryon"):
            ArrayBaryonElemental(
                sectors, (component,), elemental_structures={
                    ("vvv", component, 0): "diagonal",
                },
            )


class _ReaderWithSpec:
    def __init__(self, reader, spec):
        self.reader = reader
        self.spec = spec
        self.data_id = reader.data_id

    def read(self, *args, **kwargs):
        return self.reader.read(*args, **kwargs)


class DatasetElementalStructureTests(_DatasetPointElementalAdapterFixture):
    def _with_structures(self, reader, rows):
        document = reader.spec.document
        document["conventions"].pop("point_diagonal_components", None)
        document["conventions"]["elemental_structures"] = rows
        return _ReaderWithSpec(reader, DatasetSpec(document))

    def test_persisted_structure_changes_metadata_and_selection_binding(self):
        stored = self.readers["p2p"]
        general = DatasetElemental(
            {"p2p": self._with_structures(stored, [])}, point_dataset=self.points,
        )
        declared = DatasetElemental(
            {"p2p": self._with_structures(stored, [
                {"component": 0, "momentum": 0, "structure": "diagonal"},
            ])},
            point_dataset=self.points,
        )
        component = declared.components[0]
        ordinary = general.describe_block(component, 0, 0, "p2p")
        diagonal = declared.describe_block(component, 0, 0, "p2p")
        self.assertEqual(ordinary.structure, "general")
        self.assertFalse(ordinary.point_diagonal)
        self.assertEqual(diagonal.structure, "diagonal")
        self.assertTrue(diagonal.point_diagonal)
        self.assertNotEqual(ordinary.selection_identity, diagonal.selection_identity)
        self.assertEqual(
            declared.describe_sparse_record(component, 0, 0).structure,
            "diagonal",
        )
        selected = declared.describe_block(
            component, 0, 0, "p2p",
            point_indices={"point_left": (2, 0), "point_right": (0, 2)},
        )
        self.assertEqual(selected.structure, "general")
        self.assertFalse(selected.point_diagonal)

    def test_dataset_view_reorders_stored_component_and_momentum_slots(self):
        stored = self.readers["p2p"]
        document = stored.spec.document
        document["conventions"]["elemental_structures"] = [
            {"component": 1, "momentum": 0, "structure": "point_diagonal"},
        ]
        source_spec = DatasetSpec(document)
        source_values = np.stack([
            stored.read(time=time) for time in source_spec.coordinates("time")
        ])
        source_id = self.helper.save("elemental-structure-source", source_spec, source_values)
        source = self.helper.index.open(source_id)
        self.addCleanup(source.close)

        view_document = source_spec.document
        component_axis = next(axis for axis in view_document["axes"]
                              if axis["name"] == "component")
        momentum_axis = next(axis for axis in view_document["axes"]
                             if axis["name"] == "momentum")
        original_components = component_axis["coordinates"]
        original_momenta = momentum_axis["coordinates"]
        component_axis["coordinates"] = [original_components[1]]
        momentum_axis["coordinates"] = [original_momenta[1], original_momenta[0]]
        view = DatasetView(
            source, DatasetSpec(view_document), subset_axes=("component", "momentum"),
        )
        elemental = DatasetElemental({"p2p": view}, point_dataset=self.points)
        component = elemental.components[0]
        self.assertEqual(elemental.elemental_structure(component, 1, "p2p"), "point_diagonal")
        self.assertEqual(elemental.describe_block(component, 1, 0, "p2p").structure,
                         "point_diagonal")
        self.assertEqual(elemental.describe_block(component, 0, 0, "p2p").structure,
                         "general")

    def test_structure_convention_requires_unique_valid_slots_and_names(self):
        stored = self.readers["p2p"]
        rows = (
            [{"component": True, "momentum": 0, "structure": "diagonal"}],
            [{"component": 0, "momentum": 99, "structure": "diagonal"}],
            [{"component": 0, "momentum": 0, "structure": "unknown"}],
            [{"component": 0, "momentum": 0, "structure": "diagonal"}] * 2,
            [{"component": 0, "momentum": 0, "structure": "general", "extra": 1}],
        )
        for declaration in rows:
            with self.subTest(declaration=declaration), self.assertRaises((TypeError, ValueError)):
                DatasetElemental(
                    {"p2p": self._with_structures(stored, declaration)},
                    point_dataset=self.points,
                )


class SparseElementalStructureTests(unittest.TestCase):
    class _RecordReader:
        def __init__(self, metadata):
            self.metadata = metadata

        def describe_record(self, component, momentum, t):
            return self.metadata

        def read_record(self, component, momentum, t):
            return {"type": "sparse", "indices": np.empty((0, 2), dtype=np.int64),
                    "values": np.empty((0, 3, 3), dtype=np.complex128)}

    def test_composed_sparse_reader_inherits_structure_without_scanning_colors(self):
        component = ElementalComponent(0)
        vv = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
        pp = np.zeros((1, 1, 1, 2, 3, 2, 3), dtype=np.complex128)
        pp[:, :, :, 0, :, 0, :] = 1
        dense = ArrayElemental(
            {"v2v": vv, "p2p": pp}, (component,), phase_convention="anchor",
            elemental_structures={("p2p", component, 0): "diagonal"},
        )
        record = SparsePointRecordMetadata(
            "sparse", (2, 2), nnz=0, value_dtype=np.dtype("complex128"),
            index_dtype=np.dtype("int64"), record_bytes=0,
            data_identity="sparse", selection_identity="sparse-selection",
        )
        reader = SparseElemental(dense, self._RecordReader(record))
        metadata = reader.describe_block(component, 0, 0, "p2p")
        self.assertEqual(metadata.structure, "diagonal")
        self.assertTrue(metadata.point_diagonal)
        self.assertEqual(reader.describe_sparse_record(component, 0, 0).structure,
                         "diagonal")
        dense.close()


if __name__ == "__main__":
    unittest.main()
