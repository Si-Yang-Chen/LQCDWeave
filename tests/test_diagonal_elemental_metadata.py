"""Explicit point-diagonal support handoff through elemental readers."""

from dataclasses import replace
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.adapters.sparse_elemental import SparseElemental
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import (
    ElementalComponent, SparsePointRecordMetadata,
)
from lqcdweave.runtime.datasets import DatasetView
from test_dataset_point_elemental_adapter import _DatasetPointElementalAdapterFixture


class ArrayPointDiagonalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.component = ElementalComponent(0)
        self.values = np.zeros((1, 1, 2, 4, 3, 4, 3), dtype=np.complex128)

    def reader(self, declarations=()):
        return ArrayElemental(
            {"p2p": self.values}, (self.component,), phase_convention="anchor",
            data_identity="array-p2p", point_diagonal_components=declarations,
        )

    def test_default_is_general_and_explicit_support_is_identity_bound(self):
        general = self.reader()
        declared = self.reader((self.component,))
        ordinary = general.describe_block(self.component, 0, 1, "p2p")
        diagonal = declared.describe_block(self.component, 0, 1, "p2p")

        self.assertFalse(ordinary.point_diagonal)
        self.assertTrue(diagonal.point_diagonal)
        self.assertNotEqual(ordinary.selection_identity, diagonal.selection_identity)
        self.assertTrue(declared.describe_slab(
            self.component, 0, (1, 0), "p2p",
        ).point_diagonal)

    def test_reordered_point_selection_drops_diagonal_support(self):
        reader = self.reader((self.component,))
        same_order = reader.describe_block(
            self.component, 0, 0, "p2p",
            point_indices={"point_left": (3, 1), "point_right": (3, 1)},
        )
        reordered = reader.describe_block(
            self.component, 0, 0, "p2p",
            point_indices={"point_left": (3, 1), "point_right": (1, 3)},
        )
        self.assertTrue(same_order.point_diagonal)
        self.assertFalse(reordered.point_diagonal)
        self.assertNotEqual(same_order.selection_identity, reordered.selection_identity)

    def test_declaration_is_an_immutable_subset_of_components(self):
        with self.assertRaises(TypeError):
            self.reader([self.component])
        with self.assertRaises(ValueError):
            self.reader((ElementalComponent(1),))


class _ReaderWithSpec:
    def __init__(self, reader, spec):
        self.reader = reader
        self.spec = spec
        self.data_id = reader.data_id

    def read(self, *args, **kwargs):
        return self.reader.read(*args, **kwargs)


class DatasetPointDiagonalMetadataTests(_DatasetPointElementalAdapterFixture):
    def _with_diagonal_rows(self, reader, rows):
        document = reader.spec.document
        document["conventions"].pop("elemental_structures", None)
        if rows is None:
            document["conventions"].pop("point_diagonal_components", None)
        else:
            document["conventions"]["point_diagonal_components"] = rows
        return _ReaderWithSpec(reader, DatasetSpec(document))

    def test_default_and_declared_dataset_metadata_change_identity(self):
        stored = self.readers["p2p"]
        general = DatasetElemental(
            {"p2p": self._with_diagonal_rows(stored, None)},
            point_dataset=self.points,
        )
        declared = DatasetElemental(
            {"p2p": self._with_diagonal_rows(stored, [0])},
            point_dataset=self.points,
        )
        component = declared.components[0]
        general_metadata = general.describe_block(component, 0, 0, "p2p")
        diagonal_metadata = declared.describe_block(component, 0, 0, "p2p")

        self.assertFalse(general_metadata.point_diagonal)
        self.assertTrue(diagonal_metadata.point_diagonal)
        self.assertNotEqual(general_metadata.selection_identity,
                            diagonal_metadata.selection_identity)
        self.assertTrue(declared.describe_sparse_record(component, 0, 0).point_diagonal)

    def test_component_view_rows_map_through_underlying_coordinates(self):
        stored = self.readers["p2p"]
        document = stored.spec.document
        component_axis = next(axis for axis in document["axes"]
                              if axis["name"] == "component")
        original = component_axis["coordinates"]
        order = (1, 0, 2)
        component_axis["coordinates"] = [original[index] for index in order]
        document["conventions"]["point_diagonal_components"] = [1]
        source_spec = DatasetSpec(document)
        source_values = np.stack(
            [stored.read(time=time) for time in source_spec.coordinates("time")],
        )
        source_values = np.take(source_values, order, axis=1)
        source_id = self.helper.save("diagonal-component-view", source_spec, source_values)
        source = self.helper.index.open(source_id)
        self.addCleanup(source.close)

        view_document = source_spec.document
        component_axis = next(axis for axis in view_document["axes"]
                              if axis["name"] == "component")
        component_axis["coordinates"] = [component_axis["coordinates"][1]]
        view = DatasetView(source, DatasetSpec(view_document), subset_axes=("component",))
        elemental = DatasetElemental({"p2p": view}, point_dataset=self.points)

        diagonal_component = elemental.components[0]
        self.assertEqual(elemental.point_diagonal_components, (diagonal_component,))
        self.assertTrue(elemental.describe_block(
            diagonal_component, 0, 0, "p2p",
            point_indices={"point_left": (2, 0), "point_right": (2, 0)},
        ).point_diagonal)
        reordered = elemental.describe_block(
            diagonal_component, 0, 0, "p2p",
            point_indices={"point_left": (2, 0), "point_right": (0, 2)},
        )
        self.assertFalse(reordered.point_diagonal)

    def test_dataset_convention_requires_a_unique_in_range_integer_list(self):
        stored = self.readers["p2p"]
        invalid_values = ("zero", [True], [0, 0], [len(stored.spec.coordinates("component"))])
        for rows in invalid_values:
            with self.subTest(rows=rows):
                wrapper = self._with_diagonal_rows(stored, rows)
                with self.assertRaises((TypeError, ValueError)):
                    DatasetElemental({"p2p": wrapper}, point_dataset=self.points)


class _SparseRecordReader:
    def __init__(self, metadata, record):
        self.metadata = metadata
        self.record = record

    def describe_record(self, component, momentum, t):
        return self.metadata

    def read_record(self, component, momentum, t):
        return self.record


class SparsePointDiagonalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.component = ElementalComponent(0)
        dense = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
        self.dense_reader = ArrayElemental(
            {"v2v": dense}, (self.component,), phase_convention="anchor",
            point_diagonal_components=(self.component,),
        )
        self.values = np.ones((1, 3, 3), dtype=np.complex128)
        self.metadata = SparsePointRecordMetadata(
            "sparse", (3, 3), 1, np.dtype("complex128"), np.dtype("int64"),
            2 * np.dtype("int64").itemsize + 9 * np.dtype("complex128").itemsize,
            "sparse-data", "sparse-selection",
        )

    def reader(self, indices, *, point_diagonal=False):
        metadata = replace(self.metadata, point_diagonal=point_diagonal)
        record = {"type": "sparse", "indices": np.asarray(indices, dtype=np.int64),
                  "values": self.values}
        return SparseElemental(
            self.dense_reader, _SparseRecordReader(metadata, record),
        )

    def test_sparse_elemental_inherits_declaration_and_checks_coo_diagonal(self):
        reader = self.reader(((1, 1),))
        metadata = reader.describe_block(self.component, 0, 0, "p2p", usedNp=2)
        self.assertTrue(metadata.point_diagonal)
        self.assertTrue(reader.describe_sparse_record(self.component, 0, 0).point_diagonal)
        self.assertTrue(reader.block(self.component, 0, 0, "p2p")["type"] == "sparse")

        off_diagonal = self.reader(((0, 1),))
        with self.assertRaisesRegex(ValueError, "off-diagonal"):
            off_diagonal.block(self.component, 0, 0, "p2p")

    def test_sparse_record_diagonal_declaration_and_identity_record(self):
        metadata = replace(self.metadata, point_diagonal=True)
        record = {"type": "sparse", "indices": np.asarray(((2, 2),), dtype=np.int64),
                  "values": self.values}
        reader = SparseElemental(
            ArrayElemental({"v2v": np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)},
                           (self.component,), phase_convention="anchor"),
            _SparseRecordReader(metadata, record),
        )
        self.assertTrue(reader.describe_block(self.component, 0, 0, "p2p").point_diagonal)
        self.assertTrue(reader.block(self.component, 0, 0, "p2p")["type"] == "sparse")
        identity = SparsePointRecordMetadata(
            "identity", (3, 3), 3, np.dtype("complex128"), np.dtype("int64"),
            0, "identity-data", "identity-selection",
        )
        self.assertFalse(identity.point_diagonal)
        self.assertEqual(identity.structure, "general")
        declared_identity = replace(identity, structure="diagonal")
        self.assertTrue(declared_identity.point_diagonal)


if __name__ == "__main__":
    unittest.main()
