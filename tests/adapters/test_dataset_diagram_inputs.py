"""Published Dataset adapters for native Diagram inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter


def _axis(name, coordinates):
    return {"name": name, "coordinates": list(coordinates)}


def _publish(root, name, spec, values):
    path = Path(root) / name
    with DatasetWriter(path, spec, provenance={"test": "published-diagram-input"}) as writer:
        for index in range(spec.shape[0]):
            writer.write(index, values[index])
        identity = writer.publish()
    reader = DatasetReader(path)
    if reader.data_id != identity:
        reader.close()
        raise AssertionError("published dataset identity changed on read")
    return reader


def _configuration(lt):
    return {"ensemble": "diagram-fixture", "configuration": "cfg-7",
            "lattice": [2, 2, 2, lt]}


def _basis(indices, identity="laplace-basis-7"):
    return {"data_id": identity, "indices": list(indices)}


def _elemental_spec(times=(2, 0, 3, 1)):
    sink_modes = (2, 0, 1)
    source_modes = (1, 2, 0)
    return DatasetSpec({
        "schema": 1, "kind": "elemental_vv",
        "configuration": _configuration(4),
        "axes": [
            _axis("time", times),
            _axis("component", ({"path": []}, {"path": [0]})),
            _axis("momentum", ([0, 0, 0], [0, 0, 1])),
            _axis("sink_mode", sink_modes),
            _axis("source_mode", source_modes),
        ],
        "dtype": "complex128",
        "bases": {"sink_mode": _basis(sink_modes),
                  "source_mode": _basis(source_modes)},
        "definition": {"gauge_id": "gauge-7"},
        "conventions": {"phase": "anchor", "normalization": "unscaled"},
        "parents": ["laplace-basis-7", "gauge-7"],
    })


def _vsv_spec(axis_order, *, basis_id="laplace-basis-7", source_times=(2, 0, 1),
              sink_times=(1, 2, 0)):
    coordinates = {
        "source_time": source_times,
        "sink_time": sink_times,
        "sink_spin": range(4),
        "source_spin": range(4),
        "sink_mode": (2, 0, 1),
        "source_mode": (1, 2, 0),
    }
    return DatasetSpec({
        "schema": 1, "kind": "vsv",
        "configuration": _configuration(3),
        "axes": [_axis(name, coordinates[name]) for name in axis_order],
        "dtype": "complex128",
        "bases": {name: _basis(coordinates[name], basis_id)
                  for name in ("sink_mode", "source_mode")},
        "definition": {"gauge_id": "gauge-7"},
        "conventions": {"normalization": "unscaled",
                        "spin_basis": "degrand_rossi",
                        "temporal_boundary": "antiperiodic"},
        "parents": [basis_id, "gauge-7"],
    })


def _index(spec, axis, coordinate):
    return spec.coordinates(axis).index(coordinate)


def _direct_vsv(spec, values, sink, source, *, used_ne=None, mode_indices=None):
    selectors = {
        "source_time": _index(spec, "source_time", source),
        "sink_time": _index(spec, "sink_time", sink),
        "sink_spin": slice(None),
        "source_spin": slice(None),
        "sink_mode": slice(None),
        "source_mode": slice(None),
    }
    names = spec.axis_names
    value = values[tuple(selectors[name] for name in names)]
    kept = tuple(name for name in names if name not in ("source_time", "sink_time"))
    for axis in ("sink_mode", "source_mode"):
        selection = (tuple(mode_indices[axis]) if mode_indices is not None
                     else slice(0, used_ne) if used_ne is not None else slice(None))
        value = value[(slice(None),) * kept.index(axis) + (selection,)]
    if kept != ("sink_spin", "source_spin", "sink_mode", "source_mode"):
        value = np.transpose(value, tuple(kept.index(name) for name in
                                          ("sink_spin", "source_spin", "sink_mode", "source_mode")))
    return value


class DatasetDiagramInputTests(unittest.TestCase):
    def test_elemental_descriptions_are_metadata_only_and_slab_selects_ordered_times(self):
        spec = _elemental_spec()
        values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) + 1j).astype(np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "elemental", spec, values)
        self.addCleanup(reader.close)
        with self.subTest(published=True):
            elemental = DatasetElemental(reader)
            component = ElementalComponent.path(GaugePath())
            with patch.object(reader, "read", side_effect=AssertionError("describe performed I/O")):
                metadata = elemental.describe_slab(
                    component, 1, (3, 0, 2), "v2v", usedNe=2,
                )
                block_metadata = elemental.describe_block(
                    component, 1, 0, "v2v", usedNe=2,
                )
            self.assertEqual(metadata.axes,
                             ("coordinate_batch", "sink_mode", "source_mode"))
            self.assertEqual(metadata.shape, (3, 2, 2))
            self.assertEqual(metadata.data_identity, reader.data_id)
            self.assertIn("laplace-basis-7", metadata.selection_identity)
            self.assertEqual(block_metadata.shape, (2, 2))
            self.assertEqual(elemental.read_slab_workspace_bytes(metadata, usedNe=2),
                             2 * 2 * np.dtype(np.complex128).itemsize)

            selected = elemental.read_slab(component, 1, (3, 0, 2), "v2v", usedNe=2)
            expected = np.stack([
                values[_index(spec, "time", time), 0, 1, :2, :2]
                for time in (3, 0, 2)
            ])
            np.testing.assert_array_equal(selected, expected)
            self.assertFalse(np.shares_memory(selected, values))
            self.assertEqual(elemental.block(component, 1, 0, "v2v", usedNe=2).shape,
                             (2, 2))

    def test_elemental_missing_time_fails_in_description_without_reader_access(self):
        spec = _elemental_spec(times=(3, 0, 1))
        values = np.zeros(spec.shape, dtype=np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "elemental", spec, values)
        self.addCleanup(reader.close)
        elemental = DatasetElemental(reader)
        with patch.object(reader, "read", side_effect=AssertionError("describe performed I/O")):
            with self.assertRaisesRegex(KeyError, "time coordinate 2"):
                elemental.describe_block(
                    ElementalComponent.path(GaugePath()), 0, 2, "v2v",
                )

    def test_time_first_gathered_slabs_release_each_lane_scratch(self):
        spec = _elemental_spec()
        values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) + 1j).astype(np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "elemental", spec, values)
        self.addCleanup(reader.close)
        elemental = DatasetElemental(reader)
        component = ElementalComponent.path(GaugePath())
        original_read = reader.read
        call_kwargs, prior = [], []

        def tracked_read(*args, **kwargs):
            if prior:
                gc.collect()
                self.assertIsNone(prior[-1](), "previous selected lane remained live")
            value = original_read(*args, **kwargs)
            prior.append(weakref.ref(value))
            call_kwargs.append(kwargs)
            return value

        with patch.object(reader, "read", side_effect=tracked_read):
            slab = elemental.read_slab(
                component, 1, (3, 0, 2), "v2v",
                mode_indices={"sink_mode": (2, 1, 0), "source_mode": (2, 1, 0)},
            )
        self.assertEqual(len(call_kwargs), 3)
        for kwargs, time in zip(call_kwargs, (3, 0, 2), strict=True):
            self.assertEqual(kwargs["time"], time)
            self.assertEqual(kwargs["select"]["sink_mode"], (1, 0, 2))
            self.assertEqual(kwargs["select"]["source_mode"], (0, 2, 1))
        np.testing.assert_array_equal(slab, np.stack([
            values[_index(spec, "time", time), 0, 1]
            .take((2, 1, 0), axis=0).take((2, 1, 0), axis=1)
            for time in (3, 0, 2)
        ]))
        gc.collect()
        self.assertIsNone(prior[-1](), "last lane scratch remained live after slab return")
        metadata = elemental.describe_slab(
            component, 1, (3, 0, 2), "v2v",
            mode_indices={"sink_mode": (2, 1, 0), "source_mode": (2, 1, 0)},
        )
        self.assertEqual(elemental.read_slab_workspace_bytes(
            metadata, mode_indices={"sink_mode": (2, 1, 0), "source_mode": (2, 1, 0)},
        ), 2 * 3 * 3 * np.dtype(np.complex128).itemsize)
        self.assertEqual(elemental.read_slab_workspace_bytes(metadata),
                         2 * 3 * 3 * np.dtype(np.complex128).itemsize)

    def test_vsv_reader_handles_source_sink_axis_order_noncontiguous_times_and_prefix(self):
        layouts = (
            ("source_time", "sink_time", "sink_spin", "source_spin", "sink_mode", "source_mode"),
            ("sink_time", "source_time", "source_mode", "source_spin", "sink_mode", "sink_spin"),
        )
        for layout in layouts:
            with self.subTest(layout=layout):
                directory = TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                spec = _vsv_spec(layout)
                values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) + 3j).astype(np.complex128)
                reader = _publish(directory.name, "vsv", spec, values)
                self.addCleanup(reader.close)
                propagator = DatasetPropagator(reader)
                with patch.object(reader, "read", side_effect=AssertionError("describe performed I/O")):
                    metadata = propagator.describe_slab((0, 2), 1, usedNe=2)
                    one = propagator.describe_block(0, 1, usedNe=2)
                self.assertEqual(metadata.axes, (
                    "coordinate_batch", "sink_spin", "source_spin", "sink_mode", "source_mode"))
                self.assertEqual(metadata.shape, (2, 4, 4, 2, 2))
                self.assertEqual(metadata.data_identity, reader.data_id)
                self.assertIn("laplace-basis-7", metadata.selection_identity)
                self.assertEqual(one.shape, (4, 4, 2, 2))
                self.assertEqual(propagator.read_slab_workspace_bytes(metadata),
                                 2 * 4 * 4 * 3 * 3 * np.dtype(np.complex128).itemsize)

                expected_workspace = (
                    4 * 4 * 2 * 2 * np.dtype(np.complex128).itemsize
                    if layout[0] == "sink_time" else 0
                )
                self.assertEqual(propagator.read_slab_workspace_bytes(metadata, usedNe=2),
                                 expected_workspace)

                actual = propagator.read_slab((0, 2), 1, usedNe=2)
                expected = np.stack([
                    _direct_vsv(spec, values, sink, 1, used_ne=2) for sink in (0, 2)
                ])
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.shares_memory(actual, values))

    def test_sink_first_gathered_slab_releases_each_lane_scratch(self):
        layout = ("sink_time", "source_time", "source_mode", "source_spin",
                  "sink_mode", "sink_spin")
        spec = _vsv_spec(layout)
        values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) + 3j).astype(np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "vsv", spec, values)
        self.addCleanup(reader.close)
        propagator = DatasetPropagator(reader)
        selection = {"sink_mode": (2, 1, 0), "source_mode": (2, 1, 0)}
        original_read = reader.read
        call_kwargs, prior = [], []

        def tracked_read(*args, **kwargs):
            if prior:
                gc.collect()
                self.assertIsNone(prior[-1](), "previous selected lane remained live")
            value = original_read(*args, **kwargs)
            prior.append(weakref.ref(value))
            call_kwargs.append(kwargs)
            return value

        with patch.object(reader, "read", side_effect=tracked_read):
            slab = propagator.read_slab((0, 2), 1, mode_indices=selection)
        self.assertEqual(len(call_kwargs), 2)
        self.assertEqual(tuple(call["sink_time"] for call in call_kwargs), (0, 2))
        self.assertEqual(tuple(call["select"]["sink_mode"] for call in call_kwargs), ((1, 0, 2), (1, 0, 2)))
        self.assertEqual(tuple(call["select"]["source_mode"] for call in call_kwargs), ((0, 2, 1), (0, 2, 1)))
        metadata = propagator.describe_slab((0, 2), 1, mode_indices=selection)
        self.assertEqual(propagator.read_slab_workspace_bytes(metadata, mode_indices=selection),
                         2 * 4 * 4 * 3 * 3 * np.dtype(np.complex128).itemsize)
        expected = np.stack([
            _direct_vsv(spec, values, sink, 1, mode_indices=selection)
            for sink in (0, 2)
        ])
        np.testing.assert_array_equal(slab, expected)
        gc.collect()
        self.assertIsNone(prior[-1](), "last lane scratch remained live after slab return")

    def test_relative_vsv_coordinates_are_mapped_from_physical_sink_times(self):
        source_times, relative_times = (2, 0, 1), (2, 0, 1)
        base = _vsv_spec(
            ("source_time", "sink_time", "sink_spin", "source_spin", "sink_mode", "source_mode"),
            source_times=source_times, sink_times=relative_times,
        )
        doc = base.document
        doc["axes"][1]["name"] = "relative_sink_time"
        doc["conventions"]["time_convention"] = "relative"
        spec = DatasetSpec(doc)
        values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) + 2j).astype(np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "relative-vsv", spec, values)
        self.addCleanup(reader.close)
        propagator = DatasetPropagator(reader)
        sinks, source = (0, 2), 1
        expected_offsets = tuple((sink - source) % 3 for sink in sinks)
        actual = propagator.read_slab(sinks, source, usedNe=1)
        expected = np.stack([
            values[_index(spec, "source_time", source),
                   _index(spec, "relative_sink_time", offset), :, :, :1, :1]
            for offset in expected_offsets
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_missing_time_coverage_and_unknown_basis_are_rejected_without_reading(self):
        layout = ("source_time", "sink_time", "sink_spin", "source_spin", "sink_mode", "source_mode")
        spec = _vsv_spec(layout, source_times=(0, 2), sink_times=(1, 2))
        values = np.zeros(spec.shape, dtype=np.complex128)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "partial-vsv", spec, values)
        self.addCleanup(reader.close)
        propagator = DatasetPropagator(reader)
        with patch.object(reader, "read", side_effect=AssertionError("describe performed I/O")):
            with self.assertRaisesRegex(KeyError, "source_time coordinate 1"):
                propagator.describe_block(2, 1)
            with self.assertRaisesRegex(KeyError, "sink_time coordinate 0"):
                propagator.describe_block(0, 2)

        doc = _vsv_spec(layout).document
        doc["bases"]["source_mode"]["data_id"] = None
        unknown_basis = DatasetSpec(doc)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader = _publish(directory.name, "unknown-basis", unknown_basis,
                          np.zeros(unknown_basis.shape, dtype=np.complex128))
        self.addCleanup(reader.close)
        with self.assertRaisesRegex(ValueError, "known actual basis"):
            DatasetPropagator(reader)


if __name__ == "__main__":
    unittest.main()
