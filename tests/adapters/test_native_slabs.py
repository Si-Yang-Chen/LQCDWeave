"""Native array/NPY slab APIs are metadata-only on describe and read once per slab."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.adapters.elemental import (
    ArrayElemental, LegacyGaugeLinkElemental,
)
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec, SpatialPointSet


class _RecordingArray(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values).view(cls)
        obj.advanced_index_shapes = []
        return obj

    def __array_finalize__(self, source):
        self.advanced_index_shapes = getattr(source, "advanced_index_shapes", [])

    def __getitem__(self, key):
        if isinstance(key, tuple) and any(isinstance(item, np.ndarray) for item in key):
            self.advanced_index_shapes.append(tuple(self.shape))
        return super().__getitem__(key)


class NativeSlabTests(unittest.TestCase):
    def test_propagator_describe_is_header_only_and_slab_preserves_order(self):
        rng = np.random.default_rng(521)
        data = rng.normal(size=(5, 5, 4, 4, 3, 2)) + 1j * rng.normal(
            size=(5, 5, 4, 4, 3, 2))
        data = data.astype(np.complex128)
        reader = ArrayPropagator({"vsv": data}, time_convention="absolute",
                                 data_identity="propagator-array")
        reader.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("describe/read called block"))

        metadata = reader.describe_slab([3, 1, 2], 4, "vsv", usedNe=2)
        self.assertEqual(metadata.axes,
                         ("coordinate_batch", "sink_spin", "source_spin", "sink_mode", "source_mode"))
        self.assertEqual(metadata.shape, (3, 4, 4, 2, 2))
        self.assertEqual(metadata.data_identity, "propagator-array")
        gathered = reader.read_slab([3, 1, 2], 4, "vsv", usedNe=2)
        expected = data[4].take([3, 1, 2], axis=0)[..., :2, :2]
        np.testing.assert_array_equal(gathered, expected)
        self.assertFalse(np.shares_memory(gathered, data))

    def test_propagator_absolute_and_relative_sink_slabs(self):
        lt, ne = 5, 2
        data = np.arange(lt * lt * 4 * 4 * ne * ne, dtype=np.float64).reshape(
            lt, lt, 4, 4, ne, ne).astype(np.complex128)
        absolute = ArrayPropagator({"vsv": data}, time_convention="absolute")
        selected = [4, 0, 2]
        actual = absolute.read_slab(selected, 1, "vsv", usedNe=1)
        np.testing.assert_array_equal(actual, data[1].take(selected, axis=0)[..., :1, :1])

        relative = ArrayPropagator({"vsv": data}, time_convention="relative")
        source = 1
        indices = tuple((sink - source) % lt for sink in selected)
        actual_relative = relative.read_slab(selected, source, "vsv", usedNe=1)
        np.testing.assert_array_equal(
            actual_relative, data[source].take(indices, axis=0)[..., :1, :1],
        )

    def test_large_backing_prefixes_are_sliced_before_one_joint_time_gather(self):
        lt, ne, np_count = 3, 128, 32
        backing = _RecordingArray(np.zeros(
            (lt, lt, 4, 4, np_count, 3, ne), dtype=np.complex64))
        reader = ArrayPropagator(
            {"psv": backing}, time_convention="absolute",
        )
        slab = reader.read_slab([2, 0], 1, "psv", usedNe=2, usedNp=3)
        self.assertEqual(slab.shape, (2, 4, 4, 3, 3, 2))
        self.assertEqual(backing.advanced_index_shapes, [(3, 4, 4, 3, 3, 2)])

    def test_two_discrete_p2p_point_axes_use_one_cartesian_gather(self):
        lt, nleft, nright = 3, 20, 24
        backing = _RecordingArray(np.zeros(
            (lt, lt, 4, 4, nleft, 3, nright, 3), dtype=np.complex64))
        reader = ArrayPropagator({"psp": backing}, time_convention="absolute")
        point_indices = {"sink_point": (17, 2), "source_point": (21, 1, 5)}
        slab = reader.read_slab([2, 0], 1, "psp", point_indices=point_indices)
        self.assertEqual(slab.shape, (2, 4, 4, 2, 3, 3, 3))
        self.assertEqual(backing.advanced_index_shapes,
                         [(3, nleft, nright, 4, 4, 3, 3)])

    def test_propagator_contiguous_slab_borrows_array_and_npy_slab_uses_same_api(self):
        lt = 4
        data = np.arange(lt * lt * 4 * 4, dtype=np.float64).reshape(
            lt, lt, 4, 4, 1, 1).astype(np.complex128)
        reader = ArrayPropagator({"vsv": data}, time_convention="absolute")
        borrowed = reader.read_slab([1, 2, 3], 0, "vsv")
        self.assertTrue(np.shares_memory(borrowed, data))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "peram.npy"
            np.save(path, data)
            with ArrayPropagator.from_npy({"vsv": path}, time_convention="absolute") as mapped:
                mapped.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("NPY slab called block"))
                metadata = mapped.describe_slab([1, 2], 0, "vsv")
                slab = mapped.read_slab([1, 2], 0, "vsv")
                self.assertEqual(metadata.shape, slab.shape)
                self.assertTrue(np.shares_memory(slab, mapped.sectors["vsv"]))
                np.testing.assert_array_equal(slab, data[0, 1:3])

            slice_paths = []
            for source_time in range(lt):
                path = Path(directory) / f"source-{source_time}.npy"
                np.save(path, data[source_time])
                slice_paths.append(path)
            with ArrayPropagator.from_npy_timeslices(
                    {"vsv": slice_paths}, time_convention="absolute") as sliced:
                sliced.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("timeslice slab called block"))
                slab = sliced.read_slab([1, 2], 3, "vsv")
                self.assertTrue(np.shares_memory(slab, sliced.sectors["vsv"][3]))
                np.testing.assert_array_equal(slab, data[3, 1:3])

    def test_elemental_slab_metadata_has_per_time_pool_ids_and_reads_once(self):
        component = ElementalComponent(0)
        values = np.arange(1 * 1 * 4 * 2 * 3 * 2 * 3).reshape(
            1, 1, 4, 2, 3, 2, 3).astype(np.complex128)
        pool_ids = {
            (component, 0, time, side): f"pool-{time}-{side}"
            for time in range(4) for side in ("bar", "field")
        }
        reader = ArrayElemental(
            {"p2p": values}, (component,), phase_convention="anchor",
            data_identity="elemental-array", sampling_pool_ids=pool_ids,
            sampling_groups={
                pool_id: SamplingPoolSpec(pool_id, 64, 3)
                for pool_id in pool_ids.values()
            },
        )
        reader.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("slab called block"))
        metadata = reader.describe_slab(component, 0, [2, 0, 2], "p2p", usedNp=2)
        self.assertEqual(metadata.axes,
                         ("coordinate_batch", "point_left", "color_left", "point_right", "color_right"))
        self.assertEqual(metadata.shape, (3, 2, 3, 2, 3))
        self.assertEqual(dict(metadata.point_pool_ids), {})
        self.assertEqual(tuple(dict(item) for item in metadata.point_pool_ids_by_batch), (
            {"point_left": "pool-2-bar", "point_right": "pool-2-field"},
            {"point_left": "pool-0-bar", "point_right": "pool-0-field"},
            {"point_left": "pool-2-bar", "point_right": "pool-2-field"},
        ))
        slab = reader.read_slab(component, 0, [2, 0, 2], "p2p", usedNp=2)
        expected = values[0, 0].take([2, 0, 2], axis=0)[:, :2, :, :2, :]
        np.testing.assert_array_equal(slab, expected)
        self.assertFalse(np.shares_memory(slab, values))

    def test_elemental_contiguous_view_and_npy_ordered_gather(self):
        component = ElementalComponent(0)
        canonical = np.arange(1 * 1 * 4 * 2 * 2).reshape(1, 1, 4, 2, 2).astype(np.complex128)
        memory = ArrayElemental(
            {"v2v": canonical}, (component,), phase_convention="anchor",
            data_identity="memory-elemental",
        )
        self.assertTrue(np.shares_memory(memory.read_slab(component, 0, [1, 2], "v2v"), canonical))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "elemental.npy"
            stored = canonical.transpose(2, 0, 1, 3, 4)
            np.save(path, stored)
            with ArrayElemental.from_npy(
                    {"v2v": path}, (component,),
                    axis_order=("time", "component", "momentum"),
                    phase_convention="anchor", data_identity="npy-elemental") as mapped:
                mapped.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("NPY slab called block"))
                self.assertEqual(mapped.describe_slab(component, 0, [3, 1], "v2v").shape,
                                 (2, 2, 2))
                slab = mapped.read_slab(component, 0, [3, 1], "v2v")
                np.testing.assert_array_equal(slab, canonical[0, 0].take([3, 1], axis=0))

    def test_legacy_gauge_link_slab_keeps_per_time_phase(self):
        component = ElementalComponent.path(GaugePath((0,)))
        lt, np_count, ne = 3, 4, 2
        raw = np.arange(1 * lt * ne * np_count * 3).reshape(
            1, lt, ne, np_count, 3).astype(np.complex128)
        coordinates = np.array([
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]],
            [[1, 0, 0], [2, 0, 0], [3, 1, 0], [0, 2, 1]],
            [[2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 1]],
        ])
        points = SpatialPointSet((4, 3, 2, lt), tuple(range(lt)), coordinates)
        reader = LegacyGaugeLinkElemental(
            {"v2p": raw}, (component,), points=points,
            momentum_list=((1, 1, 0),), phase_convention="midpoint",
        )
        times_selected = (2, 0)
        expected = np.stack([reader.block(component, 0, time, "v2p")
                             for time in times_selected])
        reader.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("slab called block"))
        metadata = reader.describe_slab(component, 0, times_selected, "v2p")
        slab = reader.read_slab(component, 0, times_selected, "v2p")
        self.assertEqual(metadata.shape, slab.shape)
        np.testing.assert_allclose(slab, expected, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
