"""Independent byte/layout fixtures for old data formats and propagator sectors."""

from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental, LegacyGaugeLinkElemental
from lqcdweave.adapters.legacy_fields import DenseFileHandle, IldgGauge, QDPTimesliceField
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent, GaugePath
from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.production.elemental import ElementalGenerator
from lqcdweave.storage.dataset import DatasetReader, register_binary, register_npy, register_npy_timeslices
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


def _axis(name, count):
    return dict(name=name, coordinates=list(range(count)))


def _spec():
    return DatasetSpec(dict(schema=1, kind="psv", configuration=dict(
        ensemble="toy", configuration="cfg", lattice=[2, 1, 1, 2]),
        axes=[_axis("source_time", 2), _axis("relative_sink_time", 2),
              _axis("sink_spin", 4), _axis("source_spin", 4),
              _axis("sink_point", 1), _axis("sink_color", 3), _axis("source_mode", 2)],
        dtype="complex128", bases={}, definition={}, conventions={}, parents=[]))


def _qdp(path, records, lattice=(2, 1, 1, 2)):
    def string(value):
        value = value.encode()
        return struct.pack(">i", len(value)) + value
    xml = f"<meta><lattSize>{' '.join(map(str, lattice))}</lattSize><decay_dir>3</decay_dir></meta>"
    content = bytearray(string("XXXXQDPLazyDiskMapObjFileXXXX") + struct.pack(">i", 1) + string(xml))
    pointer = len(content)
    content += b"\0" * 16
    offsets = {}
    for key, array in records.items():
        offsets[key] = len(content)
        content += np.asarray(array).tobytes()
    table = len(content)
    content += struct.pack(">I", len(records))
    for key, offset in offsets.items():
        content += struct.pack(">i", 8) + struct.pack(">ii", *key) + struct.pack(">qq", 0, offset)
    content[pointer:pointer+16] = struct.pack(">qq", 0, table)
    path.write_bytes(content)


class _LegacyReaderFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class LegacyReaderTests(_LegacyReaderFixture):
    def test_legacy_current_v2p_p2v_phased_at_consumption(self):
        components = [ElementalComponent.path(GaugePath()),
                      ElementalComponent.path(GaugePath((0,)))]
        v2p = (np.arange(3*2*4*5*3).reshape(3, 2, 4, 5, 3) + 1j).astype(np.complex128)
        p2v = (np.arange(3*2*5*3*4).reshape(3, 2, 5, 3, 4) - 2j).astype(np.complex128)
        files = {"v2p": self.root/"cfg_v2p.npy", "p2v": self.root/"cfg_p2v.npy"}
        np.save(files["v2p"], v2p)
        np.save(files["p2v"], p2v)
        coordinates = np.array([[(0, 0, 0), (1, 0, 0), (2, 1, 0),
                                 (3, 2, 1), (0, 1, 1)] for _ in range(3)])
        coordinates[:, :, 0] = (coordinates[:, :, 0] + np.arange(3)[:, None]) % 4
        points = SpatialPointSet((4, 3, 2, 3), (0, 1, 2), coordinates)

        with LegacyGaugeLinkElemental.from_npy(files, components, points=points,
                momentum_list=((0, 0, 0), (1, 1, 0))) as reader:
            self.assertEqual((reader.num_momentum, reader.Lt), (2, 3))
            self.assertTrue(np.shares_memory(reader.sectors["v2p"], reader._maps[0]))
            np.testing.assert_array_equal(reader.block(components[1], 0, 2, "v2p"), v2p[2, 1])
            xyz = coordinates[2].astype(float)
            p2v_phase = np.exp(2j*np.pi*((xyz[:, 0]+0.5)/4 + xyz[:, 1]/3))
            v2p_phase = np.exp(2j*np.pi*((xyz[:, 0]-0.5)/4 + xyz[:, 1]/3))
            np.testing.assert_allclose(reader.block(components[1], 1, 2, "p2v"),
                                       p2v[2, 1] * p2v_phase[:, None, None])
            np.testing.assert_allclose(reader.block(components[1], 1, 2, "v2p"),
                                       v2p[2, 1] * v2p_phase[None, :, None])
            np.testing.assert_allclose(reader.block(components[1], 1, 2, "v2p",
                usedNe=2, point_indices={"point_right": [4, 1]}),
                v2p[2, 1, :2].take([4, 1], axis=1) * v2p_phase[[4, 1]][None, :, None])
            operator = Operator.from_terms("current", [OperatorTerm(0, components[1], 1)])
            np.testing.assert_allclose(list(Meson(reader, operator).factors(2, "p2v"))[0].spatial,
                                       p2v[2, 1] * p2v_phase[:, None, None])
            np.testing.assert_allclose(list(Meson(reader, operator, source=True).factors(2, "v2p"))[0].spatial,
                                       (p2v[2, 1] * p2v_phase[:, None, None]).conj().transpose(2, 0, 1))
            np.testing.assert_allclose(reader.read_times(components[0], 1, [2, 0], "p2v"),
                                       np.stack([reader.block(components[0], 1, t, "p2v") for t in (2, 0)]))
            with self.assertRaises(KeyError):
                reader.block(components[0], 2, 0, "v2p")

        with self.assertRaisesRegex(ValueError, "axis_order"):
            ArrayElemental.from_npy({"v2v": files["v2p"]}, components,
                                    axis_order=("time", "component"), phase_convention="none")
        with self.assertRaisesRegex(ValueError, "spatial GaugePath"):
            LegacyGaugeLinkElemental.from_npy(files, [components[0], ElementalComponent("row")],
                                              points=points, momentum_list=((0, 0, 0),))
        with self.assertRaisesRegex(ValueError, "spatial GaugePath"):
            LegacyGaugeLinkElemental.from_npy(files, [components[0], ElementalComponent.path(GaugePath((0, 3)))],
                                              points=points, momentum_list=((0, 0, 0),))
        shorter_points = SpatialPointSet((4, 3, 2, 3), (0, 1, 2), coordinates[:, :4])
        with self.assertRaisesRegex(ValueError, "point axis"):
            LegacyGaugeLinkElemental.from_npy(files, components, points=shorter_points,
                                              momentum_list=((0, 0, 0),))

    def test_legacy_gaugelink_phase_matches_direct_production(self):
        sizes = (4, 2, 1, 2)
        lx, ly, lz, lt = sizes
        rng = np.random.default_rng(71)
        gauge = np.broadcast_to(np.eye(3, dtype=complex),
                                (lt, lz, ly, lx, 4, 3, 3)).copy()
        vectors = rng.normal(size=(lt, 2, lz, ly, lx, 3)) + 1j*rng.normal(size=(lt, 2, lz, ly, lx, 3))
        points = SpatialPointSet(sizes, tuple(range(lt)), np.array([
            [(0, 0, 0), (2, 1, 0), (3, 0, 0)],
            [(1, 1, 0), (2, 0, 0), (3, 1, 0)],
        ]))
        paths = [GaugePath(), GaugePath((0,)), GaugePath((3,))]
        components = [ElementalComponent.path(path) for path in paths]
        momenta = ((0, 0, 0), (1, -1, 0))

        class Handle:
            def __init__(self, array):
                self.array = array

            def load(self, key):
                return self.array

        for convention in ("midpoint", "anchor"):
            generator = ElementalGenerator(sizes, Handle(gauge), Handle(vectors),
                point=points, paths=paths, path_version="v2", momentum_list=momenta,
                phase_convention=convention, ne_block_size=2).load("cfg")
            slabs = []
            for t in range(lt):
                produced = generator.calc_path_blocks(t, ("v2p", "p2v"))
                slabs.append({sector: produced[sector].copy() for sector in ("v2p", "p2v")})
            files = {}
            expected = {}
            for sector in ("v2p", "p2v"):
                values = np.stack([slab[sector] for slab in slabs])
                files[sector] = self.root/f"{convention}.{sector}.npy"
                np.save(files[sector], values[:, :, 0])
                expected[sector] = values[:, :, 1]
            with LegacyGaugeLinkElemental.from_npy(files, components, points=points,
                    momentum_list=momenta, phase_convention=convention) as reader:
                for t in range(lt):
                    for pi, component in enumerate(components):
                        for sector in files:
                            np.testing.assert_allclose(reader.block(component, 1, t, sector),
                                                       expected[sector][t, pi], atol=2e-13, rtol=2e-13)

    def test_reference_weak_field_ildg_and_propagator_layout(self):
        reference = Path(__file__).parent/"data"/"weak_field"
        ildg = IldgGauge(str(reference/"weak_field")).load("")
        raw = np.load(reference/"weak_field.npy", mmap_mode="r")
        np.testing.assert_array_equal(ildg, raw.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6))
        for format, suffix, dtype in (("npy", ".npy", "<c16"),
                                      ("binary", ".bin", "<c16")):
            field = DenseFileHandle(str(reference/"weak_field"), suffix,
                                    format=format, shape=raw.shape, dtype=dtype,
                                    reshape=(4, 8, 4, 4, 4, 3, 3),
                                    transpose=(1, 2, 3, 4, 0, 5, 6)).load("")
            np.testing.assert_array_equal(field, ildg)
        file = reference/"weak_field.perambulator.npy"
        with ArrayPropagator.from_npy({"vsv": file}, time_convention="relative") as reader:
            self.assertEqual(reader.Lt, 8)
            np.testing.assert_array_equal(reader.block(0, 1), np.load(file, mmap_mode="r")[1, 7])
        vectors = DenseFileHandle(str(reference/"weak_field"), ".eigenvector.input.npy",
                                  format="npy", shape=(8, 20, 4, 4, 4, 3),
                                  dtype="<c16", Ne=20)
        selected = vectors.read_modes("", 3, 2, 5)
        np.testing.assert_array_equal(selected,
            np.load(reference/"weak_field.eigenvector.input.npy",mmap_mode="r")[3,2:5])
        self.assertFalse(selected.flags.writeable)
        with self.assertRaisesRegex(ValueError,"canonical time or mode"):
            vectors.read_modes("",3,2,21)
        momenta = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1),
                   (0, 0, 2), (0, 1, 2), (1, 1, 2))
        generator = ElementalGenerator((4, 4, 4, 8), IldgGauge(str(reference/"weak_field")),
                                       vectors, num_nabla=2, momentum_list=momenta,
                                       calc_mode="calc_deriv", ne_block_size=20).load("")
        npy_gauge = DenseFileHandle(str(reference/"weak_field"), ".npy", format="npy",
                                    shape=raw.shape, dtype="<c16",
                                    reshape=(4, 8, 4, 4, 4, 3, 3),
                                    transpose=(1, 2, 3, 4, 0, 5, 6))
        reference_generator = ElementalGenerator((4, 4, 4, 8), npy_gauge, vectors,
                                                 num_nabla=2, momentum_list=momenta,
                                                 calc_mode="calc_deriv", ne_block_size=20).load("")
        np.testing.assert_allclose(generator.calc_path_blocks(0)["v2v"], reference_generator.calc_path_blocks(0)["v2v"], atol=2e-12, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
