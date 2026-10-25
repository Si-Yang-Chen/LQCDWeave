"""Module tests split from test_legacy_readers.py; shared numerical fixtures stay unchanged."""

from test_legacy_readers import (
    ArrayBaryonElemental,
    ArrayElemental,
    ArrayPropagator,
    DatasetReader,
    DatasetSpec,
    BaryonComponent,
    ElementalComponent,
    GaugePath,
    IldgGauge,
    QDPTimesliceField,
    _LegacyReaderFixture,
    _axis,
    _qdp,
    _spec,
    np,
    register_binary,
    register_npy,
    register_npy_timeslices,
    struct,
)


class LegacyReaderTests(_LegacyReaderFixture):
    def test_all_propagator_sectors_and_time_conventions(self):
        shapes = {"vsv": (2, 2, 4, 4, 2, 2),
                  "psv": (2, 2, 4, 4, 3, 3, 2),
                  "vsp": (2, 2, 4, 4, 2, 4, 3),
                  "psp": (2, 2, 4, 4, 3, 3, 4, 3)}
        arrays = {name: np.arange(np.prod(shape)).reshape(shape).astype(np.complex128)
                  for name, shape in shapes.items()}
        with ArrayPropagator(arrays, time_convention="relative") as reader:
            for name, array in arrays.items():
                np.testing.assert_array_equal(reader.block(0, 1, name), array[1, 1])
            selected = reader.read_times([0], [1], "vsv", usedNe=1)
            self.assertEqual(selected.shape, (1, 1, 4, 4, 1, 1))
            np.testing.assert_array_equal(selected[0, 0], arrays["vsv"][1, 1, :, :, :1, :1])
            exact = reader.block(0, 1, "vsv", mode_indices={"sink_mode": [1, 0], "source_mode": [1]})
            np.testing.assert_array_equal(exact, arrays["vsv"][1, 1].take([1, 0], axis=2).take([1], axis=3))
            self.assertEqual([(sink, source) for sink, source, _ in reader.iter_blocks([0], [1])], [(0, 1)])
            with self.assertRaises(ValueError):
                reader.block(0, 1, "vsv", usedNe=3)
            psv = reader.block(0, 1, "psv", point_indices={"sink_point": [2, 0]},
                               mode_indices={"source_mode": [1]})
            np.testing.assert_array_equal(psv, arrays["psv"][1, 1].take([2, 0], axis=2).take([1], axis=4))
            vsp = reader.block(0, 1, "vsp", usedNp=2, usedNe=1)
            np.testing.assert_array_equal(vsp, arrays["vsp"][1, 1, :, :, :1, :2])
            psp = reader.block(0, 1, "psp", usedNp=(2, 3))
            np.testing.assert_array_equal(psp, arrays["psp"][1, 1, :, :, :2, :, :3])
            exact_psp = reader.block(0, 1, "psp", point_indices={
                "sink_point": [2, 0], "source_point": [3, 1]})
            np.testing.assert_array_equal(exact_psp,
                arrays["psp"][1, 1].take([2, 0], axis=2).take([3, 1], axis=4))
            with self.assertRaises(ValueError):
                reader.block(0, 1, "psp", point_indices={"sink_point": [1, 1]})
            with self.assertRaises(ValueError):
                reader.block(0, 1, "vsv", usedNp=1)
        with ArrayPropagator(arrays, time_convention="absolute") as reader:
            np.testing.assert_array_equal(reader.block(0, 1, "vsv"), arrays["vsv"][1, 0])
        npy, binary, slices = {}, {}, {}
        for name, array in arrays.items():
            npy[name] = self.root/f"{name}.npy"
            np.save(npy[name], array)
            binary[name] = self.root/f"{name}.bin"
            array.astype(">c16").tofile(binary[name])
            slices[name] = []
            for t in range(2):
                path = self.root/f"{name}.t{t:03d}.npy"
                np.save(path, array[t])
                slices[name].append(path)
        for reader in (ArrayPropagator.from_npy(npy, time_convention="relative"),
                       ArrayPropagator.from_binary(binary, shapes=shapes,
                                                   dtypes={name: ">c16" for name in shapes},
                                                   time_convention="relative"),
                       ArrayPropagator.from_npy_timeslices(slices, time_convention="relative")):
            with reader:
                for name, array in arrays.items():
                    np.testing.assert_array_equal(reader.block(0, 1, name), array[1, 1])
        slices["vsv"][0].unlink()
        with ArrayPropagator.from_npy_timeslices({"vsv": slices["vsv"]},
                time_convention="relative", source_times=[1]) as reader:
            np.testing.assert_array_equal(reader.block(0, 1, "vsv", usedNe=1),
                                          arrays["vsv"][1, 1, :, :, :1, :1])
            with self.assertRaises(KeyError):
                reader.block(0, 0, "vsv")
        with self.assertRaisesRegex(ValueError, "file size"):
            ArrayPropagator.from_binary({"vsv": binary["vsv"]}, shapes={"vsv": (2, 2, 4, 4, 1, 1)},
                                        dtypes={"vsv": ">c16"}, time_convention="relative")

    def test_binary_elemental_and_registered_dense_formats(self):
        component = ElementalComponent.path(GaugePath())
        values = (np.arange(8).reshape(1, 2, 2, 2, 1) + 1j).astype(np.complex128)
        path = self.root/"elemental.bin"
        values.astype(">c16").tofile(path)
        with ArrayElemental.from_binary({"v2v": path}, [component], shapes={"v2v": values.shape},
                dtypes={"v2v": ">c16"}, axis_order=("component", "momentum", "time"),
                phase_convention="anchor") as reader:
            np.testing.assert_array_equal(reader.block(component, 1, 0, "v2v"), values[0, 1, 0])
            np.testing.assert_array_equal(reader.read_times(component, 1, [1, 0], "v2v", usedNe=(1, 1)),
                                          values[0, 1, [1, 0], :1, :1])
            np.testing.assert_array_equal(reader.block(component, 1, 0, "v2v",
                mode_indices={"mode_left": [1], "mode_right": [0]}), values[0, 1, 0, 1:2, :1])
        spec = _spec()
        data = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(np.complex128)
        raw = self.root/"prop.bin"
        data.astype(">c16").tofile(raw)
        register_binary(self.root/"registered-binary", raw, spec, provenance={}, storage_dtype=">c16")
        with DatasetReader(self.root/"registered-binary") as reader:
            np.testing.assert_array_equal(reader.read(source_time=1, relative_sink_time=0), data[1, 0])
            subset = reader.read(source_time=1, relative_sink_time=0,
                                 select={"source_mode": [1], "sink_point": [0]})
            np.testing.assert_array_equal(subset, data[1, 0, :, :, 0:1, :, 1:2])
            self.assertFalse(subset.flags.writeable)
            self.assertEqual([time for time, _ in reader.iter_read([1],
                relative_sink_time=0, select={"source_mode": [1]})], [1])
        paths = []
        for t in range(2):
            file = self.root/f"prop.t{t:03d}.npy"
            np.save(file, data[t])
            paths.append(file)
        register_npy_timeslices(self.root/"registered-slices", paths, spec, provenance={})
        with DatasetReader(self.root/"registered-slices") as reader:
            np.testing.assert_array_equal(reader.read(source_time=1, relative_sink_time=0), data[1, 0])
        with raw.open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "file size"):
            DatasetReader(self.root/"registered-binary")

    def test_baryon_three_mode_subset(self):
        component = BaryonComponent.paths()
        values = np.arange(1*1*2*3*4*5).reshape(1, 1, 2, 3, 4, 5).astype(np.complex128)
        with ArrayBaryonElemental({"vvv": values}, [component]) as reader:
            selected = reader.block(component, 0, 1, "vvv", usedNe=(2, 3, 4))
            np.testing.assert_array_equal(selected, values[0, 0, 1, :2, :3, :4])
            exact = reader.block(component, 0, 1, "vvv", mode_indices={"mode0": [2, 0]})
            np.testing.assert_array_equal(exact, values[0, 0, 1].take([2, 0], axis=0))

    def test_registered_point_coordinates_select_both_ends(self):
        doc = _spec().document
        doc["kind"] = "psp"
        doc["axes"] = [_axis("source_time", 2), _axis("relative_sink_time", 2),
                       _axis("sink_spin", 4), _axis("source_spin", 4),
                       _axis("sink_point", 3), _axis("sink_color", 3),
                       _axis("source_point", 4), _axis("source_color", 3)]
        spec = DatasetSpec(doc)
        data = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(np.complex128)
        path = self.root/"registered-psp.npy"
        np.save(path, data)
        register_npy(self.root/"registered-psp", path, spec, provenance={})
        with DatasetReader(self.root/"registered-psp") as reader:
            value = reader.read(source_time=1, relative_sink_time=0,
                                select={"sink_point": [2, 0], "source_point": [3, 1]})
            np.testing.assert_array_equal(value, data[1, 0].take([2, 0], axis=2).take([3, 1], axis=4))

    def test_dense_elemental_point_subsets(self):
        component = ElementalComponent.path(GaugePath())
        shapes = {"v2p": (1, 1, 2, 2, 4, 3),
                  "p2v": (1, 1, 2, 4, 3, 2),
                  "p2p": (1, 1, 2, 4, 3, 5, 3)}
        arrays = {name: np.arange(np.prod(shape)).reshape(shape).astype(np.complex128)
                  for name, shape in shapes.items()}
        with ArrayElemental(arrays, [component], phase_convention="none") as reader:
            v2p = reader.block(component, 0, 1, "v2p", usedNe=1,
                               point_indices={"point_right": [3, 1]})
            np.testing.assert_array_equal(v2p, arrays["v2p"][0, 0, 1, :1].take([3, 1], axis=1))
            p2v = reader.block(component, 0, 1, "p2v", usedNp=2, usedNe=1)
            np.testing.assert_array_equal(p2v, arrays["p2v"][0, 0, 1, :2, :, :1])
            p2p = reader.block(component, 0, 1, "p2p", point_indices={
                "point_left": [2, 0], "point_right": [4, 1]})
            np.testing.assert_array_equal(p2p,
                arrays["p2p"][0, 0, 1].take([2, 0], axis=0).take([4, 1], axis=2))
            with self.assertRaises(ValueError):
                reader.block(component, 0, 1, "p2p", usedNp=2,
                             point_indices={"point_left": [1]})

    def test_ildg_and_qdp_gauge_and_vectors(self):
        gauge = np.arange(2*1*1*2*4*3*3, dtype=np.float64).reshape(2, 1, 1, 2, 4, 3, 3).astype(np.complex128)
        xml = b"<ildgFormat><lx>2</lx><ly>1</ly><lz>1</lz><lt>2</lt><precision>64</precision></ildgFormat>"
        lime = self.root/"cfg.lime"
        def record(name, payload):
            header = b"\x45\x67\x89\xab\x00\x01\0\0" + struct.pack(">Q", len(payload))
            header += name.encode().ljust(128, b"\0")
            return header + payload + b"\0" * (-len(payload) % 8)
        lime.write_bytes(record("ildg-format", xml) + record("ildg-binary-data", gauge.astype(">c16").tobytes()))
        np.testing.assert_array_equal(IldgGauge(str(self.root) + "/", suffix="cfg.lime").load("")[:], gauge)
        # Actual Chroma QIO files can append a newline outside the final LIME
        # record. This cannot reinterpret or truncate the binary payload.
        with (self.root / "cfg.lime").open("ab") as stream:
            stream.write(b"\n")
        np.testing.assert_array_equal(IldgGauge(str(self.root) + "/", suffix="cfg.lime").load("")[:], gauge)
        qdp_gauge = self.root/"gauge.mod"
        _qdp(qdp_gauge, {(t, mu): gauge[t, :, :, :, mu].astype(">c16")
                         for t in range(2) for mu in range(4)})
        gauge_reader = QDPTimesliceField(str(self.root) + "/", "gauge.mod", role="gauge").load("")
        np.testing.assert_array_equal(gauge_reader[:], gauge)
        np.testing.assert_array_equal(gauge_reader[1], gauge[1])
        np.testing.assert_array_equal(gauge_reader.read_times([1], records=[2]), gauge[1:2, :, :, :, 2:3])
        vectors = np.arange(2*2*1*1*2*3, dtype=np.float32).reshape(2, 2, 1, 1, 2, 3).astype(np.complex64)
        qdp_vectors = self.root/"vectors.mod"
        _qdp(qdp_vectors, {(t, mode): vectors[t, mode].astype(">c8")
                           for t in range(2) for mode in range(2)})
        handle = QDPTimesliceField(str(self.root) + "/", "vectors.mod", role="eigenvector", count=2).load("")
        self.assertEqual(handle.shape, vectors.shape)
        np.testing.assert_array_equal(handle[1, 0], vectors[1, 0])
        np.testing.assert_array_equal(handle.read_times([1], records=[1]), vectors[1:2, 1:2])
        np.testing.assert_array_equal(handle[1:2, 0], vectors[1:2, 0])
