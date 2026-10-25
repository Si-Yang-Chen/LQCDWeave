"""Module tests split from test_perambulator_dataset.py; shared numerical fixtures stay unchanged."""

from test_perambulator_dataset import (
    DatasetSpec,
    _CONFIGURATION,
    _DIRAC,
    _PerambulatorDatasetFixture,
    np,
    propagator_dataset_document,
    produce_source_rows,
)


class PerambulatorDatasetTests(_PerambulatorDatasetFixture):
    def test_script_consumes_borrowed_arrays_synchronously(self):
        spec = DatasetSpec(propagator_dataset_document(
            _CONFIGURATION, (0, 1), product="PSP", gauge_id="gauge",
            points_snk_id="points-sink", point_snk_labels=(6,),
            points_src_id="points-source", point_src_labels=(9,),
            dirac=_DIRAC, solver_tolerance=1e-11, temporal_boundary="periodic"))

        class Writer:
            missing = (0, 1)

            def __init__(self):
                self.spec = spec
                self.rows = []
                self.events = []

            def write(self, index, row):
                self.events.append(("write", index))
                self.rows.append(np.array(row, copy=True))

        class Generator:
            def __init__(self):
                self.value = np.zeros(spec.shape[1:], dtype=np.complex128)
                self.events = []
                self.inputs = []

            def load(self, key):
                self.events.append(("load", key))

            def calc(self, time, *, src, sink):
                self.events.append(("calc", time))
                self.inputs.append((src, sink))
                self.value.fill(time + 1)
                return [type("Result", (), {"product":"PSP", "values":self.value})()]

        writer, generator = Writer(), Generator()

        def to_host(value):
            writer.events.append(("to_host", None))
            return np.array(value, copy=True)

        src, sink = object(), object()
        produce_source_rows(writer, generator, (0, 1), "cfg-7", "PSP",src=src,sink=sink,
                                to_host=to_host)
        self.assertEqual(generator.events, [("load", "cfg-7"), ("calc", 0), ("calc", 1)])
        self.assertEqual(generator.inputs,[(src,[sink]),(src,[sink])])
        self.assertEqual(writer.events, [("to_host", None), ("write", 0),
                                        ("to_host", None), ("write", 1)])
        np.testing.assert_array_equal(writer.rows[0], 1)
        np.testing.assert_array_equal(writer.rows[1], 2)
        self.assertFalse(np.shares_memory(writer.rows[0], generator.value))
