"""Module tests split from test_perambulator_dataset.py; shared numerical fixtures stay unchanged."""

from test_perambulator_dataset import (
    DatasetSpec,
    _CONFIGURATION,
    _DIRAC,
    _PerambulatorDatasetFixture,
    propagator_dataset_document,
)


class PerambulatorDatasetTests(_PerambulatorDatasetFixture):
    def test_builder_default_scientific_document_and_registered_signature(self):
        document = propagator_dataset_document(
            dict(ensemble="test", configuration="cfg", lattice=[2, 1, 1, 3]),
            range(3), product="VSV", gauge_id="gauge-actual",
            basis_id="basis-actual", modes=(8, 3),
            dirac=dict(action="test-action", mass=.4), solver_tolerance=1e-11,
            temporal_boundary="antiperiodic")
        expected = {
            "schema": 1, "kind": "vsv",
            "configuration": dict(ensemble="test", configuration="cfg", lattice=[2, 1, 1, 3]),
            "axes": [
                dict(name="source_time", coordinates=[0, 1, 2]),
                dict(name="sink_time", coordinates=[0, 1, 2]),
                dict(name="sink_spin", coordinates=[0, 1, 2, 3]),
                dict(name="source_spin", coordinates=[0, 1, 2, 3]),
                dict(name="sink_mode", coordinates=[8, 3]),
                dict(name="source_mode", coordinates=[8, 3]),
            ],
            "dtype": "complex128",
            "bases": {
                "sink_mode": dict(data_id="basis-actual", indices=[8, 3]),
                "source_mode": dict(data_id="basis-actual", indices=[8, 3]),
            },
            "parents": ["basis-actual", "gauge-actual"],
            "definition": dict(gauge_id="gauge-actual", solver_tolerance=1e-11,
                               dirac=dict(action="test-action", mass=.4)),
            "conventions": dict(normalization="unscaled", spin_basis="degrand_rossi",
                                temporal_boundary="antiperiodic"),
        }
        self.assertEqual(DatasetSpec(document).document, expected)
        self.assertEqual(DatasetSpec(document).definition_id,
                         "fa2b46d0350a9f598fb5a4a3571565600b8a88c2c32684bfa8e3d9c6439c3e7f")
        explicit_equal = propagator_dataset_document(
            expected["configuration"], range(3), product="VSV",
            gauge_id="gauge-actual", basis_src_id="basis-actual", modes_src=(8, 3),
            basis_snk_id="basis-actual", modes_snk=(8, 3),
            dirac=expected["definition"]["dirac"], solver_tolerance=1e-11,
            temporal_boundary="antiperiodic")
        self.assertEqual(explicit_equal, expected)

    def test_builder_records_distinct_basis_roles_dtype_spin_and_normalization(self):
        document = propagator_dataset_document(
            _CONFIGURATION, range(3), product="VSV", gauge_id="gauge",
            basis_src_id="source-basis", modes_src=(8,),
            basis_snk_id="sink-basis", modes_snk=(11, 2),
            dirac=_DIRAC, solver_tolerance=1e-11, temporal_boundary="periodic",
            dtype="complex64", spin_basis="declared-spin", normalization="unscaled")
        spec = DatasetSpec(document)
        self.assertEqual(spec.document["dtype"], "complex64")
        self.assertEqual(spec.document["conventions"]["spin_basis"], "declared-spin")
        self.assertEqual(spec.document["bases"]["source_mode"],
                         dict(data_id="source-basis", indices=[8]))
        self.assertEqual(spec.document["bases"]["sink_mode"],
                         dict(data_id="sink-basis", indices=[11, 2]))
        self.assertEqual(spec.document["parents"], ["sink-basis", "source-basis", "gauge"])
