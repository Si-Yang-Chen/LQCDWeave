"""Cheap validation paths and retained scientific source contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.source import normalize_local_source_declaration
from lqcdweave.storage import dataset as storage


def _spec(dtype="float64", count=5):
    return DatasetSpec(dict(schema=1, kind="validation-cost-fixture",
        configuration=dict(ensemble="test", configuration="validation-cfg",
                           lattice=[1, 1, 1, count]),
        axes=[dict(name="point", coordinates=list(range(count))),
              dict(name="component", coordinates=[0, 1])],
        dtype=dtype, bases={}, definition={}, conventions={}, parents=[]))


class DatasetValidationCostTests(unittest.TestCase):
    def test_only_inexact_output_dtypes_scan_finiteness(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dtype in ("int32", "int64", "float32", "float64", "complex64", "complex128"):
                with self.subTest(dtype=dtype):
                    with storage.DatasetWriter(root / dtype, _spec(dtype), provenance={}) as writer:
                        with patch.object(np, "isfinite", wraps=np.isfinite) as finite:
                            writer.write(0, np.asarray([1, 2], dtype=dtype))
                        self.assertEqual(finite.call_count,
                                         int(np.dtype(dtype).kind in "fc"))
                        if np.dtype(dtype).kind in "fc":
                            for bad in (np.nan, np.inf):
                                with self.assertRaisesRegex(ValueError, "nonfinite"):
                                    writer.write(1, np.asarray([1, bad], dtype=dtype))

    def test_producer_parameters_are_detached_at_public_binding(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data"
            context = dict(run_id="run", producer_attempt_id=None, stage="test")
            with storage.DatasetWriter(directory, _spec(count=1), provenance={},
                                       attempt_context=context) as writer:
                parameters = {"solver": {"iterations": 8, "options": [1, 2, 3]}}
                writer.bind_producer_attempt("producer", execution_parameters=parameters)
                parameters["solver"]["options"].append(4)
                writer.write(0, np.asarray([1.0, 2.0]))
                writer.publish()
            saved = storage.read_dataset_metadata(directory)["execution"]
            self.assertEqual(saved["attempts"][0]["execution_parameters"]["solver"]["options"],
                             [1, 2, 3])

    def test_local_spatial_source_requires_its_graph_attempt_and_records_origin(self):
        declaration = normalize_local_source_declaration("sample_spatial_points", "a" * 64)
        lattice = [1, 1, 1, 1]
        spec = DatasetSpec(dict(schema=1, kind="spatial_points", dtype="int64",
            configuration=dict(ensemble="e", configuration="1", lattice=lattice),
            axes=[dict(name="time", coordinates=[0]),
                  dict(name="point", coordinates=[0]),
                  dict(name="xyz", coordinates=["x", "y", "z"])], bases={},
            definition=dict(realization=dict(schema=1, kind="local_spatial_sampling",
                design=dict(lattice_size=lattice, num_points=1, temporal_policy="same"),
                seed=0, algorithm=declaration["algorithm"])),
            conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
            parents=[]))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "points graph stage"):
                storage.DatasetWriter(root / "bad", spec, provenance={}, attempt_context=dict(
                    run_id=None, producer_attempt_id=None, stage=None,
                    local_source_declaration=declaration))
            context = dict(run_id="run", producer_attempt_id=None, stage="points",
                           local_source_declaration=declaration)
            with storage.DatasetWriter(root / "good", spec, provenance={},
                                       attempt_context=context) as writer:
                writer.write(0, np.zeros((1, 3), dtype=np.int64))
                with self.assertRaisesRegex(ValueError, "producer_attempt_id"):
                    writer.publish()
                writer.bind_producer_attempt("producer")
                data_id = writer.publish()
            metadata = storage.read_dataset_metadata(root / "good", cfg="1")
            self.assertEqual(metadata["data_id"], data_id)
            self.assertEqual(metadata["source_origin"]["status"], "recorded")
            self.assertEqual(metadata["source_origin"]["origin"]["operation"],
                             "sample_spatial_points")

    def test_spec_layout_cache_keeps_public_documents_detached(self):
        spec = _spec()
        expected = spec.definition_id
        with patch("lqcdweave.contracts.dataset.json.loads", wraps=json.loads) as loads:
            for _ in range(10):
                self.assertEqual(spec.shape, (5, 2))
                self.assertEqual(spec.axis_names, ("point", "component"))
                self.assertEqual(spec.definition_id, expected)
            self.assertEqual(loads.call_count, 0)
        spec.coordinates("point").clear()
        spec.document["axes"].clear()
        self.assertEqual(spec.coordinates("point"), list(range(5)))


if __name__ == "__main__":
    unittest.main()
