"""Explicit external source records and project-scoped graph receipts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from production.elemental.test_elemental import fixture
from lqcdweave.contracts.dataset import (
    DatasetRequest, DatasetSpec, canonical_json,
)
import lqcdweave.storage.dataset as dataset_storage
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader,
    read_json_records, register_binary, register_npy, register_npy_timeslices,
    write_json_records,
)
from lqcdweave.runtime.datasets import (
    apply_source_policy, read_dataset_request_receipt,
    save_dataset_request_receipt,
)
from lqcdweave.production.elemental.planning import (
    plan_indexed_point_elementals,
)


_ORIGIN_NAMES = ("source_system", "dataset_id", "revision", "producer",
                 "producer_revision")


def import_provenance(*, revision="revision-1", unknown=(), runtime=None,
                      source_system="facility-archive", dataset_id="ensemble/cfg-1",
                      producer="external-solver", producer_revision="solver-7"):
    unknown = set(unknown)
    origin = {}
    values = {
        "source_system": source_system,
        "dataset_id": dataset_id,
        "revision": revision,
        "producer": producer,
        "producer_revision": producer_revision,
    }
    for name in _ORIGIN_NAMES:
        origin[name] = ({"status": "unknown", "reason": f"{name} omitted by source"}
                        if name in unknown else
                        {"status": "known", "value": values[name]})
    return dict(schema=1, kind="external_import", origin=origin,
                registration_runtime={"status": "known", "value":
                    {"python": "3.13", "backend": "numpy-cpu"}
                    if runtime is None else runtime})


def register_known_copy(index, actual_id, directory, *, parents=None, transform=None):
    """Re-register a fully verified fixture dataset as an explicit test source.

    The stored scientific spec and bytes are copied through the public reader.
    Optional parents are explicit actual IDs supplied by the caller, never
    inferred from the fixture label or storage path.
    """
    directory = Path(directory)
    with index.open(actual_id) as reader:
        document = json.loads(canonical_json(reader.spec.document))
        if parents is not None:
            document["parents"] = list(parents)
        if transform is not None:
            transform(document)
        spec = DatasetSpec(document)
        first_axis = spec.axis_names[0]
        values = np.stack([
            reader.read(**{first_axis: coordinate})
            for coordinate in reader.spec.coordinates(first_axis)
        ])
    source = directory.with_name(directory.name + ".source.npy")
    np.save(source, values, allow_pickle=False)
    provenance = import_provenance(
        source_system="synthetic-test-fixture",
        dataset_id="small-lattice-reference/cfg-1",
        producer="deterministic-test-fixture",
        producer_revision="fixture-v1",
    )
    identity = register_npy(directory, source, spec, provenance=provenance,
                           provenance_mode="external_import")
    index.register(directory)
    return identity


def small_spec(*, kind="test_field", parents=(), definition=None):
    return DatasetSpec(dict(schema=1, kind=kind, dtype="float64",
        configuration=dict(ensemble="source-fixture", configuration="cfg-1",
                           lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[0, 1]),
              dict(name="component", coordinates=[0, 1])],
        bases={}, definition={} if definition is None else definition,
        conventions={}, parents=list(parents)))


def point_spec(sizes, labels, *, parents=(), definition=None):
    return DatasetSpec(dict(schema=1, kind="spatial_points", dtype="int64",
        configuration=dict(ensemble="synthetic", configuration="cfg",
                           lattice=list(sizes)),
        axes=[dict(name="time", coordinates=list(range(sizes[3]))),
              dict(name="point", coordinates=list(labels)),
              dict(name="xyz", coordinates=["x", "y", "z"])],
        bases={}, definition={} if definition is None else definition,
        conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
        parents=list(parents)))


class _ExternalSourceRegistrationFixture(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = small_spec()
        self.source = self.root / "input.npy"
        self.values = np.arange(4, dtype=np.float64).reshape(2, 2)
        np.save(self.source, self.values, allow_pickle=False)


class ProjectSourceWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, links, vectors, points = fixture()
        self.cfg = dict(ensemble="synthetic", configuration="cfg",
                        lattice=list(self.sizes))
        provenance = import_provenance()
        gauge_spec = DatasetSpec(dict(schema=1, kind="gauge", dtype="complex128",
            configuration=self.cfg,
            axes=[dict(name="time", coordinates=list(range(3))),
                  dict(name="z", coordinates=list(range(2))),
                  dict(name="y", coordinates=list(range(2))),
                  dict(name="x", coordinates=list(range(3))),
                  dict(name="direction", coordinates=list(range(4))),
                  dict(name="color_row", coordinates=list(range(3))),
                  dict(name="color_column", coordinates=list(range(3)))],
            bases={}, definition={"dataset_id": "this-is-not-an-dataset-id"},
            conventions={}, parents=[]))
        basis_spec = DatasetSpec(dict(schema=1, kind="explicit_basis", dtype="complex128",
            configuration=self.cfg,
            axes=[dict(name="time", coordinates=list(range(3))),
                  dict(name="mode", coordinates=[4, 1, 9]),
                  dict(name="z", coordinates=list(range(2))),
                  dict(name="y", coordinates=list(range(2))),
                  dict(name="x", coordinates=list(range(3))),
                  dict(name="color", coordinates=list(range(3)))],
            bases={}, definition={}, conventions={}, parents=[]))
        point_coordinates = points.copy()
        point_labels = [8, 2, 5]
        self.gauge_id = self._register("gauge", gauge_spec, links, provenance)
        self.basis_id = self._register("basis", basis_spec, vectors, provenance)
        points_spec = point_spec(self.sizes, point_labels)
        self.points_id = self._register("points", points_spec,
            np.asarray(point_coordinates, dtype=np.int64), provenance)
        self.paths = ((), (0,))
        self.momenta = ((0, 0, 0), (1, 0, 0))

    def _register(self, name, spec, values, provenance):
        source = self.root / f"{name}.npy"
        np.save(source, values, allow_pickle=False)
        directory = self.root / name
        identity = register_npy(directory, source, spec, provenance=provenance,
                                provenance_mode="external_import")
        self.index.register(directory)
        return identity



    def test_unknown_legacy_source_blocks_only_explicit_strict_policy(self):
        legacy_directory = self.root/"legacy-points"
        legacy_id = register_npy(legacy_directory, self.root/"points.npy",
            point_spec(self.sizes, [8, 2, 5]), provenance={"producer": "old importer"})
        self.index.register(legacy_directory)
        options = dict(paths=self.paths, momentum_list=self.momenta,
                       sectors=("p2p",), usedNp=2, phase_convention="none")
        strict = plan_indexed_point_elementals(self.index, self.gauge_id, None, legacy_id,
            source_policy="require_known", **options)
        self.assertEqual(strict["action"], "blocked")
        allowed = plan_indexed_point_elementals(self.index, self.gauge_id, None, legacy_id,
            **options)
        self.assertNotEqual(allowed.get("action"), "blocked")



    def test_existing_blocked_graph_is_preserved_without_source_policy_reason(self):
        blocked = dict(action="blocked", reasons=("an existing graph input is blocked",),
                       stages={name: {"action": "blocked",
                                      "data_id": self.gauge_id if name == "gauge" else None}
                               for name in ("gauge", "points", "p2p")})
        with self.index.open(self.gauge_id) as gauge:
            known_request = DatasetRequest(gauge.spec)
        strict = apply_source_policy(blocked, {"gauge": known_request}, self.index,
                                     source_policy="require_known")
        self.assertEqual(strict["action"], "blocked")
        self.assertEqual(strict["reasons"], blocked["reasons"])
        self.assertNotIn("source_reasons", strict)




if __name__ == "__main__":
    unittest.main()
