"""Small known-identity data for shared contract boundary tests."""

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetWriter


def specification(*, times=(6, 0), modes=(4, 1, 7), basis="actual-basis", dtype="complex128"):
    return DatasetSpec({
        "schema": 1,
        "kind": "contract-test",
        "configuration": {"ensemble": "fixture", "configuration": "cfg", "lattice": [1, 1, 1, 8]},
        "axes": [{"name": "time", "coordinates": list(times)}, {"name": "mode", "coordinates": list(modes)}],
        "dtype": dtype,
        "bases": {"mode": {"data_id": basis, "indices": list(modes)}},
        "definition": {"gauge_id": "actual-gauge", "tolerance": 1e-10},
        "conventions": {"normalization": "unscaled"},
        "parents": ["actual-parent"],
    })


def publish(directory, spec):
    with DatasetWriter(directory, spec, provenance={"producer": "contract-fixture"}) as writer:
        for index in writer.missing:
            time = spec.coordinates("time")[index]
            writer.write(index, np.asarray([100 * time + mode + 1j for mode in spec.coordinates("mode")], dtype=spec.document["dtype"]))
        return writer.publish()
