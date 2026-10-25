"""CPU memory admission for persisted raw-to-projected Elemental rows."""

import unittest

import numpy as np
import test_elemental_projection_dataset as projection_fixture

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.production import (
    elemental_projection_documents, estimate_elemental_projection_memory,
)
from lqcdweave.storage.dataset import DatasetIndex


def _fixture(test):
    value = projection_fixture.ElementalProjectionDatasetTests("runTest")
    value.setUp()
    test.addCleanup(value.doCleanups)
    return value


def _specs(value, raw_id, rows=None, used_ne=None):
    observation = value.index.observe_metadata(raw_id)
    requested, projected = elemental_projection_documents(
        observation["spec"], value.rows if rows is None else rows,
        raw_id=raw_id, usedNe=used_ne)
    return observation["spec"], requested, DatasetSpec(projected)


def _fresh_input_index(value, name, raw_directory):
    index = DatasetIndex(value.root / name)
    for directory in ("gauge", "basis", raw_directory):
        index.register(value.root / directory)
    return index


class ElementalProjectionBudgetTests(unittest.TestCase):
    def test_pure_model_hand_calculates_dtype_layout_recipe_and_lifetimes(self):
        value = _fixture(self)
        raw_id = value.raw()
        raw_spec, requested, projected_spec = _specs(value, raw_id, used_ne=2)
        self.assertEqual(requested.axis_names,
                         ("time", "component", "momentum", "sink_mode", "source_mode"))
        self.assertEqual(requested.coordinates("component"),
                         [dict(path=[]), dict(path=[0, 1]), dict(path=[3])])
        host, device = estimate_elemental_projection_memory(
            raw_spec, projected_spec, raw_id=raw_id, target="cpu",
            workspace_reserve_bytes=17, host_staging_bytes=23)
        # Raw parent B=3*4*2*3*3*16; one-time stack X=3*2*2*2*16;
        # projected Y=2*2*2*2*16; complex C=2*3*16.
        self.assertEqual(host.input_bytes, 3_456)
        self.assertEqual(host.snapshot_bytes, 0)
        self.assertEqual(host.intermediate_bytes, 384 + 96 + 256)
        self.assertEqual(host.output_bytes, 256)
        self.assertEqual(host.assembly_extra_bytes, 0)
        self.assertEqual(host.workspace_bytes, 40)
        self.assertEqual(host.logical_peak_bytes, 4_488)
        self.assertEqual(device.logical_peak_bytes, 0)

        # A component-first, reordered-time complex64 raw dataset uses the
        # same actual view, mode prefix, and duplicate path terms. Real-only
        # coefficients select the kernel's float64 coefficient matrix.
        with value.index.open(raw_id) as raw:
            values = np.stack([raw.read(time=time) for time in (2, 0, 1)])
            values = values.transpose(1, 2, 0, 3, 4)
            document = raw.spec.document
        by_name = {axis["name"]: axis for axis in document["axes"]}
        by_name["time"]["coordinates"] = [2, 0, 1]
        document["axes"] = [by_name[name] for name in
                            ("component", "momentum", "time", "sink_mode", "source_mode")]
        document["dtype"] = "complex64"
        complex64_id = value.save("component-first-c64", DatasetSpec(document),
                                  values.astype(np.complex64))
        real_rows = [
            dict(id=0, terms=[dict(path=[0, 1], coefficient=[1.5, 0]),
                              dict(path=[0, 1], coefficient=[-.5, 0]),
                              dict(path=[3], coefficient=[.25, 0])]),
            dict(id=1, terms=[dict(path=[], coefficient=[2, 0])]),
        ]
        c64_spec, c64_view, c64_output = _specs(
            value, complex64_id, real_rows, used_ne=2)
        self.assertEqual(c64_view.axis_names,
                         ("component", "momentum", "time", "sink_mode", "source_mode"))
        self.assertEqual(c64_view.coordinates("time"), [2, 0, 1])
        c64_host, _ = estimate_elemental_projection_memory(
            c64_spec, c64_output, raw_id=complex64_id,
            workspace_reserve_bytes=0, host_staging_bytes=0)
        # B=3*4*2*3*3*8, X=3*2*2*2*8, Y=256, real C=2*3*8.
        self.assertEqual(c64_host.input_bytes, 1_728)
        self.assertEqual(c64_host.intermediate_bytes, 192 + 48 + 256)
        self.assertEqual(c64_host.output_bytes, 256)
        self.assertEqual(c64_host.logical_peak_bytes, 2_480)
