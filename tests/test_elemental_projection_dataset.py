"""Raw spatial path derivation, independent site reference and exact recovery."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from production.elemental.test_elemental import fixture, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader, DatasetWriter, inspect_dataset, register_npy,
)
from lqcdweave.production.elemental.planning import plan_indexed_elemental_projection, prepare_elemental_stage
from fixtures.elemental_script import project_elemental
import fixtures.elemental_script as workflow
from lqcdweave.runtime.datasets import open_dataset_view

from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class ElementalProjectionDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, self.links, self.vectors, _ = fixture()
        cfg = dict(ensemble="test", configuration="cfg", lattice=list(self.sizes))
        gauge_axes = [dict(name=name, coordinates=list(range(extent))) for name, extent in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"), self.links.shape)]
        gauge_spec = DatasetSpec(dict(schema=1, kind="gauge", configuration=cfg, axes=gauge_axes,
            dtype="complex128", bases={}, parents=[], definition={}, conventions=dict(boundary="periodic")))
        self.gauge_id = self.save("gauge", gauge_spec, self.links)
        basis_axes = [dict(name=name, coordinates=values) for name, values in zip(
            ("time", "mode", "z", "y", "x", "color"),
            ([0, 1, 2], [4, 1, 9], [0, 1], [0, 1], [0, 1, 2], [0, 1, 2]))]
        self.basis_id = self.save("basis", DatasetSpec(dict(schema=1, kind="explicit_basis",
            configuration=cfg, axes=basis_axes, dtype="complex128", bases={}, parents=[],
            definition={}, conventions={})), self.vectors)
        self.paths = ((), (0, 1), (3,), (1,))  # The last stored path is not requested.
        self.momenta = ((1, -1, 0), (0, 0, 0))
        self.rows = [dict(id=0, terms=[dict(path=[0, 1], coefficient=[.3, .8]),
                                     dict(path=[3], coefficient=[-.2, .7]),
                                     dict(path=[0, 1], coefficient=[.1, -.6])]),
                     dict(id=1, terms=[dict(path=[], coefficient=[2, 0]),
                                      dict(path=[3], coefficient=[0, -.4])])]

    def save(self, name, spec, values):
        source = self.root / (name + ".npy")
        np.save(source, values)
        identity = register_npy(self.root / name, source, spec, provenance=dict(test="independent"))
        self.index.register(self.root / name)
        return identity

    def assert_unpublished(self, directory):
        directory = Path(directory)
        self.assertNotEqual(inspect_dataset(directory, cfg="cfg")["state"], "complete")
        self.assertFalse((directory / "cfg.npy").exists())
        self.assertFalse((directory / "cfg.h5").exists())

    def raw(self, phase="midpoint"):
        output = self.root / ("raw-" + phase)
        with DatasetReader(self.root / "gauge") as gauge, \
             DatasetReader(self.root / "basis") as basis:
            generator, spec, provenance, times, cfg = prepare_elemental_stage(
                gauge, basis, paths=self.paths, momentum_list=self.momenta,
                phase_convention=phase)
            generator.load(cfg["configuration"])
            with DatasetWriter(output, spec, provenance=provenance) as writer:
                for slot in writer.missing:
                    writer.write(slot, generator.calc_path_blocks(times[slot])["v2v"])
                identity = writer.publish()
        self.index.register(output)
        return identity

    def project(self, identity, name, rows=None, **options):
        requested_rows = self.rows if rows is None else rows
        preview = plan_indexed_elemental_projection(
            self.index, identity, requested_rows, **options)
        spec = preview["stages"]["projected"]["spec"]
        output_root = self.root / name
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / "projected"
        with open_dataset_view(
                self.index, preview["stages"]["raw"]["spec"],
                subset_axes=("component", "sink_mode", "source_mode"),
                data_id=identity) as raw, DatasetWriter(
                destination, spec, provenance={"test": "public projection writer"}) as writer:
            project_elemental(raw, writer)
            projected_id = writer.publish()
        self.assertEqual(self.index.register(destination), projected_id)
        return {"datasets": {"raw": identity, "projected": projected_id}}

    def assert_reference(self, result, rows=None, ne=3, phase="midpoint", atol=2e-12):
        rows = self.rows if rows is None else rows
        with self.index.open(result["datasets"]["projected"]) as output:
            self.assertEqual(output.spec.document["kind"], "elemental_projected")
            self.assertEqual(output.spec.coordinates("time"), [0, 1, 2])
            self.assertEqual(output.spec.document["parents"], [result["datasets"]["raw"]])
            for time in range(3):
                vectors = self.vectors[time, :ne].reshape(ne, -1).T
                for row_index, row in enumerate(rows):
                    for momentum_index, momentum in enumerate(self.momenta):
                        matrix = sum(complex(*term["coefficient"]) * reference_matrix(
                            self.links, term["path"], time, momentum, phase=phase)
                            for term in row["terms"])
                        expected = vectors.conj().T @ matrix @ vectors
                        np.testing.assert_allclose(output.read(time=time)[row_index, momentum_index],
                                                   expected, atol=atol, rtol=atol)

    def test_all_phases_match_independent_site_reference_without_production(self):
        for phase in ("anchor", "midpoint", "none"):
            with self.subTest(phase=phase):
                identity = self.raw(phase)
                with self.index.open(identity) as raw:
                    before = {time: raw.read(time=time).copy()
                              for time in raw.spec.coordinates("time")}
                with patch.object(ElementalGenerator, "calc_path_blocks", side_effect=AssertionError("elemental production")), \
                     patch.object(workflow, "re_combine", side_effect=AssertionError("preview computes")):
                    preview = plan_indexed_elemental_projection(self.index, identity, self.rows)
                self.assertEqual(preview["stages"]["projected"]["action"], "compute")
                self.assertEqual(set(preview["stages"]), {"raw", "projected"})
                with patch.object(ElementalGenerator, "calc_path_blocks", side_effect=AssertionError("elemental production")):
                    result = self.project(identity, "projected-" + phase)
                with self.index.open(identity) as raw:
                    for time, expected in before.items():
                        np.testing.assert_array_equal(raw.read(time=time), expected)
                self.assert_reference(result, phase=phase)
                with self.index.open(result["datasets"]["projected"]) as projected:
                    self.assertEqual(projected.spec.coordinates("component"),
                                     [dict(projection=0), dict(projection=1)])

    def test_component_first_reordered_time_complex64_and_mode_prefix(self):
        identity = self.raw()
        with self.index.open(identity) as raw:
            values = np.stack([raw.read(time=time) for time in (2, 0, 1)]).transpose(1, 2, 0, 3, 4)
            document = raw.spec.document
        axes = {axis["name"]: axis for axis in document["axes"]}
        axes["time"]["coordinates"] = [2, 0, 1]
        document["axes"] = [axes[name] for name in
            ("component", "momentum", "time", "sink_mode", "source_mode")]
        document["dtype"] = "complex64"
        imported = self.save("imported-raw", DatasetSpec(document), values.astype(np.complex64))
        result = self.project(imported, "prefix", usedNe=2)
        self.assert_reference(result, ne=2, atol=3e-6)
        with self.index.open(result["datasets"]["projected"]) as projected:
            self.assertEqual(projected.spec.document["dtype"], "complex128")
            self.assertEqual(projected.spec.document["bases"]["sink_mode"],
                             dict(data_id=self.basis_id, indices=[4, 1]))

    def test_missing_raw_projected_input_and_invalid_rows_reject_before_output(self):
        identity = self.raw()
        projected = self.project(identity, "valid")["datasets"]["projected"]
        cases = ((identity, [dict(id=0, terms=[dict(path=[2], coefficient=[1, 0])])]),
                 (projected, self.rows),
                 (identity, [dict(id=0, terms=[dict(path=[6], coefficient=[1, 0])])]),
                 (identity, [dict(id=0, terms=[dict(path=[], coefficient=[float("nan"), 0])])]),
                 (identity, [dict(id=True, terms=[dict(path=[], coefficient=[1, 0])])]),
                 (identity, [dict(id=0, terms=[])]))
        for position, (actual, rows) in enumerate(cases):
            with self.subTest(position=position), \
                 patch.object(workflow, "re_combine", side_effect=AssertionError("computed")):
                with self.assertRaises(ValueError):
                    self.project(actual, f"invalid-{position}", rows)
                self.assertFalse((self.root / f"invalid-{position}").exists())

    def test_direct_row_writer_rejects_different_actual_parent_before_computing(self):
        identity = self.raw()
        preview = plan_indexed_elemental_projection(self.index, identity, self.rows)
        with self.index.open(identity) as raw:
            values = np.stack([raw.read(time=t) for t in range(3)])
            different = self.save("same-definition-other-realization", raw.spec, values)
        self.assertNotEqual(different, identity)
        with self.index.open(different) as wrong_parent, \
             DatasetWriter(self.root / "wrong-parent", preview["stages"]["projected"]["spec"],
                            provenance=dict(test="independent")) as writer, \
             patch.object(workflow, "re_combine", side_effect=AssertionError("computed")):
            with self.assertRaisesRegex(ValueError, "actual raw input view"):
                project_elemental(wrong_parent, writer)
            self.assertEqual(writer.missing, (0, 1, 2))
        self.assert_unpublished(self.root / "wrong-parent")

    def test_finite_coefficients_that_overflow_cannot_publish_a_result(self):
        identity = self.raw()
        rows = [dict(id=0, terms=[dict(path=[], coefficient=[1e308, 0])])]
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite projected"):
                self.project(identity, "overflow", rows)
        self.assert_unpublished(self.root / "overflow" / "projected")

    def test_external_raw_unknown_or_invalid_scientific_identity_rejects_before_output(self):
        identity = self.raw()
        with self.index.open(identity) as raw:
            description = raw.spec.document
            values = np.stack([raw.read(time=t) for t in range(3)])
        mutations = (("definition", "gauge_id", 17),
                     ("conventions", "phase", None),
                     ("configuration", "configuration", None))
        for position, (section, key, value) in enumerate(mutations):
            with self.subTest(section=section):
                document = DatasetSpec(description).document
                document[section][key] = value
                actual = self.save(f"unknown-{position}", DatasetSpec(document), values)
                with self.assertRaises(ValueError):
                    self.project(actual, f"unknown-output-{position}")
                self.assertFalse((self.root / f"unknown-output-{position}").exists())
