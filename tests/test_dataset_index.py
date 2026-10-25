"""Verified public locators, explicit coverage views, and their real consumer."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetReader, DatasetWriter
from lqcdweave.runtime.datasets import open_dataset_view, plan_dataset_request
from contextlib import ExitStack
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.runtime.dataset_stage import completed_dataset


def spec(times=(0, 1), modes=(4, 1, 7)):
    return DatasetSpec(dict(schema=1, kind="fixture", configuration=dict(
        ensemble="known-ensemble", configuration="cfg-7", lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=list(times)),
              dict(name="mode", coordinates=list(modes))], dtype="complex128",
        bases={"mode": dict(data_id="actual-eigensystem", indices=list(modes))},
        definition={"gauge_id": "laplace-gauge"}, conventions={"normalization": "unscaled"},
        parents=["explicit-parent"]))


class _DatasetIndexFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = DatasetIndex(self.root / "index")
    def publish(self, name="a", request=None):
        request = spec() if request is None else request
        directory = self.root / name
        with DatasetWriter(directory, request, provenance={"producer": "fixture"}) as writer:
            for i in writer.missing:
                writer.write(i, np.asarray([10 * request.coordinates("time")[i] + n + 1j
                    for n in request.coordinates("mode")], dtype=np.complex128))
            identity = writer.publish()
        return directory, identity


class DatasetIndexTests(_DatasetIndexFixture):
    def test_explicit_subset_view_preserves_actual_coordinates_and_lifetime(self):
        directory, identity = self.publish()
        self.index.register(directory)
        request = spec(times=(1, 0), modes=(7, 4))
        with open_dataset_view(self.index, request, subset_axes=("time", "mode")) as view:
            self.assertEqual(view.data_id, identity)
            self.assertEqual(view.spec.document["bases"]["mode"]["indices"], [7, 4])
            value = view.read(time=1)
            np.testing.assert_array_equal(value, [17+1j, 14+1j])
            self.assertFalse(value.flags.writeable)
            np.testing.assert_array_equal(view.read(time=0, select={"mode": slice(1, None)}), [4+1j])
            self.assertEqual(view.read(time=0, mode=7), 7+1j)
            with self.assertRaises(KeyError):
                view.read(time=0, mode=1)
            with self.assertRaises(KeyError):
                view.read(time=0, select={"mode": [1]})
        with self.assertRaisesRegex(ValueError, "closed"):
            view.read(time=0)

    def test_missing_coverage_returns_exact_requested_positions_and_blocks_open(self):
        directory, identity = self.publish(request=spec(times=(1,)))
        self.index.register(directory)
        request = spec(times=(1, 0), modes=(1,))
        plan = plan_dataset_request(self.index, request, subset_axes=("time", "mode"))
        self.assertEqual(plan["data_id"], identity)
        self.assertEqual(plan["action"], "compute")
        self.assertEqual(plan["completed"], (0,))
        self.assertEqual(plan["missing"], (1,))
        with self.assertRaisesRegex(ValueError, "compute"):
            open_dataset_view(self.index, request, subset_axes=("time", "mode"))
        document = request.document
        document["bases"]["mode"]["data_id"] = None
        self.assertEqual(plan_dataset_request(self.index, DatasetSpec(document))["action"], "blocked")

    def test_read_uses_the_same_json_coordinate_identity_as_the_contract_and_planner(self):
        document = spec().document
        document["axes"] = [dict(name="label", coordinates=[1, 1.0, True]),
                            dict(name="point", coordinates=[[0, 1], [1, 0]]),
                            dict(name="value", coordinates=[1, 1.0, True])]
        document["bases"] = {}
        declared = DatasetSpec(document)
        directory = self.root / "json-labels"
        with DatasetWriter(directory, declared, provenance={}) as writer:
            for i in writer.missing:
                writer.write(i, np.asarray([[100*i + 10*j + k for k in range(3)]
                    for j in range(2)], dtype=np.complex128))
            writer.publish()
        self.index.register(directory)
        with DatasetReader(directory) as reader:
            self.assertEqual(reader.read(label=1.0, point=(1, 0), value=True), 112)
            np.testing.assert_array_equal(reader.read(label=True, point=[0, 1],
                select={"value": [True, 1.0, 1]}), [202, 201, 200])
        requested = declared.document
        requested["axes"][0]["coordinates"] = [True, 1]
        requested["axes"][2]["coordinates"] = [True, 1.0]
        with open_dataset_view(self.index, DatasetSpec(requested),
                               subset_axes=("label", "value")) as view:
            np.testing.assert_array_equal(view.read(label=True, point=(0, 1)), [202, 201])
            with self.assertRaises(KeyError):
                view.read(label=1.0, point=(0, 1))


class IndexedMesonWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Reuse the independently constructed full-space numerical fixture,
        # rather than treating a second contraction implementation as truth.
        from test_dataset_workflow import DatasetWorkflowTests
        self.fixture = DatasetWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.index = DatasetIndex(self.root / "index")
        self.identities = tuple(self.index.register(self.root / name)
            for name in ("fwd", "rev", "elemental", "elemental"))

    def run_indexed(self, requests, name, *, resume=False):
        # The fixture selects explicit views; publication uses generic storage.
        with ExitStack() as stack:
            datasets = [stack.enter_context(open_dataset_view(
                self.index, spec, subset_axes=("sink_mode", "source_mode"),
                data_id=identity)) for spec, identity in zip(requests, self.identities)]
            spec, _, _, _ = prepare_meson_stage(
                datasets, self.fixture.sink_op, self.fixture.source_op)
            output, provenance = self.root/name, local_provenance()
            if resume:
                existing = completed_dataset(output, spec, provenance)
                if existing is not None:
                    return existing
            with DatasetWriter(output, spec, provenance=provenance, resume=resume) as writer:
                write_meson_rows(datasets, self.fixture.sink_op, self.fixture.source_op, writer)
                return writer.publish()

    def test_indexed_full_time_measurement_matches_independent_full_space_trace(self):
        requests = (self.fixture.vspec,) * 2 + (self.fixture.espec,) * 2
        identity = self.run_indexed(requests, "indexed")
        with DatasetReader(self.root / "indexed") as reader:
            np.testing.assert_allclose(np.stack([reader.read(source_time=s) for s in range(2)]),
                                       self.fixture.reference(), atol=1e-14, rtol=1e-12)
            self.assertEqual(reader.spec.document["parents"], list(self.identities))
            self.assertEqual(reader.spec.document["definition"]["input_views"]["forward"],
                             self.fixture.vspec.document)
        self.assertEqual(self.run_indexed(requests, "indexed", resume=True), identity)

    def test_smaller_ne_changes_measurement_definition_and_matches_explicit_trace(self):
        requests = []
        for original in (self.fixture.vspec,) * 2 + (self.fixture.espec,) * 2:
            document = original.document
            for name in ("sink_mode", "source_mode"):
                document["bases"][name]["indices"] = [1]
                next(a for a in document["axes"] if a["name"] == name)["coordinates"] = [1]
            requests.append(DatasetSpec(document))
        self.run_indexed(requests, "small")
        full_requests = (self.fixture.vspec,) * 2 + (self.fixture.espec,) * 2
        self.run_indexed(full_requests, "full")
        expected = np.empty((2, 2), complex)
        from lqcdweave.operators.single_hadron.gamma import gamma
        for s in range(2):
            for t in range(2):
                def vertex(operator, time):
                    return sum(term.coefficient * np.kron(gamma(term.gamma),
                        self.fixture.elemental[0, term.momentum, (1, 0).index(time), 1:2, 1:2])
                        for term in operator.terms)
                a = vertex(self.fixture.sink_op, t)
                b = gamma(8) @ vertex(self.fixture.source_op, s).conj().T @ gamma(8)
                columns = [1, 3, 5, 7]
                sink_embedding = self.fixture.embeddings[t][:, columns]
                source_embedding = self.fixture.embeddings[s][:, columns]
                a = sink_embedding @ a @ sink_embedding.conj().T
                b = source_embedding @ b @ source_embedding.conj().T
                expected[s, t] = -np.trace(a @ self.fixture.inverses[0] @ b @ self.fixture.inverses[1])
        with DatasetReader(self.root / "small") as small, DatasetReader(self.root / "full") as full:
            np.testing.assert_allclose(np.stack([small.read(source_time=s) for s in range(2)]),
                                       expected, atol=1e-14, rtol=1e-12)
            self.assertNotEqual(small.spec.definition_id, full.spec.definition_id)
            self.assertEqual(small.spec.document["parents"], full.spec.document["parents"])
        with self.assertRaisesRegex(ValueError, "resume definition"):
            self.run_indexed(full_requests, "small", resume=True)

    def test_changed_mass_and_unknown_basis_fail_before_creating_output(self):
        requests = [self.fixture.vspec] * 2 + [self.fixture.espec] * 2
        for mutation in (lambda d: d["definition"]["dirac"].update(mass=.9),
                         lambda d: d["bases"]["sink_mode"].update(data_id=None)):
            document = requests[0].document
            mutation(document)
            changed = [DatasetSpec(document), *requests[1:]]
            with self.assertRaises(ValueError):
                self.run_indexed(changed, "invalid")
            self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
