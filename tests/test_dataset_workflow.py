"""Scientific identity, independent full-space trace, and failure recovery."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, stable_digest
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contraction.meson import connected_meson_pair
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.planning.meson import validate_meson_inputs
from lqcdweave.storage.dataset import (
    DatasetReader, DatasetWriter, inspect_dataset, register_npy,
)
from lqcdweave.storage.registration import (
    read_registration, registration_lock, write_registration,
)
from contextlib import ExitStack
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.runtime.dataset_stage import completed_dataset


def axis(name, values):
    return dict(name=name, coordinates=list(values))


def input_spec(kind, *, times=(1, 0)):
    mode = dict(data_id="actual-basis-realization-17", indices=[0, 1])
    axes = ([axis("sink_time", times), axis("source_time", times),
             axis("sink_spin", range(4)), axis("source_spin", range(4))]
            if kind == "vsv" else
            [axis("component", [{"path": []}]), axis("momentum", [[1, 0, 0], [0, 0, 0]]),
             axis("time", times)])
    axes += [axis("sink_mode", range(2)), axis("source_mode", range(2))]
    definition = dict(gauge_id="dirac-gauge" if kind == "vsv" else "elemental-gauge")
    conventions = dict(normalization="unscaled")
    if kind == "vsv":
        definition.update(dirac=dict(action="dense-reference", mass=.1), solver_tolerance=1e-12)
        conventions.update(spin_basis="degrand_rossi", temporal_boundary="antiperiodic")
    else:
        conventions.update(phase="anchor")
    return DatasetSpec(dict(schema=1, kind=kind, configuration=dict(
        ensemble="toy", configuration="cfg-1", lattice=[2, 1, 1, 2]),
        axes=axes, dtype="complex128", bases=dict(sink_mode=mode, source_mode=mode),
        definition=definition, conventions=conventions, parents=[]))


class DatasetWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vspec, self.espec = input_spec("vsv"), input_spec("elemental_vv")
        rng = np.random.default_rng(82)
        def random(shape):
            return rng.normal(size=shape) + 1j*rng.normal(size=shape)
        # Independent full color-space inverses, with no gamma5 symmetry assumed.
        self.inverses = [np.linalg.inv(5*np.eye(48) + .1*random((48, 48))) for _ in range(2)]
        vectors = [np.linalg.qr(random((6, 2)))[0] for _ in range(2)]
        self.embeddings = []
        for t, v in enumerate(vectors):
            embedding = np.zeros((48, 8), complex)
            embedding[t*24:(t+1)*24] = np.kron(np.eye(4), v)
            self.embeddings.append(embedding)
        order = (1, 0)
        self.propagators = []
        for inverse in self.inverses:
            a = np.empty((2, 2, 4, 4, 2, 2), complex)
            for i, t in enumerate(order):
                for j, s in enumerate(order):
                    projected = self.embeddings[t].conj().T @ inverse @ self.embeddings[s]
                    a[i, j] = projected.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3)
            self.propagators.append(a)
        self.elemental = np.empty((1, 2, 2, 2, 2), complex)
        for i, t in enumerate(order):
            for m, phase in enumerate((np.array([1]*3 + [-1]*3), np.ones(6))):
                self.elemental[0, m, i] = vectors[t].conj().T @ (phase[:, None]*vectors[t])
        component = ElementalComponent.path(GaugePath())
        self.sink_op = Operator.from_terms("sink", [OperatorTerm(1, component, 0, .3+.2j),
                                                    OperatorTerm(15, component, 1, -.7j)])
        self.source_op = Operator.from_terms("source", [OperatorTerm(2, component, 1, .6-.3j)])
        for name, data, spec in (("fwd", self.propagators[0], self.vspec),
                                 ("rev", self.propagators[1], self.vspec),
                                 ("elemental", self.elemental, self.espec)):
            path = self.root / f"{name}.npy"
            np.save(path, data)
            register_npy(self.root/name, path, spec, provenance={"producer": "independent-test"})

    def run_workflow(self, *, resume=False):
        # Test-local composition of public readers, checked rows and storage.
        with ExitStack() as stack:
            datasets = [stack.enter_context(DatasetReader(self.root/name))
                         for name in ("fwd", "rev", "elemental", "elemental")]
            spec, _, _, _ = prepare_meson_stage(datasets, self.sink_op, self.source_op)
            provenance = local_provenance()
            output = self.root/"result"
            if resume:
                existing = completed_dataset(output, spec, provenance)
                if existing is not None:
                    return existing
            with DatasetWriter(output, spec, provenance=provenance, resume=resume) as writer:
                write_meson_rows(datasets, self.sink_op, self.source_op, writer)
                return writer.publish()

    def reference(self):
        result = np.empty((2, 2), complex)
        for s in range(2):
            for t in range(2):
                def vertex(operator, time):
                    return sum(term.coefficient*np.kron(gamma(term.gamma),
                        self.elemental[0, term.momentum, (1, 0).index(time)]) for term in operator.terms)
                a = vertex(self.sink_op, t)
                b = vertex(self.source_op, s)
                g4 = np.kron(gamma(8), np.eye(2))
                b = g4 @ b.conj().T @ g4
                # Lift both vertices back into full time/spin/space/color space.
                a = self.embeddings[t] @ a @ self.embeddings[t].conj().T
                b = self.embeddings[s] @ b @ self.embeddings[s].conj().T
                result[s, t] = -np.trace(a @ self.inverses[0] @ b @ self.inverses[1])
        return result

    def test_workflow_matches_full_space_trace_with_reordered_times(self):
        identity = self.run_workflow()
        with DatasetReader(self.root/"result") as result:
            values = np.stack([result.read(source_time=s) for s in range(2)])
            np.testing.assert_allclose(values, self.reference(), atol=1e-14, rtol=1e-12)
            self.assertEqual(result.data_id, identity)
            self.assertEqual(result.spec.axis_names, ("source_time", "sink_time"))
            self.assertEqual(len(result.spec.document["parents"]), 4)
            self.assertIn("code_sha256", result.provenance)
            self.assertEqual(result.spec.document["definition"]["sink_operator"][0]["momentum"], [1, 0, 0])
        self.assertEqual(self.run_workflow(resume=True),identity)

    def test_strict_identity_and_coverage_fail_before_output(self):
        mutations = [
            lambda d: d["bases"]["sink_mode"].update(data_id="different-realization"),
            lambda d: d["bases"]["sink_mode"].update(data_id=None),
            lambda d: d["configuration"].update(configuration="other-cfg"),
            lambda d: d["configuration"].update(ensemble=None),
            lambda d: d["conventions"].update(spin_basis="unknown"),
            lambda d: d["definition"].update(gauge_id=None),
            lambda d: d["axes"][0].update(coordinates=[0]),
            lambda d: d["conventions"].update(temporal_boundary="periodic"),
        ]
        for mutate in mutations:
            d = self.vspec.document
            mutate(d)
            with self.subTest(document=d), self.assertRaises(ValueError):
                validate_meson_inputs(DatasetSpec(d), self.vspec, self.espec, self.espec)
        d = self.vspec.document
        for name in ("sink_mode", "source_mode"):
            d["bases"][name]["indices"] = [1, 0]
            next(a for a in d["axes"] if a["name"] == name)["coordinates"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "ordered subset"):
            validate_meson_inputs(DatasetSpec(d), self.vspec, self.espec, self.espec)

    def test_resume_skips_only_verified_completed_source_rows(self):
        calls = []
        def interrupted(*args):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("simulated interruption")
            return connected_meson_pair(*args)
        with patch("lqcdweave.contraction.meson_measurement.connected_meson_pair", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.run_workflow()
        output = self.root / "result"
        state = inspect_dataset(output, cfg="cfg-1")
        self.assertEqual(state["state"], "partial")
        self.assertEqual(state["completed"], (0,))
        self.assertFalse((output / "cfg-1.npy").exists())
        self.assertFalse((output / "cfg-1.h5").exists())
        with self.assertRaises(FileNotFoundError):
            DatasetReader(output)
        partial = np.load(output / ".state" / "cfg-1.npy", mmap_mode="r", allow_pickle=False)
        try:
            completed_bytes = {position: np.asarray(partial[position]).tobytes()
                               for position in state["completed"]}
        finally:
            partial._mmap.close()
        with patch("lqcdweave.contraction.meson_measurement.connected_meson_pair", wraps=connected_meson_pair) as contract:
            self.run_workflow(resume=True)
            self.assertEqual(contract.call_count, 2)
        with DatasetReader(output) as result:
            for position, time in enumerate(result.spec.coordinates("source_time")):
                if position in completed_bytes:
                    self.assertEqual(result.read(source_time=time).tobytes(), completed_bytes[position])
            np.testing.assert_allclose(np.stack([result.read(source_time=s) for s in range(2)]),
                                       self.reference(), atol=1e-14)

    def test_changed_operator_cannot_resume_old_measurement(self):
        with patch("lqcdweave.contraction.meson_measurement.connected_meson_pair", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                self.run_workflow()
        self.source_op = self.sink_op
        with self.assertRaisesRegex(ValueError, "resume definition"):
            self.run_workflow(resume=True)
        state = inspect_dataset(self.root / "result", cfg="cfg-1")
        self.assertNotEqual(state["state"], "complete")
        self.assertFalse((self.root / "result" / "cfg-1.npy").exists())
        self.assertFalse((self.root / "result" / "cfg-1.h5").exists())

    def test_data_identity_and_locator_are_separate(self):
        old = self.root/"fwd.npy"
        moved = self.root/"relocated.npy"
        record = read_registration(self.root / "fwd")["cfg-1"]
        identity = record["data_id"]
        self.assertEqual(stable_digest(record["identity"]), identity)
        self.assertNotIn("storage", record["identity"])
        self.assertEqual(Path(record["storage"]["path"]).resolve(), old.resolve())
        old.rename(moved)
        record["storage"]["path"] = str(moved.resolve())
        with registration_lock(self.root / "fwd"):
            write_registration(self.root / "fwd", {"cfg-1": record})
        with DatasetReader(self.root/"fwd") as reader:
            self.assertEqual(reader.data_id, identity)
            block = reader.read(sink_time=0, source_time=1)
            np.testing.assert_array_equal(block, self.propagators[0][1, 0])
            self.assertFalse(block.flags.writeable)

    def test_definition_is_detached_and_stable(self):
        document = self.vspec.document
        other = DatasetSpec(dict(reversed(list(document.items()))))
        self.assertEqual(other.definition_id, self.vspec.definition_id)
        document["configuration"]["configuration"] = "changed"
        self.assertNotEqual(stable_digest(document), self.vspec.definition_id)
        with self.assertRaises(ValueError):
            DatasetSpec(dict(self.vspec.document, dtype="float16"))

    def test_writer_rejects_incomplete_and_tampered_recovery(self):
        spec = DatasetSpec(dict(self.vspec.document, kind="test"))
        output = self.root/"partial"
        with DatasetWriter(output, spec, provenance={"code": "test"}) as writer:
            writer.write(0, self.propagators[0][0])
            with self.assertRaises(ValueError):
                writer.publish()
            with self.assertRaises(FileExistsError):
                DatasetWriter(output, spec, provenance={"code": "test"}, resume=True)
        self.assertEqual(inspect_dataset(output, cfg="cfg-1")["state"], "partial")
        np.save(output/".state"/"cfg-1.npy", np.ones(1, dtype=np.complex128), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            DatasetWriter(output, spec, provenance={"code": "test"}, resume=True)
        self.assertEqual(inspect_dataset(output, cfg="cfg-1")["state"], "partial")
        self.assertFalse((output / "cfg-1.npy").exists())
        self.assertFalse((output / "cfg-1.h5").exists())


if __name__ == "__main__":
    unittest.main()
