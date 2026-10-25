"""Raw-path regrouping changes only the measurement, with a full-space oracle."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from production.elemental.test_elemental import reference_matrix
from test_wilson_pipeline import random_gauge
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.production.perambulator.api import PerambulatorGenerator
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.production.elemental.planning import prepare_elemental_stage
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class OperatorDatasetReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.configuration = dict(ensemble="test", configuration="cfg", lattice=[2, 1, 1, 2])
        self.links = random_gauge((2, 1, 1, 2), seed=93)
        gauge_axes = self.axes(("time", "z", "y", "x", "direction", "color_row", "color_column"),
                               self.links.shape)
        gauge_id = self.save("gauge", self.spec("gauge", gauge_axes, conventions=dict(boundary="periodic")),
                             [self.links[time] for time in range(2)])
        rng = np.random.default_rng(948)
        self.vectors = [np.linalg.qr(rng.normal(size=(6, 2))+1j*rng.normal(size=(6, 2)))[0]
                        for time in range(2)]
        basis_axes = [dict(name="time", coordinates=[1, 0]), dict(name="mode", coordinates=[4, 1]),
                      *self.axes(("z", "y", "x", "color"), (1, 1, 2, 3))]
        basis_id = self.save("basis", self.spec("explicit_basis", basis_axes),
            [self.vectors[time].T.reshape(2, 1, 1, 2, 3) for time in (1, 0)])
        self.embeddings = []
        for time in range(2):
            embedding = np.zeros((48, 8), complex)
            embedding[time*24:(time+1)*24] = np.kron(np.eye(4), self.vectors[time])
            self.embeddings.append(embedding)
        self.inverses = [np.linalg.inv(5*np.eye(48)+.1*(rng.normal(size=(48, 48))+
                          1j*rng.normal(size=(48, 48)))) for _ in range(2)]
        modes = dict(data_id=basis_id, indices=[4, 1])
        axes = self.axes(("source_time", "sink_time", "sink_spin", "source_spin"), (2, 2, 4, 4))
        axes += [dict(name=name, coordinates=[4, 1]) for name in ("sink_mode", "source_mode")]
        self.propagators = []
        for name, mass, inverse in zip(("forward", "reverse"), (.1, .2), self.inverses):
            spec = self.spec("vsv", axes, bases=dict(sink_mode=modes, source_mode=modes),
                parents=[basis_id, gauge_id], definition=dict(gauge_id=gauge_id,
                    dirac=dict(action="dense-reference", mass=mass), solver_tolerance=1e-12),
                conventions=dict(normalization="unscaled", spin_basis="degrand_rossi",
                                 temporal_boundary="antiperiodic"))
            rows = [np.stack([(self.embeddings[t].conj().T @ inverse @ self.embeddings[s]).reshape(
                4, 2, 4, 2).transpose(0, 2, 1, 3) for t in range(2)]) for s in range(2)]
            self.propagators.append((spec, self.save(name, spec, rows)))
        self.paths = ((), (0,), (3,))
        self.momenta = ((1, 0, 0), (0, 0, 0))
        with self.index.open(gauge_id) as gauge, self.index.open(basis_id) as basis:
            generator, elemental_spec, provenance, times, cfg = prepare_elemental_stage(
                gauge, basis, paths=self.paths, momentum_list=self.momenta,
                phase_convention="anchor",
            )
            with DatasetWriter(self.root / "raw", elemental_spec,
                                provenance=provenance) as writer:
                if writer.missing:
                    generator.load(cfg["configuration"])
                    for row in writer.missing:
                        writer.write(row, generator.calc_path_blocks(times[row])["v2v"])
                self.elemental_id = writer.publish()
        self.index.register(self.root / "raw")
        with self.index.open(self.elemental_id) as elemental:
            self.elemental_spec = elemental.spec

    @staticmethod
    def axes(names, extents):
        return [dict(name=name, coordinates=list(range(extent))) for name, extent in zip(names, extents)]

    def spec(self, kind, axes, **fields):
        document = dict(schema=1, kind=kind, configuration=self.configuration, axes=axes,
                        dtype="complex128", bases={}, parents=[], definition={}, conventions={})
        document.update(fields)
        return DatasetSpec(document)

    def save(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance=dict(test="independent")) as writer:
            for position, row in enumerate(rows):
                writer.write(position, row)
            identity = writer.publish()
        self.index.register(directory)
        return identity

    @staticmethod
    def operator(name, terms):
        return Operator.from_terms(name, [OperatorTerm(spin, ElementalComponent.path(GaugePath(path)),
            momentum, coefficient) for spin, path, momentum, coefficient in terms])

    def run_measurement(self, name, sink, source, *, elemental_spec=None, elemental_id=None):
        actual_elemental_id = self.elemental_id if elemental_id is None else elemental_id
        with ExitStack() as stack:
            datasets = tuple(stack.enter_context(self.index.open(identity)) for identity in (
                self.propagators[0][1], self.propagators[1][1],
                actual_elemental_id, actual_elemental_id,
            ))
            spec, _, _, _ = prepare_meson_stage(datasets, sink, source)
            with DatasetWriter(self.root / name, spec,
                                provenance={"test": "operator-dataset-reuse"}) as writer:
                write_meson_rows(datasets, sink, source, writer)
                identity = writer.publish()
        self.index.register(self.root / name)
        return identity

    def assert_reference(self, identity, sink, source):
        def vertex(operator, time):
            spatial = self.vectors[time]
            return sum(term.coefficient*np.kron(gamma(term.gamma), spatial.conj().T @
                reference_matrix(self.links, term.component.key.steps, time,
                                 self.momenta[term.momentum], phase="anchor") @ spatial)
                       for term in operator.terms)
        g4 = np.kron(gamma(8), np.eye(2))
        with self.index.open(identity) as result:
            self.assertEqual(result.spec.document["parents"], [self.propagators[0][1],
                self.propagators[1][1], self.elemental_id, self.elemental_id])
            for s in range(2):
                for t in range(2):
                    sink_full = self.embeddings[t] @ vertex(sink, t) @ self.embeddings[t].conj().T
                    source_full = self.embeddings[s] @ (g4 @ vertex(source, s).conj().T @ g4) @ self.embeddings[s].conj().T
                    expected = -np.trace(sink_full @ self.inverses[0] @ source_full @ self.inverses[1])
                    np.testing.assert_allclose(result.read(source_time=s, sink_time=t), expected, atol=2e-11)

    def test_new_complex_projection_combination_only_recontracts_existing_raw_inputs(self):
        initial = self.operator("sink", [(15, (), 0, 1)])
        initial_source = self.operator("source", [(2, (), 1, .3-.6j)])
        combined = self.operator("sink", [(1, (0,), 0, .3+.2j), (15, (), 1, -.7j),
                                           (2, (3,), 0, -.2+.5j)])
        combined_source = self.operator("source", [(2, (0,), 1, .6-.3j), (8, (3,), 0, 1+.4j)])
        inputs = [path for name in ("forward", "reverse", "raw")
                  for path in (self.root / name).glob("*") if path.is_file()]
        original = {path: path.read_bytes() for path in inputs}
        with patch.object(LaplaceEigenvectorGenerator, "calc", side_effect=AssertionError("basis recomputed")), \
             patch.object(PerambulatorGenerator, "calc", side_effect=AssertionError("VSV recomputed")), \
             patch.object(ElementalGenerator, "calc_path_blocks", side_effect=AssertionError("raw paths recomputed")):
            first = self.run_measurement("first", initial, initial_source)
            second = self.run_measurement("combined", combined, combined_source)
        self.index.register(self.root / "first")
        self.index.register(self.root / "combined")
        self.assertNotEqual(first, second)
        self.assert_reference(first, initial, initial_source)
        self.assert_reference(second, combined, combined_source)
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)

    def test_missing_raw_path_and_projected_only_dataset_cannot_fake_coverage(self):
        missing = self.operator("missing", [(15, (1,), 0, 1)])
        with self.assertRaisesRegex(ValueError, "missing elemental coverage"):
            self.run_measurement("missing-output", missing, missing)
        self.assertFalse((self.root / "missing-output").exists())
        document = self.elemental_spec.document
        document.update(kind="elemental_projected", parents=[self.elemental_id])
        document["axes"][1]["coordinates"] = [dict(projection="fixed-combination")]
        document["definition"]["projection"] = dict(paths=[[], [0]], coefficients=[[1, 0], [0, 1]])
        with self.index.open(self.elemental_id) as raw:
            rows = [(raw.read(time=time)[0] + 1j*raw.read(time=time)[1])[None] for time in range(2)]
        projected = DatasetSpec(document)
        projected_id = self.save("projected", projected, rows)
        local = self.operator("local", [(15, (), 0, 1)])
        with self.assertRaisesRegex(ValueError, "wrong dataset kind"):
            self.run_measurement("projected-output", local, local,
                                 elemental_spec=projected, elemental_id=projected_id)
        self.assertFalse((self.root / "projected-output").exists())
