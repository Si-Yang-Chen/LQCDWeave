"""Preset-to-calculation integration against an independent matrix trace."""

from pathlib import Path
from tempfile import TemporaryDirectory
import runpy
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave import preset
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calculate
from test_dataset_workflow import input_spec
from lqcdweave.storage.dataset import register_npy
from lqcdweave.planning.laplace import laplace_basis_document


class PresetFrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_legacy_declarations_load_and_calculate_without_storage_objects(self):
        rng = np.random.default_rng(480)
        def random(shape):
            return rng.normal(size=shape) + 1j*rng.normal(size=shape)
        e = random((1, 1, 2, 2, 2))
        p = random((2, 2, 4, 4, 2, 2))
        np.save(self.root / "e_cfg.npy", e)
        np.save(self.root / "p_cfg.npy", p)
        elemental = preset.ElementalNpy(str(self.root / "e_"), ".npy", e.shape, 2)
        perambulator = preset.PerambulatorNpy(str(self.root / "p_"), ".npy", p.shape, 2)
        op = Operator.from_terms("local", [OperatorTerm(1, ElementalComponent(0), 0)])
        source = Meson(elemental, op, True).load("cfg")
        sink = Meson(elemental, op).load("cfg")
        self.addCleanup(source.reader.close)
        self.assertIs(source.reader, sink.reader)
        propagator = perambulator.load("cfg")
        self.addCleanup(propagator.close)
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                          ["source", "sink"], [None, "q"])
        # Metadata planning must never read a numerical block.
        with patch.object(propagator, "block", side_effect=AssertionError("plan read propagator")), patch.object(
                sink.reader, "block", side_effect=AssertionError("plan read elemental")):
            planned = calculate(diagram, propagator_map={"q": propagator},
                                vertex_map={"source": source, "sink": sink},
                                configuration="cfg", usedNe=2,
                                time_axes={0: 0, 1: 1}, optimize="none", execute=False,
                                memory_limit=16 * 2**20)
        actual = planned.run().value
        # Flatten only tiny fixture data to an independent spin-mode trace.
        phi_source = np.kron(gamma(8), np.eye(2)) @ np.kron(gamma(1), e[0, 0, 0]).conj().T @ np.kron(gamma(8), np.eye(2))
        phi_sink = np.kron(gamma(1), e[0, 0, 1])
        reverse = p[1, 1].transpose(0, 2, 1, 3).reshape(8, 8)
        forward = p[0, 1].transpose(0, 2, 1, 3).reshape(8, 8)
        # A bare Diagram contracts indices; Wick signs belong to its symbolic
        # coefficient rather than the data reader or this explicit topology.
        expected = np.trace(phi_source @ reverse @ phi_sink @ forward)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "configuration|cfg"):
            calculate(diagram, propagator_map={"q": propagator},
                      vertex_map={"source": source, "sink": sink},
                      configuration="other", usedNe=2, optimize="none", execute=False)
        propagator.close()
        source.reader.close()

    def test_published_dataset_hides_reader_and_keeps_identity_checks(self):
        spec = input_spec("vsv")
        values = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(complex)
        path = self.root / "p.npy"
        np.save(path, values)
        directory = self.root / "p"
        actual_id = register_npy(directory, path, spec, provenance={"producer": "test"})
        data = preset.Dataset(str(self.root / "p")).load("cfg-1")
        self.addCleanup(data.dataset.close)
        self.assertEqual(data.dataset.data_id, actual_id)
        np.testing.assert_array_equal(data.block(0, 1), values[1, 0])
        with self.assertRaises(ValueError):
            data.block(0, 1, usedNe=3)
        # A simple declaration still validates the actual registration cfg.
        wrong = preset.Dataset(directory)
        with self.assertRaisesRegex((ValueError, FileNotFoundError, KeyError), "configuration|cfg|No such|cannot find"):
            wrong.load("")
        data.dataset.close()

    def test_published_declarations_bind_automatically_with_physical_time_order(self):
        rng = np.random.default_rng(109)
        arrays = {}
        for label, kind in (("e", "elemental_vv"), ("p", "vsv")):
            spec = input_spec(kind)  # Storage time order is (1, 0).
            values = rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
            path = self.root / f"{label}.npy"
            np.save(path, values)
            register_npy(self.root / label, path, spec, provenance={"producer": "test"})
            arrays[label] = values
        elemental = preset.Dataset(str(self.root / "e"))
        perambulator = preset.Dataset(str(self.root / "p"))
        op = Operator.from_terms("local", [OperatorTerm(1, ElementalComponent.path(()), 0)])
        source = Meson(elemental, op, True, phase_convention="anchor")
        sink = Meson(elemental, op, phase_convention="anchor")
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                          ["source", "sink"], [None, "q"])
        # Retain and explicitly close only test-owned mappings after execution.
        for declaration in (elemental, perambulator):
            original = declaration.load
            def tracked_load(cfg, original=original):
                loaded = original(cfg)
                self.addCleanup(loaded.dataset.close)
                return loaded
            declaration.load = tracked_load
        actual = calculate(diagram, propagator_map={"q": perambulator},
                           vertex_map={"source": source, "sink": sink},
                           configuration="cfg-1", time_axes={0: 0, 1: 1}, usedNe=2,
                           optimize="none", memory_limit=16 * 2**20).value
        e, p = arrays["e"], arrays["p"]
        g4 = np.kron(gamma(8), np.eye(2))
        a = g4 @ np.kron(gamma(1), e[0, 0, 1]).conj().T @ g4
        b = np.kron(gamma(1), e[0, 0, 0])
        reverse = p[1, 0].transpose(0, 2, 1, 3).reshape(8, 8)
        forward = p[0, 1].transpose(0, 2, 1, 3).reshape(8, 8)
        np.testing.assert_allclose(actual, np.trace(a @ reverse @ b @ forward), rtol=1e-12, atol=1e-12)
        self.assertIsNone(source.reader)
        self.assertIsNone(sink.reader)

    def test_full_time_public_example_matches_explicit_trace(self):
        rng = np.random.default_rng(81)
        e = rng.normal(size=(1, 1, 2, 1, 1)).astype(complex)
        p = rng.normal(size=(2, 2, 4, 4, 1, 1)).astype(complex)
        np.save(self.root / "e_cfg.npy", e)
        np.save(self.root / "p_cfg.npy", p)
        elemental = preset.ElementalNpy(str(self.root / "e_"))
        peram = preset.PerambulatorNpy(str(self.root / "p_"))
        from tests.fixtures.legacy_two_point import two_point as function
        actual = function(elemental, peram, "cfg", 1)
        self.addCleanup(elemental.load("cfg").close)
        self.addCleanup(peram.load("cfg").close)
        expected = np.empty((2, 2), complex)
        for source in range(2):
            for sink in range(2):
                a = gamma(8) @ (gamma(15) * e[0, 0, source, 0, 0]).conj().T @ gamma(8)
                b = gamma(15) * e[0, 0, sink, 0, 0]
                expected[source, sink] = -np.trace(
                    a @ p[sink, (source-sink) % 2, :, :, 0, 0] @
                    b @ p[source, (sink-source) % 2, :, :, 0, 0])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_published_basis_uses_internal_checked_handle(self):
        configuration = dict(ensemble="synthetic", configuration="cfg", lattice=[2, 1, 1, 2])
        spec = DatasetSpec(laplace_basis_document(configuration, (0, 1), "gauge-test", 2, 1e-12))
        values = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(complex)
        path = self.root / "basis.npy"
        np.save(path, values)
        actual_id = register_npy(self.root / "basis", path, spec, provenance={"producer": "test"})
        source = preset.Dataset(str(self.root / "basis"))
        block = source.read_modes("cfg", 1, 0, 1)
        np.testing.assert_array_equal(block, values[1, :1])
        self.assertEqual(source.Ne, 2)
        self.assertEqual(source.basis_id, actual_id)
        with patch("lqcdweave.storage.dataset.DatasetReader", side_effect=AssertionError("reopened same basis")):
            np.testing.assert_array_equal(source.read_modes("cfg", 0, 1, 2), values[0, 1:2])
        # Holding a view, rather than a user-managed reader, keeps its mapping.
        del source
        np.testing.assert_array_equal(block, values[1, :1])
        del block

    def test_published_elemental_uses_registration_components_and_phase(self):
        spec = input_spec("elemental_vv")
        values = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(complex)
        path = self.root / "e.npy"
        np.save(path, values)
        register_npy(self.root / "e", path, spec, provenance={"producer": "test"})
        data = preset.Dataset(str(self.root / "e")).load("cfg-1")
        self.addCleanup(data.dataset.close)
        self.assertEqual(data.phase_convention, "anchor")
        component = data.components[0]
        np.testing.assert_array_equal(data.block(component, 0, 0, "v2v"), values[0, 0, 1])
        op = Operator.from_terms("local", [OperatorTerm(0, component, 0)])
        with self.assertRaisesRegex(ValueError, "phase"):
            Meson(data, op, phase_convention="midpoint").load("cfg-1")


if __name__ == "__main__":
    unittest.main()
