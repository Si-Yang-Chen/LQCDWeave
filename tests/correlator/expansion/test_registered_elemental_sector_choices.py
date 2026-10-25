"""Sector capability and explicit request semantics at the B/E boundary."""

import unittest

import numpy as np

import test_diagram_expansion as diagram_fixture
from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contraction.expansion.api import expand_correlator
from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.baryon import (
    BaryonOperator, BaryonOperatorTerm, SparseSpinTensor,
)
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm


class RegisteredElementalSectorChoiceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = diagram_fixture._DiagramExpansionMetadataFixture("runTest")
        self.diagram, _reader, self.propagator, self.vertices, self.propagator_map = (
            self.fixture._two_current_fixture()
        )

    def _expand(self, reader, operator, *, sink_request=None, source_request=None,
                source_dagger=False):
        vertices = {
            "sink": Meson(reader, operator, phase_convention="anchor",
                          requested_sectors=sink_request),
            "source": Meson(reader, operator, source=source_dagger or source_request is not None,
                            phase_convention="anchor", requested_sectors=source_request),
        }
        expanded = expand_correlator(
            self.diagram,
            adapter=LegacyDiagramBinding(),
            bindings=DiagramBindings(
                self.propagator_map, vertices, configuration="cfg-0",
                usedNe=1, usedNp=4,
            ),
            time_axes={0: 0, 1: 2},
        )
        vertex_requests = tuple(
            request for request in expanded.inputs.tensor_requests.values()
            if request[0] == "vertex"
        )
        return expanded, vertex_requests

    @staticmethod
    def _ordinary_reader(*, raw_sectors, data_identity="cfg-0/ordinary"):
        component = ElementalComponent(0)
        lt, ne, np_count = 4, 1, 4
        shapes = {
            "v2v": (1, 1, lt, ne, ne),
            "v2p": (1, 1, lt, ne, np_count, 3),
            "p2v": (1, 1, lt, np_count, 3, ne),
            "p2p": (1, 1, lt, np_count, 3, np_count, 3),
        }
        arrays = {name: np.ones(shapes[name], dtype=np.complex128)
                  for name in raw_sectors}
        pool_ids = {
            (component, 0, time, side): "ordered-points"
            for time in range(lt) for side in ("bar", "field")
        }
        reader = ArrayElemental(
            arrays, (component,), phase_convention="anchor", cfg="cfg-0",
            data_identity=data_identity, sampling_pool_ids=pool_ids,
            sampling_groups={"ordered-points": diagram_fixture.SamplingPoolSpec(
                "ordered-points", 64, np_count,
            )},
        )
        return reader, Operator.from_terms(
            "ordinary-local", (OperatorTerm(0, component, 0, 1.0 + 0.2j),),
        )

    def test_single_raw_vp_maps_to_only_logical_pv_for_source_dagger(self):
        reader, operator = self._ordinary_reader(raw_sectors=("v2p",))
        reader.available_sectors = lambda: ("v2p",)
        sink = Meson(reader, operator, phase_convention="anchor")
        source = Meson(reader, operator, source=True, phase_convention="anchor")
        self.assertEqual(sink.available_sectors(), ("v2p",))
        self.assertEqual(source.available_sectors(), ("p2v",))

        expanded, requests = self._expand(
            reader, operator, sink_request=("v2p",), source_request=("p2v",),
        )
        self.assertEqual({request[3] for request in requests}, {"v2p", "p2v"})
        self.assertGreater(len(expanded.terms), 0)

    def test_both_raw_mixed_sectors_are_available_without_filling_other_sectors(self):
        reader, operator = self._ordinary_reader(raw_sectors=("v2p", "p2v"))
        reader.available_sectors = lambda: ("v2p", "p2v")
        sink = Meson(reader, operator, phase_convention="anchor")
        source = Meson(reader, operator, source=True, phase_convention="anchor")
        self.assertEqual(sink.available_sectors(), ("v2p", "p2v"))
        self.assertEqual(source.available_sectors(), ("p2v", "v2p"))

        expanded, requests = self._expand(
            reader, operator, sink_request=("v2p", "p2v"),
            source_request=("v2p", "p2v"),
        )
        self.assertEqual({request[3] for request in requests}, {"v2p", "p2v"})
        self.assertGreater(len(expanded.terms), 0)

    def test_unavailable_logical_request_is_rejected_before_metadata_or_payload_getters(self):
        reader, operator = self._ordinary_reader(raw_sectors=("v2p",))
        reader.available_sectors = lambda: ("v2p",)
        describe_calls, block_calls = [], []
        reader.describe_block = lambda *a, **k: describe_calls.append((a, k))
        reader.block = lambda *a, **k: block_calls.append((a, k))
        with self.assertRaisesRegex(ValueError, "not available"):
            self._expand(
                reader, operator, sink_request=("v2p",), source_request=("v2p",),
            )
        self.assertEqual(describe_calls, [])
        self.assertEqual(block_calls, [])

    def test_explicit_capability_metadata_errors_are_rejected(self):
        reader, operator = self._ordinary_reader(raw_sectors=("v2p",))
        meson = Meson(reader, operator, phase_convention="anchor")
        for declaration, error, message in (
            ((), ValueError, "no available"),
            (("v2p", "v2p"), ValueError, "unique"),
            (("unsupported",), ValueError, "unsupported"),
            (("v2p", 2), TypeError, "strings"),
            (["v2p"], TypeError, "tuple"),
        ):
            reader.available_sectors = lambda value=declaration: value
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(error, message):
                    meson.available_sectors()

    def test_ordinary_default_remains_vv_even_when_reader_has_all_four_sectors(self):
        full_reader, operator = self._ordinary_reader(
            raw_sectors=("v2v", "v2p", "p2v", "p2p"),
        )
        full_reader.available_sectors = lambda: ("v2v", "v2p", "p2v", "p2p")
        vv_reader, _ = self._ordinary_reader(raw_sectors=("v2v",))
        vv_reader.available_sectors = lambda: ("v2v",)

        full, full_requests = self._expand(full_reader, operator)
        vv, vv_requests = self._expand(vv_reader, operator)
        self.assertEqual({request[3] for request in full_requests}, {"v2v"})
        self.assertEqual({request[3] for request in vv_requests}, {"v2v"})
        self.assertEqual(
            tuple(term.subscripts for term in full.problem.terms),
            tuple(term.subscripts for term in vv.problem.terms),
        )
        self.assertEqual(
            tuple(term.coefficient for term in full.problem.terms),
            tuple(term.coefficient for term in vv.problem.terms),
        )

    def test_current_still_requires_all_four_sectors_before_describing_any_block(self):
        reader = self.vertices["sink"].reader
        reader.available_sectors = lambda: ("v2v", "v2p", "p2v")
        describe_calls, block_calls = [], []
        reader.describe_block = lambda *a, **k: describe_calls.append((a, k))
        reader.block = lambda *a, **k: block_calls.append((a, k))
        with self.assertRaisesRegex(ValueError, "not available"):
            self._expand(reader, self.vertices["sink"].operator)
        self.assertEqual(describe_calls, [])
        self.assertEqual(block_calls, [])

    def test_current_four_sector_decomposition_is_unchanged(self):
        reader = self.vertices["sink"].reader
        reader.available_sectors = lambda: ("v2v", "v2p", "p2v", "p2p")
        expanded = expand_correlator(
            self.diagram, adapter=LegacyDiagramBinding(),
            bindings=DiagramBindings(
                self.propagator_map, self.vertices, configuration="cfg-0",
                usedNe=1, usedNp=4,
            ),
            time_axes={0: 0, 1: 2},
        )
        sectors = {
            request[3] for request in expanded.inputs.tensor_requests.values()
            if request[0] == "vertex"
        }
        self.assertEqual(sectors, {"v2v", "v2p", "p2v", "p2p"})
        self.assertEqual(len(expanded.terms), 52)

    def test_baryon_binding_keeps_its_vvv_default(self):
        component = BaryonComponent.paths()
        spin = SparseSpinTensor(((0, 1, 2, 1.0),))
        operator = BaryonOperator.from_terms(
            "baryon-default", (BaryonOperatorTerm(spin, component, momentum=0),),
        )
        reader = ArrayBaryonElemental(
            {"vvv": np.ones((1, 1, 2, 1, 1, 1), dtype=np.complex128)},
            (component,), cfg="cfg-baryon",
        )
        baryon = Baryon(reader, operator)
        self.assertEqual(baryon.describe_factors(0)[0].tensor.axes,
                         ("mode0", "mode1", "mode2"))


if __name__ == "__main__":
    unittest.main()
