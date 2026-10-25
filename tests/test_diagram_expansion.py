"""Metadata-only M2 checks for copied sector and endpoint lowering."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contraction.expansion.legacy_diagram import (
    LegacyDiagramBinding, expand_state_sampling_variants, lower_state_graph,
)
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.contraction.expansion.api import expand_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings, parse_axis_pattern
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.expansion.sampling import SamplingGroup
from lqcdweave.operators.correlator.api import Propagator
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval
from lqcdweave.operators.single_hadron.current import (
    ConservedVectorInsertion, LocalVectorCurrent,
)
from lqcdweave.operators.single_hadron.operator import Operator, OperatorPath, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram


class _DiagramExpansionMetadataFixture(unittest.TestCase):
    def _two_current_fixture(self, *, include_p2p=True):
        components = tuple(dict.fromkeys(
            term.component for _ in range(2)
            for term in LocalVectorCurrent().operator(3).terms
        ))
        lt, ne, np_count = 4, 1, 4
        sectors = {
            "v2v": np.zeros((len(components), 1, lt, ne, ne), dtype=np.complex128),
            "v2p": np.zeros((len(components), 1, lt, ne, np_count, 3), dtype=np.complex128),
            "p2v": np.zeros((len(components), 1, lt, np_count, 3, ne), dtype=np.complex128),
        }
        if include_p2p:
            sectors["p2p"] = np.zeros(
                (len(components), 1, lt, np_count, 3, np_count, 3), dtype=np.complex128)
        pool_ids = {
            (component, 0, time, side): "one-ordered-point-set"
            for component in components for time in range(lt)
            for side in ("bar", "field")
        }
        reader = ArrayElemental(
            sectors, components, phase_convention="anchor", cfg="cfg-0",
            data_identity="cfg-0/elemental-basis", sampling_pool_ids=pool_ids,
            sampling_groups={"one-ordered-point-set": SamplingPoolSpec(
                "one-ordered-point-set", 64, np_count)},
        )
        propagator_sectors = {
            "vsv": np.zeros((lt, lt, 4, 4, ne, ne), dtype=np.complex128),
            "psv": np.zeros((lt, lt, 4, 4, np_count, 3, ne), dtype=np.complex128),
            "vsp": np.zeros((lt, lt, 4, 4, ne, np_count, 3), dtype=np.complex128),
            "psp": np.zeros((lt, lt, 4, 4, np_count, 3, np_count, 3), dtype=np.complex128),
        }
        propagator = ArrayPropagator(
            propagator_sectors, time_convention="absolute", cfg="cfg-0",
            data_identity="cfg-0/full-propagator-basis",
        )
        current = LocalVectorCurrent().operator(3)
        vertices = {
            name: Meson(reader, current, source=False, phase_convention="anchor")
            for name in ("sink", "source")
        }
        quark_a = Propagator("u", Tag(0, 0), Tag(1, 2))
        quark_b = Propagator("u", Tag(1, 2), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ["sink", "source"],
            [None, quark_a, quark_b],
        )
        return diagram, reader, propagator, vertices, {quark_a: propagator, quark_b: propagator}


class DiagramExpansionMetadataTests(_DiagramExpansionMetadataFixture):
    def test_source_vertex_conjugates_operator_coefficient_once(self):
        rng = np.random.default_rng(739)
        component = ElementalComponent(0)
        coefficient = 1.0 + 2.0j
        operator = Operator.from_terms(
            "complex-source", [OperatorTerm(0, component, 0, coefficient)])
        raw = rng.normal(size=(1, 1, 2, 2, 2)) + 1j * rng.normal(size=(1, 1, 2, 2, 2))
        reader = ArrayElemental({"v2v": raw}, (component,), phase_convention="anchor")
        peram = rng.normal(size=(2, 2, 4, 4, 2, 2)) + 1j * rng.normal(size=(2, 2, 4, 4, 2, 2))
        prop = ArrayPropagator({"vsv": peram}, time_convention="absolute")
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                          ["sink", "source"], [None, q])
        result = calc_diagram_eval(
            diagram, propagator_map={q: prop},
            vertex_map={"sink": Meson(reader, operator, phase_convention="anchor"),
                        "source": Meson(reader, operator, True, phase_convention="anchor")},
            usedNe=2, optimize="none")
        spin = gamma(0)
        source_spin = gamma(8) @ spin.conj().T @ gamma(8)
        sink = np.einsum("ab,ij->abij", spin, raw[0, 0, 0], optimize=False)
        source = np.einsum("ab,ij->abij", source_spin, raw[0, 0, 1].conj().T,
                           optimize=False)
        expected = abs(coefficient) ** 2 * np.einsum(
            "BAON,DCQP,ADNQ,CBPO->", peram[0, 1], peram[1, 0], sink, source,
            optimize=False)
        np.testing.assert_allclose(result.value, expected, rtol=1e-12, atol=1e-12)

    def test_native_maps_none_execution_matches_independent_numpy_reference(self):
        rng = np.random.default_rng(20260928)
        component = ElementalComponent(0)
        coefficient = 1.0 - 0.75j
        operator = Operator.from_terms(
            "complex-weighted-local", [OperatorTerm(0, component, 0, coefficient)],
        )
        vertex_data = rng.normal(size=(1, 1, 2, 1, 1)) + 1j * rng.normal(size=(1, 1, 2, 1, 1))
        elemental = ArrayElemental(
            {"v2v": vertex_data.astype(np.complex128)}, (component,),
            phase_convention="anchor", cfg="cfg-direct",
            data_identity="cfg-direct/elemental",
        )
        propagator_data = rng.normal(size=(2, 2, 4, 4, 1, 1)) + 1j * rng.normal(
            size=(2, 2, 4, 4, 1, 1))
        propagator_data = propagator_data.astype(np.complex128)
        propagator = ArrayPropagator(
            {"vsv": propagator_data}, time_convention="absolute", cfg="cfg-direct",
            data_identity="cfg-direct/peram",
        )
        meson = Meson(elemental, operator, phase_convention="anchor")
        quark = Propagator("u", Tag(0, 0), Tag(1, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), [0, 0], ["same", "same"],
            [None, quark],
        )
        read_counts = {"elemental": 0, "propagator": 0}
        elemental_block = elemental.block
        propagator_block = propagator.block

        def count_elemental(*args, **kwargs):
            read_counts["elemental"] += 1
            return elemental_block(*args, **kwargs)

        def count_propagator(*args, **kwargs):
            read_counts["propagator"] += 1
            return propagator_block(*args, **kwargs)

        elemental.block = count_elemental
        propagator.block = count_propagator
        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("optimize='none' must never search")), \
             patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("optimize='none' must never search storage paths")):
            calculated = calc_diagram_eval(
                diagram,
                propagator_map={quark: propagator},
                vertex_map={"same": meson},
                configuration="cfg-direct", usedNe=1,
                optimize="none", execute=True,
            )

        # Two occurrences share one physical vertex selection and one propagator
        # selection. The adapter materializes unweighted factors; c is applied
        # only by the explicit output recipe.
        vertex = np.einsum("ab,...->ab...", gamma(0), vertex_data[0, 0, 0], optimize=False)
        prop = propagator_data[0, 0]
        expected_equation = "BAON,DCQP,ADNQ,CBPO->"
        self.assertEqual(calculated.expanded.problem.terms[0].subscripts.normalized(),
                         parse_axis_pattern(expected_equation).normalized())
        unweighted = np.einsum(expected_equation, prop, prop, vertex, vertex, optimize=False)
        np.testing.assert_allclose(calculated.value, coefficient ** 2 * unweighted,
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(calculated.execution_mode, "direct")
        self.assertEqual(read_counts, {"elemental": 1, "propagator": 1})
        self.assertEqual(len(calculated.expanded.terms), 1)


if __name__ == "__main__":
    unittest.main()
