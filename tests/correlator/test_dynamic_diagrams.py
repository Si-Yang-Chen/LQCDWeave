from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    BaryonTerm as ContractBaryonTerm,
    BaryonTopology as ContractBaryonTopology,
    BaryonVertex as ContractBaryonVertex,
    TensorSpec, parse_axis_pattern,
)
from lqcdweave.contraction.binding.api import (
    read_baryon_diagram_inputs,
    read_simple_diagram_inputs,
)
from lqcdweave.contraction.expansion.api import (
    expand_baryon_diagrams,
    expand_simple_diagrams,
    extract_baryon_topology,
    validate_baryon_diagram_batch,
)
from lqcdweave.contraction.planning.api import compile_plan
from lqcdweave.contraction.execution.api import execute_plan
from lqcdweave.contraction.results.api import (
    assemble_baryon_diagram_outputs,
    assemble_simple_diagram_outputs,
)


def _simple_diagram(groups, equations, propagator_types=None):
    return SimpleNamespace(
        operands=groups,
        subscripts=equations,
        propagator_types=propagator_types,
    )


class _ScratchSimplePropagator:
    def __init__(self, values):
        self.values = values
        self.scratch = np.zeros_like(next(iter(values.values())))
        self.calls = []

    def get(self, source, sink):
        self.calls.append(("get", source, sink))
        self.scratch[...] = self.values[(source, sink)]
        return self.scratch

    def get_v2p(self, source, sink):
        self.calls.append(("get_v2p", source, sink))
        self.scratch[...] = self.values[(source, sink)]
        return self.scratch, "effective-v2p"

    def get_p2v(self, source, sink):
        self.calls.append(("get_p2v", source, sink))
        self.scratch[...] = self.values[(source, sink)]
        return self.scratch, "effective-p2v"

    def get_p2p(self, source, sink):
        self.calls.append(("get_p2p", source, sink))
        self.scratch[...] = self.values[(source, sink)]
        return self.scratch, "effective-p2p"


class _ScratchVertex:
    def __init__(self, values):
        self.values = values
        self.scratch = np.zeros_like(next(iter(values.values())))
        self.calls = []

    def get(self, time):
        self.calls.append(time)
        self.scratch[...] = self.values[time]
        return self.scratch


class DynamicSimpleDiagramTests(unittest.TestCase):
    def test_snapshot_reuse_expansion_execution_and_multiple_output_recipes(self):
        p01 = np.array([[1 + 1j, 2 - 1j], [3 + 0j, -1 + 2j]], dtype=np.complex128)
        p10 = np.array([[2 - 1j, 0.5 + 3j], [-2 + 1j, 4 - 0.5j]], dtype=np.complex128)
        p_other = np.array([[0.5 + 0.25j, -1j], [2 + 2j, 3 - 1j]], dtype=np.complex128)
        v11 = np.array([1 - 1j, 2 + 0.5j], dtype=np.complex128)
        v17 = np.array([0.25 + 2j, -1 + 0.75j], dtype=np.complex128)
        prop = _ScratchSimplePropagator({(11, 17): p01, (17, 11): p10})
        other_prop = _ScratchSimplePropagator({(11, 17): p_other})
        vertex11 = _ScratchVertex({11: v11})
        vertex17 = _ScratchVertex({17: v17})

        first = _simple_diagram(
            [([(0, 0, 1)], [0]), ([(0, 1, 0)], [1])],
            ["ij,j->i", "ij,j->i"],
        )
        repeated = _simple_diagram([([(0, 0, 1)], [0])], ["ij,j->i"])
        distinct_handle = _simple_diagram([([(1, 0, 1)], [0])], ["ij,j->i"])
        specs, bindings, keys = read_simple_diagram_inputs(
            (first, repeated, distinct_handle),
            [11, 17],
            [vertex11, vertex17],
            [prop, other_prop],
            backend=np,
        )

        self.assertEqual(prop.calls, [("get", 11, 17), ("get", 17, 11)])
        self.assertEqual(other_prop.calls, [("get", 11, 17)])
        self.assertEqual(vertex11.calls, [11])
        self.assertEqual(vertex17.calls, [17])
        self.assertEqual(keys[0][0], keys[1][0])
        self.assertNotEqual(keys[0][0][0], keys[2][0][0])
        first_prop_key = keys[0][0][0]
        np.testing.assert_array_equal(bindings[first_prop_key], p01)
        self.assertFalse(np.shares_memory(bindings[first_prop_key], prop.scratch))
        self.assertEqual(specs[first_prop_key], TensorSpec(p01.shape, p01.dtype))

        terms, recipes = expand_simple_diagrams((first, repeated, distinct_handle), keys, specs)
        self.assertEqual(tuple(term.subscripts for term in terms),
                         (parse_axis_pattern("ij,j->i"),) * 4)
        self.assertEqual(recipes, ((0, 1), (2,), (3,)))
        plan = compile_plan(specs, terms, optimize="greedy")
        values = execute_plan(plan, bindings, backend=np).values
        result = assemble_simple_diagram_outputs(values, recipes, np)

        term0 = np.einsum("ij,j->i", p01, v11, optimize=False)
        term1 = np.einsum("ij,j->i", p10, v17, optimize=False)
        term3 = np.einsum("ij,j->i", p_other, v11, optimize=False)
        reference = np.stack((term0 * term1, term0, term3))
        np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-13)
        reverse = assemble_simple_diagram_outputs(values, tuple(reversed(recipes)), np)
        np.testing.assert_allclose(reverse, np.stack((term3, term0, term0 * term1)), rtol=1e-13, atol=1e-13)

        scalar_graph = _simple_diagram([([(0, 0, 1)], [0])], ["ij,j->"])
        scalar_specs, scalar_bindings, scalar_keys = read_simple_diagram_inputs(
            (scalar_graph,), [11, 17], [vertex11], [prop], backend=np,
        )
        scalar_terms, scalar_recipes = expand_simple_diagrams(
            (scalar_graph,), scalar_keys, scalar_specs,
        )
        scalar_plan = compile_plan(scalar_specs, scalar_terms, optimize="greedy")
        scalar_values = execute_plan(scalar_plan, scalar_bindings, backend=np).values
        scalar_result = assemble_simple_diagram_outputs(scalar_values, scalar_recipes, np)
        expected_scalar = np.einsum("ij,j->", p01, v11, optimize=False)
        np.testing.assert_allclose(scalar_result, np.array([expected_scalar]), rtol=1e-13, atol=1e-13)

    def test_all_existing_propagator_getters_and_legacy_wrapping(self):
        values = {(2, 5): np.array([1 + 2j, 3 - 1j], dtype=np.complex128)}
        prop = _ScratchSimplePropagator(values)
        vertex = _ScratchVertex({2: np.array([2 - 1j, 0.5j], dtype=np.complex128)})
        types = [["VSV"], ["VSP"], ["PSV"], ["PSP"]]
        graph = _simple_diagram(
            [([(0, 0, 1)], [0])] * 4,
            ["i,i->"] * 4,
            types,
        )
        specs, bindings, keys = read_simple_diagram_inputs(
            (graph,), [2, 5], [vertex], [prop], backend=np,
        )
        self.assertEqual(
            [name for name, _source, _sink in prop.calls],
            ["get", "get_v2p", "get_p2v", "get_p2p"],
        )
        # The requested sector remains part of the route identity even when
        # each getter returns the same shape and reuses one scratch array.
        self.assertEqual(len({group[0] for group in keys[0]}), 4)
        self.assertEqual(len(specs), 5)
        self.assertEqual(len(bindings), 5)

        bad_prop = SimpleNamespace()
        graph = _simple_diagram([([(0, 0, 1)], [0])], ["i,i->"], [["VSP"]])
        with self.assertRaisesRegex(RuntimeError, r"Error getting propagator\[0\] type VSP"):
            read_simple_diagram_inputs((graph,), [2, 5], [vertex], [bad_prop], backend=np)

        failing_vertex = SimpleNamespace(get=lambda _time: (_ for _ in ()).throw(ValueError("vertex read")))
        graph = _simple_diagram([([], [0])], ["i->i"])
        with self.assertRaisesRegex(ValueError, "vertex read"):
            read_simple_diagram_inputs((graph,), [2], [failing_vertex], [], backend=np)

    def test_equation_parsing_belongs_to_expansion_and_invalid_indices_are_clear(self):
        prop = _ScratchSimplePropagator({(1, 2): np.eye(2, dtype=np.complex128)})
        vertex = _ScratchVertex({1: np.ones(2, dtype=np.complex128)})
        graph = _simple_diagram([([(0, 0, 1)], [0])], ["i...,i->"])
        specs, _bindings, keys = read_simple_diagram_inputs(
            (graph,), [1, 2], [vertex], [prop], backend=np,
        )
        self.assertEqual(len(prop.calls), 1)
        with self.assertRaisesRegex(NotImplementedError, "ellipses"):
            expand_simple_diagrams((graph,), keys, specs)

        prop.calls.clear()
        bad_index = _simple_diagram([([(4, 0, 1)], [0])], ["i,i->"])
        with self.assertRaisesRegex(ValueError, "propagator handle index.*outside"):
            read_simple_diagram_inputs((bad_index,), [1, 2], [vertex], [prop], backend=np)
        self.assertEqual(prop.calls, [])

    def test_binding_failure_precedes_later_expansion_parse_failure(self):
        prop = _ScratchSimplePropagator({(1, 2): np.eye(2, dtype=np.complex128)})

        class FailingPropagator:
            def get(self, _source, _sink):
                raise ValueError("later getter failure")

        vertex = _ScratchVertex({1: np.ones(2, dtype=np.complex128)})
        graph = _simple_diagram(
            [([(0, 0, 1)], [0]), ([(1, 0, 1)], [0])],
            ["i...,i->", "i,i->"],
        )
        with self.assertRaisesRegex(RuntimeError, r"Error getting propagator\[1\].*later getter failure"):
            read_simple_diagram_inputs(
                (graph,), [1, 2], [vertex], [prop, FailingPropagator()], backend=np,
            )
        self.assertEqual(len(prop.calls), 1)

    def test_simple_expansion_preserves_zip_grouping_and_result_edge_cases(self):
        prop = _ScratchSimplePropagator({(1, 2): np.eye(2, dtype=np.complex128)})
        vertex = _ScratchVertex({1: np.ones(2, dtype=np.complex128)})
        graph = _simple_diagram(
            [([(0, 0, 1)], [0]), ([(0, 0, 1)], [0])],
            ["ij,j->i"],
        )
        specs, _bindings, keys = read_simple_diagram_inputs(
            (graph,), [1, 2], [vertex], [prop], backend=np,
        )
        terms, recipes = expand_simple_diagrams((graph,), keys, specs)
        self.assertEqual(len(terms), 1)
        self.assertEqual(recipes, ((0,),))
        empty_graph = _simple_diagram([], [])
        empty_terms, empty_recipes = expand_simple_diagrams((empty_graph,), ((),), {})
        self.assertEqual(empty_terms, ())
        self.assertEqual(empty_recipes, ((),))
        single = assemble_simple_diagram_outputs((np.array([2.0, 3.0]),), recipes, np)
        np.testing.assert_array_equal(single, np.array([[2.0, 3.0]]))
        empty = assemble_simple_diagram_outputs((), empty_recipes, np)
        np.testing.assert_array_equal(empty, np.array([1.0]))
        empty = assemble_simple_diagram_outputs((), (), np)
        self.assertEqual(empty.shape, (0,))


def _baryon_diagram(source_slots=(2, 0, 1), **metadata):
    zero = [[0, 0, 0] for _ in range(3)]
    edge = [[0, 0, 0] for _ in range(3)]
    for sink_slot, source_slot in enumerate(source_slots):
        edge[source_slot][sink_slot] = sink_slot + 1
    values = {"adjacency_matrix": [[zero, edge], [zero, zero]]}
    values.update(metadata)
    return SimpleNamespace(**values)


class _BaryonVertexHandle:
    def __init__(self, offset):
        self.calls = []
        self.spin = np.full((4, 4, 4), offset + 1j, dtype=np.complex128)
        self.elemental = np.full((2, 2, 2), offset - 0.5j, dtype=np.complex128)

    def get_components(self, time):
        self.calls.append(time)
        self.spin[...] = time + 1j
        self.elemental[...] = time - 0.5j
        return self.spin, self.elemental


class _BaryonPropagatorHandle:
    def __init__(self, offset):
        self.calls = []
        self.buffer = np.empty((4, 4, 2, 2), dtype=np.complex128)
        self.offset = offset

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        self.buffer[...] = self.offset + source_time - sink_time * 1j
        return self.buffer


class DynamicBaryonDiagramTests(unittest.TestCase):
    def test_topology_preflight_binding_snapshot_and_contract_records(self):
        diagrams = (_baryon_diagram(), _baryon_diagram())
        topologies, times, coefficients = validate_baryon_diagram_batch(
            diagrams, (5, 9), 2, coefficients=(0.5 + 0.25j, np.complex128(-2j)),
        )
        self.assertIsInstance(topologies[0], ContractBaryonTopology)
        self.assertEqual(topologies[0], extract_baryon_topology(diagrams[0].adjacency_matrix))
        self.assertEqual(times, (5, 9))
        self.assertEqual(tuple(line.source_slot for line in topologies[0].lines), (2, 0, 1))

        source, sink = _BaryonVertexHandle(1), _BaryonVertexHandle(2)
        props = {1: _BaryonPropagatorHandle(10), 2: _BaryonPropagatorHandle(20), 3: _BaryonPropagatorHandle(30)}
        specs, bindings, input_keys = read_baryon_diagram_inputs(
            topologies, times, [source, sink], props, backend=np,
        )
        self.assertEqual(source.calls, [5])
        self.assertEqual(sink.calls, [9])
        self.assertEqual([handle.calls for handle in props.values()], [[(5, 9)]] * 3)
        self.assertEqual(input_keys[0], input_keys[1])
        self.assertEqual(len(specs), 7)
        self.assertEqual(len(bindings), 7)
        self.assertFalse(np.shares_memory(bindings[input_keys[0][0][0]], source.spin))
        terms = expand_baryon_diagrams(topologies, input_keys, coefficients)
        self.assertTrue(all(isinstance(term, ContractBaryonTerm) for term in terms))
        self.assertTrue(all(isinstance(term.source, ContractBaryonVertex) for term in terms))
        self.assertEqual(terms[0].source_slots, (2, 0, 1))
        self.assertEqual(terms[0].coefficient, 0.5 + 0.25j)
        self.assertEqual(terms[1].coefficient, np.complex128(-2j))

        assembled = assemble_baryon_diagram_outputs((1 + 2j, -3 + 0.5j), np)
        np.testing.assert_array_equal(assembled, np.array([1 + 2j, -3 + 0.5j]))
        empty = assemble_baryon_diagram_outputs((), np)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.dtype(np.complex128))

    def test_baryon_preflight_rejects_unsupported_metadata_before_any_reads(self):
        source, sink = _BaryonVertexHandle(1), _BaryonVertexHandle(2)
        props = {1: _BaryonPropagatorHandle(10), 2: _BaryonPropagatorHandle(20), 3: _BaryonPropagatorHandle(30)}
        for bad, error in (
            (_baryon_diagram(propagator_types=[["PSP"]]), NotImplementedError),
            (_baryon_diagram(leg_offsets=(1, None, None)), NotImplementedError),
        ):
            with self.subTest(error=error):
                with self.assertRaises(error):
                    topologies, times, coefficients = validate_baryon_diagram_batch((bad,), (5, 9), 2)
                    read_baryon_diagram_inputs(topologies, times, [source, sink], props, backend=np)
                self.assertEqual(source.calls, [])
                self.assertTrue(all(handle.calls == [] for handle in props.values()))
        with self.assertRaisesRegex(TypeError, "scalar integer time"):
            validate_baryon_diagram_batch((_baryon_diagram(),), (True, 9), 2)
        with self.assertRaisesRegex(ValueError, "one value per diagram"):
            validate_baryon_diagram_batch((_baryon_diagram(),), (5, 9), 2, coefficients=())

    def test_empty_baryon_batch_does_not_inspect_handles(self):
        self.assertEqual(
            read_baryon_diagram_inputs((), (), (), (), backend=np),
            ({}, {}, ()),
        )
        self.assertEqual(
            read_baryon_diagram_inputs((), (5, 9), (object(), object()), object(), backend=np),
            ({}, {}, ()),
        )


if __name__ == "__main__":
    unittest.main()
