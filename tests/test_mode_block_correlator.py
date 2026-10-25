"""Small complex CPU references for native meson mode partitions."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.vsv_graph import evaluate_vsv_expression
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.correlator.api import (
    Diagram, QuarkDiagram, ModeBlockedDiagram, split_eigenmodes, project_meson_mode_pairs,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calculate


def fixture(ne=7, nt=2):
    rng = np.random.default_rng(928)
    random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    component = ElementalComponent(0)
    elemental = ArrayElemental({"v2v": random((1, 1, nt, ne, ne))}, (component,),
                              phase_convention="anchor", cfg="cfg", data_identity="test-elemental")
    propagator = ArrayPropagator({"vsv": random((nt, nt, 4, 4, ne, ne))},
                                time_convention="absolute", cfg="cfg", data_identity="test-propagator")
    op = Operator.from_terms("complex", [OperatorTerm(5, component, 0, .3 + .7j)])
    vertices = {"a": Meson(elemental, op, phase_convention="anchor"),
                "b": Meson(elemental, op, True, phase_convention="anchor")}
    connected = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
    loops = [Diagram(QuarkDiagram([[1]]), [i], [name], [None, "q"])
             for i, name in enumerate(("a", "b"))]
    return connected, loops, vertices, elemental, propagator


class ModeBlockTests(unittest.TestCase):
    def run_expression(self, expr, vertices, propagator, ne=7, **kwargs):
        return calculate(expr, vertex_map=vertices, propagator_map={"q": propagator},
                         configuration="cfg", usedNe=ne, optimize="none", **kwargs)

    def test_unequal_blocks_reconstruct_connected_and_disconnected(self):
        diagram, loops, vertices, elemental, propagator = fixture()
        full = -(1 + .2j) * diagram + .4 * loops[0] * loops[1]
        split = split_eigenmodes({"full": full, "connected": diagram,
                                 "disconnected": loops[0] * loops[1]}, partitions=[0, 2, 7])
        self.assertEqual(split["full"].shape, (2, 2, 2, 2))
        with patch.object(elemental, "block", side_effect=AssertionError("plan read elemental")), \
             patch.object(propagator, "block", side_effect=AssertionError("plan read propagator")):
            plan = self.run_expression(split, vertices, propagator, execute=False)
        specs = plan.expanded.problem.specs
        self.assertTrue(all(max(spec.shape[2:]) <= 5 for spec in specs.values()))
        values = plan.run().value
        reference = evaluate_vsv_expression(full, lambda row, t: vertices[row].get(t),
                                            lambda tag, s, t: propagator.block(s, t), {0: 0, 1: 1})
        np.testing.assert_allclose(values["full"].sum(), reference, rtol=3e-13)
        whole = self.run_expression(diagram, vertices, propagator).value
        np.testing.assert_allclose(values["connected"].sum(), whole, rtol=3e-13)
        np.testing.assert_allclose(values["full"], -(1 + .2j) * values["connected"]
                                   + .4 * values["disconnected"], rtol=3e-13)
        paired = project_meson_mode_pairs(values["connected"])
        weights = np.array([1, np.sqrt(2), 1])
        np.testing.assert_allclose(weights @ paired @ weights, whole, rtol=3e-13)

    def test_each_block_source_dagger_and_pc_pairing(self):
        diagram, _, vertices, _, propagator = fixture(ne=4)
        split = split_eigenmodes(diagram, partitions=[0, 1, 4])
        values = self.run_expression(split, vertices, propagator, ne=4).value
        # Independent full-size zero-masked tensors, without selected readers.
        raw = {name: vertex.get(i) for i, (name, vertex) in enumerate(vertices.items())}
        for block in np.ndindex(values.shape):
            masks = [np.zeros(4) for _ in range(4)]
            for mask, index in zip(masks, block):
                mask[slice(0, 1) if index == 0 else slice(1, 4)] = 1
            masked = {"a": raw["a"] * np.outer(masks[0], masks[1]),
                      "b": raw["b"] * np.outer(masks[2], masks[3])}
            expected = evaluate_vsv_expression(diagram, lambda row, t: masked[row],
                                               lambda tag, s, t: propagator.block(s, t), {0: 0, 1: 1})
            np.testing.assert_allclose(values[block], expected, rtol=5e-13, atol=5e-13)
        symbolic = project_meson_mode_pairs(split)
        actual = self.run_expression(symbolic, vertices, propagator, ne=4).value
        np.testing.assert_allclose(actual, project_meson_mode_pairs(values), rtol=5e-13)

    def test_nested_arrays_identity_deepcopy_and_time_scan(self):
        diagram, _, vertices, _, propagator = fixture(ne=4)
        split = split_eigenmodes(np.array([[diagram, 2j * diagram]], dtype=object), partitions=[0, 2, 4])
        self.assertEqual(split.shape, (1, 2, 2, 2, 2, 2))
        self.assertNotEqual(split[0, 0, 0, 0, 0, 0], split[0, 0, 0, 0, 0, 1])
        self.assertNotEqual(split[0, 0, 0, 0, 0, 0], diagram)
        self.assertEqual(deepcopy(split[0, 0, 0, 0, 0, 0]), split[0, 0, 0, 0, 0, 0])
        values = self.run_expression(split, vertices, propagator, ne=4,
                                     time_axes={0: [0, 1], 1: [1, 0]}).value
        whole = self.run_expression(np.array([[diagram, 2j * diagram]], dtype=object),
                                    vertices, propagator, ne=4,
                                    time_axes={0: [0, 1], 1: [1, 0]}).value
        np.testing.assert_allclose(values.sum(axis=(2, 3, 4, 5)), whole, rtol=5e-13)
        with_zeros = split_eigenmodes(np.array([[diagram, 0], [0, diagram]], dtype=object),
                                     partitions=[0, 2, 4])
        self.assertEqual(with_zeros.shape, (2, 2, 2, 2, 2, 2))
        np.testing.assert_array_equal(with_zeros[0, 1], 0)

    def test_invalid_partitions_and_coverage_fail_before_reads(self):
        diagram, _, vertices, elemental, propagator = fixture()
        for boundaries in ([1, 7], [0, 2, 2], [0, True, 7], [0, 2.5, 7]):
            with self.assertRaises(ValueError):
                split_eigenmodes(diagram, partitions=boundaries)
        split = split_eigenmodes(diagram, partitions=[0, 3, 8])
        with patch.object(elemental, "block", side_effect=AssertionError("read")), \
             self.assertRaisesRegex(ValueError, "requested Ne"):
            self.run_expression(split, vertices, propagator, execute=False)

    def test_anytime_joint_path_matches_whole(self):
        diagram, _, vertices, _, propagator = fixture(ne=4)
        split = split_eigenmodes(diagram, partitions=[0, 2, 4])
        result = calculate(split, vertex_map=vertices, propagator_map={"q": propagator},
                           configuration="cfg", usedNe=4, optimize="anytime",
                           search={"deadline_seconds": 10, "root_width": 1,
                                   "branch_width": 1})
        whole = self.run_expression(diagram, vertices, propagator, ne=4).value
        np.testing.assert_allclose(result.value.sum(), whole, rtol=5e-13)

    def test_four_partitions_share_physical_blocks(self):
        diagram, _, vertices, elemental, propagator = fixture(ne=4)
        split = split_eigenmodes(diagram, partitions=[0, 1, 2, 3, 4])
        planned = self.run_expression(split, vertices, propagator, ne=4, execute=False)
        self.assertEqual(len(planned.expanded.problem.terms), 256)
        self.assertEqual(len(planned.expanded.inputs.blocks), 64)
        with patch.object(elemental, "block", wraps=elemental.block) as e, \
             patch.object(propagator, "block", wraps=propagator.block) as p:
            values = planned.run().value
        self.assertEqual(values.shape, (4, 4, 4, 4))
        self.assertEqual(e.call_count, 32)
        self.assertEqual(p.call_count, 32)
        self.assertTrue(all(len(indices) == 1 for call in (*e.call_args_list, *p.call_args_list)
                            for indices in call.kwargs["mode_indices"].values()))
        whole = self.run_expression(diagram, vertices, propagator, ne=4).value
        np.testing.assert_allclose(values.sum(), whole, rtol=5e-13)

    def test_repeated_fields_construct_selected_symbols_once_and_own_arrays(self):
        diagram, loops, *_ = fixture(ne=4)
        full = diagram + loops[0]*loops[1]
        constructor = ModeBlockedDiagram._from_partition
        with patch.object(ModeBlockedDiagram, '_from_partition', wraps=constructor) as build:
            blocked = split_eigenmodes(
                {'full': full, 'connected': diagram, 'disconnected': loops[0]*loops[1],
                 'again': diagram}, partitions=(0, 1, 2, 3, 4),
            )
        self.assertEqual(build.call_count, 256 + 16 + 16)
        self.assertEqual(blocked['connected'][0, 0, 0, 0], blocked['again'][0, 0, 0, 0])
        blocked['connected'][0, 0, 0, 0] = 0
        self.assertNotEqual(blocked['again'][0, 0, 0, 0], 0)

    def test_projection_only_combines_selected_entries_with_leading_and_tail_axes(self):
        rng = np.random.default_rng(702)
        data = rng.normal(size=(2, 4, 4, 4, 4, 3)) + 1j*rng.normal(size=(2, 4, 4, 4, 4, 3))
        actual = project_meson_mode_pairs(data, mode_axis=1)
        pairs = [(a, b) for a in range(4) for b in range(a, 4)]
        for row, (a, b) in enumerate(pairs):
            left = [(a, b)] if a == b else [(a, b), (b, a)]
            for column, (c, d) in enumerate(pairs):
                right = [(c, d)] if c == d else [(c, d), (d, c)]
                expected = sum(data[:, i, j, k, l, :] for i, j in left for k, l in right)
                expected /= np.sqrt(len(left)*len(right))
                np.testing.assert_allclose(actual[:, row, column, :], expected, rtol=1e-14, atol=1e-14)
        data = np.zeros((4, 4))
        data[-1, -1] = np.inf
        actual = project_meson_mode_pairs(data, vertices=1)
        np.testing.assert_array_equal(actual[:-1], 0)
        self.assertTrue(np.isinf(actual[-1]))
        integer = np.full((2, 2), 100, dtype=np.int8)
        actual = project_meson_mode_pairs(integer, vertices=1)
        self.assertEqual(actual.dtype, np.dtype('float64'))
        np.testing.assert_allclose(actual, (100, 200/np.sqrt(2), 100))
        large = np.full((2, 2), 1e308)
        actual = project_meson_mode_pairs(large, vertices=1)
        self.assertTrue(np.isfinite(actual).all())

    def test_selected_symbol_hash_is_independent_of_value_cache(self):
        diagram, *_ = fixture(ne=4)
        symbol = split_eigenmodes(diagram, partitions=(0, 2, 4))[0, 0, 0, 0]
        before = hash(symbol)
        symbol.value = np.ones((2, 2))
        symbol.value_pointer = object()
        self.assertEqual(hash(symbol), before)
        self.assertEqual(hash(deepcopy(symbol)), before)

    def test_partition_factory_matches_public_constructor_and_direct_leaf_axes(self):
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [1, 0], ['b', 'a'], [None, 'q'])
        with patch.object(Diagram, 'xreplace', side_effect=AssertionError('leaf traversal')):
            blocked = split_eigenmodes(diagram, partitions=(0, 1, 4))
        symbol = blocked[0, 1, 1, 0]
        self.assertEqual(symbol.mode_selections, (((1, 2, 3), (0,)), ((0,), (1, 2, 3))))
        public = ModeBlockedDiagram(diagram, symbol.mode_selections, mode_extent=4)
        self.assertEqual(public, symbol)
        self.assertEqual(hash(public), hash(symbol))
        self.assertEqual(public.name, symbol.name)
        self.assertEqual(hash(symbol), hash((diagram, 4, symbol.mode_selections)))
        self.assertEqual(hash(deepcopy(symbol)), hash(symbol))
        for selections, extent in (((), 4), ((((0, 0), (1,)), ((0,), (1,))), 4),
                                   ((((0,), (4,)), ((0,), (1,))), 4)):
            with self.subTest(selections=selections), self.assertRaises(ValueError):
                ModeBlockedDiagram(diagram, selections, mode_extent=extent)

    def test_persistent_planner_rejects_missing_block_science_schema(self):
        from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
        diagram, _, vertices, _, propagator = fixture(ne=4)
        block = split_eigenmodes(diagram, partitions=[0, 2, 4])[0, 0, 0, 0]
        with self.assertRaisesRegex(NotImplementedError, "selection-aware science schema"):
            plan_indexed_diagram_measurement(None, block,
                bindings=DiagramBindings({"q": propagator}, vertices, "cfg", 4),
                source_time_symbol=0, sink_time_symbol=1)


if __name__ == "__main__":
    unittest.main()
