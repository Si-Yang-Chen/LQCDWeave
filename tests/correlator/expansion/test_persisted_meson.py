from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec, parse_axis_pattern
from lqcdweave.contraction.expansion.api import describe_meson_diagrams


SHAPE = TensorSpec((2,), np.complex128)


def diagram(groups, equations=None, propagator_types=None):
    if equations is None:
        equations = ["i,i->"] * len(groups)
    return SimpleNamespace(
        operands=groups,
        subscripts=equations,
        propagator_types=propagator_types,
    )


def two_recipe_diagrams():
    return (
        diagram(
            [([(0, 0, 1)], [0]), ([(0, 0, 1)], [1])],
            propagator_types=[["VSV"], ["VSV"]],
        ),
        diagram([([(1, 0, 1)], [0])], propagator_types=[["VSV"]]),
    )


def declarations():
    return {
        "vertex_specs": {(0, 11): SHAPE, (1, 17): SHAPE},
        "propagator_specs": {(0, 11, 17): SHAPE, (1, 11, 17): SHAPE},
        "vertex_sources": {0: {"declared": "vertex-0"}, 1: {"declared": "vertex-1"}},
        "propagator_sources": {0: {"declared": "propagator-0"}, 1: {"declared": "propagator-1"}},
    }


def describe(diagrams=None, times=(11, 17), **overrides):
    values = declarations()
    values.update(overrides)
    return describe_meson_diagrams(
        two_recipe_diagrams() if diagrams is None else diagrams,
        times,
        **values,
    )


class PersistedMesonDescriptionTests(unittest.TestCase):
    def test_public_api_preserves_keys_term_order_and_recipe_groups(self):
        workload = describe()

        self.assertEqual(workload.convention, "meson-simple-v1")
        self.assertEqual(workload.output_groups, ((0, 1), (2,)))
        self.assertEqual(workload.metadata["output_recipes"], ((0, 1), (2,)))
        self.assertEqual(workload.metadata["assembly_extra_bytes"], 96)
        self.assertEqual(
            set(workload.specs),
            {
                "propagator:0:11:17",
                "vertex:0:11",
                "vertex:1:17",
                "propagator:1:11:17",
            },
        )
        self.assertEqual(
            workload.terms[0].inputs,
            ("propagator:0:11:17", "vertex:0:11"),
        )
        self.assertEqual(
            workload.terms[1].inputs,
            ("propagator:0:11:17", "vertex:1:17"),
        )
        self.assertEqual(
            workload.terms[2].inputs,
            ("propagator:1:11:17", "vertex:0:11"),
        )
        self.assertEqual(workload.input_sources["propagator:0:11:17"]["handle"], 0)
        self.assertEqual(workload.input_sources["propagator:1:11:17"]["handle"], 1)
        self.assertEqual(workload.terms[0].subscripts, parse_axis_pattern("i,i->"))

    def test_times_must_be_scalar_integer_labels(self):
        for invalid in (True, np.bool_(False), 1.5, np.array([4])):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    describe(times=(invalid, 17))
        with self.assertRaisesRegex(TypeError, "times must be a sequence"):
            describe(times=11)

    def test_invalid_source_sink_and_vertex_positions_fail_before_lookup(self):
        for bad_graph, role in (
            (diagram([([(0, 0, 2)], [0])]), "sink-time"),
            (diagram([([(0, -1, 1)], [0])]), "source-time"),
            (diagram([([(0, 0, 1)], [2])]), "vertex position"),
        ):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, role):
                    describe(diagrams=(bad_graph,))

    def test_bool_and_negative_indices_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "propagator handle index"):
            describe(diagrams=(diagram([([(True, 0, 1)], [0])]),))
        with self.assertRaisesRegex(ValueError, "negative propagator handle"):
            describe(diagrams=(diagram([([(-1, 0, 1)], [0])]),))
        with self.assertRaisesRegex(TypeError, "vertex position"):
            describe(diagrams=(diagram([([(0, 0, 1)], [True])]),))
        with self.assertRaisesRegex(ValueError, "negative vertex position"):
            describe(diagrams=(diagram([([(0, 0, 1)], [-1])]),))

    def test_explicit_vsv_types_are_checked_per_group_and_edge(self):
        with self.assertRaisesRegex(NotImplementedError, "VSV"):
            describe(diagrams=(diagram([([(0, 0, 1)], [0])], propagator_types=[["PSP"]]),))
        with self.assertRaisesRegex(ValueError, "one propagator_types group"):
            describe(diagrams=(diagram([([(0, 0, 1)], [0])], propagator_types=[]),))
        with self.assertRaisesRegex(ValueError, "one propagator type per edge"):
            describe(diagrams=(diagram([([(0, 0, 1)], [0])], propagator_types=[[]]),))
        # Omitted metadata retains the current simple-Diagram VSV convention.
        workload = describe(diagrams=(diagram([([(0, 0, 1)], [0])]),))
        self.assertEqual(workload.convention, "meson-simple-v1")

    def test_only_scalar_group_outputs_are_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "scalar group outputs"):
            describe(diagrams=(diagram([([(0, 0, 1)], [0])], equations=["i,i->i"]),))

    def test_missing_specs_and_invalid_source_declarations_are_clear(self):
        missing_spec = declarations()["propagator_specs"].copy()
        missing_spec.pop((0, 11, 17))
        with self.assertRaisesRegex(KeyError, "missing propagator TensorSpec"):
            describe(propagator_specs=missing_spec)
        with self.assertRaisesRegex(KeyError, "missing vertex source"):
            describe(vertex_sources={1: {"declared": "vertex-1"}})
        with self.assertRaisesRegex(ValueError, "non-empty JSON source metadata"):
            describe(vertex_sources={0: {}, 1: {"declared": "vertex-1"}})
        bad_json = {0: {"declared": object()}, 1: {"declared": "vertex-1"}}
        with self.assertRaises(TypeError):
            describe(vertex_sources=bad_json)


if __name__ == "__main__":
    unittest.main()
