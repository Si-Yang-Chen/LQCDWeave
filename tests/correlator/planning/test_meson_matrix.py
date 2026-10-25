"""Unified planning and numerical semantics for native meson matrices."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import PathSearchExhausted
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.operators.correlator.api import (
    Flavor,
    Hadron,
    HadronIrrep,
    Tag,
    gen_correlator,
    split_meson_mode_pairs,
)
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SparseSpinTensor
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
import lqcdweave.contraction.planning.path_planning as path_planning


def bounded_search(**overrides):
    controls = dict(deadline_seconds=1.0, max_states=1000, component_workers=1)
    controls.update(overrides)
    return controls


def multi_component_fixture():
    """A two-component V/V reader with complex coefficients and tiny CPU arrays."""
    rng = np.random.default_rng(1907)
    random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    components = (ElementalComponent(0), ElementalComponent(1))
    elemental = ArrayElemental(
        {"v2v": random((2, 1, 2, 3, 3))}, components,
        phase_convention="anchor", cfg="cfg", data_identity="meson-matrix-components",
    )
    propagator = ArrayPropagator(
        {"vsv": random((2, 2, 4, 4, 3, 3))},
        time_convention="absolute", cfg="cfg", data_identity="meson-matrix-propagator",
    )
    operators = {
        "z": Operator.from_terms("z", (
            OperatorTerm(5, components[0], 0, 0.3 + 0.7j),
            OperatorTerm(1, components[1], 0, -0.4j),
        )),
        "a": Operator.from_terms("a", (
            OperatorTerm(5, components[1], 0, 1 - 0.2j),
        )),
        "copy": Operator.from_terms("copy", (
            OperatorTerm(5, components[1], 0, 1 - 0.2j),
        )),
    }
    bindings = {}

    def bind(row):
        key = (row.hadron_name, row.dagger)
        if key not in bindings:
            bindings[key] = Meson(
                elemental, operators[row.hadron_name], row.dagger,
                phase_convention="anchor",
            )
        return bindings[key]

    return bind, elemental, propagator


class MesonMatrixPlanningTests(unittest.TestCase):
    @staticmethod
    def _options(bind, propagator, *, used_ne=3):
        return dict(
            vertex_map=bind,
            propagator_map={"S^q": propagator, "S^q_\\mathrm{local}": propagator},
            configuration="cfg",
            usedNe=used_ne,
        )

    def test_scalar_anytime_searches_for_charged_and_isospin(self):
        bind, _elemental, propagator = fixture()
        options = self._options(bind, propagator)

        for flavor in ("charged", "I0", "I1"):
            with self.subTest(flavor=flavor):
                with patch.object(path_planning, "solve_storage",
                                  wraps=path_planning.solve_storage) as search:
                    planned = calculate(
                        task(flavor=flavor), execute=False, search=bounded_search(), **options,
                    )

                planner = planned.prediction["planner"]
                search.assert_called_once()
                self.assertEqual(planner["optimizer"], "anytime")
                self.assertTrue(planner["search_performed"])
                self.assertTrue(planned.plan.state.actions)

                actual = planned.run().value
                reference = calculate(
                    task(flavor=flavor), optimize="none", **options,
                ).value
                np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_four_unequal_mode_partitions_match_symbolic_with_search(self):
        rng = np.random.default_rng(20261006)
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        component = ElementalComponent(0)
        elemental = ArrayElemental(
            {"v2v": random((1, 1, 2, 7, 7))}, (component,),
            phase_convention="anchor", cfg="cfg", data_identity="ne7-four-blocks",
        )
        propagator = ArrayPropagator(
            {"vsv": random((2, 2, 4, 4, 7, 7))},
            time_convention="absolute", cfg="cfg", data_identity="ne7-four-blocks-prop",
        )
        operator = Operator.from_terms(
            "a", (OperatorTerm(5, component, 0, 0.6 - 0.25j),),
        )
        bindings = {}

        def bind(row):
            key = (row.hadron_name, row.dagger)
            if key not in bindings:
                bindings[key] = Meson(
                    elemental, operator, row.dagger, phase_convention="anchor",
                )
            return bindings[key]

        options = self._options(bind, propagator, used_ne=7)
        partitions = [0, 1, 3, 5, 7]
        time_pairs = ({0: 0, 1: 1},)
        request = split_meson_mode_pairs(
            task(structured=True, flavor="I0", names=("a",)),
            partitions=partitions,
        )

        with patch.object(path_planning, "solve_storage",
                          wraps=path_planning.solve_storage) as search:
            planned = calculate(
                request, time_pairs=time_pairs, execute=False,
                search=bounded_search(), **options,
            )

        search.assert_called_once()
        self.assertTrue(planned.prediction["planner"]["search_performed"])
        self.assertTrue(planned.plan.state.actions)
        actual = planned.run().value

        reference_request = split_meson_mode_pairs(
            task(flavor="I0", names=("a",)), partitions=partitions,
        )
        reference = calculate(
            reference_request, time_pairs=time_pairs, optimize="none", **options,
        ).value
        self.assertEqual(actual.shape[-3:-1], (10, 10))
        self.assertEqual(actual.shape, reference.shape)
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_mode_blocks_time_pairs_archive_and_unique_source_reads(self):
        bind, elemental, propagator = multi_component_fixture()
        options = self._options(bind, propagator)
        request = split_meson_mode_pairs(
            task(structured=True, flavor="I0"), partitions=[0, 1, 3],
        )
        pairs = ({0: 0, 1: 1}, {0: 0, 1: 0})
        hardware = TargetHardwareMemory(host_available_bytes=1 << 30)
        profile = CostProfile(hardware_fingerprint="fixed-meson-cpu")
        planning_options = dict(
            options,
            optimize="meson_matrix",
            search=bounded_search(),
            time_pairs=pairs,
            execute=False,
            target_hardware=hardware,
            profile=profile,
            memory_limit=1 << 30,
            host_memory_limit=1 << 30,
        )

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixed-meson-plan.json.gz"
            with patch.object(elemental, "read_slab", wraps=elemental.read_slab) as elemental_reads, \
                 patch.object(propagator, "read_slab", wraps=propagator.read_slab) as propagator_reads:
                planned = calculate(request, **planning_options)
                self.assertEqual(planned.plan.statistics["templates"],1)
                self.assertEqual(len(planned.plan.instances),len(pairs))
                self.assertEqual(elemental_reads.call_count, 0)
                self.assertEqual(propagator_reads.call_count, 0)
                save_calculation_plan(archive, planned)
                actual = planned.run().value

                for reads in (elemental_reads.call_args_list, propagator_reads.call_args_list):
                    signatures = [repr((call.args, call.kwargs)) for call in reads]
                    self.assertEqual(len(signatures), len(set(signatures)))
                    self.assertTrue(signatures)

            with patch.object(
                path_planning, "solve_storage",
                side_effect=AssertionError("archive loading must not search"),
            ):
                restored = calculate(None,plan_archive=archive,configuration="cfg",
                    propagator_map=options["propagator_map"],vertex_map=elemental,
                    memory_limit=1<<30,host_memory_limit=1<<30,execute=False)
            self.assertEqual(len(restored.plan.instances),len(pairs))
            restored_value = restored.run().value

        reference_request = split_meson_mode_pairs(
            task(flavor="I0"), partitions=[0, 1, 3],
        )
        reference = calculate(
            reference_request,
            **dict(options, time_pairs=pairs, optimize="none"),
        ).value
        self.assertEqual(actual.shape, (2, 2, 3, 3, 2))
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(restored_value, reference, rtol=3e-12, atol=3e-12)

    def test_complete_archive_contains_all_pairs_and_loads_raw_readers(self):
        bind, elemental, propagator = fixture()
        options = self._options(bind, propagator)
        request = task(structured=True, flavor="I0", names=("a",))
        hardware = TargetHardwareMemory(host_available_bytes=1 << 30)
        profile = CostProfile(hardware_fingerprint="fixed-meson-cpu")
        saved_options = dict(
            options,
            optimize="meson_matrix",
            search=bounded_search(),
            target_hardware=hardware,
            profile=profile,
            memory_limit=1 << 30,
            host_memory_limit=1 << 30,
            batch=1,
            execute=False,
        )
        pairs = ({0: 0, 1: 1}, {0: 1, 1: 0})

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "one-pair-template.json.gz"
            complete = calculate(request,time_pairs=pairs,**saved_options)
            save_calculation_plan(archive,complete)
            with patch.object(
                path_planning, "solve_storage",
                side_effect=AssertionError("archive loading must not search"),
            ):
                restored = calculate(None,plan_archive=archive,configuration="cfg",
                    propagator_map=options["propagator_map"],vertex_map=elemental,
                    memory_limit=1<<30,host_memory_limit=1<<30,execute=False)
            actual = restored.run().value

        reference = calculate(
            task(flavor="I0", names=("a",)),
            time_pairs=pairs,
            optimize="none",
            **options,
        ).value
        self.assertEqual(actual.shape, (1, 1, 2))
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_fixed_source_and_scanned_sink_broadcasts_constant_factors(self):
        bind, _elemental, propagator = fixture()
        options = self._options(bind, propagator)
        axes = {0: 0, 1: (0, 1)}
        request = task(structured=True, flavor="I0")

        with patch.object(path_planning, "solve_storage",
                          wraps=path_planning.solve_storage) as search:
            planned = calculate(request, time_axes=axes, execute=False,
                                search=bounded_search(), **options)

        inputs = planned.expanded.inputs
        self.assertEqual(inputs.coordinates.fixed_axes, (0,))
        self.assertEqual(inputs.coordinates.scan_axes, (1,))
        self.assertGreater(search.call_count, 0)
        self.assertEqual(planned.prediction["planner"]["optimizer"], "anytime")
        self.assertTrue(planned.prediction["planner"]["search_performed"])
        self.assertTrue(inputs.task_batches)

        term = next(term for term in planned.expanded.problem.terms
                    if term.subscripts.output)
        input_labels, output = term.subscripts.inputs, term.subscripts.output
        self.assertEqual(len(input_labels), len(term.inputs))
        self.assertTrue(any(output[0] in item for item in input_labels))
        self.assertTrue(any(output[0] not in item for item in input_labels))

        actual = planned.run().value
        reference = calculate(
            task(flavor="I0"), time_axes=axes, optimize="none", **options,
        ).value
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_matrix_budget_fails_before_reader_io(self):
        bind, elemental, propagator = fixture()
        options = self._options(bind, propagator)
        with patch.object(elemental, "read_slab", side_effect=AssertionError("elemental IO")), \
             patch.object(propagator, "read_slab", side_effect=AssertionError("propagator IO")):
            with self.assertRaisesRegex(PathSearchExhausted, "storage budget"):
                calculate(
                    task(names=("a",)),
                    memory_limit=1,
                    host_memory_limit=1,
                    search=bounded_search(),
                    **options,
                )

    def test_unsupported_meson_relation_falls_back_to_generic_search(self):
        bind, _elemental, propagator = fixture()
        planned = calculate(
            task(), execute=False,
            **self._options(bind, propagator),
        )
        expanded = planned.expanded
        terms = list(expanded.problem.terms)
        # Same four factor roles and valid spin/mode extents, with a different
        # contraction relation outside the fixed dense matrix domain.
        terms[0] = replace(terms[0], subscripts="BAON,DCQP,BDNQ,CAPO->")
        problem = replace(expanded.problem, terms=tuple(terms))
        inputs = replace(expanded.inputs, problem=problem)
        unsupported = replace(expanded, inputs=inputs)

        class GenericSearchReached(Exception):
            pass

        with patch.object(path_planning, "solve_storage", side_effect=GenericSearchReached) as search:
            with self.assertRaises(GenericSearchReached):
                path_planning.find_contraction_path(
                    unsupported,
                    memory_limit=1 << 30,
                    host_memory_limit=1 << 30,
                    optimize="anytime",
                    search=bounded_search(),
                )
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["optimizer"], "anytime")

    def test_baryon_request_keeps_generic_anytime_search(self):
        rng = np.random.default_rng(1911)
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        component = BaryonComponent.paths()
        reader = ArrayBaryonElemental(
            {"vvv": random((1, 1, 2, 1, 1, 1))}, (component,), cfg="cfg",
        )
        operator = Operator.from_terms("baryon", (
            OperatorTerm.from_baryon(
                component, SparseSpinTensor(((0, 1, 2, 1),)), coefficient=0.4 + 0.3j,
            ),
        ))
        vertices = {False: Baryon(reader, operator), True: Baryon(reader, operator, True)}
        propagator = ArrayPropagator(
            {"vsv": random((2, 2, 4, 4, 1, 1))},
            time_convention="absolute", cfg="cfg",
        )
        row = HadronIrrep("N", (0, 0, 0), "G_1", 1, Tag(0, 0))[0]
        axis = [Hadron(row, Flavor("uds"))]
        request = gen_correlator(
            [axis.copy(), axis.copy()], [0, 1], [False, True],
        )

        with patch.object(path_planning, "solve_storage", wraps=path_planning.solve_storage) as search, \
             patch.object(reader, "block", side_effect=AssertionError("baryon numeric IO")), \
             patch.object(propagator, "block", side_effect=AssertionError("propagator numeric IO")):
            planned = calculate(
                request,
                vertex_map=lambda item: vertices[item.dagger],
                propagator_map={"S^q": propagator, "S^s": propagator},
                configuration="cfg",
                usedNe=1,
                execute=False,
                search=bounded_search(),
                baryon_vertices="joined",
            )

        self.assertTrue(planned.expanded.problem.baryon_joins)
        self.assertEqual(search.call_count, 1)
        self.assertTrue(planned.prediction["planner"]["search_performed"])
        self.assertEqual(planned.prediction["planner"]["optimizer"], "anytime")
        self.assertTrue(planned.plan.state.actions)


if __name__ == "__main__":
    unittest.main()
