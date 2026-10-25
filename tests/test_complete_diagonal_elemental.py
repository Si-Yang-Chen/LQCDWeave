"""Numerical regressions for producer-declared complete elemental diagonals.

The reference calculations below assemble full spin-major matrices and take
literal traces.  They deliberately do not use the compact preparation or the
contraction network's Einstein labels.
"""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval, calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _spin_major(block):
    """Convert ``(spin_sink, spin_source, space_sink, space_source)`` to matrix form."""
    return block.transpose(0, 2, 1, 3).reshape(
        block.shape[0] * block.shape[2], block.shape[1] * block.shape[3],
    )


def _elemental_matrix(sectors, component_index, time, points, modes):
    v2v = sectors["v2v"][component_index, 0, time]
    v2p = sectors["v2p"][component_index, 0, time].reshape(modes, points * 3)
    p2v = sectors["p2v"][component_index, 0, time].reshape(points * 3, modes)
    p2p = sectors["p2p"][component_index, 0, time].reshape(points * 3, points * 3)
    return np.block([[v2v, v2p], [p2v, p2p]])


def _propagator_matrix(sectors, source_time, sink_time, points, modes):
    point_dimension = points * 3
    top = np.concatenate((
        sectors["vsv"][source_time, sink_time],
        sectors["vsp"][source_time, sink_time].reshape(4, 4, modes, point_dimension),
    ), axis=3)
    bottom = np.concatenate((
        sectors["psv"][source_time, sink_time].reshape(4, 4, point_dimension, modes),
        sectors["psp"][source_time, sink_time].reshape(4, 4, point_dimension, point_dimension),
    ), axis=3)
    block = np.concatenate((top, bottom), axis=2)
    return _spin_major(block)


class CompleteDiagonalV2VTests(unittest.TestCase):
    modes = 3
    times = 3

    def _fixture(self, seed, *, diagonal, source_dagger):
        rng = np.random.default_rng(seed)
        source_component = ElementalComponent(("complete-diagonal-source", seed))
        sink_component = ElementalComponent(("general-sink", seed))
        components = (source_component, sink_component)
        component_index = {component: index for index, component in enumerate(components)}
        sectors = {"v2v": _complex(rng, (2, 1, self.times, self.modes, self.modes))}
        source_coefficients = _complex(rng, (self.modes,))
        sink_space = _complex(rng, (self.modes, self.modes))
        sectors["v2v"][0, 0, 0] = np.diag(source_coefficients)
        sectors["v2v"][1, 0, 2] = sink_space
        reader = ArrayElemental(
            sectors, components, phase_convention="anchor", cfg="cfg-complete-diagonal",
            data_identity=f"complete-diagonal-vv/{seed}",
            elemental_structures=(
                {("v2v", source_component, 0): "diagonal"} if diagonal else {}
            ),
        )
        source_coefficient = 0.37 + 0.81j
        sink_coefficient = -0.22 + 0.41j
        source = Operator.from_terms("complex-source", (
            OperatorTerm(3, source_component, 0, source_coefficient),
        ))
        sink = Operator.from_terms("complex-sink", (
            OperatorTerm(14, sink_component, 0, sink_coefficient),
        ))

        propagators = {"vsv": _complex(rng, (self.times, self.times, 4, 4, self.modes, self.modes))}
        prop_reader = ArrayPropagator(
            propagators, time_convention="absolute", cfg="cfg-complete-diagonal",
            data_identity=f"complete-diagonal-propagators/{seed}",
        )
        forward = Propagator("u", Tag(0, 0), Tag(1, 0))
        backward = Propagator("d", Tag(1, 0), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [0, 2], ["source", "sink"],
            [None, forward, backward],
        )
        vertices = {
            "source": Meson(reader, source, source=source_dagger, phase_convention="anchor"),
            "sink": Meson(reader, sink, phase_convention="anchor"),
        }
        return {
            "reader": reader,
            "source_component": source_component,
            "source_coefficients": source_coefficients,
            "sink_space": sink_space,
            "source_operator": source,
            "sink_operator": sink,
            "source_dagger": source_dagger,
            "source_coefficient": source_coefficient,
            "sink_coefficient": sink_coefficient,
            "sectors": sectors,
            "component_index": component_index,
            "propagators": propagators,
            "prop_reader": prop_reader,
            "prop_map": {forward: prop_reader, backward: prop_reader},
            "diagram": diagram,
            "vertices": vertices,
        }

    def _calculate(self, context, *, execute=True):
        return calc_diagram_eval(
            context["diagram"], propagator_map=context["prop_map"],
            vertex_map=context["vertices"], configuration="cfg-complete-diagonal",
            usedNe=self.modes, optimize="none", execute=execute,
        )

    @staticmethod
    def _reference(context):
        gamma5 = gamma(8)
        source_term = context["source_operator"].terms[0]
        sink_term = context["sink_operator"].terms[0]
        source_space = context["sectors"]["v2v"][
            context["component_index"][context["source_component"]], 0, 0,
        ]
        sink_space = context["sink_space"]
        source_spin = gamma(source_term.gamma)
        source_coefficient = context["source_coefficient"]
        if context["source_dagger"]:
            source_space = source_space.conj().T
            source_spin = gamma5 @ source_spin.conj().T @ gamma5
            source_coefficient = source_coefficient.conjugate()
        source_matrix = np.kron(source_spin, source_space)
        sink_matrix = np.kron(gamma(sink_term.gamma), sink_space)
        outgoing = _spin_major(context["propagators"]["vsv"][0, 2])
        incoming = _spin_major(context["propagators"]["vsv"][2, 0])
        return (source_coefficient * context["sink_coefficient"]
                * np.trace(source_matrix @ incoming @ sink_matrix @ outgoing))

    def test_marked_and_unmarked_complex_diagonal_vv_match_full_trace_with_source_dagger(self):
        for dagger in (False, True):
            with self.subTest(source_dagger=dagger):
                unmarked = self._fixture(441, diagonal=False, source_dagger=dagger)
                marked = self._fixture(441, diagonal=True, source_dagger=dagger)
                expected = self._reference(marked)
                self.assertGreater(np.linalg.norm(
                    marked["sink_space"] - marked["sink_space"].conj().T,
                ), 1e-3)
                self.assertGreater(np.linalg.norm(
                    np.diag(marked["source_coefficients"]).imag,
                ), 1e-3)
                self.assertEqual(marked["reader"].describe_block(
                    marked["source_component"], 0, 0, "v2v",
                ).structure, "diagonal")

                general_result = self._calculate(unmarked)
                diagonal_result = self._calculate(marked)
                np.testing.assert_allclose(general_result.value, expected, rtol=3e-12, atol=4e-11)
                np.testing.assert_allclose(diagonal_result.value, expected, rtol=3e-12, atol=4e-11)

                planned = self._calculate(marked, execute=False)
                source_specs = [
                    spec for spec in planned.expanded.tensor_specs.values()
                    if spec.description is not None
                    and spec.description.role == "elemental"
                    and spec.description.bindings.get("anchor_time") == 0
                ]
                sink_specs = [
                    spec for spec in planned.expanded.tensor_specs.values()
                    if spec.description is not None
                    and spec.description.role == "elemental"
                    and spec.description.bindings.get("anchor_time") == 2
                ]
                self.assertEqual([spec.shape for spec in source_specs], [(4, 4, self.modes)])
                self.assertEqual([spec.shape for spec in sink_specs], [(4, 4, self.modes, self.modes)])

    def test_identity_and_zero_coefficients_remain_valid_diagonal_data(self):
        for diagonal_coefficients, expected in (
                (np.ones(self.modes, dtype=np.complex128), None),
                (np.zeros(self.modes, dtype=np.complex128), 0j)):
            with self.subTest(coefficient=diagonal_coefficients[0]):
                context = self._fixture(887, diagonal=True, source_dagger=True)
                context["sectors"]["v2v"][0, 0, 0] = np.diag(diagonal_coefficients)
                context["reader"].sectors["v2v"][0, 0, 0] = np.diag(diagonal_coefficients)
                self.assertEqual(context["reader"].describe_block(
                    context["source_component"], 0, 0, "v2v",
                ).structure, "diagonal")
                value = self._calculate(context).value
                if expected is not None:
                    self.assertEqual(value, expected)
                else:
                    np.testing.assert_allclose(value, self._reference(context), rtol=3e-12, atol=4e-11)

    def test_complete_plan_replays_diagonal_and_rejects_changed_structure_before_read(self):
        context = self._fixture(1759, diagonal=True, source_dagger=True)
        selected = self._calculate(context, execute=False)
        complete = complete_calculation(
            selected, readers=selected.inputs.loader.input_readers,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "complete-diagonal-vv-plan.json.gz"
            save_calculation_plan(path, complete)
            replayed = calculate(
                plan_archive=path, configuration="cfg-complete-diagonal",
                propagator_map=context["prop_map"], vertex_map=context["reader"],
            )
            np.testing.assert_allclose(
                replayed.value, selected.run().value, rtol=3e-12, atol=4e-11,
            )

            reader = context["reader"]
            describe_block = reader.describe_block

            def generalize_source(component, momentum, time, sector, **options):
                metadata = describe_block(component, momentum, time, sector, **options)
                if component == context["source_component"] and sector == "v2v":
                    return replace(metadata, structure="general")
                return metadata

            with patch.object(reader, "describe_block", side_effect=generalize_source):
                with patch.object(reader, "block", side_effect=AssertionError("numeric read")):
                    with self.assertRaisesRegex(ValueError, "elemental structure"):
                        calculate(
                            plan_archive=path, configuration="cfg-complete-diagonal",
                            propagator_map=context["prop_map"], vertex_map=reader,
                        )


class CompleteDiagonalP2PTests(unittest.TestCase):
    population = 3
    modes = 1
    times = 3
    pool_id = "complete-diagonal-pool"

    def _fixture(self, *, seed, current_count, selected=None, weighted=False):
        rng = np.random.default_rng(seed)
        points = self.population if selected is None else len(selected)
        ordered_points = tuple(range(self.population)) if selected is None else tuple(selected)
        current_operator = LocalVectorCurrent().operator(3)
        current_component = current_operator.terms[0].component
        normal_component = ElementalComponent(("normal-v2v", seed))
        components = (current_component, normal_component)
        ci = {component: index for index, component in enumerate(components)}
        full_sectors = {
            "v2v": np.zeros((2, 1, self.times, self.modes, self.modes), dtype=np.complex128),
            "v2p": np.zeros((2, 1, self.times, self.modes, self.population, 3), dtype=np.complex128),
            "p2v": np.zeros((2, 1, self.times, self.population, 3, self.modes), dtype=np.complex128),
            "p2p": np.zeros((2, 1, self.times, self.population, 3, self.population, 3), dtype=np.complex128),
        }
        # Each complete P basis coefficient is an independent complex number;
        # color is not restricted to an identity or scalar matrix.
        diagonal_coefficients = _complex(rng, (self.times, self.population, 3))
        for time in (0, 2):
            for point in range(self.population):
                for color in range(3):
                    full_sectors["p2p"][ci[current_component], 0, time,
                                        point, color, point, color] = diagonal_coefficients[time, point, color]
        full_sectors["v2v"][ci[normal_component], 0, 2, 0, 0] = _complex(rng, ())

        full_propagators = {
            "vsv": _complex(rng, (self.times, self.times, 4, 4, self.modes, self.modes)),
            "vsp": _complex(rng, (self.times, self.times, 4, 4, self.modes, self.population, 3)),
            "psv": _complex(rng, (self.times, self.times, 4, 4, self.population, 3, self.modes)),
            "psp": _complex(rng, (self.times, self.times, 4, 4, self.population, 3, self.population, 3)),
        }
        sectors = {
            "v2v": full_sectors["v2v"],
            "v2p": np.take(full_sectors["v2p"], ordered_points, axis=4).copy(),
            "p2v": np.take(full_sectors["p2v"], ordered_points, axis=3).copy(),
            "p2p": np.take(np.take(full_sectors["p2p"], ordered_points, axis=3),
                            ordered_points, axis=5).copy(),
        }
        if weighted:
            factor = self.population / points
            sectors["v2p"] *= factor
            sectors["p2v"] *= factor
            sectors["p2p"][ci[current_component]] *= factor
        sampling_pool_ids = {
            (current_component, 0, time, side): self.pool_id
            for time in range(self.times) for side in ("bar", "field")
        }
        reader = ArrayElemental(
            sectors, components, phase_convention="anchor", cfg="cfg-complete-p2p",
            data_identity=f"complete-diagonal-p2p/{seed}/{ordered_points}",
            sampling_pool_ids=sampling_pool_ids,
            sampling_groups={self.pool_id: SamplingPoolSpec(
                self.pool_id, self.population, points,
            )},
            elemental_weighted=weighted,
            elemental_structures={("p2p", current_component, 0): "diagonal"},
        )
        prop_sectors = {
            "vsv": full_propagators["vsv"],
            "vsp": np.take(full_propagators["vsp"], ordered_points, axis=5).copy(),
            "psv": np.take(full_propagators["psv"], ordered_points, axis=4).copy(),
            "psp": np.take(np.take(full_propagators["psp"], ordered_points, axis=4),
                            ordered_points, axis=6).copy(),
        }
        prop_reader = ArrayPropagator(
            prop_sectors, time_convention="absolute", cfg="cfg-complete-p2p",
            data_identity=f"complete-diagonal-propagators/{seed}/{ordered_points}",
        )
        current_vertices = {
            f"current-{index}": Meson(
                reader, current_operator, source=True, phase_convention="anchor",
            ) for index in range(current_count)
        }
        q01 = Propagator("u", Tag(0, 0), Tag(1, 0))
        if current_count == 1:
            normal = Operator.from_terms(
                "normal-v2v", (OperatorTerm(0, normal_component, 0, 1),),
            )
            current_vertices["normal"] = Meson(reader, normal, phase_convention="anchor")
            diagram = Diagram(
                QuarkDiagram([[0, 1], [1, 0]]), [0, 2],
                ["current-0", "normal"], [None, q01],
            )
            prop_map = {q01: prop_reader}
        else:
            q10 = Propagator("u", Tag(1, 0), Tag(0, 0))
            diagram = Diagram(
                QuarkDiagram([[0, 1], [2, 0]]), [0, 2],
                ["current-0", "current-1"], [None, q01, q10],
            )
            prop_map = {q01: prop_reader, q10: prop_reader}
        return {
            "population": self.population,
            "sample_count": points,
            "ordered_points": ordered_points,
            "current_count": current_count,
            "current_component": current_component,
            "normal_component": normal_component,
            "component_index": ci,
            "full_sectors": full_sectors,
            "full_propagators": full_propagators,
            "reader": reader,
            "prop_reader": prop_reader,
            "diagram": diagram,
            "vertices": current_vertices,
            "prop_map": prop_map,
            "weighted": weighted,
        }

    def _calculate(self, context, *, execute=True):
        return calc_diagram_eval(
            context["diagram"], propagator_map=context["prop_map"],
            vertex_map=context["vertices"], configuration="cfg-complete-p2p",
            usedNe=self.modes, usedNp=context["sample_count"], optimize="none",
            execute=execute, elemental_weighted=context["weighted"],
        )

    @staticmethod
    def _full_trace(context):
        points = context["population"]
        sectors = context["full_sectors"]
        propagators = context["full_propagators"]
        ci = context["component_index"]
        current_term = LocalVectorCurrent().operator(3).terms[0]
        gamma5 = gamma(8)
        current_spin = gamma5 @ gamma(current_term.gamma).conj().T @ gamma5
        current_matrix = np.kron(
            current_spin,
            _elemental_matrix(sectors, ci[context["current_component"]], 0, points, 1).conj().T,
        )
        outgoing = _propagator_matrix(propagators, 0, 2, points, 1)
        incoming = _propagator_matrix(propagators, 2, 0, points, 1)
        if context["current_count"] == 1:
            normal_spatial = np.zeros((1 + points * 3, 1 + points * 3), dtype=np.complex128)
            normal_spatial[0, 0] = sectors["v2v"][ci[context["normal_component"]], 0, 2, 0, 0]
            normal_matrix = np.kron(gamma(0), normal_spatial)
            return np.trace(outgoing @ current_matrix @ incoming @ normal_matrix)
        second_matrix = np.kron(
            current_spin,
            _elemental_matrix(sectors, ci[context["current_component"]], 2, points, 1).conj().T,
        )
        return np.trace(outgoing @ current_matrix @ incoming @ second_matrix)

    @staticmethod
    def _point_diagonal_scene_count(calculation, *, current_count):
        problem = calculation.expanded.problem
        result = 0
        for term in problem.terms:
            p2p_count = sum(
                problem.specs[key].description.semantics.get("logical_sector") == "p2p"
                for key in term.inputs
                if problem.specs[key].description is not None
                and problem.specs[key].description.role == "elemental"
            )
            if p2p_count == current_count:
                result += 1
        return result

    def test_dense_complete_diagonal_p2p_sampling_matches_full_trace(self):
        for current_count, sample_count in ((1, 1), (2, 2)):
            with self.subTest(currents=current_count, sample_count=sample_count):
                for weighted in (False, True):
                    full = self._fixture(
                        seed=2801 + current_count, current_count=current_count,
                        weighted=weighted,
                    )
                    expected = self._full_trace(full)
                    self.assertEqual(full["reader"].describe_block(
                        full["current_component"], 0, 0, "p2p", usedNp=3,
                    ).structure, "diagonal")
                    diagonal_coefficients = np.diagonal(
                        full["full_sectors"]["p2p"][
                            full["component_index"][full["current_component"]], 0, 0,
                        ].reshape(9, 9),
                    )
                    self.assertGreater(np.linalg.norm(diagonal_coefficients.imag), 1e-3)
                    self.assertFalse(np.allclose(diagonal_coefficients, 1))
                    self.assertGreater(
                        np.linalg.norm(diagonal_coefficients - diagonal_coefficients[0]),
                        1e-3,
                    )

                    np.testing.assert_allclose(
                        self._calculate(full).value, expected, rtol=8e-11, atol=1e-8,
                    )

                    estimates = []
                    subset_scene_counts = []
                    for selected in combinations(range(self.population), sample_count):
                        sampled = self._fixture(
                            seed=2801 + current_count, current_count=current_count,
                            selected=selected, weighted=weighted,
                        )
                        calculation = self._calculate(sampled)
                        estimates.append(calculation.value)
                        subset_scene_counts.append(self._point_diagonal_scene_count(
                            calculation, current_count=current_count,
                        ))
                    np.testing.assert_allclose(
                        np.mean(estimates), expected, rtol=9e-11, atol=2e-8,
                    )
                    # The one-current graph has one equal-point scene; the
                    # two-current graph has same-point and distinct-point scenes.
                    expected_scenes = 1 if current_count == 1 else 2
                    self.assertEqual(subset_scene_counts, [expected_scenes] * len(estimates))


if __name__ == "__main__":
    unittest.main()
