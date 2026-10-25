"""Finite-population checks for declared diagonal P/P current inputs."""

from dataclasses import replace
from itertools import combinations, permutations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.adapters.sparse_elemental import SparseElemental
from lqcdweave.contracts.elemental import ElementalComponent, SparsePointRecordMetadata
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorPath, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval, calculate


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _matrix_from_sector_blocks(sectors, component_index, time, point_count, mode_count=1):
    """Assemble an explicit mode-plus-point matrix in spin-major order."""
    v2v = sectors["v2v"][component_index, 0, time]
    v2p = sectors["v2p"][component_index, 0, time].reshape(mode_count, point_count * 3)
    p2v = sectors["p2v"][component_index, 0, time].reshape(point_count * 3, mode_count)
    p2p = sectors["p2p"][component_index, 0, time].reshape(point_count * 3, point_count * 3)
    return np.block([[v2v, v2p], [p2v, p2p]])


def _matrix_from_propagator_blocks(sectors, source_time, sink_time, point_count, mode_count=1):
    """Assemble a propagator endpoint block as a 2D spin-major matrix."""
    point_dimension = point_count * 3
    top = np.concatenate((
        sectors["vsv"][source_time, sink_time],
        sectors["vsp"][source_time, sink_time].reshape(4, 4, mode_count, point_dimension),
    ), axis=3)
    bottom = np.concatenate((
        sectors["psv"][source_time, sink_time].reshape(4, 4, point_dimension, mode_count),
        sectors["psp"][source_time, sink_time].reshape(4, 4, point_dimension, point_dimension),
    ), axis=3)
    block = np.concatenate((top, bottom), axis=2)
    dimension = mode_count + point_dimension
    return block.transpose(0, 2, 1, 3).reshape(4 * dimension, 4 * dimension)


class _RecordReader:
    def __init__(self, records, point_count, identity):
        self.records = dict(records)
        self.point_count = point_count
        self.identity = identity
        self.reads = []

    def describe_record(self, component, momentum, time):
        record = self.records[component, momentum, time]
        indices, values = record["indices"], record["values"]
        return SparsePointRecordMetadata(
            "sparse", (self.point_count, self.point_count), len(indices),
            values.dtype, indices.dtype, indices.nbytes + values.nbytes,
            data_identity=f"{self.identity}/{component.key!r}/t{time}",
            selection_identity=f"{self.identity}/{component.key!r}/t{time}",
        )

    def read_record(self, component, momentum, time):
        self.reads.append((component, momentum, time))
        return self.records[component, momentum, time]

    def close_record(self, component, momentum, time, record):
        return None


class DiagonalCurrentFrontendTests(unittest.TestCase):
    population = 3
    mode_count = 1
    time_extent = 3
    pool_id = "same-ordered-point-sample"

    def _operators(self, *, mixed=False):
        base = LocalVectorCurrent().operator(3)
        diagonal_component = base.terms[0].component
        diagonal = base
        if not mixed:
            return (diagonal,), (diagonal_component,), (diagonal_component,)

        # This second imported path is also local in net displacement, but it
        # has a different registered identity and is deliberately undeclared.
        general_component = ElementalComponent.path((0, 3))
        general_term = replace(base.terms[0], component=general_component)
        general = Operator.from_terms("general-local-path-current", (general_term,))
        return (diagonal, general), (diagonal_component, general_component), (diagonal_component,)

    def _fixture(self, *, seed, current_count, mixed=False, selected=None,
                 population=None,
                 elemental_weighted=False, declared_diagonal=True):
        rng = np.random.default_rng(seed)
        population = self.population if population is None else population
        # ``selected`` identifies an ordered subset of the full population.
        point_count = population if selected is None else len(selected)
        operators, current_components, diagonal_components = self._operators(mixed=mixed)
        if not mixed:
            operators = tuple(operators[0] for _ in range(current_count))
            current_components = tuple(current_components[0] for _ in range(current_count))
        elif current_count != 2:
            raise ValueError("mixed fixtures require two current vertices")

        normal_component = ElementalComponent(("normal-v2v", seed))
        components = tuple(dict.fromkeys((*current_components, normal_component)))
        component_index = {component: index for index, component in enumerate(components)}
        ne, lt = self.mode_count, self.time_extent
        source_points = tuple(range(population)) if selected is None else tuple(selected)

        full_sectors = {
            "v2v": _complex(rng, (len(components), 1, lt, ne, ne)),
            "v2p": _complex(rng, (len(components), 1, lt, ne, population, 3)),
            "p2v": _complex(rng, (len(components), 1, lt, population, 3, ne)),
            "p2p": np.zeros(
                (len(components), 1, lt, population, 3, population, 3),
                dtype=np.complex128,
            ),
        }
        for component in current_components:
            ci = component_index[component]
            for time in range(lt):
                if component == diagonal_components[0]:
                    color_blocks = _complex(rng, (population, 3, 3))
                    for point in range(population):
                        full_sectors["p2p"][ci, 0, time, point, :, point, :] = color_blocks[point]
                else:
                    full_sectors["p2p"][ci, 0, time] = _complex(
                        rng, (population, 3, population, 3),
                    )

        # The normal insertion is V/V only. Its point sectors stay zero.
        normal_index = component_index[normal_component]
        for time in range(lt):
            full_sectors["v2v"][normal_index, 0, time] = _complex(rng, (ne, ne))
            full_sectors["v2p"][normal_index, 0, time] = 0
            full_sectors["p2v"][normal_index, 0, time] = 0

        full_propagators = {
            "vsv": _complex(rng, (lt, lt, 4, 4, ne, ne)),
            "vsp": _complex(rng, (lt, lt, 4, 4, ne, population, 3)),
            "psv": _complex(rng, (lt, lt, 4, 4, population, 3, ne)),
            "psp": _complex(rng, (lt, lt, 4, 4, population, 3, population, 3)),
        }

        if selected is None:
            sectors = full_sectors
            propagators = full_propagators
            pool_sample_count = population
            dense_elemental = ArrayElemental(
                sectors, components, phase_convention="anchor", cfg="cfg-diagonal",
                data_identity=f"full-elementals/{seed}",
                elemental_weighted=elemental_weighted,
                point_diagonal_components=(diagonal_components[0],) if declared_diagonal else (),
            )
            provider = None
        else:
            sectors = {
                "v2v": full_sectors["v2v"],
                "v2p": np.take(full_sectors["v2p"], source_points, axis=4).copy(),
                "p2v": np.take(full_sectors["p2v"], source_points, axis=3).copy(),
            }
            if elemental_weighted:
                point_factor = population / point_count
                sectors["v2p"] *= point_factor
                sectors["p2v"] *= point_factor
            records = {}
            for component in current_components:
                ci = component_index[component]
                for time in range(lt):
                    indices, values = [], []
                    for local_left, global_left in enumerate(source_points):
                        for local_right, global_right in enumerate(source_points):
                            if component == diagonal_components[0] and local_left != local_right:
                                continue
                            indices.append((local_left, local_right))
                            values.append(
                                full_sectors["p2p"][ci, 0, time,
                                                   global_left, :, global_right, :]
                            )
                    indices = np.asarray(indices, dtype=np.int64).reshape((-1, 2))
                    values = np.asarray(values, dtype=np.complex128).reshape((-1, 3, 3))
                    if elemental_weighted:
                        for row, (left, right) in enumerate(indices):
                            if left == right:
                                factor = population / point_count
                            else:
                                factor = population * (population - 1) / (
                                    point_count * (point_count - 1)
                                )
                            values[row] *= factor
                    records[component, 0, time] = {
                        "type": "sparse", "indices": indices, "values": values,
                    }
            dense_elemental = ArrayElemental(
                sectors, components, phase_convention="anchor", cfg="cfg-diagonal",
                data_identity=f"sampled-dense-sectors/{seed}/{source_points}",
                sampling_pool_ids={
                    (component, 0, time, side): self.pool_id
                    for component in current_components
                    for time in range(lt)
                    for side in ("bar", "field")
                },
                sampling_groups={self.pool_id: SamplingPoolSpec(
                    self.pool_id, population, point_count,
                )},
                elemental_weighted=elemental_weighted,
                point_diagonal_components=(diagonal_components[0],) if declared_diagonal else (),
            )
            provider = _RecordReader(records, point_count, f"sampled-records/{seed}/{source_points}")
            dense_elemental = SparseElemental(dense_elemental, provider)
            propagators = {
                "vsv": full_propagators["vsv"],
                "vsp": np.take(full_propagators["vsp"], source_points, axis=5).copy(),
                "psv": np.take(full_propagators["psv"], source_points, axis=4).copy(),
                "psp": np.take(
                    np.take(full_propagators["psp"], source_points, axis=4),
                    source_points, axis=6,
                ).copy(),
            }

        # Separate point populations are unnecessary here: every P endpoint
        # in a current instance uses the same explicitly ordered pool.
        if selected is None:
            dense_elemental.sampling_pool_ids = {
                (component, 0, time, side): self.pool_id
                for component in current_components
                for time in range(lt)
                for side in ("bar", "field")
            }
            dense_elemental.sampling_groups = {
                self.pool_id: SamplingPoolSpec(self.pool_id, population, pool_sample_count),
            }

        prop_reader = ArrayPropagator(
            propagators, time_convention="absolute", cfg="cfg-diagonal",
            data_identity=f"propagators/{seed}/{source_points}",
        )
        vertices = {
            f"current-{index}": Meson(
                dense_elemental, operator, source=True, phase_convention="anchor",
            )
            for index, operator in enumerate(operators)
        }
        q01 = Propagator("u", Tag(0, 0), Tag(1, 0))
        if current_count == 1:
            normal = Operator.from_terms(
                "normal-v2v", (OperatorTerm(0, normal_component, 0, 1),),
            )
            vertices["normal"] = Meson(dense_elemental, normal, phase_convention="anchor")
            graph = Diagram(
                QuarkDiagram([[0, 1], [1, 0]]), [0, 2],
                ["current-0", "normal"], [None, q01],
            )
            prop_map = {q01: prop_reader}
        else:
            q10 = Propagator("u", Tag(1, 0), Tag(0, 0))
            graph = Diagram(
                QuarkDiagram([[0, 1], [2, 0]]), [0, 2],
                ["current-0", "current-1"], [None, q01, q10],
            )
            prop_map = {q01: prop_reader, q10: prop_reader}

        return {
            "population": population,
            "sample_count": point_count,
            "selected": source_points,
            "operators": operators,
            "components": current_components,
            "diagonal_component": diagonal_components[0],
            "normal_component": normal_component,
            "component_indices": component_index,
            "full_sectors": full_sectors,
            "full_propagators": full_propagators,
            "reader": dense_elemental,
            "provider": provider,
            "prop_reader": prop_reader,
            "diagram": graph,
            "vertices": vertices,
            "prop_map": prop_map,
            "elemental_weighted": elemental_weighted,
        }

    @staticmethod
    def _evaluate(context, *, usedNp=None, execute=True):
        return calc_diagram_eval(
            context["diagram"], propagator_map=context["prop_map"],
            vertex_map=context["vertices"], configuration="cfg-diagonal",
            usedNe=1, usedNp=context["sample_count"] if usedNp is None else usedNp,
            optimize="none", execute=execute,
            elemental_weighted=context["elemental_weighted"],
        )

    @staticmethod
    def _full_trace(context):
        points = context["population"]
        full_sectors = context["full_sectors"]
        full_propagators = context["full_propagators"]
        ci = context["component_indices"]
        current = context["operators"][0].terms[0]
        g5 = gamma(8)
        current_spin = g5 @ gamma(current.gamma).conj().T @ g5
        current_spatial = _matrix_from_sector_blocks(
            full_sectors, ci[current.component], 0, points,
        ).conj().T
        current_matrix = np.kron(current_spin, current_spatial)

        if len(context["operators"]) == 1:
            normal_component = context["normal_component"]
            mode_count = 1
            normal_spatial = np.zeros(
                (mode_count + points * 3, mode_count + points * 3),
                dtype=np.complex128,
            )
            normal_spatial[:mode_count, :mode_count] = (
                full_sectors["v2v"][ci[normal_component], 0, 2]
            )
            normal_matrix = np.kron(gamma(0), normal_spatial)
            outgoing = _matrix_from_propagator_blocks(full_propagators, 0, 2, points)
            incoming = _matrix_from_propagator_blocks(full_propagators, 2, 0, points)
            return np.trace(outgoing @ current_matrix @ incoming @ normal_matrix)

        second = context["operators"][1].terms[0]
        second_spin = g5 @ gamma(second.gamma).conj().T @ g5
        second_spatial = _matrix_from_sector_blocks(
            full_sectors, ci[second.component], 2, points,
        ).conj().T
        second_matrix = np.kron(second_spin, second_spatial)
        forward = _matrix_from_propagator_blocks(full_propagators, 0, 2, points)
        backward = _matrix_from_propagator_blocks(full_propagators, 2, 0, points)
        return np.trace(forward @ current_matrix @ backward @ second_matrix)

    @staticmethod
    def _sparse_scene_count(calculation, *, bridges_per_scene, same_point=None):
        by_scene = {}
        for bridge in calculation.expanded.problem.sparse_bridges:
            by_scene.setdefault(bridge.scene, []).append(bridge)
        return sum(
            len(bridges) == bridges_per_scene
            and (same_point is None or all(item.same_point is same_point for item in bridges))
            for bridges in by_scene.values()
        )

    def test_single_point_diagonal_current_uses_one_point_and_is_unbiased(self):
        population = self.population
        for weighted in (False, True):
            full = self._fixture(
                seed=221, current_count=1, elemental_weighted=weighted,
            )
            expected = self._full_trace(full)
            actual = self._evaluate(full).value
            np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=6e-9)

            estimates = []
            for selected in combinations(range(population), 1):
                sampled = self._fixture(
                    seed=221, current_count=1, selected=selected,
                    elemental_weighted=weighted,
                )
                calculation = self._evaluate(sampled)
                estimates.append(calculation.value)
                self.assertEqual(
                    self._sparse_scene_count(calculation, bridges_per_scene=1, same_point=True),
                    1,
                )
                self.assertTrue(sampled["provider"].reads)
            np.testing.assert_allclose(np.mean(estimates), expected, rtol=5e-11, atol=8e-9)

    def test_two_same_pool_diagonal_currents_use_two_scenes_and_are_unbiased(self):
        population = self.population
        for weighted in (False, True):
            full = self._fixture(
                seed=488, current_count=2, elemental_weighted=weighted,
            )
            expected = self._full_trace(full)
            actual = self._evaluate(full).value
            np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=8e-9)

            estimates = []
            raw_scene_counts = []
            for selected in combinations(range(population), 2):
                sampled = self._fixture(
                    seed=488, current_count=2, selected=selected,
                    elemental_weighted=weighted,
                )
                calculation = self._evaluate(sampled)
                estimates.append(calculation.value)
                raw_scene_counts.append(self._sparse_scene_count(
                    calculation, bridges_per_scene=2, same_point=True,
                ))
                self.assertGreater(len(sampled["provider"].reads), 0)
            if not weighted:
                self.assertEqual(raw_scene_counts, [2, 2, 2])
            else:
                self.assertTrue(all(1 <= count <= 2 for count in raw_scene_counts))
            np.testing.assert_allclose(np.mean(estimates), expected, rtol=7e-11, atol=1e-8)

    def test_mixed_diagonal_and_general_path_keeps_three_independent_point_legs(self):
        population = 4
        estimates = []
        for selected in combinations(range(population), 3):
            sampled = self._fixture(
                seed=671, current_count=2, mixed=True, selected=selected,
                population=population,
                elemental_weighted=False,
            )
            calculation = self._evaluate(sampled)
            estimates.append(calculation.value)
            self.assertEqual(self._sparse_scene_count(
                calculation, bridges_per_scene=2,
            ), 5)
        full = self._fixture(
            seed=671, current_count=2, mixed=True, population=population,
            elemental_weighted=False,
        )
        expected = self._full_trace(full)
        np.testing.assert_allclose(np.mean(estimates), expected, rtol=8e-11, atol=1e-8)

    def test_undeclared_local_path_stays_general_and_cannot_use_one_point(self):
        context = self._fixture(
            seed=927, current_count=1, selected=(0,), declared_diagonal=False,
        )
        p2p = context["reader"].describe_block(
            context["components"][0], 0, 0, "p2p", usedNp=1,
        )
        self.assertFalse(p2p.point_diagonal)
        self.assertEqual(context["reader"].point_diagonal_components, ())
        with self.assertRaisesRegex(ValueError, "not estimable from a sample"):
            self._evaluate(context, execute=False)
        self.assertEqual(context["provider"].reads, [])

    def test_elemental_weighted_prefix_narrowing_keeps_the_full_population_trace(self):
        population, produced_count, used_count = 4, 3, 2
        expected_context = self._fixture(
            seed=1203, current_count=2, population=population,
            elemental_weighted=True,
        )
        expected = self._full_trace(expected_context)
        estimates = []
        for selected in permutations(range(population), produced_count):
            sampled = self._fixture(
                seed=1203, current_count=2, selected=selected,
                population=population,
                elemental_weighted=True,
            )
            calculation = self._evaluate(sampled, usedNp=used_count)
            estimates.append(calculation.value)
        np.testing.assert_allclose(np.mean(estimates), expected, rtol=1e-10, atol=2e-8)

    def test_saved_dense_plan_preserves_point_diagonal_declaration(self):
        from lqcdweave.contraction.correlator import save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation

        context = self._fixture(
            seed=1451, current_count=2, selected=None, population=3,
            elemental_weighted=False, declared_diagonal=True,
        )
        selected = calculate(
            context["diagram"], propagator_map=context["prop_map"],
            vertex_map=context["vertices"], configuration="cfg-diagonal",
            usedNe=1, usedNp=3, optimize="none", execute=False,
        )
        complete = complete_calculation(
            selected, readers=selected.inputs.loader.input_readers,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "point-diagonal-plan.json.gz"
            save_calculation_plan(path, complete)

            replayed = calculate(
                plan_archive=path, configuration="cfg-diagonal",
                propagator_map=context["prop_map"], vertex_map=context["reader"],
            )
            np.testing.assert_allclose(
                replayed.value, selected.run().value, rtol=5e-11, atol=8e-9,
            )

            reader = context["reader"]
            describe_block = reader.describe_block

            def changed_point_diagonal(component, momentum, time, sector, **options):
                metadata = describe_block(component, momentum, time, sector, **options)
                if sector == "p2p" and component == context["diagonal_component"]:
                    return replace(metadata, point_diagonal=False, structure="general")
                return metadata

            with patch.object(reader, "describe_block", side_effect=changed_point_diagonal):
                with patch.object(reader, "block", side_effect=AssertionError("numeric read")):
                    with self.assertRaisesRegex(ValueError, "saved input point support"):
                        calculate(
                            plan_archive=path, configuration="cfg-diagonal",
                            propagator_map=context["prop_map"], vertex_map=reader,
                        )


if __name__ == "__main__":
    unittest.main()
