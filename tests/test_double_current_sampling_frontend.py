"""Finite-population estimator checks for two P2P currents at real times."""

import unittest
from itertools import combinations

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.adapters.sparse_elemental import SparseElemental
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent, SparsePointRecordMetadata
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Propagator
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calc_diagram_eval


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


class _SelectedCurrentRecords:
    def __init__(self, component, records, point_count, selection_key):
        self.component = component
        self.records = dict(records)
        self.point_count = point_count
        self.selection_key = selection_key
        self.read_times = []
        self.closed = []

    def describe_record(self, component, momentum, t):
        record = self.records[component, momentum, t]
        indices, values = record["indices"], record["values"]
        return SparsePointRecordMetadata(
            "sparse", (self.point_count, self.point_count), len(indices),
            values.dtype, indices.dtype,
            indices.nbytes + values.nbytes,
            data_identity=f"double-current/{self.selection_key}/t{t}",
            selection_identity=(
                f"component={component.key!r};momentum={momentum};time={t};"
                f"sample={self.selection_key}"
            ),
        )

    def read_record(self, component, momentum, t):
        self.read_times.append(t)
        return self.records[component, momentum, t]

    def close_record(self, component, momentum, t, record):
        self.closed.append((component, momentum, t, record))


def _coo_from_dense(value, *, population=None, sampled_points=None, elemental_weighted=False):
    n = value.shape[0]
    indices = np.asarray([(i, j) for i in range(n) for j in range(n)], dtype=np.int64)
    values = np.asarray([value[i, :, j, :] for i, j in indices], dtype=np.complex128)
    if elemental_weighted:
        one_point_weight = population / sampled_points
        factors = np.full(len(indices), one_point_weight, dtype=np.float64)
        distinct = indices[:, 0] != indices[:, 1]
        if np.any(distinct):
            factors[distinct] = (
                population * (population - 1)
                / (sampled_points * (sampled_points - 1))
            )
        values *= factors[:, None, None]
    return {"type": "sparse", "indices": indices, "values": values}


class DoubleCurrentSamplingFrontendTests(unittest.TestCase):
    def _context(self, population, selected_count, *, pool_mode, sets, seed,
                 elemental_weighted=False):
        rng = np.random.default_rng(seed)
        early_time, late_time, lt, ne = 0, 2, 3, 1
        current = LocalVectorCurrent().operator(3)
        component = current.terms[0].component
        gamma_index = current.terms[0].gamma
        weight = complex(current.terms[0].coefficient)

        full_early = _complex(rng, (population, 3, population, 3))
        full_late = _complex(rng, (population, 3, population, 3))
        # The two physical insertions and both propagator time directions are
        # independent arrays so that a wrong anchor/time lookup changes the
        # result rather than merely relabeling an identical fixture.
        peram_early_to_late = _complex(rng, (4, 4, population, 3, population, 3))
        peram_late_to_early = _complex(rng, (4, 4, population, 3, population, 3))
        self.assertGreater(np.max(np.abs(full_early - full_late)), 1e-6)
        self.assertGreater(np.max(np.abs(peram_early_to_late - peram_late_to_early)), 1e-6)

        group_at_time = (
            {early_time: "shared-across-times", late_time: "shared-across-times"}
            if pool_mode == "same"
            else {early_time: "early-pool", late_time: "late-pool"}
        )
        selection_key = tuple(
            (group, tuple(sets[group])) for group in sorted(set(group_at_time.values()))
        )
        groups = {
            group_id: SamplingPoolSpec(group_id, population, len(sets[group_id]))
            for group_id in set(group_at_time.values())
        }
        point_pool_ids = {
            (component, 0, time, side): group_at_time[time]
            for time in (early_time, late_time)
            for side in ("bar", "field")
        }

        local_n = population if selected_count == population else selected_count
        zero_elemental = {
            "v2v": np.zeros((1, 1, lt, ne, ne), dtype=np.complex128),
            "v2p": np.zeros((1, 1, lt, ne, local_n, 3), dtype=np.complex128),
            "p2v": np.zeros((1, 1, lt, local_n, 3, ne), dtype=np.complex128),
        }
        if selected_count == population:
            p2p = np.zeros((1, 1, lt, population, 3, population, 3), dtype=np.complex128)
            p2p[0, 0, early_time] = full_early
            p2p[0, 0, late_time] = full_late
            elemental = ArrayElemental(
                {**zero_elemental, "p2p": p2p}, (component,),
                phase_convention="anchor", cfg="cfg-double",
                data_identity=f"full-elemental/{population}",
                sampling_pool_ids=point_pool_ids, sampling_groups=groups,
                elemental_weighted=elemental_weighted,
            )
            provider = None
            selected_sets = {group: tuple(range(population)) for group in groups}
        else:
            elemental = ArrayElemental(
                zero_elemental, (component,), phase_convention="anchor",
                cfg="cfg-double", data_identity=f"dense-v-sectors/{selection_key}",
                sampling_pool_ids=point_pool_ids, sampling_groups=groups,
                elemental_weighted=elemental_weighted,
            )
            selected_sets = {group: tuple(sets[group]) for group in groups}
            records = {}
            for time, matrix in ((early_time, full_early), (late_time, full_late)):
                chosen = selected_sets[group_at_time[time]]
                block = np.take(np.take(matrix, chosen, axis=0), chosen, axis=2)
                records[component, 0, time] = _coo_from_dense(
                    block, population=population, sampled_points=selected_count,
                    elemental_weighted=elemental_weighted,
                )
            provider = _SelectedCurrentRecords(
                component, records, selected_count,
                selection_key=tuple((group, selected_sets[group]) for group in sorted(selected_sets)),
            )
            elemental = SparseElemental(elemental, provider)

        psp_size = local_n if selected_count != population else population
        psp = np.zeros((lt, lt, 4, 4, psp_size, 3, psp_size, 3), dtype=np.complex128)
        early_points = selected_sets[group_at_time[early_time]]
        late_points = selected_sets[group_at_time[late_time]]
        # ArrayPropagator stores [source_time, sink_time, ...].
        psp[early_time, late_time] = np.take(
            np.take(peram_early_to_late, late_points, axis=2), early_points, axis=4,
        )
        psp[late_time, early_time] = np.take(
            np.take(peram_late_to_early, early_points, axis=2), late_points, axis=4,
        )
        propagators = {
            "vsv": np.zeros((lt, lt, 4, 4, ne, ne), dtype=np.complex128),
            "psv": np.zeros((lt, lt, 4, 4, psp_size, 3, ne), dtype=np.complex128),
            "vsp": np.zeros((lt, lt, 4, 4, ne, psp_size, 3), dtype=np.complex128),
            "psp": psp,
        }
        prop_reader = ArrayPropagator(
            propagators, time_convention="absolute", cfg="cfg-double",
            data_identity=f"peram/{pool_mode}/{selection_key if selected_count != population else 'full'}",
        )
        prop_calls = []
        original_describe = prop_reader.describe_block

        def record_describe(sink, source, sector="vsv", **selection):
            prop_calls.append((sink, source, sector))
            return original_describe(sink, source, sector, **selection)

        prop_reader.describe_block = record_describe
        q01 = Propagator("u", Tag(0, 0), Tag(1, 0))
        q10 = Propagator("u", Tag(1, 0), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [early_time, late_time],
            ["early", "late"], [None, q01, q10],
        )
        vertices = {
            "early": Meson(elemental, current, phase_convention="anchor"),
            "late": Meson(elemental, current, phase_convention="anchor"),
        }
        return {
            "population": population, "selected_count": selected_count,
            "pool_mode": pool_mode, "group_at_time": group_at_time,
            "groups": groups, "sets": selected_sets,
            "current": current, "component": component, "gamma": gamma_index,
            "weight": weight, "full_early": full_early, "full_late": full_late,
            "peram_early_to_late": peram_early_to_late,
            "peram_late_to_early": peram_late_to_early,
            "provider": provider, "prop_calls": prop_calls,
            "prop_reader": prop_reader, "diagram": diagram, "vertices": vertices,
            "prop_map": {q01: prop_reader, q10: prop_reader},
        }

    def _evaluate(self, context, *, groups_independent=None, execute=True,
                  elemental_weighted=False):
        return calc_diagram_eval(
            context["diagram"],
            propagator_map=context["prop_map"], vertex_map=context["vertices"],
            configuration="cfg-double", usedNe=1,
            usedNp=context["selected_count"], groups_independent=groups_independent,
            optimize="none", execute=execute, elemental_weighted=elemental_weighted,
        )

    @staticmethod
    def _explicit_full_point_trace(context):
        current = context["current"].terms[0]
        gamma_matrix = gamma(current.gamma)
        m = context["population"]
        spin_color = 4 * m * 3
        peram_early_to_late = context["peram_early_to_late"].transpose(
            0, 2, 3, 1, 4, 5,
        ).reshape(spin_color, spin_color)
        peram_late_to_early = context["peram_late_to_early"].transpose(
            0, 2, 3, 1, 4, 5,
        ).reshape(spin_color, spin_color)
        current_early = np.einsum(
            "AD,naod->AnaDod", gamma_matrix, context["full_early"], optimize=False,
        ).reshape(spin_color, spin_color)
        current_late = np.einsum(
            "CB,dfgb->CdfBgb", gamma_matrix, context["full_late"], optimize=False,
        ).reshape(spin_color, spin_color)
        total = np.trace(
            peram_early_to_late @ current_early
            @ peram_late_to_early @ current_late
        )
        return context["weight"] ** 2 * total

    def test_same_pool_cross_time_n4_of_m5_averages_to_matrix_trace(self):
        population, selected_count = 5, 4
        for elemental_weighted in (False, True):
            full_context = self._context(
                population, population, pool_mode="same",
                sets={"shared-across-times": tuple(range(population))}, seed=711,
                elemental_weighted=elemental_weighted,
            )
            expected = self._explicit_full_point_trace(full_context)
            full_result = self._evaluate(
                full_context, elemental_weighted=elemental_weighted,
            ).value
            np.testing.assert_allclose(full_result, expected, rtol=2e-11, atol=2e-9)

            estimates = []
            for sample in combinations(range(population), selected_count):
                sampled = self._context(
                    population, selected_count, pool_mode="same",
                    sets={"shared-across-times": sample}, seed=711,
                    elemental_weighted=elemental_weighted,
                )
                result = self._evaluate(
                    sampled, elemental_weighted=elemental_weighted,
                ).value
                estimates.append(result)
                self.assertEqual(set(sampled["provider"].read_times), {0, 2})
                self.assertIn((2, 0, "psp"), sampled["prop_calls"])
                self.assertIn((0, 2, "psp"), sampled["prop_calls"])
            self.assertEqual(len(estimates), 5)
            np.testing.assert_allclose(np.mean(estimates), expected, rtol=3e-11, atol=4e-9)

    def test_independent_current_pools_need_declaration_and_are_unbiased(self):
        population, selected_count = 3, 2
        full_sets = {"early-pool": tuple(range(population)),
                     "late-pool": tuple(range(population))}
        first_sample = (0, 1)
        subsets = tuple(combinations(range(population), selected_count))
        for elemental_weighted in (False, True):
            full_context = self._context(
                population, population, pool_mode="independent", sets=full_sets, seed=881,
                elemental_weighted=elemental_weighted,
            )
            expected = self._explicit_full_point_trace(full_context)
            full_result = self._evaluate(
                full_context, groups_independent=True,
                elemental_weighted=elemental_weighted,
            ).value
            np.testing.assert_allclose(full_result, expected, rtol=2e-11, atol=2e-9)

            undisclosed = self._context(
                population, selected_count, pool_mode="independent",
                sets={"early-pool": first_sample, "late-pool": (1, 2)}, seed=881,
                elemental_weighted=elemental_weighted,
            )
            with self.assertRaisesRegex(ValueError, "multiple nonempty sampling groups require an explicit independence"):
                self._evaluate(
                    undisclosed, groups_independent=None, execute=False,
                    elemental_weighted=elemental_weighted,
                )
            self.assertEqual(undisclosed["provider"].read_times, [])

            estimates = []
            for early in subsets:
                for late in subsets:
                    sampled = self._context(
                        population, selected_count, pool_mode="independent",
                        sets={"early-pool": early, "late-pool": late}, seed=881,
                        elemental_weighted=elemental_weighted,
                    )
                    estimates.append(self._evaluate(
                        sampled, groups_independent=True,
                        elemental_weighted=elemental_weighted,
                    ).value)
                    self.assertEqual(set(sampled["provider"].read_times), {0, 2})
            self.assertEqual(len(estimates), len(subsets) ** 2)
            np.testing.assert_allclose(np.mean(estimates), expected, rtol=4e-11, atol=5e-9)

    def test_same_pool_four_legs_rejects_unidentifiable_n3_of_m5(self):
        context = self._context(
            5, 3, pool_mode="same", sets={"shared-across-times": (0, 1, 4)},
            seed=1199,
        )
        with self.assertRaisesRegex(ValueError, "not estimable from a sample of 3 out of 5"):
            self._evaluate(context, execute=False)
        self.assertEqual(context["provider"].read_times, [])

    def test_temporal_current_uses_actual_bar_field_times_and_p2p_records(self):
        rng = np.random.default_rng(2301)
        population, ne, lt = 3, 1, 3
        current = ConservedVectorInsertion(direction=3).operator()
        normal_component = ElementalComponent(0)
        normal = Operator.from_terms(
            "normal", (OperatorTerm(0, normal_component, 0, 1),),
        )
        current_terms = current.terms
        current_record_keys = tuple(dict.fromkeys(
            (term.component, term.anchor_time_offset) for term in current_terms
        ))
        components = tuple(dict.fromkeys(
            [term.component for term in current_terms] + [normal_component]
        ))
        component_indices = {component: index for index, component in enumerate(components)}
        full_records = {
            key: _complex(rng, (population, 3, population, 3))
            for key in current_record_keys
        }

        pool_at_time = {0: "pool-time-0", 1: "pool-time-1"}
        groups = {
            group_id: SamplingPoolSpec(group_id, population, population)
            for group_id in pool_at_time.values()
        }
        pool_ids = {
            (component, 0, time, side): pool_at_time[time]
            for component, _anchor in current_record_keys
            for time in pool_at_time
            for side in ("bar", "field")
        }
        zero_v2v = np.zeros((len(components), 1, lt, ne, ne), dtype=np.complex128)
        normal_data = _complex(rng, (ne, ne)).astype(np.complex128)
        zero_v2v[component_indices[normal_component], 0, 2] = normal_data
        elemental_sectors = {
            "v2v": zero_v2v,
            "v2p": np.zeros((len(components), 1, lt, ne, population, 3), dtype=np.complex128),
            "p2v": np.zeros((len(components), 1, lt, population, 3, ne), dtype=np.complex128),
        }
        dense_p2p = np.zeros(
            (len(components), 1, lt, population, 3, population, 3), dtype=np.complex128,
        )
        records = {}
        for (component, anchor), matrix in full_records.items():
            dense_p2p[component_indices[component], 0, anchor] = matrix
            # Full-population weighting is the identity, but explicitly build
            # the sparse actualvalues through the weighted COO path as well.
            records[component, 0, anchor] = _coo_from_dense(
                matrix, population=population, sampled_points=population,
                elemental_weighted=True,
            )
        record_reader = _SelectedCurrentRecords(
            current_record_keys[0][0], records, population, selection_key="temporal-full-pool",
        )

        peram_vsp = np.zeros((lt, lt, 4, 4, ne, population, 3), dtype=np.complex128)
        peram_psv = np.zeros((lt, lt, 4, 4, population, 3, ne), dtype=np.complex128)
        forward_out = _complex(rng, (4, 4, ne, population, 3))
        backward_out = _complex(rng, (4, 4, ne, population, 3))
        forward_in = _complex(rng, (4, 4, population, 3, ne))
        backward_in = _complex(rng, (4, 4, population, 3, ne))
        # ArrayPropagator storage is [source_time, sink_time, ...].
        peram_vsp[1, 2] = forward_out
        peram_vsp[0, 2] = backward_out
        peram_psv[2, 0] = forward_in
        peram_psv[2, 1] = backward_in
        prop_reader = ArrayPropagator(
            {
                "vsv": np.zeros((lt, lt, 4, 4, ne, ne), dtype=np.complex128),
                "vsp": peram_vsp,
                "psv": peram_psv,
                "psp": np.zeros((lt, lt, 4, 4, population, 3, population, 3), dtype=np.complex128),
            },
            time_convention="absolute", cfg="cfg-temporal",
            data_identity="temporal-endpoint-propagators",
        )
        prop_calls = []
        describe = prop_reader.describe_block

        def log_prop_request(sink, source, sector="vsv", **selection):
            prop_calls.append((sink, source, sector))
            return describe(sink, source, sector, **selection)

        prop_reader.describe_block = log_prop_request
        q01 = Propagator("u", Tag(0, 0), Tag(1, 0))
        q10 = Propagator("u", Tag(1, 0), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [0, 2], ["temporal", "normal"],
            [None, q01, q10],
        )
        prop_map = {q01: prop_reader, q10: prop_reader}
        temporal_values = {}
        for elemental_weighted in (False, True):
            sampling_elemental = ArrayElemental(
                elemental_sectors, components, phase_convention="anchor", cfg="cfg-temporal",
                data_identity="temporal-dense-v-sectors",
                sampling_pool_ids=pool_ids, sampling_groups=groups,
                elemental_weighted=elemental_weighted,
            )
            sparse_elemental = SparseElemental(sampling_elemental, record_reader)
            dense_elemental = ArrayElemental(
                {**elemental_sectors, "p2p": dense_p2p}, components,
                phase_convention="anchor", cfg="cfg-temporal",
                data_identity="temporal-full-elemental",
                sampling_pool_ids=pool_ids, sampling_groups=groups,
                elemental_weighted=elemental_weighted,
            )
            sparse_vertices = {
                "temporal": Meson(sparse_elemental, current, phase_convention="anchor"),
                "normal": Meson(sparse_elemental, normal, phase_convention="anchor"),
            }
            dense_vertices = {
                "temporal": Meson(dense_elemental, current, phase_convention="anchor"),
                "normal": Meson(dense_elemental, normal, phase_convention="anchor"),
            }
            sparse_value = calc_diagram_eval(
                diagram, propagator_map=prop_map, vertex_map=sparse_vertices,
                configuration="cfg-temporal", usedNe=1, usedNp=population,
                groups_independent=True, optimize="none",
                elemental_weighted=elemental_weighted,
            ).value
            dense_value = calc_diagram_eval(
                diagram, propagator_map=prop_map, vertex_map=dense_vertices,
                configuration="cfg-temporal", usedNe=1, usedNp=population,
                groups_independent=True, optimize="none",
                elemental_weighted=elemental_weighted,
            ).value
            temporal_values[elemental_weighted] = (sparse_value, dense_value)

        self.assertNotEqual(full_records[current_record_keys[0]][0, 0, 0, 0],
                            full_records[current_record_keys[1]][0, 0, 0, 0])
        self.assertIn((2, 1, "vsp"), prop_calls)
        self.assertIn((0, 2, "psv"), prop_calls)
        self.assertIn((2, 0, "vsp"), prop_calls)
        self.assertIn((1, 2, "psv"), prop_calls)
        self.assertEqual(set(record_reader.read_times), {0, 1})
        for sparse_value, dense_value in temporal_values.values():
            np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-11, atol=3e-9)

        # Independently enumerate the full point assignments for both actual
        # current endpoint pairs. These records use phase_convention="anchor":
        # the p2p block is stored at the path anchor, with its raw left/right
        # point axes representing physical bar/field respectively. In the
        # contraction network the outgoing VSP point is attached to raw-left
        # (bar_time), while the incoming PSV point is attached to raw-right
        # (field_time). The propagator arrays below are source-time first.
        term_endpoint_pairs = {
            term.current_term.link: (
                (term.anchor_time_offset + term.component.time_offsets[0]) % lt,
                (term.anchor_time_offset + term.component.time_offsets[1]) % lt,
            )
            for term in current.terms
        }
        self.assertEqual(term_endpoint_pairs, {"forward": (0, 1), "backward": (1, 0)})
        gamma_matrix = gamma(current.terms[0].gamma)
        expected = 0j
        old_field_bar_mapping = 0j
        normal_factor = np.einsum("CB,PO->CBPO", gamma(0), normal_data, optimize=False)
        for term in current.terms:
            bar_time = (term.anchor_time_offset + term.component.time_offsets[0]) % lt
            field_time = (term.anchor_time_offset + term.component.time_offsets[1]) % lt
            # The first index of peram_vsp is its point/source endpoint, so it
            # follows the raw-left physical bar endpoint. The second index of
            # peram_psv is its point/sink endpoint, so it follows raw-right
            # physical field. No midpoint phase conversion is involved here.
            out = peram_vsp[bar_time, 2]
            incoming = peram_psv[2, field_time]
            raw = full_records[term.component, term.anchor_time_offset]
            current_factor = np.einsum("AD,naod->ADnaod", gamma(term.gamma), raw, optimize=False)
            term_value = np.einsum(
                "BAOna,DCodP,ADnaod,CBPO->",
                out, incoming, current_factor, normal_factor, optimize=False,
            )
            expected += term.coefficient * term_value
            # Negative control: this was the old assertion's field->VSP and
            # bar->PSV selection. The endpoint labels look plausible but attach
            # each point leg to the opposite physical endpoint.
            old_out = peram_vsp[field_time, 2]
            old_incoming = peram_psv[2, bar_time]
            old_term_value = np.einsum(
                "BAOna,DCodP,ADnaod,CBPO->",
                old_out, old_incoming, current_factor, normal_factor, optimize=False,
            )
            old_field_bar_mapping += term.coefficient * old_term_value
        for sparse_value, _ in temporal_values.values():
            np.testing.assert_allclose(sparse_value, expected, rtol=2e-11, atol=3e-9)
        self.assertGreater(abs(sparse_value - old_field_bar_mapping), 1e-6)


if __name__ == "__main__":
    unittest.main()
