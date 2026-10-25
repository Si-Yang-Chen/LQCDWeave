"""CPU graph-level checks for the first native baryon Diagram binding."""

from itertools import permutations, product
import unittest
from unittest.mock import patch

import numpy as np
import opt_einsum

from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction import (
    BaryonTerm,
    BaryonVertex,
    execute_baryon_terms,
)
from lqcdweave.contraction.expansion.api import expand_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.single_hadron.baryon import (
    BaryonComponent,
    BaryonOperator,
    BaryonOperatorTerm,
    SparseSpinTensor,
)
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calc_diagram_eval


_PERMUTATIONS = tuple(permutations(range(3)))


def _epsilon_elemental(vectors):
    """Independent explicit color epsilon and spatial sum for zero momentum."""
    result = np.zeros(tuple(vector.shape[0] for vector in vectors), dtype=np.complex128)
    color_permutations = (
        ((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
        ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1),
    )
    for modes in product(*(range(vector.shape[0]) for vector in vectors)):
        for site in range(vectors[0].shape[1]):
            for colors, sign in color_permutations:
                result[modes] += sign * (
                    vectors[0][modes[0], site, colors[0]]
                    * vectors[1][modes[1], site, colors[1]]
                    * vectors[2][modes[2], site, colors[2]]
                )
    return result


def _spin_entries(offset):
    return SparseSpinTensor((
        (0, 1, 2, 0.5 + 0.25j * offset),
        (3, 2, 1, -0.25 + 0.125j * offset),
    ))


def _diagram(source_slots):
    zero = [[0, 0, 0] for _ in range(3)]
    edge = [[0, 0, 0] for _ in range(3)]
    for sink_slot, source_slot in enumerate(source_slots):
        edge[source_slot][sink_slot] = sink_slot + 1
    adjacency = [[zero, edge], [zero, zero]]
    return Diagram(QuarkDiagram(adjacency), [0, 1], ["source", "sink"],
                   [None, "p0", "p1", "p2"])


def _case(source_slots=(0, 1, 2)):
    rng = np.random.default_rng(413 + sum((index + 1) * value
                                          for index, value in enumerate(source_slots)))
    component = BaryonComponent.paths(flavors=("u", "d", "u"))
    source_shape, sink_shape = (2, 3, 1), (3, 1, 2)
    spatial = []
    for offset, mode_shape in ((1, source_shape), (4, sink_shape)):
        vectors = [
            rng.normal(size=(count, 3, 3))
            + 1j * (offset + q + rng.normal(size=(count, 3, 3)))
            for q, count in enumerate(mode_shape)
        ]
        spatial.append(_epsilon_elemental(vectors))

    source_coefficient = 0.75 - 0.4j
    sink_coefficient = -0.3 + 0.8j
    source_term = BaryonOperatorTerm(
        _spin_entries(2), component, source_coefficient, component.flavors, 0,
    )
    sink_term = BaryonOperatorTerm(
        _spin_entries(5), component, sink_coefficient, component.flavors, 0,
    )
    source_operator = BaryonOperator.from_terms("source", (source_term,))
    sink_operator = BaryonOperator.from_terms("sink", (sink_term,))
    source_reader = ArrayBaryonElemental(
        {"vvv": np.repeat(spatial[0][None, None, None, ...], 2, axis=2)},
        (component,), cfg="cfg-a",
    )
    sink_reader = ArrayBaryonElemental(
        {"vvv": np.repeat(spatial[1][None, None, None, ...], 2, axis=2)},
        (component,), cfg="cfg-a",
    )
    source = Baryon(source_reader, source_operator, source=True)
    sink = Baryon(sink_reader, sink_operator)

    propagators = {}
    arrays = []
    for q in range(3):
        source_axis = (2, 1, 0)[source_slots[q]]
        used_modes = (sink_shape[q], source_shape[source_axis])
        data = rng.normal(size=(2, 2, 4, 4, 3, 3)) + 1j * rng.normal(size=(2, 2, 4, 4, 3, 3))
        arrays.append(data[0, 1, :, :, :used_modes[0], :used_modes[1]])
        propagators[f"p{q}"] = ArrayPropagator(
            {"vsv": data}, time_convention="absolute", cfg="cfg-a",
            data_identity=f"prop-{q}",
        )

    return {
        "diagram": _diagram(source_slots),
        "vertices": {"source": source, "sink": sink},
        "propagators": propagators,
        "arrays": tuple(arrays),
        "source_spatial": spatial[0],
        "sink_spatial": spatial[1],
        "source_spin": source_term.spin.dense(),
        "sink_spin": sink_term.spin.dense(),
        "source_coefficient": source_coefficient,
        "sink_coefficient": sink_coefficient,
        "source_slots": source_slots,
        "source_shape": source_shape,
        "sink_shape": sink_shape,
    }


def _mixed_case(source_slots, *, include_normal=False):
    rng = np.random.default_rng(941 + sum((index + 2) * slot
                                          for index, slot in enumerate(source_slots)))
    component = BaryonComponent.paths(flavors=("u", "d", "u"))
    vectors = [
        rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
        for _ in range(3)
    ]
    source_spatial = _epsilon_elemental(vectors)
    vectors = [
        rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
        for _ in range(3)
    ]
    sink_spatial = _epsilon_elemental(vectors)
    source_coefficient, sink_coefficient = 0.75 - 0.4j, -0.3 + 0.8j
    source_term = BaryonOperatorTerm(
        _spin_entries(3), component, source_coefficient, component.flavors, 0,
    )
    sink_term = BaryonOperatorTerm(
        _spin_entries(6), component, sink_coefficient, component.flavors, 0,
    )
    source_reader = ArrayBaryonElemental(
        {"vvv": np.repeat(source_spatial[None, None, None, ...], 5, axis=2)},
        (component,), cfg="cfg-mixed", data_identity="cfg-mixed/source-baryon",
    )
    sink_reader = ArrayBaryonElemental(
        {"vvv": np.repeat(sink_spatial[None, None, None, ...], 5, axis=2)},
        (component,), cfg="cfg-mixed", data_identity="cfg-mixed/sink-baryon",
    )
    source = Baryon(
        source_reader, BaryonOperator.from_terms("source", (source_term,)), source=True,
    )
    sink = Baryon(sink_reader, BaryonOperator.from_terms("sink", (sink_term,)))

    current_operator = ConservedVectorInsertion(direction=3, wilson_r=1.25).operator()
    current_components = tuple(dict.fromkeys(term.component for term in current_operator.terms))
    current_data = rng.normal(size=(len(current_components), 1, 5, 2, 2))
    current_data = current_data + 1j * rng.normal(size=current_data.shape)
    current_reader = ArrayElemental(
        {"v2v": current_data.astype(np.complex128)},
        current_components,
        phase_convention="anchor",
        cfg="cfg-mixed",
        data_identity="cfg-mixed/current",
    )
    current = Meson(current_reader, current_operator, phase_convention="anchor")

    normal = None
    if include_normal:
        normal_component = ElementalComponent(0)
        normal_term = OperatorTerm(
            5, normal_component, 0, 0.6 + 0.2j,
        )
        normal_operator = Operator.from_terms("normal", (normal_term,))
        normal_data = rng.normal(size=(1, 1, 5, 2, 2))
        normal_data = normal_data + 1j * rng.normal(size=normal_data.shape)
        normal_reader = ArrayElemental(
            {"v2v": normal_data.astype(np.complex128)},
            (normal_component,),
            phase_convention="anchor",
            cfg="cfg-mixed",
            data_identity="cfg-mixed/normal-meson",
        )
        normal = Meson(normal_reader, normal_operator, phase_convention="anchor")

    propagators = {}
    arrays = {}
    labels = (("in", "mid", "direct0", "direct1", "out") if include_normal
              else ("in", "direct0", "direct1", "out"))
    for label in labels:
        data = rng.normal(size=(5, 5, 4, 4, 2, 2))
        data = data + 1j * rng.normal(size=data.shape)
        data = data.astype(np.complex128)
        arrays[label] = data
        propagators[f"p-{label}"] = ArrayPropagator(
            {"vsv": data},
            time_convention="absolute",
            cfg="cfg-mixed",
            data_identity=f"cfg-mixed/{label}",
        )

    # Baryon source slots feed two direct sink legs and one temporal current
    # chain; an optional ordinary Meson tests more than one intermediate.
    source_to_intermediate = [[0], [0], [0]]
    source_to_intermediate[source_slots[2]][0] = 1
    source_to_sink = [[0, 0, 0] for _ in range(3)]
    source_to_sink[source_slots[0]][0] = 3 if include_normal else 2
    source_to_sink[source_slots[1]][1] = 4 if include_normal else 3
    zero3 = [[0, 0, 0] for _ in range(3)]
    if include_normal:
        adjacency = [
            [zero3, source_to_intermediate, 0, source_to_sink],
            [0, 0, 2, 0],
            [0, 0, 0, [[0, 0, 5]]],
            [0, 0, 0, zero3],
        ]
        vertices = {"source": source, "normal": normal, "current": current, "sink": sink}
        times = [0, 1, 2, 4]
        vertex_keys = ["source", "normal", "current", "sink"]
        propagator_list = [None, "p-in", "p-mid", "p-direct0", "p-direct1", "p-out"]
        adjacency[2][3] = [[0, 0, 5]]
    else:
        adjacency = [
            [zero3, source_to_intermediate, source_to_sink],
            [0, 0, [[0, 0, 4]]],
            [0, 0, zero3],
        ]
        vertices = {"source": source, "current": current, "sink": sink}
        times = [0, 1, 4]
        vertex_keys = ["source", "current", "sink"]
        propagator_list = [None, "p-in", "p-direct0", "p-direct1", "p-out"]
    diagram = Diagram(
        QuarkDiagram(adjacency),
        times,
        vertex_keys,
        propagator_list,
    )
    return {
        "diagram": diagram,
        "vertices": vertices,
        "normal": normal,
        "propagators": propagators,
        "arrays": arrays,
        "source_spatial": source_spatial,
        "sink_spatial": sink_spatial,
        "source_spin": source_term.spin.dense(),
        "sink_spin": sink_term.spin.dense(),
        "source_coefficient": source_coefficient,
        "sink_coefficient": sink_coefficient,
        "source_slots": source_slots,
    }


def _independent_oracle(case):
    """Scalar loop reference, independent of Diagram lowering/einsum labels."""
    source_g4 = gamma(8)
    source_spin = np.einsum(
        "abc,ia,jb,kc->ijk", case["source_spin"].conj(),
        source_g4, source_g4, source_g4,
    )
    sink_spin = case["sink_spin"]
    source_mode = case["source_spatial"].conj()
    sink_mode = case["sink_spatial"]
    source_field_order = (2, 1, 0)
    coefficient = case["source_coefficient"].conjugate() * case["sink_coefficient"]
    total = 0j

    source_spin_indices = [index for index in np.ndindex(source_spin.shape)
                            if source_spin[index] != 0]
    sink_spin_indices = [index for index in np.ndindex(sink_spin.shape)
                         if sink_spin[index] != 0]
    source_modes = [index for index in np.ndindex(source_mode.shape)
                    if source_mode[index] != 0]
    sink_modes = [index for index in np.ndindex(sink_mode.shape)
                  if sink_mode[index] != 0]
    for source_spin_index in source_spin_indices:
        for source_mode_index in source_modes:
            source_value = source_spin[source_spin_index] * source_mode[source_mode_index]
            for sink_spin_index in sink_spin_indices:
                for sink_mode_index in sink_modes:
                    value = source_value * sink_spin[sink_spin_index] * sink_mode[sink_mode_index]
                    for sink_slot, source_slot in enumerate(case["source_slots"]):
                        source_axis = source_field_order[source_slot]
                        value *= case["arrays"][sink_slot][
                            sink_spin_index[sink_slot],
                            source_spin_index[source_axis],
                            sink_mode_index[sink_slot],
                            source_mode_index[source_axis],
                        ]
                    total += value
    return coefficient * total


def _hand_current_factor(reader, description):
    term = description.operator_term
    spatial = reader.block(
        term.component, term.momentum, description.anchor_time % reader.Lt,
        "v2v", usedNe=2,
    )
    spin = gamma(term.gamma)
    result = np.empty((4, 4, 2, 2), dtype=np.complex128)
    for field_spin, bar_spin, field_mode, bar_mode in product(range(4), range(4), range(2), range(2)):
        result[field_spin, bar_spin, field_mode, bar_mode] = (
            spin[field_spin, bar_spin] * spatial[field_mode, bar_mode]
        )
    return result


def _hand_current_chain(propagator_in, current_factor, propagator_out):
    """Manual matrix-chain sum: sink <- current <- source, one line only."""
    result = np.zeros((4, 4, 2, 2), dtype=np.complex128)
    for sink_spin, source_spin, sink_mode, source_mode in product(
        range(4), range(4), range(2), range(2),
    ):
        total = 0j
        for field_spin, bar_spin, field_mode, bar_mode in product(
            range(4), range(4), range(2), range(2),
        ):
            total += (
                propagator_out[sink_spin, field_spin, sink_mode, field_mode]
                * current_factor[field_spin, bar_spin, field_mode, bar_mode]
                * propagator_in[bar_spin, source_spin, bar_mode, source_mode]
            )
        result[sink_spin, source_spin, sink_mode, source_mode] = total
    return result


class BaryonDiagramFrontendTests(unittest.TestCase):
    def test_operator_sum_stays_inside_binding_core_and_replays(self):
        case = _case()
        sink = case["vertices"]["sink"]
        first = sink.operator.terms[0]
        extra = OperatorTerm.from_baryon(first.component, _spin_entries(8), momentum=first.momentum,
                                       coefficient=-.2 + .3j)
        sink.operator = Operator.from_terms("complex-sink", (first, extra))
        extra_case = dict(case, sink_spin=extra.spin.dense(), sink_coefficient=extra.coefficient)
        expected = _independent_oracle(case) + _independent_oracle(extra_case)
        expanded = expand_correlator(case["diagram"], adapter=BaryonDiagramBinding(sector_scope="dense-v", vertices="joined"),
            bindings=DiagramBindings(case["propagators"], case["vertices"], "cfg-a"),
            time_axes=None, time_pairs=None, batch="auto")
        self.assertEqual(len(expanded.terms), 1)
        self.assertEqual(tuple(len(j.core.components) for j in expanded.problem.baryon_joins), (1, 2))
        for mode in ("none", "anytime"):
            with self.subTest(mode=mode):
                result = calc_diagram_eval(case["diagram"], propagator_map=case["propagators"],
                    vertex_map=case["vertices"], configuration="cfg-a", optimize=mode,
                    diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v", vertices="joined"))
                np.testing.assert_allclose(result.value, expected, rtol=2e-12, atol=1e-10)

    def test_fixed_baryon_diagram_matches_independent_source_dagger_oracle(self):
        for source_slots in _PERMUTATIONS:
            with self.subTest(source_slots=source_slots):
                case = _case(source_slots)
                expected = _independent_oracle(case)
                result = calc_diagram_eval(
                    case["diagram"],
                    propagator_map=case["propagators"],
                    vertex_map=case["vertices"],
                    configuration="cfg-a",
                    optimize="none",
                    diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                )
                np.testing.assert_allclose(result.value, expected, rtol=2e-12, atol=1e-10)

    def test_description_reads_headers_only_and_none_never_calls_path_search(self):
        case = _case((2, 0, 1))
        bindings = DiagramBindings(case["propagators"], case["vertices"], "cfg-a")
        with (
            patch.object(case["vertices"]["source"].reader, "block",
                         side_effect=AssertionError("metadata description read elemental data")),
            patch.object(case["vertices"]["sink"].reader, "block",
                         side_effect=AssertionError("metadata description read elemental data")),
            patch.object(case["propagators"]["p0"], "block",
                         side_effect=AssertionError("metadata description read a propagator")),
        ):
            expanded = expand_correlator(
                case["diagram"], adapter=BaryonDiagramBinding(sector_scope="dense-v", vertices="joined"), bindings=bindings,
                time_axes=None, time_pairs=None, batch="auto",
            )
        self.assertEqual(len(expanded.terms), 1)
        self.assertEqual(len(expanded.terms[0].inputs), 3)
        self.assertEqual(len(expanded.problem.baryon_joins), 2)
        self.assertTrue(all(join.sparse_spin is not None for join in expanded.problem.baryon_joins))

        with patch.object(opt_einsum, "contract_path",
                          side_effect=AssertionError("optimize='none' searched a path")):
            result = calc_diagram_eval(
                case["diagram"],
                propagator_map=case["propagators"],
                vertex_map=case["vertices"],
                configuration="cfg-a",
                optimize="none",
                diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v", vertices="joined"),
            )
        np.testing.assert_allclose(result.value, _independent_oracle(case), rtol=2e-12, atol=1e-10)

    def test_repeated_diagram_preserves_nested_output_and_reuses_bound_values(self):
        case = _case((1, 2, 0))
        counts = {"elemental": 0, "propagator": 0}
        for vertex in case["vertices"].values():
            original = vertex.reader.block

            def count_elemental(*args, _original=original, **kwargs):
                counts["elemental"] += 1
                return _original(*args, **kwargs)

            vertex.reader.block = count_elemental
        for reader in case["propagators"].values():
            original = reader.block

            def count_propagator(*args, _original=original, **kwargs):
                counts["propagator"] += 1
                return _original(*args, **kwargs)

            reader.block = count_propagator

        result = calc_diagram_eval(
            {"one": case["diagram"], "again": [case["diagram"]]},
            propagator_map=case["propagators"],
            vertex_map=case["vertices"],
            configuration="cfg-a",
            optimize="none",
            diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
        )
        expected = _independent_oracle(case)
        self.assertIsInstance(result.value, dict)
        self.assertIsInstance(result.value["again"], list)
        np.testing.assert_allclose(result.value["one"], expected, rtol=2e-12, atol=1e-10)
        np.testing.assert_allclose(result.value["again"][0], expected, rtol=2e-12, atol=1e-10)
        self.assertEqual(counts, {"elemental": 2, "propagator": 3})

    def test_mixed_graph_backends_fail_during_describe_before_any_block_read(self):
        case = _case((0, 1, 2))

        class CupyMarker:
            __name__ = "cupy"

        source_cupy = Baryon(
            case["vertices"]["source"].reader,
            case["vertices"]["source"].operator,
            source=True,
            array_module=CupyMarker(),
        )
        sink_cupy = Baryon(
            case["vertices"]["sink"].reader,
            case["vertices"]["sink"].operator,
            array_module=CupyMarker(),
        )
        second = Diagram(
            case["diagram"].diagram,
            case["diagram"].time_list,
            ["source-cupy", "sink-cupy"],
            case["diagram"].propagator_list,
        )
        vertices = {
            **case["vertices"],
            "source-cupy": source_cupy,
            "sink-cupy": sink_cupy,
        }
        with (
            patch.object(case["vertices"]["source"].reader, "block",
                         side_effect=AssertionError("mixed backend reached elemental read")) as source_read,
            patch.object(case["propagators"]["p0"], "block",
                         side_effect=AssertionError("mixed backend reached propagator read")) as prop_read,
        ):
            with self.assertRaisesRegex(ValueError, "same array backend"):
                expand_correlator(
                    (case["diagram"], second),
                    adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                    bindings=DiagramBindings(case["propagators"], vertices, "cfg-a"),
                    time_axes=None,
                    time_pairs=None,
                    batch="auto",
                )
        source_read.assert_not_called()
        prop_read.assert_not_called()

    def test_provider_keyerror_is_not_relabelled_as_missing_binding(self):
        case = _case((1, 2, 0))
        with patch.object(
            case["vertices"]["source"].reader,
            "block",
            side_effect=KeyError("underlying elemental section missing"),
        ):
            with self.assertRaisesRegex(KeyError, "underlying elemental section missing"):
                calc_diagram_eval(
                    case["diagram"],
                    propagator_map=case["propagators"],
                    vertex_map=case["vertices"],
                    configuration="cfg-a",
                    optimize="none",
                    diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                )

    def test_runtime_block_backend_must_match_described_backend(self):
        case = _case((0, 2, 1))

        class CupyBlockMarker:
            __module__ = "cupy.fake"
            shape = case["source_spatial"].shape
            dtype = case["source_spatial"].dtype

        with patch.object(case["vertices"]["source"].reader, "block",
                          return_value=CupyBlockMarker()):
            with self.assertRaisesRegex(TypeError, "block backend"):
                calc_diagram_eval(
                    case["diagram"],
                    propagator_map=case["propagators"],
                    vertex_map=case["vertices"],
                    configuration="cfg-a",
                    optimize="none",
                    diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                )

    def test_baryon_current_baryon_temporal_chain_matches_hand_reduced_kernel(self):
        for source_slots in _PERMUTATIONS:
            with self.subTest(source_slots=source_slots):
                case = _mixed_case(source_slots)
                current = case["vertices"]["current"]
                descriptors = current.describe_factors(1, "v2v", usedNe=2)
                self.assertEqual(len(descriptors), 4)
                self.assertEqual(
                    {(item.bar_time, item.field_time) for item in descriptors},
                    {(1, 2), (2, 1)},
                )

                endpoint_calls = {"in": [], "out": []}
                for name in ("p-in", "p-out"):
                    reader = case["propagators"][name]
                    original = reader.describe_block

                    def record(sink_time, source_time, sector="vsv", *,
                               _original=original, _name=name, **kwargs):
                        endpoint_calls["in" if _name == "p-in" else "out"].append(
                            (sink_time, source_time, sector),
                        )
                        return _original(sink_time, source_time, sector, **kwargs)

                    reader.describe_block = record
                expanded = expand_correlator(
                    case["diagram"],
                    adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                    bindings=DiagramBindings(
                        case["propagators"], case["vertices"], "cfg-mixed", usedNe=2,
                    ),
                    time_axes=None,
                    time_pairs=None,
                    batch="auto",
                )
                self.assertEqual(len(expanded.terms), 2)
                self.assertEqual(
                    {item[:2] for item in endpoint_calls["in"]}, {(1, 0), (2, 0)},
                )
                self.assertEqual(
                    {item[:2] for item in endpoint_calls["out"]}, {(4, 2), (4, 1)},
                )

                effective = np.zeros((4, 4, 2, 2), dtype=np.complex128)
                current_reader = current.reader
                for descriptor in descriptors:
                    propagator_in = case["propagators"]["p-in"].block(
                        descriptor.bar_time, 0, "vsv", usedNe=2,
                    )
                    propagator_out = case["propagators"]["p-out"].block(
                        4, descriptor.field_time, "vsv", usedNe=2,
                    )
                    effective += (
                        descriptor.operator_term.coefficient
                        * _hand_current_chain(
                            propagator_in,
                            _hand_current_factor(current_reader, descriptor),
                            propagator_out,
                        )
                    )

                direct0 = case["propagators"]["p-direct0"].block(4, 0, "vsv", usedNe=2)
                direct1 = case["propagators"]["p-direct1"].block(4, 0, "vsv", usedNe=2)
                oracle_case = {
                    "source_spin": case["source_spin"],
                    "sink_spin": case["sink_spin"],
                    "source_spatial": case["source_spatial"],
                    "sink_spatial": case["sink_spatial"],
                    "source_coefficient": case["source_coefficient"],
                    "sink_coefficient": case["sink_coefficient"],
                    "source_slots": source_slots,
                    "arrays": (direct0, direct1, effective),
                }
                expected = _independent_oracle(oracle_case)

                source_spin = np.einsum(
                    "abc,ia,jb,kc->ijk", case["source_spin"].conj(),
                    gamma(8), gamma(8), gamma(8),
                )
                mapped_source_slots = tuple((2, 1, 0)[slot] for slot in source_slots)
                baryon_result = execute_baryon_terms(
                    {
                        "S": source_spin,
                        "E": case["source_spatial"].conj(),
                        "R": case["sink_spin"],
                        "F": case["sink_spatial"],
                        "P0": direct0,
                        "P1": direct1,
                        "P2": effective,
                    },
                    (BaryonTerm(
                        BaryonVertex("S", "E"), BaryonVertex("R", "F"),
                        ("P0", "P1", "P2"), mapped_source_slots,
                        case["source_coefficient"].conjugate() * case["sink_coefficient"],
                    ),),
                )
                np.testing.assert_allclose(baryon_result.values[0], expected,
                                           rtol=2e-12, atol=1e-10)

                with patch.object(opt_einsum, "contract_path",
                                  side_effect=AssertionError("dense-V none path searched")):
                    result = calc_diagram_eval(
                        case["diagram"],
                        propagator_map=case["propagators"],
                        vertex_map=case["vertices"],
                        configuration="cfg-mixed",
                        usedNe=2,
                        optimize="none",
                        diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
                    )
                np.testing.assert_allclose(result.value, expected,
                                           rtol=2e-12, atol=1e-10)

    def test_multiple_intermediate_meson_current_vertices_use_each_real_endpoint(self):
        source_slots = (2, 0, 1)
        case = _mixed_case(source_slots, include_normal=True)
        normal = case["normal"]
        current = case["vertices"]["current"]
        normal_description = normal.describe_factors(1, "v2v", usedNe=2)[0]
        current_descriptions = current.describe_factors(2, "v2v", usedNe=2)
        self.assertEqual(
            {(item.bar_time, item.field_time) for item in current_descriptions},
            {(2, 3), (3, 2)},
        )

        endpoint_calls = {name: [] for name in ("p-in", "p-mid", "p-out")}
        for name in endpoint_calls:
            reader = case["propagators"][name]
            original = reader.describe_block

            def record(sink_time, source_time, sector="vsv", *,
                       _original=original, _name=name, **kwargs):
                endpoint_calls[_name].append((sink_time, source_time, sector))
                return _original(sink_time, source_time, sector, **kwargs)

            reader.describe_block = record

        expanded = expand_correlator(
            case["diagram"],
            adapter=BaryonDiagramBinding(sector_scope="dense-v"),
            bindings=DiagramBindings(
                case["propagators"], case["vertices"], "cfg-mixed", usedNe=2,
            ),
            time_axes=None,
            time_pairs=None,
            batch="auto",
        )
        self.assertEqual(len(expanded.terms), 2)
        self.assertEqual({item[:2] for item in endpoint_calls["p-in"]}, {(1, 0)})
        self.assertEqual({item[:2] for item in endpoint_calls["p-mid"]}, {(2, 1), (3, 1)})
        self.assertEqual({item[:2] for item in endpoint_calls["p-out"]}, {(4, 2), (4, 3)})

        normal_factor = _hand_current_factor(normal.reader, normal_description)
        effective = np.zeros((4, 4, 2, 2), dtype=np.complex128)
        for descriptor in current_descriptions:
            propagator_in = case["propagators"]["p-in"].block(1, 0, "vsv", usedNe=2)
            propagator_mid = case["propagators"]["p-mid"].block(
                descriptor.bar_time, 1, "vsv", usedNe=2,
            )
            propagator_out = case["propagators"]["p-out"].block(
                4, descriptor.field_time, "vsv", usedNe=2,
            )
            after_normal = _hand_current_chain(
                propagator_in, normal_factor, propagator_mid,
            )
            after_current = _hand_current_chain(
                after_normal, _hand_current_factor(current.reader, descriptor),
                propagator_out,
            )
            effective += (
                normal_description.operator_term.coefficient
                * descriptor.operator_term.coefficient
                * after_current
            )

        oracle_case = {
            "source_spin": case["source_spin"],
            "sink_spin": case["sink_spin"],
            "source_spatial": case["source_spatial"],
            "sink_spatial": case["sink_spatial"],
            "source_coefficient": case["source_coefficient"],
            "sink_coefficient": case["sink_coefficient"],
            "source_slots": source_slots,
            "arrays": (
                case["propagators"]["p-direct0"].block(4, 0, "vsv", usedNe=2),
                case["propagators"]["p-direct1"].block(4, 0, "vsv", usedNe=2),
                effective,
            ),
        }
        expected = _independent_oracle(oracle_case)
        with patch.object(opt_einsum, "contract_path",
                          side_effect=AssertionError("multi-meson none path searched")):
            result = calc_diagram_eval(
                case["diagram"],
                propagator_map=case["propagators"],
                vertex_map=case["vertices"],
                configuration="cfg-mixed",
                usedNe=2,
                optimize="none",
                diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
            )
        np.testing.assert_allclose(result.value, expected, rtol=3e-12, atol=1e-10)
