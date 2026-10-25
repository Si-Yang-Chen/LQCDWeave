"""Mode-block tests for dense baryon Diagram contractions."""

from itertools import combinations_with_replacement, permutations, product
import unittest
from unittest.mock import patch

import numpy as np
import opt_einsum
import sympy

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
from lqcdweave.operators.correlator.api import (
    Diagram,
    Flavor,
    Hadron,
    HadronIrrep,
    QuarkDiagram,
    Tag,
    gen_correlator,
    project_baryon_mode_triples,
    split_baryon_eigenmodes,
)
from lqcdweave.operators.single_hadron.baryon import (
    BaryonComponent,
    BaryonOperator,
    BaryonOperatorTerm,
    SparseSpinTensor,
)
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.correlator import calculate


_FIELD_ORDER = (2, 1, 0)
_PARTITIONS = (0, 1, 3)
_INTERVALS = ((0,), (1, 2))
_SOURCE_SLOTS = tuple(permutations(range(3)))


def _selections_for_block_index(block_index):
    """Split output axes are ordered by time: sink@0, then source@1."""
    sink = tuple(_INTERVALS[index] for index in block_index[:3])
    source = tuple(_INTERVALS[index] for index in block_index[3:])
    return source, sink


def _epsilon_elemental(vectors):
    """Build a VVV tensor from an explicit color epsilon contraction."""
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


def _spin_tensor(offset):
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
    # Vertex 0 is the source at t=1; vertex 1 is the sink at t=0.
    return Diagram(QuarkDiagram(adjacency), [1, 0], ["source", "sink"],
                   [None, "p0", "p1", "p2"])


def _case(source_slots=(0, 1, 2), *, mode_count=3, seed=413):
    rng = np.random.default_rng(seed + sum((i + 1) * slot
                                           for i, slot in enumerate(source_slots)))
    component = BaryonComponent.paths(flavors=("u", "d", "u"))
    spatial = []
    for offset in (1, 4):
        vectors = [
            rng.normal(size=(mode_count, 3, 3))
            + 1j * (offset + q + rng.normal(size=(mode_count, 3, 3)))
            for q in range(3)
        ]
        spatial.append(_epsilon_elemental(vectors))

    source_coefficient = 0.75 - 0.4j
    sink_coefficient = -0.3 + 0.8j
    source_term = BaryonOperatorTerm(
        _spin_tensor(2), component, source_coefficient, component.flavors, 0,
    )
    sink_term = BaryonOperatorTerm(
        _spin_tensor(5), component, sink_coefficient, component.flavors, 0,
    )
    source_data = np.zeros((1, 1, 2, mode_count, mode_count, mode_count),
                           dtype=np.complex128)
    sink_data = np.zeros_like(source_data)
    source_data[0, 0, 1] = spatial[0]
    sink_data[0, 0, 0] = spatial[1]
    source_reader = ArrayBaryonElemental(
        {"vvv": source_data}, (component,), cfg="cfg-a",
        data_identity="cfg-a/source-vvv",
    )
    sink_reader = ArrayBaryonElemental(
        {"vvv": sink_data}, (component,), cfg="cfg-a",
        data_identity="cfg-a/sink-vvv",
    )
    source = Baryon(
        source_reader, BaryonOperator.from_terms("source", (source_term,)), source=True,
    )
    sink = Baryon(sink_reader, BaryonOperator.from_terms("sink", (sink_term,)))

    propagators = {}
    propagator_arrays = []
    for q in range(3):
        data = (rng.normal(size=(2, 2, 4, 4, mode_count, mode_count))
                + 1j * rng.normal(size=(2, 2, 4, 4, mode_count, mode_count)))
        propagator_arrays.append(data[1, 0])
        propagators[f"p{q}"] = ArrayPropagator(
            {"vsv": data}, time_convention="absolute", cfg="cfg-a",
            data_identity=f"cfg-a/prop-{q}",
        )

    return {
        "diagram": _diagram(source_slots),
        "vertices": {"source": source, "sink": sink},
        "source_reader": source_reader,
        "sink_reader": sink_reader,
        "propagators": propagators,
        "propagator_arrays": tuple(propagator_arrays),
        "source_spatial": spatial[0],
        "sink_spatial": spatial[1],
        "source_spin": source_term.spin.dense(),
        "sink_spin": sink_term.spin.dense(),
        "source_coefficient": source_coefficient,
        "sink_coefficient": sink_coefficient,
        "source_slots": tuple(source_slots),
        "mode_count": mode_count,
    }


def _independent_einsum(case, source_selection=None, sink_selection=None):
    """Direct scalar formula with explicit spin, mode, and edge labels."""
    count = case["mode_count"]
    source_selection = tuple(
        tuple(range(count)) for _ in range(3)
    ) if source_selection is None else tuple(tuple(s) for s in source_selection)
    sink_selection = tuple(
        tuple(range(count)) for _ in range(3)
    ) if sink_selection is None else tuple(tuple(s) for s in sink_selection)

    source_spin = np.einsum(
        "abc,ia,jb,kc->ijk", case["source_spin"].conj(),
        gamma(8), gamma(8), gamma(8), optimize=False,
    )
    source_elemental = case["source_spatial"][np.ix_(*source_selection)].conj()
    sink_spin = case["sink_spin"]
    sink_elemental = case["sink_spatial"][np.ix_(*sink_selection)]

    subscripts = ["abc", "ijk", "xyz", "uvw"]
    operands = [source_spin, source_elemental, sink_spin, sink_elemental]
    for sink_slot, source_slot in enumerate(case["source_slots"]):
        source_axis = _FIELD_ORDER[source_slot]
        propagator = case["propagator_arrays"][sink_slot]
        propagator = np.take(propagator, sink_selection[sink_slot], axis=2)
        propagator = np.take(propagator, source_selection[source_axis], axis=3)
        subscripts.append(
            "xyz"[sink_slot] + "abc"[source_axis]
            + "uvw"[sink_slot] + "ijk"[source_axis]
        )
        operands.append(propagator)
    value = opt_einsum.contract(
        ",".join(subscripts) + "->", *operands, optimize="greedy",
    )
    return (case["source_coefficient"].conjugate()
            * case["sink_coefficient"] * value)


def _calculate(case, correlator, *, usedNe=3, optimize="none", execute=True):
    return calculate(
        correlator,
        propagator_map=case["propagators"],
        vertex_map=case["vertices"],
        configuration="cfg-a",
        usedNe=usedNe,
        optimize=optimize,
        time_axes=None,
        execute=execute,
        diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"),
    )


class BaryonModeBlockTests(unittest.TestCase):
    def test_all_64_ordered_blocks_match_independent_einsum_and_restore(self):
        case = _case()
        blocks = split_baryon_eigenmodes(case["diagram"], partitions=_PARTITIONS)
        self.assertEqual(blocks.shape, (2, 2, 2, 2, 2, 2))
        whole = _calculate(case, case["diagram"]).value
        np.testing.assert_allclose(
            whole, _independent_einsum(case), rtol=3e-12, atol=1e-10,
        )

        values = np.empty(blocks.shape, dtype=np.complex128)
        for block_index in np.ndindex(blocks.shape):
            source_selection, sink_selection = _selections_for_block_index(block_index)
            block = blocks[block_index]
            self.assertEqual(block.mode_selections, (source_selection, sink_selection))
            result = _calculate(case, block).value
            expected = _independent_einsum(case, source_selection, sink_selection)
            np.testing.assert_allclose(result, expected, rtol=3e-12, atol=1e-10)
            values[block_index] = result

        np.testing.assert_allclose(values.sum(), whole, rtol=3e-12, atol=1e-10)
        projected = project_baryon_mode_triples(values, mode_axis=0, vertices=2)
        self.assertEqual(projected.shape, (4, 4))
        multiplicities = tuple(
            len(set(permutations(triple)))
            for triple in combinations_with_replacement(range(2), 3)
        )
        restored = sum(
            projected[i, j] * np.sqrt(multiplicities[i] * multiplicities[j])
            for i in range(4) for j in range(4)
        )
        np.testing.assert_allclose(restored, whole, rtol=3e-12, atol=1e-10)

    def test_six_source_slot_permutations_and_an_exchanged_block(self):
        exchanged_block_checked = False
        for source_slots in _SOURCE_SLOTS:
            with self.subTest(source_slots=source_slots):
                case = _case(source_slots, seed=733)
                whole = _calculate(case, case["diagram"]).value
                np.testing.assert_allclose(
                    whole, _independent_einsum(case), rtol=3e-12, atol=1e-10,
                )
                blocks = split_baryon_eigenmodes(
                    case["diagram"], partitions=_PARTITIONS,
                )
                # The direct reference covers every block for each edge ordering;
                # the executable blocked path is also checked for exchanged edges.
                oracle_sum = 0j
                for block_index in np.ndindex(blocks.shape):
                    source_selection, sink_selection = _selections_for_block_index(block_index)
                    oracle_sum += _independent_einsum(
                        case, source_selection, sink_selection,
                    )
                np.testing.assert_allclose(oracle_sum, whole, rtol=3e-12, atol=1e-10)
                if source_slots != (0, 1, 2) and not exchanged_block_checked:
                    block_index = (1, 0, 1, 0, 1, 1)
                    source_selection, sink_selection = _selections_for_block_index(block_index)
                    result = _calculate(case, blocks[block_index]).value
                    np.testing.assert_allclose(
                        result,
                        _independent_einsum(case, source_selection, sink_selection),
                        rtol=3e-12,
                        atol=1e-10,
                    )
                    exchanged_block_checked = True
        self.assertTrue(exchanged_block_checked)

    def test_planning_is_data_free_and_block_reads_follow_selected_ranges(self):
        case = _case()
        blocks = split_baryon_eigenmodes(case["diagram"], partitions=_PARTITIONS)
        block_index = (1, 0, 1, 0, 1, 0)
        block = blocks[block_index]
        source_selection, sink_selection = _selections_for_block_index(block_index)

        with (
            patch.object(case["source_reader"], "block",
                         side_effect=AssertionError("description read source elemental")) as source_read,
            patch.object(case["sink_reader"], "block",
                         side_effect=AssertionError("description read sink elemental")) as sink_read,
            patch.object(case["propagators"]["p0"], "block",
                         side_effect=AssertionError("description read propagator")) as prop_read,
        ):
            calculation = _calculate(case, block, execute=False)
            source_read.assert_not_called()
            sink_read.assert_not_called()
            prop_read.assert_not_called()

        reads = {"source": [], "sink": [], "p0": [], "p1": [], "p2": []}
        for name, reader in (
            ("source", case["source_reader"]), ("sink", case["sink_reader"]),
            *((f"p{q}", case["propagators"][f"p{q}"]) for q in range(3)),
        ):
            original = reader.block

            def record(*args, _original=original, _name=name, **kwargs):
                value = _original(*args, **kwargs)
                reads[_name].append((args, kwargs, tuple(value.shape)))
                return value

            reader.block = record
        result = calculation.run()
        np.testing.assert_allclose(
            result.value,
            _independent_einsum(case, source_selection, sink_selection),
            rtol=3e-12,
            atol=1e-10,
        )

        self.assertEqual(len(reads["source"]), 1)
        self.assertEqual(len(reads["sink"]), 1)
        for q in range(3):
            self.assertEqual(len(reads[f"p{q}"]), 1)
        source_args, source_kwargs, source_shape = reads["source"][0]
        sink_args, sink_kwargs, sink_shape = reads["sink"][0]
        self.assertEqual(source_args[2:4], (1, "vvv"))
        self.assertEqual(sink_args[2:4], (0, "vvv"))
        self.assertEqual(source_kwargs["mode_indices"], {
            f"mode{axis}": source_selection[axis] for axis in range(3)
        })
        self.assertEqual(sink_kwargs["mode_indices"], {
            f"mode{axis}": sink_selection[axis] for axis in range(3)
        })
        self.assertEqual(source_shape, tuple(len(s) for s in source_selection))
        self.assertEqual(sink_shape, tuple(len(s) for s in sink_selection))
        for q in range(3):
            prop_args, prop_kwargs, prop_shape = reads[f"p{q}"][0]
            source_axis = _FIELD_ORDER[case["source_slots"][q]]
            self.assertEqual(prop_args[:3], (0, 1, "vsv"))
            self.assertEqual(prop_kwargs["mode_indices"], {
                "sink_mode": sink_selection[q],
                "source_mode": source_selection[source_axis],
            })
            self.assertEqual(prop_shape, (
                4, 4, len(sink_selection[q]), len(source_selection[source_axis]),
            ))

    def test_combined_64_blocks_share_each_selected_physical_reader_block(self):
        case = _case()
        blocks = split_baryon_eigenmodes(case["diagram"], partitions=_PARTITIONS)
        self.assertEqual(blocks.size, 64)
        combined = sympy.Add(*blocks.flat)

        reads = {"source": [], "sink": [], "p0": [], "p1": [], "p2": []}
        for name, reader in (
            ("source", case["source_reader"]), ("sink", case["sink_reader"]),
            *((f"p{q}", case["propagators"][f"p{q}"]) for q in range(3)),
        ):
            original = reader.block

            def record(*args, _original=original, _name=name, **kwargs):
                value = _original(*args, **kwargs)
                reads[_name].append((args, kwargs, tuple(value.shape)))
                return value

            reader.block = record

        result = _calculate(case, combined).value
        np.testing.assert_allclose(
            result, _independent_einsum(case), rtol=3e-12, atol=1e-10,
        )

        self.assertEqual(len(reads["source"]), 8)
        self.assertEqual(len(reads["sink"]), 8)
        self.assertEqual(sum(len(reads[f"p{q}"]) for q in range(3)), 12)
        for name in ("source", "sink"):
            selections = {
                tuple(record[1]["mode_indices"][f"mode{axis}"] for axis in range(3))
                for record in reads[name]
            }
            self.assertEqual(len(selections), 8)
        for q in range(3):
            selections = {
                (record[1]["mode_indices"]["sink_mode"],
                 record[1]["mode_indices"]["source_mode"])
                for record in reads[f"p{q}"]
            }
            self.assertEqual(len(reads[f"p{q}"]), 4)
            self.assertEqual(len(selections), 4)

    def test_nested_zero_arrays_and_four_partition_projection(self):
        case = _case()
        matrix = np.array([
            [case["diagram"], 0],
            [0, case["diagram"]],
        ], dtype=object)
        nested = split_baryon_eigenmodes(
            {"group": {"matrix": matrix, "leaves": [case["diagram"]]}},
            partitions=_PARTITIONS,
        )
        split_matrix = nested["group"]["matrix"]
        self.assertEqual(split_matrix.shape, (2, 2, 2, 2, 2, 2, 2, 2))
        self.assertTrue(all(value == 0 for value in split_matrix[0, 1].flat))
        self.assertTrue(all(value == 0 for value in split_matrix[1, 0].flat))
        self.assertEqual(nested["group"]["leaves"][0].shape, (2, 2, 2, 2, 2, 2))

        rng = np.random.default_rng(810)
        ordered = (rng.normal(size=(4,) * 6)
                   + 1j * rng.normal(size=(4,) * 6))
        merged = project_baryon_mode_triples(ordered, mode_axis=0, vertices=2)
        self.assertEqual(merged.shape, (20, 20))
        triples = tuple(combinations_with_replacement(range(4), 3))
        weights = tuple(np.sqrt(len(set(permutations(item)))) for item in triples)
        restored = sum(
            merged[i, j] * weights[i] * weights[j]
            for i in range(20) for j in range(20)
        )
        np.testing.assert_allclose(restored, ordered.sum(), rtol=3e-13, atol=1e-10)

        normalized = np.zeros((4, 4, 4), dtype=np.complex128)
        orbit = set(permutations((0, 0, 1)))
        for index in orbit:
            normalized[index] = 1.25 - 0.75j
        one_vertex = project_baryon_mode_triples(
            normalized, mode_axis=0, vertices=1,
        )
        target = triples.index((0, 0, 1))
        np.testing.assert_allclose(
            one_vertex[target], np.sqrt(3) * (1.25 - 0.75j),
        )

    def test_invalid_partitions_and_requested_ne_are_rejected(self):
        case = _case()
        for invalid in ((1, 3), (0, 0, 3), (0, 3, 3), (0, True, 3), (0, 1.5, 3)):
            with self.subTest(partitions=invalid), self.assertRaises((TypeError, ValueError)):
                split_baryon_eigenmodes(case["diagram"], partitions=invalid)

        blocks = split_baryon_eigenmodes(case["diagram"], partitions=_PARTITIONS)
        with self.assertRaisesRegex(ValueError, "endpoint must equal"):
            _calculate(case, blocks[(0, 0, 0, 0, 0, 0)], usedNe=2, execute=False)
        with self.assertRaisesRegex(ValueError, "endpoint must equal"):
            _calculate(case, blocks[(0, 0, 0, 0, 0, 0)], usedNe=4, execute=False)

        selected = case["vertices"]["sink"].with_mode_indices(((3,), (0,), (1,)))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            selected.describe_factors(0, usedNe=3)

        for invalid in ((), ((0,), ()), ((0, 0), (1,), (2,)), ((-1,), (1,), (2,))):
            with self.subTest(selections=invalid), self.assertRaises((TypeError, ValueError)):
                case["vertices"]["sink"].with_mode_indices(invalid)

    def test_native_one_channel_structured_correlator_is_rejected(self):
        row = HadronIrrep("N", (0, 0, 0), "G_1", 1, Tag(0, 0))[0]
        axis = [Hadron(row, Flavor("uds"))]
        structured = gen_correlator(
            [axis.copy(), axis.copy()], [0, 1], [False, True], structured=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "structured=False"):
            split_baryon_eigenmodes(structured, partitions=_PARTITIONS)

    def test_anytime_path_matches_fixed_path_for_two_mode_intervals(self):
        case = _case(mode_count=2, seed=992)
        fixed = _calculate(case, case["diagram"], usedNe=2, optimize="none").value
        try:
            anytime = _calculate(
                case, case["diagram"], usedNe=2, optimize="anytime",
            ).value
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"anytime path dependencies are unavailable: {exc}")
        np.testing.assert_allclose(anytime, fixed, rtol=3e-12, atol=1e-10)
        np.testing.assert_allclose(
            anytime, _independent_einsum(case), rtol=3e-12, atol=1e-10,
        )

    def test_with_mode_indices_get_factors_and_describe_read_share_selection(self):
        case = _case(seed=1201)
        template = case["vertices"]["sink"]
        original_full = template.get(0)
        selected_indices = ((2, 0), (1,), (2, 1))
        selected = template.with_mode_indices(selected_indices)

        self.assertIsNone(template.mode_indices)
        self.assertEqual(selected.mode_indices, selected_indices)
        factor = next(selected.factors(0))
        selected_value = selected.get(0)
        np.testing.assert_allclose(factor.dense(), selected_value)
        self.assertEqual(selected_value.shape, (4, 4, 4, 2, 1, 2))
        np.testing.assert_allclose(template.get(0), original_full)
        self.assertEqual(template.get(0).shape[-3:], (3, 3, 3))

        description = selected.describe_factors(0, usedNe=3)[0]
        self.assertEqual(description.mode_indices, selected_indices)
        self.assertEqual(description.tensor.shape, (2, 1, 2))
        raw = selected.read_factor_block(description)
        expected = case["sink_spatial"][np.ix_(*selected_indices)]
        np.testing.assert_allclose(raw, expected)
        np.testing.assert_allclose(
            selected.materialize_unweighted_elemental(description, raw), expected,
        )
        self.assertEqual(template.mode_indices, None)

        source_template = case["vertices"]["source"]
        source_selected = source_template.with_mode_indices(selected_indices)
        source_description = source_selected.describe_factors(1, usedNe=3)[0]
        self.assertEqual(source_selected.mode_indices, selected_indices)
        self.assertEqual(source_description.field_order, _FIELD_ORDER)
        self.assertEqual(source_description.mode_indices, selected_indices)
        source_raw = source_selected.read_factor_block(source_description)
        np.testing.assert_allclose(
            source_raw, case["source_spatial"][np.ix_(*selected_indices)],
        )
        np.testing.assert_allclose(
            source_selected.materialize_unweighted_elemental(source_description, source_raw),
            source_raw.conj(),
        )
        self.assertIsNone(source_template.mode_indices)


if __name__ == "__main__":
    unittest.main()
