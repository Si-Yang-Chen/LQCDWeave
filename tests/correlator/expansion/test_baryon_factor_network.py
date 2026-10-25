"""Row-packed baryon coefficient/Phi factors against the whole-join lowering."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator

from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.correlator import ContractionTask, calc_diagram_eval, calculate
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contraction.expansion.baryon_factor_network import factor_row_coefficients
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.correlator.api import Diagram
from lqcdweave.operators.single_hadron.api import baryon_diquark_operators
from lqcdweave.operators.single_hadron.baryon import BaryonOperator, BaryonOperatorTerm, SparseSpinTensor

from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries


_SOURCES = {"src0": ((_spin_entries(2), 0.75 - 0.4j),), "src1": ((_spin_entries(7), 0.2 + 0.5j),)}
# snk1 has two spin parts on one Phi; E merges them into one exact constant.
_SINKS = {"snk0": ((_spin_entries(5), -0.3 + 0.8j),),
          "snk1": ((_spin_entries(9), 0.6 - 0.1j), (_spin_entries(11), -0.4 + 0.25j))}
_PAIR = {(0, 1): 1.0, (1, 0): -1.0, (2, 3): 0.5j, (3, 2): -0.5j}


def _diquark(*weights):
    """A diquark matrix on two slots times ``sum_c weight_c`` on the row slot."""
    return SparseSpinTensor(tuple((a, b, c, value * weight) for (a, b), value in _PAIR.items()
                                  for c, weight in weights))


# Rows carried by one quark spin: the sources and snk0/snk2 read one spin each
# (folded into R's selection); snk1 mixes two spins (a separate row operand).
_DIQUARK_SOURCES = {"src0": ((_diquark((0, 1)), 0.75 - 0.4j),), "src1": ((_diquark((2, 1)), 0.2 + 0.5j),)}
_DIQUARK_SINKS = {"snk0": ((_diquark((1, 1)), -0.3 + 0.8j),),
                  "snk1": ((_diquark((2, 1), (3, 0.5)), 0.6 - 0.1j),),
                  "snk2": ((_diquark((0, 1j)), 0.4),)}

# Packaged diquark recipes: one Cg5 doublet at the source (rank 1 with a
# two-spin row operand), Cg5 and Cg4g5 rows at the sink (an exact rank-2 split).
_RECIPES = {recipe.name: recipe for recipe in baryon_diquark_operators()}
_RECIPE_SOURCES = {f"src{r}": ((_RECIPES["diquark_Cg5_G1g"].rows[r], 1.0),) for r in range(2)}
_RECIPE_SINKS = {"snk0": ((_RECIPES["diquark_Cg5_G1g"].rows[0], 1.0),),
                 "snk1": ((_RECIPES["diquark_Cg5_G1g"].rows[1], 1.0),),
                 "snk2": ((_RECIPES["diquark_Cg4g5_G1g"].rows[0], 0.5 - 0.2j),)}

# A vector diquark H quartet: two of its rows span only two C g_i (rank 2),
# all four rows need all three (rank 3).
_VECTOR_SOURCES = {f"src{r}": ((_RECIPES["diquark_Cgi_Hg"].rows[r], 1.0),) for r in range(2)}
_VECTOR_SINKS = {f"snk{r}": ((_RECIPES["diquark_Cgi_Hg"].rows[r], 1.0),) for r in range(4)}


def _row_case(source_slots=(1, 2, 0), sources=_SOURCES, sinks=_SINKS):
    case = _case(source_slots)
    component = case["vertices"]["source"].operator.terms[0].component

    def operator(name, parts):
        return BaryonOperator.from_terms(name, tuple(
            BaryonOperatorTerm(entries, component, coefficient, component.flavors, 0)
            for entries, coefficient in parts))

    vertices = {name: Baryon(case["vertices"]["source"].reader, operator(name, parts), source=True)
                for name, parts in sources.items()}
    vertices.update({name: Baryon(case["vertices"]["sink"].reader, operator(name, parts))
                     for name, parts in sinks.items()})
    template = case["diagram"]
    correlator = np.empty((len(sources), len(sinks)), dtype=object)
    expected = np.zeros(correlator.shape, dtype=np.complex128)
    for i, source in enumerate(sources):
        for j, sink in enumerate(sinks):
            correlator[i, j] = Diagram(template.diagram, [0, 1], [source, sink], template.propagator_list)
            for source_entries, source_coefficient in sources[source]:
                for sink_entries, sink_coefficient in sinks[sink]:
                    expected[i, j] += _independent_oracle(dict(
                        case, source_spin=source_entries.dense(), sink_spin=sink_entries.dense(),
                        source_coefficient=source_coefficient, sink_coefficient=sink_coefficient))
    return case, vertices, correlator, expected


def _expand(case, vertex_map, correlator, **options):
    return expand_correlator(correlator, adapter=BaryonDiagramBinding(sector_scope="dense-v", **options),
                             bindings=DiagramBindings(case["propagators"], vertex_map, "cfg-a"),
                             time_axes=None, time_pairs=None, batch="auto")


def _evaluate(case, vertex_map, correlator, mode, **options):
    result = calc_diagram_eval(correlator, propagator_map=case["propagators"], vertex_map=vertex_map,
                               configuration="cfg-a", optimize=mode,
                               diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v", **options))
    return np.asarray(result.value, dtype=np.complex128)


def _spin_shapes(problem):
    return sorted((spec.description.axes, spec.shape) for spec in problem.specs.values()
                  if spec.description is not None and spec.description.role == "baryon-spin")


def _scan_case():
    """Rows over a time scan; every coordinate reads different Phi and propagators."""
    case, vertices, correlator, expected = _row_case()
    times = 4
    readers = {}
    for side in ("source", "sink"):
        spatial = case[side + "_spatial"]
        values = np.stack([spatial * (1 + .2 * t + .1j * t) for t in range(times)])
        readers[side] = ArrayBaryonElemental({"vvv": values[None, None]},
                                             case["vertices"][side].reader.components,
                                             cfg="cfg-a", data_identity="scan-" + side)
    vertices = {name: Baryon(readers["source" if vertex.dagger else "sink"], vertex.operator,
                             source=vertex.dagger) for name, vertex in vertices.items()}
    propagators = {}
    for q in range(3):
        original = case["propagators"][f"p{q}"].sectors["vsv"][0, 1]
        values = np.empty((times, times, *original.shape), dtype=original.dtype)
        for source in range(times):
            for sink in range(times):
                values[source, sink] = original * (1 + .03 * source + .07j * sink)
        propagators[f"p{q}"] = ArrayPropagator({"vsv": values}, time_convention="absolute",
                                               cfg="cfg-a", data_identity=f"scan-p{q}")
    scanned = np.empty(correlator.shape, dtype=object)
    for index, diagram in np.ndenumerate(correlator):
        scanned[index] = Diagram(diagram.diagram, ["source", "sink"], diagram.vertex_list,
                                 diagram.propagator_list)
    factors = np.array([(1 + .2 * t + .1j * t) * (1 + .07j * t) ** 3 for t in (1, 2, 3)])
    options = dict(vertex_map=vertices, propagator_map=propagators, configuration="cfg-a",
                   time_pairs=[{"source": 0, "sink": t} for t in (1, 2, 3)], batch="auto",
                   optimize="anytime", memory_limit=1 << 28, host_memory_limit=1 << 28)
    return scanned, options, expected[..., None] * factors


class BaryonFactorNetworkTests(unittest.TestCase):
    def test_time_scan_rebinds_each_coordinate_and_replays_archive(self):
        request, options, expected = _scan_case()
        np.testing.assert_allclose(calculate(request, **options).value, expected, rtol=5e-12, atol=1e-10)
        with tempfile.TemporaryDirectory() as directory, ContractionTask(request, **options) as task:
            task.plan()
            np.testing.assert_allclose(task.run().value, expected, rtol=5e-12, atol=1e-10)
            archive = Path(directory) / "rows.json.gz"
            task.save_plan(archive)
            with ContractionTask.load_plan(archive, configuration="cfg-a", vertex_map=options["vertex_map"],
                                           propagator_map=options["propagator_map"]) as loaded:
                np.testing.assert_allclose(loaded.run().value, expected, rtol=5e-12, atol=1e-10)

    def test_every_lowering_replays_its_saved_archive(self):
        request, options, expected = _scan_case()
        for lowering in ("nonzero", "joined"):
            with self.subTest(baryon_vertices=lowering), tempfile.TemporaryDirectory() as directory, \
                    ContractionTask(request, **options, baryon_vertices=lowering) as task:
                task.plan()
                np.testing.assert_allclose(task.run().value, expected, rtol=5e-12, atol=1e-10)
                archive = Path(directory) / "rows.json.gz"
                task.save_plan(archive)
                with ContractionTask.load_plan(archive, configuration="cfg-a", vertex_map=options["vertex_map"],
                                               propagator_map=options["propagator_map"]) as loaded:
                    np.testing.assert_allclose(loaded.run().value, expected, rtol=5e-12, atol=1e-10)

    def test_full_row_product_is_one_term_with_visible_factors(self):
        case, vertices, correlator, _ = _row_case()
        problem = _expand(case, vertices, correlator, vertices="pair").problem
        self.assertTrue(problem.factorized_vertices)
        self.assertEqual(problem.baryon_joins, ())
        self.assertEqual(len(problem.terms), 1)
        # Each row family spans two spin patterns: an exact rank-2 split, not dense C.
        spin = sorted((spec.description.axes, spec.shape) for spec in problem.specs.values()
                      if spec.description is not None and spec.description.role == "baryon-spin")
        self.assertEqual([shape for _, shape in spin], [(2, 4, 4)] * 2 + [(2, 4, 2)] * 2)
        self.assertTrue(all("factor_rank" in axes for axes, _ in spin))
        self.assertEqual(len(problem.terms[0].subscripts.output), 2)

        legacy = _expand(case, vertices, correlator, vertices="joined").problem
        self.assertFalse(legacy.factorized_vertices)
        self.assertEqual(len(legacy.terms), 4)

    def test_row_packed_values_match_oracle_and_whole_join_lowering(self):
        for source_slots in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            case, vertices, correlator, expected = _row_case(source_slots)
            for mode in ("none", "anytime"):
                with self.subTest(source_slots=source_slots, mode=mode):
                    packed = _evaluate(case, vertices, correlator, mode)
                    legacy = _evaluate(case, vertices, correlator, mode, vertices="joined")
                    np.testing.assert_allclose(packed, expected, rtol=2e-12, atol=1e-10)
                    np.testing.assert_allclose(packed, legacy, rtol=2e-12, atol=1e-10)

    def test_diquark_rows_split_to_spin_pair_operands(self):
        for source_slots in ((0, 1, 2), (2, 0, 1)):
            case, vertices, correlator, expected = _row_case(source_slots, _DIQUARK_SOURCES, _DIQUARK_SINKS)
            problem = _expand(case, vertices, correlator, vertices="pair").problem
            axes = sorted(spec.description.axes[:1] + (spec.shape,) for spec in problem.specs.values()
                          if spec.description is not None and spec.description.role == "baryon-spin")
            # No dense (R,4,4,4) constant; 4x4 diquark pairs and one (3,4) row operand remain.
            self.assertNotIn(4, {len(shape) for *_, shape in axes})
            self.assertIn(("operator_row", (3, 4)), axes)
            for mode in ("none", "anytime"):
                with self.subTest(source_slots=source_slots, mode=mode):
                    packed = _evaluate(case, vertices, correlator, mode)
                    legacy = _evaluate(case, vertices, correlator, mode, vertices="joined")
                    np.testing.assert_allclose(packed, expected, rtol=2e-12, atol=1e-10)
                    np.testing.assert_allclose(packed, legacy, rtol=2e-12, atol=1e-10)

    def test_packaged_diquark_recipes_split_without_dense_coefficients(self):
        for source_slots in ((0, 1, 2), (1, 2, 0)):
            case, vertices, correlator, expected = _row_case(source_slots, _RECIPE_SOURCES, _RECIPE_SINKS)
            problem = _expand(case, vertices, correlator, vertices="pair").problem
            shapes = sorted(spec.shape for spec in problem.specs.values()
                            if spec.description is not None and spec.description.role == "baryon-spin")
            self.assertEqual(shapes, [(2, 4), (2, 4, 4), (3, 4, 2), (4, 4)])
            for mode in ("none", "anytime"):
                with self.subTest(source_slots=source_slots, mode=mode):
                    packed = _evaluate(case, vertices, correlator, mode)
                    legacy = _evaluate(case, vertices, correlator, mode, vertices="joined")
                    np.testing.assert_allclose(packed, expected, rtol=2e-12, atol=1e-10)
                    np.testing.assert_allclose(packed, legacy, rtol=2e-12, atol=1e-10)

    def test_packaged_vector_diquark_recipes_split_at_their_exact_rank(self):
        case, vertices, correlator, expected = _row_case((2, 0, 1), _VECTOR_SOURCES, _VECTOR_SINKS)
        problem = _expand(case, vertices, correlator, vertices="pair").problem
        shapes = sorted(spec.shape for spec in problem.specs.values()
                        if spec.description is not None and spec.description.role == "baryon-spin")
        self.assertEqual(shapes, [(2, 4, 2), (2, 4, 4), (3, 4, 4), (4, 4, 3)])
        packed = _evaluate(case, vertices, correlator, "anytime")
        np.testing.assert_allclose(packed, expected, rtol=2e-12, atol=1e-10)
        np.testing.assert_allclose(packed, _evaluate(case, vertices, correlator, "anytime", vertices="joined"),
                                   rtol=2e-12, atol=1e-10)

    def test_partial_row_request_splits_into_complete_sub_products(self):
        case, vertices, correlator, expected = _row_case((2, 0, 1), _DIQUARK_SOURCES, _DIQUARK_SINKS)
        cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        request = np.array([correlator[cell] for cell in cells], dtype=object)
        problem = _expand(case, vertices, request).problem
        self.assertTrue(problem.factorized_vertices)
        self.assertEqual(len(problem.terms), 2)  # a 2x2 block and the single (0, 2) cell
        reference = np.array([expected[cell] for cell in cells])
        for mode in ("none", "anytime"):
            with self.subTest(mode=mode):
                packed = _evaluate(case, vertices, request, mode)
                np.testing.assert_allclose(packed, reference, rtol=2e-12, atol=1e-10)
                np.testing.assert_allclose(packed, _evaluate(case, vertices, request, mode,
                                                             vertices="joined"), rtol=2e-12, atol=1e-10)

    def test_nonzero_entry_split_matches_oracle_for_every_spin_structure(self):
        cases = {"rows": (_SOURCES, _SINKS), "diquark": (_DIQUARK_SOURCES, _DIQUARK_SINKS),
                 "recipes": (_RECIPE_SOURCES, _RECIPE_SINKS), "vector": (_VECTOR_SOURCES, _VECTOR_SINKS)}
        for name, (sources, sinks) in cases.items():
            case, vertex_map, correlator, expected = _row_case((2, 0, 1), sources, sinks)
            problem = _expand(case, vertex_map, correlator, vertices="nonzero").problem
            self.assertTrue(problem.factorized_vertices)
            self.assertEqual(len(problem.terms), 1)
            shapes = _spin_shapes(problem)
            # Weights per row family plus three one-hot selections; no dense (R,4,4,4) constant.
            self.assertTrue(all(axes in (("operator_row", "spin_entry"),) or axes[0] == "spin_entry"
                                for axes, _ in shapes))
            entries = {shape[1] for axes, shape in shapes if axes[0] == "operator_row"}
            self.assertTrue(all(axes[0] == "operator_row" or (shape[0] in entries and shape[1] == 4)
                                for axes, shape in shapes))
            for mode in ("none", "anytime"):
                with self.subTest(case=name, mode=mode):
                    packed = _evaluate(case, vertex_map, correlator, mode, vertices="nonzero")
                    np.testing.assert_allclose(packed, expected, rtol=2e-12, atol=1e-10)

    def test_nonzero_entries_cover_partial_row_requests(self):
        case, vertex_map, correlator, expected = _row_case((2, 0, 1), _DIQUARK_SOURCES, _DIQUARK_SINKS)
        cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        request = np.array([correlator[cell] for cell in cells], dtype=object)
        self.assertEqual(len(_expand(case, vertex_map, request, vertices="nonzero").problem.terms), 2)
        np.testing.assert_allclose(_evaluate(case, vertex_map, request, "anytime", vertices="nonzero"),
                                   np.array([expected[cell] for cell in cells]), rtol=2e-12, atol=1e-10)

    def test_frontend_switch_selects_the_baryon_vertex_lowering(self):
        case, vertex_map, correlator, expected = _row_case()
        options = dict(propagator_map=case["propagators"], vertex_map=vertex_map, configuration="cfg-a",
                       optimize="anytime")
        for lowering, factorized in ((None, True), ("auto", True), ("pair", True), ("nonzero", True),
                                    ("joined", False)):
            with self.subTest(baryon_vertices=lowering):
                result = calc_diagram_eval(correlator, **options, baryon_vertices=lowering)
                self.assertEqual(result.expanded.problem.factorized_vertices, factorized)
                # Only the default plans the opened forms and keeps the least predicted one.
                selection = result.path.prediction.get("baryon_vertex_selection")
                self.assertEqual(selection is not None, lowering in (None, "auto"))
                if selection is not None:
                    predicted = selection["predicted_seconds"]
                    self.assertLessEqual({"pair", "nonzero"} - set(predicted), set(selection["infeasible"])
                                         | set(selection["row_choices"]))
                    self.assertEqual(predicted[selection["selected"]], min(predicted.values()))
                np.testing.assert_allclose(np.asarray(result.value), expected, rtol=2e-12, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "baryon_vertices must be one of"):
            calc_diagram_eval(correlator, **options, baryon_vertices="full")
        with self.assertRaisesRegex(ValueError, "explicit adapter owns its own"):
            calc_diagram_eval(correlator, **options, baryon_vertices="joined",
                              diagram_adapter=BaryonDiagramBinding(sector_scope="dense-v"))

    def test_coefficient_split_rank_and_dense_fallback(self):
        rng = np.random.default_rng(7)
        def random(*shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)
        dense = random(2, 4, 4, 4)
        self.assertIsNone(factor_row_coefficients(dense))
        # Rank 3 at the middle slot: factors (3*16 + 3*4*3) are fewer than 3*64 entries.
        values = np.einsum("kab,rsk->rasb", random(3, 4, 4), random(3, 4, 3))
        factor = factor_row_coefficients(values)
        self.assertEqual((factor.rank, factor.slot, factor.pairs), (3, 1, (0, 2)))
        rebuilt = np.einsum("kab,rsk->rasb", factor.pair, factor.row)
        np.testing.assert_allclose(rebuilt, values, rtol=0, atol=1e-12 * np.abs(values).max())
        # One spin per row stays the rank-1 fold read by R.
        monomial = np.einsum("ab,rc->rabc", random(4, 4), np.eye(4)[[2, 0]] * [[1.5], [-1j]])
        factor = factor_row_coefficients(monomial)
        self.assertEqual((factor.rank, factor.row, factor.spins), (1, None, (2, 0)))

    def test_diagonal_row_request_opens_single_rows(self):
        # Opened single rows run several times faster than the whole join.
        case, vertices, correlator, expected = _row_case()
        diagonal = np.array([correlator[0, 0], correlator[1, 1]], dtype=object)
        self.assertTrue(_expand(case, vertices, diagonal, vertices="joined").problem.baryon_joins)
        for lowering in ("auto", "pair", "nonzero", "dense"):
            with self.subTest(vertices=lowering):
                self.assertTrue(_expand(case, vertices, diagonal, vertices=lowering).problem.factorized_vertices)
                np.testing.assert_allclose(_evaluate(case, vertices, diagonal, "anytime", vertices=lowering),
                                           np.array([expected[0, 0], expected[1, 1]]), rtol=2e-12, atol=1e-10)
        np.testing.assert_allclose(_evaluate(case, vertices, diagonal, "anytime"),
                                   np.array([expected[0, 0], expected[1, 1]]), rtol=2e-12, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
