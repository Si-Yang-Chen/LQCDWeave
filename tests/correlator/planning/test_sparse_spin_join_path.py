"""Sparse baryon paths keep their explicit order and scientific coefficients."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import opt_einsum

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
    parse_axis_pattern,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.actions import _apply_selected
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contraction.planning.paths import iter_paths, rebind_scene_path
from lqcdweave.contraction.planning.state import (
    ContractionState, _join_path_cost,
)
from lqcdweave.contraction.planning import state as state_module
from correlator.planning.test_planning_api import _expanded


ENTRIES = ((0, 1, 2, 0.5 + 0.25j), (3, 2, 1, -0.75j))
EQUATION = "ijk,abc,ilad,jmbe,kncf->ldmenf"
PATTERN = parse_axis_pattern(EQUATION)
SHAPES = ((4, 4, 4), (2, 3, 1), (4, 2, 2, 1), (4, 1, 3, 2), (4, 3, 1, 1))
LIMIT = 1 << 26


def sparse_spin_problem(entries=ENTRIES):
    keys = ("G", "E", "P", "Q", "R")
    specs = {key: TensorSpec(shape, "complex128") for key, shape in zip(keys, SHAPES)}
    sources = {key: SourceSpec(key, math.prod(spec.shape) * spec.dtype.itemsize)
               for key, spec in specs.items()}
    return ContractionProblem(
        specs, (ContractionTerm(keys[2:], "ilad,jmbe,kncf->ldmenf"),), sources,
        baryon_joins=(BaryonJoinSpec(0, "sink", "G", "E", "ijk", "abc",
                                   sparse_spin=entries),),
    )


def resident_state(problem):
    state = ContractionState.initial(problem)
    for key in problem.sources:
        state = state.load(key, CostProfile(), LIMIT)
    return state


class SparseSpinJoinPathTests(unittest.TestCase):
    def test_explicit_cost_matches_opt_einsum_for_every_binary_order(self):
        # Unequal mode and open extents expose incorrect elimination/outer-product costs.
        for path in iter_paths(len(SHAPES)):
            with self.subTest(path=path):
                _, info = opt_einsum.contract_path(EQUATION, *SHAPES,
                                                   shapes=True, optimize=path)
                self.assertEqual(_join_path_cost(PATTERN, SHAPES, path),
                                 (int(info.opt_cost), int(info.largest_intermediate)))

    def test_multi_operand_and_scalar_output_costs_match_opt_einsum(self):
        for output, path in (("ldmenf", ((0, 1, 2, 3, 4),)),
                             ("", ((0, 2), (0, 3), (0, 2), (0, 1)))):
            equation = EQUATION.split("->")[0] + "->" + output
            pattern = parse_axis_pattern(equation)
            _, info = opt_einsum.contract_path(equation, *SHAPES, shapes=True, optimize=path)
            self.assertEqual(_join_path_cost(pattern, SHAPES, path),
                             (int(info.opt_cost), int(info.largest_intermediate)))

    def test_malformed_or_incomplete_archived_paths_are_rejected(self):
        for path in ((), ((0, 0),), ((-1, 0),), ((0, 5),), ((True, 2),),
                     ((),), ((0, 2),), ((0, 1, 2, 3, 4), (0, 1))):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "kernel path"):
                _join_path_cost(PATTERN, SHAPES, path)

    def test_direct_plan_and_storage_replay_do_not_search_kernel_paths(self):
        expanded = _expanded(sparse_spin_problem())
        with patch("lqcdweave.contraction.planning.state._prepared_join_path",
                   side_effect=AssertionError("direct must not prepare a searched path")), \
             patch("opt_einsum.contract_path", side_effect=AssertionError("direct must not search")):
            planned = plan_direct_execution(expanded, memory_limit=LIMIT)
        joins = [action for action in planned.plan.state.actions if action.kind == "baryon_join"]
        self.assertEqual(len(joins), 1)
        self.assertEqual(joins[0].kernel_path, ((0, 2), (0, 3), (0, 2), (0, 1)))
        self.assertEqual(joins[0].sparse_spin, ENTRIES)
        self.assertTrue(planned.plan.state.done)
        self.assertEqual(expanded.inputs.loader.reads, [])

    def test_selected_join_replays_chosen_path_instead_of_greedy(self):
        state = resident_state(sparse_spin_problem())
        join = state.problem.baryon_joins[0]
        chosen = state.contract_join(join, CostProfile(), LIMIT,
                                     search_kernel_path=False).actions[-1]
        with patch("lqcdweave.contraction.planning.state._prepared_join_path",
                   side_effect=AssertionError("selected path must be reused")):
            replayed = _apply_selected(state, 0, chosen, CostProfile(), LIMIT)
        self.assertIsNotNone(replayed)
        self.assertEqual(replayed.actions[-1].kernel_path, chosen.kernel_path)
        self.assertEqual(replayed.actions[-1].sparse_spin, chosen.sparse_spin)

    def test_rebind_preserves_path_but_rejects_changed_spin_coefficients(self):
        problem = sparse_spin_problem()
        template = resident_state(problem).contract_join(
            problem.baryon_joins[0], CostProfile(), LIMIT, search_kernel_path=False,
        )
        names = {key: "new-" + key for key in problem.sources}
        renamed = replace(problem,
            specs={names[key]: value for key, value in problem.specs.items()},
            sources={names[key]: replace(value, region=names[key]) for key, value in problem.sources.items()},
            terms=tuple(replace(term, inputs=tuple(names[key] for key in term.inputs))
                        for term in problem.terms),
            baryon_joins=tuple(replace(join, spin=names[join.spin], elemental=names[join.elemental])
                               for join in problem.baryon_joins))
        with patch("opt_einsum.contract_path", side_effect=AssertionError("rebind must not search")):
            rebound = rebind_scene_path(template, renamed, names, profile=CostProfile(), memory_limit=LIMIT)
        self.assertEqual(rebound.actions[-1].kernel_path, template.actions[-1].kernel_path)
        self.assertEqual(rebound.actions[-1].sparse_spin, ENTRIES)
        changed = replace(renamed, baryon_joins=(replace(renamed.baryon_joins[0],
            sparse_spin=((0, 1, 2, 0.5 - 0.25j), ENTRIES[1])),))
        with self.assertRaisesRegex(ValueError, "baryon join topology differs"):
            rebind_scene_path(template, changed, names, profile=CostProfile(), memory_limit=LIMIT)

    def test_ne2_workspace_includes_live_intermediate_packs_and_new_output(self):
        shapes = ((4, 4, 4), (2, 2, 2)) + ((4, 4, 2, 2),) * 3
        fixed = ((0, 2), (0, 3), (0, 2), (0, 1))
        greedy, _ = opt_einsum.contract_path(EQUATION, *shapes, shapes=True, optimize="greedy")
        for path in (fixed, tuple(greedy)):
            with self.subTest(path=path):
                _, largest = _join_path_cost(PATTERN, shapes, path)
                self.assertEqual(largest, 512)
                self.assertEqual(sum(math.prod(shape) for shape in shapes), 264)
                # Last step: old generated tensor512 + two worst-case
                # operand packs(512+64) + newly produced tensor512 =1600.
                # The separately reserved final output512 leaves1088 workspace.
                workspace = state_module._join_path_workspace(PATTERN, shapes, path)
                self.assertEqual(workspace, 1088)
                self.assertGreater(workspace, largest + 264)
                self.assertGreaterEqual(512 + workspace, 2 * 512)  # owned snapshot

    def test_ne2_fixed_and_greedy_join_reject_old_cap_and_fit_new_cap(self):
        original = sparse_spin_problem()
        shapes = ((4, 4, 4), (2, 2, 2)) + ((4, 4, 2, 2),) * 3
        specs = {key: TensorSpec(shape, "complex128")
                 for key, shape in zip(original.specs, shapes)}
        problem = replace(original, specs=specs,
            sources={key: SourceSpec(key, math.prod(spec.shape) * spec.dtype.itemsize)
                     for key, spec in specs.items()})
        state = resident_state(problem)
        join = problem.baryon_joins[0]
        itemsize = 16
        old_cap = state.resident_bytes + (512 + 512 + 264) * itemsize
        new_cap = state.resident_bytes + 1600 * itemsize
        for fixed in (True, False):
            with self.subTest(fixed=fixed):
                options = {"search_kernel_path": not fixed}
                with self.assertRaisesRegex(MemoryError, "baryon join workspace"):
                    state.contract_join(join, CostProfile(), old_cap, **options)
                planned = state.contract_join(join, CostProfile(), new_cap, **options)
                self.assertEqual(planned.peak_bytes, new_cap)
                self.assertTrue(planned.done)


if __name__ == "__main__":
    unittest.main()
