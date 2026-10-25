"""Public planning guarantees: conservative splitting and audited direct paths."""

from unittest.mock import patch
import unittest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    CoordinateBatch,
    DiagramBindings,
    DiagramInputs,
    ExpandedCorrelator,
    OutputNode,
    OutputRecipe,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
    describe_coordinates,
    group_coordinate_batches,
)
from lqcdweave.contraction.planning.api import (
    ContractionState,
    PathSearchExhausted,
    find_contraction_path,
    logical_cost_profile,
    logical_source_problem,
    plan_direct_execution,
    plan_scene_storage,
    split_independent_terms,
    validate_execution_binding,
)
from lqcdweave.contraction.performance.api import CostProfile


class _Loader:
    def __init__(self):
        self.reads = []

    def read_block(self, key, block):
        self.reads.append(key)
        return object()

    def materialize(self, key, source, blocks):
        return object()


def _problem(*, shared_pair=False):
    specs = {
        "A": TensorSpec((2, 3), "float64"),
        "B": TensorSpec((3, 4), "float64"),
        "C": TensorSpec((3, 5), "float64"),
    }
    terms = [ContractionTerm(("A", "B"), "ij,jk->ik")]
    if shared_pair:
        terms.append(ContractionTerm(("A", "B"), "ij,jk->ik"))
    else:
        terms.append(ContractionTerm(("A", "C"), "ij,jk->ik"))
    sources = {key: SourceSpec(key, spec.shape[0] * spec.shape[1] * 8)
               for key, spec in specs.items()}
    return ContractionProblem(specs, tuple(terms), sources)


def _expanded(problem, *, loader=None, batch_inputs=(), top_blocks=None,
              top_preparations=None):
    coordinates, batch = describe_coordinates()
    loader = _Loader() if loader is None else loader
    term_ids = tuple(f"term-{index}" for index in range(len(problem.terms)))
    blocks = top_blocks or {
        key: StorageBlock(f"dataset-{key}", "full", source.region_bytes,
                          host_bytes=source.region_bytes)
        for key, source in problem.sources.items()
    }
    preparations = top_preparations or {
        key: SourcePreparation((key,)) for key in problem.sources
    }
    inputs = DiagramInputs(
        problem, blocks, preparations, loader, coordinates, batch, term_ids,
    )
    recipe = OutputRecipe(OutputNode("constant", 0), term_ids, coordinates)
    return ExpandedCorrelator(
        object(), inputs, recipe, DiagramBindings(), coordinates, batch,
    )


class PlanningApiTests(unittest.TestCase):
    def test_both_exact_strategies_optimize_full_preparation_cost(self):
        problem = ContractionProblem(
            {k: TensorSpec((2,), "float64") for k in "ABC"},
            tuple(ContractionTerm(tuple(pair), "i,i->") for pair in ("AB", "AC", "BC")),
            {k: SourceSpec(k, 0) for k in "ABC"})
        blocks = {k: StorageBlock(k, "all", 16) for k in "ABC"}
        preps = {k: SourcePreparation((k,), flops=int(v * 1e9)) for k, v in zip("ABC", (3, 2, 1))}
        expanded = _expanded(problem, top_blocks=blocks, top_preparations=preps)
        for method in ("exact", "dynamic_programming"):
            path = find_contraction_path(expanded, optimize=method, memory_limit=96)
            self.assertEqual(path.search.status, "OPTIMAL")
            self.assertEqual(path.prediction["planner"]["objective"], "modeled_compute_cost_under_storage_limits")
            self.assertEqual(path.prediction["planner"]["storage"], "lifetime")
            self.assertAlmostEqual(path.plan.seconds, 0.007049007006, places=14)
            loads = [a.tensor for a in path.plan.state.actions if a.kind == "load"]
            self.assertEqual(loads.count("C"), 2)
            self.assertEqual(expanded.inputs.loader.reads, [])


    def test_split_proves_single_shared_source_is_not_a_shared_operation(self):
        problem = _problem()
        state = ContractionState.initial(problem)
        before = state.pending

        self.assertEqual(split_independent_terms(state), ((0,), (1,)))
        self.assertEqual(state.problem, problem)
        self.assertEqual(state.pending, before)
        self.assertEqual(set(state.problem.sources), {"A", "B", "C"})

    def test_split_keeps_terms_joint_when_they_can_share_a_contraction(self):
        problem = _problem(shared_pair=True)
        self.assertEqual(
            split_independent_terms(ContractionState.initial(problem)), ((0, 1),)
        )

    def test_search_controls_and_unresolved_status_are_queryable(self):
        expanded = _expanded(_problem())
        path = find_contraction_path(
            expanded, memory_limit=1 << 20, optimize="exact",
            search={"max_states": 1000, "deadline_seconds": 5},
        )
        self.assertEqual(path.search.status, "OPTIMAL")
        self.assertEqual(path.prediction["planner"]["search_controls"]["max_states"], 1000)
        self.assertTrue(path.prediction["planner"]["search_performed"])

        with self.assertRaises(PathSearchExhausted):
            find_contraction_path(
                expanded, memory_limit=1 << 20, optimize="exact",
                search={"max_states": 0},
            )

    def test_public_anytime_options_and_effective_metadata(self):
        expanded = _expanded(_problem())
        for mode, options in (("anytime", {}), ("exact", {}), ("dynamic_programming", {}),
                             ("anytime", {"root_width": 2, "branch_width": 3,
                                          "disk_limit_bytes": 1024}),
                             ("anytime", {"debug": True})):
            with self.subTest(mode=mode):
                path = find_contraction_path(expanded, memory_limit=1 << 20, optimize=mode, search=options)
                effective = path.search.diagnostic["strategy_options"]
                controls = path.prediction["planner"]["search_controls"]
                self.assertTrue(all(controls[k] == v for k, v in effective.items()))
                self.assertEqual(expanded.inputs.loader.reads, [])
                if mode == "dynamic_programming":
                    self.assertEqual(path.search.diagnostic["engine"], "layered")
        for mode in ("greedy", "beam", "lookahead", "storage"):
            with self.assertRaises(ValueError):
                find_contraction_path(expanded, memory_limit=1 << 20, optimize=mode)
        for mode, options in (("exact", {"disk_limit_bytes": None}), ("exact", {"widen": True}),
                              ("exact", {"debug": True}), ("dynamic_programming", {"debug": True}),
                              ("anytime", {"widen": True}), ("anytime", {"widen": False}),
                              ("anytime", {"debug": 1}), ("anytime", {"prove": True}),
                              ("anytime", {"beam_width": 4})):
            with self.assertRaises(ValueError):
                find_contraction_path(expanded, memory_limit=1 << 20, optimize=mode, search=options)

    def test_direct_plan_audits_sources_workspace_and_output_before_execution(self):
        problem = ContractionProblem(
            {"A": TensorSpec((2, 3), "float64"),
             "B": TensorSpec((3, 2), "float64")},
            (ContractionTerm(("A", "B"), "ij,jk->ik"),),
            {"A": SourceSpec("A", 48), "B": SourceSpec("B", 48)},
        )
        loader = _Loader()
        expanded = _expanded(problem, loader=loader)
        with patch(
            "lqcdweave.contraction.planning.path_planning.solve_storage",
            side_effect=AssertionError("direct planning must not search"),
        ):
            path = find_contraction_path(
                expanded, memory_limit=1 << 20, optimize="none"
            )

        self.assertEqual(path.execution_mode, "direct")
        self.assertIsNone(path.search)
        self.assertIsNotNone(path.plan)
        self.assertFalse(path.prediction["planner"]["search_performed"])
        self.assertTrue(path.plan.feasible)
        self.assertTrue(path.plan.state.done)
        self.assertEqual(path.plan.state.actions[0].kind, "load")
        self.assertTrue(all(item.kind != "release" for item in path.plan.state.actions))
        self.assertCountEqual(
            [event.block for event in path.plan.events if event.kind == "read"],
            ("A", "B"),
        )
        self.assertEqual(loader.reads, [])
        from lqcdweave.contraction.planning import path_planning
        with patch.object(path_planning, 'validate_executable_resources',
                          wraps=path_planning.validate_executable_resources) as audit:
            validate_execution_binding(expanded, path)
            self.assertEqual(audit.call_count, 1)
        with self.assertRaisesRegex(ValueError, "plan object"):
            validate_execution_binding(expanded, path.__class__(
                path.expanded, path.plan, path.search, path.prediction,
                path.planning_seconds, path.execution_mode, path.memory_limit,
                path.host_memory_limit, -1, path.profile, path.reserved_host_bytes,
                path.reserved_device_bytes,
            ))

        with self.assertRaisesRegex(MemoryError, "audited physical storage budget"):
            plan_direct_execution(expanded, memory_limit=1)
        self.assertEqual(loader.reads, [])

    def test_direct_dense_order_joins_the_first_shared_label_pair_before_outer(self):
        extent = 32
        specs = {key: TensorSpec((extent, extent), "float64")
                 for key in ("A", "B", "C", "D")}
        problem = ContractionProblem(
            specs,
            (ContractionTerm(("A", "B", "C", "D"), "ab,cd,be,ce->"),),
            {key: SourceSpec(key, extent * extent * 8) for key in specs},
        )
        expanded = _expanded(problem)
        with patch(
            "lqcdweave.contraction.planning.path_planning.solve_storage",
            side_effect=AssertionError("direct planning must not search"),
        ), patch(
            "opt_einsum.contract_path",
            side_effect=AssertionError("dense fixed direct order must not ask for a path"),
        ):
            path = find_contraction_path(expanded, memory_limit=1 << 30, optimize="none")

        first = next(action for action in path.plan.state.actions if action.kind == "contract")
        self.assertEqual(first.operands, ("A", "C"))
        self.assertEqual(path.plan.state.catalog[first.tensor].shape, (extent,))

        # Compare with the prior fixed (0, 1) first pair on the same logical
        # problem and physical source schedule.
        profile = CostProfile()
        model = logical_cost_profile(profile)
        legacy = ContractionState.initial(logical_source_problem(problem))
        for key in problem.sources:
            legacy = legacy.load(key, model, 1 << 60)
        while not legacy.done:
            term_index = next(index for index, pending in enumerate(legacy.pending)
                              if pending is not None)
            pending = legacy.pending[term_index]
            pair = (0,) if len(pending.factors) == 1 else (0, 1)
            legacy = legacy.contract(term_index, pair, model, 1 << 60)
        legacy_plan = plan_scene_storage(
            legacy, expanded.inputs.blocks, expanded.inputs.preparations,
            profile=profile, memory_limit=1 << 60,
        )
        self.assertLess(path.plan.peak_host_bytes, legacy_plan.peak_host_bytes)

    def test_direct_task_batches_join_shared_physical_reads_and_peak(self):
        problem = ContractionProblem(
            {"A": TensorSpec((2,), "float64")},
            (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("A", 16)},
        )
        coordinates, batch = describe_coordinates(time_pairs=({0: 1}, {0: 2}), batch=1)
        chunks = group_coordinate_batches(coordinates, batch)
        loader = _Loader()
        blocks = {"shared": StorageBlock("dataset", "all", 16, host_bytes=16)}
        preparations = {"A": SourcePreparation(("shared",))}
        term_ids = ("term-A",)
        task_batches = tuple(
            DiagramInputs(
                problem, blocks, preparations, loader, coordinates, batch, term_ids,
                coordinate_batch=chunk,
            )
            for chunk in chunks
        )
        inputs = DiagramInputs(
            problem, blocks, preparations, loader, coordinates, batch, term_ids,
            task_batches=task_batches,
        )
        recipe = OutputRecipe(OutputNode("constant", 0), term_ids, coordinates)
        expanded = ExpandedCorrelator(
            object(), inputs, recipe, DiagramBindings(), coordinates, batch,
        )

        path = find_contraction_path(
            expanded, memory_limit=1 << 20, optimize="none"
        )

        self.assertEqual(path.execution_mode, "direct")
        self.assertEqual(len(path.batch_paths), 2)
        for method in ("exact", "dynamic_programming"):
            with self.assertRaisesRegex(NotImplementedError, "one complete problem"):
                find_contraction_path(expanded, optimize=method, memory_limit=1 << 20)
        self.assertEqual(path.batch_shape_keys[0], path.batch_shape_keys[1])
        self.assertEqual(path.prediction["physical_reads"], 1)
        self.assertEqual(
            sum(event.kind == "read" for item in path.batch_paths for event in item.plan.events), 1
        )
        self.assertEqual(
            sum(event.kind == "release" for item in path.batch_paths for event in item.plan.events), 1
        )
        self.assertTrue(all(not item.plan.retained_before and not item.plan.retained_after
                            for item in path.batch_paths))
        self.assertFalse(path.prediction["planner"]["search_performed"])
        self.assertEqual(loader.reads, [])
        from lqcdweave.contraction.planning import path_planning
        with patch.object(path_planning, 'validate_executable_resources',
                          wraps=path_planning.validate_executable_resources) as audit:
            validate_execution_binding(expanded, path)
            self.assertEqual([call.args[0] for call in audit.call_args_list],
                             [item.plan for item in path.batch_paths])


if __name__ == "__main__":
    unittest.main()
