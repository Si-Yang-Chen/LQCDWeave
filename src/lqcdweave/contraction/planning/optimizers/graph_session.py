"""Component first solutions, integer graph DFS, and a full fixed-path audit."""
import time
import math

from .budget import DEFAULT_PLANNING_SECONDS, BudgetExhausted, SearchBudget, StaticMemory
from .graph_dfs import FingerprintDFS
from .integer_space import IntegerRegistry, IntegerSpace
from .integer_scoring import IntegerScorer
from .macro_space import MacroSearchSpace
from .results import StorageSearchResult, SearchResult
from .shared_components import shared_components, component_problem


def solve_graph_storage(problem, blocks, preparations, *, memory_limit, profile=None,
                        host_memory_limit=None, disk_limit_bytes=None,
                        reserved_host_bytes=0, reserved_device_bytes=0,
                        root_width=4, branch_width=4, include_ties=True, debug=False, component_workers=None,
                        storage_branch_width=4,
                        scorer="shared_savings", profile_scoring=False, kernel_batching=False,
                        deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000, max_transitions=None,
                        search_memory_limit=None, memory_sampling="adaptive", clock=time.monotonic,
                        verify=False, verify_index_every=0, verify_full_every=1):
    from ..path_planning import PathSearchExhausted
    if type(verify) is not bool:
        raise ValueError("verify must be bool")
    if type(verify_index_every) is not int or verify_index_every < 0:
        raise ValueError("verify_index_every must be a nonnegative integer")
    if type(verify_full_every) is not int or verify_full_every < 1:
        raise ValueError("verify_full_every must be a positive integer")
    budget = SearchBudget(deadline_seconds=deadline_seconds, max_states=max_states,
        max_transitions=max_transitions, search_memory_limit=search_memory_limit,
        memory_sampling=memory_sampling, clock=clock)
    physical = MacroSearchSpace(problem, blocks, preparations, memory_limit=memory_limit,
        profile=profile, host_memory_limit=host_memory_limit, disk_limit_bytes=disk_limit_bytes,
        reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes)
    # Prepared sources use the existing logical-source contract. The search,
    # final logical replay and physical replay must price the same descriptors.
    problem = physical.problem
    if ((problem.baryon_joins or problem.factorized_vertices) and not debug and not verify
            and component_workers in (None, 1) and not profile_scoring):
        from .lightweight_paths import solve_lightweight_storage
        return solve_lightweight_storage(physical, budget,
            root_width=root_width, branch_width=branch_width, scorer=scorer,
            include_ties=include_ties, debug=debug, kernel_batching=kernel_batching,
            storage_branch_width=storage_branch_width)
    # The dense fixed-path audit can consume search-time occurrence bindings.
    # Debug keeps the fully independent replay, joins/bridges keep their own
    # exhaustive audit, and the parallel owner rebinds routes itself. Defined
    # before any budget check so a budget stop still reaches delivery.
    reuse_bindings = not debug and not problem.baryon_joins and not problem.sparse_bridges
    sessions, curve, groups = [], [], ()
    registry = None
    reason, limited, first_complete = "completed", False, None
    first_scoring_seconds = first_scoring_evaluations = None
    preprocessing_seconds = None
    static_memory = None
    static_index_seconds = 0.0
    registry_memory_counts = (0, 0, 0)

    def sample_memory(*, force=False):
        """One global session walk when the adaptive schedule is due.

        Every component shares one budget and one schedule, so the walk covers
        all live sessions and is amortized over the steps of all of them. A
        forced request always walks; the first request of a session always
        walks so the static index exists before the first growth.
        """
        nonlocal static_index_seconds, registry_memory_counts
        if budget.memory_limit is None:
            return
        if not force and not budget.memory_due():
            return
        if static_memory is not None and registry is not None:
            indexed = time.perf_counter()
            roots, registry_memory_counts = registry.memory_constants_since(registry_memory_counts)
            static_memory.include(roots)
            static_index_seconds += time.perf_counter() - indexed
        budget.sample_memory((sessions, registry, physical, registry_memory_counts), static=static_memory)

    def checkpoint():
        budget.check(transition=True)

    def record():
        nonlocal first_complete, first_scoring_seconds, first_scoring_evaluations
        if groups and len(sessions) == len(groups) and all(s.best_id is not None for s in sessions):
            cost = sum(s.best_cost for s in sessions)
            if not curve or cost < curve[-1][1]:
                elapsed = clock() - budget.started
                curve.append((elapsed, cost))
                if first_complete is None:
                    first_complete = elapsed
                    first_scoring_seconds = sum(s.scoring_seconds for s in sessions)
                    first_scoring_evaluations = sum(s.scoring_evaluations for s in sessions)
                    sample_memory(force=True)

    preprocessing_started = clock()
    classification_stats = {}
    try:
        if clock is not time.monotonic and component_workers not in (None, 1):
            raise ValueError("parallel search requires the real monotonic clock")
        effective_workers = 1 if clock is not time.monotonic else component_workers
        if effective_workers is None:
            from .parallel_components import _worker_count
            if _worker_count(None, 2, 0, None) == 1:
                effective_workers = 1
        groups = shared_components(problem, checkpoint, component_workers=effective_workers,
                                   budget=budget, statistics=classification_stats)
        classification_stats['classification_seconds'] = clock() - preprocessing_started
        if effective_workers != 1:
            if verify:
                # Never silently skip the insurance check: the parallel worker
                # path is out of the verification scope, so demand the serial
                # route instead of returning an unverified plan.
                raise ValueError(
                    "verify=True covers the serial in-process graph search; "
                    "pass component_workers=1 (the parallel worker path is not verified)")
            from .parallel_components import solve_parallel_components
            result = solve_parallel_components(
                physical, groups, budget, component_workers=component_workers,
                root_width=root_width, branch_width=branch_width, include_ties=include_ties, debug=debug,
                scorer=scorer, profile_scoring=profile_scoring, clock=clock, kernel_batching=kernel_batching,
                storage_branch_width=storage_branch_width,
            )
            result.statistics.update(classification_stats)
            return result
        registry = IntegerRegistry(problem, memory_limit=physical.allocation_limit)
        if budget.memory_limit is not None:
            indexed = time.perf_counter()
            static_memory = StaticMemory()
            static_memory.include((problem,))
            static_index_seconds += time.perf_counter() - indexed
        checkpoint()
        for scenes in groups:
            space = IntegerSpace(component_problem(problem, scenes), profile=physical.logical_profile,
                                 registry=registry, checkpoint=checkpoint, debug=debug)
            if static_memory is not None:
                indexed = time.perf_counter()
                static_memory.include(space.memory_constants())
                static_index_seconds += time.perf_counter() - indexed
            session = FingerprintDFS(space, root_width=root_width, branch_width=branch_width,
                                     include_ties=include_ties,
                                     debug=debug, budget=budget, operations=registry.operations,
                                     operation_costs=registry.operation_costs,
                                     scorer=None if scorer == "incremental_time" else IntegerScorer(scorer),
                                     profile_scoring=profile_scoring,
                                     capture_bindings=reuse_bindings)
            # Local scene rows map back to the full problem for the final audit.
            session.scene_global = scenes
            registry.connection_tables.append(session.connections)
            sessions.append(session)
            session.memory_roots = lambda: (sessions, registry, physical)
            session.memory_sampler = sample_memory
            session.on_solution = record
            session.advance(first_solution=True)
            if session.best_id is None:
                break
        record()
        if preprocessing_seconds is None:
            preprocessing_seconds = clock() - preprocessing_started
        if first_complete is not None:
            for session in sessions:
                session.advance()
                record()
        sample_memory(force=True)
    except BudgetExhausted as exc:
        reason, limited = exc.reason, True
        try:
            sample_memory(force=True)
        except BudgetExhausted:
            pass
        if preprocessing_seconds is None:
            preprocessing_seconds = clock() - preprocessing_started
    except BaseException:
        for session in sessions:
            session.space.release()
            session.frames.clear()
        raise
    results = [s.finish(reason, limited, audit=False) for s in sessions]
    record()
    stats = dict(expanded_states=budget.expanded, evaluated_transitions=budget.transitions,
        sampled_peak_search_bytes=budget.sampled_peak_memory, component_count=len(groups),
        static_memory_index_seconds=static_index_seconds,
        static_memory_index_bytes=0 if static_memory is None else static_memory.overhead,
        static_memory_index_objects=0 if static_memory is None else len(static_memory.ids),
        memory_sampling=budget.memory_sampling_statistics,
        static_registry_definition_counts=registry_memory_counts,
        component_scenes=groups, component_first_solution_seconds=tuple(s.first_solution for s in sessions),
        component_statistics=tuple(r.statistics for r in results), improvement_curve=tuple(curve),
        first_solution_seconds=first_complete, preprocessing_and_first_paths_seconds=preprocessing_seconds,
        first_solution_compute_seconds=curve[0][1] if curve else None,
        first_solution_scoring_seconds=first_scoring_seconds if profile_scoring else None,
        first_solution_scoring_evaluations=first_scoring_evaluations,
        scoring_evaluations=sum(s.scoring_evaluations for s in sessions),
        scoring_seconds=sum(s.scoring_seconds for s in sessions) if profile_scoring else None,
        elapsed_seconds=clock() - budget.started, historical_reexpansions=0)
    stats.update(expression_enumerations=sum(s.space.enumerations for s in sessions),
        expression_applications=sum(s.space.applied for s in sessions),
        expression_undos=sum(s.space.undo_count for s in sessions),
        candidate_row_rebuilds=sum(s.space.row_rebuilds for s in sessions),
        candidate_row_descriptions=sum(s.space.row_descriptions for s in sessions),
        candidate_row_description_reuses=sum(s.space.row_description_reuses for s in sessions),
        dense_recipe_hits=sum(s.space.dense_recipe_hits for s in sessions),
        dense_recipe_builds=sum(s.space.dense_recipe_builds for s in sessions),
        incremental_rankings=sum(s.space.incremental_rankings for s in sessions),
        rank_heap_compactions=sum(s.space.rank_heap_compactions for s in sessions))
    stats.update(classification_stats)
    stats['parallel_fallback_reason'] = (('custom_clock' if clock is not time.monotonic else 'cpu_limit')
                                          if component_workers is None and effective_workers == 1 else None)
    widths = {}
    for result in results:
        for width, count in result.statistics["actual_width_histogram"].items():
            widths[width] = widths.get(width, 0) + count
    stats.update(actual_width_histogram=dict(sorted(widths.items())),
                 extra_tied_choices=sum(r.statistics["extra_tied_choices"] for r in results))
    for name in ("retained_states", "retained_connections", "duplicates", "complete_plans",
                 "cost_relaxations", "propagated_connections", "state_replays", "pooled_operations",
                 "attempted_operations", "selected_materializations", "truncated_candidates", "cost_pruned"):
        stats[name] = sum(r.statistics[name] for r in results)
    stats["pooled_operations"] = 0 if registry is None else len(registry.operations)
    stats['rejected_allocation_bounds'] = 0 if registry is None else registry.rejected_allocations
    if first_complete is None:
        detail = '; per-operation storage budget excluded candidates' if stats['rejected_allocation_bounds'] else ''
        error = PathSearchExhausted(f"logical DFS found no complete task path: {reason}{detail}; infeasibility is unproved")
        error.search_result = StorageSearchResult("UNSOLVED", None, budget.expanded, limited,
            len(sessions), 0, None, 0, {}, reason, None, stats)
        raise error
    # Sampling is complete; the final Action audit owns its own fixed inputs.
    static_memory = None
    operations, route_bindings = [], ([] if reuse_bindings else None)
    for session in sessions:
        selected, session_bindings = session.selected_operations_and_bindings()
        operations.extend(selected)
        if route_bindings is not None:
            if session_bindings is None:
                route_bindings = None
            else:
                mapping = session.scene_global
                route_bindings.extend(
                    None if binding is None else
                    tuple((mapping[row[0]], *row[1:]) for row in binding)
                    for binding in session_bindings)
    expected = sum(s.best_cost for s in sessions)
    audit_started = clock()
    try:
        if reuse_bindings:
            from .selected_path import materialize_dense_path
            audited, snippets = materialize_dense_path(
                problem, operations, expected, registry, bindings=route_bindings)
        else:
            audit_space = IntegerSpace(problem, profile=physical.logical_profile, registry=registry,
                                       debug=debug)
            audited = audit_space.audit(tuple(operations), expected)
            snippets = audit_space.snippets
    finally:
        # The side tables are delivery-only; drop them before verification and
        # storage scheduling.
        for session in sessions:
            session.bindings = None
            session.completion_bindings = None
            session.completion_owner = None
            session.scene_global = None
        route_bindings = None
    if verify:
        # Delivery insurance: an independent non-incremental reference re-walks
        # the winning route and cross-checks the delivered Action table.  This
        # stage is off by default and never changes the selected route.
        from .verification import verify_route
        stats['verification'] = verify_route(
            problem, physical.logical_profile, registry, operations, expected,
            fast_state=audited, debug=debug, index_every=verify_index_every,
            full_every=verify_full_every).summary()
    stats.update(final_audit_seconds=clock() - audit_started, logical_best_seconds=expected,
        terminal_candidates_deferred=sum(s.space.terminal_deferred for s in sessions),
        terminal_family_resolutions=sum(s.space.terminal_resolutions for s in sessions),
        terminal_layout_candidates=sum(s.space.layout_candidates for s in sessions),
        audited_logical_steps=len(snippets), registered_data=len(registry.data),
        registered_patterns=len(registry.patterns), registered_calculations=len(registry.calculations))
    for name in ('terminal_summary_requests', 'terminal_summary_hits', 'terminal_family_cache_hits',
                 'terminal_family_cache_builds', 'terminal_summary_builds', 'terminal_summary_fallbacks',
                 'terminal_summary_logical_steps', 'terminal_summary_expansions'):
        stats[name] = sum(s.space.terminal_statistics().get(name, 0) for s in sessions)
    workspaces = tuple(registry.calculations[operation.calculation].workspace
                       for operation in operations)
    plan = finish_fixed_storage(physical, audited, snippets, workspaces,
                               stats=stats, clock=clock, limited=limited, debug=debug, kernel_batching=kernel_batching,
                               storage_branch_width=storage_branch_width)
    return StorageSearchResult("FEASIBLE", plan, budget.expanded, limited,
        sum(len(s.branch_ticks) for s in sessions), sum(s.complete_plans for s in sessions),
        first_complete, physical.rejected_memory,
        dict(optimizer="anytime", engine="integer_fingerprint_graph_dfs", scorer=scorer,
             objective="compute_time_then_storage_peak", proof_scope="none",
             operation_domain="logical_complete_contractions", storage_prediction="audited_fixed_dag_schedule",
             component_schedule="first_paths_then_serial_completion",
             state_identity="blake2b_256_probabilistic_with_final_replay",
             strategy_options=dict(root_width=root_width, branch_width=branch_width,
                                   storage_branch_width=storage_branch_width,
                                   include_ties=include_ties,
                                   debug=debug, component_workers=component_workers,
                                   disk_limit_bytes=disk_limit_bytes,
                                   scorer=scorer, profile_scoring=profile_scoring),
             search_memory_accounting="sampled_python_objects"), reason, None, stats)


def finish_fixed_storage(physical, audited, snippets, workspaces, *, stats, clock, limited=False, debug=False,
                         kernel_batching=False, storage_branch_width=4):
    """Mandatory final stage: optimize the fixed DAG and audit its resources.

    This stage runs after logical search, even when its budget is exhausted.
    Mark completion only after a complete physically feasible plan is audited.
    """
    from ..path_planning import PathSearchExhausted
    started = clock()
    reference = _diagnostic_storage_baseline(physical, audited, snippets) if debug else None
    from .storage_schedule import schedule_fixed_dag
    plan, scheduling = schedule_fixed_dag(audited, snippets, workspaces,
                                        physical, clock=clock, debug=debug, branch_width=storage_branch_width)
    if reference is not None and not math.isclose(plan.seconds, reference.seconds, rel_tol=1e-11, abs_tol=1e-14):
        raise RuntimeError("diagnostic storage replay changed selected compute-model cost")
    scheduling.update(required=True, completed=False)
    stats["storage_schedule"] = scheduling
    if not plan.feasible:
        error = PathSearchExhausted("selected computation DAG exceeds the post-search storage budget after bounded scheduling; alternative computations were not searched")
        error.search_result = SearchResult("FEASIBLE", audited, stats["expanded_states"], limited,
            stop_reason="post_search_storage_limit", statistics=stats)
        raise error
    scheduling["completed"] = True
    from ..dense_batch_schedule import lower_dense_batches
    lowering_started = clock()
    if kernel_batching:
        plan, lowering = lower_dense_batches(plan, profile=physical.profile, debug=debug)
    else:
        lowering = dict(applied=False, reason='disabled')
    lowering["wall_seconds"] = clock() - lowering_started
    stats["dense_kernel_lowering"] = lowering
    stats["execution_compute_seconds"] = plan.compute_seconds
    stats.update(post_search_storage_seconds=clock() - started,
                 rejected_memory=physical.rejected_memory, resource_branches=0)
    return plan


def _diagnostic_storage_baseline(physical, audited, snippets):
    """Independent legacy state replay for explicitly requested diagnostics."""
    baseline = None
    try:
        cursor, actions = physical.initial_state(), []
        future = {}
        for action in audited.actions:
            keys = (action.tensor,) if action.reused else action.operands
            for key in set(keys):
                future[key] = future.get(key, 0) + 1
        for begin, end in snippets:
            snippet = audited.actions[begin:end]
            for action in snippet:
                keys = (action.tensor,) if action.reused else action.operands
                for key in set(keys):
                    future[key] -= 1
                    if future[key] == 0:
                        del future[key]
            edge = physical.materialize_actions(cursor, snippet, retained=future)
            cursor = edge.state
            actions.extend(edge.actions)
        if not cursor.state.done:
            raise RuntimeError("post-search storage replay did not finish the logical route")
        baseline = physical.plan(cursor, actions)
    except MemoryError:
        # A ready-node order with earlier final-use release can rescue the same
        # computations. No contraction choices or temporary storage are added.
        pass
    return baseline
