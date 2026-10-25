"""One bounded process pool for component/root tasks, followed by one DAG audit.

Only internal metadata and selected Actions cross the process boundary. Readers,
numeric payloads, storage scheduling and execution stay in the owner process.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
import io
import multiprocessing
import os
import pickle
import time
from types import MappingProxyType

from ..actions import _live_inputs
from ..canonical_operations import computation_key, describe, replay
from ..state import ContractionState
from .budget import BudgetExhausted, SearchBudget, retained_size
from .graph_dfs import FingerprintDFS
from .integer_space import IntegerSpace
from .integer_scoring import IntegerScorer
from .results import StorageSearchResult
from .protocols import SearchEvent
from .shared_components import component_problem


_WORKER_CANCEL = None
_WORKER_QUOTAS = None
_WORKER_SHARED_STATES = None
_TRANSITION_EVENT = SearchEvent("transition")


def _initialize_worker(cancel, quotas=None, shared_states=None):
    global _WORKER_CANCEL, _WORKER_QUOTAS, _WORKER_SHARED_STATES
    _WORKER_CANCEL = cancel
    _WORKER_QUOTAS = quotas
    _WORKER_SHARED_STATES = shared_states
    # Process parallelism owns the planning CPU budget; nested BLAS teams do
    # not improve Python graph work and must not oversubscribe the allocation.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


class _WorkerBudget(SearchBudget):
    def __init__(self, *, shared_quotas=False, **kwargs):
        super().__init__(**kwargs)
        self.shared_quotas = shared_quotas
        self.reserved = [0, 0]
        self.granted = [0, 0]
        self.cancel_checks = 0

    def check(self, *, transition=False):
        self.cancel_checks += 1
        if self.cancel_checks % 256 == 1 and _WORKER_CANCEL is not None and _WORKER_CANCEL.is_set():
            raise BudgetExhausted("owner_cancelled")
        super().check(transition=transition and not self.shared_quotas)
        if transition and self.shared_quotas:
            self.consume(_TRANSITION_EVENT)

    def consume(self, event):
        index = 0 if event.kind == 'expand' else 1
        shared = None if not self.shared_quotas else _WORKER_QUOTAS[index]
        if shared is not None:
            if not self.reserved[index]:
                with shared.get_lock():
                    amount = min(shared.value, 16 if index == 0 else 256)
                    shared.value -= amount
                if not amount:
                    raise BudgetExhausted('max_states' if index == 0 else 'max_transitions')
                self.reserved[index] = amount
                self.granted[index] += amount
            self.reserved[index] -= 1
        super().consume(event)

    def close(self):
        if self.shared_quotas:
            for shared, amount in zip(_WORKER_QUOTAS, self.reserved):
                if shared is not None and amount:
                    with shared.get_lock():
                        shared.value += amount
            self.reserved[:] = [0, 0]


def _restore_mapping(values):
    return MappingProxyType(values)


class _MetadataPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, MappingProxyType):
            return _restore_mapping, (dict(value),)
        return NotImplemented


def _pack(value):
    # This codec is private trusted IPC, never an archive or external input.
    stream = io.BytesIO()
    _MetadataPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return stream.getvalue()


def _root_key(registry, operation):
    """Registry-independent algebra and implementation identity for a root."""
    calculation = registry.calculations[operation.calculation]
    action = calculation.recipe
    return (computation_key(action), action.tensor, calculation.seconds,
            calculation.workspace, action.kernel_path)


def _bind_root(space, key, checkpoint):
    for operation, cost in space.search_candidates(checkpoint):
        if _root_key(space.registry, operation) == key:
            return ((operation, cost),)
    raise RuntimeError('canonical root operation is unavailable in its worker registry')


def _search_component(packet):
    problem, profile, options, deadline = pickle.loads(packet)
    started = time.monotonic()
    budget = _WorkerBudget(deadline_seconds=max(0, deadline - started), **options['budget'])
    try:
        return _search_component_inner(problem, profile, options, budget, started)
    finally:
        # The wrapper also runs on unexpected worker/model exceptions.
        budget.close()


def _search_component_inner(problem, profile, options, budget, started):
    space = session = None
    reason, limited = "completed", False

    def checkpoint():
        budget.check(transition=True)

    try:
        space = IntegerSpace(problem, profile=profile, checkpoint=checkpoint,
                             debug=options["debug"], memory_limit=options.get('allocation_limit'))
        registry = space.registry
        shared = (None if _WORKER_SHARED_STATES is None or 'shared_component' not in options
                  else _WORKER_SHARED_STATES.client(options['shared_component']))
        if shared is not None:
            space.enable_shared_fingerprint()
        session = FingerprintDFS(
            space, budget=budget, operations=registry.operations,
            operation_costs=registry.operation_costs,
            root_width=options["root_width"], branch_width=options["branch_width"],
            include_ties=options.get("include_ties", True),
            debug=options["debug"], profile_scoring=options["profile_scoring"],
            scorer=None if options["scorer"] == "incremental_time" else IntegerScorer(options["scorer"]),
            shared_states=shared,
        )
        if options.get('root_key') is not None:
            # Reset before binding: keep the initial index for the first apply.
            session.start()
            session.root_choices = _bind_root(space, options['root_key'], checkpoint)
        session.advance()
        # One forced walk per component so a worker's reported peak covers the
        # end of its own search; scheduled walks alone may all be deferred.
        budget.sample_memory(session.memory_roots())
    except BudgetExhausted as exc:
        reason, limited = exc.reason, True
    if session is None:
        if space is not None:
            space.release()
        return _pack(dict(status="UNSOLVED", reason=reason, limited=limited,
                          states=budget.expanded, transitions=budget.transitions,
                          peak_memory=budget.peak_memory, pid=os.getpid(), started=started,
                          finished=time.monotonic(), quota_granted=tuple(budget.granted)))
    result = session.finish(reason, limited, audit=False)
    stats = dict(result.statistics)
    stats.update(expression_enumerations=space.enumerations, expression_applications=space.applied,
                 expression_undos=space.undo_count, candidate_row_rebuilds=space.row_rebuilds,
                 terminal_candidates_deferred=space.terminal_deferred,
                 terminal_family_resolutions=space.terminal_resolutions,
                 terminal_layout_candidates=space.layout_candidates,
                 incremental_rankings=space.incremental_rankings,
                 rank_heap_compactions=space.rank_heap_compactions,
                 candidate_row_descriptions=space.row_descriptions,
                 candidate_row_description_reuses=space.row_description_reuses,
                 dense_recipe_hits=space.dense_recipe_hits,
                 rejected_allocation_bounds=space.registry.rejected_allocations,
                 dense_recipe_builds=space.dense_recipe_builds,
                 registered_data=len(registry.data), registered_patterns=len(registry.patterns),
                 registered_calculations=len(registry.calculations))
    answer = dict(status=result.status, reason=reason, limited=limited,
                  states=budget.expanded, transitions=budget.transitions,
                  peak_memory=budget.peak_memory, pid=os.getpid(), statistics=stats,
                  started=started, finished=time.monotonic(), quota_granted=tuple(budget.granted))
    if session.best_id is not None:
        if options.get('defer_route', False):
            from .choices import OperationPool
            _, route = session.route(session.best_id)
            definitions = copy(registry)
            definitions.operations = OperationPool()
            definitions.operation_costs, definitions.connection_tables = [], []
            answer['logical_route'] = dict(registry=definitions,
                prefix=tuple(session.operations[i] for i in route),
                completion=session.nodes[session.best_id].completion)
        else:
            operations = session.selected_operations()
            stats.update(space.terminal_statistics())
            audited = space.audit(operations, session.best_cost)
            answer.update(actions=audited.actions, snippets=tuple(space.snippets),
                          workspaces=tuple(registry.calculations[op.calculation].workspace for op in operations),
                          catalog={a.tensor: audited.catalog[a.tensor] for a in audited.actions},
                          results=audited.results, completed_joins=audited.completed_joins)
    answer['finished'] = time.monotonic()
    return _pack(answer)


def _materialize_winner(problem, report, profile, *, debug=False):
    """Expand and audit only the component's finally selected root route."""
    if 'logical_route' not in report:
        return report
    saved = report.pop('logical_route')
    registry, operations = saved['registry'], saved['prefix']
    if saved['completion'] is not None:
        from .terminal_completion import expand_completion
        operations = expand_completion(saved['completion'], operations, registry)
        report['statistics']['terminal_summary_expansions'] = 1
    if not debug and not problem.baryon_joins and not problem.sparse_bridges:
        from .selected_path import materialize_dense_path
        audited, snippets = materialize_dense_path(problem, operations,
            report['statistics']['logical_best_seconds'], registry)
    else:
        space = IntegerSpace(problem, profile=profile, registry=registry, debug=debug)
        audited = space.audit(operations, report['statistics']['logical_best_seconds'])
        snippets = space.snippets
    report.update(actions=audited.actions, snippets=tuple(snippets),
        workspaces=tuple(registry.calculations[op.calculation].workspace for op in operations),
        catalog={a.tensor: audited.catalog[a.tensor] for a in audited.actions},
        results=audited.results, completed_joins=audited.completed_joins)
    return report


def _share(total, count, index):
    if total is None:
        return None
    quotient, remainder = divmod(total, count)
    return quotient + (index < remainder)


def _worker_count(requested, count, parent_bytes, memory_limit):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and allocated.isdigit():
        cpus = min(cpus, max(1, int(allocated)))
    workers = min(cpus if requested is None else requested, count, cpus)
    if memory_limit is not None:
        remaining = memory_limit - parent_bytes
        if remaining < 2:
            raise BudgetExhausted('search_memory_limit')
        # Reserve owner/IPC headroom. A complete input graph is a conservative
        # floor for each worker's metadata quota; actual growth remains guarded.
        workers = min(workers, max(1, remaining // (2 * max(1, parent_bytes))))
    return max(1, workers)


def _weighted_shares(total, weights):
    if total is None:
        return (None,) * len(weights)
    # Give each component a minimal chance before distributing extra work.
    baseline = int(total >= len(weights))
    remaining = total - baseline * len(weights)
    weight = sum(weights)
    shares = [baseline + remaining * w // weight for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(remaining * weights[i] % weight), i))
    for i in order[:total - sum(shares)]:
        shares[i] += 1
    return tuple(shares)


def _classification_chunk(packet):
    from .shared_components import shared_operation_keys
    # The parent states whether its whole task is dense: this worker sees only
    # one component, but every chunk's keys must stay mutually comparable.
    problem, deadline, max_transitions, memory_limit, canonical = pickle.loads(packet)
    budget = _WorkerBudget(deadline_seconds=max(0, deadline - time.monotonic()),
                           max_transitions=max_transitions, search_memory_limit=memory_limit)
    transitions = 0
    def checkpoint():
        nonlocal transitions
        budget.check()
        transitions += 1
        budget.transitions += 1
    keys = tuple(shared_operation_keys(problem, checkpoint, canonical=canonical))
    budget.sample_memory((problem, keys))
    return _pack((keys, transitions, os.getpid()))


def classified_chunks(problem, budget, requested, statistics):
    """Yield bounded, deterministic initial-operation keys, never numerical data."""
    if budget.clock is not time.monotonic:
        raise ValueError('parallel classification requires the real monotonic clock')
    chunks = tuple(tuple(range(i, min(i + 256, len(problem.terms))))
                   for i in range(0, len(problem.terms), 256))
    parent_bytes = 0 if budget.memory_limit is None else retained_size((problem,))
    workers = _worker_count(requested, len(chunks), parent_bytes, budget.memory_limit)
    quota = (None if budget.memory_limit is None else
             (budget.memory_limit - parent_bytes) // (2 * workers))
    remaining = None if budget.max_transitions is None else max(0, budget.max_transitions-budget.transitions)
    canonical = not problem.baryon_joins and not problem.sparse_bridges
    context = multiprocessing.get_context('spawn')
    cancel = context.Event()
    pids, pending, cursor = set(), {}, 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_worker, initargs=(cancel,)) as pool:
        try:
            while cursor < len(chunks) or pending:
                while cursor < len(chunks) and len(pending) < workers:
                    budget.check()
                    scenes = chunks[cursor]
                    packet = _pack((component_problem(problem, scenes), budget.deadline,
                                    _share(remaining, len(chunks), cursor), quota, canonical))
                    if quota is not None:
                        reservation = parent_bytes + 2 * (len(packet) + sum(v[1] for v in pending.values()))
                        reservation += (len(pending) + 1) * quota
                        if reservation > budget.memory_limit:
                            raise BudgetExhausted('search_memory_limit')
                        budget.peak_memory = max(budget.peak_memory, reservation)
                    pending[pool.submit(_classification_chunk, packet)] = (scenes, len(packet))
                    cursor += 1
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    scenes, _ = pending.pop(future)
                    keys, transitions, pid = pickle.loads(future.result())
                    budget.transitions += transitions
                    budget.check()
                    pids.add(pid)
                    yield tuple((key, scenes[scene]) for key, scene in keys)
        finally:
            cancel.set()
            for future in pending:
                future.cancel()
    statistics.update(classification_workers=workers, classification_worker_pids=tuple(sorted(pids)))


def _replay_routes(problem, reports, groups, profile):
    state = ContractionState.initial(problem)
    state = replace(state, loaded=frozenset(state.catalog) | _live_inputs(state))
    snippets, workspaces, computations = [], [], {}
    for scenes, report in zip(groups, reports, strict=True):
        for (begin, end), workspace in zip(report["snippets"], report["workspaces"], strict=True):
            start = len(state.actions)
            for saved in report["actions"][begin:end]:
                scene = scenes[saved.term_index]
                join = None
                if saved.kind == "baryon_join":
                    join = next(j for j in problem.baryon_joins
                                if j.scene == scene and j.name == saved.join_id)
                actual, *_ = describe(state, scene, saved.positions, join=join, profile=profile,
                                      kernel_path=saved.kernel_path if join else None,
                                      output_order=saved.equation.output if join is None else None)
                if (actual.tensor != saved.tensor
                        or computation_key(actual) != computation_key(saved)):
                    raise RuntimeError("parallel branch changed its independently audited computation")
                definition = computation_key(actual)
                if computations.setdefault(actual.tensor, definition) != definition:
                    raise RuntimeError("parallel branches merged different full computations under one hash")
                state = replay(state, actual, profile)
            snippets.append((start, len(state.actions)))
            workspaces.append(workspace)
    if not state.done:
        raise RuntimeError("parallel branch routes do not finish the complete original problem")
    return state, tuple(snippets), tuple(workspaces)


def _merge_routes(problem, reports, groups, profile, *, debug=False):
    """Assemble independently audited worker routes in linear passes.

    Workers and this owner share a trusted, frozen IPC protocol. Global data
    dependencies and physical resource feasibility are checked by the existing
    fixed-DAG scheduler. An independent full logical replay is diagnostic only.
    """
    initial = ContractionState.initial(problem)
    catalog = dict(initial.catalog)
    loaded = set(catalog) | set(_live_inputs(initial))
    actions, snippets, workspaces, computations = [], [], [], {}
    results, completed = [None] * len(problem.terms), set()
    for scenes, report in zip(groups, reports, strict=True):
        # Check computation collisions once per occurrence without re-running
        # cost prediction or copying all previously assembled state.
        for saved in report['actions']:
            definition = computation_key(saved)
            if computations.setdefault(saved.tensor, definition) != definition:
                raise RuntimeError('parallel branches merged different full computations under one hash')
        for key, spec in report['catalog'].items():
            if key in catalog and catalog[key] != spec:
                raise RuntimeError('parallel branches disagree on a shared tensor specification')
            catalog[key] = spec
        for (begin, end), workspace in zip(report['snippets'], report['workspaces'], strict=True):
            start = len(actions)
            for saved in report['actions'][begin:end]:
                reused = saved.tensor in loaded
                actions.append(replace(saved, term_index=scenes[saved.term_index],
                                       reused=reused, seconds=0.0 if reused else saved.seconds))
                loaded.add(saved.tensor)
            snippets.append((start, len(actions)))
            workspaces.append(workspace)
        for scene, result in zip(scenes, report['results'], strict=True):
            if results[scene] is not None or result is None or result not in catalog:
                raise RuntimeError('parallel routes omit or repeat a completed Term')
            results[scene] = result
        completed.update((scenes[scene], name) for scene, name in report['completed_joins'])
    if any(result is None for result in results):
        raise RuntimeError('parallel branch routes do not finish the complete original problem')
    state = replace(initial, catalog=catalog, loaded=loaded, actions=tuple(actions),
                    pending=(None,) * len(results), results=tuple(results),
                    completed_joins=frozenset(completed), seconds=sum(a.seconds for a in actions))
    if debug:
        reference, reference_snippets, reference_workspaces = _replay_routes(problem, reports, groups, profile)
        if (state.actions != reference.actions or state.results != reference.results
                or dict(state.catalog) != dict(reference.catalog)
                or state.completed_joins != reference.completed_joins
                or tuple(snippets) != reference_snippets or tuple(workspaces) != reference_workspaces):
            raise RuntimeError('assembled component DAG differs from independent replay')
    return state, tuple(snippets), tuple(workspaces)


def _discover_roots(physical, groups, budget, *, root_width, branch_width,
                    include_ties, debug, scorer, profile_scoring):
    """Rank initial alternatives once in the owner using the ordinary policy."""
    roots, statistics = [], []
    def checkpoint():
        budget.check(transition=True)
    for scenes in groups:
        space = IntegerSpace(component_problem(physical.problem, scenes),
                             profile=physical.logical_profile, checkpoint=checkpoint, debug=debug,
                             memory_limit=physical.allocation_limit)
        registry = space.registry
        session = FingerprintDFS(space, budget=budget, operations=registry.operations,
            operation_costs=registry.operation_costs, root_width=root_width,
            branch_width=branch_width, include_ties=include_ties, debug=debug,
            scorer=None if scorer == 'incremental_time' else IntegerScorer(scorer),
            profile_scoring=profile_scoring)
        session.memory_roots = lambda: (physical, groups, roots, statistics, session)
        try:
            session.start()
            identities = () if space.is_done() else session.rank_candidates(root_width, root=True)
            roots.append(tuple(_root_key(registry, session.operations[i])
                               for i in identities) or (None,))
            statistics.append(dict(actual_width_histogram=dict(session.actual_width_histogram),
                extra_tied_choices=session.extra_tied_choices,
                scoring_evaluations=session.scoring_evaluations,
                scoring_seconds=session.scoring_seconds,
                rejected_allocation_bounds=space.registry.rejected_allocations,
                truncated_candidates=session.dropped))
        finally:
            space.release()
    return tuple(roots), tuple(statistics)


def solve_parallel_components(physical, groups, budget, *, component_workers,
                              root_width, branch_width, debug, scorer,
                              profile_scoring, clock, include_ties=True, kernel_batching=False,
                              storage_branch_width=4):
    from ..path_planning import PathSearchExhausted
    from .graph_session import finish_fixed_storage
    if clock is not time.monotonic:
        raise ValueError("parallel search requires the real monotonic clock")
    count = len(groups)
    roots, root_stats = _discover_roots(physical, groups, budget, root_width=root_width,
        branch_width=branch_width, include_ties=include_ties, debug=debug,
        scorer=scorer, profile_scoring=profile_scoring)
    # Round-robin components: they are complementary tasks. Roots within each
    # component are alternative complete routes and must never be concatenated.
    tasks = tuple((component, root) for root in range(max(map(len, roots)))
                  for component in range(count) if root < len(roots[component]))
    parent_bytes = (0 if budget.memory_limit is None else
                    retained_size((physical, groups, roots, root_stats, tasks)))
    budget.peak_memory = max(budget.peak_memory, parent_bytes)
    workers = _worker_count(component_workers, len(tasks), parent_bytes, budget.memory_limit)
    local = component_workers is None and len(tasks) == 1
    worker_memory = (None if budget.memory_limit is None else
                     (budget.memory_limit - parent_bytes) // (2 * workers))
    context = multiprocessing.get_context('spawn')
    shared_states = None
    if workers > 1 and include_ties and any(len(items) > 1 for items in roots):
        from .shared_states import SharedStates
        # Reserve native shared bytes in the same owner/worker memory budget.
        slots = 262144 if budget.memory_limit is None else min(262144,
            max(4, (budget.memory_limit - parent_bytes) // (16 * 48)))
        shared_states = SharedStates(context, count, slots=slots)
        parent_bytes += shared_states.nbytes
        workers = _worker_count(component_workers, len(tasks), parent_bytes, budget.memory_limit)
        worker_memory = (None if budget.memory_limit is None else
                        (budget.memory_limit - parent_bytes) // (2 * workers))
    cancel = context.Event()
    remaining_states = max(0, budget.max_states - budget.expanded)
    remaining_transitions = (None if budget.max_transitions is None else
                             max(0, budget.max_transitions - budget.transitions))
    quotas = (context.Value('q', remaining_states),
              None if remaining_transitions is None else context.Value('q', remaining_transitions))
    reports, selected, root_reports = {}, [None] * count, [[] for _ in groups]
    pending, cursor = {}, 0
    searching_started = clock()
    # Finite per-root wall slices give queued alternatives a chance. All slices
    # remain bounded by the one absolute owner deadline, including spawn/queue.
    wall_remaining = max(0.0, budget.deadline - searching_started)
    later_waves = (max(0, len(tasks) - workers) + workers - 1) // workers
    initial_wall_seconds = wall_remaining if not later_waves else wall_remaining / 2
    later_wall_seconds = 0.0 if not later_waves else wall_remaining / (2 * later_waves)
    dispatch_reason = 'completed'

    def sample_parent(packet=None, *, admitting=False, active=()):
        if budget.memory_limit is None:
            return
        # Completed futures still retain their serialized result until this
        # collection pass drops them; a losing report also stays live here.
        parent = parent_bytes + retained_size((selected, packet, root_reports, reports, active))
        parent += 2 * sum(size for _, size in pending.values())
        size = parent + (len(pending) + admitting) * worker_memory
        budget.peak_memory = max(budget.peak_memory, size)
        if size > budget.memory_limit:
            raise BudgetExhausted('search_memory_limit')

    def submit(pool, task, wall_seconds):
        component, root = task
        deadline = min(budget.deadline, clock() + wall_seconds)
        options = dict(root_width=root_width, branch_width=branch_width, debug=debug,
            allocation_limit=physical.allocation_limit,
            defer_route=not local,
            include_ties=include_ties, root_key=roots[component][root], scorer=scorer,
            profile_scoring=profile_scoring, budget=dict(max_states=remaining_states,
                max_transitions=remaining_transitions if local else None,
                search_memory_limit=worker_memory, shared_quotas=not local))
        if shared_states is not None:
            options['shared_component'] = component
        packet = _pack((component_problem(physical.problem, groups[component]),
                        physical.logical_profile, options, deadline))
        sample_parent(packet, admitting=True)
        if local:
            future = Future()
            future.set_result(_search_component(packet))
        else:
            future = pool.submit(_search_component, packet)
        pending[future] = ((component, root, deadline, wall_seconds), len(packet))

    pool_context = (nullcontext(None) if local else ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_initialize_worker,
        initargs=(cancel, quotas, shared_states)))
    with pool_context as pool:
        try:
            while cursor < len(tasks) or pending:
                while cursor < len(tasks) and len(pending) < workers and dispatch_reason == 'completed':
                    try:
                        budget.check()
                        submit(pool, tasks[cursor], initial_wall_seconds if cursor < workers else later_wall_seconds)
                        cursor += 1
                    except BudgetExhausted as exc:
                        dispatch_reason = exc.reason
                        break
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    (component, root, deadline, wall_seconds), _ = pending.pop(future)
                    report = pickle.loads(future.result())
                    budget.expanded += report['states']
                    budget.transitions += report['transitions']
                    record = {key: report[key] for key in ('status', 'reason', 'limited', 'states',
                        'transitions', 'peak_memory', 'pid', 'started', 'finished', 'quota_granted')}
                    record.update(root_index=root, root_key=roots[component][root],
                        deadline=deadline, wall_seconds=wall_seconds,
                        logical_best_seconds=report.get('statistics', {}).get('logical_best_seconds'))
                    root_reports[component].append(record)
                    reports[component, root] = report.get('statistics', {})
                    previous = selected[component]
                    if report['status'] == 'FEASIBLE' and (previous is None or
                            (report['statistics']['logical_best_seconds'], root) <
                            (previous[1]['statistics']['logical_best_seconds'], previous[0])):
                        selected[component] = (root, report)
                    sample_parent(active=(finished, report, previous))
                finished.clear()
                del future, report, previous
                if dispatch_reason != 'completed' and not pending:
                    break
        except BaseException:
            cancel.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            cancel.set()
    all_reports = [report for component in root_reports for report in component]
    limited = dispatch_reason != 'completed' or any(item['limited'] for item in all_reports)
    reason = dispatch_reason if dispatch_reason != 'completed' else next(
        (item['reason'] for item in all_reports if item['limited']), 'completed')
    stats = dict(expanded_states=budget.expanded, evaluated_transitions=budget.transitions,
        shared_state_cache_bytes=0 if shared_states is None else shared_states.nbytes,
        shared_state_reuse=shared_states is not None,
        shared_state_reuse_scope='completed_no_more_expensive_prefixes',
        shared_state_reuse_disabled_reason=(None if shared_states is not None else
            ('truncated_boundary_ties' if not include_ties else 'no_parallel_root_alternatives')),
        sampled_peak_search_bytes=budget.sampled_peak_memory, component_count=count,
        component_scenes=groups, component_workers=workers,
        component_worker_pids=tuple(sorted({item['pid'] for item in all_reports})),
        actual_search_workers=len({item['pid'] for item in all_reports if item['pid'] != os.getpid()}),
        parallel_fallback_reason='single_root_task' if local else None,
        component_root_counts=tuple(map(len, roots)),
        component_roots_started=tuple(len(items) for items in root_reports),
        component_roots_completed=tuple(sum(not r['limited'] for r in items) for items in root_reports),
        component_roots_feasible=tuple(sum(r['status'] == 'FEASIBLE' for r in items) for items in root_reports),
        component_selected_roots=tuple(None if item is None else item[0] for item in selected),
        root_branch_statistics=tuple(tuple(sorted(items, key=lambda r:r['root_index'])) for items in root_reports),
        component_search_wall_seconds=clock() - searching_started,
        component_search_memory_limit=worker_memory, historical_reexpansions=0,
        root_state_budget=remaining_states, root_transition_budget=remaining_transitions,
        root_state_reservation_batch=16, root_transition_reservation_batch=256)
    stats['component_statistics'] = tuple({} if item is None else item[1]['statistics'] for item in selected)
    widths = {}
    for item in tuple(root_stats) + tuple(reports.values()):
        for width, occurrences in item.get('actual_width_histogram', {}).items():
            widths[width] = widths.get(width, 0) + occurrences
    stats.update(actual_width_histogram=dict(sorted(widths.items())),
        extra_tied_choices=sum(item.get('extra_tied_choices', 0) for item in tuple(root_stats)+tuple(reports.values())))
    for name in ('retained_states', 'retained_connections', 'duplicates', 'complete_plans',
                 'cost_relaxations', 'propagated_connections', 'state_replays', 'pooled_operations',
                 'attempted_operations', 'selected_materializations', 'truncated_candidates', 'cost_pruned',
                 'scoring_evaluations', 'expression_enumerations', 'expression_applications',
                 'expression_undos', 'candidate_row_rebuilds', 'terminal_candidates_deferred',
                 'terminal_family_resolutions', 'terminal_layout_candidates', 'registered_data',
                 'registered_patterns', 'registered_calculations', 'incremental_rankings',
                 'rank_heap_compactions', 'candidate_row_descriptions', 'candidate_row_description_reuses',
                 'dense_recipe_hits', 'dense_recipe_builds'):
        stats[name] = sum(item.get(name, 0) for item in reports.values())
    for name in ('shared_state_lookups', 'shared_state_hits', 'shared_state_cross_process_hits',
                 'shared_state_rechecks', 'shared_state_publications',
                 'shared_state_evictions'):
        stats[name] = sum(item.get(name, 0) for item in reports.values())
    for name in ('terminal_summary_requests', 'terminal_summary_hits', 'terminal_family_cache_hits',
                 'terminal_family_cache_builds', 'terminal_summary_builds', 'terminal_summary_fallbacks',
                 'terminal_summary_logical_steps', 'terminal_summary_expansions'):
        stats[name] = sum(item.get(name, 0) for item in reports.values())
    for name in ('truncated_candidates', 'scoring_evaluations'):
        stats[name] += sum(item[name] for item in root_stats)
    stats['rejected_allocation_bounds'] = (sum(item.get('rejected_allocation_bounds',0) for item in reports.values())
                                          + sum(item['rejected_allocation_bounds'] for item in root_stats))
    if any(item is None for item in selected):
        detail = '; per-operation storage budget excluded candidates' if stats['rejected_allocation_bounds'] else ''
        error = PathSearchExhausted(f'parallel DFS found no complete task path: {reason}{detail}; infeasibility is unproved')
        error.search_result = StorageSearchResult('UNSOLVED', None, budget.expanded, limited,
            count, 0, None, 0, {}, reason, None, stats)
        raise error
    events = sorted((record['started'] - budget.started + elapsed, component, cost)
        for component, items in enumerate(root_reports) for record in items
        for elapsed, cost in reports[component, record['root_index']].get('improvement_curve', ()))
    best, curve = {}, []
    for elapsed, component, cost in events:
        best[component] = min(cost, best.get(component, float('inf')))
        if len(best) == count:
            total = sum(best.values())
            if not curve or total < curve[-1][1]:
                curve.append((elapsed, total))
    first, first_cost = curve[0]
    audit_started = clock()
    chosen = [_materialize_winner(component_problem(physical.problem, groups[i]), item[1],
                physical.logical_profile, debug=debug) for i, item in enumerate(selected)]
    stats['terminal_summary_expansions'] = sum(
        item['statistics'].get('terminal_summary_expansions', 0) for item in chosen)
    expected = sum(item['statistics']['logical_best_seconds'] for item in chosen)
    winner_materialization_seconds = clock() - audit_started
    started = clock()
    audited, snippets, workspaces = _merge_routes(physical.problem, chosen, groups,
                                                 physical.logical_profile, debug=debug)
    stats.update(final_audit_seconds=clock()-audit_started, audited_logical_steps=len(snippets),
        owner_winner_materialization_seconds=winner_materialization_seconds,
        owner_route_merge_seconds=clock()-started, owner_independent_replay=debug,
        logical_best_seconds=audited.seconds, cross_component_reused_seconds=expected-audited.seconds,
        first_solution_seconds=first, first_solution_compute_seconds=first_cost,
        component_first_solution_seconds=tuple(min((r['started']-budget.started+
            reports[i,r['root_index']]['first_solution_seconds'] for r in items
            if r['status']=='FEASIBLE'), default=None) for i,items in enumerate(root_reports)),
        # Workers expose per-route counts, not a time-aligned aggregate of all
        # concurrent work at the first full-task solution.
        first_solution_scoring_evaluations=None,
        scoring_seconds=(sum(item.get('scoring_seconds', 0) for item in reports.values())+
                         sum(item['scoring_seconds'] for item in root_stats)) if profile_scoring else None,
        first_solution_scoring_seconds=None,
        improvement_curve=tuple(curve)+((clock()-budget.started, audited.seconds),))
    # One walk deduplicates the shared catalogs, curves and root records that
    # remain live through the full-problem route assembly.
    budget.sample_memory((physical, groups, roots, root_stats, tasks, selected,
        reports, root_reports, all_reports, stats, events, best, curve,
        chosen, audited, snippets, workspaces))
    stats['sampled_peak_search_bytes'] = budget.sampled_peak_memory
    plan = finish_fixed_storage(physical, audited, snippets, workspaces, stats=stats,
                               clock=clock, limited=limited, debug=debug, kernel_batching=kernel_batching,
                               storage_branch_width=storage_branch_width)
    stats['elapsed_seconds'] = clock()-budget.started
    return StorageSearchResult('FEASIBLE', plan, budget.expanded, limited,
        len(tasks), stats['complete_plans'], first, physical.rejected_memory,
        dict(optimizer='anytime', engine='integer_fingerprint_graph_dfs', scorer=scorer,
             objective='compute_time_then_storage_peak', proof_scope='none',
             operation_domain='logical_complete_contractions', storage_prediction='audited_fixed_dag_schedule',
             component_schedule='parallel_root_alternatives_then_fixed_dag_assembly',
             state_identity='blake2b_256_probabilistic_with_final_replay',
             strategy_options=dict(root_width=root_width, branch_width=branch_width,
                storage_branch_width=storage_branch_width,
                include_ties=include_ties, component_workers=component_workers, debug=debug,
                scorer=scorer, profile_scoring=profile_scoring, disk_limit_bytes=physical.disk_limit_bytes),
             search_memory_accounting='sampled_python_objects_with_shared_work_quotas'), reason, None, stats)
