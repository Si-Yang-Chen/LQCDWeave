"""Independent exact search session and public optimizer dispatch."""
import math
import time

from .budget import DEFAULT_PLANNING_SECONDS, SearchBudget, BudgetExhausted
from .protocols import SearchView
from .results import CandidateMemoryExceeded, SearchResult


class SearchSession:
    """Audit incumbents and preserve one exhaustive engine's proof bounds."""
    def __init__(self, space, strategy, *, audit=None, dataset_cost=None,
                 deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000, max_transitions=None,
                 search_memory_limit=None, memory_sampling="adaptive", clock=time.monotonic):
        from .search import ExactSearch
        if not isinstance(strategy, ExactSearch):
            raise TypeError("exact session requires one ExactSearch")
        self.space, self.strategy = space, strategy
        self.audit = audit or (lambda state: state)
        self.dataset_cost = dataset_cost or space.cost
        self.budget = SearchBudget(deadline_seconds=deadline_seconds, max_states=max_states,
            max_transitions=max_transitions, search_memory_limit=search_memory_limit,
            memory_sampling=memory_sampling, clock=clock)
        self.best_state = self.best_plan = self.first_solution = None
        self.best_cost, self.revision = math.inf, 0
        self.complete_plans = self.rejected = self.duplicates = 0
        self.rejected_audits = 0
        self.curve = []
        self._ran = False

    def view(self):
        return SearchView(self.best_cost, self.best_state, self.best_plan, self.revision)

    def submit(self, state):
        from .search import _cost, _finish
        cost = _cost(self.space, state)
        final = _finish(self.space, state, cost)
        if final is None:
            raise ValueError("only complete feasible states can be submitted")
        if cost >= self.best_cost:
            return False
        try:
            dataset = self.audit(final)
        except CandidateMemoryExceeded:
            self.rejected_audits += 1
            self.rejected += 1
            return False
        if not math.isclose(self.dataset_cost(dataset), cost, rel_tol=1e-11, abs_tol=1e-14):
            raise RuntimeError("audited plan cost differs from search cost")
        if self.budget.clock() >= self.budget.deadline:
            raise BudgetExhausted("deadline")
        self.complete_plans += 1
        self.best_state, self.best_plan, self.best_cost = final, dataset, cost
        self.revision += 1
        elapsed = self.budget.clock() - self.budget.started
        if self.first_solution is None:
            self.first_solution = elapsed
        self.curve.append((elapsed, cost))
        return True

    def run(self):
        if self._ran:
            raise ValueError("create a new session to run again")
        self._ran = True
        stream = iter(self.strategy.steps(self.space, self.view))
        reason, limited = "proof_complete", False
        try:
            while True:
                self.budget.check()
                # The schedule amortizes each complete walk; a request outside
                # it costs nothing. The final request below is unconditional.
                if self.budget.memory_due():
                    self.budget.sample_memory([stream, self.strategy, self.best_plan])
                try:
                    event = next(stream)
                except StopIteration:
                    break
                self.budget.consume(event)
                self.rejected += event.rejected
                self.duplicates += event.duplicate
                if event.candidate is not None:
                    self.submit(event.candidate)
            if self.budget.memory_limit is not None:
                self.budget.sample_memory([stream, self.strategy, self.best_plan])
        except BudgetExhausted as exc:
            reason, limited = exc.reason, True
        # Final physical constraints can be stricter than the search key's
        # logical dominance relation. Rejected goals must not certify a proof.
        proved = self.strategy.complete and not self.rejected_audits
        if self.rejected_audits and not limited:
            reason = "physical_admission"
        status = ("OPTIMAL" if proved else "FEASIBLE") if self.best_state is not None else (
            "INFEASIBLE" if proved else "UNSOLVED")
        lower = self.strategy.lower_bound
        if self.best_state is not None:
            lower = self.best_cost if proved else min(lower, self.best_cost)
        statistics = dict(rejected_physical_candidates=self.rejected_audits,
            expanded_states=self.budget.expanded,
            evaluated_transitions=self.budget.transitions, rejected_memory=self.rejected,
            duplicates=self.duplicates, complete_plans=self.complete_plans,
            first_solution_seconds=self.first_solution, sampled_peak_search_bytes=self.budget.sampled_peak_memory,
            elapsed_seconds=self.budget.clock() - self.budget.started, improvement_curve=tuple(self.curve))
        stream.close()
        return SearchResult(status, self.best_state, self.budget.expanded, limited,
            stop_reason=reason, lower_bound_seconds=(lower if math.isfinite(lower) else None)
            if not self.rejected_audits else None,
            statistics=statistics)


def solve_storage(problem, blocks, preparations, *, memory_limit, profile=None,
                  host_memory_limit=None, reserved_host_bytes=0, reserved_device_bytes=0,
                  optimizer="anytime", kernel_batching=False, deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000,
                  max_transitions=None, search_memory_limit=None, memory_sampling="adaptive",
                  clock=time.monotonic, **strategy_options):
    from .options import resolve_strategy_options
    from .results import StorageSearchResult
    if type(kernel_batching) is not bool:
        raise TypeError('kernel_batching must be bool')
    options = resolve_strategy_options(optimizer, **strategy_options)
    controls = dict(deadline_seconds=deadline_seconds, max_states=max_states,
                    max_transitions=max_transitions, search_memory_limit=search_memory_limit,
                    memory_sampling=memory_sampling, clock=clock)
    physical = dict(memory_limit=memory_limit, profile=profile, host_memory_limit=host_memory_limit,
                    reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes)
    if optimizer == "anytime":
        from .graph_session import solve_graph_storage
        from .delivery import executable_delivery
        from dataclasses import replace
        result = solve_graph_storage(problem, blocks, preparations, kernel_batching=kernel_batching,
                                     **physical, **controls, **options)
        scheduling = result.statistics.get("storage_schedule", {})
        if (scheduling.get("required") is not True or scheduling.get("completed") is not True
                or result.plan is None or not result.plan.feasible):
            raise RuntimeError("anytime delivery requires completed storage optimization and resource audit")
        return replace(result, plan=executable_delivery(result.plan))
    from .storage_space import StorageSearchSpace
    if any(j.core is not None and (len(j.core.components) != 1 or j.core.components[0].coefficient != 1)
           for j in problem.baryon_joins):
        raise ValueError("exact modes do not include fused logical cores; supply the existing expanded operation domain")
    from .search import ExactSearch
    space = StorageSearchSpace(problem, blocks, preparations, **physical)
    strategy = ExactSearch(layered=optimizer == "dynamic_programming")
    session = SearchSession(space, strategy, audit=space.plan, dataset_cost=lambda plan: plan.seconds,
                            **controls)
    result = session.run()
    stats = dict(result.statistics)
    stats["rejected_memory"] += space.rejected_memory
    return StorageSearchResult(result.status, session.best_plan, result.explored, result.limited,
        0, session.complete_plans, session.first_solution, stats["rejected_memory"],
        dict(optimizer=optimizer, objective="modeled_compute_cost_under_storage_limits", strategy_options=options,
             engine="layered" if strategy.layered else "dijkstra",
             proof_scope="single_complete_problem", operation_domain="primitive_actions",
             search_memory_accounting="sampled_python_objects"), result.stop_reason,
        result.lower_bound_seconds, stats)
