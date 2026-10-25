"""Adaptive retained-size walks: linear total cost and explicit bounds.

The former schedule walked the complete retained object graph every fixed
number of expanded states. Because that graph grows with the retained state
count, the total cost was quadratic. These tests pin the replacement: one walk
is amortized over an interval proportional to the measured graph, the interval
never drops below the former cadence, the sampled peak trails the truth by at
most one scheduled interval, and the former cadence stays reachable.
"""
from dataclasses import dataclass
import sys
import unittest
from unittest.mock import patch

from lqcdweave.contraction.planning.optimizers import budget as budget_module
from lqcdweave.contraction.planning.optimizers.budget import (
    BudgetExhausted, MemorySampling, SearchBudget, StaticMemory, retained_size,
)
from lqcdweave.contraction.planning.optimizers.options import effective_search_options
from correlator.planning.test_integer_search import make_problem, solve


@dataclass
class MutableNode:
    value: object


class CountingSet(set):
    """A static index that reports any attempt to copy it."""

    copies = 0

    def copy(self):
        type(self).copies += 1
        return set(self)


class GrowingGraph:
    """A root list that retains one fresh byte string per expanded state."""

    def __init__(self, payload=1000):
        self.rows = []
        self.payload = payload

    def expand(self):
        self.rows.append(bytes(self.payload))


class WalkLog:
    def __init__(self):
        self.rows = []          # [expansion, bytes, visited objects, interval]
        self.breached = False

    def __len__(self):
        return len(self.rows)

    @property
    def visits(self):
        return sum(row[2] for row in self.rows)


def drives(states, *, mode="adaptive", limit=1 << 40, payload=1000,
           objects_per_state=16, minimum_interval=256):
    """Drive the real schedule for a known number of expanded states.

    Every iteration expands one state, which is the clock the schedule reads,
    and samples when the schedule reports a due walk. That is the same sequence
    a search loop produces through SearchBudget.consume.
    """
    budget = SearchBudget(search_memory_limit=limit, memory_sampling=mode,
                          memory_interval=minimum_interval,
                          memory_objects_per_state=objects_per_state)
    graph = GrowingGraph(payload)
    log = WalkLog()
    original = budget_module.retained_measurement
    original_recorded = MemorySampling.recorded

    def measure(roots, *, static=None):
        size, objects = original(roots, static=static)
        log.rows.append([budget.expanded, size, objects, None])
        return size, objects

    def recorded(sampling, expansions, size, objects, seconds):
        original_recorded(sampling, expansions, size, objects, seconds)
        # The interval this walk scheduled for the next one.
        log.rows[-1][3] = sampling.interval

    with patch.object(budget_module, "retained_measurement", measure), \
         patch.object(MemorySampling, "recorded", recorded):
        for _ in range(states):
            if budget.memory_due():
                try:
                    budget.sample_memory((graph,))
                except BudgetExhausted:
                    log.breached = True
                    break
            budget.expanded += 1
            graph.expand()
    return budget, graph, log


class MemorySamplingScheduleTests(unittest.TestCase):
    def test_adaptive_walks_are_amortized_and_never_beat_the_former_cadence(self):
        # One retained object per state keeps the visited object count equal to
        # the expansion count, so the amortized budget is directly observable.
        budget, _, log = drives(4096, objects_per_state=4, minimum_interval=16)
        self.assertGreater(len(log), 2)
        schedule = budget.memory_sampling_statistics
        self.assertEqual(schedule["walks"], len(log))
        self.assertEqual(schedule["expansions"], 4096)
        self.assertGreater(max(row[3] for row in log.rows), 16)
        # Amortized accounting budget: total visits stay within the declared
        # objects per state times the growth the interval allows.
        self.assertLessEqual(schedule["visited_objects"], 4 * 4096 * 1.5)
        for _, _, objects, interval in log.rows:
            self.assertGreaterEqual(interval, 16)
            self.assertLessEqual(interval, max(16, objects // 4))
        # The documented lag bound: consecutive walks are at most one
        # scheduled interval apart, so the sampled peak trails the true peak by
        # at most that many states of retention.
        for earlier, later in zip(log.rows, log.rows[1:]):
            self.assertLessEqual(later[0] - earlier[0], earlier[3])
        self.assertLessEqual(budget.memory_sampling.max_lag,
                             max(row[3] for row in log.rows))

    def test_total_walk_cost_grows_linearly_not_quadratically(self):
        small = drives(2048, objects_per_state=4, minimum_interval=16)[2]
        large = drives(4096, objects_per_state=4, minimum_interval=16)[2]
        adaptive = large.visits / small.visits
        fixed_small = drives(2048, mode="fixed", minimum_interval=16)[2]
        fixed_large = drives(4096, mode="fixed", minimum_interval=16)[2]
        fixed = fixed_large.visits / fixed_small.visits
        # Doubling the states doubles the amortized cost, but roughly
        # quadruples the fixed-cadence cost because every walk rescans the
        # grown graph.
        self.assertLess(adaptive, 2.6)
        self.assertGreater(fixed, 3.4)
        self.assertLess(large.visits, fixed_large.visits)

    def test_fixed_mode_restores_the_former_cadence_exactly(self):
        _, _, log = drives(2048, mode="fixed")
        self.assertEqual([row[0] for row in log.rows],
                         [256 * index for index in range(8)])
        self.assertTrue(all(row[3] == 256 for row in log.rows))

    def test_limit_breach_is_detected_within_one_scheduled_interval(self):
        limit = 400_000
        budget, _, log = drives(4096, limit=limit, payload=1000, objects_per_state=4,
                                minimum_interval=16)
        self.assertTrue(log.breached)
        reported = log.rows[-1]
        self.assertGreater(reported[1], limit)
        self.assertGreater(budget.peak_memory, limit)
        # The walk before the breach was still inside the limit, so the first
        # walk after the true crossing is the one that reported it.
        self.assertLessEqual(log.rows[-2][1], limit)
        # The reported peak cannot exceed the limit by more than the growth of
        # one scheduled interval, and that interval is the amortized budget.
        growth = (reported[1] - log.rows[-2][1]) / (reported[0] - log.rows[-2][0])
        self.assertLessEqual(reported[1] - limit, growth * log.rows[-2][3])

    def test_sampling_disabled_without_a_limit_and_mode_is_validated(self):
        budget = SearchBudget()
        self.assertIsNone(budget.memory_sampling)
        self.assertIsNone(budget.memory_sampling_statistics)
        self.assertFalse(budget.memory_due())
        budget.sample_memory((object(),))
        self.assertEqual(budget.peak_memory, 0)
        with self.assertRaisesRegex(ValueError, "memory_sampling"):
            SearchBudget(search_memory_limit=1 << 20, memory_sampling="sometimes")
        with self.assertRaisesRegex(ValueError, "mode"):
            MemorySampling(mode="sometimes")
        for objects_per_state in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                MemorySampling(objects_per_state=objects_per_state)

    def test_memory_sampling_is_a_public_search_control(self):
        controls = effective_search_options("anytime", {"memory_sampling": "fixed"})
        self.assertEqual(controls["memory_sampling"], "fixed")
        self.assertEqual(effective_search_options("anytime", {})["memory_sampling"], "adaptive")
        with self.assertRaisesRegex(ValueError, "memory_sampling"):
            effective_search_options("anytime", {"memory_sampling": "sometimes"})


class StaticIndexReuseTests(unittest.TestCase):
    def test_a_dynamic_sample_never_copies_or_extends_the_static_index(self):
        shared = ('stable', tuple(range(50)))
        static = StaticMemory()
        static.include((shared,))
        static.ids = CountingSet(static.ids)
        indexed = set(static.ids)
        dynamic = MutableNode([shared, bytes(64)])
        self.assertEqual(retained_size((dynamic, shared), static=static),
                         retained_size((dynamic, shared)) + static.overhead)
        self.assertEqual(static.ids.copies, 0)
        self.assertEqual(set(static.ids), indexed)


class ScheduledSearchIntegrationTests(unittest.TestCase):
    def test_adaptive_and_fixed_sampling_do_not_change_the_search_result(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il')] * 8,
                               {'A': (2, 3), 'B': (3, 4), 'C': (4, 2)})
        counts = {}
        original = budget_module.retained_measurement

        def measure(roots, *, static=None):
            size, objects = original(roots, static=static)
            counts["walks"] = counts.get("walks", 0) + 1
            return size, objects

        results, walks = {}, {}
        for mode in ("fixed", "adaptive"):
            counts["walks"] = 0
            with patch.object(budget_module, "retained_measurement", measure):
                results[mode] = solve(problem, memory_sampling=mode,
                                      search_memory_limit=1 << 28,
                                      root_width=2, branch_width=2)
            walks[mode] = counts["walks"]
        fixed, adaptive = results["fixed"], results["adaptive"]
        self.assertEqual(adaptive.state.actions, fixed.state.actions)
        self.assertEqual(adaptive.state.results, fixed.state.results)
        self.assertEqual(adaptive.plan.seconds, fixed.plan.seconds)
        self.assertEqual(adaptive.plan.kernel_batches, fixed.plan.kernel_batches)
        self.assertEqual(adaptive.statistics["expanded_states"],
                         fixed.statistics["expanded_states"])
        self.assertEqual(adaptive.statistics["evaluated_transitions"],
                         fixed.statistics["evaluated_transitions"])
        self.assertEqual(adaptive.statistics["actual_width_histogram"],
                         fixed.statistics["actual_width_histogram"])
        self.assertGreater(walks["fixed"], 0)
        self.assertLessEqual(walks["adaptive"], walks["fixed"])
        self.assertEqual(adaptive.statistics["memory_sampling"]["mode"], "adaptive")
        self.assertEqual(fixed.statistics["memory_sampling"]["mode"], "fixed")
        self.assertGreater(adaptive.statistics["sampled_peak_search_bytes"], 0)


if __name__ == '__main__':
    unittest.main()
