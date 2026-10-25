"""Storage-schedule candidate selection: optimized walk vs whole-ready-set scan.

The shipped scheduler finds a release target's ready operations by walking the
target's requirement bits. The pre-optimization rule rescanned the whole ready
set with a full-width integer per target. These tests prove both rules yield the
same candidates in the same order, and that the delivered plans are field-by-
field identical, on small CPU/device fixtures and on a wider DAG.
"""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec, StorageBlock,
    TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.storage import logical_source_problem
from lqcdweave.contraction.planning.optimizers import storage_schedule as scheduler
from correlator.planning.test_storage_schedule import _fixed_path, _physical, _problem


def _frontier(problem, blocks, preparations, steps, *, cap=1_000_000, profile=None):
    state, spans, workspaces = _fixed_path(problem, profile or CostProfile(), steps)
    dag = scheduler._Dag(state, spans, workspaces,
                         _physical(problem, blocks, preparations, cap=cap, profile=profile))
    return scheduler._Frontier(dag)


def _drain(frontier, width, legacy):
    rule = scheduler._legacy_choices if legacy else scheduler._Frontier._ranked_choices
    return [(index, None if goal is None else (goal.domain, goal.key, goal.required,
                                               goal.computations))
            for index, goal in rule(frontier, width, lambda: True)]


def _walk_and_compare(frontier, *, steps, width, seed):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        assert _drain(frontier, width, False) == _drain(frontier, width, True)
        if not frontier.ready:
            return
        frontier.advance(int(rng.choice(sorted(frontier.ready))), reversible=True)
    assert _drain(frontier, width, False) == _drain(frontier, width, True)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_candidate_rules_agree_on_every_frontier(backend, shared, width):
    problem, _, blocks, preparations = _problem(shared=shared)
    problem = replace(problem, backend=backend)
    if backend == "cupy":
        blocks = {key: replace(block, device_bytes=block.host_bytes, host_bytes=16)
                  for key, block in blocks.items()}
    steps = (((0, ("A", "B")), (0, None), (1, None)) if shared else
             ((0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None)))
    frontier = _frontier(problem, blocks, preparations, steps)
    _walk_and_compare(frontier, steps=len(steps), width=width, seed=7)


def test_candidate_rules_agree_after_an_uncounted_finished_prefix():
    # A late-built ranking must still agree once earlier steps are already done.
    problem, _, blocks, preparations = _problem()
    frontier = _frontier(problem, blocks, preparations,
                         ((0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None)))
    frontier.advance(0, reversible=True)
    _walk_and_compare(frontier, steps=3, width=4, seed=11)


def test_candidate_rules_agree_on_a_wide_dag():
    # 60 independent two-factor terms produce many simultaneously ready nodes.
    shapes = {**{f"A{i}": (30, 20) for i in range(60)},
              **{f"B{i}": (20, 12) for i in range(60)}}
    pairs = tuple((f"A{i}", f"B{i}") for i in range(60))
    problem = logical_source_problem(ContractionProblem(
        {key: TensorSpec(shape, "float64") for key, shape in shapes.items()},
        tuple(ContractionTerm(pair, "ij,jk->ik") for pair in pairs),
        {key: SourceSpec(key, 0) for key in shapes}))
    blocks = {key: StorageBlock(key, "full", int(np.prod(shape)) * 8,
                                host_bytes=int(np.prod(shape)) * 8)
              for key, shape in shapes.items()}
    preparations = {key: SourcePreparation((key,)) for key in shapes}
    frontier = _frontier(problem, blocks, preparations, tuple(enumerate(pairs)),
                         cap=1 << 30)
    _walk_and_compare(frontier, steps=12, width=4, seed=23)


def test_delivered_plan_is_identical_under_both_candidate_rules():
    problem, _, blocks, preparations = _problem(shared=True)
    profile = CostProfile(preparation_call_seconds=0.003)
    state, spans, workspaces = _fixed_path(problem, profile,
                                           ((0, ("A", "B")), (0, None), (1, None)))
    physical = _physical(problem, blocks, preparations, profile=profile)
    report = scheduler.verify_schedule_equivalence(
        state, spans, workspaces, physical, max_states=64, branch_width=4)
    assert report["equivalent"], report["differences"]
    assert report["candidate_actions"] == report["reference_actions"]
    assert report["candidate_peak_host_bytes"] == report["reference_peak_host_bytes"]
    assert report["candidate_seconds"] == report["reference_seconds"]
    assert scheduler._LEGACY_CHOICES is False


def test_equivalence_check_leaves_the_production_switch_off_after_failure():
    problem, _, blocks, preparations = _problem()
    state, spans, workspaces = _fixed_path(problem, CostProfile(),
                                           ((0, ("A", "B")), (1, ("D", "E")),
                                            (0, None), (1, None)))
    physical = _physical(problem, blocks, preparations)
    with patch.object(scheduler, "schedule_fixed_dag", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            scheduler.verify_schedule_equivalence(state, spans, workspaces, physical)
    assert scheduler._LEGACY_CHOICES is False


def test_legacy_rule_remains_the_whole_ready_set_scan():
    # The diagnostic must keep scanning `ready`; using the ready bit mask
    # there would make the equivalence check compare a rule with itself.
    problem, _, blocks, preparations = _problem()
    frontier = _frontier(problem, blocks, preparations,
                         ((0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None)))
    legacy = _drain(frontier, 4, True)
    assert legacy
    assert legacy == _drain(frontier, 4, False)
    # No raw owner is resident yet, so every candidate is the unclaimed
    # fallback: the legacy rule must yield those too, not a shortened list.
    assert all(goal is None for _, goal in legacy)
    frontier.advance(0)
    ranked = _drain(frontier, 4, False)
    assert ranked == _drain(frontier, 4, True)
    assert any(goal is not None for _, goal in ranked)


def test_goal_requirements_select_the_same_ready_nodes_without_position_tables():
    problem, _, blocks, preparations = _problem(shared=True)
    frontier = _frontier(problem, blocks, preparations,
                         ((0, ("A", "B")), (0, None), (1, None)))
    frontier.advance(0)
    goals = frontier.release_goals(4)
    assert goals
    for goal in goals:
        eligible = sorted(i for i in frontier.ready if goal.required & (1 << i))
        assert list(scheduler._first_indices(frontier.ready_mask & goal.required, 4)) == eligible[:4]


if __name__ == "__main__":
    import unittest
    unittest.main()
