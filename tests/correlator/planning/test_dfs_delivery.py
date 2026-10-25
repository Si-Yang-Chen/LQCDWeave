"""Fixed-policy delivery, archive replay and metadata-only resource boundaries."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from correlator.planning.test_macro_steps import fixture, ContractionTerm
from correlator.planning.test_planning_api import _expanded
from lqcdweave.contraction.planning.api import find_contraction_path, PathSearchExhausted
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule


class Loader:
    def __init__(self):
        self.reads, self.closed = [], []
    def read_block(self, key, block):
        self.reads.append(key)
        return np.arange(4.) + 1
    def materialize(self, key, source, arrays):
        return arrays[key].copy()
    def release_block(self, key, block, array):
        self.closed.append(key)


class DFSDeliveryTests(unittest.TestCase):
    def test_delivery_releases_after_last_use_and_roundtrip_preserves_events(self):
        space = fixture()
        result = solve_storage(space.problem, space.blocks, space.preparations, memory_limit=10000)
        plan = result.plan
        self.assertFalse(plan.explicit_resources)
        releases = [(index, action) for index, action in enumerate(plan.state.actions)
                    if action.kind == "release"]
        self.assertTrue(releases)
        for index, action in releases:
            self.assertFalse(any(action.tensor in later.operands
                                 for later in plan.state.actions[index + 1:]))
        self.assertEqual(plan.state.loaded, frozenset(plan.state.results))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "schedule.json"
            shapes = {key: spec.shape for key, spec in space.problem.specs.items()}
            save_storage_schedule(target, (plan,), shapes=shapes, hardware={"target": "cpu"})
            restored, _ = load_storage_schedule(target, shapes=shapes, hardware={"target": "cpu"},
                expected=space.problem, blocks=space.blocks, preparations=space.preparations,
                profile=space.profile)
            self.assertEqual(restored[0].events, plan.events)
            loader = Loader()
            values = replay_scene_storage(restored[0], loader)
            np.testing.assert_allclose(values, (np.sum((np.arange(4.) + 1) ** 3),))
            self.assertCountEqual(loader.reads, "ABC")

    def test_public_logical_search_rejects_failed_post_storage_audit_before_io(self):
        terms = (ContractionTerm(("A", "B", "C"), "i,i,i->"),
                 ContractionTerm(("A", "C"), "i,i->i"), ContractionTerm(("B", "C"), "i,i->i"))
        space = fixture(terms, cap=264)
        loader = Loader()
        expanded = _expanded(space.problem, loader=loader, top_blocks=space.blocks,
                             top_preparations=space.preparations)
        with self.assertRaisesRegex(PathSearchExhausted, "post-search storage") as caught:
            find_contraction_path(expanded, memory_limit=264, search={"deadline_seconds": 5, 'component_workers': 1})
        self.assertEqual(caught.exception.search_result.status, "FEASIBLE")
        self.assertGreater(caught.exception.search_result.statistics["complete_plans"], 0)
        self.assertEqual(loader.reads, [])

    def test_host_cache_stays_when_only_device_is_under_pressure(self):
        terms = (ContractionTerm(("A", "B", "C"), "i,i,i->"),
                 ContractionTerm(("A", "C"), "i,i->i"), ContractionTerm(("B", "C"), "i,i->i"))
        space = fixture(terms, cap=264, backend="cupy", host=10000)
        first = space.successors(space.initial_state())[0].state
        host = space.apply(first, space._move(first, "spill_host", "tensor", "B"))
        successors = space.successors(host)
        self.assertTrue(successors)
        self.assertGreater(space.resource_branches, 0)
        self.assertFalse(any(edge.actions[0].kind == "spill_disk" for edge in successors))
        self.assertFalse(any(edge.actions[0].kind == "discard" and edge.actions[0].tensor == "B"
                             and edge.actions[0].resource.domain == "tensor" for edge in successors))

    def test_old_options_and_exact_temp_controls_are_rejected(self):
        space = fixture()
        for mode, options in (("greedy", {}), ("beam", {}), ("lookahead", {}), ("storage", {}),
                              ("anytime", {"max_restarts": 1}), ("anytime", {"portfolio": "round_robin"}),
                              ("anytime", {"widen": True}), ("anytime", {"widen": False}),
                              ("exact", {"disk_limit_bytes": None}), ("dynamic_programming", {"root_width": 4})):
            with self.subTest(mode=mode, options=options), self.assertRaises(ValueError):
                solve_storage(space.problem, space.blocks, space.preparations,
                              optimizer=mode, memory_limit=10000, **options)
