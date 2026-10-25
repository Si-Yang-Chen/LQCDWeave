"""New sparse spin metadata persists while old dense archives retain defaults."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import Action, BaryonJoinSpec
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.storage.contraction_plans.api import (
    PlanArchiveError, decode_plan_value, encode_plan_value,
    load_storage_schedule, save_storage_schedule,
)
from correlator.planning.test_planning_api import _expanded
from correlator.planning.test_sparse_spin_join_path import ENTRIES, LIMIT, sparse_spin_problem


class SparseSpinArchiveTests(unittest.TestCase):
    def test_sparse_coefficients_and_explicit_path_round_trip(self):
        values = (BaryonJoinSpec(0, "sink", "G", "E", "ijk", "abc", sparse_spin=ENTRIES),
                  Action("baryon_join", "output", ("G", "E", "P", "Q", "R"),
                         kernel_path=((0, 2), (0, 3), (0, 2), (0, 1)), sparse_spin=ENTRIES))
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.assertEqual(decode_plan_value(encode_plan_value(value)), value)

    def test_current_type_names_accept_missing_new_default_fields(self):
        for value in (Action("baryon_join", "output"),
                      BaryonJoinSpec(0, "sink", "G", "E", "ijk", "abc")):
            encoded = encode_plan_value(value)
            self.assertTrue(encoded["type"].startswith("lqcdweave."))
            del encoded["value"]["sparse_spin"]
            del encoded["value"]["core"]
            self.assertEqual(decode_plan_value(encoded), value)
            if isinstance(value, Action):
                del encoded["value"]["used_np_right"]
                self.assertEqual(decode_plan_value(encoded), value)

    def test_compatibility_does_not_admit_other_missing_or_extra_fields(self):
        encoded = encode_plan_value(Action("baryon_join", "output"))
        missing = deepcopy(encoded)
        del missing["value"]["kernel_path"]
        extra = deepcopy(encoded)
        extra["value"]["unknown_spin_hint"] = None
        for malformed in (missing, extra):
            with self.assertRaisesRegex(PlanArchiveError, "schema"):
                decode_plan_value(malformed)

    def test_archive_rejects_invalid_sparse_coefficients(self):
        encoded = encode_plan_value(BaryonJoinSpec(0, "sink", "G", "E", "ijk", "abc"))
        for entries in (((4, 0, 0, 1j),), ((0, 0, 0, 0j),),
                        ((0, 0, 0, 1j), (0, 0, 0, 2j)),
                        ((1, 0, 0, 1j), (0, 0, 0, 2j))):
            malformed = deepcopy(encoded)
            malformed["value"]["sparse_spin"] = encode_plan_value(entries)
            with self.subTest(entries=entries), self.assertRaises(PlanArchiveError):
                decode_plan_value(malformed)

    def test_full_storage_schedule_restores_sparse_join_without_search(self):
        expanded = _expanded(sparse_spin_problem())
        plan = plan_direct_execution(expanded, memory_limit=LIMIT).plan
        shapes = {key: list(spec.shape) for key, spec in plan.state.problem.specs.items()}
        hardware = {"backend": "numpy", "memory": LIMIT}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sparse-spin.gz"
            save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
            with patch("opt_einsum.contract_path", side_effect=AssertionError("archive must not search")):
                restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
                    expected=(plan.state.problem,), blocks=plan.blocks, preparations=plan.preparations)
        self.assertEqual(restored[0].state.problem.baryon_joins, plan.state.problem.baryon_joins)
        self.assertEqual(restored[0].state.actions, plan.state.actions)
        join = next(action for action in restored[0].state.actions if action.kind == "baryon_join")
        self.assertEqual(join.sparse_spin, ENTRIES)
        self.assertEqual(join.kernel_path, ((0, 2), (0, 3), (0, 2), (0, 1)))


if __name__ == "__main__":
    unittest.main()
