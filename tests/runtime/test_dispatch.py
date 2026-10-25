"""File queue behavior without requiring an MPI installation on CPU CI."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lqcdweave.runtime import dispatch
from lqcdweave.runtime.dispatch import AtomicOpen, Dispatch, combine


class SingleRankComm:
    def Get_rank(self):
        return 0

    def bcast(self, value, root=0):
        return value

    def Barrier(self):
        pass


class DispatchTests(unittest.TestCase):
    def test_existing_queue_is_shared_and_completed_items_are_not_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / "cfgs.txt"
            tasks.write_text("# comment\n2000\n\n2010\n2020\n", encoding="utf-8")
            first = Dispatch(str(tasks), "same-run", comm=SingleRankComm())
            iterator = iter(first)
            self.assertEqual(next(iterator), "2000")

            second = Dispatch(str(tasks), "same-run", comm=SingleRankComm())
            self.assertEqual(list(second), ["2010", "2020"])
            self.assertEqual(list(iterator), [])
            self.assertEqual(Path(first.tmp).read_text(encoding="utf-8"), "")

            completed = Path(directory) / "completed.txt"
            combine(str(completed), "2000")
            combine(str(completed), "2010")
            self.assertEqual(completed.read_text(encoding="utf-8"), "2000\n2010\n")
            first.clear()
            self.assertFalse(Path(first.tmp).exists())
            self.assertFalse(Path(first.tmp + ".lock").exists())

    def test_missing_cfglist_fails_before_consuming_items(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / "missing.txt"
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                Dispatch(str(tasks), "missing-run", comm=SingleRankComm())
            self.assertFalse(Path(str(tasks) + ".missing-run.tmp").exists())

    def test_lock_failure_closes_file_opened_by_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.txt"
            opened = open(path, "a+")
            with mock.patch("builtins.open", return_value=opened), \
                 mock.patch.object(dispatch, "lock", side_effect=OSError("lock failed")):
                with self.assertRaisesRegex(OSError, "lock failed"):
                    AtomicOpen(path, "a+")
            self.assertTrue(opened.closed)

    def test_sync_failure_still_unlocks_and_closes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.txt"
            path.write_text("existing", encoding="utf-8")
            observed_lock_positions = []
            observed_unlock_positions = []

            def check_lock_position(file):
                observed_lock_positions.append(file.tell())

            def check_unlock_position(file):
                observed_unlock_positions.append(file.tell())

            with mock.patch.object(dispatch, "lock", side_effect=check_lock_position), \
                 mock.patch.object(dispatch, "unlock") as unlock, \
                 mock.patch.object(dispatch.os, "fsync", side_effect=OSError("sync failed")):
                unlock.side_effect = check_unlock_position
                context = AtomicOpen(path, "a+")
                self.assertEqual(context.begin, len("existing"))
                self.assertEqual(context.file.tell(), context.begin)
                context.file.write("pending")
                with self.assertRaisesRegex(OSError, "sync failed"):
                    context.__exit__(None, None, None)
            unlock.assert_called_once_with(context.file)
            self.assertEqual(observed_lock_positions, [0])
            self.assertEqual(observed_unlock_positions, [0])
            self.assertTrue(context.file.closed)
            self.assertEqual(path.read_text(encoding="utf-8"), "existingpending")

    def test_body_failure_remains_primary_when_sync_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.txt"
            with mock.patch.object(dispatch, "unlock") as unlock, \
                 mock.patch.object(dispatch.os, "fsync", side_effect=OSError("sync failed")):
                with self.assertRaisesRegex(RuntimeError, "body failed"):
                    with AtomicOpen(path, "a+"):
                        raise RuntimeError("body failed")
            unlock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
