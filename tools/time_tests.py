"""Run the full unittest discovery with per-test wall-clock timing.

Writes timing and failure details to files so long runs stay inspectable.
"""

import json
import sys
import time
import unittest
from pathlib import Path


class TimingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = []

    def startTest(self, test):
        self._started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.perf_counter() - self._started
        outcome = getattr(self, "_outcome", None)
        failed = bool(getattr(outcome, "errors", None) or getattr(outcome, "failures", None)) \
            if outcome is not None else False
        self.timings.append((round(elapsed, 3), test.id(), not failed))
        super().stopTest(test)


class TimingRunner(unittest.TextTestRunner):
    resultclass = TimingResult


def signature(result):
    seen, rows = set(), []
    for kind, entries in (("FAIL", result.failures), ("ERROR", result.errors)):
        for test, trace in entries:
            name = test.id()
            key = (kind, name, test._testMethodName if hasattr(test, "_testMethodName") else "")
            if key in seen:
                continue
            seen.add(key)
            last = [line for line in trace.strip().splitlines() if line.strip()][-1]
            rows.append({"kind": kind, "test": name, "last_line": last[:300]})
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    runner = TimingRunner(stream=sys.stderr, verbosity=1)
    result = runner.run(suite)
    out = root / "scratch"
    out.mkdir(exist_ok=True)
    (out / "test-timings.json").write_text(json.dumps(
        sorted(result.timings, reverse=True), indent=1), encoding="utf-8")
    (out / "test-failures.json").write_text(json.dumps(
        signature(result), indent=1), encoding="utf-8")
    print(f"ran={len(result.timings)} wall={sum(t for t, _, _ in result.timings):.1f}s "
          f"failures={len(result.failures)} errors={len(result.errors)} "
          f"skipped={len(result.skipped)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
