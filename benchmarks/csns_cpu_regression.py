"""Run the complete CPU unittest entry point as one CSNS Slurm job."""

import argparse
import json
from pathlib import Path
import platform
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", default=None,
                        help="optional focused unittest modules; default discovers all tests")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))
    if args.modules:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                                   for name in args.modules)
    else:
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"python": platform.python_version(), "modules": args.modules,
              "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
