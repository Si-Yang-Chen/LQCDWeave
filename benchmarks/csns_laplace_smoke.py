"""Single-GPU full-time Laplace regression against the bundled weak_field data."""

import argparse
import json
from pathlib import Path

from csns_laplace_benchmark import check_legacy_weak_field


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    import cupy as cp
    import pyquda
    from lqcdweave.adapters.laplace_quda import QudaLaplaceBackend

    pyquda.init(grid_size=[1, 1, 1, 1], backend="cupy")
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    if "V100" not in name:
        raise RuntimeError(f"expected CSNS V100, got {name}")
    fixture = root / "tests" / "data" / "weak_field"
    for filename in ("weak_field.npy", "weak_field.eigenvector.ref.npy",
                     "weak_field.eigenvalue.npy"):
        if not (fixture / filename).is_file():
            raise FileNotFoundError(f"required fixture is missing: {filename}")
    result = check_legacy_weak_field(cp, QudaLaplaceBackend(), fixture, 1e-9, 10, 0.12)
    report = {"gpu": name, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
              "fixture": "weak_field", "result": result}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
