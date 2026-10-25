"""Run the CPU/GPU elemental regression and save a hardware performance report."""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import unittest


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    default_weak_field_dir=root/"tests"/"data"/"weak_field"
    parser.add_argument("--weak-field-dir",type=Path,default=default_weak_field_dir,
                        help="weak_field fixture directory (defaults to the bundled test data)")
    parser.add_argument("--report",required=True)
    args=parser.parse_args()
    weak_field_dir=args.weak_field_dir.resolve()
    sys.path.insert(0,str(root/"src"))
    sys.path.insert(0,str(root/"tests"))
    os.environ["ED_RUN_GPU_BENCHMARK"]="1"
    os.environ["ED_WEAK_FIELD_DIR"]=str(weak_field_dir)
    import numpy as np
    import cupy as cp
    ndevice=cp.cuda.runtime.getDeviceCount()
    if ndevice<1:
        raise RuntimeError("Slurm job has no visible CUDA device")
    from test_elemental_cuda import ElementalCudaTests
    device=cp.cuda.runtime.getDeviceProperties(0)
    gpu_name=device.get("name",b"unknown")
    if isinstance(gpu_name,bytes):
        gpu_name=gpu_name.decode(errors="replace")
    buffer=io.StringIO()
    loader=unittest.TestLoader()
    suite=unittest.TestSuite()
    for module in ("production.elemental.test_elemental","test_single_hadron",
                   "correlator.binding.test_single_hadron","correlator.operators.test_single_hadron",
                   "test_elemental_cuda","test_baryon_elemental",
                   "production.elemental.test_baryon_elemental","production.elemental.test_elemental_workspace",
                   "production.perambulator.test_propagator_stream_cuda"):
        suite.addTests(loader.loadTestsFromName(module))
    with redirect_stdout(buffer):
        result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    output=buffer.getvalue()
    print(output,end="")
    marker="ELEMENTAL_GPU_BENCHMARK="
    bench=next((json.loads(line.split(marker,1)[1]) for line in output.splitlines() if marker in line),None)
    report={"host":socket.gethostname(),"python":platform.python_version(),"numpy":np.__version__,
        "cupy":cp.__version__,"cuda_driver":cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),"device_count":ndevice,"gpu":gpu_name,
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "skipped":len(result.skipped),"test_output":output,"benchmark":bench}
    target=Path(args.report)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("ELEMENTAL_GPU_REPORT="+str(target))
    if not result.wasSuccessful():
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
