"""Compare legacy and new propagator fronts on one CSNS GPU.

Example (on the staged CSNS checkout)::

    python -u benchmarks/compare_quda.py --old-root /path/to/old \
      --data-root /path/to/test --output /path/to/result.json

The benchmark uses the same gauge/eigenvector files and Dirac parameters for
both fronts. It does not apply stout smearing; use --stout only when the
adapter and benchmark environment provide that operation consistently.
"""

import argparse
import gc
import statistics
import numpy as np
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--cfg', default='weak_field')
    p.add_argument('--source-times', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stout', action='store_true')
    p.add_argument('--mrhs', action='store_true',
                   help='use the backend multi-RHS path for batches of four spins')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--atol', type=float, default=1e-8)
    p.add_argument('--rtol', type=float, default=1e-7)
    ns = p.parse_args()
    if ns.repeats < 1 or any(t < 0 or t >= 8 for t in ns.source_times):
        p.error('repeats must be positive and source times must be in 0..7')
    if len(set(ns.source_times)) != len(ns.source_times):
        p.error('source times must be unique')
    if not all(np.isfinite(v) and v >= 0 for v in (ns.atol, ns.rtol)):
        p.error('comparison tolerances must be finite and nonnegative')
    return ns


def setup_old(old_root, data_root, mrhs=False):
    sys.path.insert(0, str(old_root))
    from lattice import check_QUDA, set_backend
    set_backend('cupy')
    if not check_QUDA():
        raise RuntimeError('PyQUDA initialization failed')
    from lattice import GaugeFieldIldg, EigenvectorNpy, PerambulatorGenerator
    from lattice import Nc, Nd
    latt_size = [4, 4, 4, 8]
    gauge = GaugeFieldIldg(str(data_root) + os.sep, '.lime', [8, 4, 4, 4, Nd, Nc, Nc])
    eigen = EigenvectorNpy(str(data_root) + os.sep, '.eigenvector.input.npy',
                           [8, 20, 4, 4, 4, Nc], 20)
    return PerambulatorGenerator(
        latt_size=latt_size, gauge_field=gauge, eigenvector_src=eigen,
        mass=0.09253, tol=1e-9, maxiter=1000,
        xi_0=4.8965, nu=0.86679,
        clover_coeff_t=0.8549165664, clover_coeff_r=2.32582045,
        t_boundary=-1, multigrid=False, contract_prec='<c16',
        MRHS=mrhs, use_vectorized=True,
    )


def setup_new(old_root, data_root, mrhs=False):
    # Both fronts read the same gauge file into a PyQUDA gauge payload.
    from lattice import GaugeFieldIldg, EigenvectorNpy
    from lattice import Nc, Nd
    new_src = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(new_src))
    from lqcdweave.adapters.quda import QudaBackend
    from lqcdweave.production import PerambulatorGenerator
    from lqcdweave.production.perambulator.api import VectorBasis
    latt_size = [4, 4, 4, 8]
    gauge = GaugeFieldIldg(str(data_root) + os.sep, '.lime', [8, 4, 4, 4, Nd, Nc, Nc])
    eigen = EigenvectorNpy(str(data_root) + os.sep, '.eigenvector.input.npy',
                           [8, 20, 4, 4, 4, Nc], 20)
    class GaugePayload:
        def load(self, key):
            from pyquda_utils import io
            return io.readQIOGauge(gauge.load(key).file)

    vector_basis = VectorBasis(eigen, 20)
    generator = PerambulatorGenerator(
        latt_size=latt_size, gauge_field=GaugePayload(),
        mass=0.09253, tol=1e-9, maxiter=1000,
        xi_0=4.8965, nu=0.86679,
        clover_coeff_t=0.8549165664, clover_coeff_r=2.32582045,
        t_boundary=-1, multigrid=False, contract_prec='<c16',
        MRHS=mrhs, rhs_batch_size=4 if mrhs else None,
        backend=QudaBackend(measure=True), measure=True,
    )
    return generator, vector_basis


def run_case(label, generator, ns, source_basis=None):
    import cupy as cp

    generator.load(ns.cfg)
    if ns.stout:
        generator.stout_smear(20, 0.1)
    def calculate(source_time):
        if label == 'legacy':
            return generator.calc_new(source_time, products=('VSV',))[0]
        return generator.calc(source_time, src=source_basis,
                              sink=[source_basis])[0].values
    for t in ns.source_times:
        calculate(t)
    cp.cuda.runtime.deviceSynchronize()
    backend = getattr(generator, 'backend', None)
    if backend is not None:
        backend.reset_metrics()
    values, snapshots = [], {}
    for repeat in range(ns.repeats):
        for t in ns.source_times:
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
            result = calculate(t)
            cp.cuda.runtime.deviceSynchronize()
            elapsed = perf_counter() - started
            value = cp.asnumpy(result)
            if value.shape != (8, 4, 4, 20, 20) or not np.isfinite(value).all():
                raise ArithmeticError('invalid VSV shape or non-finite result')
            if t in snapshots and not np.allclose(value, snapshots[t], atol=ns.atol, rtol=ns.rtol):
                raise ArithmeticError('VSV changed between repeats')
            snapshots[t] = value
            values.append(dict(source_time=t, repeat=repeat, seconds=elapsed,
                               metrics=dict(getattr(generator, 'last_metrics', {}))))
    metrics = {}
    if backend is not None:
        metrics = {key: getattr(backend, key) for key in
                   ('solve_seconds', 'to_quda_seconds', 'from_quda_seconds',
                    'solve_calls', 'rhs_count', 'multirhs_calls')}
        if metrics['rhs_count'] != ns.repeats * len(ns.source_times) * 80:
            raise AssertionError('RHS counter includes unmeasured work')
    timings = [v['seconds'] for v in values]
    return dict(label=label, samples=values, snapshots=snapshots,
                mean_seconds=statistics.mean(timings), median_seconds=statistics.median(timings),
                measured_backend_metrics=metrics)


def main():
    ns = args()
    sys.path.insert(0, str(ns.old_root))
    from lattice import set_backend, check_QUDA
    set_backend('cupy')
    if not check_QUDA():
        raise RuntimeError('PyQUDA initialization failed')
    import cupy as cp
    runs, baseline, comparisons = [], None, []
    # ABBA reduces fixed-order bias. Release owned arrays between cases.
    for label in ('legacy', 'new', 'new', 'legacy'):
        generator, source_basis = None, None
        try:
            created = (setup_old if label == 'legacy' else setup_new)(
                ns.old_root, ns.data_root, ns.mrhs)
            if label == 'legacy':
                generator = created
            else:
                generator, source_basis = created
            case = run_case(label, generator, ns, source_basis)
        finally:
            if generator is not None:
                owner = getattr(generator, 'backend', None) or generator.dirac
                owner.destroy()
                del owner, generator
            del source_basis
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        snapshots = case.pop('snapshots')
        if baseline is None:
            baseline = snapshots
        errors = [float(np.max(np.abs(snapshots[t]-baseline[t]))) for t in ns.source_times]
        norm = np.sqrt(sum(float(np.vdot(baseline[t], baseline[t]).real) for t in ns.source_times))
        diff = np.sqrt(sum(float(np.vdot(snapshots[t]-baseline[t], snapshots[t]-baseline[t]).real)
                           for t in ns.source_times))
        passed = all(np.allclose(snapshots[t], baseline[t], atol=ns.atol, rtol=ns.rtol)
                     for t in ns.source_times)
        comparisons.append(dict(label=label, passed=passed, max_abs=max(errors),
                                relative_l2=float(diff/norm) if norm else (0.0 if diff == 0 else None)))
        runs.append(case)
    means = {label: statistics.mean(sample['seconds'] for run in runs if run['label'] == label
                                    for sample in run['samples']) for label in ('legacy', 'new')}
    result = dict(config=dict(lattice=[4,4,4,8], cfg=ns.cfg, source_times=ns.source_times,
                              stout=ns.stout, repeats=ns.repeats, order=['legacy','new','new','legacy'],
                              warmup='all requested sources before each case; excluded from metrics',
                              mass=0.09253, tol=1e-9, maxiter=1000, xi_0=4.8965, nu=0.86679,
                              clover_coeff_t=0.8549165664, clover_coeff_r=2.32582045,
                              t_boundary=-1, product='VSV', MRHS=ns.mrhs,
                              atol=ns.atol, rtol=ns.rtol),
                  runs=runs, numerical_comparisons=comparisons, mean_seconds=means,
                  passed=all(c['passed'] for c in comparisons),
                  speedup_legacy_over_new=means['legacy']/means['new'])
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit('numerical comparison failed')


if __name__ == '__main__':
    main()
