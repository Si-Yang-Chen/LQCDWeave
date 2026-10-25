"""Replay all 80 saved pairs with CuPy preparation, contraction and GPU output assembly."""
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback


# This exact CSNS environment module was inspected: the optional function only
# scans package metadata and emits a duplicate-installation warning. It must
# not depend on other CuPy modules; all DLL/library/runtime setup stays intact.
_CSNS_CUPY_ENV_SHA256 = '9ca19756f917832d22a7de2ac72e9d5f1ca36379b3c509a8d208886855f9a69f'


def _import_cupy(debug=False):
    """Omit only the known optional startup scan; restore the function after import."""
    if debug or 'cupy' in sys.modules:
        return importlib.import_module('cupy'), False
    spec = importlib.util.find_spec('cupy')
    if spec is None or spec.origin is None:
        return importlib.import_module('cupy'), False
    path = Path(spec.origin).with_name('_environment.py')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != _CSNS_CUPY_ENV_SHA256:
        return importlib.import_module('cupy'), False
    name = 'cupy._environment'
    if name in sys.modules:
        return importlib.import_module('cupy'), False
    environment_spec = importlib.util.spec_from_file_location(name, path)
    environment = importlib.util.module_from_spec(environment_spec)
    sys.modules[name] = environment
    try:
        environment_spec.loader.exec_module(environment)
        original = environment._detect_duplicate_installation
        environment._detect_duplicate_installation = lambda: None
        try:
            return importlib.import_module('cupy'), True
        finally:
            environment._detect_duplicate_installation = original
    except BaseException:
        sys.modules.pop(name, None)
        raise


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


class Monitor:
    def __init__(self, directory, backend):
        import psutil
        self.psutil, self.cp = psutil, backend
        self.directory, self.stage = directory, 'startup'
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self.host, self.gpu = [], []
        self.threads = []

    def start(self):
        bus = self.cp.cuda.Device(0).pci_bus_id
        query = 'timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu'
        self.gpu_process = subprocess.Popen(['nvidia-smi', '-i', bus,
            '--query-gpu='+query, '--format=csv,noheader,nounits', '--loop-ms=200'],
            stdout=subprocess.PIPE, stderr=(self.directory/'nvidia-smi.stderr').open('w'), text=True)
        def gpu_samples():
            with (self.directory/'gpu.jsonl').open('w') as stream:
                for line in self.gpu_process.stdout:
                    values = next(csv.reader([line]))
                    row = dict(elapsed=time.perf_counter()-self.started, stage=self.stage,
                               values=[value.strip() for value in values])
                    self.gpu.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
        def host_samples():
            with self.cp.cuda.Device(0), (self.directory/'host.jsonl').open('w') as stream:
                while not self.stop.is_set():
                    vm = self.psutil.virtual_memory()
                    io = self.process.io_counters()
                    pool = self.cp.get_default_memory_pool()
                    row = dict(elapsed=time.perf_counter()-self.started, stage=self.stage,
                        cpu_percent_one_core=self.process.cpu_percent(None),
                        host_cpu_percent=self.psutil.cpu_percent(None),
                        rss_bytes=self.process.memory_info().rss,
                        host_available_bytes=vm.available, threads=self.process.num_threads(),
                        io_read_bytes=io.read_bytes, io_write_bytes=io.write_bytes,
                        cupy_pool_used_bytes=pool.used_bytes(), cupy_pool_reserved_bytes=pool.total_bytes())
                    self.host.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    self.stop.wait(0.2)
        for target in (gpu_samples, host_samples):
            thread = threading.Thread(target=target, daemon=True)
            self.threads.append(thread); thread.start()

    def finish(self):
        self.stop.set()
        self.gpu_process.terminate()
        self.gpu_process.wait(timeout=10)
        for thread in self.threads:
            thread.join(timeout=10)

    def summary(self):
        host = [row for row in self.host if row['stage'] == 'execution']
        gpu = [row for row in self.gpu if row['stage'] == 'execution']
        summary = dict(sample_period_seconds=0.2, execution_host_samples=len(host),
                       execution_gpu_samples=len(gpu), peaks_are_sampled=True)
        for key in ('cpu_percent_one_core', 'host_cpu_percent', 'rss_bytes',
                    'cupy_pool_used_bytes', 'cupy_pool_reserved_bytes'):
            values = [row[key] for row in host]
            summary[key] = dict(mean=sum(values)/len(values), maximum=max(values)) if values else None
        for key, index in (('gpu_util_percent', 3), ('memory_controller_util_percent', 4),
                           ('gpu_memory_mib', 5), ('gpu_power_watts', 7),
                           ('gpu_sm_clock_mhz', 8), ('gpu_temperature_celsius', 10)):
            values = []
            for row in gpu:
                try: values.append(float(row['values'][index]))
                except ValueError: pass
            summary[key] = dict(mean=sum(values)/len(values), maximum=max(values)) if values else None
        return summary


def main():
    def terminate(signum, frame):
        raise TimeoutError('received termination signal at the one-hour limit')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cfg', default='2000')
    parser.add_argument('--debug', action='store_true', help='enable environment and detailed input diagnostics')
    args = parser.parse_args()
    import resource
    args.output.mkdir(exist_ok=False)
    report = dict(status='starting', cfg=args.cfg, hostname=socket.gethostname(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        logical_cpu_count=os.cpu_count(), allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
        execution_kind='direct SSH on ln005', blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        python=sys.version, plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        repetitions=1, time_pairs=80, diagnostics_per_action=False, numerical_execution=False,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), debug=args.debug, output_numerics='GPU assembly and packing; Dataset pair output')
    monitor = None
    started = time.perf_counter()
    try:
        import numpy as np
        cp, omitted_install_scan = _import_cupy(args.debug)
        report['optional_cupy_install_scan_omitted'] = omitted_install_scan
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.contraction.correlator import load_calculation_plan
        import lqcdweave
        report.update(source_module=lqcdweave.__file__, numpy=np.__version__, cupy=cp.__version__)
        begin = time.perf_counter()
        cp.cuda.Device(0).use()
        report['cuda_context_seconds'] = time.perf_counter()-begin
        properties = cp.cuda.runtime.getDeviceProperties(0)
        report['gpu'] = dict(name=properties['name'].decode() if isinstance(properties['name'], bytes) else properties['name'],
            total_memory_bytes=properties['totalGlobalMem'], pci_bus_id=cp.cuda.Device(0).pci_bus_id,
            runtime=cp.cuda.runtime.runtimeGetVersion(), driver=cp.cuda.runtime.driverGetVersion())
        report['cpu_model'] = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                    if line.startswith('model name')), None)
        monitor = Monitor(args.output, cp); monitor.start()
        monitor.stage = 'plan_load'
        begin = time.perf_counter()
        calculation = load_calculation_plan(args.plan, configuration=args.cfg, readers={}, output=args.output/'correlators')
        plan = calculation.plan
        assert len(plan.templates) == 1 and len(plan.instances) == 80
        pairs = [dict(instance.coordinate) for instance in plan.instances]
        assert pairs == [dict(sink=(source+offset)%40,source=source)
                         for source in range(40) for offset in (1,2)]
        assert plan.coordinate_shape == (80,)
        assert len(plan.output_fields) == 22 and sum(int(np.prod(f.shape)) for f in plan.output_fields) == 522
        report.update(plan_load_seconds=time.perf_counter()-begin, channels=14, row_matrices=22,
            matrix_entries_per_pair=522, total_matrix_entries=522*80, Ne=70, coordinates=pairs,
            kernel_batches=len(plan.instances[0].kernel_batches),
            modeled_peak_device_bytes=plan.peak_device_bytes, modeled_peak_host_bytes=plan.peak_host_bytes)
        monitor.stage = 'bind_readers'
        root, cfg = args.input_root, args.cfg
        lp = root/'03.perambulator.light.single.prec1e-9.nev70.t40.select_t.v1'/f'{cfg}.npy'
        hp = root/'03.perambulator.charm.single.prec1e-15.nev120.t40'/f'{cfg}.npy'
        ed = root/'04.meson.basis55.mom33.nev120.t40.stout20rho0.12.v1'/cfg
        lm = json.loads(lp.with_suffix('.json').read_text())
        em = json.loads((ed/'manifest.json').read_text())
        es = em['identity']['spec']
        axes = {a['name']: a['coordinates'] for a in es['axes']}
        assert tuple(lm['physical_times']) == tuple(axes['time'])
        assert es['definition']['rows'][0]['terms'] == [dict(coefficient=[1.0, 0.0], path=[])]
        light_data, charm_data, elemental_data = (np.load(path, mmap_mode='r', allow_pickle=False)
                                                 for path in (lp, hp, ed/'data.npy'))
        assert light_data.shape == (40,40,4,4,70,70) and charm_data.shape == (40,40,4,4,120,120)
        assert elemental_data.shape == (40,55,33,120,120)
        stat = hp.stat()
        charm_id = f'npy:{hp}:{stat.st_size}:{stat.st_mtime_ns}'
        reader_by_id = {
            lm['instance_id']: ArrayPropagator({'vsv': light_data}, time_convention='absolute', cfg=cfg, data_identity=lm['instance_id']),
            charm_id: ArrayPropagator({'vsv': charm_data}, time_convention='absolute', cfg=cfg, data_identity=charm_id),
            em['artifact_id']: ArrayElemental({'v2v': elemental_data[:,0:1].transpose(1,2,0,3,4)},
                (ElementalComponent.path(()),), momentum_list=axes['momentum'], cfg=cfg,
                phase_convention=es['conventions']['phase'], data_identity=em['artifact_id']),
        }
        readers = {}
        for request in plan.requests.values():
            for r in (request, *request.aliases):
                readers[r.reader_role] = reader_by_id[r.metadata.data_identity]
        report['reader_roles'] = {role: type(reader).__name__ for role, reader in readers.items()}
        report['input_interpretation'] = dict(storage_slot_pairs=pairs,
            known_light_elemental_times=[dict(sink=lm['physical_times'][p['sink']],source=lm['physical_times'][p['source']]) for p in pairs],
            charm_physical_time_mapping='not independently established',
            propagator_common_basis_identity='not independently established',
            scope='execute 80 saved storage bindings for performance; no ensemble-spectrum claim')
        if args.debug:
            report['logical_reads'] = {}
            distinct_readers = {}
            for role, reader in readers.items():
                entry = distinct_readers.setdefault(id(reader), (reader, []))
                entry[1].append(role)
            for reader, roles in distinct_readers.values():
                role = ' + '.join(sorted(roles))
                for method in ('block', 'read_slab'):
                    if not hasattr(reader, method): continue
                    original = getattr(reader, method)
                    counter = dict(calls=0, returned_bytes=0, host_call_seconds=0.0)
                    report['logical_reads'][role+'/'+method] = counter
                    def block(*values, _original=original, _counter=counter, **options):
                        before = time.perf_counter()
                        result = _original(*values, **options)
                        _counter['host_call_seconds'] += time.perf_counter()-before
                        _counter['calls'] += 1
                        _counter['returned_bytes'] += int(getattr(result, 'nbytes', 0))
                        return result
                    setattr(reader, method, block)
        begin = time.perf_counter()
        calculation = replace(calculation, options=dict(calculation.options, readers=readers))
        report['workflow_binding_seconds'] = time.perf_counter()-begin
        plan = calculation.plan
        assert plan.output_mode == 'dataset'
        assert plan.statistics.get('postprocess_backend') == 'gpu'
        assert plan.output_layout == 'pairs_and_dt' and plan.physical_lt is None
        report['numeric_routing'] = dict(input_preparation='cupy', contractions='cupy',
            output_assembly='gpu', output_packing='gpu', host_term_assembly=False,
            output_semantics='all 80 ordered pairs, 22 row matrices, 522 complex entries per pair',
            cpu_scope='file I/O, metadata, launch submission, final packed D2H and Dataset persistence')
        save(args.output/'status.json', report)
        monitor.stage = 'execution'
        before_cpu = monitor.process.cpu_times()
        before_io = monitor.process.io_counters()
        cp.cuda.Device(0).synchronize()
        begin = time.perf_counter()
        completed = calculation.run(backend=cp, debug=args.debug)
        cp.cuda.Device(0).synchronize()
        wall = time.perf_counter()-begin
        monitor.stage = 'verify_report'
        after_cpu, after_io = monitor.process.cpu_times(), monitor.process.io_counters()
        execution_report = completed.report
        assert execution_report['postprocess_backend'] == 'gpu'
        device_report = execution_report['device']
        assert device_report['output_postprocess_calls'] == 80
        expected_bytes = int(np.prod(plan.output_storage_shape))*8*80
        assert device_report['downloaded_bytes'] == expected_bytes
        assert len(completed.value) == 1
        report.update(execution_seconds=wall, numerical_execution=True,
            cpu_user_seconds=after_cpu.user-before_cpu.user,
            cpu_system_seconds=after_cpu.system-before_cpu.system,
            process_read_bytes=after_io.read_bytes-before_io.read_bytes,
            process_write_bytes=after_io.write_bytes-before_io.write_bytes,
            execution_report=execution_report, published_outputs=completed.value,
            average_seconds_per_pair=wall/80, expected_packed_output_bytes=expected_bytes,
            timing_scope='input preparation, contraction, GPU output assembly and packing, final D2H, Dataset write and publication',
            verification_scope='GPU-route execution and exact transfer counters; production scientific cross-reference still required',
            status='completed')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if monitor is not None:
            monitor.finish()
            report['hardware_samples'] = monitor.summary()
        report.update(total_seconds=time.perf_counter()-started,
            process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(args.output/'execution-report.json', report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('execution_report','outputs','reader_roles','traceback')}, default=str), flush=True)


if __name__ == '__main__':
    main()
