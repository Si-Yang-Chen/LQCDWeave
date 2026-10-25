"""Produce HSV/HSD together from actual Q-projected point sources.

H is stored as a native P endpoint with its complete projector provenance.
Only the directory/display name changes; no sampling weight enters a PR.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import numpy as np

from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle
from lqcdweave.contracts.propagator import PropagatorEndpoint
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.perambulator.api import (
    HighmodeBasis, ModePartition, PerambulatorGenerator, PointBasis, VectorBasis,
    PropagatorReader,
    propagator_pair_spec,
)
from lqcdweave.production.perambulator.highmode import sample_high_points
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset

PRODUCT_NAMES = {'PSV': 'H64SV100', 'PSD': 'H64SD40'}
GAMMA5 = np.diag([1., 1., -1., -1.])


def make_generator(manifest, gauge, v, d, high, backend, *, source_times=None):
    sizes = tuple(manifest['lattice']); par = manifest['dirac']
    selection = TimeSelectionRecord(sizes[3], tuple(range(sizes[3])) if source_times is None else tuple(source_times))
    return PerambulatorGenerator(sizes, gauge, mass=par['mass'], tol=par['tol'], maxiter=par['maxiter'],
        xi_0=par['xi_0'], nu=par['nu'], clover_coeff_t=par['clover_coeff_t'],
        clover_coeff_r=par['clover_coeff_r'], t_boundary=par['t_boundary'], multigrid=par['multigrid'],
        source_selection=selection, backend=backend, MRHS=False, rhs_batch_size=1,
        stream_eigenvectors=True, sink_ne_block_size=32, sink_time_block_size=min(4, sizes[3]),
        sink_cache_limit_bytes=int(sizes[3]*(v.Ne+d.Ne)*np.prod(sizes[:3])*3*16),
        gamma5=GAMMA5, gamma5_hermitian=True, measure=True)


def specifications(generator, configuration, source_basis, sink_bases, dirac):
    info = generator.describe()
    high = generator.endpoint(source_basis, source=True)
    sink_v, sink_d = (generator.endpoint(basis) for basis in sink_bases)
    pairs = [(h, t) for h in generator.source_times for t in generator.sink_times]
    return {product: propagator_pair_spec(product, configuration=configuration, sink=high,
        source=endpoint, time_pairs=pairs, gauge_id=info['gauge_id'], dirac=dirac,
        gamma5=GAMMA5, spin_basis='QUDA_DEGRAND_ROSSI_GAMMA_BASIS', gamma5_hermitian=True)
        for product, endpoint in (('PSV', sink_v), ('PSD', sink_d))}


def save_source(writers, specs, h_time, results, *, array_module=np):
    reverse = {result.source.kind + 'S' + result.sink.kind: result for result in results}
    for product, writer in writers.items():
        result = reverse[product]
        pairs = specs[product].coordinates('time_pair')
        for index in writer.missing:
            h, t = pairs[index]
            if h != h_time:
                continue
            block = result.reverse(t)
            if (block.product, block.sink_time, block.source_time) != (product, h_time, t):
                raise ValueError('reverse result differs from the requested H/vector physical pair')
            host = np.asarray(block.values) if array_module is np else array_module.asnumpy(block.values)
            writer.write(index, host)


def reserve_writers(stack, output, cfg, specs, provenance, campaign):
    import fcntl
    import shutil
    output.mkdir(parents=True, exist_ok=True)
    pending = {}
    for product, spec in specs.items():
        directory = output/PRODUCT_NAMES[product]
        if directory.is_symlink():
            raise ValueError('H products require independent native Dataset directories')
        if inspect_dataset(directory, cfg=cfg).get('state') == 'complete':
            with DatasetReader(directory, cfg=cfg) as reader:
                if reader.spec.document != spec.document:
                    raise ValueError('published H product has a different scientific definition')
        else:
            pending[product] = spec
    with (campaign/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        required = 0
        for product, spec in pending.items():
            path = output/PRODUCT_NAMES[product]/'.state'/(cfg+'.npy')
            already = path.stat().st_blocks*512 if path.exists() else 0
            required += max(0, int(np.prod(spec.shape))*16+4096-already)
        if shutil.disk_usage(output).free < required + 64*1024**3:
            raise RuntimeError('insufficient space for remaining H payload plus 64 GiB reserve')
        writers = {}
        for product, spec in pending.items():
            directory = output/PRODUCT_NAMES[product]
            writer = stack.enter_context(DatasetWriter(directory, spec, provenance=provenance))
            path = directory/'.state'/(cfg+'.npy')
            with path.open('r+b') as stream:
                os.posix_fallocate(stream.fileno(), 0, os.fstat(stream.fileno()).st_size)
            writers[product] = writer
        return writers


def verify_real_reverse(generator, results, v, d, high, point_basis, backend, manifest, cfg):
    """Four independent reverse-direction QUDA solves, never a round trip."""
    xp = backend.array_module; par = manifest['dirac']; lt = manifest['lattice'][3]
    t = generator.sink_times[1] if len(generator.sink_times) > 1 else generator.sink_times[0]
    h_time = generator.source_times[0]
    parent = xp.asarray(high.read_modes(cfg, h_time, 0, high.Ne))
    point_slots = point_basis.handle.load(cfg)
    mapping = getattr(point_basis.handle, 'time_selection', None)
    point_slot = h_time if mapping is None else mapping.slot(h_time)
    points = xp.asarray(point_slots[point_slot, :point_basis.count])
    actual = {result.source.kind + 'S' + result.sink.kind: result for result in results}
    rhs = xp.zeros_like(generator._rhs[:1]); solution = xp.empty_like(rhs)
    errors = []
    for product, handle in (('PSV', v), ('PSD', d)):
        vector = xp.asarray(handle.read_modes(cfg, t, 0, 1)[0])
        for spin in (0, 2):
            rhs.fill(0); rhs[0, t, ..., spin, :] = vector
            backend.solve_into(rhs, solution, tol=par['tol'], maxiter=par['maxiter'])
            expected = sample_high_points(solution[:, h_time], points, parent, array_module=xp)[0, 0]
            observed = actual[product].reverse(t).values[:, spin, 0, :, 0]
            error = float(xp.linalg.norm(observed-expected)/max(float(xp.linalg.norm(expected)), np.finfo(float).tiny))
            errors.append(dict(product=product,source_spin=spin,relative_error=error))
            if not np.isfinite(error) or error > max(100*par['tol'], 1e-10):
                raise RuntimeError(f'independent H gamma5 reverse failed: {errors[-1]}')
    return errors


def run(config, cfg, campaign, *, pilot=False):
    from tools.campaigns.partner_point_production import (
        LegacyBasisHandle, _GaugeHandle, _dirac_definition, _initialize_quda, CheckedQudaBackend,
    )
    campaign = Path(campaign); cfg = str(cfg)
    if cfg not in [str(key) for key in config['cfgs']]:
        raise ValueError('configuration outside the approved input list')
    manifest = json.loads(Path(config['partner_manifest']).read_text())
    core = Path(config['core']); output = campaign/'pilot-data' if pilot else core/'4.propegater'
    began = time.monotonic(); source_reports = []
    mpi, native, runtime = _initialize_quda(manifest, campaign/'quda-cache'/cfg)
    backend = CheckedQudaBackend(tolerance=manifest['dirac']['tol'])
    try:
        with ExitStack() as owners:
            parent_data = owners.enter_context(DatasetReader(core/'1.eig_vector428', cfg=cfg))
            d_data = owners.enter_context(DatasetReader(core/'1.noise_vector40', cfg=cfg))
            p_data = owners.enter_context(DatasetReader(core/'1.sparsen_point64', cfg=cfg))
            template = owners.enter_context(DatasetReader(core/'4.propegater'/'P64SV100', cfg=cfg))
            parent = DatasetFieldHandle(parent_data); d = DatasetFieldHandle(d_data); p = DatasetPointHandle(p_data)
            v = LegacyBasisHandle(manifest, cfg); owners.callback(v.close)
            source = d.dilution_source
            if (parent.Ne != 428 or v.Ne != 100 or d.Ne != 40 or p.Np != 64
                    or tuple(parent.mode_labels) != tuple(range(428))
                    or source.parent_basis_id != parent.basis_id
                    or tuple(g.m for g in source.groups) != (4,)*10):
                raise ValueError('actual V100/D40/P64/full-parent inputs differ')
            partition = ModePartition(parent.basis_id, tuple(range(100)), tuple(g.parent_modes for g in source.groups))
            if partition.projection_modes != tuple(range(428)):
                raise ValueError('H must subtract the complete ordered 428-dimensional parent')
            if np.max(np.abs(v.read_modes(cfg,0,0,100)-parent.read_modes(cfg,0,0,100))) > 1e-12:
                raise ValueError('V100 differs from the retained actual parent prefix')
            definition = template.spec.document['definition']
            if (definition['source_endpoint']['data_id'] != v.basis_id
                    or definition['sink_endpoint']['data_id'] != p.basis_id):
                raise ValueError('actual historical V/point identities differ')
            high = HighmodeBasis(parent, partition)
            point_basis = PointBasis(p, count=1 if pilot else p.Np,
                                     highmode_basis=high)
            sink_bases = [VectorBasis(v, count=v.Ne), VectorBasis(d, count=d.Ne)]
            generator = make_generator(manifest, _GaugeHandle(manifest), v, d, high, backend,
                                       source_times=(0,) if pilot else None)
            generator.load(cfg)
            sm = manifest['dirac']['stout']; generator.stout_smear(sm['nstep'],sm['rho'],sm['dir_ignore'])
            if generator.describe()['gauge_id'] != definition['gauge_id']:
                raise ValueError('actual Dirac gauge differs from existing inputs')
            specs = specifications(generator, template.spec.document['configuration'],
                                   point_basis, sink_bases, _dirac_definition(manifest))
            provenance = dict(method='native-high-point-source-shared-VD-sink-gamma5-v1',
                display_names=PRODUCT_NAMES, parent_basis_id=parent.basis_id, point_data_id=p.basis_id,
                d_data_id=d.basis_id, v_basis_id=v.basis_id, solver=manifest['dirac'], normalization='raw',
                inversion_source='H=Q428 coordinate delta', saved_coordinates='H sink, V/D source',
                highmode_stream='one complete physical source-time slice', pilot=pilot)
            # Pilot retains full 72-time V/D sink cache and reserves the missing
            # P63 main-output capacity. Budget reverse-write temporaries separately.
            capacity_probe = None
            if pilot:
                capacity_probe = backend.array_module.empty((len(generator.sink_times),4,4,v.Ne+d.Ne,63,3),dtype=complex)
            with ExitStack() as storage:
                writers = reserve_writers(storage, output, cfg, specs, provenance, campaign)
                for h_time in generator.source_times:
                    needed = any(any(specs[pr].coordinates('time_pair')[i][0] == h_time for i in writer.missing)
                                 for pr,writer in writers.items())
                    if not needed:
                        continue
                    start = time.monotonic(); count = len(backend.solve_records)
                    results = generator.calc(h_time, src=point_basis, sink=sink_bases)
                    production_solves = len(backend.solve_records)-count
                    if production_solves != (1 if pilot else 64)*12:
                        raise RuntimeError('unexpected number of shared H-source solves')
                    checks = (verify_real_reverse(generator, results, v, d, high,
                                                  point_basis, backend, manifest, cfg)
                              if pilot else [])
                    save_source(writers,specs,h_time,results,
                                array_module=backend.array_module)
                    record = dict(event='source_complete',cfg=cfg,inversion_source_time=h_time,
                        rhs_solves=production_solves,seconds=time.monotonic()-start,
                        max_true_residual=max(row['true_residual'] for row in backend.solve_records[count:]),
                        independent_reverse_checks=checks,metrics=generator.last_metrics)
                    source_reports.append(record); print(json.dumps(record),flush=True)
                    del results
                published = {pr: writer.publish() for pr,writer in writers.items()}
            with ExitStack() as readers:
                saved = {pr: readers.enter_context(DatasetReader(output/name,cfg=cfg)) for pr,name in PRODUCT_NAMES.items()}
                consumer = PropagatorReader(tuple(saved.values()))
                for pr,reader in saved.items():
                    if reader.spec.document != specs[pr].document:
                        raise ValueError('published H definition differs')
                    de = reader.spec.document['definition']
                    sink = PropagatorEndpoint.from_document(de['sink_endpoint'])
                    source_ep = PropagatorEndpoint.from_document(de['source_endpoint'])
                    for t,s in specs[pr].coordinates('time_pair')[:1]:
                        direct = consumer.read(pr,t,s,sink=sink,source=source_ep)
                        reverse = consumer.read(pr[2]+'SP',s,t,sink=source_ep,source=sink)
                        if not np.isfinite(direct.values).all() or not np.isfinite(reverse.values).all():
                            raise ValueError('nonfinite published H product')
            free,total = backend.array_module.cuda.runtime.memGetInfo()
            report = dict(status='complete',cfg=cfg,pilot=pilot,seconds=time.monotonic()-began,
                data_ids={pr:r.data_id for pr,r in saved.items()},rhs_solves=len(backend.solve_records),
                sources=source_reports,runtime=runtime,device_memory_free=int(free),device_memory_total=int(total),
                highmode_partition=partition.document)
            directory = campaign/('pilot-reports' if pilot else 'reports'); directory.mkdir(exist_ok=True)
            (directory/(cfg+'.json')).write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(event='configuration_complete',cfg=cfg,pilot=pilot,rhs_solves=report['rhs_solves'])),flush=True)
            return report
    finally:
        backend.destroy()


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('terminated; retaining partial H rows')))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path); parser.add_argument('--cfg'); parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args(argv); config=json.loads(args.config.read_text())
    cfg=args.cfg or config['cfgs'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    run(config,cfg,args.config.parent,pilot=args.pilot)


if __name__ == '__main__':
    main()
