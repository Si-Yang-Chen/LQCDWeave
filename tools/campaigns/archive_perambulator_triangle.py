"""Archive explicitly selected legacy rank slabs with the native pair Dataset IO.

This is an archival campaign, not a producer. Both products of a configuration
must pass a complete read-back before its original numerical files are removed.
The input JSON is an explicit deletion allowlist; no file discovery is
used to expand that allowlist during cleanup.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from lqcdweave.contracts.dataset import canonical_json, stable_digest
from lqcdweave.contracts.propagator import PropagatorEndpoint
from lqcdweave.production.perambulator import PropagatorReader, propagator_pair_spec
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected a regular, non-symlink file: {path}')
    info = path.stat()
    if info.st_nlink != 1:
        raise ValueError(f'archive requires independently owned files: {path}')
    return dict(path=str(path.resolve()), size=info.st_size, mtime_ns=info.st_mtime_ns,
                device=info.st_dev, inode=info.st_ino)


def digest_file(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def check_running_references(source_root):
    """Refuse a newly submitted consumer of the historical source directory."""
    queue = subprocess.run(['squeue', '-h', '-u', 'siyangchen', '-o', '%i|%Z'],
                           capture_output=True, text=True, check=True, timeout=60)
    seen = set()
    for line in queue.stdout.splitlines():
        job, workdir = line.split('|', 1)
        if job == os.environ.get('SLURM_JOB_ID') or workdir in seen:
            continue
        seen.add(workdir)
        record = subprocess.run(['scontrol', 'show', 'job', job], capture_output=True,
                                text=True, check=True, timeout=60).stdout
        roots = [Path(workdir)]
        if Path(workdir).name.startswith('source-'):
            roots.append(Path(workdir).parent)
        candidates = []
        command = re.search(r'\bCommand=(\S+)', record)
        if command and Path(command.group(1)).is_file():
            candidates.append(Path(command.group(1)))
        for root in roots:
            if root.is_dir():
                candidates.extend(p for p in root.iterdir()
                                  if p.is_file() and p.suffix in ('.json', '.slurm', '.sh'))
        for path in candidates:
            if path.stat().st_size > 2*1024**2:
                raise ValueError(f'cannot audit oversized active task definition: {path}')
            if Path(source_root).parent.name in path.read_text(encoding='utf-8', errors='replace'):
                raise ValueError(f'active task {job} references the legacy data: {path}')


def reserve_file(path):
    """Reserve actual filesystem blocks without changing existing data."""
    if not hasattr(os, 'posix_fallocate'):
        raise RuntimeError('this archival worker requires POSIX allocation support')
    with Path(path).open('r+b') as stream:
        os.posix_fallocate(stream.fileno(), 0, os.fstat(stream.fileno()).st_size)
        os.fsync(stream.fileno())


@contextmanager
def legacy_row(records, source_time, *, local_lt, tail):
    arrays = []
    try:
        for record in records:
            array = np.load(record['path'], mmap_mode='r', allow_pickle=False)
            arrays.append(array)
            if array.shape != (local_lt, *tail) or array.dtype != np.dtype('complex128'):
                raise ValueError(f'legacy shape/dtype mismatch: {record["path"]}')
        yield lambda t: arrays[t // local_lt][t % local_lt]
    finally:
        for array in arrays:
            array._mmap.close()


def definition(config, cfg, product, entry):
    manifest = entry['manifest']
    lattice = manifest['global_lattice']
    lt = lattice[3]
    if (manifest['source_times'] != list(range(lt))
            or manifest['time_convention'] != 'absolute-global-source-and-sink'
            or manifest['grid_size'][:3] != [1, 1, 1]):
        raise ValueError('only complete absolute-time, temporal rank slabs are supported')
    local_lt = manifest['local_lattice'][3]
    ranks = manifest['grid_size'][3]
    if local_lt * ranks != lt or len(entry['sources']) != lt * ranks:
        raise ValueError('incomplete physical source/sink coverage')
    for i, record in enumerate(entry['sources']):
        expected = Path(config['source_root']) / product / f'{cfg}.t{i//ranks:03d}.rank{i%ranks:04d}.npy'
        if Path(record['path']) != expected.resolve():
            raise ValueError('source rows do not match physical source/rank order')
    if product not in ('VSV', 'PSP'):
        raise ValueError('only explicitly authorized VSV and PSP can be archived')
    n = manifest['Ne'] if product == 'VSV' else manifest['np_snk']
    if product == 'PSP' and n != manifest['np_src']:
        raise ValueError('triangle requires the same point endpoints')
    tail = (4, 4, n, n) if product == 'VSV' else (4, 4, n, 3, n, 3)
    # Archival import identity belongs only to this historical source. It never
    # asserts phase/order equivalence with another basis of the same dimension.
    endpoint = PropagatorEndpoint(product[0], 'legacy-archive:' + stable_digest(dict(
        root=config['source_root'], cfg=cfg, kind=product[0],
        inputs=entry['metadata']['inputs'], metadata=entry['metadata'])), tuple(range(n)))
    pairs = [(t, s) for s in range(lt) for t in range(s, lt)]
    spec = propagator_pair_spec(product,
        configuration=dict(ensemble=config['ensemble'], configuration=cfg, lattice=lattice),
        sink=endpoint, source=endpoint, time_pairs=pairs,
        gauge_id='legacy-gauge:' + stable_digest(entry['metadata']['inputs']['gauge']),
        # Product, solver MRHS, tail layout and file paths are provenance,
        # not differences in the Dirac action of VSV versus PSP.
        dirac=dict(action='legacy_wilson_clover', parameters={key: manifest[key]
                   for key in ('mass', 'clover', 't_boundary', 'stout_steps', 'stout_rho')
                   if key in manifest}),
        gamma5=np.diag(config['gamma5_diagonal']).astype(np.complex128),
        spin_basis=config['spin_basis'], gamma5_hermitian=True)
    return spec, endpoint, tail, local_lt, ranks


def copy_product(config, cfg, product, entry, output):
    spec, endpoint, tail, local_lt, ranks = definition(config, cfg, product, entry)
    provenance = dict(operation='legacy-temporal-triangle-archive',
                      legacy_manifest=entry['manifest'], legacy_metadata=entry['metadata'],
                      sources=[dict(path=r['path']) for r in entry['sources']],
                      migration_id=config['migration_id'])
    observed = inspect_dataset(output, cfg=cfg)
    if observed['state'] == 'complete':
        with DatasetReader(output, cfg=cfg) as reader:
            if (reader.spec.document != spec.document or reader.provenance != provenance):
                raise ValueError('published archive definition/provenance mismatch')
        return
    positions = {tuple(pair): i for i, pair in enumerate(spec.coordinates('time_pair'))}
    with DatasetWriter(output, spec, provenance=provenance, resume=True) as writer:
        # Reserve before the first numerical write, including on a resumed run.
        reserve_file(Path(output) / '.state' / (cfg + '.npy'))
        missing = set(writer.missing)
        for s in range(spec.document['configuration']['lattice'][3]):
            with legacy_row(entry['sources'][s*ranks:(s+1)*ranks], s,
                            local_lt=local_lt, tail=tail) as read:
                for t in range(s, spec.document['configuration']['lattice'][3]):
                    position = positions[t, s]
                    if position in missing:
                        writer.write(position, read(t))
            print(json.dumps(dict(event='copy', cfg=cfg, product=product, source=s)), flush=True)
        writer.publish()


def verify_product(config, cfg, product, entry, output):
    spec, endpoint, tail, local_lt, ranks = definition(config, cfg, product, entry)
    lt = spec.document['configuration']['lattice'][3]
    maximum = relative = error2 = norm2 = 0.0
    exact = recovered = 0
    with DatasetReader(output, cfg=cfg) as dataset:
        access = PropagatorReader([dataset])
        for s in range(lt):
            with legacy_row(entry['sources'][s*ranks:(s+1)*ranks], s,
                            local_lt=local_lt, tail=tail) as read:
                for t in range(lt):
                    original = read(t)
                    actual = access.read(product, t, s, sink=endpoint, source=endpoint).values
                    if not np.isfinite(original).all() or not np.isfinite(actual).all():
                        raise ValueError(f'nonfinite data: {cfg} {product} {t},{s}')
                    if t >= s:
                        if not np.array_equal(actual.view(np.uint64), original.view(np.uint64)):
                            raise ValueError(f'archive copy differs: {cfg} {product} {t},{s}')
                        exact += 1
                    else:
                        delta = actual - original
                        err = float(np.vdot(delta, delta).real)
                        norm = float(np.vdot(original, original).real)
                        error2 += err
                        norm2 += norm
                        maximum = max(maximum, float(np.max(np.abs(delta))))
                        relative = max(relative, math.sqrt(err / norm) if norm else 0.)
                        if not np.allclose(actual, original, rtol=config['rtol'], atol=config['atol']):
                            raise ValueError(f'gamma5 recovery failed: {cfg} {product} {t},{s}; maxabs={maximum}')
                        recovered += 1
            print(json.dumps(dict(event='verify', cfg=cfg, product=product, source=s)), flush=True)
        result = dict(product=product, data_id=dataset.data_id,
                      payload=str(Path(output) / (cfg+'.npy')), exact_pairs=exact,
                      recovered_pairs=recovered, max_absolute_error=maximum,
                      max_pair_relative_error=relative,
                      global_relative_error=math.sqrt(error2/norm2) if norm2 else 0.)
    # Hash the closed, published file. A resumed cleanup rechecks this hash.
    for path in (Path(result['payload']), Path(output) / 'dataset.json'):
        with path.open('r+b') as stream:
            os.fsync(stream.fileno())
    if os.name == 'posix':
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    result['payload_sha256'] = digest_file(result['payload'])
    return result


def cleanup(config, cfg, receipt, receipt_path):
    if receipt['config_digest'] != stable_digest(config) or receipt['cfg'] != cfg:
        raise ValueError('cleanup receipt does not belong to the audited campaign')
    if receipt['status'] not in ('verified', 'deleted') or set(receipt['products']) != {'VSV', 'PSP'}:
        raise ValueError('both products must be fully verified before cleanup')
    base = Path(config['source_root']).resolve()
    expected = config['configurations'][cfg]
    # Never trust target paths from a receipt instead of the audited allowlist.
    records = [r for product in ('VSV','PSP') for r in expected[product]['sources']]
    for product in ('VSV', 'PSP'):
        for record in expected[product]['sources']:
            path = Path(record['path'])
            if (path.parent != base / product or path.resolve() != path
                    or not re.fullmatch(re.escape(cfg)+r'\.t\d{3}\.rank\d{4}\.npy', path.name)):
                raise ValueError(f'cleanup target escapes the authorized configuration: {path}')
    for product, result in receipt['products'].items():
        lt = expected[product]['manifest']['global_lattice'][3]
        if (result['exact_pairs'] != lt*(lt+1)//2
                or result['recovered_pairs'] != lt*(lt-1)//2):
            raise ValueError('cleanup requires complete direct and reverse verification')
        expected_payload = Path(config['output_root']).resolve() / product / (cfg+'.npy')
        if Path(result['payload']).resolve() != expected_payload:
            raise ValueError('archive payload path differs from the allowlist')
        if digest_file(expected_payload) != result['payload_sha256']:
            raise ValueError('archive changed since verification; originals preserved')
        with DatasetReader(expected_payload.parent, cfg=cfg) as reader:
            if reader.data_id != result['data_id']:
                raise ValueError('archive registration changed since verification')
    if config.get('check_slurm_references'):
        check_running_references(config['source_root'])
    # Only individual audited numerical files are unlinked. No recursive delete,
    # no glob expansion, no deletion of manifests, metadata, PSV or other cfgs.
    for record in records:
        path = Path(record['path'])
        if path.exists():
            path.unlink()
    receipt['status'] = 'deleted'
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(receipt_path, receipt)


def run_configuration(config, cfg, *, delete=False):
    destination = Path(config['output_root'])
    receipt_path = destination / 'receipts' / (cfg+'.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if delete:
            cleanup(config, cfg, receipt, receipt_path)
        return receipt
    entries = config['configurations'][cfg]
    if set(entries) != {'VSV', 'PSP'}:
        raise ValueError('each configuration must explicitly include both products')
    destination.mkdir(parents=True, exist_ok=True)
    for product in ('VSV','PSP'):
        copy_product(config, cfg, product, entries[product], destination / product)
    results = {product: verify_product(config, cfg, product, entries[product], destination / product)
               for product in ('VSV','PSP')}
    receipt = dict(status='verified', cfg=cfg, config_digest=stable_digest(config),
                   criteria=dict(rtol=config['rtol'], atol=config['atol'], retained='bitwise exact'),
                   products=results, verified_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt_path, receipt)
    atomic_json(destination / 'campaign.json', config)
    if delete:
        cleanup(config, cfg, receipt, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--delete-verified-originals', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source = Path(config['source_root']).resolve()
    authorized = Path('/public/home/siyangchen/BASE/localized-fulltime-25cfg-20260915-attempt-01/data')
    if source != authorized or str(source) != config['source_root']:
        raise ValueError('this campaign only authorizes the old 25-cfg root')
    if Path(config['output_root']).resolve().is_relative_to(source):
        raise ValueError('archive must be outside the original data directory')
    if len(config['configurations']) != 25:
        raise ValueError('the audited campaign must contain exactly 25 configurations')
    # A POSIX advisory lock permits restart after a terminated Slurm worker and
    # prevents two archival jobs from performing the same migration concurrently.
    import fcntl
    destination = Path(config['output_root'])
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for cfg in config['configurations']:
            check_running_references(config['source_root'])
            receipt = run_configuration(config, cfg, delete=args.delete_verified_originals)
            print(json.dumps(dict(event='configuration_complete', cfg=cfg, status=receipt['status'])), flush=True)
    atomic_json(destination / 'complete.json', dict(config_digest=stable_digest(config),
        configurations=list(config['configurations']), deleted_originals=args.delete_verified_originals))


if __name__ == '__main__':
    main()
