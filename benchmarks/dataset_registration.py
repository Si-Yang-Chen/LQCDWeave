"""Measure atomic cfg registration using a read-only real metadata snapshot.

All writes go to a temporary directory under --scratch, never to --snapshot.
No numerical payload is opened. Optional baseline uses the original source.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import threading
import time

from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.storage import metadata, registration


def _load_reference(path):
    spec = importlib.util.spec_from_file_location('lqcdweave.storage._reference_registration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify(directory, expected, common):
    document = registration.read_registration_document(directory)
    if canonical_json(document['common']) != canonical_json(common):
        raise AssertionError('common metadata changed unexpectedly')
    for cfg, changes in document['configurations'].items():
        actual = registration._restore(document['common'], changes)
        if canonical_json(actual) != canonical_json(expected[cfg]):
            raise AssertionError(f'configuration {cfg} metadata differs')
    if set(document['configurations']) != set(expected):
        raise AssertionError('configuration registrations were lost')


def measure(snapshot, scratch, repeats, concurrent, reference=None):
    snapshot = snapshot.resolve()
    source = snapshot.read_bytes()
    document = json.loads(source)
    expected = {cfg: registration._restore(document['common'], changes)
                for cfg, changes in document['configurations'].items()}
    if not expected:
        raise ValueError('the snapshot must contain registered configurations')
    sample = next(iter(expected.values()))
    source_hashes = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for name, module in [('registration', registration), ('metadata', metadata)]}
    if reference is not None:
        source_hashes['reference_registration'] = hashlib.sha256(reference.read_bytes()).hexdigest()
    # Reuse only the metadata structure; these scratch entries are not claimed
    # as new scientific data or registered in a user DatasetIndex.
    scratch.mkdir(parents=True, exist_ok=True)
    timings = {}
    with tempfile.TemporaryDirectory(prefix='registration-', dir=scratch) as temporary:
        root = Path(temporary)
        implementations = [('candidate', registration, repeats)]
        if reference is not None:
            implementations.insert(0, ('reference', _load_reference(reference), 1))
        for name, module, count in implementations:
            values = []
            for index in range(count):
                directory = root / f'{name}-{index}'
                directory.mkdir()
                shutil.copyfile(snapshot, directory / 'dataset.json')
                cfg = f'benchmark-{index}'
                while cfg in expected:
                    cfg += '-new'
                started = time.perf_counter()
                module.register_record(directory, cfg, sample)
                values.append(time.perf_counter() - started)
                _verify(directory, {**expected, cfg: sample}, document['common'])
            timings[name] = dict(seconds=values, max_seconds=max(values),
                                 median_seconds=statistics.median(values))
            print(json.dumps(dict(stage=name, **timings[name])), flush=True)
        if concurrent:
            directory = root / 'concurrent'
            directory.mkdir()
            shutil.copyfile(snapshot, directory / 'dataset.json')
            barrier = threading.Barrier(concurrent)
            lock_seconds = []
            original_append = registration.append_registration

            def timed_append(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_append(*args, **kwargs)
                finally:
                    lock_seconds.append(time.perf_counter() - started)

            def publish(index):
                cfg = f'concurrent-benchmark-{index}'
                if cfg in expected:
                    raise ValueError('benchmark cfg already exists')
                barrier.wait()
                started = time.perf_counter()
                error = None
                try:
                    registration.register_record(directory, cfg, sample)
                except Exception as exception:
                    error = f'{type(exception).__name__}: {exception}'
                return cfg, time.perf_counter() - started, error

            registration.append_registration = timed_append
            try:
                with ThreadPoolExecutor(max_workers=concurrent) as pool:
                    outcomes = list(pool.map(publish, range(concurrent)))
            finally:
                registration.append_registration = original_append
            successful = {cfg: sample for cfg, _, error in outcomes if error is None}
            _verify(directory, {**expected, **successful}, document['common'])
            timings['concurrent'] = dict(workers=concurrent,
                completed=len(successful), failures={cfg: error for cfg, _, error in outcomes if error},
                max_elapsed_seconds=max(seconds for _, seconds, _ in outcomes),
                max_append_seconds=max(lock_seconds, default=0))
    if snapshot.read_bytes() != source:
        raise AssertionError('source snapshot changed during the audit')
    return dict(host=platform.node(), python=platform.python_version(),
        snapshot=str(snapshot), snapshot_sha256=hashlib.sha256(source).hexdigest(),
        source_sha256=source_hashes,
        registered_cfgs=len(expected), snapshot_bytes=len(source),
        record_bytes=len(canonical_json(sample).encode('utf-8')),
        timing_scope='register_record including normalization, lock, read, factoring, atomic write and fsync; no payload IO',
        verified='all old and appended metadata exactly matches the source records',
        timings=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--reference-registration', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrent', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=1.0)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrent < 0:
        parser.error('repeats must be positive and concurrency nonnegative')
    report = measure(args.snapshot, args.scratch, args.repeats, args.concurrent,
                     args.reference_registration)
    report['passed'] = (report['timings']['candidate']['max_seconds'] < args.max_seconds
        and not report['timings'].get('concurrent', {}).get('failures'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
