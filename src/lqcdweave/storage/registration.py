"""One shared registration document for all configurations of a dataset."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import time

from lqcdweave.contracts.dataset import canonical_json
from .metadata import write_json_document

FORMAT = 'easy-distillation.dataset'
VERSION = 1


def configuration_key(cfg):
    value = '_unknown' if cfg is None else str(cfg)
    if not value or value in ('.', '..') or any(c in value for c in '/\\\0'):
        raise ValueError('configuration must be a nonempty filename component')
    return value


def _equal(left, right):
    """Canonical JSON equality without serializing every nested subtree."""
    kind = type(left)
    if kind is not type(right):
        # Public registration normalizes JSON first; keep tuple/list and other
        # json-encodable inputs compatible at internal factoring boundaries.
        if kind not in (dict, list, str, int, float, bool, type(None)) or type(right) not in (
                dict, list, str, int, float, bool, type(None)):
            return canonical_json(left) == canonical_json(right)
        return False
    if kind is dict:
        return left.keys() == right.keys() and all(_equal(value, right[key])
                                                  for key, value in left.items())
    if kind is list:
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if kind is float:
        return left == right and (left != 0 or math.copysign(1, left) == math.copysign(1, right))
    if kind in (str, int, bool, type(None)):
        return left == right
    return canonical_json(left) == canonical_json(right)


def _common(values):
    first = values[0]
    if all(_equal(first, value) for value in values[1:]):
        return first
    if all(type(value) is dict for value in values):
        keys = set(first).intersection(*(set(value) for value in values[1:]))
        return {key: _common([value[key] for value in values]) for key in first if key in keys}
    if (all(type(value) is list and len(value) == len(first) for value in values)
            and first and all(type(item) is dict for value in values for item in value)):
        return [_common([value[i] for value in values]) for i in range(len(first))]
    return None


def _difference(common, value, path=()):
    if _equal(common, value):
        return []
    if type(common) is dict and type(value) is dict:
        result = []
        for key, item in value.items():
            result.extend(_difference(common[key], item, (*path, key)) if key in common
                          else [[list((*path, key)), item]])
        return result
    if type(common) is list and type(value) is list and len(common) == len(value):
        result = []
        for i, item in enumerate(value):
            result.extend(_difference(common[i], item, (*path, i)))
        return result
    return [[list(path), value]]


def _restore(common, changes):
    value = json.loads(canonical_json(common))
    for path, item in changes:
        if not path:
            value = item
            continue
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = item
    return value


def read_registration_document(directory):
    """Read the compressed table without expanding unrelated configurations."""
    path = Path(directory) / 'dataset.json'
    if not path.exists():
        return dict(format=FORMAT, version=VERSION, common={}, configurations={})
    document = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(document, dict) or document.get('format') != FORMAT
            or document.get('version') != VERSION
            or set(document) != {'format', 'version', 'common', 'configurations'}):
        raise ValueError('invalid dataset registration format')
    if not isinstance(document['configurations'], dict):
        raise ValueError('configuration registrations must be a mapping')
    return document


def read_registration(directory):
    document = read_registration_document(directory)
    return {cfg: _restore(document['common'], changes)
            for cfg, changes in document['configurations'].items()}


@contextmanager
def registration_lock(directory):
    state = Path(directory) / '.state'
    path = state / 'registration.lock'
    # Even subsecond appends can queue behind dozens of independent workers.
    # Keep stale locks bounded without treating a normal burst as a failure.
    deadline = time.monotonic() + 60
    delay = .01
    while True:
        try:
            state.mkdir(parents=True, exist_ok=True)
            stream = path.open('x', encoding='ascii')
            break
        except (FileExistsError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise FileExistsError('dataset registration is locked; check the owning process')
            time.sleep(delay)
            delay = min(.1, delay * 2)
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)
        try:
            state.rmdir()
        except OSError:
            pass


def write_registration(directory, records):
    """Caller holds registration_lock; common scientific definitions occur once."""
    values = list(records.values())
    common = _common(values) if values else {}
    document = dict(format=FORMAT, version=VERSION, common=common,
        configurations={cfg: _difference(common, record) for cfg, record in records.items()})
    write_json_document(Path(directory) / 'dataset.json', document)


def _rebase_changes(baseline, changes):
    """Overlay old cfg differences on the parts removed from the common record."""
    result = list(changes)
    for path, value in baseline:
        # An existing cfg replacement at or above this path already contains
        # its complete value. Otherwise restore only the affected subtree.
        if any(path[:len(previous)] == previous for previous, _ in changes):
            continue
        descendants = [(previous[len(path):], item) for previous, item in changes
                       if previous[:len(path)] == path]
        if descendants:
            value = _restore(value, descendants)
            result = [[previous, item] for previous, item in result
                      if previous[:len(path)] != path]
        result.append([path, value])
    return result


def append_registration(directory, document, cfg, record):
    """Caller holds registration_lock; append one record to the v1 table.

    Keep old cfg differences compressed. If the common record shrinks, rebase
    only its changed subtrees, rather than expanding and refactoring all cfgs.
    """
    key = configuration_key(cfg)
    configurations = document['configurations']
    if key in configurations:
        raise FileExistsError(f'configuration {key} is already registered')
    if configurations:
        previous = document['common']
        common = _common([previous, record])
        baseline = _difference(common, previous)
        configurations = ({cfg: _rebase_changes(baseline, changes)
                           for cfg, changes in configurations.items()} if baseline
                          else dict(configurations))
    else:
        common = record
        configurations = {}
    configurations[key] = _difference(common, record)
    write_json_document(Path(directory) / 'dataset.json',
        dict(format=FORMAT, version=VERSION, common=common, configurations=configurations))


def register_record(directory, cfg, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = configuration_key(cfg)
    snapshot = json.loads(canonical_json(record))
    with registration_lock(directory):
        append_registration(directory, read_registration_document(directory), key, snapshot)
    return record['data_id']


def selected_record(directory, cfg=None):
    document = read_registration_document(directory)
    configurations = document['configurations']
    if not configurations:
        raise FileNotFoundError('no complete dataset registration')
    if cfg is None and len(configurations) == 1:
        key = next(iter(configurations))
    else:
        key = configuration_key(cfg)
    if key not in configurations:
        raise KeyError(f'configuration {key} is not registered')
    return key, _restore(document['common'], configurations[key])
