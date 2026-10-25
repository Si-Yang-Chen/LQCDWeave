"""Whole NPY/HDF5 files per cfg, with one shared dataset registration."""
from contextlib import contextmanager
from pathlib import Path
from math import prod
import json
import os
import shutil
import uuid
import threading
import weakref
from urllib.parse import quote

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json, stable_digest
from lqcdweave.contracts.source import external_import_identity, source_origin_from_registration
from .metadata import write_json_document, write_json_records, read_json_records
from .temporary_arrays import TemporaryNpyArrayStore, _npy_array_storage_bytes
from .registration import (configuration_key, read_registration, write_registration,
    registration_lock, register_record, selected_record, read_registration_document,
    append_registration)
from .execution_records import (_execution_record, _published_record, _new_attempt,
    _normalize_attempt_context, _normalize_scientific_validation, _validate_execution,
    _validate_attempt, _utc_now)

_ACTIVE_READERS = weakref.WeakSet()


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError('HDF5 storage requires the hdf5 optional dependency') from error
    return h5py


def _check_array(value, shape, dtype):
    if tuple(value.shape) != tuple(shape) or value.dtype != np.dtype(dtype):
        raise ValueError('array shape/dtype differs from the declared contract')


def _dataset_positions(spec):
    return {axis: {canonical_json(v): i for i, v in enumerate(spec.coordinates(axis))}
            for axis in spec.axis_names}


def _storage_path(directory, storage):
    path = Path(storage['path'])
    return path if path.is_absolute() else Path(directory) / path


def _validate_record(record):
    if not isinstance(record, dict) or not {'identity', 'data_id', 'storage'} <= record.keys():
        raise ValueError('incomplete dataset registration')
    identity = record['identity']
    if identity.get('state') != 'complete' or stable_digest(identity) != record['data_id']:
        raise ValueError('dataset identity or completion state differs')
    spec = DatasetSpec(identity['spec'])
    storage = record['storage']
    if storage['format'] not in ('npy', 'hdf5', 'binary'):
        raise ValueError('unsupported registered data format')
    execution = record.get('execution')
    if execution is not None:
        _validate_execution(execution, spec, data_id=record['data_id'], draft=False,
                            provenance=identity['provenance'])
    origin = source_origin_from_registration(identity['provenance'], execution,
        data_id=record['data_id'], definition_id=spec.definition_id, spec_document=spec.document)
    return dict(spec=spec, data_id=record['data_id'], provenance=identity['provenance'],
        execution=execution, source_origin=origin, storage=storage, identity=identity,
        scientific_validation=({'status': 'unknown'} if execution is None
                               else execution['scientific_validation']),
        registration_runtime=(None if execution is None else
            execution.get('registration_event', {}).get('runtime')))


def read_dataset_metadata(directory, *, cfg=None):
    key, record = selected_record(directory, cfg)
    return dict(_validate_record(record), state='metadata', directory=Path(directory), configuration=key,
                record=record, fields=record.get('fields', {}))


_read_dataset_metadata = read_dataset_metadata


def inspect_dataset(directory, *, cfg=None):
    try:
        metadata = read_dataset_metadata(directory, cfg=cfg)
    except (FileNotFoundError, KeyError):
        state = Path(directory) / '.state'
        candidates = list(state.glob('*.json')) if state.exists() else []
        if cfg is not None:
            candidates = [state / (configuration_key(cfg) + '.json')]
            candidates = [p for p in candidates if p.exists()]
        if not candidates:
            return dict(state='missing', data_id=None, completed=())
        if len(candidates) != 1:
            raise ValueError('select a configuration when inspecting production state')
        record = json.loads(candidates[0].read_text(encoding='utf-8'))
        return dict(state='partial', spec=DatasetSpec(record['spec']), data_id=None,
                    provenance=record['provenance'], completed=tuple(record['completed']),
                    execution=record['execution'], latest_attempt_status=record['execution']['attempts'][-1]['status'])
    return dict(state='complete', spec=metadata['spec'], data_id=metadata['data_id'],
        provenance=metadata['provenance'], completed=tuple(range(metadata['spec'].shape[0] if metadata['spec'].shape else 1)),
        execution=metadata['execution'])


class DatasetReader:
    """One registered cfg; NPY slices borrow a map, HDF5 slices own their data."""
    def __init__(self, directory, *, cfg=None, selected_coordinates=None, _metadata=None):
        self.directory = Path(directory)
        metadata = read_dataset_metadata(directory, cfg=cfg) if _metadata is None else _metadata
        self.configuration = metadata['configuration']
        for name in ('spec', 'data_id', 'provenance', 'execution', 'source_origin',
                     'scientific_validation', 'registration_runtime'):
            setattr(self, name, metadata[name])
        self.fields = metadata['fields']
        self._storage = metadata['storage']
        self._positions = _dataset_positions(self.spec)
        self._closed = False
        self._array = self._file = None
        first = self.spec.coordinates(self.spec.axis_names[0]) if self.spec.axis_names else []
        self._full_read = selected_coordinates is None
        selected = first if selected_coordinates is None else list(selected_coordinates)
        keys = [canonical_json(v) for v in selected]
        if self.spec.axis_names and (not selected or len(set(keys)) != len(keys) or any(
                key not in self._positions[self.spec.axis_names[0]] for key in keys)):
            raise ValueError('selected coordinates must be a nonempty ordered subset')
        self.read_scope = tuple(selected)
        self._scope_keys = frozenset(keys)
        self._selected_positions = (tuple(sorted(self._positions[self.spec.axis_names[0]][k] for k in keys))
                                    if self.spec.axis_names else ())
        path = _storage_path(directory, self._storage)
        try:
            if self._storage['format'] == 'npy':
                self._array = np.load(path, mmap_mode='r', allow_pickle=False)
                _check_array(self._array, self.spec.shape, self.spec.document['dtype'])
                if path.stat().st_size != self._array.offset + self._array.nbytes:
                    raise ValueError('NPY file length differs from its declared payload')
            elif self._storage['format'] == 'binary':
                dtype = np.dtype(self._storage['dtype']); offset = self._storage['offset']
                if path.stat().st_size != offset + prod(self.spec.shape) * dtype.itemsize:
                    raise ValueError('binary file size differs from its declared payload')
                self._array = np.memmap(path, mode='r', shape=self.spec.shape, dtype=dtype, offset=offset)
            else:
                self._file = _h5py().File(path, 'r')
                if self._storage.get('dataset') is not None:
                    self._array = self._file[self._storage['dataset']]
                    _check_array(self._array, self.spec.shape, self.spec.document['dtype'])
        except BaseException:
            self.close()
            raise
        _ACTIVE_READERS.add(self)

    def observation(self):
        return dict(state='complete' if self._full_read else 'selected', spec=self.spec,
            data_id=self.data_id, provenance=self.provenance,
            completed=tuple(range(self.spec.shape[0] if self.spec.shape else 1)) if self._full_read else self._selected_positions)

    def coverage_observation(self):
        return dict(state='complete', spec=self.spec, data_id=self.data_id,
                    provenance=self.provenance, completed=tuple(range(self.spec.shape[0] if self.spec.shape else 1)))

    def read(self, *, select=None, **coordinates):
        if self._closed or self._array is None:
            raise ValueError('reader is closed or this file contains named fields')
        names = self.spec.axis_names
        select = {} if select is None else dict(select)
        if (names[0] not in coordinates or set(coordinates) - set(names)
                or set(select) - set(names[1:]) or set(select) & set(coordinates)):
            raise ValueError('select the first axis and only unbound known axes')
        if canonical_json(coordinates[names[0]]) not in self._scope_keys:
            raise KeyError('first coordinate is outside the selected read scope')
        selection, gathers = [], []
        remaining_axis = 0
        for name in names:
            positions = self._positions[name]
            if name in coordinates:
                selection.append(positions[canonical_json(coordinates[name])])
                continue
            requested = select.get(name, slice(None))
            if isinstance(requested, slice):
                if not range(*requested.indices(len(positions))):
                    raise ValueError('empty axis selection')
                if self._file is not None and (requested.step or 1) < 0:
                    ids = tuple(range(*requested.indices(len(positions))))
                    selection.append(slice(min(ids), max(ids)+1))
                    gathers.append((remaining_axis, [i-min(ids) for i in ids]))
                else:
                    selection.append(requested)
            else:
                ids = [positions[canonical_json(v)] for v in requested]
                if not ids or len(set(ids)) != len(ids):
                    raise ValueError('empty or repeated axis selection')
                selection.append(slice(min(ids), max(ids)+1))
                gathers.append((remaining_axis, [i-min(ids) for i in ids]))
            remaining_axis += 1
        value = self._array[tuple(selection)]
        for axis, ids in sorted(gathers, key=lambda item: len(item[1])):
            value = np.take(np.asarray(value), ids, axis=axis)
        value = np.asarray(value)
        value.setflags(write=False)
        return value

    def read_field(self, path, *, selection=None):
        if self._closed or self._file is None:
            raise ValueError('named fields require an open HDF5 reader')
        key = canonical_json(list(path))
        declaration = self.fields[key]
        dataset = self._file[declaration['dataset']]
        _check_array(dataset, declaration['shape'], declaration['dtype'])
        value = dataset[()] if selection is None else dataset[selection]
        return np.asarray(value, dtype=dataset.dtype)

    def iter_read(self, first_values, *, select=None, **coordinates):
        first = self.spec.axis_names[0]
        for value in first_values:
            yield value, self.read(select=select, **{first: value, **coordinates})

    def close(self):
        if getattr(self, '_closed', True):
            return
        if isinstance(self._array, np.memmap):
            self._array._mmap.close()
        if self._file is not None:
            self._file.close()
        self._array = self._file = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _chunks(shape, dtype):
    if not shape or any(n == 0 for n in shape):
        return None
    if prod(shape) * np.dtype(dtype).itemsize <= 1024 * 1024:
        return None
    chunk = list(shape)
    for axis in range(min(2, len(chunk))):
        chunk[axis] = 1
    while prod(chunk) * np.dtype(dtype).itemsize > 1024 * 1024:
        axis = next(i for i, n in enumerate(chunk) if n > 1)
        chunk[axis] = (chunk[axis]+1)//2
    return tuple(chunk)


class DatasetWriter:
    """Fill one whole array per cfg; production state is separate and temporary."""
    def __init__(self, directory, spec, *, provenance, resume=False, attempt_context=None,
                 format='npy', structured=False, chunks=None):
        if not isinstance(spec, DatasetSpec) or type(resume) is not bool:
            raise TypeError('a DatasetSpec and boolean resume are required')
        if format not in ('npy', 'hdf5') or structured and format != 'hdf5':
            raise ValueError('use whole NPY or HDF5; named fields require HDF5')
        self.directory, self.spec = Path(directory), spec
        self.provenance = json.loads(canonical_json(provenance))
        self.configuration = configuration_key(spec.document['configuration']['configuration'])
        self.format, self.structured = format, structured
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_dir = self.directory / '.state'; self._state_dir.mkdir(exist_ok=True)
        self._state_path = self._state_dir / (self.configuration+'.json')
        self._lock_path = self._state_dir / (self.configuration+'.lock')
        self._temporary = self._state_dir / (self.configuration+('.h5' if format == 'hdf5' else '.npy'))
        self.path = self.directory / (self.configuration+('.h5' if format == 'hdf5' else '.npy'))
        self._closed = self._published = False
        self._file = self._array = None; self._completed = set(); self.fields = {}
        context = _normalize_attempt_context(attempt_context)
        self._context = context
        if self.configuration in read_registration_document(directory)['configurations'] or self.path.exists():
            raise FileExistsError('registered configuration is immutable')
        while True:
            try:
                stream = self._lock_path.open('x', encoding='ascii')
                break
            except FileNotFoundError:
                self._state_dir.mkdir(exist_ok=True)
        with stream:
            stream.write(str(os.getpid()))
        try:
            recovered_from=None
            if self._state_path.exists():
                if not resume:
                    raise FileExistsError('unfinished production requires resume=True')
                state = json.loads(self._state_path.read_text(encoding='utf-8'))
                if (canonical_json(state['spec']) != canonical_json(spec.document)
                        or canonical_json(state['provenance']) != canonical_json(self.provenance)
                        or state['format'] != format or state['structured'] != structured):
                    raise ValueError('resume definition or provenance differs')
                self._completed = set(state['completed']); self.fields = state['fields']
                self._realization = state['realization_id']; self._execution = state['execution']
                if self._execution['scientific_validation']['status']=='failed':
                    raise ValueError('failed scientific validation cannot be resumed')
                recovered_from=self._execution['attempts'][-1]['writer_session_id']
                if self._execution['attempts'][-1]['status'] == 'running':
                    self._execution['attempts'][-1].update(status='interrupted', ended_at=_utc_now())
            else:
                self._realization = uuid.uuid4().hex
                self._execution = _execution_record(spec, schema=5 if 'local_source_declaration' in context else 1,
                    local_source_declaration=context.get('local_source_declaration'))
                if 'local_source_declaration' in context:
                    self._execution['local_source_declaration'] = context['local_source_declaration']
            attempt=_new_attempt(context,recovered_from=recovered_from)
            if self._execution['schema'] in (4,5):attempt['execution_parameters']=None
            self._execution['attempts'].append(attempt)
            if format == 'npy':
                self._array = (np.load(self._temporary, mmap_mode='r+', allow_pickle=False)
                    if self._temporary.exists() else np.lib.format.open_memmap(self._temporary,
                        mode='w+', dtype=spec.document['dtype'], shape=spec.shape))
                _check_array(self._array, spec.shape, spec.document['dtype'])
            else:
                self._file = _h5py().File(self._temporary, 'a')
                if not structured:
                    self._array = self._file.require_dataset('data', shape=spec.shape,
                        dtype=spec.document['dtype'], chunks=chunks or _chunks(spec.shape, spec.document['dtype']))
            self._save_state()
        except BaseException:
            self.close()
            raise

    def _save_state(self):
        write_json_document(self._state_path, dict(spec=self.spec.document,
            provenance=self.provenance, format=self.format, structured=self.structured,
            realization_id=self._realization, completed=sorted(self._completed),
            fields=self.fields, execution=self._execution))

    @property
    def missing(self):
        return () if self.structured else tuple(i for i in range(self.spec.shape[0]) if i not in self._completed)

    def write(self, index, value):
        if self._closed or self._published or self.structured:
            raise ValueError('writer is closed or contains named fields')
        if type(index) is not int or not 0 <= index < self.spec.shape[0]:
            raise ValueError('invalid first-axis index')
        if index in self._completed:
            raise FileExistsError('completed coordinates cannot be overwritten')
        value = np.asarray(value); _check_array(value, self.spec.shape[1:], self.spec.document['dtype'])
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError('cannot publish nonfinite production output')
        self._array[index] = value
        if self._file is None:
            self._array.flush()
        else:
            self._file.flush()
        self._completed.add(index); self._save_state()

    def write_field(self, path, value, *, selection=None, shape=None, chunks=None, fillvalue=None):
        if self._closed or self._published or not self.structured:
            raise ValueError('writer is closed or does not contain named fields')
        value = np.asarray(value)
        if value.dtype.hasobject or value.dtype.kind not in 'biufc':
            raise TypeError('fields require plain numeric arrays')
        key = canonical_json(list(path))
        declaration = self.fields.get(key)
        if declaration is None:
            name = ('/'.join(quote(part, safe='') or '%00' for part in path)
                    if path and all(type(part) is str for part in path)
                    else 'fields/'+str(len(self.fields)))
            target_shape = value.shape if shape is None else tuple(shape)
            dataset = self._file.create_dataset(name, shape=target_shape, dtype=value.dtype,
                chunks=chunks or _chunks(target_shape, value.dtype), fillvalue=fillvalue)
            self.fields[key] = dict(dataset=name, shape=list(target_shape), dtype=value.dtype.str)
        else:
            dataset = self._file[declaration['dataset']]
        if selection is None:
            _check_array(value, dataset.shape, dataset.dtype)
            dataset[()] = value
        else:
            dataset[selection] = value
        self._file.flush()

    def bind_producer_attempt(self, producer_attempt_id, *, execution_parameters=None):
        attempt = self._execution['attempts'][-1]
        if not producer_attempt_id or attempt['producer_attempt_id'] is not None:
            raise ValueError('producer attempt must be bound once')
        attempt['producer_attempt_id'] = producer_attempt_id
        if execution_parameters is not None:
            from lqcdweave.contracts.execution import normalize_execution_parameters
            if self._execution['schema'] != 5:self._execution['schema'] = 4
            for entry in self._execution['attempts']:
                entry.setdefault('execution_parameters', None)
            attempt['execution_parameters'] = normalize_execution_parameters(execution_parameters)
        self._save_state()

    def update_definition(self, **metadata):
        """Finalize output metadata before publishing a native field collection."""
        if self._closed or self._published or not self.structured:
            raise ValueError('metadata updates require active named-field production')
        document=self.spec.document
        document['definition'].update(json.loads(canonical_json(metadata)))
        self.spec=DatasetSpec(document)
        self._execution['definition_id']=self.spec.definition_id
        self._save_state()

    def record_scientific_validation(self, validation):
        self._execution['scientific_validation'] = _normalize_scientific_validation(validation, self.spec)
        self._save_state()
        if self._execution['scientific_validation']['status'] == 'failed':
            self._execution['attempts'][-1].update(status='failed', ended_at=_utc_now(),
                failure=dict(type='ScientificValidationError', message='scientific validation failed'))
            self._save_state()
            self.close(); raise ValueError('scientific validation failed; data was not published')

    def publish(self, *, scientific_validation=None):
        if self._closed or self._published or self.missing:
            raise ValueError('only complete active production can publish')
        if scientific_validation is not None:
            self.record_scientific_validation(scientific_validation)
        identity = dict(schema=1, spec=self.spec.document, provenance=self.provenance,
                        realization_id=self._realization, state='complete')
        record = _published_record(identity, self._execution)
        record['storage'] = dict(format=self.format, path=self.path.name,
                                dataset=None if self.structured else 'data')
        record['fields'] = self.fields
        self._close_payload()
        with registration_lock(self.directory):
            document=read_registration_document(self.directory)
            if self.configuration in document['configurations'] or self.path.exists():
                raise FileExistsError('configuration was published by another writer')
            os.replace(self._temporary, self.path)
            try:
                append_registration(self.directory,document,self.configuration,record)
            except BaseException:
                os.replace(self.path, self._temporary)
                raise
        self._published = True; self._execution = record['execution']
        self._state_path.unlink(missing_ok=True)
        return record['data_id']

    def _close_payload(self):
        if isinstance(self._array, np.memmap):
            self._array.flush(); self._array._mmap.close()
        if self._file is not None:
            self._file.flush(); self._file.close()
        self._array = self._file = None

    def close(self):
        if self._closed:
            return
        self._close_payload()
        if not self._published and hasattr(self, '_execution'):
            attempt = self._execution['attempts'][-1]
            if attempt['status'] == 'running':
                attempt.update(status='interrupted', ended_at=_utc_now())
                self._save_state()
        self._lock_path.unlink(missing_ok=True)
        try:
            self._state_dir.rmdir()
        except OSError:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc and exc[0] is not None and not self._published and not self._closed:
            error=exc[1]
            self._execution['attempts'][-1].update(status='failed', ended_at=_utc_now(),
                failure=dict(type=type(error).__name__, message=str(error)))
            self._save_state()
        self.close()


def _registered_record(spec, provenance, storage, provenance_mode, operation):
    execution = None
    if provenance_mode == 'external_import':
        provenance, runtime = external_import_identity(provenance)
    elif provenance_mode != 'legacy':
        raise ValueError('unknown provenance mode')
    identity = dict(schema=1, spec=spec.document, provenance=provenance,
                    realization_id=uuid.uuid4().hex, state='complete')
    actual_id = stable_digest(identity)
    if provenance_mode == 'external_import':
        now = _utc_now()
        execution = dict(schema=3, definition_id=spec.definition_id, data_id=actual_id,
            registration_event=dict(operation=operation, started_at=now, completed_at=now, runtime=runtime),
            scientific_validation=dict(status='unknown'))
    return dict(identity=identity, data_id=actual_id, execution=execution, storage=storage, fields={})


def register_npy(directory, path, spec, *, provenance, provenance_mode='legacy'):
    path = Path(path).resolve()
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        _check_array(value, spec.shape, spec.document['dtype'])
    finally:
        value._mmap.close()
    record = _registered_record(spec, provenance, dict(format='npy', path=str(path), dataset='data'),
                                provenance_mode, 'register_npy')
    return register_record(directory, spec.document['configuration']['configuration'], record)


def register_hdf5(directory, path, spec, *, dataset='data', provenance, provenance_mode='legacy'):
    path = Path(path).resolve()
    with _h5py().File(path, 'r') as stream:
        _check_array(stream[dataset], spec.shape, spec.document['dtype'])
    record = _registered_record(spec, provenance, dict(format='hdf5', path=str(path), dataset=dataset),
                                provenance_mode, 'register_hdf5')
    return register_record(directory, spec.document['configuration']['configuration'], record)


def register_binary(directory, path, spec, *, provenance, storage_dtype, offset=0, provenance_mode='legacy'):
    """Explicit legacy input registration; production defaults remain NPY/HDF5."""
    path = Path(path).resolve(); dtype = np.dtype(storage_dtype)
    declared=np.dtype(spec.document['dtype'])
    if dtype.hasobject or dtype.kind != declared.kind or dtype.itemsize != declared.itemsize:
        raise ValueError('binary storage dtype differs from the declared contract')
    if type(offset) is not int or offset < 0 or path.stat().st_size != offset + prod(spec.shape)*dtype.itemsize:
        raise ValueError('binary length differs from its declared layout')
    record = _registered_record(spec, provenance,
        dict(format='binary', path=str(path), dtype=dtype.str, offset=offset), provenance_mode, 'register_binary')
    return register_record(directory, spec.document['configuration']['configuration'], record)


def register_npy_timeslices(directory, paths, spec, *, provenance, provenance_mode='legacy'):
    """Import old slices into one whole NPY file; no slice-file storage protocol."""
    if len(paths) != spec.shape[0]:
        raise ValueError('one input per first-axis coordinate is required')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    key=configuration_key(spec.document['configuration']['configuration'])
    destination=directory/(key+'.npy')
    if key in read_registration_document(directory)['configurations'] or destination.exists():
        raise FileExistsError('configuration is already registered')
    temporary=directory/('.'+key+'.'+uuid.uuid4().hex+'.tmp')
    array=np.lib.format.open_memmap(temporary,mode='w+',shape=spec.shape,dtype=spec.document['dtype'])
    try:
        for index,path in enumerate(paths):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                _check_array(value,spec.shape[1:],spec.document['dtype'])
                array[index]=value
            finally:value._mmap.close()
        array.flush(); array._mmap.close(); array=None
        record=_registered_record(spec,provenance,dict(format='npy',path=destination.name,dataset='data'),
            provenance_mode,'register_npy_timeslices')
        with registration_lock(directory):
            document=read_registration_document(directory)
            if key in document['configurations'] or destination.exists():raise FileExistsError('configuration is already registered')
            os.replace(temporary,destination)
            try:
                append_registration(directory,document,key,record)
            except BaseException:
                os.replace(destination,temporary)
                raise
        return record['data_id']
    finally:
        if array is not None:array._mmap.close()
        temporary.unlink(missing_ok=True)


class _DatasetSnapshot:
    def __init__(self, directory, *, cfg=None):
        self.directory = Path(directory)
        self.metadata = read_dataset_metadata(directory, cfg=cfg)
    def reader(self, *, selected_coordinates=None):
        return DatasetReader(self.directory, selected_coordinates=selected_coordinates, _metadata=self.metadata)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass


class DatasetView:
    def __init__(self, reader, spec):
        self.reader, self.spec = reader, spec
        self.data_id, self.provenance = reader.data_id, reader.provenance
    def read(self, *, select=None, **coordinates):
        names = self.spec.axis_names
        select = {} if select is None else dict(select)
        if (names[0] not in coordinates or set(coordinates)-set(names)
                or set(select)-set(names[1:]) or set(select)&set(coordinates)):
            raise ValueError('select the first axis and only unbound known axes')
        chosen = {}
        parent = _dataset_positions(self.reader.spec)
        for name in names:
            available = self.spec.coordinates(name)
            if name in coordinates:
                if canonical_json(coordinates[name]) not in {canonical_json(v) for v in available}:
                    raise KeyError('coordinate outside the requested view')
                continue
            requested = select.get(name, slice(None))
            values = available[requested] if isinstance(requested, slice) else list(requested)
            if not values or len({canonical_json(v) for v in values}) != len(values):
                raise KeyError('empty or repeated view selection')
            allowed = {canonical_json(v) for v in available}
            if any(canonical_json(v) not in allowed for v in values):
                raise KeyError('coordinate outside the requested view')
            indices = [parent[name][canonical_json(v)] for v in values]
            chosen[name] = (slice(indices[0], indices[0]+len(indices))
                if indices == list(range(indices[0], indices[0]+len(indices))) else values)
        return self.reader.read(select=chosen, **coordinates)
    def iter_read(self, first_values, *, select=None, **coordinates):
        first = self.spec.axis_names[0]
        for value in first_values:
            yield value, self.read(select=select, **{first: value, **coordinates})
    def close(self):
        self.reader.close()
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()


class DatasetIndex:
    """Optional explicit locator index, independent of per-dataset registrations."""
    def __init__(self, directory, *, managed_cache_root=None):
        self.directory = Path(directory).resolve(); self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'index.json'
        self.managed_cache_root = None if managed_cache_root is None else Path(managed_cache_root).resolve()
        self._transaction_depth = 0
        self._thread_lock = threading.RLock()
        self._transaction_owner = None
        self._owner_id = stable_digest({'dataset_index': str(self.directory)})
    def _read(self):
        if not self.path.exists():
            return dict(schema=1, publication_revision=0, datasets={})
        return json.loads(self.path.read_text(encoding='utf-8'))
    def _write(self, document):
        document['publication_revision'] += 1
        write_json_document(self.path, document)
    @property
    def publication_revision(self):
        return self._read()['publication_revision']
    def bump_publication_revision(self):
        record = self._read(); self._write(record); return record['publication_revision']
    @contextmanager
    def transaction(self):
        with self._thread_lock:
            if self._transaction_depth:
                yield self
                return
            with registration_lock(self.directory):
                self._transaction_depth += 1
                self._transaction_owner = threading.get_ident()
                try:
                    yield self
                finally:
                    self._transaction_depth -= 1
                    self._transaction_owner = None
    def register(self, directory, *, cfg=None):
        if not self._transaction_depth or self._transaction_owner != threading.get_ident():
            with self.transaction():
                return self.register(directory, cfg=cfg)
        metadata = read_dataset_metadata(directory, cfg=cfg)
        document = self._read(); identity = metadata['data_id']
        location = dict(directory=str(Path(directory).resolve()), configuration=metadata['configuration'])
        locations = document['datasets'].setdefault(identity, [])
        if location not in locations:
            locations.append(location); self._write(document)
        return identity
    def _metadata(self, identity):
        locations = self._read()['datasets'].get(identity)
        if not locations:
            raise KeyError('unknown registered data ID')
        records = []
        for location in locations:
            try:
                metadata = read_dataset_metadata(location['directory'], cfg=location['configuration'])
            except FileNotFoundError:
                continue
            if metadata['data_id'] != identity:
                raise ValueError('registered location identifies different data')
            records.append(metadata)
        if not records:
            raise FileNotFoundError('no available registered data location')
        if any(canonical_json(r['source_origin']) != canonical_json(records[0]['source_origin']) for r in records[1:]):
            raise ValueError('registered copies disagree on source origin')
        return records
    def observe_metadata(self, identity):
        return self._metadata(identity)[0]
    def open(self, identity):
        return self._open(identity)
    def _open(self, identity, *, selected_coordinates=None):
        for record in self._metadata(identity):
            try:
                return DatasetReader(record['directory'], cfg=record['configuration'],
                    selected_coordinates=selected_coordinates, _metadata=record)
            except FileNotFoundError:
                continue
        raise FileNotFoundError('no available registered data file')
    def open_selected(self, identity, coordinates):
        return self._open(identity,selected_coordinates=coordinates)
    def unregister(self, identity, location=None):
        if not self._transaction_depth or self._transaction_owner != threading.get_ident():
            with self.transaction():return self.unregister(identity,location)
        document=self._read()
        if identity not in document['datasets']:raise KeyError('unknown registered data ID')
        if location is None:del document['datasets'][identity]
        else:
            canonical=str(Path(location).resolve())
            remaining=[item for item in document['datasets'][identity] if item['directory'] != canonical]
            if len(remaining)==len(document['datasets'][identity]):raise KeyError('unknown registered location')
            if remaining:document['datasets'][identity]=remaining
            else:del document['datasets'][identity]
        self._write(document)
    def observations(self):
        return tuple(dict(state='complete', spec=m['spec'], data_id=identity,
            provenance=m['provenance'], completed=tuple(range(m['spec'].shape[0] if m['spec'].shape else 1)))
            for identity in self._read()['datasets'] for m in self._metadata(identity)[:1])
    def materialize(self, identity, destination, *, retention_managed=False):
        metadata = self.observe_metadata(identity); destination = Path(destination)
        cfg = metadata['configuration']
        if any((destination/'.state'/(cfg+suffix)).exists() for suffix in ('.json','.lock','.npy','.h5')):
            raise ValueError('destination contains unfinished production')
        if retention_managed:
            if self.managed_cache_root is None:
                raise ValueError('a configured managed cache root is required')
            if destination.is_symlink() or destination.resolve().parent != self.managed_cache_root:
                raise ValueError('managed destinations must be direct nonsymlink cache-root children')
        if cfg in read_registration_document(destination)['configurations']:
            if read_dataset_metadata(destination, cfg=cfg)['data_id'] != identity:
                raise FileExistsError('destination contains different data')
            self.register(destination, cfg=cfg); return identity
        destination.mkdir(parents=True, exist_ok=True)
        record = json.loads(canonical_json(metadata['record']))
        source = _storage_path(metadata['directory'], record['storage'])
        filename = cfg+source.suffix
        target = destination / filename
        if target.exists():
            raise FileExistsError('destination data file already exists')
        shutil.copyfile(source, target)
        record['storage']['path'] = filename
        if retention_managed:
            record['management'] = dict(owner=self._owner_id, root=str(self.managed_cache_root))
        else:
            record.pop('management', None)
        try:
            register_record(destination, cfg, record)
            self.register(destination, cfg=cfg)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return identity
    def metadata_inventory(self):
        return tuple(dict(data_id=identity, location=str(m['directory']), spec=m['spec'].document,
            source_origin=m['source_origin'], provenance=m['provenance'])
            for identity in self._read()['datasets'] for m in self._metadata(identity))

    def canonical_registered_location(self, identity, location):
        candidate = str(Path(location).resolve())
        if not any(str(m['directory'].resolve()) == candidate for m in self._metadata(identity)):
            raise ValueError('location is not registered for this actual data ID')
        return candidate

    def describe_managed_location(self, identity, location):
        if self.managed_cache_root is None:
            raise ValueError('a configured managed cache root is required')
        path = Path(location)
        if path.is_symlink() or path.resolve().parent != self.managed_cache_root:
            raise ValueError('managed location must be a direct nonsymlink cache-root child')
        path = path.resolve()
        record = next((m for m in self._metadata(identity) if m['directory'].resolve() == path), None)
        if record is None or record['record'].get('management') != dict(
                owner=self._owner_id, root=str(self.managed_cache_root)):
            raise ValueError('location is not owned and active')
        payload = _storage_path(path, record['storage'])
        if payload.is_symlink() or payload.resolve().parent != path:
            raise ValueError('managed payload is not a controlled local file')
        return dict(state='active', location=str(path), data_id=identity,
                    configuration=record['configuration'])

    def verify_managed_location(self, identity, location):
        descriptor = self.describe_managed_location(identity, location)
        with DatasetReader(descriptor['location'], cfg=descriptor['configuration']):
            pass
        return descriptor

    def retire_managed_location(self, identity, location, *, expected_revision=None):
        if not self._transaction_depth or self._transaction_owner != threading.get_ident():
            with self.transaction():
                return self.retire_managed_location(identity, location,expected_revision=expected_revision)
        if expected_revision is not None and expected_revision != self.publication_revision:
            raise ValueError('stale dataset index publication revision')
        descriptor = self.verify_managed_location(identity, location)
        directory = Path(descriptor['location']); cfg = descriptor['configuration']
        if any(not reader._closed and reader.directory.resolve() == directory
               and reader.configuration == cfg for reader in _ACTIVE_READERS):
            raise FileExistsError('managed data has an active reader')
        with registration_lock(directory):
            records = read_registration(directory)
            previous = records[cfg]
            payload = _storage_path(directory, previous['storage'])
            quarantine = directory / '.state' / (uuid.uuid4().hex + payload.suffix)
            os.replace(payload, quarantine)
            try:
                del records[cfg]
                write_registration(directory, records)
                document = self._read()
                remaining = [item for item in document['datasets'][identity]
                             if Path(item['directory']).resolve() != directory or item['configuration'] != cfg]
                if remaining:
                    document['datasets'][identity] = remaining
                else:
                    del document['datasets'][identity]
                self._write(document)
            except BaseException:
                records[cfg] = previous
                write_registration(directory, records)
                os.replace(quarantine, payload)
                raise
            quarantine.unlink()
        return dict(data_id=identity, location=str(directory), state='retired')
