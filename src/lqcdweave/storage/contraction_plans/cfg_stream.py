"""Bounded serialization of the existing CfgPlan, using A's contract codec.

The framing contains a versioned envelope, ordered field chunks and an explicit
end record. It adds no execution IR. Sharing spans chunks, so selected batches
and constants preserve their object identity after loading.
"""
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields
import gzip
import io
from itertools import islice
import json
import os
from pathlib import Path

from lqcdweave.contracts.correlator.api import CfgPlan
from .api import (
    PlanArchiveError, _decode_envelope, _make_envelope, _json_bytes,
    _reject_constant, _decode_plan_value,
)
from .archive_io import _atomic_archive_file
from .compact import StreamEncoder, StreamDecoder


_MAGIC = b'EDCFG-STREAM\n'
_ENCODING = 'unified-device-stream-v5'
_COMPACT_LEGACY_ENCODING = 'unified-device-stream-v4'
_LEGACY_ENCODING = 'unified-device-stream-v3'
_CHUNK_ITEMS = 256
_STREAM_ITEMS = 100_000
_STREAM_BINDING_BYTES = 32 * 1024 * 1024


def needs_stream(plan):
    return (sum(len(getattr(plan, name)) for name in
                ('requests', 'blocks', 'preparations', 'specs')) >= _STREAM_ITEMS
            or sum((len(i.node_ids.data) + len(i.reused.data)) // 2 for i in plan.instances)
            >= _STREAM_BINDING_BYTES)


def _layout(plan):
    result = []
    for field in fields(CfgPlan):
        value = getattr(plan, field.name)
        if isinstance(value, Mapping):
            kind, count = 'mapping', len(value)
        elif type(value) is tuple:
            kind, count = 'tuple', len(value)
        else:
            kind, count = 'value', 1
        result.append([field.name, kind, count])
    return result


def write_stream(path, plan):
    destination = Path(path)
    layout = _layout(plan)
    encoder = StreamEncoder()
    with _atomic_archive_file(destination) as raw, ExitStack() as stack:
        output = raw
        if destination.suffix.lower() == '.gz':
            output = stack.enter_context(gzip.GzipFile(filename='', fileobj=raw,
                                                       mode='wb', compresslevel=3, mtime=0))
        output.write(_MAGIC)
        output.write(_json_bytes(_make_envelope('complete-cfg-plan',
            {'encoding': _ENCODING, 'fields': layout})) + b'\n')
        for name, kind, count in layout:
            value = getattr(plan, name)
            iterator = iter(value.items() if kind == 'mapping' else
                            value if kind == 'tuple' else (value,))
            # A template or one coordinate's packed tables are indivisible;
            # growing global collections never create one giant JSON tree.
            chunk_size = 1 if name in ('templates', 'instances') else _CHUNK_ITEMS
            for offset in range(0, count, chunk_size):
                chunk = tuple(islice(iterator, chunk_size))
                output.write(_json_bytes([name, offset, *encoder.chunk(chunk)]) + b'\n')
        output.write(b'{"complete":true}\n')
    return destination


def _json_record(stream):
    line = stream.readline()
    if not line:
        raise PlanArchiveError('truncated complete cfg stream')
    try:
        return json.loads(line, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PlanArchiveError(f'invalid complete cfg stream record: {exc}') from exc


def _read_stream(stream):
    payload = _decode_envelope(_json_record(stream), expected_type='complete-cfg-plan')
    if (set(payload) != {'encoding', 'fields'}
            or payload['encoding'] not in (_ENCODING, _COMPACT_LEGACY_ENCODING, _LEGACY_ENCODING)):
        raise PlanArchiveError('unsupported complete cfg stream encoding')
    compact = payload['encoding'] != _LEGACY_ENCODING
    decoder = StreamDecoder(allow_defaults=payload['encoding']==_ENCODING) if compact else None
    layout = payload['fields']
    names = [field.name for field in fields(CfgPlan)]
    if (not isinstance(layout, list) or len(layout) != len(names)
            or any(not isinstance(item, list) or len(item) != 3 for item in layout)
            or [item[0] for item in layout] != names):
        raise PlanArchiveError('complete cfg stream fields do not match schema')
    shared, values = {}, {}
    for name, kind, count in layout:
        if (kind not in ('mapping', 'tuple', 'value') or type(count) is not int
                or count < 0 or (kind == 'value' and count != 1)):
            raise PlanArchiveError('invalid complete cfg stream field layout')
        result = {} if kind == 'mapping' else []
        offset = 0
        while offset < count:
            record = _json_record(stream)
            if (not isinstance(record, list) or len(record) != (5 if compact else 3) or record[0] != name
                    or type(record[1]) is not int or record[1] != offset
                    or not isinstance(record[-1], list) or not record[-1]
                    or len(record[-1]) > min(_CHUNK_ITEMS, count - offset)):
                raise PlanArchiveError('invalid, duplicate or out-of-order cfg stream chunk')
            decoded_items = (decoder.chunk(record[2], record[3], record[4]) if compact else
                             (_decode_plan_value(item, shared=shared) for item in record[2]))
            for decoded in decoded_items:
                if kind == 'mapping':
                    if not isinstance(decoded, tuple) or len(decoded) != 2:
                        raise PlanArchiveError('invalid complete cfg mapping pair')
                    key, value = decoded
                    try:
                        if key in result:
                            raise PlanArchiveError('duplicate complete cfg mapping key')
                        result[key] = value
                    except TypeError as exc:
                        raise PlanArchiveError('invalid complete cfg mapping key') from exc
                else:
                    result.append(decoded)
            offset += len(record[-1])
        values[name] = result if kind == 'mapping' else tuple(result) if kind == 'tuple' else result[0]
    terminator = _json_record(stream)
    if (not isinstance(terminator, dict) or set(terminator) != {'complete'}
            or terminator['complete'] is not True or stream.read(1)):
        raise PlanArchiveError('missing cfg stream terminator or trailing records')
    try:
        return CfgPlan(**values)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise PlanArchiveError(f'invalid complete cfg stream plan: {exc}') from exc


def read_stream_if_present(source):
    """Return None for the existing JSON encoding without reading it in full."""
    with ExitStack() as stack:
        if isinstance(source, bytes):
            raw = io.BytesIO(source)
        elif isinstance(source, str) and source.lstrip().startswith('{'):
            return None
        elif isinstance(source, (str, os.PathLike)):
            try:
                raw = stack.enter_context(open(source, 'rb'))
            except OSError as exc:
                raise PlanArchiveError(f'unable to read plan archive {source}: {exc}') from exc
        else:
            raise TypeError('archive source must be a path, UTF-8 JSON text, or bytes')
        prefix = raw.read(2)
        raw.seek(0)
        stream = raw
        if prefix == b'\x1f\x8b':
            stream = stack.enter_context(gzip.GzipFile(fileobj=raw, mode='rb'))
        try:
            if stream.read(len(_MAGIC)) != _MAGIC:
                return None
            return _read_stream(stream)
        except (OSError, EOFError) as exc:
            raise PlanArchiveError(f'invalid complete cfg stream: {exc}') from exc
