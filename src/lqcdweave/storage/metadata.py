"""Atomic JSON metadata and compressed record streams."""
import json
import os
from pathlib import Path
import uuid
import time
from lqcdweave.contracts.dataset import canonical_json


def _replace_document(temporary, path):
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        if getattr(error, 'winerror', None) not in (5, 32):
            raise
        # Windows readers can briefly deny rename while their handle is open.
        # Retry only this sharing failure; normal atomic writes do no waiting.
        deadline = time.monotonic() + 1
        while True:
            time.sleep(.01)
            try:
                os.replace(temporary, path)
                return
            except PermissionError as error:
                if (getattr(error, 'winerror', None) not in (5, 32)
                        or time.monotonic() >= deadline):
                    raise


def write_json_document(path, document):
    """Atomically save JSON metadata; the destination parent must exist."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(document))
            stream.flush()
            os.fsync(stream.fileno())
        _replace_document(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_records(path, records):
    """Atomically write a compressed stream of JSON metadata records.

    Extends the existing atomic JSON publication to large planning metadata
    without constructing one giant JSON string. No numerical arrays are stored.
    """
    import gzip
    path=Path(path)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb') as stream:
                for record in records:
                    line=(canonical_json({'record':record})+'\n').encode('utf-8')
                    stream.write(line)
            raw.flush();os.fsync(raw.fileno())
        _replace_document(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_records(path):
    """Yield records from a compressed JSON metadata stream."""
    import gzip
    with gzip.open(path,'rb') as stream:
        for line in stream:
            value=json.loads(line)
            if not isinstance(value, dict) or set(value) != {'record'}:
                raise ValueError('invalid metadata stream record')
            yield value['record']

