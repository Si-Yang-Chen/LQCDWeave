"""Scoped temporary NPY arrays, independent of persistent data formats."""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import io
import os
import tempfile
import uuid
import numpy as np

@dataclass(frozen=True)
class _TemporaryNpyReceipt:
    scope_id: str
    filename: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def _npy_array_storage_bytes(shape, dtype):
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    if (any(type(extent) is not int or extent < 0 for extent in shape)
            or dtype.hasobject or dtype.kind not in "biufc"):
        raise ValueError("scratch size requires a numeric dtype and nonnegative array shape")
    header_data = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    sizes = []
    for fortran_order in (False, True):
        header_data["fortran_order"] = fortran_order
        stream = io.BytesIO()
        try:
            np.lib.format.write_array_header_1_0(stream, header_data)
        except ValueError:
            stream = io.BytesIO()
            np.lib.format.write_array_header_2_0(stream, header_data)
        sizes.append(stream.tell())
    return max(sizes) + prod(shape) * dtype.itemsize


class TemporaryNpyArrayStore:
    """Create private per-execution scratch scopes for NumPy arrays.

    This is temporary execution storage, not a DatasetWriter: it creates no
    registration, scientific identity, index entry, or resumable completion state.
    The configured parent directory must already exist. Arrays must be C/F
    contiguous so publication never creates a full-size staging copy.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError("temporary array parent must be an existing directory")

    def open_scope(self):
        if not self.directory.is_dir():
            raise ValueError("temporary array parent is unavailable")
        return _TemporaryNpyArrayScope(self.directory)

    def storage_bytes(self, shape, dtype):
        """Return a safe NPY header-plus-payload bound without allocating data."""
        return _npy_array_storage_bytes(shape, dtype)

    @property
    def host_reserve_bytes(self):
        """Return the extra host buffer required by this scratch implementation."""
        return 0


class _TemporaryNpyArrayScope:
    def __init__(self, parent):
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=".lqcdweave-spill-", dir=parent,
        )
        self._root = Path(self._temporary_directory.name).resolve()
        self._scope_id = uuid.uuid4().hex
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise ValueError("temporary array scope is closed")

    def _path(self, receipt):
        if not isinstance(receipt, _TemporaryNpyReceipt):
            raise TypeError("temporary array receipt belongs to another store")
        if receipt.scope_id != self._scope_id:
            raise ValueError("temporary array receipt belongs to another scope")
        if (not receipt.filename.endswith(".npy")
                or Path(receipt.filename).name != receipt.filename
                or len(receipt.filename) != 36):
            raise ValueError("invalid temporary array receipt path")
        path = self._root / receipt.filename
        if path.parent.resolve() != self._root or path.is_symlink():
            raise ValueError("temporary array receipt escaped its scope")
        return path

    def write_array(self, array):
        self._ensure_open()
        value = np.asarray(array)
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise TypeError("temporary arrays require a non-object numeric dtype")
        if not (value.flags.c_contiguous or value.flags.f_contiguous):
            raise ValueError("temporary array writes require a contiguous source")
        filename = uuid.uuid4().hex + ".npy"
        destination = self._root / filename
        temporary = self._root / (filename + ".tmp")
        try:
            with temporary.open("xb", buffering=0) as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            if destination.stat().st_size > _npy_array_storage_bytes(value.shape, value.dtype):
                raise ValueError("NPY scratch file exceeds its declared byte upper bound")
            return _TemporaryNpyReceipt(
                self._scope_id, filename, tuple(value.shape), value.dtype.str,
                int(value.nbytes),
            )
        except BaseException as exc:
            for path in (temporary, destination):
                try:
                    path.unlink(missing_ok=True)
                except BaseException as cleanup_error:
                    add_note = getattr(exc, "add_note", None)
                    if add_note is not None:
                        add_note(f"temporary array write cleanup also failed: {cleanup_error!r}")
            raise

    def read_array(self, receipt, *, shape, dtype):
        self._ensure_open()
        path = self._path(receipt)
        expected_shape = tuple(shape)
        expected_dtype = np.dtype(dtype)
        if expected_dtype.hasobject or expected_dtype.kind not in "biufc":
            raise TypeError("temporary array restore requires a non-object numeric dtype")
        if (expected_shape != receipt.shape or expected_dtype.str != receipt.dtype
                or expected_dtype.itemsize * prod(expected_shape)
                != receipt.nbytes):
            raise ValueError("temporary array receipt disagrees with the expected shape/dtype")
        with path.open("rb", buffering=0) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                actual_shape, fortran_order, actual_dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                actual_shape, fortran_order, actual_dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("unsupported temporary NPY header version")
            header_end = stream.tell()
            if (tuple(actual_shape) != expected_shape or actual_dtype != expected_dtype
                    or actual_dtype.hasobject or not isinstance(fortran_order, bool)):
                raise ValueError("temporary NPY header disagrees with the receipt")
            if os.fstat(stream.fileno()).st_size != header_end + receipt.nbytes:
                raise ValueError("temporary NPY payload size disagrees with its header")
            value = np.empty(expected_shape, dtype=expected_dtype,
                             order="F" if fortran_order else "C")
            if receipt.nbytes:
                # Transpose only the view: an F-order payload is C-contiguous
                # with its axes reversed. Read directly into the sole owner.
                payload = memoryview(value.T if fortran_order else value).cast("B")
                offset = 0
                while offset < receipt.nbytes:
                    count = stream.readinto(payload[offset:])
                    if not count:
                        raise ValueError("temporary NPY payload is truncated")
                    offset += count
        if (not isinstance(value, np.ndarray) or tuple(value.shape) != expected_shape
                or value.dtype != expected_dtype or not value.flags.owndata):
            raise ValueError("temporary NPY restore did not produce an owned expected array")
        return value

    def remove_array(self, receipt):
        self._ensure_open()
        self._path(receipt).unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        self._temporary_directory.cleanup()
        self._closed = True

