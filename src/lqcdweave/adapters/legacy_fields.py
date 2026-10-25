"""Read old ILDG/LIME gauge and QDP timeslice gauge/eigenvector files."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .selection import times


def _xml_value(root, name):
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == name:
            return node.text
    raise ValueError(f"missing XML field {name}")


class DenseFileHandle:
    """NPY or headerless binary input with optional explicit checks and axis map.

    Useful for gauge, eigenvector, point and overlap handles accepted by the
    production frontends. NPY shape and dtype come from its header; optional
    declarations are checked rather than used to reinterpret the payload.
    a separate reshape handles historical files whose header uses flat volume
    axes. A returned read-only map borrows the file until its consumer finishes.
    """

    def __init__(self, prefix, suffix, *, format, shape=None, dtype=None,
                 reshape=None, transpose=None, Ne=None, Np=None):
        if format not in ("npy", "binary"):
            raise ValueError("dense input format must be npy or binary")
        self.prefix, self.suffix, self.format = str(prefix), suffix, format
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        if format == "binary" and (self.shape is None or self.dtype is None):
            raise ValueError("binary input requires explicit shape and dtype")
        if ((self.shape is not None and (not self.shape or min(self.shape) < 1))
                or (self.dtype is not None and self.dtype.hasobject)):
            raise ValueError("dense input needs positive shape and non-object dtype")
        self.reshape = None if reshape is None else tuple(reshape)
        self.transpose = None if transpose is None else tuple(transpose)
        if (self.reshape is not None and self.shape is not None
                and int(np.prod(self.reshape)) != int(np.prod(self.shape))):
            raise ValueError("reshape changes the number of elements")
        rank = len(self.reshape or self.shape or self.transpose or ())
        if self.transpose is not None and tuple(sorted(self.transpose)) != tuple(range(rank)):
            raise ValueError("transpose must permute all axes")
        self.Ne, self.Np = Ne, Np

    def load(self, key):
        self._mode_key = None
        self._mode_array = None
        path = Path(f"{self.prefix}{key}{self.suffix}")
        if self.format == "npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if ((self.shape is not None and array.shape != self.shape)
                    or (self.dtype is not None and array.dtype != self.dtype)):
                array._mmap.close()
                raise ValueError("NPY input differs from declared shape/dtype")
        else:
            if path.stat().st_size != int(np.prod(self.shape)) * self.dtype.itemsize:
                raise ValueError("binary input length differs from declared shape/dtype")
            array = np.memmap(path, dtype=self.dtype, mode="r", shape=self.shape)
        mapping = array
        try:
            if self.reshape is not None:
                array = array.reshape(self.reshape)
            if self.transpose is not None:
                array = array.transpose(self.transpose)
        except Exception:
            mapping._mmap.close()
            raise
        self.file = str(path)
        return array

    def read_modes(self, key, time, start, stop):
        """Borrow a canonical NPY/binary time-mode slice without stacking times."""
        if self.Ne is None:
            raise ValueError("mode blocks require an explicit Ne")
        if getattr(self, "_mode_key", None) != key:
            self._mode_array = self.load(key)
            self._mode_key = key
        array = self._mode_array
        if (array.ndim != 6 or array.shape[1] < self.Ne or array.shape[-1] != 3
                or type(time) is not int or not 0 <= time < array.shape[0]
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Ne):
            raise ValueError("invalid canonical time or mode block")
        return array[time, start:stop]


class IldgGauge:
    """LIME ildg-format plus ildg-binary-data, returned in (t,z,y,x,mu,a,b)."""

    format = "ildg"

    def __init__(self, prefix, suffix=".lime"):
        self.prefix, self.suffix = str(prefix), suffix

    def load(self, key):
        path = Path(f"{self.prefix}{key}{self.suffix}")
        records = {}
        with path.open("rb") as stream:
            while header := stream.read(144):
                if len(header) < 144 and not header.strip(b"\x00\r\n\t "):
                    break  # Some Chroma files end with a newline after LIME padding.
                if len(header) != 144 or header[:4] != b"\x45\x67\x89\xab":
                    raise ValueError("invalid LIME header")
                length = struct.unpack(">Q", header[8:16])[0]
                name = header[16:144].split(b"\0", 1)[0].decode("ascii")
                position = stream.tell()
                if name in records:
                    raise ValueError("duplicate LIME record")
                records[name] = (position, length)
                stream.seek((length + 7) // 8 * 8, 1)
            if not {"ildg-format", "ildg-binary-data"}.issubset(records):
                raise ValueError("missing ILDG format or data record")
            pos, length = records["ildg-format"]
            stream.seek(pos)
            root = ET.fromstring(stream.read(length).rstrip(b"\0"))
        lx, ly, lz, lt = (int(_xml_value(root, name)) for name in ("lx", "ly", "lz", "lt"))
        precision = int(_xml_value(root, "precision"))
        if min(lx, ly, lz, lt) < 1 or precision not in (32, 64):
            raise ValueError("invalid ILDG lattice or precision")
        dtype = np.dtype(">c8" if precision == 32 else ">c16")
        shape = (lt, lz, ly, lx, 4, 3, 3)
        pos, length = records["ildg-binary-data"]
        if length != int(np.prod(shape)) * dtype.itemsize or pos + length > path.stat().st_size:
            raise ValueError("ILDG payload length differs from XML lattice")
        self.file = str(path)
        return np.memmap(path, dtype=dtype, mode="r", offset=pos, shape=shape)


def _read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated QDP file")
    return data


def _read_string(stream):
    length = struct.unpack(">i", _read_exact(stream, 4))[0]
    if not 0 <= length <= 16 * 1024 * 1024:
        raise ValueError("invalid QDP string length")
    return _read_exact(stream, length).decode("utf-8")


def _read_position(stream):
    return struct.unpack(">qq", _read_exact(stream, 16))[1]


class _QDPLazyArray:
    def __init__(self, path, role, count, dtype):
        self.path, self.role, self.dtype = Path(path), role, np.dtype(dtype)
        if self.dtype.kind != "c" or self.dtype.itemsize not in (8, 16):
            raise ValueError("QDP record dtype must be complex64 or complex128")
        with self.path.open("rb") as stream:
            if _read_string(stream) != "XXXXQDPLazyDiskMapObjFileXXXX":
                raise ValueError("invalid QDP magic")
            _read_exact(stream, 4)  # version
            root = ET.fromstring(_read_string(stream))
            dims = [int(x) for x in _xml_value(root, "lattSize").split()]
            if len(dims) != 4 or min(dims) < 1 or int(_xml_value(root, "decay_dir")) != 3:
                raise ValueError("invalid QDP lattice or decay direction")
            lx, ly, lz, lt = dims
            self.lattice_size = tuple(dims)
            table = _read_position(stream)
            if not 0 <= table < self.path.stat().st_size:
                raise ValueError("invalid QDP record table offset")
            stream.seek(table)
            entries = struct.unpack(">I", _read_exact(stream, 4))[0]
            offsets = {}
            for _ in range(entries):
                size = struct.unpack(">i", _read_exact(stream, 4))[0]
                if size != 8:
                    raise ValueError("QDP record key must contain two integers")
                key = struct.unpack(">ii", _read_exact(stream, size))
                if key in offsets:
                    raise ValueError("duplicate QDP record key")
                offsets[key] = _read_position(stream)
        self.shape = ((lt, lz, ly, lx, 4, 3, 3) if role == "gauge" else
                      (lt, count, lz, ly, lx, 3))
        self._record_count = 4 if role == "gauge" else count
        self._record_shape = (lz, ly, lx, 3, 3) if role == "gauge" else (lz, ly, lx, 3)
        self._record_bytes = int(np.prod(self._record_shape)) * self.dtype.itemsize
        expected = {(t, n) for t in range(lt) for n in range(self._record_count)}
        if set(offsets) != expected or any(offset < 0 or offset + self._record_bytes > self.path.stat().st_size for offset in offsets.values()):
            raise ValueError("incomplete or invalid QDP record coverage")
        self.offsets = offsets

    def _record(self, t, n):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[t, n])
            return np.frombuffer(_read_exact(stream, self._record_bytes), dtype=self.dtype).reshape(self._record_shape)

    def _time(self, t):
        records = [self._record(t, n) for n in range(self._record_count)]
        return np.stack(records, axis=3 if self.role == "gauge" else 0)

    def read_times(self, selected_times, *, records=None):
        """Read selected physical times and directions/modes only."""
        selected = times(selected_times, self.shape[0], "times")
        record_ids = (tuple(range(self._record_count)) if records is None else
                      times(records, self._record_count, "records"))
        axis = 3 if self.role == "gauge" else 0
        return np.stack([np.stack([self._record(t, n) for n in record_ids], axis=axis)
                         for t in selected])

    def __getitem__(self, key):
        # Elemental producers request [time, mode_slice]. Apply that slice to
        # record IDs before reading, rather than materializing every mode.
        if (self.role == "eigenvector" and isinstance(key, tuple) and len(key) >= 2
                and (type(key[0]) is int or isinstance(key[0], slice))
                and isinstance(key[1], slice)):
            time, modes = key[:2]
            if type(time) is int:
                if not 0 <= time < self.shape[0]:
                    raise IndexError("QDP time outside coverage")
                selected_times = (time,)
            else:
                selected_times = range(*time.indices(self.shape[0]))
            selected_modes = range(*modes.indices(self.shape[1]))
            result = np.empty((len(selected_times), len(selected_modes),
                               *self._record_shape), dtype=self.dtype)
            for ti, t in enumerate(selected_times):
                for ni, n in enumerate(selected_modes):
                    result[ti, ni] = self._record(t, n)
            if type(time) is int:
                return result[0][(slice(None), *key[2:])]
            return result[(slice(None), slice(None), *key[2:])]
        if self.role == "eigenvector" and isinstance(key, tuple) and len(key) >= 2 and all(type(x) is int for x in key[:2]):
            t, n = key[:2]
            if not (0 <= t < self.shape[0] and 0 <= n < self.shape[1]):
                raise IndexError("QDP record coordinate outside coverage")
            return self._record(t, n)[key[2:]]
        first = key[0] if isinstance(key, tuple) else key
        if type(first) is int:
            if not 0 <= first < self.shape[0]:
                raise IndexError("QDP time outside coverage")
            return self._time(first)[key[1:] if isinstance(key, tuple) else ()]
        if isinstance(first, slice):
            selected = range(*first.indices(self.shape[0]))
            result = self.read_times(selected)
            return result[(slice(None), *key[1:])] if isinstance(key, tuple) else result
        if isinstance(first, (tuple, list)):
            result = self.read_times(first)
            return result[(slice(None), *key[1:])] if isinstance(key, tuple) else result
        full = np.empty(self.shape, dtype=self.dtype)
        for t in range(self.shape[0]):
            for n in range(self._record_count):
                if self.role == "gauge":
                    full[t, :, :, :, n] = self._record(t, n)
                else:
                    full[t, n] = self._record(t, n)
        return full[key]


class QDPTimesliceField:
    """Old QDPLazyDiskMapObjFile for gauge or Laplace eigenvectors."""

    def __init__(self, prefix, suffix, *, role, count=None, dtype=None):
        if role not in ("gauge", "eigenvector") or (role == "eigenvector" and (type(count) is not int or count < 1)):
            raise ValueError("QDP role must be gauge or eigenvector with mode count")
        self.prefix, self.suffix, self.role = str(prefix), suffix, role
        self.Ne = count if role == "eigenvector" else None
        self.dtype = dtype or (">c16" if role == "gauge" else ">c8")

    def load(self, key):
        path = Path(f"{self.prefix}{key}{self.suffix}")
        self.file = str(path)
        self._stream_key = None
        self._stream_array = None
        return _QDPLazyArray(path, self.role, self.Ne, self.dtype)

    def read_modes(self, key, time, start, stop):
        """Return only requested QDP eigenvector records as an owned block."""
        if self.role != "eigenvector":
            raise ValueError("mode blocks require an eigenvector field")
        if (type(time) is not int or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Ne):
            raise ValueError("invalid time or mode block")
        if getattr(self, "_stream_key", None) != key:
            self._stream_array = self.load(key)
            self._stream_key = key
        if time >= self._stream_array.shape[0]:
            raise ValueError("time outside QDP coverage")
        return self._stream_array[time, start:stop]
