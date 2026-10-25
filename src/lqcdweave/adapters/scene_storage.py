"""Explicit NPY selections for shared-block replay; transformations stay in memory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import math
import time

import numpy as np

from ..contracts.correlator.api import StorageBlock


@dataclass(frozen=True)
class NpyBlockBinding:
    path: Path
    selection: tuple
    shape: tuple[int, ...]
    dtype: str
    read_mode: str = "mmap"
    file_shape: tuple[int, ...] | None = None

    def __post_init__(self):
        if not all(isinstance(index, (int, slice)) for index in self.selection):
            raise TypeError("NPY block selection supports basic integer/slice indexing only")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "selection", tuple(self.selection))
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.read_mode not in ("mmap", "whole_file"):
            raise ValueError("read_mode must be mmap or whole_file")
        if self.file_shape is not None:
            object.__setattr__(self, "file_shape", tuple(self.file_shape))
        if self.read_mode == "whole_file" and self.file_shape is None:
            raise ValueError("whole_file reads require an explicit file_shape")


@dataclass(frozen=True)
class NpyPackedBinding:
    """Several explicit selections packed after one whole-file payload read.

    Segments must already be deduplicated by the caller using verified dataset
    and basis identities. Offsets refer to elements of the returned flat array.
    Whole-file host staging is still required and fully charged.
    """

    segments: tuple[NpyBlockBinding, ...]

    def __post_init__(self):
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError("packed binding requires at least one segment")
        first = self.segments[0]
        for segment in self.segments:
            if (segment.read_mode != "whole_file" or segment.path != first.path
                    or segment.file_shape != first.file_shape or segment.dtype != first.dtype):
                raise ValueError("packed segments require one whole-file identity and dtype")
            # Check the selected shape from metadata before any payload allocation.
            shape = []
            if len(segment.selection) > len(segment.file_shape):
                raise ValueError("too many selection axes")
            indices = segment.selection + (slice(None),) * (len(segment.file_shape)-len(segment.selection))
            for index, extent in zip(indices, segment.file_shape):
                if isinstance(index, int):
                    if not -extent <= index < extent:
                        raise ValueError("packed selection index outside file shape")
                else:
                    shape.append(len(range(*index.indices(extent))))
            if tuple(shape) != segment.shape:
                raise ValueError("packed segment shape disagrees with selection")

    @property
    def path(self):
        return self.segments[0].path

    @property
    def dtype(self):
        return self.segments[0].dtype

    @property
    def file_shape(self):
        return self.segments[0].file_shape

    @property
    def read_mode(self):
        return "whole_file"

    @property
    def shape(self):
        return (sum(math.prod(segment.shape) for segment in self.segments),)

    def segment_view(self, packed, index):
        """Borrow one segment without allocating or reading a file."""
        if not isinstance(index, int) or not 0 <= index < len(self.segments):
            raise IndexError("packed segment index outside binding")
        if tuple(packed.shape) != self.shape or np.dtype(packed.dtype) != np.dtype(self.dtype):
            raise ValueError("packed buffer disagrees with binding")
        offset = sum(math.prod(segment.shape) for segment in self.segments[:index])
        shape = self.segments[index].shape
        return packed[offset:offset+math.prod(shape)].reshape(shape)


class NpySceneBlockLoader:
    """Read each scheduled selection into owned storage; no additional cache.

    Callbacks receive (source_metadata, dependency_arrays) and may only perform
    the declared in-memory transformations. Paths bind locations, while the
    plan's dataset/selection identities declare verified physical sharing.
    Device reads use a host staging copy then one device transfer.
    """

    prefetch_across_batches = True

    def __init__(self, bindings: Mapping[str, NpyBlockBinding | NpyPackedBinding],
                 materializers: Mapping[str, Callable], *, backend=np,
                 temporary_array_store=None):
        self.bindings, self.materializers = dict(bindings), dict(materializers)
        self.backend = backend
        self.temporary_array_store = temporary_array_store
        self._io_events = None

    def set_measurement(self, enabled):
        """Called by the replay measurement switch; no timing when disabled."""
        self._io_events = [] if enabled else None

    def drain_io_events(self):
        result = tuple(self._io_events or ())
        if self._io_events is not None:
            self._io_events.clear()
        return result

    def _event(self, binding, operation, size, started):
        if self._io_events is not None:
            self._io_events.append(dict(path=str(binding.path), operation=operation,
                                        bytes=size, observed_seconds=time.perf_counter() - started))

    def can_prefetch_block(self, key, block):
        return key in self.bindings

    def prefetch_host_bytes(self, key, block):
        return block.host_bytes + block.staging_host_bytes

    def prefetch_block(self, key, block):
        return self._read_host(key, block)

    def finish_prefetch(self, key, block, value):
        result = self._to_backend(self.bindings[key], value)
        if self.backend.__name__ == "cupy":
            # The future's host buffer must outlive the queued transfer.
            self.backend.cuda.get_current_stream().synchronize()
        return result

    def close_prefetch(self, key, value):
        # _read_host returns owned arrays and closes all file handles itself.
        pass

    def read_block(self, key: str, block: StorageBlock):
        return self._to_backend(self.bindings[key], self._read_host(key, block))

    def _read_host(self, key, block):
        binding = self.bindings[key]
        gpu = self.backend.__name__ == "cupy"
        size = math.prod(binding.shape) * np.dtype(binding.dtype).itemsize
        whole = binding.read_mode == "whole_file"
        read_bytes = math.prod(binding.file_shape) * np.dtype(binding.dtype).itemsize if whole else size
        staging = size + (read_bytes if whole else 0)
        if block.read_bytes != read_bytes:
            raise ValueError("read_bytes disagrees with the declared NPY read strategy")
        if gpu:
            if block.device_bytes < size or block.staging_host_bytes < staging or block.transfer_bytes != size:
                raise ValueError("device read needs declared device storage, host staging, and transfer")
        elif block.host_bytes < size or (whole and block.staging_host_bytes < read_bytes):
            raise ValueError("CPU read needs selected storage and whole-file staging")
        # Validate the header before np.load can allocate an unexpectedly large file.
        if whole:
            with binding.path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("whole_file supports NPY header versions 1 and 2")
                if shape != binding.file_shape or dtype != np.dtype(binding.dtype) or dtype.hasobject:
                    raise ValueError("NPY file header disagrees with declared shape/dtype")
                if binding.path.stat().st_size != stream.tell() + read_bytes:
                    raise ValueError("NPY file size disagrees with its header")
        started = time.perf_counter() if self._io_events is not None else 0
        mapped = np.load(binding.path, mmap_mode=None if whole else "r", allow_pickle=False)
        try:
            if isinstance(binding, NpyPackedBinding):
                host = np.empty(binding.shape, dtype=binding.dtype)
                for index, segment in enumerate(binding.segments):
                    selected = mapped[segment.selection]
                    if tuple(selected.shape) != segment.shape:
                        raise ValueError(f"NPY segment disagrees with declared shape: {key}")
                    binding.segment_view(host, index)[...] = selected
            else:
                selected = mapped[binding.selection]
                if tuple(selected.shape) != binding.shape or selected.dtype != np.dtype(binding.dtype):
                    raise ValueError(f"NPY selection disagrees with declared shape/dtype: {key}")
                host = np.array(selected, copy=True)
        finally:
            # A copied selection no longer needs the mmap or an open file handle.
            selected = None
            if not whole:
                mapped._mmap.close()
            mapped = None
        self._event(binding, "npy_whole_read" if whole else "npy_slice_read", read_bytes, started)
        return host

    def _to_backend(self, binding, host):
        if self.backend.__name__ != "cupy":
            return host
        size = host.nbytes
        if self._io_events is not None:
            self.backend.cuda.get_current_stream().synchronize()
        started = time.perf_counter() if self._io_events is not None else 0
        device = self.backend.asarray(host)
        if self._io_events is not None:
            self.backend.cuda.get_current_stream().synchronize()
        self._event(binding, "npy_slice_transfer", size, started)
        return device

    def materialize(self, key, source, blocks):
        return self.materializers[key](source.metadata, blocks)


@dataclass(frozen=True)
class NpyPrefixRequest:
    """A prefix in a caller-verified dataset and ordered basis, not a filename identity."""

    dataset: str
    basis_identity: str
    path: Path
    file_shape: tuple[int, ...]
    prefix_shape: tuple[int, ...]
    dtype: str = "complex128"


def merge_npy_prefixes(requests: Mapping[str, NpyPrefixRequest], *, backend="numpy", read_mode="whole_file"):
    """Return blocks, bindings, and request->block keys for maximum shared prefixes.

    All axes are explicit; full-time requests keep their full time extent.
    Whole-file mode avoids shared-filesystem mmap faults, and charges the whole
    payload once plus the cropped host staging. It does not claim selected disk IO.
    No payload files or arrays are opened here. Path resolution is reused only
    within this call; there is no persistent input cache.
    """
    if backend not in ("numpy", "cupy"):
        raise ValueError("backend must be numpy or cupy")
    groups, aliases, paths, resolved_paths = {}, {}, {}, {}
    for key, request in requests.items():
        r = request
        if not r.dataset or not r.basis_identity:
            raise ValueError("verified dataset and ordered basis identities are required")
        shape, prefix = tuple(r.file_shape), tuple(r.prefix_shape)
        if len(shape) != len(prefix) or any(type(x) is not int or x < 1 for x in shape + prefix):
            raise ValueError("file and prefix shapes must have equal rank and positive integer extents")
        if any(p > n for p, n in zip(prefix, shape)) or np.dtype(r.dtype).kind not in "biufc":
            raise ValueError("invalid prefix extent or numeric dtype")
        supplied_path = Path(r.path)
        if supplied_path not in resolved_paths:
            resolved_paths[supplied_path] = supplied_path.resolve()
        path = resolved_paths[supplied_path]
        identity = (r.basis_identity, path, shape, np.dtype(r.dtype).str)
        if path in paths and paths[path] != r.dataset:
            raise ValueError("one prefix file cannot have conflicting dataset identities")
        paths[path] = r.dataset
        if r.dataset in groups:
            previous, maximum = groups[r.dataset]
            if previous != identity:
                raise ValueError("shared dataset requires identical basis, file, shape and dtype")
            prefix = tuple(max(a, b) for a, b in zip(prefix, maximum))
        groups[r.dataset] = (identity, prefix)
        aliases[key] = r.dataset
    blocks, bindings = {}, {}
    for key, ((basis, path, shape, dtype), maximum) in groups.items():
        size = math.prod(maximum) * np.dtype(dtype).itemsize
        payload = math.prod(shape) * np.dtype(dtype).itemsize
        whole = read_mode == "whole_file"
        bindings[key] = NpyBlockBinding(path, tuple(slice(0, n) for n in maximum), maximum, dtype,
                                         read_mode=read_mode, file_shape=shape)
        blocks[key] = StorageBlock(key, "prefix:" + str(maximum), payload if whole else size,
            host_bytes=size if backend == "numpy" else 0, device_bytes=size if backend == "cupy" else 0,
            transfer_bytes=size if backend == "cupy" else 0,
            staging_host_bytes=(payload if whole else 0) + (size if backend == "cupy" else 0))
    return blocks, bindings, aliases
