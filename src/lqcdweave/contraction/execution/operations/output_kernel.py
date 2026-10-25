"""Bounded CuPy output snapshot copies for fixed result layouts."""
from collections import OrderedDict
from functools import lru_cache
from math import prod

import numpy as np
from ..diagnostics import debug_enabled
from ..raw_arguments import bind_pointer_arguments
from .result_views import _DeferredResultView, _array_pointer


_MAX_LANES = 16
_MAX_ARGUMENT_BYTES = 3072
# Raw arguments carry pointers rather than the legacy CArray descriptors.
# Keep code size bounded, while using more of the existing 3 KiB argument budget.
_MAX_RAW_LANES = 128
_MAX_RAW_VARIANTS = 8

_RAW_TYPES = {np.dtype(t): c for t, c in (
    ('bool', 'bool'), ('int8', 'signed char'), ('uint8', 'unsigned char'),
    ('int16', 'short'), ('uint16', 'unsigned short'), ('int32', 'int'),
    ('uint32', 'unsigned int'), ('int64', 'long long'), ('uint64', 'unsigned long long'),
    ('float32', 'float'), ('float64', 'double'),
    ('complex64', 'complex<float>'), ('complex128', 'complex<double>'))}


@lru_cache(maxsize=64)
def _raw_copy_kernel(factory, source):
    # Factory and source only: no arrays, pointers, layout IDs, or device state.
    return factory(source, 'ed_output_snapshot_copy', options=('--std=c++17',))


def _raw_copy_source(shape, strides, dtype, lanes):
    count = prod(shape)
    params = [f'const {_RAW_TYPES[dtype]}* x{lane}' for lane in range(lanes)]
    params.append('long long destination_lane_offset')
    params.append(f'{_RAW_TYPES[dtype]}* y')
    branches = []
    for lane, input_strides in enumerate(strides):
        terms = []
        for axis, (dimension, stride) in enumerate(zip(shape, input_strides, strict=True)):
            divisor = prod(shape[axis + 1:])
            terms.append(f'((q / {divisor}LL) % {dimension}LL) * ({stride}LL)')
        offset = ' + '.join(terms) or '0LL'
        branches.append(
            f'{"else " if lane else ""}if (lane == {lane}LL) {{ '
            f'const char* address = reinterpret_cast<const char*>(x{lane}) + ({offset}); '
            f'y[(destination_lane_offset + {lane}LL) * {count}LL + q] = '
            f'*reinterpret_cast<const {_RAW_TYPES[dtype]}*>(address); }}')
    total = count * lanes
    return ('#include <cupy/complex.cuh>\n'
            'extern "C" __global__ void ed_output_snapshot_copy('
            + ', '.join(params) + ') {\n'
            'const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
            + f'if (i >= {total}LL) return;\n'
            + f'const long long lane = i / {count}LL, q = i % {count}LL;\n'
            + '\n'.join(branches) + '\n}')


class _BoundRawOutputCopy:
    """A frozen shape/dtype copier with only a bounded stride-variant cache."""

    __slots__ = ('shape', 'dtype', 'lanes', 'factory', 'backend', '_chunks', '_variants')
    accepts_deferred_views = True

    def __init__(self, shape, dtype, lanes, factory, backend):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.lanes = int(lanes)
        self.factory = factory
        self.backend = backend
        # One pointer per lane, one int64 offset and one destination pointer.
        width = min(_MAX_RAW_LANES, _MAX_ARGUMENT_BYTES // 8 - 2)
        self._chunks = tuple((start, min(start + width, self.lanes))
                             for start in range(0, self.lanes, width))
        self._variants = OrderedDict()

    def _bind(self, key):
        calls = []
        for start, stop in self._chunks:
            source = _raw_copy_source(self.shape, key[start:stop], self.dtype,
                                      stop - start)
            kernel = _raw_copy_kernel(self.factory, source)
            size = prod(self.shape) * (stop - start)
            grid, block, lane_offset = ((size + 255) // 256,), (256,), np.int64(start)
            native = bind_pointer_arguments(self.backend, kernel, grid, block,
                (None,) * (stop - start) + (lane_offset, None),
                tuple((lane, index) for lane, index in enumerate(range(start, stop))),
                pointer_getter=_array_pointer)
            calls.append((kernel, native, start, stop, lane_offset, grid, block))
        return tuple(calls)

    def __call__(self, values, destination):
        count = prod(self.shape)
        if count == 0:
            return True
        key = tuple(value.strides for value in values)
        calls = self._variants.get(key)
        if calls is None:
            calls = self._bind(key)
            self._variants[key] = calls
            if len(self._variants) > _MAX_RAW_VARIANTS:
                self._variants.popitem(last=False)
        else:
            self._variants.move_to_end(key)
        for kernel, native, start, stop, lane_offset, grid, block in calls:
            # A uint64 carries the same pointer-sized RawKernel ABI. The live
            # references keep each original owner alive through submission.
            if native is not None and not debug_enabled():
                native(values, destination)
            else:
                arguments = tuple(value.pointer_argument() if type(value) is _DeferredResultView
                                  else value for value in values[start:stop])
                kernel(grid, block, (*arguments, lane_offset, destination))
        return True


def bind_packed_output_copy(backend, shape, dtype, lanes):
    """Bind a bounded stride-specialized CuPy snapshot copier, if supported."""
    dtype = np.dtype(dtype)
    factory = getattr(backend, 'RawKernel', None)
    if (getattr(backend, '__name__', '') != 'cupy' or not callable(factory)
            or dtype not in _RAW_TYPES or lanes < 1):
        return None
    return _BoundRawOutputCopy(shape, dtype, lanes, factory, backend)


@lru_cache(maxsize=32)
def _copy_kernel(kernel_type, signature):
    # Include actual specialization metadata in the outer bounded cache rather
    # than leave an unbounded set of dtype/rank variants inside one kernel.
    count = len(signature) - 1
    branches = '\nelse '.join(
        f'if (lane == {lane}) output[(offset + lane) * count + q] = x{lane}[q];'
        for lane in range(count))
    params = ', '.join(f'raw T x{lane}' for lane in range(count))
    return kernel_type(params + ', int64 count, int64 offset', 'raw T output',
        'const long long lane = i / count, q = i % count;\n' + branches,
        'ed_pack_outputs')


def copy_packed_outputs(values, owner, backend):
    """Return whether the supported CuPy copy path handled this output group.

    It adds no arrays beyond the final owner, including for strided/scalar
    sources. A bounded number of pointers/strides travel as kernel arguments;
    large groups split launches rather than allocating device pointer tables.
    """
    kernel_type = getattr(backend, 'ElementwiseKernel', None)
    if (getattr(backend, '__name__', '') != 'cupy' or not callable(kernel_type)
            or owner.dtype.kind not in 'biufc'):
        return False
    count = prod(owner.shape[1:])
    if count == 0:
        return True
    output_bytes = 48 + 16 * owner.ndim
    # Unsupported exceptionally high-rank arrays keep the ordinary copy ABI.
    if any(output_bytes + 16 + 16 * value.ndim > _MAX_ARGUMENT_BYTES for value in values):
        return False
    offset = 0
    while offset < len(values):
        end, argument_bytes = offset, output_bytes
        while end < len(values) and end - offset < _MAX_LANES:
            size = 16 + 16 * values[end].ndim
            if argument_bytes + size > _MAX_ARGUMENT_BYTES:
                break
            argument_bytes += size
            end += 1
        chunk = values[offset:end]
        signature = tuple((v.dtype.str, v.ndim, bool(v.flags.c_contiguous),
                           tuple(v.shape), tuple(v.strides)) for v in (*chunk, owner))
        _copy_kernel(kernel_type, signature)(*chunk, np.int64(count), np.int64(offset), owner,
                                             size=(end - offset) * count)
        offset = end
    return True
