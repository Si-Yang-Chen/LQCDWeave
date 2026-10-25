"""Fixed-layout concatenation kernels for already selected dense batches."""
from math import prod
from collections import OrderedDict
from functools import lru_cache
from operator import itemgetter
import numpy as np
from ..diagnostics import debug_enabled
from ..raw_arguments import bind_pointer_arguments
from .result_views import _DeferredResultView, _array_pointer

# General raw calls use 264 bytes at 32 inputs. Equal stacks add two int64
# offsets, using 280 bytes, independent of rank and the number of stack lanes.
# Mixed-stride stacks also carry 32 signed strides per remaining dimension;
# their complete fixed pointer/scalar ABI must stay within the same 3 KiB cap.
# Keep the CArray-based fallback's independent eight-input/3 KiB limits.
_MAX_RAW_INPUTS = 32
_MAX_LEGACY_INPUTS = 8
_MAX_ARGUMENT_BYTES = 3072


def raw_packing_kernel_calls_upper_bound(input_count):
    """Bound raw copy submissions; current contiguous owners may avoid them."""
    return (input_count + _MAX_RAW_INPUTS - 1) // _MAX_RAW_INPUTS


def _packing_values(arrays, operands):
    """Resolve current references without materializing pointer-only inputs."""
    raw = getattr(arrays, '_raw_output_value', None)
    get = raw if callable(raw) and not debug_enabled() else arrays.__getitem__
    return tuple(get(key) for key, _ in operands)


def _bind_packing_values(operands):
    """Bind fixed keys, resolving only the current instance's live references."""
    if not operands:
        return lambda arrays: ()
    keys = tuple(key for key, _ in operands)
    key_getter = itemgetter(*keys)
    single = len(keys) == 1

    def values(arrays):
        bulk = getattr(arrays, '_raw_output_values', None)
        if not debug_enabled() and callable(bulk):
            try:
                return bulk(key_getter, single)
            except KeyError:
                # Resolve an invalid handoff in original operand order, so a
                # missing earlier owner still precedes a later missing key.
                return _packing_values(arrays, operands)
        return _packing_values(arrays, operands)

    return values


def _pointer_arguments(values):
    return tuple(value.pointer_argument() if type(value) is _DeferredResultView
                 else value for value in values)

_TYPES = {np.dtype(t): c for t, c in (
    ('bool', 'bool'), ('int8', 'signed char'), ('uint8', 'unsigned char'),
    ('int16', 'short'), ('uint16', 'unsigned short'), ('int32', 'int'),
    ('uint32', 'unsigned int'), ('int64', 'long long'), ('uint64', 'unsigned long long'),
    ('float32', 'float'), ('float64', 'double'),
    ('complex64', 'complex<float>'), ('complex128', 'complex<double>'))}


@lru_cache(maxsize=64)
def _raw_kernel(factory, source):
    # Code/factory only. CuPy manages its per-device JIT and binary cache.
    return factory(source, 'ed_dense_batch_pack_raw', options=('--std=c++17',))


def _stack_address(shape, strides):
    """Canonical fixed-stride address, joining adjacent contiguous dimensions."""
    groups = []
    for dimension, step in zip(shape, strides, strict=True):
        if dimension == 1:
            continue
        if groups and groups[-1][1] == dimension * step:
            groups[-1] = (groups[-1][0] * dimension, step)
        else:
            groups.append((dimension, step))
    terms = []
    for index, (dimension, step) in enumerate(groups):
        if not step:
            continue
        divisor = prod(size for size, _ in groups[index + 1:])
        coordinate = 'q' if divisor == 1 else f'(q / {divisor}LL)'
        # q < total size is checked before indexing, so the leading coordinate
        # cannot reach its dimension. Inner coordinates still need their modulo.
        if index:
            coordinate = f'({coordinate} % {dimension}LL)'
        terms.append(f'{coordinate} * ({step}LL)')
    return ' + '.join(terms) or '0LL'


def _stack_packing_call(backend, factory, shape, strides, axis, dtype, output_dtype,
                        span, offset, start, stop):
    """Keep geometry constant while sharing code across equal-stack widths.

    A CTA reads one input lane. Only destination span and chunk offset vary;
    every integer divisor in source remains a compile-time constant. The fixed
    pointer ABI is padded with null values, never a device pointer table.
    """
    size, width = prod(shape), prod(shape[axis + 1:])
    input_type, output_type = _TYPES[dtype], _TYPES[output_dtype]
    params = [f'const {input_type}* x{i}' for i in range(_MAX_RAW_INPUTS)]
    params += ['long long span', 'long long offset', f'{output_type}* y']
    address = _stack_address(shape, strides)
    choices = '\n'.join(f'case {i}: input = x{i}; break;' for i in range(_MAX_RAW_INPUTS))
    source = ('#include <cupy/complex.cuh>\n'
        'extern "C" __global__ void ed_dense_batch_pack_raw('
        + ', '.join(params) + ') {\n'
        'const long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
        + f'if (q >= {size}LL) return;\n'
        + f'const {input_type}* input;\nswitch (blockIdx.y) {{\n'
        + choices + '\ndefault: return;\n}\n'
        + f'const auto value = *reinterpret_cast<const {input_type}*>('
          f'reinterpret_cast<const char*>(input) + ({address}));\n'
        + f'y[(q / {width}LL) * span + offset + '
          f'static_cast<long long>(blockIdx.y) * {width}LL + q % {width}LL]'
          f' = static_cast<{output_type}>(value);\n}}')
    kernel = _raw_kernel(factory, source)
    grid, block = ((size + 255) // 256, stop - start), (256,)
    padding = (np.uint64(0),) * (_MAX_RAW_INPUTS - (stop - start))
    tail = (*padding, np.int64(span), np.int64(offset))
    native = bind_pointer_arguments(backend, kernel, grid, block,
        (*(None,) * (stop - start), *tail, None),
        tuple((lane, index) for lane, index in enumerate(range(start, stop))),
        pointer_getter=_array_pointer)
    return kernel, native, start, stop, size, grid, block, tail


def _mixed_stack_packing_call(backend, factory, shape, strides, axis, dtype, output_dtype,
                              span, offset, start, stop):
    """Share equal-shape copies across per-lane strides, with static divisors.

    Shape-one coordinates are zero. Adjacent coordinates can share one stride
    only when every lane proves the same signed contiguous relation. This is
    done on a layout-cache miss; the uniform CTA lane then selects one pointer
    and its remaining strides from host argument slots, never a device table.
    """
    groups = []
    for source_axis, dimension in enumerate(shape):
        if dimension == 1:
            continue
        if groups and all(steps[groups[-1][1]] == dimension * steps[source_axis]
                          for steps in strides):
            groups[-1] = (groups[-1][0] * dimension, source_axis)
        else:
            groups.append((dimension, source_axis))
    axes = tuple(source_axis for _, source_axis in groups)
    if 8 * (_MAX_RAW_INPUTS * (1 + len(axes)) + 3) > _MAX_ARGUMENT_BYTES:
        return None
    size, width = prod(shape), prod(shape[axis + 1:])
    input_type, output_type = _TYPES[dtype], _TYPES[output_dtype]
    params = [f'const {input_type}* x{i}' for i in range(_MAX_RAW_INPUTS)]
    params += [f'long long t{lane}_{index}' for lane in range(_MAX_RAW_INPUTS)
               for index in range(len(axes))]
    params += ['long long span', 'long long offset', f'{output_type}* y']
    choices = '\n'.join(f'case {lane}: input = x{lane}; '
        + ' '.join(f's{index} = t{lane}_{index};' for index in range(len(axes)))
        + ' break;' for lane in range(_MAX_RAW_INPUTS))
    terms = []
    for index, (dimension, _) in enumerate(groups):
        divisor = prod(size for size, _ in groups[index + 1:])
        coordinate = 'q' if divisor == 1 else f'(q / {divisor}LL)'
        if index:
            coordinate = f'({coordinate} % {dimension}LL)'
        terms.append(f'{coordinate} * s{index}')
    address = ' + '.join(terms) or '0LL'
    source = ('#include <cupy/complex.cuh>\n'
        'extern "C" __global__ void ed_dense_batch_pack_raw('
        + ', '.join(params) + ') {\n'
        'const long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
        + f'if (q >= {size}LL) return;\n'
        + f'const {input_type}* input;\n'
        + ''.join(f'long long s{index};\n' for index in range(len(axes)))
        + 'switch (blockIdx.y) {\n' + choices + '\ndefault: return;\n}\n'
        + f'const auto value = *reinterpret_cast<const {input_type}*>('
          f'reinterpret_cast<const char*>(input) + ({address}));\n'
        + f'y[(q / {width}LL) * span + offset + '
          f'static_cast<long long>(blockIdx.y) * {width}LL + q % {width}LL]'
          f' = static_cast<{output_type}>(value);\n}}')
    kernel = _raw_kernel(factory, source)
    grid, block = ((size + 255) // 256, stop - start), (256,)
    count = stop - start
    padding = (np.uint64(0),) * (_MAX_RAW_INPUTS - count)
    stride_words = tuple(np.int64(strides[lane][index] if lane < count else 0)
                         for lane in range(_MAX_RAW_INPUTS) for index in axes)
    tail = (*padding, *stride_words, np.int64(span), np.int64(offset))
    native = bind_pointer_arguments(backend, kernel, grid, block,
        (*(None,) * count, *tail, None),
        tuple((lane, index) for lane, index in enumerate(range(start, stop))),
        pointer_getter=_array_pointer)
    return kernel, native, start, stop, size, grid, block, tail


def _raw_packing(backend, shapes, dtypes, axis, output_dtype, input_layouts=None):
    factory = getattr(backend, 'RawKernel', None)
    output_dtype = np.dtype(output_dtype)
    dtypes = tuple(np.dtype(t) for t in dtypes)
    if (not callable(factory) or output_dtype not in _TYPES or any(t not in _TYPES for t in dtypes)
            or output_dtype.kind not in 'fc'
            or output_dtype.kind != 'c' and any(t.kind == 'c' for t in dtypes)):
        return None
    inner = prod(shapes[0][axis+1:])
    span = sum(shape[axis] for shape in shapes) * inner
    variants = OrderedDict()
    equal_stack = (shapes[0][axis] == 1 and prod(shapes[0]) > 0
                   and all(tuple(shape) == tuple(shapes[0]) for shape in shapes)
                   and len(set(dtypes)) == 1)

    def bind(strides):
        calls, offset = [], 0
        for start in range(0, len(shapes), _MAX_RAW_INPUTS):
            stop = min(start + _MAX_RAW_INPUTS, len(shapes))
            if equal_stack and all(step == strides[start] for step in strides[start:stop]):
                calls.append(_stack_packing_call(backend, factory, shapes[start], strides[start],
                    axis, dtypes[start], output_dtype, span, offset, start, stop))
                offset += (stop - start) * inner
                continue
            if equal_stack:
                call = _mixed_stack_packing_call(backend, factory, shapes[start],
                    strides[start:stop], axis, dtypes[start], output_dtype,
                    span, offset, start, stop)
                if call is not None:
                    calls.append(call)
                    offset += (stop - start) * inner
                    continue
            params, body, end = [], [], 0
            for lane, index in enumerate(range(start, stop)):
                params.append(f'const {_TYPES[dtypes[index]]}* x{lane}')
                shape = shapes[index]
                size, width = prod(shape), shape[axis] * inner
                if size:
                    divisions = [prod(shape[i+1:]) for i in range(len(shape))]
                    address = ' + '.join(f'((q / {div}LL) % {dim}LL) * ({step}LL)'
                        for div, dim, step in zip(divisions, shape, strides[index])) or '0LL'
                    body.append(f'{"else " if body else ""}if (i < {end+size}LL) {{'
                        f' const long long q = i - {end}LL;'
                        f' const auto value = *reinterpret_cast<const {_TYPES[dtypes[index]]}*>('
                        f' reinterpret_cast<const char*>(x{lane}) + ({address}));'
                        f' y[(q / {width}LL) * {span}LL + {offset}LL + q % {width}LL]'
                        f' = static_cast<{_TYPES[output_dtype]}>(value); }}')
                end += size
                offset += width
            params.append(f'{_TYPES[output_dtype]}* y')
            # The launch size is fixed by the frozen descriptor, not an input.
            source = ('#include <cupy/complex.cuh>\n'
                'extern "C" __global__ void ed_dense_batch_pack_raw('
                + ', '.join(params) + ') {\n'
                'const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
                + f'if (i >= {end}LL) return;\n' + '\n'.join(body) + '\n}')
            kernel = _raw_kernel(factory, source) if end else None
            grid, block = ((end + 255) // 256,), (256,)
            native = (bind_pointer_arguments(backend, kernel, grid, block,
                (None,) * (stop - start + 1),
                tuple((lane, index) for lane, index in enumerate(range(start, stop))),
                pointer_getter=_array_pointer) if kernel is not None else None)
            calls.append((kernel, native, start, stop, end, grid, block, ()))
        return tuple(calls)

    fallback = None

    def pack_strides(values, destination, strides, *, original=False):
        nonlocal fallback
        if not destination.flags.c_contiguous:
            if fallback is None:
                fallback = _elementwise_packing(backend, shapes, dtypes, axis, output_dtype)
            return fallback(tuple(value.array() if type(value) is _DeferredResultView
                                  else value for value in values), destination)
        # Only live layout metadata varies. Neither cached programs nor their
        # signatures capture numerical arrays, base pointers or cfg identities.
        # Layouts are fixed for this binding. Transform raw byte strides only
        # on a miss, and distinguish logical views from original pointers even
        # when their stride tuples happen to match.
        key = (original, strides)
        calls = variants.get(key)
        if calls is None:
            logical_strides = (tuple(tuple(0 if source_axis is None else steps[source_axis]
                                          for source_axis in layout)
                                     for steps, layout in zip(strides, input_layouts))
                               if original else strides)
            calls = bind(logical_strides)
            variants[key] = calls
            if len(variants) > 8:
                variants.popitem(last=False)
        else:
            variants.move_to_end(key)
        for kernel, native, start, stop, size, grid, block, tail in calls:
            if size:
                if native is not None and not debug_enabled():
                    native(values, destination)
                else:
                    kernel(grid, block, (*_pointer_arguments(values[start:stop]), *tail, destination))
        return destination

    def pack(values, destination):
        return pack_strides(values, destination, tuple(value.strides for value in values))

    # Reporting metadata only. Non-C destinations use the separately chunked
    # Elementwise fallback, so this bound explicitly requires a C destination.
    pack.c_contiguous_kernel_calls_upper_bound = raw_packing_kernel_calls_upper_bound(len(shapes))
    if input_layouts is not None:
        def pack_original(values, destination):
            if not destination.flags.c_contiguous:
                raise ValueError('original-input raw packing requires a C-contiguous destination')
            return pack_strides(values, destination, tuple(value.strides for value in values),
                                original=True)

        pack_original.c_contiguous_kernel_calls_upper_bound = pack.c_contiguous_kernel_calls_upper_bound
        pack.pack_original = pack_original
    return pack


def bind_partition_packing(backend, shapes, dtypes, strides, offsets, output_dtype):
    """Copy independent fixed layouts to disjoint, prebudgeted byte partitions.

    Shapes/strides describe the desired C iteration order, including arbitrary
    signed source strides. Offsets and partition sizes are proved at binding;
    invocation receives only current pointers and a caller-owned C arena.
    """
    factory = getattr(backend, 'RawKernel', None)
    output_dtype = np.dtype(output_dtype)
    dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
    if (getattr(backend, '__name__', '') != 'cupy' or not callable(factory)
            or output_dtype not in _TYPES or output_dtype.kind not in 'fc'
            or any(dtype not in _TYPES for dtype in dtypes)
            or output_dtype.kind != 'c' and any(dtype.kind == 'c' for dtype in dtypes)):
        return None
    if (not 0 < len(shapes) <= _MAX_RAW_INPUTS or len(shapes) != len(dtypes)
            or len(shapes) != len(strides) or len(shapes) != len(offsets)):
        raise ValueError('partition packing requires aligned fixed input layouts')
    sizes = tuple(prod(shape)*output_dtype.itemsize for shape in shapes)
    if any(type(offset) is not int or offset < 0 or offset % output_dtype.itemsize
           for offset in offsets):
        raise ValueError('partition packing offsets must be aligned nonnegative bytes')
    ranges = sorted(zip(offsets,sizes))
    if any(start+size > following for (start,size),(following,_) in zip(ranges,ranges[1:])):
        raise ValueError('partition packing destinations overlap')
    required = max(offset+size for offset,size in zip(offsets,sizes))
    params,body,end = [],[],0
    for lane,(shape,steps,dtype,offset) in enumerate(zip(shapes,strides,dtypes,offsets)):
        if len(shape) != len(steps) or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError('partition packing requires positive fixed dimensions and byte strides')
        size = prod(shape)
        address = _stack_address(shape,steps)
        params.append(f'const char* x{lane}')
        body.append(f'{"else " if body else ""}if (i < {end+size}LL) {{'
            f' const long long q = i - {end}LL;'
            f' const auto value = *reinterpret_cast<const {_TYPES[dtype]}*>(x{lane}+({address}));'
            f' y[{offset//output_dtype.itemsize}LL+q] = static_cast<{_TYPES[output_dtype]}>(value); }}')
        end += size
    if (end+255)//256 > 2**31-1:
        return None
    params.append(f'{_TYPES[output_dtype]}* y')
    source = ('#include <cupy/complex.cuh>\n'
        'extern "C" __global__ void ed_dense_batch_pack_raw('+', '.join(params)+') {\n'
        'const long long i = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;\n'
        +'\n'.join(body)+'\n}')
    kernel = _raw_kernel(factory,source)
    grid,block = ((end+255)//256,),(256,)
    native = bind_pointer_arguments(backend,kernel,grid,block,(None,)*(len(shapes)+1),
        tuple((i,i) for i in range(len(shapes))),pointer_getter=_array_pointer)

    def pack(values,destination):
        if (destination.dtype != output_dtype or not destination.flags.c_contiguous
                or destination.nbytes < required):
            raise ValueError('partition packing arena differs from its admitted extent')
        if native is not None and not debug_enabled():
            native(values,destination)
        else:
            kernel(grid,block,(*_pointer_arguments(values),destination))
        return destination

    return pack


def bind_packing(backend, shapes, dtypes, axis, output_dtype, *, input_layouts=None):
    """Bind bounded raw-array copies; storage is the saved packing workspace.

    Stride variants carry code only; transposed and reversed inputs need no
    extra contiguous snapshots. No pointer table or numerical input is kept.
    """
    factory = getattr(backend, 'ElementwiseKernel', None)
    if getattr(backend, '__name__', '') != 'cupy' or not callable(factory):
        return None
    if any(np.dtype(t).kind not in 'fciub' for t in (*dtypes, output_dtype)):
        return None
    raw = _raw_packing(backend, shapes, dtypes, axis, output_dtype, input_layouts)
    if raw is not None:
        return raw
    return _elementwise_packing(backend, shapes, dtypes, axis, output_dtype)


def bind_gemm_group_packing(backend, sides, catalog, matrix_shape, dtype):
    """Coalesce one saved group's required copies inside its packing allowance.

    Return None for unsupported descriptors, or per call when an existing owner
    can already supply a child without copying. No input or buffer is cached.
    """
    factory = getattr(backend, 'RawKernel', None)
    dtype = np.dtype(dtype)
    if (getattr(backend, '__name__', '') != 'cupy' or not callable(factory)
            or dtype not in _TYPES or dtype.kind not in 'fc'
            or len(sides) < 2 or any(side.prepared or side.original_packer is None
                                   or len(side.operands) < 2 for side in sides)):
        return None
    elements = prod(matrix_shape)
    members, inputs = [], []
    for child, side in enumerate(sides):
        if prod(side.shape) != elements or side.input_layouts is None:
            return None
        inner = prod(side.shape[side.axis + 1:])
        span, offset = side.shape[side.axis] * inner, 0
        for (key, _), layout in zip(side.operands, side.input_layouts, strict=True):
            spec = catalog[key]
            input_dtype = np.dtype(spec.dtype)
            if input_dtype not in _TYPES or dtype.kind != 'c' and input_dtype.kind == 'c':
                return None
            shape = tuple(1 if axis is None else spec.shape[axis] for axis in layout)
            width = shape[side.axis] * inner
            inputs.append((key, layout, shape, input_dtype, child * elements + offset,
                           width, span))
            offset += width
        members.append(side)
    member_values = tuple(_bind_packing_values(side.operands) for side in members)
    variants = OrderedDict()
    block = (256,)

    def bind_chunk(strides, start, stop):
        params, body, end = [], [], 0
        for index, position in enumerate(range(start, stop)):
            _, _, shape, input_dtype, offset, width, span = inputs[position]
            steps = strides[position]
            params.append(f'const {_TYPES[input_dtype]}* x{index}')
            size = prod(shape)
            if size:
                address = ' + '.join(
                    f'((q / {prod(shape[axis + 1:])}LL) % {dimension}LL) * ({step}LL)'
                    for axis, (dimension, step) in enumerate(zip(shape, steps, strict=True))) or '0LL'
                body.append(f'{"else " if body else ""}if (i < {end + size}LL) {{'
                    f' const long long q = i - {end}LL;'
                    f' const auto value = *reinterpret_cast<const {_TYPES[input_dtype]}*>('
                    f' reinterpret_cast<const char*>(x{index}) + ({address}));'
                    f' y[{offset}LL + (q / {width}LL) * {span}LL + q % {width}LL]'
                    f' = static_cast<{_TYPES[dtype]}>(value); }}')
            end += size
        params.append(f'{_TYPES[dtype]}* y')
        source = ('#include <cupy/complex.cuh>\n'
            'extern "C" __global__ void ed_dense_batch_pack_raw('
            + ', '.join(params) + ') {\n'
            'const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
            + f'if (i >= {end}LL) return;\n' + '\n'.join(body) + '\n}')
        kernel = _raw_kernel(factory, source) if end else None
        grid = ((end + 255) // 256,)
        native = (bind_pointer_arguments(backend, kernel, grid, block,
            (None,) * (stop - start + 1),
            tuple((index, position) for index, position in enumerate(range(start, stop))),
            pointer_getter=_array_pointer) if kernel is not None else None)
        return kernel, grid, native, start, stop

    def bind(strides):
        # Every chunk writes original global child offsets in the same owner;
        # a boundary may split a child without allocating intermediate cubes.
        return tuple(bind_chunk(strides, start, min(start + _MAX_RAW_INPUTS, len(inputs)))
                     for start in range(0, len(inputs), _MAX_RAW_INPUTS))

    def pack_buffer(arrays):
        # Preserve every existing zero-copy opportunity before allocating.
        values = []
        for side, get_values in zip(members, member_values):
            current = get_values(arrays)
            if side._borrow_original(current, dtype) is not None:
                return None
            values.extend(current)
        strides = tuple(value.strides for value in values)
        call = variants.get(strides)
        if call is None:
            logical_strides = tuple(tuple(0 if axis is None else steps[axis] for axis in layout)
                                    for steps, (_, layout, *_) in zip(strides, inputs, strict=True))
            call = variants[strides] = bind(logical_strides)
            if len(variants) > 8:
                variants.popitem(last=False)
        else:
            variants.move_to_end(strides)
        destination = backend.empty((len(sides), *matrix_shape), dtype=dtype, order='C')
        if not destination.flags.c_contiguous:
            return None
        for kernel, grid, native, start, stop in call:
            if kernel is not None:
                if native is not None and not debug_enabled():
                    native(values, destination)
                else:
                    kernel(grid, block, (*_pointer_arguments(values[start:stop]), destination))
        return destination

    def pack(arrays):
        destination = pack_buffer(arrays)
        if destination is None:
            return None
        return tuple(destination[index] for index in range(len(sides)))

    # The grouped executor needs the whole owner, not per-child matrix views.
    # Retain the view handoff for other private consumers of this binding.
    pack.pack_buffer = pack_buffer

    return pack


def _elementwise_packing(backend, shapes, dtypes, axis, output_dtype):
    factory = backend.ElementwiseKernel
    rank = len(shapes[0])
    count = min(_MAX_LEGACY_INPUTS,
                (_MAX_ARGUMENT_BYTES - (32 + 16 * rank)) // (16 + 16 * rank))
    if count < 1:
        return None
    inner = prod(shapes[0][axis + 1:])
    span = sum(s[axis] for s in shapes) * inner
    calls, offset = [], 0
    for start in range(0, len(shapes), count):
        stop = min(start + count, len(shapes))
        params, code, end = [], [], 0
        for lane, index in enumerate(range(start, stop)):
            size, width = prod(shapes[index]), shapes[index][axis] * inner
            params.append(f'raw {np.dtype(dtypes[index]).name} x{lane}')
            if size:
                code.append(f'{"else " if code else ""}if (i < {end + size}LL) {{'
                            f' const long long q = i - {end}LL;'
                            f' y[(q / {width}LL) * {span}LL + {offset}LL + q % {width}LL] = x{lane}[q]; }}')
            end += size
            offset += width
        kernel = factory(', '.join(params), f'raw {np.dtype(output_dtype).name} y',
                         '\n'.join(code), 'ed_dense_batch_pack') if end else None
        calls.append((kernel, start, stop, end))

    def pack(values, destination):
        for kernel, start, stop, size in calls:
            if size:
                kernel(*values[start:stop], destination, size=size)
        return destination
    pack.c_contiguous_kernel_calls_upper_bound = sum(bool(size) for _, _, _, size in calls)
    return pack
