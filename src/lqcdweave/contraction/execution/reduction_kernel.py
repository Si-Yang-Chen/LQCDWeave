"""Two-stage wide-precision GPU reductions, without casts or packing kernels."""
from collections import OrderedDict
from functools import lru_cache
from math import prod

import numpy as np

from .raw_arguments import bind_pointer_arguments


_THREADS = 256
# Saved Cartesian producers can expose more than eight row pitches to one
# scalar-reduction binding. Retain bounded metadata, not numerical owners, so
# warm replay need not rebuild raw arguments and context setup at each cycle.
_MAX_LAYOUTS = 64


def _coordinate(label, labels, dimensions, index):
    position = labels.index(label)
    tail = prod(dimensions[key] for key in labels[position + 1:])
    return f'(({index} / {tail}LL) % {dimensions[label]}LL)'


def _offset(labels, strides, output, reduced, dimensions):
    return ' + '.join(f'({stride}LL * {_coordinate(label, output if label in output else reduced, dimensions, "o" if label in output else "i")})'
                      for label, stride in zip(labels, strides, strict=True)) or '0LL'


def _wide_parts(dtype):
    """One accumulation rule for specialized and runtime-geometry sources."""
    complex_values = np.dtype(dtype).kind == 'c'
    double_input = np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128'))
    wide = 'complex<double>' if complex_values else 'double'
    narrow = wide if double_input else 'complex<float>' if complex_values else 'float'
    declaration = 'double total_re = 0.0;' + (' double total_im = 0.0;' if complex_values else '')
    # Every CTA has eight complete CUDA warps. Reduce within each warp in
    # registers, then exchange only eight sums. This needs one block barrier
    # instead of one for each of the eight steps of a shared-memory tree.
    shared = '__shared__ double shared_re[8];' + (' __shared__ double shared_im[8];' if complex_values else '')
    accumulate = ('total_re += term.real(); total_im += term.imag();' if complex_values else 'total_re += term;')
    combine = 'total_re += __shfl_down_sync(0xffffffffU, total_re, step);' + (
        ' total_im += __shfl_down_sync(0xffffffffU, total_im, step);' if complex_values else '')
    store = 'shared_re[warp] = total_re;' + (' shared_im[warp] = total_im;' if complex_values else '')
    load = 'total_re = lane < 8 ? shared_re[lane] : 0.0;' + (
        ' total_im = lane < 8 ? shared_im[lane] : 0.0;' if complex_values else '')
    tree = f'''const int lane = t & 31;
    const int warp = t >> 5;
    for (int step = 16; step; step >>= 1) {{ {combine} }}
    if (lane == 0) {{ {store} }}
    __syncthreads();
    if (warp == 0) {{
        {load}
        for (int step = 4; step; step >>= 1) {{ {combine} }}
    }}'''
    wide_result = 'complex<double>(total_re, total_im)' if complex_values else 'total_re'
    final_result = (wide_result if double_input
                    else 'complex<float>(float(total_re), float(total_im))' if complex_values else 'float(total_re)')
    return narrow, wide, declaration, shared, accumulate, tree, wide_result, final_result


def reduction_sources(axes, output, shapes, strides, dtype, blocks):
    """Static geometry for the general strided/diagonal reduction route."""
    dimensions = {label: size for labels, shape in zip(axes, shapes, strict=True)
                  for label, size in zip(labels, shape, strict=True)}
    reduced = tuple(label for label in dimensions if label not in output)
    size = prod(dimensions[label] for label in reduced)
    outputs = prod(dimensions[label] for label in output)
    narrow, wide, declaration, shared, accumulate, tree, wide_result, final_result = _wide_parts(dtype)
    parameters = ', '.join(f'const char* p{index}' for index in range(len(axes)))
    loads = [f'{wide} a{index} = {wide}(*reinterpret_cast<const {narrow}*>(p{index} + ({_offset(labels, steps, output, reduced, dimensions)})));'
             for index, (labels, steps) in enumerate(zip(axes, strides, strict=True))]
    expression = ' * '.join(f'a{index}' for index in range(len(axes)))
    load_source = '\n        '.join(loads)
    first = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_partial({parameters}, {wide}* partials) {{
    const int t = threadIdx.x;
    const long long o = blockIdx.x / {blocks}LL;
    const long long b = blockIdx.x % {blocks}LL;
    {shared}
    {declaration}
    for (long long i = b * 256LL + t; i < {size}LL; i += {blocks * _THREADS}LL) {{
        {load_source}
        {wide} term = {expression};
        {accumulate}
    }}
    {tree}
    if (t == 0) partials[o * {blocks}LL + b] = {wide_result};
}}
'''
    finish = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_finish(const {wide}* partials, {narrow}* result) {{
    const int t = threadIdx.x;
    const long long o = blockIdx.x;
    {shared}
    {declaration}
    for (long long i = t; i < {blocks}LL; i += 256LL) {{
        {wide} term = partials[o * {blocks}LL + i];
        {accumulate}
    }}
    {tree}
    if (t == 0) result[o] = {final_result};
}}
'''
    return first, finish, outputs


def _matrix_axes(axes, output):
    """Find independent row/column axes; exclude diagonals and aligned axes."""
    if len(axes) != 2 or len(output) != 2 or any(len(labels) != 2 for labels in axes):
        return None
    reduced = set(axes[0]) & set(axes[1])
    if len(reduced) != 1 or reduced & set(output):
        return None
    key = next(iter(reduced))
    if set(axes[0]) != {output[0], key} or set(axes[1]) != {key, output[1]}:
        return None
    return axes[0].index(output[0]), axes[0].index(key), axes[1].index(key), axes[1].index(output[1])


@lru_cache(maxsize=2)
def matrix_reduction_sources(dtype):
    """Two dtype-only sources, shared across matrix sizes and byte strides.

    Output row/column use grid.z/grid.y. This avoids runtime integer division
    by matrix dimensions or split count, unlike flattening the whole grid.
    """
    narrow, wide, declaration, shared, accumulate, tree, wide_result, final_result = _wide_parts(dtype)
    first = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_matrix_partial(
    const char* p0, const char* p1, const long long k, const long long n,
    const long long blocks, const long long lm, const long long lk,
    const long long rk, const long long rn, {wide}* partials) {{
    const int t = threadIdx.x;
    const long long row = blockIdx.z, col = blockIdx.y, b = blockIdx.x;
    const long long o = row * n + col;
    {shared}
    {declaration}
    for (long long i = b * 256LL + t; i < k; i += blocks * 256LL) {{
        {wide} a = {wide}(*reinterpret_cast<const {narrow}*>(p0 + row * lm + i * lk));
        {wide} b_value = {wide}(*reinterpret_cast<const {narrow}*>(p1 + i * rk + col * rn));
        {wide} term = a * b_value;
        {accumulate}
    }}
    {tree}
    if (t == 0) partials[o * blocks + b] = {wide_result};
}}
'''
    finish = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_matrix_finish(
    const {wide}* partials, const long long blocks, {narrow}* result) {{
    const int t = threadIdx.x;
    const long long o = blockIdx.x;
    {shared}
    {declaration}
    for (long long i = t; i < blocks; i += 256LL) {{
        {wide} term = partials[o * blocks + i];
        {accumulate}
    }}
    {tree}
    if (t == 0) result[o] = {final_result};
}}
'''
    return first, finish


@lru_cache(maxsize=2)
def tiled_matrix_reduction_source(dtype):
    """Reuse four row values in registers without changing any output's sum.

    Each physical thread still represents the same one of 256 logical lanes
    for every output. Only independent output accumulators are interleaved;
    K partitions, arithmetic order and both levels of the tree remain intact.
    """
    narrow, wide, declaration, shared, accumulate, tree, wide_result, _ = _wide_parts(dtype)
    before, after = tree.split('__syncthreads();')
    declarations, products, warp_sums, block_sums = [], [], [], []
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        suffix = f'_{r}{c}'
        valid = ('true' if r == c == 0 else
                 'has_col' if r == 0 else 'has_row' if c == 0 else 'has_row && has_col')

        def named(text):
            for name in ('total_re', 'total_im', 'shared_re', 'shared_im'):
                text = text.replace(name, name + suffix)
            return text

        declarations.append(named(shared + '\n' + declaration))
        products.append(f'if ({valid}) {{ {wide} term = a{r} * b{c}; {named(accumulate)} }}')
        warp_sums.append(f'if ({valid}) {{ {named(before)} }}')
        block_sums.append(f'''if ({valid}) {{
        const int lane = t & 31, warp = t >> 5;
        {named(after)}
        if (t == 0) partials[((row + {r}) * n + col + {c}) * blocks + split] = {named(wide_result)};
    }}''')
    return f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_matrix_tile_partial(
    const char* p0, const char* p1, const long long k, const long long m, const long long n,
    const long long blocks, const long long lm, const long long lk,
    const long long rk, const long long rn, {wide}* partials) {{
    const int t = threadIdx.x;
    const long long row = 2LL * blockIdx.z, col = 2LL * blockIdx.y, split = blockIdx.x;
    const bool has_row = row + 1 < m, has_col = col + 1 < n;
    {chr(10).join(declarations)}
    for (long long i = split * 256LL + t; i < k; i += blocks * 256LL) {{
        {wide} a0 = {wide}(*reinterpret_cast<const {narrow}*>(p0 + row * lm + i * lk));
        {wide} a1 = has_row ? {wide}(*reinterpret_cast<const {narrow}*>(p0 + row * lm + i * lk + lm)) : {wide}(0.0);
        {wide} b0 = {wide}(*reinterpret_cast<const {narrow}*>(p1 + i * rk + col * rn));
        {wide} b1 = has_col ? {wide}(*reinterpret_cast<const {narrow}*>(p1 + i * rk + col * rn + rn)) : {wide}(0.0);
        {chr(10).join(products)}
    }}
    {chr(10).join(warp_sums)}
    __syncthreads();
    {chr(10).join(block_sums)}
}}
'''


@lru_cache(maxsize=64)
def _kernel(factory, source, name):
    return factory(source, name, options=('--std=c++17', '--fmad=false'))


def _bind_input_strides(shapes, dtype):
    """Validate current geometry and read current strides without retaining inputs."""
    if len(shapes) == 2:
        left_shape, right_shape = shapes

        def read(values):
            left, right = values  # Exact input count, before any submission.
            if (left.shape != left_shape or left.dtype != dtype
                    or right.shape != right_shape or right.dtype != dtype):
                raise ValueError('precise reduction inputs differ from their bound geometry')
            return left.strides, right.strides
        return read

    def read(values):
        if any(value.shape != shape or value.dtype != dtype
               for value, shape in zip(values, shapes, strict=True)):
            raise ValueError('precise reduction inputs differ from their bound geometry')
        return tuple(value.strides for value in values)
    return read


def bind_raw_reduction(equation, shapes, dtype, backend, allowance):
    factory = getattr(backend, 'RawKernel', None)
    if (getattr(backend, '__name__', '') != 'cupy' or not callable(factory)
            or np.dtype(dtype) not in tuple(map(np.dtype, ('float32', 'complex64', 'float64', 'complex128')))):
        return None
    is_hip = getattr(getattr(getattr(backend, 'cuda', None), 'runtime', None), 'is_hip', False)
    if is_hip() if callable(is_hip) else is_hip:
        return None
    from .device import active_device_execution
    axes, output = equation.inputs, equation.output
    dimensions = {label: size for labels, shape in zip(axes, shapes, strict=True)
                  for label, size in zip(labels, shape, strict=True)}
    output_shape = tuple(dimensions[label] for label in output)
    outputs = prod(output_shape)
    size = prod(dimensions[label] for label in dimensions if label not in output)
    wide = np.dtype('complex128' if np.dtype(dtype).kind == 'c' else 'float64')
    blocks = min(1024, (size + _THREADS - 1) // _THREADS,
                 (allowance - outputs * np.dtype(dtype).itemsize) // (outputs * wide.itemsize))
    if blocks < 1 or outputs * blocks >= 2**31:
        return None
    matrix_axes = _matrix_axes(axes, output)
    if any(size > 65535 for size in output_shape):
        matrix_axes = None  # CUDA grid.y/grid.z limits; general route stays flat.
    tiled = matrix_axes is not None and outputs > 1
    layouts = OrderedDict()
    input_strides = _bind_input_strides(shapes, dtype)

    def bind_layout(strides):
        if matrix_axes is None:
            first, finish, _ = reduction_sources(axes, output, shapes, strides, dtype, blocks)
            first_name, finish_name = 'ed_precise_partial', 'ed_precise_finish'
            first_grid = (outputs * blocks,)
            first_constants = finish_constants = ()
        else:
            first, finish = matrix_reduction_sources(np.dtype(dtype).name)
            first_name, finish_name = 'ed_precise_matrix_partial', 'ed_precise_matrix_finish'
            lm, lk, rk, rn = matrix_axes
            if tiled:
                first = tiled_matrix_reduction_source(np.dtype(dtype).name)
                first_name = 'ed_precise_matrix_tile_partial'
            first_grid = ((blocks, (output_shape[1] + 1)//2, (output_shape[0] + 1)//2) if tiled
                          else (blocks, output_shape[1], output_shape[0]))
            sizes = (size, *output_shape) if tiled else (size, output_shape[1])
            first_constants = tuple(np.int64(value) for value in (*sizes, blocks,
                strides[0][lm], strides[0][lk], strides[1][rk], strides[1][rn]))
            finish_constants = (np.int64(blocks),)
        first_kernel = _kernel(factory, first, first_name)
        finish_kernel = _kernel(factory, finish, finish_name)
        second_grid, threads = (outputs,), (_THREADS,)
        positions = tuple((index, index) for index in range(len(axes)))
        first_arguments = bind_pointer_arguments(backend, first_kernel, first_grid, threads,
            (None,) * len(axes) + first_constants + (None,), positions)
        finish_arguments = bind_pointer_arguments(backend, finish_kernel, second_grid, threads,
            (None, *finish_constants, None), ((0, 0),))

        def call(values):
            partials = result = None
            try:
                device = active_device_execution()
                borrow = (getattr(device, 'reduction_partial', None)
                          if device is not None and device.backend is backend else None)
                if callable(borrow):
                    partials = borrow(outputs * blocks * wide.itemsize, wide)
                if partials is None:
                    partials = backend.empty((outputs, blocks), dtype=wide)
                result = backend.empty(output_shape, dtype=dtype)
                if first_arguments is None:
                    first_kernel(first_grid, threads, (*values, *first_constants, partials))
                else:
                    first_arguments(values, partials)
                if finish_arguments is None:
                    finish_kernel(second_grid, threads, (partials, *finish_constants, result))
                else:
                    finish_arguments((partials,), result)
                return result
            except BaseException as error:
                # Preserve input and temporary owners through failed submissions.
                try:
                    backend.cuda.get_current_stream().synchronize()
                except BaseException as cleanup:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'precise reduction cleanup also failed: {cleanup!r}')
                raise
        return call

    def run(*values):
        strides = input_strides(values)
        call = layouts.get(strides)
        if call is None:
            call = bind_layout(strides)
            layouts[strides] = call
            if len(layouts) > _MAX_LAYOUTS:
                layouts.popitem(last=False)
        else:
            layouts.move_to_end(strides)
        return call(values)
    run.workspace_bytes = outputs * (blocks * wide.itemsize + np.dtype(dtype).itemsize)
    run.workspace_allowance = allowance
    run.raw_reduction = True
    run.raw_source_family = 'matrix_runtime_geometry' if matrix_axes is not None else 'static_geometry'
    run.raw_output_tile = (2, 2) if tiled else (1, 1)
    return run
