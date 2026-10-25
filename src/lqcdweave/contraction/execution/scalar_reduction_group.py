"""Wide scalar submission grouping with shared host binding and ownership.

Only independent scalar dot products with one common geometry are represented.
The caller must prove dependencies, input residency and simultaneous workspace
admission. Child producer layouts and output owners are never replaced here.
"""
from functools import lru_cache
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern, scalar_reduction_group_limit
from .diagnostics import debug_enabled
from .raw_arguments import bind_pointer_arguments
from .reduction_kernel import _coordinate, _wide_parts, _THREADS
from .reduction_group_runtime import (bind_group_runtime, current_pointer as _pointer,
                                      FAILED_SUBMISSIONS as _FAILED_SUBMISSIONS)


_MAX_TASKS = 32


@lru_cache(maxsize=64)
def scalar_group_sources(axes, shapes, dtype, capacity=8):
    """Share sources across task counts, blocks and all runtime byte strides."""
    dimensions = {label: size for labels, shape in zip(axes, shapes, strict=True)
                  for label, size in zip(labels, shape, strict=True)}
    reduced = tuple(dimensions)
    narrow, wide, declaration, shared, accumulate, tree, wide_result, final_result = _wide_parts(dtype)
    strides = [f's{side}_{axis}' for side, labels in enumerate(axes) for axis in range(len(labels))]
    parameters, cases = [], []
    for task in range(capacity):
        parameters.extend((f'const char* a{task}', f'const char* b{task}'))
        parameters.extend(f'const long long {stride}_{task}' for stride in strides)
        assignments = ' '.join(f'{stride} = {stride}_{task};' for stride in strides)
        cases.append(f'case {task}: p0 = a{task}; p1 = b{task}; {assignments} break;')
    offsets = [' + '.join(f's{side}_{axis} * {_coordinate(label, reduced, dimensions, "i")}'
                         for axis, label in enumerate(labels)) for side, labels in enumerate(axes)]
    loads = '\n        '.join(f'{wide} v{side} = {wide}(*reinterpret_cast<const {narrow}*>(p{side} + ({offset})));'
                              for side, offset in enumerate(offsets))
    first = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_scalar_group_partial(
    {', '.join(parameters)}, const long long blocks, {wide}* partials) {{
    const int t = threadIdx.x;
    const long long task = blockIdx.y, b = blockIdx.x;
    const char* p0 = nullptr; const char* p1 = nullptr;
    long long {', '.join(strides)};
    switch (task) {{ {' '.join(cases)} default: return; }}
    {shared}
    {declaration}
    for (long long i = b * 256LL + t; i < {prod(dimensions.values())}LL; i += blocks * 256LL) {{
        {loads}
        {wide} term = v0 * v1;
        {accumulate}
    }}
    {tree}
    if (t == 0) partials[task * blocks + b] = {wide_result};
}}
'''
    # The ABI helper's last parameter is the output allocation. Keep output 0
    # there, with remaining independent outputs in ordinary pointer slots.
    outputs = ', '.join(f'{narrow}* out{i}' for i in range(1, capacity))
    select = ' '.join(f'case {i}: result = out{i}; break;' for i in range(capacity))
    finish = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_scalar_group_finish(
    const {wide}* partials, {outputs}, const long long blocks, {narrow}* out0) {{
    const int t = threadIdx.x;
    const long long task = blockIdx.x;
    {narrow}* result = nullptr;
    switch (task) {{ {select} default: return; }}
    {shared}
    {declaration}
    for (long long i = t; i < blocks; i += 256LL) {{
        {wide} term = partials[task * blocks + i];
        {accumulate}
    }}
    {tree}
    if (t == 0) result[0] = {final_result};
}}
'''
    return first, finish


def bind_scalar_reduction_group(equation, shapes, dtype, allowances, backend, *, share_small_outputs=False):
    """Bind 2..32 same-geometry scalar reductions, or return None for fallback.

    ``run(tasks)`` takes a sequence of independent ``(left, right)`` pairs and
    returns a tuple of non-overlapping scalar arrays. ``allowances``
    contains each original reduction's input-plus-output byte allowance. All
    children must choose the same original split count; it is never adjusted
    to make grouping fit. The caller admits their simultaneous workspace.
    """
    factory = getattr(backend, 'RawKernel', None)
    dtype = np.dtype(dtype)
    if (debug_enabled() or getattr(backend, '__name__', '') != 'cupy'
            or not callable(factory) or dtype not in (np.dtype('float32'), np.dtype('complex64'))):
        return None
    is_hip = getattr(getattr(getattr(backend, 'cuda', None), 'runtime', None), 'is_hip', False)
    if is_hip() if callable(is_hip) else is_hip:
        return None
    allowances = tuple(allowances)
    if not 2 <= len(allowances) <= _MAX_TASKS or any(type(n) is not int or n < 0 for n in allowances):
        return None
    equation = parse_axis_pattern(equation).normalized()
    axes, output = equation.inputs, equation.output
    shapes = tuple(tuple(shape) for shape in shapes)
    if (output or len(axes) != 2 or len(shapes) != 2 or not axes[0]
            or any(len(labels) != len(shape) or len(set(labels)) != len(labels)
                   for labels, shape in zip(axes, shapes, strict=True))
            or set(axes[0]) != set(axes[1])):
        return None
    dimensions = {}
    for labels, shape in zip(axes, shapes, strict=True):
        for label, size in zip(labels, shape, strict=True):
            if type(size) is not int or size <= 0 or dimensions.setdefault(label, size) != size:
                return None
    # Reuse the original fixed-eight ABI for every old group. Larger groups
    # use shared 16/32-slot sources, without specializing on the actual count.
    capacity = next(n for n in (8, 16, 32) if len(allowances) <= n)
    if capacity > scalar_reduction_group_limit(shapes):
        return None
    size = prod(dimensions.values())
    wide = np.dtype('complex128' if dtype.kind == 'c' else 'float64')
    splits = tuple(min(1024, (size + _THREADS - 1)//_THREADS,
                       (n - dtype.itemsize)//wide.itemsize) for n in allowances)
    if size < 2 or splits[0] < 1 or len(set(splits)) != 1:
        return None
    count, blocks = len(allowances), splits[0]
    return bind_group_runtime(backend, shapes, dtype, allowances, blocks, (),
        scalar_group_sources(axes, shapes, dtype.name, capacity),
        ('ed_precise_scalar_group_partial', 'ed_precise_scalar_group_finish'),
        ((blocks, count), (count,)), ((blocks,), (blocks,)), bind_pointer_arguments,
        share_small_outputs=share_small_outputs, capacity=capacity)
