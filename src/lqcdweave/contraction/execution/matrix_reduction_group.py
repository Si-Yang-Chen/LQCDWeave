"""Bounded same-geometry ik,jk->ij submission groups for prepared matrices.

Inputs are already prepared matrices. No Cartesian packing, plan annotation,
cfg publication or additional resource admission is performed here.
"""
from functools import lru_cache

import numpy as np

from .diagnostics import debug_enabled
from .raw_arguments import bind_pointer_arguments
from .reduction_group_runtime import bind_group_runtime, MAX_TASKS, FAILED_SUBMISSIONS
from .reduction_kernel import (_THREADS, _wide_parts, tiled_matrix_reduction_source,
                               matrix_reduction_sources)


@lru_cache(maxsize=2)
def matrix_group_sources(dtype):
    """Keep the production tile body intact; sources are shared by dtype only."""
    narrow, wide = _wide_parts(dtype)[:2]
    parameters, cases = [], []
    for task in range(MAX_TASKS):
        parameters.extend((f'const char* a{task}', f'const char* b{task}'))
        parameters.extend(f'const long long {stride}_{task}' for stride in ('lm', 'lk', 'rn', 'rk'))
        cases.append(f'case {task}: p0=a{task}; p1=b{task}; lm=lm_{task}; lk=lk_{task}; '
                     f'rn=rn_{task}; rk=rk_{task}; break;')
    # Reuse the complete tested arithmetic body. Only pointer selection,
    # output-grid decoding and the child partial offset are substituted.
    # For d=ceil(m/2), z=blockIdx.z, magic=ceil(2**32/d):
    # z*(d-1) < 2**32 under admission, so (z*magic)>>32 is exactly z//d.
    # Its product fits uint64; row and signed byte/partial offsets stay int64.
    tile = tiled_matrix_reduction_source(dtype)
    body = tile[tile.index('    const int t = threadIdx.x;'):]
    body = body.replace('const long long row = 2LL * blockIdx.z, col = 2LL * blockIdx.y, split = blockIdx.x;',
        'const unsigned int tile_rows = (static_cast<unsigned int>(m) + 1U) >> 1;\n'
        '    const unsigned int task = (static_cast<unsigned long long>(blockIdx.z) * magic) >> 32;\n'
        '    const long long row = 2LL * (blockIdx.z - task * tile_rows), col = 2LL * blockIdx.y, split = blockIdx.x;\n'
        '    const char* p0 = nullptr; const char* p1 = nullptr;\n'
        '    long long lm, lk, rn, rk;\n'
        f'    switch (task) {{ {" ".join(cases)} default: return; }}\n'
        '    partials += task * m * n * blocks;')
    first = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_matrix_group_partial(
    {', '.join(parameters)}, const long long k, const long long m, const long long n,
    const long long blocks, const unsigned long long magic, {wide}* partials) {{
{body}'''
    finish = matrix_reduction_sources(dtype)[1]
    body = finish[finish.index('    const int t = threadIdx.x;'):]
    select = ' '.join(f'case {task}: result = out{task}; break;' for task in range(MAX_TASKS))
    body = body.replace('const long long o = blockIdx.x;',
        'const long long o = blockIdx.x, task = blockIdx.y;\n'
        f'    {narrow}* result = nullptr;\n'
        f'    switch (task) {{ {select} default: return; }}\n'
        '    partials += task * outputs * blocks;')
    outputs = ', '.join(f'{narrow}* out{task}' for task in range(1,MAX_TASKS))
    second = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_matrix_group_finish(
    const {wide}* partials, {outputs}, const long long blocks,
    const long long outputs, {narrow}* out0) {{
{body}'''
    return first, second


def bind_matrix_reduction_group(shapes, dtype, allowances, backend, *, share_small_outputs=False):
    """Bind 2..8 already-packed same-shaped matrix pairs, or return None.

    Each allowance is the original precise-batch input/output scratch budget.
    Original block counts must agree; a caller must additionally admit all
    simultaneous inputs, child output payloads and summed partials.
    """
    dtype = np.dtype(dtype)
    if (debug_enabled() or getattr(backend, '__name__', '') != 'cupy'
            or not callable(getattr(backend, 'RawKernel', None))
            or dtype not in (np.dtype('float32'), np.dtype('complex64'))):
        return None
    is_hip = getattr(getattr(getattr(backend, 'cuda', None), 'runtime', None), 'is_hip', False)
    if is_hip() if callable(is_hip) else is_hip:
        return None
    shapes = tuple(tuple(shape) for shape in shapes)
    allowances = tuple(allowances)
    if (len(shapes) != 2 or any(len(shape) != 2 for shape in shapes)
            or any(type(size) is not int or size <= 0 for shape in shapes for size in shape)
            or shapes[0][1] != shapes[1][1]
            or not 2 <= len(allowances) <= MAX_TASKS
            or any(type(n) is not int or n < 0 for n in allowances)):
        return None
    (m, k), (n, _) = shapes
    count, outputs = len(allowances), m*n
    tile_rows = (m+1)//2
    if (k < 2 or k > np.iinfo(np.int64).max or m > 65535 or n > 65535
            or count*tile_rows > 65535 or (n+1)//2 > 65535 or outputs > 2**31-1):
        return None
    wide = np.dtype('complex128' if dtype.kind == 'c' else 'float64')
    splits = tuple(min(1024, (k+_THREADS-1)//_THREADS,
                       (allowance-outputs*dtype.itemsize)//(outputs*wide.itemsize))
                   for allowance in allowances)
    blocks = splits[0]
    if blocks < 1 or len(set(splits)) != 1 or outputs*blocks >= 2**31:
        return None
    magic = ((1 << 32) + tile_rows - 1)//tile_rows
    return bind_group_runtime(backend, shapes, dtype, allowances, blocks, (m,n),
        matrix_group_sources(dtype.name),
        ('ed_precise_matrix_group_partial', 'ed_precise_matrix_group_finish'),
        ((blocks,(n+1)//2,count*tile_rows), (outputs,count)),
        ((k,m,n,blocks,magic), (blocks,outputs)), bind_pointer_arguments,
        share_small_outputs=share_small_outputs)
