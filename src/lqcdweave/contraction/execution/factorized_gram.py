"""Register reuse for an exclusive-left/right tiled wide Gram partial.

Only independent output points are interleaved. Each keeps the original
256 lanes, K partitions, input product and two-stage block reduction.
"""
from math import prod

from .reduction_kernel import _offset, _wide_parts


def matrix_gram_geometry(core, *, row_tile=2, column_tile=2):
    if core.tile_elements is None:
        return None
    axes, output, dims = core.equation.inputs, core.equation.output, core.dimensions
    a, b = (set(axes[i]) for i in core.large_operands)
    shared = tuple(x for x in core.gram_axes if x in output)
    inner = tuple(x for x in core.gram_axes if x not in output)
    left = tuple(x for x in inner if x in a - b)
    right = tuple(x for x in inner if x in b - a)
    # Internal aligned axes or interleaved left/right output order retain the
    # original general kernel. Output lanes can occur on either large input.
    if inner != left + right or not left or not right:
        return None
    rows, columns = prod(dims[x] for x in left), prod(dims[x] for x in right)
    if rows < row_tile or columns < column_tile or columns > core.tile_width:
        return None
    shared_count = prod(dims[x] for x in shared)
    intersecting_rows = min(rows, (core.tile_width + 2 * columns - 2) // columns)
    row_groups = (intersecting_rows + row_tile - 1) // row_tile
    column_groups = (columns + column_tile - 1) // column_tile
    if column_groups > 65535 or shared_count * row_groups > 65535:
        return None
    return shared, left, right, rows, columns, row_groups, column_groups


def matrix_gram_partial(core, strides, *, row_tile=2, column_tile=2):
    geometry = matrix_gram_geometry(core, row_tile=row_tile, column_tile=column_tile)
    if geometry is None:
        return None
    shared_axes, left, right, rows, columns, row_groups, column_groups = geometry
    dims, reduced, blocks, width = core.dimensions, core.reduced_axes, core.blocks, core.tile_width
    narrow, wide, declaration, shared, accumulate, tree, wide_result, _ = _wide_parts(core.dtype)
    before, after = tree.split('__syncthreads();')
    valid, declarations, products, warp_sums, block_sums = [], [], [], [], []
    for r in range(row_tile):
        for c in range(column_tile):
            suffix = f'_{r}{c}'
            point = f'(row + {r}LL) * {columns}LL + col + {c}LL'
            valid.append(f'''const long long point{suffix} = {point};
    const bool valid{suffix} = row + {r}LL < {rows}LL && col + {c}LL < {columns}LL
        && point{suffix} >= tile_start && point{suffix} < tile_start + tile_count;''')
            def named(text):
                for name in ('total_re', 'total_im', 'shared_re', 'shared_im'):
                    text = text.replace(name, name + suffix)
                return text
            declarations.append(named(shared + '\n    ' + declaration))
            products.append(f'if (valid{suffix}) {{ {wide} term = a{r} * b{c}; {named(accumulate)} }}')
            warp_sums.append(f'if (valid{suffix}) {{ {named(before)} }}')
            block_sums.append(f'''if (valid{suffix}) {{
        const int lane = t & 31, warp = t >> 5;
        {named(after)}
        if (t == 0) partials[(batch * {width}LL + point{suffix} - tile_start) * {blocks}LL + split] = {named(wide_result)};
    }}''')
    loads = []
    for side, labels, total, count in ((0, left, rows, row_tile), (1, right, columns, column_tile)):
        operand = core.large_operands[side]
        coordinate = 'row' if side == 0 else 'col'
        letter = 'a' if side == 0 else 'b'
        for index in range(count):
            offset = _offset(core.equation.inputs[operand], strides[operand],
                (*shared_axes, *labels), reduced, dims).replace('o /', f'(batch * {total}LL + {coordinate} + {index}LL) /')
            take = ' || '.join(f'valid_{index}{c}' for c in range(column_tile)) if side == 0 else ' || '.join(
                f'valid_{r}{index}' for r in range(row_tile))
            loads.append(f'''{wide} {letter}{index}(0.0);
        if ({take}) {letter}{index} = {wide}(*reinterpret_cast<const {narrow}*>(p{side} + ({offset})));''')
    any_valid = ' || '.join(f'valid_{r}{c}' for r in range(row_tile) for c in range(column_tile))
    source = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_precise_partial(
    const char* p0, const char* p1, const long long tile_start,
    const long long tile_count, {wide}* partials) {{
    const int t = threadIdx.x;
    const long long split = blockIdx.x;
    const long long batch = blockIdx.z / {row_groups}LL;
    const long long row = tile_start / {columns}LL + {row_tile}LL * (blockIdx.z % {row_groups}LL);
    const long long col = {column_tile}LL * blockIdx.y;
    {chr(10).join(valid)}
    if (!({any_valid})) return;
    {chr(10).join(declarations)}
    for (long long i = split * 256LL + t; i < {core.reduction_elements}LL; i += {blocks * 256}LL) {{
        {chr(10).join(loads)}
        {chr(10).join(products)}
    }}
    {chr(10).join(warp_sums)}
    __syncthreads();
    {chr(10).join(block_sums)}
}}
'''
    return source, (blocks, column_groups, prod(dims[x] for x in shared_axes) * row_groups)
