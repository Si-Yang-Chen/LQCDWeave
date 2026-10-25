"""Fixed thin-wide Action execution; no graph selection or numerical cache."""
from collections import OrderedDict
from itertools import product
from math import prod
import numpy as np

from .reduction_kernel import reduction_sources, _wide_parts, _offset, _kernel
from .raw_arguments import bind_pointer_arguments
from .operations.result_views import _DeferredResultView


def factorized_sources(core, strides):
    if core.tile_elements is not None:
        return _tiled_sources(core, strides)
    big, small = core.large_operands, core.small_operands
    axes, output, gram, dims = core.equation.inputs, core.equation.output, core.gram_axes, core.dimensions
    inner = tuple(label for label in gram if label not in output)
    first, _, outputs = reduction_sources(tuple(axes[i] for i in big), gram,
        tuple(core.shapes[i] for i in big), tuple(strides[i] for i in big), core.dtype, core.blocks)
    _, _, declaration, shared, accumulate, tree, _, final = _wide_parts(core.dtype)
    if core.result_dtype == 'complex128':
        final = 'complex<double>(total_re, total_im)'
    result_type = 'complex<double>' if core.result_dtype == 'complex128' else 'complex<float>'
    loads = [f'complex<double> c{n} = complex<double>(*reinterpret_cast<const complex<float>*>(p{n} + ({_offset(axes[i], strides[i], output, inner, dims)})));'
             for n, i in enumerate(small)]
    # Preserve the unbatched source verbatim. Retained lanes select their
    # explicit slice of the same packed internal partial owner.
    partial = 'partials[j]'
    if set(gram).intersection(output):
        steps = tuple(prod(dims[label] for label in gram[position+1:]) * core.blocks * 16
                      for position in range(len(gram)))
        offset = _offset(gram, steps, output, inner, dims)
        partial = f'partials[({offset}) / 16LL + j % {core.blocks}LL]'
    finish = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_factorized_wide_finish(
 const complex<double>* partials, const char* p0, const char* p1, {result_type}* result) {{
 const int t = threadIdx.x;
 const long long o = blockIdx.x;
 {shared}
 {declaration}
 for (long long j = t; j < {core.epilogue_elements * core.blocks}LL; j += 256LL) {{
   const long long i = j / {core.blocks}LL;
   {chr(10).join(loads)}
   complex<double> term = {partial} * c0 * c1;
   {accumulate}
 }}
 {tree}
 if (t == 0) result[o] = {final};
}}
'''
    return first, finish


def _tiled_sources(core, strides):
    """Fixed epilogue tiles; only internal wide owners cross tile boundaries."""
    big, small = core.large_operands, core.small_operands
    axes, output, dims = core.equation.inputs, core.equation.output, core.dimensions
    shared_axes = tuple(a for a in core.gram_axes if a in output)
    inner = tuple(a for a in core.gram_axes if a not in output)
    gram = (*shared_axes, *inner)
    first, _, _ = reduction_sources(tuple(axes[i] for i in big), gram,
        tuple(core.shapes[i] for i in big), tuple(strides[i] for i in big), core.dtype, core.blocks)
    first = first.replace('complex<double>* partials)',
        'const long long tile_start, const long long tile_count, complex<double>* partials)')
    first = first.replace(f'const long long o = blockIdx.x / {core.blocks}LL;',
        f'''const long long local = blockIdx.x / {core.blocks}LL;
    const long long e = local % {core.tile_width}LL;
    if (e >= tile_count) return;
    const long long o = (local / {core.tile_width}LL) * {core.epilogue_elements}LL + tile_start + e;''')
    first = first.replace(f'partials[o * {core.blocks}LL + b]', f'partials[local * {core.blocks}LL + b]')
    _, _, declaration, shared, accumulate, tree, wide, final = _wide_parts(core.dtype)
    if core.result_dtype == 'complex128':
        final = wide
    result_type = 'complex<double>' if core.result_dtype == 'complex128' else 'complex<float>'
    loads = []
    for n, (operand, extra) in enumerate(zip(small, core.small_reduced_axes, strict=True)):
        common = tuple((a,s) for a,s in zip(axes[operand], strides[operand], strict=True) if a not in extra)
        extras = tuple((a,s) for a,s in zip(axes[operand], strides[operand], strict=True) if a in extra)
        base = _offset(tuple(a for a,_ in common), tuple(s for _,s in common), output, inner, dims)
        offset = _offset(tuple(a for a,_ in extras), tuple(s for _,s in extras), (), extra, dims).replace('i /', 'k /')
        width = prod(dims[a] for a in extra)
        loads.append(f'''complex<double> c{n}(0.0);
   for (long long k = 0; k < {width}LL; ++k)
     c{n} += complex<double>(*reinterpret_cast<const complex<float>*>(p{n} + ({base}) + ({offset})));''')
    shared_steps = tuple(prod(dims[a] for a in shared_axes[k+1:]) for k in range(len(shared_axes)))
    shared_index = _offset(shared_axes, shared_steps, output, inner, dims)
    finish = f'''#include <cupy/complex.cuh>
extern "C" __global__ void ed_factorized_wide_finish(
 const complex<double>* partials, {', '.join(f'const char* p{i}' for i in range(len(small)))},
 complex<double>* accumulator, const long long tile_start, const long long tile_count,
 {result_type}* result) {{
 const int t = threadIdx.x;
 const long long o = blockIdx.x;
 {shared}
 {declaration}
 for (long long j = t; j < tile_count * {core.blocks}LL; j += 256LL) {{
   const long long i = tile_start + j / {core.blocks}LL;
   {chr(10).join(loads)}
   const long long at = (({shared_index}) * {core.tile_width}LL + j / {core.blocks}LL) * {core.blocks}LL + j % {core.blocks}LL;
   complex<double> term = partials[at] * {' * '.join(f'c{i}' for i in range(len(small)))};
   {accumulate}
 }}
 {tree}
 if (t == 0) {{
   if (tile_start) {{ total_re += accumulator[o].real(); total_im += accumulator[o].imag(); }}
   if (tile_start + tile_count == {core.epilogue_elements}LL) result[o] = {final};
   else accumulator[o] = {wide};
 }}
}}
'''
    return first, finish


def _scalar_oracle(core, backend):
    """Bounded small CPU oracle: four factors and only scalar wide temporaries."""
    axes, output, dims = core.equation.inputs, core.equation.output, core.dimensions
    reduced = tuple(label for label in dims if label not in output)
    indices = tuple(tuple((label in output, (output if label in output else reduced).index(label))
                          for label in labels) for labels in axes)
    def run(values):
        result = backend.empty(core.output_shape, dtype=core.result_dtype)
        for point in product(*(range(n) for n in core.output_shape)):
            total = backend.zeros((), dtype='complex128')
            for inner in product(*(range(dims[label]) for label in reduced)):
                term = backend.ones((), dtype='complex128')
                for value, take in zip(values, indices, strict=True):
                    scalar = backend.asarray(value[tuple((point if outer else inner)[i] for outer, i in take)], dtype='complex128')
                    term *= scalar
                total += term
            result[point] = total
        return result, True
    return run


def _numpy_wide(core):
    """One-tile CPU reference with explicitly reserved wide temporaries."""
    numbers={a:i for i,a in enumerate(core.dimensions)}
    axes=tuple(tuple(numbers[a] for a in labels) for labels in core.equation.inputs)
    gram=tuple(numbers[a] for a in core.gram_axes)
    output=tuple(numbers[a] for a in core.equation.output)
    def run(values):
        args=tuple(item for i in core.large_operands for item in (values[i],axes[i]))+(gram,)
        partial=np.einsum(*args,dtype='complex128',optimize=False)
        small=[]
        for i,extra in zip(core.small_operands,core.small_reduced_axes,strict=True):
            positions=tuple(core.equation.inputs[i].index(a) for a in extra)
            value=(values[i].sum(axis=positions,dtype='complex128') if positions else
                   values[i].astype('complex128'))
            small.extend((value,tuple(numbers[a] for a in core.equation.inputs[i] if a not in extra)))
        result=np.einsum(partial,gram,*small,output,dtype='complex128',optimize=False)
        return result.astype(core.result_dtype, copy=False),True
    return run


def _numpy_tiled(core):
    """Bounded NumPy tiles; do not enumerate the full factor product in Python."""
    dims, output = core.dimensions, core.equation.output
    shared = tuple(a for a in core.gram_axes if a in output)
    inner = tuple(a for a in core.gram_axes if a not in output)
    reduced = {a:i for i,a in enumerate(core.reduced_axes)}
    big_axes = tuple(tuple(reduced[a] for a in core.equation.inputs[i] if a in reduced)
                     for i in core.large_operands)
    shared_shape = tuple(dims[a] for a in shared)
    def run(values):
        partial = np.empty((prod(shared_shape),core.tile_width),dtype='complex128')
        total = np.zeros(core.output_shape,dtype='complex128')
        small = []
        for i,extra in zip(core.small_operands,core.small_reduced_axes,strict=True):
            positions = tuple(core.equation.inputs[i].index(a) for a in extra)
            value = (values[i].sum(axis=positions,dtype='complex128') if positions else
                     values[i].astype('complex128'))
            small.append((value,tuple(a for a in core.equation.inputs[i] if a not in extra)))
        for start in range(0,core.epilogue_elements,core.tile_width):
            count = min(core.tile_width,core.epilogue_elements-start)
            coordinates = np.unravel_index(np.arange(start,start+count),tuple(dims[a] for a in inner))
            for shared_index,point in enumerate(np.ndindex(shared_shape)):
                at = dict(zip(shared,point,strict=True))
                for j in range(count):
                    at.update((a,int(coordinates[k][j])) for k,a in enumerate(inner))
                    left,right = (values[i][tuple(at[a] if a in at else slice(None)
                                   for a in core.equation.inputs[i])].astype('complex128')
                                  for i in core.large_operands)
                    partial[shared_index,j] = np.einsum(left,big_axes[0],right,big_axes[1],(),optimize=False)
                    del left,right
            for point in np.ndindex(core.output_shape):
                at = dict(zip(output,point,strict=True))
                shared_index = np.ravel_multi_index(tuple(at[a] for a in shared),tuple(dims[a] for a in shared)) if shared else 0
                weight = None
                for value,axes in small:
                    current = value[tuple(at[a] if a in at else coordinates[inner.index(a)] for a in axes)]
                    weight = current if weight is None else weight*current
                total[point] += np.sum(partial[shared_index,:count]*weight,dtype='complex128')
        return total.astype(core.result_dtype, copy=False),True
    return run


def validate_factorized_action(action, catalog):
    """Validate every Action before sharing its name-free numerical factory."""
    core = action.core
    dtype = np.dtype(core.dtype)
    if (action.equation != core.equation or not action.wide_accumulation
            or tuple(catalog[key].shape for key in action.operands) != core.shapes
            or any(catalog[key].dtype != dtype for key in action.operands)
            or catalog[action.tensor].shape != core.output_shape
            or catalog[action.tensor].dtype != np.dtype(core.result_dtype)):
        raise ValueError('thin wide Action differs from its declared layouts')


def bind_factorized_reduction(action, catalog, backend, *, factorized_gram_tiling=None):
    validate_factorized_action(action, catalog)
    core = action.core
    if backend is np:
        return ((_numpy_wide(core) if core.gram_elements <= core.partial_elements else _numpy_tiled(core))
                if core.tile_elements is not None else _scalar_oracle(core, backend))
    factory = getattr(backend, 'RawKernel', None)
    if getattr(backend, '__name__', '') != 'cupy' or not callable(factory):
        raise NotImplementedError('thin wide Actions require NumPy or CUDA CuPy')
    is_hip = getattr(getattr(getattr(backend, 'cuda', None), 'runtime', None), 'is_hip', False)
    if is_hip() if callable(is_hip) else is_hip:
        raise NotImplementedError('thin wide Actions require the saved CUDA target')
    layouts = OrderedDict()
    big, small = core.large_operands, core.small_operands
    # The core is immutable. Resolve recursive geometry once per factory,
    # rather than rebuilding dimensions/axis sets on every call and tile.
    shapes, dtype, output_shape = core.shapes, np.dtype(core.dtype), core.output_shape
    result_dtype = np.dtype(core.result_dtype)
    blocks, partial_elements, partial_bytes = core.blocks, core.partial_elements, core.partial_bytes
    epilogue_elements, tile_width, tiles = core.epilogue_elements, core.tile_width, core.tiles
    output_elements, workspace_bytes = core.output_elements, core.workspace_bytes
    tiled = core.tile_elements is not None
    gemm = core.gram_strategy == 'gemm'
    if gemm:
        from lqcdweave.contracts.correlator.factorized_geometry import gram_matrix_axes
        left,right = gram_matrix_axes(core.equation,big,core._geometry)
        reduced,dims = core.reduced_axes,core.dimensions
        rows,columns = prod(dims[a] for a in left),prod(dims[a] for a in right)
        reduction = core.reduction_elements
        permutations = (tuple(core.equation.inputs[big[0]].index(a) for a in (*left,*reduced)),
                        tuple(core.equation.inputs[big[1]].index(a) for a in (*reduced,*right)))
        from lqcdweave.contracts.correlator.allocation import device_allocation_bytes
        packed_sizes = (rows*reduction*16,reduction*columns*16)
        packed_offsets = (0,device_allocation_bytes(packed_sizes[0]))
        gram_offset = packed_offsets[1]+device_allocation_bytes(packed_sizes[1])
        arena_bytes = gram_offset+device_allocation_bytes(rows*columns*16)
        packing_shapes = tuple(tuple(shapes[index][axis] for axis in permutation)
            for index,permutation in zip(big,permutations))
        from .matrix_product import bind_pointer_gemm
        packed_shapes = ((rows,reduction),(reduction,columns))
        packed_gemm = bind_pointer_gemm(backend,packed_shapes,(1,1),(None,None),
            'complex128',(rows,columns),input_offsets=packed_offsets)
        if packed_gemm is not None:
            packed_gemm = packed_gemm.bind_layout(tuple(
                (shape,np.dtype('complex128'),(shape[1]*16,16)) for shape in packed_shapes))
    if factorized_gram_tiling is None:
        from .binding_cache import factorized_gram_tiling_enabled
        factorized_gram_tiling = factorized_gram_tiling_enabled()

    def bind_layout(strides):
        first, finish = factorized_sources(core, strides)
        if gemm:
            from .operations.packing_kernel import bind_partition_packing
            pack = bind_partition_packing(backend,packing_shapes,(dtype,dtype),
                tuple(tuple(strides[index][axis] for axis in permutation)
                    for index,permutation in zip(big,permutations)),
                packed_offsets,np.dtype('complex128'))
            finish_kernel = _kernel(factory,finish,'ed_factorized_wide_finish')
            finish_grid,threads = (output_elements,),(256,)
            finish_bound = bind_pointer_arguments(backend,finish_kernel,finish_grid,threads,
                (None,None,None,None,np.int64(0),np.int64(epilogue_elements),None),
                ((0,0),(1,1),(2,2),(3,0)))
            def call(values):
                arena = packed_left = packed_right = gram = result = None
                try:
                    from .device import active_device_execution
                    device = active_device_execution()
                    borrow = (getattr(device,'reduction_partial',None)
                              if device is not None and device.backend is backend else None)
                    arena = borrow(arena_bytes,np.dtype('complex128')) if callable(borrow) else None
                    if arena is None:
                        arena = backend.empty((arena_bytes//16,),dtype='complex128')
                    gram = backend.ndarray((rows,columns),dtype='complex128',memptr=arena.data+gram_offset)
                    if pack is not None:
                        pack(tuple(values[i] for i in big),arena)
                    else:
                        for index,shape,offset,permutation in zip(big,packing_shapes,packed_offsets,permutations):
                            view = backend.ndarray(shape,dtype='complex128',memptr=arena.data+offset)
                            backend.copyto(view,values[index].transpose(permutation))
                    submitted = packed_gemm(arena,arena,out=gram) if packed_gemm is not None else None
                    if submitted is None:
                        # Preserve nondefault CuPy compute/math policies and
                        # missing private ABIs within the same admitted arena.
                        packed_left = backend.ndarray(packed_shapes[0],dtype='complex128',memptr=arena.data)
                        packed_right = backend.ndarray(packed_shapes[1],dtype='complex128',memptr=arena.data+packed_offsets[1])
                        backend.matmul(packed_left,packed_right,out=gram)
                    result = backend.empty(output_shape,dtype=result_dtype)
                    if finish_bound is None:
                        finish_kernel(finish_grid,threads,(gram,*(values[i] for i in small),gram,
                            np.int64(0),np.int64(epilogue_elements),result))
                    else:
                        finish_bound((gram,*(values[i] for i in small)),result)
                    return result,True
                except BaseException as error:
                    try:
                        backend.cuda.get_current_stream().synchronize()
                    except BaseException as cleanup:
                        if callable(getattr(error,'add_note',None)):
                            error.add_note(f'wide Gram cleanup also failed: {cleanup!r}')
                    raise
            return call
        first_grid = (partial_elements * blocks,)
        if factorized_gram_tiling and tiled and tile_width >= 1024:
            from .factorized_gram import matrix_gram_partial
            matrix = matrix_gram_partial(core, strides)
            if matrix is not None:
                first, first_grid = matrix
        first_kernel = _kernel(factory, first, 'ed_precise_partial')
        finish_kernel = _kernel(factory, finish, 'ed_factorized_wide_finish')
        finish_grid, threads = (output_elements,), (256,)
        first_args = None if tiled else bind_pointer_arguments(backend, first_kernel, first_grid, threads,
            (None, None, None), ((0, 0), (1, 1)))
        finish_args = None if tiled else bind_pointer_arguments(backend, finish_kernel, finish_grid, threads,
            (None, None, None, None), ((0, 0), (1, 1), (2, 2)))
        tile_arguments = {}
        if tiled:
            # At most a full and a tail template, regardless of tile count.
            # Only multi-tile starts vary; constants and pointer slots remain
            # thread-local through the ordinary raw argument ABI.
            for count in {tile_width, epilogue_elements % tile_width} - {0}:
                start = ((2, 0),) if tiles > 1 else ()
                first_bound = bind_pointer_arguments(backend, first_kernel, first_grid, threads,
                    (None, None, np.int64(0), np.int64(count), None), ((0, 0), (1, 1)),
                    scalar_positions=start)
                n = len(small)
                finish_bound = bind_pointer_arguments(backend, finish_kernel, finish_grid, threads,
                    (None,) * (n + 2) + (np.int64(0), np.int64(count), None),
                    tuple((i, i) for i in range(n + 2)),
                    scalar_positions=((n + 2, 0),) if tiles > 1 else ())
                tile_arguments[count] = first_bound, finish_bound
        def call(values):
            partials = result = accumulator = None
            try:
                from .device import active_device_execution
                device = active_device_execution()
                borrow = (getattr(device, 'reduction_partial', None)
                          if device is not None and device.backend is backend else None)
                partials = borrow(partial_bytes, np.dtype('complex128')) if callable(borrow) else None
                if partials is None:
                    partials = backend.empty((partial_elements, blocks), dtype='complex128')
                result = backend.empty(output_shape, dtype=result_dtype)
                large_values = tuple(values[i] for i in big)
                small_values = tuple(values[i] for i in small)
                if tiled:
                    accumulator = (backend.empty(output_shape, dtype='complex128') if tiles > 1 else result)
                    for start in range(0, epilogue_elements, tile_width):
                        count = min(tile_width, epilogue_elements-start)
                        first_bound, finish_bound = tile_arguments[count]
                        if first_bound is None:
                            first_kernel(first_grid, threads, (*large_values, np.int64(start), np.int64(count), partials))
                        else:
                            first_bound(large_values, partials, (start,))
                        if finish_bound is None:
                            finish_kernel(finish_grid, threads, (partials, *small_values, accumulator,
                                                                np.int64(start), np.int64(count), result))
                        else:
                            finish_bound((partials, *small_values, accumulator), result, (start,))
                    return result, True
                if first_args is None:
                    first_kernel(first_grid, threads, (*large_values, partials))
                else:
                    first_args(large_values, partials)
                if finish_args is None:
                    finish_kernel(finish_grid, threads, (partials, *small_values, result))
                else:
                    finish_args((partials, *small_values), result)
                return result, True
            except BaseException as error:
                try:
                    backend.cuda.get_current_stream().synchronize()
                except BaseException as cleanup:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'thin wide cleanup also failed: {cleanup!r}')
                raise
        return call

    def run(operands):
        # Ordinary cfg mappings can retain an owned result's deferred view.
        values = tuple(value.array() if type(value) is _DeferredResultView else value for value in operands)
        if len(values) != len(shapes) or any(value.shape != shape or value.dtype != dtype
                for value, shape in zip(values, shapes, strict=True)):
            raise ValueError('current thin wide inputs differ from the saved layouts')
        strides = tuple(value.strides for value in values)
        call = layouts.get(strides)
        if call is None:
            call = layouts[strides] = bind_layout(strides)
            if len(layouts) > 64:
                layouts.popitem(last=False)
        else:
            try:
                layouts.move_to_end(strides)
            except KeyError:
                # Another caller can evict this metadata while binding a new
                # layout. The local callable remains valid for this launch.
                pass
        return call(values)
    run.workspace_bytes = workspace_bytes
    return run
