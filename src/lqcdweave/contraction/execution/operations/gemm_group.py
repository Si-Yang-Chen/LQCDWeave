"""Execute a fixed group of independent, same-shaped dense GEMMs."""
from functools import lru_cache, partial
from math import prod

import numpy as np
from lqcdweave.contracts.correlator.api import GEMM_GROUP_MAX_BATCHES, GEMM_GROUP_POINTER_BYTES

from ..diagnostics import debug_enabled
from ..raw_arguments import bind_pointer_arguments
from .packing_kernel import bind_gemm_group_packing
from .result_views import bind_result_views


_MAX_BATCH = GEMM_GROUP_MAX_BATCHES
_MAX_INT32 = np.iinfo(np.int32).max
_GEMM_BY_DTYPE = {
    np.dtype(np.float32): "sgemmBatched",
    np.dtype(np.float64): "dgemmBatched",
    np.dtype(np.complex64): "cgemmBatched",
    np.dtype(np.complex128): "zgemmBatched",
}


@lru_cache(maxsize=64)
def _pointer_kernel(factory, source):
    """Cache only bounded source-specialized code, never arrays or pointers."""
    return factory(source, "ed_gemm_group_pointer_setup", options=("--std=c++17",))


def _pointer_source(lanes, matrix_elements, itemsize, *, input_elements, packed_mask):
    params = []
    stores = []
    # cuBLAS sees row-major A/B as column-major B/A. Each side independently
    # supplies either one complete packed slab or the current child matrices.
    for side, name, table_offset in ((1, 'b', 0), (0, 'a', lanes)):
        if packed_mask & (1 << side):
            params.append(f"const void* {name}")
        else:
            params.extend(f"const void* {name}{index}" for index in range(lanes))
        for index in range(lanes):
            address = (f"reinterpret_cast<const char*>({name}) + "
                       f"{index * input_elements[side] * itemsize}LL"
                       if packed_mask & (1 << side) else f"{name}{index}")
            stores.append(f"table[{table_offset + index}] = "
                          f"reinterpret_cast<unsigned long long>({address});")
    for index in range(lanes):
        stores.append(
            f"table[{2 * lanes + index}] = reinterpret_cast<unsigned long long>("
            f"reinterpret_cast<const char*>(cube) + {index * matrix_elements * itemsize}LL);")
    params.extend(("unsigned long long* table", "const void* cube"))
    return (
        'extern "C" __global__ void ed_gemm_group_pointer_setup('
        + ", ".join(params)
        + ") {\n"
        + "if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
        + "\n".join(stores)
        + "\n}\n"
    )


def _bind_group_pointer_setup(backend, kernels, lanes):
    """Bind the saved setup ABI, using fresh matrix/table/result owners."""
    native = {}
    for mask, kernel in kernels.items():
        matrix_count = sum(1 if mask & (1 << side) else lanes for side in range(2))
        count = matrix_count + 2  # Matrices, table, then the result cube.
        native[mask] = bind_pointer_arguments(backend, kernel, (1,), (1,),
            (None,) * count, tuple((index, index) for index in range(count - 1)))

    def setup(packed, mask, table, cube):
        bound = native[mask]
        if bound is not None and not debug_enabled():
            bound((*packed, table), cube)
        else:
            kernels[mask]((1,), (1,), (*packed, table, cube))

    setup.native_arguments = native
    return setup


def _cupy_requirements(backend, dtype):
    """Reject unsupported CuPy/HIP routes during binding, without making a context."""
    cuda = getattr(backend, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    is_hip = getattr(runtime, "is_hip", None)
    if callable(is_hip):
        is_hip = is_hip()
    if is_hip is None:
        raise NotImplementedError("CuPy GEMM groups require CUDA runtime capability metadata")
    if is_hip:
        raise NotImplementedError("CuPy GEMM groups currently support CUDA only; HIP is unsupported")

    missing = []
    for name in ("RawKernel", "ElementwiseKernel"):
        if not callable(getattr(backend, name, None)):
            missing.append(f"cupy.{name}")
    if not callable(getattr(backend, "ndarray", None)):
        missing.append("cupy.ndarray")
    for name in ("cublas", "Device", "get_current_stream"):
        if getattr(cuda, name, None) is None:
            missing.append(f"cupy.cuda.{name}")
    if not callable(getattr(backend, "empty", None)):
        missing.append("cupy.empty")
    if missing:
        raise NotImplementedError("CuPy CUDA GEMM groups require " + ", ".join(missing))

    cublas = cuda.cublas
    required = (
        "CUBLAS_OP_N", "CUBLAS_POINTER_MODE_HOST", "getPointerMode",
        _GEMM_BY_DTYPE[dtype],
    )
    missing = [f"cupy.cuda.cublas.{name}" for name in required
               if not hasattr(cublas, name)]
    if missing:
        raise NotImplementedError("CuPy CUDA GEMM groups require " + ", ".join(missing))
    return cublas


def _bind_layouts(group, catalog, backend):
    batches = tuple(group.batches)
    if not 2 <= len(batches) <= _MAX_BATCH:
        raise ValueError(f"a dense GEMM group must contain between 2 and {_MAX_BATCH} batches")
    try:
        m, k, n = tuple(group.matrix_shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("a dense GEMM group requires a (M, K, N) matrix_shape") from error
    if any(type(value) is not int or value <= 0 or value > _MAX_INT32 for value in (m, k, n)):
        raise ValueError("dense GEMM group dimensions must be positive int32 values")
    dtype = np.dtype(group.packing_dtype)
    if dtype not in _GEMM_BY_DTYPE:
        raise TypeError("dense GEMM groups support float32, float64, complex64, and complex128")

    from .dense_batch import bind_batch_layout

    # Saved groups use the canonical (M,K)/(K,N) pointer ABI, including legacy
    # groups whose thin children now execute through the wide reference route.
    layouts = tuple(bind_batch_layout(batch, catalog, backend, allow_fortran=False,
                                     coalesce_reductions=False) for batch in batches)
    expected_matrices = ((m, k), (k, n))
    for batch, layout in zip(batches, layouts, strict=True):
        if batch.packing_dtype != dtype.name or layout.dtype != dtype:
            raise ValueError("dense GEMM group children must share the saved packing dtype")
        if tuple(tuple(shape) for shape in layout.matrix_shapes) != expected_matrices:
            raise ValueError("dense GEMM group children must share the saved (M, K, N) shape")
        if prod(layout.output_shape) != m * n:
            raise ValueError("dense GEMM child output layout does not match its matrix shape")
    return batches, layouts, (m, k, n), dtype


def _bind_group_result_views(layouts):
    """Bind one reshape and direct result indexing for a shared child layout."""
    output_shape = layouts[0].output_shape
    if not output_shape or any(layout.output_shape != output_shape for layout in layouts[1:]):
        return None
    target_shape = (len(layouts), *output_shape)
    result_views = tuple(
        (key, (lane, *bounds), squeeze)
        for lane, layout in enumerate(layouts)
        for key, bounds, squeeze in layout.result_slices
    )

    def views(output):
        grouped = output.reshape(target_shape)
        results = {}
        for key, indexer, squeeze in result_views:
            value = grouped[indexer]
            results[key] = value.squeeze(axis=squeeze) if squeeze else value
        return results

    views.deferred_layouts = bind_result_views(target_shape, layouts[0].dtype, result_views)
    return views


def _validate_packed(packed, layouts, dtype):
    for matrices, layout in zip(packed, layouts, strict=True):
        for value, shape in zip(matrices, layout.matrix_shapes, strict=True):
            if (tuple(value.shape) != tuple(shape) or np.dtype(value.dtype) != dtype
                    or not value.flags.c_contiguous):
                raise ValueError("packed dense GEMM inputs differ from their saved matrix layout")


def _bind_group_output_allocation(backend, shape, dtype):
    """Use one current allocation for the saved 512-byte table and result."""
    table_bytes = GEMM_GROUP_POINTER_BYTES
    table_words = table_bytes // np.dtype(np.uint64).itemsize
    allocate = partial(backend.empty, (table_bytes + prod(shape) * dtype.itemsize,),
                       dtype=np.uint8)
    array = backend.ndarray

    def current():
        storage = allocate()
        pointer = storage.data
        table = array((table_words,), dtype=np.uint64, memptr=pointer)
        cube = array(shape, dtype=dtype, memptr=pointer + table_bytes)
        return cube, table
    return current


def _bind_group_inputs(layouts, grouped_packers, backend, dtype):
    """Bind side geometry; hand current buffers to the pointer kernel directly."""
    sides = tuple(tuple((layout.sides[index], layout.matrix_shapes[index])
                        for layout in layouts) for index in range(2))
    buffer_packers = tuple(packer.pack_buffer if packer is not None else None
                           for packer in grouped_packers)

    def pack(arrays, owners):
        values, mask = [], 0
        debug = debug_enabled()
        for index, (members, buffer_packer) in enumerate(zip(sides, buffer_packers)):
            buffer = buffer_packer(arrays) if buffer_packer is not None and not debug else None
            if buffer is not None:
                owners.append(buffer)
                values.append((buffer,))
                mask |= 1 << index
            else:
                current = []
                for side, shape in members:
                    value = side.pack(arrays, backend, dtype)
                    # Keep earlier copies alive if a later side raises, until
                    # run_cupy fences the submitted work during cleanup.
                    owners.append(value)
                    # The raw pointer kernel needs the contiguous allocation,
                    # not a 2-D view: cuBLAS geometry was fixed during binding.
                    current.append(value.reshape(shape) if debug and value.shape != shape else value)
                values.append(tuple(current))
        if debug:
            _validate_packed(tuple(zip(*values)), layouts, dtype)
        return (*values[1], *values[0]), mask

    return pack


def _bind_precise_group(layouts, backend, shape, dtype, group_result_views):
    """Legacy groups containing thin reductions must not bypass wide sums."""
    gpu = getattr(backend, '__name__', '') == 'cupy'

    def run(arrays, *, timed=None, device=False):
        packed, cube, product = [], None, None
        try:
            def pack_all():
                for layout in layouts:
                    packed.append(tuple(side.pack(arrays, backend, dtype) for side in layout.sides))
            if timed is None:
                pack_all()
            else:
                timed('dense_batch_pack', pack_all, device=device)

            def contract_all():
                nonlocal cube, product
                cube = backend.empty(shape, dtype=dtype)
                for index, (layout, values) in enumerate(zip(layouts, packed, strict=True)):
                    if layout.precise_product is None:
                        left, right = (value.reshape(matrix_shape) for value, matrix_shape in
                                       zip(values, layout.matrix_shapes, strict=True))
                        backend.matmul(left, right, out=cube[index])
                    else:
                        product = layout.contract(values, backend)
                        cube[index] = product.reshape(shape[1:])
                        product = None
                return cube
            if timed is None:
                contract_all()
            else:
                timed('contract_kernel', contract_all, device=device)
            if (group_result_views is not None and not debug_enabled() and timed is None
                    and callable(getattr(arrays, '_publish_result_views', None))):
                arrays._publish_result_views(cube, group_result_views.deferred_layouts)
            else:
                results = {}
                for lane, layout in zip(cube, layouts, strict=True):
                    results.update(layout.views(lane.reshape(layout.output_shape)))
                arrays.update(results)
            return cube
        except BaseException as error:
            if gpu:
                try:
                    backend.cuda.get_current_stream().synchronize()
                except BaseException as cleanup:
                    error.add_note(f'precise group cleanup also failed: {cleanup!r}')
            raise
    return run


def bind_dense_gemm_group(group, catalog, backend=np):
    """Bind child packing once, returning a stateless-input execution callable.

    NumPy is the independent numerical reference path. CuPy is accepted only
    when its CUDA and cuBLAS facilities are available; this binding never
    initializes a device or chooses another execution route.
    """
    if backend is not np and getattr(backend, "__name__", "") != "cupy":
        raise NotImplementedError("dense GEMM groups require NumPy or CUDA CuPy")
    cublas = None
    if backend is not np:
        try:
            requested_dtype = np.dtype(group.packing_dtype)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("dense GEMM groups require a supported packing dtype") from error
        if requested_dtype not in _GEMM_BY_DTYPE:
            raise TypeError("dense GEMM groups support float32, float64, complex64, and complex128")
    batches, layouts, (m, k, n), dtype = _bind_layouts(group, catalog, backend)
    lanes = len(batches)
    group_result_views = _bind_group_result_views(layouts) if backend is not np else None
    if any(layout.precise_product is not None for layout in layouts):
        return _bind_precise_group(layouts, backend, (lanes, m, n), dtype, group_result_views)
    if backend is not np:
        cublas = _cupy_requirements(backend, dtype)

    if backend is np:
        def run_numpy(arrays, *, timed=None, device=False):
            def pack_all():
                packed = []
                for layout in layouts:
                    values = tuple(side.pack(arrays, backend, dtype) for side in layout.sides)
                    packed.append(tuple(value if value.shape == shape else value.reshape(shape) for value, shape in
                                        zip(values, layout.matrix_shapes, strict=True)))
                return tuple(packed)

            packed = pack_all() if timed is None else timed(
                "dense_batch_pack", pack_all, device=device)
            if debug_enabled():
                _validate_packed(packed, layouts, dtype)

            def contract_all():
                cube = backend.empty((lanes, m, n), dtype=dtype)
                for index, (left, right) in enumerate(packed):
                    backend.matmul(left, right, out=cube[index])
                return cube

            cube = contract_all() if timed is None else timed(
                "contract_kernel", contract_all, device=device)
            results = {}
            for layout, lane in zip(layouts, cube, strict=True):
                results.update(layout.views(lane if lane.shape == layout.output_shape else lane.reshape(layout.output_shape)))
            arrays.update(results)
            return cube

        return run_numpy

    gemm = getattr(cublas, _GEMM_BY_DTYPE[dtype])
    alpha = np.asarray(1, dtype=dtype)
    beta = np.asarray(0, dtype=dtype)
    alpha_pointer = int(alpha.ctypes.data)
    beta_pointer = int(beta.ctypes.data)
    operation = cublas.CUBLAS_OP_N
    cuda = backend.cuda
    allocate_outputs = _bind_group_output_allocation(backend, (lanes, m, n), dtype)
    pointer_bytes = lanes * np.dtype(np.uint64).itemsize
    m, k, n = (int(value) for value in (m, k, n))
    grouped_packers = tuple(bind_gemm_group_packing(
        backend, tuple(layout.sides[index] for layout in layouts), catalog,
        layouts[0].matrix_shapes[index], dtype) for index in range(2))
    pack_inputs = _bind_group_inputs(layouts, grouped_packers, backend, dtype)
    pointer_kernels = {
        mask: _pointer_kernel(backend.RawKernel, _pointer_source(
            lanes, m * n, dtype.itemsize, input_elements=(m * k, k * n), packed_mask=mask))
        for mask in range(4)
        if all(not mask & (1 << index) or packer is not None
               for index, packer in enumerate(grouped_packers))
    }

    setup_pointers = _bind_group_pointer_setup(backend, pointer_kernels, lanes)

    def contract_group(handle, table, _scalar_owners=(alpha, beta)):
        # Keep the arrays owning these bound host addresses alive with the call.
        table_pointer = int(table.data.ptr)
        gemm(handle, operation, operation, n, m, k, alpha_pointer,
             table_pointer, n, table_pointer + pointer_bytes, k, beta_pointer,
             table_pointer + 2 * pointer_bytes, n, lanes)

    def run_cupy(arrays, *, timed=None, device=False):
        stream = cuda.get_current_stream()
        handle = cuda.Device().cublas_handle
        if cublas.getPointerMode(handle) != cublas.CUBLAS_POINTER_MODE_HOST:
            raise RuntimeError("cuBLAS handle must already use host pointer mode")

        submitted = False
        packed = ()
        packing_owners = []
        cube = table = output = None
        try:
            with stream:
                # Packing can submit copies before a later child fails.
                submitted = True
                if timed is None:
                    packed, packing_mask = pack_inputs(arrays, packing_owners)
                    cube, table = allocate_outputs()
                    setup_pointers(packed, packing_mask, table, cube)
                    contract_group(handle, table)
                else:
                    packed, packing_mask = timed(
                        "dense_batch_pack", partial(pack_inputs, arrays, packing_owners),
                        device=device)

                    def timed_setup_pointers():
                        nonlocal cube, table
                        cube, table = allocate_outputs()
                        setup_pointers(packed, packing_mask, table, cube)
                        return table

                    timed("pointer_setup", timed_setup_pointers, device=device)
                    timed("contract_kernel", partial(contract_group, handle, table),
                          device=device)

                # Every result view retains the shared MemoryPointer allocation,
                # including the table prefix consumed by asynchronous cuBLAS.
                output = cube
                if group_result_views is None:
                    results = {}
                    for layout, lane in zip(layouts, output, strict=True):
                        results.update(layout.views(lane if lane.shape == layout.output_shape else lane.reshape(layout.output_shape)))
                elif (timed is None and not debug_enabled()
                      and callable(getattr(arrays, '_publish_result_views', None))):
                    arrays._publish_result_views(output, group_result_views.deferred_layouts)
                    return output
                else:
                    results = group_result_views(output)
                arrays.update(results)
                return output
        except BaseException as error:
            if submitted:
                try:
                    stream.synchronize()
                except BaseException as cleanup_error:
                    error.add_note(
                        "CuPy GEMM group cleanup synchronization also failed: "
                        f"{cleanup_error!r}"
                    )
            raise

    return run_cupy
