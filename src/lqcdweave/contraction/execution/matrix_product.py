"""Bind the existing CUDA GEMM to fixed matrix geometry, without array views."""
from math import prod
from functools import partial

import numpy as np

from .diagnostics import debug_enabled


_GEMM = {np.dtype(t): name for t, name in (
    (np.float32, 'sgemm'), (np.float64, 'dgemm'),
    (np.complex64, 'cgemm'), (np.complex128, 'zgemm'))}


def _matrix_strides(shape, split, permutation, itemsize):
    """Original-axis byte strides for C/F matrices with C-flattened groups."""
    candidates = []
    for transpose in (False, True):
        order = ((*range(split, len(shape)), *range(split)) if transpose
                 else tuple(range(len(shape))))
        steps = [0] * len(shape)
        stride = itemsize
        for axis in reversed(order):
            steps[axis] = stride
            stride *= shape[axis]
        original = [0] * len(shape)
        for axis, step in zip(permutation, steps):
            original[axis] = step
        candidates.append(tuple(original))
    return tuple(candidates)


def bind_pointer_gemm(backend, shapes, splits, permutations, dtype, output_shape,
                      output_permutation=None, *, input_offsets=None):
    """Return a private fixed GEMM, or None when the backend lacks its ABI.

    Each call proves the current operand's saved shape/dtype and byte strides.
    ``bind_layout`` permits a caller already dispatching on that metadata to
    reuse the classification, without retaining operands or numerical modes.
    It returns None for a layout needing packing or for debug; the caller keeps
    its existing array-based path. Only two host scalar constants are retained.
    No input, numerical result, device handle or stream is captured.
    A supplied ``out`` is the canonical C output before any permutation.
    Private arena callers may bind aligned byte offsets with virtual matrix
    layouts; they retain the arena and prove its disjoint matrix partitions.
    """
    dtype = np.dtype(dtype)
    if getattr(backend, '__name__', '') != 'cupy' or dtype not in _GEMM:
        return None
    cuda = getattr(backend, 'cuda', None)
    runtime = getattr(cuda, 'runtime', None)
    is_hip = getattr(runtime, 'is_hip', None)
    if callable(is_hip):
        is_hip = is_hip()
    cublas = getattr(cuda, 'cublas', None)
    linalg = getattr(getattr(backend, '_core', None), '_routines_linalg', None)
    compute_type = getattr(linalg, 'get_compute_type', None)
    default_compute_type = getattr(linalg, 'COMPUTE_TYPE_DEFAULT', None)
    required = (_GEMM[dtype], 'CUBLAS_OP_N', 'CUBLAS_OP_T',
                'CUBLAS_POINTER_MODE_HOST', 'getPointerMode',
                'CUBLAS_DEFAULT_MATH', 'getMathMode')
    if (is_hip is None or bool(is_hip) or not callable(getattr(cuda, 'Device', None))
            or not callable(getattr(cuda, 'get_current_stream', None))
            or not callable(compute_type) or default_compute_type is None
            or any(not hasattr(cublas, name) for name in required)):
        return None
    shapes = tuple(tuple(shape) for shape in shapes)
    if input_offsets is not None:
        if (len(input_offsets) != 2 or any(type(offset) is not int or offset < 0
                or offset % dtype.itemsize for offset in input_offsets)):
            raise ValueError('fixed GEMM arena offsets must be two aligned nonnegative byte offsets')
        input_offsets = tuple(input_offsets)
    permutations = tuple(tuple(range(len(s))) if p is None else tuple(p)
                         for s, p in zip(shapes, permutations))
    m, k, n = prod(shapes[0][:splits[0]]), prod(shapes[0][splits[0]:]), prod(shapes[1][splits[1]:])
    if (not 0 < min(m, k, n) <= max(m, k, n) <= np.iinfo(np.int32).max
            or prod(shapes[1][:splits[1]]) != k or prod(output_shape) != m * n):
        return None
    original_shapes = tuple(tuple(shape[p.index(i)] for i in range(len(shape)))
                            for shape, p in zip(shapes, permutations))
    strides = tuple(_matrix_strides(s, split, p, dtype.itemsize)
                    for s, split, p in zip(shapes, splits, permutations))
    gemm = getattr(cublas, _GEMM[dtype])
    alpha, beta = np.asarray(1, dtype=dtype), np.asarray(0, dtype=dtype)
    alpha_pointer, beta_pointer = int(alpha.ctypes.data), int(beta.ctypes.data)
    operations = cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T
    output_shape = tuple(output_shape)
    input_bytes = tuple(prod(shape)*dtype.itemsize for shape in shapes)
    output_bytes = prod(output_shape)*dtype.itemsize
    offsets = (0,0) if input_offsets is None else input_offsets
    if (input_offsets is not None and offsets[0] < offsets[1]+input_bytes[1]
            and offsets[1] < offsets[0]+input_bytes[0]):
        raise ValueError('fixed GEMM arena input partitions overlap')

    def multiply(left, right, orientations, out=None, _scalar_owners=(alpha, beta)):
        # CuPy's public dot can select pedantic/TF32 gemmEx compute modes.
        # Keep that numerical policy on its original path, even if it changes
        # after binding. These compatibility reads submit no device work.
        if debug_enabled() or compute_type(dtype) != default_compute_type:
            return None
        handle = cuda.Device().cublas_handle
        if cublas.getMathMode(handle) != cublas.CUBLAS_DEFAULT_MATH:
            return None
        if cublas.getPointerMode(handle) != cublas.CUBLAS_POINTER_MODE_HOST:
            raise RuntimeError('cuBLAS handle must already use host pointer mode')
        if input_offsets is not None:
            # Only the explicit arena path needs these extent checks. Ordinary
            # matrices retain their existing layout classification unchanged.
            if any(value.dtype != dtype or not value.flags.c_contiguous
                   or value.nbytes < offset+size
                   for value,offset,size in zip((left,right),offsets,input_bytes)):
                raise ValueError('fixed GEMM arena is smaller than its saved matrix partitions')
        left_pointer, right_pointer = left.data.ptr+offsets[0], right.data.ptr+offsets[1]
        if out is None:
            output = backend.empty(output_shape, dtype=dtype, order='C')
        else:
            if out.shape != output_shape or out.dtype != dtype or not out.flags.c_contiguous:
                raise ValueError('fixed GEMM output differs from its saved C layout')
            output_pointer = out.data.ptr
            if any(pointer < output_pointer+output_bytes and output_pointer < pointer+size
                   for pointer,size in zip((left_pointer,right_pointer),input_bytes)):
                raise ValueError('fixed GEMM output overlaps an input matrix')
            output = out
        left_orientation, right_orientation = orientations
        try:
            # Row-major C = A B is column-major C.T = B.T A.T. CuPy's
            # cuBLAS wrapper binds the current stream for this GEMM call.
            gemm(handle, operations[right_orientation], operations[left_orientation],
                 n, m, k, alpha_pointer, right_pointer, k if right_orientation else n,
                 left_pointer, m if left_orientation else k, beta_pointer, output.data.ptr, n)
        except BaseException as error:
            # Keep current owners alive until possibly submitted work finishes.
            # Normal execution never synchronizes here or retries another kernel.
            try:
                cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                error.add_note(f'GEMM cleanup also failed: {cleanup!r}')
            raise
        return output if output_permutation is None else backend.transpose(output, output_permutation)

    def classify(layouts):
        orientations = []
        for (shape, actual_dtype, actual), expected, patterns in zip(layouts, original_shapes, strides):
            if shape != expected or actual_dtype != dtype:
                return None
            if actual == patterns[0]:
                orientations.append(0)
            elif actual == patterns[1]:
                orientations.append(1)
            else:
                return None
        return tuple(orientations)

    def bind_layout(layouts):
        orientations = classify(layouts)
        # Store only the two structural orientation bits. Current device,
        # stream, pointers and precision policy remain invocation-local.
        return None if orientations is None else partial(multiply, orientations=orientations)

    def run(left, right, out=None):
        orientations = classify(((left.shape, left.dtype, left.strides),
                                 (right.shape, right.dtype, right.strides)))
        return None if orientations is None else multiply(left, right, orientations, out=out)

    run.bind_layout = bind_layout
    return run
