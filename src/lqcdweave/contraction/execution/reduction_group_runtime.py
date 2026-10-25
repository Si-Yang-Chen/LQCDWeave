"""Shared host ABI and ownership for bound, independent raw reduction groups."""
from collections import OrderedDict
from math import prod
from types import ModuleType

import numpy as np

from .reduction_kernel import _kernel, _THREADS, _MAX_LAYOUTS


MAX_TASKS = 8
FAILED_SUBMISSIONS = []
_SHARED_OUTPUT_MAX_BYTES = 512


def current_pointer(value):
    pointer = value.ctypes.data if isinstance(value, np.ndarray) else value.data.ptr
    if type(pointer) is not int or not 0 < pointer < 2**64:
        raise ValueError('reduction group requires a current nonzero array pointer')
    return pointer


def _address(pointer):
    return pointer


def _native_output_rounding(state):
    """Only the ordinary dense domain has the per-owner allocation proof.

    Mixed opaque/core/sparse scenes retain independent outputs. Complete cfg
    could cover more domains, but this common binder does not infer that proof.
    """
    return all(action.kind in ('load', 'release', 'contract')
               and action.core is None and action.resource is None
               for action in state.actions)


def _native_cupy_views(backend, array_type):
    # Do not import/probe CUDA or apply this API contract to protocol backends.
    return type(backend) is ModuleType and getattr(array_type, '__module__', None) == 'cupy'


def bind_group_runtime(backend, shapes, dtype, allowances, blocks, output_shape,
                       sources, names, grids, constants, argument_binder, *, share_small_outputs=False, capacity=MAX_TASKS):
    """Fix the mode at binding; the warm call only consumes current arrays.

    The source family fixes its input-pair/stride and non-overlapping result
    slot capacity at binding; matrix groups retain the default eight slots.
    Geometry, launch grids and trailing constants are already selected. No front-end description or execution policy is inspected here.
    """
    count = len(allowances)
    left_shape, right_shape = shapes
    factory, array_type = backend.RawKernel, getattr(backend, 'ndarray', None)
    wide = np.dtype('complex128' if dtype.kind == 'c' else 'float64')
    outputs = prod(output_shape)
    partial_elements = count * outputs * blocks
    partial_bytes = partial_elements * wide.itemsize
    first_constants, finish_constants = (tuple(np.int64(n) for n in values) for values in constants)
    first_grid, finish_grid = grids
    threads = (_THREADS,)
    layouts = OrderedDict()
    output_bytes = count * outputs * dtype.itemsize
    # CuPy rounds each positive allocation request to at least 512 bytes.
    # If the entire payload fits one request unit, retaining any subset of
    # these views costs no more than one of the previous independent owners.
    # Fix the strategy here; do not inspect frontend policy on a warm call.
    output_slices = (tuple(slice(i*outputs, (i+1)*outputs) for i in range(count))
                     if share_small_outputs and output_bytes <= _SHARED_OUTPUT_MAX_BYTES else ())
    reshape_in_place = bool(output_slices) and _native_cupy_views(backend, array_type)

    def bind_layout(strides):
        # Normal ndarray strides are immutable integers. Audit only unseen
        # layouts, not each axis on every warm call.
        for pair in strides:
            for steps, shape in zip(pair, shapes, strict=True):
                if len(steps) != len(shape) or any(type(n) is not int or not -(2**63) <= n < 2**63 for n in steps):
                    raise ValueError('reduction group strides must be signed 64-bit byte offsets')
        first, finish = (_kernel(factory, source, name) for source, name in zip(sources, names, strict=True))
        template, positions = [], []
        for task in range(capacity):
            for side in range(2):
                if task < count:
                    positions.append((len(template), 2*task+side)); template.append(None)
                else:
                    template.append(np.uint64(0))
            template.extend(np.int64(step) for side in range(2)
                            for step in (strides[task][side] if task < count else (0,)*len(shapes[side])))
        template.extend((*first_constants, None))
        last = (None,) + (None,)*(count-1) + (np.uint64(0),)*(capacity-count) + (*finish_constants, None)
        first_args = argument_binder(backend, first, first_grid, threads, tuple(template), positions, pointer_getter=_address)
        finish_args = argument_binder(backend, finish, finish_grid, threads, last,
                                     tuple((i, i) for i in range(count)), pointer_getter=_address)
        return first, finish, first_args, finish_args, tuple(template), tuple(positions), last

    def run(tasks):
        tasks = tuple(tuple(pair) for pair in tasks)
        if len(tasks) != count or any(len(pair) != 2 for pair in tasks):
            raise ValueError('reduction group task count differs from bound geometry')
        values, pointers, strides = [], [], []
        for left, right in tasks:
            if array_type is not None and (not isinstance(left, array_type) or not isinstance(right, array_type)):
                raise ValueError('reduction group inputs must belong to the bound array backend')
            if (left.shape != left_shape or left.dtype != dtype
                    or right.shape != right_shape or right.dtype != dtype):
                raise ValueError('reduction group inputs differ from bound geometry')
            values.extend((left, right))
            pointers.extend((current_pointer(left), current_pointer(right)))
            strides.append((tuple(left.strides), tuple(right.strides)))
        strides = tuple(strides)
        bound = layouts.get(strides)
        if bound is None:
            bound = bind_layout(strides); layouts[strides] = bound
            if len(layouts) > _MAX_LAYOUTS:
                layouts.popitem(last=False)
        else:
            layouts.move_to_end(strides)
        first, finish, first_args, finish_args, template, positions, last = bound
        partials = device = shared_output = None
        results = []
        try:
            from .device import active_device_execution
            device = active_device_execution()
            borrow = getattr(device, 'reduction_partial', None) if device is not None and device.backend is backend else None
            if callable(borrow):
                partials = borrow(partial_bytes, wide)
            if partials is None:
                partials = backend.empty((partial_elements,), dtype=wide)
            if output_slices:
                shared_output = backend.empty((count*outputs,), dtype=dtype)
                for section in output_slices:
                    # Slicing first keeps scalar results as zero-dimensional
                    # arrays instead of extracting a NumPy/CuPy scalar.
                    view = shared_output[section]
                    if reshape_in_place:
                        # Public CuPy shape assignment cannot copy. Preserve
                        # the common ndarray.base root used by packing's
                        # adjacent-view borrowing, with no second ndarray.
                        view.shape = output_shape
                    else:
                        view = view.reshape(output_shape)
                    results.append(view)
            else:
                results.extend(backend.empty(output_shape, dtype=dtype) for _ in range(count))
            if first_args is None:
                arguments = list(template)
                for position, index in positions:arguments[position] = values[index]
                arguments[-1] = partials
                first(first_grid, threads, tuple(arguments))
            else:
                first_args(pointers, partials)
            if finish_args is None:
                arguments = list(last); arguments[0] = partials; arguments[-1] = results[0]
                arguments[1:count] = results[1:]
                finish(finish_grid, threads, tuple(arguments))
            else:
                finish_args((current_pointer(partials), *(current_pointer(value) for value in results[1:])), results[0])
            return tuple(results)
        except BaseException as error:
            try:
                backend.cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                FAILED_SUBMISSIONS.append((backend, tuple(values), partials, tuple(results),
                                           getattr(device, 'pool', None), shared_output))
                if callable(getattr(error, 'add_note', None)):
                    error.add_note(f'reduction group cleanup also failed: {cleanup!r}')
            raise

    run.output_owner_count = 1 if output_slices else count
    run.shared_output_bytes = output_bytes if output_slices else 0
    run.output_view_method = ('inplace-shape' if reshape_in_place else
                             'slice-reshape' if output_slices else 'independent')
    run.partial_bytes = partial_bytes
    run.workspace_bytes = partial_bytes + count*outputs*dtype.itemsize
    run.workspace_allowance = sum(allowances)
    run.blocks, run.task_count, run.capacity = blocks, count, capacity
    run.parameter_bytes = ((capacity*(2+sum(map(len, shapes)))+len(first_constants)+1)*8,
                           (capacity+len(finish_constants)+1)*8)
    run.raw_reduction = True
    return run
