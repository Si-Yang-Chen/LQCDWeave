"""Accurate thin reductions within the existing dense workspace allowance."""
from functools import lru_cache
from itertools import product
from math import prod
import numpy as np
from lqcdweave.contracts.correlator.api import parse_axis_pattern
from .diagnostics import debug_enabled


@lru_cache(maxsize=256)
def _description(equation, shapes, dtypes, force_matrix=False):
    if not dtypes or len(set(dtypes)) != 1:
        return None
    dtype = np.dtype(dtypes[0])
    if dtype not in (np.dtype('float32'), np.dtype('complex64')):
        return None
    equation = equation.normalized()
    axes, output = equation.inputs, equation.output
    if len(axes) not in (1, 2):
        return None
    dimensions = {}
    for labels, shape in zip(axes, shapes, strict=True):
        for label, size in zip(labels, shape, strict=True):
            if size <= 0 or dimensions.setdefault(label, size) != size:
                return None
    reduced = tuple(label for label in dimensions if label not in output)
    if not reduced or (not force_matrix and prod(dimensions[label] for label in reduced) < 2):
        return None
    if len(axes) == 2:
        # Ordinary matrix products retain their selected BLAS implementation.
        # An explicit saved wide Action may also describe Cartesian thin dots.
        left, right = map(set, axes)
        if not force_matrix and (left - right) & set(output) and (right - left) & set(output):
            return None
    wide = np.dtype('complex128' if dtype.kind == 'c' else 'float64')
    output_shape = tuple(dimensions[label] for label in output)
    output_bytes = prod(output_shape) * dtype.itemsize
    allowance = sum(prod(shape) * dtype.itemsize for shape in shapes) + output_bytes
    widths = {label: dimensions[label] for label in reduced}

    def scratch():
        raw = sum(prod(widths.get(label, dimensions[label]) for label in labels)
                  * dtype.itemsize for labels in axes)
        # Upcast inputs, packing/reduction scratch, two wide partials and the
        # final cast all fit inside P's ordinary input-bytes + output allowance.
        return 8 * raw + 4 * prod(output_shape) * wide.itemsize + 256

    while scratch() > allowance:
        candidates = [label for label in reduced if widths[label] > 1]
        if not candidates:
            return None
        label = max(candidates, key=lambda label: widths[label])
        widths[label] = (widths[label] + 1) // 2
    counts = tuple((dimensions[label] + widths[label] - 1) // widths[label]
                   for label in reduced)
    if prod(counts) > 256:
        return None
    return (axes, output, reduced, tuple(widths[label] for label in reduced),
            tuple(dimensions[label] for label in reduced), counts, wide.name, dtype.name,
            scratch(), allowance)


def bind_precise_reduction(equation, shapes, dtypes, backend, *, force_matrix=False):
    """Bind bounded wide partial sums; inputs/results keep the planned dtype.

    This is an implementation of one saved Action, never a path search. Only
    contracted labels are tiled, including every repeated diagonal occurrence.
    CPU/debug use tiled einsum; native GPU fuses loads and wide partial sums.
    """
    if not callable(getattr(backend, 'einsum', None)):
        return None
    equation = parse_axis_pattern(equation).normalized()
    if force_matrix and dtypes and len({np.dtype(value) for value in dtypes}) == 1:
        dtype = np.dtype(dtypes[0])
        if dtype in (np.dtype('float32'), np.dtype('complex64')):
            allowance = sum(prod(shape)*dtype.itemsize for shape in shapes)
            dimensions = {label:size for axes,shape in zip(equation.inputs,shapes,strict=True)
                          for label,size in zip(axes,shape,strict=True)}
            allowance += prod(dimensions[label] for label in equation.output)*dtype.itemsize
            # The native kernel needs only wide partials, not CPU upcast or
            # einsum buffers. Admit it against its own actual workspace bound.
            if not debug_enabled():
                from .reduction_kernel import bind_raw_reduction
                raw = bind_raw_reduction(equation,tuple(map(tuple,shapes)),dtype,backend,allowance)
                if raw is not None:
                    return raw
    description = _description(parse_axis_pattern(equation), tuple(map(tuple, shapes)),
                               tuple(np.dtype(dtype).str for dtype in dtypes), force_matrix)
    if description is None:
        if force_matrix:
            return _bind_scalar_wide(equation, shapes, dtypes, backend)
        return None
    axes, output, reduced, widths, dimensions, counts, wide, dtype, scratch, allowance = description
    if not debug_enabled():
        from .reduction_kernel import bind_raw_reduction
        raw = bind_raw_reduction(parse_axis_pattern(equation).normalized(),
            tuple(map(tuple, shapes)), np.dtype(dtype), backend, allowance)
        if raw is not None:
            return raw
    # Only immutable geometry is retained; every invocation owns its accumulator.
    tiles = tuple(tuple(slice(i * width, min((i + 1) * width, size))
                        for i, width, size in zip(index, widths, dimensions, strict=True))
                  for index in product(*(range(count) for count in counts)))
    selections = tuple(tuple(tuple(dict(zip(reduced, tile)).get(label, slice(None))
                                   for label in labels) for labels in axes) for tile in tiles)

    def run(*operands):
        accumulator = None
        for selection in selections:
            arguments = tuple(item for value, labels, take in zip(operands, axes, selection, strict=True)
                              for item in (value[take], labels)) + (output,)
            partial = backend.einsum(*arguments, optimize=False, dtype=wide)
            if accumulator is None:
                accumulator = partial
            else:
                accumulator += partial
        return accumulator.astype(dtype, copy=False)
    run.workspace_bytes = scratch
    run.workspace_allowance = allowance
    run.tile_count = len(tiles)
    return run


def _bind_scalar_wide(equation, shapes, dtypes, backend):
    """Tiny CPU/debug oracle when conservative vector tiles cannot fit."""
    if not dtypes or len(set(map(np.dtype,dtypes))) != 1:
        return None
    dtype=np.dtype(dtypes[0])
    if dtype not in (np.dtype('float32'),np.dtype('complex64')):
        return None
    dimensions={label:size for axes,shape in zip(equation.inputs,shapes,strict=True)
                for label,size in zip(axes,shape,strict=True)}
    output=equation.output
    reduced=tuple(label for label in dimensions if label not in output)
    wide=np.dtype('complex128' if dtype.kind=='c' else 'float64')
    output_shape=tuple(dimensions[label] for label in output)
    output_bytes=prod(output_shape)*dtype.itemsize
    allowance=sum(prod(shape)*dtype.itemsize for shape in shapes)+output_bytes
    scratch=5*wide.itemsize+output_bytes
    if not reduced or scratch>allowance:
        return None
    indices=tuple(tuple((label in output,(output if label in output else reduced).index(label))
                       for label in axes) for axes in equation.inputs)
    def run(*operands):
        result=backend.empty(output_shape,dtype=dtype)
        for point in product(*(range(size) for size in output_shape)):
            total=backend.zeros((),dtype=wide)
            for inner in product(*(range(dimensions[label]) for label in reduced)):
                values=tuple(backend.asarray(value[tuple((point if outer else inner)[i]
                             for outer,i in axes)],dtype=wide)
                             for value,axes in zip(operands,indices,strict=True))
                term=values[0] if len(values)==1 else values[0]*values[1]
                total+=term
                del values,term
            result[point]=total
        return result
    run.workspace_bytes=scratch
    run.workspace_allowance=allowance
    run.tile_count=prod(dimensions[label] for label in reduced)
    return run


__all__ = ['bind_precise_reduction']
