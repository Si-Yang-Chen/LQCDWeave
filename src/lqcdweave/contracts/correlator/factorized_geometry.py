"""Bounded sharing of validated immutable thin-wide integer geometry only."""
from functools import lru_cache
from math import prod
from typing import NamedTuple

import numpy as np

from .allocation import device_allocation_bytes


class ReductionGeometry(NamedTuple):
    dimensions: tuple
    small_operands: tuple
    small_reduced_axes: tuple
    small_reduction_elements: tuple
    gram_axes: tuple
    reduced_axes: tuple
    gram_elements: int
    epilogue_elements: int
    reduction_elements: int
    output_shape: tuple
    output_elements: int
    blocks: int
    partial_bytes: int
    tile_width: int
    tiles: int
    partial_elements: int
    accumulator_bytes: int
    kernel_calls: int
    workspace_bytes: int
    flops: int
    traffic_bytes: int


def gram_matrix_axes(equation, large, geometry):
    """A conservative full-Gram GEMM domain, proved by labels rather than shape."""
    if (len(equation.inputs) != 4 or geometry.blocks != 1 or geometry.reduction_elements < 256
            or not 4096 <= geometry.gram_elements < 2**31 or geometry.tile_width < 1024
            or any(geometry.small_reduced_axes)):
        return None
    a, b = (set(equation.inputs[i]) for i in large)
    output = set(equation.output)
    if output.intersection(a | b):
        return None  # Retained coordinate lanes keep the ordinary reduction.
    gram = geometry.gram_axes
    left = tuple(axis for axis in gram if axis in a-b)
    right = tuple(axis for axis in gram if axis in b-a)
    if not left or not right or gram != left+right:
        return None
    small = tuple(equation.inputs[i] for i in geometry.small_operands)
    internal = tuple(set(axes)-output for axes in small)
    retained = tuple(set(axes)&output for axes in small)
    if (internal[0].intersection(internal[1]) or internal[0] | internal[1] != set(gram)
            or retained[0].intersection(retained[1]) or retained[0] | retained[1] != output):
        return None
    return left, right


def _calculate_geometry(equation, shapes, large, tile_elements, result_dtype, gram_strategy='reduction'):
    dimensions = {}
    for axes, shape in zip(equation.inputs, shapes, strict=True):
        if len(axes) != len(shape) or len(set(axes)) != len(axes):
            raise ValueError('thin wide inputs require non-diagonal ordered axes')
        for label, size in zip(axes, shape, strict=True):
            if type(size) is not int or size <= 0 or dimensions.get(label, size) != size:
                raise ValueError('thin wide dimensions must be positive and aligned')
            dimensions[label] = size
    output = equation.output
    if len(set(output)) != len(output) or not set(output) <= dimensions.keys():
        raise ValueError('invalid thin wide output axes')
    small_operands = tuple(i for i in range(len(shapes)) if i not in large)
    big = tuple(equation.inputs[i] for i in large)
    small = tuple(equation.inputs[i] for i in small_operands)
    small_axes = set().union(*small)
    needed = set(output) | small_axes
    gram_axes = tuple(dict.fromkeys(label for axes in big for label in axes if label in needed))
    gram_set = set(gram_axes)
    reduced_axes = tuple(dict.fromkeys(label for axes in big for label in axes if label not in gram_set))
    retained = gram_set | set(output)
    small_reduced_axes = tuple(tuple(axis for axis in axes if axis not in retained) for axes in small)
    if (not reduced_axes or not set(reduced_axes) <= set(big[0]) & set(big[1])
            or set(reduced_axes).intersection(small_axes)
            or tile_elements is None and not small_axes <= gram_set | set(output)
            or len(small) == 2 and set(small_reduced_axes[0]).intersection(small_reduced_axes[1])):
        raise ValueError('integer factors are outside the bounded thin wide domain')
    gram_elements = prod(dimensions[axis] for axis in gram_axes)
    epilogue_elements = prod(dimensions[axis] for axis in gram_axes if axis not in output)
    reduction_elements = prod(dimensions[axis] for axis in reduced_axes)
    output_shape = tuple(dimensions[axis] for axis in output)
    output_elements = prod(output_shape)
    blocks = min(64, max(1, (reduction_elements + 1023) // 1024))
    tile_width = min(epilogue_elements, tile_elements or epilogue_elements)
    partial_elements = gram_elements // epilogue_elements * tile_width
    if (epilogue_elements < 1 or tile_elements is None and epilogue_elements > 4096
            or reduction_elements < 2 or output_elements >= 2**31 or partial_elements * blocks >= 2**31):
        raise ValueError('integer factors are outside the bounded thin wide domain')
    tiles = (epilogue_elements + tile_width - 1) // tile_width
    partial_bytes = partial_elements * blocks * 16
    accumulator_bytes = output_elements * 16 if tiles > 1 else 0
    small_reduction_elements = tuple(prod(dimensions[axis] for axis in axes) for axes in small_reduced_axes)
    output_bytes = output_elements * np.dtype(result_dtype).itemsize
    device = (max(128, device_allocation_bytes(partial_bytes))
              + (device_allocation_bytes(accumulator_bytes) if accumulator_bytes else 0)
              + device_allocation_bytes(output_bytes) - output_bytes)
    workspace = device
    if tile_elements is not None and gram_elements <= partial_elements:
        reference = (gram_elements * 16 + output_elements * 16
                     + sum(prod(shape) * 16 for shape in shapes))
        workspace = max(device, reference)
    elif tile_elements is not None:
        reference = (partial_elements * 16 + output_elements * 16
                     + sum(prod(shapes[i]) * 16 for i in small_operands)
                     + 2 * reduction_elements * 16 + 4 * tile_width * 16
                     + (len(gram_axes) + 1) * tile_width * 8)
        workspace = max(device, reference)
    flops = (gram_elements * reduction_elements
             + sum(small_reduction_elements) * output_elements * epilogue_elements * blocks
             + output_elements * max(0, tiles - 1))
    traffic = (2 * gram_elements * reduction_elements * 8
               + (output_elements + 1) * gram_elements * blocks * 16
               + output_elements * gram_elements * blocks * 16 + output_bytes
               + 2 * max(0, tiles - 1) * output_elements * 16
               + (sum(small_reduction_elements) * output_elements * epilogue_elements * blocks * 8
                  if tile_elements is not None else 0))
    geometry = ReductionGeometry(tuple(dimensions.items()), small_operands, small_reduced_axes,
        small_reduction_elements, gram_axes, reduced_axes, gram_elements, epilogue_elements,
        reduction_elements, output_shape, output_elements, blocks, partial_bytes, tile_width,
        tiles, partial_elements, accumulator_bytes, 2 * tiles, workspace, flops, traffic)
    if gram_strategy == 'gemm':
        if tile_elements is None or gram_matrix_axes(equation, large, geometry) is None:
            raise ValueError('full wide Gram GEMM is outside its declared matrix domain')
        # One aligned arena contains both wide packs and the full Gram. Its
        # owner remains live while the independent final result is allocated.
        packed = tuple(prod(shapes[i])*16 for i in large)
        gram_bytes = gram_elements*16
        arena_bytes = sum(device_allocation_bytes(n) for n in (*packed,gram_bytes))
        device_workspace = arena_bytes+device_allocation_bytes(output_bytes)-output_bytes
        reference = gram_bytes+output_elements*16+sum(prod(shape)*16 for shape in shapes)
        gemm_traffic = (sum(prod(shapes[i])*8 for i in large)+2*sum(packed)+gram_bytes
                        + 2*output_elements*gram_bytes+output_bytes)
        return geometry._replace(partial_bytes=gram_bytes,tile_width=epilogue_elements,
            tiles=1,partial_elements=gram_elements,accumulator_bytes=0,kernel_calls=3,
            workspace_bytes=max(device_workspace,reference),
            flops=gram_elements*reduction_elements+sum(small_reduction_elements)*output_elements*epilogue_elements,
            traffic_bytes=gemm_traffic)
    return geometry


# Exact ordered integer specifications only. Core/Action identities stay distinct;
# the bounded cache contains no arrays, readers, tensor identities or pointers.
_cached_geometry = lru_cache(maxsize=2048)(_calculate_geometry)


def bind_geometry(equation, shapes, large, tile_elements, result_dtype, gram_strategy='reduction'):
    if all(type(size) is int for shape in shapes for size in shape):
        return _cached_geometry(equation, shapes, large, tile_elements, result_dtype, gram_strategy)
    # Preserve normal validation/error ordering for unsupported, possibly
    # unhashable sizes rather than allowing the cache key to raise TypeError.
    return _calculate_geometry(equation, shapes, large, tile_elements, result_dtype, gram_strategy)
