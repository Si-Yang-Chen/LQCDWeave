"""Cheap metadata cost model for a declared local whole-vertex algorithm.

This returns work and storage estimates, never an executable path. Its fixed
shape projection is a model of the declared implementation, not a path search.
No production arrays, spin support discovery or trial execution are involved.
"""
from math import prod


def estimate_vertex_kernel(description, equation, specs):
    if description.implementation != 'ordered-join-v1':
        raise ValueError('unsupported vertex cost model')
    labels = list(equation.inputs)
    shapes = [tuple(spec.shape) for spec in specs]
    dimensions = {label: size for axes, shape in zip(labels, shapes)
                  for label, size in zip(axes, shape)}
    itemsize = max(spec.dtype.itemsize for spec in specs)
    input_bytes = sum(prod(shape) * spec.dtype.itemsize for shape, spec in zip(shapes, specs))
    work, generated, largest = 0, 0, 0
    # The model uses one spin projection then absorbs elemental/remaining legs.
    # It intentionally ignores sparse speedups, GEMM packing reuse and BLAS.
    positions = (0, 2)
    while len(labels) > 1:
        selected = [labels[index] for index in positions]
        remaining = [axes for index, axes in enumerate(labels) if index not in positions]
        needed = set(equation.output).union(*(set(axes) for axes in remaining))
        output = (tuple(dict.fromkeys(label for axes in selected for label in axes if label in needed))
                  if remaining else equation.output)
        extent = prod(dimensions[label] for label in output)
        work += prod(dimensions[label] for label in set().union(*map(set, selected)))
        # Current and previous generated values plus possible contiguous packs.
        largest = max(largest, generated + 3 * extent * itemsize)
        generated = extent * itemsize
        labels = remaining + [output]
        positions = (0, len(labels) - 1)
    traffic = input_bytes + generated
    return work, traffic, input_bytes + largest
