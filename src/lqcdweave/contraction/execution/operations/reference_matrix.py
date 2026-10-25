"""Prebind supported reference binary reductions with the same workspace bound."""
from math import prod
import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern
from ..lowering import lower_binary_matmul, bind_matmul


def bind_reference_matrix(equation, specs, backend):
    if getattr(backend, '__name__', '') != 'cupy' or len(specs) != 2:
        return None
    if specs[0].dtype != specs[1].dtype:
        return None
    local = equation.normalized()
    axes, output = local.inputs, local.output
    if (len(axes) != 2 or any(len(a) != len(set(a)) for a in (*axes, output))
            or any(s.dtype.kind not in 'fc' or s.dtype.itemsize not in (4, 8, 16) for s in specs)
            or any(n <= 1 for s in specs for n in s.shape)):
        return None
    dtype = np.result_type(*(s.dtype for s in specs))
    if dtype not in tuple(map(np.dtype, (np.float32,np.float64,np.complex64,np.complex128))):
        return None
    if any(not callable(getattr(backend, name, None)) for name in ('transpose','reshape','matmul')):
        return None
    output_set = set(output)
    reductions = tuple(tuple(i for i,label in enumerate(a) if label not in output_set | set(axes[1-side]))
                       for side,a in enumerate(axes))
    labels = tuple(tuple(label for i,label in enumerate(a) if i not in r)
                   for a,r in zip(axes,reductions))
    common = set(labels[0]) & set(labels[1])
    contracted = tuple(sorted(common - output_set))
    if not contracted:
        return None
    shapes = tuple(tuple(n for i,n in enumerate(s.shape) if i not in r)
                   for s,r in zip(specs,reductions))
    # Match CuPy's sorted contraction-label order before the existing GEMM.
    orders = tuple(tuple(a.index(label) for label in (*[x for x in a if x not in contracted], *contracted))
                   for a in labels)
    arranged = tuple(tuple(a[i] for i in order) for a,order in zip(labels,orders))
    arranged_shapes = tuple(tuple(s[i] for i in order) for s,order in zip(shapes,orders))
    recipe = lower_binary_matmul(AxisPattern(arranged, output), arranged_shapes, (dtype,dtype))
    if recipe is None:
        return None
    sizes = tuple(prod(s)*dtype.itemsize for s in shapes)
    preparation = sum(size for size,s,r in zip(sizes,specs,reductions) if r or s.dtype != dtype)
    input_bytes = sum(prod(s.shape)*s.dtype.itemsize for s in specs)
    output_bytes = prod(recipe.grouped_shape)*dtype.itemsize
    # P's ordinary binary allowance is input bytes + one output, in addition
    # to the owned result. Include both possible matrix packing copies here.
    if preparation + sum(sizes) > input_bytes + output_bytes:
        return None
    multiply = bind_matmul(recipe, backend)

    def run(left, right):
        values = []
        for value, reduced, order in zip((left,right), reductions, orders):
            if reduced:
                value = value.sum(axis=reduced, dtype=dtype)
            if order != tuple(range(len(order))):
                value = value.transpose(order)
            values.append(value)
        return multiply(*values)
    return run
