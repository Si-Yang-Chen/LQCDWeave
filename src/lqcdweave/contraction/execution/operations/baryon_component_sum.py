"""Accumulate whole-vertex components before their shared final leg.

This is a local implementation of an ordered-join-v1 vertex. It does not
change the saved Action, its external operands, or the global schedule.
Only immutable expressions are cached; preparation retains current inputs.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import AxisPattern
from lqcdweave.contracts.correlator.cores import baryon_component_pattern
from ..lowering import bind_matmul, lower_binary_matmul
from .baryon_kernel import _fixed_step_equations, _join_expression


@lru_cache(maxsize=256)
def _program(equation, shapes, dtypes, backend):
    # Two separate merged legs, optionally sharing one retained leading lane.
    # Other repeated axes and mixed precision keep their implementation.
    if (len(shapes) != 4 or shapes[0] != (4, 4, 4)
            or len(set(dtypes)) != 1
            or np.dtype(dtypes[0]).kind != 'c'
            or len(equation.output) not in (6, 7)
            or any(len(set(axes)) != len(axes) for axes in equation.inputs)):
        return None
    spin, spatial, left, right = equation.inputs
    lane = equation.output[0] if len(equation.output) == 7 else None
    shared = set() if lane is None else {lane}
    if lane is None:
        if tuple(map(len, shapes)) != (3, 3, 6, 6):
            return None
    elif (len(spatial) not in (3, 4) or len(left) != 7 or len(right) != 7
          or left[0] != lane or right[0] != lane or lane in spin
          or (len(spatial) == 4 and spatial[0] != lane)):
        return None
    modes = spatial[-3:]
    internal = set(spin + modes)
    if (len(internal) != 6 or internal.intersection(equation.output)
            or len(set(left).intersection(spin)) != 2
            or len(set(left).intersection(modes)) != 2
            or len(set(right).intersection(spin)) != 1
            or len(set(right).intersection(modes)) != 1
            or set(left).intersection(right) != shared):
        return None
    steps = _fixed_step_equations(equation, ((0, 2), (0, 2), (0, 1)))
    partial = AxisPattern(equation.inputs[:-1], steps[-2].output)
    sizes = dict(item for axes, shape in zip(equation.inputs, shapes)
                 for item in zip(axes, shape))
    partial_shape = tuple(sizes[axis] for axis in partial.output)
    # Keep the accumulating partials smaller than the old full-size total
    # and current component. This does not introduce persistent residency.
    if 2 * prod(partial_shape) > prod(sizes[axis] for axis in equation.output):
        return None
    if lane is not None and not (backend.__name__ == 'numpy'
                                 and all(sizes[axis] == 1 for axis in modes)):
        # Reuse X's existing matrix lowering for the same saved binary steps.
        # Generic batch einsum otherwise misses BLAS for the shared lane.
        # Degenerate one-mode CPU tensors keep the smaller expression setup.
        first_shape = tuple(sizes[axis] for axis in steps[0].output)
        first = _join_expression(steps[0], (shapes[0], shapes[2]),
                                 (dtypes[0], dtypes[2]), ((0, 1),))
        second = bind_matmul(lower_binary_matmul(steps[1], (shapes[1], first_shape),
                             (dtypes[1], dtypes[0])), backend)
        final = bind_matmul(lower_binary_matmul(steps[2], (shapes[3], partial_shape),
                            (dtypes[3], dtypes[0])), backend)

        def prefix(spin, spatial, left, **kwargs):
            return second(spatial, first(spin, left, backend=backend.__name__))

        def suffix(right, accumulated, **kwargs):
            return final(right, accumulated)

        return prefix, suffix
    prefix = _join_expression(partial, shapes[:-1], dtypes[:-1], ((0, 2), (0, 1)))
    suffix = _join_expression(steps[-1], (shapes[-1], partial_shape),
                              (dtypes[-1], dtypes[-1]), ((0, 1),))
    return prefix, suffix


@dataclass
class PreparedComponentSum:
    prefix: object
    suffix: object
    components: tuple
    legs: tuple
    backend: object

    def compute(self):
        backend = self.backend
        total = None
        for spin, spatial, coefficient in self.components:
            value = backend.array(self.prefix(spin, spatial, self.legs[0],
                                  backend=backend.__name__), copy=True)
            if coefficient != 1:
                value *= coefficient
            if total is None:
                total = value
            else:
                total += value
        return self.suffix(self.legs[1], total, backend=backend.__name__)

    def close(self):
        self.components = self.legs = ()
        self.prefix = self.suffix = self.backend = None


def prepare_component_sum(action, arrays, backend, dtype):
    """Bind the supported local representation, or leave the old path intact."""
    core = action.core
    description = getattr(action, 'kernel_description', None)
    if (len(core.components) < 2 or description is None
            or description.implementation != 'ordered-join-v1'):
        return None
    legs = action.operands[len(core.inputs):]
    if len(legs) != 2:
        return None
    first = core.components[0]
    values = tuple(arrays[key] for key in first.inputs + legs)
    if backend.__name__ != 'numpy' and any(isinstance(value, np.ndarray) for value in values):
        return None
    shapes = tuple(value.shape for value in values)
    dtypes = tuple(str(value.dtype) for value in values)
    if any(value.dtype != dtype for component in core.components
           for value in (arrays[key] for key in component.inputs)):
        return None
    if any(tuple(arrays[key].shape for key in component.inputs) != shapes[:2]
           for component in core.components):
        return None
    equation = baryon_component_pattern(action.equation.inputs[0],
        action.equation.inputs[1:], action.equation.output, values[1].ndim)
    program = _program(equation, shapes, dtypes, backend)
    if program is None:
        return None
    return PreparedComponentSum(*program,
        tuple((arrays[part.inputs[0]], arrays[part.inputs[1]], part.coefficient)
              for part in core.components), values[2:], backend)
