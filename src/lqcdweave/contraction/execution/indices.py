"""Backend invocation adapters; derived einsum spelling stays inside X."""
from functools import lru_cache

import opt_einsum

from lqcdweave.contracts.correlator.api import AxisPattern


def einsum_arguments(pattern: AxisPattern, operands):
    """Interleaved array/integer-list arguments for NumPy-compatible einsum."""
    operands = tuple(operands)
    if len(operands) != len(pattern.inputs):
        raise ValueError("one array is required per input axis sequence")
    local = pattern.normalized()
    arguments = []
    for operand, axes in zip(operands, local.inputs, strict=True):
        arguments.extend((operand, list(axes)))
    arguments.append(list(local.output))
    return tuple(arguments)


@lru_cache(maxsize=256)
def einsum_equation(pattern: AxisPattern) -> str:
    """Render only for backend APIs requiring string expressions."""
    local = pattern.normalized()
    def render(axes):
        return "".join(opt_einsum.get_symbol(axis) for axis in axes)
    return ",".join(render(axes) for axes in local.inputs) + "->" + render(local.output)
