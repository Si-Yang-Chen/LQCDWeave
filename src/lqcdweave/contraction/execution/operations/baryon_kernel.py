"""Bind and prepare the saved baryon operation without replanning its path.

This is X's compatibility execution of an existing Action, not a new global
planner. All paths and workspace decisions still arrive from the saved plan.
Only immutable programs are cached; current numeric preparation is per call.
"""
from dataclasses import dataclass
from functools import lru_cache

import opt_einsum

from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern
from ..indices import einsum_arguments, einsum_equation
from .contract import _contract_value
from .baryon_sparse_spin import (
    contract_sparse_spin, prepare_sparse_spin_elemental,
    sparse_spin_pattern_supported, sparse_spin_recipe,
)


@lru_cache(maxsize=256)
def _join_expression(equation, shapes, dtypes, kernel_path):
    """Fixed native-array expression with no numerical constants or owners."""
    del dtypes
    return opt_einsum.contract_expression(
        einsum_equation(parse_axis_pattern(equation)), *shapes, optimize=list(kernel_path),
    )


@lru_cache(maxsize=256)
def _fixed_step_equations(equation, kernel_path):
    """Lower only saved positions; no contraction-path discovery."""
    equation = parse_axis_pattern(equation)
    final_output = equation.output
    pending = list(equation.inputs)
    equations = []
    for positions in kernel_path:
        if not positions or len(set(positions)) != len(positions) or any(
            index < 0 or index >= len(pending) for index in positions
        ):
            raise ValueError('invalid explicit baryon kernel step')
        selected = [pending[index] for index in positions]
        remaining = [labels for index, labels in enumerate(pending) if index not in positions]
        needed = set(final_output).union(*(set(axes) for axes in remaining))
        output = (tuple(dict.fromkeys(name for axes in selected for name in axes if name in needed))
                  if remaining else final_output)
        equations.append(AxisPattern(tuple(selected), output))
        pending = remaining + [output]
    if len(pending) != 1 or pending[0] != final_output:
        raise ValueError('explicit baryon path does not finish the equation')
    return tuple(equations)


def _contract_fixed_steps(equation, operands, kernel_path, backend, device):
    current = list(operands)
    for positions, step_equation in zip(kernel_path, _fixed_step_equations(equation, kernel_path)):
        selected = tuple(current[index] for index in positions)
        value, _ = _contract_value(step_equation, selected, backend, device)
        current = [item for index, item in enumerate(current) if index not in positions]
        current.append(value)
        del selected
    return current[0]


@dataclass(frozen=True)
class BaryonKernelBinding:
    equation: AxisPattern
    shapes: tuple
    dtypes: tuple
    path: tuple
    entries: tuple | None
    sparse_recipe: object
    foreign_inputs: bool
    no_search: bool

    @property
    def implementation(self):
        if self.sparse_recipe is not None:
            return 'saved-sparse-spin'
        return 'saved-fixed-steps' if self.no_search else 'saved-expression'

    def prepare(self, spin, spatial, backend):
        """Use current prepared input blocks; no reader, queue or upload owner.

        Sparse Phi is packed at the same operation's start, then released at
        its end. This does not activate the isolated cross-operation prepack
        strategy or increase the plan's persistent residency.
        """
        packed, expression = None, None
        if self.sparse_recipe is not None:
            packed = prepare_sparse_spin_elemental(self.sparse_recipe, spatial, backend)
        elif not self.no_search and not self.foreign_inputs:
            expression = _join_expression(self.equation, self.shapes, self.dtypes, self.path)
        return PreparedBaryonKernel(self, backend, spin, spatial, packed, expression)


@lru_cache(maxsize=256)
def bind_baryon_kernel(equation, shapes, dtypes, path, entries, foreign_inputs, no_search,
                       kernel_description=None):
    """Bind only metadata of a saved Action's existing numerical behavior.

    Legacy Actions declare sparse entries and a dense fallback path. Resolve
    that established compatibility dispatch at binding, never in compute.
    No alternative path, precision, tile width or hardware is selected here.
    """
    if kernel_description is not None:
        if kernel_description.implementation != 'ordered-join-v1':
            raise ValueError('unknown baryon kernel implementation')
        # Fixed implementation, not global path search. A leg may already
        # contain several endpoints, including a closed trace topology.
        count = len(shapes)
        steps = [(0, 2)]
        count -= 1
        while count > 1:
            steps.append((0, count - 1))
            count -= 1
        path = tuple(steps)
        # P's coarse workspace estimate is not a runtime allocation bound.
        # The existing execution allocator enforces the actual resource cap;
        # binding neither reconstructs TensorSpecs nor repeats M's estimate.
    recipe = None
    if entries is not None and not foreign_inputs and sparse_spin_pattern_supported(entries):
        recipe = sparse_spin_recipe(equation, shapes, dtypes, path)
    return BaryonKernelBinding(equation, shapes, dtypes, path, entries, recipe,
                              foreign_inputs, no_search)


class PreparedBaryonKernel:
    """One operation's current numerical state; never cached across calls."""
    def __init__(self, binding, backend, spin, spatial, packed, expression):
        self.binding, self.backend = binding, backend
        self.spin, self.spatial, self.packed = spin, spatial, packed
        self.expression = expression
        self.closed = False

    def compute(self, legs, *, device=False):
        if self.closed:
            raise RuntimeError('baryon preparation has already been released')
        binding, backend = self.binding, self.backend
        operands = (self.spin, self.spatial, *legs)
        if binding.sparse_recipe is not None:
            return contract_sparse_spin(binding.sparse_recipe, binding.entries, operands,
                                        backend, prepared_elemental=self.packed), device
        if binding.foreign_inputs and not binding.no_search:
            # Foreign backend conversion must never enter a cached expression.
            return opt_einsum.contract(*einsum_arguments(binding.equation, operands),
                optimize=list(binding.path), backend=backend.__name__), False
        if binding.no_search:
            return _contract_fixed_steps(binding.equation, operands, binding.path, backend, device), False
        return self.expression(*operands, backend=backend.__name__), False

    def close(self):
        self.spin = self.spatial = self.packed = self.expression = None
        self.closed = True


__all__ = ['bind_baryon_kernel', 'BaryonKernelBinding', 'PreparedBaryonKernel']
