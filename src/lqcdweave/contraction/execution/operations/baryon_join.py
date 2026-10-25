"""Baryon join operation over the existing shape-selected kernel path."""

import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern

from lqcdweave.contracts.correlator.cores import baryon_component_pattern
from .transfer import copy_to_backend
from .baryon_component_sum import prepare_component_sum
from .baryon_scalar_sum import CPU_MIN_MODES, prepare_scalar_sum
from .baryon_kernel import (
    bind_baryon_kernel,
    _join_expression, _fixed_step_equations, _contract_fixed_steps,
)


def contract_baryon_join(action, arrays, backend, *, timed, device=False, no_search=False):
    """Execute a preplanned baryon join and store its owned result."""
    if action.reused:
        timed("reuse", lambda: None)
        return
    equation = parse_axis_pattern(action.equation)
    core = getattr(action, "core", None)
    if core is not None:
        core_labels, *leg_labels = action.equation.inputs
        output = action.equation.output
        core_inputs = core.inputs
        legs = action.operands[len(core_inputs):]
        dtype = np.result_type(core.spec.dtype, *(arrays[key].dtype for key in core_inputs + legs))
        if any(part.coefficient.imag for part in core.components):
            dtype = np.result_type(dtype, np.complex64)
        if (not no_search and len(legs) == 1
                and (backend is not np or (core.spec.shape[-1] >= CPU_MIN_MODES
                     and core.spec.shape[-2] >= CPU_MIN_MODES
                     and core.spec.shape[-3] >= CPU_MIN_MODES))):
            prepared = timed('baryon_scalar_prepare',
                lambda: prepare_scalar_sum(action, arrays, backend, dtype), device=device)
            if prepared is not None:
                try:
                    value = timed('baryon_scalar_kernel', prepared.compute, device=device)
                finally:
                    prepared.close()
                arrays[action.tensor] = timed('result_snapshot', lambda: value,
                                              nbytes=0, device=device)
                return
        if not no_search and len(legs) == 2 and len(core.components) > 1:
            prepared = timed('baryon_prepare',
                lambda: prepare_component_sum(action, arrays, backend, dtype), device=device)
            if prepared is not None:
                try:
                    value = timed('baryon_join_kernel', prepared.compute, device=device)
                finally:
                    prepared.close()
                arrays[action.tensor] = copy_to_backend(value, backend,
                    phase='result_snapshot', timed=timed, device=device)
                return
        leg_values = tuple(arrays[key] for key in legs)
        total = None
        for component in core.components:
            equation = baryon_component_pattern(core_labels, leg_labels, output, arrays[component.inputs[1]].ndim)
            operands = tuple(arrays[key] for key in component.inputs) + leg_values
            part = _contract_baryon_part(equation, operands, action.kernel_path, component.sparse_spin,
                                        backend, timed=timed, device=device, no_search=no_search,
                                        kernel_description=getattr(action, "kernel_description", None))
            # _contract_baryon_part has already returned an independent owner.
            # Weighting it in place cannot mutate the inputs or another part.
            value = timed("core_copy", lambda: part if part.dtype == dtype else
                          backend.array(part, dtype=dtype, copy=True), device=device)
            del part
            if component.coefficient != 1:
                weight = component.coefficient if dtype.kind == "c" else component.coefficient.real
                timed("core_weight", lambda: value.__imul__(weight), device=device)
            if total is None:
                total = value
            else:
                timed("core_accumulate", lambda: total.__iadd__(value), device=device)
        arrays[action.tensor] = total
        return
    operands = tuple(arrays[key] for key in action.operands)
    arrays[action.tensor] = _contract_baryon_part(equation, operands, action.kernel_path,
        getattr(action, "sparse_spin", None), backend, timed=timed, device=device, no_search=no_search,
        kernel_description=getattr(action, "kernel_description", None))


def _contract_baryon_part(equation, operands, kernel_path, entries, backend, *, timed, device, no_search,
                         kernel_description=None):
    """Execute one internal component without manufacturing another Action.

    The returned array owns an independent result, including when the backend
    kernel itself returned a view. Core accumulation may therefore consume it
    directly when its dtype already matches the declared accumulator.
    """
    shapes = tuple(tuple(value.shape) for value in operands)
    dtypes = tuple(str(value.dtype) for value in operands)
    path = tuple(tuple(step) for step in kernel_path)
    foreign_inputs = backend.__name__ != "numpy" and all(isinstance(value, np.ndarray) for value in operands)
    binding = bind_baryon_kernel(equation, shapes, dtypes, path,
                                entries, foreign_inputs, no_search, kernel_description)
    prepared = timed("baryon_prepare", lambda: binding.prepare(*operands[:2], backend), device=device)
    try:
        value, independent = timed("baryon_join_kernel", lambda: prepared.compute(operands[2:], device=device),
                                   device=device)
    finally:
        prepared.close()
    return (
        timed("result_snapshot", lambda: value, nbytes=0, device=device)
        if independent else copy_to_backend(
            value, backend, phase="result_snapshot", timed=timed, device=device,
        )
    )


__all__ = ["contract_baryon_join"]
