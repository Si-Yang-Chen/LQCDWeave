"""Retain the original whole meson topology across coordinate lifting."""
from dataclasses import replace

from lqcdweave.contracts.correlator.api import AxisPattern


def lift_meson_kernel(kernel, source_map, dynamic, lane_axis):
    inputs = tuple(source_map[key] for key in kernel.inputs)
    flags = tuple(key in dynamic for key in inputs)
    equation = kernel.private_equation
    varies = any(flags)
    labels = {a for axes in (*equation.inputs, equation.output) for a in axes}
    if varies and lane_axis in labels:
        # Private contribution axes do not reserve global labels. A global
        # lane may use the same integer; rename that private dummy locally.
        fresh = max((*labels, lane_axis)) + 1
        equation = AxisPattern(tuple(tuple(fresh if a == lane_axis else a for a in axes)
            for axes in equation.inputs), tuple(fresh if a == lane_axis else a for a in equation.output))
    private = AxisPattern(tuple((lane_axis, *axes) if flag else axes
        for axes, flag in zip(equation.inputs, flags, strict=True)),
        (lane_axis, *equation.output) if varies else equation.output)
    return replace(kernel, cores=tuple(source_map[key] for key in kernel.cores),
        inputs=inputs, private_equation=private, lane_axis=lane_axis if varies else None)


def describe_meson_core_batch(problem, source_map, dynamic, top_specs, *,
                              key_prefix, width, lane_name):
    """Add virtual whole specs, never fake numerical preparations or owners."""
    cores = []
    for core in problem.meson_cores:
        key = source_map.setdefault(core.key, key_prefix + core.key)
        spec = problem.specs[core.key]
        if any(source_map[member] in dynamic for member in core.inputs):
            dynamic.add(key)
            spec = replace(spec, shape=(width, *spec.shape),
                description=(None if spec.description is None else replace(spec.description,
                    axes=((lane_name, *spec.axes) if spec.axes else ()))))
        top_specs[key] = spec
        cores.append(replace(core, key=key, inputs=tuple(source_map[member] for member in core.inputs)))
    return tuple(cores)
