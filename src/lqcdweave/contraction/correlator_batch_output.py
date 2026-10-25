"""Assemble bounded coordinate batches using the existing fixed R program."""
from dataclasses import replace
import logging
from time import perf_counter

import numpy as np

from lqcdweave.contracts.correlator.api import CoordinateSpec
from lqcdweave.contraction.execution.api import execute_contraction_path
from lqcdweave.contraction.correlator_cfg import stack_point_results


def _owned(value):
    if isinstance(value, dict):
        return {key: _owned(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_owned(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    array = np.asarray(value)
    if array.dtype.kind == 'O':
        result = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            result[index] = _owned(array[index])
        return result
    return np.array(array, copy=True)


def run_coordinate_output(calculation, *, backend, debug, runtime_device_limit):
    """Keep final coordinate fields, release each batch's raw terms on return."""
    expanded = calculation.expanded
    recipe = expanded.output_recipe
    scalar = replace(recipe, coordinates=CoordinateSpec(
        'bound', (), (), (), (recipe.coordinates.points[0],), ()))
    assemble, = calculation._output_bindings.bind(recipe, (scalar,), debug=debug)
    results = [None] * len(recipe.coordinates.points)
    completed = 0
    started = perf_counter()

    def consume(index, raw_values, output_axes):
        nonlocal completed
        group = expanded.inputs.task_batches[index].coordinate_batch
        for value, axis in zip(raw_values, output_axes, strict=True):
            if axis is not None and (axis >= value.ndim or value.shape[axis] != group.actual_size):
                raise ValueError('coordinate output lane has the wrong extent')
        for lane, position in enumerate(group.point_indices):
            values = tuple(value if axis is None else
                           value[(slice(None),) * axis + (lane,)]
                           for value, axis in zip(raw_values, output_axes, strict=True))
            # Even an identity recipe can return a raw view: detach the final
            # fields so a tiny selected leaf cannot pin a whole raw batch.
            results[position] = _owned(assemble(values))
            completed += 1
        logging.getLogger('lqcdweave').info(
            'coordinate replay: %d/%d points assembled', completed, len(results))

    executed = execute_contraction_path(expanded, calculation.path, backend=backend,
        measure=False, debug=debug, execution_cache=calculation._execution_bindings,
        runtime_device_limit=runtime_device_limit, batch_consumer=consume)
    if completed != len(results) or any(value is None for value in results):
        raise ValueError('coordinate assembly did not cover the full requested scan')
    value = stack_point_results(results, recipe.coordinates.scan_shape)
    return value, dict(execution=executed.report, output_retention='assembled-coordinates',
        coordinate_points=completed, replay_and_assembly_seconds=perf_counter()-started)


def measured_coordinate_path(calculation):
    """Admit optional retained raw reports without changing the chosen DAG."""
    reserve = calculation.prediction['coordinate_batching']['raw_measurement_host_reserve_bytes']
    children = []
    for child in calculation.path.batch_paths:
        plan = child.plan.with_reserves(reserved_host_bytes=max(reserve, child.plan.reserved_host_bytes))
        if not plan.feasible:
            raise MemoryError('retained raw measurement reports exceed the coordinate host budget')
        children.append(replace(child, plan=plan, plan_identity=id(plan),
                                reserved_host_bytes=plan.reserved_host_bytes))
    first = children[0]
    return replace(calculation.path, plan=first.plan, plan_identity=id(first.plan),
                   reserved_host_bytes=first.plan.reserved_host_bytes, batch_paths=tuple(children))
