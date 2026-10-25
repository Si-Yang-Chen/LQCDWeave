"""Whole-cfg output description, independent of numerical coordinate binding.

Packing and reservation formulas are adapted from correlator_cfg.complete_calculation
and correlator_dt_output.configure_packed_dt. Writers and scientific encoding stay
shared with the complete-archive workflow.
"""
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
import numpy as np

from lqcdweave.contracts.correlator.api import CfgOutputField, json_digest
from lqcdweave.contraction.results.api import describe_output_fields
from lqcdweave.storage.contraction_plans.api import encode_plan_value


@dataclass(frozen=True)
class CfgWindowOutput:
    """W-only packing description; contains no actions or execution instances."""
    coordinates: tuple
    coordinate_shape: tuple
    output_fields: tuple
    output_words: int
    dt_words: int
    physical_lt: int | None
    output_storage_shape: tuple
    dt_storage_shape: tuple
    time_roles: tuple
    output_definition: object
    output_layout: str
    dt_coordinates: tuple
    dt_packed_storage_shape: tuple
    one_point_storage_shape: tuple
    one_point_slots: int
    extra_host_bytes: int


def _shape(words):
    columns = min(words, 4096)
    return ((words + columns - 1) // columns, columns) if columns else (0, 0)


def describe_window_output(recipe, problem, preparations, specs, coordinates, *,
                           physical_lt=None, time_roles=('0', '1'),
                           dt_fields=('two_point', 'connected', 'disconnected'),
                           output_definition=None, output_layout='pairs_and_dt',
                           dt_coordinates=None):
    if physical_lt is not None and (type(physical_lt) is not int or physical_lt < 1):
        raise ValueError('physical Lt must be a positive integer')
    if output_layout not in {'pairs_and_dt', 'dt_packed'}:
        raise ValueError('unknown output layout')
    roles = tuple(map(str, time_roles))
    if len(roles) != 2 or len(set(roles)) != 2:
        raise ValueError('two distinct physical time roles are required')
    points = tuple(tuple((str(k), int(v)) for k, v in point.values) for point in coordinates.points)
    if not points:
        raise ValueError('cfg output requires at least one coordinate')
    fields = []
    words = dt_words = 0
    for path, shape, dtype in describe_output_fields(recipe, problem):
        count = (prod(shape) * np.dtype(dtype).itemsize + 7) // 8
        dt_offset = dt_words if physical_lt is not None and path and path[-1] in dt_fields else -1
        role = None
        if output_layout == 'dt_packed' and dt_offset < 0:
            role = {'vacuum_sink': roles[0], 'vacuum_source': roles[1]}.get(path[-1] if path else None)
            if role is None:
                raise ValueError(f'no time role for one-point field {path!r}')
        fields.append(CfgOutputField(path, shape, dtype, words, count, dt_offset, role))
        words += count
        if dt_offset >= 0:
            dt_words += count
    output_shape, dt_shape = _shape(words), _shape(dt_words)
    largest_elements = max((prod(f.shape) for f in fields), default=0)
    largest_bytes = max((prod(f.shape) * np.dtype(f.dtype).itemsize for f in fields), default=0)
    finite_bytes = max(largest_elements, prod(output_shape))
    packing = prod(output_shape) * 8 + finite_bytes + largest_bytes
    dt_buffer = sum(prod(f.shape) * np.dtype(f.dtype).itemsize
                    for f in fields if f.dt_offset >= 0) * (physical_lt or 0)
    extra = packing + dt_buffer + prod(dt_shape) * 8 + (physical_lt or 0) * 8
    dts = ()
    slots = 0
    dt_packed = one_point = (0, 0)
    if output_layout == 'dt_packed':
        if not physical_lt or not dt_words:
            raise ValueError('packed dt requires physical Lt and dt fields')
        dts = tuple(range(1, physical_lt)) if dt_coordinates is None else tuple(dt_coordinates)
        if not dts or len(set(dts)) != len(dts) or any(type(dt) is not int or not 1 <= dt < physical_lt for dt in dts):
            raise ValueError('invalid packed physical dt coordinates')
        auxiliary = [f for f in fields if f.dt_offset < 0]
        slots = 1 + max(v for point in points for role, v in point if role in roles) if auxiliary else 0
        dt_packed, one_point = _shape(dt_words * len(dts)), _shape(sum(f.words for f in auxiliary) * slots)
        largest = max((prod(f.shape) * np.dtype(f.dtype).itemsize *
                       (len(dts) if f.dt_offset >= 0 else slots) for f in fields), default=0)
        extra += (prod(dt_packed) + prod(one_point)) * 9 + largest
        extra += sum(prod(f.shape) * np.dtype(f.dtype).itemsize * slots + slots for f in auxiliary)
    elif dt_coordinates is not None:
        raise ValueError('dt coordinates require packed output')
    definition = dict(output_definition or {})
    definition['task_signature'] = json_digest(encode_plan_value(dict(
        output=recipe.root, preparations=preparations, specs=specs)))
    return CfgWindowOutput(points, tuple(coordinates.scan_shape), tuple(fields), words, dt_words,
        physical_lt, output_shape, dt_shape, roles, MappingProxyType(definition), output_layout,
        dts, dt_packed, one_point, slots, extra)
