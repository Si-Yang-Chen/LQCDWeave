"""Validate compact-time metadata once at an input binding boundary."""

from copy import copy
from .glueball import Glueball
from lqcdweave.contracts.time_selection import TimeSelectionRecord


def validate_slot_compatibility(readers):
    """Require a shared ordered selection for the common same-time slot path.

    Direct readers may expose asymmetric endpoint selections. Such readers
    cannot enter a common-coordinate Diagram: that needs separate endpoint
    coordinate spaces and must never be approximated by matching Nt alone.
    """
    readers = tuple({id(reader): reader for reader in readers if reader is not None}.values())
    common = None
    unmarked = False
    for reader in readers:
        selections = getattr(reader, 'time_selections', {})
        if not selections:
            unmarked = True
            continue
        for record in selections.values():
            if not isinstance(record, TimeSelectionRecord):
                raise TypeError('reader time selections must use TimeSelectionRecord')
            if common is None:
                common = record
            elif record != common:
                raise ValueError('common slot binding requires identical ordered time selections')
    if common is not None and unmarked:
        raise ValueError('cannot mix slot-selected and undeclared physical-time readers')
    return common


def _has_offsets(operator):
    return any(getattr(term, 'anchor_time_offset', 0)
               or any(getattr(term.component, 'time_offsets', ()))
               or any(getattr(term, 'endpoint_time_offsets', ()) or ())
               for term in operator.terms)


def validate_slot_operator(reader, operator, tables=None):
    """Require an explicit binding context before interpreting physical offsets."""
    if getattr(reader, 'time_selections', {}) and _has_offsets(operator) and tables is None:
        raise NotImplementedError('compact temporal offsets require a bound physical endpoint slot map')


def _offset_table(anchors, destination, offset, boundary):
    slots = {time: slot for slot, time in enumerate(destination.times)}
    result = []
    for time in anchors.times:
        physical = time + offset
        if boundary == 'periodic':
            physical %= anchors.Lt
        result.append(slots.get(physical))
    return tuple(result)


def _endpoint_union(propagators):
    records = []
    for reader in propagators:
        selections = getattr(reader, 'time_selections', {})
        if not selections:
            raise ValueError('temporal slot binding requires declared propagator endpoint selections')
        records.extend(selections.values())
    if not records or any(record.Lt != records[0].Lt for record in records):
        raise ValueError('temporal slot binding endpoint physical Lt differs')
    times = tuple(dict.fromkeys(time for record in records for time in record.times))
    return TimeSelectionRecord(records[0].Lt, times)


def bind_slot_vertices(vertices, propagators):
    """Bind non-affine physical offsets to the existing vertex metadata.

    A metadata-only union of actual endpoint records permits independent source
    and sink coverage. E maps each endpoint slot to its actual edge reader once;
    missing coverage is checked for that edge, never inferred from shape.
    No numerical arrays are read or copied.
    """
    vertices, propagators = tuple(vertices), tuple(propagators)
    if any(isinstance(vertex, Glueball) for vertex in vertices):
        validate_slot_compatibility([*(vertex.reader for vertex in vertices), *propagators])
        quarks = iter(bind_slot_vertices(
            (vertex for vertex in vertices if not isinstance(vertex, Glueball)), propagators)
            if any(not isinstance(vertex, Glueball) for vertex in vertices) else ())
        return tuple(vertex if isinstance(vertex, Glueball) else next(quarks) for vertex in vertices)
    if vertices and all(getattr(vertex, '_time_slot_tables', None) is not None for vertex in vertices):
        validate_vertex_time_bindings(vertices, propagators)
        return vertices
    marked = tuple(getattr(vertex.reader, 'time_selection', None) for vertex in vertices)
    if not any(record is not None for record in marked):
        validate_slot_compatibility([*(vertex.reader for vertex in vertices), *propagators])
        return vertices
    if any(record is None for record in marked):
        raise ValueError('cannot mix slot-selected and undeclared physical-time vertices')
    endpoint_union = _endpoint_union(propagators)
    if any(record.Lt != endpoint_union.Lt for record in marked):
        raise ValueError('temporal slot binding requires compatible physical endpoint selections')
    physical_offsets = any(_has_offsets(vertex.operator) for vertex in vertices)
    result = []
    for vertex in vertices:
        anchors = vertex.reader.time_selection
        endpoints = endpoint_union if physical_offsets else anchors
        if not callable(getattr(vertex, 'term_time_offsets', None)):
            if _has_offsets(vertex.operator):
                raise NotImplementedError('compact Baryon vertices require same-time legs and zero temporal anchor offsets')
            selected = copy(vertex)
            # BaryonFactorMetadata.time remains its own elemental anchor slot.
            # Only the connected propagator edge may need a different slot map.
            selected._time_slot_tables = ()
            selected.endpoint_time_selection = anchors
            selected._physical_offset_mapping = physical_offsets
            selected.cores = {}
            result.append(selected)
            continue
        selected = copy(vertex)
        tables, links, pools = [], [], []
        for index, term in enumerate(vertex.operator.terms):
            if term.profile is not None:
                raise NotImplementedError('temporal slot maps require an explicitly mapped spectral profile')
            anchor = _offset_table(anchors, anchors, term.anchor_time_offset, vertex.boundary)
            bar_offset, field_offset = vertex.term_time_offsets(index)
            bar = _offset_table(anchors, endpoints, bar_offset, vertex.boundary)
            field = _offset_table(anchors, endpoints, field_offset, vertex.boundary)
            tables.append((anchor, bar, field))
            sample = getattr(vertex.reader, 'sampling_time_selection', anchors)
            pools.append((_offset_table(anchors, sample, bar_offset, vertex.boundary),
                          _offset_table(anchors, sample, field_offset, vertex.boundary)))
            current = getattr(term, 'current_term', None)
            links.append(None if current is None else _offset_table(
                anchors, endpoints, current.link_origin_offset[3], vertex.boundary))
        selected._time_slot_tables = tuple(tables)
        selected._time_link_tables = tuple(links)
        selected._time_pool_tables = tuple(pools)
        selected.endpoint_time_selection = endpoints
        selected._physical_offset_mapping = physical_offsets
        result.append(selected)
    return tuple(result)


def validate_vertex_time_bindings(vertices, propagators):
    if any(isinstance(vertex, Glueball) for vertex in vertices):
        validate_slot_compatibility([*(vertex.reader for vertex in vertices), *propagators])
        vertices = tuple(vertex for vertex in vertices if not isinstance(vertex, Glueball))
    if not any(getattr(vertex, 'endpoint_time_selection', None) is not None for vertex in vertices):
        return validate_slot_compatibility([*(vertex.reader for vertex in vertices), *propagators])
    endpoints = _endpoint_union(propagators)
    for vertex in vertices:
        anchor = getattr(vertex.reader, 'time_selection', None)
        expected = (endpoints if getattr(vertex, '_physical_offset_mapping', False)
                    and callable(getattr(vertex, 'term_time_offsets', None)) else anchor)
        if (anchor is None or anchor.Lt != endpoints.Lt
                or getattr(vertex, 'endpoint_time_selection', None) != expected):
            raise ValueError('bound slot context differs from actual endpoint selections')
    # Each exact source/sink role is checked against its connected edge in E.
    return None


def map_edge_time_slot(time, source_selection, propagator_reader, axis, *, physical_offsets=False):
    """Resolve one bound vertex endpoint onto an actual propagator axis.

    This is a metadata boundary, called before the numerical block reader.
    Same-time inputs require exact ordered records. An explicitly bound
    temporal context may map an endpoint to a larger/differently ordered record.
    """
    target = getattr(propagator_reader, 'time_selections', {}).get(axis)
    if source_selection is None and target is None:
        return time
    if source_selection is None or target is None:
        raise ValueError('cannot mix slot-selected and undeclared physical-time endpoints')
    if not physical_offsets:
        if source_selection != target:
            raise ValueError('connected same-time edge requires identical ordered time selections for each endpoint role')
        return time
    if source_selection.Lt != target.Lt:
        raise ValueError('connected temporal edge has incompatible physical Lt')
    return target.slot(source_selection.physical_time(time))
