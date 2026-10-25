"""Save correlator results as native HDF5 fields through the dataset lifecycle.

Complete-cfg and structured results retain each field's real shape and dtype in
HDF5. Literal container descriptions are persistence metadata, not an execution IR.
"""
from collections.abc import Mapping
import hashlib
import json
from math import prod

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.correlator.api import CorrelatorResult
from lqcdweave.contracts.time_selection import normalize_time_selection, TimeSelectionRecord
from .dataset import DatasetReader, DatasetWriter
from lqcdweave.sampling.temporal.api import (
    pair_time_differences, count_time_pairs, mean_time_pairs,
)


_ENCODING = 'native-hdf5-fields-v1'
_FORMAT = 'structured-correlator-result-v2'


def _document(value, label):
    """Require JSON-native metadata rather than silently changing Python keys."""
    def normalize(item):
        if item is None or type(item) in (str, bool, int, float):
            return item
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise TypeError(f'{label} requires string mapping keys')
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        raise TypeError(f'{label} must contain JSON-native data')
    return json.loads(canonical_json(normalize(value)))


def _key(value):
    if type(value) is tuple:
        return {'tuple': [_key(item) for item in value]}
    if value is None or type(value) in (str, bool, int, float):
        return {'scalar': _document(value, 'container key')}
    raise TypeError('container keys must be JSON scalars or tuples of those scalars')


def _decode_key(value):
    if not isinstance(value, dict):
        raise ValueError('invalid saved container key')
    if set(value) == {'tuple'} and isinstance(value['tuple'], list):
        return tuple(_decode_key(item) for item in value['tuple'])
    if set(value) == {'scalar'}:
        scalar = value['scalar']
        if scalar is None or type(scalar) in (str, bool, int, float):
            _document(scalar, 'container key')
            return scalar
    raise ValueError('invalid saved container key')


def _path(path):
    if not isinstance(path, tuple):
        raise TypeError('field paths must be tuples')
    return [_key(item) for item in path]


def _names(axes):
    if isinstance(axes, str):
        raise TypeError('axes must be a sequence of names')
    names = tuple(axes)
    if any(not (type(name) is int or (type(name) is str and name)) for name in names) or len(set(names)) != len(names):
        raise ValueError('axis keys must be unique integers or nonempty strings')
    return names


def _coordinates(axes, coordinates, shape=None):
    if not isinstance(coordinates, Mapping) or set(coordinates) != set(axes):
        raise ValueError('coordinates must describe exactly the declared axes')
    result = {}
    for index, name in enumerate(axes):
        values = coordinates[name]
        if isinstance(values, (str, bytes, Mapping)):
            raise TypeError('coordinates must be ordered sequences')
        values = _document(list(values), 'coordinates')
        if any(value is None for value in values):
            raise ValueError('axis coordinates must be known')
        if shape is not None and len(values) != shape[index]:
            raise ValueError(f'axis {name!r} coordinate count differs from field shape')
        result[name] = values
    return result


def _coords_document(coordinates):
    return [dict(axis=name, values=values) for name, values in coordinates.items()]


def _coords_from(document):
    if not isinstance(document, list):
        raise ValueError('invalid saved coordinate entries')
    result = {}
    for entry in document:
        if not isinstance(entry, dict) or set(entry) != {'axis', 'values'}:
            raise ValueError('invalid saved coordinate entry')
        name = _names((entry['axis'],))[0]
        if name in result:
            raise ValueError('duplicate saved coordinate axis')
        result[name] = entry['values']
    return result


def _describe(value):
    fields = []
    active = set()
    def visit(item, path):
        container = isinstance(item, (dict, list, tuple)) or (
            isinstance(item, np.ndarray) and item.dtype == object)
        if container:
            if id(item) in active:
                raise ValueError('cyclic numerical result container')
            active.add(id(item))
            try:
                if isinstance(item, dict):
                    return dict(kind='dict', keys=[_key(key) for key in item],
                        children=[visit(child, path + (key,)) for key, child in item.items()])
                if isinstance(item, np.ndarray):
                    return dict(kind='object_array', shape=list(item.shape),
                        children=[visit(item[index], path + (index,)) for index in np.ndindex(item.shape)])
                return dict(kind='tuple' if isinstance(item, tuple) else 'list',
                    children=[visit(child, path + (index,)) for index, child in enumerate(item)])
            finally:
                active.remove(id(item))
        if not isinstance(item, (np.ndarray, np.generic)) and type(item) not in (int, float, complex, bool):
            raise TypeError('result leaves must be numerical arrays or numerical scalars')
        array = np.asarray(item)
        if array.dtype.hasobject or array.dtype.kind not in 'biufc':
            raise TypeError('result leaves require a plain numerical dtype')
        if isinstance(item, np.ndarray):
            kind = 'ndarray'
        elif isinstance(item, np.generic):
            kind = 'numpy_scalar'
        else:
            kind = type(item).__name__
        index = len(fields)
        fields.append(dict(path=path, array=array, value_kind=kind))
        return dict(kind='field', index=index)
    return visit(value, ()), fields


def _field_metadata(fields, axes, coordinates, field_axes, field_coordinates):
    field_axes = {} if field_axes is None else dict(field_axes)
    field_coordinates = {} if field_coordinates is None else dict(field_coordinates)
    paths = {field['path'] for field in fields}
    if set(field_axes) - paths or set(field_coordinates) - paths:
        raise ValueError('field metadata refers to an unknown numerical leaf')
    for path in set(field_axes) | set(field_coordinates):
        _path(path)
    for field in fields:
        path, array = field['path'], field['array']
        leading = array.ndim - len(axes)
        if leading < 0:
            raise ValueError('result field has fewer dimensions than shared scan axes')
        names = field_axes.get(path)
        if names is None:
            names = tuple(f'operator_axis_{index}' for index in range(leading)) + axes
        names = _names(names)
        if len(names) != array.ndim or (axes and names[-len(axes):] != axes):
            raise ValueError('full field axes must match ndim and end with the shared scan axes')
        supplied = field_coordinates.get(path, {})
        if not isinstance(supplied, Mapping) or set(supplied) - set(names):
            raise ValueError('field coordinates name an unknown field axis')
        coords = {name: list(range(array.shape[index])) for index, name in enumerate(names)}
        coords.update(coordinates)
        for name, values in supplied.items():
            values = _coordinates((name,), {name: values})[name]
            if name in coordinates and canonical_json(values) != canonical_json(coordinates[name]):
                raise ValueError('field scan coordinates differ from shared scan coordinates')
            coords[name] = values
        field['axes'] = names
        field['coordinates'] = _coordinates(names, coords, array.shape)


def _reduction(axes, coordinates, *, source_axis, sink_axis, relative_axis, physical_lt,
               boundary, missing, coordinate_domain, selection=None, source_selection=None,
               sink_selection=None, time_layout='absolute', dt_coordinates=None):
    if source_axis is None or source_axis not in axes:
        raise ValueError('averaging requires an explicit source_axis among the shared scan axes')
    if coordinate_domain not in ('physical', 'slot'):
        raise ValueError("averaging requires coordinate_domain='physical' or 'slot'")
    if boundary != 'periodic':
        raise ValueError('source averaging supports only explicitly periodic boundary conditions')
    absolute = sink_axis is not None
    other = sink_axis if absolute else relative_axis
    if other is None or other == source_axis or set(axes) != {source_axis, other}:
        raise ValueError('averaging requires exactly source/sink or source/relative shared time axes')
    output_axis = ('dt' if relative_axis is None else relative_axis) if absolute else other
    if output_axis == source_axis:
        raise ValueError('invalid relative-time output axis')
    _names((output_axis,))
    if missing not in ('error', 'nan'):
        raise ValueError("missing must be 'error' or 'nan'")
    selections = None
    if coordinate_domain == 'slot':
        if selection is not None:
            if source_selection is not None or sink_selection is not None:
                raise ValueError('selection cannot be combined with endpoint selections')
            source_selection = sink_selection = selection
        if source_selection is None:
            raise ValueError('slot coordinates require an explicit ordered selection')
        source_selection = normalize_time_selection(source_selection, Lt=physical_lt)
        sink_selection = (source_selection if sink_selection is None else
                          normalize_time_selection(sink_selection, Lt=source_selection.Lt))
        physical_lt = source_selection.Lt
        if time_layout not in ('absolute', 'cyclic_difference'):
            raise ValueError("time_layout must be 'absolute' or 'cyclic_difference'")
        if absolute != (time_layout == 'absolute'):
            raise ValueError('absolute slot layout requires sink_axis; cyclic_difference requires relative_axis')
        if not absolute and source_selection != sink_selection:
            raise ValueError('cyclic_difference requires identical ordered endpoint selections')
        selections = dict(source=source_selection.document, sink=sink_selection.document)
        extents = {source_axis: source_selection.Nt, other: sink_selection.Nt}
    else:
        if any(value is not None for value in (selection, source_selection, sink_selection)):
            raise ValueError('physical coordinates cannot be combined with slot selections')
        if time_layout != 'absolute':
            raise ValueError('cyclic_difference is a slot layout, never a physical dt axis')
        if type(physical_lt) is not int or physical_lt < 1:
            raise ValueError('averaging requires an explicit positive physical_lt')
        extents = {name: physical_lt for name in axes}
    for name in axes:
        if (not coordinates[name] or any(type(t) is not int or not 0 <= t < extents[name]
                                        for t in coordinates[name])):
            raise ValueError('time coordinates must be nonempty actual integers within the declared domain')
        if len(set(coordinates[name])) != len(coordinates[name]):
            raise ValueError('source averaging requires unique time coordinates')
    output_times = list(range(physical_lt)) if dt_coordinates is None else list(dt_coordinates)
    if (not output_times or len(set(output_times)) != len(output_times)
            or any(type(dt) is not int or not 0 <= dt < physical_lt for dt in output_times)):
        raise ValueError('dt_coordinates must be unique physical time differences within Lt')
    result = dict(kind='source_time_mean', mode='absolute' if absolute else 'relative',
        source_axis=source_axis, sink_axis=sink_axis, relative_axis=output_axis,
        physical_lt=physical_lt, boundary=boundary, coordinate_domain=coordinate_domain, input_axes=list(axes),
        input_coordinates=_coords_document(coordinates), counts=[], missing_dt=missing,
        aggregation='mean over covered source/dt contributions', output_coordinates=output_times)
    if selections is not None:
        result.update(selections=selections, time_layout=time_layout)
    differences = _pair_differences(result)
    counts = count_time_pairs(differences, Lt=physical_lt)
    if missing == 'error' and not np.all(counts[output_times]):
        raise ValueError("missing physical dt coverage; pass missing='nan' explicitly to retain NaN gaps")
    result['counts'] = counts.tolist()
    return result


def _pair_differences(reduction):
    """Resolve coordinates once at the output boundary, in source × other order."""
    coordinates = _coords_from(reduction['input_coordinates'])
    source = reduction['source_axis']
    other = reduction['sink_axis'] if reduction['sink_axis'] is not None else reduction['relative_axis']
    options = {}
    layout = 'relative' if reduction['sink_axis'] is None else 'absolute'
    if reduction['coordinate_domain'] == 'slot':
        selections = reduction['selections']
        options = dict(source_times=selections['source']['times'],
                       sink_times=selections['sink']['times'])
        layout = reduction['time_layout']
    return pair_time_differences(coordinates[source], coordinates[other],
        Lt=reduction['physical_lt'], time_layout=layout, **options)


def _average_fields(fields, reduction):
    source, sink = reduction['source_axis'], reduction['sink_axis']
    relative = reduction['relative_axis']
    other = relative if sink is None else sink
    lt = reduction['physical_lt']
    differences = _pair_differences(reduction)
    for field in fields:
        array, names = field['array'], field['axes']
        if not np.isfinite(array).all():
            raise ValueError('source averaging requires finite measured contributions')
        spectator_names = tuple(name for name in names if name not in (source, other))
        if relative in spectator_names:
            raise ValueError('relative-time axis collides with an operator axis')
        dtype = array.dtype if array.dtype.kind in 'fc' else np.dtype('float64')
        result = mean_time_pairs(array, differences, Lt=lt,
            source_axis=names.index(source), other_axis=names.index(other),
            missing='nan', dtype=dtype, sum_dtype=np.result_type(dtype, np.float64))
        out_names = tuple(relative if name == other else name for name in names if name != source)
        field['array'] = (result.values if reduction['output_coordinates'] == list(range(lt)) else
            np.take(result.values, reduction['output_coordinates'], axis=out_names.index(relative)))
        field['value_kind'] = 'ndarray'
        field['axes'] = out_names
        field['coordinates'] = {name: (list(reduction['output_coordinates']) if name == relative
                                      else field['coordinates'][name]) for name in out_names}


def save_correlator(path, value, *, axes=None, coordinates=None, field_axes=None,
                    field_coordinates=None, average_tsrc=False, source_axis=None,
                    sink_axis=None, relative_axis=None, physical_lt=None,
                    boundary=None, missing='error', coordinate_domain=None, selection=None,
                    source_selection=None, sink_selection=None, time_layout='absolute',
                    metadata=None, provenance=None, configuration=None, dt_coordinates=None):
    """Publish a numerical tree, unchanged unless ``average_tsrc=True``.

    ``axes`` names common trailing scan dimensions; coordinates are actual
    ordered labels. Leading operator dimensions are preserved and may be named
    per numerical leaf using tuple container paths in ``field_axes`` and
    ``field_coordinates``. No role is inferred from any axis name.
    """
    if type(average_tsrc) is not bool:
        raise TypeError('average_tsrc must be bool')
    if dt_coordinates is not None and not average_tsrc:
        raise ValueError('dt_coordinates requires source averaging')
    previous_reduction = None
    if isinstance(value, CorrelatorResult):
        result = value
        value = result.value
        axes = result.axes if axes is None else axes
        coordinates = result.coordinates if coordinates is None else coordinates
        field_axes = result.field_axes if field_axes is None else field_axes
        field_coordinates = result.field_coordinates if field_coordinates is None else field_coordinates
        metadata = result.metadata if metadata is None else metadata
        provenance = result.provenance if provenance is None else provenance
        previous_reduction = result.reduction
        if average_tsrc and previous_reduction is not None:
            raise ValueError('cannot source-average an already reduced result')
    axes = _names(() if axes is None else axes)
    coordinates = _coordinates(axes, {} if coordinates is None else coordinates)
    metadata = _document({} if metadata is None else metadata, 'metadata')
    provenance = _document({} if provenance is None else provenance, 'provenance')
    if not isinstance(metadata, dict) or not isinstance(provenance, dict):
        raise TypeError('metadata and provenance must be mappings')
    tree, fields = _describe(value)
    _field_metadata(fields, axes, coordinates, field_axes, field_coordinates)
    reduction = previous_reduction
    if average_tsrc:
        reduction = _reduction(axes, coordinates, source_axis=source_axis, sink_axis=sink_axis,
            relative_axis=relative_axis, physical_lt=physical_lt, boundary=boundary, missing=missing,
            coordinate_domain=coordinate_domain, selection=selection, source_selection=source_selection,
            sink_selection=sink_selection, time_layout=time_layout, dt_coordinates=dt_coordinates)
        _average_fields(fields, reduction)
        axes = (reduction['relative_axis'],)
        coordinates = {axes[0]: list(reduction['output_coordinates'])}
    elif any(item is not None for item in (source_axis, sink_axis, relative_axis, physical_lt, boundary, coordinate_domain, selection, source_selection, sink_selection)) or time_layout != 'absolute':
        # These explicit declarations are still valuable for an unreduced file.
        declaration = dict(source_axis=source_axis, sink_axis=sink_axis, relative_axis=relative_axis,
                           physical_lt=physical_lt, boundary=boundary, coordinate_domain=coordinate_domain)
        if coordinate_domain == 'slot':
            checked = _reduction(axes, coordinates, **declaration, missing='nan', selection=selection,
                                 source_selection=source_selection, sink_selection=sink_selection, time_layout=time_layout)
            declaration.update(physical_lt=checked['physical_lt'], selections=checked['selections'], time_layout=time_layout)
        elif any(item is not None for item in (selection, source_selection, sink_selection)) or time_layout != 'absolute':
            raise ValueError('slot selections/layout require coordinate_domain=slot')
        metadata = dict(metadata, time_declaration=declaration)
    _validate_reduction(reduction, axes, coordinates,
        {field['path']: field['axes'] for field in fields}, [field['array'] for field in fields])
    descriptors = []
    for field in fields:
        array = np.ascontiguousarray(field['array']).reshape(field['array'].shape)
        descriptors.append(dict(path=_path(field['path']), shape=list(array.shape), dtype=array.dtype.str,
            value_kind=field['value_kind'], axes=list(field['axes']), coordinates=_coords_document(field['coordinates']),
            sha256=hashlib.sha256(array.reshape(-1).view(np.uint8)).hexdigest()))
    definition = dict(format=_FORMAT, encoding=_ENCODING, tree=tree, fields=descriptors,
        axes=list(axes), coordinates=_coords_document(coordinates), metadata=metadata,
        reduction=_document(reduction, 'reduction'))
    spec = DatasetSpec(dict(schema=1, kind='correlator',
        configuration=dict(ensemble=None, configuration=None if configuration is None else str(configuration), lattice=[1, 1, 1, 1]),
        axes=[], dtype=None, bases={}, definition=definition,
        conventions=dict(numeric_dtypes='stored per field; no numerical cast', lattice_declared=False), parents=[]))
    with DatasetWriter(path, spec, provenance=provenance, format='hdf5', structured=True) as writer:
        for descriptor, field in zip(descriptors, fields, strict=True):
            writer.write_field(descriptor['path'], np.asarray(field['array']))
        return writer.publish()


def _restore(tree, values, paths, current=(), used=None):
    used = set() if used is None else used
    if not isinstance(tree, dict) or type(tree.get('kind')) is not str:
        raise ValueError('invalid numerical container description')
    kind = tree['kind']
    if kind == 'field':
        index = tree.get('index')
        if (set(tree) != {'kind', 'index'} or type(index) is not int or not 0 <= index < len(values)
                or index in used or paths[index] != current):
            raise ValueError('invalid or duplicate saved field reference')
        used.add(index)
        return values[index]
    expected = {'kind', 'children'} | ({'keys'} if kind == 'dict' else {'shape'} if kind == 'object_array' else set())
    if kind not in ('dict', 'list', 'tuple', 'object_array') or set(tree) != expected or not isinstance(tree['children'], list):
        raise ValueError('invalid numerical container description')
    children = tree['children']
    if kind == 'dict':
        if not isinstance(tree['keys'], list) or len(tree['keys']) != len(children):
            raise ValueError('saved dictionary keys differ from child count')
        keys = [_decode_key(key) for key in tree['keys']]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate saved dictionary key')
        return {key: _restore(child, values, paths, current+(key,), used) for key, child in zip(keys, children)}
    if kind == 'object_array':
        shape = _shape(tree['shape'])
        if prod(shape) != len(children):
            raise ValueError('saved object-array shape differs from child count')
        value = np.empty(shape, dtype=object)
        for index, child in zip(np.ndindex(shape), children):
            value[index] = _restore(child, values, paths, current+(index,), used)
        return value
    items = [_restore(child, values, paths, current+(index,), used) for index, child in enumerate(children)]
    return tuple(items) if kind == 'tuple' else items


def _shape(shape):
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError('invalid saved numerical shape')
    return tuple(shape)


def load_correlator(path, *, configuration=None):
    """Validate a complete dataset and restore an owned numerical result tree."""
    with DatasetReader(path, cfg=configuration) as reader:
        spec = reader.spec.document
        definition = spec['definition']
        if definition.get('format') != _FORMAT or definition.get('encoding') != _ENCODING:
            raise ValueError('not a supported standalone correlator result')
        expected = {'format', 'encoding', 'tree', 'fields', 'axes', 'coordinates', 'metadata', 'reduction'}
        if set(definition) != expected:
            raise ValueError('invalid standalone correlator result definition')
        axes = _names(definition['axes'])
        coordinates = _coordinates(axes, _coords_from(definition['coordinates']))
        fields = definition['fields']
        if not isinstance(fields, list):
            raise ValueError('invalid saved field table')
        values, paths, field_axes, field_coordinates = [], [], {}, {}
        for field in fields:
            if not isinstance(field, dict) or set(field) != {'path', 'shape', 'dtype', 'value_kind', 'axes', 'coordinates', 'sha256'}:
                raise ValueError('invalid saved numerical field')
            if not isinstance(field['path'], list):
                raise ValueError('invalid saved field path')
            field_path = tuple(_decode_key(key) for key in field['path'])
            if field_path in field_axes:
                raise ValueError('duplicate saved field path')
            shape = _shape(field['shape'])
            try:
                dtype = np.dtype(field['dtype'])
            except (TypeError, ValueError) as error:
                raise ValueError('invalid saved numerical dtype') from error
            if dtype.hasobject or dtype.kind not in 'biufc' or dtype.str != field['dtype']:
                raise ValueError('invalid saved numerical dtype')
            try:
                value = reader.read_field(field['path'])
            except KeyError as error:
                raise ValueError('saved numerical field path is missing') from error
            if tuple(value.shape) != shape or value.dtype.str != field['dtype']:
                raise ValueError('saved numerical field shape/dtype differs')
            if hashlib.sha256(value.reshape(-1).view(np.uint8)).hexdigest() != field['sha256']:
                raise ValueError('saved numerical payload checksum differs')
            names = _names(field['axes'])
            if len(names) != len(shape) or (axes and names[-len(axes):] != axes):
                raise ValueError('saved field axes differ from numerical shape or shared axes')
            coords = _coordinates(names, _coords_from(field['coordinates']), shape)
            if any(canonical_json(coords[name]) != canonical_json(coordinates[name]) for name in axes):
                raise ValueError('saved field coordinates differ from shared coordinates')
            kind = field['value_kind']
            if kind != 'ndarray':
                if shape != () or kind not in ('numpy_scalar', 'int', 'float', 'complex', 'bool'):
                    raise ValueError('invalid saved scalar type or shape')
                value = value[()] if kind == 'numpy_scalar' else value.item()
                if kind != 'numpy_scalar' and type(value).__name__ != kind:
                    raise ValueError('saved scalar dtype differs from its type')
            values.append(value)
            paths.append(field_path)
            field_axes[field_path] = names
            field_coordinates[field_path] = coords
        used = set()
        value = _restore(definition['tree'], values, paths, used=used)
        if len(used) != len(values):
            raise ValueError('saved field table contains unreferenced values')
        metadata = _document(definition['metadata'], 'metadata')
        reduction = _document(definition['reduction'], 'reduction')
        if not isinstance(metadata, dict):
            raise ValueError('saved metadata must be a mapping')
        declaration = metadata.get('time_declaration')
        if reduction is None and isinstance(declaration, dict) and declaration.get('coordinate_domain') == 'slot':
            arguments = {key: value for key, value in declaration.items() if key != 'selections'}
            _reduction(axes, coordinates, **arguments, missing='nan', **_selection_options(declaration))
        _validate_reduction(reduction, axes, coordinates, field_axes, values)
        return CorrelatorResult(value=value, axes=axes, coordinates=coordinates,
            field_axes=field_axes, field_coordinates=field_coordinates, metadata=metadata,
            reduction=reduction, provenance=dict(reader.provenance), data_id=reader.data_id)


def _selection_options(record):
    if record.get('coordinate_domain') != 'slot':
        return {}
    selections = record.get('selections')
    if not isinstance(selections, dict) or set(selections) != {'source', 'sink'}:
        raise ValueError('invalid saved slot selections')
    return dict(source_selection=TimeSelectionRecord.from_document(selections['source']),
                sink_selection=TimeSelectionRecord.from_document(selections['sink']))


def _validate_reduction(reduction, axes, coordinates, field_axes, values):
    if reduction is None:
        return
    expected = {'kind', 'mode', 'source_axis', 'sink_axis', 'relative_axis', 'physical_lt', 'boundary',
                'input_axes', 'input_coordinates', 'counts', 'missing_dt', 'aggregation', 'output_coordinates', 'coordinate_domain'}
    if isinstance(reduction, dict) and reduction.get('coordinate_domain') == 'slot':
        expected |= {'time_layout', 'selections'}
    if not isinstance(reduction, dict) or set(reduction) != expected:
        raise ValueError('invalid saved reduction declaration')
    original_axes = _names(reduction['input_axes'])
    original_coords = _coordinates(original_axes, _coords_from(reduction['input_coordinates']))
    checked = _reduction(original_axes, original_coords, source_axis=reduction['source_axis'],
        sink_axis=reduction['sink_axis'], relative_axis=reduction['relative_axis'],
        physical_lt=reduction['physical_lt'], boundary=reduction['boundary'], missing=reduction['missing_dt'],
        coordinate_domain=reduction['coordinate_domain'],
        time_layout=reduction.get('time_layout', 'absolute'),
        dt_coordinates=reduction['output_coordinates'], **_selection_options(reduction))
    if canonical_json(checked) != canonical_json(reduction):
        raise ValueError('saved reduction counts or semantics differ from declared coverage')
    relative = reduction['relative_axis']
    if axes != (relative,) or coordinates != {relative: reduction['output_coordinates']}:
        raise ValueError('saved reduction output coordinates differ')
    counts = np.asarray(reduction['counts'])[coordinates[relative]]
    for names, value in zip(field_axes.values(), values):
        array = np.asarray(value)
        moved = np.moveaxis(array, names.index(relative), -1)
        if not np.isfinite(moved[..., counts > 0]).all() or not np.isnan(moved[..., counts == 0]).all():
            raise ValueError('saved reduction must contain finite coverage and NaN gaps')


# Complete-cfg and standalone-tree layouts share this storage entry point while
# retaining their separate schema and decoder contracts.
from ._cfg_correlator_output import (
    CfgOutputWriter, PackedDtOutputWriter, load_correlator_output,
    validate_output_binding as validate_cfg_output_binding,
)


__all__ = ['CorrelatorResult', 'save_correlator', 'load_correlator',
           'CfgOutputWriter', 'PackedDtOutputWriter', 'load_correlator_output',
           'validate_cfg_output_binding']
