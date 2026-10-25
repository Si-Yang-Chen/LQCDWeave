"""Pure registration and output contracts for fixed full-time or explicitly slot-selected Meson Diagrams.

Checks adapt the existing planning.meson input contract. Publication documents
adapt meson_measurement_document and directed_current_pair_document; numerical
expansion, operators, readers and execution stay in their existing modules.
"""

from collections.abc import Mapping
import json
import math

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.time_selection import dataset_time_selection, bind_time_selections
from lqcdweave.contracts.correlator.api import (
    TensorDescription, canonical_coordinate_identity,
)
from .production import validate_elemental_fields


_PROPAGATOR_BASES = {
    'vsv': ('sink_mode', 'source_mode'),
    'psv': ('sink_point', 'source_mode'),
    'vsp': ('sink_mode', 'source_point'),
    'psp': ('sink_point', 'source_point'),
}
_ELEMENTAL_BASES = {
    'elemental_vv': ('sink_mode', 'source_mode'),
    'elemental_v2p': ('mode_left', 'point_right'),
    'elemental_p2v': ('point_left', 'mode_right'),
    'elemental_p2p_row_sparse': ('point_left',),
}
_BARYON_PRODUCT_ENCODINGS = {
    'raw_paths': 'ordered_path_triples_v1',
    'projected': 'projected_row_ids_v1',
    'distance': 'distance_triples_v1',
}


def _known(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f'unknown {label}; explicit registration is required')


def _full_time_axis(spec, axis, extent):
    if axis not in spec.axis_names:
        raise ValueError(f'missing {axis} axis')
    selection = dataset_time_selection(spec, axis)
    values = spec.coordinates(axis)
    expected = range(extent if selection is None else selection.Nt)
    if any(type(value) is not int for value in values) or sorted(values) != list(expected):
        raise ValueError(f'{axis} must cover all physical times or all explicitly declared slots')


def _shared_time_selection(input_specs):
    """One coordinate space for this fixed two-axis helper; never infer by Nt."""
    selections = []
    for spec in input_specs.values():
        spec = spec if isinstance(spec, DatasetSpec) else DatasetSpec(spec)
        axes = ('time',) if 'time' in spec.axis_names else (
            'source_time', 'relative_sink_time' if 'relative_sink_time' in spec.axis_names else 'sink_time')
        selections.extend(dataset_time_selection(spec, axis) for axis in axes)
    selected = [record for record in selections if record is not None]
    if not selected:
        return None
    if len(selected) != len(selections) or any(record != selected[0] for record in selected):
        raise ValueError('fixed two-axis Diagram requires identical ordered slot selections; '
                         'use native calculate with explicit endpoint bindings for other domains')
    return selected[0]


def validate_diagram_input_specs(input_specs):
    """Validate explicit scientific registrations without resolving or reading IDs.

    Inputs are only the actual required propagator, elemental and point views.
    Different quark lines may declare different Dirac masses or gauge variants.
    Their compatibility at each mode/point endpoint belongs to E, and complete
    bytes/coverage and actual ID resolution belong to the public storage index.
    """
    if not isinstance(input_specs, Mapping) or not input_specs:
        raise TypeError('input_specs must be a nonempty role-to-DatasetSpec mapping')
    configuration = None
    for role, spec in input_specs.items():
        _known(role, 'input role')
        if not isinstance(spec, DatasetSpec):
            raise TypeError(f'{role} must supply a DatasetSpec')
        document = spec.document
        cfg = document['configuration']
        _known(cfg['ensemble'], 'ensemble identity')
        _known(cfg['configuration'], 'configuration identity')
        if configuration is None:
            configuration = cfg
        elif cfg != configuration:
            raise ValueError('configuration or lattice mismatch')
        extent = cfg['lattice'][3]
        kind = document['kind']
        definition, conventions = document['definition'], document['conventions']
        if kind == 'spatial_points':
            _full_time_axis(spec, 'time', extent)
            if (spec.axis_names != ('time', 'point', 'xyz') or
                    spec.coordinates('xyz') != ['x', 'y', 'z'] or
                    document['dtype'] not in ('int32', 'int64')):
                raise ValueError('explicit signed XYZ spatial point coordinates are required')
            if (conventions.get('representation') != 'coordinate_delta' or
                    conventions.get('normalization') != 'unit_delta'):
                raise ValueError('explicit coordinate-delta point normalization is required')
            continue
        if (kind not in _PROPAGATOR_BASES and kind not in _ELEMENTAL_BASES
                and kind != 'elemental_baryon_vvv'):
            raise ValueError(f'unsupported fixed Diagram input kind {kind!r}')
        if document['dtype'] not in ('complex64', 'complex128'):
            raise ValueError('Diagram numerical inputs require complex storage')
        _known(definition.get('gauge_id'), 'gauge variant identity')
        if conventions.get('normalization') != 'unscaled':
            raise ValueError('declare unscaled inputs; no implicit normalization conversion')
        if kind == 'elemental_baryon_vvv':
            _validate_baryon_spec(spec)
            _full_time_axis(spec, 'time', extent)
            continue
        propagator = kind in _PROPAGATOR_BASES
        bases = _PROPAGATOR_BASES[kind] if propagator else _ELEMENTAL_BASES[kind]
        for axis in bases:
            binding = document['bases'].get(axis)
            if binding is None:
                raise ValueError(f'missing actual {axis} basis binding')
            _known(binding['data_id'], f'actual {axis} basis identity')
        for axis in spec.axis_names:
            labels = spec.coordinates(axis)
            if axis in ('sink_spin', 'source_spin') and (
                    any(type(value) is not int for value in labels) or labels != [0, 1, 2, 3]):
                raise ValueError('spin axes must be ordered 0,1,2,3')
            if 'color' in axis and (
                    any(type(value) is not int for value in labels) or labels != [0, 1, 2]):
                raise ValueError('color axes must be ordered 0,1,2')
        if propagator:
            _full_time_axis(spec, 'source_time', extent)
            sink_axis = 'relative_sink_time' if 'relative_sink_time' in spec.axis_names else 'sink_time'
            _full_time_axis(spec, sink_axis, extent)
            if (not {'sink_spin', 'source_spin'} <= set(spec.axis_names) or
                    conventions.get('spin_basis') != 'degrand_rossi'):
                raise ValueError('explicit DeGrand-Rossi spin axes and basis are required')
            if conventions.get('temporal_boundary') not in ('periodic', 'antiperiodic'):
                raise ValueError('explicit fermion temporal boundary is required')
            dirac = definition.get('dirac')
            if not isinstance(dirac, dict):
                raise ValueError('explicit Dirac definition is required')
            _known(dirac.get('action'), 'Dirac action')
            mass = dirac.get('mass')
            if type(mass) not in (int, float) or not math.isfinite(mass):
                raise ValueError('explicit finite Dirac mass is required')
            tolerance = definition.get('solver_tolerance')
            if type(tolerance) not in (int, float) or not 0 < tolerance < 1:
                raise ValueError('explicit solver tolerance is required separately from dtype')
        else:
            _full_time_axis(spec, 'time', extent)
            if conventions.get('phase') not in ('anchor', 'midpoint', 'none'):
                raise ValueError('explicit elemental phase convention is required')
            if definition.get('path_boundary') not in ('periodic', 'open'):
                raise ValueError('explicit elemental path boundary is required')
            if kind == 'elemental_p2p_row_sparse':
                _known(definition.get('point_right_id'), 'actual right point identity')
                if conventions.get('point_pair_layout') != 'left-point-row-slots-v1':
                    raise ValueError('P/P rows require the declared left-point row-slot layout')
                left_id = document['bases']['point_left']['data_id']
                right_id = definition['point_right_id']
                selection = definition.get('point_right_selection')
                if selection == 'same_ordered_prefix_as_point_left':
                    if (right_id != left_id or ('point_right_labels' in definition and
                            canonical_json(definition['point_right_labels']) !=
                            canonical_json(spec.coordinates('point_left')))):
                        raise ValueError('P/P same-pool rows require identical actual ID and ordered labels')
                elif selection == 'explicit_ordered_labels':
                    labels = definition.get('point_right_labels')
                    if (not isinstance(labels, list) or not labels or
                            any(type(label) is not int or label < 0 for label in labels) or
                            len(set(labels)) != len(labels)):
                        raise ValueError('P/P right endpoint requires explicit unique ordered integer labels')
                else:
                    raise ValueError('P/P right endpoint selection must be explicitly declared')
                if left_id not in document['parents'] or right_id not in document['parents']:
                    raise ValueError('P/P rows must bind both actual point parents')
    selection = _shared_time_selection(input_specs)
    extent = configuration['lattice'][3] if selection is None else selection.Nt
    return configuration, tuple(range(extent))


def _validate_baryon_spec(spec):
    """Validate the producer's tagged VVV contract without reading its values."""
    document = spec.document
    if (spec.axis_names != ('time', 'component', 'momentum', 'mode0', 'mode1', 'mode2')
            or document['dtype'] != 'complex128'):
        raise ValueError('baryon Diagram input requires canonical complex128 vvv axes')
    definition = document['definition']
    product = definition.get('product')
    if _BARYON_PRODUCT_ENCODINGS.get(product) != definition.get('component_encoding'):
        raise ValueError('baryon product and component encoding are missing or inconsistent')
    if (not isinstance(definition.get('gauge_id'), str) or not definition['gauge_id']
            or definition.get('gauge_purpose') != 'elemental_path_transport'):
        raise ValueError('baryon Diagram input requires its actual elemental gauge identity')
    conventions = document['conventions']
    if (conventions.get('normalization') != 'unscaled'
            or conventions.get('phase') not in ('anchor', 'none')
            or conventions.get('path_boundary') not in ('periodic', 'open')
            or conventions.get('color_contraction') != 'epsilon'
            or conventions.get('vector_conjugation') != 'none'):
        raise ValueError('baryon Diagram input conventions are incomplete or unsupported')
    roles, views = definition.get('input_roles'), definition.get('input_views')
    role_names = {'gauge', 'basis_leg0', 'basis_leg1', 'basis_leg2'}
    if (not isinstance(roles, dict) or set(roles) != role_names
            or not isinstance(views, dict) or set(views) != role_names):
        raise ValueError('baryon Diagram input must declare all four actual role views')
    gauge_view = DatasetSpec(views['gauge'])
    if (roles['gauge'] != definition['gauge_id']
            or gauge_view.document['kind'] != 'gauge'
            or gauge_view.document['configuration'] != document['configuration']):
        raise ValueError('baryon gauge role does not match its actual output configuration')
    selection = dataset_time_selection(spec, 'time')
    extent = document['configuration']['lattice'][3] if selection is None else selection.Nt
    if spec.coordinates('time') != list(range(extent)):
        raise ValueError('baryon Diagram output time coordinates must be canonical physical order or explicit slots')
    parents = [definition['gauge_id']]
    for leg in range(3):
        axis = f'mode{leg}'
        binding = document['bases'].get(axis)
        if (not isinstance(binding, dict) or not isinstance(binding.get('data_id'), str)
                or not binding['data_id']):
            raise ValueError(f'baryon Diagram input requires an actual {axis} basis')
        if roles[f'basis_leg{leg}'] != binding['data_id']:
            raise ValueError(f'baryon {axis} actual identity disagrees with its input role')
        view = DatasetSpec(views[f'basis_leg{leg}'])
        view_labels = tuple(view.coordinates('mode')) if 'mode' in view.axis_names else ()
        requested_labels = tuple(spec.coordinates(axis))
        bound_labels = tuple(binding['indices'])
        if (view.document['kind'] not in ('explicit_basis', 'laplace_basis')
                or view.document['configuration'] != document['configuration']
                or view.axis_names[:2] != ('time', 'mode')
                or requested_labels != bound_labels
                or bound_labels != view_labels[:len(bound_labels)]):
            raise ValueError(f'baryon {axis} role view does not match its ordered mode labels')
        validate_elemental_fields(gauge_view, view, selection=selection)
        parents.extend([binding['data_id']] if binding['data_id'] not in parents else [])
    if document['parents'] != parents:
        raise ValueError('baryon parents must preserve gauge and first-use ordered basis identities')
    if product == 'projected':
        rows = definition.get('projected_rows')
        if not isinstance(rows, list) or not rows or len(rows) != len(spec.coordinates('component')):
            raise ValueError('projected baryon rows must exactly cover component coordinates')
        if [row.get('id') for row in rows if isinstance(row, dict)] != list(range(len(rows))):
            raise ValueError('projected baryon row IDs must be consecutive and ordered')
    elif product == 'distance':
        if definition.get('distance_rule') != 'independent_six_straight_paths_per_nonzero_leg':
            raise ValueError('distance baryon input must declare its fixed six-direction rule')
    elif 'projected_rows' in definition or 'distance_rule' in definition:
        raise ValueError('raw baryon input cannot carry projected/distance product metadata')


def validate_diagram_endpoint_descriptions(descriptions):
    """Require verified known endpoints for a persisted measurement.

    Generic in-memory providers may retain unknown declarations. This stricter
    registration gate does not change that existing calculator compatibility.
    """
    propagators = 0
    for description in descriptions:
        if not isinstance(description, TensorDescription):
            raise TypeError('persistent Diagram tensors require TensorDescription metadata')
        if description.role != 'propagator':
            continue
        propagators += 1
        checks = description.bindings.get('endpoint_axis_bindings')
        if not isinstance(checks, Mapping) or set(checks) != {'source', 'sink'}:
            raise ValueError('persistent Diagram requires both endpoint declarations')
        for check in checks.values():
            if (not isinstance(check, Mapping) or check.get('status') != 'verified' or
                    check.get('knowledge') != 'both-bound'):
                raise ValueError('persistent Diagram endpoint identity is unknown or unverified')
            legacy_meson = 'meson' in check
            neutral_vertex = 'vertex' in check
            if legacy_meson == neutral_vertex:
                raise ValueError('persistent Diagram endpoint record is ambiguous or lacks a vertex binding')
            vertex_side = 'vertex' if neutral_vertex else 'meson'
            if neutral_vertex:
                if (check.get('vertex_kind') not in ('baryon', 'meson')
                        or not isinstance(check.get('propagator_axis'), str)
                        or not isinstance(check.get('vertex_axis'), str)):
                    raise ValueError('neutral Diagram endpoint record lacks recognized vertex/axis metadata')
                if check['vertex_kind'] == 'baryon':
                    slot = check.get('vertex_slot')
                    if type(slot) is not int or not 0 <= slot < 3 or check.get('vertex_endpoint') != 'leg':
                        raise ValueError('baryon endpoint record must identify its ordered leg slot')
                elif check.get('vertex_endpoint') not in ('field', 'bar'):
                    raise ValueError('Meson endpoint record must identify its field or bar side')
            values = []
            for side in ('propagator', vertex_side):
                binding = check.get(side)
                if not isinstance(binding, (tuple, list)) or len(binding) != 2:
                    raise ValueError('persistent Diagram endpoint binding is missing')
                identity, labels = binding
                _known(identity, 'actual endpoint identity')
                if not isinstance(labels, (tuple, list)) or not labels:
                    raise ValueError('persistent Diagram endpoint requires ordered known labels')
                values.append((identity, tuple(canonical_coordinate_identity(value) for value in labels)))
            if values[0] != values[1]:
                raise ValueError('persistent Diagram endpoint actual ID or coordinate order mismatch')
    if not propagators:
        raise ValueError('persistent Diagram must describe at least one propagator')


def diagram_measurement_document(configuration, times, input_specifications, input_ids, science):
    """Construct the shared scientific output for concrete parents or request refs.

    ``science`` is an already detached JSON-native Diagram/operator/sampling
    snapshot. It contains no execution strategy, live readers or large arrays.
    Runtime provenance separately records the execution environment.
    """
    if (not isinstance(input_specifications, Mapping) or not input_specifications or
            not isinstance(input_ids, Mapping) or set(input_ids) != set(input_specifications)):
        raise ValueError('input views and actual ID/reference roles must agree')
    for role, identity in input_ids.items():
        _known(role, 'input role')
        if isinstance(identity, dict) and set(identity) == {'request'}:
            _known(identity['request'], 'logical input request')
        else:
            _known(identity, 'actual input identity')
    selection = _shared_time_selection(input_specifications)
    extent = configuration['lattice'][3] if selection is None else selection.Nt
    times = tuple(times)
    if times != tuple(range(extent)) or any(type(value) is not int for value in times):
        raise ValueError('persistent Diagram output must include all physical times or declared slots in order')
    if not isinstance(science, Mapping):
        raise TypeError('science must be a detached JSON-native mapping')
    snapshot = json.loads(canonical_json(dict(science)))
    views = {role: spec.document if isinstance(spec, DatasetSpec) else spec
             for role, spec in input_specifications.items()}
    document = dict(schema=1, kind='diagram_measurement', configuration=configuration,
        axes=[dict(name='source_time', coordinates=list(times)),
              dict(name='sink_time', coordinates=list(times))],
        dtype='complex128', bases={}, parents=list(input_ids.values()),
        definition=dict(measurement='fixed_diagram_full_time', science=snapshot,
                        input_roles=dict(input_ids), input_views=views),
        conventions=dict(normalization='unscaled', time_coordinates='absolute',
                         spin_basis='degrand_rossi', coefficient_owner='OutputRecipe',
                         time_coverage='all_physical'))
    if selection is not None:
        document['definition']['measurement'] = 'fixed_diagram_selected_slots'
        document['conventions']['time_coverage'] = 'selected_physical'
        for axis in document['axes']:
            axis['coordinates'] = list(selection.times)
        document = bind_time_selections(document, {'source_time': selection, 'sink_time': selection})
    return json.loads(canonical_json(document))


__all__ = ['validate_diagram_input_specs', 'validate_diagram_endpoint_descriptions',
           'diagram_measurement_document']
