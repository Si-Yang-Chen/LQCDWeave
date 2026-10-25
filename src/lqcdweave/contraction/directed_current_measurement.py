"""Read-only directed-current planning and shared row writers."""
import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.contracts.elemental import integer
from lqcdweave.operators.single_hadron.api import validate_one_link_current_term
from lqcdweave.planning.directed_current import (
    directed_current_pair_views, directed_current_pair_times, directed_current_pair_document,
    directed_current_chain_views, directed_current_chain_times, directed_current_chain_document,
)
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.adapters.directed_current_dataset import load_registered_directed_current
from lqcdweave.contraction.directed_current import (
    resolve_current_term_endpoints, contract_directed_current_pair_v2v,
    contract_directed_current_v2v,
)
from lqcdweave.runtime.datasets import apply_source_policy


_ROLES = ('first_current', 'second_current', 'vsv')
_CHAIN_ROLES = ('current', 'incoming_vsv', 'outgoing_vsv')


def _operator_snapshot(terms, momentum, spec, times):
    terms = tuple(validate_one_link_current_term(term) for term in terms)
    if not terms:
        raise ValueError('nonempty linked current term collection required')
    momentum = integer(momentum, 'momentum index', 0)
    momenta = spec.coordinates('momentum')
    if momentum >= len(momenta):
        raise ValueError('momentum index exceeds registered current coverage')
    boundary = spec.document['definition']['path_boundary']
    extent = spec.document['configuration']['lattice'][3]
    serialized = []
    for term in terms:
        for time in times:
            resolve_current_term_endpoints(term, anchor_time=time,
                temporal_extent=extent, boundary=boundary)
        document = term.as_dict()
        for name in ('coefficient', 'normalization', 'wilson_r'):
            value = complex(document[name])
            document[name] = [float(value.real), float(value.imag)]
        for name in ('displacement', 'bar_offset', 'field_offset', 'link_origin_offset'):
            document[name] = list(document[name])
        serialized.append(document)
    return dict(terms=serialized, momentum=momenta[momentum]), terms


def _snapshot_terms(definition):
    terms = []
    for stored in definition['terms']:
        document = dict(stored)
        for name in ('coefficient', 'normalization', 'wilson_r'):
            pair = document[name]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('current snapshot scalar must be [real,imag]')
            document[name] = complex(*pair)
        terms.append(validate_one_link_current_term(document))
    return tuple(terms)


def _select_momentum(spec, coordinate):
    document = spec.document
    for axis in document['axes']:
        if axis['name'] == 'momentum':
            axis['coordinates'] = [coordinate]
    return DatasetSpec(document)


def _pair_requests(index, first_id, second_id, vsv_id, first_terms, second_terms,
                   first_momentum, second_momentum, usedNe, first_times, second_times):
    with index.open(first_id) as first, index.open(second_id) as second, index.open(vsv_id) as vsv:
        specs = directed_current_pair_views(first.spec, second.spec, vsv.spec, usedNe=usedNe)
        extent = specs[0].document['configuration']['lattice'][3]
        first_times = directed_current_pair_times(extent, first_times, 'first_times')
        second_times = directed_current_pair_times(extent, second_times, 'second_times')
        first_definition, _ = _operator_snapshot(first_terms, first_momentum, specs[0], first_times)
        second_definition, _ = _operator_snapshot(second_terms, second_momentum, specs[1], second_times)
        specs = (_select_momentum(specs[0], first_definition['momentum']),
                 _select_momentum(specs[1], second_definition['momentum']), specs[2])
        document = directed_current_pair_document(*specs,
            [dict(request=role) for role in _ROLES], first_definition, second_definition,
            first_times=first_times, second_times=second_times)
    requests = {role: DatasetRequest(spec, data_id=identity,
        subset_axes=('sink_mode', 'source_mode', 'momentum') if role != 'vsv' else
                    ('sink_mode', 'source_mode'))
        for role, spec, identity in zip(_ROLES, specs, (first_id, second_id, vsv_id))}
    requests['measurement'] = DatasetRequest(document)
    return requests


def plan_indexed_directed_current_pair(index, first_id, second_id, vsv_id,
                                      first_terms, second_terms, *, first_momentum=0,
                                      second_momentum=0, usedNe=None, first_times=None,
                                      second_times=None, data_ids=None,
                                      source_policy="allow_unknown"):
    """Pure numerical preview over verified existing fields; default all times.

    Terms are ordered CurrentTerm objects or v1 mappings (e.g. insertion.terms).
    An explicit anchor window is preserved; invalid open endpoints are rejected,
    never removed or silently zero-filled. No producer or output is constructed.
    """
    requests = _pair_requests(index, first_id, second_id, vsv_id, first_terms, second_terms,
        first_momentum, second_momentum, usedNe, first_times, second_times)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def write_directed_current_pair_rows(first, second, vsv, writer):
    """Borrow actual readers/views and save missing first-anchor rows synchronously.

    The original copied pair kernel owns the contraction formula. This adapter
    materializes one selected momentum of each distinct full-time raw basis;
    VSV blocks are read on demand. No total working-set limit is established.
    Publication and input lifetime belong to the calling runtime.
    """
    output = writer.spec
    first_times = output.coordinates('first_anchor_time')
    second_times = output.coordinates('second_anchor_time')
    definitions = output.document['definition']
    normalized = []
    terms = []
    momenta = []
    for role, dataset, times in zip(('first_operator', 'second_operator'),
                                    (first, second), (first_times, second_times)):
        definition = definitions[role]
        coordinate = definition['momentum']
        available = dataset.spec.coordinates('momentum')
        if coordinate not in available:
            raise ValueError('operator requires missing actual momentum coverage')
        momentum = available.index(coordinate)
        snapshot, collection = _operator_snapshot(_snapshot_terms(definition), momentum, dataset.spec, times)
        normalized.append(snapshot)
        terms.append(collection)
        momenta.append(momentum)
    document = directed_current_pair_document(first.spec, second.spec, vsv.spec,
        [first.data_id, second.data_id, vsv.data_id], *normalized,
        first_times=first_times, second_times=second_times)
    if DatasetSpec(document).definition_id != output.definition_id:
        raise ValueError('current pair writer definition differs from actual input views')
    if not writer.missing:
        return
    adapted_first = load_registered_directed_current(first, vsv)
    adapted_second = (adapted_first if first.data_id == second.data_id and
        first.spec.definition_id == second.spec.definition_id else load_registered_directed_current(second, vsv))
    ne = len(first.spec.coordinates('sink_mode'))
    for position in writer.missing:
        row = np.empty(len(second_times), dtype=np.complex128)
        for column, second_time in enumerate(second_times):
            result = contract_directed_current_pair_v2v(terms[0], adapted_first['raw'],
                adapted_first['contract'], terms[1], adapted_second['raw'],
                adapted_second['contract'], adapted_first['vsv'][0],
                first_anchor_time=first_times[position], second_anchor_time=second_time,
                first_field_ne=ne, first_bar_ne=ne, second_field_ne=ne, second_bar_ne=ne,
                first_momentum=momenta[0], second_momentum=momenta[1], array_backend=np)
            value = np.asarray(result['value'])
            if value.shape != () or not np.isfinite(value):
                raise FloatingPointError('nonfinite or nonscalar current pair value cannot be published')
            row[column] = value
        writer.write(position, row)


def _chain_requests(index, current_id, incoming_vsv_id, outgoing_vsv_id, terms,
                    momentum, usedNe, source_times, sink_times, anchor_times):
    identities = (current_id, incoming_vsv_id, outgoing_vsv_id)
    with index.open(current_id) as current, index.open(incoming_vsv_id) as incoming, \
            index.open(outgoing_vsv_id) as outgoing:
        views = directed_current_chain_views(current.spec, incoming.spec, outgoing.spec,
                                             usedNe=usedNe)
        extent = views[0].document['configuration']['lattice'][3]
        source_times = directed_current_chain_times(extent, source_times, 'source_times')
        sink_times = directed_current_chain_times(extent, sink_times, 'sink_times')
        anchor_times = directed_current_chain_times(extent, anchor_times, 'anchor_times')
        operator, _ = _operator_snapshot(terms, momentum, views[0], anchor_times)
        views = (_select_momentum(views[0], operator['momentum']), *views[1:])
        document = directed_current_chain_document(*views,
            [dict(request=role) for role in _CHAIN_ROLES], operator,
            source_times=source_times, sink_times=sink_times, anchor_times=anchor_times)
    requests = {role: DatasetRequest(view, data_id=identity,
        subset_axes=('sink_mode', 'source_mode', 'momentum') if role == 'current' else
                    ('sink_mode', 'source_mode'))
        for role, view, identity in zip(_CHAIN_ROLES, views, identities)}
    requests['measurement'] = DatasetRequest(document)
    return requests


def plan_indexed_directed_current_chain(index, current_id, incoming_vsv_id,
                                        outgoing_vsv_id, terms, *, momentum=0,
                                        usedNe=None, source_times=None, sink_times=None,
                                        anchor_times=None, data_ids=None,
                                        source_policy="allow_unknown"):
    """Pure preview of one persistent open V/V current tensor chain.

    The requested modes are an explicit ordered prefix. Source, sink and
    anchor times default to all physical times and preserve caller order.
    Actual basis/configuration and each role's gauge/Dirac metadata are
    retained; no identity is inferred from array shape.
    """
    requests = _chain_requests(index, current_id, incoming_vsv_id, outgoing_vsv_id,
        terms, momentum, usedNe, source_times, sink_times, anchor_times)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def write_directed_current_chain_rows(current, incoming_vsv, outgoing_vsv, writer):
    """Write missing source-time rows using the original open V/V current kernel.

    Input readers/views are borrowed. The row axis partitions writes by source
    time, but this function does not establish a total memory bound.
    """
    output = writer.spec
    source_times = output.coordinates('source_time')
    sink_times = output.coordinates('sink_time')
    anchor_times = output.coordinates('anchor_time')
    definition = output.document['definition']['operator']
    coordinate = definition['momentum']
    available = current.spec.coordinates('momentum')
    if coordinate not in available:
        raise ValueError('operator requires missing actual momentum coverage')
    momentum = available.index(coordinate)
    operator, terms = _operator_snapshot(_snapshot_terms(definition), momentum,
                                         current.spec, anchor_times)
    document = directed_current_chain_document(current.spec, incoming_vsv.spec,
        outgoing_vsv.spec, [current.data_id, incoming_vsv.data_id,
        outgoing_vsv.data_id], operator, source_times=source_times,
        sink_times=sink_times, anchor_times=anchor_times)
    if DatasetSpec(document).definition_id != output.definition_id:
        raise ValueError('current chain writer definition differs from actual input views')
    if not writer.missing:
        return

    adapted = load_registered_directed_current(current, incoming_vsv, outgoing_vsv)
    ne = len(current.spec.coordinates('sink_mode'))
    for position in writer.missing:
        row = np.empty((len(sink_times), len(anchor_times), 4, 4, ne, ne),
                       dtype=np.complex128)
        source_time = source_times[position]
        for sink_position, sink_time in enumerate(sink_times):
            for anchor_position, anchor_time in enumerate(anchor_times):
                result = contract_directed_current_v2v(terms, adapted['raw'],
                    adapted['contract'], adapted['vsv'][0], adapted['vsv'][1],
                    source_time=source_time, sink_time=sink_time,
                    anchor_time=anchor_time, current_source_ne=ne,
                    current_sink_ne=ne, momentum=momentum)
                value = np.asarray(result['value'])
                if (tuple(result['axes']) != ('external_sink_spin', 'external_source_spin',
                                              'external_sink_ne', 'external_source_ne') or
                        value.shape != (4, 4, ne, ne) or
                        not np.issubdtype(value.dtype, np.complexfloating) or
                        not np.all(np.isfinite(value))):
                    raise FloatingPointError('invalid or nonfinite open current-chain tensor')
                row[sink_position, anchor_position] = value
        writer.write(position, row)
