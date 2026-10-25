"""Read-only indexed perambulator-pair coverage planning."""

from contextlib import ExitStack
import numpy as np
from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.contracts.propagator import PropagatorEndpoint
from lqcdweave.contracts.mode_partition import ModePartition
from lqcdweave.production.perambulator.api import (
    PropagatorReader, propagator_pair_spec,
)


def _validated_document(spec):
    doc = spec.document
    if doc['kind'] != 'perambulator_pairs' or doc['definition'].get('encoding') != 'pr-six-pairs-v1':
        raise ValueError('expected a six-representative pair specification')
    definition, conventions = doc['definition'], doc['conventions']
    encoded = conventions['gamma5']
    expected = propagator_pair_spec(definition['product'], configuration=doc['configuration'],
        sink=PropagatorEndpoint.from_document(definition['sink_endpoint']),
        source=PropagatorEndpoint.from_document(definition['source_endpoint']), time_pairs=spec.coordinates('time_pair'),
        gauge_id=definition['gauge_id'], dirac=definition['dirac'],
        gamma5=np.asarray(encoded['real']) + 1j*np.asarray(encoded['imag']),
        spin_basis=conventions['spin_basis'], gamma5_hermitian=conventions['gamma5_hermitian'], dtype=doc['dtype'])
    if expected.document != doc:
        raise ValueError('pair specification differs from its actual endpoints or conventions')
    return doc


def _context(doc):
    return canonical_json(dict(configuration=doc['configuration'], gauge_id=doc['definition']['gauge_id'],
        dirac=doc['definition']['dirac'], conventions=doc['conventions']))


def plan_indexed_perambulator_pairs(index, spec):
    """Preview exact coverage, including disjoint shards and Gamma5 reversal.

    Inputs must be registered actual parents. This performs metadata inspection
    only and never loads numeric rows, starts solves or creates output files.
    Returned missing pairs are informational; the caller chooses what to compute.
    """
    doc = _validated_document(spec)
    parents = {identity: index.observe_metadata(identity)['spec'] for identity in doc['parents']}
    if any(parent.document['configuration'] != doc['configuration'] for parent in parents.values()):
        raise ValueError('pair parent configuration differs from the requested output')
    if parents[doc['definition']['gauge_id']].document['kind'] != 'gauge':
        raise ValueError('pair requires an actual registered gauge parent')
    endpoints = [PropagatorEndpoint.from_document(doc['definition'][side+'_endpoint']) for side in ('sink', 'source')]
    for endpoint in endpoints:
        parent = parents[endpoint.data_id]
        axis = 'point' if endpoint.kind == 'P' else 'mode'
        if (parent.document['kind'] not in (('spatial_points',) if axis == 'point' else ('laplace_basis', 'explicit_basis'))
                or any(label not in parent.coordinates(axis) for label in endpoint.indices)):
            raise ValueError('pair endpoint lacks registered actual coverage')
        projection = endpoint.metadata.get('high_mode_projection')
        if projection is not None:
            part = ModePartition.from_document(projection)
            parent = parents[part.parent_basis_id]
            if (parent.document['conventions'].get('normalization') != 'orthonormal'
                    or any(label not in parent.coordinates('mode') for label in part.projection_modes)):
                raise ValueError('high pair parent must cover the complete orthonormal low+medium space')
    with ExitStack() as stack:
        candidates = []
        ids = []
        context = _context(doc)
        for observation in index.observations():
            observed = observation['spec'].document
            if observed['kind'] != 'perambulator_pairs' or observed['definition'].get('encoding') != 'pr-six-pairs-v1':
                continue
            definition = observed['definition']
            same = (definition.get('sink_endpoint'), definition.get('source_endpoint')) == tuple(ep.document for ep in endpoints)
            reverse = (doc['definition']['product'][0] == doc['definition']['product'][2]
                       and (definition.get('source_endpoint'), definition.get('sink_endpoint')) == tuple(ep.document for ep in endpoints))
            if (observed['dtype'] == doc['dtype']
                    and definition.get('product') == doc['definition']['product'] and (same or reverse)
                    and _context(observed) == context):
                ids.append(observation['data_id'])
                candidates.append(stack.enter_context(index.open(observation['data_id'])))
        requested = tuple(tuple(pair) for pair in spec.coordinates('time_pair'))
        covered = (set(PropagatorReader(candidates).coverage(doc['definition']['product'],
                       sink=endpoints[0], source=endpoints[1])) if candidates else set())
    missing = tuple(pair for pair in requested if pair not in covered)
    return dict(action='compute' if missing else 'reuse', spec=spec,
                covered_pairs=tuple(pair for pair in requested if pair in covered),
                missing_pairs=missing, data_ids=tuple(ids))
