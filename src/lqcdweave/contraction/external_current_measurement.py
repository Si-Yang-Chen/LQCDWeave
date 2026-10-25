"""Expert one-current external-leg scan using complete current-time solutions.

The ordinary multi-current workflow remains contraction.correlator.calculate.
This entry implements a linear one-identity estimator, not a new Wick engine.
"""
import numpy as np

from lqcdweave.contracts.sampling_weights import expected_weight
from lqcdweave.contraction.external_current import (
    apply_source_current, project_fields, sample_fields, leg_contributions,
)


def calculate_densities(fields, source_operator, sink_vertex, *, rows, links, momenta,
                        left_projections, low_vectors, old_d_vectors, medium_vectors,
                        coordinates, array_module=np, backend='numpy'):
    """Share all X projections across channels; compute unweighted leg densities."""
    xp = array_module
    applied = apply_source_current(fields, source_operator, rows, links, momenta, array_module=xp)
    result = {}
    for name, vectors in (('V', low_vectors), ('Dfull', old_d_vectors), ('Dmedium', medium_vectors)):
        right = project_fields(applied, vectors, array_module=xp)
        result[name] = leg_contributions(left_projections[name], right, sink_vertex,
                                         array_module=xp, backend=backend)
    right = sample_fields(applied, coordinates, array_module=xp).reshape(-1, 4, 4, fields.shape[-1])
    for name in ('Q100', 'Q428'):
        left = left_projections[name].reshape(-1, 4, 4, fields.shape[-1])
        values = leg_contributions(left, right, sink_vertex, array_module=xp, backend=backend)
        result[name] = values.reshape(len(coordinates), 3).sum(axis=1)
    return result


def assemble_cases(densities, cases, *, medium_source, spatial_sites, low_modes, array_module=np):
    """Linear inclusion weights: one sampled identity, exactly once per graph.

    Different D/P leg labels are not multiplied together in this estimator.
    Actual group sizes and saved column positions come from DilutionSource.
    """
    xp = array_module
    if len(densities['V']) != low_modes:
        raise ValueError('low mode count differs from actual V leg coverage')
    low = densities['V'].sum()
    groups = medium_source.groups
    if len(densities['Dmedium']) != len(medium_source.vector_labels):
        raise ValueError('medium density columns differ from actual saved dilution labels')
    values = []
    for case in cases:
        value = low
        if case['route'] == 'VD':
            m = case['d_columns']
            if type(m) is not int or not 0 < m <= len(densities['Dfull']):
                raise ValueError('requested full-complement D columns exceed saved leg coverage')
            value = value + float(expected_weight(3*spatial_sites-low_modes, m, 1)) * densities['Dfull'][:m].sum()
        elif case['route'] in ('VP', 'VDP'):
            if case['p_projection'] != ('Q100' if case['route']=='VP' else 'Q428'):
                raise ValueError('point complement differs from the declared route')
            if case['route'] == 'VDP':
                m = case['columns_per_group']
                for gi, group in enumerate(groups):
                    if m > group.m:
                        raise ValueError('requested medium columns exceed actual saved group')
                    positions = xp.asarray(medium_source.group_indices(gi)[:m])
                    value = value + float(expected_weight(group.n, m, 1)) * densities['Dmedium'][positions].sum()
            count = case['point_count']
            if type(count) is not int or not 0 < count <= len(densities[case['p_projection']]):
                raise ValueError('requested points exceed actual external-leg coverage')
            value = value + float(expected_weight(spatial_sites, count, 1)) * densities[case['p_projection']][:count].sum()
        else:
            raise ValueError('unknown route')
        values.append(value)
    return xp.stack(values)
