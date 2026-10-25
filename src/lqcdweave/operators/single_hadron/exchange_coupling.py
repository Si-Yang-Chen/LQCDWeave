"""Coefficient-only exchange projection after spin x spatial cubic coupling.

Spatial recipes represent epsilon-contracted VVV coefficients. Slot actions
therefore include one color sign. Flavor is combined later in the correlator.
"""

from collections import defaultdict
from copy import copy

from lqcdweave.contracts.elemental import (
    BaryonComponent, GaugePath, elemental_operator_rows, integer,
)
from lqcdweave.contracts.elemental_catalog import embedded_catalog
from .baryon import SparseSpinTensor
from .operator import InsertionTerm


def _exact(value, sp):
    value = complex(value)
    return sp.Rational(str(value.real)) + sp.I*sp.Rational(str(value.imag))


def _add(target, source, weight, sp):
    for key, value in source.items():
        combined = sp.expand(target.get(key, 0)+weight*value)
        if combined == 0:
            target.pop(key, None)
        else:
            target[key] = combined


def _spatial_definitions(insertion, operator_table):
    if operator_table is not None:
        return dict(elemental_operator_rows(operator_table, rank=3)), getattr(operator_table, 'catalog', None)
    definitions, catalog = {}, None
    for row in insertion.rows:
        for term in row:
            key = term.component.key
            if isinstance(key,tuple) and key[:3] == ('catalog','baryon-vvv','v1'):
                from .baryon_catalog import baryon_preset
                if catalog is None:
                    catalog = baryon_preset().catalog
                definitions[term.component] = catalog.rows[key[3]]
            elif isinstance(key,tuple) and len(key) == 3 and all(isinstance(path, GaugePath) for path in key):
                definitions[term.component] = ((key,1),)
            else:
                raise ValueError('exchange projection needs the matching spatial operator_table')
    return definitions, catalog


def _expanded_rows(insertion, definitions, sp):
    rows = []
    for row in insertion.rows:
        result = {}
        for term in row:
            if not isinstance(term.component,BaryonComponent) or term.spin is None:
                raise ValueError('quark exchange projection requires baryon spin x spatial rows')
            try:
                spatial = definitions[term.component]
            except KeyError as exc:
                raise ValueError('spatial operator_table lacks a referenced component') from exc
            for paths, coefficient in spatial:
                if any(step >= 6 for path in paths for step in path.steps):
                    raise NotImplementedError('exchange projection currently requires spatial baryon paths')
                paths = tuple(path.steps for path in paths)
                for a,b,c,weight in term.spin.entries:
                    key = (paths,term.anchor_time_offset,term.endpoint_time_offsets,(a,b,c))
                    _add(result,{key:_exact(term.coefficient,sp)*_exact(coefficient,sp)*_exact(weight,sp)},1,sp)
        rows.append(result)
    return rows


def _move(row, permutation):
    inverse = tuple(permutation.index(i) for i in range(3))
    sign = (-1)**sum(permutation[i]>permutation[j] for i in range(3) for j in range(i+1,3))
    return {(tuple(paths[i] for i in inverse),anchor,tuple(offsets[i] for i in inverse),
             tuple(spins[i] for i in inverse)): sign*value
            for (paths,anchor,offsets,spins),value in row.items()}


def _unit(row, representation, target, source, sp):
    result = {}
    factor = sp.Rational(representation.dimension,len(representation.elements))
    for permutation,matrix in representation.elements:
        weight = factor*sp.conjugate(matrix[target,source])
        if weight != 0:
            _add(result,_move(row,permutation),weight,sp)
    return result


def _independent_copies(seed, representation, sp):
    """Align copies in row zero, retaining arbitrary small user coefficients.

    At most two candidate columns exist. Exact Gram--Schmidt avoids a numeric
    rank tolerance and the same coefficients transport every cubic/S3 row.
    No normalization of the user's spatial or spin coefficients is applied.
    """
    vectors, combinations = [], []
    for source in range(representation.dimension):
        candidate = _unit(seed,representation,0,source,sp)
        weights = [sp.S.One if j == source else sp.S.Zero for j in range(representation.dimension)]
        for previous, combination in zip(vectors,combinations):
            norm = sum(sp.conjugate(value)*value for value in previous.values())
            overlap = sum(sp.conjugate(value)*candidate.get(key,0) for key,value in previous.items())
            ratio = sp.cancel(sp.expand(overlap)/sp.expand(norm))
            _add(candidate,previous,-ratio,sp)
            weights = [sp.cancel(a-ratio*b) for a,b in zip(weights,combination)]
        if candidate:
            vectors.append(candidate)
            combinations.append(tuple(weights))
    return tuple(combinations)


def _raw_table(paths, catalog):
    """Reuse the original directory's unit path triples where it covers them."""
    paths = tuple(sorted(paths))
    if catalog is not None:
        lookup = {tuple(path.steps for path in terms[0][0]):i for i,terms in reversed(tuple(enumerate(catalog.rows)))
                  if len(terms) == 1 and terms[0][1] == 1}
        if all(path in lookup for path in paths):
            table = catalog.table(tuple(lookup[path] for path in paths))
            return table, {path:row['component'] for path,row in zip(paths,table)}
    records = [dict(terms=[dict(paths=path,coefficient=[1,0])]) for path in paths]
    table = embedded_catalog(records,rank=3).table()
    return table, {path:row['component'] for path,row in zip(paths,table)}


def project_exchange(insertion, target, *, operator_table=None, copy_index=0, slots=None):
    """Project spin x epsilon-contracted spatial coefficients, before flavor.

    S/A return one Insertion; mixed returns the aligned (MS, MA) Insertions.
    MS or MA selects one partner. copy_index selects independent matrix-unit
    seed columns; slots restricts S/A to one S2 pair of the three baryon legs.
    """
    import sympy as sp
    from lqcdweave.contracts.quark_exchange import QuarkExchange
    from .insertion import Insertion
    aliases = {'symmetric':'S','antisymmetric':'A','M':'mixed'}
    target = aliases.get(target,target)
    if target not in ('S','A','MS','MA','mixed'):
        raise ValueError('exchange target must be S, A, MS, MA or mixed')
    if any(getattr(term,'spin',None) is None for row in insertion.rows for term in row):
        raise ValueError('quark exchange projection requires baryon spin x spatial rows')
    index = integer(copy_index,'exchange copy_index',0)
    kind = {'S':'symmetric','A':'antisymmetric'}.get(target,'mixed')
    if slots is not None:
        if kind == 'mixed':
            raise ValueError('S2 pair exchange has only S and A irreducible representations')
        slots = tuple(slots)
        representation = QuarkExchange.pair(kind,slots=slots)
    else:
        representation = QuarkExchange(kind)
    definitions,catalog = _spatial_definitions(insertion,operator_table)
    expanded = _expanded_rows(insertion,definitions,sp)
    combinations = _independent_copies(expanded[0],representation,sp)
    if index >= len(combinations):
        raise ValueError(f'exchange projection has {len(combinations)} independent copies of {kind}')
    weights = combinations[index]
    projected = []
    for component in range(representation.dimension):
        rows = []
        for original in expanded:
            row = {}
            for source,weight in enumerate(weights):
                if weight != 0:
                    _add(row,_unit(original,representation,component,source,sp),weight,sp)
            rows.append(row)
        projected.append(rows)
    all_paths = {key[0] for rows in projected for row in rows for key in row}
    table,components = _raw_table(all_paths,catalog)
    basis = insertion.rows[0][0].spin.basis
    results = []
    for component,rows in enumerate(projected):
        output = []
        for row in rows:
            grouped = defaultdict(list)
            for (paths,anchor,offsets,indices),value in sorted(row.items()):
                grouped[paths,anchor,offsets].append((*indices,complex(value)))
            output.append(tuple(InsertionTerm(components[paths],spin=SparseSpinTensor(entries,basis),
                anchor_time_offset=anchor,endpoint_time_offsets=offsets)
                for (paths,anchor,offsets),entries in grouped.items()))
        # Matrix units can remove an absent component of a malformed cubic
        # multiplet; report the missing support rather than inventing a row.
        if any(not row for row in output):
            raise ValueError('exchange projection has zero support in a supplied cubic row')
        result = copy(insertion)
        result.rows = Insertion.from_rows(output,insertion.momentum_dict).rows
        result._generic_rows = True
        result.operator_table = table
        result.exchange_irrep = 'M' if kind == 'mixed' else target
        result.exchange_component = ('MS','MA')[component] if kind == 'mixed' else target
        result.exchange_copy_index = index
        result.exchange_multiplicity = len(combinations)
        result.exchange_slots = slots
        result.exchange_matrices = representation.matrices
        results.append(result)
    if target == 'mixed':
        return tuple(results)
    return results[1 if target == 'MA' else 0]
