"""Frozen v1 VVV spatial directory: one-link triples, distances and exchanges.

The directory lists available recipes; it never chooses a measurement, basis,
momentum, or spin/flavor coefficient. Exchange matrices refer to the spatial
coefficient including color epsilon, for three identical ordered vector bases.
"""

from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
from math import sqrt

import numpy as np

from lqcdweave.contracts.elemental_catalog import ElementalCatalog


def _parity(p):
    return (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))


def _matrices():
    identity = (0, 1, 2)
    matrices = {identity: np.eye(2)}
    generators = (((1, 0, 2), np.diag([1, -1])),
                  ((0, 2, 1), np.array([[-.5, sqrt(3)/2], [sqrt(3)/2, .5]])))
    pending = [identity]
    while pending:
        p = pending.pop()
        for q, matrix in generators:
            pq = tuple(q[p[i]] for i in range(3))
            if pq not in matrices:
                matrices[pq] = matrix @ matrices[p]
                pending.append(pq)
    return matrices


@lru_cache(maxsize=1)
def _directory():
    paths = ((), *((i,) for i in range(6)))
    rows, metadata = [], []

    def add(name, terms, **extra):
        number = len(rows)
        rows.append(dict(id=number, terms=[dict(paths=triple, coefficient=[float(c), 0])
                                          for triple, c in terms]))
        metadata.append(dict(name=name, **extra))

    for indices in product(range(7), repeat=3):
        name = 'local' if indices == (0, 0, 0) else 'paths/' + ','.join(map(str, indices))
        add(name, [(tuple(paths[i] for i in indices), 1)], kind='raw_paths')
    for distances in product(range(2), repeat=3):
        choices = [((),) if d == 0 else tuple((direction,)*d for direction in range(6)) for d in distances]
        weight = 1/np.prod([len(values) for values in choices])
        add('distance/'+','.join(map(str, distances)), [(triple, weight) for triple in product(*choices)],
            kind='distance', distances=distances)
    matrices = _matrices()
    for indices in combinations_with_replacement(range(7), 3):
        seed = tuple(paths[i] for i in indices)
        prefix = 's3/'+','.join(map(str, indices))
        for name in ('S', 'A'):
            add(prefix+'/'+name, [(tuple(seed[i] for i in p),
                                   (_parity(p) if name == 'S' else 1)/sqrt(6)) for p in permutations(range(3))],
                kind='projected', exchange=name, seed=indices)
        for occurrence, component in product(range(2), range(2)):
            name = ('MS', 'MA')[component]
            add(prefix+f'/mixed{occurrence}/'+name,
                [(tuple(seed[i] for i in p), _parity(p)*matrix[occurrence, component]/sqrt(3))
                 for p, matrix in sorted(matrices.items())],
                kind='projected', exchange=name, occurrence=occurrence, seed=indices)
    for slots in ((0, 1), (0, 2), (1, 2)):
        other = next(i for i in range(3) if i not in slots)
        for pair in combinations_with_replacement(range(7), 2):
            for third in range(7):
                indices = [None]*3
                indices[slots[0]], indices[slots[1]], indices[other] = *pair, third
                seed = tuple(paths[i] for i in indices)
                moved = list(seed)
                moved[slots[0]], moved[slots[1]] = moved[slots[1]], moved[slots[0]]
                for name, sign in (('S', 1), ('A', -1)):
                    add(f's2/{slots[0]}{slots[1]}/'+','.join(map(str, indices))+'/'+name,
                        [(seed, 1/sqrt(2)), (tuple(moved), -sign/sqrt(2))],
                        kind='projected', exchange=name, slots=slots, seed=tuple(indices))
    catalog = ElementalCatalog.from_rows('baryon-vvv', 'v1', rows, rank=3)
    return catalog, tuple(metadata)


def baryon_preset(*, version='v1'):
    """Return the versioned spatial directory, with stable numeric component IDs.

    v1 has 343 raw triples from local and six one-link paths, 8 independent
    six-direction distance averages (each leg 0/1), 504 S3 rows (S/A and both
    MS/MA occurrences), and 1176 S2 rows at all three pair placements.
    Repeated-path projection rows retain their specified normalization and
    exact algebraic cancellation; they are not independently renormalized.
    """
    if version != 'v1':
        raise ValueError('baryon preset version must be v1')
    catalog, metadata = _directory()
    return catalog.table(metadata=metadata)
