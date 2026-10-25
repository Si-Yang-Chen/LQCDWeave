"""Freeze diquark x free-quark recipes as exact baryon-spin catalog combinations.

Each recipe is ``(q0^T Gamma q1) q2`` with ``Gamma = C Gamma'`` for every one of
the 16 Clifford elements (``C = g2 g4`` in DR), and the free quark on slot 2,
like the original distillation contraction programs (``G13`` there is C g5).
Rotation-scalar diquarks (C, C g5, C g4, C g4 g5) give one G1 doublet per
parity; rotation-vector diquarks (C g_i, C g_i g5, C g_i g4, C g_i g4 g5,
i = 1..3) give one G1 doublet and one H quartet per parity. Together the 24
recipes are a complete orthonormal 64-row basis.

Each span is reduced against the frozen catalog multiplets of the same irrep
and parity: the result is one multiplet whose canonical rows are
``sum_m w_m * catalog multiplet m``, with row-independent weights ``w``.
Runtime reads the JSON, just as for the 64-row catalog.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lqcdweave.operators.single_hadron.baryon_spin import baryon_spin_operators
from lqcdweave.operators.single_hadron.gamma import gamma

CHARGE_CONJUGATION = gamma(2) @ gamma(8)
_G5, _G4, _SPATIAL = gamma(15), gamma(8), (gamma(1), gamma(2), gamma(4))
# label -> rotation components of Gamma' (one for scalars, three for vectors).
DIQUARKS = (('C', (np.eye(4),)), ('Cg5', (_G5,)), ('Cg4', (_G4,)), ('Cg4g5', (_G4 @ _G5,)),
            ('Cgi', _SPATIAL), ('Cgig5', tuple(g @ _G5 for g in _SPATIAL)),
            ('Cgig4', tuple(g @ _G4 for g in _SPATIAL)),
            ('Cgig4g5', tuple(g @ _G4 @ _G5 for g in _SPATIAL)))


def diquark_matrices(label):
    """The ``C Gamma'`` matrices of one recipe label, in DR."""
    return tuple(CHARGE_CONJUGATION @ g for g in dict(DIQUARKS)[label])


def _multiplet(matrices, multiplets):
    """Row-independent catalog weights of the projection of ``matrices x e_c``."""
    rows = np.stack([np.stack([spin.dense().reshape(64) for spin in m.rows]) for m in multiplets])
    free = np.stack([np.einsum('ab,c->abc', matrix, np.eye(4)[c]).reshape(64)
                     for matrix in matrices for c in range(4)], axis=1)
    # One canonical multiplet means overlap[m, r, c] = w_m * Y[r, c]: rank one over m.
    overlap = np.einsum('mrk,kc->mrc', rows.conj(), free)            # (multiplet, row, free)
    weights = np.linalg.svd(overlap.reshape(len(multiplets), -1), full_matrices=False)[0][:, 0]
    weights[np.abs(weights) < 1e-12] = 0
    pivot = next(w for w in weights if w != 0)
    weights *= pivot.conjugate() / abs(pivot)
    projected = np.einsum('m,mrc->rc', weights.conj(), overlap)
    if np.abs(projected).max() < 1e-10:
        raise ArithmeticError('diquark has no part in this irrep and parity')
    if np.abs(np.einsum('m,rc->mrc', weights, projected) - overlap).max() > 1e-12:
        raise ArithmeticError('diquark part is not one canonical catalog multiplet')
    return weights, np.einsum('m,mrk->rk', weights, rows)


def build_document():
    catalog = baryon_spin_operators(version='v1')
    recipes = []
    for label, components in DIQUARKS:
        for irrep in ('G1',) if len(components) == 1 else ('G1', 'H'):
            for suffix, parity in (('g', 1), ('u', -1)):
                multiplets = [m for m in catalog if m.irrep == irrep and m.parity == parity]
                weights, rows = _multiplet(diquark_matrices(label), multiplets)
                terms = []
                for vector in rows:
                    vector.real[abs(vector.real) < 1e-12] = 0
                    vector.imag[abs(vector.imag) < 1e-12] = 0
                    terms.append([dict(spin=list(map(int, np.unravel_index(i, (4, 4, 4)))),
                                       coefficient=[float(z.real), float(z.imag)])
                                  for i, z in enumerate(vector) if z != 0])
                recipes.append(dict(name=f'diquark_{label}_{irrep}{suffix}', diquark=label, irrep=irrep,
                                    parity=parity,
                                    weights=[dict(multiplet=m.name, coefficient=[float(w.real), float(w.imag)])
                                             for m, w in zip(multiplets, weights) if w != 0],
                                    rows=terms))
    return dict(schema=1, name='baryon-diquark', version='v1', basis='degrand_rossi', catalog='baryon-spin/v1',
                convention='(q0^T C Gamma q1) q2 with C = gamma(2) gamma(8) in DR and Gamma over the 16 '
                           'Clifford elements; canonical baryon-spin G1/H rows; global phase makes the first '
                           'nonzero catalog weight real positive',
                recipes=recipes)


def build(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    document = build_document()
    (output / 'baryon_diquark_v1.json').write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return dict(recipes=len(document['recipes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output_dir)))
