"""Freeze the complete DR three-spin basis under double cubic x S3 x parity.

Offline dependencies are the existing correlator extra. Runtime reads JSON,
just as meson55 does. Ordered matrix-unit columns fix copies and phases.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lqcdweave.symmetry import irrep_matrices, row_subduction

from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.correlator.api import QuarkExchange


def _columns(projector):
    """Deterministic orthonormal projected coordinate columns, without eigengauges."""
    columns = []
    for column in projector.T:
        value = column.copy()
        for _ in range(2):
            for previous in columns:
                value -= previous * np.vdot(previous, value)
        norm = np.linalg.norm(value)
        if norm > 1e-10:
            value /= norm
            pivot = next(z for z in value if abs(z) > 1e-10)
            columns.append(value * (pivot.conjugate()/abs(pivot)))
    return np.column_stack(columns)


def _spin_rotations():
    fundamental = irrep_matrices((0, 0, 0), 'G1')
    pauli = np.asarray([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
    physical = [-1j*gamma(j)@gamma(k) for j,k in ((2,4),(4,1),(1,2))]
    result = []
    for matrix in fundamental.values():
        matrix = np.asarray(matrix, complex)
        dr = np.trace(matrix)/2*np.eye(4, dtype=complex)
        for sigma, generator in zip(pauli, physical):
            dr += np.trace(sigma@matrix)/2*generator
        result.append(np.kron(np.kron(dr.conj(),dr.conj()),dr.conj()))
    return tuple(fundamental), result


def build_document():
    names, rotations = _spin_rotations()
    parity = np.kron(np.kron(gamma(8),gamma(8)),gamma(8)).conj()
    exchanges = {label: QuarkExchange(kind).matrices for label,kind in
                 (('S','symmetric'),('A','antisymmetric'),('M','mixed'))}
    permutations = tuple(exchanges['M'])
    identity = np.eye(64).reshape(4,4,4,64)
    actions = [identity.transpose(*(p.index(i) for i in range(3)),3).reshape(64,64)
               for p in permutations]
    rows, multiplets = [], []
    for suffix, eta in (('g',1),('u',-1)):
        parity_columns = _columns((np.eye(64)+eta*parity)/2)
        exchange_actions = [parity_columns.conj().T@u@parity_columns for u in actions]
        for exchange, matrices in exchanges.items():
            reduction = row_subduction(exchange_actions,[np.asarray(matrices[p],complex) for p in permutations])
            exchange_columns = [parity_columns@basis for basis in reduction['row_bases']]
            seed = exchange_columns[0]
            cubic_actions = [seed.conj().T@u@seed for u in rotations]
            for irrep in ('G1','H'):
                targets = irrep_matrices((0,0,0),irrep)
                cubic = row_subduction(cubic_actions,[np.asarray(targets[g],complex) for g in names])
                for copy in range(cubic['multiplicity']):
                    components = ('MS','MA') if exchange == 'M' else (exchange,)
                    for e, component in enumerate(components):
                        name = f'spin_{irrep}{suffix}_{component}_copy{copy}'
                        row_ids = []
                        for r,basis in enumerate(cubic['row_bases']):
                            vector = exchange_columns[e]@basis[:,copy]
                            # Remove projector roundoff only in the offline definition.
                            vector.real[abs(vector.real)<1e-12] = 0
                            vector.imag[abs(vector.imag)<1e-12] = 0
                            row_id = len(rows)
                            row_ids.append(row_id)
                            rows.append(dict(id=row_id,catalog='baryon-spin',version='v1',
                                name=f'{name}/row{r}',multiplet=name,
                                irrep=irrep,parity=eta,exchange_irrep=exchange,
                                exchange_component=component,copy=copy,row=r,
                                terms=[dict(spin=list(map(int,np.unravel_index(i,(4,4,4)))),
                                            coefficient=[float(z.real),float(z.imag)])
                                       for i,z in enumerate(vector) if z != 0]))
                        multiplets.append(dict(name=name,irrep=irrep,parity=eta,exchange_irrep=exchange,
                            exchange_component=component,copy=copy,row_ids=row_ids,
                            partner=(f'spin_{irrep}{suffix}_{components[1-e]}_copy{copy}'
                                     if exchange == 'M' else None)))
    matrix = np.column_stack([sum((complex(*t['coefficient'])*np.eye(64)[:,
        np.ravel_multi_index(t['spin'],(4,4,4))] for t in row['terms']),np.zeros(64,complex)) for row in rows])
    if matrix.shape != (64,64) or not np.allclose(matrix.conj().T@matrix,np.eye(64),atol=1e-11,rtol=0):
        raise ArithmeticError('joint projections do not form a complete orthonormal 64-row basis')
    return dict(schema=1,name='baryon-spin',version='v1',basis='degrand_rossi',
        dimension=64,group='Oh-double x S3',
        convention='coefficient columns transform by S* on each spin leg; exchange is transpose(inverse(p)); canonical lattice_symmetry cubic rows and QuarkExchange (MS,MA) rows; ordered projected columns fix copies and phases',
        reference='https://arxiv.org/abs/hep-lat/0508018',
        rows=rows,multiplets=multiplets)


def build(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    document = build_document()
    (output/'baryon_spin_v1.json').write_bytes((json.dumps(document,indent=2)+'\n').encode('utf-8'))
    return dict(rows=len(document['rows']),multiplets=len(document['multiplets']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output_dir)))
