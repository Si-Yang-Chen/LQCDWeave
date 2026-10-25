"""Single-current connected meson trace with one sampled spatial identity.

Expert numerical boundary. X[r,s,z,y,x,c,n] is S(t_current,t_meson)V[n].
The other current leg stays complete. This is not the two sampled-leg
estimator and does not support two currents or disconnected contractions.
"""
import numpy as np
from opt_einsum import contract

from lqcdweave.production.elemental.transport import path_transport, momentum_phase
from lqcdweave.production.perambulator.highmode import sample_high_points
from lqcdweave.operators.single_hadron.gamma import gamma


def project_fields(fields, vectors, *, array_module=np):
    """Return <vector|X> as (vector,current_spin,meson_spin,meson_mode)."""
    xp = array_module
    return xp.einsum('kq,rsqn->krsn', vectors.reshape(len(vectors), -1).conj(),
                     fields.reshape(4, 4, -1, fields.shape[-1]), optimize=True)


def sample_fields(fields, coordinates, *, parent_vectors=None, array_module=np):
    """Sample X or Q X. Q acts only on this leg, never on the operator leg."""
    xp = array_module
    zsize, ysize, xsize = fields.shape[2:5]
    x, y, z = xp.asarray(coordinates).T
    if parent_vectors is None:
        return fields[:, :, z, y, x, :, :].transpose(2, 3, 0, 1, 4).copy()
    n = fields.shape[-1]
    rhs = fields.transpose(1, 6, 2, 3, 4, 0, 5).reshape(4*n, zsize, ysize, xsize, 4, 3)
    sampled = sample_high_points(rhs, xp.asarray(coordinates), parent_vectors, array_module=xp)
    return sampled.reshape(4, n, len(coordinates), 4, 3).transpose(2, 4, 3, 0, 1)


def apply_source_current(fields, operator, rows, links, momenta, *, array_module=np):
    """Return gamma5 * (gamma4 Gamma† gamma4) * O_spatial† X.

    Uses the same path transport and anchor phases as ElementalGenerator.
    Complete X supplies displaced endpoints, including points not sampled.
    """
    xp = array_module
    n = fields.shape[-1]
    lz, ly, lx = fields.shape[2:5]
    vectors = fields.transpose(0, 1, 6, 2, 3, 4, 5).reshape(16*n, lz, ly, lx, 3)
    result = xp.zeros_like(fields)
    g4, g5 = gamma(8, array_module=xp), gamma(15, array_module=xp)
    for term in operator.terms:
        if term.anchor_time_offset or term.component.time_offsets != (0, 0):
            raise NotImplementedError('external-current kernel supports purely spatial equal-time operators')
        spin = g5 @ g4 @ gamma(term.gamma, array_module=xp).conj().T @ g4
        for path, coefficient in rows[term.component]:
            if path.displacement[3]:
                raise NotImplementedError('temporal paths require additional full solution slices')
            transport = path_transport(links, path, 0, array_module=xp)
            phase = momentum_phase((lx, ly, lz, 1), momenta[term.momentum],
                                   path.displacement, 'anchor', array_module=xp)
            moved = xp.einsum('zyxba,ezyxb->ezyxa', transport.conj(), vectors, optimize=True)
            moved *= phase.conj()[None, ..., None]
            dx, dy, dz, _ = path.displacement
            moved = xp.roll(moved, (dz, dy, dx), axis=(1, 2, 3))
            moved = moved.reshape(4, 4, n, lz, ly, lx, 3).transpose(0, 1, 3, 4, 5, 6, 2)
            result += (term.coefficient * coefficient).conjugate() * xp.einsum(
                'rv,vszyxcn->rszyxcn', spin, moved, optimize=True)
    return result


def leg_contributions(left, operator_leg, sink_vertex, *, array_module=np, backend='numpy'):
    """Return each identity-insertion contribution, including connected Wick minus.

    left/operator_leg: (leg,4,4,Nmeson), sink_vertex: (4,4,Nmeson,Nmeson).
    Source dagger and gamma5 on the current spin are already in operator_leg.
    No sampling weight, volume factor or flavor factor is inserted here.
    """
    if left.shape != operator_leg.shape or left.ndim != 4 or left.shape[1:3] != (4, 4):
        raise ValueError('external legs must have matching (leg,4,4,Nmeson) shapes')
    n = left.shape[-1]
    if sink_vertex.shape != (4, 4, n, n):
        raise ValueError('sink vertex has incompatible spin/mode axes')
    xp = array_module
    middle = contract('swmn,wt->stmn', sink_vertex, gamma(15, array_module=xp),
                      optimize='greedy', backend=backend)
    return -contract('stmn,krtn,krsm->k', middle, left.conj(), operator_leg,
                     optimize='greedy', backend=backend)


def reverse_flow_leg(tau, *, array_module=np):
    """V(t)† S(t,0) xi -> xi† S(0,t)V(t), (leg,4,4,Nmeson)."""
    xp=array_module
    g5=gamma(15,array_module=xp)
    adjoint=tau.transpose(3,1,0,2).conj()
    return xp.einsum('ru,kuvn,vs->krsn',g5,adjoint,g5,optimize=True)


def flow_operator_leg(tau_operator_source, gamma_index, coefficient=1, *, array_module=np):
    """Use S O xi with all RHS at 0 to build xi† gamma5 A_source S V."""
    xp=array_module
    g4,g5=gamma(8,array_module=xp),gamma(15,array_module=xp)
    spin=complex(coefficient).conjugate()*g5@g4@gamma(gamma_index,array_module=xp).conj().T@g4
    reverse=reverse_flow_leg(tau_operator_source,array_module=xp)
    return xp.einsum('rv,kvsn->krsn',spin,reverse,optimize=True)
