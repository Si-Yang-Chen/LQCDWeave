"""Connected meson matrices with mode-block Heaviside profiles.

This is a numerical kernel: callers must supply the spin matrices and the
gamma5-transformed backward perambulator in the same convention as
``one_particle.twopoint_matrix``.  No eigenbasis transformation is implicit.
"""

from opt_einsum import backends, contract

import numpy as np


_ARRAY_MODULE_FUNCTIONS = ('empty', 'zeros', 'result_type', 'cumsum', 'stack')


def _validate_execution_backend(array_module, backend):
    """Validate caller-selected array and opt_einsum backends before work."""
    missing = [name for name in _ARRAY_MODULE_FUNCTIONS if not callable(getattr(array_module, name, None))]
    add = getattr(array_module, 'add', None)
    if not callable(getattr(add, 'reduceat', None)):
        missing.append('add.reduceat')
    if missing:
        raise ValueError(f'array_module is missing required operations: {", ".join(missing)}')

    if not isinstance(backend, str) or not backend or backend == 'auto':
        raise ValueError("backend must name an explicit opt_einsum backend (for example, 'numpy')")
    try:
        if not backends.has_einsum(backend):
            backends.get_func('tensordot', backend)
            backends.get_func('transpose', backend)
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        raise ValueError(f'unsupported opt_einsum backend: {backend!r}') from exc


def _resolve_strategy(strategy, backend):
    if strategy == 'auto':
        # CuPy benefits from a fused contraction; other explicit backends use
        # block reuse. This decision depends on the declared backend, not data types.
        return 'batched' if backend == 'cupy' else 'blocks'
    if strategy not in ('blocks', 'weighted', 'batched'):
        raise ValueError('strategy must be auto, blocks, weighted or batched')
    return strategy


def _validate(forward, backward, sink, source, cuts_a, cuts_c):
    na, nc = backward.shape[-1], forward.shape[-1]
    if backward.shape != (4, 4, na, na) or forward.shape != (4, 4, nc, nc):
        raise ValueError("perambulators must have shape [4,4,Ne,Ne]")
    if sink.ndim != 5 or source.ndim != 5 or sink.shape[1:] != (4, 4, na, nc) or source.shape[1:] != (4, 4, na, nc):
        raise ValueError("vertices must have shape [operator,4,4,backward-Ne,forward-Ne]")
    for cuts, ne in ((cuts_a, na), (cuts_c, nc)):
        if not cuts or cuts[-1] != ne or any(not isinstance(n, int) for n in cuts):
            raise ValueError("cuts must end at the corresponding Ne")
        if any(lo >= hi for lo, hi in zip((0, *cuts[:-1]), cuts)):
            raise ValueError("cuts must be strictly increasing positive integers")
    if len(cuts_a) != len(cuts_c):
        raise ValueError("both mode legs must have the same number of profiles")


def contract_meson_vertices_blocked(
    forward, backward, sink, source, cuts_a, cuts_c, *, strategy='auto', array_module=np, backend='numpy'
):
    """Return C[sink,source,sink_profile,source_profile] for nested cutoffs.

    ``backward[i,j,a,b]`` and ``forward[k,l,c,d]`` contract with
    ``sink[s,j,k,b,c]`` and ``source[r,l,i,a,d]``. Source vertex includes
    complex conjugation and gamma4 sandwich in the existing convention.
    Multi-term spin/gauge operator sums are supported by the vertex arrays.
    Profiles act on BOTH quark legs of each vertex. ``blocks`` and ``weighted``
    evaluate each of the 3**4 block terms once and recombine prefix sums;
    ``batched`` fuses all profiles into one contraction. ``array_module`` and
    ``backend`` must name compatible array and opt_einsum implementations. The
    ``auto`` strategy selects ``batched`` for the explicitly declared CuPy
    backend and ``blocks`` for every other backend.
    """
    _validate_execution_backend(array_module, backend)
    strategy = _resolve_strategy(strategy, backend)
    _validate(forward, backward, sink, source, cuts_a, cuts_c)
    xp = array_module
    ba = (0, *cuts_a)
    bc = (0, *cuts_c)
    n = len(cuts_a)
    if strategy == 'batched':
        # Fuse profile pairs into one contraction. Nested prefixes deliberately
        # repeat arithmetic, but on a GPU one large GEMM can beat 90 launches.
        sv = xp.zeros((n, *sink.shape), dtype=sink.dtype)
        rv = xp.zeros((n, *source.shape), dtype=source.dtype)
        for p, (a, c) in enumerate(zip(cuts_a, cuts_c)):
            sv[p, :, :, :, :a, :c] = sink[:, :, :, :a, :c]
            rv[p, :, :, :, :a, :c] = source[:, :, :, :a, :c]
        return -contract(
            'ijab,psjkbc,klcd,qrliad->srpq', backward, sv, forward, rv,
            optimize='greedy', backend=backend,
        )
    terms = xp.empty((n, n, n, n, len(sink), len(source)), dtype=xp.result_type(forward, backward, sink, source))

    # Source vertices are reused for all nine sink mode blocks.
    if strategy == 'blocks':
        src_blocks = {}
        for ia in range(n):
            aa = slice(ba[ia], ba[ia + 1])
            for id_ in range(n):
                dd = slice(bc[id_], bc[id_ + 1])
                src_blocks[ia, id_] = source[:, :, :, aa, dd]

    for ib in range(n):
        bb = slice(ba[ib], ba[ib + 1])
        for ic in range(n):
            cc = slice(bc[ic], bc[ic + 1])
            vertex = sink[:, :, :, bb, cc]
            # The expensive part is shared across all source operators and
            # source profile blocks. Never materialize all profile vertices.
            middle = contract(
                'ijab,sjkbc,klcd->silad', backward[:, :, :, bb], vertex, forward[:, :, cc, :],
                optimize='greedy', backend=backend,
            )
            if strategy == 'weighted':
                weighted = contract('silad,rliad->srad', middle, source, optimize='greedy', backend=backend)
                bins = xp.add.reduceat(weighted, ba[:-1], axis=2)
                bins = xp.add.reduceat(bins, bc[:-1], axis=3)
                terms[ib, ic] = bins.transpose(2, 3, 0, 1)
            else:
                for ia in range(n):
                    aa = slice(ba[ia], ba[ia + 1])
                    for id_ in range(n):
                        dd = slice(bc[id_], bc[id_ + 1])
                        terms[ib, ic, ia, id_] = contract(
                            'silad,rliad->sr', middle[:, :, :, aa, dd], src_blocks[ia, id_],
                            optimize='greedy', backend=backend,
                        )

    for axis in range(4):
        terms = xp.cumsum(terms, axis=axis)
    result = xp.stack([xp.stack([terms[i, i, j, j] for j in range(n)], axis=-1) for i in range(n)], axis=-2)
    return -result


def contract_meson_matrix_blocked(
    forward, backward, sink, source, gamma_sink, gamma_source, cuts_a, cuts_c, *,
    strategy='auto', array_module=np, backend='numpy'
):
    """Fast single spin/gauge term wrapper around the general vertex kernel.

    ``gamma_source`` is already conjugated and gamma4-sandwiched. The source
    elemental is conjugated here, matching ``one_particle.twopoint_matrix``.
    """
    _validate_execution_backend(array_module, backend)
    strategy = _resolve_strategy(strategy, backend)
    if sink.ndim != 3 or source.ndim != 3:
        raise ValueError("elementals must have shape [operator,backward-Ne,forward-Ne]")
    if gamma_sink.shape != (len(sink), 4, 4) or gamma_source.shape != (len(source), 4, 4):
        raise ValueError("spin vertices must have shape [operator,4,4]")
    vertex_sink = gamma_sink[:, :, :, None, None] * sink[:, None, None, :, :]
    vertex_source = gamma_source[:, :, :, None, None] * source[:, None, None, :, :].conj()
    return contract_meson_vertices_blocked(
        forward, backward, vertex_sink, vertex_source, cuts_a, cuts_c,
        strategy=strategy, array_module=array_module, backend=backend,
    )


def meson_matrix_from_operator_data(
    forward, backward_raw, operator_data, t_source, t_sink, gamma4, gamma5, cuts_a, cuts_c,
    *, strategy='auto', array_module=np, backend='numpy'
):
    """One sink time, all operators and cutoff pairs, in the legacy convention.

    ``operator_data`` has the same (spin_terms, elemental_terms) layout as
    ``get_elemental_data``; each elemental term is [time,NeA,NeC]. This
    adapter handles coupled irrep operators without duplicating contractions.
    The caller owns cfg/basis checks and supplies perambulator slices for the
    requested pair of times (including a separate backward flavor if needed).
    """
    _validate_execution_backend(array_module, backend)
    strategy = _resolve_strategy(strategy, backend)
    if not operator_data:
        raise ValueError("at least one operator required")
    xp = array_module
    if gamma4.shape != (4, 4) or gamma5.shape != (4, 4):
        raise ValueError("gamma4 and gamma5 must be 4x4")
    sink, source = [], []
    for spin, elemental in operator_data:
        if spin.ndim != 3 or elemental.ndim != 4 or len(spin) != len(elemental) or len(spin) == 0:
            raise ValueError("operator terms must pair [term,4,4] with [term,time,NeA,NeC]")
        sink.append(contract(
            'xjk,xbc->jkbc', spin, elemental[:, t_sink], optimize='greedy', backend=backend,
        ))
        adjoint_spin = contract(
            'lj,xkj,ki->xli', gamma4, spin.conj(), gamma4, optimize='greedy', backend=backend,
        )
        source.append(contract(
            'xli,xad->liad', adjoint_spin, elemental[:, t_source].conj(),
            optimize='greedy', backend=backend,
        ))
    sink = xp.stack(sink)
    source = xp.stack(source)
    # gamma5 hermiticity: gamma5[i,i] * raw[j,i,b,a]^* * gamma5[j,j].
    backward = contract(
        'ii,jiba,jj->ijab', gamma5, backward_raw.conj(), gamma5,
        optimize='greedy', backend=backend,
    )
    return contract_meson_vertices_blocked(
        forward, backward, sink, source, cuts_a, cuts_c,
        strategy=strategy, array_module=array_module, backend=backend,
    )
