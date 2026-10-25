"""Checked low-spectrum extension without rotating the supplied prefix."""

from .laplacian import spatial_laplacian


def extend_basis(links, prefix, total, tol, xp, *, solve=None, fix_phase=True):
    shape = tuple(links.shape[1:4]) + (3,)
    prefix = xp.asarray(prefix)
    if (prefix.ndim != 5 or tuple(prefix.shape[1:]) != shape
            or prefix.dtype.kind != 'c' or not 0 < len(prefix) <= total):
        raise ValueError('existing basis must be complex (Nold,Lz,Ly,Lx,3), with 0 < Nold <= Ne')
    prefix = prefix.astype(xp.complex128, copy=True)
    if not bool(xp.isfinite(prefix).all()):
        raise ValueError('existing basis contains nonfinite values')
    old = len(prefix)
    u = prefix.reshape(old, -1).T
    dimension = u.shape[0]
    # This is a scientific acceptance gate, not a hot-path debug scan. An
    # arbitrary saved basis may belong to a different gauge or smearing.
    threshold = max(10 * tol, 1e-11)

    def apply(v):
        result = spatial_laplacian(v.reshape(*shape, -1), links, xp).reshape(dimension, -1)
        return result[:, 0] if v.ndim == 1 else result

    gram = u.conj().T @ u
    gram_error = float(xp.linalg.norm(gram - xp.eye(old), ord=2))
    if gram_error > threshold:
        raise ValueError('existing basis is not orthonormal')
    au = apply(u)
    old_values = xp.sum(u.conj() * au, axis=0).real / xp.diag(gram).real
    residual = xp.linalg.norm(au - u * old_values, axis=0) / xp.maximum(1., xp.abs(old_values))
    old_residual = float(residual.max())
    if old_residual > threshold:
        raise ValueError('existing basis fails the current Laplace residual check')
    order_slack = threshold * max(1., float(xp.max(xp.abs(old_values))))
    if bool(xp.any(xp.diff(old_values) < -order_slack)):
        raise ValueError('existing basis is not in ascending spectral order')

    # A small accepted Gram error must not make Q non-idempotent. QR is used
    # ONLY for the internal projector; the output prefix is never changed.
    q, _ = xp.linalg.qr(u, mode='reduced')

    def complement(v):
        return v - q @ (q.conj().T @ v)

    missing = total - old
    count = max(1, missing)  # Even a no-growth request checks prefix completeness.
    if solve is None:
        if xp.__name__ == 'numpy':
            from scipy.sparse import linalg
        else:
            from cupyx.scipy.sparse import linalg
        # Frobenius norms bound link operator norms, also for nonunitary input.
        bound = 6. + 2. * sum(float(xp.sqrt(xp.sum(xp.abs(link)**2, axis=(-2, -1))).max())
                             for link in links)
        shift = max(bound, float(old_values.max())) + 1.

        def projected(v):
            inside = q @ (q.conj().T @ v)
            return complement(apply(v - inside)) + shift * inside

        operator = linalg.LinearOperator((dimension, dimension), matvec=projected,
            matmat=projected, dtype=xp.complex128)
        _, columns = linalg.eigsh(operator, k=count, which='SA', tol=tol)
    else:
        vectors, _ = solve(prefix.copy(), count)
        vectors = xp.asarray(vectors)
        if vectors.shape != (count, *shape) or not bool(xp.isfinite(vectors).all()):
            raise ValueError('solve_complement returned invalid vectors')
        columns = vectors.reshape(count, dimension).T

    columns = complement(columns)
    if float(xp.linalg.svd(columns, compute_uv=False).min()) < .5:
        raise ValueError('complement solver did not return independent normalized new modes')
    columns, _ = xp.linalg.qr(columns, mode='reduced')
    ac = apply(columns)
    reduced = columns.conj().T @ ac
    new_values, rotation = xp.linalg.eigh((reduced + reduced.conj().T) * .5)
    columns = columns @ rotation
    new_residual = float((xp.linalg.norm(apply(columns) - columns * new_values, axis=0)
                          / xp.maximum(1., xp.abs(new_values))).max())
    if new_residual > threshold:
        raise ValueError('new modes fail the Laplace residual check')
    if float(new_values[0]) < float(old_values[-1]) - order_slack:
        raise ValueError('existing basis is not a complete low-spectrum prefix')
    new = columns.T.reshape(count, *shape)[:missing].copy()
    if fix_phase and missing:
        phase = xp.angle(new.reshape(missing, -1)[:, 0])
        new *= xp.exp(-1j * phase)[:, None, None, None, None]
    vectors = xp.concatenate((prefix, new), axis=0)
    flat = vectors.reshape(total, -1)
    complete_gram = float(xp.linalg.norm(flat.conj() @ flat.T - xp.eye(total), ord=2))
    if complete_gram > threshold:
        raise ValueError('extended basis fails the full orthonormality check')
    return vectors, xp.concatenate((old_values, new_values[:missing])), dict(
        method='preserved-prefix-complement-v1',
        new_phase='first_component_real' if fix_phase else 'solver_phase',
        prefix_count=old, added_count=missing, threshold=threshold,
        prefix_residual=old_residual, new_residual=new_residual, gram_error=complete_gram)
