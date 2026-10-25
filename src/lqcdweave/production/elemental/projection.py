"""Shared spatial projection: combine equal endpoints before acting on modes."""

import numpy as np

from .transport import path_transport, apply_transport


_TRANSPORT_CACHE_BYTES = 512 * 1024**2
_WEIGHT_BYTES = 256 * 1024**2


def project_rows(engine, parsed, t, out, *, write=None):
    """Evaluate already-normalized spatial rows without raw-path mode fields.

    Only paths with identical *unwrapped* displacement share a transport sum;
    winding paths retain their distinct midpoint phases. Numerical kernels are
    local to this call, so load/smear/time changes never reuse stale links.
    """
    xp = engine.xp
    groups, destinations = [], {}
    for ri, recipe in enumerate(parsed):
        row = {}
        for path, coefficient in recipe.items():
            if coefficient == 0:
                continue
            displacement = path.displacement[:3]
            row.setdefault(displacement, []).append((path, coefficient))
            destinations.setdefault(path, []).append((ri, displacement, coefficient))
        groups.append(row)

    shape = engine._workspace.links.shape[2:]
    kernel_bytes = int(np.prod(shape)) * np.dtype(np.complex128).itemsize
    cached = {}
    if kernel_bytes * sum(map(len, groups)) <= _TRANSPORT_CACHE_BYTES:
        for path, targets in destinations.items():
            transport = path_transport(engine._workspace.links, path, t,
                                       engine.path_boundary, array_module=xp)
            for ri, displacement, coefficient in targets:
                key = ri, displacement
                if key not in cached:
                    cached[key] = coefficient * transport
                else:
                    cached[key] += coefficient * transport

    def transport_sum(ri, displacement, terms):
        value = cached.get((ri, displacement))
        if value is not None:
            return value
        # Very wide tables retain a bounded streaming fallback.
        value = xp.zeros(shape, dtype=xp.complex128)
        for path, coefficient in terms:
            value += coefficient * path_transport(engine._workspace.links, path, t,
                                                  engine.path_boundary, array_module=xp)
        return value

    ne, nmom = engine.usedNe, engine.num_momentum
    vectors = engine._v(t, slice(0, ne))
    left = vectors.reshape(ne, -1).conj().T
    field_bytes = vectors.size * np.dtype(np.complex128).itemsize
    # Dense delivery can fill the existing output a few right modes at a time.
    # This admits more momenta per transported field without changing callback
    # granularity or retaining all endpoint fields for large spatial volumes.
    mode_bytes = max(1, (field_bytes + ne - 1) // ne)
    modebatch = (ne if write is not None else
                 min(ne, max(1, _WEIGHT_BYTES // (nmom * mode_bytes))))
    tile_bytes = max(1, (field_bytes * modebatch + ne - 1) // ne)
    budget = max(1, _WEIGHT_BYTES // tile_bytes)
    mombatch = min(nmom, budget)
    rowbatch = max(1, min(len(parsed), budget // mombatch))
    phase_convention = 'none' if engine.phase_convention == 'none' else 'anchor'
    cache_spare = max(0, _TRANSPORT_CACHE_BYTES - kernel_bytes * len(cached))
    phase_bytes = nmom * int(np.prod(vectors.shape[1:-1])) * 16
    phases_all = None
    # The list-to-stack construction temporarily needs both copies. Callback
    # delivery keeps its established streaming phase lifetime and error order.
    if (write is None and (modebatch < ne or rowbatch < len(parsed))
            and 2 * phase_bytes <= cache_spare):
        phases_all = xp.stack([engine._phase(momentum, convention=phase_convention)
                               for momentum in engine.momentum_list])
        cache_spare -= phase_bytes
    for rfirst in range(0, len(parsed), rowbatch):
        rstop = min(rfirst + rowbatch, len(parsed))
        for kfirst in range(0, ne, modebatch):
            kstop = min(kfirst + modebatch, ne)
            tile = vectors[kfirst:kstop]
            transported_cache = {}
            remaining = cache_spare
            for mfirst in range(0, nmom, mombatch):
                mstop = min(mfirst + mombatch, nmom)
                momenta = engine.momentum_list[mfirst:mstop]
                phases = (phases_all[mfirst:mstop] if phases_all is not None else
                          xp.stack([engine._phase(momentum, convention=phase_convention)
                                    for momentum in momenta]))
                weights = xp.zeros((rstop-rfirst, mstop-mfirst, *tile.shape), complex)
                for ri in range(rfirst, rstop):
                    for displacement, terms in groups[ri].items():
                        key = ri, displacement
                        transported = transported_cache.get(key)
                        if transported is None:
                            kernel = transport_sum(ri, displacement, terms)
                            transported = apply_transport(kernel, tile, displacement, array_module=xp)
                            if mombatch < nmom and tile_bytes <= remaining:
                                transported_cache[key] = transported
                                remaining -= tile_bytes
                        if engine.phase_convention == 'midpoint':
                            half = np.exp(1j*np.pi*(np.asarray(momenta)
                                @ (np.asarray(displacement)/np.asarray(engine.latt_size[:3]))))
                        else:
                            half = np.ones(mstop-mfirst)
                        # Host constants avoid scalar device-to-host syncs.
                        for mi, coefficient in enumerate(half):
                            weights[ri-rfirst, mi] += coefficient * transported
                weights *= phases[None, :, None, :, :, :, None]
                contracted = weights.reshape((rstop-rfirst)*(mstop-mfirst)*(kstop-kfirst), -1) @ left
                values = contracted.reshape(rstop-rfirst, mstop-mfirst, kstop-kfirst, ne).swapaxes(-1, -2)
                if engine.mode_pair_weights is not None:
                    values *= engine.mode_pair_weights[:ne, kfirst:kstop]
                if write is None:
                    out[rfirst:rstop, mfirst:mstop, :, kfirst:kstop] = values
                else:
                    for ri in range(rfirst, rstop):
                        for mi in range(mfirst, mstop):
                            write(ri, mi, 'v2v', (slice(None), slice(None)),
                                  values[ri-rfirst, mi-mfirst])
    return out
