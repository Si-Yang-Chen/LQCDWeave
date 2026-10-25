"""Connected equal-time meson Wick contraction with explicit reverse data."""

import numpy as np


def connected_meson_pair(sink_factors, source_factors, forward, reverse):
    """Return -Tr[Phi_sink tau_f(t,s) Phi_source tau_r(s,t)].

    Phi_source is already the source vertex (including its dagger convention),
    not a sink vertex to conjugate here. tau axes are (spin,spin,mode,mode).
    The minus sign is the connected Wick loop sign; there is no extra source
    average, volume normalization, flavor coefficient, or gamma5 reconstruction.
    Caller validates identities, time endpoints and coverage before invoking.
    """
    forward = np.asarray(forward, dtype=np.complex128)
    reverse = np.asarray(reverse, dtype=np.complex128)
    if (forward.ndim != 4 or forward.shape[:2] != (4, 4) or
            reverse.shape != forward.shape or forward.shape[2] != forward.shape[3]):
        raise ValueError("expected square common-basis propagators (4,4,Ne,Ne)")
    sink_factors, source_factors = tuple(sink_factors), tuple(source_factors)
    if not sink_factors or not source_factors:
        raise ValueError("nonempty source and sink vertices are required")
    result = 0j
    for sink in sink_factors:
        for source in source_factors:
            if (sink.sector != "v2v" or source.sector != "v2v" or
                    np.shape(sink.spin) != (4, 4) or np.shape(source.spin) != (4, 4) or
                    np.shape(sink.spatial) != forward.shape[2:] or
                    np.shape(source.spatial) != forward.shape[2:]):
                raise ValueError("incompatible meson vertex block")
            result -= sink.coefficient * source.coefficient * np.einsum(
                "ab,ij,bcjk,cd,kl,dali->", sink.spin, sink.spatial,
                forward, source.spin, source.spatial, reverse,
                optimize=True, dtype=np.complex128)
    return complex(result)
