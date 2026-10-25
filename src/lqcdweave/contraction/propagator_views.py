"""Explicit in-memory views derived from propagator blocks.

The legacy PSV dagger name forwards to PR's unique Gamma5 implementation.
Highmode projection remains here; it does not produce or store PR datasets.
"""
import numpy as np
from lqcdweave.production.perambulator.api import psv_to_vsp_dagger


_HIGHMODE_COMPLEX_DTYPES = (np.dtype("complex64"), np.dtype("complex128"))


def _highmode_arrays(array_module, names, values, ranks):
    """Validate one fixed-pair input set and promote it to one explicit dtype."""
    required = ("asarray", "array", "all", "isfinite", "einsum")
    if any(not callable(getattr(array_module, name, None)) for name in required):
        raise TypeError("array_module must provide NumPy-compatible array operations")

    shapes = []
    dtypes = []
    for name, value, rank in zip(names, values, ranks):
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != rank:
            raise ValueError(f"{name} must have rank {rank}")
        try:
            dtype = np.dtype(value.dtype)
        except (AttributeError, TypeError) as exc:
            raise TypeError(f"{name} must be a complex array") from exc
        if dtype not in _HIGHMODE_COMPLEX_DTYPES:
            raise TypeError(f"{name} must have complex64 or complex128 dtype")
        shape = tuple(int(size) for size in shape)
        if any(size <= 0 for size in shape):
            raise ValueError(f"{name} extents must be positive")
        shapes.append(shape)
        dtypes.append(dtype)

    for name, value in zip(names, values):
        if not bool(array_module.all(array_module.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")

    dtype = np.dtype(np.result_type(*dtypes))
    arrays = tuple(array_module.asarray(value, dtype=dtype) for value in values)
    return tuple(shapes), arrays, dtype


def _highmode_result(value, dtype, array_module, shape):
    if tuple(int(size) for size in value.shape) != shape:
        raise ValueError("projected block has an unexpected shape")
    if not bool(array_module.all(array_module.isfinite(value))):
        raise ValueError("projected block must contain only finite values")
    return array_module.array(value, dtype=dtype, copy=True)


def _project_vsp_cast(vsv, vsp, source_overlap, *, array_module):
    """Source-side contraction shared by VSP and the two-sided PSP path."""
    # Adapted from localized_blending/source/lattice/propagators.py,
    # get_VSP_highmode single-time branch (lines 1925-1932):
    # contract("abij,jxc->abixc", S_vsv, M_conj_t); tilde_S = S_vsp - correction.
    correction = array_module.einsum(
        "abij,jpc->abipc", vsv, source_overlap.conj(),
    )
    return vsp - correction


def project_vsp_highmode(vsv, vsp, source_overlap, *, array_module=np):
    """Apply the source/right orthonormal low-mode complement to one VSP pair.

    Inputs are VSV ``(4,4,Nv_sink,Nv_source)``, raw VSP
    ``(4,4,Nv_sink,Np_source,3)``, and source overlap
    ``(Nv_source,Np_source,3)``. The returned owned array has the VSP shape.
    Endpoint identity and orthonormality are caller/planner responsibilities.
    """
    names = ("vsv", "vsp", "source_overlap")
    shapes, arrays, dtype = _highmode_arrays(
        array_module, names, (vsv, vsp, source_overlap), (4, 5, 3),
    )
    vsv_shape, vsp_shape, overlap_shape = shapes
    nv_sink, nv_source = vsv_shape[2:]
    if vsv_shape[:2] != (4, 4):
        raise ValueError("vsv must have spin axes of extent four")
    if (vsp_shape[:3] != (4, 4, nv_sink) or vsp_shape[-1] != 3
            or overlap_shape != (nv_source, vsp_shape[3], 3)):
        raise ValueError("VSV, VSP and source-overlap axes are incompatible")
    vsv_value, vsp_value, overlap_value = arrays
    projected = _project_vsp_cast(
        vsv_value, vsp_value, overlap_value, array_module=array_module,
    )
    return _highmode_result(projected, dtype, array_module, vsp_shape)


def project_psv_highmode(vsv, psv, sink_overlap, *, array_module=np):
    """Apply the sink/left orthonormal low-mode complement to one PSV pair.

    Inputs are VSV ``(4,4,Nv_sink,Nv_source)``, raw PSV
    ``(4,4,Np_sink,3,Nv_source)``, and sink overlap
    ``(Nv_sink,Np_sink,3)``. The returned owned array has the PSV shape.
    Endpoint identity and orthonormality are caller/planner responsibilities.
    """
    names = ("vsv", "psv", "sink_overlap")
    shapes, arrays, dtype = _highmode_arrays(
        array_module, names, (vsv, psv, sink_overlap), (4, 5, 3),
    )
    vsv_shape, psv_shape, overlap_shape = shapes
    nv_sink, nv_source = vsv_shape[2:]
    if vsv_shape[:2] != (4, 4):
        raise ValueError("vsv must have spin axes of extent four")
    if (psv_shape[:2] != (4, 4) or psv_shape[3] != 3
            or psv_shape[4] != nv_source
            or overlap_shape != (nv_sink, psv_shape[2], 3)):
        raise ValueError("VSV, PSV and sink-overlap axes are incompatible")
    vsv_value, psv_value, overlap_value = arrays
    # Adapted from localized_blending/source/lattice/propagators.py,
    # get_PSV_highmode single-time branch (lines 2098-2105):
    # contract("jxc,abji->abxci", M_t, S_vsv); tilde_S = S_psv - correction.
    correction = array_module.einsum(
        "ipc,abij->abpcj", overlap_value, vsv_value,
    )
    projected = psv_value - correction
    return _highmode_result(projected, dtype, array_module, psv_shape)


def project_psp_highmode(
    vsv, psv, vsp, psp, sink_overlap, source_overlap, *, array_module=np,
):
    """Apply both endpoint complements to one PSP physical-time pair.

    VSV, PSV, VSP, PSP and overlap shapes are respectively
    ``(4,4,Nv_sink,Nv_source)``, ``(4,4,Np_sink,3,Nv_source)``,
    ``(4,4,Nv_sink,Np_source,3)``, ``(4,4,Np_sink,3,Np_source,3)``,
    ``(Nv_sink,Np_sink,3)``, and ``(Nv_source,Np_source,3)``. The result is
    owned with the PSP shape. The right-side correction uses the source overlap
    conjugate; the left-side term uses source-projected VSP, retaining the
    positive two-sided cross term in ``Q_sink S Q_source``.
    """
    names = (
        "vsv", "psv", "vsp", "psp", "sink_overlap", "source_overlap",
    )
    ranks = (4, 5, 5, 6, 3, 3)
    shapes, arrays, dtype = _highmode_arrays(
        array_module, names,
        (vsv, psv, vsp, psp, sink_overlap, source_overlap), ranks,
    )
    (vsv_shape, psv_shape, vsp_shape, psp_shape,
     sink_shape, source_shape) = shapes
    nv_sink, nv_source = vsv_shape[2:]
    if vsv_shape[:2] != (4, 4):
        raise ValueError("vsv must have spin axes of extent four")
    np_sink = psv_shape[2]
    np_source = vsp_shape[3]
    if (psv_shape[:4] != (4, 4, np_sink, 3)
            or psv_shape[4] != nv_source
            or vsp_shape[:3] != (4, 4, nv_sink) or vsp_shape[4] != 3
            or psp_shape != (4, 4, np_sink, 3, np_source, 3)
            or sink_shape != (nv_sink, np_sink, 3)
            or source_shape != (nv_source, np_source, 3)):
        raise ValueError("VSV, PSV, VSP, PSP and overlap axes are incompatible")

    vsv_value, psv_value, vsp_value, psp_value, sink_value, source_value = arrays
    vsp_high = _project_vsp_cast(
        vsv_value, vsp_value, source_value, array_module=array_module,
    )
    # Adapted from localized_blending/source/lattice/propagators.py,
    # _compute_PSP_highmode single-time branch (lines 2320-2352):
    # term2 = contract("ixc,abiyd->abxcyd", M_sink_t, S_vsp_tilde)
    # term3 = contract("abxcj,jyd->abxcyd", S_psv, M_source_t)
    term2 = array_module.einsum(
        "ipc,abiqd->abpcqd", sink_value, vsp_high,
    )
    term3 = array_module.einsum(
        "abpcj,jqd->abpcqd", psv_value, source_value.conj(),
    )
    projected = psp_value - term2 - term3
    return _highmode_result(projected, dtype, array_module, psp_shape)
