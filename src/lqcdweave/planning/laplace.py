"""Pure records and resource estimates for registered Laplace products."""

import math
import numbers

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import normalize_time_selection, bind_time_selections, dataset_time_selection
from lqcdweave.contracts.memory import MemoryEstimate


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


def _spec_document(configuration, kind, axes, *, dtype="complex128", bases=None,
                   definition=None, conventions=None, parents=()):
    return dict(schema=1, kind=kind, configuration=configuration, axes=axes,
        dtype=dtype, bases=bases or {}, definition=definition or {},
        conventions=conventions or {}, parents=list(parents))


def laplace_basis_document(cfg, times, gauge_id, ne, tolerance, *, selection=None):
    """Describe the physical-time basis using the established Wilson contract.

    Logical DatasetRequest references may be supplied for ``gauge_id``; the
    request graph resolves them to actual IDs before a DatasetSpec is used.
    """
    lx, ly, lz, lt = cfg["lattice"]
    if selection is not None:
        selection = normalize_time_selection(selection, Lt=lt)
        times = selection.times
    document = _spec_document(cfg, "laplace_basis", [
        _axis("time", times), _axis("mode", range(ne)),
        _axis("z", range(lz)), _axis("y", range(ly)), _axis("x", range(lx)),
        _axis("color", range(3)),
    ], definition=dict(gauge_id=gauge_id,
        operator="positive_spatial_laplacian", solver_tolerance=tolerance),
        conventions=dict(normalization="orthonormal", spatial_boundary="periodic",
                         phase="first_component_real"),
        parents=[gauge_id])
    return document if selection is None else bind_time_selections(document, {"time": selection})


def laplace_eigenvalues_document(cfg, times, ne, basis_id, gauge_id,
                                 basis_document):
    """Describe eigenvalues bound to the realized ordered basis prefix."""
    binding = dict(data_id=basis_id, indices=list(range(ne)))
    selection = dataset_time_selection(basis_document)
    if selection is not None:
        times = selection.times
    document = _spec_document(cfg, "laplace_eigenvalues", [
        _axis("time", times), _axis("mode", range(ne)),
    ], dtype="float64", bases=dict(mode=binding),
        definition=basis_document["definition"], parents=[basis_id, gauge_id])
    return document if selection is None else bind_time_selections(document, {"time": selection})


def estimate_laplace_memory(gauge_spec, ne, *, target,
                            workspace_reserve_bytes=0,
                            host_staging_bytes=0, selection=None):
    """Estimate logical host/device peaks for one complete Laplace run.

    The returned pair contains the shared C MemoryEstimate type.  The model
    includes the full registered gauge, the loader's raw and owned link arrays,
    known eigensolver result/copy arrays, spatial-operator temporaries, cached
    eigenvalue rows, host output rows and finite masks. ARPACK, CuPyX, QUDA and other backend/native workspaces
    are opaque and must be covered by the caller's reserve.  This is a logical
    allocation model, not an RSS or native-library hard bound.

    This model covers ordinary independent solves, not preserved-prefix
    extension's additional projection, Gram and QR workspaces.

    A GPU target conservatively charges the opaque workspace to both domains;
    host staging is always charged to host.  The caller is responsible for
    supplying reserves that cover its backend and storage path.
    """
    if not isinstance(gauge_spec, DatasetSpec):
        raise TypeError("gauge_spec must be a DatasetSpec")
    if type(ne) is not int or ne < 1:
        raise ValueError("ne must be a positive integer")
    if target not in ("cpu", "gpu"):
        raise ValueError("target must be 'cpu' or 'gpu'")
    workspace_reserve_bytes = _nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _nonnegative_int(host_staging_bytes, "host_staging_bytes")

    document = gauge_spec.document
    if document["kind"] != "gauge":
        raise ValueError("gauge_spec must describe a gauge")
    lattice = document["configuration"]["lattice"]
    lx, ly, lz, lt = lattice
    expected_shape = (lt, lz, ly, lx, 4, 3, 3)
    if gauge_spec.shape != expected_shape:
        raise ValueError("gauge_spec must have complete canonical gauge extents")
    input_itemsize = {"complex64": 8, "complex128": 16}.get(document["dtype"])
    if input_itemsize is None:
        raise ValueError("gauge_spec dtype must be complex64 or complex128")
    nt = lt if selection is None else normalize_time_selection(selection, Lt=lt).Nt
    volume = lx * ly * lz
    if ne > 3 * volume - 2:
        raise ValueError("ne must be at most 3*Lx*Ly*Lz - 2")

    gauge_bytes = math.prod(expected_shape) * input_itemsize
    links_bytes = 4 * lt * volume * 3 * 3 * 16
    basis_row_bytes = ne * volume * 3 * 16
    value_row_bytes = ne * 8
    value_cache_bytes = nt * value_row_bytes
    prior_value_cache_bytes = max(0, nt - 1) * value_row_bytes
    basis_elements = ne * volume * 3
    basis_mask_bytes = basis_elements
    value_mask_bytes = ne
    writer_scratch_bytes = max(basis_mask_bytes, value_mask_bytes)

    # The matrix-free operator receives one or more color vectors.  Its
    # explicit vector, roll/einsum and backward-link conjugate arrays have
    # shape-derived sizes; solver-owned Krylov state remains opaque.
    vector_field_bytes = volume * 3 * 16
    spatial_dagger_links_bytes = 3 * volume * 3 * 3 * 16
    operator_rhs_bytes = max(1, ne) * vector_field_bytes
    operator_temporaries = 6 * operator_rhs_bytes + spatial_dagger_links_bytes
    # eigsh returns columns; the core reshapes, indexes by eigenvalue order,
    # and makes an owned complex128 result.  Three complete result-sized arrays
    # conservatively cover these known overlapping conversions.
    sorted_result_temporaries = 3 * basis_row_bytes + 3 * value_row_bytes
    overlap_temporaries = basis_row_bytes + ne * 16

    host_workspace = workspace_reserve_bytes
    device_workspace = workspace_reserve_bytes if target == "gpu" else 0

    def estimate(*, input_bytes=0, snapshot_bytes=0, intermediate_bytes=0,
                 output_bytes=0, assembly_bytes=0, workspace_bytes=0):
        return MemoryEstimate(
            input_bytes=input_bytes,
            snapshot_bytes=snapshot_bytes,
            intermediate_bytes=intermediate_bytes,
            output_bytes=output_bytes,
            assembly_extra_bytes=assembly_bytes,
            workspace_bytes=workspace_bytes,
        )

    host_phases = [
        # Reader and loader conversion phases.
        estimate(input_bytes=gauge_bytes,
                 snapshot_bytes=gauge_bytes + (links_bytes if target == "cpu" else 0),
                 assembly_bytes=host_staging_bytes,
                 workspace_bytes=host_workspace),
        # Matrix-free solve; previous eigenvalue rows remain cached while a
        # later time slice is solved.
        estimate(input_bytes=gauge_bytes,
                 snapshot_bytes=(links_bytes if target == "cpu" else 0) +
                                prior_value_cache_bytes,
                 intermediate_bytes=operator_temporaries,
                 assembly_bytes=host_staging_bytes,
                 workspace_bytes=host_workspace),
        estimate(input_bytes=gauge_bytes,
                 snapshot_bytes=(links_bytes if target == "cpu" else 0) +
                                prior_value_cache_bytes,
                 intermediate_bytes=sorted_result_temporaries,
                 assembly_bytes=host_staging_bytes,
                 workspace_bytes=host_workspace),
        # Basis calculation and host copy overlap; cache append follows the
        # synchronous writer call and the current locals are explicitly freed.
        estimate(input_bytes=gauge_bytes,
                 snapshot_bytes=(links_bytes if target == "cpu" else 0) +
                                prior_value_cache_bytes,
                 output_bytes=2 * basis_row_bytes + 2 * value_row_bytes,
                 assembly_bytes=host_staging_bytes + writer_scratch_bytes,
                 workspace_bytes=host_workspace),
        estimate(input_bytes=gauge_bytes,
                 snapshot_bytes=(links_bytes if target == "cpu" else 0) +
                                prior_value_cache_bytes,
                 output_bytes=2 * basis_row_bytes + 3 * value_row_bytes,
                 assembly_bytes=host_staging_bytes,
                 workspace_bytes=host_workspace),
        # The eigenvalue producer opens the complete saved basis view.  Normal
        # cache reuse and the uncached recomputation branch are both included.
        estimate(input_bytes=gauge_bytes + nt * basis_row_bytes,
                 snapshot_bytes=value_cache_bytes,
                 output_bytes=value_row_bytes,
                 assembly_bytes=host_staging_bytes + writer_scratch_bytes,
                 workspace_bytes=host_workspace),
        estimate(input_bytes=gauge_bytes + nt * basis_row_bytes,
                 snapshot_bytes=(links_bytes if target == "cpu" else 0) +
                                prior_value_cache_bytes,
                 intermediate_bytes=(sorted_result_temporaries +
                                     overlap_temporaries),
                 output_bytes=basis_row_bytes + 2 * value_row_bytes,
                 assembly_bytes=host_staging_bytes + writer_scratch_bytes,
                 workspace_bytes=host_workspace),
    ]
    host_peak = max(host_phases, key=lambda item: item.logical_peak_bytes)

    if target == "cpu":
        return host_peak, MemoryEstimate()

    device_phases = [
        estimate(input_bytes=gauge_bytes, snapshot_bytes=gauge_bytes + links_bytes,
                 workspace_bytes=device_workspace),
        estimate(input_bytes=links_bytes,
                 intermediate_bytes=operator_temporaries,
                 workspace_bytes=device_workspace),
        estimate(input_bytes=links_bytes,
                 intermediate_bytes=sorted_result_temporaries,
                 workspace_bytes=device_workspace),
        estimate(input_bytes=links_bytes,
                 output_bytes=basis_row_bytes + value_row_bytes,
                 workspace_bytes=device_workspace),
        estimate(input_bytes=links_bytes,
                 output_bytes=value_row_bytes,
                 workspace_bytes=device_workspace),
    ]
    device_peak = max(device_phases, key=lambda item: item.logical_peak_bytes)
    return host_peak, device_peak


__all__ = [
    "estimate_laplace_memory",
    "laplace_basis_document",
    "laplace_eigenvalues_document",
]
