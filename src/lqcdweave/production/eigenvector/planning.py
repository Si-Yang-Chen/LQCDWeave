"""Read-only Laplace dataset planning."""

from numbers import Real

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest
from lqcdweave.contracts.time_selection import normalize_time_selection
from lqcdweave.contracts.elemental import integer
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.planning.production import validate_gauge_field
from lqcdweave.runtime.datasets import apply_source_policy


def _checked_tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("tolerance must be a finite real number in (0,1)")
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError("tolerance must be a finite real number in (0,1)")
    return result


def _indexed_laplace_requests(index, gauge_id, *, Ne, tolerance, selection=None):
    ne = integer(Ne, "Ne", 1)
    tol = _checked_tolerance(tolerance)
    with index.open(gauge_id) as gauge:
        times = validate_gauge_field(gauge.spec)
        if gauge.spec.document["conventions"].get("boundary") != "periodic":
            raise ValueError("Laplace production requires an explicitly periodic gauge boundary")
        cfg = gauge.spec.document["configuration"]
        if selection is not None:
            times = normalize_time_selection(selection, Lt=cfg["lattice"][3]).times
        lx, ly, lz, _ = cfg["lattice"]
        spatial_dimension = 3 * lx * ly * lz
        if ne > spatial_dimension - 2:
            raise ValueError("Ne must be at most 3*Lx*Ly*Lz - 2")
        reference_gauge = dict(request="gauge")
        reference_basis = dict(request="basis")
        basis_document = laplace_basis_document(
            cfg, times, reference_gauge, ne, tol, selection=selection)
        requests = {
            "gauge": DatasetRequest(gauge.spec, data_id=gauge_id),
            "basis": DatasetRequest(basis_document,
                subset_axes=("time", "mode")),
            "eigenvalues": DatasetRequest(laplace_eigenvalues_document(
                cfg, times, ne, reference_basis, reference_gauge, basis_document),
                subset_axes=("time", "mode")),
        }
    return requests, tuple(times), cfg, ne, tol


def _preview(index, requests, *, data_ids, source_policy):
    preview = plan_dataset_graph(
        requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def plan_indexed_laplace_basis(index, gauge_id, *, Ne=2, tolerance=1e-11,
                               data_ids=None, source_policy="allow_unknown", selection=None):
    """Preview the full-time gauge -> Laplace basis -> eigenvalues graph.

    The planner reads registered metadata and may inspect input arrays, but it
    never runs a numerical solve or creates output directories.
    """
    requests, _, _, _, _ = _indexed_laplace_requests(
        index, gauge_id, Ne=Ne, tolerance=tolerance, selection=selection)
    return _preview(index, requests, data_ids=data_ids,
                    source_policy=source_policy)


__all__ = [
    "plan_indexed_laplace_basis",
]
