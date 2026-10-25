"""Read-only planning for overlap datasets backed by registered inputs."""

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest
from lqcdweave.contracts.time_selection import dataset_time_selection
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.overlap import overlap_dataset_document
from lqcdweave.runtime.datasets import apply_source_policy


def _payload_row(basis, points, time, modes, labels):
    basis_selection = dataset_time_selection(basis.spec)
    point_selection = dataset_time_selection(points.spec)
    vectors = basis.read(time=time if basis_selection is None else basis_selection.slot(time), select={"mode": modes})
    coordinates = points.read(time=time if point_selection is None else point_selection.slot(time), select={"point": labels})
    if not np.all(np.isfinite(vectors)):
        raise ValueError("selected basis contains nonfinite values")
    sizes = np.asarray(basis.spec.document["configuration"]["lattice"][:3])
    if (coordinates.dtype.kind not in "iu" or coordinates.shape != (len(labels), 3)
            or np.any(coordinates < 0) or np.any(coordinates >= sizes)
            or len(np.unique(coordinates, axis=0)) != len(labels)):
        raise ValueError("selected point coordinates must be integer, in bounds and unique per time")
    return vectors, coordinates


def _requests(index, basis_id, points_id, usedNe, usedNp, selection=None):
    if any(not isinstance(value, str) or not value for value in (basis_id, points_id)):
        raise ValueError("basis_id and points_id must be actual registered IDs")
    with index.open(basis_id) as basis, index.open(points_id) as points:
        basis_view, points_view, document, times = overlap_dataset_document(
            basis.spec, points.spec, basis_id={"request": "basis"},
            points_id={"request": "points"}, usedNe=usedNe, usedNp=usedNp, selection=selection)
        # Inspect the whole requested domain before any writer exists, including
        # time rows that a later resume/reuse might otherwise avoid reading.
        for time in times:
            _payload_row(basis, points, time, basis_view.coordinates("mode"),
                         points_view.coordinates("point"))
    return dict(basis=DatasetRequest(basis_view, data_id=basis_id, subset_axes=("mode",)),
                points=DatasetRequest(points_view, data_id=points_id, subset_axes=("point",)),
                overlap=DatasetRequest(document, subset_axes=("mode", "point"))), times


def _preview(index, requests, data_ids, source_policy):
    graph = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(graph, requests, index, source_policy=source_policy)


def plan_indexed_overlap(index, basis_id, points_id, *, usedNe=None, usedNp=None,
                         data_ids=None, source_policy="allow_unknown", selection=None):
    """Verify requested input rows and preview without gathering or writing.

    The preview reads requested rows to check their numerical/domain
    constraints. It is independent of source knownness.
    """
    if source_policy not in ("allow_unknown", "require_known"):
        raise ValueError("source_policy must be allow_unknown or require_known")
    requests, times = _requests(index, basis_id, points_id, usedNe, usedNp, selection=selection)
    graph = _preview(index, requests, data_ids, source_policy)
    blocked = (graph.get("action") == "blocked" or
               any(stage["action"] == "blocked" for stage in graph["stages"].values()))
    return dict(requests=requests, graph=graph,
                dataset_spec=(None if blocked else
                               graph["stages"]["overlap"].get("spec")), times=times)


__all__ = ["plan_indexed_overlap"]
