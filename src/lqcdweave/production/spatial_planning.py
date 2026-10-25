"""Plan caller point realizations and reproducible local samples."""

from lqcdweave.contracts.dataset import DatasetRequest
from lqcdweave.contracts.time_selection import dataset_physical_times
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.spatial_points import sampled_spatial_points_spec
from lqcdweave.runtime.datasets import apply_source_policy


def _source_preview(graph, requests, index, source_policy):
    return apply_source_policy(graph, requests, index, source_policy=source_policy)


def _plan_single_request(index, spec, *, data_ids=None):
    requests = {"points": DatasetRequest(spec)}
    graph = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return requests, graph


def plan_indexed_spatial_points(index, design, *, configuration, seed,
                                data_ids=None, source_policy="allow_unknown", selection=None):
    """Preview the dataset definition for a sampled spatial-point set."""
    spec = sampled_spatial_points_spec(design, configuration=configuration, seed=seed, selection=selection)
    requests, graph = _plan_single_request(index, spec, data_ids=data_ids)
    graph = _source_preview(graph, requests, index, source_policy)
    stage = graph["stages"]["points"]
    return dict(requests=requests, graph=graph,
        dataset_spec=stage["spec"], times=dataset_physical_times(spec),
        design=design, seed=spec.document["definition"]["realization"]["seed"],
        source_policy=graph.get("source_policy", source_policy),
        source_status=graph.get("source_status", "unknown"))
