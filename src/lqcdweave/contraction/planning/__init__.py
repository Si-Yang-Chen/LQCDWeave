"""Lazy package facade; implementations stay importable without cycles."""

from importlib import import_module

__all__ = [
    "SearchSpace", "LayeredSearchSpace", "StorageSearchSpace", "solve_shortest_path",
    "solve_layered_search", "solve_storage_exact", "solve_storage",
    "ContractionPlanCache", "ContractionState", "CostProfile",
    "NoLegalContractionPath", "PathGraph", "PathLimitExceeded", "PathSearchExhausted",
    "PlannedContraction", "SceneBasisBatch", "SceneStoragePlan", "SearchResult",
    "StateGraphLimitExceeded", "StorageSearchResult", "build_path_graph",
    "combine_basis_problems", "compile_plan", "compile_plan_cached", "find_contraction_path", "iter_action_paths",
    "iter_join_storage_batches", "iter_paths", "iter_retain_storage_tensors",
    "join_storage_batches", "logical_cost_profile", "logical_source_problem", "path_count",
    "plan_direct_execution", "plan_scene_storage",
    "rebind_scene_path", "retain_storage_tensors",
    "solve_storage_batch",
    "split_independent_terms", "validate_execution_binding",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".api", __name__), name)
    globals()[name] = value
    return value
