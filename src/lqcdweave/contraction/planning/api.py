"""Public API for contraction path, state, and storage planning."""

from .target import retarget_expanded

from ..performance.api import CostProfile
from lqcdweave.contracts.correlator.api import PlannedContraction, SceneStoragePlan
from .basis_batch import SceneBasisBatch, combine_basis_problems
from .components import split_independent_terms
from .coordinate_path import lift_coordinate_path
from .coordinate_schedule import (CoordinateReplayPlan, CoordinateDeliverySelection,
                                  plan_coordinate_batches, plan_coordinate_delivery,
                                  supports_coordinate_batches)
from .dense_batches import describe_dense_batch, find_dense_batches
from .dense_batch_schedule import lower_dense_batches
from .gemm_groups import lower_independent_dense_batches, lower_independent_gemm_path
from .factorized_gemm import lower_factorized_gemm_path
from .scalar_reduction_groups import lower_scalar_reduction_groups
from .cartesian_reduction_groups import lower_cartesian_reduction_groups
from .path_planning import (
    NoLegalContractionPath,
    PathSearchExhausted,
    find_contraction_path,
    plan_direct_execution,
    validate_execution_binding,
)
from .paths import (
    PathGraph,
    PathLimitExceeded,
    SearchResult,
    StateGraphLimitExceeded,
    build_path_graph,
    iter_action_paths,
    iter_paths,
    path_count,
    rebind_scene_path,
)
from .resources import audit_resource_trajectory, transition_resource, resource_point
from .sparse_workspace import coordinate_batch_sparse_workspace_bytes, prepare_coordinate_batch_workspace
from .state import ContractionState, has_native_catalog
from .cache import ContractionPlanCache, compile_plan_cached
from .baryon import compile_baryon_terms, compile_baryon_terms_cached
from .compiler import compile_plan
from .matrix_kernel import estimate_matrix_kernel, estimate_matrix_action, describe_matrix_action, replay_matrix_action, plan_matrix_actions
from .cfg_plan import build_cfg_plan, iter_cfg_plan_windows, audit_cfg_plan, validate_cfg_allocation_envelope, configure_cfg_output_layout
from .cfg_coordinate_batches import build_coordinate_cfg_plan
from .batch import solve_storage_batch
from .workload_schedule import list_plan_candidates, plan_workload
from .memory import (
    estimate_plan_memory,
    resolve_budget_limits,
    user_memory_cap,
    workspace_reserve,
)
from .storage import (
    iter_join_storage_batches,
    iter_retain_storage_tensors,
    join_storage_batches,
    logical_cost_profile,
    logical_source_problem,
    plan_scene_storage,
    retain_storage_tensors,
)
from .optimizers import (
    solve_storage,
    SearchSpace,
    LayeredSearchSpace,
    StorageSearchSpace,
    solve_shortest_path,
    solve_layered_search,
    solve_storage_exact,
    StorageSearchResult,
)

__all__ = [
    "lift_coordinate_path",
    "CoordinateReplayPlan", "CoordinateDeliverySelection", "plan_coordinate_batches",
    "plan_coordinate_delivery", "supports_coordinate_batches",
    "retarget_expanded",
    "describe_dense_batch", "find_dense_batches", "lower_dense_batches", "lower_independent_dense_batches", "lower_independent_gemm_path", "lower_factorized_gemm_path",
    "lower_scalar_reduction_groups",
    "lower_cartesian_reduction_groups",
    "build_cfg_plan", "build_coordinate_cfg_plan", "iter_cfg_plan_windows", "audit_cfg_plan", "validate_cfg_allocation_envelope", "configure_cfg_output_layout",
    "estimate_matrix_kernel",
    "estimate_matrix_action",
    "describe_matrix_action", "replay_matrix_action",
    "plan_matrix_actions",
    "audit_resource_trajectory", "transition_resource", "resource_point",
    "coordinate_batch_sparse_workspace_bytes", "prepare_coordinate_batch_workspace",
    "list_plan_candidates",
    "plan_workload",
    "estimate_plan_memory",
    "resolve_budget_limits",
    "user_memory_cap",
    "workspace_reserve",
    "solve_storage",
    "solve_storage_batch",
    "SearchSpace",
    "LayeredSearchSpace",
    "StorageSearchSpace",
    "solve_shortest_path",
    "solve_layered_search",
    "solve_storage_exact",
    "ContractionState",
    "has_native_catalog",
    "ContractionPlanCache",
    "CostProfile",
    "NoLegalContractionPath",
    "PathGraph",
    "PathLimitExceeded",
    "PathSearchExhausted",
    "PlannedContraction",
    "SceneBasisBatch",
    "SceneStoragePlan",
    "SearchResult",
    "StateGraphLimitExceeded",
    "StorageSearchResult",
    "build_path_graph",
    "combine_basis_problems",
    "compile_plan",
    "compile_plan_cached",
    "compile_baryon_terms",
    "compile_baryon_terms_cached",
    "find_contraction_path",
    "iter_action_paths",
    "iter_join_storage_batches",
    "iter_paths",
    "iter_retain_storage_tensors",
    "join_storage_batches",
    "logical_cost_profile",
    "logical_source_problem",
    "path_count",
    "plan_direct_execution",
    "plan_scene_storage",
    "rebind_scene_path",
    "retain_storage_tensors",
    "split_independent_terms",
    "validate_execution_binding",
]
