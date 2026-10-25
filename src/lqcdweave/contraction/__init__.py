"""Public contraction facade; module implementations load only when requested."""
from importlib import import_module
from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    TensorSpec,
    ContractionPlan,
    ExecutionResult,
    ExecutionStats,
    ContractionProblem,
    SourceSpec,
    StorageBlock,
    SourcePreparation,
    SceneStoragePlan,
)

_PUBLIC = {
    "solve_storage": "lqcdweave.contraction.planning.api",
    "ContractionPlanCache": "lqcdweave.contraction.planning.api",
    "CostProfile": "lqcdweave.contraction.performance.api",
    "ContractionState": "lqcdweave.contraction.planning.api",
    "SceneBasisBatch": "lqcdweave.contraction.planning.api",
    "combine_basis_problems": "lqcdweave.contraction.planning.api",
    "rebind_scene_path": "lqcdweave.contraction.planning.api",
    "logical_source_problem": "lqcdweave.contraction.planning.api",
    "logical_cost_profile": "lqcdweave.contraction.planning.api",
    "plan_scene_storage": "lqcdweave.contraction.planning.api",
    "join_storage_batches": "lqcdweave.contraction.planning.api",
    "retain_storage_tensors": "lqcdweave.contraction.planning.api",
    "iter_join_storage_batches": "lqcdweave.contraction.planning.api",
    "iter_retain_storage_tensors": "lqcdweave.contraction.planning.api",
    "solve_storage_batch": "lqcdweave.contraction.planning.api",
    "StorageSearchResult": "lqcdweave.contraction.planning.api",
    "compile_plan": "lqcdweave.contraction.planning.api",
    "execute_plan": "lqcdweave.contraction.execution.api",
    "BaryonTerm": "lqcdweave.contracts.correlator.api",
    "BaryonVertex": "lqcdweave.contracts.correlator.api",
    "compile_baryon_terms": "lqcdweave.contraction.planning.api",
    "execute_baryon_terms": "lqcdweave.contraction.contraction_session",
    "replay_scene_storage": "lqcdweave.contraction.replay",
    "replay_storage_batches": "lqcdweave.contraction.replay",
}

def __getattr__(name):
    try:
        module = _PUBLIC[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value

__all__ = [
    "solve_storage",
    "ContractionTerm",
    "TensorSpec",
    "ContractionPlan",
    "ExecutionResult",
    "ExecutionStats",
    "ContractionProblem",
    "SourceSpec",
    "StorageBlock",
    "SourcePreparation",
    "SceneStoragePlan",
    "ContractionPlanCache",
    "CostProfile",
    "ContractionState",
    "SceneBasisBatch",
    "combine_basis_problems",
    "rebind_scene_path",
    "logical_source_problem",
    "logical_cost_profile",
    "plan_scene_storage",
    "join_storage_batches",
    "retain_storage_tensors",
    "iter_join_storage_batches",
    "iter_retain_storage_tensors",
    "solve_storage_batch",
    "StorageSearchResult",
    "compile_plan",
    "execute_plan",
    "BaryonTerm",
    "BaryonVertex",
    "compile_baryon_terms",
    "execute_baryon_terms",
    "replay_scene_storage",
    "replay_storage_batches",
]
