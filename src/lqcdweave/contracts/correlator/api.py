"""Public numerical-free correlator contracts; unique definitions only."""
from .allocation import DEVICE_ALLOCATION_ALIGNMENT, device_allocation_bytes, device_allocator_headroom
from .vertex_kernel import VertexKernelDescription
from .meson_core import MesonWholeCore, MesonWholeKernel
from .time import normalize_time_key, time_key_order
from .numerical_result import CorrelatorResult
from .target import ExecutionTarget
from .baryon import BaryonLine, BaryonTerm, BaryonTopology, BaryonVertex
from .indices import AxisPattern, axis_ids, parse_axis_pattern
from .dense_batch import (
    DenseBatchOperand, DenseBatchResult, DenseKernelBatch, dense_batch_memory, dense_batch_producer_eligible,
)
from .gemm_group import (DenseGemmGroup, dense_batch_leaves, dense_batch_matrix_shape,
                         GEMM_GROUP_MAX_BATCHES, GEMM_GROUP_POINTER_BYTES)
from .scalar_reduction_group import (
    ScalarReductionGroup, ScalarReductionGeometry, scalar_reduction_group_limit, scalar_reduction_geometry,
    describe_scalar_reduction_group, scalar_reduction_group_memory, validate_scalar_reduction_groups,
)
from .cartesian_reduction_group import (
    CartesianReductionGroup, CartesianReductionGeometry, cartesian_reduction_geometry,
    describe_cartesian_reduction_group, cartesian_reduction_group_workspace,
    cartesian_reduction_group_memory, validate_cartesian_reduction_groups,
)
from .ir import (
    TensorSpec,
    ContractionTerm,
    InputRef,
    NodeRef,
    Operation,
    CompiledTerm,
    ContractionPlan,
    ExecutionStats,
    ExecutionResult,
    Ref,
)
from .semantics import TensorDescription
from .preparation import ArrayConstant, TensorPreparation, SourceReadRequest
from .cfg_plan import CfgPlan, CfgTemplate, CfgInstance, CfgOutputField
from .inputs import (
    DiagramBindings,
    TensorAxisMetadata,
    CoordinatePoint,
    CoordinateSpec,
    BatchSpec,
    CoordinateBatch,
    PhysicalSlabBinding,
    BatchTensorBinding,
    canonical_coordinate_identity,
    canonical_axis_binding_identity,
    group_coordinate_batches,
    describe_coordinates,
    TermContribution,
    DiagramInputs,
    DiagramData,
)
from .equations import (
    ParsedTerm,
    parse_term,
)
from .scene import (
    ResourceTransition,
    SourceSpec,
    Action,
    Factor,
    PendingTerm,
    ContractionProblem,
    SparseBridgeSpec,
    BaryonJoinSpec,
    normalized_scene_relations,
)
from .storage import (
    RESOURCE_ACTION_KINDS,
    COMPUTE_ACTION_KINDS,
    validate_executable_resources,
    ResourceAllocation,
    ResourceSnapshot,
    StorageBlock,
    SourcePreparation,
    StorageEvent,
    MemoryPoint,
    SceneStoragePlan,
)
from .outputs import (
    OutputNode,
    OutputRecipe,
)
from .expanded import (
    ExpandedCorrelator,
)
from .plan import (
    PlannedContraction,
)
from .protocols import (
    CostModel,
    SourceLoader,
    DiagramBindingProvider,
    StateDescriptor,
    BatchLoader,
    MeasurementAwareLoader,
    IOReportingLoader,
    BackgroundBlockLoader,
    TemporaryArrayScope,
    TemporaryArrayStore,
)
from .vertices import (
    MesonCoreMetadata,
    VertexFactorMetadata,
    BaryonFactorMetadata,
    BaryonCoreMetadata,
)
from .matrix import (
    MatrixLinearAxis, MatrixComponent, MatrixNetwork, MatrixWickProduct,
    MatrixWickBlock, MatrixGroup, MatrixCatalog, BoundMatrixCatalog,
)
from .cores import CoreComponent, TensorCoreSpec
from .sparse import (
    cycle_point_equivalence,
    cycle_point_equalities_covered,
    cycle_requires_generic_join,
    cycle_intermediate_labels,
    bridge_point_equivalence,
    bridge_point_equalities_covered,
    bridge_intermediate_labels,
    cycle_external_labels,
    DEFAULT_JOIN_INDEX_BLOCK_SIZE,
    ESTIMATED_JOIN_INDEX_BYTES_PER_ROW,
    JOIN_COST_STATUS,
)
from .workload import (
    ExecutionBudget,
    WorkloadSpec,
    validate_frontend_workload,
    PlanBatch,
    PlanBundle,
    encode_scalar,
    decode_scalar,
    thaw_json,
)
from .signatures import (
    canonical_json_bytes,
    json_digest,
    encode_ref,
    encode_workload,
    encode_plan,
    structural_payload,
    structural_signature,
)

__all__ = [
    "DEVICE_ALLOCATION_ALIGNMENT", "device_allocation_bytes", "device_allocator_headroom",
    "CorrelatorResult",
    "ExecutionTarget",
    "ArrayConstant", "TensorPreparation", "SourceReadRequest",
    "MatrixLinearAxis", "MatrixComponent", "MatrixNetwork", "MatrixWickProduct",
    "MatrixWickBlock", "MatrixGroup", "MatrixCatalog", "BoundMatrixCatalog",
    "BackgroundBlockLoader",
    "RESOURCE_ACTION_KINDS", "COMPUTE_ACTION_KINDS",
    "ResourceTransition", "ResourceAllocation", "ResourceSnapshot", "validate_executable_resources",
    "TensorSpec",
    "ContractionTerm",
    "InputRef",
    "NodeRef",
    "Operation",
    "CompiledTerm",
    "ContractionPlan",
    "ExecutionStats",
    "ExecutionResult",
    "Ref",
    "BaryonLine",
    "BaryonTerm",
    "BaryonTopology",
    "BaryonVertex",
    "DiagramBindings",
    "TensorAxisMetadata",
    "canonical_coordinate_identity",
    "canonical_axis_binding_identity",
    "CoordinatePoint",
    "CoordinateSpec",
    "BatchSpec",
    "CoordinateBatch",
    "PhysicalSlabBinding",
    "BatchTensorBinding",
    "group_coordinate_batches",
    "describe_coordinates",
    "TermContribution",
    "DiagramInputs",
    "DiagramData",
    "AxisPattern",
    "axis_ids",
    "parse_axis_pattern",
    "ParsedTerm",
    "parse_term",
    "SourceSpec",
    "Action",
    "Factor",
    "PendingTerm",
    "ContractionProblem",
    "SparseBridgeSpec",
    "BaryonJoinSpec",
    "normalized_scene_relations",
    "StorageBlock",
    "SourcePreparation",
    "StorageEvent",
    "MemoryPoint",
    "SceneStoragePlan",
    "OutputNode",
    "OutputRecipe",
    "ExpandedCorrelator",
    "PlannedContraction",
    "CostModel",
    "SourceLoader",
    "DiagramBindingProvider",
    "StateDescriptor",
    "BatchLoader",
    "MeasurementAwareLoader",
    "IOReportingLoader",
    "TemporaryArrayScope",
    "TemporaryArrayStore",
    "VertexFactorMetadata",
    "BaryonFactorMetadata",
    "cycle_point_equivalence",
    "cycle_point_equalities_covered",
    "cycle_requires_generic_join",
    "cycle_intermediate_labels",
    "bridge_point_equivalence",
    "bridge_point_equalities_covered",
    "bridge_intermediate_labels",
    "cycle_external_labels",
    "DEFAULT_JOIN_INDEX_BLOCK_SIZE",
    "ESTIMATED_JOIN_INDEX_BYTES_PER_ROW",
    "JOIN_COST_STATUS",
    "ExecutionBudget",
    "WorkloadSpec",
    "validate_frontend_workload",
    "PlanBatch",
    "PlanBundle",
    "encode_scalar",
    "decode_scalar",
    "thaw_json",
    "canonical_json_bytes",
    "json_digest",
    "encode_ref",
    "encode_workload",
    "encode_plan",
    "structural_payload",
    "structural_signature",
]

from .sparse import parse_sparse_equation, sparse_cycle_endpoint_axes, sparse_cycle_slots, SPARSE_ENDPOINT_NAMES
__all__ += ["parse_sparse_equation", "sparse_cycle_endpoint_axes", "sparse_cycle_slots", "SPARSE_ENDPOINT_NAMES"]

from .sparse import bridge_external_labels
__all__ += ["bridge_external_labels"]
__all__ += ["TensorDescription"]

from .matrix_kernel import MatrixKernelSpec, MatrixKernelStageEstimate, MatrixKernelEstimate
from .factorized_reduction import FactorizedWideReduction
__all__ += ["FactorizedWideReduction"]
__all__ += ["MatrixKernelSpec", "MatrixKernelStageEstimate", "MatrixKernelEstimate"]
__all__ += ["CfgPlan", "CfgTemplate", "CfgInstance", "CfgOutputField"]
__all__ += ["DenseBatchOperand", "DenseBatchResult", "DenseKernelBatch", "dense_batch_memory", "dense_batch_producer_eligible"]
__all__ += ["ScalarReductionGroup", "ScalarReductionGeometry", "scalar_reduction_group_limit", "scalar_reduction_geometry",
            "describe_scalar_reduction_group", "scalar_reduction_group_memory", "validate_scalar_reduction_groups"]
__all__ += ["CartesianReductionGroup", "CartesianReductionGeometry", "cartesian_reduction_geometry",
            "describe_cartesian_reduction_group", "cartesian_reduction_group_workspace",
            "cartesian_reduction_group_memory", "validate_cartesian_reduction_groups"]

__all__ += ["normalize_time_key", "time_key_order"]
__all__ += ['MesonCoreMetadata', 'VertexKernelDescription']
__all__ += ['MesonWholeCore', 'MesonWholeKernel']
