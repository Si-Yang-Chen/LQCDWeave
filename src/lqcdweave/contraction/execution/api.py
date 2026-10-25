"""Public path execution and atomic-operation boundary."""

from .preparation import PreparedSourceLoader, bind_prepared_loader
from .kernel_metadata import describe_vertex_kernel
from .binding_cache import ExecutionBindingCache
from .run_resources import RunResources
from .diagnostics import execution_debug
from .calibration import probe_compute_profile
from .device import device_execution, active_device_execution
from .output_program import (bind_device_output_assembly, describe_device_output_fields,
                             estimate_device_output_bytes, estimate_device_output_metadata_bytes,
                             describe_device_output_program, describe_device_output_reservation)
from .output_postprocessing import (DeviceOutputProcessor, pack_device_fields,
                                    describe_device_download_layout)

from .direct import execute_direct_contraction
from .domain import BoundContractionExecution, execute_contraction_path
from .executor import execute_plan, prepare_plan_kernels
from .errors import PlanMismatchError, SourceIdentityError, UnsupportedLoaderError
from .scheduled import (
    BundleExecutionResult,
    BundleExecutionStats,
    PreparedBundle,
    execute_bundle,
    prepare_bundle_execution,
)
from .operations import (
    contract_baryon_join,
    contract_dense,
    contract_sparse_bridge,
    copy_output,
    copy_to_backend,
    load_retained_source,
    load_source,
    materialize_from_blocks,
    materialize_source,
    read_block,
    release_tensor,
    select_output_lane,
    sparse_pairs,
)
from .scene_runtime import replay
from .cfg_runtime import CfgExecutionCache,execute_cfg_plan,iter_execute_cfg_plan
from .cfg_windows import execute_cfg_windows,iter_execute_cfg_windows
from .matrix_kernels import contract_meson_matrix_pc, contract_meson_loop_pc
from .storage_runtime import (
    audit_execution_memory,
    replay_scene_storage,
    replay_storage_batches,
)

__all__ = [
    "execution_debug",
    "ExecutionBindingCache",
    "RunResources",
    "DeviceOutputProcessor", "describe_device_download_layout", "pack_device_fields", "active_device_execution",
    "bind_device_output_assembly", "describe_device_output_fields", "estimate_device_output_bytes",
    "estimate_device_output_metadata_bytes", "describe_device_output_program", "describe_device_output_reservation",
    "PreparedSourceLoader", "bind_prepared_loader", "probe_compute_profile", "device_execution",
    "CfgExecutionCache", "execute_cfg_plan", "iter_execute_cfg_plan",
    "execute_cfg_windows", "iter_execute_cfg_windows",
    "contract_meson_matrix_pc", "contract_meson_loop_pc",
    "PlanMismatchError",
    "SourceIdentityError",
    "UnsupportedLoaderError",
    "BoundContractionExecution",
    "BundleExecutionResult",
    "BundleExecutionStats",
    "contract_baryon_join",
    "contract_dense",
    "contract_sparse_bridge",
    "audit_execution_memory",
    "copy_output",
    "copy_to_backend",
    "execute_contraction_path",
    "execute_bundle",
    "execute_direct_contraction",
    "execute_plan",
    "load_retained_source",
    "load_source",
    "materialize_from_blocks",
    "materialize_source",
    "prepare_plan_kernels",
    "prepare_bundle_execution",
    "PreparedBundle",
    "read_block",
    "release_tensor",
    "replay",
    "replay_scene_storage",
    "replay_storage_batches",
    "select_output_lane",
    "sparse_pairs",
]
