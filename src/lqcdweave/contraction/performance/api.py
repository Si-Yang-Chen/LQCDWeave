"""Public measurement and cost-model interface for correlator planning."""

from .targets import load_execution_target, register_execution_target, resolve_target_profile
from .targets import target_implementation_coverage
from .calibration_cache import IMPLEMENTATION_VERSION, calibrate_target_profile, load_target_calibration

from .model import CostProfile, cost_profile_calibration_status
from .vertex_kernel import estimate_vertex_kernel
from .meson_core import estimate_meson_whole
from .measurement import ExecutionMeasurement, PhaseCollector
from .profile_fit import (
    ProfileFitReport,
    calibrate_cost_profile,
    evaluate_cost_profile,
    execution_cost_samples,
    fit_cost_profile,
    fit_cost_profile_sources,
    normalize_profile_weights,
    update_cost_profile,
    weighted_profile_merge,
)
from .profile_io import (
    fit_profile_files,
    fit_profile_sources_files,
    load_cost_profile,
    load_cost_profile_document,
    merge_profile_files,
    profile_file_path,
    save_cost_profile,
)
from .scene_cost_report import predict_plan_cost

__all__ = [
    "load_execution_target", "register_execution_target", "resolve_target_profile",
    "IMPLEMENTATION_VERSION", "calibrate_target_profile", "load_target_calibration",
    "target_implementation_coverage",
    "CostProfile",
    "estimate_vertex_kernel",
    "estimate_meson_whole",
    "cost_profile_calibration_status",
    "ExecutionMeasurement",
    "PhaseCollector",
    "ProfileFitReport",
    "calibrate_cost_profile",
    "evaluate_cost_profile",
    "execution_cost_samples",
    "fit_cost_profile",
    "fit_cost_profile_sources",
    "fit_profile_files",
    "fit_profile_sources_files",
    "load_cost_profile",
    "load_cost_profile_document",
    "merge_profile_files",
    "normalize_profile_weights",
    "predict_plan_cost",
    "profile_file_path",
    "save_cost_profile",
    "update_cost_profile",
    "weighted_profile_merge",
]
