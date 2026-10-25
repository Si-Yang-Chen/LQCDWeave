"""Fit and merge saved cost profiles without running correlator calculations."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .profile_fit import (
    ProfileFitReport,
    fit_cost_profile,
    fit_cost_profile_sources,
    normalize_profile_weights,
    weighted_profile_merge,
)
from .model import CostProfile, cost_profile_calibration_status
from ...storage.scene_cost import (
    read_cost_profile_document,
    save_cost_profile as _save_cost_profile_document,
)
from ...runtime.measurement import load_measurement


PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"


def profile_file_path(profile=None):
    """Resolve a package name/default or explicit path at the workflow edge."""
    if profile is None:
        path = PROFILE_DIRECTORY / "default.json"
    elif isinstance(profile, (str, Path)):
        candidate = Path(profile).expanduser()
        if candidate.is_file():
            path = candidate
        else:
            path = PROFILE_DIRECTORY / (str(profile) + ".json")
            if not path.is_file():
                raise FileNotFoundError(f"CostProfile not found: {profile}")
    else:
        raise TypeError("profile must be CostProfile, a built-in name, a path, or None")
    if not path.is_file():
        raise FileNotFoundError(f"CostProfile not found: {profile}")
    return path


def load_cost_profile_document(profile=None):
    """Load a CostProfile and its checked sidecar metadata in a workflow."""
    if isinstance(profile, CostProfile):
        provenance = {}
        return {
            "profile": profile,
            "provenance": provenance,
            "source": (f"CostProfile object (version={profile.version!r}, "
                       f"hardware={profile.hardware_fingerprint!r})"),
            "calibration_status": cost_profile_calibration_status(profile, provenance),
        }
    path = profile_file_path(profile)
    document = read_cost_profile_document(path)
    profile_fields = document.get("profile")
    if not isinstance(profile_fields, dict):
        raise ValueError("CostProfile JSON must contain a profile object")
    known = {field.name for field in fields(CostProfile)}
    unknown = set(profile_fields) - known
    if unknown:
        raise ValueError(f"unknown CostProfile fields: {sorted(unknown)}")
    provenance = document.get("provenance", {})
    loaded_profile = CostProfile(**profile_fields)
    return {
        "profile": loaded_profile,
        "provenance": provenance,
        "source": str(path),
        "calibration_status": cost_profile_calibration_status(loaded_profile, provenance),
    }


def load_cost_profile(profile=None) -> CostProfile:
    """Resolve a CostProfile object, built-in name, path, or package default."""
    return load_cost_profile_document(profile)["profile"]


def save_cost_profile(profile: CostProfile, path, *, provenance=None) -> Path:
    """Save a profile through the shared cost-profile storage API."""
    if not isinstance(profile, CostProfile):
        raise TypeError("profile must be CostProfile")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _save_cost_profile_document(target, profile, provenance=provenance)
    return target


def _report_path(path):
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"measurement report not found: {path}")
    return resolved


def fit_profile_files(base, training_report, validation_report, *, verified,
                      hardware_fingerprint, training_run_id, validation_run_id,
                      return_provenance=False):
    """Fit one report with unit source weight and hold out a separate run."""
    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise ValueError("training_run_id must be a nonempty user-supplied run identity")
    if not isinstance(validation_run_id, str) or not validation_run_id.strip():
        raise ValueError("validation_run_id must be a nonempty user-supplied run identity")
    return fit_profile_sources_files(
        base,
        ((training_report, 1.0, training_run_id, training_run_id, None),),
        ((validation_report, validation_run_id, validation_run_id, None),),
        verified=verified, hardware_fingerprint=hardware_fingerprint,
        validation_by="run", return_provenance=return_provenance,
    )


def _optional_label(value, name):
    if value in (None, "", "-"):
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string label or None")
    return value


def fit_profile_sources_files(base, training_sources, validation_sources, *, verified,
                              hardware_fingerprint, validation_by="run",
                              return_provenance=False):
    """Fit weighted report sources and validate on disjoint run/family groups.

    Training entries are ``(path, weight, source_id, run_id, shape_family)``;
    validation entries are ``(path, source_id, run_id, shape_family)``. The
    selected split identity is a caller declaration, not authenticated by the
    operation report format.
    """
    training_specs = tuple(training_sources)
    validation_specs = tuple(validation_sources)
    if not training_specs or not validation_specs:
        raise ValueError("at least one training and one validation report are required")
    if any(not isinstance(spec, (tuple, list)) or len(spec) != 5 for spec in training_specs):
        raise ValueError("training source entries must be (path, weight, source_id, run_id, shape_family)")
    normalized, _ = normalize_profile_weights(
        ((spec, spec[1]) for spec in training_specs)
    )
    active_training = []
    for spec, weight, _ in normalized:
        path, _, source_id, run_id, shape_family = spec
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("training source_id must be nonempty")
        run_id = _optional_label(run_id, "run_id")
        shape_family = _optional_label(shape_family, "shape_family")
        active_training.append((path, weight, source_id, run_id, shape_family))
    validation_specs = tuple(
        (path, source_id, _optional_label(run_id, "run_id"),
         _optional_label(shape_family, "shape_family"))
        for path, source_id, run_id, shape_family in validation_specs
    )
    if any(not isinstance(source_id, str) or not source_id.strip()
           for _, source_id, _, _ in validation_specs):
        raise ValueError("validation source_id must be nonempty")

    base_document = load_cost_profile_document(base)
    base_profile = base_document["profile"]
    used_paths = set()
    used_reports = []

    def load_report(path):
        resolved = _report_path(path)
        if resolved in used_paths:
            raise ValueError("training and validation report paths must differ")
        used_paths.add(resolved)
        report_document = json.loads(resolved.read_text(encoding="utf-8"))
        if any(report_document == prior for prior in used_reports):
            raise ValueError("training and validation report contents are identical")
        used_reports.append(report_document)
        return resolved, load_measurement(resolved)

    fit_sources = []
    training_metadata = []
    for path, weight, source_id, run_id, shape_family in active_training:
        resolved, report = load_report(path)
        fit_sources.append(ProfileFitReport(
            report, source_id, run_id=run_id, shape_family=shape_family,
            weight=weight, reference=str(resolved),
        ))
        training_metadata.append({
            "source_id": source_id, "weight": weight, "run_id": run_id,
            "shape_family": shape_family, "source": str(resolved),
        })

    heldout_sources = []
    validation_metadata = []
    for path, source_id, run_id, shape_family in validation_specs:
        resolved, report = load_report(path)
        heldout_sources.append(ProfileFitReport(
            report, source_id, run_id=run_id, shape_family=shape_family,
            reference=str(resolved),
        ))
        validation_metadata.append({
            "source_id": source_id, "run_id": run_id, "shape_family": shape_family,
            "source": str(resolved),
        })

    fitted, diagnostics = fit_cost_profile_sources(
        base_profile, fit_sources, heldout_sources,
        verified=verified, hardware_fingerprint=hardware_fingerprint,
        validation_by=validation_by,
    )
    provenance = {
        "operation": "fit",
        "prediction_structure": "CostProfile.compute-model/v1",
        "base_profile": {
            "source": base_document["source"],
            "version": base_profile.version,
            "hardware_fingerprint": base_profile.hardware_fingerprint,
            "sample_count": base_profile.sample_count,
            **({"prior_provenance": base_document["provenance"]}
                if base_document["provenance"] else {}),
        },
        "training_sources": training_metadata,
        "validation_sources": validation_metadata,
        "validation_by": validation_by,
        "independence": (
            "run IDs and shape-family labels are user declarations; ReplayReport does not "
            "carry group identity that can be authenticated automatically"
        ),
        "verified_outputs_asserted_by_user": bool(verified),
        "hardware_fingerprint_asserted_by_user": hardware_fingerprint,
        "fit_diagnostics": diagnostics,
    }
    if return_provenance:
        return fitted, diagnostics, provenance
    return fitted, diagnostics


def merge_profile_files(components, *, allow_cross_hardware=False,
                        return_provenance=False):
    """Merge named or explicit profile files with their sidecar provenance."""
    normalized, weight_mode = normalize_profile_weights(components)
    resolved = []
    source_metadata = []
    for reference, input_weight, _ in normalized:
        document = load_cost_profile_document(reference)
        resolved.append((document["profile"], input_weight))
        metadata = {"source": document["source"]}
        if document["provenance"]:
            metadata["prior_provenance"] = document["provenance"]
        source_metadata.append(metadata)
    merged, provenance = weighted_profile_merge(
        resolved, allow_cross_hardware=allow_cross_hardware,
        return_provenance=True, source_metadata=source_metadata,
        input_weight_mode=weight_mode,
    )
    if return_provenance:
        return merged, provenance
    return merged


__all__ = [
    "fit_profile_files", "fit_profile_sources_files", "load_cost_profile",
    "load_cost_profile_document", "merge_profile_files",
    "normalize_profile_weights", "profile_file_path", "save_cost_profile",
    "weighted_profile_merge",
]
