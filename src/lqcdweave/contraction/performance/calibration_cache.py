"""User-local cache for verified execution-target compute calibration."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.measurement import ReplayReport

from .model import CostProfile
from .profile_fit import calibrate_cost_profile, execution_cost_samples
from .profile_io import load_cost_profile_document, save_cost_profile


IMPLEMENTATION_VERSION = "unified-device-v1"
_CALIBRATABLE_CATEGORIES = {"dense_contraction", "logical_snapshot"}


def _cache_key_digest(cache_key):
    if not isinstance(cache_key, str) or not cache_key.strip():
        raise ValueError("cache_key must be a nonempty string")
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def _cache_directory(directory=None):
    if directory is not None:
        return Path(directory).expanduser()
    configured = os.environ.get("LQCDWEAVE_PROFILE_DIR")
    root = (Path(configured).expanduser() if configured else
            Path.home() / ".cache" / "lqcdweave" / "profiles")
    return root / "calibration"


def _cache_path(cache_key, directory=None):
    digest = _cache_key_digest(cache_key)
    return _cache_directory(directory) / f"{digest}.json", digest


def _attempt_result(document, *, calibrated, missing, path=None):
    result = dict(document)
    result["calibration"] = {
        "calibrated": bool(calibrated),
        "missing": list(missing),
        "cache_path": None if path is None else str(path),
    }
    return result


def calibrate_target_profile(
    target: ExecutionTarget,
    document: Mapping,
    report: ReplayReport,
    *,
    cache_key: str,
    directory=None,
) -> dict:
    """Fit and cache eligible contract/snapshot samples from a verified report.

    A nonqualifying report returns the supplied profile document, with a
    ``calibration`` field describing what is missing. No device work is started
    by this function.
    """
    if not isinstance(target, ExecutionTarget):
        raise TypeError("target must be ExecutionTarget")
    if not isinstance(document, Mapping):
        raise TypeError("document must be a CostProfile document mapping")
    original = dict(document)
    missing = []
    try:
        path, key_digest = _cache_path(cache_key, directory)
    except ValueError:
        path = None
        key_digest = None
        missing.append("nonempty cache_key")

    profile = original.get("profile")
    if not isinstance(profile, CostProfile):
        missing.append("CostProfile in document['profile']")
    base_provenance = original.get("provenance", {})
    if base_provenance is None:
        base_provenance = {}
    if not isinstance(base_provenance, Mapping):
        missing.append("mapping in document['provenance']")

    if not isinstance(report, ReplayReport):
        missing.append("ReplayReport")
        execution_profile = None
    else:
        execution_profile = report.execution_profile
        if not isinstance(execution_profile, Mapping):
            missing.append("report execution_profile metadata")
            execution_profile = {}
        if execution_profile.get("verified") is not True:
            missing.append("report verified=True")
        if execution_profile.get("implementation_version") != IMPLEMENTATION_VERSION:
            missing.append(f"implementation_version={IMPLEMENTATION_VERSION}")
        covered = execution_profile.get("covered_operations", ())
        if isinstance(covered, (str, bytes)):
            covered = ()
        try:
            covered = tuple(covered)
        except TypeError:
            covered = ()
        if any(not isinstance(operation, str) for operation in covered):
            covered = ()
        if not {"contract", "snapshot"}.issubset(covered):
            missing.append("contract and snapshot coverage")

    samples = ()
    if isinstance(report, ReplayReport):
        try:
            samples = execution_cost_samples(report)
        except (KeyError, TypeError, ValueError) as exc:
            missing.append(f"valid cost samples ({exc})")
    if not samples:
        missing.append("nonempty contract/snapshot samples")
    elif any(sample.get("category") not in _CALIBRATABLE_CATEGORIES
             for sample in samples):
        missing.append("contract/snapshot samples only")

    if missing:
        return _attempt_result(original, calibrated=False, missing=missing)

    execution_profile = report.execution_profile
    fingerprint = (target.hardware_fingerprint
                   if target.hardware_fingerprint != "unmeasured" else cache_key)
    try:
        fitted, diagnostics = calibrate_cost_profile(
            profile,
            (report,),
            verified=execution_profile.get("verified") is True,
            hardware_fingerprint=fingerprint,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _attempt_result(
            original,
            calibrated=False,
            missing=[f"compatible calibration inputs ({exc})"],
        )

    timing_mode = execution_profile.get("timing_mode")
    covered_operations = tuple(
        operation for operation in execution_profile.get("covered_operations", ())
        if operation in {"contract", "snapshot"}
    )
    provenance = dict(base_provenance)
    provenance.update({
        "calibration_status": "partially-calibrated",
        "operation": "target-compute-calibration",
        "timing_mode": timing_mode,
        "implementation_version": IMPLEMENTATION_VERSION,
        "covered_operations": list(covered_operations),
        "covered_dtypes": list(execution_profile.get('covered_dtypes', ())),
        "cache_key_sha256": key_digest,
        "target_name": target.name,
        "target_backend": target.backend,
        "target_runtime": target.runtime,
        "target_software": dict(target.software),
        "target_hardware_fingerprint": target.hardware_fingerprint,
        "diagnostics": diagnostics,
    })
    saved = save_cost_profile(fitted, path, provenance=provenance)

    result = dict(original)
    result.update({
        "profile": fitted,
        "provenance": provenance,
        "source": str(saved),
        "calibration_status": "partially-calibrated",
    })
    return _attempt_result(
        result,
        calibrated=True,
        missing=(),
        path=saved,
    )


def load_target_calibration(
    target: ExecutionTarget,
    *,
    cache_key: str,
    directory=None,
) -> dict | None:
    """Load a calibration only when its key and implementation still match."""
    if not isinstance(target, ExecutionTarget):
        raise TypeError("target must be ExecutionTarget")
    path, key_digest = _cache_path(cache_key, directory)
    if not path.is_file():
        return None
    try:
        document = load_cost_profile_document(path)
    except (OSError, TypeError, ValueError):
        return None
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    if (provenance.get("cache_key_sha256") != key_digest
            or provenance.get("implementation_version") != IMPLEMENTATION_VERSION):
        return None
    if not isinstance(document.get("profile"), CostProfile):
        return None
    fingerprint = document["profile"].hardware_fingerprint
    if (target.hardware_fingerprint != "unmeasured"
            and fingerprint not in ("unmeasured", target.hardware_fingerprint)):
        return None
    return document


__all__ = [
    "IMPLEMENTATION_VERSION",
    "calibrate_target_profile",
    "load_target_calibration",
]
