"""Persistence and CostProfile selection for declared execution targets."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
from dataclasses import replace
from collections.abc import Mapping

from ...contracts.correlator.api import ExecutionTarget
from ...storage.dataset import write_json_document


TARGET_DIRECTORY = Path(__file__).resolve().parent / "profiles" / "targets"
_TARGET_FORMAT = "easy-distillation.execution-target"
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+\Z", re.ASCII)


def _validate_name(name):
    if (not isinstance(name, str) or not name.strip() or name != name.strip()
            or name in {'.', '..'} or any(c in name for c in '/\\\0')
            or not name.isprintable()):
        raise ValueError("execution target name must be a nonempty display name without path separators")


def _user_target_directory():
    configured = os.environ.get("LQCDWEAVE_PROFILE_DIR")
    root = (Path(configured).expanduser() if configured else
            Path.home() / ".cache" / "lqcdweave" / "profiles")
    return root / "targets"


def _normalize_directories(directories):
    if directories is None:
        values = (_user_target_directory(), TARGET_DIRECTORY)
    elif isinstance(directories, (str, os.PathLike)):
        values = (Path(directories).expanduser(),)
    else:
        try:
            values = tuple(Path(value).expanduser() for value in directories)
        except TypeError as exc:
            raise TypeError("directories must be a path or iterable of paths") from exc

    normalized = []
    seen = set()
    for value in values:
        path = Path(value).expanduser()
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            normalized.append(path)
    return tuple(normalized)


def register_execution_target(target: ExecutionTarget, directory=None) -> Path:
    """Atomically register or explicitly update a target JSON document.

    ``directory`` is the target archive directory itself. With no override,
    records are written below the user profile directory.
    """
    if not isinstance(target, ExecutionTarget):
        raise TypeError("target must be ExecutionTarget")
    _validate_name(target.name)
    destination = (Path(directory).expanduser() if directory is not None
                   else _user_target_directory())
    destination.mkdir(parents=True, exist_ok=True)
    filename = (target.name if _SAFE_NAME.fullmatch(target.name) else
                hashlib.sha256(target.name.encode('utf-8')).hexdigest())
    path = destination / f"{filename}.json"
    document = {
        "format": _TARGET_FORMAT,
        "target": target.to_dict(),
        "aliases": [],
    }
    write_json_document(path, document)
    return path


def _read_target_document(path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid execution target document: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid execution target document: {path}")

    aliases = document.get("aliases", [])
    if not isinstance(aliases, list):
        raise ValueError(f"execution target aliases must be a list: {path}")
    for alias in aliases:
        _validate_name(alias)

    if "target" in document:
        if document.get("format") != _TARGET_FORMAT:
            raise ValueError(f"unsupported execution target format: {path}")
        record = document["target"]
    else:
        # Accept a flat target mapping as a convenient hand-authored archive.
        if document.get("format") not in (None, _TARGET_FORMAT):
            raise ValueError(f"unsupported execution target format: {path}")
        record = {key: value for key, value in document.items()
                  if key not in {"format", "aliases"}}
    if not isinstance(record, Mapping):
        raise ValueError(f"execution target payload must be a mapping: {path}")
    return document, record, tuple(aliases)


def load_execution_target(name, directories=None) -> ExecutionTarget:
    """Load a uniquely matching target name or alias without probing hardware."""
    _validate_name(name)
    matches = {}
    for directory in _normalize_directories(directories):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            document, record, aliases = _read_target_document(path)
            if record.get("name") != name and name not in aliases:
                continue
            try:
                target = ExecutionTarget.from_dict(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid execution target document: {path}: {exc}") from exc
            identity = json.dumps(
                document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            matches.setdefault(identity, target)

    if not matches:
        raise ValueError(f"ExecutionTarget not found: {name}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous ExecutionTarget name or alias: {name}")
    return next(iter(matches.values()))


def resolve_target_profile(target: ExecutionTarget, profile=None) -> dict:
    """Resolve a target's compute-cost model without numerical calibration."""
    if not isinstance(target, ExecutionTarget):
        raise TypeError("target must be ExecutionTarget")

    from .profile_io import load_cost_profile_document, PROFILE_DIRECTORY

    selected_profile = profile if profile is not None else target.profile
    if selected_profile is None and target.hardware_fingerprint != 'unmeasured':
        from .calibration_cache import load_target_calibration, IMPLEMENTATION_VERSION
        cached = load_target_calibration(target,
            cache_key=target.hardware_fingerprint + ':' + IMPLEMENTATION_VERSION)
        if cached is not None:
            selected_profile = cached['source']
    if selected_profile is None:
        # Match registered device names in the existing model directories.
        candidates = (_user_target_directory().parent / (target.name + '.json'),
                      PROFILE_DIRECTORY / (target.name + '.json'))
        selected_profile = next((path for path in candidates if path.is_file()), None)
    if selected_profile is None:
        from .calibration_cache import IMPLEMENTATION_VERSION
        matches = []
        for path in sorted((_user_target_directory().parent / 'calibration').glob('*.json')):
            cached = load_cost_profile_document(path)
            provenance = cached.get('provenance', {})
            if (provenance.get('target_name') == target.name
                    and provenance.get('target_backend') == target.backend
                    and provenance.get('target_runtime') == target.runtime
                    and provenance.get('implementation_version') == IMPLEMENTATION_VERSION
                    and all(provenance.get('target_software', {}).get(key) == value
                            for key, value in target.software)
                    and (target.hardware_fingerprint == 'unmeasured'
                         or cached['profile'].hardware_fingerprint == target.hardware_fingerprint)):
                matches.append(path)
        # An alias used on multiple environments does not identify a model.
        # Keep the explicitly uncalibrated fallback until the target is specified.
        if len(matches) == 1:
            selected_profile = matches[0]
    use_default = selected_profile is None
    document = dict(load_cost_profile_document(None if use_default else selected_profile))
    model = document["profile"]

    if use_default:
        if target.hardware_fingerprint != "unmeasured":
            document["profile"] = replace(
                model, hardware_fingerprint=target.hardware_fingerprint
            )
        document["calibration_status"] = "uncalibrated"
    elif (model.hardware_fingerprint != "unmeasured"
          and target.hardware_fingerprint != "unmeasured"
          and model.hardware_fingerprint != target.hardware_fingerprint):
        raise ValueError(
            f"CostProfile hardware fingerprint {model.hardware_fingerprint!r} "
            f"does not match ExecutionTarget {target.name!r} "
            f"({target.hardware_fingerprint!r})"
        )

    resolved_model = document["profile"]
    modeled = ('contract', 'snapshot', 'preparation', 'packing', 'accumulation',
               'baryon_join', 'sparse_bridge')
    calibrated = tuple(document.get('provenance', {}).get('covered_operations', ()))
    document["target"] = target.to_dict()
    document["coverage"] = {
        "target_operations": list(target.operations),
        "profile_source": document.get("source"),
        "calibration_status": document["calibration_status"],
        "hardware_fingerprint": resolved_model.hardware_fingerprint,
        "calibrated_operations": calibrated,
        "missing_operations": tuple(operation for operation in modeled if operation not in calibrated),
        "description": (
            "CostProfile covers modeled preparation, snapshots, and compute costs. "
            "Declared target operations do not by themselves establish calibrated coverage."
        ),
    }
    return document


def target_implementation_coverage(target, operations):
    """Return the current implementation ABI coverage without importing X."""
    implemented = {'contract', 'baryon_join', 'sparse_bridge'}
    required = tuple(sorted(set(operations)))
    available = implemented.intersection(target.operations)
    return dict(implementation_version='unified-device-v1', backend=target.backend,
                runtime=target.runtime, required=required,
                missing=tuple(operation for operation in required if operation not in available))


__all__ = [
    "TARGET_DIRECTORY",
    "load_execution_target",
    "register_execution_target",
    "resolve_target_profile",
    "target_implementation_coverage",
]
