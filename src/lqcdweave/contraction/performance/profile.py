"""Command-line workflow for standalone CostProfile fitting and fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from lqcdweave.contraction.performance.api import (
    CostProfile,
    fit_profile_sources_files,
    merge_profile_files,
    normalize_profile_weights,
    profile_file_path,
    save_cost_profile,
)


def _report_path(path):
    """Resolve a report input path before delegating profile fitting."""
    # Small validation helper copied from contraction.performance.profile_io;
    # that module still uses its own version for non-CLI profile services.
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"measurement report not found: {path}")
    return resolved


def _optional_label(value, name):
    """Normalize an optional CLI label using the profile service contract."""
    # This intentionally mirrors contraction.performance.profile_io's helper:
    # the M API also uses it when validating direct service calls.
    if value in (None, "", "-"):
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string label or None")
    return value


def _reject_output_alias(output, inputs):
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"profile output already exists: {destination}")
    for source in inputs:
        if source is None:
            continue
        try:
            candidate = Path(source).expanduser().resolve(strict=True)
        except (TypeError, OSError):
            continue
        if destination == candidate:
            raise ValueError("profile output must be a new path, separate from its inputs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit weighted operation reports and validate on a held-out group")
    fit.add_argument("--base", default=None)
    training_inputs = fit.add_mutually_exclusive_group(required=True)
    training_inputs.add_argument(
        "--training-report",
        help="legacy single training report; use weight 1 and its run ID as source ID",
    )
    training_inputs.add_argument(
        "--training-source", nargs=5, action="append",
        metavar=("REPORT", "WEIGHT", "SOURCE_ID", "RUN_ID", "SHAPE_FAMILY"),
        help="repeat a weighted source; use - for an unknown optional group label",
    )
    validation_inputs = fit.add_mutually_exclusive_group(required=True)
    validation_inputs.add_argument("--validation-report", help="legacy single held-out report")
    validation_inputs.add_argument(
        "--validation-source", nargs=4, action="append",
        metavar=("REPORT", "SOURCE_ID", "RUN_ID", "SHAPE_FAMILY"),
        help="repeat held-out reports; use - for an unknown optional group label",
    )
    fit.add_argument("--training-run-id")
    fit.add_argument("--validation-run-id")
    fit.add_argument("--validation-by", choices=("run", "shape_family"), default="run")
    fit.add_argument("--hardware", required=True)
    fit.add_argument(
        "--verified", action="store_true",
        help="assert that numerical outputs were independently verified",
    )
    fit.add_argument("--output", required=True)
    merge = commands.add_parser("merge", help="weighted fusion of compatible profiles")
    merge_components = merge.add_mutually_exclusive_group(required=True)
    merge_components.add_argument(
        "--component", nargs=2, action="append", metavar=("PROFILE", "WEIGHT"),
        help="add a profile with an explicit weight",
    )
    merge_components.add_argument(
        "--profile", action="append", help="add a profile with equal weight; repeat as needed",
    )
    merge.add_argument("--allow-cross-hardware", action="store_true")
    merge.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "fit":
        if args.training_source is not None:
            training_sources = tuple(
                (path, float(weight), source_id,
                 _optional_label(run_id, "run_id"),
                 _optional_label(shape_family, "shape_family"))
                for path, weight, source_id, run_id, shape_family in args.training_source
            )
        else:
            if not args.training_run_id:
                parser.error("--training-run-id is required with --training-report")
            training_sources = ((args.training_report, 1.0, args.training_run_id,
                                 args.training_run_id, None),)
        if args.validation_source is not None:
            validation_sources = tuple(
                (path, source_id, _optional_label(run_id, "run_id"),
                 _optional_label(shape_family, "shape_family"))
                for path, source_id, run_id, shape_family in args.validation_source
            )
        else:
            if not args.validation_run_id:
                parser.error("--validation-run-id is required with --validation-report")
            validation_sources = ((args.validation_report, args.validation_run_id,
                                   args.validation_run_id, None),)
        positive_training, _ = normalize_profile_weights(
            ((spec, spec[1]) for spec in training_sources)
        )
        input_paths = [profile_file_path(args.base)]
        input_paths.extend(profile_file_path(spec[0]) for spec, _, _ in positive_training)
        input_paths.extend(_report_path(spec[0]) for spec in validation_sources)
        _reject_output_alias(args.output, input_paths)
        profile, diagnostics, provenance = fit_profile_sources_files(
            args.base, training_sources, validation_sources,
            verified=args.verified, hardware_fingerprint=args.hardware,
            validation_by=args.validation_by, return_provenance=True,
        )
        save_cost_profile(profile, args.output, provenance=provenance)
        print(diagnostics)
    else:
        components = (args.profile if args.profile is not None else
                      [(path, float(weight)) for path, weight in args.component])
        component_sources = []
        normalized, _ = normalize_profile_weights(components)
        for reference, _, _ in normalized:
            if not isinstance(reference, CostProfile):
                component_sources.append(profile_file_path(reference))
        _reject_output_alias(args.output, component_sources)
        profile, provenance = merge_profile_files(
            components, allow_cross_hardware=args.allow_cross_hardware,
            return_provenance=True,
        )
        save_cost_profile(profile, args.output, provenance=provenance)
        print(f"saved weighted CostProfile to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
