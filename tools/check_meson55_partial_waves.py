"""Audit endpoint angular geometry in the meson55 catalog.

This offline tool uses catalog ``terms`` only. It does not import the
本软件 runtime or trust cached endpoint weights. For each radial
shell it evaluates

    S_l = (2*l + 1) * w.conj() @ P_l(cosine) @ w

from the Legendre addition theorem. A shell's zero threshold is the requested
relative tolerance times ``sum(abs(w)**2)`` for that shell.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    ROOT
    / "src/lqcdweave/operators/single_hadron/presets/meson55_v2.json"
)
SCHEMA = "lqcdweave.meson55.partial-wave-report"
REPORT_VERSION = 1
DEFAULT_TOLERANCE = 1e-10


def _finite_real(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def endpoint_weights(terms: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> dict[tuple[int, int, int], complex]:
    """Return complex weights summed at endpoints derived only from ``terms``.

    Direction codes 0/1/2 add one unit along x/y/z; 3/4/5 subtract one unit
    along x/y/z. Passing a row mapping is accepted as shorthand for its
    ``terms`` field.
    """
    if isinstance(terms, Mapping):
        if "terms" not in terms:
            raise ValueError("row mapping is missing 'terms'")
        terms = terms["terms"]
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise ValueError("terms must be a sequence")

    weights: defaultdict[tuple[int, int, int], complex] = defaultdict(complex)
    for term_index, term in enumerate(terms):
        if not isinstance(term, Mapping):
            raise ValueError(f"term {term_index} must be an object")
        path = term.get("path")
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
            raise ValueError(f"term {term_index} path must be a sequence")
        endpoint = [0, 0, 0]
        for step_index, code in enumerate(path):
            if isinstance(code, bool) or not isinstance(code, int) or code not in range(6):
                raise ValueError(
                    f"term {term_index} path step {step_index} must be an integer from 0 to 5"
                )
            axis = code % 3
            endpoint[axis] += 1 if code < 3 else -1

        coefficient = term.get("coefficient")
        if not isinstance(coefficient, Sequence) or isinstance(coefficient, (str, bytes)) or len(coefficient) != 2:
            raise ValueError(f"term {term_index} coefficient must be [real, imaginary]")
        real = _finite_real(coefficient[0], f"term {term_index} coefficient real part")
        imag = _finite_real(coefficient[1], f"term {term_index} coefficient imaginary part")
        weights[tuple(endpoint)] += complex(real, imag)

    return dict(weights)


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "kind": row.get("kind"),
        "family": row.get("family"),
        "irrep": row.get("irrep"),
        "target_l": row.get("target_l"),
    }


def _strength_for_shell(
    endpoints: Sequence[tuple[int, int, int]],
    weights: np.ndarray,
    l: int,
    epsilon_scale: float,
) -> float:
    """Evaluate one shell's addition-theorem strength with a roundoff guard."""
    if not endpoints:
        return 0.0
    if endpoints[0] == (0, 0, 0):
        # The origin contributes to l=0 only; it has no angular direction for
        # any higher partial wave.
        if l > 0:
            return 0.0
        raw = np.vdot(weights, np.ones((len(weights), len(weights))) @ weights)
    else:
        positions = np.asarray(endpoints, dtype=float)
        radius = math.sqrt(sum(component * component for component in endpoints[0]))
        cosine = (positions @ positions.T) / (radius * radius)
        cosine = np.clip(cosine, -1.0, 1.0)
        legendre = np.polynomial.legendre.legval(
            cosine, [0.0] * l + [1.0]
        )
        raw = (2 * l + 1) * np.vdot(weights, legendre @ weights)

    if abs(float(raw.imag)) > epsilon_scale:
        raise ValueError(
            f"l={l} strength has non-negligible imaginary roundoff: {raw.imag}"
        )
    strength = float(raw.real)
    if strength < 0.0:
        if abs(strength) <= epsilon_scale:
            return 0.0
        raise ValueError(
            f"l={l} partial-wave strength is negative beyond roundoff: {strength}"
        )
    return strength


def analyze_row(row: Mapping[str, Any], *, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Analyze one catalog row, returning a JSON-ready row report."""
    tolerance = _finite_real(tolerance, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not isinstance(row, Mapping):
        raise ValueError("row must be an object")

    report: dict[str, Any] = {
        **_identity(row),
        "shells": [],
        "lower_contamination": [],
        "leading_l": None,
        "passed": False,
        "status": "failed",
        "reasons": [],
    }
    if row.get("kind") != "ordinary":
        report["status"] = "unsupported"
        report["reasons"].append(
            f"kind {row.get('kind')!r} is not supported by the ordinary-row partial-wave check"
        )
        return report

    target_l = row.get("target_l")
    if isinstance(target_l, bool) or not isinstance(target_l, int) or target_l < 0:
        raise ValueError("ordinary row target_l must be a non-negative integer")

    weights_by_endpoint = endpoint_weights(row)
    by_radius: defaultdict[int, list[tuple[tuple[int, int, int], complex]]] = defaultdict(list)
    for endpoint, weight in weights_by_endpoint.items():
        if weight == 0j:
            continue
        radius_squared = sum(component * component for component in endpoint)
        by_radius[radius_squared].append((endpoint, weight))

    row_lower_contamination: list[dict[str, Any]] = []
    row_leading_ls: list[int] = []
    target_present = False
    all_shells_passed = True
    for radius_squared in sorted(by_radius):
        entries = sorted(by_radius[radius_squared])
        shell_endpoints = [endpoint for endpoint, _ in entries]
        shell_weights = np.asarray([weight for _, weight in entries], dtype=np.complex128)
        epsilon_scale = tolerance * float(np.vdot(shell_weights, shell_weights).real)

        strengths: dict[str, float] = {}
        nonzero_ls: list[int] = []
        for l in range(target_l + 1):
            strength = _strength_for_shell(
                shell_endpoints, shell_weights, l, epsilon_scale
            )
            if strength <= epsilon_scale:
                strength = 0.0
            else:
                nonzero_ls.append(l)
            strengths[str(l)] = strength
        if nonzero_ls:
            row_leading_ls.append(min(nonzero_ls))
        if strengths[str(target_l)] > epsilon_scale:
            target_present = True

        shell_lower: list[dict[str, Any]] = []
        shell_passed = True
        for l in range(target_l):
            strength = strengths[str(l)]
            contaminated = strength > epsilon_scale
            shell_lower.append(
                {
                    "l": l,
                    "strength": strength,
                    "eps": epsilon_scale,
                    "contaminated": contaminated,
                }
            )
            if contaminated:
                shell_passed = False
                row_lower_contamination.append(
                    {
                        "radius_squared": radius_squared,
                        "l": l,
                        "strength": strength,
                        "eps": epsilon_scale,
                    }
                )
        all_shells_passed = all_shells_passed and shell_passed
        report["shells"].append(
            {
                "radius_squared": radius_squared,
                "epsilon_scale": epsilon_scale,
                "strengths": strengths,
                "lower_contamination": shell_lower,
                "leading_l": min(nonzero_ls) if nonzero_ls else None,
                "target_present": strengths[str(target_l)] > epsilon_scale,
                "passed": shell_passed,
            }
        )

    report["lower_contamination"] = row_lower_contamination
    report["leading_l"] = min(row_leading_ls) if row_leading_ls else None
    if not target_present:
        report["reasons"].append(
            f"target_l={target_l} has no strength above the shell epsilon scale"
        )
    if row_lower_contamination:
        report["reasons"].append("one or more shells contain lower-l contamination")

    report["passed"] = target_present and all_shells_passed
    report["status"] = "passed" if report["passed"] else "failed"
    return report


def analyze_catalog(
    catalog: Mapping[str, Any], *, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    """Analyze all rows in a decoded catalog and return the report object."""
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must be an object")
    rows = catalog.get("rows")
    if not isinstance(rows, list):
        raise ValueError("catalog must contain a rows array")

    row_reports = [analyze_row(row, tolerance=tolerance) for row in rows]
    passed_ids = [row["id"] for row in row_reports if row["status"] == "passed"]
    failed_ids = [row["id"] for row in row_reports if row["status"] == "failed"]
    unsupported_ids = [row["id"] for row in row_reports if row["status"] == "unsupported"]
    return {
        "schema": SCHEMA,
        "version": REPORT_VERSION,
        "definition": {
            "source": "terms only; direction codes 0/1/2=+x/+y/+z and 3/4/5=-x/-y/-z",
            "endpoint_weights": "complex coefficients are summed at identical endpoints",
            "radial_shells": "endpoints are grouped by integer radius_squared, with no cross-shell sums",
            "strength": "S_l=(2*l+1)*w.conj() @ P_l(cosine) @ w",
            "l_range": "each ordinary row is evaluated for l=0..target_l inclusive",
            "epsilon_scale": "per shell: tolerance * sum(abs(endpoint_weight)**2)",
            "default_tolerance": DEFAULT_TOLERANCE,
            "zero_rule": "strengths with abs(strength) <= epsilon_scale are reported as zero",
            "negative_rule": "negative strengths are zero only when abs(strength) <= epsilon_scale; otherwise analysis raises",
            "pass_rule": "target_l must be nonzero in at least one shell and every lower-l strength must be at or below that shell's epsilon_scale",
            "local_terms": "the origin contributes |w|^2 to l=0 and zero to l>0; for target_l > 0 this is reported as lower-l contamination",
            "unsupported_kinds": "every kind other than ordinary is explicitly unsupported",
        },
        "tolerance": tolerance,
        "rows": row_reports,
        "passed_ids": passed_ids,
        "failed_ids": failed_ids,
        "unsupported_ids": unsupported_ids,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"catalog JSON (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the full report JSON",
    )
    args = parser.parse_args(argv)

    try:
        catalog_bytes = args.catalog.read_bytes()
        catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        catalog = json.loads(catalog_bytes.decode("utf-8"))
        report = analyze_catalog(catalog)
        report["source"] = {
            "catalog_sha256": catalog_sha256,
            "catalog_path": str(args.catalog.resolve()),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(f"partial-wave check failed: {error}", file=sys.stderr)
        return 1

    print(
        "Partial-wave check: "
        f"passed={len(report['passed_ids'])}, "
        f"failed={len(report['failed_ids'])}, "
        f"unsupported={len(report['unsupported_ids'])}"
    )
    if report["failed_ids"]:
        print("failed ids: " + ", ".join(map(str, report["failed_ids"])))
    if report["unsupported_ids"]:
        print("unsupported ids: " + ", ".join(map(str, report["unsupported_ids"])))
    if report["failed_ids"]:
        return 1
    if report["unsupported_ids"]:
        return 2
    return 0 if len(report["passed_ids"]) == len(report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
