"""Draft the single-current V/D/P campaign; no solves, submission or contraction.

The JSON is a reviewable task specification, not an executable Calculation
archive. Numerical workers must bind actual native Dataset identities and
jointly plan the declared measurements before running it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CFGS = ("10000", "13000", "14000", "15000", "16000", "17000", "18000",
        "19000", "25000", "26000", "27000", "28000", "29000", "30000",
        "31000", "32000", "33000", "34000")
PARENT_COUNTS = (8, 10, 12, 14, 18, 23, 30, 43, 64, 106)
POINT_COUNTS = (3, 6, 9, 12, 16, 24, 32)
TIME_DIFFERENCES = tuple(range(1, 19))


def medium_columns(columns_per_group: int) -> tuple[int, ...]:
    """Select the first m actual columns of EACH saved four-column group."""
    if type(columns_per_group) is not int or columns_per_group not in (2, 3, 4):
        raise ValueError("columns_per_group must be 2, 3 or 4")
    return tuple(4 * group + column for group in range(10)
                 for column in range(columns_per_group))


def scan_cases() -> list[dict]:
    cases = []
    for nd in (48, 63, 96):
        cases.append(dict(case_id=f"VD-D{nd}", route="VD", d_columns=nd,
                          point_count=0, sampled_components=nd,
                          d_parent="complete-complement-of-V100", p_projection=None))
    # Include the P=21 matched-budget reference as well as the P scan grid.
    for npnt in sorted(set(POINT_COUNTS) | {21}):
        cases.append(dict(case_id=f"VP-P{npnt}", route="VP", d_columns=0,
                          point_count=npnt, sampled_components=3 * npnt,
                          d_parent=None, p_projection="Q100"))
    matched = {2: (14, 25), 3: (11, 22), 4: (8, 19)}
    for m in (2, 3, 4):
        for npnt in sorted(set(POINT_COUNTS) | set(matched[m])):
            cases.append(dict(case_id=f"VDP-m{m}-P{npnt}", route="VDP",
                              d_columns=10 * m, columns_per_group=m,
                              d_column_positions=list(medium_columns(m)),
                              point_count=npnt, sampled_components=10 * m + 3 * npnt,
                              d_parent="Laplace-101-through-428", p_projection="Q428"))
    return cases


def storage_budget(*, configurations: int = 18) -> dict:
    """Complex128 payloads, excluding headers, elemental and result sizes.

    The solution slice is S(0,t) V100(t), (spin_out,spin_in,x_color,mode).
    It is NOT the entire 72-time solution of every RHS.
    """
    spin_pairs, complex_bytes, spatial_colors = 16, 16, 24**3 * 3
    pairs, nm = len(TIME_DIFFERENCES), 100
    per_cfg = {
        "parent_U428_dagger_S_V100": pairs * spin_pairs * 428 * nm * complex_bytes,
        "point_anchor_and_six_neighbors_upper_bound":
            pairs * spin_pairs * (7 * 64 * 3) * nm * complex_bytes,
        "overlap_U428_anchor_and_six_neighbors_upper_bound": 428 * (7 * 64 * 3) * complex_bytes,
        "optional_current_time_solution_slices":
            pairs * spin_pairs * spatial_colors * nm * complex_bytes,
    }
    return dict(per_configuration_payload_bytes=per_cfg,
                ensemble_payload_bytes={k: n * configurations for k, n in per_cfg.items()},
                excludes=["NPY headers", "elemental", "correlators", "execution reports"],
                neighbor_budget="upper bound; deduplicate actual coordinates and use actual C3 paths")


def build_plan(*, retain_current_slices: bool = False) -> dict:
    cases = scan_cases()
    return {
        "schema": 1, "kind": "single-current-vdp-task-draft", "status": "planned_not_executed",
        "numerical_worker_included": False,
        "input_root": "/public/home/siyangchen/DATA/C24P29",
        "configurations": list(CFGS), "lattice": [24, 24, 24, 72],
        "source": {"role": "current", "physical_time": 0, "dagger": True},
        "sink": {"role": "meson", "basis": "actual-V100", "physical_times": list(TIME_DIFFERENCES)},
        "time_differences": list(TIME_DIFFERENCES),
        "measurements": [
            {"channel": "C2", "momentum": [0, 0, 0], "directions": [0]},
            {"channel": "C2p", "momentum": [0, 0, 1], "directions": [0]},
            {"channel": "C3", "momentum": [0, 0, 0], "directions": [0, 1, 2]},
        ],
        "operator_recipe": "scripts.localized_blending_measurements.measurement_operators",
        "low_modes": 100, "medium_parent_counts": list(PARENT_COUNTS),
        "medium_saved_columns_per_group": 4, "cases": cases,
        "sampling": {
            "point_candidates": 64, "color": "all-three", "spin": "all-four",
            "point_subset": "prefix of saved actual ordered P64; same labels across routes",
            "medium_subset": "per-group actual QR column prefix; preserve parent labels and time pool IDs",
            "current_external_legs": "prepare complete operator-dependent legs before subset selection",
            "c3": "include displaced endpoints even when they are outside the selected anchors",
            "weights": "native inclusion weights from selected D/P metadata; applied exactly once",
            "subset_identity": "rebase marginal D source identity and m; do not only slice original metadata",
            "higher_order_tests": "retain equality patterns and actual pool identities; no assumed independent weights",
        },
        "highmode": {
            "projectors": {"Q100": list(range(100)), "Q428": list(range(428))},
            "persistent_output": False,
            "reuse_scope": "one cfg and bounded physical-time window; all channels and P/D subsets",
            "reuse_key_fields": ["cfg", "gauge_id", "dirac", "spin_basis", "dtype",
                                 "parent_basis_id", "projection_mode_labels", "point_data_id",
                                 "ordered_point_labels", "physical_time_pair", "endpoint_roles"],
            "routes": {
                "existing_auxiliary": "raw point block minus U(point) times full-parent auxiliary; native project_highmode_block",
                "current_solution_slice": "apply Q to complete current-time solution then sample; native sample_high_points",
                "direct_projected_source": "apply Q to full point RHS then solve; native PerambulatorGenerator highmode_basis",
            },
            "selection_rule": "use compatible actual available input; record chosen route and all new RHS solves",
            "forbidden_substitute": "D20/D30/D40 must not replace the complete 428-mode subtraction",
        },
        "shared_stages": [
            {"stage": "bind", "once_per": "cfg", "work": "bind actual inputs, identities, coverage and stable parents"},
            {"stage": "operators", "once_per": "cfg and gauge recipe",
             "work": "reuse gauge/stout, actual operator tables and phase convention"},
            {"stage": "joint_plan", "once_per": "compatible scientific template",
             "work": "Hadron/Flavor -> gen_correlator; nest all channels, directions and cases; calculate(execute=False) once"},
            {"stage": "raw_inputs", "once_per": "cfg and physical-time window",
             "work": "read existing data or produce missing complete external legs; share Dirac solutions across compatible outputs"},
            {"stage": "highmode", "once_per": "cfg, time window and projector",
             "work": "derive maximum P64 in RAM; share Q428 across m=2/3/4; never publish highmode"},
            {"stage": "replay", "once_per": "saved compatible complete plan",
             "work": "run native bounded-window execution; subset labels before contraction; reuse common expressions via planner"},
            {"stage": "publish", "once_per": "cfg",
             "work": "publish correlators, per-direction values, timing and actual provenance using native Dataset services"},
        ],
        "retention": {
            "default": ["actual point and D subset labels and pools", "overlaps",
                        "raw complete-parent auxiliary if obtained", "raw operator-dependent P64 legs if obtained",
                        "unweighted base elemental and actual endpoint metadata", "compatible selected plan archive",
                        "per-direction complex correlators before averaging", "solver residuals and stage timing"],
            "optional_current_time_solution_slices": retain_current_slices,
            "optional_slice_definition": "raw S(0,t) V100(t), t=1..18; full spatial/color and both spin axes",
            "never": ["persistent highmode products", "one copy of common inputs per P/D case",
                      "full 72-time 4D solution for every RHS"],
            "storage": "native DatasetWriter, whole cfg NPY and shared dataset.json; real disk allocation before writes",
            "reuse_validation": "exact actual parent IDs, labels, physical coordinates, operator and numerical conventions",
        },
        "timing": {
            "stages": ["input_read", "gauge_stout", "elemental", "planning", "host_to_device",
                       "point_source_subtraction", "dirac_solve", "sink_projection", "highmode_projection",
                       "contraction", "device_to_host", "intermediate_write", "result_write"],
            "device_timing": "stream events; synchronize bounded stage boundaries, not every RHS or primitive",
            "report": "wall seconds, device seconds where available, bytes, RHS count, reuse hit/miss and parent IDs",
            "accounting": "shared preparation reported once; per-case marginal contraction separately; do not multiply shared costs",
        },
        "statistics": {"unit": "cfg", "source_average": False, "C3_direction_average": "within cfg",
                       "error": "real-part sample standard deviation / sqrt(Ncfg)",
                       "aggregate": "RMS of standard-error ratios over dt=1..18 with a common channel reference",
                       "ratio_uncertainty": "paired delete-one-cfg jackknife",
                       "incomplete": "common cfg and dt only, explicitly recorded; never zero-fill"},
        "storage_budget": storage_budget(),
        "execution_gates": [
            "Actual full-parent auxiliary OR complete current-time solution slices OR newly solved projected point sources",
            "Actual complete C3 displaced external legs; anchor-only P64 data do not establish this",
            "Native per-group marginal D selection and one-leg external-current binding verified against an independent reference",
            "Native joint planning/replay tested for all declared cases; this JSON is not a Calculation plan archive",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON path outside the source tree")
    parser.add_argument("--retain-current-slices", action="store_true",
                        help="request optional full-spatial current-time slices; inspect budget before production")
    args = parser.parse_args(argv)
    plan = build_plan(retain_current_slices=args.retain_current_slices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {len(plan['cases'])} cases to {args.output}; draft only, no numerical execution or submission.")


if __name__ == "__main__":
    main()
