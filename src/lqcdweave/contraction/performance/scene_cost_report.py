"""Metadata-only reporting for selected compute plans.

The report describes the operation, snapshot, and source-preparation model.
It retains physical byte and lifetime metadata without assigning time costs to
reads, transfers, or temporary resource moves.
"""

from dataclasses import asdict
from math import fsum, isclose
from typing import Iterable, cast

from lqcdweave.contracts.correlator.api import (
    SceneStoragePlan, StateDescriptor, validate_executable_resources, DenseGemmGroup,
)
from .model import CostProfile, cost_profile_calibration_status


def predict_plan_cost(
    plans: Iterable[SceneStoragePlan], profile: CostProfile, *, workload, provenance
):
    """Summarize modeled preparation and numerical work for selected plans."""
    plans = tuple(plans)
    if not plans:
        raise ValueError("at least one storage plan is required")
    parts = {key: [] for key in (
        "source_preparation", "logical_snapshot", "dense_contraction",
        "sparse_contraction", "baryon_contraction", "release",
    )}
    kinds = dict(load="logical_snapshot", contract="dense_contraction",
                 sparse_bridge="sparse_contraction", baryon_join="baryon_contraction",
                 release="release")
    resource_kinds = ("discard", "spill_host", "spill_disk", "restore_host", "restore_device")
    resource_counts = {kind: 0 for kind in resource_kinds}
    opened, identities = set(), set()
    read_bytes = transfer_bytes = 0
    for plan in plans:
        state = cast(StateDescriptor, plan.state)
        if plan.cost_profile != profile:
            raise ValueError("plan profile is absent or differs from the supplied profile; rebuild plans")
        for event in plan.events:
            if event.kind == "release":
                continue
            if event.kind != "read":
                raise ValueError(f"unknown storage event {event.kind!r}")
            block = plan.blocks[event.block]
            identity = (block.dataset, block.selection)
            if event.block in opened or identity in identities:
                raise ValueError("cost reporting requires globally joined physical lifetimes")
            opened.add(event.block)
            identities.add(identity)
            read_bytes += block.read_bytes
            transfer_bytes += block.transfer_bytes

        prep = fsum(plan.preparations[action.tensor].seconds(profile)
                    for action in state.actions if action.kind == "load" and not action.reused)
        if not isclose(prep, plan.preparation_seconds, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("preparation cost does not match the supplied profile")
        parts["source_preparation"].append(plan.preparation_seconds)
        action_seconds = []
        for action in state.actions:
            if action.kind in ("read_block", "close_block", *resource_kinds):
                if action.seconds != 0.0:
                    raise ValueError("storage lifecycle actions cannot carry predicted time costs")
                if action.kind in resource_counts:
                    resource_counts[action.kind] += 1
                continue
            category = kinds.get(action.kind)
            if category is None:
                raise ValueError(f"unknown modeled action {action.kind!r}")
            parts[category].append(action.seconds)
            action_seconds.append(action.seconds)
        if not isclose(fsum(action_seconds), state.seconds, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("modeled action costs do not match the selected state")
        if plan.kernel_batches:
            validate_executable_resources(plan)
            parts["dense_contraction"].append(plan.kernel_cost_adjustment)

    parts = {key: fsum(values) for key, values in parts.items()}
    total = fsum(parts.values())
    if not isclose(total, fsum(plan.seconds for plan in plans), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("modeled compute cost breakdown does not equal plan total")
    report = dict(
        format="easy-distillation.scene-compute-cost-prediction", version=2,
        profile=asdict(profile), profile_version=profile.version,
        hardware_fingerprint=profile.hardware_fingerprint, sample_count=profile.sample_count,
        calibration_status=cost_profile_calibration_status(profile),
        workload=dict(workload), provenance=dict(provenance),
        total_seconds=total, parts_seconds=parts,
        parts_fraction={key: value / total if total else 0.0 for key, value in parts.items()},
        batches=len(plans), physical_reads=len(opened), read_bytes=read_bytes,
        dense_kernel_batches=sum(len(plan.kernel_batches) for plan in plans),
        transfer_bytes=transfer_bytes, resource_action_counts=resource_counts,
        retained_releases=sum(len(plan.retained_releases) for plan in plans),
        retained_source_hits=sum(
            action.kind == "load" and action.reused
            for plan in plans for action in cast(StateDescriptor, plan.state).actions
        ),
        retained_peak_bytes=max((point.retained_host_bytes + point.retained_device_bytes
                                 for plan in plans for point in plan.memory), default=0),
        feasible=all(plan.feasible for plan in plans),
        peak_host_bytes=max(plan.peak_host_bytes for plan in plans),
        peak_device_bytes=max(plan.peak_device_bytes for plan in plans),
        budgets=[dict(backend=cast(StateDescriptor, plan.state).problem.backend,
                      memory_limit=plan.memory_limit, host_memory_limit=plan.host_memory_limit,
                      reserved_host_bytes=plan.reserved_host_bytes,
                      reserved_device_bytes=plan.reserved_device_bytes) for plan in plans],
        scope="Modeled snapshots, source preparation, and numerical operations; excludes physical IO, resource movement, search, input hashing, and publication",
    )
    if any(plan.explicit_resources for plan in plans):
        report["peak_disk_bytes"] = max(plan.peak_disk_bytes for plan in plans)
        report["temporary_read_bytes"] = sum(
            action.resource.nbytes for plan in plans for action in cast(StateDescriptor, plan.state).actions
            if action.kind == "restore_host" and action.resource is not None
        )
        report["temporary_write_bytes"] = sum(
            action.resource.nbytes for plan in plans for action in cast(StateDescriptor, plan.state).actions
            if action.kind == "spill_disk" and action.resource is not None
        )
        for budget, plan in zip(report["budgets"], plans, strict=True):
            budget["disk_limit_bytes"] = plan.disk_limit_bytes
    groups = [batch for plan in plans for batch in plan.kernel_batches if isinstance(batch, DenseGemmGroup)]
    if groups:
        report['kernel_models'] = {
            'cuda-cublas-gemm-batched-v1': dict(groups=len(groups),
                grouped_batches=sum(len(group.batches) for group in groups),
                calibration_status='estimated-from-dense-v1',
                description='Generic dense cost estimate; no dedicated batched-GEMM calibration.')}
    return report


__all__ = ["predict_plan_cost"]
