from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lqcdweave.contracts.correlator.api import ExecutionBudget, InputRef, NodeRef
from lqcdweave.contracts.memory import (
    BudgetError,
    BudgetExceededError,
    MemoryEstimate,
    MemoryLimits,
)
from lqcdweave.contraction.planning.memory import (
    estimate_plan_memory,
    resolve_budget_limits,
    user_memory_cap,
    workspace_reserve,
)


@dataclass(frozen=True)
class _Spec:
    shape: tuple[int, ...]
    dtype: str


def test_contract_types_keep_fields_peak_and_backend_selection():
    estimate = MemoryEstimate(
        input_bytes=1,
        snapshot_bytes=2,
        intermediate_bytes=3,
        output_bytes=4,
        assembly_extra_bytes=5,
        workspace_bytes=6,
    )
    assert estimate.logical_peak_bytes == estimate.peak_bytes == 21
    assert estimate.as_dict() == {
        "input_bytes": 1,
        "snapshot_bytes": 2,
        "intermediate_bytes": 3,
        "output_bytes": 4,
        "assembly_extra_bytes": 5,
        "workspace_bytes": 6,
        "logical_peak_bytes": 21,
    }
    with pytest.raises(TypeError, match="input_bytes"):
        MemoryEstimate(input_bytes=True)
    with pytest.raises(ValueError, match="output_bytes"):
        MemoryEstimate(output_bytes=-1)

    limits = MemoryLimits(
        host_bytes=100,
        device_bytes=60,
        host_available_bytes=140,
        device_free_bytes=90,
        device_total_bytes=120,
        device_reserve_bytes=30,
    )
    assert limits.limit_for("numpy") == 100
    assert limits.limit_for("CuPy") == 60
    assert limits.limit_for(SimpleNamespace(__name__="cupy")) == 60
    assert issubclass(BudgetExceededError, BudgetError)
    assert issubclass(BudgetExceededError, MemoryError)


def test_estimate_plan_memory_counts_last_use_snapshots_and_workspace():
    specs = {
        "a": _Spec((2, 3), "float32"),
        "b": _Spec((3, 4), "float32"),
        "c": _Spec((4, 1), "int16"),
    }
    operation0 = SimpleNamespace(
        index=0,
        inputs=(InputRef("a"), InputRef("b")),
        output_shape=(2, 4),
        output_dtype="float64",
    )
    operation1 = SimpleNamespace(
        index=1,
        inputs=(NodeRef(0), InputRef("c")),
        output_shape=(2, 1),
        output_dtype="complex64",
    )
    term = SimpleNamespace(
        output=NodeRef(1),
        output_shape=(2, 1),
        output_dtype="complex64",
    )
    plan = SimpleNamespace(specs=specs, operations=(operation0, operation1), terms=(term,))

    estimate = estimate_plan_memory(
        plan,
        budget=SimpleNamespace(workspace_bytes=7),
        assembly_extra_bytes=5,
    )

    # 80 input bytes; node 0 remains live through op 1, peaking at 80;
    # op 0 needs a 72-byte packed operand plus a 64-byte eager result.
    assert type(estimate) is MemoryEstimate
    assert estimate.input_bytes == 80
    assert estimate.snapshot_bytes == 80
    assert estimate.intermediate_bytes == 80
    assert estimate.output_bytes == 16
    assert estimate.assembly_extra_bytes == 5
    assert estimate.workspace_bytes == 143
    assert estimate.logical_peak_bytes == 404

    without_snapshot = estimate_plan_memory(
        plan,
        budget=SimpleNamespace(workspace_bytes=7),
        output_bytes=3,
        snapshot_inputs=False,
    )
    assert without_snapshot.snapshot_bytes == 0
    assert without_snapshot.output_bytes == 3
    assert without_snapshot.workspace_bytes == 143


def test_budget_field_helpers_validate_declarations():
    budget = SimpleNamespace(host_bytes=90, device_bytes=70, workspace_bytes=11)
    assert user_memory_cap(budget, kind="host") == 90
    assert user_memory_cap(budget, kind="device") == 70
    assert workspace_reserve(budget) == 11
    assert user_memory_cap(None, kind="host") is None
    assert workspace_reserve(None) == 0
    with pytest.raises(ValueError, match="kind"):
        user_memory_cap(budget, kind="other")
    with pytest.raises(TypeError, match="host memory cap"):
        user_memory_cap(SimpleNamespace(host_bytes=True), kind="host")


def test_pure_budget_limits_apply_cpu_caps_and_unknown_observations():
    cpu = resolve_budget_limits(
        SimpleNamespace(host_bytes=400, device_bytes=33),
        target="cpu",
        host_available=1_000,
        device_info=(1,),  # The legacy CPU path ignores device observations.
    )
    assert cpu.host_bytes == 400  # min(user cap, 70% of observed host)
    assert cpu.device_bytes == 33
    assert cpu.host_available_bytes == 1_000
    assert cpu.device_free_bytes is None

    declared_host_only = resolve_budget_limits(
        SimpleNamespace(host_bytes=42),
        target="cpu",
        host_available=None,
    )
    assert declared_host_only.host_bytes == 42
    assert declared_host_only.host_available_bytes is None

    with pytest.raises(BudgetError, match="available host memory"):
        resolve_budget_limits(None, target="cpu", host_available=None)
    with pytest.raises(ValueError, match="target"):
        resolve_budget_limits(None, target="dcu", host_available=10)


def test_gpu_limits_keep_legacy_reserve_truncation_and_zero_total():
    budget = SimpleNamespace(
        host_bytes=2_000,
        device_bytes=5_000,
        reserve_fraction=0.25,
        min_reserve_bytes=10,
    )
    limits = resolve_budget_limits(
        budget,
        target="gpu",
        host_available=1_000,
        device_info=(800, 2_000),
    )
    assert limits.host_bytes == 700
    assert limits.device_bytes == 300
    assert limits.device_free_bytes == 800
    assert limits.device_total_bytes == 2_000
    assert limits.device_reserve_bytes == 500

    fractional = ExecutionBudget(reserve_fraction=0.1, min_reserve_bytes=0)
    one_byte = resolve_budget_limits(
        fractional,
        target="gpu",
        host_available=10,
        device_info=(1, 1),
    )
    assert one_byte.device_reserve_bytes == 0
    assert one_byte.device_bytes == 1
    # C's declared helper currently uses ceil and rejects total=0. This
    # prototype keeps the old root resolver's truncation and zero-total input.
    assert fractional.device_reserve(1) == 1

    zero_total = resolve_budget_limits(
        SimpleNamespace(reserve_fraction=0.1, min_reserve_bytes=0),
        target="gpu",
        host_available=10,
        device_info=(0, 0),
    )
    assert zero_total.device_reserve_bytes == 0
    assert zero_total.device_bytes == 0
    with pytest.raises(ValueError, match="total_bytes"):
        fractional.device_reserve(0)

    with pytest.raises(BudgetError, match="available GPU memory"):
        resolve_budget_limits(None, target="gpu", host_available=10)
    with pytest.raises(ValueError, match="pair"):
        resolve_budget_limits(
            None,
            target="gpu",
            host_available=10,
            device_info=(1,),
        )
