"""Allocator headroom reduces admission width before numerical I/O."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from correlator.planning.test_coordinate_inputs import fixture, materialize, evaluate
from correlator.planning.test_coordinate_path import selected
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.planning.api import (
    CostProfile, logical_source_problem, plan_coordinate_batches, plan_scene_storage,
)


def template(backend='cupy', caller_reserve=0):
    raw, requests, preparations, specs, _, metadata, readers, _ = fixture(backend)
    state = selected(logical_source_problem(raw.state.problem))
    plan = plan_scene_storage(state, raw.blocks, preparations, profile=CostProfile(),
        memory_limit=1 << 20, reserved_device_bytes=caller_reserve)
    coordinates, _ = describe_coordinates(time_axes={'source': 3, 'sink': range(40)})
    return plan, coordinates, requests, preparations, specs, metadata, readers


def schedule(values, *, cap=16000, reserve=0):
    plan, coordinates, requests, preparations, specs, metadata, _ = values
    return plan_coordinate_batches(plan, SimpleNamespace(term_ids=('loop', 'static')),
        coordinates, requests, preparations, specs, read_metadata=metadata,
        memory_limit=cap, allocator_reserve_bytes=reserve)


def test_reserve_rejects_sixteen_and_selects_eight_before_reads_with_equal_algebra():
    values = template()
    baseline = schedule(values)
    reserved = schedule(values, reserve=4000)
    assert baseline.width == 16
    assert reserved.width == 8
    assert reserved.statistics['allocator_reserve_bytes'] == 4000
    assert reserved.statistics['hard_device_limit_bytes'] == 16000
    assert reserved.statistics['numerical_device_limit_bytes'] == 12000
    assert 'not a mathematical' in reserved.statistics['allocator_reserve_policy']
    assert any(row['width'] == 16 and not row['feasible']
               for row in reserved.statistics['candidate_attempts'])
    readers = values[-1]
    assert sum(reader.reads for reader in readers.values()) == 0
    for plan in reserved.plans:
        assert plan.memory_limit == 16000
        assert plan.peak_device_bytes <= plan.memory_limit
        assert plan.reserved_device_bytes == 4000
        assert all(point.reserved_device_bytes == 4000 for point in plan.memory)
        # X's existing spare calculation must see the reserve inside peak.
        from lqcdweave.contraction.execution.read_ahead import _resolve_device_capacity
        spare = _resolve_device_capacity(plan, None)
        assert spare + (plan.peak_device_bytes-4000) <= 12000
    actual = []
    for batch in reserved.batches:
        # The production replay binds all batch requests together. This local
        # independent NumPy reference loads only the present batch closure.
        local = replace(batch, inputs=replace(batch.inputs,
            blocks={key: batch.blocks[key] for key in batch.requests}, read_requests=batch.requests))
        computed = evaluate(batch.problem, materialize(local, readers))
        actual.extend(np.atleast_1d(computed[0]))
    expected = [np.einsum('ij,jk,kl,li->', readers['forward'].arrays[sink, 3],
        readers['sink'].arrays[sink,], readers['reverse'].arrays[3, sink],
        readers['source'].arrays[3,]) for sink in range(40)]
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)


def test_adds_allocator_to_existing_caller_reserve_once():
    values = template(caller_reserve=512)
    result = schedule(values, reserve=3200)
    assert values[0].reserved_device_bytes == 512
    assert all(plan.reserved_device_bytes == 3712 for plan in result.plans)
    assert result.statistics['caller_reserved_device_bytes'] == 512


@pytest.mark.parametrize('reserve', [-1, True, 1.5])
def test_invalid_reserve_is_rejected_before_coordinate_binding(reserve):
    values = template()
    with patch('lqcdweave.contraction.planning.coordinate_schedule.bind_coordinate_inputs',
               side_effect=AssertionError('unexpected binding')):
        with pytest.raises(ValueError):
            schedule(values, reserve=reserve)


def test_exhausted_reserve_fails_before_coordinate_binding_and_reads():
    values = template(caller_reserve=512)
    with patch('lqcdweave.contraction.planning.coordinate_schedule.bind_coordinate_inputs',
               side_effect=AssertionError('unexpected binding')):
        with pytest.raises(MemoryError, match='reserves exhaust'):
            schedule(values, reserve=15500)
    assert sum(reader.reads for reader in values[-1].values()) == 0


def test_cpu_zero_default_keeps_selection_and_nonzero_device_reserve_is_invalid():
    values = template('numpy')
    result = schedule(values, cap=1 << 20)
    assert result.width == 40
    assert result.statistics['allocator_reserve_bytes'] == 0
    assert result.statistics['hard_device_limit_bytes'] == 0
    assert all(plan.peak_device_bytes == 0 for plan in result.plans)
    with pytest.raises(ValueError, match='CPU coordinate'):
        schedule(values, reserve=1)
