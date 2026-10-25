"""Only static native resource proofs survive an execution owner scope."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.execution import storage_runtime, bound_program
from lqcdweave.contraction.execution.binding_cache import (
    ExecutionBindingCache, native_resource_audit_binding,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.planning.api import has_native_catalog
from correlator.execution.test_debug_execution import _make_dense_execution
from correlator.execution.test_cartesian_group_execution import fixture, Backend
from correlator.execution.test_scalar_group_execution import ArrayLoader


def audit(plan, backend, cache):
    with cache.scope(plan, backend):
        storage_runtime._validate_static_resources(plan, backend)


def test_native_audit_once_per_plan_backend_budget_and_debug():
    plan, _ = fixture()
    cache = ExecutionBindingCache()
    backend = object()
    assert native_resource_audit_binding(plan)
    with patch.object(storage_runtime, 'validate_executable_resources',
                      wraps=storage_runtime.validate_executable_resources) as checked:
        audit(plan, backend, cache)
        audit(plan, backend, cache)
        assert checked.call_count == 1
        with execution_debug(True):
            audit(plan, backend, cache)
            audit(plan, backend, cache)
        assert checked.call_count == 3
        audit(plan, backend, cache)
        assert checked.call_count == 3
        audit(plan, object(), cache)
        assert checked.call_count == 4
        for changed in (replace(plan), replace(plan, memory_limit=plan.memory_limit + 1),
                        plan.with_reserves(reserved_device_bytes=512)):
            audit(changed, backend, cache)
            audit(changed, backend, cache)
        assert checked.call_count == 7


@pytest.mark.parametrize('field', ['memory', 'point', 'events', 'extension', 'catalog', 'action'])
def test_mutable_or_extension_audit_descriptions_are_never_cached(field):
    plan, _ = fixture()
    if field in ('memory', 'events'):
        plan = replace(plan, **{field: list(getattr(plan, field))})
    elif field == 'point':
        class ExtensionPoint(type(plan.memory[0])):
            pass
        plan = replace(plan, memory=(ExtensionPoint(**vars(plan.memory[0])), *plan.memory[1:]))
    elif field == 'extension':
        plan = SimpleNamespace(**vars(plan))
    elif field == 'catalog':
        class MutableCatalog(type(plan.state.catalog)):
            def __getitem__(self, key):
                return current[key]
        current = dict(plan.state.catalog)
        state = replace(plan.state, catalog=MutableCatalog(current))
        assert not has_native_catalog(state)
        plan = replace(plan, state=state)
    else:
        actions = list(plan.state.actions)
        actions[0] = replace(actions[0], left_axes=[-3, -2, -1])
        plan = replace(plan, state=replace(plan.state, actions=tuple(actions)))
    assert not native_resource_audit_binding(plan)
    cache = ExecutionBindingCache()
    with patch.object(storage_runtime, 'validate_executable_resources',
                      wraps=storage_runtime.validate_executable_resources) as checked:
        for _ in range(2):
            audit(plan, np, cache)
        assert checked.call_count == 2
    assert cache._binding is None
    if field == 'memory':
        index = next(i for i, p in enumerate(plan.memory) if p.cartesian_reduction_device_bytes)
        plan.memory[index] = replace(plan.memory[index], cartesian_reduction_device_bytes=0)
        with pytest.raises(ValueError, match='resource|Cartesian'):
            audit(plan, np, cache)
    if field == 'catalog':
        key = plan.kernel_batches[0].left[0].tensor
        current[key] = replace(current[key], shape=(79,))
        with pytest.raises(ValueError):
            audit(plan, np, cache)


def test_invalid_first_audit_and_failed_first_scope_never_publish():
    plan, _ = fixture()
    cache = ExecutionBindingCache()
    group, = plan.cartesian_reduction_groups
    bad = replace(plan, cartesian_reduction_groups=(replace(group, workspace_bytes=1),))
    for _ in range(2):
        with pytest.raises(ValueError, match='workspace'):
            audit(bad, np, cache)
        assert cache._binding is None
    with pytest.raises(RuntimeError, match='later binding failed'):
        with cache.scope(plan, np):
            storage_runtime._validate_static_resources(plan, np)
            raise RuntimeError('later binding failed')
    assert cache._binding is None


def test_public_cpu_replay_checks_current_inputs_loader_and_feasibility():
    _, path, loader, expected = _make_dense_execution()
    plan = path.plan
    cache = ExecutionBindingCache()
    assert native_resource_audit_binding(plan)
    def run(current=loader, selected=plan):
        return storage_runtime.replay_scene_storage(selected, current, execution_cache=cache,
                                                    io_overlap=False)
    with patch.object(storage_runtime, 'validate_executable_resources',
                      wraps=storage_runtime.validate_executable_resources) as checked:
        np.testing.assert_allclose(run()[0], expected)
        key = next(iter(loader.arrays))
        loader.arrays[key] *= 2
        np.testing.assert_allclose(run()[0], expected * 2)
        assert checked.call_count == 1
        old = loader.arrays[key]
        for bad in (old.reshape(-1)[:-1], old.astype(np.complex64)):
            loader.arrays[key] = bad
            with pytest.raises(ValueError, match='mismatched tensor'):
                run()
        loader.arrays[key] = old
        with pytest.raises(TypeError):
            run(SimpleNamespace())
        with pytest.raises(MemoryError, match='infeasible'):
            run(selected=replace(plan, feasible=False))
        assert checked.call_count == 1
        np.testing.assert_allclose(run()[0], expected * 2)


def test_grouped_public_protocol_reuses_audit_and_preserves_current_input_checks():
    plan, values = fixture()
    backend, cache = Backend(), ExecutionBindingCache()
    def run(current):
        return storage_runtime.replay_scene_storage(plan, ArrayLoader(current, backend),
            backend=backend, io_overlap=False, execution_cache=cache)
    with patch.object(bound_program, 'bind_output_layout', return_value=None), patch.object(
            storage_runtime, 'validate_executable_resources',
            wraps=storage_runtime.validate_executable_resources) as checked:
        first = run(values)
        second = run({k: v * 2 for k, v in values.items()})
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(4 * a, b)
        assert checked.call_count == 1
        key = next(iter(values))
        for bad in (values[key][:-1], values[key].astype(np.complex128)):
            current = dict(values, **{key: bad})
            with pytest.raises(ValueError, match='mismatched tensor'):
                run(current)
        assert checked.call_count == 1
        with execution_debug(True):
            run(values)
        assert checked.call_count > 1


def test_nonstreaming_batches_share_static_proof_but_no_scope_does_not():
    _, path, loader, expected = _make_dense_execution()
    cache = ExecutionBindingCache()
    with patch.object(storage_runtime, 'validate_executable_resources',
                      wraps=storage_runtime.validate_executable_resources) as checked:
        for _ in range(2):
            received = []
            storage_runtime.replay_storage_batches((path.plan,), loader,
                lambda i, value: received.append(value), io_overlap=False, execution_cache=cache)
            np.testing.assert_allclose(received[0][0], expected)
        assert checked.call_count == 1
        for _ in range(2):
            storage_runtime._validate_static_resources(path.plan, np)
        assert checked.call_count == 3


def test_calculation_run_uses_the_common_static_proof():
    from test_common_execution_bindings import CommonExecutionBindingsTests
    calculation = CommonExecutionBindingsTests().calculation()
    with patch.object(storage_runtime, 'validate_executable_resources',
                      wraps=storage_runtime.validate_executable_resources) as checked:
        first = calculation.run().value
        initial = checked.call_count
        assert initial > 0
        np.testing.assert_array_equal(calculation.run().value, first)
        assert checked.call_count == initial
        np.testing.assert_allclose(calculation.run(debug=True).value, first)
        assert checked.call_count > initial
