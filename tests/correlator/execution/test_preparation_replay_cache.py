"""Cross-run source programs retain recipes, never current readers or values."""
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.binding_cache import ExecutionBindingCache
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution import preparation as module
from correlator.execution.test_prebound_preparation import _outer_oracle


@pytest.mark.parametrize('kind', ['identity', 'constant', 'spin_spatial'])
def test_native_leaf_replay_reuses_structure_with_current_values_and_no_owners(kind):
    spin = np.eye(4, dtype=np.complex64)
    constants = () if kind == 'identity' else (ArrayConstant.from_array(spin),)
    recipe = TensorPreparation(kind, () if kind == 'constant' else ('raw',),
        constants, compute_dtype='complex64')
    prep = SourcePreparation(recipe.inputs, recipe=recipe)
    shape = (4, 4, 2, 3) if kind == 'spin_spatial' else (4, 4) if kind == 'constant' else (2, 3)
    spec = TensorSpec(shape, np.complex64)
    cache, plan = ExecutionBindingCache(), object()
    references, values = [], []
    with patch.object(module, '_bind_leaf_operation', wraps=module._bind_leaf_operation) as build:
        for scale in (1, 3):
            raw = np.full((2, 3), scale + 2j, dtype=np.complex64)
            physical = SimpleNamespace(array=raw)
            loader = module.bind_prepared_loader(physical, {'x': prep}, {'x': spec})
            with cache.scope(plan, np):
                value = loader.materialize('x', None, {'raw': raw})
            expected = (_outer_oracle(spin, raw) if kind == 'spin_spatial' else
                        spin if kind == 'constant' else raw)
            np.testing.assert_array_equal(value, expected)
            if kind != 'constant':
                assert not np.shares_memory(value, raw)
            values.append(value)
            references.extend((weakref.ref(raw), weakref.ref(loader)))
            del raw, physical, loader, expected
        assert build.call_count == 1
    gc.collect()
    assert all(ref() is None for ref in references)
    assert values[0] is not values[1] or kind == 'constant'


def test_mutable_recipe_and_spec_invalidate_run_local_constants_and_program():
    first = ArrayConstant.from_array(np.eye(4, dtype=np.float32))
    prep = SourcePreparation(('raw',), recipe=TensorPreparation('spin_spatial', ('raw',), (first,)))
    tables, specs = {'x': prep}, {'x': TensorSpec((4, 4, 2, 3), np.float32)}
    loader = module.bind_prepared_loader(None, tables, specs)
    cache, plan = ExecutionBindingCache(), object()
    raw = np.ones((2, 3), np.float32)
    with cache.scope(plan, np):
        one = loader.materialize('x', None, {'raw': raw})
        tables['x'] = replace(prep, recipe=replace(prep.recipe,
            constants=(ArrayConstant.from_array(2 * np.eye(4, dtype=np.float32)),)))
        two = loader.materialize('x', None, {'raw': raw})
        specs['x'] = TensorSpec((4, 4, 3, 2), np.float32)
        transposed = loader.materialize('x', None, {'raw': raw.T})
    np.testing.assert_array_equal(two, 2 * one)
    np.testing.assert_array_equal(transposed, two.transpose(0, 1, 3, 2))


def test_debug_uses_original_binder_and_profile_rounding_matches_reference():
    rng = np.random.default_rng(89)
    arrays = [rng.normal(size=shape).astype(np.complex64) for shape in
              ((4, 4), (3, 2), (3, 3), (3,))]
    recipe = TensorPreparation('spin_spatial', ('raw',),
        tuple(ArrayConstant.from_array(a) for a in arrays), configuration='cfg',
        conjugate=True, permutation=(1, 0), compute_dtype='complex64')
    preps = {'x': SourcePreparation(('raw',), recipe=recipe)}
    specs = {'x': TensorSpec((4, 4, 3, 2), np.complex64)}
    raw = rng.normal(size=(2, 3)).astype(np.complex64)[:, ::-1]
    cache, plan = ExecutionBindingCache(), object()
    with cache.scope(plan, np):
        actual = module.bind_prepared_loader(None, preps, specs).materialize('x', None, {'raw': raw})
    with execution_debug(True), cache.scope(plan, np), patch.object(module,
            '_bind_leaf_operation', side_effect=AssertionError('debug must use reference')):
        expected = module.bind_prepared_loader(None, preps, specs).materialize('x', None, {'raw': raw})
    np.testing.assert_array_equal(actual, expected)


def test_replacing_native_recipe_keeps_one_slot_and_rebinds_current_geometry():
    cache, plan = ExecutionBindingCache(), object()
    for size in range(1, 25):
        prep = SourcePreparation(('raw',), recipe=TensorPreparation(
            'identity', ('raw',), compute_dtype='float32'))
        raw = np.arange(size, dtype=np.float64)
        loader = module.bind_prepared_loader(None, {'x': prep},
            {'x': TensorSpec(raw.shape, np.float32)})
        with cache.scope(plan, np):
            result = loader.materialize('x', None, {'raw': raw})
        np.testing.assert_array_equal(result, raw.astype(np.float32))
    assert len(cache._binding[2]) == 1


def test_backend_change_rebinds_and_parallel_replays_do_not_share_numerical_state():
    from concurrent.futures import ThreadPoolExecutor
    from correlator.execution.test_prebound_preparation import _backend
    cache, plan, backend = ExecutionBindingCache(), object(), _backend()
    recipe = TensorPreparation('spin_spatial', ('raw',),
        (ArrayConstant.from_array(np.eye(4, dtype=np.float32)),))
    prep, spec = SourcePreparation(('raw',), recipe=recipe), TensorSpec((4, 4, 2, 3), np.float32)
    def run(scale):
        loader = module.bind_prepared_loader(None, {'x': prep}, {'x': spec}, backend=backend)
        with cache.scope(plan, backend):
            return loader.materialize('x', None, {'raw': np.full((2, 3), scale, np.float32)})
    # Warm publication; concurrent runs only share the pure operation.
    one = run(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(run, (2, 3, 4)))
    for scale, value in zip((2, 3, 4), values):
        np.testing.assert_array_equal(value, scale * one)
    backend = _backend()
    with patch.object(module, '_bind_leaf_operation', wraps=module._bind_leaf_operation) as build:
        np.testing.assert_array_equal(run(5), 5 * one)
        assert build.call_count == 1


def test_public_calculation_reuses_leaf_bindings_without_reusing_changed_propagators():
    from test_structured_correlator import fixture, task
    from lqcdweave.contraction.correlator import calculate
    binding, elemental, propagator = fixture()
    selected = calculate(task(structured=True, names=('a',)), vertex_map=binding,
        propagator_map={'S^q': propagator}, configuration='cfg', usedNe=3,
        time_axes={0: 1, 1: 0}, optimize='none', memory_limit=1 << 26, execute=False)
    with patch.object(module, '_bind_leaf_operation', wraps=module._bind_leaf_operation) as build:
        first = selected.run()
        cold_calls = build.call_count
        assert cold_calls > 0
        second = first.run()
        assert build.call_count == cold_calls
        np.testing.assert_array_equal(second.value, first.value)
        propagator.sectors['vsv'] *= 2
        third = selected.run()
        assert build.call_count == cold_calls
        np.testing.assert_allclose(third.value, 4 * first.value, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(selected.run(debug=True).value, third.value, rtol=1e-13, atol=1e-13)
    selected.close()
    np.testing.assert_array_equal(first.value, second.value)
