"""Whole merged vertices retain current values and independent result owners."""
from dataclasses import replace
import gc
import os
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    AxisPattern, CoreComponent, TensorCoreSpec, TensorSpec, VertexKernelDescription,
)
from lqcdweave.contracts.correlator.cores import baryon_component_pattern
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contraction.execution.operations import baryon_join, baryon_component_sum


def timed(phase, call, **kwargs):
    return call()


def fixture(backend, dtype, layout, batch=None):
    rng = np.random.default_rng(741)
    pattern = AxisPattern(((22, 18, 14, 23, 19, 15),
        (10, 11, 18, 19, 22, 23), (2, 3, 6, 7, 14, 15)), (10, 11, 2, 3, 6, 7))
    sizes = {axis: (4 if axis in (22, 18, 14, 10, 2, 6) else 2)
             for axes in pattern.inputs for axis in axes}
    if layout == 'permuted':
        pattern = AxisPattern((pattern.inputs[0], tuple(reversed(pattern.inputs[1])),
                              tuple(reversed(pattern.inputs[2]))), tuple(reversed(pattern.output)))
    if batch is not None:
        pattern = AxisPattern(tuple((30, *axes) for axes in pattern.inputs), (30, *pattern.output))
        sizes[30] = 2
    def array(shape):
        if layout == 'strided':
            shape = tuple(2 * size for size in shape)
        value = backend.asarray((rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype))
        if layout == 'strided':
            value = value[tuple(slice(None, None, -2) for _ in shape)]
        return value
    arrays, parts = {}, []
    for index, coefficient in enumerate((.7+.2j, -.4j, -.3)):
        arrays[f'S{index}'] = array((4, 4, 4))
        arrays[f'E{index}'] = array((2, 2, 2, 2) if batch == 'dynamic' else (2, 2, 2))
        parts.append(CoreComponent((f'S{index}', f'E{index}'), coefficient))
    for name, axes in zip(('P', 'Q'), pattern.inputs[1:]):
        arrays[name] = array(tuple(sizes[axis] for axis in axes))
    axes, shape = ('a', 'b', 'c', 'i', 'j', 'k'), (4, 4, 4, 2, 2, 2)
    if batch is not None:
        axes, shape = ('lane', *axes), (2, *shape)
    core = TensorCoreSpec('baryon', axes, TensorSpec(shape, dtype), tuple(parts))
    return SimpleNamespace(reused=False, equation=pattern, core=core,
        operands=core.inputs + ('P', 'Q'), tensor='result', kernel_path=(),
        kernel_description=VertexKernelDescription('baryon', 'ordered-join-v1')), arrays


@pytest.mark.parametrize('backend_name', ('numpy', 'cupy'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('layout', ('contiguous', 'strided', 'permuted'))
@pytest.mark.parametrize('batch', (None, 'static', 'dynamic'))
def test_component_sum_against_independent_dense_reference(backend_name, dtype, layout, batch):
    if backend_name == 'cupy':
        if os.environ.get('LQCDWEAVE_TEST_GPU') != '1':
            pytest.skip('requires an explicitly allocated GPU')
        import cupy as backend
        host = backend.asnumpy
    else:
        backend, host = np, np.asarray
    action, arrays = fixture(backend, dtype, layout, batch)
    equation = baryon_component_pattern(action.equation.inputs[0],
        action.equation.inputs[1:], action.equation.output, arrays['E0'].ndim)
    previous = None
    for repeat in range(2):
        if repeat:
            for value in arrays.values():
                value *= .8+.1j
        snapshots = {key: host(value).copy() for key, value in arrays.items()}
        expected = sum(part.coefficient * np.einsum(*einsum_arguments(equation,
            tuple(snapshots[key].astype('complex128') for key in part.inputs + ('P', 'Q'))),
            optimize=False) for part in action.core.components)
        prepared = baryon_component_sum.prepare_component_sum(action, arrays, backend, np.dtype(dtype))
        assert prepared is not None
        prepared.close()
        assert prepared.components == prepared.legs == ()
        with patch.object(baryon_join, 'prepare_component_sum', return_value=None):
            baryon_join.contract_baryon_join(action, arrays, backend, timed=timed,
                                            device=backend_name == 'cupy')
        baseline = host(arrays.pop('result')).copy()
        with patch('opt_einsum.paths.get_path_fn', side_effect=AssertionError('path search')):
            baryon_join.contract_baryon_join(action, arrays, backend, timed=timed,
                                            device=backend_name == 'cupy')
        result = arrays.pop('result')
        actual = host(result).copy()
        assert result.dtype == np.dtype(dtype)
        tolerance = 3e-6 if dtype == 'complex64' else 3e-13
        for value in (baseline, actual):
            np.testing.assert_allclose(value, expected, rtol=tolerance,
                                       atol=tolerance * np.max(np.abs(expected)))
        for key, value in snapshots.items():
            np.testing.assert_array_equal(host(arrays[key]), value)
        if previous is not None:
            assert not np.array_equal(actual, previous)
        previous = actual
        # Changing current inputs cannot mutate the returned output owner.
        first_key = next(iter(arrays))
        arrays[first_key][...] *= 1.1
        np.testing.assert_array_equal(host(result), actual)


def test_explicit_steps_and_single_component_keep_existing_route():
    action, arrays = fixture(np, 'complex128', 'contiguous')
    with patch.object(baryon_join, 'prepare_component_sum', side_effect=AssertionError('changed route')):
        baryon_join.contract_baryon_join(action, arrays, np, timed=timed, no_search=True)
        single = replace(action.core, components=(action.core.components[0],))
        action.core, action.operands = single, single.inputs + ('P', 'Q')
        baryon_join.contract_baryon_join(action, arrays, np, timed=timed)


def test_failed_component_compute_releases_current_inputs():
    action, arrays = fixture(np, 'complex128', 'contiguous')
    refs = [weakref.ref(value) for value in arrays.values()]
    prepared = []
    original = baryon_join.prepare_component_sum
    def capture(*args):
        result = original(*args)
        prepared.append(result)
        return result
    with (patch.object(baryon_join, 'prepare_component_sum', capture),
          patch.object(baryon_component_sum.PreparedComponentSum, 'compute', side_effect=RuntimeError('failed'))):
        with pytest.raises(RuntimeError, match='failed'):
            baryon_join.contract_baryon_join(action, arrays, np, timed=timed)
    assert 'result' not in arrays
    assert prepared[0].components == prepared[0].legs == ()
    del arrays
    gc.collect()
    assert all(ref() is None for ref in refs)
