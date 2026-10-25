"""Formal whole-vertex dispatch preserves fixed plans and current owners."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.execution.operations import baryon_join, baryon_scalar_sum
from correlator.execution.test_baryon_scalar_sum import backend_for, fixture, reference


def timed(phase, call, **kwargs):
    return call()


@pytest.mark.parametrize('backend_name', ('numpy', 'cupy'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('batch', (None, 'static', 'dynamic', 'mixed'))
def test_formal_dispatch_current_values_without_search_or_result_copy(backend_name, dtype, batch):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend, dtype, 'strided', batch)
    if backend_name == 'numpy':
        # CPU policy retains the old route below eight modes on any leg axis.
        # Exercise actual formal dispatch above that boundary, with negative strides.
        rng = np.random.default_rng(746)
        modes = set(action.equation.inputs[0][-3:])
        for part in action.core.components:
            shape = ((2,) if arrays[part.inputs[1]].ndim == 4 else ()) + (8, 8, 8)
            value = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
            arrays[part.inputs[1]] = value[(slice(None, None, -1),)*len(shape)]
        shape = tuple(8 if axis in modes else size for axis, size in
                      zip(action.equation.inputs[1], arrays['P'].shape))
        value = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
        arrays['P'] = value[(slice(None, None, -1),)*len(shape)]
        action.core = replace(action.core, spec=replace(action.core.spec,
            shape=(*action.core.spec.shape[:-3], 8, 8, 8)))
    original = (action.equation, action.core, action.operands, action.kernel_path)
    for current in range(2):
        if current:
            for value in arrays.values():
                value *= np.dtype(dtype).type(.6+.2j)
        expected = reference(action, arrays, host)
        with (patch('opt_einsum.paths.get_path_fn', side_effect=AssertionError('search')),
              patch.object(baryon_join, '_contract_baryon_part', side_effect=AssertionError('old route')),
              patch.object(baryon_join, 'copy_to_backend', side_effect=AssertionError('redundant copy'))):
            baryon_join.contract_baryon_join(action, arrays, backend, timed=timed,
                                            device=backend_name == 'cupy')
        result = arrays.pop('result')
        actual = host(result).copy()
        tolerance = 3e-6 if dtype == 'complex64' else 2e-11
        np.testing.assert_allclose(actual, expected, rtol=tolerance,
                                   atol=tolerance*float(np.max(np.abs(expected))))
        assert result.dtype == np.dtype(dtype)
        assert (action.equation, action.core, action.operands, action.kernel_path) == original
        arrays['P'] *= .9
        np.testing.assert_array_equal(host(result), actual)


def test_reused_and_explicit_steps_do_not_prepare_scalar():
    action, arrays = fixture(np, 'complex128', 'contiguous', None)
    with patch.object(baryon_join, 'prepare_scalar_sum', side_effect=AssertionError('changed route')):
        baryon_join.contract_baryon_join(action, arrays, np, timed=timed, no_search=True)
        saved = arrays['result']
        action.reused = True
        baryon_join.contract_baryon_join(action, arrays, np, timed=timed)
        assert arrays['result'] is saved


@pytest.mark.parametrize('modes', ((1, 1, 1), (2, 3, 2), (7, 8, 8)))
def test_small_cpu_skips_preparation_probe(modes):
    action, arrays = fixture(np, 'complex128', 'permuted', 'mixed', modes=modes)
    expected = reference(action, arrays, np.asarray)
    with patch.object(baryon_join, 'prepare_scalar_sum', side_effect=AssertionError('small probe')):
        baryon_join.contract_baryon_join(action, arrays, np, timed=timed)
    np.testing.assert_allclose(arrays['result'], expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('backend_name', ('numpy', 'cupy'))
def test_missing_support_falls_back_with_same_values(backend_name):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend, 'complex128', 'permuted', 'mixed')
    action.core = replace(action.core, components=tuple(
        replace(part, sparse_spin=None) for part in action.core.components))
    expected = reference(action, arrays, host)
    baryon_join.contract_baryon_join(action, arrays, backend, timed=timed,
                                    device=backend_name == 'cupy')
    np.testing.assert_allclose(host(arrays['result']), expected, rtol=2e-11, atol=2e-12)


def test_failure_closes_preparation_without_publishing_result():
    action, arrays = fixture(np, 'complex128', 'contiguous', 'mixed', modes=(8, 8, 8))
    prepared = baryon_scalar_sum.prepare_scalar_sum(action, arrays, np, np.dtype('complex128'))
    with (patch.object(baryon_join, 'prepare_scalar_sum', return_value=prepared),
          patch.object(prepared, 'compute', side_effect=RuntimeError('compute failed'))):
        with pytest.raises(RuntimeError, match='compute failed'):
            baryon_join.contract_baryon_join(action, arrays, np, timed=timed)
    assert prepared.closed and prepared.components == () and prepared.leg is None
    assert 'result' not in arrays
