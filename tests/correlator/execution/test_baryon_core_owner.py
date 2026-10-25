"""Whole-core accumulation consumes independent parts without a second copy."""
from dataclasses import replace
import os
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.cores import baryon_component_pattern
from lqcdweave.contraction.execution.operations import baryon_join
from correlator.planning.test_cfg_whole_baryon_identity import fixture


def timed(phase, call, **options):
    return call()


@pytest.mark.parametrize('backend_name', ('numpy', 'cupy'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('no_search', (False, True))
def test_owned_parts_preserve_values_layout_inputs_and_current_data(
        backend_name, dtype, sparse, no_search):
    if backend_name == 'cupy':
        if os.environ.get('LQCDWEAVE_TEST_GPU') != '1':
            pytest.skip('requires an explicitly allocated GPU')
        import cupy as backend
        host = backend.asnumpy
    else:
        backend, host = np, np.asarray
    selected, _, _, _, originals, *_ = fixture(dtype, backend_name)
    action = next(item for item in selected.state.actions if item.core is not None)
    if not sparse:
        action = replace(action, core=replace(action.core, components=tuple(
            replace(component, sparse_spin=None) for component in action.core.components)))
    core = action.core
    core_labels, *leg_labels = action.equation.inputs
    legs = action.operands[len(core.inputs):]
    real_part = baryon_join._contract_baryon_part
    previous = None
    for scale in (1, .8+.15j):
        arrays = {key: backend.asarray(value if key.startswith('S') else value * scale)
                  for key, value in originals.items()}
        snapshots = {key: host(value).copy() for key, value in arrays.items()}
        # Keep the old snapshot -> copy -> weight -> add order as the exact
        # reference. Independent physical/formula references live in the
        # existing baryon precision, sparse-spin and full archive suites.
        expected = None
        for component in core.components:
            equation = baryon_component_pattern(core_labels, leg_labels,
                action.equation.output, arrays[component.inputs[1]].ndim)
            operands = tuple(arrays[key] for key in component.inputs + legs)
            part = real_part(equation, operands, action.kernel_path, component.sparse_spin,
                backend, timed=timed, device=backend_name == 'cupy', no_search=no_search,
                kernel_description=action.kernel_description)
            value = backend.array(part, dtype=dtype, copy=True)
            value *= component.coefficient
            if expected is None:
                expected = value
            else:
                expected += value
        parts, consumed = [], []
        def capture(*args, **kwargs):
            value = real_part(*args, **kwargs)
            parts.append(value)
            return value
        def record(phase, call, **options):
            value = call()
            if phase == 'core_copy':
                consumed.append(value)
            return value
        with patch.object(baryon_join, '_contract_baryon_part', capture):
            baryon_join.contract_baryon_join(action, arrays, backend, timed=record,
                device=backend_name == 'cupy', no_search=no_search)
        actual = arrays[action.tensor]
        assert all(a is b for a, b in zip(parts, consumed, strict=True))
        assert actual is parts[0]
        assert actual.dtype == expected.dtype
        assert actual.strides == expected.strides
        np.testing.assert_array_equal(host(actual), host(expected))
        for key, value in snapshots.items():
            np.testing.assert_array_equal(host(arrays[key]), value)
        retained = host(actual).copy()
        for key in originals:
            arrays[key][...] = 0
        np.testing.assert_array_equal(host(actual), retained)
        if previous is not None:
            assert not np.array_equal(retained, previous)
        previous = retained


def test_core_keeps_dtype_conversion_when_accumulator_is_wider():
    selected, _, _, _, arrays, *_ = fixture('complex64')
    action = next(item for item in selected.state.actions if item.core is not None)
    action = replace(action, core=replace(action.core,
        spec=replace(action.core.spec, dtype=np.dtype('complex128'))))
    baryon_join.contract_baryon_join(action, arrays, np, timed=timed, no_search=True)
    assert arrays[action.tensor].dtype == np.dtype('complex128')
