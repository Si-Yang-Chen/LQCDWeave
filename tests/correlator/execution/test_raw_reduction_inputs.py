"""Dynamic raw input checks remain before allocations and native submission."""
import gc
import weakref

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution.reduction_kernel import (
    _bind_input_strides, _kernel, bind_raw_reduction)
from correlator.execution.test_raw_precise_reduction import _SourceBackend


class _CountingBackend(_SourceBackend):
    def __init__(self):
        super().__init__()
        self.allocations = 0

    def empty(self, shape, dtype):
        self.allocations += 1
        return np.empty(shape, dtype=dtype)


@pytest.fixture(autouse=True)
def clear_kernels():
    yield
    _kernel.cache_clear()


def test_invalid_binary_inputs_fail_before_allocation_binding_or_submission():
    backend = _CountingBackend()
    a = np.ones((3, 17), np.float32)
    b = np.ones((5, 17), np.float32)
    run = bind_raw_reduction(parse_axis_pattern('ik,jk->ij'), (a.shape, b.shape),
                             a.dtype, backend, 8192)
    invalid = ((), (a,), (a, b, b), (a[:2], b), (a, b[:4]),
               (a.astype(np.float64), b), (a, b.astype(np.float64)))
    for values in invalid:
        with pytest.raises(ValueError):
            run(*values)
        assert backend.allocations == backend.launches == 0
        assert backend.sources == []
        backend.synchronize.assert_not_called()
    np.testing.assert_array_equal(run(a, b), np.full((3, 5), 17, np.float32))
    counts = backend.allocations, backend.launches, len(backend.sources)
    for values in invalid:
        with pytest.raises(ValueError):
            run(*values)
        assert (backend.allocations, backend.launches, len(backend.sources)) == counts


@pytest.mark.parametrize('dtype', [np.float32, np.complex64])
def test_binary_layout_changes_and_fresh_values_match_independent_sum(dtype):
    backend = _CountingBackend()
    rng = np.random.default_rng(49)
    a = rng.normal(size=(3, 17)).astype(dtype)
    b = rng.normal(size=(5, 17)).astype(dtype)
    if dtype is np.complex64:
        a += .7j * a[:, ::-1]
        b -= .2j * b[:, ::-1]
    run = bind_raw_reduction(parse_axis_pattern('ik,jk->ij'), (a.shape, b.shape),
                             a.dtype, backend, 8192)
    wide = np.float64 if dtype is np.float32 else np.complex128
    outputs = []
    for left, right in ((a, b), (a[:, ::-1], b), (a, b[:, ::-1]),
                         (np.asfortranarray(a), b), (a * 2, b * 3)):
        expected = np.einsum('ik,jk->ij', left, right, dtype=wide).astype(dtype)
        outputs.append(run(left, right))
        np.testing.assert_allclose(outputs[-1], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outputs[-1], 6 * outputs[0], rtol=2e-6, atol=2e-6)
    assert all(not np.shares_memory(outputs[0], other) for other in outputs[1:])
    backend.fail_at = backend.launches + 2
    with pytest.raises(RuntimeError, match='submission failed'):
        run(a, b)
    backend.synchronize.assert_called_once()


def test_generic_unary_checks_and_shape_equal_different_strides_remain_dynamic():
    backend = _CountingBackend()
    a = np.arange(17, dtype=np.float32)
    run = bind_raw_reduction(parse_axis_pattern('i->'), (a.shape,), a.dtype, backend, 8192)
    for invalid in ((), (a, a), (a[:-1],), (a.astype(np.float64),)):
        with pytest.raises(ValueError):
            run(*invalid)
    assert backend.launches == backend.allocations == 0
    assert run(a) == np.sum(a, dtype=np.float64).astype(a.dtype)
    assert run(a[::-1]) == np.sum(a, dtype=np.float64).astype(a.dtype)


def test_bound_checks_do_not_capture_numerical_inputs_or_strides():
    value = np.ones((2, 3), np.float32)
    read = _bind_input_strides((value.shape, value.shape), value.dtype)
    assert read((value, value)) == (value.strides, value.strides)
    assert read((value[:, ::-1], value.T.copy().T)) == ((12, -4), (4, 8))
    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is None
