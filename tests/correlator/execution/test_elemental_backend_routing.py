"""Selected device scope owns elemental preparations and their constant cache.

The instrumented NumPy-backed CuPy protocol does not establish GPU correctness
or speed; the separately collected optional test needs a real device.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from correlator.execution.test_device import _FakeCupy, _Tracker
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.preparation import bind_prepared_loader


def _inputs():
    rng = np.random.default_rng(508)
    def complex_values(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    raw = complex_values((3, 2)).astype(np.complex64)
    left, right, weights = complex_values((2, 3)), complex_values((2, 2)), complex_values((2,))
    spins = (complex_values((4, 4)), complex_values((4, 4)))
    preps, specs = {}, {}
    expected = np.empty((2, 4, 4, 2, 3), np.complex128)
    for member, spin in enumerate(spins):
        constants = tuple(ArrayConstant.from_array(value) for value in (spin, left, right, weights))
        preps[str(member)] = SourcePreparation(("raw",), recipe=TensorPreparation(
            "spin_spatial", ("raw",), constants, conjugate=True, permutation=(1, 0), configuration="cfg"))
        specs[str(member)] = TensorSpec((4, 4, 2, 3), np.complex128)
        # Independent scalar profile -> dagger -> spin reference.
        for i, j in np.ndindex(raw.shape):
            value = sum(weights[p] * left[p, i].conjugate() * raw[i, j] * right[p, j]
                        for p in range(len(weights)))
            for a, b in np.ndindex(spin.shape):
                expected[member, a, b, j, i] = spin[a, b] * value.conjugate()
    preps["stack"] = SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("0", "1")))
    specs["stack"] = TensorSpec(expected.shape, expected.dtype)
    return raw, preps, specs, expected


def _check_device_binding(test, backend, to_host):
    raw, preps, specs, expected = _inputs()
    physical = SimpleNamespace(
        input_readers={}, read_block=Mock(return_value=raw), close_block=Mock(),
        materialize=Mock(side_effect=AssertionError("opaque CPU preparation was used")))
    host = bind_prepared_loader(physical, preps, specs)
    # Start from an already-used NumPy loader: rebinding must replace its cache.
    np.testing.assert_allclose(host.materialize("stack", None, {"raw": raw}), expected,
                               rtol=3e-14, atol=3e-14)
    with device_execution(backend, device_memory_limit=1 << 26) as device:
        loader = device.bind_loader(host)
        test.assertIs(loader.prepared.backend, backend)
        test.assertIsNot(loader.prepared.constants, host.constants)
        value = loader.read_block("raw", None)
        try:
            before = raw.copy()
            for _ in range(2):
                actual = loader.materialize("stack", None, {"raw": value})
                test.assertIsInstance(actual, backend.ndarray)
                np.testing.assert_allclose(to_host(actual), expected, rtol=3e-13, atol=3e-13)
            np.testing.assert_array_equal(raw, before)
            test.assertEqual(len(loader.prepared._constant_pool), 5)
        finally:
            loader.close_block("raw", value)
    physical.close_block.assert_called_once_with("raw", raw)
    physical.materialize.assert_not_called()


class ElementalBackendRoutingTests(unittest.TestCase):
    def test_profile_dagger_spin_stack_rebind_to_device_not_host_materializer(self):
        backend = _FakeCupy(_Tracker())
        backend.asarray = Mock(wraps=backend.asarray)
        backend.zeros_like = Mock(wraps=np.zeros_like)
        backend.multiply = Mock(wraps=np.multiply)
        _check_device_binding(self, backend, np.asarray)
        # One raw upload plus five distinct constants, reused on the second call.
        self.assertEqual(backend.asarray.call_count, 6)
        self.assertEqual(backend.zeros_like.call_count, 4)
        self.assertEqual(backend.multiply.call_count, 4)


class ElementalBackendRoutingCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as error:
            raise unittest.SkipTest(f"requires working CuPy/CUDA: {error}")
        cls.cp = cupy

    def test_profile_dagger_spin_stack_stays_on_device(self):
        _check_device_binding(self, self.cp, self.cp.asnumpy)
