"""Bound dense stores resolve each call's arrays and preserve ownership."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import gc
import weakref
from unittest.mock import patch
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import Action, TensorSpec
from lqcdweave.contraction.execution import scene_runtime
from lqcdweave.contraction.execution.operations.contract import (
    bind_dense_action,
    bind_dense_call,
)


def _matrix_input_pairs(seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(2):
        left = (rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))).astype(np.complex128)
        right = (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))).astype(np.complex128)
        pairs.append((left, right))
    return tuple(pairs)


def _load_execution_fixture(filename, alias):
    tests_root = Path(__file__).resolve().parents[2]
    if str(tests_root) not in sys.path:
        sys.path.insert(0, str(tests_root))
    path = Path(__file__).with_name(filename)
    spec = spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load execution fixture from {path}")
    module = module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


class DenseStoreTests(unittest.TestCase):
    def test_cpu_fixed_binary_matmul_handles_strides_batches_and_axis_order(self):
        rng = np.random.default_rng(2819)
        cases = (
            ('ij,jk->ki', (4, 5), (5, 3)),
            ('bij,bjk->bki', (2, 4, 5), (2, 5, 3)),
            ('ijk,jkl->li', (3, 4, 5), (4, 5, 2)),
        )
        for equation, left_shape, right_shape in cases:
            left = rng.normal(size=left_shape) + 1j*rng.normal(size=left_shape)
            right = (rng.normal(size=right_shape) + 1j*rng.normal(size=right_shape))[..., ::-1]
            action = Action('contract', 'out', ('left', 'right'), equation)
            catalog = {key: TensorSpec(value.shape, value.dtype)
                       for key, value in (('left', left), ('right', right))}
            expected = np.einsum(equation, left, right, optimize=False)
            kernel = bind_dense_action(action, catalog, np)
            with patch.object(np, 'einsum', side_effect=AssertionError('eligible CPU binary used einsum')):
                value, independent = kernel((left, right))
            self.assertTrue(independent)
            self.assertFalse(np.shares_memory(value, left))
            self.assertFalse(np.shares_memory(value, right))
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)

    def test_cpu_diagonal_keeps_reference(self):
        for equation, left, right in (
            ('ii,ij->j', np.arange(9.).reshape(3, 3), np.arange(6.).reshape(3, 2)),
        ):
            action = Action('contract', 'out', ('left', 'right'), equation)
            catalog = {key: TensorSpec(value.shape, value.dtype)
                       for key, value in (('left', left), ('right', right))}
            expected = np.einsum(equation, left, right, optimize=False)
            kernel = bind_dense_action(action, catalog, np)
            with patch.object(np, 'einsum', wraps=np.einsum) as reference:
                value, _ = kernel((left, right))
            reference.assert_called_once()
            np.testing.assert_array_equal(value, expected)

    def test_owned_cpu_reference_result_is_not_copied_again(self):
        left=np.arange(12,dtype=np.float32).reshape(3,4)
        action=Action('contract','out',('left',),'ij->j')
        catalog={'left':TensorSpec(left.shape,left.dtype)}
        kernel=bind_dense_action(action,catalog,np)
        arrays={'left':left}
        expected=np.einsum('ij->j',left,optimize=False)
        with patch.object(np,'array',side_effect=AssertionError('owned reference result copied')):
            bind_dense_call(action,kernel,np)((arrays,None))
        self.assertTrue(arrays['out'].flags.owndata)
        np.testing.assert_array_equal(arrays['out'],expected)

    def test_mixed_precision_is_rejected_at_kernel_binding(self):
        action=Action('contract','out',('left','right'),'ij,jk->ik')
        catalog={'left':TensorSpec((3,4),'complex64'),'right':TensorSpec((4,2),'complex128')}
        with self.assertRaisesRegex(ValueError,'mixed-precision'):
            bind_dense_action(action,catalog,np)

    def test_unary_view_store_rebinds_arrays_and_keeps_owned_snapshots(self):
        action = Action("contract", "out", ("x",), "i->i")
        catalog = {"x": TensorSpec((4,), np.float64)}
        kernel = bind_dense_action(action, catalog, np)
        store = bind_dense_call(action, kernel, np)

        first_input = np.array([1.0, 2.0, -3.0, 4.5])
        first_expected = first_input.copy()
        first_arrays = {"x": first_input}
        store((first_arrays, None))
        first_output = first_arrays["out"]
        first_input[:] = 99.0
        np.testing.assert_array_equal(first_output, first_expected)
        self.assertFalse(np.shares_memory(first_output, first_input))

        second_input = np.array([-8.0, 0.25, 7.0, 2.0])
        second_expected = second_input.copy()
        second_arrays = {"x": second_input}
        store((second_arrays, None))
        second_output = second_arrays["out"]
        self.assertIsNot(first_output, second_output)
        np.testing.assert_array_equal(second_output, second_expected)
        self.assertFalse(np.shares_memory(second_output, second_input))

        first_ref, second_ref = weakref.ref(first_input), weakref.ref(second_input)
        first_arrays.clear()
        second_arrays.clear()
        del first_input, second_input
        gc.collect()
        self.assertIsNone(first_ref())
        self.assertIsNone(second_ref())
        np.testing.assert_array_equal(first_output, first_expected)
        np.testing.assert_array_equal(second_output, second_expected)

    def test_independent_binary_kernel_matches_reference_without_store_copy(self):
        left, right = _matrix_input_pairs(2209)[0]
        action = Action("contract", "out", ("left", "right"), "ij,jk->ik")
        catalog = {
            "left": TensorSpec(left.shape, left.dtype),
            "right": TensorSpec(right.shape, right.dtype),
        }
        kernel = bind_dense_action(action, catalog, np, output_layout=(0, 1))
        expected = np.einsum("ij,jk->ik", left, right, optimize=False)
        direct_value, independent = kernel((left, right))
        self.assertTrue(independent)
        self.assertFalse(np.shares_memory(direct_value, left))
        self.assertFalse(np.shares_memory(direct_value, right))
        np.testing.assert_allclose(direct_value, expected, rtol=2e-12, atol=2e-12)

        kernel_outputs = []

        def record_kernel(operands):
            value, independent_result = kernel(operands)
            kernel_outputs.append(value)
            return value, independent_result

        store = bind_dense_call(action, record_kernel, np)
        arrays = {"left": left, "right": right}
        with patch.object(np, "array", wraps=np.array) as array_factory:
            store((arrays, None))
            array_factory.assert_not_called()

        self.assertIs(arrays["out"], kernel_outputs[0])
        np.testing.assert_allclose(arrays["out"], expected, rtol=2e-12, atol=2e-12)

    def test_store_preserves_duplicate_operand_order_and_current_mapping(self):
        action = Action('contract', 'out', ('right', 'left', 'right'), 'i,j,i->ij')
        catalog = {key: TensorSpec((2,), np.complex128) for key in ('left', 'right')}
        store = bind_dense_call(action, bind_dense_action(action, catalog, np), np)

        class TracedArrays(dict):
            def __getitem__(self, key):
                self.reads.append(key)
                return super().__getitem__(key)

        for scale in (1, -2j):
            right = np.array([1 + 2j, -3j]) * scale
            left = np.array([4 - 1j, 0.25j])
            arrays = TracedArrays(right=right, left=left)
            arrays.reads = []
            store((arrays, None))
            self.assertEqual(arrays.reads, ['right', 'left', 'right'])
            np.testing.assert_array_equal(arrays['out'],
                np.einsum('i,j,i->ij', right, left, right, optimize=False))

        missing = TracedArrays(right=right)
        missing.reads = []
        with self.assertRaises(KeyError):
            store((missing, None))
        self.assertEqual(missing.reads, ['right', 'left'])
        self.assertNotIn('out', missing)

    def test_real_cupy_unary_snapshot_and_binary_store_rebind(self):
        try:
            import cupy as cp
        except ImportError as exc:
            raise unittest.SkipTest(f"CuPy unavailable: {exc}") from exc
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise unittest.SkipTest(f"CUDA device unavailable: {exc}") from exc
        if device_count < 1:
            raise unittest.SkipTest("no CUDA device")

        unary = Action("contract", "unary_out", ("x",), "i->i")
        unary_catalog = {"x": TensorSpec((6,), np.complex128)}
        unary_store = bind_dense_call(unary, bind_dense_action(unary, unary_catalog, cp), cp)
        unary_inputs = (
            np.array([1 + 2j, 2 - 1j, -3j, 4, 0.5 + 0.25j, -2 - 3j], dtype=np.complex128),
            np.array([-4j, 3 + 2j, 0.5, -1 + 7j, 2 - 4j, 6j], dtype=np.complex128),
        )
        unary_snapshots = []
        for host_input in unary_inputs:
            device_input = cp.asarray(host_input)
            arrays = {"x": device_input}
            unary_store((arrays, None))
            cp.cuda.get_current_stream().synchronize()
            snapshot = arrays["unary_out"]
            unary_snapshots.append((snapshot, host_input.copy()))
            device_input[...] = 17 - 9j
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_array_equal(cp.asnumpy(snapshot), host_input)
            del arrays["x"]
            del device_input
            gc.collect()
            np.testing.assert_array_equal(cp.asnumpy(snapshot), host_input)
        self.assertIsNot(unary_snapshots[0][0], unary_snapshots[1][0])
        np.testing.assert_array_equal(cp.asnumpy(unary_snapshots[0][0]), unary_snapshots[0][1])
        np.testing.assert_array_equal(cp.asnumpy(unary_snapshots[1][0]), unary_snapshots[1][1])

        matrix_inputs = _matrix_input_pairs(1009)
        binary = Action("contract", "binary_out", ("left", "right"), "ij,jk->ik")
        binary_catalog = {
            "left": TensorSpec((3, 4), np.complex128),
            "right": TensorSpec((4, 2), np.complex128),
        }
        binary_store = bind_dense_call(binary, bind_dense_action(binary, binary_catalog, cp), cp)
        binary_snapshots = []
        for left_host, right_host in matrix_inputs:
            expected = np.einsum("ij,jk->ik", left_host, right_host, optimize=False)
            left_device, right_device = cp.asarray(left_host), cp.asarray(right_host)
            arrays = {"left": left_device, "right": right_device}
            binary_store((arrays, None))
            cp.cuda.get_current_stream().synchronize()
            snapshot = arrays["binary_out"]
            np.testing.assert_allclose(cp.asnumpy(snapshot), expected, rtol=2e-12, atol=2e-12)
            left_device[...] = 0
            right_device[...] = 0
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_allclose(cp.asnumpy(snapshot), expected, rtol=2e-12, atol=2e-12)
            del arrays["left"], arrays["right"]
            del left_device, right_device
            gc.collect()
            np.testing.assert_allclose(cp.asnumpy(snapshot), expected, rtol=2e-12, atol=2e-12)
            binary_snapshots.append((snapshot, expected))
        self.assertIsNot(binary_snapshots[0][0], binary_snapshots[1][0])
        for snapshot, expected in binary_snapshots:
            np.testing.assert_allclose(cp.asnumpy(snapshot), expected, rtol=2e-12, atol=2e-12)

    def test_scene_runtime_custom_contract_wrapper_remains_in_dispatch(self):
        fixtures = _load_execution_fixture("test_debug_execution.py",
                                           "_dense_store_debug_execution_helpers")
        expanded, path, _loader, expected = fixtures._make_dense_execution()
        original = scene_runtime.contract_dense
        calls = []

        def wrapped_contract(*args, **kwargs):
            calls.append(args[0].tensor)
            return original(*args, **kwargs)

        with patch.object(scene_runtime, "contract_dense", wrapped_contract):
            actual = fixtures.domain.execute_contraction_path(
                expanded, path, backend=np, measure=False, debug=False)

        expected_contracts = [action.tensor for action in path.plan.state.actions
                              if action.kind == "contract" and not action.reused]
        self.assertEqual(calls, expected_contracts)
        np.testing.assert_allclose(actual.term_values[0].value, expected,
                                   rtol=2e-12, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
