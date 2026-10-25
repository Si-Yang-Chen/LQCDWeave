"""Atomic dense GPU dispatch, ownership and independent numerical references."""

import importlib
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.operations.contract import contract_dense
from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern


CASES = (
    ("ij,jk->ik", ((3, 4), (4, 2))),
    ("abij,abjk->bkai", ((2, 3, 4, 5), (2, 3, 5, 2))),
    ("ikjl,jlmn->nmki", ((2, 3, 4, 5), (4, 5, 2, 3))),
    ("i,j->ji", ((4,), (3,))),
    ("ij,ij->", ((3, 4), (3, 4))),
    ("ab,ab->ba", ((2, 3), (2, 3))),
    ("abi,abj->bija", ((2, 3, 4), (2, 3, 5))),
)

FALLBACK_CASES = (
    ("ii,ij->j", ((3, 3), (3, 4))),
    ("ij,jk->k", ((3, 4), (4, 2))),
)

NO_REDUCTION_PATTERNS = frozenset(
    parse_axis_pattern(equation).normalized()
    for equation in ("i,j->ji", "ab,ab->ba", "abi,abj->bija")
)


def _inputs(shapes, dtype):
    rng = np.random.default_rng(194)
    return tuple((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
                 if np.dtype(dtype).kind == "c" else rng.normal(size=shape).astype(dtype)
                 for shape in shapes)


def _loop_reference(equation, operands):
    """Explicit index summation, including output permutation and complex products."""
    inputs, output = equation.split("->")
    labels = inputs.split(",")
    dimensions = {label: size for subscript, value in zip(labels, operands)
                  for label, size in zip(subscript, value.shape)}
    all_labels = tuple(dimensions)
    result = np.zeros(tuple(dimensions[label] for label in output),
                      dtype=np.result_type(*operands))
    for coordinates in np.ndindex(*(dimensions[label] for label in all_labels)):
        position = dict(zip(all_labels, coordinates))
        product = 1
        for subscript, value in zip(labels, operands):
            product *= value[tuple(position[label] for label in subscript)]
        result[tuple(position[label] for label in output)] += product
    return result


def _execute(equation, operands, backend, *, device, copy_result=True):
    arrays = {"left": operands[0], "right": operands[1]}
    action = SimpleNamespace(reused=False, core=None, equation=parse_axis_pattern(equation),
                             operands=("left", "right"), tensor="result",
                             wide_accumulation=False)
    phases = []

    def timed(name, function, **kwargs):
        phases.append((name, kwargs))
        return function()

    contract_dense(action, arrays, backend, timed=timed, device=device,
                   copy_result=copy_result)
    return arrays["result"], phases


class DenseAtomicKernelTests(unittest.TestCase):
    def test_owned_matmul_snapshot_adopts_storage_and_survives_input_release_and_reexecution(self):
        equation = "abij,abjk->bkai"
        shapes = ((2, 3, 4, 5), (2, 3, 5, 2))
        for copy_result in (False, True):
            with self.subTest(copy_result=copy_result):
                operands = tuple(value[..., ::-1] for value in _inputs(shapes, np.complex128))
                expected = _loop_reference(equation, operands)
                references = tuple(weakref.ref(value) for value in operands)
                with patch.object(np, "array", wraps=np.array) as array:
                    result, phases = _execute(equation, operands, np, device=True,
                                               copy_result=copy_result)
                array.assert_not_called()
                if copy_result:
                    self.assertEqual(phases[-1], ("result_snapshot", {"nbytes": 0, "device": True}))
                saved_inputs = tuple(value.copy() for value in operands)
                result[0, 0, 0, 0] += 5j
                for value, saved in zip(operands, saved_inputs):
                    np.testing.assert_array_equal(value, saved)
                del value, saved
                result[0, 0, 0, 0] -= 5j
                operands[0][...] = 0
                del operands
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
                np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
                fresh = _inputs(shapes, np.complex128)
                fresh[0][...] *= 2
                next_result, _ = _execute(equation, fresh, np, device=True,
                                           copy_result=copy_result)
                self.assertFalse(np.shares_memory(result, next_result))
                next_result[...] = 0
                np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_cpu_and_fallback_snapshots_still_copy_owned_values(self):
        for equation, shapes in (CASES[0], FALLBACK_CASES[0]):
            for device in (False, True):
                if device and equation == CASES[0][0]:
                    continue
                with self.subTest(equation=equation, device=device):
                    operands = _inputs(shapes, np.complex128)
                    with patch.object(np, "array", wraps=np.array) as array:
                        result, phases = _execute(equation, operands, np, device=device)
                    self.assertEqual(array.call_count, 1)
                    self.assertEqual(phases[-1][1]["nbytes"], result.nbytes)
                    np.testing.assert_allclose(result, _loop_reference(equation, operands),
                                               rtol=1e-12, atol=1e-12)

    def test_reference_only_gpu_actions_skip_lowering_and_matmul_capability_checks(self):
        class ReferenceBackend:
            einsum = staticmethod(np.einsum)
            array = staticmethod(np.array)

            def __getattr__(self, name):
                raise AssertionError(f"reference-only operation queried backend capability {name}")

        cases = FALLBACK_CASES + tuple(
            case for case in CASES
            if parse_axis_pattern(case[0]).normalized() in NO_REDUCTION_PATTERNS
        )
        for equation, shapes in cases:
            with self.subTest(equation=equation):
                operands = _inputs(shapes, np.complex128)
                with patch("lqcdweave.contraction.execution.operations.contract.lower_binary_matmul",
                           side_effect=AssertionError("reference-only operation normalized shapes/dtypes")):
                    result, _ = _execute(equation, operands, ReferenceBackend(), device=True)
                np.testing.assert_allclose(result, _loop_reference(equation, operands),
                                           rtol=1e-12, atol=1e-12)

    def test_lowering_lookup_is_inside_measured_kernel_phase(self):
        from lqcdweave.contraction.execution import lowering

        active_phase = None
        seen = []
        operands = _inputs(((3, 4), (4, 2)), np.complex128)
        arrays = dict(zip(("left", "right"), operands))
        external_equation = "ij,jk->ik"
        action = SimpleNamespace(reused=False, core=None,
                                 equation=parse_axis_pattern(external_equation),
                                 operands=("left", "right"), tensor="result",
                                 wide_accumulation=False)

        def lookup(*args, **kwargs):
            self.assertEqual(active_phase, "contract_kernel")
            seen.append(active_phase)
            return lowering.lower_binary_matmul(*args, **kwargs)

        def timed(name, function, **kwargs):
            nonlocal active_phase
            active_phase = name
            try:
                return function()
            finally:
                active_phase = None

        with patch("lqcdweave.contraction.execution.operations.contract.lower_binary_matmul",
                   side_effect=lookup):
            contract_dense(action, arrays, np, timed=timed, device=True)
        self.assertEqual(seen, ["contract_kernel"])
        np.testing.assert_allclose(arrays["result"], _loop_reference(external_equation, operands),
                                   rtol=1e-12, atol=1e-12)

    def test_gpu_recipe_preserves_axes_values_and_ownership_without_path_search(self):
        calls = []

        def no_reduction_einsum(*arguments, **kwargs):
            self.assertEqual(kwargs, {"optimize": False})
            operand_count = (len(arguments) - 1) // 2
            pattern = AxisPattern(
                tuple(tuple(arguments[2 * index + 1]) for index in range(operand_count)),
                tuple(arguments[-1]),
            )
            self.assertIn(pattern, NO_REDUCTION_PATTERNS)
            calls.append(pattern)
            return np.einsum(*arguments, **kwargs)

        backend = SimpleNamespace(transpose=np.transpose, reshape=np.reshape,
                                  matmul=np.matmul, array=np.array,
                                  einsum=no_reduction_einsum)
        for equation, shapes in CASES:
            for dtype in (np.float32, np.float64, np.complex64, np.complex128):
                for copy_result in (False, True):
                    with self.subTest(equation=equation, dtype=dtype, copy_result=copy_result):
                        operands = _inputs(shapes, dtype)
                        # Negative strides and noncontiguous axis groups exercise packing.
                        operands = tuple(value[..., ::-1] for value in operands)
                        expected = _loop_reference(equation, operands)
                        calls.clear()
                        with patch("opt_einsum.contract_path", side_effect=AssertionError("path search")), \
                             patch.object(backend, "matmul", wraps=np.matmul) as matmul:
                            result, phases = _execute(equation, operands, backend, device=True,
                                                      copy_result=copy_result)
                        is_no_reduction = parse_axis_pattern(equation).normalized() in NO_REDUCTION_PATTERNS
                        self.assertEqual(calls, [parse_axis_pattern(equation).normalized()]
                                         if is_no_reduction else [])
                        self.assertEqual(matmul.call_count, 0 if is_no_reduction else 1)
                        tolerance = 3e-5 if dtype in (np.float32, np.complex64) else 1e-12
                        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
                        self.assertEqual(result.dtype, np.dtype(dtype))
                        self.assertFalse(any(np.shares_memory(result, value) for value in operands))
                        operands[0][...] = 0
                        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
                        self.assertEqual([name for name, _ in phases],
                                         ["contract_kernel", "result_snapshot"] if copy_result
                                         else ["contract_kernel"])

    def test_cpu_and_unsupported_equations_keep_no_search_einsum(self):
        equations = FALLBACK_CASES + (("ij,jk->ik", ((3, 4), (4, 2))),)
        for equation, shapes in equations:
            for device in (False, True):
                with self.subTest(equation=equation, device=device):
                    operands = _inputs(shapes, np.complex128)
                    with patch.object(np, "einsum", wraps=np.einsum) as einsum:
                        result, _ = _execute(equation, operands, np, device=device)
                    if device and equation == "ij,jk->ik":
                        einsum.assert_not_called()
                    else:
                        self.assertEqual(einsum.call_args.kwargs, {"optimize": False})
                    np.testing.assert_allclose(result, _loop_reference(equation, operands),
                                               rtol=1e-12, atol=1e-12)

    def test_reused_action_never_resolves_operands(self):
        phases = []
        action = SimpleNamespace(reused=True)
        contract_dense(action, {}, None, timed=lambda name, fn, **kw: phases.append(name))
        self.assertEqual(phases, ["reuse"])


class DenseAtomicCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module("cupy")
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc

    def test_owned_gpu_result_mutation_release_and_later_execution_are_independent(self):
        equation = "abij,abjk->bkai"
        shapes = ((2, 3, 4, 5), (2, 3, 5, 2))
        for copy_result in (False, True):
            with self.subTest(copy_result=copy_result):
                host = _inputs(shapes, np.complex128)
                operands = tuple(self.cp.asarray(value)[..., ::-1] for value in host)
                expected = _loop_reference(equation, tuple(value[..., ::-1] for value in host))
                result, phases = _execute(equation, operands, self.cp, device=True,
                                           copy_result=copy_result)
                if copy_result:
                    self.assertEqual(phases[-1][1]["nbytes"], 0)
                result[0, 0, 0, 0] += 5j
                for value, reference in zip(operands, host):
                    np.testing.assert_array_equal(self.cp.asnumpy(value), reference[..., ::-1])
                del value, reference
                result[0, 0, 0, 0] -= 5j
                operands[0][...] = 0
                del operands
                gc.collect()
                fresh = tuple(2 * self.cp.asarray(value) for value in host)
                next_result, _ = _execute(equation, fresh, self.cp, device=True,
                                           copy_result=copy_result)
                next_result[...] = 0
                np.testing.assert_allclose(self.cp.asnumpy(result), expected,
                                           rtol=1e-12, atol=1e-12)

    def test_real_gpu_binary_actions_against_index_loops(self):
        for equation, shapes in CASES + FALLBACK_CASES:
            for dtype in (np.complex64, np.complex128):
                for copy_result in (False, True):
                    with self.subTest(equation=equation, dtype=dtype, copy_result=copy_result):
                        host = _inputs(shapes, dtype)
                        operands = tuple(self.cp.asarray(value)[..., ::-1] for value in host)
                        result, _ = _execute(equation, operands, self.cp, device=True,
                                             copy_result=copy_result)
                        expected = _loop_reference(equation, tuple(value[..., ::-1] for value in host))
                        tolerance = 3e-5 if dtype == np.complex64 else 1e-12
                        np.testing.assert_allclose(self.cp.asnumpy(result), expected,
                                                   rtol=tolerance, atol=tolerance)
                        operands[0][...] = 0
                        np.testing.assert_allclose(self.cp.asnumpy(result), expected,
                                                   rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
