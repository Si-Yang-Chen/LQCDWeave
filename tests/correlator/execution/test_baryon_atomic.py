"""Fixed-path baryon joins retain structure, never operand buffers."""

import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import opt_einsum

from lqcdweave.contraction.execution.operations.baryon_join import (
    _join_expression, contract_baryon_join,
)
from lqcdweave.contracts.correlator.api import parse_axis_pattern


def timed(phase, function, **kwargs):
    return function()


class BaryonAtomicTests(unittest.TestCase):
    def setUp(self):
        _join_expression.cache_clear()
        self.rng = np.random.default_rng(513)
        self.equation = "ijk,abc,ia,jb,kc->"
        self.shapes = ((2, 3, 2), (3, 2, 4), (2, 3), (3, 2), (2, 4))
        self.path = ((0, 2), (0, 2), (0, 1), (0, 1))

    def action(self, path=None):
        return SimpleNamespace(
            reused=False, equation=parse_axis_pattern(self.equation), operands=tuple("sepqr"),
            tensor="result", kernel_path=self.path if path is None else path,
        )

    def operands(self, dtype="complex128"):
        return {
            key: (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)).astype(dtype)
            for key, shape in zip("sepqr", self.shapes)
        }

    def test_independent_complex_reference_and_owned_result(self):
        arrays = self.operands()
        s, e, p, q, r = (arrays[key] for key in "sepqr")
        expected = sum(
            s[i, j, k] * e[a, b, c] * p[i, a] * q[j, b] * r[k, c]
            for i, j, k in np.ndindex(s.shape)
            for a, b, c in np.ndindex(e.shape)
        )
        phases = []

        def record(phase, function, **kwargs):
            phases.append(phase)
            return function()

        contract_baryon_join(self.action(), arrays, np, timed=record)
        np.testing.assert_allclose(arrays["result"], expected, rtol=2e-13)
        self.assertTrue(arrays["result"].flags.owndata)
        self.assertEqual(phases, ["baryon_prepare", "baryon_join_kernel", "result_snapshot"])

    def test_cache_reuses_structure_with_fresh_operands(self):
        first = self.operands()
        contract_baryon_join(self.action(), first, np, timed=timed)
        second = self.operands()
        expected = np.einsum(self.equation, *(second[key] for key in "sepqr"))
        with patch.object(opt_einsum, "contract_expression", side_effect=AssertionError("recompiled")), \
             patch.object(opt_einsum.paths, "get_path_fn", side_effect=AssertionError("searched")):
            contract_baryon_join(self.action(), second, np, timed=timed)
        np.testing.assert_allclose(second["result"], expected, rtol=2e-13)
        self.assertEqual(_join_expression.cache_info().hits, 1)

    def test_dtype_shape_and_explicit_path_separate_cache_entries(self):
        for dtype in ("complex128", "complex64"):
            contract_baryon_join(self.action(), self.operands(dtype), np, timed=timed)
        other_path = ((0, 2), (0, 1), (0, 2), (0, 1))
        arrays = self.operands()
        expected = np.einsum(self.equation, *(arrays[key] for key in "sepqr"))
        contract_baryon_join(self.action(other_path), arrays, np, timed=timed)
        np.testing.assert_allclose(arrays["result"], expected, rtol=2e-13)
        changed = self.operands()
        changed["s"] = changed["s"][:1]
        changed["p"] = changed["p"][:1]
        contract_baryon_join(self.action(), changed, np, timed=timed)
        self.assertEqual(_join_expression.cache_info().misses, 4)
        self.assertEqual(_join_expression.cache_info().maxsize, 256)

    def test_expression_does_not_retain_arrays(self):
        arrays = self.operands()
        refs = [weakref.ref(array) for array in arrays.values()]
        contract_baryon_join(self.action(), arrays, np, timed=timed)
        del arrays
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_reused_action_never_reads_inputs_or_builds_expression(self):
        action = self.action()
        action.reused = True
        arrays = {"result": np.array(17)}
        with patch.object(opt_einsum, "contract_expression", side_effect=AssertionError("compiled")):
            contract_baryon_join(action, arrays, np, timed=timed)
        self.assertEqual(arrays["result"], 17)
        self.assertEqual(_join_expression.cache_info().currsize, 0)

    def test_non_native_arrays_keep_fixed_path_without_cached_conversion(self):
        arrays = self.operands()
        backend = SimpleNamespace(__name__="cupy", array=np.array)
        with patch.object(opt_einsum, "contract", return_value=np.array(19)) as contract, \
             patch.object(opt_einsum, "contract_expression", side_effect=AssertionError("conversion")):
            contract_baryon_join(self.action(), arrays, backend, timed=timed)
        self.assertEqual(contract.call_args.kwargs["optimize"], list(self.path))
        self.assertEqual(contract.call_args.kwargs["backend"], "cupy")
        self.assertEqual(_join_expression.cache_info().currsize, 0)


if __name__ == "__main__":
    unittest.main()
