"""Fixed preparation kernels: CPU oracles, dispatch counts and ownership.

NumPy-backed instrumentation is not GPU execution/performance evidence.
"""
import gc
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import bind_prepared_loader


def _spin_preparation(spin, *, block="raw", conjugate=False, permutation=(), profile=()):
    recipe = TensorPreparation("spin_spatial", (block,),
        tuple(ArrayConstant.from_array(value) for value in (spin, *profile)),
        conjugate=conjugate, permutation=permutation,
        configuration="cfg" if profile else None)
    return SourcePreparation((block,), recipe=recipe)


def _outer_oracle(spin, raw, *, conjugate=False, permutation=()):
    """Explicit scalar indices, independent of the fixed broadcast kernel."""
    spatial = raw.conj() if conjugate else raw
    spatial = spatial.transpose(permutation) if permutation else spatial
    dtype = np.result_type(spin, spatial)
    spin, spatial = spin.astype(dtype), spatial.astype(dtype)
    result = np.empty((4, 4, *spatial.shape), dtype=dtype)
    for a, b in product(range(4), repeat=2):
        for index in np.ndindex(spatial.shape):
            result[(a, b, *index)] = spin[a, b] * spatial[index]
    return result


def _backend():
    return SimpleNamespace(
        asarray=Mock(wraps=np.asarray), empty=Mock(wraps=np.empty),
        multiply=Mock(wraps=np.multiply), zeros_like=Mock(wraps=np.zeros_like),
        result_type=np.result_type,
        einsum=Mock(side_effect=AssertionError("fixed preparation used einsum")),
    )


class PreboundPreparationTests(unittest.TestCase):
    def test_slab_selection_precedes_cast_spin_dagger_and_destination_write(self):
        from dataclasses import replace
        rng = np.random.default_rng(573)
        spin = (rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))).astype('complex64')
        slab = rng.normal(size=(3, 2, 5)) + 1j*rng.normal(size=(3, 2, 5))
        for index in (0, 2):
            for kind in ('identity', 'spin_spatial'):
                with self.subTest(index=index, kind=kind):
                    recipe = (TensorPreparation('identity', ('raw',)) if kind == 'identity'
                              else _spin_preparation(spin, conjugate=True, permutation=(1, 0)).recipe)
                    recipe = replace(recipe, raw_index=index, compute_dtype='complex64')
                    expected = (slab[index].astype('complex64') if kind == 'identity' else
                                _outer_oracle(spin, slab[index].astype('complex64'),
                                              conjugate=True, permutation=(1, 0)))
                    prep = SourcePreparation(('raw',), recipe=recipe)
                    loader = bind_prepared_loader(None, {'x': prep},
                        {'x': TensorSpec(expected.shape, expected.dtype)}, backend=np)
                    actual = loader.materialize('x', None, {'raw': slab})
                    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)
                    self.assertFalse(np.shares_memory(actual, slab))
                    # A fresh raw owner must not reuse values or addresses.
                    changed = loader.materialize('x', None, {'raw': 2*slab})
                    np.testing.assert_allclose(changed, 2*expected, rtol=3e-7, atol=6e-7)

    def test_outer_matches_scalar_oracle_for_sectors_dtypes_and_strides(self):
        rng = np.random.default_rng(418)
        layouts = (((3, 2), (1, 0)), ((2, 3, 2), (1, 2, 0)),
                   ((3, 2, 2), (2, 0, 1)), ((2, 3, 2, 2), (2, 3, 0, 1)))
        dtypes = (np.float32, np.float64, np.complex64, np.complex128)
        for (shape, permutation), spin_dtype, raw_dtype, dagger in product(
                layouts, dtypes, dtypes, (False, True)):
            with self.subTest(shape=shape, spin=spin_dtype, raw=raw_dtype, dagger=dagger):
                spin = rng.normal(size=(4, 4))
                raw = rng.normal(size=shape)
                if np.dtype(spin_dtype).kind == "c":
                    spin = spin + 1j * rng.normal(size=spin.shape)
                if np.dtype(raw_dtype).kind == "c":
                    raw = raw + 1j * rng.normal(size=raw.shape)
                spin, raw = spin.astype(spin_dtype), raw.astype(raw_dtype)[::-1]
                axes = permutation if dagger else ()
                expected = _outer_oracle(spin, raw, conjugate=dagger, permutation=axes)
                prep = _spin_preparation(spin, conjugate=dagger, permutation=axes)
                backend = _backend()
                loader = bind_prepared_loader(None, {"x": prep},
                    {"x": TensorSpec(expected.shape, expected.dtype)}, backend=backend)
                before = raw.copy()
                actual = loader.materialize("x", None, {"raw": raw})
                self.assertEqual(actual.dtype, expected.dtype)
                tolerance = 2e-6 if expected.real.dtype.itemsize == 4 else 3e-14
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                np.testing.assert_array_equal(raw, before)
                self.assertFalse(np.shares_memory(actual, raw))
                self.assertTrue(loader.materializes_owned("x"))
                self.assertEqual(backend.multiply.call_count, 1)
                backend.einsum.assert_not_called()

    def test_scalar_empty_and_non_native_inputs_preserve_einsum_dtype(self):
        # Scalar promotion in NumPy 1.x is value-sensitive. The old expression
        # is used only to check that ABI, not as the numerical value oracle.
        for shape, dtype in product(((), (0, 2), (2, 3)), ("<f4", ">f8", ">c16")):
            raw = np.ones(shape, dtype=dtype)
            spin = np.full((4, 4), 1.2345678 + 0.12345j, np.complex64)
            expected = _outer_oracle(spin, raw)
            loader = bind_prepared_loader(None, {"x": _spin_preparation(spin)},
                {"x": TensorSpec(expected.shape, expected.dtype)})
            actual = loader.materialize("x", None, {"raw": raw})
            self.assertEqual(actual.dtype, np.einsum("ab,...->ab...", spin, raw).dtype)
            np.testing.assert_array_equal(actual, expected)

    def test_profile_dagger_matches_explicit_complex_scalar_sum(self):
        rng = np.random.default_rng(312)
        spin = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        raw = (rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))).astype(np.complex64)
        left = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        right = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        weights = np.array([1.5 + 2j, -0.2j, -3.5 + 0.7j])
        projected = np.empty(raw.shape, dtype=np.complex128)
        for i, j in np.ndindex(raw.shape):
            projected[i, j] = sum(weights[p] * left[p, i].conjugate()
                                  * raw[i, j] * right[p, j] for p in range(3))
        for dagger in (False, True):
            axes = (1, 0) if dagger else ()
            expected = _outer_oracle(spin, projected, conjugate=dagger, permutation=axes)
            prep = _spin_preparation(spin, conjugate=dagger, permutation=axes,
                                     profile=(left, right, weights))
            backend = _backend()
            loader = bind_prepared_loader(None, {"x": prep},
                {"x": TensorSpec(expected.shape, expected.dtype)}, backend=backend)
            actual = loader.materialize("x", None, {"raw": raw})
            self.assertEqual(actual.dtype, np.dtype(np.complex128))
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
            self.assertEqual(backend.zeros_like.call_count, 1)
            self.assertEqual(backend.multiply.call_count, 1)

    def test_stack_writes_once_into_final_slices_and_binds_once(self):
        raw = np.arange(6, dtype=np.float32).reshape(2, 3) + np.float32(0.31)
        spins = (np.full((4, 4), 1.2345678, np.complex64), np.eye(4, dtype=np.complex128))
        preps = {str(index): _spin_preparation(spin) for index, spin in enumerate(spins)}
        preps["stack"] = SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("0", "1")))
        specs = {str(index): TensorSpec((4, 4, 2, 3), spin.dtype) for index, spin in enumerate(spins)}
        specs["stack"] = TensorSpec((2, 4, 4, 2, 3), np.complex128)
        backend = _backend()
        loader = bind_prepared_loader(None, preps, specs, backend=backend)
        with patch.object(loader, "_bind_spin", wraps=loader._bind_spin) as bind:
            for run in range(2):
                with execution_debug(True):
                    actual = loader.materialize("stack", None, {"raw": raw})
                expected = np.stack([_outer_oracle(spin, raw) for spin in spins])
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(backend.empty.call_count, run + 1)
                for call in backend.multiply.call_args_list[-2:]:
                    self.assertTrue(np.shares_memory(call.kwargs["out"], actual))
                raw += np.float32(0.5)
            self.assertEqual(bind.call_count, 2)
        self.assertEqual(backend.asarray.call_count, 2)
        self.assertEqual(backend.multiply.call_count, 4)
        backend.einsum.assert_not_called()

    def test_nested_stack_preserves_intermediate_rounding_and_opaque_fallback(self):
        raw = np.array([1.00000007], dtype=np.float64)
        opaque = Mock(side_effect=lambda key, source, blocks: raw)
        physical = SimpleNamespace(materialize=opaque)
        preps = {
            "opaque": SourcePreparation(("raw",)),
            "inner": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("opaque",))),
            "outer": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("inner",))),
        }
        specs = {"opaque": TensorSpec((1,), np.float64), "inner": TensorSpec((1, 1), np.float32),
                 "outer": TensorSpec((1, 1, 1), np.float64)}
        loader = bind_prepared_loader(physical, preps, specs)
        with execution_debug(True):
            actual = loader.materialize("outer", None, {"raw": raw})
        np.testing.assert_array_equal(actual, raw.astype(np.float32).astype(np.float64).reshape(1, 1, 1))
        self.assertEqual(opaque.call_count, 1)
        self.assertFalse(loader.materializes_owned("opaque"))

    def test_outputs_and_mutable_raw_blocks_are_not_retained_or_reused(self):
        spin = np.eye(4, dtype=np.complex128)
        prep = _spin_preparation(spin, conjugate=True, permutation=(1, 0))
        loader = bind_prepared_loader(None, {"x": prep}, {"x": TensorSpec((4, 4, 3, 2), spin.dtype)})
        raw = np.ones((2, 3), dtype=np.complex128)
        first = loader.materialize("x", None, {"raw": raw})
        raw[:] = 2 + 3j
        second = loader.materialize("x", None, {"raw": raw})
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(first, _outer_oracle(spin, np.ones_like(raw),
                                                          conjugate=True, permutation=(1, 0)))
        np.testing.assert_array_equal(second, _outer_oracle(spin, raw, conjugate=True, permutation=(1, 0)))
        refs = [weakref.ref(value) for value in (raw, first, second)]
        del raw, first, second
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(loader._bound_preparations), 1)

    def test_identity_and_constant_remain_borrowed(self):
        raw = np.ones(3)
        constant = ArrayConstant.from_array(np.arange(3.0))
        preps = {
            "identity": SourcePreparation(("raw",), recipe=TensorPreparation("identity", ("raw",))),
            "constant": SourcePreparation((), recipe=TensorPreparation("constant", constants=(constant,))),
        }
        loader = bind_prepared_loader(None, preps, {key: TensorSpec((3,), np.float64) for key in preps})
        self.assertIs(loader.materialize("identity", None, {"raw": raw}), raw)
        first = loader.materialize("constant", None, {})
        self.assertIs(loader.materialize("constant", None, {}), first)
        self.assertFalse(loader.materializes_owned("identity"))
        self.assertFalse(loader.materializes_owned("constant"))

    def test_scalar_stack_members_write_into_zero_dimensional_views(self):
        raw = np.array(2.5)
        preps = {
            "a": SourcePreparation(("raw",), recipe=TensorPreparation("identity", ("raw",))),
            "b": SourcePreparation((), recipe=TensorPreparation("constant",
                constants=(ArrayConstant.from_array(np.array(-1.5)),))),
            "s": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("a", "b"))),
        }
        specs = {"a": TensorSpec((), np.float64), "b": TensorSpec((), np.float64),
                 "s": TensorSpec((2,), np.float64)}
        loader = bind_prepared_loader(None, preps, specs)
        with execution_debug(True):
            actual = loader.materialize("s", None, {"raw": raw})
        np.testing.assert_array_equal(actual, [2.5, -1.5])
        raw[...] = 7
        np.testing.assert_array_equal(actual, [2.5, -1.5])

    def test_mutable_supplied_constants_cannot_override_saved_records(self):
        spin = np.eye(4, dtype=np.complex128)
        preps = {"x": _spin_preparation(spin)}
        physical = SimpleNamespace(constant_values={"x": (spin,)})
        loader = bind_prepared_loader(physical, preps, {"x": TensorSpec((4, 4, 2, 3), spin.dtype)})
        spin[:] = 9
        actual = loader.materialize("x", None, {"raw": np.ones((2, 3))})
        np.testing.assert_array_equal(actual, _outer_oracle(np.eye(4), np.ones((2, 3))))

    def test_replaced_preparation_invalidates_callable_constants_and_ownership(self):
        raw = np.ones((2, 3), dtype=np.complex128)
        preps = {"x": _spin_preparation(np.eye(4))}
        specs = {"x": TensorSpec((4, 4, 2, 3), np.complex128)}
        loader = bind_prepared_loader(None, preps, specs)
        loader.materialize("x", None, {"raw": raw})
        preps["x"] = SourcePreparation(("raw",), recipe=TensorPreparation("identity", ("raw",)))
        specs["x"] = TensorSpec(raw.shape, raw.dtype)
        self.assertFalse(loader.materializes_owned("x"))
        self.assertIs(loader.materialize("x", None, {"raw": raw}), raw)
        self.assertEqual(len(loader._constant_pool), 0)

    def test_replaced_stack_member_invalidates_existing_and_new_stacks(self):
        for warm_stack in (False, True):
            preps = {"x": _spin_preparation(np.eye(4)),
                     "s": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("x",)))}
            specs = {"x": TensorSpec((4, 4, 2, 3), np.float64),
                     "s": TensorSpec((1, 4, 4, 2, 3), np.float64)}
            loader = bind_prepared_loader(None, preps, specs)
            raw = np.ones((2, 3))
            loader.materialize("s" if warm_stack else "x", None, {"raw": raw})
            preps["x"] = _spin_preparation(np.full((4, 4), 7.0))
            actual = loader.materialize("s", None, {"raw": raw})
            np.testing.assert_array_equal(actual, np.full(specs["s"].shape, 7.0))
            self.assertEqual(len(loader._constant_pool), 1)

    def test_backend_rebinding_rebuilds_constants_and_numeric_callables(self):
        preps = {"x": _spin_preparation(np.eye(4))}
        specs = {"x": TensorSpec((4, 4, 2, 3), np.float64)}
        first_backend, second_backend = _backend(), _backend()
        first = bind_prepared_loader(None, preps, specs, backend=first_backend)
        raw = np.ones((2, 3))
        first.materialize("x", None, {"raw": raw})
        second = first.with_backend(second_backend)
        second.materialize("x", None, {"raw": raw})
        self.assertEqual(first_backend.multiply.call_count, 1)
        self.assertEqual(second_backend.multiply.call_count, 1)
        self.assertFalse(np.shares_memory(first.constants["x"][0], second.constants["x"][0]))
        new_preps = {"x": _spin_preparation(np.full((4, 4), 11.0))}
        rebound = bind_prepared_loader(first, new_preps, specs, backend=first_backend)
        np.testing.assert_array_equal(rebound.materialize("x", None, {"raw": raw}),
                                      np.full(specs["x"].shape, 11.0))

    def test_debug_checks_component_shape_and_dtype_before_direct_writes(self):
        preps = {"x": _spin_preparation(np.eye(4, dtype=np.complex128)),
                 "s": SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("x",)))}
        specs = {"x": TensorSpec((4, 4, 2, 3), np.complex64),
                 "s": TensorSpec((1, 4, 4, 2, 3), np.complex128)}
        loader = bind_prepared_loader(None, preps, specs)
        with execution_debug(True), self.assertRaisesRegex(ValueError, "metadata"):
            loader.materialize("s", None, {"raw": np.ones((2, 3))})
        specs["x"] = TensorSpec((4, 4, 1, 3), np.complex128)
        with execution_debug(True), self.assertRaisesRegex(ValueError, "metadata"):
            loader.materialize("s", None, {"raw": np.ones((1, 3))})

    def test_legacy_backends_can_keep_original_einsum_protocol(self):
        backend = SimpleNamespace(asarray=np.asarray, einsum=Mock(wraps=np.einsum))
        loader = bind_prepared_loader(None, {"x": _spin_preparation(np.eye(4))},
            {"x": TensorSpec((4, 4, 2, 3), np.float64)}, backend=backend)
        actual = loader.materialize("x", None, {"raw": np.ones((2, 3))})
        np.testing.assert_array_equal(actual, _outer_oracle(np.eye(4), np.ones((2, 3))))
        self.assertEqual(backend.einsum.call_count, 1)


class CuPyPreboundPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no device")
        except Exception as exc:
            raise unittest.SkipTest(f"CuPy device unavailable: {exc}")
        cls.backend = cupy

    def test_fixed_outer_dagger_and_mixed_precision_stack(self):
        cp = self.backend
        rng = np.random.default_rng(981)
        raw = (rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))).astype(np.complex64)
        spins = tuple((rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(dtype)
                      for dtype in (np.complex64, np.complex128))
        preps = {str(index): _spin_preparation(spin, conjugate=True, permutation=(1, 0))
                 for index, spin in enumerate(spins)}
        preps["s"] = SourcePreparation(("raw",), recipe=TensorPreparation("stack", ("0", "1")))
        specs = {str(index): TensorSpec((4, 4, 2, 3), spin.dtype)
                 for index, spin in enumerate(spins)}
        specs["s"] = TensorSpec((2, 4, 4, 2, 3), np.complex128)
        loader = bind_prepared_loader(None, preps, specs, backend=cp)
        device_raw = cp.asarray(raw)[:, ::-1]
        expected = np.stack([_outer_oracle(spin, raw[:, ::-1], conjugate=True, permutation=(1, 0))
                             for spin in spins])
        with execution_debug(True), patch.object(cp, "einsum", side_effect=AssertionError("einsum")):
            actual = loader.materialize("s", None, {"raw": device_raw})
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(cp.asnumpy(actual[0]), expected[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(cp.asnumpy(actual[1]), expected[1], rtol=3e-14, atol=3e-14)
        device_raw[:] = 0
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
