"""CPU protocol checks for bound GEMM group input slabs and legacy views."""
import gc
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide
from lqcdweave.contraction.execution.operations.gemm_group import _bind_group_inputs
from lqcdweave.contraction.execution.operations.packing_kernel import (
    _raw_kernel,
    bind_gemm_group_packing,
)


_MATRIX_SHAPES = ((2, 3), (3, 4))
_SIDE_KEYS = (("a0", "a1"), ("b0", "b1"))


class _FullSlab:
    """A complete owner that fails if the executor tries to make child views."""

    def __init__(self, label, shape=None, dtype=np.float64):
        self.label = label
        self.shape = None if shape is None else tuple(shape)
        self.dtype = np.dtype(dtype)
        self.flags = SimpleNamespace(c_contiguous=True)

    def __getitem__(self, key):
        raise AssertionError("the bound group path must pass its complete slab")


class _FakeSide:
    def __init__(self, key, transform=None):
        self.key = key
        self.transform = transform
        self.seen = []

    def pack(self, arrays, backend, dtype):
        value = arrays[self.key]
        self.seen.append(value)
        return value if self.transform is None else self.transform(value)


def _layouts(count=2):
    layouts = []
    for lane in range(count):
        layouts.append(SimpleNamespace(
            sides=(_FakeSide(f"a{lane}"), _FakeSide(f"b{lane}")),
            matrix_shapes=_MATRIX_SHAPES,
        ))
    return tuple(layouts)


def _fresh_arrays(count=2, generation=0):
    arrays = {}
    for lane in range(count):
        arrays[f"a{lane}"] = np.full((2, 3), generation + lane, dtype=np.float64)
        arrays[f"b{lane}"] = np.full((3, 4), generation + lane + 10, dtype=np.float64)
    return arrays


def _buffer_packer(side, *, return_none=False):
    seen, created = [], []
    keys = tuple(f"{side}{lane}" for lane in range(2))

    def pack_buffer(arrays):
        current = tuple(arrays[key] for key in keys)
        seen.append(current)
        if return_none:
            return None
        matrix_shape = _MATRIX_SHAPES[0 if side == "a" else 1]
        slab = _FullSlab(f"{side}-{len(created)}", (len(keys), *matrix_shape))
        created.append(slab)
        return slab

    return SimpleNamespace(pack_buffer=pack_buffer), seen, created


class GroupInputBindingProtocolTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_four_masks_keep_b_then_a_order_and_use_current_inputs(self):
        for mask in range(4):
            with self.subTest(mask=mask):
                layouts = _layouts()
                packer_info = {}
                packers = []
                for index, side in enumerate(("a", "b")):
                    if mask & (1 << index):
                        info = _buffer_packer(side)
                        packer_info[index] = info[1:]
                        packers.append(info[0])
                    else:
                        packers.append(None)
                pack_inputs = _bind_group_inputs(layouts, tuple(packers), np, np.dtype(np.float64))
                previous = None

                for generation in (1, 2):
                    arrays = _fresh_arrays(generation=generation)
                    owners = []
                    arguments, actual_mask = pack_inputs(arrays, owners)
                    self.assertEqual(actual_mask, mask)

                    expected_sides = []
                    for index, side in enumerate(("a", "b")):
                        if mask & (1 << index):
                            seen, created = packer_info[index]
                            self.assertEqual(len(seen[-1]), 2)
                            for key, value in zip(_SIDE_KEYS[index], seen[-1], strict=True):
                                self.assertIs(value, arrays[key])
                            self.assertEqual(created[-1].shape,
                                             (2, *_MATRIX_SHAPES[index]))
                            self.assertFalse(callable(getattr(created[-1], "reshape", None)))
                            expected_sides.append((created[-1],))
                            self.assertTrue(any(owner is created[-1] for owner in owners))
                        else:
                            expected_sides.append(tuple(arrays[key] for key in _SIDE_KEYS[index]))
                            for lane, key in enumerate(_SIDE_KEYS[index]):
                                self.assertIs(layouts[lane].sides[index].seen[-1], arrays[key])

                    # cuBLAS pointer setup consumes B first, then A.
                    expected = (*expected_sides[1], *expected_sides[0])
                    self.assertEqual(len(arguments), len(expected))
                    for actual, wanted in zip(arguments, expected, strict=True):
                        self.assertIs(actual, wanted)
                    if previous is not None:
                        for key in arrays:
                            self.assertIsNot(previous[key], arrays[key])
                    previous = arrays

    def test_one_none_side_keeps_its_child_matrices_and_other_side_slab(self):
        for missing_index in (0, 1):
            with self.subTest(missing_index=missing_index):
                layouts = _layouts()
                fallback_index = missing_index
                for layout in layouts:
                    side = layout.sides[fallback_index]
                    side.transform = lambda value: value.reshape(-1)

                packer_a, _, _ = _buffer_packer("a", return_none=missing_index == 0)
                packer_b, _, _ = _buffer_packer("b", return_none=missing_index == 1)
                pack_inputs = _bind_group_inputs(
                    layouts, (packer_a, packer_b), np, np.dtype(np.float64)
                )
                arrays = _fresh_arrays(generation=5)
                owners = []
                arguments, mask = pack_inputs(arrays, owners)

                slab_index = 1 - missing_index
                expected_mask = 1 << slab_index
                self.assertEqual(mask, expected_mask)
                if slab_index == 1:  # B is packed; A is two matrix arguments.
                    self.assertIsInstance(arguments[0], _FullSlab)
                    fallback_arguments = arguments[1:]
                    fallback_keys = _SIDE_KEYS[0]
                else:  # A is packed; B is two matrix arguments before it.
                    self.assertIsInstance(arguments[-1], _FullSlab)
                    fallback_arguments = arguments[:2]
                    fallback_keys = _SIDE_KEYS[1]

                self.assertEqual(len(fallback_arguments), 2)
                for value, key in zip(fallback_arguments, fallback_keys, strict=True):
                    self.assertEqual(value.shape, (np.prod(_MATRIX_SHAPES[fallback_index]),))
                    np.testing.assert_array_equal(value, arrays[key].reshape(-1))
                    self.assertTrue(np.shares_memory(value, arrays[key]))
                    self.assertEqual(value.ctypes.data, arrays[key].ctypes.data)
                    self.assertIsNot(value, arrays[key])
                for lane, layout in enumerate(layouts):
                    self.assertEqual(len(layout.sides[fallback_index].seen), 1)
                    self.assertEqual(len(layout.sides[slab_index].seen), 0)

    def test_debug_skips_buffer_packers_and_rejects_bad_matrices(self):
        layouts = _layouts()
        packers = tuple(SimpleNamespace(pack_buffer=Mock(return_value=_FullSlab(index)))
                        for index in range(2))
        pack_inputs = _bind_group_inputs(layouts, packers, np, np.dtype(np.float64))
        bad_values = (
            np.zeros((5,), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float32),
            np.ones((2, 6), dtype=np.float64)[:, ::2],
        )
        for bad in bad_values:
            with self.subTest(shape=bad.shape, dtype=bad.dtype,
                              contiguous=bad.flags.c_contiguous):
                arrays = _fresh_arrays()
                arrays["a0"] = bad
                with execution_debug(True):
                    with self.assertRaises(ValueError):
                        pack_inputs(arrays, [])
        for packer in packers:
            packer.pack_buffer.assert_not_called()

        with execution_debug(True):
            arguments, mask = pack_inputs(_fresh_arrays(generation=9), [])
        self.assertEqual(mask, 0)
        self.assertEqual(len(arguments), 4)
        for value, shape in zip(arguments, ((3, 4), (3, 4), (2, 3), (2, 3)), strict=True):
            self.assertEqual(value.shape, shape)
            self.assertEqual(value.dtype, np.dtype(np.float64))
            self.assertTrue(value.flags.c_contiguous)
        for packer in packers:
            packer.pack_buffer.assert_not_called()

    def test_multidimensional_full_slab_has_no_reshape_protocol(self):
        layouts = _layouts()
        a_slabs = tuple(_FullSlab(f"a-{lane}", (2, 1, 3)) for lane in range(2))
        b_slabs = tuple(_FullSlab(f"b-{lane}", (3, 1, 4)) for lane in range(2))
        for lane, layout in enumerate(layouts):
            layout.sides[0].transform = lambda _value, slab=a_slabs[lane]: slab
            layout.sides[1].transform = lambda _value, slab=b_slabs[lane]: slab
        owners = []
        arguments, mask = _bind_group_inputs(
            layouts, (None, None), np, np.dtype(np.float64)
        )(_fresh_arrays(generation=12), owners)

        self.assertEqual(mask, 0)
        self.assertEqual(tuple(value.shape for value in arguments),
                         ((3, 1, 4), (3, 1, 4), (2, 1, 3), (2, 1, 3)))
        self.assertTrue(all(actual is expected for actual, expected in zip(
            arguments, (b_slabs[0], b_slabs[1], a_slabs[0], a_slabs[1]), strict=True)))
        self.assertTrue(all(actual is expected for actual, expected in zip(
            owners, (*a_slabs, *b_slabs), strict=True)))
        self.assertFalse(any(callable(getattr(value, "reshape", None)) for value in arguments))

    def test_failed_second_side_keeps_first_owner_until_caller_clears_it(self):
        layouts = _layouts()
        refs, fail_second = [], [True]

        def owner_packer(label):
            def pack_buffer(arrays):
                slab = _FullSlab(label)
                refs.append(weakref.ref(slab))
                return slab
            return SimpleNamespace(pack_buffer=pack_buffer)

        def second_pack_buffer(arrays):
            if fail_second[0]:
                raise RuntimeError("second slab failed")
            slab = _FullSlab("right-next-call")
            refs.append(weakref.ref(slab))
            return slab

        pack_inputs = _bind_group_inputs(
            layouts, (owner_packer("left"), SimpleNamespace(pack_buffer=second_pack_buffer)),
            np, np.dtype(np.float64),
        )
        first_owners = []
        try:
            pack_inputs(_fresh_arrays(generation=1), first_owners)
        except RuntimeError as error:
            self.assertEqual(str(error), "second slab failed")
            error.__traceback__ = None
        else:
            self.fail("the second-side failure was not propagated")
        self.assertEqual(len(first_owners), 1)
        self.assertIs(first_owners[0], refs[0]())
        first_owners.clear()
        gc.collect()
        self.assertIsNone(refs[0]())

        fail_second[0] = False
        second_owners = []
        arguments, mask = pack_inputs(_fresh_arrays(generation=2), second_owners)
        self.assertEqual(mask, 3)
        self.assertIs(arguments[0], refs[2]())
        self.assertIs(arguments[1], refs[1]())
        del arguments
        second_owners.clear()
        gc.collect()
        self.assertIsNone(refs[1]())
        self.assertIsNone(refs[2]())

    def test_group_packing_buffer_is_whole_owner_while_legacy_pack_makes_views(self):
        count = 3
        calls, allocations = [], []
        counters = {"getitem": 0, "reshape": 0}
        state = {"deny_views": True}

        class TrackedSlab(np.ndarray):
            def __new__(cls, shape, dtype):
                value = np.empty(shape, dtype=dtype, order="C").view(cls)
                value.counters = counters
                value.deny_views = state["deny_views"]
                return value

            def __array_finalize__(self, source):
                self.counters = getattr(source, "counters", counters)
                self.deny_views = getattr(source, "deny_views", state["deny_views"])

            def __getitem__(self, key):
                if isinstance(key, (int, np.integer)):
                    if self.deny_views:
                        raise AssertionError("pack_buffer unexpectedly created child views")
                    self.counters["getitem"] += 1
                return super().__getitem__(key)

            def reshape(self, *args, **kwargs):
                if self.deny_views:
                    raise AssertionError("pack_buffer unexpectedly reshaped its owner")
                self.counters["reshape"] += 1
                return super().reshape(*args, **kwargs)

        def raw_factory(source, name, **kwargs):
            def noop_raw_kernel(grid, block, arguments):
                # This protocol fake records the launch only; it performs no math.
                calls.append((grid, block, tuple(arguments), arguments[-1]))
            return noop_raw_kernel

        def empty(shape, *, dtype, order):
            value = TrackedSlab(shape, dtype)
            allocations.append(value)
            return value

        backend = SimpleNamespace(__name__="cupy", RawKernel=raw_factory, empty=empty)
        catalog, left_sides, right_sides = {}, [], []
        for lane in range(count):
            left_keys = (f"left{lane}_0", f"left{lane}_1")
            right_keys = (f"right{lane}_0", f"right{lane}_1")
            for key, shape in zip(left_keys, ((2, 1), (2, 2)), strict=True):
                catalog[key] = TensorSpec(shape, np.float64)
            for key, shape in zip(right_keys, ((3, 2), (3, 2)), strict=True):
                catalog[key] = TensorSpec(shape, np.float64)
            left_sides.append(_BatchSide(
                tuple((key, (0, 1)) for key in left_keys), (2, 3), 1, False,
                original_packer=lambda *args: None, input_layouts=((0, 1), (0, 1)),
            ))
            right_sides.append(_BatchSide(
                tuple((key, (0, 1)) for key in right_keys), (3, 4), 1, False,
                original_packer=lambda *args: None, input_layouts=((0, 1), (0, 1)),
            ))

        arrays = {
            key: np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape).copy()
            for key, spec in catalog.items()
        }
        left_packer = bind_gemm_group_packing(
            backend, tuple(left_sides), catalog, (2, 3), np.float64
        )
        right_packer = bind_gemm_group_packing(
            backend, tuple(right_sides), catalog, (3, 4), np.float64
        )
        layouts = tuple(SimpleNamespace(
            sides=(left_sides[lane], right_sides[lane]),
            matrix_shapes=((2, 3), (3, 4)),
        ) for lane in range(count))

        owners = []
        arguments, mask = _bind_group_inputs(
            layouts, (left_packer, right_packer), backend, np.dtype(np.float64)
        )(arrays, owners)
        self.assertEqual(mask, 3)
        self.assertEqual([tuple(value.shape) for value in allocations],
                         [(count, 2, 3), (count, 3, 4)])
        self.assertEqual(len(owners), 2)
        self.assertIs(owners[0], allocations[0])
        self.assertIs(owners[1], allocations[1])
        self.assertIs(arguments[0], allocations[1])
        self.assertIs(arguments[1], allocations[0])
        self.assertEqual(counters, {"getitem": 0, "reshape": 0})
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][-1], allocations[0])
        self.assertIs(calls[1][-1], allocations[1])

        state["deny_views"] = True
        whole = left_packer.pack_buffer(arrays)
        self.assertIs(whole, allocations[-1])
        self.assertEqual(tuple(whole.shape), (count, 2, 3))
        self.assertEqual(counters, {"getitem": 0, "reshape": 0})
        self.assertIs(calls[-1][-1], whole)

        state["deny_views"] = False
        legacy = left_packer(arrays)
        legacy_owner = allocations[-1]
        self.assertEqual(len(legacy), count)
        self.assertEqual([tuple(value.shape) for value in legacy], [(2, 3)] * count)
        self.assertEqual(counters["getitem"], count)
        self.assertTrue(all(np.shares_memory(value, legacy_owner) for value in legacy))
        self.assertIs(calls[-1][-1], legacy_owner)


if __name__ == "__main__":
    unittest.main()
