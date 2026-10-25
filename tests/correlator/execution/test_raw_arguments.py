"""CPU protocol tests for per-thread RawKernel host argument slots."""
import ctypes
import gc
import threading
import unittest
import weakref
from types import SimpleNamespace

import numpy as np

from lqcdweave.contraction.execution.raw_arguments import _BoundPointerArguments, bind_pointer_arguments


class _FakeCNumpyArray:
    """Model CuPy's host-pointer wrapper around a one-word NumPy view."""

    constructed = 0

    def __init__(self, value):
        if not isinstance(value, np.ndarray) or value.size != 1:
            raise TypeError("CNumpyArray expects a one-element NumPy view")
        type(self).constructed += 1
        self.value = value
        self.slot_address = int(value.__array_interface__["data"][0])


class _HostArray:
    """Expose a NumPy buffer through the small array interface the binder uses."""

    def __init__(self, array):
        self.array = np.ascontiguousarray(array)
        self.data = SimpleNamespace(ptr=int(self.array.ctypes.data))


class _RawKernelImplementation:
    def __init__(self, *, barrier=None, error=None):
        self.barrier = barrier
        self.error = error
        self.calls = []

    @staticmethod
    def _read_slot(pointer):
        return int(ctypes.c_uint64.from_address(pointer.slot_address).value)

    @staticmethod
    def _signed_slot(pointer):
        return int(ctypes.c_int64.from_address(pointer.slot_address).value)

    def __call__(self, grid, block, arguments):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)

        slot_addresses = tuple(pointer.slot_address for pointer in arguments)
        words = tuple(self._read_slot(pointer) for pointer in arguments)
        bias = self._signed_slot(arguments[1])
        offset = self._signed_slot(arguments[2])
        record = {
            "grid": grid,
            "block": block,
            "slot_addresses": slot_addresses,
            "words": words,
            "bias": bias,
            "offset": offset,
        }
        self.calls.append(record)

        if self.error is not None:
            raise self.error

        source = ctypes.cast(words[0], ctypes.POINTER(ctypes.c_int64))
        constant = ctypes.cast(words[3], ctypes.POINTER(ctypes.c_int64))
        target = ctypes.cast(words[4], ctypes.POINTER(ctypes.c_int64))
        for index in range(3):
            target[index] = source[offset + index] + constant[index] + bias


# Capability checks intentionally inspect the concrete class name and module.
_RawKernelImplementation.__name__ = "RawKernel"
_RawKernelImplementation.__module__ = "cupy._core.raw"


class _CustomKernel:
    def __call__(self, grid, block, arguments):
        raise AssertionError("unsupported custom kernel must not be called")


def _backend(*, name="cupy", is_hip=False,
             argument_type=_FakeCNumpyArray, cpointer_type=None):
    function = SimpleNamespace()
    if argument_type is not None:
        function.CNumpyArray = argument_type
    if cpointer_type is not None:
        function.CPointer = cpointer_type
    return SimpleNamespace(
        __name__=name,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(is_hip=is_hip),
            function=function,
        ),
    )


def _template(constant):
    # The negative scalar checks signed int64 packing; offset exercises a
    # positive scalar that the fake kernel applies to its input pointer.
    return (None, np.int64(-7), np.int64(2), constant, None)


def _bind(kernel, constant, *, backend=None, positions=((0, 0),)):
    return bind_pointer_arguments(
        backend or _backend(), kernel, (1,), (32,), _template(constant), positions
    )


def _uint64_bits(value):
    return ctypes.c_uint64(int(value)).value


class RawArgumentsCpuTests(unittest.TestCase):
    def setUp(self):
        _FakeCNumpyArray.constructed = 0

    def _assert_idle_slots(self, binding, constant_pointer):
        words = binding._local.storage[0]
        self.assertEqual(int(words[0]), 0)
        self.assertEqual(int(words[1]), _uint64_bits(-7))
        self.assertEqual(int(words[2]), 2)
        self.assertEqual(int(words[3]), constant_pointer)
        self.assertEqual(int(words[4]), 0)

    def test_fragmented_parameters_restore_fixed_bytes_after_success_and_failure(self):
        template = tuple(value for index in range(8)
                         for value in (None, np.int64(-index - 1))) + (None,)
        records, fail_next = [], []

        def kernel(grid, block, parameters):
            words = tuple(int(parameter.value[0]) for parameter in parameters)
            records.append(words)
            if fail_next:
                raise fail_next.pop()
            value = sum(ctypes.c_int64.from_address(words[2 * index]).value
                        + ctypes.c_int64(words[2 * index + 1]).value for index in range(8))
            ctypes.c_int64.from_address(words[-1]).value = value

        binding = _BoundPointerArguments(kernel, (1,), (1,), _FakeCNumpyArray,
            template, tuple((index * 2, index) for index in range(8)))
        refs = []
        for phase in range(3):
            inputs = tuple(_HostArray(np.array([phase * 10 + index], dtype=np.int64)) for index in range(8))
            output = _HostArray(np.zeros(1, dtype=np.int64))
            refs.extend(weakref.ref(value) for value in (*inputs, output))
            if phase == 1:
                error = RuntimeError('fragmented launch failure')
                fail_next.append(error)
                with self.assertRaises(RuntimeError) as caught:
                    binding(inputs, output)
                self.assertIs(caught.exception, error)
                # The retained traceback must not be mistaken for a binding leak.
                error.__traceback__ = None
            else:
                binding(inputs, output)
                self.assertEqual(output.array[0], sum(phase * 10 - 1 for _ in range(8)))
            self.assertEqual(records[-1][::2], tuple(value.data.ptr for value in inputs) + (output.data.ptr,))
            self.assertEqual(tuple(binding._local.storage[0]), tuple(map(int, binding.fixed_words)))
            self.assertEqual(_FakeCNumpyArray.constructed, len(template))
            del inputs, output
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))

    def test_reuses_slots_updates_values_and_does_not_retain_dynamic_arrays(self):
        constant_array = np.array([10, -4, 7], dtype=np.int64)
        constant = _HostArray(constant_array)
        constant_pointer = constant.data.ptr
        constant_ref = weakref.ref(constant)
        constant_array_ref = weakref.ref(constant_array)
        kernel = _RawKernelImplementation()
        binding = _bind(kernel, constant)

        # The binding owns fixed pointer constants for as long as its ABI lives.
        del constant, constant_array
        gc.collect()
        self.assertIsNotNone(constant_ref())
        self.assertIsNotNone(constant_array_ref())

        cases = (
            np.array([31, 14, 6, -9, 25], dtype=np.int64),
            np.array([-20, 8, 70, 12, -5], dtype=np.int64),
        )
        first_constructor_count = None
        source_pointers = []
        output_pointers = []
        for values in cases:
            source = values.copy()
            source_wrapper = _HostArray(source)
            target = np.full(3, 999, dtype=np.int64)
            target_wrapper = _HostArray(target)
            source_pointer = source_wrapper.data.ptr
            output_pointer = target_wrapper.data.ptr
            expected = source[2:5].copy() + np.array([10, -4, 7], dtype=np.int64) - 7
            source_wrapper_ref = weakref.ref(source_wrapper)
            source_ref = weakref.ref(source)
            target_wrapper_ref = weakref.ref(target_wrapper)
            target_ref = weakref.ref(target)

            binding((source_wrapper,), target_wrapper)
            np.testing.assert_array_equal(target, expected)
            source_pointers.append(source_pointer)
            output_pointers.append(output_pointer)
            self._assert_idle_slots(binding, constant_pointer)

            if first_constructor_count is None:
                first_constructor_count = _FakeCNumpyArray.constructed
                self.assertEqual(first_constructor_count, 5)
            else:
                self.assertEqual(_FakeCNumpyArray.constructed, first_constructor_count)

            del source_wrapper, source, target_wrapper, target
            gc.collect()
            self.assertIsNone(source_wrapper_ref())
            self.assertIsNone(source_ref())
            self.assertIsNone(target_wrapper_ref())
            self.assertIsNone(target_ref())

        self.assertEqual(len(kernel.calls), 2)
        self.assertEqual([call["words"][0] for call in kernel.calls], source_pointers)
        self.assertEqual([call["words"][4] for call in kernel.calls], output_pointers)
        self.assertEqual([call["words"][3] for call in kernel.calls], [constant_pointer] * 2)
        self.assertEqual([call["bias"] for call in kernel.calls], [-7, -7])
        self.assertEqual([call["offset"] for call in kernel.calls], [2, 2])

        del binding
        gc.collect()
        self.assertIsNone(constant_ref())
        self.assertIsNone(constant_array_ref())

    def test_kernel_error_clears_dynamic_slots_and_preserves_exception(self):
        constant = _HostArray(np.array([1, 2, 3], dtype=np.int64))
        kernel_error = RuntimeError("kernel failure sentinel")
        kernel = _RawKernelImplementation(error=kernel_error)
        binding = _bind(kernel, constant)
        source = np.array([4, 5, 6, 7, 8], dtype=np.int64)
        target = np.full(3, -100, dtype=np.int64)
        source_wrapper = _HostArray(source)
        target_wrapper = _HostArray(target)

        with self.assertRaises(RuntimeError) as caught:
            binding((source_wrapper,), target_wrapper)

        self.assertIs(caught.exception, kernel_error)
        self._assert_idle_slots(binding, constant.data.ptr)
        self.assertEqual(_FakeCNumpyArray.constructed, 5)
        self.assertEqual(kernel.calls[0]["words"][0], source.ctypes.data)
        self.assertEqual(kernel.calls[0]["words"][4], target.ctypes.data)

    def test_thread_local_argument_storage_isolated_during_simultaneous_calls(self):
        barrier = threading.Barrier(2)
        kernel = _RawKernelImplementation(barrier=barrier)
        constant_array = np.array([3, 5, 7], dtype=np.int64)
        constant = _HostArray(constant_array)
        constant_pointer = constant.data.ptr
        binding = _bind(kernel, constant)
        outcomes = {}
        errors = []

        def worker(name, values):
            try:
                source = np.array(values, dtype=np.int64)
                target = np.full(3, -1, dtype=np.int64)
                source_wrapper = _HostArray(source)
                target_wrapper = _HostArray(target)
                expected = source[2:5].copy() + constant_array - 7
                source_pointer = source.ctypes.data
                output_pointer = target.ctypes.data
                binding((source_wrapper,), target_wrapper)
                outcomes[name] = (
                    source_pointer,
                    output_pointer,
                    target.copy(),
                    expected,
                )
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=worker, args=("first", [9, 8, 21, 22, 23]))
        second = threading.Thread(target=worker, args=("second", [100, 101, -3, 4, 5]))
        first.start()
        second.start()
        first.join(timeout=15)
        second.join(timeout=15)

        self.assertFalse(first.is_alive(), "first worker did not finish")
        self.assertFalse(second.is_alive(), "second worker did not finish")
        self.assertEqual(errors, [])
        self.assertEqual(set(outcomes), {"first", "second"})
        self.assertEqual(_FakeCNumpyArray.constructed, 10)
        self.assertEqual(len(kernel.calls), 2)

        for result in outcomes.values():
            np.testing.assert_array_equal(result[2], result[3])
        records = {record["words"][0]: record for record in kernel.calls}
        for source_pointer, output_pointer, _, _ in outcomes.values():
            record = records[source_pointer]
            self.assertEqual(record["words"][4], output_pointer)
            self.assertEqual(record["words"][3], constant_pointer)
            self.assertEqual(record["bias"], -7)
            self.assertEqual(record["offset"], 2)

        first_slots, second_slots = (record["slot_addresses"] for record in kernel.calls)
        self.assertTrue(set(first_slots).isdisjoint(second_slots))

    def test_unsupported_backends_and_kernels_fall_back_to_none(self):
        constant = _HostArray(np.array([1, 2, 3], dtype=np.int64))
        template = _template(constant)
        raw_kernel = _RawKernelImplementation()

        self.assertIsNone(bind_pointer_arguments(
            _backend(is_hip=True), raw_kernel, (1,), (32,), template, ((0, 0),)
        ))
        self.assertIsNone(bind_pointer_arguments(
            _backend(argument_type=None), raw_kernel, (1,), (32,), template, ((0, 0),)
        ))
        legacy_pointer = lambda value: self.fail("legacy CPointer must not be used")
        self.assertIsNone(bind_pointer_arguments(
            _backend(argument_type=None, cpointer_type=legacy_pointer),
            raw_kernel, (1,), (32,), template, ((0, 0),)
        ))
        self.assertIsNone(bind_pointer_arguments(
            _backend(), _CustomKernel(), (1,), (32,), template, ((0, 0),)
        ))
        self.assertIsNone(bind_pointer_arguments(
            _backend(name="numpy"), raw_kernel, (1,), (32,), template, ((0, 0),)
        ))


if __name__ == "__main__":
    unittest.main()
