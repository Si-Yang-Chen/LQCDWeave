"""CPU protocol and opt-in CUDA tests for RawKernel driver argument tables.

Importing this module is CPU-safe.  CuPy is loaded only by the CUDA test
fixture, so callers can select ``RawDriverArgumentsCpuTests`` without loading
the CUDA runtime.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import itertools
import threading
import unittest
import weakref
from types import SimpleNamespace
from weakref import WeakValueDictionary

import numpy as np

from lqcdweave.contraction.execution.raw_arguments import (
    _BoundDriverArguments,
    _BoundPointerArguments,
    bind_pointer_arguments,
)


_POINTERS = itertools.count(0x10000, 0x100)


class _FakeCNumpyArray:
    """Keep the one-word NumPy view used by the fallback CPointer wrapper."""

    constructed = 0

    def __init__(self, value):
        if not isinstance(value, np.ndarray) or value.size != 1:
            raise TypeError("CNumpyArray expects a one-element NumPy view")
        type(self).constructed += 1
        self.value = value
        self.slot_address = int(value.__array_interface__["data"][0])


class _FakeArray:
    def __init__(self, registry: WeakValueDictionary, values, label="array"):
        self.array = np.ascontiguousarray(values, dtype=np.int64)
        self.data = SimpleNamespace(ptr=int(self.array.ctypes.data))
        self.label = label
        registry[self.data.ptr] = self


class _FakeFunction:
    def __init__(self, ptr, device):
        self.ptr = ptr
        self.device = device


class _FakeStream:
    def __init__(self, ptr, device):
        self.ptr = ptr
        self.device = device


class _FakeRuntime:
    def __init__(self, events, *, mem_get_info=True):
        self.events = events
        self.is_hip = False
        self.calls = []
        self.fail_next = None
        if mem_get_info:
            self.memGetInfo = self._mem_get_info
        else:
            self.memGetInfo = None
        self.device_getter = lambda: None

    def _mem_get_info(self):
        device = self.device_getter()
        self.calls.append({"device": device})
        self.events.append(("mem_get_info", device))
        error, self.fail_next = self.fail_next, None
        if error is not None:
            raise error
        return 1024, 2048


class _FakeDriver:
    def __init__(self, registry, events, *, launch=True, barrier=None):
        self.registry = registry
        self.events = events
        self.barrier = barrier
        self.fail_next = None
        self.records = []
        if launch:
            self.launchKernel = self._launch_kernel
        else:
            self.launchKernel = None

    @staticmethod
    def _read_words(table_pointer, count):
        table = ctypes.cast(int(table_pointer), ctypes.POINTER(ctypes.c_void_p))
        addresses = tuple(int(table[index]) for index in range(count))
        words = tuple(ctypes.c_uint64.from_address(address).value for address in addresses)
        signed = tuple(ctypes.c_int64.from_address(address).value for address in addresses)
        return addresses, words, signed

    def _launch_kernel(self, function_ptr, gx, gy, gz, bx, by, bz,
                       shared_bytes, stream_ptr, table_pointer, extra):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        error, self.fail_next = self.fail_next, None
        if error is not None:
            self.events.append(("launch_error", threading.current_thread().name))
            raise error

        addresses, words, signed = self._read_words(table_pointer, 4)
        source = self.registry[words[0]].array
        constant = self.registry[words[2]].array
        destination = self.registry[words[3]].array
        destination[:] = source[:destination.size] + constant + signed[1]
        record = {
            "thread": threading.current_thread().name,
            "function_ptr": int(function_ptr),
            "geometry": (gx, gy, gz, bx, by, bz),
            "shared_bytes": shared_bytes,
            "stream_ptr": int(stream_ptr),
            "table_pointer": int(table_pointer),
            "parameter_addresses": addresses,
            "words": words,
            "signed_words": signed,
            "input_ptr": words[0],
            "scalar": signed[1],
            "constant_ptr": words[2],
            "output_ptr": words[3],
            "output": tuple(int(value) for value in destination),
        }
        self.records.append(record)
        self.events.append(("launch", record["thread"], int(function_ptr), int(stream_ptr)))


class _FakeBackend:
    def __init__(self, *, launch=True, mem_get_info=True, current_stream=True,
                 barrier=None):
        self.__name__ = "cupy"
        self.events = []
        self.registry = WeakValueDictionary()
        self.device = 0
        self.functions = {0: _FakeFunction(0xA001, 0), 1: _FakeFunction(0xB002, 1)}
        self.streams = {0: _FakeStream(0xC003, 0), 1: _FakeStream(0xD004, 1)}
        self.driver = _FakeDriver(self.registry, self.events,
                                  launch=launch, barrier=barrier)
        self.runtime = _FakeRuntime(self.events, mem_get_info=mem_get_info)
        self.runtime.device_getter = lambda: self.device
        self.current_stream_enabled = current_stream
        cuda = SimpleNamespace(
            driver=self.driver,
            runtime=self.runtime,
            function=SimpleNamespace(CNumpyArray=_FakeCNumpyArray),
            get_current_stream=self.get_current_stream if current_stream else None,
        )
        self.cuda = cuda

    def get_current_stream(self):
        stream = self.streams[self.device]
        self.events.append(("get_stream", self.device, stream.ptr))
        return stream


class _FakeRawKernel:
    """Numerical fake for the normal CPointer path; metadata matches CuPy RawKernel."""

    def __init__(self, backend, *, barrier=None, error=None, cooperative=False):
        self.backend = backend
        self.barrier = barrier
        self.error = error
        self.enable_cooperative_groups = cooperative
        self.calls = []

    @property
    def kernel(self):
        function = self.backend.functions[self.backend.device]
        self.backend.events.append(("function_lookup", self.backend.device, function.ptr))
        return function

    @staticmethod
    def _read_slot(pointer, signed=False):
        scalar = ctypes.c_int64 if signed else ctypes.c_uint64
        return int(scalar.from_address(pointer.slot_address).value)

    def __call__(self, grid, block, arguments):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        addresses = tuple(pointer.slot_address for pointer in arguments)
        words = tuple(self._read_slot(pointer) for pointer in arguments)
        signed = tuple(self._read_slot(pointer, signed=True) for pointer in arguments)
        record = {"grid": grid, "block": block, "addresses": addresses,
                  "words": words, "signed_words": signed}
        self.calls.append(record)
        if self.error is not None:
            raise self.error
        source = ctypes.cast(words[0], ctypes.POINTER(ctypes.c_int64))
        constant = ctypes.cast(words[2], ctypes.POINTER(ctypes.c_int64))
        destination = ctypes.cast(words[3], ctypes.POINTER(ctypes.c_int64))
        for index in range(3):
            destination[index] = source[index] + constant[index] + signed[1]


# bind_pointer_arguments deliberately checks the concrete class metadata.
_FakeRawKernel.__name__ = "RawKernel"
_FakeRawKernel.__module__ = "cupy._core.raw"


def _template(constant):
    return (None, np.int64(-7), constant, None)


def _new_driver_binding(backend=None, *, kernel=None, barrier=None):
    backend = backend or _FakeBackend(barrier=barrier)
    constant = _FakeArray(backend.registry, [10, -4, 7], "constant")
    constant_ref = weakref.ref(constant)
    constant_array_ref = weakref.ref(constant.array)
    kernel = kernel or _FakeRawKernel(backend, barrier=barrier)
    binding = bind_pointer_arguments(
        backend, kernel, (7,), (3,), _template(constant), ((0, 0),)
    )
    return backend, kernel, binding, constant, constant_ref, constant_array_ref


def _expected(source, constant):
    return source[:3].copy() + constant + np.int64(-7)


class RawDriverArgumentsCpuTests(unittest.TestCase):
    def setUp(self):
        _FakeCNumpyArray.constructed = 0

    def test_driver_table_reuses_slots_signed_values_and_owner_lifetimes(self):
        backend, _kernel, binding, constant, constant_ref, constant_array_ref = (
            _new_driver_binding()
        )
        self.assertIs(type(binding), _BoundDriverArguments)
        constant_values = constant.array.copy()
        constant_pointer = constant.data.ptr
        del constant
        gc.collect()
        self.assertIsNotNone(constant_ref())
        self.assertIsNotNone(constant_array_ref())

        inputs, outputs = [], []
        source_refs, output_refs = [], []
        dynamic_owners = []
        records = []
        for values in (np.array([31, 14, 6, -9, 25], dtype=np.int64),
                       np.array([-20, 8, 70, 12, -5], dtype=np.int64)):
            source = _FakeArray(backend.registry, values, "input")
            destination = _FakeArray(backend.registry, [999, 999, 999], "output")
            inputs.append(source.data.ptr)
            outputs.append(destination.data.ptr)
            source_refs.append(weakref.ref(source))
            output_refs.append(weakref.ref(destination))
            dynamic_owners.append((source, destination))
            binding((source,), destination)
            np.testing.assert_array_equal(destination.array, _expected(values, constant_values))
            records.append(backend.driver.records[-1])
            words, parameters = binding._local.storage
            self.assertEqual((int(words[0]), int(words[3])), (source.data.ptr, destination.data.ptr))
            self.assertEqual(int(words[1]), ctypes.c_uint64(-7).value)
            self.assertEqual(int(words[2]), constant_pointer)
            self.assertIsInstance(parameters[0], np.ndarray)
            self.assertEqual(parameters[1], parameters[0].ctypes.data)
            self.assertEqual(parameters[2].ptr, backend.functions[0].ptr)

        first, second = records
        with self.subTest(protocol="fixed host table and parameter slots"):
            self.assertEqual(first["geometry"], (7, 1, 1, 3, 1, 1))
            self.assertEqual(first["parameter_addresses"], tuple(
                first["parameter_addresses"][0] + index * 8 for index in range(4)
            ))
            self.assertEqual(first["table_pointer"], second["table_pointer"])
            self.assertEqual(first["parameter_addresses"], second["parameter_addresses"])
            self.assertEqual(first["words"][1], ctypes.c_uint64(-7).value)
            self.assertEqual(first["signed_words"][1], -7)
            self.assertEqual(first["constant_ptr"], second["constant_ptr"])
        with self.subTest(protocol="fresh dynamic pointers and values"):
            self.assertEqual((first["input_ptr"], second["input_ptr"]), tuple(inputs))
            self.assertEqual((first["output_ptr"], second["output_ptr"]), tuple(outputs))
            self.assertNotEqual(first["input_ptr"], second["input_ptr"])
            self.assertNotEqual(first["output_ptr"], second["output_ptr"])
        del dynamic_owners
        del source, destination
        gc.collect()
        with self.subTest(lifetime="dynamic arrays are not cached"):
            self.assertTrue(all(reference() is None for reference in source_refs + output_refs))
        with self.subTest(lifetime="constant is held by binding"):
            self.assertIsNotNone(constant_ref())
            self.assertIsNotNone(constant_array_ref())
        del binding
        gc.collect()
        with self.subTest(lifetime="constant is released with binding"):
            self.assertIsNone(constant_ref())
            self.assertIsNone(constant_array_ref())
        self.assertEqual(_FakeCNumpyArray.constructed, 0)

    def test_context_stream_function_refresh_and_original_errors(self):
        backend, _kernel, binding, _constant, _, _ = _new_driver_binding()
        self.assertIs(type(binding), _BoundDriverArguments)

        def invoke(label):
            values = np.array([31, 14, 6, -9, 25], dtype=np.int64)
            source = _FakeArray(backend.registry, values, f"input-{label}")
            destination = _FakeArray(backend.registry, [0, 0, 0], f"output-{label}")
            binding((source,), destination)
            return backend.driver.records[-1]

        first = invoke("first")
        event_names = [event[0] for event in backend.events]
        with self.subTest(device="first-use context before stream and launch"):
            self.assertLess(event_names.index("mem_get_info"), event_names.index("get_stream"))
            self.assertLess(event_names.index("mem_get_info"), event_names.index("launch"))
            self.assertEqual(first["function_ptr"], backend.functions[0].ptr)
            self.assertEqual(first["stream_ptr"], backend.streams[0].ptr)
            self.assertEqual(len(backend.runtime.calls), 1)

        backend.streams[0] = _FakeStream(0xC103, 0)
        second = invoke("new-stream")
        with self.subTest(device="same function with current stream refresh"):
            self.assertEqual(second["function_ptr"], first["function_ptr"])
            self.assertEqual(second["stream_ptr"], backend.streams[0].ptr)
            self.assertNotEqual(second["stream_ptr"], first["stream_ptr"])
            self.assertEqual(len(backend.runtime.calls), 1)

        backend.device = 1
        third = invoke("new-device")
        with self.subTest(device="function change rechecks context"):
            self.assertEqual([event[0] for event in backend.events[-4:]],
                             ["function_lookup", "mem_get_info", "get_stream", "launch"])
            self.assertEqual(third["function_ptr"], backend.functions[1].ptr)
            self.assertEqual(third["stream_ptr"], backend.streams[1].ptr)
            self.assertEqual(len(backend.runtime.calls), 2)

        failing_backend = _FakeBackend()
        failing_binding = bind_pointer_arguments(
            failing_backend, _FakeRawKernel(failing_backend), (7,), (3,),
            _template(_FakeArray(failing_backend.registry, [10, -4, 7], "constant")),
            ((0, 0),),
        )
        source = _FakeArray(failing_backend.registry, [5, 6, 7, 8, 9], "failure-input")
        destination = _FakeArray(failing_backend.registry, [0, 0, 0], "failure-output")
        context_error = RuntimeError("original memGetInfo failure")
        failing_backend.runtime.fail_next = context_error
        with self.subTest(error="first context establishment"):
            with self.assertRaises(RuntimeError) as caught:
                failing_binding((source,), destination)
            self.assertIs(caught.exception, context_error)
            words, parameters = failing_binding._local.storage
            self.assertEqual((int(words[0]), int(words[-1])), (source.data.ptr, destination.data.ptr))
            self.assertIsNone(parameters[2])
            self.assertEqual(len(failing_backend.driver.records), 0)

        driver_error = ValueError("original driver launch failure")
        failing_backend.driver.fail_next = driver_error
        with self.subTest(error="driver launch"):
            with self.assertRaises(ValueError) as caught:
                failing_binding((source,), destination)
            self.assertIs(caught.exception, driver_error)
            words, parameters = failing_binding._local.storage
            self.assertEqual((int(words[0]), int(words[-1])), (source.data.ptr, destination.data.ptr))
            self.assertIs(parameters[2], failing_backend.functions[0])

    def test_partial_address_failure_does_not_submit_or_retain_owners(self):
        backend, _kernel, binding, constant, _, _ = _new_driver_binding()
        first = _FakeArray(backend.registry, [10, 20, 30, 40, 50], "first")
        output = _FakeArray(backend.registry, [0, 0, 0], "output")
        first_ref, output_ref = weakref.ref(first), weakref.ref(output)
        binding((first,), output)
        self.assertFalse(binding._clear_after_launch)
        self.assertEqual(tuple(map(int, binding._local.storage[0])),
                         backend.driver.records[-1]["words"])
        del first, output
        gc.collect()
        self.assertIsNone(first_ref())
        self.assertIsNone(output_ref())

        error = RuntimeError("destination address unavailable")
        class BrokenOutput:
            @property
            def data(self):
                raise error

        current = _FakeArray(backend.registry, [3, 4, 5, 6, 7], "partial-input")
        submitted = len(backend.driver.records)
        with self.assertRaises(RuntimeError) as caught:
            binding((current,), BrokenOutput())
        self.assertIs(caught.exception, error)
        self.assertEqual(len(backend.driver.records), submitted)
        error.__traceback__ = None

        # The next complete fill replaces every current pointer before launch.
        fresh = _FakeArray(backend.registry, [80, 70, 60, 50, 40], "fresh-input")
        target = _FakeArray(backend.registry, [0, 0, 0], "fresh-output")
        binding((fresh,), target)
        record = backend.driver.records[-1]
        self.assertEqual((record["words"][0], record["words"][-1]),
                         (fresh.data.ptr, target.data.ptr))
        np.testing.assert_array_equal(target.array, _expected(fresh.array, constant.array))

    def test_simultaneous_threads_use_separate_parameter_tables_and_slots(self):
        barrier = threading.Barrier(2)
        backend, _kernel, binding, _constant, _, _ = _new_driver_binding(
            _FakeBackend(barrier=barrier), barrier=barrier
        )
        self.assertIs(type(binding), _BoundDriverArguments)
        sources = [
            _FakeArray(backend.registry, [9, 8, 21, 22, 23], "thread-a-input"),
            _FakeArray(backend.registry, [100, 101, -3, 4, 5], "thread-b-input"),
        ]
        destinations = [
            _FakeArray(backend.registry, [0, 0, 0], "thread-a-output"),
            _FakeArray(backend.registry, [0, 0, 0], "thread-b-output"),
        ]
        errors = []

        def invoke(index):
            try:
                binding((sources[index],), destinations[index])
            except BaseException as error:  # Surface exceptions from worker threads.
                errors.append(error)

        threads = [threading.Thread(target=invoke, args=(index,), name=f"driver-cpu-{index}")
                   for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread barrier timed out")
        self.assertEqual(errors, [])
        self.assertEqual(len(backend.driver.records), 2)
        expected_by_ptr = {
            sources[0].data.ptr: (12, -3, 21),
            sources[1].data.ptr: (103, 90, -3),
        }
        for record in backend.driver.records:
            with self.subTest(thread=record["thread"]):
                self.assertEqual(record["output"], expected_by_ptr[record["input_ptr"]])
                self.assertEqual(record["scalar"], -7)
        first, second = backend.driver.records
        with self.subTest(thread_storage="separate host tables"):
            self.assertNotEqual(first["table_pointer"], second["table_pointer"])
            self.assertNotEqual(first["parameter_addresses"], second["parameter_addresses"])
        self.assertEqual(len(backend.runtime.calls), 2)

    def test_incomplete_driver_or_cooperative_kernel_uses_pointer_fallback(self):
        cases = (
            ("missing launchKernel", {"launch": False}, False),
            ("missing memGetInfo", {"mem_get_info": False}, False),
            ("missing current stream", {"current_stream": False}, False),
            ("cooperative groups", {}, True),
        )
        for label, backend_options, cooperative in cases:
            with self.subTest(capability=label):
                _FakeCNumpyArray.constructed = 0
                backend = _FakeBackend(**backend_options)
                constant = _FakeArray(backend.registry, [10, -4, 7], "constant")
                kernel = _FakeRawKernel(backend, cooperative=cooperative)
                binding = bind_pointer_arguments(
                    backend, kernel, (7,), (3,), _template(constant), ((0, 0),)
                )
                self.assertIs(type(binding), _BoundPointerArguments)
                source_values = np.array([31, 14, 6, -9, 25], dtype=np.int64)
                source = _FakeArray(backend.registry, source_values, "fallback-input")
                destination = _FakeArray(backend.registry, [0, 0, 0], "fallback-output")
                binding((source,), destination)
                np.testing.assert_array_equal(
                    destination.array, _expected(source_values, constant.array)
                )
                self.assertEqual(_FakeCNumpyArray.constructed, 4)
                self.assertEqual(len(kernel.calls), 1)
                self.assertEqual(len(backend.driver.records), 0)
                words = binding._local.storage[0]
                self.assertEqual((int(words[0]), int(words[-1])), (0, 0))
                self.assertEqual(int(words[2]), constant.data.ptr)


class _RecordingCudaDriver:
    """Observe host parameter slots, then forward to CuPy's real driver API."""

    def __init__(self, driver, parameter_count):
        self.driver = driver
        self.parameter_count = parameter_count
        self.records = []

    def launchKernel(self, function_ptr, gx, gy, gz, bx, by, bz,
                     shared_bytes, stream_ptr, table_pointer, extra):
        table = ctypes.cast(int(table_pointer), ctypes.POINTER(ctypes.c_void_p))
        addresses = tuple(int(table[index]) for index in range(self.parameter_count))
        words = tuple(ctypes.c_uint64.from_address(address).value for address in addresses)
        signed = tuple(ctypes.c_int64.from_address(address).value for address in addresses)
        self.records.append({
            "thread": threading.current_thread().name,
            "function_ptr": int(function_ptr),
            "stream_ptr": int(stream_ptr),
            "table_pointer": int(table_pointer),
            "parameter_addresses": addresses,
            "words": words,
            "signed_words": signed,
            "geometry": (gx, gy, gz, bx, by, bz),
            "shared_bytes": shared_bytes,
        })
        return self.driver.launchKernel(function_ptr, gx, gy, gz, bx, by, bz,
                                        shared_bytes, stream_ptr, table_pointer, extra)


def _cupy_backend(cp, parameter_count):
    driver = _RecordingCudaDriver(cp.cuda.driver, parameter_count)
    cuda = SimpleNamespace(
        driver=driver,
        runtime=cp.cuda.runtime,
        function=cp.cuda.function,
        get_current_stream=cp.cuda.get_current_stream,
    )
    return SimpleNamespace(__name__="cupy", cuda=cuda), driver


def _vector_kernel_source(name, size):
    return f'''extern "C" __global__ void {name}(
        const long long* input, long long offset,
        const long long* constant_value, long long* output) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}) output[i] = input[i] + offset + constant_value[0];
}}'''


def _many_pointer_kernel_source(name, pointer_count, size):
    parameters = [f"const long long* x{index}" for index in range(pointer_count)]
    parameters += ["long long offset", "long long* output"]
    terms = " + ".join(f"x{index}[i]" for index in range(pointer_count))
    return (f'extern "C" __global__ void {name}(' + ", ".join(parameters) + ") {\n"
            "    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"    if (i < {size}) output[i] = {terms} + offset;\n"
            "}")


class RawDriverArgumentsCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Do not skip: an unavailable CuPy/CUDA environment is a gate error.
        try:
            cls.cp = importlib.import_module("cupy")
            cls.device_id = int(cls.cp.cuda.runtime.getDevice())
            cls.cp.cuda.Device(cls.device_id).use()
        except Exception as error:
            raise RuntimeError("RawDriverArgumentsCuPyTests requires a working CuPy CUDA device") from error

    def test_signed_pointer_constant_table_reuse_on_nondefault_stream(self):
        cp = self.cp
        size = 256
        scalar = np.int64(-7)
        host_constant = np.asarray([19], dtype=np.int64)
        constant = cp.asarray(host_constant)
        kernel = cp.RawKernel(
            _vector_kernel_source("raw_driver_parameter_vector", size),
            "raw_driver_parameter_vector",
            enable_cooperative_groups=False,
        )
        backend, driver = _cupy_backend(cp, 4)
        warm_input = cp.asarray(np.arange(size, dtype=np.int64))
        warm_output = cp.empty_like(warm_input)
        kernel((1,), (size,), (warm_input, scalar, constant, warm_output))
        cp.cuda.get_current_stream().synchronize()

        binding = bind_pointer_arguments(
            backend, kernel, (1,), (size,), (None, scalar, constant, None), ((0, 0),)
        )
        self.assertIs(type(binding), _BoundDriverArguments)
        stream = cp.cuda.Stream(non_blocking=True)
        self.assertNotEqual(int(stream.ptr), 0, "expected a nondefault CUDA stream")
        first_table = None
        first_table_owner = None
        first_expected = None
        for iteration, host_input in enumerate((
                np.arange(size, dtype=np.int64) * 3 - 91,
                np.arange(size, dtype=np.int64) * -2 + 307)):
            # Upload, allocate, and submit in order on the same stream.
            with stream:
                device_input = cp.asarray(host_input)
                device_output = cp.empty(size, dtype=cp.int64)
                binding((device_input,), device_output)
            stream.synchronize()
            expected = host_input + int(scalar) + host_constant[0]
            np.testing.assert_array_equal(cp.asnumpy(device_output), expected)

            record = driver.records[-1]
            self.assertEqual(record["words"][0], device_input.data.ptr)
            self.assertEqual(record["words"][1], int(np.asarray(scalar).view(np.uint64)))
            self.assertEqual(record["signed_words"][1], int(scalar))
            self.assertEqual(record["words"][2], constant.data.ptr)
            self.assertEqual(record["words"][3], device_output.data.ptr)
            self.assertEqual(record["stream_ptr"], stream.ptr)
            self.assertEqual(record["geometry"], (1, 1, 1, size, 1, 1))
            table = (record["table_pointer"], record["parameter_addresses"])
            if first_table is None:
                first_table = table
                first_table_owner = binding._local.storage[1][0]
                first_expected = expected.copy()
            else:
                self.assertEqual(table, first_table)
                self.assertFalse(np.array_equal(expected, first_expected))
            words, parameters = binding._local.storage
            self.assertIs(parameters[0], first_table_owner)
            self.assertEqual(parameters[1], record["table_pointer"])
            self.assertEqual((int(words[0]), int(words[3])), (device_input.data.ptr, device_output.data.ptr))
            self.assertEqual(int(words[1]), int(np.asarray(scalar).view(np.uint64)))
            self.assertEqual(int(words[2]), constant.data.ptr)

    def test_32_pointer_table_reuses_binding_for_fresh_inputs(self):
        cp = self.cp
        pointer_count = 32
        size = 64
        offset = np.int64(-13)
        kernel = cp.RawKernel(
            _many_pointer_kernel_source("raw_driver_parameter_many", pointer_count, size),
            "raw_driver_parameter_many",
            enable_cooperative_groups=False,
        )
        backend, driver = _cupy_backend(cp, pointer_count + 2)

        def host_batch(shift):
            return [np.arange(size, dtype=np.int64) * ((index % 5) + 1) + index - shift
                    for index in range(pointer_count)]

        warm_hosts = host_batch(0)
        warm_inputs = [cp.asarray(values) for values in warm_hosts]
        warm_output = cp.empty(size, dtype=cp.int64)
        kernel((1,), (size,), (*warm_inputs, offset, warm_output))
        cp.cuda.get_current_stream().synchronize()

        binding = bind_pointer_arguments(
            backend, kernel, (1,), (size,), (None,) * pointer_count + (offset, None),
            tuple((index, index) for index in range(pointer_count)),
        )
        self.assertIs(type(binding), _BoundDriverArguments)
        self.assertLess((pointer_count + 2) * size * 8, 64 * 1024 * 1024)
        first_table = None
        stream = cp.cuda.Stream(non_blocking=True)
        for generation in range(2):
            hosts = host_batch(7 + generation * 31)
            with stream:
                inputs = [cp.asarray(values) for values in hosts]
                output = cp.empty(size, dtype=cp.int64)
                binding(tuple(inputs), output)
            stream.synchronize()
            expected = np.full(size, int(offset), dtype=np.int64)
            for values in hosts:
                expected += values
            np.testing.assert_array_equal(cp.asnumpy(output), expected)
            record = driver.records[-1]
            for index, device_input in enumerate(inputs):
                self.assertEqual(record["words"][index], device_input.data.ptr)
            self.assertEqual(record["signed_words"][pointer_count], int(offset))
            self.assertEqual(record["words"][pointer_count + 1], output.data.ptr)
            table = (record["table_pointer"], record["parameter_addresses"])
            if first_table is None:
                first_table = table
            else:
                self.assertEqual(table, first_table)
            words = binding._local.storage[0]
            self.assertEqual(tuple(map(int, words[:pointer_count])), tuple(value.data.ptr for value in inputs))
            self.assertEqual(int(words[-1]), output.data.ptr)
            self.assertEqual(int(words[pointer_count]),
                             int(np.asarray(offset).view(np.uint64)))

    def test_warmed_function_uses_thread_current_stream(self):
        cp = self.cp
        size = 256
        scalar = np.int64(-5)
        constant = cp.asarray(np.asarray([29], dtype=np.int64))
        kernel = cp.RawKernel(
            _vector_kernel_source("raw_driver_parameter_thread", size),
            "raw_driver_parameter_thread",
            enable_cooperative_groups=False,
        )
        backend, driver = _cupy_backend(cp, 4)
        warm_input = cp.asarray(np.arange(size, dtype=np.int64))
        warm_output = cp.empty(size, dtype=cp.int64)
        kernel((1,), (size,), (warm_input, scalar, constant, warm_output))
        cp.cuda.get_current_stream().synchronize()
        expected_function_ptr = int(kernel.kernel.ptr)

        binding = bind_pointer_arguments(
            backend, kernel, (1,), (size,), (None, scalar, constant, None), ((0, 0),)
        )
        self.assertIs(type(binding), _BoundDriverArguments)
        host_input = np.arange(size, dtype=np.int64) * 4 - 61
        device_input = cp.asarray(host_input)
        device_output = cp.empty(size, dtype=cp.int64)
        # Make the host-thread upload visible before the worker uses another stream.
        cp.cuda.get_current_stream().synchronize()
        result = {}
        errors = []

        def worker():
            try:
                with cp.cuda.Device(self.device_id):
                    thread_stream = cp.cuda.Stream(non_blocking=True)
                    with thread_stream:
                        binding((device_input,), device_output)
                    thread_stream.synchronize()
                    result["stream_ptr"] = int(thread_stream.ptr)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker, name="raw-driver-new-thread")
        thread.start()
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "CUDA worker thread did not finish")
        if errors:
            raise errors[0]

        expected = host_input + int(scalar) + 29
        np.testing.assert_array_equal(cp.asnumpy(device_output), expected)
        record = driver.records[-1]
        self.assertEqual(record["thread"], "raw-driver-new-thread")
        self.assertEqual(record["function_ptr"], expected_function_ptr)
        self.assertEqual(record["stream_ptr"], result["stream_ptr"])
        self.assertNotEqual(record["stream_ptr"], 0)
        self.assertEqual(record["words"][0], device_input.data.ptr)
        self.assertEqual(record["words"][3], device_output.data.ptr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
