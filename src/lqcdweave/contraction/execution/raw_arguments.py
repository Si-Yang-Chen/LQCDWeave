"""X-local host parameter slots for fixed 64-bit raw kernel arguments."""
from threading import local

import numpy as np


class _BoundPointerArguments:
    _clear_after_launch = True

    def __init__(self, kernel, grid, block, argument_type, template, positions,
                 pointer_getter=None, scalar_positions=()):
        self.kernel, self.grid, self.block = kernel, grid, block
        self.argument_type = argument_type
        self.positions = tuple(positions)
        self.scalar_positions = tuple(scalar_positions)
        self.pointer_getter = pointer_getter
        # Operations also import this ABI service; resolve the optional
        # consumer contract during binding, after module initialization.
        from .operations.result_views import _DeferredResultView, _array_pointer
        self.array_pointers = pointer_getter is _array_pointer
        self.deferred_view_type = _DeferredResultView
        self.output_position = len(template) - 1
        dynamic_positions = (sorted({position for position, _ in self.positions}
                                    | {position for position, _ in self.scalar_positions}
                                    | {self.output_position})
                             if self._clear_after_launch else ())
        ranges = []
        for position in dynamic_positions:
            if ranges and position == ranges[-1][1]:
                ranges[-1][1] = position + 1
            else:
                ranges.append([position, position + 1])
        # Dense pointer runs can be cleared by copying fixed zero bytes. Leave
        # small or fragmented layouts on the original scalar path.
        self.clear_ranges = (tuple((start * 8, stop * 8, bytes((stop - start) * 8))
                                   for start, stop in ranges)
                             if len(dynamic_positions) >= 8
                             and len(ranges) * 2 <= len(dynamic_positions) else None)
        self.fixed_words = np.zeros(len(template), dtype=np.uint64)
        owners = []
        for position, value in enumerate(template):
            if value is None:
                continue
            if type(value) in (np.int64, np.uint64):
                self.fixed_words[position] = value.view(np.uint64)
            else:
                self.fixed_words[position] = value.data.ptr
                owners.append(value)
        # Immutable recipe constants already belong to the loader. Keeping
        # their owners here makes the fixed addresses safe in this binding too.
        self.constant_owners = tuple(owners)
        # Interleaved pointers/constants make many tiny clear operations. One
        # bounded host copy restores exactly the same fixed/zero slot bytes.
        self.clear_template = (self.fixed_words.tobytes()
                               if len(dynamic_positions) >= 8 and self.clear_ranges is None else None)
        self._local = local()

    def __call__(self, inputs, result, scalars=()):
        storage = getattr(self._local, 'storage', None)
        if storage is None:
            words = self.fixed_words.copy()
            parameters = self._parameters(words)
            # A native uint64 view writes the same slots without NumPy scalar
            # conversion on every dynamic address assignment. It owns words;
            # CNumpyArray slices and the driver's address table stay unchanged.
            byte_slots = self._local.byte_slots = memoryview(words).cast('B')
            slots = byte_slots.cast('Q')
            storage = self._local.storage = slots, parameters
        words, parameters = storage
        try:
            if self.pointer_getter is None:
                for position, index in self.positions:
                    words[position] = inputs[index].data.ptr
            elif self.array_pointers:
                # These consumers use the same current owner/address contract
                # as _array_pointer. Fill all slots without a Python getter and
                # deferred-view method call for every operand.
                deferred_type = self.deferred_view_type
                for position, index in self.positions:
                    value = inputs[index]
                    if type(value) is deferred_type:
                        words[position] = value.owner.address + value.layout.offset
                    elif isinstance(value, np.ndarray):
                        words[position] = value.__array_interface__['data'][0]
                    else:
                        words[position] = value.data.ptr
            else:
                for position, index in self.positions:
                    words[position] = self.pointer_getter(inputs[index])
            words[self.output_position] = result.data.ptr
            for position, index in self.scalar_positions:
                words[position] = scalars[index] & 0xffffffffffffffff
            # Both native submissions preserve current-device/current-stream
            # dispatch. Slots carry values, never Python object addresses.
            self._launch(parameters)
        finally:
            if self._clear_after_launch:
                if self.clear_template is not None:
                    self._local.byte_slots[:] = self.clear_template
                elif self.clear_ranges is None:
                    for position, _ in self.positions:
                        words[position] = 0
                    for position, _ in self.scalar_positions:
                        words[position] = 0
                    words[self.output_position] = 0
                else:
                    byte_slots = self._local.byte_slots
                    for start, stop, zeros in self.clear_ranges:
                        byte_slots[start:stop] = zeros


    def _parameters(self, words):
        return tuple(self.argument_type(words[position:position + 1])
                     for position in range(len(words)))

    def _launch(self, parameters):
        self.kernel(self.grid, self.block, parameters)


class _BoundDriverArguments(_BoundPointerArguments):
    """Same pointer/int64 ABI, with a thread-local host parameter table."""
    # Native launchKernel consumes the host argument values before returning.
    # Every dynamic word is overwritten before the next launch; an incomplete
    # fill raises before submission. Integer addresses own no input arrays.
    # Keep clearing only on the original/custom parameter-wrapper fallback.
    _clear_after_launch = False

    def __init__(self, *args, launch, ensure_context, current_stream, **kwargs):
        super().__init__(*args, **kwargs)
        self.launch = launch
        self.ensure_context = ensure_context
        self.current_stream = current_stream
        self.geometry = tuple(max(1, x) for x in (self.grid + (1, 1))[:3]) + tuple(
            max(1, x) for x in (self.block + (1, 1))[:3])

    def _parameters(self, words):
        table = np.arange(len(words), dtype=np.uintp) * words.itemsize + words.ctypes.data
        # The storage also owns words; table entries address those host slots.
        return [table, int(table.ctypes.data), None]

    def _launch(self, parameters):
        function = self.kernel.kernel
        if function is not parameters[2]:
            # A cached Function can first be consumed from a fresh host thread.
            # Runtime API establishes that thread's current CUDA context, as
            # CuPy Function does before its own driver submission.
            self.ensure_context()
            parameters[2] = function
        self.launch(function.ptr, *self.geometry, 0,
                    self.current_stream().ptr, parameters[1], 0)


def bind_pointer_arguments(backend, kernel, grid, block, template, positions, *,
                           pointer_getter=None, scalar_positions=()):
    """Bind the pointer/int64 ABI; unsupported kernels keep normal calls.

    ``scalar_positions`` maps parameter slots to Python signed-int64 values
    in the call's third argument. Slots, including scalar slots, are local to
    the calling thread; no input or result owners enter the binding.
    """
    cuda = getattr(backend, 'cuda', None)
    runtime = getattr(cuda, 'runtime', None)
    argument_type = getattr(getattr(cuda, 'function', None), 'CNumpyArray', None)
    if (getattr(backend, '__name__', '') != 'cupy'
            or getattr(runtime, 'is_hip', True)
            or not callable(argument_type)
            or np.dtype(np.uintp).itemsize != 8
            or type(kernel).__module__ != 'cupy._core.raw'
            or type(kernel).__name__ != 'RawKernel'):
        return None
    launch = getattr(getattr(cuda, 'driver', None), 'launchKernel', None)
    ensure_context = getattr(runtime, 'memGetInfo', None)
    current_stream = getattr(cuda, 'get_current_stream', None)
    if (not getattr(kernel, 'enable_cooperative_groups', True)
            and all(callable(value) for value in (launch, ensure_context, current_stream))):
        return _BoundDriverArguments(kernel, grid, block, argument_type, template, positions,
            pointer_getter, scalar_positions, launch=launch, ensure_context=ensure_context,
            current_stream=current_stream)
    return _BoundPointerArguments(kernel, grid, block, argument_type, template, positions,
                                 pointer_getter, scalar_positions)
