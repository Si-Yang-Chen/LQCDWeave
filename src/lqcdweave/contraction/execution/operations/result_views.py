"""X-local lazy slices of a current, owned C-contiguous batch result."""
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True, slots=True)
class _ResultViewLayout:
    full_shape: tuple
    indexer: tuple
    squeeze: tuple
    shape: tuple
    strides: tuple
    offset: int
    dtype: object


def bind_result_views(full_shape, dtype, results):
    """Prove fixed slice geometry once; retain no numerical owner or pointer."""
    full_shape, dtype = tuple(full_shape), np.dtype(dtype)
    strides = [dtype.itemsize] * len(full_shape)
    for axis in range(len(full_shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * full_shape[axis + 1]
    layouts = []
    for key, indexer, squeeze in results:
        if len(indexer) != len(full_shape):
            raise ValueError('batch result slice rank differs from its owner')
        shape, steps, offset = [], [], 0
        for extent, step, index in zip(full_shape, strides, indexer, strict=True):
            if type(index) is int:
                if not 0 <= index < extent:
                    raise ValueError('batch result lane is outside its owner')
                offset += index * step
            elif isinstance(index, slice) and index.step in (None, 1):
                start = 0 if index.start is None else index.start
                stop = extent if index.stop is None else index.stop
                if not 0 <= start <= stop <= extent:
                    raise ValueError('batch result slice is outside its owner')
                offset += start * step
                shape.append(stop - start)
                steps.append(step)
            else:
                raise ValueError('batch result requires fixed unit-stride slices')
        squeeze = tuple(squeeze)
        if (len(set(squeeze)) != len(squeeze)
                or any(type(axis) is not int or not 0 <= axis < len(shape)
                       or shape[axis] != 1 for axis in squeeze)):
            raise ValueError('batch result squeeze differs from its saved shape')
        if prod(shape) and (offset + sum((n - 1) * step for n, step in zip(shape, steps))
                            + dtype.itemsize > prod(full_shape) * dtype.itemsize):
            raise ValueError('batch result byte span is outside its owner')
        shape = tuple(n for axis, n in enumerate(shape) if axis not in squeeze)
        steps = tuple(step for axis, step in enumerate(steps) if axis not in squeeze)
        layouts.append((key, _ResultViewLayout(full_shape, tuple(indexer), squeeze,
                                              shape, steps, offset, dtype)))
    return tuple(layouts)


class _ResultOwner:
    __slots__ = ('value', '_grouped', 'address')

    def __init__(self, value):
        self.value, self._grouped = value, None
        # This current result owns fixed storage until its last slice is freed.
        # Keep the address with that numerical lifetime, never with a template.
        self.address = value.ctypes.data if isinstance(value, np.ndarray) else value.data.ptr

    def grouped(self, shape):
        if self._grouped is None:
            self._grouped = self.value.reshape(shape)
        return self._grouped


class _DeferredResultView:
    __slots__ = ('owner', 'layout')

    def __init__(self, owner, layout):
        self.owner, self.layout = owner, layout

    @property
    def shape(self):
        return self.layout.shape

    @property
    def strides(self):
        return self.layout.strides

    @property
    def dtype(self):
        return self.layout.dtype

    def pointer_address(self):
        return self.owner.address + self.layout.offset

    def pointer_argument(self):
        return np.uint64(self.pointer_address())

    def array(self):
        value = self.owner.grouped(self.layout.full_shape)[self.layout.indexer]
        return value.squeeze(axis=self.layout.squeeze) if self.layout.squeeze else value


def _array_pointer(value):
    """Read current owner metadata without array views or scalar wrappers."""
    if type(value) is _DeferredResultView:
        return value.pointer_address()
    if isinstance(value, np.ndarray):
        return value.__array_interface__['data'][0]
    return value.data.ptr
