"""CPU address oracles and optional real-CUDA checks for deferred group views."""
import ctypes
import gc
import itertools
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.gemm_group import (
    _bind_group_result_views,
    _bind_layouts,
    bind_dense_gemm_group,
)
from lqcdweave.contraction.execution.operations.dense_batch import (
    _contiguous_concatenation,
    bind_batch_layout,
)
from lqcdweave.contraction.execution.operations.output import bind_output_layout
from lqcdweave.contraction.execution.operations.output_kernel import _raw_copy_kernel
from lqcdweave.contraction.execution.operations.packing_kernel import (
    _raw_kernel as _packing_raw_kernel,
    _packing_values,
    bind_gemm_group_packing,
    bind_packing,
)
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView,
    bind_result_views,
)
from .test_gemm_group_result_views import _multiaxis_group, _stacked_group


_DTYPE = np.dtype(np.complex128)


def _bound_fixture(group, catalog):
    batches, layouts, _, dtype = _bind_layouts(group, catalog, np)
    views = _bind_group_result_views(layouts)
    if views is None:
        raise AssertionError("fixture must have one fixed result layout")
    return batches, layouts, views, dtype


def _new_node_view(groups, arrays=None):
    arrays = {} if arrays is None else arrays
    keys = tuple(result.tensor for group, _catalog in groups
                 for batch in group.batches for result in batch.results)
    node = _NodeView({key: key for key in keys}, arrays)
    return node, arrays


def _publish_group(node, group, catalog, owner):
    batches, layouts, views, _dtype = _bound_fixture(group, catalog)
    node._publish_result_views(owner, views.deferred_layouts)
    return batches, layouts, views


def _eager_fixture_results(group, catalog, owner):
    batches, layouts, views, _dtype = _bound_fixture(group, catalog)
    grouped = owner.reshape((len(batches), *layouts[0].output_shape))
    results = {}
    for lane, (batch, layout) in enumerate(zip(batches, layouts, strict=True)):
        for key, bounds, squeeze in layout.result_slices:
            value = grouped[(lane, *bounds)]
            results[key] = value.squeeze(axis=squeeze) if squeeze else value
    return results


def _independent_fixture_products(group, catalog, host_inputs):
    """Build fixture products in their declared tensor-axis order."""
    batches, layouts, _, _dtype = _bind_layouts(group, catalog, np)
    results = {}
    for batch, layout in zip(batches, layouts, strict=True):
        left = host_inputs[batch.left[0].tensor]
        original_rights = tuple(host_inputs[item.tensor] for item in batch.right)
        if left.ndim == 3:
            product = np.einsum(
                "abc,cde->abde", left, np.concatenate(original_rights, axis=2))
        elif left.ndim == 1:
            product = np.asarray([
                [np.sum(left * right) for right in original_rights]
            ])
        else:
            raise AssertionError(f"unexpected fixture left rank: {left.ndim}")

        for key, bounds, squeeze in layout.result_slices:
            value = product[bounds]
            results[key] = value.squeeze(axis=squeeze) if squeeze else value
    return results


def _independent_batch_results(batch, host_inputs):
    left = host_inputs[batch.left[0].tensor]
    rights = tuple(host_inputs[item.tensor] for item in batch.right)
    if left.ndim == 3:
        product = np.einsum("abc,cde->abde", left, np.concatenate(rights, axis=2))
    elif left.ndim == 1:
        product = np.asarray([[np.sum(left * right) for right in rights]])
    else:
        raise AssertionError(f"unexpected fixture left rank: {left.ndim}")
    results = {}
    for result in batch.results:
        bounds = tuple(slice(start, stop) for start, stop in result.bounds)
        value = product[bounds]
        if result.squeeze_axes:
            value = value.squeeze(axis=result.squeeze_axes)
        results[result.tensor] = np.array(value, copy=True)
    return results


def _fixture_batch_inputs(batch, catalog, generation):
    inputs = {}
    for index, operand in enumerate((*batch.left, *batch.right)):
        spec = catalog[operand.tensor]
        real = np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape)
        real = real + generation * 19 + index + 0.25
        if np.dtype(spec.dtype).kind == "c":
            value = real + 1j * (real[::-1] / 7 + generation + 0.5)
        else:
            value = real
        inputs[operand.tensor] = value.astype(spec.dtype)
    return inputs


def _read_pointer(pointer, shape, strides, dtype):
    """Read an array through a uint64 pointer and an independent strided declaration."""
    shape, strides, dtype = tuple(shape), tuple(strides), np.dtype(dtype)
    result = np.empty(shape, dtype=dtype)
    indices = itertools.product(*(range(extent) for extent in shape)) if shape else ((),)
    for index in indices:
        offset = sum(axis * stride for axis, stride in zip(index, strides, strict=True))
        raw = ctypes.string_at(int(pointer) + offset, dtype.itemsize)
        result[index] = np.frombuffer(raw, dtype=dtype, count=1)[0]
    return result


def _publish_owner_slices(node, owner, keys, *, lane_indices=None, columns=None):
    """Publish fixed owner slices through the production layout binder."""
    keys = tuple(keys)
    lane_indices = tuple(range(len(keys))) if lane_indices is None else tuple(lane_indices)
    if len(lane_indices) != len(keys):
        raise AssertionError("each owner slice needs one lane index")
    results = []
    for lane, (lane_index, key) in enumerate(zip(lane_indices, keys, strict=True)):
        indexer = [slice(None)] * (owner.ndim - 1)
        if columns is not None:
            indexer[-1] = columns[lane]
        results.append((key, (lane_index, *indexer), ()))
    layouts = bind_result_views(owner.shape, owner.dtype, tuple(results))
    node._publish_result_views(owner, layouts)
    return tuple(node._raw_output_value(key) for key in keys)


def _apply_input_layout(value, layout):
    if layout is None:
        return value
    source_axes = tuple(axis for axis in layout if axis is not None)
    if source_axes != tuple(range(value.ndim)):
        value = value.transpose(source_axes)
    for axis, source_axis in enumerate(layout):
        if source_axis is None:
            value = np.expand_dims(value, axis=axis)
    return value


def _declare_pointer(backend, value):
    pointer = int(value.pointer_argument())
    backend.declarations[pointer] = (
        tuple(value.shape), tuple(value.strides), np.dtype(value.dtype))
    return pointer


class _PointerOracleBackend:
    """CPU fake-CuPy surface whose kernels consume only declared raw addresses."""

    __name__ = "cupy"

    def __init__(self, kernel):
        self.kernel = kernel
        self.declarations = {}
        self.launches = []
        self.empty_calls = []

    def empty(self, shape, dtype, order="C"):
        shape, dtype = tuple(shape), np.dtype(dtype)
        self.empty_calls.append((shape, dtype, order))
        return np.empty(shape, dtype=dtype, order=order)

    def RawKernel(self, _source, _name, options=()):
        if tuple(options) != ("--std=c++17",):
            raise AssertionError(f"unexpected RawKernel options: {options}")
        return self.kernel

    def ElementwiseKernel(self, *_args, **_kwargs):
        raise AssertionError("the saved descriptors must select RawKernel packing")


class _ConcatenatePointerOracleKernel:
    def __init__(self, backend, layouts, axis, output_dtype):
        self.backend = backend
        self.layouts = tuple(layouts)
        self.axis = axis
        self.output_dtype = np.dtype(output_dtype)

    def __call__(self, _grid, _block, arguments):
        pointers, destination = arguments[:-1], arguments[-1]
        if len(arguments) == 35:
            # Equal stacks use 32 pointer slots, then span/offset scalars.
            # The unused slots are null; only declared input lanes are read.
            slots, span, offset = arguments[:32], arguments[32], arguments[33]
            if any(int(pointer) != 0 for pointer in slots[len(self.layouts):]):
                raise AssertionError('non-null stack padding')
            if int(span) != destination.shape[self.axis] or int(offset) != 0:
                raise AssertionError('unexpected stack destination geometry')
            pointers = slots[:len(self.layouts)]
        self.backend.launches.append({
            "argument_types": tuple(type(pointer) for pointer in pointers),
            "addresses": tuple(int(pointer) for pointer in pointers),
            "destination_shape": tuple(destination.shape),
        })
        values = []
        for pointer, layout in zip(pointers, self.layouts, strict=True):
            shape, strides, dtype = self.backend.declarations[int(pointer)]
            value = _read_pointer(pointer, shape, strides, dtype)
            values.append(_apply_input_layout(value, layout))
        expected = np.concatenate(values, axis=self.axis).astype(
            self.output_dtype, copy=False)
        if tuple(expected.shape) != tuple(destination.shape):
            raise AssertionError(
                f"oracle shape {expected.shape} differs from destination {destination.shape}")
        destination[...] = expected


class _GroupPointerOracleKernel:
    def __init__(self, backend, sides, output_dtype):
        self.backend = backend
        self.sides = tuple(sides)
        self.output_dtype = np.dtype(output_dtype)

    def __call__(self, _grid, _block, arguments):
        pointers, destination = arguments[:-1], arguments[-1]
        self.backend.launches.append({
            "argument_types": tuple(type(pointer) for pointer in pointers),
            "addresses": tuple(int(pointer) for pointer in pointers),
            "destination_shape": tuple(destination.shape),
        })
        cursor = 0
        for lane, side in enumerate(self.sides):
            values = []
            for layout in side.input_layouts:
                pointer = pointers[cursor]
                cursor += 1
                shape, strides, dtype = self.backend.declarations[int(pointer)]
                value = _read_pointer(pointer, shape, strides, dtype)
                values.append(_apply_input_layout(value, layout))
            packed = np.concatenate(values, axis=side.axis)
            destination[lane] = packed.reshape(destination[lane].shape).astype(
                self.output_dtype, copy=False)
        if cursor != len(pointers):
            raise AssertionError("oracle did not consume every declared input pointer")


class _AddressOracleKernel:
    def __init__(self, backend, source):
        self.backend, self.source = backend, source

    def __call__(self, _grid, _block, arguments):
        pointers, lane_offset, destination = arguments[:-2], int(arguments[-2]), arguments[-1]
        self.backend.launches.append(tuple(type(pointer) for pointer in pointers))
        for lane, pointer in enumerate(pointers):
            declaration = self.backend.declarations.get(int(pointer))
            if declaration is None:
                raise AssertionError(f"copy kernel received an undeclared address: {int(pointer)}")
            shape, strides, dtype = declaration
            value = _read_pointer(pointer, shape, strides, dtype)
            for index in (itertools.product(*(range(extent) for extent in shape))
                          if shape else ((),)):
                destination[(lane_offset + lane, *index)] = value[index]


class _AddressOracleBackend:
    __name__ = "cupy"

    def __init__(self):
        self.declarations = {}
        self.launches = []

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def RawKernel(self, source, _name, options=()):
        if tuple(options) != ("--std=c++17",):
            raise AssertionError(f"unexpected RawKernel options: {options}")
        return _AddressOracleKernel(self, source)


class DeferredResultViewCpuTests(unittest.TestCase):
    def setUp(self):
        _raw_copy_kernel.cache_clear()

    def tearDown(self):
        _raw_copy_kernel.cache_clear()

    def test_current_owner_address_is_read_once_and_fresh_outputs_remain_independent(self):
        class DeviceResult:
            def __init__(self, value):
                self.value = value
                self.address_reads = 0

            @property
            def data(self):
                self.address_reads += 1
                return SimpleNamespace(ptr=self.value.ctypes.data)

            def reshape(self, shape):
                return self.value.reshape(shape)

        layouts = bind_result_views((2, 3), np.float64,
            ((str(lane), (lane, slice(None)), ()) for lane in range(2)))
        arrays = {}
        node = _NodeView({'0': 20, '1': 21}, arrays)
        first = DeviceResult(np.arange(6, dtype=np.float64).reshape(2, 3))
        node._publish_result_views(first, layouts)
        previous = tuple(node._raw_output_value(key) for key, _ in layouts)
        first.value += 10
        for _ in range(3):
            for lane, reference in enumerate(previous):
                self.assertEqual(reference.pointer_address(), first.value[lane].ctypes.data)
                np.testing.assert_array_equal(
                    _read_pointer(reference.pointer_argument(), reference.shape,
                                  reference.strides, reference.dtype), first.value[lane])
        self.assertEqual(first.address_reads, 1)

        second = DeviceResult((np.arange(6, dtype=np.float64) + 50).reshape(2, 3))
        node._publish_result_views(second, layouts)
        for lane, (key, _) in enumerate(layouts):
            current = node._raw_output_value(key)
            self.assertIsNot(current.owner, previous[lane].owner)
            self.assertNotEqual(current.pointer_address(), previous[lane].pointer_address())
            np.testing.assert_array_equal(current.array(), second.value[lane])
            np.testing.assert_array_equal(previous[lane].array(), first.value[lane])
        self.assertEqual(first.address_reads, 1)
        self.assertEqual(second.address_reads, 1)

    def test_multiaxis_and_scalar_geometry_and_uint64_address_oracle(self):
        multi_group, multi_catalog = _multiaxis_group()
        scalar_group, scalar_catalog = _stacked_group()
        multi_owner = (np.arange(3 * 6 * 10, dtype=np.float64) + 1j).astype(_DTYPE).reshape(3, 6, 10)
        scalar_owner = (np.arange(4, dtype=np.float64) + 11j).astype(_DTYPE).reshape(2, 1, 2)
        node, _arrays = _new_node_view(((multi_group, multi_catalog), (scalar_group, scalar_catalog)))
        multi_batches, multi_layouts, multi_views = _publish_group(
            node, multi_group, multi_catalog, multi_owner)
        scalar_batches, scalar_layouts, _scalar_views = _publish_group(
            node, scalar_group, scalar_catalog, scalar_owner)

        cases = (
            (multi_group, multi_batches, multi_layouts, multi_owner,
             (len(multi_batches), *multi_layouts[0].output_shape)),
            (scalar_group, scalar_batches, scalar_layouts, scalar_owner,
             (len(scalar_batches), *scalar_layouts[0].output_shape)),
        )
        for group, batches, layouts, owner, full_shape in cases:
            grouped = owner.reshape(full_shape)
            self.assertEqual(tuple(layouts[0].output_shape), full_shape[1:])
            for lane, (batch, layout) in enumerate(zip(batches, layouts, strict=True)):
                for result in batch.results:
                    key, bounds, squeeze = next(
                        item for item in layout.result_slices if item[0] == result.tensor)
                    expected = grouped[(lane, *bounds)]
                    if squeeze:
                        expected = expected.squeeze(axis=squeeze)
                    reference = node._raw_output_value(key)
                    self.assertIs(type(reference), _DeferredResultView)
                    self.assertEqual(reference.shape, expected.shape)
                    self.assertEqual(reference.strides, expected.strides)
                    self.assertEqual(reference.layout.full_shape, full_shape)
                    self.assertEqual(reference.layout.offset,
                                     expected.ctypes.data - owner.ctypes.data)
                    pointer = reference.pointer_argument()
                    self.assertIs(type(pointer), np.uint64)
                    self.assertEqual(int(pointer), expected.ctypes.data)
                    actual = _read_pointer(pointer, reference.shape, reference.strides,
                                           reference.dtype)
                    np.testing.assert_array_equal(actual, expected)

        key = multi_group.batches[1].results[0].tensor
        first = node[key]
        self.assertIs(type(first), np.ndarray)
        self.assertIs(node[key], first)
        self.assertIs(type(node._raw_output_value(key)), np.ndarray)
        self.assertTrue(np.shares_memory(first, multi_owner))

    def test_output_copy_uses_raw_addresses_and_creates_independent_owners(self):
        multi_group, multi_catalog = _multiaxis_group()
        scalar_group, scalar_catalog = _stacked_group()
        multi_owner = (np.arange(3 * 6 * 10).reshape(3, 6, 10) + (2 + 3j)).astype(_DTYPE)
        scalar_owner = (np.arange(4).reshape(2, 1, 2) + (7 - 2j)).astype(_DTYPE)
        groups = ((multi_group, multi_catalog), (scalar_group, scalar_catalog))
        node, _arrays = _new_node_view(groups)
        _publish_group(node, multi_group, multi_catalog, multi_owner)
        _publish_group(node, scalar_group, scalar_catalog, scalar_owner)

        expected = {}
        source_owners = (multi_owner, scalar_owner)
        for (group, catalog), owner in zip(groups, source_owners, strict=True):
            eager = _eager_fixture_results(group, catalog, owner)
            expected.update({key: value.copy() for key, value in eager.items()})

        backend = _AddressOracleBackend()
        for (group, catalog), owner in zip(groups, source_owners, strict=True):
            for key, value in _eager_fixture_results(group, catalog, owner).items():
                backend.declarations[value.ctypes.data] = (
                    tuple(value.shape), tuple(value.strides), np.dtype(value.dtype))

        catalog = dict(multi_catalog)
        catalog.update(scalar_catalog)
        keys = tuple(expected)
        output_layout = bind_output_layout(keys, catalog, backend=backend, validate=False)
        self.assertEqual(len(output_layout.groups), 2)
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("output copy must consume the raw pointer")):
            packed = output_layout.copy(node, backend)

        self.assertEqual(len(packed.owners), 2)
        self.assertEqual(len(backend.launches), 2)
        self.assertTrue(all(all(argument_type is np.uint64 for argument_type in launch)
                            for launch in backend.launches))
        for actual, key in zip(packed, keys, strict=True):
            np.testing.assert_array_equal(actual, expected[key])

        for copied_owner in packed.owners:
            self.assertFalse(np.shares_memory(copied_owner, multi_owner))
            self.assertFalse(np.shares_memory(copied_owner, scalar_owner))

        multi_owner.fill(-99)
        scalar_owner.fill(123)
        for actual, key in zip(packed, keys, strict=True):
            np.testing.assert_array_equal(actual, expected[key])

    def test_numpy_group_results_match_independent_fixture_products(self):
        rng = np.random.default_rng(30491)
        for fixture_name, fixture in (("multiaxis", _multiaxis_group),
                                      ("scalar", _stacked_group)):
            group, catalog = fixture()
            batches, _layouts, _views, _dtype = _bound_fixture(group, catalog)
            host_inputs = {}
            for batch in batches:
                for operand in (*batch.left, *batch.right):
                    spec = catalog[operand.tensor]
                    real = rng.normal(size=spec.shape)
                    imag = rng.normal(size=spec.shape)
                    host_inputs[operand.tensor] = (real + 1j * imag).astype(spec.dtype)

            expected = _independent_fixture_products(group, catalog, host_inputs)
            arrays = dict(host_inputs)
            cube = bind_dense_gemm_group(group, catalog, np)(arrays)
            self.assertEqual(tuple(cube.shape),
                             (len(batches), *group.matrix_shape[::2]), fixture_name)
            for batch in batches:
                for result in batch.results:
                    actual = arrays[result.tensor]
                    self.assertEqual(tuple(actual.shape), tuple(expected[result.tensor].shape),
                                     f"{fixture_name} result {result.tensor}")
                    np.testing.assert_allclose(
                        actual, expected[result.tensor], rtol=2e-12, atol=2e-12,
                        err_msg=f"{fixture_name} result {result.tensor}")

    def test_single_batch_lazy_owner_is_fresh_and_raw_consumers_need_no_array(self):
        _packing_raw_kernel.cache_clear()
        self.addCleanup(_packing_raw_kernel.cache_clear)
        cupy_named_numpy = SimpleNamespace(
            __name__="cupy", empty=np.empty, array=np.array, concatenate=np.concatenate,
            dot=np.dot, matmul=np.matmul,
        )
        for fixture_name, fixture in (("multiaxis", _multiaxis_group),
                                      ("scalar", _stacked_group)):
            with self.subTest(fixture=fixture_name):
                group, catalog = fixture()
                batch = group.batches[0]
                layout = bind_batch_layout(batch, catalog, cupy_named_numpy)
                self.assertIsNotNone(layout.deferred_layouts)
                keys = tuple(result.tensor for result in batch.results)
                binding = {key: key for key in catalog}
                runs = []
                for generation in (1, 2):
                    inputs = _fixture_batch_inputs(batch, catalog, generation)
                    node = _NodeView(binding, dict(inputs))
                    output = layout.run(node, backend=cupy_named_numpy)
                    expected = _independent_batch_results(batch, inputs)
                    references = tuple(node._raw_output_value(key) for key in keys)
                    self.assertTrue(all(type(value) is _DeferredResultView
                                        for value in references))
                    self.assertTrue(all(value.owner.value is output for value in references))
                    for key, value in zip(keys, references, strict=True):
                        self.assertEqual(tuple(value.shape), tuple(expected[key].shape))
                        actual = _read_pointer(value.pointer_argument(), value.shape,
                                               value.strides, value.dtype)
                        np.testing.assert_allclose(actual, expected[key], rtol=2e-12, atol=2e-12)
                    runs.append((node, output, expected, references))

                first_node, first_output, first_expected, first_refs = runs[0]
                second_node, second_output, second_expected, second_refs = runs[1]
                first_owner, second_owner = weakref.ref(first_output), weakref.ref(second_output)
                self.assertIsNot(first_output, second_output)
                self.assertFalse(np.shares_memory(first_output, second_output))

                copy_backend = _AddressOracleBackend()
                for reference in first_refs:
                    _declare_pointer(copy_backend, reference)
                snapshot_layout = bind_output_layout(
                    keys, catalog, backend=copy_backend, validate=False)
                with execution_debug(False), patch.object(
                        _DeferredResultView, "array",
                        side_effect=AssertionError("raw snapshot must consume deferred pointers")):
                    snapshot = snapshot_layout.copy(first_node, copy_backend)
                for actual, key, reference in zip(snapshot, keys, first_refs, strict=True):
                    source = _read_pointer(reference.pointer_argument(), reference.shape,
                                           reference.strides, reference.dtype)
                    np.testing.assert_array_equal(actual, source)
                    np.testing.assert_allclose(actual, first_expected[key], rtol=2e-12, atol=2e-12)

                if fixture_name == "multiaxis":
                    pointer_backend = _PointerOracleBackend(None)
                    pointer_backend.kernel = _ConcatenatePointerOracleKernel(
                        pointer_backend, ((0, 1, 2, 3),) * 2, 3, np.complex128)
                    operands = tuple((key, None) for key in keys)
                    with execution_debug(False), patch.object(
                            _DeferredResultView, "array",
                            side_effect=AssertionError("downstream packing must use raw refs")):
                        current = _packing_values(first_node, operands)
                        for reference in current:
                            _declare_pointer(pointer_backend, reference)
                        downstream_packer = bind_packing(
                            pointer_backend, tuple(value.shape for value in current),
                            tuple(value.dtype for value in current), 3, np.complex128,
                            input_layouts=((0, 1, 2, 3),) * 2)
                        downstream_shape = np.concatenate(
                            tuple(first_expected[key] for key in keys), axis=3).shape
                        downstream = np.empty(downstream_shape, dtype=np.complex128)
                        downstream_packer(current, downstream)
                    np.testing.assert_allclose(
                        downstream, np.concatenate(
                            tuple(first_expected[key] for key in keys), axis=3),
                        rtol=2e-12, atol=2e-12)

                first_key = keys[0]
                materialized = first_node[first_key]
                self.assertIs(type(materialized), np.ndarray)
                self.assertIs(first_node._raw_output_value(first_key), materialized)
                np.testing.assert_allclose(materialized, first_expected[first_key],
                                           rtol=2e-12, atol=2e-12)
                for key in keys:
                    del first_node[key]
                    del second_node[key]
                del materialized, first_refs, second_refs, first_output, second_output
                del reference, value, source
                del first_node, second_node, node, output, references, runs
                if fixture_name == "multiaxis":
                    del current, downstream_packer, downstream, pointer_backend
                del snapshot, snapshot_layout, copy_backend
                gc.collect()
                self.assertIsNone(first_owner())
                self.assertIsNone(second_owner())

    def test_single_batch_numpy_dict_debug_and_timed_paths_stay_eager(self):
        cupy_named_numpy = SimpleNamespace(
            __name__="cupy", empty=np.empty, array=np.array, concatenate=np.concatenate,
            dot=np.dot, matmul=np.matmul,
        )

        def assert_results(arrays, expected):
            for key, value in expected.items():
                actual = arrays[key]
                self.assertFalse(type(actual) is _DeferredResultView)
                np.testing.assert_allclose(actual, value, rtol=2e-12, atol=2e-12)

        for fixture_name, fixture in (("multiaxis", _multiaxis_group),
                                      ("scalar", _stacked_group)):
            with self.subTest(fixture=fixture_name):
                group, catalog = fixture()
                batch = group.batches[0]
                inputs = _fixture_batch_inputs(batch, catalog, 7)
                layout = bind_batch_layout(batch, catalog, cupy_named_numpy)
                expected = _independent_batch_results(batch, inputs)

                numpy_layout = bind_batch_layout(batch, catalog, np)
                self.assertIsNone(numpy_layout.deferred_layouts)
                numpy_arrays = dict(inputs)
                numpy_layout.run(numpy_arrays, backend=np)
                assert_results(numpy_arrays, expected)

                dict_arrays = dict(inputs)
                layout.run(dict_arrays, backend=cupy_named_numpy)
                assert_results(dict_arrays, expected)

                binding = {key: key for key in catalog}
                debug_node = _NodeView(binding, dict(inputs))
                with execution_debug(True):
                    layout.run(debug_node, backend=cupy_named_numpy)
                assert_results(debug_node.arrays, expected)

                timed_node = _NodeView(binding, dict(inputs))
                labels = []

                def timed(label, operation, *, device):
                    labels.append((label, device))
                    return operation()

                layout.run(timed_node, backend=cupy_named_numpy, timed=timed, device=True)
                self.assertEqual(labels, [
                    ("dense_batch_pack", True), ("dense_batch_pack", True),
                    ("contract_kernel", True),
                ])
                assert_results(timed_node.arrays, expected)

    def test_invalid_slice_squeeze_and_bounds_are_rejected_at_bind(self):
        invalid = (
            ((slice(None),), ()),                         # wrong rank
            ((-1, slice(None)), ()),                     # negative integer index
            ((2, slice(None)), ()),                      # integer index at the bound
            ((slice(-1, 2), slice(None)), ()),           # negative start
            ((slice(None), slice(0, 4)), ()),             # stop beyond the bound
            ((slice(None), slice(0, 2, 2)), ()),          # non-unit stride
            ((slice(None, None, -1), slice(None)), ()),   # reverse slice
            ((slice(None), slice(None)), (0, 0)),         # duplicate squeeze
            ((slice(None), slice(None)), (2,)),           # squeeze axis outside rank
            ((slice(None), slice(None)), (1,)),           # non-singleton squeeze
            ((slice(None), slice(None)), (True,)),        # bool is not an integer axis
        )
        for indexer, squeeze in invalid:
            with self.subTest(indexer=indexer, squeeze=squeeze):
                with self.assertRaises(ValueError):
                    bind_result_views((2, 3), _DTYPE, (("bad", indexer, squeeze),))

    def test_debug_resolution_eager_dict_and_fresh_publications(self):
        group, catalog = _multiaxis_group()
        batches, layouts, views, _dtype = _bound_fixture(group, catalog)
        first_owner = np.arange(3 * 6 * 10, dtype=np.float64).astype(_DTYPE).reshape(3, 6, 10)
        second_owner = first_owner + 5000
        node, arrays = _new_node_view(((group, catalog),))
        _publish_group(node, group, catalog, first_owner)
        keys = tuple(result.tensor for batch in group.batches for result in batch.results)
        with execution_debug(True):
            first_values = {key: node[key] for key in keys}
        self.assertTrue(all(type(value) is np.ndarray for value in first_values.values()))
        self.assertTrue(all(type(arrays[key]) is np.ndarray for key in keys))

        eager = views(second_owner)
        self.assertIs(type(eager), dict)
        self.assertEqual(set(eager), set(keys))
        arrays.update(eager)
        for key in keys:
            np.testing.assert_array_equal(arrays[key], _eager_fixture_results(
                group, catalog, second_owner)[key])
            self.assertFalse(np.shares_memory(first_values[key], arrays[key]))

        _publish_group(node, group, catalog, first_owner)
        third = {key: node[key] for key in keys}
        _publish_group(node, group, catalog, second_owner)
        fourth = {key: node[key] for key in keys}
        for key in keys:
            np.testing.assert_array_equal(third[key], _eager_fixture_results(
                group, catalog, first_owner)[key])
            np.testing.assert_array_equal(fourth[key], _eager_fixture_results(
                group, catalog, second_owner)[key])
            self.assertFalse(np.shares_memory(third[key], fourth[key]))

    def test_multiple_owners_follow_partial_and_complete_logical_release(self):
        multi_group, multi_catalog = _multiaxis_group()
        scalar_group, scalar_catalog = _stacked_group()
        groups = ((multi_group, multi_catalog), (scalar_group, scalar_catalog))
        node, arrays = _new_node_view(groups)
        multi_owner = np.empty((3, 6, 10), dtype=_DTYPE)
        scalar_owner = np.empty((2, 1, 2), dtype=_DTYPE)
        _publish_group(node, multi_group, multi_catalog, multi_owner)
        _publish_group(node, scalar_group, scalar_catalog, scalar_owner)
        multi_ref, scalar_ref = weakref.ref(multi_owner), weakref.ref(scalar_owner)
        multi_keys = tuple(result.tensor for batch in multi_group.batches for result in batch.results)
        scalar_keys = tuple(result.tensor for batch in scalar_group.batches for result in batch.results)
        del multi_owner, scalar_owner
        gc.collect()
        self.assertIsNotNone(multi_ref())
        self.assertIsNotNone(scalar_ref())

        del arrays[multi_keys[0]]
        gc.collect()
        self.assertIsNotNone(multi_ref())
        self.assertIsNotNone(scalar_ref())
        for key in multi_keys[1:]:
            del arrays[key]
        gc.collect()
        self.assertIsNone(multi_ref())
        self.assertIsNotNone(scalar_ref())
        for key in scalar_keys:
            del arrays[key]
        gc.collect()
        self.assertIsNone(scalar_ref())


class DeferredPackingCpuTests(unittest.TestCase):
    def setUp(self):
        _packing_raw_kernel.cache_clear()

    def tearDown(self):
        _packing_raw_kernel.cache_clear()

    def test_raw_original_pack_reads_strided_deferred_lane_slices_by_uint64(self):
        owner = (np.arange(24).reshape(2, 3, 4) + (2.5 + 1.25j)).astype(_DTYPE)
        keys = ("left-slab", "right-slab")
        node = _NodeView({key: key for key in keys}, {})
        values = _publish_owner_slices(
            node, owner, keys, columns=(slice(0, 2), slice(2, 4)))
        self.assertEqual(tuple(value.shape for value in values), ((3, 2), (3, 2)))
        self.assertEqual(tuple(value.strides for value in values),
                         ((4 * _DTYPE.itemsize, _DTYPE.itemsize),) * 2)
        self.assertTrue(all(type(value) is _DeferredResultView for value in values))

        layouts = ((0, 1), (0, 1))
        backend = _PointerOracleBackend(None)
        backend.kernel = _ConcatenatePointerOracleKernel(backend, layouts, 1, _DTYPE)
        input_shape = (3, 2)
        packer = bind_packing(
            backend, (input_shape, input_shape), (_DTYPE, _DTYPE), 1, _DTYPE,
            input_layouts=layouts)
        self.assertIsNotNone(packer)
        self.assertTrue(callable(packer.pack_original))
        pointers = tuple(_declare_pointer(backend, value) for value in values)
        destination = np.empty((3, 4), dtype=_DTYPE)
        expected = np.concatenate((owner[0, :, :2], owner[1, :, 2:]), axis=1)

        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("raw original pack must use pointer metadata")):
            packed = packer.pack_original(values, destination)

        self.assertIs(packed, destination)
        np.testing.assert_array_equal(packed, expected)
        self.assertEqual(len(backend.launches), 1)
        launch = backend.launches[0]
        self.assertEqual(launch["argument_types"], (np.uint64, np.uint64))
        self.assertEqual(launch["addresses"], pointers)

    def test_group_pack_uses_six_fresh_pointers_and_only_the_exact_workspace(self):
        group, catalog = _multiaxis_group()
        backend = _PointerOracleBackend(None)
        batches, layouts, _matrix_shape, dtype = _bind_layouts(group, catalog, backend)
        sides = tuple(layout.sides[1] for layout in layouts)
        backend.kernel = _GroupPointerOracleKernel(backend, sides, dtype)
        packer = bind_gemm_group_packing(
            backend, sides, catalog, layouts[0].matrix_shapes[1], dtype)
        self.assertIsNotNone(packer)

        node = _NodeView({key: key for key in catalog}, {})
        arrays = node.arrays
        right_a_keys = tuple(batch.right[0].tensor for batch in batches)
        right_b_keys = tuple(batch.right[1].tensor for batch in batches)
        first_a = (np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5, 1)
                   + (1.0 + 2.0j)).astype(dtype)
        first_b = (np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5, 1)
                   + (101.0 - 3.0j)).astype(dtype)
        first_refs_a = _publish_owner_slices(node, first_a, right_a_keys)
        first_refs_b = _publish_owner_slices(node, first_b, right_b_keys)
        first_refs = tuple(
            value for pair in zip(first_refs_a, first_refs_b, strict=True) for value in pair)
        self.assertEqual(len(first_refs), 6)
        self.assertEqual(len({id(value.owner.value) for value in first_refs}), 2)
        first_pointers = tuple(_declare_pointer(backend, value) for value in first_refs)
        first_owner_refs = (weakref.ref(first_a), weakref.ref(first_b))
        expected_first = tuple(
            np.concatenate((first_a[lane], first_b[lane]), axis=2).reshape(4, 10).copy()
            for lane in range(3))

        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("group pack must pass deferred pointers")):
            first_packed = packer.pack_buffer(node)

        self.assertEqual(tuple(first_packed.shape), (3, 4, 10))
        self.assertEqual(first_packed.nbytes, 3 * 4 * 10 * np.dtype(dtype).itemsize)
        self.assertEqual(backend.empty_calls, [((3, 4, 10), np.dtype(dtype), "C")])
        np.testing.assert_array_equal(first_packed, np.stack(expected_first))
        self.assertEqual(backend.launches[-1]["argument_types"], (np.uint64,) * 6)
        self.assertEqual(backend.launches[-1]["addresses"], first_pointers)

        second_a = (np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5, 1)
                    + (701.0 + 5.0j)).astype(dtype)
        second_b = (np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5, 1)
                    + (901.0 - 7.0j)).astype(dtype)
        second_refs_a = _publish_owner_slices(node, second_a, right_a_keys)
        second_refs_b = _publish_owner_slices(node, second_b, right_b_keys)
        second_refs = tuple(
            value for pair in zip(second_refs_a, second_refs_b, strict=True) for value in pair)
        second_pointers = tuple(_declare_pointer(backend, value) for value in second_refs)
        second_owner_refs = (weakref.ref(second_a), weakref.ref(second_b))
        expected_second = tuple(
            np.concatenate((second_a[lane], second_b[lane]), axis=2).reshape(4, 10).copy()
            for lane in range(3))
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("reused group binding must read current refs")):
            second_packed = packer.pack_buffer(node)

        self.assertEqual(len(backend.launches), 2)
        self.assertEqual(backend.empty_calls,
                         [((3, 4, 10), np.dtype(dtype), "C")] * 2)
        self.assertEqual(backend.launches[-1]["addresses"], second_pointers)
        self.assertNotEqual(first_pointers, second_pointers)
        np.testing.assert_array_equal(second_packed, np.stack(expected_second))
        self.assertFalse(np.shares_memory(first_packed, second_packed))

        del first_refs_a, first_refs_b, first_refs, first_a, first_b
        gc.collect()
        self.assertIsNone(first_owner_refs[0]())
        self.assertIsNone(first_owner_refs[1]())
        del second_refs_a, second_refs_b, second_refs, second_a, second_b
        del node, arrays
        gc.collect()
        self.assertIsNone(second_owner_refs[0]())
        self.assertIsNone(second_owner_refs[1]())

    def test_contiguous_borrow_and_fallback_boundaries_for_deferred_values(self):
        owner = (np.arange(24).reshape(2, 3, 4) + (0.75 - 1.5j)).astype(_DTYPE)
        keys = ("lane-0", "lane-1")
        node = _NodeView({key: key for key in keys}, {})
        refs = _publish_owner_slices(node, owner, keys)
        ordinary_first = owner[0]
        expected = owner.reshape(6, 4)

        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("borrow proof must not materialize deferred views")):
            deferred_borrow = _contiguous_concatenation(
                refs, 0, (6, 4), _DTYPE)
            mixed_borrow = _contiguous_concatenation(
                (ordinary_first, refs[1]), 0, (6, 4), _DTYPE)

        self.assertIsNotNone(deferred_borrow)
        self.assertIsNotNone(mixed_borrow)
        np.testing.assert_array_equal(deferred_borrow, expected)
        np.testing.assert_array_equal(mixed_borrow, expected)
        self.assertTrue(np.shares_memory(deferred_borrow, owner))
        self.assertTrue(np.shares_memory(mixed_borrow, owner))

        gap_owner = np.arange(36, dtype=np.float64).astype(_DTYPE).reshape(3, 3, 4)
        gap_node = _NodeView({key: key for key in keys}, {})
        gap_refs = _publish_owner_slices(gap_node, gap_owner, keys, lane_indices=(0, 2))
        reverse_node = _NodeView({key: key for key in keys}, {})
        reverse_refs = _publish_owner_slices(
            reverse_node, owner, keys, lane_indices=(1, 0))
        transpose_layouts = ((1, 0), (1, 0))

        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("fallback classification must stay pointer-only")):
            self.assertIsNone(_contiguous_concatenation(
                gap_refs, 0, (6, 4), _DTYPE))
            self.assertIsNone(_contiguous_concatenation(
                reverse_refs, 0, (6, 4), _DTYPE))
            self.assertIsNone(_contiguous_concatenation(
                refs, 0, (8, 3), _DTYPE, input_layouts=transpose_layouts))

        owner_a = owner[:1].copy()
        owner_b = (owner[1:2] + 1000).copy()
        separate_node = _NodeView({key: key for key in keys}, {})
        separate_a = _publish_owner_slices(separate_node, owner_a, (keys[0],))
        separate_b = _publish_owner_slices(separate_node, owner_b, (keys[1],))
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("different-owner check must not materialize")):
            self.assertIsNone(_contiguous_concatenation(
                (separate_a[0], separate_b[0]), 0, (6, 4), _DTYPE))

        promoted_owner = owner.astype(np.complex64)
        promoted_node = _NodeView({key: key for key in keys}, {})
        promoted = _publish_owner_slices(promoted_node, promoted_owner, keys)
        self.assertIsNone(_contiguous_concatenation(
            promoted, 0, (6, 4), _DTYPE))

        ordinary = owner[0]
        identity = _contiguous_concatenation((ordinary,), 0, ordinary.shape, _DTYPE)
        self.assertIs(identity, ordinary)

    def test_debug_pack_resolves_with_getitem_but_fast_path_reads_raw_values(self):
        owner = (np.arange(24).reshape(2, 3, 4) + (3.0 + 4.0j)).astype(_DTYPE)
        keys = ("debug-left", "debug-right")
        node = _NodeView({key: key for key in keys}, {})
        _publish_owner_slices(node, owner, keys)
        operands = tuple((key, None) for key in keys)

        with execution_debug(False):
            with patch.object(node, "__getitem__", wraps=node.__getitem__) as getitem:
                with patch.object(node, "_raw_output_value",
                                  wraps=node._raw_output_value) as raw_value:
                    with patch.object(
                            _DeferredResultView, "array",
                            side_effect=AssertionError("fast path resolved a deferred view")):
                        raw_values = _packing_values(node, operands)
        self.assertEqual(getitem.call_count, 0)
        self.assertEqual(raw_value.call_count, 2)
        self.assertEqual(tuple(type(value) for value in raw_values),
                         (_DeferredResultView, _DeferredResultView))

        array_calls = []
        original_array = _DeferredResultView.array

        def record_array(value):
            array_calls.append(value)
            return original_array(value)

        with execution_debug(True):
            with patch.object(node, "__getitem__", wraps=node.__getitem__) as getitem:
                with patch.object(node, "_raw_output_value",
                                  wraps=node._raw_output_value) as raw_value:
                    with patch.object(_DeferredResultView, "array", new=record_array):
                        debug_values = _packing_values(node, operands)
        self.assertEqual(getitem.call_count, 2)
        self.assertEqual(raw_value.call_count, 0)
        self.assertEqual(len(array_calls), 2)
        self.assertTrue(all(type(value) is np.ndarray for value in debug_values))
        for key, value, lane in zip(keys, debug_values, owner, strict=True):
            self.assertIs(node[key], value)
            np.testing.assert_array_equal(value, lane)


class RealCuPyDeferredResultViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            cupy.cuda.runtime.getDeviceProperties(0)
        except Exception as error:
            raise unittest.SkipTest(f"real CUDA deferred-view checks require CuPy and a device: {error}")
        cls.cp = cupy

    def test_cupy_raw_pointer_copy_matches_independent_numpy_and_owns_snapshots(self):
        cp = self.cp
        multi_group, multi_catalog = _multiaxis_group()
        scalar_group, scalar_catalog = _stacked_group()
        multi_host = (np.arange(3 * 6 * 10).reshape(3, 6, 10) + (0.5 + 1.25j)).astype(_DTYPE)
        scalar_host = (np.arange(4).reshape(2, 1, 2) + (8 - 0.75j)).astype(_DTYPE)
        multi_device, scalar_device = cp.asarray(multi_host), cp.asarray(scalar_host)
        groups = ((multi_group, multi_catalog), (scalar_group, scalar_catalog))
        node, _arrays = _new_node_view(groups)
        _publish_group(node, multi_group, multi_catalog, multi_device)
        _publish_group(node, scalar_group, scalar_catalog, scalar_device)

        expected = {}
        for (group, catalog), host_owner in zip(
                groups, (multi_host, scalar_host), strict=True):
            expected.update({key: np.array(value, copy=True) for key, value in
                             _eager_fixture_results(group, catalog, host_owner).items()})
        catalog = dict(multi_catalog)
        catalog.update(scalar_catalog)
        layout = bind_output_layout(tuple(expected), catalog, backend=cp, validate=False)
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("GPU snapshot must use deferred pointers")):
            packed = layout.copy(node, cp)
        cp.cuda.get_current_stream().synchronize()

        self.assertEqual(len(packed.owners), 2)
        for actual, key in zip(packed, expected, strict=True):
            np.testing.assert_array_equal(cp.asnumpy(actual), expected[key])
        for copied_owner in packed.owners:
            for source_owner in (multi_device, scalar_device):
                self.assertFalse(
                    int(copied_owner.data.ptr) < int(source_owner.data.ptr) + source_owner.nbytes
                    and int(source_owner.data.ptr) < int(copied_owner.data.ptr) + copied_owner.nbytes)

        multi_device.fill(-50)
        scalar_device.fill(90)
        cp.cuda.get_current_stream().synchronize()
        for actual, key in zip(packed, expected, strict=True):
            np.testing.assert_array_equal(cp.asnumpy(actual), expected[key])

    def test_real_group_run_publishes_node_views_for_multiaxis_and_scalar_results(self):
        cp = self.cp
        rng = np.random.default_rng(30491)
        for fixture_name, fixture in (("multiaxis", _multiaxis_group),
                                      ("scalar", _stacked_group)):
            group, catalog = fixture()
            batches, layouts, _views, _dtype = _bound_fixture(group, catalog)
            host_inputs = {}
            for batch in batches:
                for operand in (*batch.left, *batch.right):
                    spec = catalog[operand.tensor]
                    real = rng.normal(size=spec.shape)
                    imag = rng.normal(size=spec.shape)
                    host_inputs[operand.tensor] = (real + 1j * imag).astype(spec.dtype)
            expected_results = _independent_fixture_products(group, catalog, host_inputs)
            global_arrays = {key: cp.asarray(value) for key, value in host_inputs.items()}
            binding = {key: key for key in catalog}
            node = _NodeView(binding, global_arrays)
            output = bind_dense_gemm_group(group, catalog, cp)(node)
            cp.cuda.get_current_stream().synchronize()
            self.assertEqual(tuple(output.shape), (len(batches), *group.matrix_shape[::2]))

            keys_seen = set()
            for batch, layout in zip(batches, layouts, strict=True):
                for key, bounds, squeeze in layout.result_slices:
                    raw = node._raw_output_value(key)
                    self.assertIs(type(raw), _DeferredResultView,
                                  msg=f"{fixture_name} group did not publish a deferred result")
                    value = node[key]
                    expected = expected_results[key]
                    self.assertEqual(tuple(value.shape), tuple(expected.shape))
                    np.testing.assert_allclose(cp.asnumpy(value), expected,
                                               rtol=2e-12, atol=2e-12)
                    keys_seen.add(key)
            self.assertEqual(keys_seen, {result.tensor for batch in batches for result in batch.results})

    def test_real_single_batch_lazy_results_snapshot_and_downstream_pack(self):
        cp = self.cp
        for fixture_name, fixture in (("multiaxis", _multiaxis_group),
                                      ("scalar", _stacked_group)):
            with self.subTest(fixture=fixture_name):
                group, catalog = fixture()
                batch = group.batches[0]
                layout = bind_batch_layout(batch, catalog, cp)
                host_inputs = _fixture_batch_inputs(batch, catalog, 23)
                expected = _independent_batch_results(batch, host_inputs)
                arrays = {key: cp.asarray(value) for key, value in host_inputs.items()}
                node = _NodeView({key: key for key in catalog}, arrays)

                output = layout.run(node, backend=cp)
                self.assertEqual(tuple(output.shape), tuple(layout.output_shape))
                keys = tuple(result.tensor for result in batch.results)
                raw_values = tuple(node._raw_output_value(key) for key in keys)
                self.assertTrue(all(type(value) is _DeferredResultView
                                    for value in raw_values))
                for key, value in zip(keys, raw_values, strict=True):
                    self.assertEqual(tuple(value.shape), tuple(expected[key].shape))
                    self.assertEqual(value.dtype, expected[key].dtype)

                snapshot_layout = bind_output_layout(
                    keys, catalog, backend=cp, validate=False)
                with execution_debug(False), patch.object(
                        _DeferredResultView, "array",
                        side_effect=AssertionError("GPU snapshot must consume deferred pointers")):
                    snapshots = snapshot_layout.copy(node, cp)
                cp.cuda.get_current_stream().synchronize()
                self.assertEqual(len(snapshots.owners), 1)
                self.assertNotEqual(int(snapshots.owners[0].data.ptr), int(output.data.ptr))
                for actual, key in zip(snapshots, keys, strict=True):
                    np.testing.assert_allclose(cp.asnumpy(actual), expected[key],
                                               rtol=2e-12, atol=2e-12)

                if fixture_name == "multiaxis":
                    operands = tuple((key, None) for key in keys)
                    current = _packing_values(node, operands)
                    input_layouts = ((0, 1, 2, 3),) * len(current)
                    axis = 3
                    output_shape = list(current[0].shape)
                    output_shape[axis] *= len(current)
                    packer = bind_packing(
                        cp, tuple(value.shape for value in current),
                        tuple(value.dtype for value in current), axis, current[0].dtype,
                        input_layouts=input_layouts)
                    self.assertIsNotNone(packer)
                    self.assertTrue(callable(packer.pack_original))
                    destination = cp.empty(tuple(output_shape), dtype=current[0].dtype,
                                           order="C")
                    expected_packed = np.concatenate(
                        tuple(expected[key] for key in keys), axis=axis)
                    with execution_debug(False), patch.object(
                            _DeferredResultView, "array",
                            side_effect=AssertionError(
                                "GPU downstream pack must consume deferred pointers")):
                        packer.pack_original(current, destination)
                    cp.cuda.get_current_stream().synchronize()
                    np.testing.assert_allclose(cp.asnumpy(destination), expected_packed,
                                               rtol=2e-12, atol=2e-12)

                for key, expected_value in expected.items():
                    value = node[key]
                    self.assertIs(node._raw_output_value(key), value)
                    np.testing.assert_allclose(cp.asnumpy(value), expected_value,
                                               rtol=2e-12, atol=2e-12)


class RealCuPyDeferredPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            cupy.cuda.runtime.getDeviceProperties(0)
        except Exception as error:
            raise unittest.SkipTest(f"deferred packing checks require CuPy and a device: {error}")
        cls.cp = cupy

    def test_raw_original_pointer_pack_covers_dtype_permutation_and_singleton_layouts(self):
        cp = self.cp
        cases = (
            ("complex64-identity", np.dtype(np.complex64), (3, 4), (0, 1), 1),
            ("complex128-identity", np.dtype(np.complex128), (3, 4), (0, 1), 0),
            ("complex128-permutation", np.dtype(np.complex128), (4, 3), (1, 0), 1),
            ("complex64-singleton", np.dtype(np.complex64), (3, 4),
             (None, 0, 1), 2),
        )
        for case_name, dtype, source_shape, input_layout, axis in cases:
            with self.subTest(case=case_name):
                base = np.arange(2 * int(np.prod(source_shape)), dtype=np.float64)
                base = base.reshape((2, *source_shape))
                owner_host = (base + (len(case_name) + 0.5)
                              + 1j * (base / 13.0 + 0.75)).astype(dtype)
                owner = cp.asarray(owner_host)
                keys = (f"{case_name}-0", f"{case_name}-1")
                node = _NodeView({key: key for key in keys}, {})
                values = _publish_owner_slices(node, owner, keys)
                logical_shape = tuple(
                    1 if source_axis is None else source_shape[source_axis]
                    for source_axis in input_layout)
                layouts = (input_layout, input_layout)
                packer = bind_packing(
                    cp, (logical_shape, logical_shape), (dtype, dtype),
                    axis, dtype, input_layouts=layouts)
                self.assertIsNotNone(packer, case_name)
                self.assertTrue(callable(packer.pack_original), case_name)
                output_shape = list(logical_shape)
                output_shape[axis] *= 2
                destination = cp.empty(tuple(output_shape), dtype=dtype, order="C")
                expected = np.concatenate(
                    tuple(_apply_input_layout(owner_host[lane], input_layout)
                          for lane in range(2)),
                    axis=axis).astype(dtype, copy=False)

                with execution_debug(False), patch.object(
                        _DeferredResultView, "array",
                        side_effect=AssertionError(
                            f"{case_name} raw pack materialized a deferred view")):
                    packer.pack_original(values, destination)
                cp.cuda.get_current_stream().synchronize()
                self.assertEqual(tuple(destination.shape), tuple(output_shape), case_name)
                np.testing.assert_array_equal(cp.asnumpy(destination), expected,
                                              err_msg=case_name)

        with self.subTest(case="complex128-strided-column-slices"):
            dtype = np.dtype(np.complex128)
            base = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
            owner_host = (base + (23.5 + 0.25j * base)).astype(dtype)
            owner = cp.asarray(owner_host)
            keys = ("strided-columns-0", "strided-columns-1")
            node = _NodeView({key: key for key in keys}, {})
            values = _publish_owner_slices(
                node, owner, keys, columns=(slice(0, 2), slice(2, 4)))
            self.assertEqual(tuple(value.shape for value in values), ((3, 2), (3, 2)))
            self.assertEqual(tuple(value.strides for value in values),
                             ((4 * dtype.itemsize, dtype.itemsize),) * 2)
            layout = ((0, 1), (0, 1))
            packer = bind_packing(
                cp, ((3, 2), (3, 2)), (dtype, dtype), 1, dtype,
                input_layouts=layout)
            self.assertIsNotNone(packer)
            destination = cp.empty((3, 4), dtype=dtype, order="C")
            expected = np.concatenate(
                (owner_host[0, :, :2], owner_host[1, :, 2:]), axis=1)
            with execution_debug(False), patch.object(
                    _DeferredResultView, "array",
                    side_effect=AssertionError(
                        "strided GPU copy must use pointer plus runtime strides")):
                packer.pack_original(values, destination)
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_array_equal(cp.asnumpy(destination), expected)

            borrowed_keys = ("borrowed-columns-0", "borrowed-columns-1")
            borrowed_node = _NodeView({key: key for key in borrowed_keys}, {})
            borrowed_refs = _publish_owner_slices(
                borrowed_node, owner, borrowed_keys, lane_indices=(0, 0),
                columns=(slice(0, 2), slice(2, 4)))
            with execution_debug(False), patch.object(
                    _DeferredResultView, "array",
                    side_effect=AssertionError(
                        "adjacent same-owner columns must be borrowed without materializing")):
                borrowed = _contiguous_concatenation(
                    borrowed_refs, 1, (3, 4), dtype,
                    input_layouts=((0, 1), (0, 1)))
            self.assertIsNotNone(borrowed)
            self.assertEqual(int(borrowed.data.ptr), int(owner.data.ptr))
            self.assertEqual(tuple(borrowed.shape), (3, 4))
            np.testing.assert_array_equal(cp.asnumpy(borrowed), owner_host[0])

    def test_real_group_packer_copies_six_current_refs_from_two_owners(self):
        cp = self.cp
        group, catalog = _multiaxis_group()
        batches, layouts, _matrix_shape, dtype = _bind_layouts(group, catalog, cp)
        sides = tuple(layout.sides[1] for layout in layouts)
        packer = bind_gemm_group_packing(
            cp, sides, catalog, layouts[0].matrix_shapes[1], dtype)
        self.assertIsNotNone(packer)

        node = _NodeView({key: key for key in catalog}, {})
        right_a_keys = tuple(batch.right[0].tensor for batch in batches)
        right_b_keys = tuple(batch.right[1].tensor for batch in batches)
        rng = np.random.default_rng(30491)

        def make_sources(offset):
            shape = (3, 4, 5, 1)
            first = (rng.normal(size=shape) + 1j * rng.normal(size=shape) + offset).astype(dtype)
            second = (rng.normal(size=shape) + 1j * rng.normal(size=shape) - offset).astype(dtype)
            return first, second, cp.asarray(first), cp.asarray(second)

        def expected_rows(first, second):
            return np.stack(tuple(
                np.concatenate((first[lane], second[lane]), axis=2).reshape(4, 10)
                for lane in range(3)))

        first_host_a, first_host_b, first_owner_a, first_owner_b = make_sources(2.0)
        refs_a = _publish_owner_slices(node, first_owner_a, right_a_keys)
        refs_b = _publish_owner_slices(node, first_owner_b, right_b_keys)
        self.assertEqual(len((*refs_a, *refs_b)), 6)
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("real group packing must use pointer arguments")):
            first_packed = packer.pack_buffer(node)
        cp.cuda.get_current_stream().synchronize()
        self.assertEqual(tuple(first_packed.shape), (3, 4, 10))
        np.testing.assert_array_equal(cp.asnumpy(first_packed),
                                     expected_rows(first_host_a, first_host_b))

        second_host_a, second_host_b, second_owner_a, second_owner_b = make_sources(17.0)
        _publish_owner_slices(node, second_owner_a, right_a_keys)
        _publish_owner_slices(node, second_owner_b, right_b_keys)
        with execution_debug(False), patch.object(
                _DeferredResultView, "array",
                side_effect=AssertionError("fresh group refs must remain pointer-only")):
            second_packed = packer.pack_buffer(node)
        cp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(cp.asnumpy(second_packed),
                                     expected_rows(second_host_a, second_host_b))
        self.assertNotEqual(int(first_packed.data.ptr), int(second_packed.data.ptr))
        for copied in (first_packed, second_packed):
            for source in (first_owner_a, first_owner_b, second_owner_a, second_owner_b):
                self.assertFalse(
                    int(copied.data.ptr) < int(source.data.ptr) + source.nbytes
                    and int(source.data.ptr) < int(copied.data.ptr) + copied.nbytes)


if __name__ == "__main__":
    unittest.main()
