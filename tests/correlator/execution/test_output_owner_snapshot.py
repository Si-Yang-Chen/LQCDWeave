"""CPU address/ownership oracles for X's private device-output handoff.

Protocol arrays below use real host addresses, never CUDA. Actual CuPy
tests remain separately marked and cannot be replaced by these oracles.
"""
import ctypes
from dataclasses import replace
import gc
import pickle
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

from lqcdweave.contraction.execution import output_program
from lqcdweave.contraction.execution.bound_program import bind_program
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.output import (
    PackedOutputs, _OutputOwnerSnapshot, bind_output_layout,
)
from lqcdweave.contraction.results.api import bind_output_assembly
from correlator.execution.test_device_output_program import _fixture, _term
from lqcdweave.contracts.correlator.api import OutputNode, TensorSpec


class _HostAddressArray(np.ndarray):
    @property
    def data(self):
        return SimpleNamespace(ptr=self.ctypes.data)


class _AddressBackend:
    """Reads the submitted address/stride ABI with an independent CPU oracle."""
    __name__ = 'cupy'
    ndarray = _HostAddressArray

    def __init__(self):
        self.uploads = []
        self.kernel_groups = ()
        self.descriptor = None
        self.arguments = ()
        self.kernel_count = 0

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype).view(_HostAddressArray)

    def asarray(self, value):
        self.uploads.append(value)
        return np.array(value, copy=True).view(_HostAddressArray)

    def RawKernel(self, source, name, options=()):
        def copy(grid, block, args):
            values, offset, destination = args[:-2], int(args[-2]), args[-1]
            for lane, value in enumerate(values):
                destination[offset + lane] = value
        return copy

    def ElementwiseKernel(self, inputs, outputs, operation, name, **kwargs):
        if name == 'ed_pack_outputs':
            def copy(*args, size):
                values, offset, destination = args[:-3], int(args[-2]), args[-1]
                for lane, value in enumerate(values):
                    destination[offset + lane] = value
            return copy
        number = self.kernel_count
        self.kernel_count += 1
        indirect = inputs == 'raw uint64 addresses'

        def launch(*args, size):
            output = args[-1]
            output.fill(0)
            if indirect:
                table = args[0]
                for term in self.kernel_groups[number]:
                    shape, dtype = self.descriptor.term_specs[term]
                    offset = self.descriptor.table_offsets['input', term]
                    address = int(table[offset])
                    strides = table.view('int64')[offset + 1:offset + 1 + len(shape)]
                    for index in np.ndindex(shape):
                        pointer = address + sum(i * int(s) for i, s in zip(index, strides, strict=True))
                        # Independent reads of submitted bytes; no snapshot views.
                        value = np.frombuffer(ctypes.string_at(pointer, dtype.itemsize),
                                              dtype=dtype, count=1)[0]
                        output[index] += value
            else:
                for array in args[:-1]:
                    output[...] += array
        return launch


def _bind(root, host_values):
    recipe, problem = _fixture(root, *host_values)
    backend = _AddressBackend()
    bound = output_program.bind_device_output_assembly(recipe, problem, backend)
    backend.descriptor = bound.description
    backend.kernel_groups = tuple(tuple(index for kind, index in args if kind == 'input')
                                  for _, args, *_ in bound.programs)
    arrays = dict(zip(recipe.term_ids, host_values, strict=True))
    layout = bind_output_layout(recipe.term_ids, problem.specs, backend=backend, validate=False)
    return recipe, backend, bound, arrays, layout


class OutputOwnerSnapshotTests(unittest.TestCase):
    def setUp(self):
        output_program._output_kernel.cache_clear()

    def test_lazy_scalar_vector_order_and_repeated_lane_survive_source_release(self):
        arrays = {'v': np.arange(6.).reshape(2, 3), 's': np.array(7.)}
        backend = _AddressBackend()
        layout = bind_output_layout(('s', 'v', 'v'), {
            key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()
        }, validate=False)
        snapshot = layout.copy(arrays, backend, owner_snapshot=True)
        self.assertIs(type(snapshot), _OutputOwnerSnapshot)
        self.assertEqual(snapshot._views, {})
        for source in arrays.values():
            source[...] = -1
        self.assertEqual(snapshot[0].shape, ())
        self.assertEqual(snapshot[0].item(), 7.)
        self.assertIs(snapshot[1], snapshot[1])
        self.assertIs(snapshot[-1], snapshot[2])
        np.testing.assert_array_equal(snapshot[1], np.arange(6.).reshape(2, 3))
        self.assertTrue(np.shares_memory(snapshot[1], snapshot.owners[1]))
        self.assertEqual(len(snapshot[:2]), 2)
        for index in (-4, 3):
            with self.assertRaises(IndexError):
                snapshot[index]
        with self.assertRaises(TypeError):
            snapshot[1.5]
        # The private provenance never becomes a serialized contract.
        restored = pickle.loads(pickle.dumps(snapshot))
        self.assertIs(type(restored), PackedOutputs)
        self.assertFalse(hasattr(restored, 'provenance'))
        for value, expected in zip(restored, snapshot, strict=True):
            np.testing.assert_array_equal(value, expected)

    def test_mixed_scalar_vector_indirect_table_matches_R_without_any_term_views(self):
        rng = np.random.default_rng(81047)
        values = tuple((rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
                       for _ in range(100)) + tuple(
            np.array(rng.normal() + 1j * rng.normal()) for _ in range(200))
        root = OutputNode('dict', keys=('vectors', 'scalars'), children=(
            OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(100))),
            OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(100, 300)))))
        recipe, backend, bound, arrays, layout = _bind(root, values)
        self.assertTrue(all(bound.description.indirect_stages))
        reference = bind_output_assembly(recipe)(values)
        for shift in (0., 2.5):
            shifted = {key: value + shift for key, value in arrays.items()}
            snapshot = layout.copy(shifted, backend, owner_snapshot=True)
            actual = bound(snapshot)
            self.assertEqual(snapshot._views, {})
            for key, count in (('vectors', 100), ('scalars', 200)):
                np.testing.assert_allclose(actual[key], reference[key] + count * shift,
                                           rtol=3e-12, atol=3e-12)
            # Each pointer is the matching current owner lane, with real strides.
            table = backend.uploads[-1]
            for (kind, term), shape, dtype, offset in bound.description.table_layout:
                self.assertEqual(kind, 'input')
                group, lane = layout.positions[term]
                owner = snapshot.owners[group]
                self.assertEqual(int(table[offset]), owner.ctypes.data + lane * owner.strides[0])
                self.assertEqual(tuple(table.view('int64')[offset + 1:offset + 1 + len(shape)]),
                                 owner.strides[1:])
        self.assertEqual(len(bound._owner_layout_schemas), 1)

    def test_direct_arguments_are_lazy_reused_and_stay_inside_the_current_snapshot(self):
        values = (np.arange(6.).reshape(2, 3), np.ones((2, 3)))
        root = OutputNode('dict', keys=('sum', 'again'), children=(
            OutputNode('add', children=(_term(0), _term(1))), _term(0)))
        recipe, backend, bound, arrays, layout = _bind(root, values)
        snapshot = layout.copy(arrays, backend, owner_snapshot=True)
        owners = tuple(weakref.ref(owner) for owner in snapshot.owners)
        actual = bound(snapshot)
        self.assertEqual(set(snapshot._views), {0, 1})
        reference = bind_output_assembly(recipe)(values)
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key])
        del snapshot
        gc.collect()
        self.assertTrue(all(owner() is None for owner in owners))
        # The bounded program cache owns metadata only, including after errors.
        fresh = layout.copy({key: value + 3 for key, value in arrays.items()},
                            backend, owner_snapshot=True)
        np.testing.assert_array_equal(bound(fresh)['sum'], reference['sum'] + 6)

    def test_first_layout_and_every_owner_keep_boundary_checks(self):
        values = (np.ones((2, 3)), np.array(1.))
        _, backend, bound, arrays, layout = _bind(
            OutputNode('tuple', children=(_term(0), _term(1))), values)
        snapshot = layout.copy(arrays, backend, owner_snapshot=True)
        for bad_layout in (
                replace(layout, positions=tuple(reversed(layout.positions))),
                replace(layout, positions=((0, 99), (1, 0)))):
            with self.assertRaises(ValueError):
                bound(_OutputOwnerSnapshot(snapshot.owners, bad_layout))
        bound(snapshot)
        for owners in (snapshot.owners[:1],
                       (np.empty((1, 2, 4)).view(_HostAddressArray), snapshot.owners[1]),
                       (np.empty((1, 2, 3), dtype='float32').view(_HostAddressArray), snapshot.owners[1])):
            with self.assertRaises(ValueError):
                bound(_OutputOwnerSnapshot(owners, layout))
        with self.assertRaises(TypeError):
            bound(_OutputOwnerSnapshot((np.empty((1, 2, 3)), snapshot.owners[1]), layout))
        self.assertLessEqual(len(bound._owner_layout_schemas), 16)

    def test_debug_numpy_and_default_program_return_the_existing_tuple(self):
        arrays = {'out': np.arange(4.)}
        catalog = {'out': TensorSpec((4,), 'float64')}
        backend = _AddressBackend()
        layout = bind_output_layout(('out',), catalog, validate=False)
        with execution_debug(True):
            checked = layout.copy(arrays, backend, owner_snapshot=True)
        self.assertIs(type(checked), PackedOutputs)
        self.assertIs(type(layout.copy(arrays, np, owner_snapshot=True)), PackedOutputs)
        state = SimpleNamespace(actions=(), results=('out',), catalog=catalog, problem=None)
        program = bind_program(state, backend, {}, {}, {}, {}, (), (), (None,) * 4)
        self.assertIsInstance(program.run(None, arrays), tuple)
        compact = program.run(None, arrays, owner_snapshot=True)
        self.assertIs(type(compact), _OutputOwnerSnapshot)
        np.testing.assert_array_equal(compact[0], arrays['out'])

    def test_indirect_table_uses_current_negative_owner_strides(self):
        values = tuple(np.arange(6.).reshape(2, 3) + i for i in range(100))
        root = OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(100)))
        recipe, backend, bound, arrays, layout = _bind(root, values)
        packed = layout.copy(arrays, backend, owner_snapshot=True)
        # Reversed lanes and axes demonstrate that the table derives actual
        # current strides, not a fabricated contiguous byte-offset formula.
        owner = packed.owners[0][::-1, ::-1, ::-1]
        snapshot = _OutputOwnerSnapshot((owner,), layout)
        actual = bound(snapshot)
        reference = bind_output_assembly(recipe)(tuple(owner[i] for i in range(100)))
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual(snapshot._views, {})


class RealCuPyOwnerSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceProperties(0)
        except Exception as error:
            raise unittest.SkipTest(f'actual CUDA owner snapshot test requires a device: {error}')
        cls.cp = cupy

    def test_actual_indirect_and_direct_outputs_match_independent_host_sum(self):
        cp = self.cp
        for count, shape in ((2, (2, 3)), (100, (2, 3)), (2, ()), (200, ())):
            with self.subTest(count=count, shape=shape):
                base = np.arange(6).reshape(2, 3) if shape else np.array(1.25)
                host = tuple((base + i * .25).astype('complex128') for i in range(count))
                root = OutputNode('diagram_sum', tuple((f't{i}', 1) for i in range(count)))
                recipe, problem = _fixture(root, *host)
                bound = output_program.bind_device_output_assembly(recipe, problem, cp)
                layout = bind_output_layout(recipe.term_ids, problem.specs, backend=cp, validate=False)
                arrays = {key: cp.asarray(value) for key, value in zip(recipe.term_ids, host, strict=True)}
                first = layout.copy(arrays, cp, owner_snapshot=True)
                actual = bound(first)
                expected = sum(host[1:], host[0].copy())
                np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=3e-12, atol=3e-12)
                self.assertEqual(len(first._views), count if count == 2 else 0)
                second = layout.copy({key: value + 2 for key, value in arrays.items()},
                                     cp, owner_snapshot=True)
                np.testing.assert_allclose(cp.asnumpy(bound(second)), expected + 2 * count,
                                           rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(cp.asnumpy(bound(first)), expected, rtol=3e-12, atol=3e-12)
