"""Exact monomial source representations before ordinary dense planning."""
from dataclasses import replace
from types import MappingProxyType
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, TensorPreparation, TensorSpec, SourcePreparation,
)
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.operators.single_hadron.api import gamma
from lqcdweave.storage.contraction_plans.api import _encode_plan_value, _decode_plan_value


def _program(backend, recipe, shape, dtype):
    return bind_prepared_loader(None,
        MappingProxyType({'x': SourcePreparation(('raw',), recipe=recipe)}),
        MappingProxyType({'x': TensorSpec(shape, dtype)}), backend=backend)


class MonomialPreparationTests(unittest.TestCase):
    def test_spin_compression_retains_profile_dagger_and_selected_mode_values(self):
        rng = np.random.default_rng(261021)
        for dtype in (np.complex64, np.complex128):
            spin = gamma(7).astype(dtype)
            columns = tuple(map(int, np.nonzero(spin)[1]))
            constants = (spin, rng.normal(size=(3, 2)).astype(dtype),
                         rng.normal(size=(3, 3)).astype(dtype),
                         np.array([.7, -.2, .4], dtype))
            base = TensorPreparation('spin_spatial', ('raw',),
                tuple(ArrayConstant.from_array(v) for v in constants),
                conjugate=True, permutation=(1, 0), configuration='cfg', compute_dtype=np.dtype(dtype).name)
            compressed = replace(base, spin_columns=columns)
            full = _program(np, base, (4, 4, 3, 2), dtype)
            small = _program(np, compressed, (4, 3, 2), dtype)
            for repeat in range(2):
                value = (rng.normal(size=(2, 3))+1j*rng.normal(size=(2, 3))).astype(dtype)[:, ::-1]
                expected = full.materialize('x', None, {'raw': value})
                actual = small.materialize('x', None, {'raw': value})
                retained = np.stack([expected[row, col] for row, col in enumerate(columns)])
                np.testing.assert_array_equal(actual, retained)
                self.assertFalse(np.shares_memory(actual, value))
            restored = _decode_plan_value(_encode_plan_value(compressed))
            self.assertEqual(restored, compressed)

    def test_identity_gathers_are_owned_and_archive_roundtrip_keeps_both_axes(self):
        order = (3, 1, 0, 2)
        recipe = TensorPreparation('identity', ('raw',),
            index_permutations=((0, order), (1, order)), compute_dtype='complex64')
        loader = _program(np, recipe, (4, 4, 2, 3), np.complex64)
        value = np.arange(96).reshape(4, 4, 2, 3).astype(np.complex128)
        actual = loader.materialize('x', None, {'raw': value})
        np.testing.assert_array_equal(actual, value[np.array(order)][:, np.array(order)].astype(np.complex64))
        self.assertFalse(np.shares_memory(actual, value))
        self.assertTrue(loader.materializes_owned('x'))
        self.assertEqual(_decode_plan_value(_encode_plan_value(recipe)), recipe)
        legacy = _encode_plan_value(TensorPreparation('identity', ('raw',)))
        del legacy['value']['spin_columns'], legacy['value']['index_permutations']
        self.assertEqual(_decode_plan_value(legacy), TensorPreparation('identity', ('raw',)))

    def test_invalid_permutations_are_rejected_without_reading_numerical_inputs(self):
        for columns in ((0, 0, 2, 3), (0, 1, 2), (0, 1, 2, 4)):
            with self.assertRaises(ValueError):
                TensorPreparation('identity', ('raw',), spin_columns=columns)
        for axes in (((-1, (0, 1, 2, 3)),), ((0, (0, 0, 2, 3)),)):
            with self.assertRaises(ValueError):
                TensorPreparation('identity', ('raw',), index_permutations=axes)


class MonomialPreparationCuPyTests(unittest.TestCase):
    def test_public_matrix_and_fixed_archive_match_independent_cpu_workflow(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from test_structured_correlator import fixture, task
        from lqcdweave.contraction.correlator import calculate, save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        bind, elemental, propagator = fixture()
        options = dict(vertex_map=bind, configuration='cfg', usedNe=3,
            propagator_map={'S^q':propagator, r'S^q_\mathrm{local}':propagator},
            memory_limit=1<<30, host_memory_limit=1<<30)
        selected = calculate(task(structured=True), backend='cupy', execute=False,
            time_axes={0:0,1:1}, partitions=(0,1,3),
            search={'deadline_seconds':1.0, 'max_states':1000, 'component_workers':1}, **options)
        self.assertTrue(any(p.recipe.spin_columns or p.recipe.spatial_only for p in selected.inputs.loader.preparations.values()
                            if p.recipe is not None))
        pairs=({0:0,1:1},{0:1,1:0})
        reference=calculate(task(structured=False), optimize='none', time_pairs=pairs,
                            partitions=(0,1,3), **options).value
        coordinates,_=describe_coordinates(time_pairs=pairs)
        complete=complete_calculation(selected, coordinates=coordinates,
                                       readers=selected.inputs.loader.input_readers)
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'spin.json.gz'
            save_calculation_plan(archive, complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('archive searched')):
                loaded=calculate(None, plan_archive=archive, configuration='cfg',
                    vertex_map=elemental, propagator_map=options['propagator_map'],
                    memory_limit=1<<30, host_memory_limit=1<<30, execute=False)
                first=loaded.run().value
                second=loaded.run().value
        np.testing.assert_allclose(first, reference, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(second, reference, rtol=3e-12, atol=3e-12)

    def test_raw_profile_compression_matches_full_kernel_nonzeros(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        rng = np.random.default_rng(261022)
        spin = gamma(3).astype(np.complex64)
        columns = tuple(map(int, np.nonzero(spin)[1]))
        values = (spin, rng.normal(size=(2, 3)).astype(np.complex64),
                  rng.normal(size=(2, 3)).astype(np.complex64), np.array([.7, -.2], np.float32))
        recipe = TensorPreparation('spin_spatial', ('raw',),
            tuple(ArrayConstant.from_array(v) for v in values), conjugate=True,
            permutation=(1, 0), configuration='cfg', compute_dtype='complex64')
        full = _program(cp, recipe, (4, 4, 3, 3), np.complex64)
        small = _program(cp, replace(recipe, spin_columns=columns), (4, 3, 3), np.complex64)
        for repeat in range(2):
            value = cp.asarray((rng.normal(size=(3, 3))+repeat).astype(np.complex64))[::-1, ::-1]
            a = full.materialize('x', None, {'raw': value})
            b = small.materialize('x', None, {'raw': value})
            expected = cp.stack([a[row, col] for row, col in enumerate(columns)])
            np.testing.assert_array_equal(cp.asnumpy(b), cp.asnumpy(expected))
