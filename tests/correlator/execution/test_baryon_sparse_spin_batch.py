"""Shared-coordinate sparse spin against independent scalar contractions."""

import unittest

import numpy as np

from lqcdweave.contraction.execution.operations.baryon_sparse_spin import (
    BatchedSparseSpinRecipe,
    contract_sparse_spin,
    prepare_sparse_spin_elemental,
    sparse_spin_recipe,
)


PATH = ((0, 2), (0, 1), (1, 2), (0, 1))
ENTRIES = ((0, 1, 2, 0.7 + 0.2j), (1, 0, 2, -0.3j),
           (3, 2, 1, -0.4 + 0.1j))


def _case(width, dtype, dynamic, permuted=False):
    dynamic = list(dynamic)
    rng = np.random.default_rng(1157 + width)
    shapes = ((2, 3, 2), (4, 2, 2, 2), (4, 2, 3, 1), (4, 2, 2, 3))
    spin = np.zeros((4, 4, 4), dtype=dtype)
    for i, j, k, weight in ENTRIES:
        spin[i, j, k] = weight
    values = [spin]
    for size, is_dynamic in zip(shapes, dynamic):
        shape = ((width,) if is_dynamic else ()) + size
        values.append((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype))
    labels = ['ijk', 'abc', 'ilad', 'jmbe', 'kncf']
    if permuted:
        values[1] = values[1].swapaxes(-1, -3)
        labels[1] = 'cba'
        values[2] = values[2].swapaxes(-1, -4)
        labels[2] = 'dlai'
        values[3], values[4] = values[4], values[3]
        labels[3], labels[4] = labels[4], labels[3]
        dynamic[2], dynamic[3] = dynamic[3], dynamic[2]
    for index, is_dynamic in enumerate(dynamic, 1):
        if is_dynamic:
            labels[index] = 't' + labels[index]
    output = 'fnemdl' if permuted else 'ldmenf'
    equation = ','.join(labels) + '->t' + output
    reference = []
    scalar_equation = ','.join(label.removeprefix('t') for label in labels) + '->' + output
    for lane in range(width):
        scalar = [values[0]] + [value[lane] if active else value
                                for value, active in zip(values[1:], dynamic)]
        reference.append(np.einsum(scalar_equation, *scalar, optimize=False))
    return equation, tuple(values), np.stack(reference)


class BatchedSparseSpinTests(unittest.TestCase):
    def test_shared_lane_matches_independent_scalar_for_shapes_and_precision(self):
        cases = (
            ((True, True, True, True), False),
            ((False, True, True, True), True),
            ((False, False, True, True), False),
            ((True, False, True, False), True),
            ((True, False, False, False), False),
        )
        for dtype in ('complex64', 'complex128'):
            for width in (1, 2, 3):
                for dynamic, permuted in cases:
                    with self.subTest(dtype=dtype, width=width, dynamic=dynamic,
                                      permuted=permuted):
                        equation, values, expected = _case(width, dtype, dynamic, permuted)
                        recipe = sparse_spin_recipe(equation, tuple(value.shape for value in values),
                                                    (dtype,) * 5, PATH)
                        self.assertIsInstance(recipe, BatchedSparseSpinRecipe)
                        packed = prepare_sparse_spin_elemental(recipe, values[1], np)
                        result = contract_sparse_spin(recipe, ENTRIES, values, np,
                                                      prepared_elemental=packed)
                        tolerance = 4e-6 if dtype == 'complex64' else 4e-13
                        np.testing.assert_allclose(result, expected, rtol=tolerance,
                                                   atol=tolerance)
                        self.assertEqual(result.dtype, np.dtype(dtype))

    def test_single_dynamic_leg_retains_scalar_route(self):
        equation, values, expected = _case(3, 'complex128',
                                           (False, True, False, False))
        recipe = sparse_spin_recipe(equation, tuple(value.shape for value in values),
                                    ('complex128',) * 5, PATH)
        self.assertIsNotNone(recipe)
        self.assertNotIsInstance(recipe, BatchedSparseSpinRecipe)
        result = contract_sparse_spin(recipe, ENTRIES, values, np)
        np.testing.assert_allclose(result, expected, rtol=4e-13, atol=4e-13)

    def test_gpu_batched_matmul_matches_independent_cpu_reference(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                self.skipTest('no CUDA device')
        except (ImportError, RuntimeError):
            self.skipTest('CuPy or CUDA is unavailable')
        for width in (1, 3):
            equation, values, expected = _case(width, 'complex128',
                                               (False, True, True, True), True)
            arrays = tuple(cp.asarray(value) for value in values)
            recipe = sparse_spin_recipe(equation, tuple(value.shape for value in arrays),
                                        ('complex128',) * 5, PATH)
            self.assertIsInstance(recipe, BatchedSparseSpinRecipe)
            packed = prepare_sparse_spin_elemental(recipe, arrays[1], cp)
            actual = contract_sparse_spin(recipe, ENTRIES, arrays, cp,
                                          prepared_elemental=packed)
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=4e-13, atol=4e-13)


if __name__ == '__main__':
    unittest.main()
