"""Numerical and lifecycle checks for fixed replay preparation/packing."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.execution.operations.packing_kernel import bind_packing
from correlator.execution.test_fused_preparation import _HostKernelBackend
from correlator.execution.test_prebound_preparation import _spin_preparation, _outer_oracle
from test_gpu_output_postprocessing import _processor_case, _device_tree, output


class PackingOracleTests(unittest.TestCase):
    def test_arbitrary_axis_chunks_mixed_dtypes_and_changed_values(self):
        backend = _HostKernelBackend()
        self.addCleanup(backend.close)
        for axis in (0, 1, 2):
            values = []
            for index in range(19):
                shape = [2, 3, 4]
                shape[axis] = index % 3  # includes empty segments
                dtype = np.complex128 if index % 2 else np.float32
                a = np.arange(np.prod(shape), dtype=np.float64).reshape(shape).astype(dtype)
                if index % 2:
                    a += 2j * a
                values.append(a[::-1, :, ::-1])
            pack = bind_packing(backend, tuple(v.shape for v in values),
                                tuple(v.dtype for v in values), axis, np.complex128)
            for scale in (1, 3):
                current = [scale * v for v in values]
                current = [v[::-1] for v in current]
                expected = np.concatenate(current, axis=axis)
                result = np.empty(expected.shape, dtype=expected.dtype)
                self.assertIs(pack(current, result), result)
                np.testing.assert_array_equal(result, expected)

    def test_equal_coordinate_recipes_reuse_kernel_without_caching_values(self):
        backend = _HostKernelBackend()
        self.addCleanup(backend.close)
        spin = np.eye(4, dtype=np.complex128)
        preps = {str(i): _spin_preparation(spin, block=f'raw{i}') for i in range(3)}
        specs = {key: TensorSpec((4, 4, 2, 3), np.complex128) for key in preps}
        loader = bind_prepared_loader(None, preps, specs, backend=backend)
        outputs = []
        for i in range(3):
            raw = np.full((2, 3), i + 1j, dtype=np.complex128)
            result = loader.materialize(str(i), None, {f'raw{i}': raw})
            np.testing.assert_array_equal(result, _outer_oracle(spin, raw))
            outputs.append(result)
            raw[:] = 9
        self.assertEqual(backend.created, 1)
        self.assertEqual(len(loader._fused_programs), 1)
        self.assertFalse(np.shares_memory(outputs[0], outputs[1]))


class OutputTemplateCacheTests(unittest.TestCase):
    def test_structural_comparison_only_on_new_template(self):
        _, _, values, backend, processor, original, _ = _processor_case()
        class Recipe:
            comparisons = 0
            def __eq__(self, other):
                self.comparisons += 1
                return isinstance(other, Recipe)
        recipe = Recipe()
        first = SimpleNamespace(output=recipe, plan=original.plan)
        second = SimpleNamespace(output=Recipe(), plan=original.plan)
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]):
            for index, point in enumerate(values):
                processor.process(index, (_device_tree(point, backend),), first if index == 0 else second)
        self.assertEqual(recipe.comparisons, 1)
        processor.close()
        self.assertFalse(processor._template_programs)
