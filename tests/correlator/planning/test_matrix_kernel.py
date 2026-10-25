import unittest
from math import prod
from unittest.mock import patch
import numpy as np

from lqcdweave.contracts.correlator.api import AxisPattern, MatrixKernelSpec
from lqcdweave.contraction.planning.api import estimate_matrix_kernel
from lqcdweave.contraction.execution import matrix_kernels as kernels


class RecordingProfile:
    def __init__(self):
        self.calls = []

    def contract_seconds(self, flops, traffic_bytes, *, dtype, kernel_calls):
        self.calls.append((flops, traffic_bytes, np.dtype(dtype).name, kernel_calls))
        return flops*0.001 + traffic_bytes*0.002 + kernel_calls*0.003


class MatrixKernelEstimateTests(unittest.TestCase):
    def test_connected_counts_match_actual_arithmetic_calls(self):
        # Inspect the actual calls and their einsum indices independently of
        # the closed-form planner counts, including a partial final row tile.
        n, cl, cr, nl, nr = 7, 3, 2, 5, 4
        arrays = [np.ones(shape, dtype=dtype) for shape, dtype in (
            ((4,4,n,n), 'complex128'), ((4,4,n,n), 'complex128'),
            ((cl,4,4,n,n), 'complex128'), ((cr,4,4,n,n), 'complex128'),
            ((nl,cl), 'complex128'), ((nr,cr), 'complex128'))]
        for partitions in (None, (0,1,3,4,7)):
            for row_block in (1,2,8):
                with self.subTest(partitions=partitions, row_block=row_block):
                    spec = MatrixKernelSpec('connected', tuple(a.shape for a in arrays),
                        tuple(a.dtype.name for a in arrays), partitions, row_block)
                    profile = RecordingProfile()
                    estimate = estimate_matrix_kernel(spec, profile=profile)
                    observed = [0,0,0]
                    binary, linear = kernels._binary, kernels._linear

                    def record_binary(equation, left, right, *, backend=np):
                        self.assertIsInstance(equation, AxisPattern)
                        sizes = {}
                        for indices, value in zip(equation.inputs, (left,right), strict=True):
                            self.assertEqual(len(indices), value.ndim)
                            for label, size in zip(indices, value.shape, strict=True):
                                previous = sizes.setdefault(label, size)
                                self.assertEqual(previous, size)
                        result = binary(equation,left,right,backend=backend)
                        self.assertEqual(result.shape, tuple(sizes[label] for label in equation.output))
                        observed[0] += 2*prod(sizes.values())
                        observed[1] += left.nbytes+right.nbytes+result.nbytes
                        observed[2] += 1
                        return result

                    def record_linear(weights, products):
                        result = linear(weights,products)
                        observed[0] += 2*len(weights)*products.size
                        observed[1] += weights.nbytes+products.nbytes+result.nbytes
                        observed[2] += 1
                        return result

                    with patch.object(kernels,'_binary',record_binary), patch.object(kernels,'_linear',record_linear):
                        output = kernels.contract_meson_matrix_pc(*arrays,partitions=partitions,row_block=row_block)
                    stages = [s for s in estimate.stages if s.name != 'accumulate']
                    self.assertEqual(observed, [sum(s.flops for s in stages),
                        sum(s.traffic_bytes for s in stages),sum(s.kernel_calls for s in stages)])
                    self.assertEqual(estimate.output_bytes,output.nbytes)
                    self.assertEqual(estimate.promotion_bytes,0)
                    self.assertAlmostEqual(estimate.seconds, sum(f*0.001+t*0.002+c*0.003
                        for f,t,d,c in profile.calls))
                    self.assertTrue(all(d=='complex128' for f,t,d,c in profile.calls))

    def test_peak_components_and_reserve(self):
        spec = MatrixKernelSpec('connected', ((4,4,70,70),)*2+
            ((12,4,4,70,70),(8,4,4,70,70),(30,12),(20,8)),
            ('complex64',)*2+('complex128',)*4,(0,16,32,48,70),3)
        estimate = estimate_matrix_kernel(spec,profile=RecordingProfile(),workspace_reserve_bytes=4096)
        self.assertEqual(estimate.retained_product_bytes,4*20*16*70*70*16)
        self.assertEqual(estimate.output_bytes,30*20*100*16)
        # Worst K map retains the primitive, its full reshape pack and
        # weighted row tile together. Original inputs and H also remain live.
        self.assertGreaterEqual(estimate.temporary_bytes,(2*12+3)*16*70*22*16)
        self.assertEqual(estimate.peak_array_bytes, sum((estimate.input_bytes,
            estimate.promotion_bytes, estimate.output_bytes, estimate.retained_product_bytes,
            estimate.temporary_bytes,4096)))

    def test_loop_and_empty_components(self):
        for comp in (0,3):
            spec = MatrixKernelSpec('loop', ((4,4,7,7),(comp,4,4,7,7),(5,comp)),
                ('float32','complex128','complex128'),(0,1,3,4,7))
            profile = RecordingProfile()
            estimate = estimate_matrix_kernel(spec,profile=profile)
            self.assertEqual(estimate.output_bytes,5*10*16)
            self.assertEqual(estimate.retained_product_bytes,0)
            if comp:
                stages = {s.name:s for s in estimate.stages}
                self.assertEqual(stages['loop_product'].flops,2*comp*16*7*7)
                self.assertEqual(stages['loop_product'].kernel_calls,16)
                self.assertEqual(stages['loop_map'].kernel_calls,16)
                self.assertEqual(stages['accumulate'].kernel_calls,32)
            else:
                self.assertEqual(profile.calls,[])
                self.assertEqual(estimate.temporary_bytes,0)

    def test_invalid_layouts_and_reserves(self):
        args = ('loop', ((4,4,7,7),(3,4,4,7,7),(5,3)), ('complex128',)*3)
        for kwargs in ({'partitions':(0,3,2,7)}, {'row_block':0}):
            with self.assertRaises(ValueError):
                MatrixKernelSpec(*args,**kwargs)
        with self.assertRaises(ValueError):
            MatrixKernelSpec('loop', ((4,4,7,7),(3,4,4,7,7),(5,2)), ('complex128',)*3)
        with self.assertRaises(TypeError):
            MatrixKernelSpec(args[0],args[1],('int32',)*3)
        with self.assertRaises(ValueError):
            estimate_matrix_kernel(MatrixKernelSpec(*args),profile=RecordingProfile(),workspace_reserve_bytes=-1)


if __name__ == '__main__':
    unittest.main()
