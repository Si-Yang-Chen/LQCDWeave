"""Direct nine-product solves versus Gamma5 reversal, including Wilson high P."""

import unittest

import numpy as np

from production.perambulator import test_dilution as raw_reference
from production.perambulator import test_highmode_partition as high_reference
from production.perambulator.test_dilution import Handle, CountingBackend
from tests.fixtures.laplace_reference import random_gauge
from test_wilson_pipeline import stencil
from lqcdweave.adapters.dense_wilson import DenseWilsonBackend
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.production.perambulator.api import (
    ALL_PRODUCTS, PerambulatorGenerator,
    VectorBasis, PointBasis,
)


class Gamma5DirectProductionTests(unittest.TestCase):
    def _verify(self, *, wilson, high):
        # The two fixtures provide independent full-space embeddings, unequal
        # endpoint widths, complex raw D amplitudes and actual QR high sources.
        fixture = high_reference.HighmodePartitionTests() if high else raw_reference.DilutionTests()
        fixture.setUp()
        f = fixture.f if high else fixture
        if wilson:
            links = random_gauge(f.lattice, seed=817)
            f.gamma = gamma(15)
            dimension = int(np.prod(f.field_shape))
            unit = np.eye(dimension)
            matrix = np.column_stack([
                stencil(links, unit[:, column].reshape(f.field_shape), .4, -1).ravel()
                for column in range(dimension)
            ])
            f.inverse = np.linalg.inv(matrix)
        oracle = fixture.oracle
        errors = {product: [0.0, 0.0] for product in ALL_PRODUCTS}
        comparisons = 0
        for streamed in (False, True):
            direct = {}
            owners = {}
            for source_kind in ('V', 'D'):
                for sink_kind in ('V', 'D'):
                    handles = {kind: Handle(f.data[kind], Ne=f.data[kind].shape[1],
                        endpoint=f.endpoints[kind],
                        dilution_source=fixture.source if high and kind == 'D' else None)
                        for kind in ('V', 'D')}
                    source_basis=VectorBasis(handles[source_kind])
                    sink_vector_basis=VectorBasis(handles[sink_kind])
                    point_source_basis=PointBasis(
                        Handle(f.points,Np=2,endpoint=f.endpoints['P']),
                        highmode_basis=fixture.high if high else None)
                    point_sink_basis=PointBasis(
                        Handle(f.points,Np=2,endpoint=f.endpoints['P']),
                        highmode_basis=fixture.high if high else None)
                    generator = PerambulatorGenerator(f.lattice,
                        Handle(links if wilson else f.matrix), .4 if wilson else 0,
                        1e-12, 200, clover_coeff_r=0, t_boundary=-1 if wilson else 1,
                        backend=DenseWilsonBackend() if wilson else CountingBackend(),
                        MRHS=True, rhs_batch_size=5, stream_eigenvectors=streamed,
                        gamma5=f.gamma,gamma5_hermitian=True,
                        sink_ne_block_size=1,
                        source_selection=TimeSelectionRecord(3, (2, 0, 1)),
                        sink_selection=TimeSelectionRecord(3, (1, 2, 0)))
                    generator.load('cfg')
                    for source_time in (2, 0, 1):
                        for source, products in ((source_basis,(sink_kind+'S'+source_kind,'PS'+source_kind)),
                                                 (point_source_basis,(sink_kind+'SP','PSP'))):
                            values_list=generator.calc(source_time,src=source,
                                sink=[sink_vector_basis,point_sink_basis])
                            for product, result in zip(products,values_list):
                                for slot, sink_time in enumerate((1, 2, 0)):
                                    block=result.at(sink_time)
                                    expected = oracle(product, sink_time, source_time).reshape(result.values[slot].shape)
                                    np.testing.assert_allclose(result.values[slot], expected, rtol=3e-12, atol=2e-13)
                                    key = product, sink_time, source_time
                                    if key in direct:
                                        np.testing.assert_allclose(result.values[slot], direct[key].values,
                                                                   rtol=3e-12, atol=2e-13)
                                    direct[key] = block
                                    owners[key] = result
            self.assertEqual({key[0] for key in direct}, set(ALL_PRODUCTS))
            for (product, sink_time, source_time), original in direct.items():
                with self.subTest(wilson=wilson, high=high, streamed=streamed,
                                  product=product,
                                  sink=sink_time, source=source_time):
                    reverse = owners[(product,sink_time,source_time)].reverse(sink_time)
                    opposite = direct[product[2]+'S'+product[0], source_time, sink_time]
                    self.assertEqual((reverse.product, reverse.sink_time, reverse.source_time),
                                     (opposite.product, opposite.sink_time, opposite.source_time))
                    self.assertEqual(reverse.sink, opposite.sink)
                    self.assertEqual(reverse.source, opposite.source)
                    np.testing.assert_allclose(reverse.values, opposite.values,
                                               rtol=3e-12, atol=2e-13)
                    # Independent ordinary-matrix dagger checks every spin,
                    # mode, point and color permutation without the PR helper.
                    left = len(original.sink.indices)*(3 if product[0] == 'P' else 1)
                    right = len(original.source.indices)*(3 if product[2] == 'P' else 1)
                    matrix = original.values.reshape(4, 4, left, right).transpose(0, 2, 1, 3)
                    matrix = matrix.reshape(4*left, 4*right)
                    literal = (np.kron(f.gamma, np.eye(right)) @ matrix.conj().T
                               @ np.kron(f.gamma, np.eye(left)))
                    literal = literal.reshape(4, right, 4, left).transpose(0, 2, 1, 3)
                    np.testing.assert_allclose(reverse.values, literal.reshape(reverse.values.shape),
                                               rtol=3e-12, atol=2e-13)
                    difference = reverse.values - opposite.values
                    errors[product][0] = max(errors[product][0], float(np.max(np.abs(difference))))
                    errors[product][1] = max(errors[product][1],
                        float(np.linalg.norm(difference)/np.linalg.norm(opposite.values)))
                    comparisons += 1
        self.assertEqual(comparisons, 2*9*9)
        label = 'Wilson-high' if high else 'Wilson-raw' if wilson else 'complex-spin-raw'
        for product, (absolute, relative) in errors.items():
            print(f'Gamma5 direct {label} {product}: max_abs={absolute:.6e}, max_rel={relative:.6e}')

    def test_raw_nine_products_in_complex_spin_basis(self):
        self._verify(wilson=False, high=False)

    def test_raw_nine_wilson_products(self):
        self._verify(wilson=True, high=False)

    def test_high_nine_wilson_products(self):
        self._verify(wilson=True, high=True)


if __name__ == '__main__':
    unittest.main()
