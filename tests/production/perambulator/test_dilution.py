"""Independent full-space oracle for V/P/D, Gamma5, and canonical persistence."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import h5py  # noqa: F401
except ImportError:
    HDF5_AVAILABLE = False
else:
    HDF5_AVAILABLE = True

from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.production.perambulator.api import (
    ALL_PRODUCTS, SAVED_PRODUCTS, PerambulatorGenerator, PropagatorEndpoint,
    VectorBasis, PointBasis,
    PropagatorBlock, gamma5_reverse, reverse_propagator, propagator_pair_spec,
    PropagatorWriter, PropagatorReader, source_time_blocks,
)
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader
from lqcdweave.contracts.time_selection import TimeSelectionRecord


class Handle:
    def __init__(self, values, **metadata):
        self.values = values
        self.__dict__.update(metadata)

    def load(self, key):
        return self.values

    def read_modes(self, key, time, start, stop):
        # Deliberately borrowed and overwritten by the next read.
        if not hasattr(self, 'scratch') or self.scratch.shape != self.values[time, start:stop].shape:
            self.scratch = np.empty_like(self.values[time, start:stop])
        self.scratch[:] = self.values[time, start:stop]
        return self.scratch


class CountingBackend(DenseDiracBackend):
    def __init__(self):
        super().__init__()
        self.rhs_count = 0

    def solve_into(self, rhs, solution, **options):
        self.rhs_count += len(rhs)
        return super().solve_into(rhs, solution, **options)


class DilutionTests(unittest.TestCase):
    def setUp(self):
        self.lattice = (2, 1, 1, 3)
        self.field_shape = (3, 1, 1, 2, 4, 3)
        rng = np.random.default_rng(20261011)
        unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))
        self.gamma = unitary @ np.diag([1, 1, -1, -1]) @ unitary.conj().T
        full_gamma = np.kron(np.eye(6), np.kron(self.gamma, np.eye(3)))
        random = rng.normal(size=(72, 72)) + 1j*rng.normal(size=(72, 72))
        hermitian = 8*np.eye(72) + .05*(random + random.conj().T)
        self.matrix = full_gamma @ hermitian
        self.inverse = np.linalg.inv(self.matrix)
        self.v = rng.normal(size=(3, 2, 1, 1, 2, 3)) + 1j*rng.normal(size=(3, 2, 1, 1, 2, 3))
        self.d = rng.normal(size=(3, 3, 1, 1, 2, 3)) + 1j*rng.normal(size=(3, 3, 1, 1, 2, 3))
        self.d *= np.array([.4, 1.3, 2.1])[None, :, None, None, None, None]
        self.points = np.array([[[0, 0, 0], [1, 0, 0]],
                                [[1, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [1, 0, 0]]])
        self.data = {'V': self.v, 'D': self.d, 'P': self.points}
        self.endpoints = {
            'V': PropagatorEndpoint('V', 'actual-v', [0, 1]),
            'P': PropagatorEndpoint('P', 'actual-p', [0, 1]),
            'D': PropagatorEndpoint('D', 'actual-d', [0, 1, 2], metadata=dict(
                vector_labels=[[0, 0], [1, 0], [1, 1]], mode_groups=[[2, 3], [4, 5, 6]],
                parent_basis_id='actual-parent', realization_id='realization-A',
                noise_pool_id='pool-A', normalization='raw')),
        }
        self.configuration = dict(ensemble='toy', configuration='cfg', lattice=list(self.lattice))

    def embedding(self, kind, data, time):
        count = data.shape[1]
        width = count*3 if kind == 'P' else count
        embedding = np.zeros((72, 4*width), complex)
        for spin in range(4):
            for index in range(width):
                for x in range(2):
                    for color in range(3):
                        row = np.ravel_multi_index((time, 0, 0, x, spin, color), self.field_shape)
                        if kind == 'P':
                            point, point_color = divmod(index, 3)
                            value = int(data[time, point, 0] == x and point_color == color)
                        else:
                            value = data[time, index, 0, 0, x, color]
                        embedding[row, spin*width+index] = value
        return embedding

    def oracle(self, product, sink_time, source_time, *, sink_data=None, source_data=None):
        a, b = product[0], product[2]
        sink_data = self.data[a] if sink_data is None else sink_data
        source_data = self.data[b] if source_data is None else source_data
        sink, source = self.embedding(a, sink_data, sink_time), self.embedding(b, source_data, source_time)
        value = sink.conj().T @ self.inverse @ source
        count_a = sink_data.shape[1]
        count_b = source_data.shape[1]
        tail_a = (count_a, 3) if a == 'P' else (count_a,)
        tail_b = (count_b, 3) if b == 'P' else (count_b,)
        return value.reshape(4, sink.shape[1]//4, 4, source.shape[1]//4).transpose(
            0, 2, 1, 3).reshape((4, 4) + tail_a + tail_b)

    def generator(self, source_kind, sink_kind, **options):
        source_handle = Handle(self.data[source_kind], Ne=self.data[source_kind].shape[1],
                               endpoint=self.endpoints[source_kind])
        vector_sink_handle = Handle(self.data[sink_kind], Ne=self.data[sink_kind].shape[1],
                                    endpoint=self.endpoints[sink_kind])
        point_source_handle = Handle(self.points, Np=2, endpoint=self.endpoints['P'])
        point_sink_handle = Handle(self.points, Np=2, endpoint=self.endpoints['P'])
        self.src_basis = VectorBasis(source_handle)
        self.point_src_basis = PointBasis(point_source_handle)
        self.sink_v_basis = VectorBasis(vector_sink_handle)
        self.sink_p_basis = PointBasis(point_sink_handle)
        self.sink_bases = [self.sink_v_basis, self.sink_p_basis]
        return PerambulatorGenerator(self.lattice, Handle(self.matrix), 0, 1e-11, 200,
            backend=CountingBackend(), MRHS=True, rhs_batch_size=5, **options)

    def test_all_nine_direct_products_and_shared_solves_match_full_inverse(self):
        for streamed in (False, True):
            for source_kind in ('V', 'D'):
                for sink_kind in ('V', 'D'):
                    gen = self.generator(source_kind, sink_kind,
                        stream_eigenvectors=streamed,
                        sink_ne_block_size=2, sink_time_block_size=2 if streamed else 1,
                        source_selection=TimeSelectionRecord(3, (2, 0)),
                        sink_selection=TimeSelectionRecord(3, (0, 2)))
                    gen.load('cfg')
                    for source_time in (2, 0):
                        results = gen.calc(source_time,src=self.src_basis,sink=self.sink_bases)
                        point_results = gen.calc(source_time,src=self.point_src_basis,sink=self.sink_bases)
                        products = (sink_kind+'S'+source_kind, 'PS'+source_kind,
                                    sink_kind+'SP', 'PSP')
                        for product, result in zip(products,
                                                   (results[0],results[1],point_results[0],point_results[1])):
                            expected = np.stack([self.oracle(product, t, source_time) for t in (0, 2)])
                            np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=3e-14)
                            self.assertEqual(result.product,product)
                    self.assertEqual(gen.backend.rhs_count, 2*(4*self.data[source_kind].shape[1]+24))

    def test_exact_dilution_is_the_same_numerical_input_and_raw_amplitude_is_preserved(self):
        noise = NoisevectorGenerator(Handle(self.v, Ne=2), ([1, 1], [1, 1]))
        noise.load('cfg')
        exact = noise.calc(seed=7)
        self.data['D'] = exact
        self.endpoints['D'] = PropagatorEndpoint('D', 'exact-d', [0, 1])
        v = self.generator('V', 'V')
        v_src, v_sinks = self.src_basis, self.sink_bases
        d = self.generator('D', 'D')
        d_src, d_sinks = self.src_basis, self.sink_bases
        v.load('cfg'); d.load('cfg')
        for time in (0, 2):
            vr = v.calc(time,src=v_src,sink=v_sinks)
            dr = d.calc(time,src=d_src,sink=d_sinks)
            for v_result, d_result in zip(vr, dr):
                np.testing.assert_array_equal(d_result.values, v_result.values)
        # The preceding all-nine oracle covers nonunit, nonorthogonal raw D.

    def test_independent_rectangular_d_realizations_and_endpoint_snapshot(self):
        other = self.d[:, :2]*np.exp(.3j) + .1*self.v
        sink_endpoint = PropagatorEndpoint('D', 'different-actual-d', [0, 1],
            metadata={**self.endpoints['D'].metadata, 'vector_labels': [[0, 0], [0, 1]],
                      'mode_groups': [[7, 8, 9]], 'realization_id': 'realization-B'})
        source_handle = Handle(self.d, Ne=3, endpoint=self.endpoints['D'])
        sink_handle = Handle(other, Ne=2, endpoint=sink_endpoint)
        source_basis = VectorBasis(source_handle,count=2)
        sink_basis = VectorBasis(sink_handle)
        gen = PerambulatorGenerator(self.lattice, Handle(self.matrix), 0, 1e-11, 200,
            backend=CountingBackend())
        gen.load('cfg')
        self.assertEqual(gen.endpoint(source_basis,source=True).metadata['vector_labels'], [[0, 0], [1, 0]])
        self.assertEqual(gen.endpoint(sink_basis).document, sink_endpoint.document)
        result = gen.calc(2,src=source_basis,sink=[sink_basis])[0]
        source_handle.endpoint = PropagatorEndpoint('D', 'replacement', [0, 1, 2])
        self.assertEqual(result.source.data_id,'actual-d')
        self.assertEqual(gen.endpoint(source_basis,source=True).data_id,'replacement')
        expected = np.stack([self.oracle('DSD', t, 2, sink_data=other, source_data=self.d[:, :2]) for t in range(3)])
        np.testing.assert_allclose(result.values, expected, atol=3e-14)
        # A new load must observe the new actual record, not retain the old one.
        gen.load('cfg')
        self.assertEqual(gen.endpoint(source_basis,source=True).data_id,'replacement')

    def test_point_basis_rejects_a_vector_endpoint(self):
        gen = self.generator('D', 'V')
        gen.load('cfg')
        malformed = PointBasis(Handle(self.points,Np=2,endpoint=self.endpoints['D']))
        with self.assertRaisesRegex(ValueError,'basis type differs'):
            gen.calc(0,src=self.src_basis,sink=[malformed])
        self.assertEqual(gen.backend.rhs_count,0)

    def test_gamma5_all_nine_in_complex_spin_basis_and_rectangular_endpoint_groups(self):
        for product in ALL_PRODUCTS:
            parent_product = product[2]+'S'+product[0]
            parent = PropagatorBlock(parent_product, 2, 0,
                self.endpoints[product[2]], self.endpoints[product[0]],
                self.oracle(parent_product, 2, 0), parent_id='actual-parent-propagator')
            forward = reverse_propagator(parent, self.gamma, gamma5_hermitian=True)
            np.testing.assert_allclose(forward.values, self.oracle(product, 0, 2), atol=4e-14)
            self.assertEqual((forward.sink_time, forward.source_time), (0, 2))
            self.assertEqual(forward.sink, parent.source)
            self.assertEqual(forward.parent_id, parent.parent_id)
            self.assertFalse(np.shares_memory(forward.values, parent.values))
            restored = reverse_propagator(forward, self.gamma, gamma5_hermitian=True)
            np.testing.assert_allclose(restored.values, parent.values, atol=4e-14)
            low_precision = gamma5_reverse(parent.values.astype('complex64'), parent_product, self.gamma)
            self.assertEqual(low_precision.dtype, np.dtype('complex64'))
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            reverse_propagator(parent, self.gamma, gamma5_hermitian=False)

    def spec(self, product, pairs):
        return propagator_pair_spec(product, configuration=self.configuration,
            sink=self.endpoints[product[0]], source=self.endpoints[product[2]],
            time_pairs=pairs, gauge_id='actual-gauge', dirac=dict(action='explicit-toy-matrix', mass=0),
            gamma5=self.gamma, spin_basis='complex-test-basis', gamma5_hermitian=True)

    def test_six_saved_products_read_nine_and_noncanonical_writes_swap_real_times(self):
        pairs = ((2, 0), (0, 2), (0, 0), (2, 2))
        for storage_format in ('npy', 'hdf5'):
            with self.subTest(storage_format=storage_format):
                if storage_format == 'hdf5' and not HDF5_AVAILABLE:
                    self.skipTest('h5py is not installed')
                with tempfile.TemporaryDirectory() as temporary:
                    readers = []
                    try:
                        for product in SAVED_PRODUCTS:
                            directory = Path(temporary)/product
                            with DatasetWriter(directory, self.spec(product, pairs),
                                               provenance={'fixture': 'dense'},
                                               format=storage_format) as writer:
                                output = PropagatorWriter(writer)
                                # Exercise every noncanonical input at the save boundary.
                                incoming = {'PSV': 'VSP', 'VSD': 'DSV', 'PSD': 'DSP'}.get(product, product)
                                for sink_time, source_time in pairs:
                                    t, s = ((source_time, sink_time) if incoming != product
                                            else (sink_time, source_time))
                                    output.write(PropagatorBlock(incoming, t, s,
                                        self.endpoints[incoming[0]], self.endpoints[incoming[2]],
                                        self.oracle(incoming, t, s)))
                                writer.publish()
                            readers.append(DatasetReader(directory, cfg='cfg'))
                        access = PropagatorReader(readers)
                        extension = '.h5' if storage_format == 'hdf5' else '.npy'
                        files = sorted(path.relative_to(temporary).as_posix()
                                       for path in Path(temporary).glob(f'*/cfg{extension}'))
                        self.assertEqual(files,
                                         sorted(f'{product}/cfg{extension}' for product in SAVED_PRODUCTS))
                        other_extension = '.npy' if storage_format == 'hdf5' else '.h5'
                        self.assertEqual(list(Path(temporary).glob(f'*/cfg{other_extension}')), [])
                        for product in ALL_PRODUCTS:
                            for t, s in pairs:
                                block = access.read(product, t, s, sink=self.endpoints[product[0]],
                                                    source=self.endpoints[product[2]])
                                np.testing.assert_allclose(block.values, self.oracle(product, t, s), atol=4e-14)
                                self.assertEqual(block.derived, product not in SAVED_PRODUCTS)
                                self.assertIsNotNone(block.parent_id)
                        altered = PropagatorEndpoint('D', 'actual-d', [0, 1, 2],
                            metadata={**self.endpoints['D'].metadata, 'noise_pool_id': 'different-pool'})
                        with self.assertRaises(KeyError):
                            access.read('VSD', 0, 2, sink=self.endpoints['V'], source=altered)
                        with self.assertRaises(KeyError):
                            access.read('VSP', 1, 0, sink=self.endpoints['V'], source=self.endpoints['P'])
                    finally:
                        for reader in readers:
                            reader.close()

    def test_resume_and_rejections_preserve_coordinate_coverage(self):
        pairs = ((2, 0), (0, 2))
        product = 'PSD'
        block = PropagatorBlock(product, 2, 0, self.endpoints['P'], self.endpoints['D'],
                                self.oracle(product, 2, 0))
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.spec(product, pairs)
            with DatasetWriter(temporary, spec, provenance={'test': 1}) as writer:
                output = PropagatorWriter(writer)
                with self.assertRaises(KeyError):
                    output.write(replace(block, sink_time=1))
                with self.assertRaises(ValueError):
                    output.write(replace(block, source=self.endpoints['V']))
                self.assertEqual(writer.missing, (0, 1))
                output.write(block)
            with DatasetWriter(temporary, spec, provenance={'test': 1}, resume=True) as writer:
                self.assertEqual(writer.missing, (1,))
                output = PropagatorWriter(writer)
                with self.assertRaises(FileExistsError):
                    output.write(block)
                output.write(replace(block, sink_time=0, source_time=2, values=self.oracle(product, 0, 2)))
                writer.publish()
            with DatasetReader(temporary, cfg='cfg') as reader:
                direct = PropagatorReader([reader]).read(product, 0, 2,
                    sink=self.endpoints['P'], source=self.endpoints['D'])
                np.testing.assert_allclose(direct.values, self.oracle(product, 0, 2))
            with self.assertRaises(ValueError):
                propagator_pair_spec('VSP', configuration=self.configuration,
                    sink=self.endpoints['V'], source=self.endpoints['P'], time_pairs=pairs,
                    gauge_id='g', dirac={'action': 'toy'}, gamma5=self.gamma,
                    spin_basis='test', gamma5_hermitian=True)

    def test_self_products_triangle_restores_all_physical_time_pairs(self):
        triangle = [(t, s) for s in range(3) for t in range(s, 3)]
        full_pairs = [(t, s) for s in range(3) for t in range(3)]
        for product in ('VSV', 'DSD', 'PSP'):
            with self.subTest(product=product), tempfile.TemporaryDirectory() as directory:
                with DatasetWriter(directory, self.spec(product, triangle),
                                   provenance={'fixture': 'triangle'}) as writer:
                    output = PropagatorWriter(writer)
                    for t, s in triangle:
                        output.write(PropagatorBlock(product, t, s,
                            self.endpoints[product[0]], self.endpoints[product[2]],
                            self.oracle(product, t, s)))
                    writer.publish()
                with DatasetReader(directory, cfg='cfg') as reader:
                    access = PropagatorReader([reader])
                    kwargs = dict(sink=self.endpoints[product[0]],
                                  source=self.endpoints[product[2]])
                    self.assertEqual(set(access.coverage(product, **kwargs)), set(full_pairs))
                    self.assertEqual(access.missing(product, full_pairs, **kwargs), ())
                    for t, s in full_pairs:
                        block = access.read(product, t, s, **kwargs)
                        np.testing.assert_allclose(block.values, self.oracle(product, t, s),
                                                   atol=4e-14)
                        self.assertEqual(block.derived, t < s)
                        self.assertEqual((block.sink_time, block.source_time), (t, s))
                    different = PropagatorEndpoint(product[0], 'different-input',
                                                   self.endpoints[product[0]].indices)
                    self.assertEqual(access.missing(product, [(0, 2)],
                        sink=different, source=kwargs['source']), ((0, 2),))
                    with self.assertRaises(KeyError):
                        access.read(product, 0, 2, sink=different, source=kwargs['source'])

    def test_reader_disjoint_pair_shards_and_action_ambiguity(self):
        class PairReader:
            def __init__(reader, spec, values, data_id):
                reader.spec, reader.values, reader.data_id = spec, values, data_id

            def read(reader, *, time_pair):
                return reader.values

        a = PairReader(self.spec('VSD', ((2, 0),)), self.oracle('VSD', 2, 0), 'pair-A')
        b = PairReader(self.spec('VSD', ((0, 2),)), self.oracle('VSD', 0, 2), 'pair-B')
        access = PropagatorReader([a, b])
        self.assertEqual(access.coverage('VSD', sink=self.endpoints['V'],
            source=self.endpoints['D']), ((2, 0), (0, 2)))
        self.assertEqual(access.coverage('DSV', sink=self.endpoints['D'],
            source=self.endpoints['V']), ((0, 2), (2, 0)))
        self.assertEqual(access.missing('DSV', ((2, 0), (1, 2), (0, 2)),
            sink=self.endpoints['D'], source=self.endpoints['V']), ((1, 2),))
        other = PropagatorEndpoint('D', 'different-actual-d',
            self.endpoints['D'].indices, metadata=self.endpoints['D'].metadata)
        self.assertEqual(access.missing('VSD', ((0, 2),),
            sink=self.endpoints['V'], source=other), ((0, 2),))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            access.missing('VSD', ((2, 0), (2, 0)),
                sink=self.endpoints['V'], source=self.endpoints['D'])
        with self.assertRaisesRegex(ValueError, 'physical'):
            access.missing('VSD', ((3, 0),),
                sink=self.endpoints['V'], source=self.endpoints['D'])
        for t, s, parent in ((0, 2, 'pair-A'), (2, 0, 'pair-B')):
            block = access.read('DSV', t, s, sink=self.endpoints['D'], source=self.endpoints['V'])
            np.testing.assert_allclose(block.values, self.oracle('DSV', t, s), atol=4e-14)
            self.assertEqual(block.parent_id, parent)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            PropagatorReader([a, a])
        from lqcdweave.contracts.dataset import DatasetSpec
        incompatible = b.spec.document
        incompatible['definition']['dirac']['mass'] = 1
        different_action = PairReader(DatasetSpec(incompatible), b.values, 'other-action')
        with self.assertRaisesRegex(ValueError, 'action'):
            PropagatorReader([a, different_action])

    def test_complex64_reverse_storage_preserves_declared_precision(self):
        spec = self.spec('PSD', ((0, 2),)).document
        spec['dtype'] = 'complex64'
        from lqcdweave.contracts.dataset import DatasetSpec
        with tempfile.TemporaryDirectory() as temporary:
            with DatasetWriter(temporary, DatasetSpec(spec), provenance={}) as writer:
                PropagatorWriter(writer).write(PropagatorBlock('DSP', 2, 0,
                    self.endpoints['D'], self.endpoints['P'], self.oracle('DSP', 2, 0).astype('complex64')))
                writer.publish()
            with DatasetReader(temporary, cfg='cfg') as reader:
                block = PropagatorReader([reader]).read('DSP', 2, 0,
                    sink=self.endpoints['D'], source=self.endpoints['P'])
                self.assertEqual(block.values.dtype, np.dtype('complex64'))
                np.testing.assert_allclose(block.values, self.oracle('DSP', 2, 0), rtol=2e-6, atol=2e-6)

    def test_borrowed_row_blocks_retain_order_without_reinterpreting_dilution_labels(self):
        row = np.stack([self.oracle('VSD', t, 2) for t in (2, 0)])
        blocks = list(source_time_blocks('VSD', row, 2, (2, 0),
            sink=self.endpoints['V'], source=self.endpoints['D']))
        self.assertEqual([block.sink_time for block in blocks], [2, 0])
        self.assertTrue(np.shares_memory(row, blocks[0].values))
        self.assertEqual(blocks[0].source.metadata['vector_labels'], [[0, 0], [1, 0], [1, 1]])


if __name__ == '__main__':
    unittest.main()
