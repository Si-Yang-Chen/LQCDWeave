"""Complete parent complements against an independent full inverse."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

from production.perambulator import test_dilution as reference
from production.perambulator.test_dilution import Handle, CountingBackend
from lqcdweave.contracts.dilution import DilutionSource
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, ModePartition, HighmodeBasis, PropagatorEndpoint,
    VectorBasis, PointBasis,
    PropagatorBlock, project_highmode_block, propagator_pair_spec,
    PropagatorReader, PropagatorWriter, ALL_PRODUCTS, SAVED_PRODUCTS,
)
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader
from lqcdweave.production.perambulator.products import product_shape


class HighmodePartitionTests(unittest.TestCase):
    def setUp(self):
        self.f = reference.DilutionTests()
        self.f.setUp()
        rng = np.random.default_rng(914)
        parent = np.stack([np.linalg.qr(rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6)))[0].T
                           for _ in range(3)]).reshape(3, 6, 1, 1, 2, 3)
        self.parent = Handle(parent, Ne=6, basis_id='parent', mode_labels=tuple(range(6)),
                             basis_orthogonal=True, basis_orthonormal=True)
        self.partition = ModePartition('parent', (0, 1), ((2, 3, 4),))
        self.high = HighmodeBasis(self.parent, self.partition)
        noise = NoisevectorGenerator(self.parent, ((3,), (2,)), mode_offset=2).load('cfg')
        self.f.v, self.f.d = parent[:, :2], noise.calc(seed=331)
        self.source = DilutionSource.from_description(noise.describe(), data_id='actual-d')
        self.f.data.update(V=self.f.v, D=self.f.d)
        self.f.endpoints['V'] = PropagatorEndpoint('V', 'parent', (0, 1))
        self.f.endpoints['D'] = PropagatorEndpoint('D', 'actual-d', (0, 1), metadata=dict(
            vector_labels=[list(v) for v in self.source.vector_labels], mode_groups=[[2, 3, 4]],
            parent_basis_id='parent', realization_id='draw', noise_pool_id='per-time', normalization='raw',
            dilution_source=self.source.document))

    def generator(self, source_kind, sink_kind, **options):
        handles = {kind: Handle(self.f.data[kind], Ne=self.f.data[kind].shape[1],
                               endpoint=self.f.endpoints[kind],
                               dilution_source=self.source if kind == 'D' else None)
                   for kind in ('V', 'D')}
        self.src_basis=VectorBasis(handles[source_kind])
        self.point_src_basis=PointBasis(
            Handle(self.f.points,Np=2,endpoint=self.f.endpoints['P']),highmode_basis=self.high)
        self.sink_v_basis=VectorBasis(handles[sink_kind])
        self.sink_p_basis=PointBasis(
            Handle(self.f.points,Np=2,endpoint=self.f.endpoints['P']),highmode_basis=self.high)
        self.sink_bases=[self.sink_v_basis,self.sink_p_basis]
        return PerambulatorGenerator(self.f.lattice, Handle(self.f.matrix), 0, 1e-11, 200,
            backend=CountingBackend(), MRHS=True, rhs_batch_size=5, **options)

    def oracle(self, product, t, s):
        sink = self.f.embedding(product[0], self.f.data[product[0]], t)
        source = self.f.embedding(product[2], self.f.data[product[2]], s)
        if product[0] == 'P':
            u = self.f.embedding('V', self.parent.values[:, :5], t)
            sink = (np.eye(72) - u @ u.conj().T) @ sink
        if product[2] == 'P':
            u = self.f.embedding('V', self.parent.values[:, :5], s)
            source = (np.eye(72) - u @ u.conj().T) @ source
        value = sink.conj().T @ self.f.inverse @ source
        return value.reshape(4, sink.shape[1]//4, 4, source.shape[1]//4).transpose(0, 2, 1, 3)

    def test_nine_products_remove_all_parents_without_extra_medium_solves(self):
        for streamed in (False, True):
            for a in ('V', 'D'):
                for b in ('V', 'D'):
                    gen = self.generator(a, b, stream_eigenvectors=streamed,
                        sink_ne_block_size=1,
                        source_selection=TimeSelectionRecord(3, (2, 0)),
                        sink_selection=TimeSelectionRecord(3, (0, 2)))
                    gen.load('cfg')
                    for s in (2, 0):
                        vector_results=gen.calc(s,src=self.src_basis,sink=self.sink_bases)
                        point_results=gen.calc(s,src=self.point_src_basis,sink=self.sink_bases)
                        products=(b+'S'+a,'PS'+a,b+'SP','PSP')
                        for product, result in zip(products,(*vector_results,*point_results)):
                            expected = np.stack([self.oracle(product, t, s) for t in (0, 2)])
                            np.testing.assert_allclose(result.values.reshape(expected.shape), expected, atol=4e-14)
                    self.assertEqual(gen.backend.rhs_count, 2*(8+24))
                    self.assertEqual(point_results[0].source.metadata['high_mode_projection'],self.partition.document)
                    self.assertEqual(vector_results[1].sink.metadata['high_mode_projection'],self.partition.document)

    def test_block_delivery_keeps_complete_highmode_subtraction(self):
        gen = self.generator('V', 'D',
            sink_ne_block_size=1, sink_selection=TimeSelectionRecord(3, (2, 0)))
        gen.load('cfg')
        for source_basis, products in ((self.src_basis,('DSV','PSV')),
                                       (self.point_src_basis,('DSP','PSP'))):
            actual={p:np.empty((2,*product_shape(p,2,2)),complex) for p in products}
            gen.calc(2,src=source_basis,sink=self.sink_bases,
                write=lambda index,selection,value:actual[products[index]].__setitem__(selection,value))
            for product,value in actual.items():
                expected=np.stack([self.oracle(product,t,2) for t in (2,0)])
                np.testing.assert_allclose(value.reshape(expected.shape),expected,atol=4e-14)

    def test_streamed_high_parent_never_materializes_all_times(self):
        self.parent.load = lambda key: (_ for _ in ()).throw(AssertionError('eager parent read'))
        reads=[]
        original_read=self.parent.read_modes
        self.parent.read_modes=lambda key,time,start,stop:(reads.append(time),original_read(key,time,start,stop))[1]
        for source_kind, sink_kind in (('V','V'), ('D','D')):
            gen=self.generator(source_kind,sink_kind,
                stream_eigenvectors=True,source_selection=TimeSelectionRecord(3,(2,0)),
                sink_selection=TimeSelectionRecord(3,(0,2)),sink_ne_block_size=1)
            gen.load('cfg')
            for s in (2,0):
                vp,pp=gen.calc(s,src=self.point_src_basis,
                               sink=[self.sink_v_basis,self.sink_p_basis])
                for product,actual in ((sink_kind+'SP',vp),('PSP',pp)):
                    expected=np.stack([self.oracle(product,t,s) for t in (0,2)])
                    np.testing.assert_allclose(actual.values.reshape(expected.shape),expected,atol=4e-14)
        self.assertEqual(set(reads),{0,2})

    def test_derived_d_point_and_double_point_match_native_and_full_inverse(self):
        u = PropagatorEndpoint('V', 'parent', range(5))
        for product in ('PSV', 'VSP', 'PSD', 'DSP', 'PSP'):
            for t, s in ((0, 2), (2, 0)):
                block = PropagatorBlock(product, t, s, self.f.endpoints[product[0]],
                    self.f.endpoints[product[2]], self.f.oracle(product, t, s))
                sides = [side for side, kind in (('sink', product[0]), ('source', product[2])) if kind == 'P']
                overlaps, auxiliary = {}, {}
                for side in sides:
                    time = t if side == 'sink' else s
                    coords = self.f.points[time]
                    overlaps[side] = self.parent.values[time, :5, coords[:, 2], coords[:, 1], coords[:, 0]].transpose(1, 0, 2)
                    ap = 'VS'+product[2] if side == 'sink' else product[0]+'SV'
                    auxiliary[side] = PropagatorBlock(ap, t, s, u if side == 'sink' else block.sink,
                        u if side == 'source' else block.source,
                        self.f.oracle(ap, t, s, sink_data=self.parent.values[:, :5] if side == 'sink' else None,
                                      source_data=self.parent.values[:, :5] if side == 'source' else None))
                if len(sides) == 2:
                    auxiliary['cross'] = PropagatorBlock('VSV', t, s, u, u,
                        self.f.oracle('VSV', t, s, sink_data=self.parent.values[:, :5], source_data=self.parent.values[:, :5]))
                result = project_highmode_block(block, partition=self.partition, overlaps=overlaps, auxiliary=auxiliary)
                expected = self.oracle(product, t, s)
                np.testing.assert_allclose(result.values.reshape(expected.shape), expected, atol=5e-14)
                self.assertTrue(result.derived)
                bad = dict(auxiliary)
                first = sides[0]
                from dataclasses import replace
                bad[first] = replace(bad[first], source_time=1)
                with self.assertRaisesRegex(ValueError, 'physical time'):
                    project_highmode_block(block, partition=self.partition, overlaps=overlaps, auxiliary=bad)

    def test_rectangular_derivation_matches_full_space_complements(self):
        rng = np.random.default_rng(7382)
        for np_sink, np_source, ml, mr in ((2, 3, 4, 6), (3, 1, 5, 2)):
            for dtype in (np.complex64, np.complex128):
                with self.subTest(points=(np_sink, np_source), parents=(ml, mr), dtype=dtype):
                    def sample(shape):
                        return (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)

                    us = np.linalg.qr(sample((9, ml)))[0].astype(dtype)
                    ur = np.linalg.qr(sample((12, mr)))[0].astype(dtype)
                    points_s = (2, 0, 1)[:np_sink]
                    points_r = (3, 1, 0)[:np_source]
                    ids_s = [3*p+c for p in points_s for c in range(3)]
                    ids_r = [3*p+c for p in points_r for c in range(3)]
                    ps = np.eye(9, dtype=dtype)[:, ids_s]
                    pr = np.eye(12, dtype=dtype)[:, ids_r]
                    full = sample((36, 48))
                    spin = np.eye(4, dtype=dtype)
                    partitions = dict(sink=ModePartition('left-parent', range(ml), ()),
                                      source=ModePartition('right-parent', range(mr), ()))
                    parent_s = PropagatorEndpoint('V', 'left-parent', range(ml))
                    parent_r = PropagatorEndpoint('V', 'right-parent', range(mr))
                    ds = np.linalg.qr(sample((9, 2)))[0].astype(dtype)
                    dr = np.linalg.qr(sample((12, 3)))[0].astype(dtype)
                    ends_s = dict(P=PropagatorEndpoint('P', 'left-points', points_s),
                                  V=PropagatorEndpoint('V', 'left-parent', range(ml)),
                                  D=PropagatorEndpoint('D', 'left-noise', range(2)))
                    ends_r = dict(P=PropagatorEndpoint('P', 'right-points', points_r),
                                  V=PropagatorEndpoint('V', 'right-parent', range(mr)),
                                  D=PropagatorEndpoint('D', 'right-noise', range(3)))
                    bases_s, bases_r = dict(P=ps, V=us, D=ds), dict(P=pr, V=ur, D=dr)

                    def raw(a, b, left, right):
                        value = np.kron(spin, left).conj().T @ full @ np.kron(spin, right)
                        value = value.reshape(4, left.shape[1], 4, right.shape[1]).transpose(0, 2, 1, 3)
                        sink_tail = (len(a.indices), 3) if a.kind == 'P' else (len(a.indices),)
                        source_tail = (len(b.indices), 3) if b.kind == 'P' else (len(b.indices),)
                        value = value.reshape(4, 4, *sink_tail, *source_tail)
                        return PropagatorBlock(a.kind+'S'+b.kind, 2, 0, a, b, value)

                    for product in ('PSV', 'VSP', 'PSD', 'DSP', 'PSP'):
                        a, b = product[0], product[2]
                        block = raw(ends_s[a], ends_r[b], bases_s[a], bases_r[b])
                        before = block.values.copy()
                        overlaps, auxiliary = {}, {}
                        left, right = bases_s[a], bases_r[b]
                        if a == 'P':
                            overlaps['sink'] = us[ids_s].T.reshape(ml, np_sink, 3)
                            auxiliary['sink'] = raw(parent_s, ends_r[b], us, right)
                            left = (np.eye(9, dtype=dtype)-us@us.conj().T) @ left
                        if b == 'P':
                            overlaps['source'] = ur[ids_r].T.reshape(mr, np_source, 3)
                            auxiliary['source'] = raw(ends_s[a], parent_r, bases_s[a], ur)
                            right = (np.eye(12, dtype=dtype)-ur@ur.conj().T) @ right
                        if a == b == 'P':
                            auxiliary['cross'] = raw(parent_s, parent_r, us, ur)
                        expected = np.kron(spin, left).conj().T @ full @ np.kron(spin, right)
                        result = project_highmode_block(block, partition=partitions,
                            overlaps=overlaps, auxiliary=auxiliary)
                        actual = result.values.reshape(4, 4, left.shape[1], right.shape[1])
                        actual = actual.transpose(0, 2, 1, 3).reshape(expected.shape)
                        tolerance = 1e-5 if dtype == np.complex64 else 1e-12
                        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                        np.testing.assert_array_equal(block.values, before)
                        self.assertEqual(result.values.dtype, dtype)
                        self.assertFalse(np.shares_memory(result.values, block.values))

    def test_saved_six_restore_high_nine_and_reject_raw_point_alias(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stored = []
            for a, b, products in (('V', 'V', ('VSV', 'PSV', 'PSP')),
                                   ('D', 'V', ('VSD', 'PSD')), ('D', 'D', ('DSD',))):
                gen = self.generator(a, b)
                gen.load('cfg')
                eps = dict(V=gen.endpoint(self.sink_v_basis),
                           D=PropagatorEndpoint.from_document(self.f.endpoints['D'].document),
                           P=gen.endpoint(self.sink_p_basis))
                for product in products:
                    sink = eps[product[0]]
                    source = (gen.endpoint(self.point_src_basis,source=True)
                              if product[2]=='P' else gen.endpoint(self.src_basis,source=True))
                    spec = propagator_pair_spec(product, configuration=self.f.configuration, sink=sink, source=source,
                        time_pairs=((0, 2), (2, 0)), gauge_id='gauge', dirac=dict(action='test'),
                        gamma5=self.f.gamma, spin_basis='complex', gamma5_hermitian=True)
                    if 'P' in product:
                        self.assertIn('parent', spec.document['parents'])
                    path = Path(directory) / product
                    with DatasetWriter(path, spec, provenance={}) as writer:
                        output = PropagatorWriter(writer)
                        for s, t in ((2, 0), (0, 2)):
                            src_basis=self.point_src_basis if product[2]=='P' else self.src_basis
                            sink_basis=self.sink_p_basis if product[0]=='P' else self.sink_v_basis
                            result=gen.calc(s,src=src_basis,sink=[sink_basis])[0]
                            output.write(result.at(t))
                        writer.publish()
                    stored.append(stack.enter_context(DatasetReader(path)))
            reader = PropagatorReader(stored)
            endpoints = dict(self.f.endpoints)
            endpoints['P'] = self.high.point_endpoint(endpoints['P'])
            for product in ALL_PRODUCTS:
                value = reader.read(product, 0, 2, sink=endpoints[product[0]], source=endpoints[product[2]])
                expected = self.oracle(product, 0, 2)
                np.testing.assert_allclose(value.values.reshape(expected.shape), expected, atol=6e-14)
            self.assertEqual(reader.coverage('PSD', sink=self.f.endpoints['P'], source=endpoints['D']), ())

    def test_partition_and_basis_contracts_reject_invalid_inputs_before_solving(self):
        with self.assertRaises(ValueError):
            ModePartition('parent', (0, 1), ((1, 2),))
        with self.assertRaisesRegex(ValueError, 'complete'):
            HighmodeBasis(Handle(self.parent.values[:, :4], basis_id='parent', mode_labels=range(4),
                                 basis_orthonormal=True), self.partition)
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            HighmodeBasis(Handle(self.parent.values, basis_id='parent', mode_labels=range(6),
                                 basis_orthogonal=True), self.partition)
        gen = self.generator('D', 'V')
        bad_source=VectorBasis(Handle(self.f.d,Ne=self.f.d.shape[1],
            basis_id='wrong-data-id',dilution_source=self.source))
        self.src_basis=bad_source
        gen.load('cfg')
        with patch.object(gen.backend, 'solve_into', side_effect=AssertionError('numeric solve')):
            with self.assertRaisesRegex(ValueError,'D handle identity'):
                gen.calc(0,src=bad_source,sink=[self.sink_v_basis])
        gen = self.generator('V', 'V')
        gen.load('cfg')
        high_point=self.high.point_endpoint(self.f.endpoints['P'])
        raw_point=PointBasis(Handle(self.f.points,Np=2),endpoint=high_point)
        with self.assertRaisesRegex(ValueError, 'actual highmode_basis'):
            gen.calc(0,src=self.src_basis,sink=[raw_point])

    def test_imported_d_source_supplies_saved_endpoint_without_manual_metadata(self):
        gen = self.generator('D', 'V')
        source_handle=Handle(self.f.d,Ne=self.f.d.shape[1],endpoint=None,
            basis_id=self.source.data_id,mode_labels=self.source.ordered_indices,
            dilution_source=self.source)
        source_basis=VectorBasis(source_handle)
        gen.load('cfg')
        metadata = gen.endpoint(source_basis,source=True).metadata
        self.assertEqual(metadata['dilution_source'], self.source.document)
        self.assertEqual(metadata['noise_pool_id'], [list(pair) for pair in self.source.time_pools])

    def test_explicit_pair_write_highmode_roundtrip_against_full_inverse(self):
        gen = self.generator('D', 'V')
        gen.gauge_field.gauge_id = 'gauge'
        gen.load('cfg')
        sink_endpoint=gen.endpoint(self.sink_p_basis)
        source_endpoint=gen.endpoint(self.src_basis,source=True)
        spec = propagator_pair_spec('PSD', configuration=self.f.configuration, sink=sink_endpoint,
            source=source_endpoint, time_pairs=((0, 2), (2, 0), (1, 2)), gauge_id='gauge',
            dirac=dict(action='reference', mass=0, temporal_boundary='periodic', solver_tolerance=1e-11), gamma5=self.f.gamma,
            spin_basis='complex', gamma5_hermitian=True)
        with TemporaryDirectory() as directory:
            with DatasetWriter(directory, spec, provenance={}) as writer:
                output = PropagatorWriter(writer)
                for s, sink_times in ((2, (0, 1)), (0, (2,))):
                    result = gen.calc(s, src=self.src_basis, sink=[self.sink_p_basis])[0]
                    for t in sink_times:
                        output.write(result.at(t))
                writer.publish()
            with DatasetReader(directory) as stored:
                for t, s in spec.coordinates('time_pair'):
                    np.testing.assert_allclose(stored.read(time_pair=[t, s]),
                        self.oracle('PSD', t, s).reshape(spec.shape[1:]), atol=5e-14)


if __name__ == '__main__':
    unittest.main()
