"""Compact temporal point readers respect independent physical endpoint maps."""
from contextlib import ExitStack
import unittest
import numpy as np
import test_sparse_elemental_dataset as fixture
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.elemental import TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.production.elemental.planning import plan_indexed_point_elementals
from fixtures.elemental_script import produce_point_elemental
from production.elemental.test_elemental import reference_matrix, point_basis


class SparsePointConsumptionTests(unittest.TestCase):
    def test_temporal_coo_and_mixed_blocks_use_each_parent_point_map(self):
        f = fixture.SparseElementalDatasetTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        anchors = TimeSelectionRecord(6, (5, 1))
        basis_id = f.publish_basis('basis', TimeSelectionRecord(6, (2, 5, 0, 1)))
        left_id = f.publish_points('left', anchors)
        right_id = f.publish_points('right', TimeSelectionRecord(6, (2, 0)))
        plan = plan_indexed_point_elementals(f.index, f.gauge_id, basis_id, left_id,
            paths=((6, 0),), momentum_list=f.momenta, selection=anchors,
            points_right_id=right_id, point_right_labels=(9, 4),
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, phase_convention='midpoint')
        with ExitStack() as stack:
            gauge = stack.enter_context(f.index.open(f.gauge_id))
            basis = stack.enter_context(f.index.open(basis_id))
            left = stack.enter_context(f.index.open(left_id))
            right = stack.enter_context(f.index.open(right_id))
            readers = {}
            for sector in ('v2p', 'p2v', 'p2p'):
                destination = f.root/sector
                with DatasetWriter(destination, plan['stages'][sector]['spec'], provenance={}) as writer:
                    produce_point_elemental(gauge, basis if sector != 'p2p' else None,
                                               left, writer, points_right=right)
                    identity = writer.publish()
                f.index.register(destination)
                readers[sector] = stack.enter_context(f.index.open(identity))
            adapter = DatasetElemental(readers, point_dataset=left, point_right_dataset=right)
            self.assertEqual(adapter.Lt, 2)
            component = ElementalComponent.path((6, 0))
            for slot, physical in enumerate(anchors.times):
                endpoint = (physical+1)%6
                for mi, momentum in enumerate(f.momenta):
                    kernel = reference_matrix(f.gauge, (6, 0), physical, momentum)
                    for sector in readers:
                        lb = f.vectors[physical].reshape(2, -1).T if sector == 'v2p' else point_basis(f.points[physical], f.size)
                        rb = f.vectors[endpoint].reshape(2, -1).T if sector == 'p2v' else point_basis(f.points[endpoint, [2, 0]], f.size)
                        expected = lb.conj().T @ kernel @ rb
                        actual = adapter.block(component, mi, slot, sector)
                        if sector == 'p2p':
                            dense = np.zeros((3, 3, 2, 3), complex)
                            for (row, column), value in zip(actual['indices'], actual['values']):
                                dense[row, :, column, :] = value
                            actual = dense
                        np.testing.assert_allclose(np.asarray(actual).reshape(expected.shape), expected, atol=3e-12)

    def test_native_temporal_mixed_diagram_uses_endpoint_sampling_slots(self):
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.operators.single_hadron import Operator, OperatorTerm
        from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
        from lqcdweave.contraction.correlator import calculate
        rng = np.random.default_rng(430)
        def random(shape):
            return rng.normal(size=shape)+1j*rng.normal(size=shape)
        e = random((1, 1, 8, 2, 2, 3))
        p = {'vsv': random((8, 8, 4, 4, 2, 2)), 'psp': random((8, 8, 4, 4, 2, 3, 2, 3))}
        anchors = TimeSelectionRecord(8, (7, 0, 2))
        endpoints = TimeSelectionRecord(8, (7, 0, 2, 1, 3))
        component = ElementalComponent('temporal', (0, 1))
        operator = Operator.from_terms('mixed', [OperatorTerm(1, component, 0)])
        pool = SamplingPoolSpec('census', spatial_sites=2, sampled_points=2)
        def elemental(data, record=None, sampling=None):
            extent = 8 if sampling is None else sampling.Nt
            return ArrayElemental({'v2p': data}, [component], phase_convention='midpoint', cfg='cfg',
                selection=record, sampling_selection=sampling,
                sampling_groups={'census': pool}, sampling_pool_ids={
                    (component, 0, slot, side): 'census' for slot in range(extent) for side in ('bar', 'field')})
        full_e = elemental(e)
        sparse_e = elemental(e[:, :, anchors.times], anchors, endpoints)
        full_p = ArrayPropagator(p, time_convention='absolute', cfg='cfg')
        sparse_p = ArrayPropagator({key: value[list(endpoints.times)][:, list(endpoints.times)] for key, value in p.items()},
            time_convention='absolute', cfg='cfg', source_selection=endpoints, sink_selection=endpoints)
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['source', 'sink'], [None, 'q'])
        def run(reader, prop, times):
            return calculate(diagram, propagator_map={'q': prop}, vertex_map={
                'source': Meson(reader, operator, True, requested_sectors=('p2v',)),
                'sink': Meson(reader, operator, requested_sectors=('v2p',))},
                configuration='cfg', usedNe=2, usedNp=2, groups_independent=True,
                time_axes={0: times, 1: times}, optimize='none', memory_limit=32*2**20).value
        expected = run(full_e, full_p, anchors.times)
        actual = run(sparse_e, sparse_p, range(anchors.Nt))
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-10)

    def test_compact_coo_refuses_unmapped_records_and_incomplete_cfg_archive(self):
        import test_sparse_diagram_binding as sparse_fixture
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.sparse_elemental import SparseElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        record = TimeSelectionRecord(8, (7, 2))
        case = sparse_fixture.SparseDiagramBindingTests()._case({'type': 'identity'})
        original = case['dense_reader']
        dense = ArrayElemental(original.sectors, original.components, phase_convention='anchor',
            cfg='cfg-A', sampling_pool_ids=original.sampling_pool_ids, sampling_groups=original.sampling_groups,
            selection=record)
        with self.assertRaisesRegex(ValueError, 'explicitly ordered selection'):
            SparseElemental(dense, case['record_reader'])
        reader = SparseElemental(dense, case['record_reader'], selection=record)
        prop = ArrayPropagator(case['peram'], time_convention='absolute', cfg='cfg-A',
                              source_selection=record, sink_selection=record)
        vertices = {key: Meson(reader, vertex.operator, vertex.dagger, phase_convention='anchor')
                    for key, vertex in case['sparse_vertices'].items()}
        propagators = {key: prop for key in case['bindings_sparse'].propagator_map}
        planned = calculate(case['diagram'], vertex_map=vertices, propagator_map=propagators,
            configuration='cfg-A', usedNe=1, usedNp=case['np'], optimize='none', execute=False,
            memory_limit=32*2**20)
        self.assertIsNotNone(planned.run().value)
        with self.assertRaisesRegex(ValueError, 'described preparations'):
            complete_calculation(planned)
