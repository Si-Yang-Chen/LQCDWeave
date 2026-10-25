"""All nine sectors: real Wilson solves, six saved products, literal full trace."""

from contextlib import ExitStack
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

from tests.fixtures.laplace_reference import random_gauge
from test_wilson_pipeline import stencil
from production.elemental.test_elemental import reference_matrix
from lqcdweave.adapters.dense_wilson import DenseWilsonBackend
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.adapters.propagator import PropagatorPairReader
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS
from lqcdweave.contracts.spatial_sampling import SpatialPointSet, SpatialSamplingDesign, SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram, Propagator, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.planning.production import elemental_dataset_document, point_elemental_dataset_document
from lqcdweave.planning.spatial_points import validate_complete_spatial_point_set
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, ModePartition, HighmodeBasis, PropagatorEndpoint,
    VectorBasis, PointBasis,
    PropagatorBlock, propagator_pair_spec, PropagatorWriter, PropagatorReader,
)
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader, DatasetIndex
from lqcdweave.contraction.correlator import calculate
from lqcdweave.production.perambulator.pair_planning import plan_indexed_perambulator_pairs


class Input:
    def __init__(self, values, **metadata):
        self.values = values
        self.__dict__.update(metadata)

    def load(self, cfg):
        return self.values


class ThreeSectorClosedLoopTests(unittest.TestCase):
    lattice = (2, 1, 1, 2)
    cfg = 'wilson-three-sectors'

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='ed-three-sector-reference-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = DatasetIndex(self.root/'index')
        self.configuration = dict(ensemble='bounded-wilson', configuration=self.cfg, lattice=list(self.lattice))
        self.links = random_gauge(self.lattice, seed=432)
        rng = np.random.default_rng(932)
        self.parents = np.stack([np.linalg.qr(rng.normal(size=(6, 4)) + 1j*rng.normal(size=(6, 4)))[0].T
                                 for _ in range(2)]).reshape(2, 4, 1, 1, 2, 3)
        self.rotations = [np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))[0]
                          for _ in range(2)]
        shape = (2, 1, 1, 2, 4, 3)
        self.matrix = np.column_stack([stencil(self.links, np.eye(48)[:, i].reshape(shape), .4, -1).ravel()
                                       for i in range(48)])
        self.inverse = np.linalg.inv(self.matrix)
        self.gauge_spec = self.spec('gauge', ('time', 'z', 'y', 'x', 'direction', 'color_row', 'color_column'),
                                    (range(2), (0,), (0,), range(2), range(4), range(3), range(3)))
        self.gauge_id = self.publish('gauge', self.gauge_spec, self.links)
        self.parent_spec = self.spec('explicit_basis', ('time', 'mode', 'z', 'y', 'x', 'color'),
            (range(2), range(4), (0,), (0,), range(2), range(3)), conventions=dict(normalization='orthonormal'))
        self.parent_id = self.publish('parents', self.parent_spec, self.parents)
        self.partition = ModePartition(self.parent_id, (0,), ((1, 2, 3),))
        q01, q10 = Propagator('u', Tag(0, 0), Tag(1, 0)), Propagator('d', Tag(1, 0), Tag(0, 0))
        self.symbols = q01, q10
        self.diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ['early', 'late'], [None, q01, q10])

    def spec(self, kind, names, coordinates, *, conventions=None, definition=None, parents=()):
        return DatasetSpec(dict(schema=1, kind=kind, configuration=self.configuration,
            axes=[dict(name=n, coordinates=list(c)) for n, c in zip(names, coordinates)],
            dtype='complex128', bases={}, parents=list(parents), definition=definition or {},
            conventions=conventions or {}))

    def publish(self, name, spec, values, provenance=None):
        with DatasetWriter(self.root/name, spec, provenance=provenance or {}) as writer:
            for slot in writer.missing:
                writer.write(slot, np.asarray(values[slot], dtype=spec.document['dtype']))
            identity = writer.publish()
        self.index.register(self.root/name)
        return identity

    def expected(self, path, momentum):
        matrices = []
        for time in range(2):
            spatial = reference_matrix(self.links, path, time, momentum, phase='anchor').reshape(2, 3, 2, 3)
            # Full space uses site,spin,color ordering; no production coefficient helpers.
            matrices.append(np.einsum('ab,icjd->iacjbd', gamma(15), spatial).reshape(24, 24))
        return np.trace(matrices[0] @ self.inverse[:24, 24:] @ matrices[1] @ self.inverse[24:, :24])

    def evaluate(self, subsets, point_columns, *, weighted, path, momentum, name, check_pair_planner=False):
        directory = self.root/name
        coordinates = np.asarray([[[x, 0, 0] for x in cols] for cols in point_columns], dtype=np.int64)
        points = SpatialPointSet(self.lattice, (0, 1), coordinates,
            design=SpatialSamplingDesign(self.lattice, len(point_columns[0]), 'independent'),
            pool_ids=(name+'-points-0', name+'-points-1'))
        point_spec = validate_complete_spatial_point_set(points, configuration=self.configuration)
        point_id = self.publish(name+'/points', point_spec, coordinates)
        d = np.stack([np.einsum('i...,ia->a...', self.parents[t, 1:4], self.rotations[t][:, list(cols)])
                      for t, cols in enumerate(subsets)])
        count = len(subsets[0])
        # Each time's selected columns are stored in axis order. The independent
        # pools have different labels but the same valid n/m group contract.
        description = dict(algorithm='qr-dilution-v1', parent_basis_id=self.parent_id, mode_groups=((1, 2, 3),),
            parent_counts=(3,), vector_counts=(count,), vector_labels=tuple((0, i) for i in range(count)),
            branches=('exact' if count == 3 else 'qr',), covariance_factors=(count/3,), normalization='raw',
            complete=True, time_pools=((0, name+'-D-0'), (1, name+'-D-1')))
        d_spec = self.spec('explicit_basis', ('time', 'mode', 'z', 'y', 'x', 'color'),
            (range(2), range(count), (0,), (0,), range(2), range(3)),
            conventions=dict(representation='dilution', normalization='raw', basis_orthogonal=True),
            parents=(self.parent_id,))
        d_id = self.publish(name+'/D', d_spec, d, dict(generator=description))
        with ExitStack() as stack:
            parent = stack.enter_context(DatasetReader(self.root/'parents'))
            gauge = stack.enter_context(DatasetReader(self.root/'gauge'))
            d_stored = stack.enter_context(DatasetReader(directory/'D'))
            parent_handle = DatasetFieldHandle(parent)
            d_handle = DatasetFieldHandle(d_stored)
            d_source = d_handle.dilution_source
            self.assertEqual(d_source.parent_basis_id, self.parent_id)
            high = HighmodeBasis(parent_handle, self.partition)
            v_handle = Input(self.parents[:, :1], Ne=1, basis_id=self.parent_id, mode_labels=(0,),
                basis_orthogonal=True, endpoint=PropagatorEndpoint('V', self.parent_id, (0,)))
            d_handle.endpoint = PropagatorEndpoint('D', d_id, range(count), metadata=dict(
                vector_labels=[list(v) for v in d_source.vector_labels], mode_groups=[[1, 2, 3]],
                parent_basis_id=self.parent_id, realization_id=d_id, noise_pool_id=name+'-D',
                normalization='raw', dilution_source=d_source.document))
            point_handle = SpatialPointHandle.from_dataset(directory/'points', cfg=self.cfg)
            point_input = Input(coordinates, Np=coordinates.shape[1],
                endpoint=PropagatorEndpoint('P', point_id, range(coordinates.shape[1])))
            handles = dict(V=v_handle, D=d_handle)
            saved_pr = []
            for a, b, products in (('V', 'V', ('VSV', 'PSV', 'PSP')),
                                   ('D', 'V', ('VSD', 'PSD')), ('D', 'D', ('DSD',))):
                source_basis=VectorBasis(handles[a])
                sink_vector_basis=VectorBasis(handles[b])
                point_basis=PointBasis(point_input,highmode_basis=high)
                gen = PerambulatorGenerator(self.lattice, DatasetFieldHandle(gauge), .4, 1e-12, 100,
                    clover_coeff_r=0, t_boundary=-1,
                    backend=DenseWilsonBackend(),
                    MRHS=True, rhs_batch_size=12)
                gen.load(self.cfg)
                writers = {}
                for pr in products:
                    sink_basis=point_basis if pr[0]=='P' else sink_vector_basis
                    source_input=point_basis if pr[2]=='P' else source_basis
                    sink = gen.endpoint(sink_basis)
                    source = gen.endpoint(source_input,source=True)
                    spec = propagator_pair_spec(pr, configuration=self.configuration, sink=sink, source=source,
                        time_pairs=((0, 1), (1, 0)), gauge_id=self.gauge_id, dirac=dict(action='wilson', mass=.4,
                        temporal_boundary=-1, clover_coeff_r=0), gamma5=gamma(15), spin_basis='degrand_rossi',
                        gamma5_hermitian=True)
                    writer = stack.enter_context(DatasetWriter(directory/pr, spec, provenance={}))
                    writers[pr] = writer, PropagatorWriter(writer), sink_basis, source_input
                    if check_pair_planner:
                        plan = plan_indexed_perambulator_pairs(self.index, spec)
                        self.assertEqual(plan['missing_pairs'], ((0, 1), (1, 0)))
                        self.assertEqual(plan['action'], 'compute')
                for s, t in ((0, 1), (1, 0)):
                    arrays={}
                    vector_results=gen.calc(s,src=source_basis,
                        sink=[sink_vector_basis,point_basis])
                    arrays.update((result.product,result) for result in vector_results)
                    if 'PSP' in products:
                        arrays['PSP']=gen.calc(s,src=point_basis,sink=[point_basis])[0]
                    for pr, (writer, output, sink_basis, source_input) in writers.items():
                        output.write(arrays[pr].at(t))
                for pr, (writer, *_rest) in writers.items():
                    writer.publish()
                    self.index.register(directory/pr)
                    if check_pair_planner:
                        with patch.object(DatasetReader, 'read', side_effect=AssertionError('numeric planner read')):
                            plan = plan_indexed_perambulator_pairs(self.index, writer.spec)
                        self.assertEqual(plan['action'], 'reuse')
                    saved_pr.append(stack.enter_context(DatasetReader(directory/pr)))
            propagators = PropagatorPairReader(PropagatorReader(saved_pr))
            component = ElementalComponent.path(path)
            generator = ElementalGenerator(DatasetFieldHandle(gauge), None, (momentum,),
                [dict(component=component, path=path, coefficient=1)], vector_bases=dict(V=v_handle, D=d_handle),
                points=point_handle, blocks=ELEMENTAL_SECTORS, phase_convention='anchor',
                elemental_weighted=weighted, lattice_size=self.lattice).load(self.cfg)
            rows = []
            for t in range(2):
                blocks = generator.calc_path_blocks(t)
                rows.append({sector: (np.asarray([[value.values.copy() for value in row] for row in values])
                                       if sector == 'p2p' else values.copy()) for sector, values in blocks.items()})
            d_doc = d_stored.spec.document
            d_doc['conventions']['dilution_source'] = d_source.document
            d_binding_spec = DatasetSpec(d_doc)
            specifications = dict(v=parent.spec, d=d_binding_spec)
            ids = dict(v=self.parent_id, d=d_id)
            elemental_readers = {}
            for sector in ELEMENTAL_SECTORS:
                if 'p' not in sector:
                    left, right = sector[0], sector[-1]
                    modes = (0,) if left == 'v' else tuple(range(count))
                    right_modes = (0,) if right == 'v' else tuple(range(count))
                    document = elemental_dataset_document(gauge.spec, specifications[left], gauge_id=self.gauge_id,
                        basis_id=ids[left], product=sector, components=[dict(path=list(path))], momenta=(momentum,),
                        modes=modes, right_basis=specifications[right], right_basis_id=ids[right], right_modes=right_modes,
                        phase_convention='anchor', elemental_weighted=weighted)
                else:
                    kind = sector[0] if sector[-1] == 'p' else sector[-1]
                    vector = 'v' if kind == 'p' else kind
                    document = point_elemental_dataset_document(gauge.spec, None if sector == 'p2p' else specifications[vector], point_spec,
                        product=sector, gauge_id=self.gauge_id, basis_id=None if sector == 'p2p' else ids[vector], points_id=point_id,
                        components=[dict(path=list(path))], momenta=(momentum,),
                        modes=() if sector == 'p2p' else (0,) if vector == 'v' else tuple(range(count)), point_labels=range(coordinates.shape[1]),
                        phase_convention='anchor', elemental_weighted=weighted)
                self.publish(name+'/EL-'+sector, DatasetSpec(document), [row[sector] for row in rows])
                elemental_readers[sector] = stack.enter_context(DatasetReader(directory/('EL-'+sector)))
            pools = {identity: SamplingPoolSpec(identity, 2, coordinates.shape[1]) for identity in points.pool_ids}
            elemental = DatasetElemental(elemental_readers, point_dataset=stack.enter_context(DatasetReader(directory/'points')),
                sampling_groups=pools, sampling_pool_ids={(component, 0, t, side): points.pool_ids[t]
                    for t in range(2) for side in ('bar', 'field')})
            operator = Operator.from_terms('pseudoscalar-path', (OperatorTerm(15, component, 0),))
            vertices = {name: Meson(elemental, operator, requested_sectors=ELEMENTAL_SECTORS, phase_convention='anchor')
                        for name in ('early', 'late')}
            return calculate(self.diagram, vertex_map=vertices, propagator_map={symbol: propagators for symbol in self.symbols},
                configuration=self.cfg, time_axes={0: (0,), 1: (1,)}, elemental_weighted=weighted,
                groups_independent=True, optimize='none').value.item()

    def test_all_nine_general_path_dilution_subsets_average_to_full_wilson_trace(self):
        target = self.expected((0,), (1, 0, 0))
        means = []
        for weighted in (False, True):
            values = [self.evaluate((a, b), ((0, 1), (0, 1)), weighted=weighted, path=(0,), momentum=(1, 0, 0),
                name=f'general-{weighted}-{a}-{b}') for a, b in product(combinations(range(3), 2), repeat=2)]
            means.append(np.mean(values))
            np.testing.assert_allclose(means[-1], target, rtol=2e-11, atol=1e-13)
            print(f'three-sector general weighted={weighted}: mean={means[-1]}, full={target}')
        np.testing.assert_allclose(means[0], means[1], rtol=2e-12, atol=1e-13)

    def test_random_high_points_and_exact_medium_average_to_full_wilson_trace(self):
        target = self.expected((), (0, 0, 0))
        for weighted in (False, True):
            values = [self.evaluate(((0, 1, 2), (0, 1, 2)), ((a,), (b,)), weighted=weighted,
                path=(), momentum=(0, 0, 0), name=f'points-{weighted}-{a}-{b}') for a, b in product(range(2), repeat=2)]
            mean = np.mean(values)
            np.testing.assert_allclose(mean, target, rtol=2e-11, atol=1e-13)
            print(f'three-sector random high weighted={weighted}: mean={mean}, full={target}')

    def test_indexed_pair_planner_matches_the_same_nine_sector_sum(self):
        target = self.expected((0,), (1, 0, 0))
        actual = self.evaluate(((0, 1, 2), (0, 1, 2)), ((0, 1), (0, 1)), weighted=True,
            path=(0,), momentum=(1, 0, 0), name='pair-planner', check_pair_planner=True)
        np.testing.assert_allclose(actual, target, rtol=2e-11, atol=1e-13)
        with DatasetReader(self.root/'pair-planner'/'PSD') as stored:
            doc = stored.spec.document
        doc['axes'][0]['coordinates'].append([0, 0])
        plan = plan_indexed_perambulator_pairs(self.index, DatasetSpec(doc))
        self.assertEqual(plan['covered_pairs'], ((0, 1), (1, 0)))
        self.assertEqual(plan['missing_pairs'], ((0, 0),))
        # A different requested precision is a new production definition.
        doc['dtype'] = 'complex64'
        plan = plan_indexed_perambulator_pairs(self.index, DatasetSpec(doc))
        self.assertEqual(plan['covered_pairs'], ())
        self.assertEqual(plan['missing_pairs'], ((0, 1), (1, 0), (0, 0)))


if __name__ == '__main__':
    unittest.main()
