"""Real published point products consumed by the public mixed Diagram API."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import test_indexed_point_production as point_fixture

from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import CurrentTerm, LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorPath, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval
from fixtures.c_group_production import publish_elemental_products, publish_perambulators


class PublishedPointDiagramTests(unittest.TestCase):
    def setUp(self):
        self.fixture = point_fixture.IndexedPointProductionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.basis_id = f.basis()
        self.points_id, _ = f.points('diagram-points', labels=(9, 3))
        self.elemental_ids = publish_elemental_products(
            f.index, f.root, f.gauge_id, self.basis_id, self.points_id,
            output_name='point-elementals', paths=((), (0,)),
            momentum_list=((0, 0, 0), (1, 0, 0)),
            sectors=('v2v', 'v2p', 'p2v', 'p2p'), usedNe=2, usedNp=2)
        self.propagator_ids = publish_perambulators(
            f.index, f.root, f.gauge_id, output_name='diagram-propagators',
            basis_id=self.basis_id, points_id=self.points_id,
            products=('VSV', 'PSV', 'VSP', 'PSP'), Ne=2, Np=2)

    def assert_trace(self, *, usedNe, usedNp, source=False):
        f = self.fixture
        with ExitStack() as stack:
            elemental_readers = {sector: stack.enter_context(f.index.open(identity))
                                 for sector, identity in self.elemental_ids.items()}
            propagator_readers = {sector: stack.enter_context(f.index.open(identity))
                                  for sector, identity in self.propagator_ids.items()}
            points = stack.enter_context(f.index.open(self.points_id))
            current = LocalVectorCurrent().operator(3)
            if source:
                current = OperatorPath('source_path_current',
                    ((1, GaugePath((0,)), 1, .6 + .4j),),
                    current_terms=(CurrentTerm(.6 + .4j, 0, (1, 0, 0, 0), 1),),
                    current_term_ranges=((0,),))
            component = current.terms[0].component
            momentum = current.terms[0].momentum
            normal_component = ElementalComponent.path(())
            normal = Operator.from_terms('normal', [OperatorTerm(0, normal_component, 0, 1)])
            group_id = 'explicit-uniform-ordered-prefix'
            elemental = DatasetElemental(elemental_readers, point_dataset=points,
                sampling_pool_ids={(component, momentum, t, side): group_id
                                   for t in range(3) for side in ('bar', 'field')},
                sampling_groups={group_id: SamplingPoolSpec(group_id, 2, usedNp)})
            propagator = DatasetPropagator(propagator_readers)
            q = Propagator('u', Tag(0, 0), Tag(1, 1))
            vertex_map = {'current': Meson(elemental, current, source=source, phase_convention='anchor'),
                          'normal': Meson(elemental, normal, phase_convention='anchor')}
            count = usedNp * 3
            point_weight = 2 / usedNp

            def matrix(value, left, right):
                return value.reshape(4, 4, left, right).transpose(0, 2, 1, 3).reshape(
                    4 * left, 4 * right)

            with patch.object(elemental, 'describe_selected_sparse_record',
                              wraps=elemental.describe_selected_sparse_record) as selected:
                for t in range(3):
                    vv = elemental.block(component, momentum, t, 'v2v', usedNe=usedNe)
                    vp = elemental.block(component, momentum, t, 'v2p',
                                         usedNe=usedNe, usedNp=usedNp).reshape(usedNe, count)
                    pv = elemental.block(component, momentum, t, 'p2v',
                                         usedNe=usedNe, usedNp=usedNp).reshape(count, usedNe)
                    record = elemental.block(component, momentum, t, 'p2p', usedNp=usedNp)
                    pp = np.zeros((usedNp, 3, usedNp, 3), complex)
                    for (left, right), value in zip(record['indices'], record['values'], strict=True):
                        pp[left, :, right, :] += value
                    # Local P/P has only one independent point label. Its
                    # W1 factor remains estimable for a one-point prefix;
                    # the general path uses the admitted full sample here.
                    current_space = np.block([[vv, point_weight * vp],
                        [point_weight * pv, point_weight * pp.reshape(count, count)]])
                    current_spin = gamma(current.terms[0].gamma)
                    coefficient = current.terms[0].coefficient
                    if source:
                        self.assertGreater(np.linalg.norm(current_space - current_space.conj().T), 1e-5)
                        current_space = current_space.conj().T
                        current_spin = gamma(8) @ current_spin.conj().T @ gamma(8)
                        coefficient = coefficient.conjugate()
                    for s in range(3):
                        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [t, s],
                                          ['current', 'normal'], [None, q])
                        actual = calc_diagram_eval(diagram, propagator_map={q: propagator},
                            vertex_map=vertex_map, configuration='cfg', usedNe=usedNe,
                            usedNp=usedNp, optimize='none').value
                        # The current field at t is the source of S(s,t), while
                        # its bar receives S(t,s). Keep this directed ordering.
                        outgoing = np.concatenate((propagator.block(s, t, 'vsv', usedNe=usedNe),
                            propagator.block(s, t, 'vsp', usedNe=usedNe, usedNp=usedNp).reshape(
                                4, 4, usedNe, count)), axis=3)
                        incoming = np.concatenate((propagator.block(t, s, 'vsv', usedNe=usedNe),
                            propagator.block(t, s, 'psv', usedNe=usedNe, usedNp=usedNp).reshape(
                                4, 4, count, usedNe)), axis=2)
                        expected = coefficient * np.trace(matrix(outgoing, usedNe, usedNe + count)
                            @ np.kron(current_spin, current_space)
                            @ matrix(incoming, usedNe + count, usedNe)
                            @ np.kron(gamma(0), elemental.block(
                                normal_component, 0, s, 'v2v', usedNe=usedNe)))
                        with self.subTest(sink=t, source=s, Ne=usedNe, Np=usedNp):
                            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
                self.assertGreater(selected.call_count, 0)

    def test_all_published_sectors_against_explicit_full_sample_trace(self):
        self.assert_trace(usedNe=2, usedNp=2)

    def test_smaller_mode_prefix_against_explicit_trace(self):
        self.assert_trace(usedNe=1, usedNp=2)

    def test_source_dagger_of_complex_path_current_against_independent_trace(self):
        self.assert_trace(usedNe=2, usedNp=2, source=True)

    def test_one_point_sample_estimates_declared_diagonal_current(self):
        self.assert_trace(usedNe=1, usedNp=1)

    def test_one_point_sample_cannot_estimate_general_path_current(self):
        with self.assertRaisesRegex(ValueError, 'inclusion probability is zero'):
            self.assert_trace(usedNe=1, usedNp=1, source=True)

    def test_cropped_sparse_factor_metadata_matches_its_actual_record(self):
        f = self.fixture
        with ExitStack() as stack:
            readers = {sector: stack.enter_context(f.index.open(identity))
                       for sector, identity in self.elemental_ids.items()}
            points = stack.enter_context(f.index.open(self.points_id))
            current = LocalVectorCurrent().operator(3)
            component = current.terms[0].component
            group_id = 'explicit-one-point-sample'
            elemental = DatasetElemental(readers, point_dataset=points,
                sampling_pool_ids={(component, 0, t, side): group_id
                                   for t in range(3) for side in ('bar', 'field')},
                sampling_groups={group_id: SamplingPoolSpec(group_id, 2, 1)})
            meson = Meson(elemental, current, phase_convention='anchor')
            for t in range(3):
                description, = meson.describe_factors(t, 'p2p', usedNe=1, usedNp=1)
                metadata = meson.describe_sparse_record(description)
                raw = meson.read_factor_block(description)
                self.assertEqual(metadata.point_shape, (1, 1))
                self.assertEqual(metadata.nnz, len(raw['indices']))
                self.assertEqual(description.tensor.nnz_upper, metadata.nnz)
                self.assertEqual(description.tensor.selection_identity, metadata.selection_identity)
                self.assertTrue(np.all(raw['indices'] < 1))

    def test_different_actual_point_identity_is_refused_before_numeric_reads(self):
        f = self.fixture
        wrong_points_id, _ = f.points('same-shape-different-point-identity', labels=(4, 8))
        wrong_ids = publish_perambulators(
            f.index, f.root, f.gauge_id, output_name='different-point-propagators',
            basis_id=self.basis_id, points_id=wrong_points_id,
            products=('VSV', 'PSV', 'VSP', 'PSP'), Ne=2, Np=2)
        with ExitStack() as stack:
            elemental_readers = {sector: stack.enter_context(f.index.open(identity))
                                 for sector, identity in self.elemental_ids.items()}
            propagator_readers = {sector: stack.enter_context(f.index.open(identity))
                                  for sector, identity in wrong_ids.items()}
            points = stack.enter_context(f.index.open(self.points_id))
            current = LocalVectorCurrent().operator(3)
            component = current.terms[0].component
            normal = Operator.from_terms('normal', [OperatorTerm(0, component, 0, 1)])
            group_id = 'explicit-full-point-sample'
            elemental = DatasetElemental(elemental_readers, point_dataset=points,
                sampling_pool_ids={(component, 0, t, side): group_id
                                   for t in range(3) for side in ('bar', 'field')},
                sampling_groups={group_id: SamplingPoolSpec(group_id, 2, 2)})
            propagator = DatasetPropagator(propagator_readers)
            q = Propagator('u', Tag(0, 0), Tag(1, 1))
            diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                              ['current', 'normal'], [None, q])
            with patch.object(elemental, 'block', side_effect=AssertionError('numeric elemental read')), \
                 patch.object(propagator, 'block', side_effect=AssertionError('numeric propagator read')):
                with self.assertRaisesRegex(ValueError, 'endpoint'):
                    calc_diagram_eval(diagram, propagator_map={q: propagator},
                        vertex_map={'current': Meson(elemental, current, phase_convention='anchor'),
                                    'normal': Meson(elemental, normal, phase_convention='anchor')},
                        configuration='cfg', usedNe=2, usedNp=2,
                        optimize='none', execute=False)

    def test_published_vv_named_axes_refuse_a_different_actual_mode_basis(self):
        f = self.fixture
        wrong_basis_id = f.basis('same-shape-different-mode-identity', modes=(4, 7))
        wrong_ids = publish_perambulators(
            f.index, f.root, f.gauge_id, output_name='different-mode-propagator',
            basis_id=wrong_basis_id, products=('VSV',), Ne=2)
        with f.index.open(self.elemental_ids['v2v']) as elemental_reader, \
             f.index.open(wrong_ids['vsv']) as propagator_reader:
            elemental = DatasetElemental(elemental_reader)
            propagator = DatasetPropagator(propagator_reader)
            normal = Operator.from_terms('normal', [OperatorTerm(
                0, ElementalComponent.path(()), 0, 1)])
            q = Propagator('u', Tag(0, 0), Tag(1, 1))
            diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                              ['normal', 'normal'], [None, q])
            with patch.object(elemental, 'block', side_effect=AssertionError('numeric elemental read')), \
                 patch.object(propagator, 'block', side_effect=AssertionError('numeric propagator read')):
                with self.assertRaisesRegex(ValueError, 'endpoint'):
                    calc_diagram_eval(diagram, propagator_map={q: propagator},
                        vertex_map={'normal': Meson(elemental, normal, phase_convention='anchor')},
                        configuration='cfg', usedNe=2, optimize='none', execute=False)


if __name__ == '__main__':
    unittest.main()
