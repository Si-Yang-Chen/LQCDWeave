"""Public lowering shares complete dense recipes without merging coefficients."""
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion.api import NativeDiagramBinding, expand_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.contraction.correlator import calculate


class SharedDenseFactorsTests(unittest.TestCase):
    def case(self, *, known=True, changed_gamma=False, second_reader=False):
        rng = np.random.default_rng(87)
        raw = rng.normal(size=(1, 1, 2, 2, 2)) + 1j*rng.normal(size=(1, 1, 2, 2, 2))
        values = rng.normal(size=(2, 2, 4, 4, 2, 2)) + 1j*rng.normal(size=(2, 2, 4, 4, 2, 2))
        component = ElementalComponent.path(())
        elemental = ArrayElemental({'v2v': raw}, (component,), phase_convention='anchor',
                                   cfg='test', data_identity='test/elemental' if known else None)
        other = ArrayElemental({'v2v': raw.copy()}, (component,), phase_convention='anchor',
                              cfg='test', data_identity='test/elemental' if known else None) if second_reader else elemental
        propagator = ArrayPropagator({'vsv': values}, time_convention='absolute', cfg='test',
                                    data_identity='test/propagator' if known else None)
        weights = (2+3j, -4+1j)
        operators = [Operator.from_terms(f'op-{i}', (OperatorTerm(
            1 if changed_gamma and i == 1 else 0, component, 0, weight),)) for i, weight in enumerate(weights)]
        sink = Operator.from_terms('sink', (OperatorTerm(0, component, 0),))
        vertices = {'a': Meson(elemental, operators[0], source=True, phase_convention='anchor'),
                    'b': Meson(other, operators[1], source=True, phase_convention='anchor'),
                    'sink': Meson(elemental, sink, phase_convention='anchor')}
        graphs = {name: Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], [name, 'sink'],
                               [None, 'q-forward', 'q-backward']) for name in ('a', 'b')}
        bindings = DiagramBindings({'q-forward': propagator, 'q-backward': propagator},
                                   vertices, 'test', usedNe=2)
        return graphs, bindings, elemental, propagator, raw, values, weights

    def describe(self, graphs, bindings):
        return expand_correlator(graphs, adapter=NativeDiagramBinding(), bindings=bindings)

    def test_separate_meson_objects_share_exact_recipe_before_search_without_io(self):
        graphs, bindings, elemental, propagator, *_ = self.case()
        with patch.object(elemental, 'block', side_effect=AssertionError('metadata read')), \
             patch.object(propagator, 'block', side_effect=AssertionError('metadata read')):
            expanded = self.describe(graphs, bindings)
        problem = expanded.inputs.problem
        self.assertEqual(len(problem.specs), 4)
        self.assertEqual(problem.terms[0].inputs, problem.terms[1].inputs)
        self.assertEqual(len(expanded.inputs.preparations), 4)

    def test_gamma_and_unknown_data_identities_remain_distinct(self):
        for options in ({'changed_gamma': True}, {'known': False}):
            with self.subTest(options=options):
                graphs, bindings, *_ = self.case(**options)
                problem = self.describe(graphs, bindings).inputs.problem
                self.assertEqual(len(problem.specs), 5)
                self.assertNotEqual(problem.terms[0].inputs, problem.terms[1].inputs)

    def test_separate_readers_of_same_declared_dataset_share_batch_binding(self):
        graphs, bindings, *_ = self.case(second_reader=True)
        problem = self.describe(graphs, bindings).inputs.problem
        self.assertEqual(len(problem.specs), 4)
        self.assertEqual(problem.terms[0].inputs, problem.terms[1].inputs)

    def test_known_different_dataset_is_not_merged(self):
        graphs, bindings, *_ = self.case(second_reader=True)
        bindings.vertex_map['b'].reader.data_identity = 'test/other-elemental'
        problem = self.describe(graphs, bindings).inputs.problem
        self.assertEqual(len(problem.specs), 5)

    def test_dagger_transform_remains_distinct_at_same_time(self):
        graphs, bindings, *_ = self.case()
        graphs = {name: Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 0], [name, 'sink'],
                               [None, 'q-forward', 'q-backward']) for name in ('a', 'b')}
        problem = self.describe(graphs, bindings).inputs.problem
        vertices = [spec for spec in problem.specs.values() if spec.description.role == 'elemental']
        self.assertEqual(len(vertices), 2)
        self.assertNotEqual(vertices[0].description.transform, vertices[1].description.transform)

    def test_public_execution_preserves_each_complex_weight_and_single_materialization(self):
        graphs, bindings, elemental, propagator, raw, values, weights = self.case()
        with patch.object(elemental, 'block', wraps=elemental.block) as reads:
            result = calculate(graphs, propagator_map=bindings.propagator_map,
                               vertex_map=bindings.vertex_map, configuration='test', usedNe=2,
                               memory_limit=1 << 24, optimize='anytime', storage='bounded',
                               search={'root_width': 1, 'branch_width': 1, 'max_states': 100}, execute=True)
        source = np.einsum('ab,ij->abij', gamma(0), raw[0, 0, 0].conj().T)
        sink = np.einsum('ab,ij->abij', gamma(0), raw[0, 0, 1])
        reference = np.einsum('BAON,DCQP,ADNQ,CBPO->', values[0, 1], values[1, 0], source, sink)
        for name, weight in zip(('a', 'b'), weights):
            np.testing.assert_allclose(result.value[name], weight.conjugate()*reference, rtol=1e-12)
        self.assertEqual(reads.call_count, 2)
        self.assertEqual(len(result.inputs.problem.specs), 4)

    def test_full_time_scan_keeps_shared_roles_consistent_and_preserves_values(self):
        graphs, bindings, _, _, raw, values, weights = self.case(second_reader=True)
        result = calculate(graphs, propagator_map=bindings.propagator_map,
                           vertex_map=bindings.vertex_map, configuration='test', usedNe=2,
                           time_axes={0: (0, 1), 1: (0, 1)}, batch='auto',
                           memory_limit=1 << 24, optimize='anytime', storage='bounded',
                           search={'root_width': 1, 'branch_width': 1, 'max_states': 100}, execute=True)
        reference = np.empty((2, 2), complex)
        for ts in range(2):
            for tt in range(2):
                source = np.einsum('ab,ij->abij', gamma(0), raw[0, 0, ts].conj().T)
                sink = np.einsum('ab,ij->abij', gamma(0), raw[0, 0, tt])
                reference[ts, tt] = np.einsum('BAON,DCQP,ADNQ,CBPO->',
                                             values[ts, tt], values[tt, ts], source, sink)
        for name, weight in zip(('a', 'b'), weights):
            np.testing.assert_allclose(result.value[name], weight.conjugate()*reference, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
