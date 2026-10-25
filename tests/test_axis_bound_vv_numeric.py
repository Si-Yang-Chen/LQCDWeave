"""Independent dense trace with distinct declared left/right mode identities."""

from dataclasses import replace
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.correlator import calc_diagram_eval


class AxisBoundVVNumericalTests(unittest.TestCase):
    def test_raw_mode_slots_and_source_dagger_against_independent_trace(self):
        rng = np.random.default_rng(281)

        def complex_array(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        spaces = complex_array((1, 1, 3, 2, 2))
        outgoing = complex_array((3, 3, 4, 4, 2, 2))
        incoming = complex_array((3, 3, 4, 4, 2, 2))
        left = ('actual-left-basis', (1, 2))
        right = ('actual-right-basis', (7, 3))
        component = ElementalComponent.path(())
        coefficient_source, coefficient_sink = .4 + .9j, -.2 + .3j
        source_operator = Operator.from_terms('source', (
            OperatorTerm(3, component, 0, coefficient_source),))
        sink_operator = Operator.from_terms('sink', (
            OperatorTerm(5, component, 0, coefficient_sink),))
        forward = Propagator('u', Tag(0, 0), Tag(1, 2))
        reverse = Propagator('d', Tag(1, 2), Tag(0, 0))
        graph = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 2],
                        ['source', 'sink'], [None, forward, reverse])

        def matrix(block):
            return block.transpose(0, 2, 1, 3).reshape(8, 8)

        for dagger in (False, True):
            with self.subTest(source_dagger=dagger):
                elemental = ArrayElemental({'v2v': spaces}, (component,),
                    phase_convention='anchor', cfg='cfg', data_identity='synthetic-vv')
                original = elemental.describe_block

                def describe_elemental(*args, **kwargs):
                    return replace(original(*args, **kwargs),
                        axes=('sink_mode', 'source_mode'), axis_bindings={
                            'sink_mode': left, 'source_mode': right})

                elemental.describe_block = describe_elemental
                readers = {}
                for quark, array, sink_binding, source_binding in (
                    (forward, outgoing, right, right if dagger else left),
                    (reverse, incoming, left if dagger else right, left),
                ):
                    reader = ArrayPropagator({'vsv': array}, time_convention='absolute',
                        cfg='cfg', data_identity=f'synthetic-{quark.flavor}-line')
                    describe = reader.describe_block

                    def describe_propagator(*args, _describe=describe,
                                            _sink=sink_binding, _source=source_binding,
                                            **kwargs):
                        return replace(_describe(*args, **kwargs), axis_bindings={
                            'sink_mode': _sink, 'source_mode': _source})

                    reader.describe_block = describe_propagator
                    readers[quark] = reader
                calculation = calc_diagram_eval(graph, propagator_map=readers,
                    vertex_map={
                        'source': Meson(elemental, source_operator, source=dagger,
                                        phase_convention='anchor'),
                        'sink': Meson(elemental, sink_operator, phase_convention='anchor'),
                    }, configuration='cfg', usedNe=2, optimize='none')

                source_space, source_spin = spaces[0, 0, 0], gamma(3)
                source_coefficient = coefficient_source
                self.assertGreater(np.linalg.norm(source_space - source_space.conj().T), 1)
                if dagger:
                    source_space = source_space.conj().T
                    source_spin = gamma(8) @ source_spin.conj().T @ gamma(8)
                    source_coefficient = source_coefficient.conjugate()
                # Adjacency directs source field -> S(2,0), and sink field ->
                # S(0,2). Matrix products use each incoming bar before its field.
                expected = source_coefficient * coefficient_sink * np.trace(
                    np.kron(source_spin, source_space) @ matrix(incoming[2, 0]) @
                    np.kron(gamma(5), spaces[0, 0, 2]) @ matrix(outgoing[0, 2]))
                np.testing.assert_allclose(calculation.value, expected, rtol=2e-13, atol=2e-12)
                descriptions = [spec.description for spec in calculation.inputs.problem.specs.values()]
                for description in descriptions:
                    if description.role == 'elemental':
                        self.assertEqual(dict(description.bindings['axis_bindings']), {
                            'sink_mode': left, 'source_mode': right})
                        with self.assertRaises(TypeError):
                            description.bindings['axis_bindings']['sink_mode'] = right
                    elif description.role == 'propagator':
                        self.assertTrue(all(check['status'] == 'verified'
                            for check in description.bindings['endpoint_axis_bindings'].values()))


if __name__ == '__main__':
    unittest.main()
