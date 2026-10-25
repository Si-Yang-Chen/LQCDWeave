"""Named S/O/W bindings and ordered physical scans versus an independent trace."""
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Flavor, Hadron, HadronIrrep, Tag, gen_correlator
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.contraction.correlator import calc_diagram_eval
from lqcdweave.contraction.correlator_inputs import bind_native_configuration


class NamedBindingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(962)
        random = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
        self.e = random((1, 1, 3, 1, 1))
        self.p = random((3, 3, 4, 4, 1, 1))
        self.component = ElementalComponent(0)
        self.reader = ArrayElemental({'v2v': self.e}, (self.component,),
                                    phase_convention='anchor', cfg='test')
        self.propagator = ArrayPropagator({'vsv': self.p}, time_convention='absolute', cfg='test')
        self.vertices = {name: Meson(self.reader, self.operator(name), source,
                                    phase_convention='anchor')
                         for name, source in (('sink_op', False), ('source_op', True))}

    def operator(self, name):
        return Operator.from_terms(name, (OperatorTerm(1, self.component, 0, .3+.7j),))

    def task(self, *, structured=False, times=('sink', 'source')):
        axes = [[Hadron(HadronIrrep(name, (0, 0, 0), 'A_1', -1, Tag(0, 0))[0], Flavor('ud'))]
                for name in ('sink_op', 'source_op')]
        return gen_correlator(axes, times, [False, True], structured=structured)

    def compute(self, task, **options):
        return calc_diagram_eval(task, vertex_map=self.vertices,
            propagator_map={'S^q': self.propagator, r'S^q_\mathrm{local}': self.propagator},
            usedNe=1, configuration='test', optimize='none', memory_limit=16*2**20,
            **options).value

    def reference(self, source, sink):
        g = (.3+.7j)*gamma(1)
        a = gamma(8) @ (g*self.e[0, 0, source, 0, 0]).conj().T @ gamma(8)
        b = g*self.e[0, 0, sink, 0, 0]
        return -np.trace(a @ self.p[sink, source, :, :, 0, 0] @ b @
                         self.p[source, sink, :, :, 0, 0])

    def test_named_scan_order_symbolic_and_structured_matches_trace(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                value = self.compute(self.task(structured=structured),
                                     time_axes={'source': (2, 0), 'sink': (1, 2, 0)},
                                     batch={'axis': 'sink', 'size': 2})
                expected = [[self.reference(source, sink) for sink in (1, 2, 0)]
                            for source in (2, 0)]
                self.assertEqual(value.shape, (1, 1, 2, 3))
                np.testing.assert_allclose(value[0, 0], expected, rtol=1e-12, atol=1e-12)

    def test_shared_time_key_is_one_axis_and_matches_diagonal(self):
        value = self.compute(self.task(times=('current', 'current')),
                             time_axes={'current': range(3)})
        self.assertEqual(value.shape, (1, 1, 3))
        np.testing.assert_allclose(value[0, 0], [self.reference(t, t) for t in range(3)],
                                   rtol=1e-12, atol=1e-12)

    def test_named_mapping_errors_precede_reader_resolution(self):
        cases = (
            ({'sink_op': self.vertices['sink_op']}, KeyError, 'source_op'),
            ({'wrong': self.vertices['sink_op'], 'source_op': self.vertices['source_op']},
             ValueError, 'must match'),
            ({'sink_op': self.vertices['sink_op'], 'alias': self.vertices['sink_op']},
             ValueError, 'duplicate'),
        )
        for vertices, error, message in cases:
            with self.subTest(message=message), patch(
                    'lqcdweave.contraction.correlator_inputs.resolve_correlator_source',
                    side_effect=AssertionError('read before name validation')):
                with self.assertRaisesRegex(error, message):
                    bind_native_configuration(self.task(), DiagramBindings(
                        {'S^q': self.propagator}, vertices, 'test'))

    def test_invalid_operator_names_are_rejected(self):
        for name in (None, 1, '', ' '):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                self.operator(name)


if __name__ == '__main__':
    unittest.main()
