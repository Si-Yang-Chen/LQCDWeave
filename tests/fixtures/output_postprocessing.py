"""Test-owned synthetic complete-cfg output expression; not an installed API."""

from hashlib import sha256

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm


def inputs(dtype):
    """An explicit synthetic expression, not a complete physical flavor channel."""
    rng = np.random.default_rng(704)

    def random(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)

    component = ElementalComponent(0)
    elemental_data = random((1, 1, 4, 2, 2))
    propagator_data = random((4, 4, 4, 4, 2, 2))
    elemental = ArrayElemental({'v2v': elemental_data}, (component,),
        phase_convention='anchor', cfg='demo', data_identity='synthetic-output-elemental')
    propagator = ArrayPropagator({'vsv': propagator_data}, time_convention='absolute',
        cfg='demo', data_identity='synthetic-output-propagator')
    operator = Operator.from_terms('example', (OperatorTerm(5, component, 0, .7 + .2j),))
    vertices = {role: Meson(elemental, operator, dagger, phase_convention='anchor')
                for role, dagger in (('sink', False), ('source', True))}
    connected = Diagram(QuarkDiagram([[0, 1], [2, 0]]), ['sink', 'source'],
        ['sink', 'source'], [None, 'q', 'q'])
    sink_loop = Diagram(QuarkDiagram([[1]]), ['sink'], ['sink'], [None, 'q'])
    source_loop = Diagram(QuarkDiagram([[1]]), ['source'], ['source'], [None, 'q'])
    request = {'two_point': .7 * connected + .2j * sink_loop * source_loop,
               'vacuum_sink': sink_loop, 'vacuum_source': source_loop}
    fingerprint = sha256(elemental_data.tobytes() + propagator_data.tobytes()).hexdigest()
    common = dict(vertex_map=vertices, propagator_map={'q': propagator},
                  configuration='demo', usedNe=2, optimize='none', profile='default')
    return request, common, fingerprint
