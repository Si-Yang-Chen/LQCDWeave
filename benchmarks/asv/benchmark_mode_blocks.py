"""Synthetic complex CPU meson replay, including all 256 mode blocks."""
import numpy as np


class MesonModeBlocks:
    params = [120, 128]
    param_names = ["Ne"]
    repeat = (5, 10, 15)
    number = 1
    warmup_time = 0.1
    timeout = 120

    def setup(self, ne):
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
        from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
        from lqcdweave.contraction.correlator import calculate
        rng = np.random.default_rng(928)
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        component = ElementalComponent(0)
        elemental = ArrayElemental({"v2v": random((1, 1, 2, ne, ne))}, (component,),
                                  phase_convention="anchor", cfg="cfg", data_identity="asv-elemental")
        propagator = ArrayPropagator({"vsv": random((2, 2, 4, 4, ne, ne))},
                                    time_convention="absolute", cfg="cfg", data_identity="asv-propagator")
        operator = Operator.from_terms("meson", [OperatorTerm(5, component, 0, .3 + .7j)])
        vertices = {"a": Meson(elemental, operator, phase_convention="anchor"),
                    "b": Meson(elemental, operator, True, phase_convention="anchor")}
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
        options = dict(vertex_map=vertices, propagator_map={"q": propagator}, usedNe=ne,
                       configuration="cfg", optimize="none", execute=False, memory_limit=2**30)
        self.whole = calculate(diagram, **options)
        self.blocked = None
        try:
            from lqcdweave.operators.correlator.api import split_eigenmodes
        except ImportError:
            return  # Historical commits retain the whole replay comparison.
        blocked = split_eigenmodes(diagram, partitions=range(0, ne + 1, ne // 4))
        self.blocked = calculate(blocked, **options)
        np.testing.assert_allclose(self.blocked.run().value.sum(), self.whole.run().value,
                                   rtol=1e-11, atol=1e-8)

    def time_whole(self, ne):
        self.whole.run()

    def time_all_blocks(self, ne):
        if self.blocked is None:
            raise NotImplementedError("mode partitions were not available on this commit")
        self.blocked.run()
