"""Owned propagation results with physical endpoints and built-in reversal."""

from dataclasses import dataclass, field

from lqcdweave.contracts.propagator import PropagatorBlock
from .symmetry import reverse_propagator


@dataclass(frozen=True)
class PropagatorResult:
    values: object
    source: object
    sink: object
    source_time: int
    sink_times: tuple
    _gamma5: object = field(repr=False)
    _gamma5_hermitian: bool = field(repr=False)
    _array_module: object = field(repr=False)

    @property
    def product(self):
        return self.sink.kind + 'S' + self.source.kind

    def at(self, sink_time):
        """Borrow a recorded physical time pair from this owned row."""
        if self.values is None:
            raise ValueError('a streamed result has no retained values')
        slot = self.sink_times.index(sink_time)
        return PropagatorBlock(self.product, sink_time, self.source_time,
                               self.sink, self.source, self.values[slot])

    def reverse(self, sink_time=None):
        """Return reverse physical pairs, using the bound action convention.

        With no time argument, return one block per original sink time. The
        reversed blocks have distinct source times; they are not a source row.
        """
        def reverse(time):
            return reverse_propagator(self.at(time), self._gamma5,
                gamma5_hermitian=self._gamma5_hermitian, array_module=self._array_module)
        if sink_time is not None:
            return reverse(sink_time)
        return [reverse(time) for time in self.sink_times]


@dataclass(frozen=True)
class PropagatorSolution:
    """Owned unsummed fields, in source-index/spin[/color],t,z,y,x,spin,color.

    Projection can reuse the solve while the producing generator is still
    loaded for this configuration. Values remain owned after reload as well.
    """

    values: object
    source: object
    source_time: int
    _project: object = field(repr=False)

    def project(self, *, sink, write=None):
        """Project onto another list of actual sinks, without a Dirac solve."""
        return self._project(self, sink, write)
