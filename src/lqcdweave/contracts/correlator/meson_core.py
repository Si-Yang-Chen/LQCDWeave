"""Whole meson endpoints and private matrix-kernel data dependencies.

These records describe factorized values, not another executable graph. Only
the ordered physical endpoints and independent output rows occur in a Term.
Private spatial/component/phase axes belong to the local implementation.
"""
from dataclasses import dataclass

from .indices import AxisPattern
from .vertex_kernel import VertexKernelDescription


@dataclass(frozen=True)
class MesonWholeCore:
    key: str
    inputs: tuple[str, ...]
    output_rows: tuple
    representation: str

    def __post_init__(self):
        if (not self.key or self.representation not in ('unit-spin', 'dense')
                or len(self.inputs) != 2 or not self.output_rows):
            raise ValueError('a whole meson core needs data dependencies and independent rows')
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'output_rows', tuple(self.output_rows))


@dataclass(frozen=True)
class MesonWholeKernel:
    """One closed connected pair or trace, with explicit operand bindings.

    ``inputs`` identifies actual numerical sources; ``cores`` identifies whole
    logical vertices in the external Term. Repeated source IDs are legal.
    ``private_equation`` is a fixed local layout, never a global search input.
    A retained coordinate axis can be present on any subset of the inputs.
    """
    kind: str
    cores: tuple[str, ...]
    inputs: tuple[str, ...]
    private_equation: AxisPattern
    representation: str
    lane_axis: int | None = None
    kernel_description: object = None

    def __post_init__(self):
        count = {'connected': 2, 'loop': 1}.get(self.kind)
        if (count is None or len(self.cores) != count or len(self.inputs) != 3 * count
                or self.representation not in ('unit-spin', 'dense')
                or len(self.private_equation.inputs) != len(self.inputs)):
            raise ValueError('invalid whole meson local topology')
        object.__setattr__(self, 'cores', tuple(self.cores))
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        if self.kernel_description is not None and (
                type(self.kernel_description) is not VertexKernelDescription
                or self.kernel_description.kind != 'meson'):
            raise ValueError('whole meson implementation requires its neutral meson declaration')

    @property
    def identity(self):
        # Data identities are supplied separately by the current execution.
        return (self.kind, self.private_equation.normalized(),
                self.representation, self.lane_axis, self.kernel_description)


__all__ = ['MesonWholeCore', 'MesonWholeKernel']
