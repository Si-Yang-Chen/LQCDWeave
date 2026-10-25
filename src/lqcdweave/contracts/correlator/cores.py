"""Logical tensor cores and their representations; no dense core allocation."""
from dataclasses import dataclass
import math

from .ir import TensorSpec


@dataclass(frozen=True, slots=True)
class CoreComponent:
    inputs: tuple[str, ...]
    coefficient: complex = 1
    sparse_spin: tuple | None = None

    def __post_init__(self):
        inputs = tuple(self.inputs)
        if not inputs or any(not isinstance(key, str) or not key for key in inputs):
            raise ValueError("core components require explicit input identities")
        value = complex(self.coefficient)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("core coefficient must be finite")
        entries = self.sparse_spin
        if entries is not None:
            from .scene import _sparse_spin_entries
            entries = _sparse_spin_entries(entries)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "coefficient", value)
        object.__setattr__(self, "sparse_spin", entries)


@dataclass(frozen=True, slots=True)
class TensorCoreSpec:
    kind: str
    axes: tuple[str, ...]
    spec: TensorSpec
    components: tuple[CoreComponent, ...]

    def __post_init__(self):
        axes, components = tuple(self.axes), tuple(self.components)
        if self.kind not in ("baryon", "p2p"):
            raise ValueError("unknown logical core representation")
        if len(axes) != len(self.spec.shape) or len(set(axes)) != len(axes):
            raise ValueError("logical core axes must match its rank")
        if not components or any(not isinstance(item, CoreComponent) for item in components):
            raise ValueError("logical core requires components")
        if self.kind == "baryon" and (len(axes) not in (6, 7) or any(len(c.inputs) != 2 for c in components)):
            raise ValueError("baryon components require spin and elemental inputs")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "components", components)

    @property
    def inputs(self):
        return tuple(dict.fromkeys(key for component in self.components for key in component.inputs))

    @property
    def identity(self):
        # Ordered components retain the declared accumulation order and precision.
        return (self.kind, self.axes, self.spec.shape, self.spec.dtype.str,
                tuple((c.inputs, c.coefficient, c.sparse_spin) for c in self.components))


def baryon_component_pattern(core_labels, leg_labels, output, elemental_rank):
    """Map one representation to the whole vertex's ordered physical ports.

    A seventh core axis is a retained coordinate lane. Spin is constant;
    each elemental either carries that lane or broadcasts its three modes.
    """
    from .indices import AxisPattern
    physical = tuple(core_labels[-6:])
    mode = physical[3:]
    if elemental_rank == 4:
        mode = (core_labels[0], *mode)
    return AxisPattern((physical[:3], mode, *leg_labels), output)


def lift_baryon_join(join, source_map, dynamic, specs, lane_axis, width):
    """Rename physical inputs and attach one retained lane to a whole vertex."""
    from dataclasses import replace
    core = join.core
    elementals = (tuple(part.inputs[1] for part in core.components)
                  if core is not None else (join.elemental,))
    varies = any(source_map[key] in dynamic for key in elementals)
    if core is not None:
        components = tuple(replace(part, inputs=tuple(source_map[key] for key in part.inputs))
                           for part in core.components)
        core = replace(core, components=components,
            axes=('coordinate_batch', *core.axes) if varies else core.axes,
            spec=TensorSpec((width, *core.spec.shape), core.spec.dtype) if varies else core.spec)
    return replace(join, spin=source_map[join.spin], elemental=source_map[join.elemental],
        core=core, batch_label=lane_axis if varies else None)
