"""Shared lowering of an isolated glueball to an ordinary single-input term."""

from dataclasses import replace

from lqcdweave.contraction.binding.api import Glueball
from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionTerm, TensorPreparation, BatchTensorBinding, PhysicalSlabBinding,
)


def glueball_term(registry, vertices, anchors, roles, *, add_tensor=None):
    """Return a data term for a singleton, or None for a quark component."""
    if len(vertices) != 1 or not isinstance(vertices[0], Glueball):
        return None
    vertex, time = vertices[0], anchors[0]
    reader = vertex.reader
    arguments = (vertex.component, vertex.momentum, time)
    metadata = reader.describe_block(*arguments)
    if metadata.axes or metadata.shape:
        raise ValueError("a fixed glueball component must be a scalar with no quark axes")
    register = registry.add_tensor if add_tensor is None else add_tensor
    key = register(
        ("glueball", id(reader), arguments, str(roles[0]), vertex.dagger),
        metadata, reader, f"glueball:{roles[0]}",
        lambda: reader.block(*arguments),
        lambda raw: vertex.xp.conjugate(raw) if vertex.dagger else raw,
        transform=(("operation", "glueball"), ("source_adjoint", vertex.dagger)),
        description_semantics={"component": vertex.component,
                               "momentum": reader.momentum_list[vertex.momentum],
                               "phase_convention": reader.phase_convention},
    )
    preparation = registry.preparations[key]
    registry.preparations[key] = replace(
        preparation,
        recipe=TensorPreparation("identity", preparation.blocks, conjugate=vertex.dagger),
        kernel_calls=int(vertex.dagger),
        traffic_bytes=2 * metadata.dtype.itemsize if vertex.dagger else 0,
    )
    registry.register_read(key, reader, "elemental", arguments, {},
                           coordinates=((2, str(roles[0]), 0, reader.Lt),))
    if callable(getattr(reader, "describe_slab", None)) and callable(getattr(reader, "read_slab", None)):
        component, momentum = arguments[:2]
        identity = metadata.data_identity if metadata.data_identity is not None else id(reader)
        slab = PhysicalSlabBinding(
            ("glueball", identity, component, momentum),
            lambda times: reader.describe_slab(component, momentum, times),
            lambda times: reader.read_slab(component, momentum, times),
        )
        registry.batch_bindings[key] = BatchTensorBinding(
            key, (slab,), (time,),
            (lambda raw: vertex.xp.conjugate(raw) if vertex.dagger else raw,),
        )
    return ContractionTerm((key,), AxisPattern(((),), ()))
