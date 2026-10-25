"""Unweighted meson components for structured matrix lowering.

Baryon vertices retain their own registered spin/spatial core. This helper
never describes or reads data and does not split a baryon core into dense legs.
"""
from dataclasses import replace

from lqcdweave.operators.single_hadron.api import Operator
from .single_hadron import Baryon, Current, Meson
from .glueball import Glueball
from .operator_description import operator_term_semantics


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def structured_vertex_components(vertex, *, registry, whole=False):
    """Return (bound component, output coefficient) pairs within this call."""
    if isinstance(vertex, (Baryon, Glueball)):
        return ((vertex, 1),)
    if not isinstance(vertex, Meson) or isinstance(vertex, Current):
        raise NotImplementedError('structured matrices require native Meson or Baryon bindings')
    if vertex.reader is None:
        raise NotImplementedError('structured mesons require an injected metadata reader')
    if any(term.current_term is not None for term in vertex.operator.terms):
        raise NotImplementedError('structured directed-current operators are not implemented')
    if whole:
        if not any(term.coefficient != 0 for term in vertex.operator.terms):
            return ()
        # Equal complete definitions can reuse one numerical vertex while each
        # output row remains a distinct output. Partial component overlap does
        # not merge different operators or expose their terms to global P.
        identity = ('whole-meson', id(vertex.reader),
            tuple((_freeze(operator_term_semantics(term)), complex(term.coefficient),
                   term.anchor_time_offset, term.endpoint_time_offsets)
                  for term in vertex.operator.terms), vertex.dagger,
            _freeze(vertex.usedNe), _freeze(vertex.usedNp),
            vertex.phase_convention, vertex.boundary, id(vertex.xp),
            vertex.requested_sectors, vertex.mode_indices, vertex.key)
        return ((registry.setdefault(identity, vertex), 1),)
    result = []
    for term in vertex.operator.terms:
        coefficient = term.coefficient.conjugate() if vertex.dagger else term.coefficient
        if coefficient == 0:
            continue
        unweighted = replace(term, coefficient=1)
        identity = (id(vertex.reader), _freeze(operator_term_semantics(unweighted)), vertex.dagger,
                    unweighted.anchor_time_offset, unweighted.endpoint_time_offsets,
                    _freeze(vertex.usedNe), _freeze(vertex.usedNp),
                    vertex.phase_convention, vertex.boundary, id(vertex.xp),
                    vertex.requested_sectors, vertex.mode_indices, vertex.key)
        if identity not in registry:
            component = Meson(vertex.reader, Operator.from_terms('structured-component', (unweighted,)),
                              vertex.dagger, phase_convention=vertex.phase_convention,
                              boundary=vertex.boundary, array_module=vertex.xp,
                              requested_sectors=vertex.requested_sectors)
            component.key, component.usedNe, component.usedNp = vertex.key, vertex.usedNe, vertex.usedNp
            component.mode_indices = vertex.mode_indices
            registry[identity] = component
        result.append((registry[identity], coefficient))
    return tuple(result)


__all__ = ['structured_vertex_components']
