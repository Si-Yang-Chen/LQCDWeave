"""Resolve native declarations into independent bindings for one calculation."""

from collections.abc import Mapping

import numpy as np

from lqcdweave.adapters.preset import resolve_correlator_source
from lqcdweave.contraction.binding.api import Baryon, Meson, Glueball
from lqcdweave.contraction.expansion.api import contains_structured
from lqcdweave.contraction.results.api import collect_diagrams
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.correlator.api import (
    HadronIrrepRow, StructuredCorrelator, StructuredOperation, structured_axis_rows,
)


def bind_native_configuration(correlator, bindings: DiagramBindings) -> DiagramBindings:
    """Resolve each shared source once per call; never mutate a vertex template.

    Existing loaded vertices stay tied to their explicit reader. Custom
    DiagramData adapters bypass this native convenience entirely.
    """
    structured = contains_structured(correlator)
    if not structured and not collect_diagrams(correlator):
        return bindings
    original = bindings.vertex_map
    named = (isinstance(original, Mapping) and bool(original)
             and all(isinstance(key, str) for key in original)
             and any(isinstance(row, HadronIrrepRow) for row in _symbolic_rows(correlator)))
    if named:
        _validate_named_vertices(original, _symbolic_rows(correlator))
    readers, vertices = {}, {}

    def reader(source, family):
        identity = (id(source), family)
        if identity not in readers:
            resolved = resolve_correlator_source(source, bindings.configuration)
            if family == "elemental":
                from lqcdweave.contraction.elemental_preparation import bind_managed_elemental
                resolved = bind_managed_elemental(resolved, bindings.elemental_weighted)
            readers[identity] = (source, resolved)
        return readers[identity][1]

    def vertex(template):
        if not isinstance(template, (Meson, Baryon, Glueball)):
            return template
        identity = id(template)
        if identity not in vertices:
            source = template.reader if template.reader is not None else template.elemental
            resolved_reader = reader(source, "glueball" if isinstance(template, Glueball) else "elemental")
            vertices[identity] = (template, template.with_reader(
                resolved_reader, configuration=bindings.configuration))
        return vertices[identity][1]

    propagators = bindings.propagator_map
    if isinstance(propagators, Mapping):
        propagators = {key: reader(source, "propagator") for key, source in propagators.items()}
    vertex_map = original
    if original is not None:
        by_key = {}

        def vertex_map(key):
            if key not in by_key:
                lookup = key.hadron_name if named and isinstance(key, HadronIrrepRow) else key
                template = original[lookup] if isinstance(original, Mapping) else original(key)
                by_key[key] = vertex(template)
            return by_key[key]

    return DiagramBindings(propagators, vertex_map, bindings.configuration,
                           bindings.usedNe, bindings.usedNp, bindings.groups_independent,
                           bindings.elemental_weighted)


def _symbolic_rows(correlator):
    """Read existing symbolic declarations without expanding structured Wick rows."""
    if isinstance(correlator, StructuredCorrelator):
        for axis in structured_axis_rows(correlator):
            for position in axis:
                for row, _coefficient in position:
                    yield row
    elif isinstance(correlator, StructuredOperation):
        for child in correlator.children:
            yield from _symbolic_rows(child)
    elif isinstance(correlator, Mapping):
        for child in correlator.values():
            yield from _symbolic_rows(child)
    elif isinstance(correlator, (list, tuple)):
        for child in correlator:
            yield from _symbolic_rows(child)
    elif isinstance(correlator, np.ndarray) and correlator.dtype == object:
        for child in correlator.flat:
            yield from _symbolic_rows(child)
    else:
        for diagram in collect_diagrams(correlator):
            yield from diagram.vertex_list


def _validate_named_vertices(vertices, rows):
    """Validate frontend names only; B still owns scientific compatibility."""
    names = {}
    for key, vertex in vertices.items():
        if not isinstance(vertex, (Meson, Baryon, Glueball)):
            raise TypeError("named vertex_map values must be Meson, Baryon, Current, or Glueball templates")
        name = vertex.name if isinstance(vertex, Glueball) else vertex.operator.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("named vertex_map requires a nonempty explicit Operator.name")
        if name in names:
            raise ValueError(f"duplicate Operator.name {name!r} in vertex_map")
        names[name] = key
    for name, key in names.items():
        if key != name:
            raise ValueError(f"vertex_map key {key!r} must match Operator.name {name!r}")
    for row in rows:
        key = row.hadron_name if isinstance(row, HadronIrrepRow) else row
        if key not in vertices:
            raise KeyError(f"vertex_map has no Operator.name binding for {key!r}")
