"""Choose a metadata-only Diagram expansion strategy from bound vertex kinds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from lqcdweave.contraction.binding.api import Baryon, Glueball, bind_slot_vertices
from lqcdweave.contraction.results.api import collect_diagrams

from .batching import BatchDiagramBinding
from .baryon_diagram import BARYON_VERTEX_LOWERINGS, BaryonDiagramBinding
from .legacy_diagram import LegacyDiagramBinding, _referenced_propagators


class NativeDiagramBinding:
    """Select the existing Baryon or Meson/Current provider from bound objects."""

    def __init__(self, *, baryon_vertices="auto", **budgets):
        if baryon_vertices not in BARYON_VERTEX_LOWERINGS:
            raise ValueError(f"baryon_vertices must be one of {BARYON_VERTEX_LOWERINGS}")
        self.baryon_vertices = baryon_vertices
        self.budgets = dict(budgets)
        self._selected = None

    @property
    def row_choices(self):
        """Per-family static choices of the last Baryon description; empty without one."""
        return tuple(getattr(getattr(self._selected, "fixed_adapter", None), "row_choices", ()))

    def describe(self, correlator, *, bindings, coordinates, batch):
        source = bindings.vertex_map
        vertices = {}
        diagrams = collect_diagrams(correlator)
        def connected(cell):
            return any(connected(item) for item in cell) if isinstance(cell, (list, tuple)) else bool(cell)
        for diagram in diagrams:
            for row in diagram.vertex_list:
                if row not in vertices:
                    vertices[row] = source[row] if isinstance(source, Mapping) else source(row)
            adjacency = diagram.diagram.adjacency_matrix
            for index, row in enumerate(diagram.vertex_list):
                has_edges = any(connected(adjacency[index][j]) or connected(adjacency[j][index])
                                for j in range(len(adjacency)))
                if isinstance(vertices[row], Glueball) == has_edges:
                    raise ValueError("zero-quark vertices require Glueball; Glueball cannot bind a quark edge")
        if bindings.propagator_map is None:
            bindings = replace(bindings, propagator_map={})
        if not isinstance(bindings.propagator_map, Mapping):
            raise TypeError('native Diagram binding requires a propagator_map mapping')
        bindings = replace(bindings, propagator_map=_referenced_propagators(
            ((diagram.diagram.adjacency_matrix, diagram.propagator_list) for diagram in diagrams),
            bindings.propagator_map))
        bound = bind_slot_vertices(vertices.values(), bindings.propagator_map.values())
        vertices = dict(zip(vertices, bound, strict=True))
        if any(isinstance(vertex, Baryon) for vertex in vertices.values()):
            provider = BatchDiagramBinding(BaryonDiagramBinding(sector_scope="dense-v",
                coordinate_slabs=bool(coordinates.scan_shape), vertices=self.baryon_vertices), **self.budgets)
        else:
            provider = BatchDiagramBinding(LegacyDiagramBinding(), **self.budgets)
        self._selected = provider
        return provider.describe(
            correlator,
            bindings=replace(bindings, vertex_map=vertices),
            coordinates=coordinates,
            batch=batch,
        )


__all__ = ["NativeDiagramBinding"]
