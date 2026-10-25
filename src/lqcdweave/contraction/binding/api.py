"""Public APIs for binding Operators to registered data readers."""

from .single_hadron import (
    Baryon,
    BaryonVertexFactor,
    Current,
    Meson,
    VertexFactor,
)
from .recipe import RecipeVertex
from .glueball import Glueball
from .precision import normalize_array_precision
from .operator_description import operator_term_semantics
from .time_coordinates import (validate_slot_compatibility, bind_slot_vertices,
                               validate_vertex_time_bindings, map_edge_time_slot)
from .structured import structured_vertex_components
from .persisted_getters import bind_meson_getters
from .dynamic_diagrams import read_simple_diagram_inputs, read_baryon_diagram_inputs
from lqcdweave.contracts.correlator.api import (
    BaryonFactorMetadata,
    VertexFactorMetadata,
)

__all__ = [
    "Glueball",
    "normalize_array_precision",
    "map_edge_time_slot", "bind_slot_vertices", "validate_vertex_time_bindings", "validate_slot_compatibility", "Meson", "RecipeVertex", "Baryon", "Current", "VertexFactor", "BaryonVertexFactor",
    "VertexFactorMetadata", "BaryonFactorMetadata",
    "operator_term_semantics", "bind_meson_getters",
    "structured_vertex_components",
    "read_simple_diagram_inputs", "read_baryon_diagram_inputs",
]
