"""Public symbols for data-independent hadron and correlator definitions."""

from .base_types import Tag
from .quark_exchange_matrix import QuarkExchange
from .flavor_structure import HadronFlavorStructure, Propagator
from .group_projection import hadron_little_group_projection, operator_transform
from .hadron import (
    Hadron,
    gen_correlator,
    operator_conjugate,
    set_dagger_in_list,
    set_time_in_expr,
    set_time_in_list,
)
from .spatial_structure import HadronIrrep, HadronIrrepRow
from .symbolic_graph import (
    Diagram,
    QuarkDiagram,
    diagram_simplify,
    diagram_vertice_replace,
    quark_contract,
    remove_disconneted_diagram,
    validate_adjacency_matrix,
)

# ``Flavor`` is the user-facing symbolic flavor structure.  The six-value
# Literal used to annotate an individual quark is an implementation detail.
Flavor = HadronFlavorStructure

from .mode_blocks import (
    ModeBlockedDiagram, split_eigenmodes, project_meson_mode_pairs,
    split_baryon_eigenmodes, project_baryon_mode_triples,
)
from .mode_projection import split_meson_mode_pairs
from .structured import (
    StructuredCorrelator, StructuredOperation, StructuredValue, iter_structured_wick_entries,
    structured_axis_rows, structured_axis_times, structured_wick_templates,
)

__all__ = [
    "Tag",
    "HadronIrrep",
    "HadronIrrepRow",
    "Flavor",
    "HadronFlavorStructure",
    "Hadron",
    "QuarkExchange",
    "hadron_little_group_projection",
    "operator_transform",
    "gen_correlator",
    "StructuredCorrelator", "StructuredOperation", "StructuredValue", "iter_structured_wick_entries",
    "structured_axis_rows", "structured_axis_times", "structured_wick_templates",
    "Diagram",
    "ModeBlockedDiagram",
    "split_eigenmodes",
    "project_meson_mode_pairs",
    "split_baryon_eigenmodes",
    "project_baryon_mode_triples",
    "split_meson_mode_pairs",
    "QuarkDiagram",
    "Propagator",
    "diagram_simplify",
    "diagram_vertice_replace",
    "quark_contract",
    "remove_disconneted_diagram",
    "validate_adjacency_matrix",
    "operator_conjugate",
    "set_time_in_expr",
    "set_time_in_list",
    "set_dagger_in_list",
]
