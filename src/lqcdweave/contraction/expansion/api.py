"""Public entry points for metadata-only correlator expansion."""

from lqcdweave.contracts.correlator.api import (
    ExpandedCorrelator,
    describe_coordinates,
    group_coordinate_batches,
)

from .diagram_expansion import expand_correlator
from .structured import contains_structured
from .baryon_diagram import BaryonDiagramBinding
from .native import NativeDiagramBinding
from .matrix_workflow import expand_meson_matrix, try_expand_search_matrices
from .matrix_catalog import describe_meson_matrix
from .matrix_inputs import bind_meson_matrix_catalog, MatrixCatalogBinder
from .saved_inputs import (bound_source_identities,restore_input_loader, describe_input_reads, describe_input_preparations,
                          input_readers,describe_coordinate_reads,unknown_reader_aliases)
from .sampling_compatibility import input_sampling_signature
from .persisted_meson import describe_meson_diagrams
from .precision import normalize_contraction_precision
from .dynamic_diagrams import (
    expand_simple_diagrams,
    extract_baryon_topology,
    validate_baryon_diagram_batch,
    expand_baryon_diagrams,
)

__all__ = [
    "normalize_contraction_precision",
    "expand_correlator",
    "contains_structured",
    "expand_meson_matrix",
    "try_expand_search_matrices",
    "describe_meson_matrix",
    "bind_meson_matrix_catalog",
    "MatrixCatalogBinder",
    "restore_input_loader", "bound_source_identities",
    "describe_input_reads",
    "describe_input_preparations",
    "input_readers",
    "describe_coordinate_reads",
    "unknown_reader_aliases",
    "input_sampling_signature",
    "ExpandedCorrelator",
    "NativeDiagramBinding",
    "BaryonDiagramBinding",
    "describe_coordinates",
    "describe_meson_diagrams",
    "group_coordinate_batches",
    "expand_simple_diagrams",
    "extract_baryon_topology",
    "validate_baryon_diagram_batch",
    "expand_baryon_diagrams",
]
