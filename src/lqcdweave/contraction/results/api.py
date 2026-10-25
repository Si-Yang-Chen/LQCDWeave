"""Public pure result description and assembly boundary."""

from .core import (
    OutputNode,
    OutputRecipe,
    TermValue,
    assemble_correlator,
    bind_output_assembly,
    assemble_workload_outputs,
    build_output_recipe,
    collect_diagrams,
    estimate_output_retention_bytes,
    describe_output_fields, describe_output_layout, freeze_output_recipe,
)
from .dynamic_diagrams import (
    assemble_simple_diagram_outputs,
    assemble_baryon_diagram_outputs,
)
from .structured import compose_output_recipe
from .binding_cache import OutputAssemblyCache

__all__ = [
    "OutputAssemblyCache",
    "OutputNode",
    "OutputRecipe",
    "TermValue",
    "assemble_correlator",
    "bind_output_assembly",
    "assemble_workload_outputs",
    "build_output_recipe",
    "collect_diagrams",
    "estimate_output_retention_bytes", "describe_output_fields", "describe_output_layout", "freeze_output_recipe",
    "compose_output_recipe",
    "assemble_simple_diagram_outputs",
    "assemble_baryon_diagram_outputs",
]
