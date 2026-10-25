"""Pure expanded correlator envelope; loader capabilities are not archival data."""
from __future__ import annotations
from dataclasses import dataclass, field
from .inputs import DiagramInputs, DiagramBindings, CoordinateSpec, BatchSpec
from .outputs import OutputRecipe

@dataclass(frozen=True)
class ExpandedCorrelator:
    correlator: object
    inputs: DiagramInputs | None
    output_recipe: OutputRecipe
    bindings: DiagramBindings
    coordinates: CoordinateSpec
    batch: BatchSpec
    adapter: object = field(compare=False, repr=False, default=None)
    identity_token: object = field(compare=False, repr=False, default_factory=object)

    @property
    def problem(self):
        return None if self.inputs is None else self.inputs.problem

    @property
    def tensor_specs(self):
        return {} if self.problem is None else self.problem.specs

    @property
    def terms(self):
        return () if self.problem is None else self.problem.terms

    @property
    def output_structure(self):
        return self.output_recipe.root

