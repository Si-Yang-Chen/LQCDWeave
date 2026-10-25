"""Metadata-only correlator lowering into the existing contraction contracts."""

from __future__ import annotations

from dataclasses import replace
from lqcdweave.contracts.correlator.api import (
    DiagramBindingProvider,
    DiagramBindings,
    DiagramInputs,
    ExpandedCorrelator,
    describe_coordinates,
)
from lqcdweave.contraction.results.api import (
    build_output_recipe,
    collect_diagrams,
    estimate_output_retention_bytes,
)




def expand_correlator(correlator, *, adapter: DiagramBindingProvider | None,
                      bindings: DiagramBindings | None = None,
                      time_axes=None, time_pairs=None, batch="auto") -> ExpandedCorrelator:
    """Describe all Terms, tensor requests, output structure, and coordinates.

    ``adapter.describe`` has one fixed protocol; no signature reflection or
    numeric getter calls occur here. The public workflow can build bindings
    from the legacy propagator/vertex/configuration parameters.
    """
    from .structured import contains_structured, expand_structured
    if contains_structured(correlator):
        return expand_structured(correlator, adapter=adapter, bindings=bindings,
                                 time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    bindings = DiagramBindings() if bindings is None else bindings
    coordinates, batch_spec = describe_coordinates(
        time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    diagrams = collect_diagrams(correlator)
    if not diagrams:
        recipe = build_output_recipe(correlator, {}, (), coordinates)
        return ExpandedCorrelator(correlator, None, recipe, bindings,
                                  coordinates, batch_spec)
    if adapter is None:
        raise NotImplementedError(
            "expand_correlator requires an explicit DiagramBindingProvider; "
            "calc_diagram_eval supplies the native legacy-map adapter"
        )
    describe = getattr(adapter, "describe", None)
    if not callable(describe):
        raise TypeError("diagram adapter must implement describe(correlator, bindings, coordinates, batch)")
    inputs = describe(correlator, bindings=bindings,
                      coordinates=coordinates, batch=batch_spec)
    if not isinstance(inputs, DiagramInputs):
        raise TypeError("diagram adapter describe must return DiagramInputs")
    if inputs.coordinates != coordinates or inputs.batch != batch_spec:
        raise ValueError("diagram adapter returned coordinates/batch different from the request")
    if not inputs.problem.terms:
        raise ValueError("a correlator containing Diagrams expanded to no ContractionTerms")
    mapped_diagrams = set(inputs.diagram_terms) | set(inputs.diagram_recipes)
    if mapped_diagrams != set(diagrams):
        raise ValueError("diagram_terms/diagram_recipes must map every unique Diagram exactly once")
    recipe = build_output_recipe(correlator, inputs.diagram_terms,
                                 inputs.term_ids, coordinates,
                                 diagram_recipes=inputs.diagram_recipes)
    reserve_problem = (inputs.task_batches[0].problem if inputs.task_batches
                       else inputs.problem)
    reserve_axes = (inputs.task_batches[0].batch_output_axes if inputs.task_batches
                    else inputs.batch_output_axes)
    task_bytes = estimate_output_retention_bytes(
        recipe, reserve_problem, reserve_axes, coordinates,
    )
    if reserve_problem.backend == "cupy":
        host_bytes = inputs.assembly_host_bytes
        device_bytes = max(inputs.assembly_device_bytes, task_bytes)
    else:
        host_bytes = max(inputs.assembly_host_bytes, task_bytes)
        device_bytes = inputs.assembly_device_bytes
    if inputs.task_batches:
        def reserve_batches(items):
            return tuple(
                replace(item,
                        assembly_host_bytes=max(item.assembly_host_bytes, host_bytes),
                        assembly_device_bytes=max(item.assembly_device_bytes, device_bytes))
                for item in items
            )

        selected_width = next((width for width, items in inputs.batch_candidates.items()
                               if len(items) == len(inputs.task_batches) and all(
                                   left is right for left, right in zip(
                                       items, inputs.task_batches, strict=True))), None)
        candidate_map = {width: reserve_batches(items)
                         for width, items in inputs.batch_candidates.items()}
        batches = (candidate_map[selected_width] if selected_width is not None
                   else reserve_batches(inputs.task_batches))
        inputs = replace(inputs, assembly_host_bytes=host_bytes,
                         assembly_device_bytes=device_bytes, task_batches=batches,
                         batch_candidates=candidate_map)
    else:
        inputs = replace(inputs, assembly_host_bytes=host_bytes,
                         assembly_device_bytes=device_bytes)
    return ExpandedCorrelator(correlator, inputs, recipe, bindings,
                              coordinates, batch_spec, adapter)


__all__ = ["ExpandedCorrelator", "expand_correlator"]
