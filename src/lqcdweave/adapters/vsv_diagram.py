"""Metadata-first V/V meson Diagram binding.

Edge/axis construction is copied from contraction.vsv_graph's reference
evaluate_vsv_diagram; numerical getters are deferred to storage replay.
"""

from dataclasses import dataclass

import numpy as np
from lqcdweave.contracts.correlator.api import (
    AxisPattern, TensorSpec, ContractionTerm, ContractionProblem, SourceSpec,
    StorageBlock, SourcePreparation, DiagramInputs, BatchSpec,
    CoordinateSpec, DiagramBindings, TermContribution, TensorDescription,
)
from lqcdweave.contraction.results.api import collect_diagrams


@dataclass(frozen=True)
class VSVData:
    """Bind validated V/V data to symbolic meson diagrams (CPU reference).

    spec callbacks have the same arguments as their getter and return TensorSpec
    without loading arrays. Propagator arguments are (tag, sink, source).
    Getters return owned arrays, not reusable scratch buffers. Dataset identity,
    basis order, coverage and physical conventions must be checked by the caller.
    This adapter treats each request as one physical block. For file slabs or
    nested-prefix reuse supply a dataset adapter implementing DiagramData instead.
    """

    time_map: object
    vertex_spec: object
    propagator_spec: object
    vertex_getter: object
    propagator_getter: object
    dataset_identity: str

    def describe(self, diagrams, *, bindings: DiagramBindings,
                 coordinates: CoordinateSpec, batch: BatchSpec):
        if not self.dataset_identity:
            raise ValueError("validated dataset identity is required")
        if any(value is not None for value in (
            bindings.propagator_map, bindings.vertex_map,
            bindings.configuration, bindings.usedNe,
        )):
            raise NotImplementedError("VSVData is already bound and does not consume legacy handle maps")
        if coordinates.scan_shape:
            raise NotImplementedError("VSVData currently binds one explicit time_map")
        time_map = self.time_map
        if coordinates.mode == "axes":
            time_map = dict(coordinates.points[0].values)
            if set(time_map) != set(self.time_map):
                raise ValueError("fixed time_axes must define every symbol in the VSVData time_map")
        if batch.size != "auto":
            raise NotImplementedError("VSVData does not implement batch selection")
        atoms = collect_diagrams(diagrams)
        specs, sources, blocks, preparations, requests, terms = {}, {}, {}, {}, {}, []
        identities = {}

        def request(kind, args):
            identity = (kind, *args)
            if identity in identities:
                return identities[identity]
            spec_fn = self.vertex_spec if kind == "vertex" else self.propagator_spec
            spec = spec_fn(*args)
            if not isinstance(spec, TensorSpec):
                raise TypeError("metadata callbacks must return TensorSpec")
            if len(spec.shape) != 4 or spec.shape[:2] != (4, 4) or spec.shape[2] != spec.shape[3]:
                raise ValueError("V/V input requires two spin and two equal mode axes")
            key = f"{kind}:{len(requests)}"
            identities[identity] = key
            requests[key] = (kind, args)
            if spec.description is None:
                if kind == "propagator":
                    axes = ("sink_spin", "source_spin", "sink_mode", "source_mode")
                    sink_time, source_time = args[1], args[2]
                    role, sector = "propagator:vsv", "vsv"
                else:
                    axes = ("spin_field", "spin_bar", "mode_field", "mode_bar")
                    sink_time, source_time = args[1], None
                    role, sector = "meson-elemental:v2v", "v2v"
                description = TensorDescription(
                    axes=axes,
                    data_identity=self.dataset_identity,
                    selection_identity=repr(identity),
                    role=role,
                    source_bindings=(key,),
                    transform=(("operation", "identity"),),
                    semantics={"sector": sector, "reference_backend": "numpy"},
                    bindings={
                        "sink_time": int(sink_time),
                        "source_time": None if source_time is None else int(source_time),
                    },
                )
                spec = TensorSpec(spec.shape, spec.dtype, description=description)
            specs[key] = spec
            size = int(np.prod(spec.shape)) * spec.dtype.itemsize
            sources[key] = SourceSpec(key, 0)
            blocks[key] = StorageBlock(self.dataset_identity, repr(identity), size, host_bytes=size)
            preparations[key] = SourcePreparation((key,))
            return key

        # Same index pairing as the existing independent numerical reference.
        diagram_terms = {}
        term_ids = []
        for diagram_index, diagram in enumerate(atoms):
            adjacency = diagram.diagram.adjacency_matrix
            count = len(diagram.vertex_list)
            if len(adjacency) != count or len(diagram.time_list) != count or any(len(r) != count for r in adjacency):
                raise ValueError("diagram vertices and adjacency matrix differ")
            times = []
            for symbol in diagram.time_list:
                if symbol not in time_map or type(time_map[symbol]) is not int:
                    raise ValueError("each symbolic vertex time needs one physical integer time")
                times.append(time_map[symbol])
            edges = []
            outgoing, incoming = [0] * count, [0] * count
            for source, row in enumerate(adjacency):
                for sink, label in enumerate(row):
                    if label == 0:
                        continue
                    if type(label) is not int or not 0 < label < len(diagram.propagator_list):
                        raise ValueError("one integer VSV label per edge is required")
                    outgoing[source] += 1
                    incoming[sink] += 1
                    edges.append((label, source, sink))
            if not edges or any(a != 1 or b != 1 for a, b in zip(outgoing, incoming)):
                raise ValueError("VSVData supports meson V/V vertices only")
            mode_offset = 2 * len(edges)
            keys, subscripts = [], []
            lefts, rights = [None] * count, [None] * count
            for i, (label, source, sink) in enumerate(edges):
                left, right = 2 * i, 2 * i + 1
                lefts[source] = (left, mode_offset + left)
                rights[sink] = (right, mode_offset + right)
                keys.append(request("propagator", (diagram.propagator_list[label], times[sink], times[source])))
                subscripts.append((right, left, mode_offset + right, mode_offset + left))
            for i, row in enumerate(diagram.vertex_list):
                keys.append(request("vertex", (row, times[i])))
                ls, lm = lefts[i]
                rs, rm = rights[i]
                subscripts.append((ls, rs, lm, rm))
            terms.append(ContractionTerm(
                tuple(keys), AxisPattern(tuple(subscripts), ()),
            ))
            term_id = f"term:{diagram_index}"
            term_ids.append(term_id)
            diagram_terms[diagram] = (TermContribution(term_id),)

        problem = ContractionProblem(specs, tuple(terms), sources)
        vertex_getter, propagator_getter = self.vertex_getter, self.propagator_getter

        class Loader:
            def read_block(self, key, block):
                kind, args = requests[key]
                getter = vertex_getter if kind == "vertex" else propagator_getter
                array = np.asarray(getter(*args))
                spec = specs[key]
                if array.shape != spec.shape or array.dtype != spec.dtype:
                    raise ValueError("actual data shape/dtype differs from declared metadata")
                return array

            def materialize(self, key, source, arrays):
                return arrays[key]

        def output_slots(value):
            if isinstance(value, dict):
                return sum(output_slots(child) for child in value.values())
            if isinstance(value, (list, tuple)):
                return sum(output_slots(child) for child in value)
            if isinstance(value, np.ndarray) and value.dtype == object:
                return sum(output_slots(child) for child in value.flat)
            return 1

        return DiagramInputs(problem, blocks, preparations, Loader(), coordinates, batch,
                             tuple(term_ids), diagram_terms=diagram_terms,
                             assembly_host_bytes=output_slots(diagrams) * 16 + 48)
