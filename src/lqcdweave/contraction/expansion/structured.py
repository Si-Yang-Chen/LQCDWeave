"""Lower compact matrices to unique native Diagrams and existing output IR.

No whole projected SymPy matrix is built. Independent Wick components remain
factored; native providers continue to own headers, sources and physical terms.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
import math

import numpy as np

from lqcdweave.operators.correlator.api import (
    Diagram, StructuredCorrelator, StructuredOperation, StructuredValue,
    iter_structured_wick_entries,
)
from lqcdweave.contraction.binding.api import Meson, structured_vertex_components
from lqcdweave.contraction.results.api import (
    build_output_recipe, collect_diagrams, compose_output_recipe, estimate_output_retention_bytes,
)
from lqcdweave.contracts.correlator.api import (
    DiagramBindings, ExpandedCorrelator, OutputNode, OutputRecipe, TermContribution,
    describe_coordinates,
)


def contains_structured(value):
    active = set()

    def visit(item):
        if isinstance(item, StructuredValue):
            return True
        if isinstance(item, Mapping):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        elif isinstance(item, np.ndarray) and item.dtype == object:
            children = item.flat
        else:
            return False
        identity = id(item)
        if identity in active:
            raise ValueError('cyclic correlator containers are not supported')
        active.add(identity)
        try:
            found = False
            for child in children:
                found = visit(child) or found
            return found
        finally:
            active.remove(identity)

    return visit(value)


def _frozen(value):
    if isinstance(value, np.ndarray):
        return _frozen(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _sum(children):
    children = tuple(children)
    if not children:
        return OutputNode('constant', 0)
    return children[0] if len(children) == 1 else OutputNode('add', children=children)


def _product(children, coefficient=1):
    children = tuple(children)
    if coefficient == 0:
        return OutputNode('constant', 0)
    if not children:
        return OutputNode('constant', coefficient)
    if coefficient != 1:
        children = (OutputNode('constant', coefficient), *children)
    return children[0] if len(children) == 1 else OutputNode('multiply', children=children)


@dataclass(frozen=True)
class _ExchangeVertexKey:
    index: int
    quark_symmetry: tuple
    quark_exchange: object


class _Compiler:
    def __init__(self, bindings):
        self.bindings = bindings
        self.original_vertices, self.components, self.component_registry = {}, {}, {}
        self.selected, self.vertices, self.vertex_keys = {}, {}, {}
        self.diagrams, self.diagram_keys, self.nodes = [], {}, {}
        self.requests = {}

    def vertex(self, row):
        if row not in self.original_vertices:
            source = self.bindings.vertex_map
            self.original_vertices[row] = source[row] if isinstance(source, Mapping) else source(row)
        return self.original_vertices[row]

    def vertex_key(self, vertex, row):
        symmetry = getattr(row, 'quark_symmetry', ())
        exchange = getattr(row, 'quark_exchange', None)
        key = (id(vertex), symmetry, exchange) if symmetry or exchange else id(vertex)
        if key not in self.vertex_keys:
            name = (_ExchangeVertexKey(len(self.vertices), symmetry, exchange) if symmetry or exchange
                    else ('structured-vertex', len(self.vertices)))
            self.vertex_keys[key] = name
            self.vertices[name] = vertex
        return self.vertex_keys[key]

    def choices(self, row, selection):
        if row not in self.components:
            # Native Diagram lowering owns the whole vertex sum. Matrix-catalog
            # lowering retains its separate internal component representation.
            self.components[row] = structured_vertex_components(
                self.vertex(row), registry=self.component_registry, whole=True)
        result = []
        for vertex, coefficient in self.components[row]:
            if selection is not None:
                if not isinstance(vertex, Meson):
                    raise NotImplementedError('mode-pair projection supports dense V/V mesons only')
                key = id(vertex), selection
                if key not in self.selected:
                    self.selected[key] = vertex.with_mode_indices(selection)
                vertex = self.selected[key]
            if isinstance(vertex, Meson) and vertex.is_structurally_zero():
                continue
            result.append((self.vertex_key(vertex, row), coefficient))
        return tuple(result)

    def primitive(self, diagram, vertices):
        key = (_frozen(diagram.diagram.adjacency_matrix), tuple(diagram.time_list),
               tuple(vertices), tuple(diagram.propagator_list))
        if key not in self.diagram_keys:
            index = len(self.diagrams)
            self.diagrams.append(Diagram(diagram.diagram, diagram.time_list,
                                         list(vertices), diagram.propagator_list))
            self.diagram_keys[key] = OutputNode('term', TermContribution(f'basis:{index}'))
        return self.diagram_keys[key]

    def diagram(self, diagram, selections):
        key = diagram, selections
        if key not in self.nodes:
            choices = [self.choices(row, selection) for row, selection
                       in zip(diagram.vertex_list, selections, strict=True)]
            self.nodes[key] = _sum(_product((self.primitive(diagram, [item[0] for item in chosen]),),
                                            math.prod(item[1] for item in chosen))
                                   for chosen in product(*choices))
        return self.nodes[key]

    def task(self, task):
        ranges = (None if task.partitions is None else
                  tuple(tuple(range(a, b)) for a, b in zip(task.partitions[:-1], task.partitions[1:])))
        if ranges is not None and self.bindings.usedNe != task.partitions[-1]:
            raise ValueError('partition endpoint must equal the explicit requested Ne')
        pairs = (None if ranges is None else
                 tuple((a, b) for a in range(len(ranges)) for b in range(a, len(ranges))))
        outputs = []
        if task.partition_mode == 'memory':
            sums = {}
            for _index, entries in iter_structured_wick_entries(task):
                summands = []
                for coefficient, diagrams in entries:
                    keys = {(time, row) for diagram in diagrams
                            for time, row in zip(diagram.time_list, diagram.vertex_list, strict=True)}
                    if diagrams and len(keys) != len(task.axes):
                        raise NotImplementedError('mode projection requires distinct single-hadron vertices')
                    factors = []
                    for diagram in diagrams:
                        if diagram not in sums:
                            sums[diagram] = _sum(self.diagram(diagram, tuple(
                                (chosen[2*i], chosen[2*i+1]) for i in range(len(diagram.vertex_list))))
                                for chosen in product(ranges, repeat=2*len(diagram.vertex_list)))
                        factors.append(sums[diagram])
                    summands.append(_product(factors, coefficient))
                outputs.append(_sum(summands))
            return OutputNode('array', children=tuple(outputs), shape=task.shape)
        for _index, entries in iter_structured_wick_entries(task):
            mode_outputs = []
            positions = ((),) if pairs is None else product(range(len(pairs)), repeat=len(task.axes))
            for position in positions:
                summands = []
                for coefficient, diagrams in entries:
                    keys = sorted({(time, row) for diagram in diagrams
                                   for time, row in zip(diagram.time_list, diagram.vertex_list, strict=True)},
                                  key=lambda key: (str(key[0]), repr(key[1])))
                    if ranges is not None and diagrams and len(keys) != len(task.axes):
                        raise NotImplementedError('mode projection requires distinct single-hadron vertices')
                    selection_by_key = {}
                    for key, pair_index in zip(keys, position):
                        a, b = pairs[pair_index]
                        selection_by_key[key] = (((ranges[a], ranges[b]), 1),) if a == b else (
                            ((ranges[a], ranges[b]), 1 / np.sqrt(2)),
                            ((ranges[b], ranges[a]), 1 / np.sqrt(2)))
                    factors = []
                    for diagram in diagrams:
                        orientations = [selection_by_key[(time, row)] if ranges is not None else ((None, 1),)
                                        for time, row in zip(diagram.time_list, diagram.vertex_list, strict=True)]
                        factors.append(_sum(_product((self.diagram(diagram, tuple(item[0] for item in chosen)),),
                                                     math.prod(item[1] for item in chosen))
                                            for chosen in product(*orientations)))
                    summands.append(_product(factors, coefficient))
                mode_outputs.append(_sum(summands))
            outputs.extend(mode_outputs)
        return OutputNode('array', children=tuple(outputs), shape=task.shape)

    def visit(self, value):
        if isinstance(value, StructuredValue):
            key = id(value)
            if key not in self.requests:
                if isinstance(value, StructuredCorrelator):
                    result = self.task(value)
                else:
                    value.shape  # Reject ambiguous broadcasting of structured arrays.
                    result = OutputNode(value.kind, children=tuple(self.visit(child) for child in value.children))
                self.requests[key] = value, result
            return self.requests[key][1]
        if isinstance(value, Mapping):
            return OutputNode('dict', children=tuple(self.visit(child) for child in value.values()), keys=tuple(value))
        if isinstance(value, (tuple, list)):
            return OutputNode('tuple' if isinstance(value, tuple) else 'list',
                              children=tuple(self.visit(child) for child in value))
        if isinstance(value, np.ndarray) and value.dtype == object:
            return OutputNode('array', children=tuple(self.visit(child) for child in value.flat), shape=value.shape)
        # Constants and ordinary symbolic siblings keep the public R semantics.
        diagrams = collect_diagrams(value)
        replacements = {}
        for diagram in diagrams:
            if type(diagram) is not Diagram:
                raise NotImplementedError('mixing selected symbolic Diagrams with structured requests is unsupported')
            refs = []
            for row in diagram.vertex_list:
                vertex = self.vertex(row)
                refs.append(self.vertex_key(vertex, row))
            replacements[diagram] = self.primitive(diagram, refs)
        coordinates, _ = describe_coordinates()
        return build_output_recipe(value, {}, tuple(f'basis:{i}' for i in range(len(self.diagrams))),
                                   coordinates, diagram_recipes=replacements).root


def expand_structured(correlator, *, adapter, bindings, time_axes, time_pairs, batch):
    from .diagram_expansion import expand_correlator

    bindings = DiagramBindings() if bindings is None else bindings
    compiler = _Compiler(bindings)
    root = compiler.visit(correlator)
    expanded = expand_correlator(compiler.diagrams, adapter=adapter,
                                bindings=replace(bindings, vertex_map=compiler.vertices),
                                time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    inputs = expanded.inputs
    replacements = {f'basis:{i}': node
                    for i, node in enumerate(expanded.output_recipe.root.children)}
    recipe = compose_output_recipe(root, replacements, expanded.output_recipe.term_ids, expanded.coordinates)
    if inputs is not None:
        problem = inputs.task_batches[0].problem if inputs.task_batches else inputs.problem
        axes = inputs.task_batches[0].batch_output_axes if inputs.task_batches else inputs.batch_output_axes
        reserve = estimate_output_retention_bytes(recipe, problem, axes, expanded.coordinates)
        host = max(inputs.assembly_host_bytes, reserve) if problem.backend == 'numpy' else inputs.assembly_host_bytes
        device = max(inputs.assembly_device_bytes, reserve) if problem.backend == 'cupy' else inputs.assembly_device_bytes

        def reserve_batch(item):
            return replace(item, assembly_host_bytes=max(item.assembly_host_bytes, host),
                           assembly_device_bytes=max(item.assembly_device_bytes, device))

        selected_width = next((width for width, items in inputs.batch_candidates.items()
                               if len(items) == len(inputs.task_batches) and all(
                                   left is right for left, right in zip(
                                       items, inputs.task_batches, strict=True))), None)
        candidates = {width: tuple(reserve_batch(item) for item in items)
                      for width, items in inputs.batch_candidates.items()}
        batches = (candidates[selected_width] if selected_width is not None else
                   tuple(reserve_batch(item) for item in inputs.task_batches))
        inputs = replace(inputs, assembly_host_bytes=host, assembly_device_bytes=device,
                         task_batches=batches, batch_candidates=candidates,
                         batch_topology=('structured-v1', inputs.batch_topology, recipe.root))
    return ExpandedCorrelator(correlator, inputs, recipe, bindings,
                              expanded.coordinates, expanded.batch, adapter)


__all__ = ['contains_structured', 'expand_structured']
