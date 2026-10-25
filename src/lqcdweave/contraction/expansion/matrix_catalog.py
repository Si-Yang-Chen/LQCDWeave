"""Describe structured matrices directly; no scalar expansion or numeric IO.

Position/flavour parsing comes from S. Unweighted component identity and source
dagger weights reuse B. These records precede executable template compilation.
"""
from collections.abc import Mapping
from itertools import product
import numbers
import numpy as np

from lqcdweave.operators.correlator.api import (
    StructuredCorrelator, StructuredValue, structured_axis_rows, structured_axis_times,
    structured_wick_templates,
)
from lqcdweave.contraction.binding.api import Meson, structured_vertex_components
from lqcdweave.contracts.correlator.api import (
    normalize_time_key, DiagramBindings, OutputNode, MatrixLinearAxis, MatrixComponent, MatrixNetwork,
    MatrixWickProduct, MatrixWickBlock, MatrixGroup, MatrixCatalog, BoundMatrixCatalog,
)


def _freeze(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze(item) for item in value)
    return value


class _CatalogBuilder:
    def __init__(self, bindings, times):
        self.bindings, self.times = bindings, dict(times)
        self.components, self.vertices, self.axes, self.networks, self.groups = [], [], [], [], []
        self.vertex_cache, self.row_cache, self.registry = {}, {}, {}
        self.component_ids, self.axis_ids, self.network_ids, self.wick_cache = {}, {}, {}, {}
        self.label_names = {}
        self.nodes, self.active = {}, set()

    def label_name(self, label):
        # Mixed-mode axes repeat the same (potentially large) symbolic linear
        # combinations. Format each object only once for this request. Keep a
        # strong reference so Python cannot recycle its id during construction;
        # display text never participates in the scientific axis identity.
        identity = id(label)
        if identity not in self.label_names:
            self.label_names[identity] = label, str(label)
        return self.label_names[identity][1]

    def row(self, row, time_label):
        row_key = row, time_label
        if row_key not in self.row_cache:
            if row not in self.vertex_cache:
                source = self.bindings.vertex_map
                self.vertex_cache[row] = source[row] if isinstance(source, Mapping) else source(row)
            vertex = self.vertex_cache[row]
            if (not isinstance(vertex, Meson) or vertex.available_sectors() not in (None, ('v2v',))
                    or vertex.requested_sectors not in (None, ('v2v',))):
                raise NotImplementedError('matrix catalogs require dense V/V Meson bindings')
            result = []
            for component, coefficient in structured_vertex_components(vertex, registry=self.registry):
                time = self.times.get(time_label, time_label)
                if component.is_structurally_zero(time):
                    continue
                key = id(component), time_label
                if key not in self.component_ids:
                    time = self.times.get(time_label, time_label)
                    descriptions = component.describe_factors(time, usedNe=self.bindings.usedNe)
                    if len(descriptions) != 1:
                        raise ValueError('a matrix component must describe one unweighted factor')
                    if descriptions[0].tensor.sparse_layout != 'dense':
                        raise NotImplementedError('matrix catalogs require dense component metadata')
                    self.component_ids[key] = len(self.components)
                    self.components.append(MatrixComponent(time_label, descriptions[0]))
                    self.vertices.append(component)
                result.append((self.component_ids[key], complex(coefficient)))
            self.row_cache[row_key] = tuple(result)
        return self.row_cache[row_key]

    def axis(self, labels, rows, time_label, *, selection=None):
        key = tuple(labels), time_label, selection
        if key not in self.axis_ids:
            offsets, ids, coefficients = [0], [], []
            for choices in rows:
                combined = {}
                for row, position_weight in choices:
                    for component, weight in self.row(row, time_label):
                        combined[component] = combined.get(component, 0) + complex(position_weight) * weight
                for component, weight in combined.items():
                    if weight != 0:
                        ids.append(component)
                        coefficients.append(weight)
                offsets.append(len(ids))
            self.axis_ids[key] = len(self.axes)
            self.axes.append(MatrixLinearAxis(tuple(map(self.label_name, labels)), offsets, ids, coefficients))
        return self.axis_ids[key]

    def network(self, diagram):
        record = MatrixNetwork(_freeze(diagram.diagram.adjacency_matrix), tuple(diagram.time_list),
                               tuple(diagram.vertex_list), tuple(diagram.propagator_list))
        if record not in self.network_ids:
            self.network_ids[record] = len(self.networks)
            self.networks.append(record)
        return self.network_ids[record]

    def task(self, task):
        if task.partitions is not None:
            boundaries = task.partitions
            if (len(boundaries) < 2 or boundaries[0] != 0
                    or any(type(item) is not int for item in boundaries)
                    or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
                    or boundaries[-1] != self.bindings.usedNe):
                raise ValueError('matrix partitions must increase from zero to explicit usedNe')
        rows = structured_axis_rows(task)
        axes = tuple(self.axis(tuple(item[0] for item in labels), choices, time)
                     for labels, choices, time in zip(task.axes, rows, structured_axis_times(task), strict=True))
        # Each matrix kernel shares one propagator for a fixed pair of physical
        # endpoint selections. Equal widths alone do not imply equal modes.
        mode_groups = []
        mixed = False
        for choices in rows:
            grouped = {}
            for position, row_choices in enumerate(choices):
                signatures = {self.vertex_cache[row].mode_indices for row, weight in row_choices
                              if weight != 0}
                if len(signatures) > 1:
                    mixed = True
                signature = next(iter(signatures), None)
                grouped.setdefault(signature, []).append(position)
            mode_groups.append(tuple(tuple(value) for value in grouped.values()))
        if mixed:
            return self.mixed_task(task, rows)
        blocks = []
        for positions, entries in structured_wick_templates(task, cache=self.wick_cache):
            products = tuple(MatrixWickProduct(weight, tuple(self.network(diagram) for diagram in diagrams))
                             for weight, diagrams in entries)
            selected_groups = [tuple(tuple(p for p in subset if p in allowed) for subset in groups)
                               for allowed, groups in zip(positions, mode_groups, strict=True)]
            for selection in product(*selected_groups):
                if all(selection):
                    blocks.append(MatrixWickBlock(selection, products))
        blocks = tuple(blocks)
        group = len(self.groups)
        self.groups.append(MatrixGroup(axes, blocks, task.partitions, task.partition_mode))
        return OutputNode('matrix', group, shape=task.shape)

    def mixed_task(self, task, rows):
        """Keep linear rows compact while each kernel has fixed physical modes."""
        choices_by_axis = []
        for labels, choices, time in zip(task.axes, rows, structured_axis_times(task), strict=True):
            signatures = dict.fromkeys(self.vertex_cache[row].mode_indices
                for position in choices for row, weight in position if weight != 0)
            variants = []
            for signature in signatures:
                filtered = tuple(tuple((row, weight) for row, weight in position
                    if weight != 0 and self.vertex_cache[row].mode_indices == signature)
                    for position in choices)
                axis = self.axis(tuple(item[0] for item in labels), filtered, time,
                                 selection=('modes', signature))
                variants.append((axis, tuple(i for i, position in enumerate(filtered) if position)))
            choices_by_axis.append(variants)
        templates = tuple((positions, tuple(MatrixWickProduct(weight,
            tuple(self.network(diagram) for diagram in diagrams)) for weight, diagrams in entries))
            for positions, entries in structured_wick_templates(task, cache=self.wick_cache))
        nodes = []
        for variants in product(*choices_by_axis):
            blocks = []
            for positions, products in templates:
                selection = tuple(tuple(p for p in selected if p in allowed)
                    for (_, selected), allowed in zip(variants, positions, strict=True))
                if all(selection):
                    blocks.append(MatrixWickBlock(selection, products))
            if blocks:
                group = len(self.groups)
                self.groups.append(MatrixGroup(tuple(axis for axis, _ in variants),
                    tuple(blocks), task.partitions, task.partition_mode))
                nodes.append(OutputNode('matrix', group, shape=task.shape))
        if not nodes:
            return OutputNode('zeros', 'complex128', shape=task.shape)
        return nodes[0] if len(nodes) == 1 else OutputNode('add', children=tuple(nodes), shape=task.shape)

    def visit(self, value):
        if isinstance(value, numbers.Number):
            return OutputNode('constant', complex(value))
        identity = id(value)
        if identity in self.active:
            raise ValueError('cyclic matrix request containers are unsupported')
        if identity in self.nodes:
            return self.nodes[identity][1]
        self.active.add(identity)
        try:
            if isinstance(value, StructuredCorrelator):
                node = self.task(value)
            elif isinstance(value, StructuredValue):
                node = OutputNode(value.kind, children=tuple(self.visit(child) for child in value.children),
                                  shape=value.shape)
            elif isinstance(value, Mapping):
                node = OutputNode('dict', keys=tuple(value), children=tuple(self.visit(v) for v in value.values()))
            elif isinstance(value, (tuple, list)):
                node = OutputNode('tuple' if isinstance(value, tuple) else 'list',
                                  children=tuple(self.visit(child) for child in value))
            elif isinstance(value, np.ndarray) and value.dtype == object:
                node = OutputNode('array', shape=value.shape, children=tuple(self.visit(v) for v in value.flat))
            else:
                raise NotImplementedError('matrix catalogs accept structured requests and numeric constants')
            self.nodes[identity] = value, node
            return node
        finally:
            self.active.remove(identity)


def describe_meson_matrix(correlator, *, bindings, representative_times=None):
    """Build a shared task catalog without cell/mode/component-pair expansion.

    representative_times maps symbolic labels to real slots for header metadata.
    The returned binding is ephemeral and has no executable plan or loader.
    """
    if not isinstance(bindings, DiagramBindings) or bindings.vertex_map is None:
        raise TypeError('describe_meson_matrix requires DiagramBindings with vertex_map')
    times = {} if representative_times is None else dict(representative_times)
    times = {normalize_time_key(label, name="representative_times key"): time
             for label, time in times.items()}
    if any(type(time) is not int or time < 0 for time in times.values()):
        raise ValueError('representative_times must map time keys to nonnegative slots')
    builder = _CatalogBuilder(bindings, times)
    output = builder.visit(correlator)
    catalog = MatrixCatalog(tuple(builder.components), tuple(builder.axes), tuple(builder.networks),
                            tuple(builder.groups), output)
    return BoundMatrixCatalog(catalog, tuple(builder.vertices), tuple(times.items()))


__all__ = ['describe_meson_matrix']
