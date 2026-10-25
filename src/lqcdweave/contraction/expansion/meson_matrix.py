"""Compile declared dense meson matrices into shared row-vector Terms.

Scientific recipes and primitive metadata reuse structured._Compiler and the
native binding provider. Only metadata probes are lowered; the Cartesian
component-pair contraction list is never constructed. Numerical IO remains
with the native loader and the existing execution/storage protocol.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from collections import defaultdict

import numpy as np

from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.results.api import (
    collect_diagrams, compose_output_recipe, estimate_output_retention_bytes,
)
from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionTerm, ExpandedCorrelator, OutputNode, SourceSpec,
    SourcePreparation, TensorSpec, TermContribution,
)
from lqcdweave.operators.correlator.api import iter_structured_wick_entries
from .structured import _Compiler, _frozen, _sum, _product, contains_structured
from .diagram_expansion import expand_correlator
from .native import NativeDiagramBinding


def _topology(diagram):
    return (_frozen(diagram.diagram.adjacency_matrix), tuple(diagram.time_list),
            tuple(diagram.propagator_list))


class _RowCompiler(_Compiler):
    def __init__(self, bindings):
        super().__init__(bindings)
        self.pools = {}
        self.members = {}
        self.rows = {}
        self.row_keys = {}
        self.choice_sets = {}
        self.column_sets = {}
        self.column_identities = {}
        self.row_nodes = {}
        self.select_nodes = {}
        self.topologies = {}

    def topology(self, diagram):
        key = id(diagram)
        if key not in self.topologies:
            # Retain the object so its address cannot be reused during expansion.
            self.topologies[key] = diagram, _topology(diagram)
        return self.topologies[key][1]

    def visit(self, value):
        if not contains_structured(value) and collect_diagrams(value):
            raise NotImplementedError('meson_matrix requires user-declared structured matrix groups')
        return super().visit(value)

    def choices(self, row, selection):
        key = row, selection
        if key not in self.choice_sets:
            vertex = self.vertex(row)
            if not isinstance(vertex, Meson) or vertex.available_sectors() not in (None, ('v2v',)):
                raise NotImplementedError('meson_matrix requires dense V/V Meson vertices')
            self.choice_sets[key] = super().choices(row, selection)
        return self.choice_sets[key]

    def task(self, task):
        # Group boundaries are the user-declared StructuredCorrelator axes.
        pools = defaultdict(dict)
        for _, entries in iter_structured_wick_entries(task):
            for _, diagrams in entries:
                for diagram in diagrams:
                    if len(diagram.vertex_list) == 2:
                        pools[self.topology(diagram)][diagram.vertex_list[1]] = None
                    elif len(diagram.vertex_list) != 1:
                        raise NotImplementedError('meson_matrix requires single-meson two-point matrices')
        previous = self.pools
        self.pools = {key: tuple(rows) for key, rows in pools.items()}
        try:
            return super().task(task)
        finally:
            self.pools = previous

    def diagram(self, diagram, selections):
        topology = self.topology(diagram)
        vertices = tuple(diagram.vertex_list)
        if len(diagram.vertex_list) == 1:
            key = topology, vertices, selections
            if key not in self.nodes:
                self.nodes[key] = _sum(_product((self.primitive(diagram, (vertex,)),), coefficient)
                    for vertex, coefficient in self.choices(vertices[0], selections[0]))
            return self.nodes[key]
        pool = self.pools[topology]
        key = topology, vertices, selections, pool
        if key in self.nodes:
            return self.nodes[key]
        left = self.choices(diagram.vertex_list[0], selections[0])
        right = self.choices(diagram.vertex_list[1], selections[1])
        if not left or not right:
            self.nodes[key] = OutputNode('constant', 0)
            return self.nodes[key]
        column_key = pool, selections[1]
        if column_key not in self.column_sets:
            columns = tuple(dict.fromkeys(vertex for row in pool
                            for vertex, _ in self.choices(row, selections[1])))
            if columns not in self.column_identities:
                self.column_identities[columns] = (len(self.column_identities), columns,
                                                   {column: i for i, column in enumerate(columns)})
            self.column_sets[column_key] = self.column_identities[columns]
        column_id, columns, positions = self.column_sets[column_key]
        if not columns or not left:
            raise ValueError('meson_matrix cannot compile empty component axes')
        member_key = topology, selections, column_id
        if member_key not in self.members:
            self.members[member_key] = tuple(
                self.primitive(diagram, (left[0][0], column)).value.term_id
                for column in columns)
        probes = self.members[member_key]
        summands = []
        for vertex, left_coefficient in left:
            row_key = member_key, vertex
            if row_key not in self.row_keys:
                probe = self.primitive(diagram, (vertex, columns[0])).value.term_id
                row_id = f'row:{len(self.rows)}'
                self.row_keys[row_key] = row_id
                self.rows[row_id] = probe, probes
                self.row_nodes[row_id] = OutputNode('term', TermContribution(row_id))
            row_id = self.row_keys[row_key]
            for column, coefficient in right:
                selected_key = row_id, positions[column]
                if selected_key not in self.select_nodes:
                    self.select_nodes[selected_key] = OutputNode(
                        'select', (positions[column],), (self.row_nodes[row_id],))
                summands.append(_product((self.select_nodes[selected_key],),
                                         left_coefficient * coefficient))
        self.nodes[key] = _sum(summands)
        return self.nodes[key]


class _RowLoader:
    """Add sequential stack materialization to the native slab loader."""
    def __init__(self, loader):
        self.loader = loader
        self.active = 0
        self.stacks = {}

    def __getattr__(self, name):
        return getattr(self.loader, name)

    @contextmanager
    def activate(self, index):
        previous, self.active = self.active, index
        activate = getattr(self.loader, 'activate', None)
        try:
            with activate(index) if activate else nullcontext():
                yield self
        finally:
            self.active = previous

    def materialize(self, key, source, raw):
        stack = self.stacks[self.active].get(key)
        if stack is None:
            return self.loader.materialize(key, source, raw)
        axis, spec, members = stack
        value = np.empty(spec.shape, dtype=spec.dtype)
        for index, (member_key, member_source, dependencies) in enumerate(members):
            member = self.loader.materialize(member_key, member_source,
                                            {block: raw[block] for block in dependencies})
            selection = [slice(None)] * len(spec.shape)
            selection[axis] = index
            value[tuple(selection)] = member
            del member
        return value


def _one_term(node):
    if node.kind == 'term':
        return node.value
    if node.kind == 'diagram_sum' and len(node.value) == 1:
        return TermContribution(*node.value[0])
    raise NotImplementedError('meson_matrix requires one dense V/V Term per primitive component')


def _used_basis(node):
    if node.kind == 'term' and node.value.term_id.startswith('basis:'):
        yield node.value.term_id
    for child in node.children:
        yield from _used_basis(child)


def expand_meson_matrix(correlator, *, adapter, bindings, time_axes=None,
                        time_pairs=None, batch='auto'):
    """Lower explicit structured matrix groups without scalar pair expansion."""
    if not contains_structured(correlator) or not isinstance(adapter, NativeDiagramBinding):
        raise NotImplementedError('meson_matrix requires structured=True and native Meson bindings')
    compiler = _RowCompiler(bindings)
    root = compiler.visit(correlator)
    base = expand_correlator(compiler.diagrams, adapter=adapter,
                            bindings=replace(bindings, vertex_map=compiler.vertices),
                            time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    if base.inputs is None:
        raise ValueError('meson_matrix requires a nonempty matrix')
    if base.problem.backend != 'numpy':
        raise NotImplementedError('compact meson_matrix stack materialization currently supports NumPy')
    if base.problem.sparse_bridges or base.problem.baryon_joins:
        raise NotImplementedError('meson_matrix requires dense V/V inputs without bridges or baryon joins')
    contributions = {f'basis:{i}': _one_term(node)
                     for i, node in enumerate(base.output_recipe.root.children)}
    original_indices = {term_id: i for i, term_id in enumerate(base.inputs.term_ids)}
    loops = tuple(dict.fromkeys(_used_basis(root)))
    term_ids = (*compiler.rows, *loops)
    replacements = {key: OutputNode('term', TermContribution(key)) for key in term_ids}
    wrappers, converted = {}, {}

    def convert(inputs):
        if id(inputs) in converted:
            return converted[id(inputs)]
        problem = inputs.problem
        specs, sources, preparations, terms, output_axes = {}, {}, {}, [], []
        stacks, stack_ids = {}, {}
        for row_id, (probe, members) in compiler.rows.items():
            index = original_indices[contributions[probe].term_id]
            term = problem.terms[index]
            if term.coefficient != 1 or contributions[probe].coefficient != 1:
                raise NotImplementedError('primitive dense components must have unit weights')
            member_terms = [problem.terms[original_indices[contributions[item].term_id]] for item in members]
            if any(item.inputs[:2] != term.inputs[:2] or len(item.inputs) != 4
                   or item.subscripts != term.subscripts or item.coefficient != 1
                   or contributions[probe_id].coefficient != 1
                   for item, probe_id in zip(member_terms, members, strict=True)):
                raise NotImplementedError('matrix columns have different propagator or axis bindings')
            member_keys = tuple(item.inputs[3] for item in member_terms)
            pattern = term.subscripts
            axis = len(pattern.inputs[3]) - 4
            signature = member_keys, axis
            if signature not in stack_ids:
                stack_id = f'meson-columns:{len(stack_ids)}'
                stack_ids[signature] = stack_id
                member_specs = [problem.specs[key] for key in member_keys]
                first = member_specs[0]
                if any(item.shape != first.shape or item.dtype != first.dtype for item in member_specs):
                    raise ValueError('matrix column components must have matching shapes and precision')
                descriptions = tuple(item.description for item in member_specs)
                description = replace(first.description,
                    axes=(*first.axes[:axis], 'meson_component', *first.axes[axis:]),
                    data_identity=None, selection_identity=None,
                    source_bindings=tuple(dict.fromkeys(block for item in descriptions for block in item.source_bindings)),
                    transform=('stack_mesons', axis),
                    semantics={'members': tuple((item.role, item.axes, item.transform, item.semantics)
                                               for item in descriptions)},
                    bindings={'members': tuple((item.data_identity, item.selection_identity, item.bindings)
                                               for item in descriptions)})
                spec = TensorSpec((*first.shape[:axis], len(member_keys), *first.shape[axis:]),
                                  first.dtype, description=description)
                size = int(np.prod(spec.shape)) * spec.dtype.itemsize
                source = SourceSpec(stack_id, 0, snapshot_bytes=size,
                                    metadata={'family': 'meson-columns', 'members': member_keys, 'axis': axis})
                member_preps = [inputs.preparations[key] for key in member_keys]
                prep = SourcePreparation(
                    tuple(dict.fromkeys(block for item in member_preps for block in item.blocks)),
                    workspace_host_bytes=size + max(item.workspace_host_bytes +
                        int(np.prod(member_spec.shape)) * member_spec.dtype.itemsize
                        for item, member_spec in zip(member_preps, member_specs, strict=True)),
                    flops=sum(item.flops for item in member_preps),
                    traffic_bytes=sum(item.traffic_bytes for item in member_preps) + 2 * size,
                    kernel_calls=sum(item.kernel_calls for item in member_preps) + len(member_keys),
                    dtype=str(spec.dtype))
                specs[stack_id], sources[stack_id], preparations[stack_id] = spec, source, prep
                stacks[stack_id] = axis, spec, tuple((key, problem.sources[key], inputs.preparations[key].blocks)
                                                   for key in member_keys)
            stack_id = stack_ids[signature]
            component_axis = max(
                (axis_id for input_axes in pattern.inputs for axis_id in input_axes),
                default=-1,
            ) + 1
            stacked_inputs = (
                *pattern.inputs[:3],
                (*pattern.inputs[3][:axis], component_axis,
                 *pattern.inputs[3][axis:]),
            )
            terms.append(ContractionTerm(
                (*term.inputs[:3], stack_id),
                AxisPattern(stacked_inputs, (*pattern.output, component_axis)),
            ))
            output_axes.append(inputs.batch_output_axes[index] if inputs.batch_output_axes else None)
        for loop in loops:
            index = original_indices[contributions[loop].term_id]
            terms.append(problem.terms[index])
            output_axes.append(inputs.batch_output_axes[index] if inputs.batch_output_axes else None)
            replacements[loop] = OutputNode('term', TermContribution(loop, contributions[loop].coefficient))
        required = tuple(dict.fromkeys(key for term in terms for key in term.inputs))
        for key in required:
            if key not in stacks:
                specs[key], sources[key], preparations[key] = problem.specs[key], problem.sources[key], inputs.preparations[key]
        new_problem = replace(problem, specs=specs, sources=sources, terms=tuple(terms))
        wrapper = wrappers.setdefault(id(inputs.loader), _RowLoader(inputs.loader))
        wrapper.stacks[inputs.coordinate_batch.batch_index if inputs.coordinate_batch else 0] = stacks
        result = replace(inputs, problem=new_problem, preparations=preparations, loader=wrapper,
                         term_ids=term_ids, diagram_terms={}, diagram_recipes={},
                         tensor_requests={key: value for key, value in inputs.tensor_requests.items() if key in specs},
                         batch_bindings={key: value for key, value in inputs.batch_bindings.items() if key in specs},
                         batch_output_axes=tuple(output_axes) if inputs.coordinate_batch else (),
                         assembly_host_bytes=0, assembly_device_bytes=0,
                         batch_topology=('meson-rows-v1', tuple((term.inputs, term.subscripts.normalized()) for term in terms)))
        converted[id(inputs)] = result
        return result

    original = base.inputs
    if original.task_batches:
        candidates = {width: tuple(convert(item) for item in items)
                      for width, items in original.batch_candidates.items()}
        batches = tuple(convert(item) for item in original.task_batches)
        first = batches[0]
        inputs = replace(first, coordinate_batch=None, batch_bindings={}, batch_output_axes=(),
                         task_batches=batches, batch_candidates=candidates)
    else:
        inputs = convert(original)
    recipe = compose_output_recipe(root, replacements, term_ids, base.coordinates)
    problem = inputs.task_batches[0].problem if inputs.task_batches else inputs.problem
    axes = inputs.task_batches[0].batch_output_axes if inputs.task_batches else inputs.batch_output_axes
    reserve = estimate_output_retention_bytes(recipe, problem, axes, base.coordinates)
    def reserve_batch(item):
        return replace(item, assembly_host_bytes=reserve)
    candidates = {width: tuple(reserve_batch(item) for item in items)
                  for width, items in inputs.batch_candidates.items()}
    selected = next((width for width, items in inputs.batch_candidates.items()
                     if len(items) == len(inputs.task_batches) and all(
                         left is right for left, right in zip(items, inputs.task_batches, strict=True))), None)
    batches = candidates[selected] if selected is not None else tuple(reserve_batch(item) for item in inputs.task_batches)
    inputs = replace(inputs, assembly_host_bytes=reserve, task_batches=batches, batch_candidates=candidates)
    return ExpandedCorrelator(correlator, inputs, recipe, bindings, base.coordinates, base.batch, adapter)


__all__ = ['expand_meson_matrix']
