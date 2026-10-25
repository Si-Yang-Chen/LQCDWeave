"""Bind the shared catalog to existing DiagramInputs, without cell expansion.

Physical endpoint/header checks and delayed read/materialize callbacks use the
same registry as ordinary Diagram expansion. No representative Diagram, scalar
Term or component-pair expansion is constructed. Coefficient maps stay explicit.
"""
from dataclasses import replace
from math import prod
from itertools import product as cartesian_product
import hashlib
import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, BoundMatrixCatalog, DiagramBindings, ContractionProblem, ContractionTerm,
    DiagramInputs, ExpandedCorrelator, OutputNode, OutputRecipe, TensorSpec,
    SourceSpec, SourcePreparation, TermContribution, MatrixKernelSpec, TensorDescription,
    describe_coordinates, CoordinateSpec,
    ArrayConstant, TensorPreparation,
    MesonWholeCore, MesonWholeKernel,
)
from lqcdweave.contraction.results.api import estimate_output_retention_bytes
from lqcdweave.contraction.binding.api import bind_slot_vertices
from .physical_inputs import PhysicalInputRegistry
from .legacy_diagram import _lookup_propagator, _resolve_time, _validate_configuration, _referenced_propagators
from .state_graph import lower_state_graph
from .monomial_spin import MonomialPreparations, monomial_columns, unit_spin


def _product(nodes):
    nodes = tuple(nodes)
    return nodes[0] if len(nodes)==1 else OutputNode('multiply',children=nodes)


def _sum(nodes, shape):
    nodes = tuple(nodes)
    if not nodes:
        return OutputNode('zeros','complex128',shape=shape)
    return nodes[0] if len(nodes)==1 else OutputNode('add',children=nodes)


def _catalog_jobs(catalog):
    jobs, networks = {}, {}

    def axis_components(axis_id, positions):
        axis = catalog.axes[axis_id]
        return tuple(sorted({int(component) for position in positions for component in
            axis.component_ids[axis.offsets[position]:axis.offsets[position+1]]}))

    def job(group, block, network_id):
        network = catalog.networks[network_id]
        slots = tuple(int(slot) for slot in network.vertex_slots)
        if slots not in ((0,),(1,),(0,1)) or any(slot>=len(group.axes) for slot in slots):
            raise NotImplementedError('matrix templates require an ordinary connected pair or independent loops')
        selections = tuple((group.axes[slot],tuple(map(int,block.axis_positions[slot]))) for slot in slots)
        key = network_id,selections,group.partitions,group.partition_mode
        if key not in jobs:
            components = tuple(axis_components(axis,positions) for axis,positions in selections)
            if any(not items for items in components):
                return None
            if len(network.time_labels) != len(slots):
                raise ValueError('matrix network time and vertex counts disagree')
            if network_id not in networks:
                lowered = lower_state_graph(network.adjacency,
                    [{'left':'v','right':'v'} for _ in slots])
                expected = AxisPattern(
                    ((0,1,2,3),(4,5,6,7),(1,4,3,6),(5,0,7,2))
                    if len(slots)==2 else ((0,1,2,3),(1,0,3,2)), ())
                if len(lowered)!=1 or lowered[0].equation.normalized()!=expected.normalized():
                    raise NotImplementedError('matrix network has a different axis relation')
                networks[network_id] = lowered[0]
            for items,time in zip(components,network.time_labels,strict=True):
                if any(catalog.components[item].time_label!=time for item in items):
                    raise ValueError('matrix component time role differs from its network')
            jobs[key] = (network,slots,selections,components,networks[network_id])
        return key

    block_jobs = {}
    for group_id,group in enumerate(catalog.groups):
        if len(group.axes) not in (1,2):
            raise NotImplementedError('matrix templates require one or two operator axes')
        for block_id,block in enumerate(group.wick_blocks):
            for product_id,product in enumerate(block.products):
                block_jobs[group_id,block_id,product_id] = tuple(job(group,block,net) for net in product.networks)
    return jobs, block_jobs


class MatrixCatalogBinder:
    """One cfg's reusable matrix construction, with fixed-coordinate binding.

    Input-role jobs and read-only coefficient arrays belong to this binder. Each
    bind checks actual endpoint metadata and returns fresh source contracts and
    a loader; source arrays and read handles are not cached between coordinates.
    The binder is runtime-only and is not a second execution-plan format.
    """
    def __init__(self, described, *, bindings, row_block=1, _factorize_spin=False,
                 _whole_vertices=False):
        if not isinstance(described,BoundMatrixCatalog) or not isinstance(bindings,DiagramBindings):
            raise TypeError('matrix binding requires BoundMatrixCatalog and DiagramBindings')
        if type(row_block) is not int or row_block <= 0:
            raise ValueError('row_block must be a positive integer')
        bindings = replace(bindings, propagator_map=_referenced_propagators(
            ((network.adjacency, network.propagators) for network in described.catalog.networks),
            bindings.propagator_map or {}))
        described = replace(described, component_bindings=bind_slot_vertices(
            described.component_bindings, (bindings.propagator_map or {}).values()))
        self.described, self.bindings, self.row_block = described, bindings, row_block
        self._factorize_spin = _factorize_spin
        self._whole_vertices = _whole_vertices
        self.jobs, self.block_jobs = _catalog_jobs(described.catalog)
        self.constants = {}

    def bind(self, *, coordinates=None):
        described, bindings = self.described, self.bindings
        if coordinates is None:
            times = dict(described.representative_times)
            coordinates,batch = describe_coordinates(time_axes=times) if times else describe_coordinates()
        else:
            _,batch = describe_coordinates()
        if coordinates.scan_shape and len(coordinates.points)>1:
            raise NotImplementedError('matrix catalog binding currently accepts one fixed coordinate')
        if not coordinates.points:
            raise ValueError('matrix binding has no selected coordinate')
        vertices = described.component_bindings
        _validate_configuration(bindings.configuration,vertices,bindings.propagator_map or {})
        if any(vertex.xp is not np for vertex in vertices):
            raise NotImplementedError('matrix catalog materialization currently requires NumPy')
        base = PhysicalInputRegistry(bindings, coordinate_values=dict(coordinates.points[0].values))
        return _bind_matrix_base(described,bindings,coordinates,batch,self.row_block,
                                 base,self.jobs,self.block_jobs,self.constants,self._factorize_spin,
                                 self._whole_vertices)

    def iter_bind(self, coordinates):
        """Bind coordinates on demand, sharing only roles and coefficient arrays.

        Each result is a fixed-coordinate ExpandedCorrelator with the original
        point key. No prior result or future loader is retained by this iterator.
        The caller chooses its execution window and owns numeric lifetimes.
        """
        if not isinstance(coordinates, CoordinateSpec):
            raise TypeError('matrix coordinate iteration requires CoordinateSpec')
        return self._iter_bind(coordinates)

    def _iter_bind(self, coordinates):
        for point in coordinates.points:
            selected = CoordinateSpec('bound', (), (), (), (point,), ())
            yield self.bind(coordinates=selected)


def bind_meson_matrix_catalog(described, *, bindings, coordinates=None, row_block=1,
                             _factorize_spin=False, _whole_vertices=False):
    """Bind one fixed coordinate through the reusable catalog constructor."""
    return MatrixCatalogBinder(described,bindings=bindings,row_block=row_block,
        _factorize_spin=_factorize_spin, _whole_vertices=_whole_vertices).bind(coordinates=coordinates)


def _bind_matrix_base(described, bindings, coordinates, batch, row_block,
                      base, jobs, block_jobs, shared_constants, factorize_spin=False,
                      whole_vertices=False):
    catalog = described.catalog
    specs,sources,preps,constants = {},{},{},{}
    terms,cores,term_ids = [],[],[]
    whole_cores = []
    stack_keys,constant_keys,job_nodes = {},{},{}
    monomial = MonomialPreparations(base)

    time_values = dict(coordinates.points[0].values)
    factor_metadata, selected_vertices, spin_columns, spin_recipes = {}, {}, {}, {}
    registered_edges, unit_spins = {}, {}

    def factor(component, selection=None):
        identity = component, selection
        if identity not in factor_metadata:
            vertex = described.component_bindings[component]
            if selection is not None:
                # Reuse the original structured mode-view rule, including
                # the source dagger's swap from logical to physical slots.
                selected_vertices[identity] = vertex.with_mode_indices(selection)
                vertex = selected_vertices[identity]
            time = _resolve_time(catalog.components[component].time_label,time_values)
            descriptions = vertex.describe_factors(time,'v2v',indices=(0,),
                usedNe=bindings.usedNe,usedNp=bindings.usedNp)
            if len(descriptions)!=1 or descriptions[0].tensor.sparse_layout!='dense':
                raise NotImplementedError('matrix component must describe one unweighted dense factor')
            factor_metadata[identity] = descriptions[0]
        return factor_metadata[identity]

    def source(key):
        specs[key],sources[key],preps[key] = base.specs[key],base.sources[key],base.preparations[key]
        return key

    def stack(keys):
        keys = tuple(keys)
        if keys not in stack_keys:
            key = f'matrix:vertices:{len(stack_keys)}'
            layouts = tuple(base.specs[item] for item in keys)
            if any(item.shape!=layouts[0].shape for item in layouts):
                raise ValueError('matrix components have different physical layouts')
            descriptions = tuple(item.description for item in layouts)
            first = descriptions[0]
            description = replace(first,axes=('meson_component',*first.axes),
                data_identity=None,selection_identity=None,
                source_bindings=tuple(dict.fromkeys(block for item in descriptions for block in item.source_bindings)),
                transform=('stack_mesons',0),
                semantics={'members':tuple((s.role,s.axes,s.transform,s.semantics) for s in descriptions)},
                bindings={'members':tuple((s.data_identity,s.selection_identity,s.bindings) for s in descriptions)})
            spec = TensorSpec((len(keys),*layouts[0].shape),np.result_type(*(item.dtype for item in layouts)),description=description)
            size = prod(spec.shape)*spec.dtype.itemsize
            dependencies = tuple(dict.fromkeys(block for item in keys for block in base.preparations[item].blocks))
            preparations = tuple(base.preparations[item] for item in keys)
            specs[key],sources[key] = spec,SourceSpec(key,0,snapshot_bytes=size,read_bytes=0,
                metadata={'family':'matrix_vertices','component_tensors':keys})
            preps[key] = SourcePreparation(dependencies,
                workspace_host_bytes=size+max(p.workspace_host_bytes+prod(s.shape)*s.dtype.itemsize
                    for p,s in zip(preparations,layouts,strict=True)),
                flops=sum(p.flops for p in preparations),
                traffic_bytes=2*size+sum(p.traffic_bytes for p in preparations),
                kernel_calls=len(keys)+sum(p.kernel_calls for p in preparations),dtype=spec.dtype.name,
                recipe=TensorPreparation('stack',keys))
            stack_keys[keys] = key
        return stack_keys[keys]

    def constant(identity, value, axes=('matrix_operator','meson_component')):
        if identity not in constant_keys:
            key = f'matrix:coefficient:{len(constant_keys)}'
            entry = shared_constants.get(identity)
            value = np.array(value,copy=True) if entry is None else entry[0]
            value.flags.writeable = False
            signature = (hashlib.sha256(repr((value.shape,value.dtype.str)).encode()+value.tobytes()).hexdigest()
                         if entry is None else entry[1])
            shared_constants.setdefault(identity,(value,signature))
            spec = TensorSpec(value.shape,value.dtype,description=TensorDescription(axes=axes,
                role='coefficient',transform=('matrix_coefficient',),
                semantics={'definition_digest':signature}))
            specs[key],sources[key] = spec,SourceSpec(key,0,snapshot_bytes=value.nbytes,read_bytes=0)
            preps[key] = SourcePreparation((),dtype=spec.dtype.name,
                recipe=TensorPreparation('constant',constants=(ArrayConstant.from_array(value),)))
            constants[key] = value
            constant_keys[identity] = key
        return constant_keys[identity]

    def weights(axis_id, positions, components):
        identity = ('axis',axis_id,positions,components)
        if identity in constant_keys:
            return constant_keys[identity]
        if identity in shared_constants:
            return constant(identity,shared_constants[identity][0])
        axis = catalog.axes[axis_id]
        column = {component:i for i,component in enumerate(components)}
        value = np.zeros((len(positions),len(components)),dtype=np.complex128)
        for row,position in enumerate(positions):
            for index in range(axis.offsets[position],axis.offsets[position+1]):
                component = int(axis.component_ids[index])
                if component in column:
                    value[row,column[component]] += axis.coefficients[index]
        return constant(identity,value)

    def block_node(key, mode_selections=None):
        if whole_vertices:
            # This is a representation proof on 4x4 frozen constants, not a
            # global component contraction search. Nonunit spin remains inside
            # the same whole kernel's dense representation.
            components = jobs[key][3]
            if not factorize_spin:
                return component_block_node(key, mode_selections)
            separated = True
            for vertex, items in enumerate(components):
                selection = None if mode_selections is None else mode_selections[vertex]
                for component in items:
                    selected = component, selection
                    descriptor = factor(component, selection)
                    binding = described.component_bindings[component]
                    if selected not in spin_recipes:
                        spin_recipes[selected] = binding.describe_factor_preparation(
                            descriptor, 'unbound', compact_structure=False, embed_structure=True)
                    if (not unit_spin(spin_recipes[selected])
                            or prod(descriptor.tensor.shape) < 2):
                        separated = False
                        break
                if not separated:
                    break
            return component_block_node(key, mode_selections, component_axis=separated)
        if not factorize_spin:
            return component_block_node(key, mode_selections)
        identity = ('monomial', key[:3], mode_selections)
        cached = job_nodes.get(identity)
        if cached is not None:
            return cached
        network,slots,selections,components,lowered = jobs[key]
        for vertex, items in enumerate(components):
            for component in items:
                selection = None if mode_selections is None else mode_selections[vertex]
                selected = component, selection
                descriptor = factor(component, selection)
                binding = described.component_bindings[component]
                if selected not in spin_recipes:
                    spin_recipes[selected] = binding.describe_factor_preparation(descriptor, 'unbound', compact_structure=False, embed_structure=True)
                recipe = spin_recipes[selected]
                # A scalar spatial block has no GEMM work to share and cannot
                # amortize the small-spin producer/terminal-wide workspace.
                if selected not in unit_spins:
                    unit_spins[selected] = unit_spin(recipe)
                if not unit_spins[selected] or prod(descriptor.tensor.shape) < 2:
                    break
                registered = base.register_dense_vertex(binding, descriptor,
                    time_role=catalog.components[component].time_label, compact_structure=False, embed_structure=True)
                monomial.separate(registered)
            else:
                continue
            break
        else:
            # Components are an explicit sum axis, not a Cartesian set of
            # separate Terms. Keep their exact phase and operator coefficient
            # attached until the same wide mode/spin/component boundary.
            job_nodes[identity] = component_block_node(key, mode_selections, component_axis=True)
            return job_nodes[identity]
        groups = []
        for vertex, items in enumerate(components):
            choices = {}
            for component in items:
                selection = None if mode_selections is None else mode_selections[vertex]
                selected = component, selection
                if selected not in spin_columns:
                    descriptor = factor(component, selection)
                    binding = described.component_bindings[component]
                    if selected not in spin_recipes:
                        spin_recipes[selected] = binding.describe_factor_preparation(descriptor, 'unbound', compact_structure=False, embed_structure=True)
                    spin_columns[selected] = monomial_columns(spin_recipes[selected])
                columns = spin_columns[selected]
                if columns is None:
                    job_nodes[identity] = component_block_node(key, mode_selections)
                    return job_nodes[identity]
                choices.setdefault(columns, []).append(component)
            groups.append(tuple((columns, tuple(items)) for columns, items in choices.items()))
        children = [component_block_node(key, mode_selections, chosen)
                    for chosen in cartesian_product(*groups)]
        node, slots, shape = children[0]
        job_nodes[identity] = _sum(tuple(child[0] for child in children), shape), slots, shape
        return job_nodes[identity]

    def component_block_node(key, mode_selections=None, spin_groups=None, component_axis=False):
        # Output modes share the same physical block computations.
        identity = key[:3], mode_selections, spin_groups, component_axis
        if identity in job_nodes:
            return job_nodes[identity]
        network,slots,selections,components,lowered = jobs[key]
        if spin_groups is not None:
            components = tuple(items for _, items in spin_groups)
        reference = tuple(factor(items[0], None if mode_selections is None else mode_selections[i])
                          for i, items in enumerate(components))
        edges = tuple(request for request in lowered.requests if request.kind=='propagator')

        def edge_key(request, selected):
            if request.tensor_label>=len(network.propagators):
                raise ValueError('matrix propagator label is outside its role table')
            symbol = network.propagators[request.tensor_label]
            reader = _lookup_propagator(bindings.propagator_map,symbol)
            source, sink = selected[request.source_vertex], selected[request.sink_vertex]
            roles = network.time_labels[request.source_vertex], network.time_labels[request.sink_vertex]
            # factor_metadata keeps these actual, checked descriptors alive for
            # this bind. Equal shapes never prove endpoint identity. Retaining
            # reader, symbol and roles also preserves every read alias/header.
            identity = (id(reader), str(symbol), request.sector, id(source), id(sink),
                        tuple(None if role is None else str(role) for role in roles))
            cached = registered_edges.get(identity)
            if cached is None:
                cached = base.register_propagator(reader,symbol,source,sink,request.sector,
                    source_role=roles[0],sink_role=roles[1])
                registered_edges[identity] = cached
            return cached

        reference_edges = tuple(edge_key(request,reference) for request in edges)
        stack_sources, full_shapes, spin_sources, phase_matrices, spatial_positions = [], [], [], [], []
        for vertex,items in enumerate(components):
            members = []
            for component in items:
                descriptor = factor(component, None if mode_selections is None else mode_selections[vertex])
                selected = list(reference)
                selected[vertex] = descriptor
                if tuple(edge_key(request,selected) for request in edges)!=reference_edges:
                    raise ValueError('matrix component changes a propagator endpoint or physical selection')
                members.append(base.register_dense_vertex(described.component_bindings[component],descriptor,
                    time_role=catalog.components[component].time_label, compact_structure=False, embed_structure=True))
            full_shapes.append((len(members), *base.specs[members[0]].shape))
            if component_axis:
                spaces = tuple(monomial.separate(member) for member in members)
                unique = tuple(dict.fromkeys(spaces))
                spatial_positions.append(tuple(unique.index(space) for space in spaces))
                phase_matrices.append(np.stack([base.preparations[member].recipe.constants[0].array()
                                                for member in members]))
                stack_sources.append(stack(unique))
                continue
            if spin_groups is not None:
                members = tuple(monomial.compress(member, spin_groups[vertex][0]) for member in members)
            stack_sources.append(stack(members))
        weight_sources = [weights(axis,positions,items) for (axis,positions),items in zip(selections,components,strict=True)]
        kind = 'connected' if len(slots)==2 else 'loop'
        full_layouts = tuple(base.specs[item].shape for item in reference_edges) + tuple(full_shapes) + tuple(specs[item].shape for item in weight_sources)
        dtypes = tuple(base.specs[item].dtype.name for item in reference_edges) + tuple(specs[item].dtype.name for item in (*stack_sources,*weight_sources))
        core = MatrixKernelSpec(kind, full_layouts, dtypes,
            None,row_block,mode_block=(mode_selections is not None or
                                      any(item.mode_indices is not None for item in reference)))
        equation = core.equation
        if component_axis:
            count = len(reference_edges)
            labels = tuple((axes[0], *axes[-2:]) if count <= i < count+len(components) else axes
                           for i, axes in enumerate(equation.inputs))
            phase_labels, contribution_labels = [], []
            fresh = max(axis for axes in (*equation.inputs, equation.output) for axis in axes) + 1
            for vertex, (weight, matrices) in enumerate(zip(weight_sources, phase_matrices, strict=True)):
                # Retain each original coefficient contribution separately.
                # Equal spatial parents share a source, but potentially
                # cancelling projection terms are never pre-summed/cast here.
                weights0, positions = constants[weight], spatial_positions[vertex]
                entries = {}
                for row, component in zip(*np.nonzero(weights0), strict=True):
                    for field, bar in zip(*np.nonzero(matrices[component]), strict=True):
                        entries.setdefault((row, positions[component], field, bar), []).append(
                            weights0[row,component] * matrices[component,field,bar])
                multiplicity = max(map(len, entries.values()))
                shape = (weights0.shape[0], max(positions)+1, 4, 4)
                value = np.zeros((*shape, multiplicity), dtype=np.result_type(weights0, matrices))
                for at, values in entries.items():
                    value[at][:len(values)] = values
                phase_axes = (equation.inputs[count+len(components)+vertex][0], *equation.inputs[count+vertex][:3])
                names = ('matrix_operator', 'meson_component', 'spin_field', 'spin_bar')
                if multiplicity == 1:
                    value = value[...,0]
                else:
                    phase_axes = (*phase_axes, fresh)
                    contribution_labels.append(fresh)
                    fresh += 1
                    names = (*names, 'coefficient_contribution')
                phase_labels.append(phase_axes)
                spin_sources.append(constant(('component-phase', ArrayConstant.from_array(value)), value, names))
            phase_labels = tuple(phase_labels)
            equation = AxisPattern((*labels[:count+len(components)], *phase_labels), equation.output)
            inputs = [source(item) for item in reference_edges] + stack_sources + spin_sources
        elif spin_groups is not None:
            count = len(reference_edges)
            merges = {}
            orders = {}
            for vertex, (columns, _) in enumerate(spin_groups):
                _, field, bar, *_ = equation.inputs[count+vertex]
                merges[bar] = field
                orders[bar] = columns
            edges = []
            for key0, labels in zip(reference_edges, equation.inputs[:count], strict=True):
                permutations = tuple((axis, orders[label]) for axis, label in enumerate(labels) if label in orders)
                edges.append(monomial.gather(key0, permutations))
            inputs = [source(item) for item in edges] + stack_sources + weight_sources
            labels = []
            for index, axes in enumerate(equation.inputs):
                if count <= index < count + len(spin_groups):
                    axes = axes[:2] + axes[3:]
                labels.append(tuple(merges.get(axis, axis) for axis in axes))
            equation = AxisPattern(tuple(labels), equation.output)
        else:
            inputs = [source(item) for item in reference_edges] + stack_sources + weight_sources
        term_id = f'matrix:term:{len(terms)}'
        barriers = ()
        if component_axis:
            spatial = tuple(dict.fromkeys(axis for axes in labels[count:count+len(components)] for axis in axes[1:]))
            coupled = tuple(dict.fromkeys(axis for axes in phase_labels for axis in axes[1:]))
            barriers = ((spatial, coupled), *(((axis,), (*spatial, *(a for a in coupled if a != axis)))
                                             for axis in coupled))
        if whole_vertices:
            logical_keys = []
            representation = 'unit-spin' if component_axis else 'dense'
            for vertex, ((axis, positions), full) in enumerate(zip(selections, full_shapes, strict=True)):
                logical = f'matrix:whole:{len(whole_cores)}'
                count = len(reference_edges)
                dependencies = (inputs[count + vertex], inputs[count + len(components) + vertex])
                rows = tuple((axis, int(position)) for position in positions)
                whole_cores.append(MesonWholeCore(logical, dependencies, rows, representation))
                specs[logical] = TensorSpec((len(rows), *full[1:]),
                    np.result_type(*(specs[item].dtype for item in dependencies)),
                    description=TensorDescription(
                        axes=('matrix_operator', 'spin_field', 'spin_bar', 'mode_left', 'mode_right'),
                        role='elemental', transform=('whole-meson-matrix',),
                        semantics={'coefficient_owner': 'vertex-core', 'structure': 'general',
                                   'independent_output_rows': rows}))
                logical_keys.append(logical)
            count = len(reference_edges)
            outer = AxisPattern((*core.equation.inputs[:count], *(
                (core.equation.inputs[count + len(components) + i][0],
                 *core.equation.inputs[count + i][1:]) for i in range(len(components)))),
                core.equation.output)
            kernel = MesonWholeKernel(kind, tuple(logical_keys), tuple(inputs), equation, representation)
            terms.append(ContractionTerm((*reference_edges, *logical_keys), outer, kernel=kernel))
            cores.append(None)
        else:
            terms.append(ContractionTerm(tuple(inputs),equation, sum_barriers=barriers))
            cores.append(core if spin_groups is None and not component_axis else None)
        term_ids.append(term_id)
        node = OutputNode('term',TermContribution(term_id))
        job_nodes[identity] = node,slots,core.output_shape
        return job_nodes[identity]

    def network_node(key):
        if key[2] is None:
            return block_node(key)
        if key in job_nodes:
            return job_nodes[key]
        boundaries = key[2]
        ranges = tuple(tuple(range(a,b)) for a,b in zip(boundaries,boundaries[1:]))
        if key[3] == 'memory':
            slots = jobs[key][1]
            leaves = []
            for chosen in cartesian_product(ranges, repeat=2*len(slots)):
                selections = tuple((chosen[2*i],chosen[2*i+1]) for i in range(len(slots)))
                leaf,_,shape = block_node(key,selections)
                leaves.append(leaf)
            node = _sum(leaves,shape)
            job_nodes[key] = node,slots,shape
            return job_nodes[key]
        pairs = tuple((a,b) for a in range(len(ranges)) for b in range(a,len(ranges)))
        slots = jobs[key][1]
        cells, output_shape = [], None
        for position in cartesian_product(pairs, repeat=len(slots)):
            orientations = tuple((((a,b),1.0),) if a==b else
                                 (((a,b),1/np.sqrt(2)),((b,a),1/np.sqrt(2)))
                                 for a,b in position)
            summands = []
            for chosen in cartesian_product(*orientations):
                selections = tuple((ranges[a],ranges[b]) for (a,b),_ in chosen)
                leaf,_,output_shape = block_node(key,selections)
                weight = prod(w for _,w in chosen)
                summands.append(leaf if weight==1 else _product((OutputNode('constant',weight),leaf)))
            cells.append(_sum(summands,output_shape))
        # PC projection combines small block results, never full-Ne masks.
        pc_shape = (len(pairs),)*len(slots)
        packed = OutputNode('array',children=tuple(cells),shape=pc_shape)
        permutation = tuple(range(len(slots),2*len(slots)))+tuple(range(len(slots)))
        shape = (*output_shape,*pc_shape)
        node = OutputNode('transpose',permutation,children=(packed,),shape=shape)
        job_nodes[key] = node,slots,shape
        return job_nodes[key]

    shared_output_groups = set()
    def shared_outputs(node):
        if node.kind == 'add':
            shared_output_groups.update(child.value for child in node.children if child.kind == 'matrix')
        for child in node.children:
            shared_outputs(child)
    shared_outputs(catalog.output)
    replacements = {}
    for group_id,group in enumerate(catalog.groups):
        shape = tuple(len(catalog.axes[axis].labels) for axis in group.axes)
        pc = (None if group.partitions is None or group.partition_mode == 'memory' else
              len(group.partitions)*(len(group.partitions)-1)//2)
        if pc is not None:
            shape += (pc,)*len(group.axes)
        blocks = []
        for block_id,block in enumerate(group.wick_blocks):
            products = []
            block_shape = tuple(len(axis) for axis in block.axis_positions)+shape[len(group.axes):]
            for product_id,product in enumerate(block.products):
                keys = block_jobs[group_id,block_id,product_id]
                if not keys or any(key is None for key in keys):
                    continue
                factors, covered = [],[]
                for key in keys:
                    leaf,slots,output_shape = network_node(key)
                    covered += list(slots)
                    if len(group.axes)==2 and len(slots)==1:
                        leading = (output_shape[0],1) if slots==(0,) else (1,output_shape[0])
                        target = leading if pc is None else (*leading,*((pc,1) if slots==(0,) else (1,pc)))
                        leaf = OutputNode('reshape',children=(leaf,),shape=target)
                    factors.append(leaf)
                if sorted(covered)!=list(range(len(group.axes))):
                    raise NotImplementedError('Wick product does not cover the declared operator axes exactly once')
                if product.coefficient != 1:
                    factors.insert(0,OutputNode('constant',product.coefficient))
                products.append(_product(factors))
            if products:
                node = _sum(products,block_shape)
                positions = tuple(tuple(map(int,axis)) for axis in block.axis_positions)
                if (group_id in shared_output_groups or
                        positions!=tuple(tuple(range(len(catalog.axes[a].labels))) for a in group.axes)):
                    node = OutputNode('scatter',positions,children=(node,),shape=shape)
                blocks.append(node)
        replacements[group_id] = _sum(blocks,shape)

    cache = {}
    def placements(node):
        if node.kind == 'scatter':
            return (node,)
        if node.kind == 'add':
            children = tuple(placements(child) for child in node.children)
            if all(child is not None for child in children):
                return tuple(item for child in children for item in child)
        return None

    def output(node):
        identity = id(node)
        if identity not in cache:
            cache[identity] = (replacements[node.value] if node.kind=='matrix' else
                               replace(node,children=tuple(output(child) for child in node.children)))
            # Physical-mode variants share one output allocation, including
            # overlapping placements from symmetric/antisymmetric linear rows.
            flattened = placements(cache[identity]) if node.kind == 'add' else None
            if flattened and all(item.shape == flattened[0].shape for item in flattened):
                cache[identity] = _sum(flattened, flattened[0].shape)
        return cache[identity]
    root = output(catalog.output)
    recipe = OutputRecipe(root,tuple(term_ids),coordinates)
    problem = None if not terms else ContractionProblem(specs,tuple(terms),sources,
        single_read_regions=True, meson_cores=tuple(whole_cores))
    bound_inputs = None
    if problem is not None:
        physical_loader = base.loader()
        physical_loader.preparations = dict(base.preparations, **preps)
        physical_loader.specs = dict(base.specs, **specs)
        physical_loader.constant_values = {
            key: (value,) for key, value in constants.items()
        }
        used_blocks = {block for prep in preps.values() for block in prep.blocks}
        bound_inputs = DiagramInputs(problem,{key:base.blocks[key] for key in used_blocks},preps,
            physical_loader,coordinates,batch,tuple(term_ids),matrix_kernels=tuple(cores) if all(core is not None for core in cores) else (),
            read_requests={key:value for key,value in base.read_requests.items() if key in used_blocks},
            tensor_requests={key:request for key,request in base.tensor_requests.items() if key in specs},
            assembly_host_bytes=(estimate_output_retention_bytes(recipe,problem,(),coordinates)
                +sum(value.nbytes for value in constants.values())+sum(axis.nbytes for axis in catalog.axes)))
    return ExpandedCorrelator(catalog,bound_inputs,recipe,bindings,coordinates,batch,adapter=described)


__all__ = ['bind_meson_matrix_catalog', 'MatrixCatalogBinder']
