"""Freeze the task's declared or inferred precision before resource planning."""
from dataclasses import replace
from math import prod
import numpy as np

from lqcdweave.contracts.correlator.api import ArrayConstant
from lqcdweave.contracts.correlator.precision import minimum_contraction_dtype, declared_contraction_dtype


class _PrecisionLoader:
    def __init__(self, physical, preparations, specs):
        self.physical = physical
        self.preparations, self.specs = preparations, specs

    def __getattr__(self, name):
        return getattr(self.physical, name)


def normalize_contraction_precision(expanded, *, dtype=None):
    """Describe conversions; raw read metadata and scientific identities stay raw.

    X performs each source conversion when its required block is materialized.
    No numerical reader, path search or full-dataset conversion runs here.
    """
    explicit_dtype = dtype is not None
    declared_dtype = declared_contraction_dtype(dtype if explicit_dtype else expanded.output_recipe.dtype)
    if expanded.inputs is None and declared_dtype is None:
        return expanded
    inputs_seen = {}
    def collect(inputs):
        if id(inputs) in inputs_seen:
            return
        inputs_seen[id(inputs)] = inputs
        for item in (*inputs.task_batches, *(v for group in inputs.batch_candidates.values() for v in group)):
            collect(item)
    if expanded.inputs is not None:
        collect(expanded.inputs)
    dtypes = [spec.dtype for inputs in inputs_seen.values() for spec in inputs.problem.specs.values()]
    loaders_seen = {id(inputs.loader): inputs.loader for inputs in inputs_seen.values()}
    dtypes.extend(dtype for loader in loaders_seen.values()
                  for dtype in getattr(loader, 'raw_dtypes', ()))
    dtypes.extend(request.metadata.dtype for inputs in inputs_seen.values()
                  for request in inputs.read_requests.values())
    pending, seen = [expanded.output_recipe.root], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node)); pending.extend(node.children)
        values = ((node.value,) if node.kind in {'constant','constant_array'} else
                  (node.value.coefficient,) if node.kind == 'term' else
                  tuple(coefficient for _, coefficient in node.value) if node.kind == 'diagram_sum' else ())
        if any((np.dtype(value.dtype).kind == 'c' if isinstance(value, ArrayConstant)
                else np.iscomplexobj(value)) for value in values):
            dtypes.append(np.dtype('complex128'))
    dtype = declared_dtype if declared_dtype is not None else minimum_contraction_dtype(dtypes)
    if dtype is None:
        return expanded
    if dtype.kind != 'c' and any(np.dtype(value).kind == 'c' for value in dtypes):
        raise TypeError('a real task dtype cannot discard complex input components')
    intermediate_dtype = {'float32': 'float64', 'complex64': 'complex128'}.get(dtype.name)
    if (expanded.output_recipe.dtype == dtype.name
            and expanded.output_recipe.intermediate_dtype == intermediate_dtype) and all(
           spec.dtype == dtype for inputs in inputs_seen.values()
           for spec in inputs.problem.specs.values()) and all(
            prep.recipe is not None and prep.recipe.compute_dtype == dtype.name
            for inputs in inputs_seen.values() for prep in inputs.preparations.values()) and all(
            join.core is None or join.core.spec.dtype == dtype
            for inputs in inputs_seen.values() for join in inputs.problem.baryon_joins):
        return expanded
    constants, loaders, converted = {}, {}, {}
    output_nodes = {}
    def output_node(node):
        if id(node) in output_nodes:
            return output_nodes[id(node)]
        value = node.value
        if explicit_dtype:
            if node.kind == 'constant':
                value = dtype.type(value).item()
            elif node.kind == 'constant_array':
                array = value.array() if isinstance(value, ArrayConstant) else np.asarray(value)
                value = ArrayConstant.from_array(array.astype(dtype, copy=False))
            elif node.kind == 'term':
                value = replace(value, coefficient=dtype.type(value.coefficient).item())
            elif node.kind == 'diagram_sum':
                value = tuple((key, dtype.type(coefficient).item()) for key, coefficient in value)
        result = replace(node, value=value, children=tuple(output_node(child) for child in node.children))
        output_nodes[id(node)] = result
        return result
    root = expanded.output_recipe.root
    if explicit_dtype:
        root = output_node(root)
    output_recipe = replace(expanded.output_recipe, root=root, dtype=dtype.name,
                            intermediate_dtype=intermediate_dtype)
    if expanded.inputs is None:
        return replace(expanded, output_recipe=output_recipe)
    assembly_bytes = 0
    if intermediate_dtype:
        from lqcdweave.contraction.results.api import estimate_output_retention_bytes
        representative = (expanded.inputs.task_batches[0] if expanded.inputs.task_batches
                          else expanded.inputs)
        uniform_specs = {key: replace(spec, dtype=dtype)
                         for key, spec in representative.problem.specs.items()}
        # The metadata-only assembly view must change its snapshot description
        # together with dtype, just like the final converted problem below.
        uniform_problem = replace(representative.problem, specs=uniform_specs,
            sources={key: replace(source, snapshot_bytes=prod(uniform_specs[key].shape)*dtype.itemsize)
                     if key in uniform_specs else source
                     for key, source in representative.problem.sources.items()})
        assembly_bytes = estimate_output_retention_bytes(output_recipe, uniform_problem,
            representative.batch_output_axes, expanded.coordinates)
    def specification(spec):
        return spec if spec.dtype == dtype else replace(spec, dtype=dtype)
    def constant(value):
        if value not in constants:
            constants[value] = ArrayConstant.from_array(value.array().astype(dtype, copy=False))
        return constants[value]
    def preparation(prep, source_spec, requests):
        recipe = prep.recipe
        if recipe is None:
            original_dtype = prep.dtype if prep.dtype is not None else (
                None if source_spec is None else source_spec.dtype)
            if original_dtype is not None and np.dtype(original_dtype) != dtype:
                raise NotImplementedError('minimum precision requires a described preparation recipe')
            return prep
        values = tuple(constant(value) for value in recipe.constants)
        coordinate_constants = tuple((position, argument, tuple(None if row is None else constant(row) for row in rows))
            for position, argument, rows in recipe.coordinate_constants)
        # Raw casting can overlap a prepared output. Keep original allowances
        # and add an explicit upper bound rather than shrinking unknown work.
        extra = sum(prod(requests[key].metadata.shape) * dtype.itemsize
                    for key in recipe.inputs if key in requests
                    and requests[key].metadata.dtype != dtype) if (
                        recipe.kind == 'spin_spatial' or
                        recipe.kind == 'identity') else 0
        gpu = expanded.problem.backend == 'cupy'
        return replace(prep, dtype=dtype.name,
            recipe=replace(recipe, constants=values, compute_dtype=dtype.name,
                           coordinate_constants=coordinate_constants),
            workspace_host_bytes=prep.workspace_host_bytes + (0 if gpu else extra),
            workspace_device_bytes=prep.workspace_device_bytes + (extra if gpu else 0))
    def convert(inputs):
        if id(inputs) in converted:
            return converted[id(inputs)]
        problem = inputs.problem
        if (problem.sparse_bridges
                and any(spec.dtype != dtype for spec in problem.specs.values())):
            raise NotImplementedError('precision conversion for sparse records needs a uniform input dataset')
        loader = inputs.loader
        if id(loader) not in loaders:
            raw_specs = dict(getattr(loader, 'specs', {})) | dict(problem.specs)
            raw_preps = dict(getattr(loader, 'preparations', {})) | dict(inputs.preparations)
            specs = {key: specification(spec) for key, spec in raw_specs.items()}
            preps = {key: preparation(prep, raw_specs.get(key), inputs.read_requests)
                     for key, prep in raw_preps.items()}
            loaders[id(loader)] = _PrecisionLoader(loader, preps, specs)
        loader = loaders[id(loader)]
        specs = {key: specification(spec) for key, spec in problem.specs.items()}
        def sparse_entries(entries):
            if entries is None:
                return None
            return tuple((i, j, k, complex(dtype.type(weight)))
                         for i, j, k, weight in entries)
        joins = []
        for join in problem.baryon_joins:
            core = join.core
            if core is not None:
                core = replace(core, spec=specification(core.spec),
                    components=tuple(replace(part, sparse_spin=sparse_entries(part.sparse_spin),
                                             coefficient=(dtype.type(part.coefficient).item()
                                                          if explicit_dtype else part.coefficient))
                                     for part in core.components))
            joins.append(replace(join, core=core, sparse_spin=sparse_entries(join.sparse_spin)))
        sources = {key: (replace(source, snapshot_bytes=prod(specs[key].shape)*dtype.itemsize)
                         if key in specs else source)
                   for key, source in problem.sources.items()}
        result = replace(inputs, problem=replace(problem, specs=specs, sources=sources,
                                                baryon_joins=tuple(joins)),
            loader=loader, preparations={key: preparation(prep, problem.specs.get(key)
                    or getattr(inputs.loader,'specs',{}).get(key), inputs.read_requests)
                                         for key, prep in inputs.preparations.items()},
            matrix_kernels=(), task_batches=tuple(convert(item) for item in inputs.task_batches),
            assembly_host_bytes=max(inputs.assembly_host_bytes, assembly_bytes),
            assembly_device_bytes=max(inputs.assembly_device_bytes,
                assembly_bytes if problem.backend == 'cupy' else 0),
            batch_candidates={width: tuple(convert(item) for item in group)
                              for width, group in inputs.batch_candidates.items()})
        converted[id(inputs)] = result
        return result
    return replace(expanded, inputs=convert(expanded.inputs),
                   output_recipe=output_recipe)
