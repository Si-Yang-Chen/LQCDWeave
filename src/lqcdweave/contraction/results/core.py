"""Explicit, inspectable output recipes for nested correlator expressions."""

from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
from types import SimpleNamespace

import numpy as np
from .matrix_output import (MATRIX_OUTPUT_KINDS, evaluate_matrix_output,
                            estimate_matrix_output, validate_matrix_output,
                            scatter_sum_blocks, evaluate_scatter_sum)

from lqcdweave.contracts.execution import TermValue

from lqcdweave.contracts.correlator.api import (
    CoordinateSpec,
    OutputNode,
    OutputRecipe,
    TermContribution,
)


def collect_diagrams(value):
    """Return Diagram atoms once, preserving first appearance order."""
    from lqcdweave.operators.correlator.api import Diagram
    from sympy import Add, Mul, Pow

    found, seen = [], set()
    active, visited = set(), {}

    def visit_container(item, children):
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic correlator containers are not supported")
        if identity in visited:
            return
        active.add(identity)
        try:
            for child in children:
                visit(child)
        finally:
            active.remove(identity)
        visited[identity] = item

    def visit(item):
        if isinstance(item, Diagram):
            if item not in seen:
                seen.add(item)
                found.append(item)
        elif isinstance(item, np.ndarray):
            if item.dtype == object:
                visit_container(item, item.flat)
        elif isinstance(item, Mapping):
            visit_container(item, item.values())
        elif isinstance(item, (list, tuple)):
            visit_container(item, item)
        elif isinstance(item, (Add, Mul)):
            if id(item) in visited:
                return
            for child in item.args:
                visit(child)
            visited[id(item)] = item
        elif isinstance(item, Pow):
            if id(item) in visited:
                return
            visit(item.base)
            visited[id(item)] = item
        elif not _is_number(item):
            raise TypeError(f"unsupported correlator leaf {type(item).__name__}")

    visit(value)
    return tuple(found)


def build_output_recipe(correlator, diagram_terms, term_ids, coordinates, *, diagram_recipes=None):
    """Translate symbolic algebra and Python structure into explicit nodes."""
    from lqcdweave.operators.correlator.api import Diagram
    from sympy import Add, Mul, Number, Pow

    term_ids = tuple(term_ids)
    known = set(term_ids)
    diagram_terms = {} if diagram_terms is None else diagram_terms
    diagram_recipes = {} if diagram_recipes is None else diagram_recipes
    active, symbolic_nodes, validated_recipes = set(), {}, set()

    def build_container(item, kind, children, *, keys=(), shape=()):
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic correlator containers are not supported")
        active.add(identity)
        try:
            nodes = tuple(build(child) for child in children)
        finally:
            active.remove(identity)
        return OutputNode(kind, keys=tuple(keys), shape=tuple(shape), children=nodes)

    def build(item):
        if isinstance(item, (Diagram, Add, Mul, Number, Pow)):
            identity = id(item)
            if identity not in symbolic_nodes:
                symbolic_nodes[identity] = item, build_item(item)
            return symbolic_nodes[identity][1]
        return build_item(item)

    def build_item(item):
        if isinstance(item, Diagram):
            if item in diagram_recipes:
                node = diagram_recipes[item]
                if id(node) not in validated_recipes:
                    _validate_diagram_recipe(node, known)
                    validated_recipes.add(id(node))
                return node
            try:
                contributions = diagram_terms[item]
            except KeyError as exc:
                raise ValueError(f"Diagram has no expanded term mapping: {item}") from exc
            if not contributions:
                raise ValueError("a Diagram must map to at least one ContractionTerm")
            if any(not isinstance(value, TermContribution) or value.term_id not in known
                   for value in contributions):
                raise ValueError("diagram mapping contains an unknown TermContribution")
            return OutputNode("diagram_sum", tuple((value.term_id, _python_number(value.coefficient))
                                                     for value in contributions))
        if getattr(item, 'is_number', False):
            return OutputNode("constant", _python_number(item))
        if isinstance(item, Add):
            return OutputNode("add", children=tuple(build(child) for child in item.args))
        if isinstance(item, Mul):
            return OutputNode("multiply", children=tuple(build(child) for child in item.args))
        if isinstance(item, Pow):
            if not item.exp.is_Integer:
                raise TypeError("correlator powers must have integer exponents")
            return OutputNode("power", int(item.exp), (build(item.base),))
        if isinstance(item, Mapping):
            return build_container(item, "dict", item.values(), keys=item.keys())
        if isinstance(item, list):
            return build_container(item, "list", item)
        if isinstance(item, tuple):
            return build_container(item, "tuple", item)
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                return build_container(item, "array", item.flat, shape=item.shape)
            constant = np.array(item, copy=True)
            constant.setflags(write=False)
            return OutputNode("constant_array", constant, shape=tuple(item.shape))
        if _is_number(item):
            return OutputNode("constant", _python_number(item))
        raise TypeError(f"unsupported correlator leaf {type(item).__name__}")

    return OutputRecipe(build(correlator), term_ids, coordinates)


def assemble_correlator(expanded, term_values):
    """Evaluate a result recipe and restore its requested coordinate axes.

    This function evaluates only the supplied recipe and values. It performs
    no data reads, planning, contractions, or backend transfers.
    """
    if not isinstance(expanded.output_recipe, OutputRecipe):
        raise TypeError("expanded correlator has no OutputRecipe")
    coordinates = expanded.coordinates
    recipe = expanded.output_recipe
    if not isinstance(coordinates, CoordinateSpec):
        raise TypeError("ExpandedCorrelator must contain a CoordinateSpec")
    if recipe.coordinates != coordinates:
        raise ValueError("OutputRecipe coordinates differ from ExpandedCorrelator coordinates")
    _validate_recipe_contents(recipe, coordinates)
    try:
        items = tuple(term_values)
    except TypeError as exc:
        raise TypeError("term_values must be an iterable of TermValue objects") from exc
    values = {}
    for item in items:
        if not isinstance(item, TermValue):
            raise TypeError("term_values must contain TermValue objects")
        key = (item.term_id, item.coordinate_key)
        if key in values:
            raise ValueError(f"duplicate term result for {key}")
        values[key] = item.value
    expected = {(term_id, point.key)
                for term_id in recipe.term_ids
                for point in coordinates.points}
    if set(values) != expected:
        missing, extra = expected - set(values), set(values) - expected
        raise ValueError(
            "term result coverage mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    return _evaluate_output_recipe(
        recipe, coordinates,
        lambda term_id, coordinate_key: values[(term_id, coordinate_key)],
    )


def bind_output_assembly(recipe, *, debug=False):
    """Bind one fixed-coordinate recipe for repeated ordered-value assembly."""
    if debug:
        if type(debug) is not bool:
            raise TypeError("debug must be bool")
        if not isinstance(recipe, OutputRecipe):
            raise TypeError("recipe must be an OutputRecipe")
        coordinates = recipe.coordinates
        if not isinstance(coordinates, CoordinateSpec):
            raise TypeError("OutputRecipe must contain a CoordinateSpec")
        if coordinates.scan_shape or len(coordinates.points) != 1:
            raise ValueError("bind_output_assembly requires fixed single-point coordinates")
        _validate_recipe_contents(recipe, coordinates)
        point_key = coordinates.points[0].key
        expanded = SimpleNamespace(output_recipe=recipe, coordinates=coordinates)
        value_count = len(recipe.term_ids)
        def assemble(values):
            ordered_values = _ordered_values(values, value_count)
            term_values = tuple(
                TermValue(term_id, point_key, value)
                for term_id, value in zip(recipe.term_ids, ordered_values, strict=True)
            )
            return assemble_correlator(expanded, term_values)
        return assemble

    from .bound import bind_fixed_output
    term_indexes = {term_id: index for index, term_id in enumerate(recipe.term_ids)}
    return bind_fixed_output(recipe, term_indexes)


def _validate_recipe_contents(recipe, coordinates):
    if not isinstance(recipe.term_ids, tuple):
        raise TypeError("OutputRecipe term_ids must be a tuple")
    if any(not isinstance(term_id, str) or not term_id for term_id in recipe.term_ids):
        raise ValueError("OutputRecipe term_ids must be nonempty strings")
    if len(recipe.term_ids) != len(set(recipe.term_ids)):
        raise ValueError("OutputRecipe term_ids must be unique")
    _validate_output_node(recipe.root, set(recipe.term_ids))
    coordinate_keys = tuple(point.key for point in coordinates.points)
    if (not coordinate_keys or any(not isinstance(key, str) or not key for key in coordinate_keys)
            or len(coordinate_keys) != len(set(coordinate_keys))):
        raise ValueError("CoordinateSpec must contain unique nonempty point keys")
    expected_point_count = math.prod(coordinates.scan_shape) if coordinates.scan_shape else 1
    if len(coordinates.points) != expected_point_count:
        raise ValueError("CoordinateSpec point count disagrees with scan_shape")
    return coordinate_keys


def _ordered_values(values, expected_count):
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError("ordered term values must be iterable") from exc
    if len(items) != expected_count:
        raise ValueError(
            f"ordered term value count mismatch; expected={expected_count}, actual={len(items)}"
        )
    return items


def _evaluate_output_recipe(recipe, coordinates, value_at, *, trusted=False):
    """Evaluate a validated recipe against a term/value lookup."""

    selected_values = {}
    computation_dtype = recipe.intermediate_dtype or recipe.dtype
    dtype = None if computation_dtype is None else np.dtype(computation_dtype)
    number = _python_number if dtype is None else lambda value: dtype.type(_python_number(value))
    scalarize = _scalarize if dtype is None else lambda value: np.asarray(value, dtype=dtype)[()]
    zero, one = (0, 1) if dtype is None else (dtype.type(0), dtype.type(1))

    def evaluate(node):
        if node.kind == "diagram_sum":
            return _assemble_coordinate_values(
                [sum((number(coefficient) * scalarize(value_at(term_id, point.key))
                      for term_id, coefficient in node.value), zero)
                 for point in coordinates.points],
                coordinates, dtype=dtype,
            )
        if node.kind in MATRIX_OUTPUT_KINDS:
            identity = id(node)
            if identity not in selected_values:
                selected_values[identity] = evaluate_matrix_output(
                    node, evaluate, coordinates.scan_shape, trusted=trusted, dtype=dtype
                )
            return selected_values[identity]
        if node.kind == "term":
            contribution = node.value
            return _assemble_coordinate_values(
                [number(contribution.coefficient)
                 * scalarize(value_at(contribution.term_id, point.key))
                 for point in coordinates.points], coordinates, dtype=dtype
            )
        if node.kind == "constant":
            return _broadcast_constant(number(node.value), coordinates)
        if node.kind == "constant_array":
            value = _constant_array(node.value)
            return _broadcast_constant(value if dtype is None else value.astype(dtype), coordinates)
        if node.kind == "add":
            blocks = scatter_sum_blocks(node)
            if blocks is not None:
                return evaluate_scatter_sum(blocks, evaluate, coordinates.scan_shape,
                                            trusted=trusted, zero=zero)
            return sum((evaluate(child) for child in node.children), zero)
        if node.kind == "multiply":
            result = one
            for child in node.children:
                result = result * evaluate(child)
            return result
        if node.kind == "power":
            value = evaluate(node.children[0])
            return value ** node.value if dtype is None else np.power(value, node.value, dtype=dtype)
        if node.kind == "select":
            child = node.children[0]
            identity = id(child)
            if identity not in selected_values:
                selected_values[identity] = evaluate(child)
            return selected_values[identity][node.value]
        if node.kind == "dict":
            return {key: evaluate(child) for key, child in zip(node.keys, node.children, strict=True)}
        if node.kind == "list":
            return [evaluate(child) for child in node.children]
        if node.kind == "tuple":
            return tuple(evaluate(child) for child in node.children)
        if node.kind == "array":
            children = [evaluate(child) for child in node.children]
            if not children:
                return np.empty(node.shape, dtype=object)
            nested_container_kinds = {"array", "constant_array", "dict", "list", "tuple"}
            has_nested_container = any(child.kind in nested_container_kinds
                                        for child in node.children)
            if not has_nested_container and _uniform_array_values(children):
                stacked = np.stack(children, axis=0)
                return stacked.reshape((*node.shape, *stacked.shape[1:]))
            out = np.empty(node.shape, dtype=object)
            for index, value in zip(np.ndindex(node.shape), children, strict=True):
                out[index] = value
            return out
        raise ValueError(f"unknown OutputRecipe node kind {node.kind!r}")

    value = evaluate(recipe.root)
    return _cast_output_fields(value, recipe.dtype) if recipe.intermediate_dtype else value


def _cast_output_fields(value, dtype):
    """Apply the stored boundary cast without flattening user containers."""
    if isinstance(value, dict):
        return {key: _cast_output_fields(child, dtype) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cast_output_fields(child, dtype) for child in value)
    if isinstance(value, np.ndarray) and value.dtype.kind == 'O':
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = _cast_output_fields(value[index], dtype)
        return result
    converted = np.asarray(value, dtype=dtype)
    return converted if isinstance(value, np.ndarray) else converted[()]


def assemble_workload_outputs(workload, values, backend):
    """Apply only the explicitly stored Meson product recipe, in order."""
    if workload.metadata.get("adapter") != "meson-simple-v1":
        return tuple(values)
    outputs = []
    for recipe in workload.metadata["output_recipes"]:
        value = 1.0
        for term_index in recipe:
            value = value * values[term_index]
        outputs.append(backend.asarray(value))
    result = backend.stack(outputs) if outputs else backend.asarray([])
    if backend.__name__ == "cupy":
        backend.cuda.get_current_stream().synchronize()
    return result


def _constant_array(value):
    from lqcdweave.contracts.correlator.api import ArrayConstant
    return value.array() if isinstance(value,ArrayConstant) else np.asarray(value)


def freeze_output_recipe(recipe):
    """Store literal output arrays as explicit read-only definition constants."""
    from dataclasses import replace
    from lqcdweave.contracts.correlator.api import ArrayConstant
    nodes={}
    def visit(node):
        if id(node) not in nodes:
            value=ArrayConstant.from_array(_constant_array(node.value)) if node.kind=='constant_array' else node.value
            nodes[id(node)]=replace(node,value=value,children=tuple(visit(c) for c in node.children))
        return nodes[id(node)]
    return replace(recipe,root=visit(recipe.root))


def _output_metadata(recipe, problem, batch_output_axes, coordinates, *, fields=False):
    """Estimate array payload bytes for task TermValues and recipe outputs.

    Repeated Diagram occurrences are counted as repeated recipe nodes. This
    intentionally overestimates NumPy payload temporaries during nested
    arithmetic so every per-compute-batch plan reserves outputs accumulated by
    the whole task. Python/container and ndarray object headers are excluded;
    this is a storage-planning estimate, not an RSS upper bound.
    """
    if not isinstance(recipe, OutputRecipe):
        raise TypeError("recipe must be an OutputRecipe")
    terms = tuple(problem.terms)
    term_ids = tuple(recipe.term_ids)
    if len(terms) != len(term_ids):
        raise ValueError("output recipe term IDs do not match the contraction problem")
    axes = tuple(batch_output_axes) or tuple(None for _ in terms)
    if len(axes) != len(terms):
        raise ValueError("batch output axes must align with contraction Terms")
    point_count = len(coordinates.points)
    term_shapes, term_dtypes = {}, {}
    for index, (term_id, term, batch_axis) in enumerate(zip(term_ids, terms, axes, strict=True)):
        pattern = term.subscripts
        labels = pattern.inputs
        if len(labels) != len(term.inputs):
            raise ValueError(f"term {term_id!r} input labels do not match its operands")
        dimensions, dtypes = {}, []
        for key, tensor_labels in zip(term.inputs, labels, strict=True):
            spec = problem.specs[key]
            if len(tensor_labels) != len(spec.shape):
                raise ValueError(f"term {term_id!r} labels do not match TensorSpec {key!r}")
            dimensions.update(zip(tensor_labels, spec.shape, strict=True))
            dtypes.append(spec.dtype)
        output_labels = list(pattern.output)
        if batch_axis is not None:
            if batch_axis >= len(output_labels):
                raise ValueError(f"term {term_id!r} batch axis lies outside its output")
            output_labels.pop(batch_axis)
        term_shapes[term_id] = tuple(dimensions[label] for label in output_labels)
        term_dtypes[term_id] = np.result_type(*dtypes)

    term_values_bytes = point_count * sum(
        math.prod(term_shapes[term_id]) * term_dtypes[term_id].itemsize
        for term_id in term_ids
    )
    scan_shape = tuple(coordinates.scan_shape)
    pointer_bytes = np.dtype(object).itemsize

    def result_bytes(shape, dtype):
        return math.prod(shape) * np.dtype(dtype).itemsize

    def broadcast_shape(shapes):
        nonempty = [tuple(shape) for shape in shapes]
        if not nonempty:
            return ()
        try:
            return np.broadcast_shapes(*nonempty)
        except ValueError:
            # Let actual assembly raise the algebra error; reserve the largest
            # output shape in the meantime without undercounting its storage.
            return max(nonempty, key=lambda shape: math.prod(shape))

    estimates = {}
    computation_dtype = recipe.intermediate_dtype or recipe.dtype

    def numeric_dtype(dtype):
        return np.dtype(recipe.intermediate_dtype) if recipe.intermediate_dtype else np.dtype(dtype)

    def estimate(node):
        key = id(node)
        if key not in estimates:
            result = estimate_node(node)
            if computation_dtype is not None and np.dtype(result[2]).kind in 'biufc':
                result = result[0], result[1], np.dtype(computation_dtype)
            estimates[key] = result
        return estimates[key]

    def estimate_node(node):
        blocks = scatter_sum_blocks(node)
        if blocks is not None:
            children = [estimate(block.children[0]) for block in blocks]
            dtype = np.result_type(*(child[2] for child in children))
            dtype = (np.empty(0, dtype=dtype) + 0).dtype
            shape = (*blocks[0].shape, *scan_shape)
            size = result_bytes(shape, dtype)
            indices = sum(sum(map(len, block.value)) for block in blocks) * np.dtype(np.intp).itemsize
            largest = max((result_bytes(child[1], child[2]) for child in children), default=0)
            # Compact cached children, gather/add scratch, and two full owners
            # cover a dtype promotion without N full-size scatter temporaries.
            return sum(child[0] for child in children) + 2*size + 2*largest + indices, shape, dtype
        if node.kind in MATRIX_OUTPUT_KINDS:
            if node.kind == 'zeros' and recipe.intermediate_dtype:
                shape = (*node.shape, *scan_shape)
                dtype = np.dtype(recipe.intermediate_dtype)
                return result_bytes(shape, dtype), shape, dtype
            return estimate_matrix_output(node,estimate,scan_shape)
        if node.kind == "term":
            contribution = node.value
            shape = (*term_shapes[contribution.term_id], *scan_shape)
            dtype = numeric_dtype(np.result_type(term_dtypes[contribution.term_id],
                                   np.asarray(contribution.coefficient).dtype))
            size = result_bytes(shape, dtype)
            return (2 * size if recipe.intermediate_dtype else size), shape, dtype
        if node.kind == "diagram_sum":
            entries = tuple(node.value)
            shapes = [(*term_shapes[term_id], *scan_shape) for term_id, _ in entries]
            dtypes = [numeric_dtype(np.result_type(term_dtypes[term_id], np.asarray(coefficient).dtype))
                      for term_id, coefficient in entries]
            shape = broadcast_shape(shapes)
            dtype = np.result_type(*dtypes)
            size = result_bytes(shape, dtype)
            # Coefficient products for array-valued terms may coexist until the
            # coordinate stack is made; scalar terms need only the sum output.
            temporary = sum(result_bytes(item_shape, item_dtype)
                            for item_shape, item_dtype in zip(shapes, dtypes, strict=True)
                            if math.prod(item_shape) > 1)
            return ((2 * temporary + 2 * size) if recipe.intermediate_dtype
                    else temporary + size), shape, dtype
        if node.kind == "constant":
            if not scan_shape:
                return 0, (), numeric_dtype(_constant_array(node.value).dtype)
            dtype = numeric_dtype(_constant_array(node.value).dtype)
            shape = scan_shape
            size = result_bytes(shape, dtype)
            return size, shape, dtype
        if node.kind == "constant_array":
            value = _constant_array(node.value)
            shape = (*value.shape, *scan_shape)
            dtype = numeric_dtype(value.dtype)
            size = result_bytes(shape, dtype)
            return size, shape, dtype
        if node.kind in {"dict", "list", "tuple"}:
            children = [estimate(child) for child in node.children]
            size = len(node.children) * pointer_bytes
            return sum(item[0] for item in children) + size, (), np.dtype(object)
        if node.kind == "array":
            children = [estimate(child) for child in node.children]
            nested = any(child.kind in {"array", "constant_array", "dict", "list", "tuple"}
                         for child in node.children)
            if nested:
                size = math.prod(node.shape) * pointer_bytes
                return sum(item[0] for item in children) + size, node.shape, np.dtype(object)
            child_shapes = [item[1] for item in children]
            child_dtypes = [item[2] for item in children]
            shape = broadcast_shape(child_shapes)
            dtype = np.result_type(*child_dtypes) if child_dtypes else np.dtype(object)
            stacked_shape = (*node.shape, *shape)
            # Mixed scalar/array children produce an object array instead of a
            # dense stack; reserve both layouts so the estimate covers either.
            size = (result_bytes(stacked_shape, dtype)
                    + math.prod(node.shape) * pointer_bytes)
            return sum(item[0] for item in children) + size, stacked_shape, dtype
        if node.kind in {"add", "multiply"}:
            children = [estimate(child) for child in node.children]
            shape = broadcast_shape([item[1] for item in children])
            dtype = np.result_type(*(item[2] for item in children))
            size = result_bytes(shape, dtype)
            operations = (len(children) if recipe.intermediate_dtype
                          else max(len(children) - 1, 0))
            return sum(item[0] for item in children) + operations * size, shape, dtype
        if node.kind == "power":
            child = estimate(node.children[0])
            size = result_bytes(child[1], child[2])
            return child[0] + size, child[1], child[2]
        if node.kind == "select":
            child = estimate(node.children[0])
            if (len(node.value) > len(child[1]) - len(scan_shape)
                    or any(index >= size for index, size in zip(node.value, child[1]))):
                raise ValueError("select indexes outside the leading result axes")
            shape = child[1][len(node.value):]
            return child[0], shape, child[2]
        raise ValueError(f"unknown OutputRecipe node kind {node.kind!r}")

    recipe_bytes = estimate(recipe.root)[0]
    if not fields and not recipe.intermediate_dtype:
        return term_values_bytes + recipe_bytes
    leaves=[]
    def visit(node,path):
        if node.kind in {'dict','list','tuple'}:
            keys=node.keys if node.kind=='dict' else range(len(node.children))
            for key,child in zip(keys,node.children,strict=True): visit(child,(*path,key))
        elif node.kind=='array' and estimate(node)[2].kind=='O':
            for index,child in enumerate(node.children):
                visit(child,(*path,tuple(map(int,np.unravel_index(index,node.shape)))))
        else:
            _,shape,dtype=estimate(node)
            if dtype.kind not in 'biufc': raise ValueError('output packing requires numeric leaves')
            if recipe.intermediate_dtype:
                dtype = np.dtype(recipe.dtype)
            leaves.append((path,tuple(shape),dtype.str))
    visit(recipe.root,())
    result = tuple(leaves)
    if recipe.intermediate_dtype:
        # Final narrow fields may coexist with every wide arithmetic owner.
        recipe_bytes += sum(result_bytes(shape, dtype) for _, shape, dtype in result)
    if not fields: return term_values_bytes + recipe_bytes
    return (term_values_bytes + recipe_bytes, result) if fields == 'layout' else result


def estimate_output_retention_bytes(recipe,problem,batch_output_axes,coordinates):
    return _output_metadata(recipe,problem,batch_output_axes,coordinates)


def describe_output_fields(recipe,problem):
    """Freeze numeric field layouts without constructing any output arrays."""
    return _output_metadata(recipe,problem,(),recipe.coordinates,fields=True)


def describe_output_layout(recipe, problem):
    """Return retention bytes and numeric fields from one metadata traversal."""
    return _output_metadata(recipe, problem, (), recipe.coordinates, fields='layout')


def _validate_output_node(node, term_ids):
    if not isinstance(node, OutputNode):
        raise TypeError("OutputRecipe children must be OutputNode objects")
    if not isinstance(node.children, tuple):
        raise TypeError("OutputNode children must be a tuple")
    if node.kind in MATRIX_OUTPUT_KINDS:
        validate_matrix_output(node,term_ids,_validate_output_node)
    elif node.kind == "diagram_sum":
        if not isinstance(node.value, tuple) or not node.value:
            raise ValueError("diagram_sum must contain at least one (term_id, coefficient) pair")
        for pair in node.value:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("diagram_sum entries must be (term_id, coefficient) pairs")
            term_id, coefficient = pair
            if term_id not in term_ids:
                raise ValueError(f"diagram_sum references unknown term_id {term_id!r}")
            if not _is_number(coefficient):
                raise TypeError("diagram_sum coefficients must be numeric")
    elif node.kind == "term":
        if not isinstance(node.value, TermContribution) or node.value.term_id not in term_ids:
            raise ValueError("term node references an unknown term_id")
        if node.children:
            raise ValueError("term nodes cannot have children")
    elif node.kind == "constant":
        if not _is_number(node.value) or node.children:
            raise TypeError("constant nodes must contain a numeric scalar and no children")
    elif node.kind == "constant_array":
        array = _constant_array(node.value)
        if array.dtype.kind not in "biufc" or tuple(array.shape) != tuple(node.shape) or node.children:
            raise TypeError("constant_array node has invalid value, shape, or children")
    elif node.kind in {"add", "multiply"}:
        if len(node.children) < 2:
            raise ValueError(f"{node.kind} nodes need at least two operands")
        for child in node.children:
            _validate_output_node(child, term_ids)
    elif node.kind == "power":
        if type(node.value) is not int or len(node.children) != 1:
            raise ValueError("power nodes need one child and an integer exponent")
        _validate_output_node(node.children[0], term_ids)
    elif node.kind == "select":
        if (not isinstance(node.value, tuple) or not node.value
                or any(type(index) is not int or index < 0 for index in node.value)
                or len(node.children) != 1):
            raise ValueError("select needs nonnegative leading-axis indices and one child")
        _validate_output_node(node.children[0], term_ids)
    elif node.kind == "dict":
        if (len(node.keys) != len(node.children)
                or len(set(node.keys)) != len(node.keys)):
            raise ValueError("dict node keys and children must be unique and have equal length")
        for child in node.children:
            _validate_output_node(child, term_ids)
    elif node.kind in {"list", "tuple"}:
        for child in node.children:
            _validate_output_node(child, term_ids)
    elif node.kind == "array":
        if (not isinstance(node.shape, tuple)
                or any(type(size) is not int or size < 0 for size in node.shape)
                or len(node.children) != math.prod(node.shape)):
            raise ValueError("array node children do not match its shape")
        for child in node.children:
            _validate_output_node(child, term_ids)
    else:
        raise ValueError(f"unknown OutputRecipe node kind {node.kind!r}")


def _broadcast_constant(value, coordinates):
    scan_shape = coordinates.scan_shape
    if not scan_shape:
        return np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    if not isinstance(value, np.ndarray):
        return np.full(scan_shape, value)
    target_shape = (*value.shape, *scan_shape)
    reshaped = value.reshape((*value.shape, *((1,) * len(scan_shape))))
    return np.broadcast_to(reshaped, target_shape).copy()


def _assemble_coordinate_values(values, coordinates, *, dtype=None):
    if not coordinates.scan_shape:
        return values[0]
    scan_shape = coordinates.scan_shape
    normalized = ([_scalarize(value) for value in values] if dtype is None else
                  [np.asarray(value, dtype=dtype)[()] for value in values])
    arrays = [np.asarray(value) for value in normalized]
    if (arrays and all(array.dtype.kind in "biufc" and array.shape == arrays[0].shape
                       for array in arrays)):
        stacked = np.stack(arrays, axis=0)
        leading_shape = (*scan_shape, *arrays[0].shape)
        stacked = stacked.reshape(leading_shape)
        scan_rank = len(scan_shape)
        value_rank = arrays[0].ndim
        if value_rank:
            stacked = stacked.transpose(
                (*range(scan_rank, scan_rank + value_rank), *range(scan_rank))
            )
        return stacked
    out = np.empty(scan_shape, dtype=object)
    if len(normalized) != math.prod(scan_shape):
        raise ValueError("coordinate value count does not match scan_shape")
    for flat_index, value in enumerate(normalized):
        out[np.unravel_index(flat_index, scan_shape)] = value
    return out


def _uniform_array_values(values):
    if not values:
        return False
    if all(not isinstance(value, (dict, list, tuple, np.ndarray)) for value in values):
        return all(_is_number(value) for value in values)
    if not all(isinstance(value, np.ndarray) for value in values):
        return False
    first_shape = values[0].shape
    return all(value.shape == first_shape for value in values)


def _validate_diagram_recipe(node, term_ids):
    """Validate local contraction algebra without expanding its polynomial."""
    if not isinstance(node, OutputNode):
        raise TypeError("diagram recipe must be an OutputNode")
    if not isinstance(node.children, tuple) or any(not isinstance(child, OutputNode) for child in node.children):
        raise TypeError("diagram recipe children must be OutputNode tuples")
    if node.kind == "term":
        if not isinstance(node.value, TermContribution) or node.value.term_id not in term_ids:
            raise ValueError("diagram recipe term leaf references an unknown term_id")
    elif node.kind == "constant":
        if not _is_number(node.value):
            raise TypeError("diagram recipe constants must be numeric scalars")
    elif node.kind in {"add", "multiply"}:
        if len(node.children) < 2:
            raise ValueError(f"diagram recipe {node.kind} needs at least two operands")
        for child in node.children:
            _validate_diagram_recipe(child, term_ids)
    elif node.kind == "power":
        if type(node.value) is not int or len(node.children) != 1:
            raise ValueError("diagram recipe power requires one child and integer exponent")
        _validate_diagram_recipe(node.children[0], term_ids)
    elif node.kind == "select":
        # Row-packed terms retain leading operator-row axes; one Diagram reads one entry.
        if (not isinstance(node.value, tuple) or len(node.children) != 1
                or any(type(index) is not int or index < 0 for index in node.value)):
            raise ValueError("diagram recipe select needs nonnegative leading-axis indices and one child")
        _validate_diagram_recipe(node.children[0], term_ids)
    else:
        raise ValueError(f"unsupported diagram recipe node {node.kind!r}")
    return node


def _scalarize(value):
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def _is_number(value):
    return isinstance(value, (numbers.Number, np.number)) or bool(getattr(value, "is_number", False))


def _python_number(value):
    if hasattr(value, "item"):
        return value.item()
    if getattr(value, "is_Integer", False):
        return int(value)
    if getattr(value, "is_real", None) is True:
        return float(value)
    if getattr(value, "is_number", False):
        return complex(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, numbers.Complex):
        return complex(value)
    return value


__all__ = [
    "OutputNode", "OutputRecipe", "TermValue", "assemble_correlator",
    "bind_output_assembly",
    "build_output_recipe", "collect_diagrams", "estimate_output_retention_bytes", "describe_output_fields", "freeze_output_recipe",
]
