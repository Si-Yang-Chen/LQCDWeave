"""Call-local evaluation of a validated, fixed-coordinate OutputRecipe DAG."""
import numpy as np

from .core import (_cast_output_fields, _constant_array, _python_number,
                   _scalarize, _uniform_array_values)
from .matrix_output import (MATRIX_OUTPUT_KINDS, evaluate_matrix_output,
                            scatter_sum_blocks, evaluate_scatter_sum)


_MISSING = object()
_LITERAL_TYPES = (bool, int, float, complex)


def _bind_term(index, coefficient, slot, scalarize=_scalarize):
    def evaluate(context):
        cache = context[1]
        value = cache[slot]
        if value is _MISSING:
            value = coefficient * scalarize(context[0][index])
            if type(value) in _LITERAL_TYPES or isinstance(value, np.number):
                cache[slot] = value
        return value
    return evaluate


def _bind_multiply(children, slot, prefix):
    def evaluate(context):
        cache = context[1]
        value = cache[slot]
        if value is _MISSING:
            value = prefix
            for child in children:
                value = value * child(context)
            if type(value) in _LITERAL_TYPES or isinstance(value, np.number):
                cache[slot] = value
        return value
    return evaluate


def bind_fixed_output(recipe, term_indexes):
    """Bind structure/coefficients, retaining no numerical call results.

    Only immutable scalar results are memoized outside the original matrix /
    select cache. Array-valued arithmetic still creates its original owners.
    Literal arithmetic keeps the same order, including the initial 0/1.
    """
    bound = {}
    literals = {}
    scalar_slots = 0
    computation_dtype = recipe.intermediate_dtype or recipe.dtype
    dtype = None if computation_dtype is None else np.dtype(computation_dtype)
    number = _python_number if dtype is None else lambda value: dtype.type(_python_number(value))
    scalarize = _scalarize if dtype is None else lambda value: np.asarray(value, dtype=dtype)[()]
    zero, one = (0, 1) if dtype is None else (dtype.type(0), dtype.type(1))

    def bind(node):
        nonlocal scalar_slots
        identity = id(node)
        if identity in bound:
            return bound[identity]
        kind = node.kind
        child_nodes = node.children
        placements = scatter_sum_blocks(node)
        if placements is not None:
            child_nodes = tuple(block.children[0] for block in placements)
        prefix = one
        if kind == 'multiply':
            prefix_count = 0
            for child in child_nodes:
                if child.kind != 'constant' or type(child.value) not in _LITERAL_TYPES:
                    break
                try:
                    prefix = prefix * number(child.value)
                except ArithmeticError:
                    prefix, prefix_count = one, 0
                    break
                prefix_count += 1
            child_nodes = child_nodes[prefix_count:]
        children = tuple(bind(child) for child in child_nodes)
        if kind == 'constant':
            value = number(node.value)
            def evaluate(context): return value
            if type(value) in _LITERAL_TYPES:
                literals[identity] = value
        elif kind == 'constant_array':
            def evaluate(context): return np.array(_constant_array(node.value), dtype=dtype, copy=True)
        elif kind == 'term':
            index = term_indexes[node.value.term_id]
            coefficient = number(node.value.coefficient)
            evaluate = _bind_term(index, coefficient, scalar_slots, scalarize)
            scalar_slots += 1
        elif kind == 'diagram_sum':
            entries = tuple((term_indexes[key], number(coefficient))
                            for key, coefficient in node.value)
            def evaluate(context):
                values = context[0]
                return sum((coefficient * scalarize(values[index])
                            for index, coefficient in entries), zero)
        elif kind == 'add':
            if placements is None:
                def evaluate(context): return sum((child(context) for child in children), zero)
            else:
                by_identity = {id(block.children[0]): child for block, child in zip(placements, children)}
                indexes = tuple(np.ix_(*block.value) for block in placements)
                def evaluate(context):
                    return evaluate_scatter_sum(placements,
                        lambda child: by_identity[id(child)](context), (),
                        trusted=True, zero=zero, indexes=indexes)
        elif kind == 'multiply':
            if all(id(child) in literals for child in child_nodes):
                try:
                    value = prefix
                    for child in child_nodes:
                        value = value * literals[id(child)]
                except ArithmeticError:
                    pass
                else:
                    if type(value) in _LITERAL_TYPES:
                        literals[identity] = value
                        def evaluate(context, value=value): return value
            if identity not in literals:
                evaluate = _bind_multiply(children, scalar_slots, prefix)
                scalar_slots += 1
        elif kind == 'power':
            child, = children
            def evaluate(context):
                value = child(context)
                return value ** node.value if dtype is None else np.power(value, node.value, dtype=dtype)
        elif kind in MATRIX_OUTPUT_KINDS:
            by_identity = {id(child): evaluate for child, evaluate in zip(node.children, children)}
            def evaluate(context):
                selected = context[2]
                if identity not in selected:
                    selected[identity] = evaluate_matrix_output(node,
                        lambda child: by_identity[id(child)](context), (), trusted=True, dtype=dtype)
                return selected[identity]
        elif kind == 'select':
            child, = children
            child_identity = id(node.children[0])
            def evaluate(context):
                selected = context[2]
                if child_identity not in selected:
                    selected[child_identity] = child(context)
                return selected[child_identity][node.value]
        elif kind == 'dict':
            pairs = tuple(zip(node.keys, children, strict=True))
            def evaluate(context): return {key: child(context) for key, child in pairs}
        elif kind == 'list':
            def evaluate(context): return [child(context) for child in children]
        elif kind == 'tuple':
            def evaluate(context): return tuple(child(context) for child in children)
        elif kind == 'array':
            nested = any(child.kind in {'array', 'constant_array', 'dict', 'list', 'tuple'}
                         for child in node.children)
            indexes = tuple(np.ndindex(node.shape)) if nested else None
            def evaluate(context):
                values = [child(context) for child in children]
                if not values:
                    return np.empty(node.shape, dtype=object)
                if not nested and _uniform_array_values(values):
                    stacked = np.stack(values, axis=0)
                    return stacked.reshape((*node.shape, *stacked.shape[1:]))
                result = np.empty(node.shape, dtype=object)
                for index, value in zip(indexes if indexes is not None else np.ndindex(node.shape),
                                        values, strict=True):
                    result[index] = value
                return result
        else:
            raise ValueError(f'unknown OutputRecipe node kind {kind!r}')

        if kind in {'add', 'power'} and all(id(child) in literals for child in node.children):
            # Do not move arithmetic errors from invocation into binding.
            try:
                value = evaluate(((), {}, {}))
            except ArithmeticError:
                pass
            else:
                if type(value) in _LITERAL_TYPES:
                    literals[identity] = value
                    def evaluate(context, value=value): return value

        if kind in {'diagram_sum', 'add', 'power'} and identity not in literals:
            slot = scalar_slots
            scalar_slots += 1
            operation = evaluate
            def evaluate(context):
                cache = context[1]
                value = cache[slot]
                if value is _MISSING:
                    value = operation(context)
                    if type(value) in _LITERAL_TYPES or isinstance(value, np.number):
                        cache[slot] = value
                return value
        bound[identity] = evaluate
        return evaluate

    evaluate = bind(recipe.root)
    # The recursive binder is no longer needed. Break its self-reference so
    # binding-only dictionaries and term indexes need not await cyclic GC.
    del bind
    if recipe.intermediate_dtype:
        return lambda values: _cast_output_fields(
            evaluate((values, [_MISSING] * scalar_slots, {})), recipe.dtype)
    return lambda values: evaluate((values, [_MISSING] * scalar_slots, {}))
