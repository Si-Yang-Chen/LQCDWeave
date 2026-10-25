"""PC mode pairs assembled from shared Diagram projections.

Projection is linear and acts independently on disjoint vertex axes. Keep
disconnected factors factored instead of constructing their ordered products
and projecting the same Diagram again at every correlator output position.
"""
import numpy as np
from sympy import Add, Float, Mul

from .mode_blocks import _expression_layout, _partitions, split_eigenmodes
from .symbolic_graph import Diagram


def _project_symbolic_pairs(ordered, vertices, weight):
    """Build the same pair algebra without eagerly flattening every expression."""
    weighted_pair = np.frompyfunc(
        lambda left, right: Add(Mul(weight, left, evaluate=False),
                                Mul(weight, right, evaluate=False), evaluate=False), 2, 1)
    value = ordered
    for vertex in range(vertices):
        count = value.shape[vertex]
        axes = np.moveaxis(value, (vertex, vertex + 1), (0, 1))
        paired = np.empty((count * (count + 1) // 2, *axes.shape[2:]), dtype=object)
        for row, (a, b) in enumerate((a, b) for a in range(count) for b in range(a, count)):
            paired[row] = axes[a, b] if a == b else weighted_pair(axes[a, b], axes[b, a])
        value = np.moveaxis(paired, 0, vertex)
    return value


def split_meson_mode_pairs(correlator, *, partitions, partition_mode='radial'):
    """Split and PC-project meson expressions without whole ordered arrays.

    Container structure is preserved. Each vertex contributes one pair axis,
    appended to original array axes in the same sorted order as
    ``split_eigenmodes``. All ordered physical blocks remain represented.
    Independent Diagram projections and scalar templates are shared only
    within this call; returned arrays own their element slots.
    With partition_mode='memory', sum ordered blocks with unit weights and
    preserve the original output shape instead of producing radial axes.
    """
    if partition_mode not in ('radial', 'memory'):
        raise ValueError("partition_mode must be 'radial' or 'memory'")
    boundaries = tuple(partitions)
    ranges = _partitions(boundaries)
    from .structured import StructuredValue, project_structured
    pair_count = len(ranges) * (len(ranges) + 1) // 2
    diagrams, expressions = {}, {}
    structured = {}
    weight = Float(1 / np.sqrt(2))
    multiply = np.frompyfunc(lambda left, right: Mul(left, right, evaluate=False), 2, 1)
    add = np.frompyfunc(lambda left, right: Add(left, right, evaluate=False), 2, 1)

    def project_diagram(diagram):
        identity = id(diagram)
        if identity not in diagrams:
            keys, _ = _expression_layout(diagram)
            ordered = split_eigenmodes(diagram, partitions=boundaries)
            projected = (Add(*ordered.flat, evaluate=False) if partition_mode == 'memory' else
                         _project_symbolic_pairs(ordered, len(keys), weight))
            diagrams[identity] = diagram, keys, projected
        return diagrams[identity][1:]

    def project_scalar(value):
        identity = id(value)
        if identity in expressions:
            result = expressions[identity][1]
            return result.copy() if isinstance(result, np.ndarray) else result
        keys, selected = _expression_layout(value)
        if not selected:
            return value
        result = None
        for term in Add.make_args(value):
            coefficient, factors = 1, []
            for factor in Mul.make_args(term):
                if isinstance(factor, Diagram):
                    factor_keys, projected = project_diagram(factor)
                    shape = tuple(pair_count if key in factor_keys else 1 for key in keys)
                    factors.append(projected if partition_mode == 'memory' else projected.reshape(shape))
                else:
                    coefficient *= factor
            if partition_mode == 'memory':
                product = Mul(coefficient, *factors, evaluate=False)
                result = product if result is None else Add(result, product, evaluate=False)
                continue
            product = factors[0]
            for factor in factors[1:]:
                product = multiply(product, factor)
            if coefficient != 1:
                product = multiply(coefficient, product)
            result = product if result is None else add(result, product)
        expressions[identity] = value, result
        return result.copy() if isinstance(result, np.ndarray) else result

    def visit(value):
        if isinstance(value, StructuredValue):
            return project_structured(value, boundaries, partition_mode=partition_mode, memo=structured)
        if isinstance(value, dict):
            return {key: visit(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(visit(child) for child in value)
        if isinstance(value, np.ndarray):
            children = [visit(child) for child in value.flat]
            arrays = [child for child in children if isinstance(child, np.ndarray)]
            if arrays and all(child.shape == arrays[0].shape for child in arrays):
                children = [np.zeros(arrays[0].shape, dtype=object)
                            if not isinstance(child, np.ndarray) and child == 0 else child
                            for child in children]
            if children and all(isinstance(child, np.ndarray) and child.shape == children[0].shape
                                for child in children):
                return np.stack(children).reshape((*value.shape, *children[0].shape))
            result = np.empty(value.shape, dtype=object)
            for index, child in zip(np.ndindex(value.shape), children):
                result[index] = child
            return result
        return project_scalar(value)

    return visit(correlator)
