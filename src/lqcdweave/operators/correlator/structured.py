"""Compact single-hadron correlator requests, without a symbolic matrix.

Wick coefficients/topologies reuse the public S implementation. Bindings,
numeric data and contraction paths do not belong to these requests.
"""
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from sympy import Add, Mul, S
from ._expressions import word_factors

from .spatial_structure import HadronIrrepRow
from .flavor_structure import HadronFlavorStructure
from .symbolic_graph import Diagram, diagram_simplify, quark_contract


class StructuredValue:
    def __add__(self, other):
        return StructuredOperation('add', (self, other))

    __radd__ = __add__

    def __mul__(self, other):
        return StructuredOperation('multiply', (self, other))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return other + -self


@dataclass(frozen=True)
class StructuredOperation(StructuredValue):
    kind: str
    children: tuple

    @property
    def shape(self):
        shapes = [child.shape for child in self.children if isinstance(child, StructuredValue)]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError('structured arithmetic requires matching output shapes')
        return shapes[0] if shapes else ()


@dataclass(frozen=True)
class StructuredCorrelator(StructuredValue):
    axes: tuple
    degenerate: bool
    partitions: tuple | None = None
    excluded_propagators: tuple = ()
    partition_mode: str = 'radial'

    def __post_init__(self):
        if self.partition_mode not in ('radial', 'memory'):
            raise ValueError("partition_mode must be 'radial' or 'memory'")
        if self.partition_mode == 'memory' and self.partitions is None:
            raise ValueError('memory partition_mode requires partitions')

    @classmethod
    def from_hadrons(cls, hadrons, *, degenerate):
        if not 1 <= len(hadrons) <= 2:
            raise NotImplementedError('structured requests support one- and two-point single-hadron matrices')
        axes = tuple(tuple((hadron.irrep_row, hadron.flavor_structure) for hadron in axis)
                     for axis in hadrons)
        for axis in axes:
            for position, _flavor in axis:
                _linear_rows(position)
        return cls(axes, degenerate)

    @property
    def shape(self):
        shape = tuple(len(axis) for axis in self.axes)
        if self.partitions is not None and self.partition_mode == 'radial':
            count = len(self.partitions) - 1
            shape += (count * (count + 1) // 2,) * len(self.axes)
        return shape


def _linear_rows(expression):
    result = []
    for term in Add.make_args(expression.expand()):
        rows, coefficient = [], S.One
        for factor in Mul.make_args(term):
            if isinstance(factor, HadronIrrepRow):
                rows.append(factor)
            else:
                coefficient *= factor
        if coefficient == 0:
            continue
        if len(rows) != 1:
            raise NotImplementedError('each structured insertion must be a linear single-hadron operator')
        if not coefficient.is_number:
            raise TypeError('structured operator coefficients must be numeric')
        result.append((rows[0], coefficient))
    return tuple(result)


def project_structured(value, partitions, *, partition_mode='radial', memo=None):
    memo = {} if memo is None else memo
    key = id(value)
    if key in memo:
        return memo[key][1]
    if isinstance(value, StructuredCorrelator):
        if value.partitions is not None:
            raise ValueError('correlator is already mode-partitioned')
        result = replace(value, partitions=tuple(partitions), partition_mode=partition_mode)
    else:
        result = StructuredOperation(value.kind, tuple(
            project_structured(child, partitions, partition_mode=partition_mode, memo=memo)
            if isinstance(child, StructuredValue) else child
            for child in value.children))
    memo[key] = value, result
    return result


def filter_structured(value, propagators):
    if isinstance(value, StructuredCorrelator):
        return replace(value, excluded_propagators=(*value.excluded_propagators, *propagators))
    return StructuredOperation(value.kind, tuple(
        filter_structured(child, propagators) if isinstance(child, StructuredValue) else child
        for child in value.children))


def iter_structured_wick_entries(task):
    """Yield matrix index and Wick monomials; never retain a whole matrix.

    This mirrors gen_correlator's position/flavor expansion and graph simplify.
    Every monomial contains its numeric Wick/position coefficient and independent
    Diagram components. Operator/data coefficients are supplied later by B/E.
    """
    choices = tuple(tuple(_linear_rows(position) for position, _ in axis) for axis in task.axes)
    templates = {}
    exchange_cache = {}
    from .quark_exchange_matrix import scalar_exchange_flavors
    scalar_flavors = scalar_exchange_flavors(task.axes)
    for index in np.ndindex(tuple(len(axis) for axis in task.axes)):
        flavors = tuple(axis[i][1] for axis, i in zip(task.axes, index, strict=True))
        if flavors not in templates:
            flavor = Mul(*flavors).expand()
            templates[flavors] = diagram_simplify(quark_contract(
                flavor, np.arange(len(task.axes)), degenerate=task.degenerate))
        template = templates[flavors]
        entries = []
        for chosen in product(*(axis[i] for axis, i in zip(choices, index, strict=True))):
            # Match the canonical factor order used by the symbolic frontend.
            rows = tuple(word_factors(Mul(*(item[0] for item in chosen))))
            position_weight = Mul(*(item[1] for item in chosen))
            for term in Add.make_args(template):
                coefficient, diagrams = position_weight, []
                for factor in Mul.make_args(term):
                    if isinstance(factor, Diagram):
                        if any(prop in task.excluded_propagators for prop in factor.propagator_list):
                            coefficient = S.Zero
                            break
                        diagrams.append(Diagram(factor.diagram, factor.time_list,
                                                [rows[v] for v in factor.vertex_list],
                                                factor.propagator_list))
                    else:
                        coefficient *= factor
                if coefficient != 0:
                    # Preserve radicals and phases through exact exchange
                    # elimination; numerical output conversion happens below.
                    entries.append((coefficient, tuple(diagrams)))
        if entries:
            from .quark_exchange_matrix import reduce_matrix_exchange
            expression = reduce_matrix_exchange(Add(*(weight * Mul(*diagrams)
                                                      for weight, diagrams in entries)), cache=exchange_cache,
                                                scalar_flavors=scalar_flavors)
            entries = []
            for term in Add.make_args(expression):
                diagrams, coefficient = [], S.One
                for factor in Mul.make_args(term):
                    if isinstance(factor, Diagram):
                        diagrams.append(factor)
                    else:
                        coefficient *= factor
                if coefficient != 0:
                    entries.append((complex(coefficient), tuple(diagrams)))
        yield index, tuple(entries)


def structured_axis_rows(task):
    """Describe each linear axis independently, preserving insertion order.

    Reuses the position parser of iter_structured_wick_entries. The Hadron rows
    are noncommutative, so their order is the input axis order, not name order.
    """
    return tuple(tuple(_linear_rows(position) for position, _ in axis)
                 for axis in task.axes)


def structured_axis_times(task):
    """Get insertion labels from the explicitly timed flavour expressions.

    Flavour symbols are the source of Wick insertion times. Rows retain their
    own time tags too; their visible Symbol names deliberately omit time.
    """
    result = []
    for axis in task.axes:
        times = {item.time for _, flavor in axis for item in flavor.atoms(HadronFlavorStructure)}
        if len(times) > 1:
            raise NotImplementedError('matrix axes require one insertion time label each')
        result.append(next(iter(times)) if times else 0)
    return tuple(result)


def structured_wick_templates(task, *, cache=None):
    """Yield rectangular flavour subsets and Wick templates with slot vertices.

    This is the Wick/flavour portion of iter_structured_wick_entries, without
    its matrix-cell and position-component products. Filtering happens before
    handing the templates to expansion. No bound vertex or data is created.
    """
    cache = {} if cache is None else cache
    axes = []
    for axis in task.axes:
        groups = {}
        for index, (_, flavor) in enumerate(axis):
            groups.setdefault(flavor, []).append(index)
        axes.append(tuple((flavor, tuple(positions)) for flavor, positions in groups.items()))
    for chosen in product(*axes):
        flavors = tuple(item[0] for item in chosen)
        key = flavors, task.degenerate, task.excluded_propagators
        if key not in cache:
            flavor = Mul(*flavors).expand()
            template = diagram_simplify(quark_contract(
                flavor, np.arange(len(task.axes)), degenerate=task.degenerate))
            entries = []
            for term in Add.make_args(template):
                coefficient, diagrams = S.One, []
                for factor in Mul.make_args(term):
                    if isinstance(factor, Diagram):
                        if any(prop in task.excluded_propagators for prop in factor.propagator_list):
                            coefficient = S.Zero
                            break
                        diagrams.append(factor)
                    else:
                        coefficient *= factor
                if coefficient != 0:
                    entries.append((complex(coefficient), tuple(diagrams)))
            cache[key] = tuple(entries)
        yield tuple(item[1] for item in chosen), cache[key]


__all__ = ['StructuredValue', 'StructuredCorrelator', 'StructuredOperation',
           'iter_structured_wick_entries', 'structured_axis_rows', 'structured_axis_times',
           'structured_wick_templates']
