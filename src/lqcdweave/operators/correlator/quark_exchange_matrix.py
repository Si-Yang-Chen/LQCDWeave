"""Finite, exact matrix representations and coupled Wick graph reduction."""

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import permutations, product
from functools import cached_property, lru_cache

import sympy as sp


from lqcdweave.contracts.quark_exchange import QuarkExchange


def _inverse(p):
    return tuple(p.index(i) for i in range(3))


def _exact(matrix):
    return sp.ImmutableMatrix(matrix).applyfunc(lambda value: sp.cancel(sp.expand(value)))


@dataclass(frozen=True)
class ExchangeComponent:
    representation: QuarkExchange
    names: tuple
    flavors: tuple
    index: int

    @cached_property
    def identity(self):
        return sha256(repr(self).encode()).hexdigest()

    @cached_property
    def active(self):
        return tuple((p, matrix) for p, matrix in self.representation.elements
                     if tuple(self.flavors[i] for i in p) == self.flavors)

    def adjoint(self):
        return replace(self, representation=self.representation.adjoint(), flavors=self.flavors[::-1])

    def row(self, original, index):
        from .spatial_structure import HadronIrrepRow
        return HadronIrrepRow(self.names[index], original.momentum, original.irrep_name,
                             original.row_idx, original.parity, original.tag, original.dagger,
                             quark_exchange=replace(self, index=index))


@dataclass(frozen=True)
class _ScalarComponent:
    representation: QuarkExchange
    index: int = 0

    @property
    def active(self):
        return self.representation.elements

    def row(self, original, index):
        return original


def scalar_exchange_flavors(axes):
    """Collect each scalar-labelled row's actual flavor words across a request."""
    from .spatial_structure import HadronIrrepRow
    from .flavor_structure import HadronFlavorStructure
    from ._expressions import word_factors
    words = {}
    for axis in axes:
        for position, flavor in axis:
            if position == 0 or flavor == 0 or not any(
                    row.quark_symmetry for row in position.atoms(HadronIrrepRow)):
                continue
            for pterm in sp.Add.make_args(position.expand()):
                rows = tuple(f for f in word_factors(pterm) if isinstance(f, HadronIrrepRow))
                if not any(row.quark_symmetry for row in rows):
                    continue
                for fterm in sp.Add.make_args(flavor.expand()):
                    factors = tuple(f for f in word_factors(fterm) if isinstance(f, HadronFlavorStructure))
                    for row, factor in zip(rows, factors, strict=True):
                        if row.quark_symmetry:
                            word = tuple(factor.quark_list if factor.baryon_num == 1 else factor.anti_quark_list)
                            words.setdefault(row, set()).add(word)
    return {row: tuple(sorted(values)) for row, values in words.items()}


@lru_cache(maxsize=128)
def _scalar_component(symmetry, words):
    generators = {p: [[sign]] for p, sign in symmetry
                  if any(tuple(word[i] for i in p) == word for word in words)}
    return _ScalarComponent(QuarkExchange(generators))


def _row_basis(row):
    component = getattr(row, "quark_exchange", None)
    return component.row(row, 0) if component is not None else row


def _vertex_order(diagram):
    """A total, deterministic order, independent of a family's component index."""
    from lqcdweave.contracts.correlator.api import time_key_order
    times = tuple(time_key_order(time) for time in diagram.time_list)
    if all(a < b for a, b in zip(times, times[1:])):
        return tuple(range(len(diagram.vertex_list)))
    return tuple(sorted(range(len(diagram.vertex_list)), key=lambda i: (
        time_key_order(diagram.time_list[i]), repr(_row_basis(diagram.vertex_list[i]).__getnewargs_ex__()))))


def _vertex_actions(rows, times, selected, dimensions):
    """Permute identical vertices and the corresponding component tensor factors."""
    groups = {}
    for vertex, (row, time) in enumerate(zip(rows, times)):
        groups.setdefault((row, time), []).append(vertex)
    interchangeable = tuple(tuple(values) for values in groups.values() if len(values) > 1)
    states = tuple(product(*(range(d) for d in dimensions)))
    positions = {vertex: i for i, (vertex, _) in enumerate(selected)}
    state_index = {state: i for i, state in enumerate(states)}
    result = []
    for chosen in product(*(permutations(group) for group in interchangeable)):
        action = list(range(len(rows)))
        for original, moved in zip(interchangeable, chosen):
            for left, right in zip(original, moved):
                action[left] = right
        entries = {}
        for column, state in enumerate(states):
            transformed = list(state)
            for vertex, position in positions.items():
                transformed[positions[action[vertex]]] = state[position]
            entries[state_index[tuple(transformed)], column] = 1
        result.append((tuple(action), sp.ImmutableSparseMatrix(len(states), len(states), entries)))
    return tuple(result)


def reduce_matrix_exchange(expression, *, cache=None, scalar_flavors=None):
    """Reduce coupled row/graph orbits with their stabilizer relations.

    The local memo contains symbolic metadata only. Stabilizer averaging and
    exact column elimination retain matrix relations even when several group
    actions produce the same graph. Original Wick signs are already present.
    """
    from .symbolic_graph import Diagram, QuarkDiagram
    cache = {} if cache is None else cache
    scalar_flavors = {} if scalar_flavors is None else scalar_flavors
    replacements = {}
    for original_diagram in expression.atoms(Diagram):
        order = _vertex_order(original_diagram)
        if (order == tuple(range(len(order)))
                and not any(getattr(row, "quark_exchange", None) or getattr(row, "quark_symmetry", ())
                            for row in original_diagram.vertex_list)
                and len(set(zip(original_diagram.vertex_list, original_diagram.time_list))) == len(order)):
            continue
        diagram = Diagram(QuarkDiagram([[original_diagram.diagram.adjacency_matrix[i][j] for j in order]
                                       for i in order], validate=False),
                          [original_diagram.time_list[i] for i in order],
                          [original_diagram.vertex_list[i] for i in order], original_diagram.propagator_list)
        selected = []
        for vertex, row in enumerate(diagram.vertex_list):
            component = getattr(row, "quark_exchange", None)
            if component is None and getattr(row, "quark_symmetry", ()) and row in scalar_flavors:
                component = _scalar_component(row.quark_symmetry, scalar_flavors[row])
            if component is not None and (len(component.active) > 1 or component.representation.dimension > 1):
                selected.append((vertex, component))
        selected = tuple(selected)
        selected_vertices = {vertex for vertex, _ in selected}
        components = dict(selected)
        rows = tuple(components[i].row(row, 0) if i in selected_vertices
                     else row for i, row in enumerate(diagram.vertex_list))
        dimensions = tuple(component.representation.dimension for _, component in selected)
        vertex_actions = _vertex_actions(rows, diagram.time_list, selected, dimensions)
        if all(len(component.active) == 1 for _, component in selected) and len(vertex_actions) == 1:
            replacements[original_diagram] = diagram
            continue
        edges = []
        adjacency = diagram.diagram.adjacency_matrix
        for source, line in enumerate(adjacency):
            for sink, entry in enumerate(line):
                if isinstance(entry, list):
                    edges.extend((source, a, sink, b, prop)
                                 for a, values in enumerate(entry) for b, prop in enumerate(values) if prop)
                elif entry:
                    edges.append((source, 0, sink, 0, entry))
        edges = tuple(sorted(edges))
        context = (rows, tuple(diagram.time_list), tuple(diagram.propagator_list),
                   tuple((vertex, component.active) for vertex, component in selected))
        key = context, edges
        if key not in cache:
            actions = []
            for chosen in product(*(component.active for _, component in selected)):
                permutations = {vertex: p for (vertex, _), (p, _) in zip(selected, chosen)}
                leg_edges = tuple((s, permutations[s][a] if s in permutations else a,
                                   t, permutations[t][b] if t in permutations else b, prop)
                                  for s, a, t, b, prop in edges)
                matrix = sp.ImmutableMatrix([[1]])
                inverse_matrix = sp.ImmutableMatrix([[1]])
                for (_, component), (p, m) in zip(selected, chosen):
                    matrix = sp.kronecker_product(matrix, m)
                    inverse_matrix = sp.kronecker_product(
                        inverse_matrix, component.representation.matrices[_inverse(p)])
                for vertex_permutation, component_permutation in vertex_actions:
                    moved = tuple(sorted((vertex_permutation[s], a, vertex_permutation[t], b, prop)
                                         for s, a, t, b, prop in leg_edges))
                    actions.append((moved, component_permutation * matrix,
                                    inverse_matrix * component_permutation.T))
            representative = min(moved for moved, _, _ in actions)
            _, transport, inverse = next(action for action in actions if action[0] == representative)
            stabilizers = [m * inverse for moved, m, _ in actions if moved == representative]
            projector = _exact(sum(stabilizers, sp.zeros(transport.rows)) / len(stabilizers))
            pivots = projector.rref()[1]
            if pivots:
                basis = projector[:, list(pivots)]
                pivot_rows = basis.T.rref()[1]
                weights = _exact(basis[list(pivot_rows), :].inv() * projector[list(pivot_rows), :])
            else:
                weights = sp.zeros(0, transport.rows)
            basis_diagrams = []
            states = tuple(product(*(range(d) for d in dimensions)))
            for pivot in pivots:
                vertex_rows = list(diagram.vertex_list)
                for (vertex, component), index in zip(selected, states[pivot]):
                    vertex_rows[vertex] = component.row(vertex_rows[vertex], index)
                graph = [[[[0 for _ in values] for values in entry] if isinstance(entry, list) else 0
                          for entry in line] for line in adjacency]
                for s, a, t, b, prop in representative:
                    if isinstance(graph[s][t], list):
                        graph[s][t][a][b] = prop
                    else:
                        graph[s][t] = prop
                basis_diagrams.append(Diagram(QuarkDiagram(graph), list(diagram.time_list),
                                              vertex_rows, list(diagram.propagator_list)))
            for moved, matrix, inverse_matrix in actions:
                member_key = context, moved
                if member_key not in cache:
                    cache[member_key] = tuple(basis_diagrams), _exact(weights * transport * inverse_matrix)
        basis_diagrams, coefficients = cache[key]
        index = 0
        for dimension, (_, component) in zip(dimensions, selected):
            index = index * dimension + component.index
        replacements[original_diagram] = sp.Add(*(coefficients[i, index] * value
                                                 for i, value in enumerate(basis_diagrams)))
    return expression.xreplace(replacements).expand()
