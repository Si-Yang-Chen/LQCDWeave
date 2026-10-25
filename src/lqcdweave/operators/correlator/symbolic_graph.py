"""Symbolic Wick graphs used before any numerical data is bound.

The algebraic kernel follows the legacy frontend. Numerical contraction and
file loading belong to separate modules.
"""
from copy import deepcopy
from itertools import product
import hashlib
import logging
from typing import Any, Dict, List, Union

import numpy as np
import sympy as sp
from sympy import Add, Expr, Mul, Pow, S, Symbol, simplify
from sympy.core.cache import cacheit

from lqcdweave.operators.correlator.flavor_structure import Propagator
from lqcdweave.operators.correlator.spatial_structure import HadronIrrepRow
from .graph_identity import graph_identity, _freeze
from lqcdweave.contracts.correlator.api import time_key_order

logger = logging.getLogger(__name__)


def _flatten_paths_matrix(path):
    """Flatten one adjacency-matrix entry into its propagator labels.

    Same recursion as ``lattice.quark_draw._flatten_paths``; kept here so the
    validity check below does not depend on the drawing module.
    """
    if isinstance(path, int):
        return [path] if path != 0 else []
    if isinstance(path, list):
        labels = []
        for entry in path:
            labels.extend(_flatten_paths_matrix(entry))
        return labels
    raise ValueError(f"Invalid value {path} in the adjacency matrix")


def validate_adjacency_matrix(adjacency_matrix):
    """Check the quark-link invariants of an adjacency matrix.

    Every propagator entry is a quark line, so at each vertex the number of
    line-ends is that vertex's quark count. Since every quark must be linked
    (quark out, antiquark in) and there are only three hadrons, the allowed
    (outgoing, incoming) degree pairs are:

        (1, 1)  meson: one quark line in, one out — possibly both a self-loop
        (3, 0)  baryon at the source: its three quarks all leave
        (0, 3)  baryon at the sink: its three quarks all arrive

    (0, 0) represents an explicit zero-quark insertion, such as a glueball.
    Native binding verifies that its numerical vertex has no quark legs.
    A disconnected meson still has its non-zero self-loop.

    Anything else describes a hadron that cannot exist — a vertex with two
    quark lines out carries four quarks, one with a single dangling end leaves a
    quark unpaired — and the diagram would fail later, deep inside contraction
    or drawing, with an error far from the cause.

    Raises:
        ValueError: naming the first vertex that violates the invariant.
    """
    num_vertices = len(adjacency_matrix)
    degree = [[0, 0] for _ in range(num_vertices)]
    for i in range(num_vertices):
        row = adjacency_matrix[i]
        if len(row) != num_vertices:
            raise ValueError(
                f"Adjacency matrix must be square; row {i} has {len(row)} "
                f"entries, expected {num_vertices}."
            )
        for j in range(num_vertices):
            for _label in _flatten_paths_matrix(row[j]):
                degree[i][0] += 1   # a line leaves i
                degree[j][1] += 1   # ... and arrives at j
    for i, (outgoing, incoming) in enumerate(degree):
        if (outgoing, incoming) not in {(0, 0), (1, 1), (3, 0), (0, 3)}:
            hint = (
                "; a hadron that connects to nothing must self-loop, e.g. a "
                "non-zero diagonal entry"
                if (outgoing, incoming) == (0, 0)
                else ""
            )
            raise ValueError(
                f"Vertex {i} has {outgoing} outgoing and {incoming} incoming "
                f"quark lines; a hadron must be a meson (1, 1), or a baryon at "
                f"the source (3, 0) or sink (0, 3)" + hint + "."
            )

class QuarkDiagram:
    def __init__(self, adjacency_matrix, validate=True):
        self.adjacency_matrix = adjacency_matrix
        if validate:
            validate_adjacency_matrix(adjacency_matrix)


@cacheit
def _diagram_symbol(cls, identity, visible_name):
    """Retain bounded interning by full identity, independent of display names."""
    return Symbol.__xnew__(cls, visible_name)


class Diagram(Symbol):
    def __new__(cls, diagram, time_list, vertex_list, propagator_list):
        # The visible vertex names omit insertion tags. SymPy's name cache
        # must not return another graph and overwrite its tagged vertices.
        # Equality and hashing below use the complete graph identity.
        identity=(_freeze(diagram.adjacency_matrix),_freeze(time_list),
                  _freeze(vertex_list),_freeze(propagator_list))
        obj = _diagram_symbol(cls,identity,
            f"{diagram.adjacency_matrix},{time_list},{vertex_list},{propagator_list}")
        return obj

    def __init__(self, diagram, time_list, vertex_list, propagator_list):
        self.diagram = diagram
        self.time_list = time_list
        self.vertex_list = vertex_list
        self.propagator_list = propagator_list
        self.value = None
        self.value_pointer = None

    def __deepcopy__(self, memo):
        # Do not reconstruct through the cached public constructor: it can
        # return this same Symbol and replace caller-owned mutable graph lists.
        clone = self.__xnew__(type(self), self.name, **self.assumptions0)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, deepcopy(value, memo))
        return clone

    def __str__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __repr__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return graph_identity(self) == graph_identity(other)

    def __hash__(self):
        # Retain the small integer domain used by the selected-symbol factory.
        return hash(graph_identity(self)) % (2**31)

    def transform(self, group_element, time=None):
        coeff_vertices_list = []
        for i in range(len(self.vertex_list)):
            assert isinstance(self.vertex_list[i], HadronIrrepRow)
            if time is None or self.time_list[i] == time:
                transformed_vertex = self.vertex_list[i].transform(group_element)
            else:
                transformed_vertex = self.vertex_list[i]
            terms = Add.make_args(transformed_vertex)
            coeff_vertices = []
            for term in terms:
                factors = Mul.make_args(term)
                coeff = S(1)
                for factor in factors:
                    if isinstance(factor, HadronIrrepRow):
                        hadron_irrep_row = factor
                    else:
                        coeff *= factor
                coeff_vertices.append((coeff, hadron_irrep_row))
            coeff_vertices_list.append(coeff_vertices)
        vertex_products = list(product(*coeff_vertices_list))
        result = S(0)
        for i in range(len(vertex_products)):
            vertex_product = vertex_products[i]
            diagram_coeff = S(1)
            new_vertex_list = []
            for coeff_vertex in vertex_product:
                diagram_coeff *= coeff_vertex[0]
                new_vertex_list.append(coeff_vertex[1])
            result += diagram_coeff * Diagram(
                self.diagram, self.time_list, new_vertex_list, self.propagator_list
            )
        return result

    def conjugate(self):
        new_vertex_list = []
        for i in range(len(self.vertex_list)):
            assert isinstance(self.vertex_list[i], HadronIrrepRow)
            new_vertex_list.append(self.vertex_list[i].conjugate())

        return Diagram(
            self.diagram, self.time_list, new_vertex_list, self.propagator_list
        )

    def simplify(self):
        """
        Simplify Diagram object, perform the following operations:
        1. Remove redundant vertices (equivalent to remove_redundant functionality)
        2. Sort vertices and propagators (equivalent to sort_vertex_and_propagator functionality)
        3. Split graph into product of different subgraphs

        Integrates the functionality of the original separate remove_redundant and sort_vertex_and_propagator methods,
        and further splits the graph into connected components, ultimately returning the optimized Diagram or Diagram product expression.

        Returns:
            sympy.Expr or Diagram: Simplified Diagram object or expression representing product of different subgraphs
        """
        from sympy import Mul
        from copy import deepcopy

        # Get graph information
        adjacency_matrix = deepcopy(self.diagram.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        # Record each vertex's connected component
        component_ids = [-1] * num_vertex
        next_component_id = 0

        # Use BFS to find all connected components
        for start_vertex in range(num_vertex):
            # If already assigned connected component, skip
            if component_ids[start_vertex] != -1:
                continue

            # An isolated zero-quark vertex is a data factor, not the identity.
            # Use BFS to find all connected vertices
            component_ids[start_vertex] = next_component_id
            queue = [start_vertex]
            while queue:
                vertex = queue.pop(0)

                # Check all possible connections
                for next_vertex in range(num_vertex):
                    # Check connection from vertex to next_vertex
                    if component_ids[next_vertex] == -1:
                        is_in_queue = False
                        if (
                            isinstance(
                                adjacency_matrix[vertex][next_vertex], np.ndarray
                            )
                            and (adjacency_matrix[vertex][next_vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif (
                            isinstance(
                                adjacency_matrix[next_vertex][vertex], np.ndarray
                            )
                            and (adjacency_matrix[next_vertex][vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif adjacency_matrix[vertex][next_vertex] != 0:
                            is_in_queue = True
                        elif adjacency_matrix[next_vertex][vertex] != 0:
                            is_in_queue = True
                        if is_in_queue:
                            component_ids[next_vertex] = next_component_id
                            queue.append(next_vertex)
            # Start a new connected component
            next_component_id += 1
        # Create a new Diagram object for each connected component
        result_diagrams = []
        for component_id in range(next_component_id):
            # Find vertices belonging to this connected component
            vertices = []
            for i in range(num_vertex):
                if component_ids[i] == component_id:
                    vertices.append(i)
            # Sort vertices based on self.vertex_list
            # When vertices are equal, consider all possible orders
            # First, sort based on time and vertex type
            vertices.sort(key=lambda v: (time_key_order(self.time_list[v]), self.vertex_list[v]))

            # Check if there are same vertices
            has_same_vertices = False
            for i in range(len(vertices) - 1):
                if (
                    self.time_list[vertices[i]] == self.time_list[vertices[i + 1]]
                    and self.vertex_list[vertices[i]]
                    == self.vertex_list[vertices[i + 1]]
                ):
                    has_same_vertices = True
                    break

            # If there are same vertices, consider all possible orders
            if has_same_vertices:
                # Group by time and vertex type
                from itertools import groupby
                from itertools import permutations

                # Group by time and vertex type
                groups = []
                for k, g in groupby(
                    vertices, key=lambda v: (self.time_list[v], self.vertex_list[v])
                ):
                    groups.append(list(g))

                # Only sort vertices in each group, keeping group order
                all_possible_orders = []
                for g in groups:
                    perms = [list(p) for p in permutations(g)]
                    all_possible_orders.append(perms)

                # Use itertools.product to get all combinations
                from itertools import product

                all_permutations = list(product(*all_possible_orders))

                # Flatten nested list to match vertices format
                all_possible_vertices = []
                for perm_combination in all_permutations:
                    flattened_vertices = []
                    for group_perm in perm_combination:
                        flattened_vertices.extend(group_perm)
                    all_possible_vertices.append(flattened_vertices)
            else:
                all_possible_vertices = [vertices]
            # Create adjacency matrix for this connected component, size of connected component
            min_hash = float("inf")
            for vertices in all_possible_vertices:
                component_size = len(vertices)
                component_matrix = [
                    [
                        adjacency_matrix[vertices[j]][vertices[i]]
                        for i in range(component_size)
                    ]
                    for j in range(component_size)
                ]
                new_quark_diagram = QuarkDiagram(component_matrix, validate=False)
                new_time_list = [self.time_list[i] for i in vertices]
                new_vertex_list = [self.vertex_list[i] for i in vertices]
                # Create new propagator_list, only retain used propagators
                used_propagators = set([])  # 0 is default value, always retained

                # Traverse component_matrix to find all used propagators
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                used_propagators.add(value)
                        elif isinstance(value, np.ndarray):
                            # For array type, add propagator indices of all non-zero elements
                            for prop_idx in value.flatten():
                                if prop_idx != 0:
                                    used_propagators.add(int(prop_idx))
                        elif isinstance(value, list):
                            # For list type, add all non-zero elements
                            # Handle nested lists, similar to ndarray's flatten operation
                            flat_values = []

                            def flatten_list(lst):
                                for item in lst:
                                    if isinstance(item, list):
                                        flatten_list(item)
                                    else:
                                        flat_values.append(item)

                            flatten_list(value)
                            for prop_idx in flat_values:
                                if prop_idx != 0:
                                    used_propagators.add(prop_idx)

                # Sort used propagator indices in original order
                used_propagators = sorted(
                    list(used_propagators), key=lambda x: self.propagator_list[x]
                )
                used_propagators = [0] + used_propagators
                new_propagator_list = [
                    self.propagator_list[i] for i in used_propagators
                ]

                # Update propagator indices in component_matrix
                old_to_new = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(used_propagators)
                }
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                component_matrix[i][j] = old_to_new[value]
                        elif isinstance(value, np.ndarray):
                            # For array type, update all non-zero elements
                            new_array = np.zeros_like(value)
                            for idx in np.ndindex(value.shape):
                                if value[idx] != 0:
                                    new_array[idx] = old_to_new[int(value[idx])]
                            component_matrix[i][j] = new_array
                        elif isinstance(value, list):
                            # For list type, update all non-zero elements
                            # Handle nested list case
                            def update_nested_list(lst):
                                result = []
                                for item in lst:
                                    if isinstance(item, list):
                                        result.append(update_nested_list(item))
                                    else:
                                        result.append(
                                            old_to_new[item] if item != 0 else 0
                                        )
                                return result

                            component_matrix[i][j] = update_nested_list(value)

                # Use stable hash algorithm to calculate graph hash
                current_hash = int(
                    hashlib.sha256(
                        str(new_quark_diagram.adjacency_matrix).encode()
                    ).hexdigest(),
                    16,
                )
                if min_hash > current_hash:
                    min_hash = current_hash
                    new_diagram = Diagram(
                        new_quark_diagram,
                        new_time_list,
                        new_vertex_list,
                        new_propagator_list,
                    )

            result_diagrams.append(new_diagram)

        # Convert results to product expression
        result = S(1)
        for diagram in result_diagrams:
            result = Mul(result, diagram, evaluate=False)
        return sp.simplify(result)

def diagram_vertice_replace(
    expr: Union[Expr, List, Any], indice_map: Dict
) -> Union[Expr, List, Any]:
    """
    Replace vertices in all Diagram object in the expr,list,dict,tuple or any data structure
    """
    if isinstance(expr, Diagram):
        # Replace vertices in the Diagram object
        new_vertice_list = [indice_map[v] for v in expr.vertex_list]
        new_diagram = Diagram(
            expr.diagram, expr.time_list, new_vertice_list, expr.propagator_list
        )
        return new_diagram
    elif isinstance(expr, list):
        # Recursively process list elements
        return [diagram_vertice_replace(item, indice_map) for item in expr]
    elif isinstance(expr, tuple):
        # Recursively process tuple elements
        return tuple(diagram_vertice_replace(item, indice_map) for item in expr)
    elif isinstance(expr, dict):
        # Recursively process dictionary values
        return {
            key: diagram_vertice_replace(value, indice_map)
            for key, value in expr.items()
        }
    elif isinstance(expr, Add):
        # Process sympy Add expression
        return Add(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif isinstance(expr, Mul):
        # Process sympy Mul expression
        return Mul(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif hasattr(expr, "args") and expr.args:
        # For other expressions with args attribute
        return expr.func(
            *[diagram_vertice_replace(arg, indice_map) for arg in expr.args]
        )
    else:
        # Return unchanged for other types
        return expr


def diagram_simplify(expr: Union[Expr, List, Any]) -> Union[Expr, List, Any]:
    """
    Recursively simplify expressions containing Diagram objects

    Call each Diagram object's simplify method, which integrates the following functionalities:
    1. Remove redundant vertices
    2. Sort vertices and propagators
    3. Split graph into connected components

    Supports processing various data structures, including:
    - Single Diagram object
    - sympy expressions (e.g., Add, Mul, Pow, etc.)
    - Nested lists, tuples, dictionaries
    - NumPy arrays

    Args:
        expr: Expression or data structure containing Diagram objects

    Returns:
        Simplified expression or data structure, original expression unchanged
    """
    from sympy import Add, Mul, Pow, Symbol
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Base case: process single Diagram object
    if isinstance(expr, Diagram):
        try:
            # Apply simplification operations
            # expr = expr.remove_redundant()
            # expr = expr.sort_vertex_and_propagator()
            splited_expr = expr.simplify()
            result = sp.simplify(splited_expr)
            return result

        except Exception as e:
            # If an exception occurs, return original expression and print error
            logger.debug(f"Warning: Simplification of Diagram failed: {e}")
            import traceback

            traceback.print_exc()
            return expr

    # Process general list - recursively process each element in the list
    elif isinstance(expr, list):
        return [diagram_simplify(item) for item in expr]

    # Process numpy ndarray
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):  # Detect ndarray
        # Get original shape
        original_shape = expr.shape

        # Flatten ndarray to 1D array, process each element, then restore original shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array([diagram_simplify(item) for item in flattened], dtype=object)

        # Restore original shape
        return result.reshape(original_shape)

    # Recursively process addition expression
    elif isinstance(expr, Add):
        terms = []
        for term in expr.args:
            simplified_term = diagram_simplify(term)
            terms.append(simplified_term)
        return Add(*terms)

    # Recursively process multiplication expression
    elif isinstance(expr, Mul):
        factors = []
        for factor in expr.args:
            simplified_factor = diagram_simplify(factor)
            # Handle multiplication nested cases
            if isinstance(simplified_factor, Mul):
                factors.extend(simplified_factor.args)
            else:
                factors.append(simplified_factor)
        return Mul(*factors)

    # Recursively process power expression
    elif isinstance(expr, Pow):
        base = diagram_simplify(expr.args[0])
        # Keep exponent unchanged
        exponent = expr.args[1]
        return Pow(base, exponent)

    # Recursively process dictionary - process value part
    elif isinstance(expr, dict):
        return {key: diagram_simplify(value) for key, value in expr.items()}

    # Recursively process tuple - similar to list but returns tuple
    elif isinstance(expr, tuple):
        return tuple(diagram_simplify(item) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return sp.simplify(expr)

def quark_contract(expr, particles, degenerate=True):
    """
    Perform quark contraction based on hadron flavor structure in the expression

    Args:
        expr: Expression containing HadronFlavorStructure objects
        particles: List of particles
        degenerate: Whether to consider u and d quark degeneracy

    Returns:
        diagrams: Contraction diagrams
        coeffs: Coefficient list
        particles: Particle list
        propagators: Propagator list
    """
    from lqcdweave.operators.correlator.flavor_structure import HadronFlavorStructure, Qurak, Propagator
    from lqcdweave.operators.correlator.base_types import Tag
    from ._expressions import word_factors
    from .quark_exchange import flavor_preserving_symmetry, reduce_exchange_terms

    diagrams = []
    coeffs = []
    propagators = [None]
    expr = expr.expand()
    num_particles = len(particles)
    # Expand expression into sum of terms
    terms = Add.make_args(expr)
    result_terms = []
    baryon_num_list = []
    quark_sizes = []
    time_list = []
    baryon_num_list_finished = False
    for term in terms:
        # Decompose factors
        factors = word_factors(term)
        coeff = S(1)
        symbol_list = []
        exchange_symmetries = []
        hadron_id = 0
        for factor in factors:
            if isinstance(factor, HadronFlavorStructure):
                if factor.quark_symmetry:
                    flavors = factor.quark_list if factor.baryon_num == 1 else factor.anti_quark_list
                    group = flavor_preserving_symmetry(factor.quark_symmetry, flavors)
                    if len(group) > 1:
                        exchange_symmetries.append((hadron_id, group))
                if not baryon_num_list_finished:
                    baryon_num_list.append(factor.baryon_num)
                    quark_sizes.append(len(factor.quark_list) + len(factor.anti_quark_list))
                    time_list.append(factor.time)
                # Collect quarks and anti-quarks
                if factor.baryon_num == 0 and factor.quark_list:
                    symbol_list.extend(
                        [
                            Qurak(
                                factor.anti_quark_list[0],
                                Tag(hadron_id * 3, factor.time),
                                True,
                            ),
                            Qurak(
                                factor.quark_list[0],
                                Tag(hadron_id * 3, factor.time),
                                False,
                            ),
                        ]
                    )
                elif factor.baryon_num == 1:
                    quark_id = 0
                    for q in factor.quark_list:
                        symbol_list.append(
                            Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), False)
                        )
                        quark_id += 1
                elif factor.baryon_num == -1:
                    quark_id = 0
                    for q in factor.anti_quark_list:
                        symbol_list.append(
                            Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), True)
                        )
                        quark_id += 1
                hadron_id += 1
            else:
                # Non-hadron flavor structure factors as coefficients
                coeff *= factor
        baryon_num_list_finished = True
        # Perform quark contraction
        result_list = []
        result = []

        _quark_contract(symbol_list, result_list, result, degenerate)
        if exchange_symmetries:
            result_list = reduce_exchange_terms(result_list, exchange_symmetries, time_list)
        result_terms.append(coeff * Add(*result_list))
    # Merge results and simplify
    contracted = simplify(Add(*result_terms)).expand()
    if contracted == 0:
        return S(0)
    terms = Add.make_args(contracted)

    for term in terms:
        diagram = [[0 for i in range(num_particles)] for j in range(num_particles)]
        for i in range(num_particles):
            for j in range(num_particles):
                if not quark_sizes[i] or not quark_sizes[j]:
                    continue
                if baryon_num_list[i] != 0 and baryon_num_list[j] != 0:
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(3)]
                elif baryon_num_list[i] != 0 and baryon_num_list[j] == 0:
                    # The first index is the source slot and the second is the
                    # sink slot (``[source_quark][sink_quark]`` below).  A
                    # baryon source therefore needs three rows, while a meson
                    # sink contributes its single column.
                    diagram[i][j] = [[0 for _ in range(1)] for _ in range(3)]
                elif baryon_num_list[i] == 0 and baryon_num_list[j] != 0:
                    # Conversely, a meson source has one row and a baryon sink
                    # has three columns.  Keep this layout consistent with the
                    # pure meson/pure baryon matrices and with the assignment
                    # indexed by source slot first below.
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(1)]
        factors = Mul.make_args(term)
        coeff = S(1)
        for factor in factors:
            if isinstance(factor, Propagator):
                if factor.tag not in propagators:
                    propagators.append(factor.tag)
                hadron_id_source = factor.source_tag.tag // 3
                hadron_id_sink = factor.sink_tag.tag // 3
                quark_id_source = factor.source_tag.tag % 3
                quark_id_sink = factor.sink_tag.tag % 3
                if (
                    baryon_num_list[hadron_id_source] == 0
                    and baryon_num_list[hadron_id_sink] == 0
                ):
                    diagram[hadron_id_source][hadron_id_sink] = propagators.index(
                        factor.tag
                    )
                else:
                    diagram[hadron_id_source][hadron_id_sink][quark_id_source][
                        quark_id_sink
                    ] = propagators.index(factor.tag)
            else:
                coeff *= factor

        diagrams.append(diagram)
        coeffs.append(coeff)
    diagram_expr = S(0)
    for i in range(len(diagrams)):
        diagram_expr += coeffs[i] * Diagram(
            QuarkDiagram(diagrams[i]), time_list, particles, propagators
        )
    return diagram_expr


def _quark_contract(symbol_list, result_list, result, degenerate):
    from lqcdweave.operators.correlator.flavor_structure import Propagator
    from lqcdweave.operators.correlator.base_types import Tag

    if symbol_list == []:
        result_list.append(Mul(*result))
        return
    antiquark_count = sum(bool(quark.anti) for quark in symbol_list)
    if 2 * antiquark_count != len(symbol_list):
        return
    source_index = next((index for index, quark in enumerate(symbol_list) if quark.anti), None)
    if source_index is None:
        return
    i = source_index
    src = symbol_list[i]
    for j, snk in enumerate(symbol_list):
        if not snk.anti and snk.flavor == src.flavor:
            if i > j:
                symbol_list.pop(i)
                symbol_list.pop(j)
                factor = S(-1) ** (i - j - 1)
            else:
                symbol_list.pop(j)
                symbol_list.pop(i)
                factor = S(-1) ** (j - i)
            if degenerate and (snk.flavor == "u" or snk.flavor == "d"):
                prop = Propagator("q", src.tag, snk.tag)
            else:
                prop = Propagator(src.flavor, src.tag, snk.tag)
            result.append(factor * prop)
            _quark_contract(symbol_list, result_list, result, degenerate)
            result.pop()
            if i > j:
                symbol_list.insert(j, snk)
                symbol_list.insert(i, src)
            else:
                symbol_list.insert(i, src)
                symbol_list.insert(j, snk)


def remove_disconneted_diagram(
    expr: Union[Expr, List, Any], propagator_list: List[Propagator]
):
    """
    Recursively find all Diagram objects and replace those with propagators meeting certain conditions with S(0).

    This function traverses through expressions, lists, dictionaries, or other nested structures to find
    Diagram objects. If a Diagram contains any propagator from the provided propagator_list, it will be
    replaced with a symbolic zero (S(0)).

    Args:
        expr: The expression or structure to process, can be a sympy expression, list, dictionary, etc.
        propagator_list: List of Propagator objects to check against

    Returns:
        The processed expression with redundant diagrams replaced by zeros
    """
    from sympy import Add, Mul, Pow, S
    import numpy as np
    from .structured import StructuredValue, filter_structured

    if isinstance(expr, StructuredValue):
        return filter_structured(expr, propagator_list)

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Process Diagram object
    if isinstance(expr, Diagram):
        # Check if this Diagram contains any propagator from the provided propagator_list
        for prop in expr.propagator_list:
            if prop in propagator_list:
                return S(0)  # If contains, return symbolic 0
        return expr  # If not contains, remain unchanged

    # Process list
    elif isinstance(expr, list):
        return [remove_disconneted_diagram(item, propagator_list) for item in expr]

    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array(
            [remove_disconneted_diagram(item, propagator_list) for item in flattened],
            dtype=object,
        )
        return result.reshape(original_shape)

    # Process addition expression
    elif isinstance(expr, Add):
        terms = [
            remove_disconneted_diagram(term, propagator_list) for term in expr.args
        ]
        return Add(*terms)

    # Process multiplication expression
    elif isinstance(expr, Mul):
        factors = [
            remove_disconneted_diagram(factor, propagator_list) for factor in expr.args
        ]
        return Mul(*factors)

    # Process power expression
    elif isinstance(expr, Pow):
        base = remove_disconneted_diagram(expr.args[0], propagator_list)
        exponent = expr.args[1]  # Keep exponent unchanged
        return Pow(base, exponent)

    # Process dictionary
    elif isinstance(expr, dict):
        return {
            key: remove_disconneted_diagram(value, propagator_list)
            for key, value in expr.items()
        }

    # Process tuple
    elif isinstance(expr, tuple):
        return tuple(remove_disconneted_diagram(item, propagator_list) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return expr


__all__ = [
    "QuarkDiagram",
    "Diagram",
    "validate_adjacency_matrix",
    "diagram_vertice_replace",
    "diagram_simplify",
    "quark_contract",
    "remove_disconneted_diagram",
]
