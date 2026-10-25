"""Data-free mode partitions of equal-time dense meson and baryon diagrams."""
from itertools import combinations_with_replacement, permutations
from numbers import Integral

import numpy as np
from sympy import Add, Expr, Mul, Symbol

from .symbolic_graph import Diagram


class ModeBlockedDiagram(Diagram):
    """A Diagram with ordered logical mode slots selected at each vertex.

    Meson slots are the spatial axes after source daggering. Baryon slots
    retain original spatial axis order; field_order maps its adjoint graph.
    Selection identity belongs to the symbol, never to its mutable value cache.
    """

    def __new__(cls, base, selections, *, mode_extent):
        selections = tuple(tuple(tuple(s) for s in pair) for pair in selections)
        if isinstance(base, ModeBlockedDiagram):
            raise ValueError("a Diagram cannot be partitioned twice")
        if len(selections) != len(base.vertex_list) or any(len(p) not in (2, 3) for p in selections):
            raise ValueError("two or three mode selections are required per vertex")
        if any(not s or len(set(s)) != len(s) or any(type(i) is not int or i < 0 for i in s)
               for pair in selections for s in pair):
            raise ValueError("mode selections must contain distinct nonnegative positions")
        if type(mode_extent) is not int or mode_extent < 1 or any(i >= mode_extent for p in selections for s in p for i in s):
            raise ValueError("mode_extent must cover all selected positions")
        return cls._from_partition(base, selections, mode_extent=mode_extent, base_hash=hash(base))

    @classmethod
    def _from_partition(cls, base, selections, *, mode_extent, base_hash):
        """Construct from selections already validated by the partition builder."""
        obj = Symbol.__new__(cls, f"mode-block:{base.name}:{mode_extent}:{selections!r}")
        obj.base_diagram = base
        obj.mode_selections = selections
        obj.mode_extent = mode_extent
        # Symbol definitions are immutable; numerical value caches are separate.
        # In particular, avoid re-stringifying/hash-digesting the base graph on
        # every dictionary lookup of the same selected symbol.
        obj._mode_hash = hash((base_hash, mode_extent, selections))
        obj.diagram, obj.time_list = base.diagram, base.time_list
        obj.vertex_list, obj.propagator_list = base.vertex_list, base.propagator_list
        obj.value = obj.value_pointer = None
        return obj

    def __init__(self, base, selections, *, mode_extent):
        self.diagram, self.time_list = base.diagram, base.time_list
        self.vertex_list, self.propagator_list = base.vertex_list, base.propagator_list
        self.value = self.value_pointer = None

    def __eq__(self, other):
        return (isinstance(other, ModeBlockedDiagram)
                and self.base_diagram == other.base_diagram
                and self.mode_extent == other.mode_extent
                and self.mode_selections == other.mode_selections)

    def __hash__(self):
        return self._mode_hash

    def __str__(self):
        return self.name

    __repr__ = __str__

    def simplify(self):
        return self

    def transform(self, *args, **kwargs):
        raise NotImplementedError("apply symmetry transformations before split_eigenmodes")

    def conjugate(self):
        raise NotImplementedError("apply source conjugation before split_eigenmodes")


def _partitions(boundaries):
    boundaries = tuple(boundaries)
    if (len(boundaries) < 2 or boundaries[0] != 0
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) for n in boundaries)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("partitions must be increasing integer boundaries starting at zero")
    return tuple(tuple(range(int(a), int(b))) for a, b in zip(boundaries, boundaries[1:]))


def _expression_layout(value):
    """Validate multilinearity and return the ordered vertex axes of a scalar."""
    atoms = ((value,) if isinstance(value, Diagram) else
             tuple(value.atoms(Diagram)) if hasattr(value, "atoms") else ())
    if not atoms:
        return (), {}
    if any(isinstance(diagram, ModeBlockedDiagram) for diagram in atoms):
        raise ValueError("correlator is already mode-partitioned")
    keys_by_diagram = {diagram: tuple(zip(diagram.time_list, diagram.vertex_list))
                       for diagram in atoms}
    if any(len(set(keys)) != len(keys) for keys in keys_by_diagram.values()):
        raise NotImplementedError("repeated identical vertices require explicit distinct vertex tags")
    keys = tuple(sorted(set(key for entries in keys_by_diagram.values() for key in entries),
                        key=lambda key: (str(key[0]), repr(key[1]))))
    for term in Add.make_args(value):
        present = [key for factor in Mul.make_args(term) if isinstance(factor, Diagram)
                   for key in keys_by_diagram[factor]]
        if len(present) != len(keys) or set(present) != set(keys):
            raise NotImplementedError("each term must contain the same vertices exactly once")
    return keys, keys_by_diagram


def split_eigenmodes(correlator, *, partitions):
    """Replace each scalar meson expression by its complete mode-block array.

    Each distinct (time, vertex) contributes two consecutive block axes, sorted
    by its symbolic time and vertex representation. Every additive term must
    cover those same vertices. Products of disconnected loops remain products.
    No data are read. Summing the new axes restores the original expression.
    """
    return _split_mode_slots(correlator, partitions=partitions, slots=2)


def split_baryon_eigenmodes(correlator, *, partitions):
    """Append three ordered mode axes per equal-time baryon vertex.

    Axis order is the original spatial slots 0,1,2, including at the source.
    BaryonDiagramBinding maps source graph slots through its field_order.
    Pure baryon graphs only; source adjoints must be constructed first.
    """
    return _split_mode_slots(correlator, partitions=partitions, slots=3)


def _split_mode_slots(correlator, *, partitions, slots):
    from .structured import StructuredCorrelator, StructuredOperation, StructuredValue

    ranges = _partitions(partitions)
    diagram_blocks, expressions = {}, {}

    def blocks_for(diagram):
        if diagram not in diagram_blocks:
            result = np.empty((len(ranges),) * (slots * len(diagram.vertex_list)), dtype=object)
            base_hash = hash(diagram)
            for block in np.ndindex(result.shape):
                selections = tuple(tuple(ranges[block[slots * i + j]] for j in range(slots))
                                   for i in range(len(diagram.vertex_list)))
                result[block] = ModeBlockedDiagram._from_partition(
                    diagram, selections, mode_extent=ranges[-1][-1] + 1, base_hash=base_hash,
                )
            diagram_blocks[diagram] = result
        return diagram_blocks[diagram]

    def visit(value):
        if slots == 3 and isinstance(value, (StructuredCorrelator, StructuredOperation, StructuredValue)):
            raise NotImplementedError(
                "baryon mode blocks require Diagram expressions; use gen_correlator(..., structured=False)"
            )
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
            if children and all(isinstance(c, np.ndarray) and c.shape == children[0].shape for c in children):
                return np.stack(children).reshape((*value.shape, *children[0].shape))
            result = np.empty(value.shape, dtype=object)
            for index, child in zip(np.ndindex(value.shape), children):
                result[index] = child
            return result
        if isinstance(value, Expr) and value in expressions:
            # Public output arrays own their slots even when their immutable
            # symbolic leaves come from the same call-local template.
            return expressions[value].copy()
        keys, keys_by_diagram = _expression_layout(value)
        if not keys_by_diagram:
            return value
        positions = {key: i for i, key in enumerate(keys)}
        selected = tuple((diagram, blocks_for(diagram),
                          tuple(axis for key in entries
                                for axis in range(slots * positions[key], slots * (positions[key] + 1))))
                         for diagram, entries in keys_by_diagram.items())
        if isinstance(value, Diagram):
            # The scalar is the leaf being replaced. Reorder its prebuilt slots
            # directly; no expression traversal or replacement dictionary is
            # needed for every ordered block.
            _, variants, axes = selected[0]
            result = variants.transpose(tuple(np.argsort(axes))).copy()
        else:
            result = np.empty((len(ranges),) * (slots * len(keys)), dtype=object)
            for block in np.ndindex(result.shape):
                replacements = {diagram: variants[tuple(block[axis] for axis in axes)]
                                for diagram, variants, axes in selected}
                result[block] = value.xreplace(replacements)
        expressions[value] = result
        return result

    return visit(correlator)


def project_meson_mode_pairs(blocks, *, mode_axis=0, vertices=2):
    """Replace contiguous ordered mode-pair axes by PC-preserving pair axes.

    Pair order is (0,0),(0,1),...,(1,1),... . Diagonal weight is one;
    off-diagonal pairs have equal weights 1/sqrt(2). Other axes stay in order.
    This is also valid for object arrays of symbolic Diagram expressions.
    """
    value = np.asarray(blocks)
    if type(mode_axis) is not int or type(vertices) is not int or vertices < 1:
        raise ValueError("mode_axis and vertices must be integers; vertices must be positive")
    if mode_axis < 0 or mode_axis + 2 * vertices > value.ndim:
        raise ValueError("requested mode axes are outside the array")
    for vertex in range(vertices):
        axis = mode_axis + vertex
        m, n = value.shape[axis:axis + 2]
        if m != n:
            raise ValueError("PC pairing requires the same partition on both mode legs")
        ordered = np.asarray(np.moveaxis(value, (axis, axis + 1), (0, 1)),
                             dtype=np.result_type(value.dtype, np.float64))
        paired = np.empty((m * (m + 1) // 2, *ordered.shape[2:]),
                          dtype=ordered.dtype)
        scale = 1 / np.sqrt(2)
        for row, (a, b) in enumerate((a, b) for a in range(m) for b in range(a, m)):
            if a == b:
                paired[row] = ordered[a, b]
            else:
                paired[row] = scale * ordered[a, b] + scale * ordered[b, a]
        value = np.moveaxis(paired, 0, axis)
    return value


def project_baryon_mode_triples(blocks, *, mode_axis=0, vertices=2):
    """Combine distinct permutations of each triple with weight 1/sqrt(m).

    Triples are ordered a<=b<=c lexicographically. Restoration weight is
    sqrt(m), with m=1,3,6. This symmetric filter preserves an already valid
    spin/flavor/spatial exchange projection; it does not construct that one.
    """
    value = np.asarray(blocks)
    if type(mode_axis) is not int or type(vertices) is not int or vertices < 1:
        raise ValueError("mode_axis and vertices must be integers; vertices must be positive")
    if mode_axis < 0 or mode_axis + 3 * vertices > value.ndim:
        raise ValueError("requested mode axes are outside the array")
    for vertex in range(vertices):
        axis = mode_axis + vertex
        sizes = value.shape[axis:axis + 3]
        if len(set(sizes)) != 1 or sizes[0] < 1:
            raise ValueError("symmetric triples require the same nonempty partition on all legs")
        ordered = np.asarray(np.moveaxis(value, (axis, axis+1, axis+2), (0, 1, 2)),
                             dtype=np.result_type(value.dtype, np.float64))
        triples = tuple(combinations_with_replacement(range(sizes[0]), 3))
        combined = np.empty((len(triples), *ordered.shape[3:]), dtype=ordered.dtype)
        for row, triple in enumerate(triples):
            orbit = tuple(sorted(set(permutations(triple))))
            combined[row] = sum(ordered[index] for index in orbit) / np.sqrt(len(orbit))
        value = np.moveaxis(combined, 0, axis)
    return value
