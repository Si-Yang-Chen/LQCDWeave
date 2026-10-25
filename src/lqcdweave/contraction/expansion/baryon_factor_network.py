"""Row-packed dense-V networks with ordinary coefficient and spatial factors.

A whole baryon vertex is ``V[r] = sum_p C_p[r, s0, s1, s2] * Phi_p[n0, n1, n2]``
over its parts ``p``: the spin parts of one leg binding that share one ordered
Phi are one exact coefficient constant. Physical rows whose parts share all
data except those constants become one retained output axis. A Meson or
current vertex keeps its described factors, one part each, and a glueball
singleton stays one data term. Each choice of one part per vertex is one
ordinary network; the planner then sees ordinary operands and may contract
either factor with an attached propagator.

This module only groups metadata; it reads no arrays and changes no Wick graph,
flavor weight, source adjoint or output coefficient.
"""
from dataclasses import dataclass, replace
from itertools import product
from math import prod

import numpy as np


@dataclass(frozen=True)
class PackedComponent:
    """One connected component of one Diagram and its row selection."""
    key: tuple
    vertices: tuple[int, ...]
    edges: tuple
    selection: tuple[int, ...]


@dataclass(frozen=True)
class FactorNetwork:
    rows: dict            # row instance (a row, or a row and its mode block) -> bound vertex
    parts: dict           # (instance, anchor) -> summed parts of that vertex
    families: dict        # component key -> per-vertex ordered instance tuples
    diagrams: tuple       # per Diagram: (anchors, instances, tuple[PackedComponent] or None for a glueball)


def _stable(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _phi_signature(vertex, description):
    from lqcdweave.contracts.correlator.api import canonical_axis_binding_identity
    term, tensor = description.operator_term, description.tensor
    return (_stable(term.component), _stable(term.momentum), description.time,
            tensor.selection_identity, tensor.shape, str(tensor.dtype),
            canonical_axis_binding_identity(tensor.axis_bindings), description.mode_indices,
            _stable(description.usedNe), tuple(description.field_order), description.dagger)


def plan_factor_network(diagrams, bindings, time_values):
    """Return the row-packed grouping of a dense-V request.

    A Baryon part is the spin parts of one leg binding sharing one Phi; Baryon
    rows with equal parts (all data but the coefficients) pack, also single
    rows. Meson and current rows stay single, one part per described factor.
    Mode blocks select their Baryon slots first. Requested row combinations
    that are not a complete product are split into complete sub-products,
    one packed term each.
    """
    from lqcdweave.contraction.binding.api import Baryon, Glueball, bind_slot_vertices
    from lqcdweave.operators.correlator.api import ModeBlockedDiagram
    from .baryon_diagram import (_components_from_edges, _extract_dense_v_graph, _group_factor_descriptions,
                                 _meson_used_ne, _resolve_dense_vertex, _resolve_time)

    raw, instances = {}, []
    for diagram in diagrams:
        blocked = isinstance(diagram, ModeBlockedDiagram)
        if blocked and bindings.usedNe != diagram.mode_extent:
            raise ValueError("partition endpoint must equal the explicit requested Ne")
        keys = []
        for index, row in enumerate(diagram.vertex_list):
            key = (row, diagram.mode_selections[index]) if blocked else row
            if key not in raw:
                vertex = _resolve_dense_vertex(bindings.vertex_map, row)
                if blocked:
                    if not isinstance(vertex, Baryon):
                        raise NotImplementedError("baryon mode partitions require pure Baryon graphs")
                    vertex = vertex.with_mode_indices(diagram.mode_selections[index])
                raw[key] = vertex
            keys.append(key)
        instances.append(tuple(keys))
    ordered = tuple(raw)
    bound = bind_slot_vertices(tuple(raw[key] for key in ordered),
                               bindings.propagator_map.values())
    rows = dict(zip(ordered, bound, strict=True))
    parts, signatures = {}, {}

    def describe(key, anchor):
        if (key, anchor) in signatures:
            return signatures[key, anchor]
        vertex = rows[key]
        if isinstance(vertex, Baryon):
            grouped = {}
            for core in vertex.describe_core_factors(anchor, usedNe=bindings.usedNe):
                binding = _stable(vertex._core_binding(core.factors[0]))
                for item in core.factors:
                    grouped.setdefault((binding, _phi_signature(vertex, item)), []).append(item)
            # One canonical part order, so rows with equal parts pack.
            order = tuple(sorted(grouped, key=repr))
            parts[key, anchor] = tuple(tuple(grouped[part]) for part in order)
            signature = ("baryon", id(vertex.reader), vertex.dagger, vertex.spin_basis,
                         vertex.phase_convention, vertex.boundary, anchor, order)
        else:
            if any("d" in sector for sector in (vertex.requested_sectors or ())):
                raise NotImplementedError("mixed baryon graphs do not yet support D elemental sectors")
            factors = vertex.describe_core_factors(anchor, "v2v", usedNe=_meson_used_ne(bindings.usedNe))
            if not factors:
                raise ValueError("Meson/current operator has no dense-v factors")
            parts[key, anchor] = tuple(item for group in _group_factor_descriptions(vertex, tuple(factors))
                                       for item in group)
            signature = ("meson", key, anchor)
        signatures[key, anchor] = signature
        return signature

    selections, packed = {}, []
    for diagram, keys in zip(diagrams, instances, strict=True):
        vertices = tuple(rows[key] for key in keys)
        anchors = tuple(_resolve_time(value, time_values) for value in diagram.time_list)
        if len(vertices) == 1 and isinstance(vertices[0], Glueball):
            packed.append((anchors, keys, None))
            continue
        edges, roles = _extract_dense_v_graph(diagram.diagram.adjacency_matrix, vertices)
        for index, (vertex, role) in enumerate(zip(vertices, roles, strict=True)):
            if isinstance(vertex, Baryon) and vertex.dagger != (role.orientation == "source"):
                raise ValueError(f"Baryon vertex {index} dagger state disagrees with its graph orientation")
        components = []
        for component_edges in _components_from_edges(len(vertices), edges):
            members = tuple(sorted({v for edge in component_edges for v in (edge.source, edge.sink)}))
            signature = tuple(describe(keys[index], anchors[index]) for index in members)
            position = {vertex: slot for slot, vertex in enumerate(members)}
            topology = tuple(sorted((position[edge.source], position[edge.sink], edge.source_slot,
                                     edge.sink_slot, str(diagram.propagator_list[edge.propagator]))
                                    for edge in component_edges))
            key = (signature, topology)
            used = selections.setdefault(key, [dict() for _ in members])
            selection = tuple(used[slot].setdefault(keys[index], len(used[slot]))
                              for slot, index in enumerate(members))
            components.append(PackedComponent(key, members, tuple(component_edges), selection))
        packed.append((anchors, keys, tuple(components)))
    families = {key: tuple(tuple(slot) for slot in used) for key, used in selections.items()}
    requested = {}
    for _, _, components in packed:
        for component in components or ():
            requested.setdefault(component.key, set()).add(component.selection)
    # A partial row request is split exactly into complete sub-products.
    covers = {}
    for key, used in requested.items():
        if len(used) == prod(len(slot) for slot in families[key]):
            continue
        covers[key] = {}
        for index, block in enumerate(_product_cover(used)):
            part = (key, index)
            families[part] = tuple(tuple(families[key][slot][i] for i in rows)
                                   for slot, rows in enumerate(block))
            for selection in product(*block):
                covers[key][selection] = part, tuple(rows.index(i) for rows, i in zip(block, selection))
        del families[key]
    if covers:
        packed = [(anchors, keys, None if components is None else tuple(
            replace(component, key=covers[component.key][component.selection][0],
                    selection=covers[component.key][component.selection][1])
            if component.key in covers else component for component in components))
            for anchors, keys, components in packed]
    return FactorNetwork(rows, parts, families, tuple(packed))


def _product_cover(selections):
    """Partition row-index tuples into complete products (greedy, no overlap)."""
    remaining, cover = set(selections), []
    while remaining:
        seed = min(remaining)
        block = [[value] for value in seed]
        for slot in range(len(seed)):
            for value in sorted({item[slot] for item in remaining} - {seed[slot]}):
                trial = block[:slot] + [[value]] + block[slot + 1:]
                if all(item in remaining for item in product(*trial)):
                    block[slot].append(value)
        block = tuple(tuple(sorted(rows)) for rows in block)
        remaining -= set(product(*block))
        cover.append(block)
    return cover


def merged_coefficients(vertex, part, spin_entries_once):
    """Exact weighted sum of one row part's spin parts in the bound spin basis."""
    value = np.zeros((4, 4, 4), dtype=np.complex128)
    for description in part:
        for a, b, c, coefficient in spin_entries_once(vertex, description):
            value[a, b, c] += description.coefficient * coefficient
    return value


@dataclass(frozen=True)
class RowFactor:
    """``C[r, s0, s1, s2] = sum_k pair[k, sa, sb] * row[r, sj, k]`` with ``(sa, sb) = pairs``.

    ``rank`` 1 drops the ``k`` axis: ``pair`` is ``(4, 4)`` and ``row`` is
    ``(R, 4)``. When every row of a rank-1 ``row`` has at most one nonzero,
    ``row`` is ``None`` and row ``r`` reads spin ``spins[r]`` of slot ``slot``
    times ``factors[r]``.
    """
    slot: int
    pairs: tuple[int, int]
    pair: np.ndarray
    row: np.ndarray | None
    spins: tuple[int, ...] | None
    factors: tuple[complex, ...] | None
    rank: int = 1


@dataclass(frozen=True)
class EntryFactor:
    """``C[r, s0, s1, s2] = sum_n weight[r, n] * prod_j select[j][n, s_j]``.

    ``n`` runs over the nonzero spin positions of any row; each ``select[j]``
    is one-hot. Every spin slot is a separate operand, whatever the Gamma
    structure, so the planner sees all the sparsity of ``C``.
    """
    weight: np.ndarray
    select: tuple[np.ndarray, np.ndarray, np.ndarray]


def entry_row_coefficients(values, tolerance=1e-13):
    """Split a row-stacked ``(R, 4, 4, 4)`` constant into its nonzero entries, or ``None``."""
    scale = float(np.abs(values).max())
    if scale == 0.0:
        return None
    positions = np.argwhere((np.abs(values) > tolerance * scale).any(axis=0))
    weight = values[:, positions[:, 0], positions[:, 1], positions[:, 2]]
    eye = np.eye(4, dtype=np.complex128)
    return EntryFactor(weight, tuple(eye[positions[:, j]] for j in range(3)))


def _row_skeleton(matrix, threshold):
    """Indices of linearly independent rows spanning ``matrix`` (pivoted Gram-Schmidt)."""
    residual = matrix.astype(np.complex128, copy=True)
    chosen = []
    while np.abs(residual).max() > threshold and len(chosen) < min(matrix.shape):
        index = int(np.linalg.norm(residual, axis=1).argmax())
        chosen.append(index)
        basis = residual[index] / np.linalg.norm(residual[index])
        residual -= np.multiply.outer(residual @ basis.conj(), basis)
    return chosen


def factor_row_coefficients(values, tolerance=1e-13, *, bounded=True):
    """Split a row-stacked ``(R, 4, 4, 4)`` constant at one spin slot, or ``None``.

    Baryon operators usually carry the row on one quark spin and a diquark
    matrix on the other two. Keeping ``C`` dense couples all three spins and
    the row in every contraction it joins; the split leaves 4x4 pair operands.
    The rank-one form is preferred; otherwise the lowest exact rank ``k`` over
    the three slots is used, while ``bounded`` only while the factors hold
    fewer entries than ``C``. Every split reconstructs ``C`` within
    ``tolerance`` of its largest entry; anything else stays dense.
    """
    scale = float(np.abs(values).max())
    if scale == 0.0:
        return None
    threshold, rows = tolerance * scale, len(values)
    general = None
    for slot in range(3):
        matrix = np.moveaxis(values, 1 + slot, 1).reshape(rows * 4, 16)
        pairs = tuple(k for k in range(3) if k != slot)
        p, q = np.unravel_index(int(np.abs(matrix).argmax()), matrix.shape)
        row, pair = matrix[:, q] / matrix[p, q], matrix[p]
        if np.abs(np.multiply.outer(row, pair) - matrix).max() <= threshold:
            row = row.reshape(rows, 4)
            nonzero = np.abs(row) > tolerance * np.abs(row).max()
            if (nonzero.sum(axis=1) <= 1).all():
                spins = nonzero.argmax(axis=1)
                factors = np.where(nonzero.any(axis=1), row[np.arange(len(row)), spins], 0)
                return RowFactor(slot, pairs, pair.reshape(4, 4), None,
                                 tuple(map(int, spins)), tuple(map(complex, factors)))
            if general is None or general.rank > 1:
                general = RowFactor(slot, pairs, pair.reshape(4, 4), row, None, None)
            continue
        if general is not None and general.rank == 1:
            continue
        chosen = _row_skeleton(matrix, threshold)
        rank = len(chosen)
        # Factors must stay smaller than C, or the split only adds operands.
        if (general is not None and rank >= general.rank) or (
                bounded and rank * (16 + 4 * rows) >= 64 * rows):
            continue
        pair = matrix[chosen]
        weights = np.linalg.lstsq(pair.T, matrix.T, rcond=None)[0].T
        weights[np.abs(weights) <= tolerance * np.abs(weights).max()] = 0
        if np.abs(weights @ pair - matrix).max() > threshold:
            continue
        general = RowFactor(slot, pairs, pair.reshape(rank, 4, 4), weights.reshape(rows, 4, rank),
                            None, None, rank)
    return general


#: Opened forms of one row family's coefficients, all exact: the two-part
#: split at any rank, the nonzero entries, or the whole ``(R, 4, 4, 4)`` constant.
ROW_LOWERINGS = ("pair", "nonzero", "dense")


def auto_row_lowering(values):
    """The opened form ``"auto"`` gives a family, by its coefficient structure.

    The split rank never exceeds the number of nonzero spin positions. Measured
    per structure (docs/modules/expansion/baryon-factor-network.md): while the split
    compresses entries (a fold, rank one, or fewer pair matrices than nonzero
    positions) it is fastest or within a few percent; when every entry needs its
    own pair matrix the nonzero entries are faster. The dense constant is never
    fastest.
    """
    split = factor_row_coefficients(values, bounded=False)
    if split is None:
        return "nonzero"
    if split.rank == 1:
        return "pair"
    return "nonzero" if split.rank >= entry_row_coefficients(values).weight.shape[1] else "pair"


__all__ = ["EntryFactor", "FactorNetwork", "PackedComponent", "ROW_LOWERINGS", "RowFactor",
           "auto_row_lowering", "entry_row_coefficients", "factor_row_coefficients", "merged_coefficients",
           "plan_factor_network"]
