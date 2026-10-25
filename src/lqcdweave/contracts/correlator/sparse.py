"""Pure sparse point-label semantics and conservative workspace features.
Copied from the existing sparse kernels; no numerical execution imports.
"""
from __future__ import annotations
from .indices import AxisPattern, axis_ids, parse_axis_pattern
_ENDPOINTS = ("first_left", "first_right", "second_left", "second_right")
DEFAULT_JOIN_INDEX_BLOCK_SIZE = 64
ESTIMATED_JOIN_INDEX_BYTES_PER_ROW = 512
JOIN_COST_STATUS = "conservative_upper_bound_unmeasured"
def _parse_cycle_equation(equation):
    pattern = parse_axis_pattern(equation)
    labels, output = pattern.inputs, pattern.output
    if len(labels) != 2 or any(not part for part in labels):
        raise ValueError("sparse join equation must describe two nonempty operands")
    if len(output) != len(set(output)):
        raise ValueError("sparse join output labels must be unique")
    return labels[0], labels[1], output


def _second_slots(first_bridge, second_bridge):
    first_ids = (first_bridge.left, first_bridge.right)
    second_ids = (second_bridge.left, second_bridge.right)
    if second_ids == first_ids:
        return (0, 1)
    if first_ids[0] != first_ids[1] and second_ids == first_ids[::-1]:
        return (1, 0)
    if first_ids[0] == first_ids[1]:
        return (0, 1)
    raise ValueError("two sparse records must join the same ordered propagator pair")


def _endpoint_axes(first_bridge, second_bridge):
    second_slots = _second_slots(first_bridge, second_bridge)
    axes = {
        "first_left": (0, tuple(first_bridge.left_axes)),
        "first_right": (1, tuple(first_bridge.right_axes)),
        "second_left": (second_slots[0], tuple(second_bridge.left_axes)),
        "second_right": (second_slots[1], tuple(second_bridge.right_axes)),
    }
    return axes


def _root(parent, value):
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent, left, right):
    left, right = _root(parent, left), _root(parent, right)
    if left != right:
        parent[right] = left


def _canonical_partition(parent):
    roots = [_root(parent, index) for index in range(len(parent))]
    names, result = {}, []
    for root in roots:
        if root not in names:
            names[root] = len(names)
        result.append(names[root])
    return tuple(result)


def cycle_point_equivalence(equation, first_bridge, second_bridge):
    """Return explicit equivalence classes for the four bridged point axes.

    Endpoint names are ``first_left``, ``first_right``, ``second_left``, and
    ``second_right``. Equal labels in the two propagator subscripts and each
    bridge's ``same_point`` flag are combined into one frozen symbolic class.
    """
    left_labels, right_labels, _ = _parse_cycle_equation(equation)
    operand_labels = (left_labels, right_labels)
    axes = _endpoint_axes(first_bridge, second_bridge)
    parent = list(range(4))
    point_label_by_endpoint = {}
    for index, name in enumerate(_ENDPOINTS):
        slot, triple = axes[name]
        labels = operand_labels[slot]
        if len(triple) != 3:
            raise ValueError("each sparse bridge endpoint requires spin, point and color axes")
        point_axis = triple[1] % len(labels)
        point_label_by_endpoint[name] = labels[point_axis]
        for previous in _ENDPOINTS[:index]:
            if point_label_by_endpoint[previous] == point_label_by_endpoint[name]:
                _union(parent, index, _ENDPOINTS.index(previous))
    if first_bridge.same_point:
        _union(parent, 0, 1)
    if second_bridge.same_point:
        _union(parent, 2, 3)
    canonical = _canonical_partition(parent)
    return {name: canonical[index] for index, name in enumerate(_ENDPOINTS)}


def cycle_point_equalities_covered(equation, first_bridge, second_bridge):
    """Whether the legacy cycle masks every explicit endpoint equality.

    It knows repeated point labels within one operand and the two record-local
    diagonal flags. A class equating endpoints in opposite propagators through
    different current records needs :func:`contract_sparse_p2p_join`.
    """
    left_labels, right_labels, _ = _parse_cycle_equation(equation)
    operands = (left_labels, right_labels)
    axes = _endpoint_axes(first_bridge, second_bridge)
    point_labels = {}
    expected = list(range(4))
    local = list(range(4))
    for index, name in enumerate(_ENDPOINTS):
        slot, triple = axes[name]
        point_labels[name] = operands[slot][triple[1] % len(operands[slot])]
        for prior in _ENDPOINTS[:index]:
            if point_labels[prior] == point_labels[name]:
                _union(expected, index, _ENDPOINTS.index(prior))
                if axes[prior][0] == slot:
                    _union(local, index, _ENDPOINTS.index(prior))
    if first_bridge.same_point:
        _union(expected, 0, 1)
        _union(local, 0, 1)
    if second_bridge.same_point:
        _union(expected, 2, 3)
        _union(local, 2, 3)
    return _canonical_partition(expected) == _canonical_partition(local)


def _cycle_endpoint_point_labels(equation, first_bridge, second_bridge):
    left, right, _ = _parse_cycle_equation(equation)
    operands = (left, right)
    return tuple(dict.fromkeys(
        operands[slot][axes[1] % len(operands[slot])]
        for slot, axes in _endpoint_axes(first_bridge, second_bridge).values()
    ))


def cycle_requires_generic_join(equation, first_bridge, second_bridge):
    """Keep exported point coordinates while joining both sparse records."""
    _, _, output = cycle_external_labels(equation, first_bridge, second_bridge)
    return bool(set(output).intersection(
        _cycle_endpoint_point_labels(equation, first_bridge, second_bridge)
    )) or not cycle_point_equalities_covered(equation, first_bridge, second_bridge)


def cycle_intermediate_labels(equation, first_bridge, second_bridge, *, needed_labels):
    """Describe the integer axes needed after consuming both sparse records.

    Endpoint point axes needed by another factor survive even when the final
    term output is scalar. Consumed spin/color axes never survive this step.
    """
    if not isinstance(needed_labels, (str, tuple)):
        raise TypeError("needed_labels must be an ordered tuple of integer axes or external labels")
    needed = axis_ids(needed_labels)
    first, external, output = cycle_external_labels(equation, first_bridge, second_bridge)
    selected = tuple(dict.fromkeys(
        label for label in external + _cycle_endpoint_point_labels(
            equation, first_bridge, second_bridge) if label in needed
    ))
    return first, selected, output


def bridge_point_equivalence(equation, left_axes, right_axes, *, same_point=False):
    """Return the explicit equality classes for one COO bridge's point axes."""
    left_labels, right_labels, _ = _parse_cycle_equation(equation)
    axes = (tuple(left_axes), tuple(right_axes))
    labels = (
        left_labels[axes[0][1] % len(left_labels)],
        right_labels[axes[1][1] % len(right_labels)],
    )
    if same_point or labels[0] == labels[1]:
        return {"left": 0, "right": 0}
    return {"left": 0, "right": 1}


def bridge_point_equalities_covered(equation, left_axes, right_axes, *, same_point=False):
    """Whether the legacy one-record bridge can enforce all point equalities.

    It handles an equality between the two COO endpoints. An endpoint label
    repeated on any additional point/operand axis requires streaming indexed
    accumulation so that external point axes follow the COO coordinate.
    """
    left_labels, right_labels, _ = _parse_cycle_equation(equation)
    left_axes, right_axes = tuple(left_axes), tuple(right_axes)
    endpoint_labels = (
        left_labels[left_axes[1] % len(left_labels)],
        right_labels[right_axes[1] % len(right_labels)],
    )
    endpoint_counts = {label: endpoint_labels.count(label) for label in set(endpoint_labels)}
    all_counts = {
        label: left_labels.count(label) + right_labels.count(label)
        for label in endpoint_counts
    }
    return all(all_counts[label] == count for label, count in endpoint_counts.items())


def bridge_intermediate_labels(
    equation: AxisPattern,
    left_axes: tuple[int, int, int],
    right_axes: tuple[int, int, int],
    *,
    same_point: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return external, raw-kernel, and planned output labels for one bridge."""
    external, output = bridge_external_labels(
        equation, left_axes, right_axes, same_point=same_point
    )
    left, right, _ = _parse_cycle_equation(equation)
    point_labels = (
        left[left_axes[1] % len(left)],
        right[right_axes[1] % len(right)],
    )
    retained = tuple(
        dict.fromkeys(label for label in point_labels if label in output)
    )
    return external, external + retained, output


def cycle_external_labels(equation: AxisPattern, first_bridge: Any, second_bridge: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Describe the raw and final index order of a two-current sparse cycle."""
    left, right, output = _parse_cycle_equation(equation)
    first_ids = (first_bridge.left, first_bridge.right)
    second_ids = (second_bridge.left, second_bridge.right)
    if set(first_ids) != set(second_ids):
        raise ValueError("P2P cycle must join the same two propagator identities")
    labels = (left, right)
    first_axes = tuple(
        {axis % len(term_labels) for axis in axes}
        for term_labels, axes in zip(labels, (first_bridge.left_axes, first_bridge.right_axes), strict=True)
    )
    if any(len(axes) != 3 for axes in first_axes):
        raise ValueError("P2P endpoint axes must be distinct")
    if second_ids == first_ids:
        second_axes_spec = (second_bridge.left_axes, second_bridge.right_axes)
    elif first_ids[0] != first_ids[1] and second_ids == first_ids[::-1]:
        second_axes_spec = (second_bridge.right_axes, second_bridge.left_axes)
    else:
        # A self edge has one tensor identity but two distinct operand slots;
        # bridge.left/right preserve those slots in equation order.
        second_axes_spec = (second_bridge.left_axes, second_bridge.right_axes)
    second_axes = tuple(
        {axis % len(term_labels) for axis in axes}
        for term_labels, axes in zip(labels, second_axes_spec, strict=True)
    )
    if any(len(axes) != 3 for axes in second_axes):
        raise ValueError("P2P endpoint axes must be distinct")
    if any(first_axes[slot] & second_axes[slot] for slot in range(2)):
        raise ValueError("two P2P currents cannot consume the same endpoint axis")
    first_external = tuple(
        label
        for slot in range(2)
        for axis, label in enumerate(labels[slot])
        if axis not in first_axes[slot]
    )
    final_external = tuple(
        label
        for slot in range(2)
        for axis, label in enumerate(labels[slot])
        if axis not in first_axes[slot] | second_axes[slot]
    )
    endpoint_labels = _cycle_endpoint_point_labels(equation, first_bridge, second_bridge)
    if not set(output) <= set(final_external) | set(endpoint_labels):
        raise ValueError("P2P cycle output is not formed from remaining or endpoint point axes")
    final_external += tuple(label for label in output
                            if label in endpoint_labels and label not in final_external)
    return first_external, final_external, output


# Public descriptor helpers consumed by both planning and numerical replay.
parse_sparse_equation = _parse_cycle_equation
sparse_cycle_endpoint_axes = _endpoint_axes
sparse_cycle_slots = _second_slots
SPARSE_ENDPOINT_NAMES = _ENDPOINTS


def bridge_external_labels(
    equation: AxisPattern,
    left_axes: tuple[int, int, int],
    right_axes: tuple[int, int, int],
    *,
    same_point: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the axes retained by the sparse kernel and the planned output.

    Either endpoint point axis may be retained independently. Exporting a
    point label does not introduce an equality between the two endpoints.
    """
    left, right, output = _parse_cycle_equation(equation)
    kept = []
    for labels, axes in ((left, left_axes), (right, right_axes)):
        contracted = {axis % len(labels) for axis in axes}
        if len(contracted) != 3:
            raise ValueError("P2P endpoint axes must be distinct")
        kept.append(tuple(label for index, label in enumerate(labels) if index not in contracted))
    external = tuple(label for axes in kept for label in axes)
    left_point_label = left[left_axes[1] % len(left)]
    right_point_label = right[right_axes[1] % len(right)]
    point_labels = {left_point_label, right_point_label}
    if not set(output) <= set(external) | point_labels or len(set(output)) != len(output):
        raise ValueError("P2P planned output is not formed from external or endpoint point axes")
    return external, output

