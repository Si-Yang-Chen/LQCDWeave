"""Local component-inclusion weights for actual D endpoint labels."""


from lqcdweave.contracts.dilution import DILUTION_WEIGHT_SCHEME


def selected_dilution_groups(source, indices):
    """Resolve scalar storage labels to their declared scientific groups."""
    positions = {index: position for position, index in enumerate(indices)}
    result = []
    for group_index, group in enumerate(source.groups):
        selected = tuple(index for index in source.group_indices(group_index) if index in positions)
        if selected:
            result.append((group_index, group, tuple(positions[index] for index in selected),
                           tuple(source.vector_labels[index] for index in selected)))
    return tuple(result)


def weight_dilution_pairs(value, left_source, right_source, left_groups, right_groups,
                          left_weights, right_weights, left_time, right_time,
                          structure, *, array_module):
    """Weight a producer-owned DD matrix without changing its component support.

    Different independent groups/pools use two first moments. Within the same
    raw QR group/pool, equal scientific components use H1 and distinct ones H2.
    The structural diagonal promise is supplied by production, never scanned.
    """
    xp = array_module
    adjustments = []
    for left_index, left_group, left_positions, left_labels in left_groups:
        for right_index, right_group, right_positions, right_labels in right_groups:
            if ((left_group.branch == "exact" or left_group.m == left_group.n)
                    and (right_group.branch == "exact" or right_group.m == right_group.n)):
                continue
            left_pool = left_source.pool_key(left_time, left_index)
            right_pool = right_source.pool_key(right_time, right_index)
            if left_pool is None or right_pool is None:
                raise ValueError("weighted random DD elemental requires explicit physical-time pool identities")
            if left_pool != right_pool:
                continue
            if (left_group.n, left_group.m, left_group.branch, left_group.covariance_factor, left_group.vector_labels) != (
                    right_group.n, right_group.m, right_group.branch, right_group.covariance_factor, right_group.vector_labels):
                raise ValueError("one DD pool has inconsistent dilution group metadata")
            if left_group.branch == "u1":
                raise ValueError("legacy u1 same-pool DD component weights are unsupported")
            n, m = left_group.n, left_group.m
            if m < 2 and structure != "diagonal":
                raise ValueError("rank-one same-pool DD requires a proven complete diagonal elemental")
            same = xp.asarray([[left == right for right in right_labels] for left in left_labels])
            h1 = n / m
            # No off-diagonal component is present in a proven diagonal block.
            h2 = h1 if structure == "diagonal" else n * (n - 1) / (m * (m - 1))
            ratio = xp.where(same, h1, h2) / (left_group.first_moment_weight * right_group.first_moment_weight)
            adjustments.append((left_positions, right_positions, ratio))
    # Validate every statistical branch before mutating the borrowed block.
    value *= left_weights[:, None] * right_weights[None, :]
    for left_positions, right_positions, ratio in adjustments:
        left = xp.asarray(left_positions)[:, None]
        right = xp.asarray(right_positions)[None, :]
        value[left, right] *= ratio
    return value
