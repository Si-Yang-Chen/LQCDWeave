"""Single-current point compensation on already computed spatial blocks."""

from lqcdweave.contracts.elemental import SparsePointBlock
from lqcdweave.contracts.sampling_weights import expected_weight


def weight_point_block(value, sector, left_pool, right_pool, groups, *, array_module):
    """Weight a producer-owned block in place; COO support stays unchanged.

    Same-pool indices refer to the same ordered sample labels. This function
    never establishes pool identity from coordinates or array shapes.
    """
    xp = array_module
    if sector == "v2v":
        return value
    if sector in ("v2p", "p2v"):
        pool = groups[right_pool if sector == "v2p" else left_pool]
        value *= float(expected_weight(pool.spatial_sites, pool.sampled_points, 1))
        return value
    left, right = groups[left_pool], groups[right_pool]
    if left_pool != right_pool:
        weight = (expected_weight(left.spatial_sites, left.sampled_points, 1)
                  * expected_weight(right.spatial_sites, right.sampled_points, 1))
        if isinstance(value, SparsePointBlock):
            value.values *= float(weight)
        else:
            value *= float(weight)
        return value
    diagonal = float(expected_weight(left.spatial_sites, left.sampled_points, 1))
    if isinstance(value, SparsePointBlock) and value.point_diagonal:
        value.values *= diagonal
        return value
    distinct = float(expected_weight(left.spatial_sites, left.sampled_points, 2))
    if isinstance(value, SparsePointBlock):
        weights = xp.where(value.rows == value.columns, diagonal, distinct)
        value.values *= weights[:, None, None]
    else:
        value *= distinct
        indices = xp.arange(min(value.shape[0], value.shape[2]))
        value[indices, :, indices, :] *= diagonal / distinct
    return value
