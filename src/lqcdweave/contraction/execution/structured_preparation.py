"""Embed only metadata-declared support into one current complete vertex."""
from itertools import product
from math import prod


def spatial_support_condition(support, shape, index='q'):
    """Generate an index-only predicate for the existing GPU preparation ABI."""
    conditions = []
    for left, right, li, ri in support:
        lhs = f'(({index} / {prod(shape[left + 1:])}LL) % {shape[left]}LL)'
        rhs = f'(({index} / {prod(shape[right + 1:])}LL) % {shape[right]}LL)'
        if li == ri == tuple(range(shape[left])) and shape[left] == shape[right]:
            conditions.append(f'({lhs} == {rhs})')
        else:
            conditions.append('(' + (' || '.join(
                f'({lhs} == {i}LL && {rhs} == {j}LL)' for i, j in zip(li, ri)) or 'false') + ')')
    return ' && '.join(conditions)


def bind_spatial_embedding(support, shape, *, spatial_only=False):
    """Freeze axis geometry and selected-index matches, never input arrays."""
    tail = shape if spatial_only else shape[2:]
    if (len(tail) == 2 and len(support) == 1 and tail[0] == tail[1]
            and {support[0][0], support[0][1]} == {0, 1}
            and support[0][2] == support[0][3] == tuple(range(tail[0]))):
        # Resolve the usual diagonal before constructing generic Cartesian
        # index geometry. Source adjoints may reverse the two support axes.
        def diagonal_operation(xp, spin, value, destination, dtype):
            output = xp.empty(shape, dtype=dtype) if destination is None else destination
            output.fill(0)
            compact = value.diagonal()
            target = output.diagonal(axis1=-2, axis2=-1)
            if getattr(xp, '__name__', '') == 'numpy':
                target.setflags(write=True)
            if spatial_only:
                target[...] = compact.astype(xp.result_type(spin, value), copy=False)
            else:
                xp.multiply(spin[..., None], compact, out=target,
                            dtype=xp.result_type(spin, value), casting='unsafe')
            return output
        return diagonal_operation
    left = tuple(item[0] for item in support)
    right = tuple(item[1] for item in support)
    free = tuple(i for i in range(len(tail)) if i not in left + right)
    order = left + right + free
    left_shape, right_shape = tuple(tail[i] for i in left), tuple(tail[i] for i in right)
    def flatten(indices, sizes):
        value = 0
        for index, size in zip(indices, sizes):
            value = value * size + index
        return value
    matches = tuple(product(*(tuple(zip(item[2], item[3])) for item in support)))
    li = tuple(flatten(tuple(pair[0] for pair in row), left_shape) for row in matches)
    ri = tuple(flatten(tuple(pair[1] for pair in row), right_shape) for row in matches)
    spatial_shape = (prod(left_shape), prod(right_shape), *(tail[i] for i in free))

    def operation(xp, spin, value, destination, dtype):
        output = xp.empty(shape, dtype=dtype) if destination is None else destination
        output.fill(0)
        if not li:
            return output
        # Only current support entries participate in numerical work. Index
        # arrays are small metadata-derived temporaries, not production caches.
        left_index, right_index = xp.asarray(li), xp.asarray(ri)
        compact = value.transpose(order).reshape(spatial_shape)[left_index, right_index]
        if spatial_only:
            target = output.transpose(order).reshape(spatial_shape)
            target[left_index, right_index] = compact.astype(xp.result_type(spin, value), copy=False)
            return output
        target = output.transpose((0, 1, *(i + 2 for i in order))).reshape((4, 4, *spatial_shape))
        target[:, :, left_index, right_index] = (
            spin.reshape((4, 4) + (1,) * compact.ndim) * compact)
        return output
    return operation
