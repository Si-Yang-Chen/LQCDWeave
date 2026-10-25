"""Independent matrix kernels within the common physical batch annotation."""
from dataclasses import dataclass
from math import isfinite, prod

from .dense_batch import DenseKernelBatch

GEMM_GROUP_POINTER_BYTES = 512
# Three uint64 device addresses per child: left, right and result.
GEMM_GROUP_MAX_BATCHES = GEMM_GROUP_POINTER_BYTES // (3 * 8)


@dataclass(frozen=True)
class DenseGemmGroup:
    batches: tuple[DenseKernelBatch, ...]
    matrix_shape: tuple[int, int, int]
    seconds: float
    cost_status: str = 'estimated-from-dense-v1'

    def __post_init__(self):
        object.__setattr__(self, 'batches', tuple(self.batches))
        object.__setattr__(self, 'matrix_shape', tuple(self.matrix_shape))
        if not 2 <= len(self.batches) <= GEMM_GROUP_MAX_BATCHES or any(type(b) is not DenseKernelBatch for b in self.batches):
            raise ValueError(f'independent GEMM groups require 2–{GEMM_GROUP_MAX_BATCHES} Cartesian batches')
        if (len(self.matrix_shape) != 3
                or any(type(n) is not int or not 0 < n < (1 << 31) for n in self.matrix_shape)):
            raise ValueError('independent GEMM dimensions require positive BLAS int values')
        dtypes = {batch.packing_dtype for batch in self.batches}
        if len(dtypes) != 1 or None in dtypes:
            raise ValueError('independent GEMMs require one explicit packing dtype')
        indices = [i for batch in self.batches for i in batch.action_indices]
        results = [r.tensor for batch in self.batches for r in batch.results]
        if len(set(indices)) != len(indices) or len(set(results)) != len(results):
            raise ValueError('independent GEMM batches cannot share a paid Action or result')
        if (not isinstance(self.seconds, (int, float)) or not isfinite(self.seconds)
                or self.seconds < 0 or self.cost_status != 'estimated-from-dense-v1'):
            raise ValueError('independent GEMM costs require the recorded finite estimate')

    @property
    def action_indices(self):
        return tuple(sorted(i for batch in self.batches for i in batch.action_indices))

    @property
    def results(self):
        return tuple(result for _, result in sorted(
            (i, result) for batch in self.batches for i, result in zip(batch.action_indices, batch.results)))

    @property
    def left(self):
        return tuple(o for batch in self.batches for o in batch.left)

    @property
    def right(self):
        return tuple(o for batch in self.batches for o in batch.right)

    @property
    def packing_dtype(self):
        return self.batches[0].packing_dtype

    @property
    def pointer_bytes(self):
        # Fixed 64 uint64 words, including padding, owned by the output cube.
        return GEMM_GROUP_POINTER_BYTES

    @property
    def output_bytes(self):
        return sum(batch.output_bytes for batch in self.batches) + self.pointer_bytes

    @property
    def packing_bytes(self):
        return sum(batch.packing_bytes for batch in self.batches)


def dense_batch_leaves(batches):
    for batch in batches:
        yield from batch.batches if isinstance(batch, DenseGemmGroup) else (batch,)


def dense_batch_matrix_shape(batch, catalog):
    shapes = []
    for side, axis, stacked, prepend in (
            (batch.left, batch.left_axis, batch.stacked_sides[0], True),
            (batch.right, batch.right_axis, batch.stacked_sides[1], False)):
        ordered = [tuple(catalog[o.tensor].shape[i] for i in o.permutation) for o in side]
        if stacked:
            ordered = [(1,) + s if prepend else s + (1,) for s in ordered]
        merged = list(ordered[0])
        merged[axis] = sum(s[axis] for s in ordered)
        shapes.append(tuple(merged))
    left, right = shapes
    return prod(left[:batch.left_free_rank]), prod(left[batch.left_free_rank:]), prod(right[batch.reduction_rank:])
