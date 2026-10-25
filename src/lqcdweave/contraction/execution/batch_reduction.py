"""Wide accumulation for Cartesian batches of originally thin reductions."""
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from .diagnostics import debug_enabled
from .reduction_kernel import bind_raw_reduction


def bind_precise_batch(batch, catalog, matrix_shapes, backend, *, right_rows=False):
    """Use only an original member's workspace, in addition to saved packing.

    Full batch output ownership and input packing already belong to P. Extra
    partials/upcasts fit in the smallest member's input+output allowance, which
    is no larger than the workspace at the batch's first original Action.
    """
    dtype = np.dtype(catalog[batch.left[0].tensor].dtype)
    if not any(batch.stacked_sides) or dtype not in (np.dtype('float32'), np.dtype('complex64')):
        return None
    # The alternate physical layout belongs only to native scalar batches.
    # Failure to bind must leave the caller free to retain its whole original
    # packing layout, including the CPU/debug oracle and legacy GEMM groups.
    if right_rows and (not all(batch.stacked_sides) or debug_enabled()
                       or getattr(backend, '__name__', '') != 'cupy'):
        return None
    sizes = lambda side: min(prod(catalog[o.tensor].shape) * dtype.itemsize for o in side)
    # Result views need not exist in an input-only execution catalog yet.
    # Saved bounds include singleton stack axes, so their volume also gives
    # the logical result size after squeezing.
    allowance = sizes(batch.left) + sizes(batch.right) + min(
        prod(stop - start for start, stop in r.bounds) * dtype.itemsize for r in batch.results)
    if not debug_enabled():
        raw = bind_raw_reduction(parse_axis_pattern('ik,jk->ij' if right_rows else 'ik,kj->ij'), matrix_shapes,
                                 dtype, backend, allowance)
        if raw is not None:
            return raw
    if right_rows:
        return None
    # Independent bounded oracle, also used on CPU. Whole-input upcasts would
    # exceed the original member workspace for a large Cartesian batch.
    (m, k), (_, n) = matrix_shapes
    wide = np.dtype('complex128' if dtype.kind == 'c' else 'float64')
    if allowance < 4 * wide.itemsize:
        return None
    fixed = m * n * (3 * wide.itemsize + dtype.itemsize)
    width = min(k, max(0, (allowance - fixed) // ((m + n) * wide.itemsize)))

    def run(left, right):
        if not width:
            # Tiny K / large output grids can have no room for a wide matrix.
            # Keep only one wide scalar and its two wide input elements live.
            result = backend.empty((m, n), dtype=dtype)
            for i in range(m):
                for j in range(n):
                    total = backend.zeros((), dtype=wide)
                    for h in range(k):
                        total += left[i, h].astype(wide) * right[h, j].astype(wide)
                    result[i, j] = total
            return result
        total = None
        for start in range(0, k, width):
            a = left[:, start:start+width].astype(wide)
            b = right[start:start+width].astype(wide)
            partial = backend.matmul(a, b)
            if total is None:
                total = partial
            else:
                total += partial
            del a, b, partial
        return total.astype(dtype)

    run.workspace_bytes = (fixed + width * (m+n) * wide.itemsize if width else
                           4 * wide.itemsize)
    run.workspace_allowance = allowance
    return run
