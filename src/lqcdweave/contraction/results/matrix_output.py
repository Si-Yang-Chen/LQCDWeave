"""Matrix block placement and shape operations on the existing OutputNode IR.

No Wick logic, tensor contraction or IO occurs here. Scan axes remain trailing.
"""
from math import prod
import numpy as np

MATRIX_OUTPUT_KINDS = frozenset(('reshape', 'scatter', 'zeros', 'transpose'))


def scatter_sum_blocks(node):
    """Recognize an ordered sum of placements with the same target layout."""
    if node.kind != 'add' or not node.children:
        return None
    blocks = node.children
    shape = blocks[0].shape
    return blocks if all(child.kind == 'scatter' and child.shape == shape for child in blocks) else None


def evaluate_scatter_sum(blocks, evaluate, scan_shape, *, trusted=False, zero=0, indexes=None):
    """Accumulate compact blocks in order, owning only one full matrix.

    Each scatter has unique indices within its block; different blocks may
    overlap. Promotion follows the original left-to-right array additions.
    No borrowed block or result from a different output occurrence is mutated.
    """
    target = (*blocks[0].shape, *scan_shape)
    result = None
    for index, node in enumerate(blocks):
        child = np.asarray(evaluate(node.children[0]))
        positions = node.value
        if not trusted:
            expected = (*(len(axis) for axis in positions), *node.shape[len(positions):], *scan_shape)
            if child.shape != expected:
                raise ValueError('scatter block shape differs from its target positions')
        if result is None:
            # Use a rank-one probe: scalar/array promotion differs from 0-D
            # arithmetic on older NumPy, and bool + integer zero yields int.
            dtype = (np.empty(0, dtype=child.dtype) + zero).dtype
            result = np.zeros(target, dtype=dtype)
        else:
            dtype = np.result_type(result.dtype, child.dtype)
            if dtype != result.dtype:
                result = result.astype(dtype)
        selected = np.ix_(*positions) if indexes is None else indexes[index]
        result[selected] = result[selected] + child
    return result


def validate_matrix_output(node, term_ids, validate):
    if (not isinstance(node.shape,tuple)
            or any(type(size) is not int or size < 0 for size in node.shape)):
        raise ValueError('matrix output shape must contain nonnegative integers')
    if node.kind == 'zeros':
        if node.children or not isinstance(node.value,str) or np.dtype(node.value).kind not in 'biufc':
            raise ValueError('zeros needs a numeric dtype and no children')
        return
    if len(node.children) != 1:
        raise ValueError('matrix output operation needs one child')
    if node.kind == 'scatter':
        positions = node.value
        if (not isinstance(positions,tuple) or not positions or len(positions)>len(node.shape)
                or any(not isinstance(axis,tuple) or len(set(axis))!=len(axis)
                       or any(type(i) is not int or i<0 or i>=node.shape[a] for i in axis)
                       for a,axis in enumerate(positions))):
            raise ValueError('scatter positions must be unique in-range axis indices')
    if node.kind == 'transpose' and (not isinstance(node.value,tuple)
            or any(type(axis) is not int for axis in node.value)
            or sorted(node.value)!=list(range(len(node.shape)))):
        raise ValueError('transpose requires a permutation of its result axes')
    validate(node.children[0],term_ids)


def evaluate_matrix_output(node, evaluate, scan_shape, *, trusted=False, dtype=None):
    target = (*node.shape,*scan_shape)
    if node.kind == 'zeros':
        return np.zeros(target,dtype=node.value if dtype is None else dtype)
    child = np.asarray(evaluate(node.children[0]))
    if node.kind == 'transpose':
        if not trusted and child.ndim!=len(node.shape)+len(scan_shape):
            raise ValueError('transpose child rank differs from the output layout')
        result = child.transpose((*node.value,*range(len(node.shape),child.ndim)))
        if not trusted and result.shape!=target:
            raise ValueError('transpose child shape differs from the output layout')
        return result
    if node.kind == 'reshape':
        if not trusted and child.size != prod(target):
            raise ValueError('reshape changes the number of result elements')
        return child.reshape(target)
    positions = node.value
    expected = (*(len(axis) for axis in positions),*node.shape[len(positions):],*scan_shape)
    if not trusted and child.shape != expected:
        raise ValueError('scatter block shape differs from its target positions')
    result = np.zeros(target,dtype=child.dtype)
    result[np.ix_(*positions)] = child
    return result


def estimate_matrix_output(node, estimate, scan_shape):
    shape = (*node.shape,*scan_shape)
    if node.kind == 'zeros':
        dtype = np.dtype(node.value)
        return prod(shape)*dtype.itemsize,shape,dtype
    child_bytes,child_shape,dtype = estimate(node.children[0])
    size = prod(shape)*dtype.itemsize
    if node.kind == 'transpose':
        expected = tuple(child_shape[i] for i in node.value)+scan_shape
        if expected!=shape:
            raise ValueError('transpose child shape differs from the output layout')
        return child_bytes,shape,dtype
    if node.kind == 'reshape':
        if prod(child_shape)!=prod(shape):
            raise ValueError('reshape changes the number of result elements')
        # A scanned result may be a noncontiguous transpose; conservatively
        # reserve a full reshape copy even when NumPy can return a view.
        return child_bytes+size,shape,dtype
    positions = node.value
    expected = (*(len(axis) for axis in positions),*node.shape[len(positions):],*scan_shape)
    if child_shape!=expected:
        raise ValueError('scatter block shape differs from its target positions')
    indices = sum(map(len,positions))*np.dtype(np.intp).itemsize
    return child_bytes+size+indices,shape,dtype
