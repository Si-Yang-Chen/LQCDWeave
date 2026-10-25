"""Shared K/H kernels for declared meson matrices, without IO or Wick signs.

The factor order follows planning.meson_matrix's fixed shared products. Binary
operations use the existing X matmul lowering. Block traversal follows the
numerical meson_matrix block kernel, replacing its three-factor middle/prefix
profiles with shared K/H products and explicit disjoint PC accumulation.
Caller supplies B-transformed vertices and coefficients; no conjugation or
gamma sandwich is added here. These kernels are not a resource-audited plan.
"""
import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern, MatrixKernelSpec
from .lowering import lower_binary_matmul, execute_matmul

_PRODUCT = AxisPattern(((0,1,2,3), (4,1,5,3,6)), (4,0,5,2,6))
_DOT = AxisPattern(((0,1,2,3,4), (5,2,1,4,3)), (0,5))
_LOOP = AxisPattern(((0,1,2,3), (4,1,0,3,2)), (4,))


def _binary(equation, left, right, *, backend=np):
    recipe = lower_binary_matmul(equation, (left.shape, right.shape), (left.dtype, right.dtype))
    if recipe is None:
        raise ValueError('matrix kernel equation must lower to a binary product')
    return execute_matmul(recipe, left, right, backend)


def _modes(partitions, ne):
    boundaries = (0, ne) if partitions is None else tuple(partitions)
    slices = tuple(slice(a,b) for a,b in zip(boundaries,boundaries[1:]))
    pairs = {(a,b):i for i,(a,b) in enumerate(
        (a,b) for a in range(len(slices)) for b in range(a,len(slices)))}
    def target(a,b):
        return pairs[min(a,b),max(a,b)], 1.0 if a==b else 1/np.sqrt(2)
    return slices,pairs,target


def _linear(weights, products):
    result = weights @ products.reshape(len(products), -1)
    return result.reshape(len(weights), *products.shape[1:])


def contract_meson_matrix_pc(p0, p1, left, right, left_weights, right_weights,
                             *, partitions=None, row_block=1, backend=np, mode_block=False):
    """Return positive connected [left_op,right_op,left_PC,right_PC].

    Inputs: P0[BAON], P1[DCQP], left[cADNQ], right[cCBPO]. Coefficient arrays
    already include source-side conjugation. With partitions=None the two
    singleton PC axes are omitted. row_block is an explicit planned row width.
    The explicit array backend owns the output; caller owns all input arrays.
    """
    inputs = (p0,p1,left,right,left_weights,right_weights)
    spec = MatrixKernelSpec('connected', tuple(value.shape for value in inputs),
        tuple(value.dtype.name for value in inputs), partitions, row_block,mode_block=mode_block)
    ne = spec.ne
    slices,pairs,target = _modes(partitions,ne)
    dtype = spec.dtype
    if any(value.dtype != dtype for value in inputs):
        raise ValueError('mixed-precision contraction is unsupported; normalize inputs before planning')
    if mode_block:
        if not left.shape[0] or not right.shape[0] or not len(left_weights) or not len(right_weights):
            return backend.zeros(spec.output_shape,dtype=dtype)
        other = _linear(right_weights,_binary(_PRODUCT,p1,right,backend=backend))
        primitive = _binary(_PRODUCT,p0,left,backend=backend)
        result = backend.zeros(spec.output_shape,dtype=dtype)
        for start in range(0,len(left_weights),row_block):
            stop = min(start+row_block,len(left_weights))
            weighted = _linear(left_weights[start:stop],primitive)
            result[start:stop] = _binary(_DOT,weighted,other,backend=backend)
        return result
    result = backend.zeros((len(left_weights),len(right_weights),len(pairs),len(pairs)),dtype=dtype)
    if not left.shape[0] or not right.shape[0] or not len(left_weights) or not len(right_weights):
        return result if partitions is not None else result[:,:,0,0]
    # Each primitive right product is formed once per ordered vertex block;
    # the coefficient map is applied before dot products, without pair expansion.
    right_products = {}
    for c,cc in enumerate(slices):
        for d,dd in enumerate(slices):
            primitive = _binary(_PRODUCT, p1[:,:,:,cc], right[:,:,:,cc,dd], backend=backend)
            right_products[c,d] = _linear(right_weights,primitive)
            del primitive
    for a,aa in enumerate(slices):
        for b,bb in enumerate(slices):
            primitive = _binary(_PRODUCT,p0[:,:,:,aa],left[:,:,:,aa,bb], backend=backend)
            left_pair,left_weight = target(a,b)
            for start in range(0,len(left_weights),row_block):
                stop = min(start+row_block,len(left_weights))
                weighted = _linear(left_weights[start:stop],primitive)
                for (c,d),other in right_products.items():
                    right_pair,right_weight = target(c,d)
                    value = _binary(_DOT,weighted[:,:,:,slices[d],:],other[:,:,:,bb,:], backend=backend)
                    result[start:stop,:,left_pair,right_pair] += value * (left_weight*right_weight)
                    del value
                del weighted
            del primitive
    return result if partitions is not None else result[:,:,0,0]


def contract_meson_loop_pc(propagator, vertices, weights, *, partitions=None, backend=np, mode_block=False):
    """Return a loop vector [operator,PC] with B's source transform preserved."""
    inputs = (propagator,vertices,weights)
    spec = MatrixKernelSpec('loop', tuple(value.shape for value in inputs),
        tuple(value.dtype.name for value in inputs), partitions,mode_block=mode_block)
    ne = spec.ne
    slices,pairs,target = _modes(partitions,ne)
    dtype = spec.dtype
    if any(value.dtype != dtype for value in inputs):
        raise ValueError('mixed-precision contraction is unsupported; normalize inputs before planning')
    if mode_block:
        if not vertices.shape[0] or not len(weights):
            return backend.zeros(spec.output_shape,dtype=dtype)
        return weights @ _binary(_LOOP,propagator,vertices,backend=backend)
    result = backend.zeros((len(weights),len(pairs)), dtype=dtype)
    if not vertices.shape[0] or not len(weights):
        return result if partitions is not None else result[:,0]
    for a,aa in enumerate(slices):
        for b,bb in enumerate(slices):
            primitive = _binary(_LOOP,propagator[:,:,bb,aa],vertices[:,:,:,aa,bb], backend=backend)
            pair,weight = target(a,b)
            result[:,pair] += (weights @ primitive) * weight
    return result if partitions is not None else result[:,0]


__all__ = ['contract_meson_matrix_pc','contract_meson_loop_pc']
