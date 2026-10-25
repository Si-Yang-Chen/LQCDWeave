"""Explicit Gamma5 reverse operations owned by the perambulator module."""

from dataclasses import replace
import numpy as np

from .products import normalize_product, product_axes, product_shape, reverse_product


def gamma5_reverse(values, product, gamma5, *, array_module=np, leading_axes=0):
    """Return owned ``gamma5 T^dagger gamma5`` with endpoint groups swapped.

    Leading axes are preserved, never interpreted as physical time. Use
    reverse_propagator for an actual time/endpoint record exchange. The supplied
    gamma matrix and action convention are explicit; no solver is invoked.
    """
    product = normalize_product(product)
    if type(leading_axes) is not int or leading_axes < 0:
        raise ValueError('leading_axes must be a nonnegative integer')
    arr = array_module.asarray(values)
    if arr.dtype.kind != 'c':
        raise ValueError('propagator values must have a complex dtype')
    gamma = array_module.asarray(gamma5, dtype=arr.dtype)
    rank = len(product_axes(product))
    if (arr.ndim != leading_axes + rank or
            arr.shape[leading_axes:leading_axes+2] != (4, 4)):
        raise ValueError('values must have leading,spin,spin,sink,source axes')
    if gamma.shape != (4, 4):
        raise ValueError('gamma5 must be a 4 by 4 matrix')
    if ((product[0] == 'P' and arr.shape[leading_axes+3] != 3) or
            (product[2] == 'P' and arr.shape[-1] != 3)):
        raise ValueError('point color axes must have extent three')
    split = leading_axes + 2 + (2 if product[0] == 'P' else 1)
    permutation = (tuple(range(leading_axes)) + (leading_axes+1, leading_axes)
                   + tuple(range(split, arr.ndim))
                   + tuple(range(leading_axes+2, split)))
    dagger = arr.conj().transpose(permutation)
    shape = dagger.shape
    flat = dagger.reshape(shape[:leading_axes+2] + (-1,))
    result = array_module.einsum('ai,...ijr,jb->...abr', gamma, flat, gamma)
    return result.reshape(shape)


def reverse_propagator(block, gamma5, *, gamma5_hermitian, array_module=np):
    """Exchange a physical pair and actual endpoints, with declared capability."""
    if gamma5_hermitian is not True:
        raise ValueError('the action must explicitly declare gamma5 hermiticity')
    product = normalize_product(block.product)
    if product[0] != block.sink.kind or product[2] != block.source.kind:
        raise ValueError('product differs from its actual endpoint representations')
    expected = product_shape(product, len(block.sink.indices), len(block.source.indices))
    if tuple(block.values.shape) != expected:
        raise ValueError('values differ from the actual endpoint axis sizes')
    return replace(block, product=reverse_product(product),
                   sink_time=block.source_time, source_time=block.sink_time,
                   sink=block.source, source=block.sink,
                   values=gamma5_reverse(block.values, product, gamma5,
                                         array_module=array_module), derived=True)


def psv_to_vsp_dagger(psv_block, gamma5, *, backend=np):
    """Legacy row operation; the caller supplies the reversed time orientation."""
    return gamma5_reverse(psv_block, 'PSV', gamma5,
                          array_module=backend, leading_axes=1)
