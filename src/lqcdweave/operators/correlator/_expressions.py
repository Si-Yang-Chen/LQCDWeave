"""Ordered factors of expanded operator words, without rebuilding operators."""

from sympy import Mul, Pow


def word_factors(expression):
    """Yield repeated factors explicitly, preserving noncommutative order.

    Rebuilding an unevaluated Mul can immediately combine its repeated factors
    back into Pow. Walk both forms instead so repeated insertions stay visible.
    """
    if isinstance(expression, Mul):
        for factor in expression.args:
            yield from word_factors(factor)
    elif isinstance(expression, Pow) and expression.exp.is_Integer and expression.exp > 0:
        for _ in range(int(expression.exp)):
            yield from word_factors(expression.base)
    else:
        yield expression
