"""Compile dense axis relations without a prediction or execution state."""
from itertools import combinations
from lqcdweave.contracts.correlator.api import AxisPattern

from ..canonical_operations import axis_key, output_axes
from ..state import _outer_product_axes


def compile_dense_choices(pattern, checkpoint, *, mask=None):
    count = len(pattern.inputs)
    pairs = ((0,),) if count == 1 else combinations(range(count), 2)
    choices = []
    for ordinal, pair in enumerate(pairs):
        checkpoint()
        if mask is not None and not mask & (1 << ordinal):
            continue
        labels = tuple(pattern.inputs[i] for i in pair)
        needed = set(pattern.output)
        for i, axes in enumerate(pattern.inputs):
            if i not in pair:
                needed.update(axes)
        if mask is None and count > 1 and _outer_product_axes(*labels, needed):
            continue
        variants = []
        for names in ((0,),) if count == 1 else ((0, 1), (1, 0)):
            output = pattern.output if count == 1 else output_axes(names, labels, needed)
            keys = tuple(axis_key(ids, labels, output)[1:]
                         for ids in (((0,),) if count == 1 else ((0, 1), (1, 0), (0, 0))))
            variants.append((output, keys, AxisPattern(labels, output)))
        choices.append((pair, tuple(variants)))
    return tuple(choices)
