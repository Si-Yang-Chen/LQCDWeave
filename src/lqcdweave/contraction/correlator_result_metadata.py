"""Frontend-only operator labels for numerical result fields.

Uses public symbolic declarations, without extending the numerical output IR.
Unknown labels remain positional rather than being inferred from an ndarray.
"""
from collections.abc import Mapping
import numpy as np
from lqcdweave.operators.correlator.api import StructuredCorrelator, StructuredOperation


def operator_coordinates(correlator):
    """Return field-path to leading operator-axis coordinates when declared."""
    result = {}

    def labels(item):
        if isinstance(item, StructuredCorrelator):
            coordinates = {
                f'operator_axis_{axis_index}': [
                    dict(index=index, operator=str(position), flavor=str(flavor))
                    for index, (position, flavor) in enumerate(axis)]
                for axis_index, axis in enumerate(item.axes)
            }
            # Remaining projected/mode axes retain their actual positional
            # coordinates unless explicitly named by the caller at save time.
            return coordinates
        if isinstance(item, StructuredOperation):
            candidates = [labels(child) for child in item.children
                          if isinstance(child, (StructuredCorrelator, StructuredOperation))]
            if candidates and all(value == candidates[0] for value in candidates):
                return candidates[0]
        return {}

    def visit(item, path):
        if isinstance(item, (StructuredCorrelator, StructuredOperation)):
            if coordinates := labels(item):
                result[path] = coordinates
        elif isinstance(item, Mapping):
            for key, value in item.items():
                visit(value, path + (key,))
        elif isinstance(item, (tuple, list)):
            for index, value in enumerate(item):
                visit(value, path + (index,))
        elif isinstance(item, np.ndarray) and item.dtype == object:
            # An object-array symbolic correlator assembles into one numeric
            # array, not independent object leaves; its axes are positional.
            return
    visit(correlator, ())
    return result
