"""Array-safe, data-free identity of a symbolic Wick graph."""
import numpy as np


def _freeze(value):
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def graph_identity(diagram):
    """Normalize graph and coordinate sequences for equality and local hashing.

    This is an in-process dictionary key, never a persistent scientific ID.
    No numeric value caches are included.
    """
    return (_freeze(diagram.diagram.adjacency_matrix), _freeze(diagram.time_list),
            _freeze(diagram.vertex_list), _freeze(diagram.propagator_list))
