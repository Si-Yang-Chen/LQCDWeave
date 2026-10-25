"""Build pure logical TensorDescription values from bound reader metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import numpy as np

from lqcdweave.contracts.correlator.api import TensorDescription


def _recipe_value_key(value, metadata_cache):
    # A module-level recursion has no self-referential closure retaining the
    # construction cache until a later cyclic-GC pass.
    container = isinstance(value, (Mapping, tuple))
    if container and metadata_cache is not None:
        previous = metadata_cache.get(id(value))
        if previous is not None and previous[0] is value:
            return previous[1]
    if isinstance(value, Mapping):
        result = tuple((name, _recipe_value_key(item, metadata_cache)) for name, item in sorted(value.items()))
    elif isinstance(value, tuple):
        result = tuple(_recipe_value_key(item, metadata_cache) for item in value)
    else:
        result = value
    if container and metadata_cache is not None:
        metadata_cache[id(value)] = (value, result)
    return result


def tensor_recipe_key(description, shape, dtype, workspace_bytes, *, metadata_cache=None):
    """Exact call-local key for a fully identified dense materialization.

    Full records, rather than a digest or shape alone, decide equality. Unknown
    dataset/selection identities keep their existing logical occurrence IDs.
    OutputRecipe owns scalar weights, so they are absent from this description.
    """
    if description.data_identity is None or description.selection_identity is None:
        return None

    return (tuple(shape), np.dtype(dtype).str, workspace_bytes,
            tuple(_recipe_value_key(getattr(description, field.name), metadata_cache)
                  for field in fields(description)))


def make_tensor_description(
    metadata,
    *,
    role,
    source_bindings,
    axes=None,
    transform=(),
    semantics=None,
    bindings=None,
    metadata_sources=(),
):
    """Combine reader identity with the final *logical* tensor axes.

    ``metadata.axes`` names the raw physical layout. A caller that adds spin
    axes or applies dagger permutations must pass the resulting logical names
    separately. Reader identities remain unknown when the reader says unknown.
    """
    logical_axes = tuple(metadata.axes if axes is None else axes)
    normalized_bindings = dict(bindings or {})
    normalized_semantics = {
        "raw_axes": tuple(metadata.axes),
        "sparse_layout": metadata.sparse_layout,
        "point_diagonal": metadata.point_diagonal,
        "structure": metadata.structure,
    }
    normalized_bindings.setdefault("raw_shape", tuple(int(size) for size in metadata.shape))
    normalized_bindings.setdefault("raw_dtype", np.dtype(metadata.dtype).str)
    normalized_bindings.setdefault("point_shape", metadata.point_shape)
    normalized_bindings.setdefault("nnz_upper", metadata.nnz_upper)
    normalized_bindings.setdefault("point_pool_ids", dict(metadata.point_pool_ids))
    normalized_bindings.setdefault("axis_bindings", dict(metadata.axis_bindings))
    normalized_bindings.setdefault(
        "point_pool_ids_by_batch",
        None if metadata.point_pool_ids_by_batch is None else tuple(
            dict(item) for item in metadata.point_pool_ids_by_batch
        ),
    )
    if semantics:
        normalized_semantics.update(semantics)
    if isinstance(source_bindings, str):
        source_bindings = (source_bindings,)
    return TensorDescription(
        axes=logical_axes,
        data_identity=metadata.data_identity,
        selection_identity=metadata.selection_identity,
        role=role,
        source_bindings=tuple(source_bindings),
        transform=tuple(transform),
        semantics=normalized_semantics,
        bindings=normalized_bindings,
        _metadata_sources=metadata_sources,
    )


def stable_binding_value(value):
    """Keep only already explicit scalar/tuple binding identities."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return None
        return {key: stable_binding_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(stable_binding_value(item) for item in value)
    return None


def point_pool_bindings(metadata):
    """Extract the registered pool description without coordinate arrays."""
    pools = {}
    for pool_id, item in metadata.point_pool_metadata.items():
        observed = getattr(item, "observed_counts", None)
        if observed is not None:
            observed = tuple(sorted((repr(key), int(count)) for key, count in observed.items()))
        pools[pool_id] = {
            "spatial_sites": int(item.spatial_sites),
            "sampled_points": int(item.sampled_points),
            "sampling_design": item.sampling_design,
            "compensation": item.compensation,
            "observed_counts": observed,
        }
    return pools


__all__ = [
    "make_tensor_description", "point_pool_bindings", "stable_binding_value",
]
