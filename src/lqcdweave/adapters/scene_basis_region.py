"""Bind one maximum-Ne/Np read to all views of a scene backing region."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from ..contracts.correlator.api import SourceSpec, TensorSpec
from ..contraction.execution.api import sparse_pairs
from ..contraction.planning.api import SceneBasisBatch


@dataclass(frozen=True)
class RegionReadRequest:
    region: str
    used_ne: int
    used_np: int
    spec: TensorSpec | None
    problem_index: int
    original_key: str
    source: SourceSpec


class MaxBasisRegionLoader:
    """Implement replay's ``load_region/view`` protocol for one basis batch.

    ``read_max`` performs the only external read for a region and receives
    a ``RegionReadRequest`` with componentwise maximum Ne/Np. ``view`` must
    slice or transform that returned data without another external read.
    Dense requests use prefix slices by default. Sparse P2P records are
    reduced from the maximum-Np record without a dense point-by-point array.
    """

    def __init__(
        self,
        batch: SceneBasisBatch,
        read_max: Callable[[RegionReadRequest], Any],
        *,
        view: Callable[[Any, int, str, int, int, SourceSpec], Any] | None = None,
    ):
        if not isinstance(batch, SceneBasisBatch) or not callable(read_max):
            raise TypeError("a SceneBasisBatch and read_max callback are required")
        if view is not None and not callable(view):
            raise TypeError("view must be callable")
        self.batch = batch
        self._read_max = read_max
        self._custom_view = view
        if view is None:
            by_region = {}
            for source in batch.problem.sources.values():
                semantics = _view_semantics(source)
                previous = by_region.setdefault(source.region, semantics)
                if previous != semantics:
                    raise ValueError("different component selections require an explicit in-memory view callback")

    def load_region(self, region: str, _first_source: SourceSpec):
        key = self.batch.representative_by_region[region]
        problem_index, original_key = self.batch.source_origins[key]
        used_ne, used_np = self.batch.maxima_by_region[region]
        spec = self.batch.spec_by_region[region]
        source = self.batch.problem.sources[key]
        if isinstance(source.metadata, Mapping):
            metadata = dict(source.metadata)
            if "used_ne" in metadata:
                metadata["used_ne"] = used_ne
            if "used_np" in metadata:
                metadata["used_np"] = used_np
            source = replace(source, metadata=metadata)
        value = self._read_max(
            RegionReadRequest(
                region,
                used_ne,
                used_np,
                spec,
                problem_index,
                original_key,
                source,
            )
        )
        if spec is not None:
            if tuple(value.shape) != spec.shape or np.dtype(value.dtype) != spec.dtype:
                raise ValueError("maximum-Ne/Np read disagrees with the declared tensor")
            if self.batch.problem.backend == "cupy" and not hasattr(value, "__cuda_array_interface__"):
                raise TypeError("CuPy maximum region must be materialized on the device before slicing")
        return value

    def view(self, region_data: Any, key: str, source: SourceSpec):
        problem_index, original_key = self.batch.source_origins[key]
        used_ne, used_np = self.batch.used_bases[problem_index]
        if self._custom_view is not None:
            value = self._custom_view(region_data, problem_index, original_key, used_ne, used_np, source)
        elif key in self.batch.problem.specs:
            spec = self.batch.problem.specs[key]
            if len(region_data.shape) != len(spec.shape):
                raise ValueError("region rank disagrees with its dense view")
            value = region_data[tuple(slice(0, extent) for extent in spec.shape)]
        elif isinstance(source.metadata, Mapping) and source.metadata.get("family") == "p2p":
            _, maximum_np = self.batch.maxima_by_region[source.region]
            value = _p2p_prefix_view(region_data, used_np, maximum_np)
        else:
            if (used_ne, used_np) != self.batch.maxima_by_region[source.region]:
                raise ValueError("a custom view is required for this non-dense source")
            value = region_data
        if key in self.batch.problem.specs:
            spec = self.batch.problem.specs[key]
            if tuple(value.shape) != spec.shape or np.dtype(value.dtype) != spec.dtype:
                raise ValueError(f"region view disagrees with TensorSpec for {key}")
        return value


def _p2p_prefix_view(parts: Any, used_np: int, maximum_np: int):
    if used_np == maximum_np:
        return parts
    selected = []
    for gamma, coefficient, record in parts:
        if not isinstance(record, Mapping):
            raise TypeError("P2P part must contain a sparse record mapping")
        if record.get("type") == "identity":
            selected.append((gamma, coefficient, record))
            continue
        # Legacy negative indices are relative to the requested Np. Matched
        # records instead declare their full stored extent in their shape.
        extent = maximum_np if record.get("layout") == "matched" else used_np
        indices, values = sparse_pairs(record, extent)
        keep = (indices[:, 0] < used_np) & (indices[:, 1] < used_np)
        selected.append(
            (gamma, coefficient, {"type": "sparse", "indices": indices[keep].copy(), "values": values[keep]})
        )
    return tuple(selected)


def _view_semantics(source: SourceSpec):
    if isinstance(source.metadata, Mapping):
        return {key: value for key, value in source.metadata.items() if key not in ("used_ne", "used_np")}
    return source.metadata


__all__ = ["MaxBasisRegionLoader", "RegionReadRequest"]
