"""Spatial sampling definitions and explicit, ordered realizations."""

from dataclasses import dataclass
from types import MappingProxyType
from numbers import Integral
from math import prod

import numpy as np


def sampling_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _geometry(value):
    sizes = tuple(sampling_integer(x, "lattice extent") for x in value)
    if len(sizes) != 4:
        raise ValueError("lattice_size must be (Lx, Ly, Lz, Lt)")
    if max(sizes) > np.iinfo(np.int64).max or prod(sizes[:3]) > np.iinfo(np.int64).max:
        raise ValueError("lattice dimensions exceed int64 indexing")
    return sizes


@dataclass(frozen=True)
class SpatialSamplingDesign:
    """Uniform sampling without replacement within each spatial time slice."""

    lattice_size: tuple[int, int, int, int]
    num_points: int
    temporal_policy: int | str = 1

    def __post_init__(self):
        sizes = _geometry(self.lattice_size)
        count = sampling_integer(self.num_points, "num_points")
        if count > prod(sizes[:3]):
            raise ValueError("num_points exceeds spatial volume")
        policy = self.temporal_policy
        if isinstance(policy, str):
            if policy not in ("independent", "same"):
                raise ValueError("temporal_policy must be a segment length, 'independent' or 'same'")
        else:
            policy = sampling_integer(policy, "temporal_policy")
            if policy > sizes[3]:
                raise ValueError("temporal_policy segment length must not exceed Lt")
        object.__setattr__(self, "lattice_size", sizes)
        object.__setattr__(self, "num_points", count)
        object.__setattr__(self, "temporal_policy", policy)

    @property
    def segment_length(self):
        """Physical-time segment length, including the historical string aliases."""
        if self.temporal_policy == "independent":
            return 1
        if self.temporal_policy == "same":
            return self.lattice_size[3]
        return self.temporal_policy


@dataclass(frozen=True)
class SamplingPoolSpec:
    """Neutral per-pool metadata shared by reader and contraction adapters.

    The pool ID is supplied by the data binding. It is not derived from a time
    coordinate, offset, Np, seed, or array shape.
    """

    group_id: str
    spatial_sites: int
    sampled_points: int
    sampling_design: str = "simple_random_without_replacement"
    compensation: str = "expected"
    observed_counts: object = None

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("sampling pool needs a stable explicit group_id")
        sites = sampling_integer(self.spatial_sites, "spatial_sites")
        count = sampling_integer(self.sampled_points, "sampled_points")
        if count > sites:
            raise ValueError("sampled_points cannot exceed spatial_sites")
        if not isinstance(self.sampling_design, str) or not self.sampling_design:
            raise ValueError("sampling_design must be explicitly named")
        if self.compensation not in ("expected", "observed"):
            raise ValueError("compensation must be expected or observed")
        if self.observed_counts is not None:
            if not isinstance(self.observed_counts, dict):
                raise TypeError("observed_counts must be a mapping")
            object.__setattr__(self, "observed_counts",
                               MappingProxyType(dict(self.observed_counts)))
        object.__setattr__(self, "spatial_sites", sites)
        object.__setattr__(self, "sampled_points", count)


@dataclass(frozen=True, eq=False)
class SpatialPointSet:
    """Coordinates [time, point, xyz]; time and point order define the basis.

    Imported coordinates need not claim a sampling design. Arrays are copied
    into immutable buffers so caller mutation cannot change the realization.
    """

    lattice_size: tuple[int, int, int, int]
    times: tuple[int, ...]
    coordinates: np.ndarray
    design: SpatialSamplingDesign | None = None
    seed: int | None = None
    algorithm: str | None = None
    pool_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        sizes = _geometry(self.lattice_size)
        times = tuple(sampling_integer(t, "time", 0) for t in self.times)
        if not times or len(set(times)) != len(times) or any(t >= sizes[3] for t in times):
            raise ValueError("times must be unique coordinates in [0, Lt)")
        coords = np.asarray(self.coordinates)
        if coords.ndim != 3 or coords.shape[0] != len(times) or coords.shape[2] != 3 or coords.shape[1] == 0:
            raise ValueError("coordinates must have shape (len(times), num_points, 3)")
        if coords.dtype.kind not in "iu":
            raise ValueError("coordinates must be integers")
        if np.any(coords < 0) or np.any(coords >= np.asarray(sizes[:3])):
            raise ValueError("spatial coordinates are outside the lattice")
        if any(len(np.unique(row, axis=0)) != coords.shape[1] for row in coords):
            raise ValueError("points must be unique within each time slice")
        if self.design is not None:
            if self.design.lattice_size != sizes or self.design.num_points != coords.shape[1]:
                raise ValueError("point set does not match its design")
            length = self.design.segment_length
            if length > 1 and length == sizes[3]:
                if not np.all(coords == coords[0]):
                    raise ValueError("temporal policy requires identical ordered point sets within each segment")
            elif length > 1:
                representatives = {}
                for time, row in zip(times, coords):
                    segment = time // length
                    if segment in representatives:
                        if not np.array_equal(row, representatives[segment]):
                            raise ValueError("temporal policy requires identical ordered point sets within each segment")
                    else:
                        representatives[segment] = row
        pool_ids = self.pool_ids
        if pool_ids is not None:
            if isinstance(pool_ids, (str, bytes)):
                raise TypeError("pool_ids must be a sequence of IDs")
            pool_ids = tuple(pool_ids)
            if len(pool_ids) != len(times):
                raise ValueError("pool_ids must have one ID per time")
            if any(not isinstance(pool_id, str) for pool_id in pool_ids):
                raise TypeError("pool_ids must contain strings")
            if any(not pool_id for pool_id in pool_ids):
                raise ValueError("pool_ids must contain nonempty strings")
            if self.design is None:
                raise ValueError("pool_ids require an explicit sampling design")
            representatives = {}
            for pool_id, row in zip(pool_ids, coords):
                if pool_id in representatives:
                    if not np.array_equal(row, representatives[pool_id]):
                        raise ValueError("times sharing a pool ID must have identical ordered point sets")
                else:
                    representatives[pool_id] = row
        seed = None if self.seed is None else sampling_integer(self.seed, "seed", 0)
        owned = np.frombuffer(coords.astype(np.int64).tobytes(), dtype=np.int64).reshape(coords.shape)
        object.__setattr__(self, "lattice_size", sizes)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coordinates", owned)
        object.__setattr__(self, "seed", seed)
        if pool_ids is not None:
            object.__setattr__(self, "pool_ids", pool_ids)
            by_segment, by_pool = {}, {}
            for time, pool_id in zip(times, pool_ids):
                segment = time // self.design.segment_length
                if by_segment.setdefault(segment, pool_id) != pool_id:
                    raise ValueError("one sampling draw segment must retain one pool ID")
                if by_pool.setdefault(pool_id, segment) != segment:
                    raise ValueError("independent sampling draw segments require different pool IDs")

    @property
    def num_points(self):
        return self.coordinates.shape[1]

    def point_pool_id(self, time):
        """Return the explicit pool ID for a retained physical time, if declared."""
        if self.pool_ids is None:
            return None
        try:
            slot = self.times.index(time)
        except ValueError:
            return None
        return self.pool_ids[slot]

    @property
    def sampling_groups(self):
        """Return explicit metadata for the known sampling pools."""
        if self.pool_ids is None or self.design is None:
            return {}
        spatial_sites = prod(self.lattice_size[:3])
        return {
            pool_id: SamplingPoolSpec(pool_id, spatial_sites, self.num_points)
            for pool_id in dict.fromkeys(self.pool_ids)
        }

