"""Pure DatasetSpec builders for full-time spatial point realizations."""

import json
from collections.abc import Mapping

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json, stable_digest
from lqcdweave.contracts.time_selection import normalize_time_selection, bind_time_selections
from lqcdweave.contracts.spatial_sampling import (
    SpatialPointSet, SpatialSamplingDesign, sampling_integer,
)


LOCAL_SPATIAL_POINT_ALGORITHM = "numpy.PCG64/partial-fisher-yates-v1"


def _configuration_document(configuration, lattice_size):
    if not isinstance(configuration, Mapping) or set(configuration) != {
            "ensemble", "configuration", "lattice"}:
        raise ValueError("configuration must explicitly contain ensemble, configuration and lattice")
    try:
        result = json.loads(canonical_json(dict(configuration)))
    except (TypeError, ValueError) as exc:
        raise ValueError("configuration must contain legal JSON values") from exc
    lattice = result["lattice"]
    if (not isinstance(lattice, list) or len(lattice) != 4 or
            any(type(size) is not int or size < 1 for size in lattice) or
            lattice != list(lattice_size)):
        raise ValueError("explicit configuration lattice differs from the point realization")
    DatasetSpec(dict(schema=1, kind="spatial_points", configuration=result,
        axes=[dict(name="time", coordinates=[0]),
              dict(name="point", coordinates=[0]),
              dict(name="xyz", coordinates=["x", "y", "z"])],
        dtype="int64", bases={}, definition={}, conventions={}, parents=[]))
    return result


def _point_document(configuration, times, count, realization, selection=None):
    if not isinstance(times, (list, tuple)) or not times:
        raise ValueError("point realization needs explicit physical times")
    times = list(times)
    if (any(type(time) is not int or time < 0 for time in times) or
            len(set(times)) != len(times) or (selection is None and sorted(times) != list(range(configuration["lattice"][3])))):
        raise ValueError("point realization must cover every physical time exactly once")
    count = sampling_integer(count, "num_points")
    document = dict(schema=1, kind="spatial_points", configuration=configuration,
        axes=[dict(name="time", coordinates=times),
              dict(name="point", coordinates=list(range(count))),
              dict(name="xyz", coordinates=["x", "y", "z"])],
        dtype="int64", bases={}, definition={"realization": realization},
        conventions={"representation": "coordinate_delta",
                     "normalization": "unit_delta"}, parents=[])
    return DatasetSpec(document if selection is None else bind_time_selections(document, {"time": selection}))


def sampled_spatial_points_spec(design, *, configuration, seed, selection=None):
    """Describe the fixed supported local sampler without sampling points."""
    if not isinstance(design, SpatialSamplingDesign):
        raise TypeError("design must be a SpatialSamplingDesign")
    seed = sampling_integer(seed, "seed", 0)
    cfg = _configuration_document(configuration, design.lattice_size)
    realization = dict(schema=1, kind="local_spatial_sampling",
        design=dict(lattice_size=list(design.lattice_size),
                    num_points=design.num_points,
                    temporal_policy=design.temporal_policy),
        seed=seed, algorithm=LOCAL_SPATIAL_POINT_ALGORITHM)
    times = tuple(range(design.lattice_size[3])) if selection is None else normalize_time_selection(selection, Lt=design.lattice_size[3]).times
    return _point_document(cfg, times, design.num_points, realization, selection)


def validate_complete_spatial_point_set(points, *, configuration, selection=None):
    """Validate an owned full-time point set against explicit configuration."""
    if not isinstance(points, SpatialPointSet):
        raise TypeError("points must be a SpatialPointSet")
    cfg = _configuration_document(configuration, points.lattice_size)
    times = tuple(points.times)
    if (any(type(time) is not int for time in times) or
            (selection is None and sorted(times) != list(range(points.lattice_size[3])))):
        raise ValueError("persistent point publication requires every physical time exactly once")
    count = points.num_points
    payload = dict(schema=1, lattice_size=list(points.lattice_size),
        time_order=list(times), point_order=list(range(count)),
        xyz_order=["x", "y", "z"], dtype="int64",
        coordinates=points.coordinates.tolist())
    realization = dict(schema=1, kind="caller_spatial_point_set",
                       sha256=stable_digest(payload))
    if points.design is not None:
        realization["sampling"] = dict(
            design=dict(lattice_size=list(points.design.lattice_size),
                        num_points=points.design.num_points,
                        temporal_policy=points.design.temporal_policy),
            seed=points.seed, algorithm=points.algorithm,
            pool_ids=None if points.pool_ids is None else list(points.pool_ids))
    return _point_document(cfg, times, count, realization, selection)
