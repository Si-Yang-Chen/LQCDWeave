"""Uniform spatial point sampling with an isolated random generator."""

from uuid import uuid4

import numpy as np
from lqcdweave.contracts.time_selection import normalize_time_selection

from lqcdweave.contracts.spatial_sampling import (
    SpatialPointSet, SpatialSamplingDesign, sampling_integer,
)


def sample_spatial_points(design: SpatialSamplingDesign, *, seed=None, selection=None) -> SpatialPointSet:
    """Generate full time by default, or retain an explicitly selected ordered subset.

    The same seed preserves the full-time RNG stream: unselected draws are
    consumed without retaining their rows. Selection reduces output allocation,
    not the number of random draws.

    Partial Fisher-Yates sampling takes O(Np) auxiliary memory, without a
    full-volume permutation or rejection loops near full occupancy.
    """
    if seed is not None:
        seed = sampling_integer(seed, "seed", 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    lx, ly, lz, lt = design.lattice_size
    volume = lx * ly * lz
    times = tuple(range(lt)) if selection is None else normalize_time_selection(selection, Lt=lt).times
    slots = {time: slot for slot, time in enumerate(times)}
    coords = np.empty((len(times), design.num_points, 3), dtype=np.int64)
    realization_token = uuid4().hex
    pool_ids = tuple(
        f"spatial-pool:{realization_token}:{time // design.segment_length}"
        for time in times
    )
    row = np.empty((design.num_points, 3), dtype=np.int64)
    length = design.segment_length
    for start in range(0, lt, length):
        swaps = {}
        for p in range(design.num_points):
            remaining = volume - p
            index = int(rng.integers(remaining))
            flat = swaps.get(index, index)
            swaps[index] = swaps.get(remaining - 1, remaining - 1)
            swaps.pop(remaining - 1, None)
            row[p] = (flat % lx, (flat // lx) % ly, flat // (lx * ly))
        for t in range(start, min(start + length, lt)):
            if t in slots:
                coords[slots[t]] = row
    return SpatialPointSet(
        design.lattice_size, times, coords, design, seed,
        "numpy.PCG64/partial-fisher-yates-v1",
        pool_ids=pool_ids,
    )
