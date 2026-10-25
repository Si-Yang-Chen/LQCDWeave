"""Representative CPU hot paths measured by ASV for each Git revision."""

from pathlib import Path

import numpy as np

from lqcdweave.contracts.spatial_sampling import SpatialSamplingDesign
from lqcdweave.sampling.temporal.api import find_min_loop, reconstruct_two_point
from lqcdweave.sampling.spatial import sample_spatial_points
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class TimeSelectionBenchmarks:
    def setup(self):
        self.lt = np.load(_fixture_dir() / "weak_field.npy", mmap_mode="r").shape[4]

    def time_find_min_loop_weak_field_full_time(self):
        find_min_loop(self.lt, self.lt)


class SpatialSamplingBenchmarks:
    def setup(self):
        gauge = np.load(_fixture_dir() / "weak_field.npy", mmap_mode="r")
        self.design = SpatialSamplingDesign((*gauge.shape[:3], gauge.shape[4]), 20)

    def time_sample_spatial_points_weak_field(self):
        sample_spatial_points(self.design, seed=20260926)


class TwoPointBenchmarks:
    def setup(self):
        elemental = np.load(_fixture_dir() / "weak_field.elemental.npy", mmap_mode="r")
        # Actual fixture values, arranged in the API's (source, index difference) order.
        values = np.trace(elemental[:4, 0], axis1=-2, axis2=-1)
        self.lt = values.shape[1]
        self.times = np.arange(self.lt)
        source, difference = np.indices((self.lt, self.lt))
        self.data = (values[:, (source + difference) % self.lt] *
                     values[:, source].conj()).transpose(1, 2, 0).copy()

    def time_reconstruct_weak_field_full_time(self):
        reconstruct_two_point(
            self.data,
            Lt=self.lt,
            selected_times=self.times,
            tsrc_axis=0,
            dt_axis=1,
        )


class _ArrayHandle:
    def __init__(self, value):
        self.value = value

    def load(self, key):
        if key != "weak_field":
            raise ValueError(key)
        return self.value


def _fixture_dir():
    return Path(__file__).resolve().parents[2] / "tests" / "data" / "weak_field"


class WeakFieldElementalBenchmarks:
    """CPU elemental production on the small, checked-in physical fixture."""

    def setup(self):
        from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator

        fixture_dir = _fixture_dir()
        gauge = np.load(fixture_dir / "weak_field.npy", mmap_mode="r")
        gauge = gauge.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6)
        vectors = np.load(fixture_dir / "weak_field.eigenvector.input.npy", mmap_mode="r")
        vectors = vectors.astype(np.complex64)
        momenta = (
            (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1),
            (0, 0, 2), (0, 1, 2), (1, 1, 2),
        )
        self.generator = ElementalGenerator(
            (4, 4, 4, 8),
            _ArrayHandle(gauge),
            _ArrayHandle(vectors),
            num_nabla=2,
            momentum_list=momenta,
            calc_mode="calc_deriv",
            ne_block_size=20,
            array_module=np,
        ).load("weak_field")

    def time_elemental_cpu_all_times(self):
        for time_slice in range(8):
            self.generator.calc_path_blocks(time_slice)["v2v"]
