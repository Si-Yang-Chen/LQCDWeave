"""Canonical spatial point handles; no sampling or path guessing."""

from copy import deepcopy
import numpy as np

from lqcdweave.contracts.time_selection import TimeSelectionRecord, normalize_time_selection
from lqcdweave.contracts.spatial_sampling import SpatialPointSet, SpatialSamplingDesign


class SpatialPointHandle:
    """Owned ordered point snapshot implementing production's Np/load protocol.

    ``cfg=None`` deliberately leaves configuration unbound. Otherwise load must
    receive exactly that key. No identity is inferred from coordinates or paths.
    File constructors eagerly copy and close mappings; no close() is needed.
    """

    def __init__(self, points, *, cfg=None):
        if not isinstance(points, SpatialPointSet):
            raise TypeError("points must be a SpatialPointSet")
        self.time_selection = TimeSelectionRecord(points.lattice_size[3], points.times)
        self._coordinates = points.coordinates
        self.points = points
        self.Np = points.num_points
        self.cfg = cfg
        self.provenance = None
        self.source_origin = None
        self.data_id = None
        self.basis_id = None

    def load(self, key):
        """Borrow immutable ``(Nt,Np,3)`` int64 coordinates for this snapshot."""
        if self.cfg is not None and key != self.cfg:
            raise ValueError("point handle configuration differs from requested key")
        return self._coordinates

    def read_points(self, key, time, start=0, stop=None):
        """Borrow one physical-time point prefix/block from the ordered snapshot."""
        values = self.load(key)
        stop = self.Np if stop is None else stop
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Np):
            raise ValueError("point block must be a nonempty interval within Np")
        return values[self.time_selection.slot(time), start:stop]

    @classmethod
    def from_npy(cls, path, *, lattice_size, cfg=None, time_selection=None):
        """Read canonical coordinate NPY with explicit geometry and unknown history."""
        from .preset import load

        selection = normalize_time_selection(time_selection, Lt=lattice_size[3])
        coordinates = load(path, kind="points")
        try:
            points = SpatialPointSet(lattice_size, selection.times, coordinates)
        finally:
            coordinates._mmap.close()
        return cls(points, cfg=cfg)

    @classmethod
    def from_dataset(cls, directory, *, cfg=None):
        """Read a complete spatial-point dataset through existing storage.

        Preserve stored provenance/source status without upgrading unknown origin.
        Restore an explicit stored sampling design and pool identities when present.
        Configuration is selected from the registration, never from the directory name.
        """
        from lqcdweave.storage.dataset import DatasetReader
        from lqcdweave.contracts.time_selection import dataset_physical_times

        with DatasetReader(directory, cfg=cfg) as reader:
            spec = reader.spec
            document = spec.document
            if (document["kind"] != "spatial_points" or
                    spec.axis_names != ("time", "point", "xyz") or
                    spec.coordinates("xyz") != ["x", "y", "z"] or
                    spec.coordinates("point") != list(range(spec.shape[1]))):
                raise ValueError("dataset must declare ordered spatial point coordinates")
            configuration = document["configuration"]
            lattice = tuple(configuration["lattice"])
            times = dataset_physical_times(spec, "time")
            stored_times = tuple(spec.coordinates("time"))
            realization = document["definition"].get("realization", {})
            design = seed = algorithm = pool_ids = None
            if realization.get("kind") == "local_spatial_sampling":
                design = SpatialSamplingDesign(**realization["design"])
                seed, algorithm = realization["seed"], realization["algorithm"]
                # One registered realization owns these explicitly declared draws.
                pool_ids = tuple(f"spatial-pool:dataset:{reader.data_id}:{t // design.segment_length}"
                                 for t in times)
            elif realization.get("kind") == "caller_spatial_point_set" and "sampling" in realization:
                sampling = realization["sampling"]
                design = SpatialSamplingDesign(**sampling["design"])
                seed, algorithm = sampling["seed"], sampling["algorithm"]
                pool_ids = sampling["pool_ids"]
            points = SpatialPointSet(lattice, times,
                np.stack([reader.read(time=t) for t in stored_times]), design, seed, algorithm, pool_ids)
            result = cls(points, cfg=configuration["configuration"])
            result.provenance = deepcopy(reader.provenance)
            result.source_origin = deepcopy(reader.source_origin)
            result.data_id = reader.data_id
            result.basis_id = reader.data_id
        return result


__all__ = ["SpatialPointHandle"]
