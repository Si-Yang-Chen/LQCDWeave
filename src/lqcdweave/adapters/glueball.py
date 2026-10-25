"""Read already measured glueball components; no gauge-loop production here."""

from pathlib import Path

import numpy as np

from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.contracts.elemental import integer
from lqcdweave.contracts.time_selection import normalize_time_selection, dataset_time_selection
from .selection import host_arrays_resident


class ArrayGlueball:
    """Array, NPY or registered Dataset reader for measured glueball values.

    ``axis_order`` describes the stored axes. Reordering is a view, including
    memory-mapped NPY inputs. Components and momenta retain their supplied order.
    """

    def __init__(self, data, *, components, momentum_list, phase_convention,
                 cfg=None, data_identity=None,
                 axis_order=("component", "momentum", "time"), time_selection=None):
        axes = ("component", "momentum", "time")
        if len(axis_order) != 3 or set(axis_order) != set(axes):
            raise ValueError("glueball axis_order must contain component, momentum, time")
        self._mapped = self._dataset = None
        if isinstance(data, (str, Path)):
            path = Path(data).resolve()
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            self._mapped = data
            if data_identity is None:
                data_identity = str(path)
        if data.ndim != 3 or np.dtype(data.dtype).kind not in "fc":
            self.close()
            raise ValueError("glueball data must be a rank-three real or complex array")
        self.data = data.transpose(tuple(axis_order.index(axis) for axis in axes))
        self.host_resident = host_arrays_resident(data)
        try:
            self._configure(self.data.shape, self.data.dtype, components, momentum_list,
                            phase_convention, cfg, data_identity, time_selection)
        except Exception:
            self.close()
            raise

    def _configure(self, shape, dtype, components, momentum_list, phase_convention,
                   cfg, data_identity, time_selection):
        self.dtype = np.dtype(dtype)
        self.components = tuple(components)
        if any(type(c) not in (str, int) for c in self.components):
            raise TypeError("glueball components must be string or integer labels")
        self.momentum_list = tuple(tuple(integer(x, "momentum") for x in p)
                                   for p in momentum_list)
        if (min(shape) < 1 or len(self.components) != shape[0]
                or len(set(self.components)) != len(self.components)):
            raise ValueError("glueball components must uniquely label the component axis")
        if (len(self.momentum_list) != shape[1]
                or any(len(p) != 3 for p in self.momentum_list)
                or len(set(self.momentum_list)) != len(self.momentum_list)):
            raise ValueError("glueball momentum_list must uniquely label the momentum axis")
        if not isinstance(phase_convention, str) or not phase_convention:
            raise ValueError("glueball input requires an explicit phase convention")
        self.phase_convention = phase_convention
        self.cfg, self.data_identity = cfg, data_identity
        self.Lt = shape[2]
        self.num_momentum = shape[1]
        self.time_selection = (None if time_selection is None else normalize_time_selection(time_selection))
        if self.time_selection is not None and len(self.time_selection.times) != self.Lt:
            raise ValueError("glueball time selection differs from stored time extent")
        self.time_selections = ({} if self.time_selection is None
                                else {"time": self.time_selection})
        self.time_selection_identity = (None if self.time_selection is None else
                                        (("time", self.time_selection.Lt, self.time_selection.times),))
        self.background_read_safe = self.host_resident
        self.glueball_definition = None
        self._closed = False

    @classmethod
    def from_dataset(cls, dataset):
        """Borrow the existing storage service; inspect only its metadata."""
        spec = dataset.spec
        document = spec.document
        axes = ("component", "momentum", "time")
        if (document["kind"] != "glueball" or set(spec.axis_names) != set(axes)
                or document["bases"] or np.dtype(document["dtype"]).kind not in "fc"):
            raise ValueError("glueball Dataset needs component,momentum,time axes and no quark bases")
        selection = dataset_time_selection(spec)
        times = spec.coordinates("time")
        if times != list(range(len(times))):
            raise ValueError("glueball Dataset time axis requires complete times or explicit compact slots")
        if selection is None and len(times) != document["configuration"]["lattice"][3]:
            raise ValueError("partial glueball time coverage needs an explicit time selection")
        result = cls.__new__(cls)
        result.data, result._mapped, result._dataset = None, None, dataset
        result.host_resident = False
        result._configure(
            tuple(len(spec.coordinates(axis)) for axis in axes), document["dtype"],
            spec.coordinates("component"), spec.coordinates("momentum"),
            document["conventions"].get("phase_convention"),
            document["configuration"]["configuration"], dataset.data_id, selection,
        )
        result.glueball_definition = canonical_json(document["definition"])
        return result

    def _indices(self, component, momentum, time):
        if self._closed:
            raise RuntimeError("glueball reader is closed")
        try:
            index = self.components.index(component)
        except ValueError as exc:
            raise KeyError(f"unknown glueball component {component!r}") from exc
        momentum, time = integer(momentum, "momentum", 0), integer(time, "time", 0)
        if momentum >= self.num_momentum or time >= self.Lt:
            raise IndexError("glueball momentum or time is outside input coverage")
        return index, momentum, time

    def describe_block(self, component, momentum, time):
        _, momentum, time = self._indices(component, momentum, time)
        selection = canonical_json({"component": component,
                                    "momentum": self.momentum_list[momentum], "time": time})
        return TensorAxisMetadata((), (), self.dtype, self.data_identity, selection)

    def block(self, component, momentum, time):
        i, p, t = self._indices(component, momentum, time)
        if self._dataset is not None:
            return self._dataset.read(component=component, momentum=list(self.momentum_list[p]), time=t)
        return self.data[i:i+1, p:p+1, t:t+1].reshape(())

    def describe_slab(self, component, momentum, times):
        times = tuple(integer(time, "time", 0) for time in times)
        for time in times:
            self._indices(component, momentum, time)
        selection = canonical_json({"component": component,
                                    "momentum": self.momentum_list[momentum], "times": times})
        return TensorAxisMetadata(("coordinate_batch",), (len(times),), self.dtype,
                                  self.data_identity, selection)

    def read_slab(self, component, momentum, times):
        times = tuple(integer(time, "time", 0) for time in times)
        for time in times:
            self._indices(component, momentum, time)
        if self._dataset is not None:
            return np.stack([self.block(component, momentum, time) for time in times])
        return self.data[self.components.index(component), momentum, list(times)]

    def close(self):
        """Release owned NPY mappings; borrowed arrays remain caller-owned."""
        self._closed = True
        if self._mapped is not None:
            self._mapped._mmap.close()
            self._mapped = None
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["ArrayGlueball"]
