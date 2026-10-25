"""Periodic templates and realized ordered times, independent of file storage."""

from dataclasses import dataclass
from numbers import Integral
from collections.abc import Mapping
from copy import deepcopy


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class TimeSelection:
    """Repeat offsets every ``loop`` sites on a periodic lattice of size ``Lt``."""

    Lt: int
    loop: int
    offsets: tuple[int, ...]

    def __post_init__(self):
        Lt = positive_integer(self.Lt, "Lt")
        loop = positive_integer(self.loop, "loop")
        if Lt % loop:
            raise ValueError("loop must divide Lt")
        offsets = tuple(self.offsets)
        if not offsets or any(
            isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x < loop
            for x in offsets
        ):
            raise ValueError("offsets must be nonempty integer coordinates in [0, loop)")
        offsets = tuple(sorted(int(x) for x in offsets))
        if len(set(offsets)) != len(offsets):
            raise ValueError("offsets must be unique")
        object.__setattr__(self, "Lt", Lt)
        object.__setattr__(self, "loop", loop)
        object.__setattr__(self, "offsets", offsets)

    @property
    def Nt(self) -> int:
        return len(self.offsets) * (self.Lt // self.loop)

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(base + o for base in range(0, self.Lt, self.loop) for o in self.offsets)


@dataclass(frozen=True)
class TimeSelectionRecord:
    """Actual physical times in their saved slot order, including wraparound."""

    Lt: int
    times: tuple[int, ...]

    def __post_init__(self):
        Lt = positive_integer(self.Lt, "Lt")
        times = tuple(self.times)
        if not times or any(
            isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < Lt
            for t in times
        ):
            raise ValueError("times must be nonempty integer coordinates in [0, Lt)")
        times = tuple(int(t) for t in times)
        if len(set(times)) != len(times):
            raise ValueError("times must be unique")
        object.__setattr__(self, "Lt", Lt)
        object.__setattr__(self, "times", times)

    @property
    def Nt(self) -> int:
        return len(self.times)

    @property
    def document(self):
        """Detached JSON data preserving the physical extent and slot order."""
        return {"Lt": self.Lt, "times": list(self.times)}

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, Mapping) or set(document) != {"Lt", "times"}:
            raise ValueError("time selection requires exactly Lt and ordered times")
        return cls(document["Lt"], tuple(document["times"]))

    def physical_time(self, slot):
        if isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < self.Nt:
            raise ValueError("time slot is outside the selection")
        return self.times[int(slot)]

    def slot(self, physical_time):
        if isinstance(physical_time, bool) or not isinstance(physical_time, Integral):
            raise ValueError("physical time must be an integer")
        try:
            return self.times.index(int(physical_time))
        except ValueError as exc:
            raise KeyError(f"missing physical time {physical_time} in selection") from exc

    def require_times(self, times):
        """Map actual endpoints to slots; missing endpoints never wrap by slot."""
        return tuple(self.slot(time) for time in times)


def normalize_time_selection(selection, *, Lt=None):
    """Normalize a template/realization without changing its actual time order."""
    if selection is None and Lt is not None:
        Lt = positive_integer(Lt, "Lt")
        return TimeSelectionRecord(Lt, tuple(range(Lt)))
    if not isinstance(selection, (TimeSelection, TimeSelectionRecord)):
        raise TypeError("selection must be a TimeSelection or TimeSelectionRecord")
    result = TimeSelectionRecord(selection.Lt, selection.times)
    if Lt is not None and result.Lt != positive_integer(Lt, "Lt"):
        raise ValueError("selection physical Lt differs from the lattice")
    return result


def validate_time_coordinates(document):
    """Validate the optional slot mapping inside an existing dataset contract."""
    metadata = document.get("conventions", {}).get("time_coordinates")
    if metadata is None or isinstance(metadata, str):
        return
    if (not isinstance(metadata, dict) or set(metadata) != {"domain", "layout", "selections"}
            or metadata["domain"] != "slot"
            or metadata["layout"] not in ("absolute", "cyclic_difference")):
        raise ValueError("time_coordinates requires explicit slot domain and known layout")
    selections = metadata["selections"]
    if not isinstance(selections, dict) or not selections:
        raise ValueError("slot time coordinates require ordered selections")
    axes = {axis["name"]: axis["coordinates"] for axis in document["axes"]}
    for axis, value in selections.items():
        record = TimeSelectionRecord.from_document(value)
        if record.Lt != document["configuration"]["lattice"][3]:
            raise ValueError("selection physical Lt differs from configuration lattice")
        if (axis not in axes or any(type(x) is not int for x in axes[axis])
                or axes[axis] != list(range(record.Nt))):
            raise ValueError("slot time axis must contain canonical slots matching its selection")
    if metadata["layout"] == "cyclic_difference":
        records = list(selections.values())
        if len(records) != 2 or records[0] != records[1]:
            raise ValueError("cyclic_difference requires two identical ordered time selections")


def bind_time_selections(document, selections, *, layout="absolute"):
    """Convert explicitly ordered physical axes to compact slots in a copied spec.

    This changes the scientific contract, never the tensor payload. Each input
    axis must already be in the selected physical order. Existing mappings may
    only be rebound identically, preventing accidental relabeling on recovery.
    """
    result = deepcopy(document)
    if not isinstance(selections, Mapping) or not selections:
        raise ValueError("selections must map time-axis names to actual records")
    records = {axis: normalize_time_selection(value, Lt=result["configuration"]["lattice"][3])
               for axis, value in selections.items()}
    metadata = dict(domain="slot", layout=layout,
                    selections={axis: record.document for axis, record in records.items()})
    existing = result["conventions"].get("time_coordinates")
    if existing is not None and existing != "absolute":
        if existing != metadata:
            raise ValueError("cannot rebind a dataset to different ordered time selections")
        validate_time_coordinates(result)
        return result
    axes = {axis["name"]: axis for axis in result["axes"]}
    for name, record in records.items():
        if name not in axes or axes[name]["coordinates"] != list(record.times):
            raise ValueError("physical axis order must exactly match its time selection")
        axes[name]["coordinates"] = list(range(record.Nt))
    result["conventions"]["time_coordinates"] = metadata
    validate_time_coordinates(result)
    return result


def dataset_time_selection(spec, axis="time"):
    """Return an explicit slot mapping, or None for legacy physical-axis data."""
    document = spec if isinstance(spec, Mapping) else spec.document
    validate_time_coordinates(document)
    metadata = document.get("conventions", {}).get("time_coordinates")
    if metadata is None or isinstance(metadata, str) or axis not in metadata["selections"]:
        return None
    return TimeSelectionRecord.from_document(metadata["selections"][axis])


def dataset_physical_times(spec, axis="time"):
    """Resolve a declared storage axis into ordered physical coordinates."""
    document = spec if isinstance(spec, Mapping) else spec.document
    selection = dataset_time_selection(document, axis)
    if selection is not None:
        return selection.times
    coordinates = next((a["coordinates"] for a in document["axes"] if a["name"] == axis), None)
    if coordinates is None:
        raise ValueError(f"missing axis {axis}")
    return TimeSelectionRecord(document["configuration"]["lattice"][3], tuple(coordinates)).times
