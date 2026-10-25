"""Adapt verified registered one-link elementals to the old V2V raw contract."""

import numpy as np
from lqcdweave.contracts.time_selection import dataset_time_selection

from lqcdweave.contracts.current import (
    build_current_raw_contract, validate_current_raw_contract,
)
from lqcdweave.planning.directed_current import validate_directed_current_datasets


class RegisteredVSV:
    """Borrow a verified DatasetReader through its public absolute-time API."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get(self, source_time, sink_time):
        return self.dataset.read(source_time=source_time, sink_time=sink_time)


class _DirectedCurrentRawView:
    """Lazy direction/time/momentum/mode view over one registered dataset."""

    def __init__(self, dataset, plan):
        self.dataset = dataset
        self.shape = (8, len(plan["times"]), len(plan["momenta"]),
                      len(plan["basis"]["indices"]), len(plan["basis"]["indices"]))
        self.dtype = np.dtype(dataset.spec.document["dtype"])
        components = dataset.spec.coordinates("component")
        self._components = {
            component["path"][0]: component
            for component in components
            if component["path"]
        }
        self._momenta = tuple(tuple(value) for value in dataset.spec.coordinates("momentum"))
        self._modes = {name: tuple(dataset.spec.coordinates(name))
                       for name in ("sink_mode", "source_mode")}

    @staticmethod
    def _scalar(value, extent, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"directed-current {name} index must be an integer")
        value = int(value)
        if value < 0:
            value += extent
        if not 0 <= value < extent:
            raise IndexError(f"directed-current {name} index is outside coverage")
        return value

    @staticmethod
    def _mode_selection(value, extent, name):
        if isinstance(value, slice):
            start, stop, step = value.indices(extent)
            if step < 0:
                raise IndexError("directed-current mode slices must have a positive step")
            return slice(start, stop, step)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"directed-current {name} selection must be an integer or slice")
        index = int(value)
        if index < 0:
            index += extent
        if not 0 <= index < extent:
            raise IndexError(f"directed-current {name} index is outside coverage")
        return slice(index, index + 1)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > 5 or any(item is None or item is Ellipsis for item in key):
            raise IndexError("directed-current raw view requires five explicit axes")
        key = (*key, *(slice(None) for _ in range(5 - len(key))))
        direction = self._scalar(key[0], 8, "direction")
        time = self._scalar(key[1], self.shape[1], "time")
        momentum = self._scalar(key[2], self.shape[2], "momentum")
        try:
            component = self._components[direction]
        except KeyError as exc:
            raise KeyError(f"registered current dataset lacks direction {direction}") from exc
        select = {
            axis: self._mode_selection(index, self.shape[position], axis)
            for position, (axis, index) in enumerate(
                (("sink_mode", key[3]), ("source_mode", key[4])), start=3)
        }
        value = self.dataset.read(
            component=component,
            time=time,
            momentum=self._momenta[momentum],
            select=select,
        )
        for axis, index in reversed(list(zip(("sink_mode", "source_mode"), key[3:5], strict=True))):
            if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
                value = np.squeeze(value, axis=("sink_mode", "source_mode").index(axis))
        return value


def load_registered_directed_current(elemental_dataset, *vsv_datasets):
    """Bind a checked raw basis without reading its values.

    DatasetReader validates the registration and complete stored coverage when
    opened. The planning check binds the actual basis, ordered
    modes, configuration, phase and time conventions. This adapter then reads
    The raw array proxy reads exactly one direction/time/momentum and selected
    mode range when the contraction requests it. VSV blocks are read on demand.
    """
    if dataset_time_selection(elemental_dataset.spec, 'time') is not None:
        raise NotImplementedError('directed-current kernels need explicit physical endpoint mappings for compact slots')
    plan = validate_directed_current_datasets(
        elemental_dataset.spec, *(dataset.spec for dataset in vsv_datasets))
    values = _DirectedCurrentRawView(elemental_dataset, plan)
    raw = {"v2v": values}
    ne = len(plan["basis"]["indices"])
    nm = len(plan["momenta"])
    contract = build_current_raw_contract(
        raw, boundary=plan["boundary"], available_ne=ne, used_ne=ne,
        momentum_count=nm)
    validate_current_raw_contract(
        raw, contract, require_temporal=True, validate_values=False,
    )
    return dict(raw=raw, contract=contract,
                vsv=tuple(RegisteredVSV(dataset) for dataset in vsv_datasets),
                elemental_data_id=elemental_dataset.data_id,
                vsv_data_ids=tuple(dataset.data_id for dataset in vsv_datasets),
                configuration=plan["configuration"], basis=plan["basis"],
                momenta=plan["momenta"])
