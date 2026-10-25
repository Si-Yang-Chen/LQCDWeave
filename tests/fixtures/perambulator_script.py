"""Test-owned production script composition; not an installed production API."""

from collections.abc import Mapping
from typing import Callable, Literal, Sequence
import numpy as np
from lqcdweave.contracts.time_selection import dataset_time_selection, dataset_physical_times
from lqcdweave.production.perambulator.api import PerambulatorGenerator
from lqcdweave.storage.dataset import DatasetWriter

def _host_row(value, to_host, expected_shape, expected_dtype):
    host = np.array(value, copy=True) if to_host is None else to_host(value)
    if not isinstance(host, np.ndarray):
        raise TypeError("to_host must return a NumPy array")
    if host.shape != expected_shape:
        raise ValueError("perambulator row shape differs from its DatasetSpec")
    if host.dtype != expected_dtype:
        raise ValueError("perambulator row dtype differs from its DatasetSpec")
    if not np.isfinite(host).all():
        raise ValueError("perambulator row contains non-finite values")
    return host

def produce_source_rows(writer: DatasetWriter, generator: PerambulatorGenerator,
                            physical_times: Sequence[int], configuration,
                            product: Literal["VSV", "PSV", "PSP", "VSP"] = "VSV", *, src, sink,
                            to_host: Callable[[object], np.ndarray] | None = None) -> None:
    """Write and release each owned product row before computing the next one.

    ``to_host=None`` makes a synchronous owned NumPy copy when the caller knows
    the result is host-resident. A non-NumPy caller supplies its own synchronous
    host-transfer function.
    """
    if product not in ("VSV", "PSV", "PSP", "VSP"):
        raise ValueError("product must be VSV, PSV, PSP or VSP")
    spec = writer.spec
    physical_times = tuple(physical_times)
    source_times = dataset_physical_times(spec, "source_time")
    sink_times = dataset_physical_times(spec, "sink_time")
    if spec.document["kind"] != product.lower() or source_times != physical_times:
        raise ValueError("writer and perambulator physical-time/product contract differ")
    declared_configuration = spec.document['configuration']
    expected_configuration = (declared_configuration if isinstance(configuration, Mapping)
                              else declared_configuration['configuration'])
    if configuration != expected_configuration:
        raise ValueError("writer and perambulator configuration differ")
    lattice = tuple(declared_configuration['lattice'])
    if tuple(getattr(generator, 'latt_size', lattice)) != lattice:
        raise ValueError("writer and perambulator physical lattice differ")
    source_record = getattr(generator, 'source_selection', None)
    sink_record = getattr(generator, 'sink_selection', None)
    for record in (source_record, sink_record):
        if record is not None and record.Lt != lattice[3]:
            raise ValueError("writer and perambulator selection physical Lt differ")
    if source_record is not None:
        source_record.require_times(source_times)
    if sink_record is not None and sink_record.times != sink_times:
        raise ValueError("writer and perambulator sink selection/order differ")
    if sink_record is None and dataset_time_selection(spec, "sink_time") is not None:
        raise ValueError("selected writer requires a generator declaring sink_selection")
    if not writer.missing:
        return
    generator.load(configuration)
    expected_shape = spec.shape[1:]
    expected_dtype = np.dtype(spec.document["dtype"])
    for position in writer.missing:
        source_time = physical_times[position]
        result = generator.calc(source_time, src=src, sink=[sink])[0]
        if result.product != product:
            raise ValueError('actual source/sink bases differ from the requested product')
        value = result.values
        row = _host_row(value, to_host, expected_shape, expected_dtype)
        writer.write(position, row)
        del row, value, result
