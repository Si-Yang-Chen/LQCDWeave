"""Adapt explicit source-time rows to the canonical pair-reader protocol.

This adapter indexes already-loaded numeric arrays; it does not open files,
decode pickle data, run a solver, or infer a time layout from array shape. The
caller must decode only trusted legacy inputs, verify basis/endpoint identity,
and keep mapped row storage alive while borrowed direct reads are in use.
"""

from numbers import Integral

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.propagator import PropagatorBlock, PropagatorEndpoint
from lqcdweave.production.perambulator.api import (
    ALL_PRODUCTS,
    SAVED_PRODUCTS,
    propagator_pair_spec,
    reverse_product,
    reverse_propagator,
)


def _physical_time(value, lt, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain physical integer times")
    value = int(value)
    if not 0 <= value < lt:
        raise ValueError(f"{name} must lie in [0, {lt})")
    return value


def _ordered_times(values, name, lt):
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of physical integer times") from error
    if not values:
        raise ValueError(f"{name} cannot be empty")
    result = tuple(_physical_time(value, lt, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain repeated times")
    return result


def _time_pair(value, lt):
    if isinstance(value, (str, bytes)):
        raise ValueError("time_pair must contain sink and source physical times")
    try:
        pair = tuple(value)
    except TypeError as error:
        raise ValueError("time_pair must contain sink and source physical times") from error
    if len(pair) != 2:
        raise ValueError("time_pair must contain sink and source physical times")
    return (_physical_time(pair[0], lt, "time_pair"),
            _physical_time(pair[1], lt, "time_pair"))


def _product_shape(sink, source):
    sink_axes = (len(sink.indices), 3) if sink.kind == "P" else (len(sink.indices),)
    source_axes = (len(source.indices), 3) if source.kind == "P" else (len(source.indices),)
    return (4, 4) + sink_axes + source_axes


def _canonical_spec(spec):
    if not isinstance(spec, DatasetSpec):
        raise TypeError("spec must be a DatasetSpec")
    document = spec.document
    try:
        definition = document["definition"]
        conventions = document["conventions"]
        if (document["kind"] != "perambulator_pairs" or
                definition["encoding"] != "pr-six-pairs-v1"):
            raise ValueError("expected a pr-six-pairs-v1 perambulator dataset")
        sink = PropagatorEndpoint.from_document(definition["sink_endpoint"])
        source = PropagatorEndpoint.from_document(definition["source_endpoint"])
        encoded_gamma5 = conventions["gamma5"]
        gamma5 = (np.asarray(encoded_gamma5["real"]) +
                  1j * np.asarray(encoded_gamma5["imag"]))
        rebuilt = propagator_pair_spec(
            definition["product"],
            configuration=document["configuration"],
            sink=sink,
            source=source,
            time_pairs=spec.coordinates("time_pair"),
            gauge_id=definition["gauge_id"],
            dirac=definition["dirac"],
            gamma5=gamma5,
            spin_basis=conventions["spin_basis"],
            gamma5_hermitian=conventions["gamma5_hermitian"],
            dtype=document["dtype"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("spec must be a canonical pr-six-pairs-v1 propagator dataset") from error
    if rebuilt.document != document:
        raise ValueError("spec must be a canonical pr-six-pairs-v1 propagator dataset")
    return document, sink, source, gamma5


class SourceTimePropagatorReader:
    """Expose ordered source-time rows as ``DatasetReader``-like pair reads.

    ``load_row`` receives an actual physical source time and must return its
    already-loaded NumPy ndarray or memmap. ``row_times`` declares the raw row
    axis: physical sink times for ``absolute`` storage, or physical cyclic
    sink-minus-source offsets for ``relative`` storage. ``source_times`` and
    ``row_times`` are explicit ordered coordinates; no layout is guessed.

    Direct reads return borrowed row views. Reads that require reversing a
    noncanonical raw product return an owned Gamma5-derived array.
    """

    def __init__(self, spec, load_row, *, product, source_times, row_times,
                 time_convention, data_id):
        if not callable(load_row):
            raise TypeError("load_row must be callable")
        if time_convention not in ("absolute", "relative"):
            raise ValueError("time_convention must be explicitly absolute or relative")
        if not isinstance(data_id, str) or not data_id:
            raise ValueError("data_id must be an actual nonempty dataset ID")

        document, spec_sink, spec_source, gamma5 = _canonical_spec(spec)
        raw_product = str(product).upper()
        if raw_product not in ALL_PRODUCTS:
            raise ValueError("product must be one of " + ", ".join(ALL_PRODUCTS))

        definition = document["definition"]
        saved_product = definition["product"]
        reverse = raw_product not in SAVED_PRODUCTS
        expected_saved_product = reverse_product(raw_product) if reverse else raw_product
        if saved_product != expected_saved_product:
            raise ValueError("spec product does not match the raw product and endpoint orientation")

        if reverse:
            raw_sink, raw_source = spec_source, spec_sink
        else:
            raw_sink, raw_source = spec_sink, spec_source

        lt = document["configuration"]["lattice"][3]
        source_times = _ordered_times(source_times, "source_times", lt)
        row_times = _ordered_times(row_times, "row_times", lt)
        source_set, row_positions = set(source_times), {time: i for i, time in enumerate(row_times)}
        pairs = tuple(_time_pair(pair, lt) for pair in spec.coordinates("time_pair"))
        if not pairs:
            raise ValueError("spec must declare at least one physical time pair")

        pair_positions = {}
        for pair in pairs:
            raw_sink_time, raw_source_time = pair[::-1] if reverse else pair
            raw_row_time = (raw_sink_time if time_convention == "absolute" else
                            (raw_sink_time - raw_source_time) % lt)
            if raw_source_time not in source_set:
                raise ValueError("spec time-pair coverage is outside source_times")
            if raw_row_time not in row_positions:
                raise ValueError("spec time-pair coverage is outside row_times")
            pair_positions[pair] = (raw_sink_time, raw_source_time,
                                    row_positions[raw_row_time])

        self.spec = spec
        self.data_id = data_id
        self._load_row = load_row
        self._lt = lt
        self._raw_product = raw_product
        self._reverse = reverse
        self._raw_sink = raw_sink
        self._raw_source = raw_source
        self._gamma5 = np.array(gamma5, copy=True)
        self._gamma5.setflags(write=False)
        self._row_shape = (len(row_times),) + _product_shape(
            raw_sink, raw_source)
        self._dtype = np.dtype(document["dtype"])
        self._pair_positions = pair_positions

    def read(self, *, time_pair):
        """Read one declared physical ``[sink_time, source_time]`` pair."""
        pair = _time_pair(time_pair, self._lt)
        try:
            raw_sink_time, raw_source_time, row_position = self._pair_positions[pair]
        except KeyError:
            raise KeyError("physical time pair is outside the declared row coverage") from None

        row = self._load_row(raw_source_time)
        if not isinstance(row, np.ndarray):
            raise TypeError("load_row must return a NumPy ndarray or memmap")
        if tuple(row.shape) != self._row_shape:
            raise ValueError("loaded source-time row shape differs from its explicit metadata")
        if row.dtype != self._dtype:
            raise ValueError("loaded source-time row dtype differs from the DatasetSpec")

        values = row[row_position]
        if not self._reverse:
            return values
        raw_block = PropagatorBlock(
            self._raw_product,
            raw_sink_time,
            raw_source_time,
            self._raw_sink,
            self._raw_source,
            values,
            parent_id=self.data_id,
        )
        return reverse_propagator(
            raw_block, self._gamma5, gamma5_hermitian=True).values


__all__ = ["SourceTimePropagatorReader"]
