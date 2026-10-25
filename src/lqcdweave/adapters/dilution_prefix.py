"""An explicit marginal QR-column prefix for a separate comparison estimator.

This view does not support joint contractions with its original larger sample.
It retains the complete parent space, rebases actual selected-column identity,
and updates local weights when consuming already weighted local elementals.
"""

from dataclasses import replace
from hashlib import sha256

import numpy as np

from ..contracts.dataset import canonical_json
from ..contracts.dilution import DilutionGroup, DilutionSource, DILUTION_WEIGHT_SCHEME
from ..contracts.sampling_weights import expected_weight


class DilutionPrefix:
    """Shared actual prefix identity for compatible elemental/propagator views."""

    def __init__(self, rank):
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        self.rank = rank
        self.sources = {}

    def bind(self, source):
        if not isinstance(source, DilutionSource) or len(source.groups) != 1:
            raise ValueError("blending prefix requires one explicit complete-complement D group")
        group = source.groups[0]
        if group.branch != "qr" or self.rank > group.m:
            raise ValueError("blending prefix requires enough actual Haar QR columns")
        previous = self.sources.get(source.data_id)
        if previous is not None:
            if previous[0] != source:
                raise ValueError("one D dataset has inconsistent actual source metadata")
            return previous[1]
        selected_group = DilutionGroup(group.parent_modes, group.vector_labels[:self.rank], "qr")
        identity = "qr-prefix:" + sha256(canonical_json(
            {"source": source.document, "rank": self.rank}).encode()).hexdigest()
        selected = DilutionSource(identity, source.parent_basis_id, (selected_group,),
                                  source.time_pools, "qr-dilution-prefix-v1")
        self.sources[source.data_id] = (source, selected)
        return selected

    def metadata(self, metadata):
        bindings = dict(metadata.axis_bindings)
        for axis, (identity, labels) in tuple(bindings.items()):
            if identity not in self.sources:
                continue
            original, selected = self.sources[identity]
            if any(type(i) is not int or not 0 <= i < self.rank for i in labels):
                raise ValueError("D prefix metadata contains columns outside its actual selection")
            bindings[axis] = (selected.data_id, labels)
        return replace(metadata, axis_bindings=bindings,
                       selection_identity=repr((metadata.selection_identity, "qr-prefix-v1", self.rank)))


class _PrefixReader:
    background_read_safe = False

    def __init__(self, reader, prefix):
        self._reader, self.prefix = reader, prefix

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def _selection(self, sector, selection, *, elemental):
        result = dict(selection)
        mode = dict(result.pop("mode_indices", None) or {})
        kinds = (sector[0], sector[2])
        axes = (("mode_left", "mode_right") if elemental else ("sink_mode", "source_mode"))
        for kind, axis in zip(kinds, axes):
            if kind != "d":
                continue
            chosen = mode.get(axis, tuple(range(self.prefix.rank)))
            if isinstance(chosen, slice):
                chosen = tuple(range(self.prefix.rank))[chosen]
            chosen = tuple(chosen)
            if any(type(i) is not int or not 0 <= i < self.prefix.rank for i in chosen):
                raise ValueError("mode selection exceeds the declared D prefix")
            mode[axis] = chosen
        if mode:
            # Preserve requested V prefixes while selecting D independently.
            ne = result.pop("usedNe", None)
            if ne is not None:
                for kind, axis in zip(kinds, axes):
                    if kind == "v" and axis not in mode:
                        mode[axis] = tuple(range(ne))
            result["mode_indices"] = mode
        return result


class DilutionPrefixElemental(_PrefixReader):
    def _bind(self, component, momentum, time, sector):
        for kind, side, axis in zip((sector[0], sector[2]), ("bar", "field"),
                                   ("mode_left", "mode_right")):
            if kind != "d":
                continue
            original = self._reader.dilution_basis(component, momentum, time, sector, side)
            self.prefix.bind(original)
            metadata = self._reader.describe_block(component, momentum, time, sector)
            if metadata.axis_bindings.get(axis) != (original.data_id, original.ordered_indices):
                raise ValueError("D prefix requires the actual complete ordered original column axis")

    def dilution_basis(self, component, momentum, time, sector, side):
        return self.prefix.bind(self._reader.dilution_basis(component, momentum, time, sector, side))

    def describe_block(self, component, momentum, time, sector, **selection):
        self._bind(component, momentum, time, sector)
        return self.prefix.metadata(self._reader.describe_block(component, momentum, time, sector,
            **self._selection(sector, selection, elemental=True)))

    def block(self, component, momentum, time, sector, **selection):
        self._bind(component, momentum, time, sector)
        selected = self._selection(sector, selection, elemental=True)
        value = self._reader.block(component, momentum, time, sector, **selected)
        if not self.elemental_weighted or "d" not in sector:
            return value
        if self.dilution_weight_scheme != DILUTION_WEIGHT_SCHEME:
            raise ValueError("weighted D prefix requires component_inclusion-v1")
        if component.time_offsets != (0, 0):
            raise NotImplementedError("weighted D prefixes currently support local equal-time elementals")
        metadata = self._reader.describe_block(component, momentum, time, sector, **selected)
        d_axes = [axis for kind, axis in zip((sector[0], sector[2]), ("mode_left", "mode_right"))
                  if kind == "d"]
        originals = [self.prefix.sources[metadata.axis_bindings[axis][0]][0] for axis in d_axes]
        if len(d_axes) == 2 and originals[0].pool_key(time, 0) == originals[1].pool_key(time, 0):
            group = originals[0].groups[0]
            left, right = (np.asarray(metadata.axis_bindings[axis][1]) for axis in d_axes)
            same = left[:, None] == right[None, :]
            w1 = float(expected_weight(group.n, self.prefix.rank, 1) / expected_weight(group.n, group.m, 1))
            w2 = float(expected_weight(group.n, self.prefix.rank, 2) / expected_weight(group.n, group.m, 2))
            return value * np.where(same, w1, w2)
        ratio = np.prod([source.groups[0].m / self.prefix.rank for source in originals])
        return value * ratio

    def describe_slab(self, component, momentum, times, sector, **selection):
        times = tuple(times)
        for time in times:
            self._bind(component, momentum, time, sector)
        return self.prefix.metadata(self._reader.describe_slab(component, momentum, times, sector,
            **self._selection(sector, selection, elemental=True)))

    def read_slab(self, component, momentum, times, sector, **selection):
        if not self.elemental_weighted or "d" not in sector:
            return self._reader.read_slab(component, momentum, times, sector,
                **self._selection(sector, selection, elemental=True))
        return np.stack([self.block(component, momentum, time, sector, **selection) for time in times])

    def read_slab_workspace_bytes(self, metadata, **selection):
        describe = getattr(self._reader, "read_slab_workspace_bytes", None)
        base = describe(metadata, **selection) if callable(describe) else 0
        return base + 2 * int(np.prod(metadata.shape)) * metadata.dtype.itemsize


class DilutionPrefixPropagator(_PrefixReader):
    def describe_block(self, sink, source, sector="vsv", **selection):
        return self.prefix.metadata(self._reader.describe_block(sink, source, sector,
            **self._selection(sector, selection, elemental=False)))

    def block(self, sink, source, sector="vsv", **selection):
        return self._reader.block(sink, source, sector,
            **self._selection(sector, selection, elemental=False))

    def describe_slab(self, sinks, source, sector="vsv", **selection):
        return self.prefix.metadata(self._reader.describe_slab(sinks, source, sector,
            **self._selection(sector, selection, elemental=False)))

    def read_slab(self, sinks, source, sector="vsv", **selection):
        return self._reader.read_slab(sinks, source, sector,
            **self._selection(sector, selection, elemental=False))
