"""Compose dense elemental sectors with an explicit sparse P2P record source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Protocol

import numpy as np

from lqcdweave.contracts.elemental import (
    ElementalComponent,
    SparsePointRecordMetadata,
    integer,
    normalize_elemental_structure,
)
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.time_selection import normalize_time_selection


class SparsePointRecordReader(Protocol):
    """Header/read interface for raw legacy identity or COO P2P records."""

    def describe_record(
        self, component: ElementalComponent, momentum: int, t: int,
    ) -> SparsePointRecordMetadata: ...

    def read_record(
        self, component: ElementalComponent, momentum: int, t: int,
    ) -> Mapping: ...


class BorrowedSparsePointRecord(Mapping):
    """Validated immutable mapping that retains its provider-owned resource."""

    __slots__ = ("_record", "resource_token")

    def __init__(self, record, resource_token):
        self._record = MappingProxyType(dict(record))
        self.resource_token = resource_token

    def __getitem__(self, key):
        return self._record[key]

    def __iter__(self):
        return iter(self._record)

    def __len__(self):
        return len(self._record)


def validate_sparse_point_record(record, metadata: SparsePointRecordMetadata):
    """Validate the returned raw record against header-only byte metadata."""
    if not isinstance(metadata, SparsePointRecordMetadata):
        raise TypeError("sparse P2P metadata must be SparsePointRecordMetadata")
    if metadata.backend != "numpy":
        raise NotImplementedError(
            "SparseElemental currently validates CPU NumPy records only; device-native COO is not implemented"
        )
    if not isinstance(record, Mapping):
        raise TypeError("sparse P2P reader must return an identity/COO mapping")
    resource_token = (record.resource_token if isinstance(record, BorrowedSparsePointRecord)
                      else record)
    if record.get("type") != metadata.kind:
        raise ValueError("sparse P2P record kind differs from its metadata")
    if metadata.kind == "identity":
        if set(record) != {"type"}:
            raise ValueError("identity P2P records contain only their type tag")
        return BorrowedSparsePointRecord({"type": "identity"}, resource_token)

    if set(record) != {"type", "indices", "values"}:
        raise ValueError("COO P2P records require exactly type, indices, and values")
    indices, values = record["indices"], record["values"]
    if (type(indices).__module__.split(".", 1)[0] != "numpy"
            or type(values).__module__.split(".", 1)[0] != "numpy"):
        raise NotImplementedError(
            "SparseElemental does not stage device COO arrays to host; use a backend-native provider"
        )
    try:
        indices_shape, values_shape = tuple(indices.shape), tuple(values.shape)
        indices_dtype, values_dtype = np.dtype(indices.dtype), np.dtype(values.dtype)
        index_bytes, value_bytes = int(indices.nbytes), int(values.nbytes)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TypeError("COO P2P indices and values must be typed array buffers") from exc
    if indices_dtype != metadata.index_dtype or values_dtype != metadata.value_dtype:
        raise ValueError("COO P2P dtypes differ from their metadata")
    if indices_shape != (metadata.nnz, 2) or values_shape != (metadata.nnz, 3, 3):
        raise ValueError("COO P2P shape/nnz differs from its metadata")
    if index_bytes + value_bytes != metadata.record_bytes:
        raise ValueError("COO P2P resident bytes differ from their metadata")
    if indices.dtype.kind not in "iu":
        raise TypeError("COO P2P point indices must be integer arrays")
    # The sparse kernel preserves Python/NumPy negative-index semantics, but
    # indices beyond either full stored extent are invalid before prefix cuts.
    if metadata.nnz:
        rows, columns = indices[:, 0], indices[:, 1]
        if (bool(np.any(rows < -metadata.point_shape[0]))
                or bool(np.any(rows >= metadata.point_shape[0]))
                or bool(np.any(columns < -metadata.point_shape[1]))
                or bool(np.any(columns >= metadata.point_shape[1]))):
            raise IndexError("COO P2P point index lies outside its declared point shape")
        if metadata.point_diagonal:
            row_positions = np.where(rows < 0, rows + metadata.point_shape[0], rows)
            column_positions = np.where(
                columns < 0, columns + metadata.point_shape[1], columns,
            )
            if bool(np.any(row_positions != column_positions)):
                raise ValueError("COO P2P point-diagonal metadata contains off-diagonal pairs")
    return BorrowedSparsePointRecord(
        {"type": "sparse", "indices": indices, "values": values}, resource_token,
    )


class SparseElemental:
    """Elemental reader combining existing dense V sectors and sparse P2P.

    ``dense_reader`` continues to provide V2V/V2P/P2V. ``record_reader`` owns
    the explicit P2P record source; no point axes are inferred from time,
    ``usedNp``, or record contents. Returned record arrays may be borrowed and
    remain valid through the caller's read/materialize lifecycle.
    """

    def __init__(self, dense_reader, record_reader, *, selection=None):
        for name in ("block", "describe_block"):
            if not callable(getattr(dense_reader, name, None)):
                raise TypeError(f"dense_reader must expose {name}(...)")
        if (not callable(getattr(record_reader, "describe_record", None))
                or not callable(getattr(record_reader, "read_record", None))):
            raise TypeError("record_reader must expose describe_record and read_record")
        self.dense_reader = dense_reader
        self.record_reader = record_reader
        self.point_diagonal_components = tuple(
            getattr(dense_reader, "point_diagonal_components", ())
        )
        self._point_diagonal_component_set = frozenset(self.point_diagonal_components)
        self.elemental_structures = getattr(dense_reader, "elemental_structures", {})
        self.phase_convention = dense_reader.phase_convention
        self.elemental_weighted = getattr(dense_reader, "elemental_weighted", False)
        self.dilution_weight_scheme = getattr(dense_reader, "dilution_weight_scheme", None)
        self.Lt = integer(dense_reader.Lt, "Lt", 1)
        self.cfg = getattr(dense_reader, "cfg", None)
        self.time_selection = getattr(dense_reader, 'time_selection', None)
        record_selection = getattr(record_reader, 'time_selection', None) if selection is None else normalize_time_selection(selection)
        if record_selection != self.time_selection:
            raise ValueError('sparse records require the same explicitly ordered selection as dense sectors')
        self.time_selections = getattr(dense_reader, 'time_selections', {})
        self.time_selection_identity = getattr(dense_reader, 'time_selection_identity', None)
        self.sampling_time_selection = getattr(dense_reader, 'sampling_time_selection', self.time_selection)
        self.Nt = self.Lt
        self.physical_Lt = getattr(dense_reader, 'physical_Lt', self.Lt)
        self.momenta = getattr(dense_reader, 'momenta', None)

    @property
    def dilution_identity(self):
        return getattr(self.dense_reader, 'dilution_identity', None)

    def dilution_basis(self, component, momentum, time, sector, side):
        provider = getattr(self.dense_reader, 'dilution_basis', None)
        return None if provider is None else provider(component, momentum, time, sector, side)

    def describe_sparse_record(self, component, momentum, t):
        momentum, t = integer(momentum, "momentum", 0), integer(t, "time", 0)
        if t >= self.Lt:
            raise KeyError("sparse P2P request is outside elemental time coverage")
        metadata = self.record_reader.describe_record(component, momentum, t)
        if not isinstance(metadata, SparsePointRecordMetadata):
            raise TypeError("record_reader.describe_record must return SparsePointRecordMetadata")
        point_diagonal_declared = component in self._point_diagonal_component_set
        dense_structure = "general"
        provider = getattr(self.dense_reader, "elemental_structure", None)
        if callable(provider):
            try:
                dense_structure = provider(component, momentum, "p2p")
            except (KeyError, ValueError):
                dense_structure = "general"
        elif isinstance(self.elemental_structures, Mapping):
            dense_structure = self.elemental_structures.get(
                ("p2p", component, momentum), "general",
            )
        dense_structure = normalize_elemental_structure(
            dense_structure, point_diagonal_declared,
        )
        record_structure = normalize_elemental_structure(
            metadata.structure, metadata.point_diagonal,
        )
        structure_order = {"general": 0, "point_diagonal": 1, "diagonal": 2}
        structure = max((dense_structure, record_structure),
                        key=structure_order.__getitem__)
        metadata = replace(
            metadata, structure=structure,
            point_diagonal=structure in {"diagonal", "point_diagonal"},
            selection_identity=(
                repr(("elemental-structure-v1", dense_structure,
                      record_structure, structure, metadata.selection_identity))
                if structure != "general" or dense_structure != "general"
                or record_structure != "general" else metadata.selection_identity
            ),
        )
        if self.elemental_weighted and metadata.kind == "identity":
            for side in ("bar", "field"):
                pool_id = self.point_pool_id(component, momentum, t, side)
                pool = self.sampling_group(pool_id)
                if pool is not None and pool.spatial_sites != pool.sampled_points:
                    raise ValueError("weighted P2P identity requires explicit COO values for non-unit compensation")
        if self.time_selection_identity is not None:
            metadata = replace(metadata, selection_identity=repr((self.time_selection_identity, metadata.selection_identity)))
        if self.elemental_weighted:
            metadata = replace(metadata, selection_identity=repr((
                "single-current-point-weighted-v1", metadata.selection_identity)))
        return metadata

    def describe_block(self, component, momentum, t, sector, *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        if sector != "p2p":
            return self.dense_reader.describe_block(
                component, momentum, t, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
        _reject_sparse_p2p_selection(
            usedNe=usedNe, mode_indices=mode_indices, point_indices=point_indices,
        )
        metadata = self.describe_sparse_record(component, momentum, t)
        selected = metadata.point_shape if usedNp is None else (
            integer(usedNp, "usedNp", 1), integer(usedNp, "usedNp", 1)
        )
        if any(size > extent for size, extent in zip(selected, metadata.point_shape, strict=True)):
            raise ValueError("usedNp exceeds sparse P2P record coverage")
        structure = metadata.structure
        if structure == "general":
            # An unmarked legacy/general record needs no stronger dense-axis
            # proof. In particular, avoid another point-pool lookup here:
            # Meson describes the dagger-aware raw sides immediately after
            # this call and owns that ordering.
            return TensorAxisMetadata(
                ("point_left", "color_left", "point_right", "color_right"),
                (selected[0], 3, selected[1], 3), metadata.value_dtype,
                metadata.data_identity, metadata.selection_identity,
                sparse_layout="coo", point_shape=metadata.point_shape,
                nnz_upper=metadata.nnz, structure="general",
            )
        left_pool = self.point_pool_id(component, momentum, t, "bar")
        right_pool = self.point_pool_id(component, momentum, t, "field")
        if left_pool is not None and right_pool is not None and left_pool != right_pool:
            structure = "general"
        dense_describer = getattr(self.dense_reader, "describe_block", None)
        if callable(dense_describer):
            try:
                dense_metadata = dense_describer(
                    component, momentum, t, "p2p", usedNp=usedNp,
                    point_indices=point_indices,
                )
            except (KeyError, ValueError, NotImplementedError):
                dense_metadata = None
            if dense_metadata is not None:
                provider = getattr(self.dense_reader, "elemental_structure", None)
                dense_declared = "general"
                if callable(provider):
                    try:
                        dense_declared = provider(component, momentum, "p2p")
                    except (KeyError, ValueError):
                        dense_declared = "general"
                elif isinstance(self.elemental_structures, Mapping):
                    dense_declared = self.elemental_structures.get(
                        ("p2p", component, momentum), "general",
                    )
                dense_declared = normalize_elemental_structure(
                    dense_declared,
                    component in self._point_diagonal_component_set,
                )
                if (dense_declared != "general"
                        and getattr(dense_metadata, "structure", "general") == "general"):
                    structure = "general"
        return TensorAxisMetadata(
            ("point_left", "color_left", "point_right", "color_right"),
            (selected[0], 3, selected[1], 3), metadata.value_dtype,
            metadata.data_identity,
            (repr(("selected-elemental-structure-v1", metadata.structure, structure,
                   metadata.selection_identity))
             if structure != metadata.structure else metadata.selection_identity),
            sparse_layout="coo",
            point_shape=metadata.point_shape,
            nnz_upper=metadata.nnz,
            structure=structure,
        )

    def block(self, component, momentum, t, sector, *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        if sector != "p2p":
            return self.dense_reader.block(
                component, momentum, t, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
        _reject_sparse_p2p_selection(
            usedNe=usedNe, mode_indices=mode_indices, point_indices=point_indices,
        )
        metadata = self.describe_sparse_record(component, momentum, t)
        if metadata.backend != "numpy":
            raise NotImplementedError(
                "SparseElemental sparse P2P materialization currently requires CPU NumPy records"
            )
        record = self.record_reader.read_record(component, momentum, t)
        try:
            return validate_sparse_point_record(record, metadata)
        except BaseException as primary_error:
            try:
                self.close_sparse_record(component, momentum, t, record)
            except BaseException as cleanup_error:
                try:
                    primary_error.add_note(
                        f"sparse record cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except Exception:
                    pass
            raise

    def close_sparse_record(self, component, momentum, t, record):
        """Release one borrowed source record after its last contraction use."""
        close = getattr(self.record_reader, "close_record", None)
        if callable(close):
            resource = (record.resource_token if isinstance(record, BorrowedSparsePointRecord)
                        else record)
            close(component, int(momentum), int(t), resource)

    def point_pool_id(self, component, momentum, t, side):
        provider = getattr(self.dense_reader, "point_pool_id", None)
        return None if not callable(provider) else provider(component, momentum, t, side)

    def sampling_group(self, group_id):
        provider = getattr(self.dense_reader, "sampling_group", None)
        return None if not callable(provider) else provider(group_id)


def _reject_sparse_p2p_selection(*, usedNe, mode_indices, point_indices):
    if usedNe is not None or mode_indices is not None or point_indices is not None:
        raise NotImplementedError(
            "sparse P2P records support only the declared usedNp prefix; "
            "Ne, mode-index, and arbitrary point-index selections are unsupported"
        )


__all__ = [
    "BorrowedSparsePointRecord", "SparseElemental", "SparsePointRecordReader",
    "validate_sparse_point_record",
]
