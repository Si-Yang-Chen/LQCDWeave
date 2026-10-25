"""Archivable constants and numerical source preparation descriptions."""
from __future__ import annotations
from dataclasses import dataclass
from math import prod

import numpy as np

from .inputs import TensorAxisMetadata


def _freeze_convention(value):
    """Keep ordered reader metadata independent of mutable source containers."""
    if isinstance(value, tuple):
        try:
            hash(value)
        except TypeError:
            pass
        else:
            return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_convention(item) for item in value)
    return value


@dataclass(frozen=True)
class SourceReadRequest:
    """Adapter-owned read locator, separate from actual storage identity.

    reader_role names an injected reader. Arguments and selections are the
    already resolved reader API request; restoring it requires no graph.
    """

    reader_role: str
    family: str
    arguments: tuple
    selection: tuple
    metadata: TensorAxisMetadata
    prefetch_host_bytes: int | None = None
    aliases: tuple[SourceReadRequest, ...] = ()
    conventions: tuple[tuple[str, object], ...] = ()
    coordinate_arguments: tuple[tuple[int, str, int, int | None], ...] = ()
    coordinate_tables: tuple[tuple[int, str, tuple[int | None, ...]], ...] = ()

    def __post_init__(self):
        if not isinstance(self.reader_role, str) or not self.reader_role:
            raise ValueError("read request requires a reader role")
        if self.family not in {"propagator", "elemental"}:
            raise ValueError("unknown read request family")
        if not isinstance(self.metadata, TensorAxisMetadata) or self.metadata.sparse_layout != "dense":
            raise TypeError("dense read request requires dense TensorAxisMetadata")
        selection = tuple(self.selection)
        if (any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                for item in selection) or len({item[0] for item in selection}) != len(selection)):
            raise ValueError("read selection must contain unique keyword pairs")
        if self.prefetch_host_bytes is not None and (
                type(self.prefetch_host_bytes) is not int or self.prefetch_host_bytes < 0):
            raise ValueError("prefetch size must be nonnegative integer bytes or None")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "selection", selection)
        aliases = tuple(self.aliases)
        if any(not isinstance(item, SourceReadRequest) or item.aliases for item in aliases):
            raise TypeError("read aliases must be flat SourceReadRequest records")
        object.__setattr__(self, "aliases", aliases)
        conventions = tuple(self.conventions)
        if any(not isinstance(item, tuple) or len(item) != 2 or item[0] not in
               {'time_convention', 'phase_convention', 'path_boundary', 'momenta', 'momentum_list', 'time_selection_identity', 'elemental_weighted', 'dilution_identity', 'glueball_definition', 'propagator_time_view', 'read_context_axes'} for item in conventions):
            raise ValueError("read conventions must name reader metadata conventions")
        object.__setattr__(self, "conventions",
                           tuple((name, _freeze_convention(value)) for name,value in conventions))
        coordinates = tuple(self.coordinate_arguments)
        if any(type(index) is not int or not 0 <= index < len(self.arguments)
               or not isinstance(role, str) or not role or type(offset) is not int
               or (period is not None and (type(period) is not int or period <= 0))
               for index, role, offset, period in coordinates):
            raise ValueError('invalid read coordinate parameter')
        if len({entry[0] for entry in coordinates}) != len(coordinates):
            raise ValueError('duplicate read coordinate argument')
        object.__setattr__(self, 'coordinate_arguments', coordinates)
        tables = tuple((index, role, tuple(values)) for index, role, values in self.coordinate_tables)
        if any(type(index) is not int or not 0 <= index < len(self.arguments)
               or not isinstance(role, str) or not role or not values
               or any(value is not None and (type(value) is not int or value < 0) for value in values)
               for index, role, values in tables):
            raise ValueError('invalid read coordinate lookup table')
        indices = [item[0] for item in (*coordinates, *tables)]
        if len(indices) != len(set(indices)):
            raise ValueError('duplicate read coordinate argument or table')
        object.__setattr__(self, 'coordinate_tables', tables)

    def at(self, values):
        """Substitute declared roles only; equal numeric times imply no alias."""
        from dataclasses import replace
        values = {str(role): value for role,value in values.items()}
        arguments = list(self.arguments)
        for index, role, offset, period in self.coordinate_arguments:
            if role in values:
                value = int(values[role]) + offset
                arguments[index] = value if period is None else value % period
        for index, role, table in self.coordinate_tables:
            if role in values:
                slot = values[role]
                if (isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
                        or not 0 <= int(slot) < len(table)):
                    raise ValueError('coordinate slot is outside saved lookup table')
                mapped = table[int(slot)]
                if mapped is None:
                    raise KeyError('physical endpoint is missing from saved time selection')
                arguments[index] = mapped
        return replace(self, arguments=tuple(arguments),
                       aliases=tuple(alias.at(values) for alias in self.aliases))


@dataclass(frozen=True)
class ArrayConstant:
    """An immutable numerical constant, never a production input array."""

    shape: tuple[int, ...]
    dtype: str
    data: str

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("constant dimensions must be nonnegative integers")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biufc":
            raise TypeError("constant dtype must be numeric")
        try:
            raw = bytes.fromhex(self.data)
        except (TypeError, ValueError) as exc:
            raise ValueError("constant data must be hexadecimal bytes") from exc
        if len(raw) != prod(shape) * dtype.itemsize:
            raise ValueError("constant byte extent differs from shape and dtype")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.str)

    @classmethod
    def from_array(cls, value):
        value = np.asarray(value)
        return cls(tuple(value.shape), value.dtype.str, value.tobytes(order="C").hex())

    def array(self):
        """Return a read-only view backed by immutable bytes."""
        return np.frombuffer(bytes.fromhex(self.data), dtype=self.dtype).reshape(self.shape)


@dataclass(frozen=True)
class TensorPreparation:
    """Fixed preparation; inputs name raw blocks or stack member tensors.

    spin_spatial constants are spin and, optionally, selected left/right
    profile rows plus their coefficients. Operator weights belong to outputs.
    raw_index selects a coordinate lane from a shared physical slab before
    casting, conjugation and the declared spatial transforms.
    """

    kind: str
    inputs: tuple[str, ...] = ()
    constants: tuple[ArrayConstant, ...] = ()
    conjugate: bool = False
    permutation: tuple[int, ...] = ()
    configuration: str | None = None
    spatial_structure: str = "general"
    compute_dtype: str | None = None
    raw_index: int | None = None
    spin_columns: tuple[int, ...] = ()
    index_permutations: tuple[tuple[int, tuple[int, ...]], ...] = ()
    spatial_only: bool = False
    # (constant position, raw read argument position, ordered constant rows).
    # Selected offline when a scalar template is bound to actual coordinates.
    coordinate_constants: tuple = ()
    # Logical spatial axis pairs and matched local positions after permutation.
    spatial_support: tuple = ()

    def __post_init__(self):
        tables = tuple((position, argument, tuple(rows))
                       for position, argument, rows in self.coordinate_constants)
        if tables and (self.kind != 'spin_spatial' or any(
                type(position) is not int or not 0 <= position < len(self.constants)
                or type(argument) is not int or argument < 0 or not rows
                or any(row is not None and type(row) is not ArrayConstant for row in rows)
                for position, argument, rows in tables)):
            raise ValueError('coordinate constants require spin-spatial constant rows')
        object.__setattr__(self, 'coordinate_constants', tables)
        support = tuple((a, b, tuple(left), tuple(right))
                        for a, b, left, right in self.spatial_support)
        if support and (self.kind != 'spin_spatial' or self.spin_columns
                        or any(type(a) is not int or type(b) is not int or a < 0 or b < 0
                               or a == b or len(left) != len(right)
                               or any(type(i) is not int or i < 0 for i in left + right)
                               for a, b, left, right in support)):
            raise ValueError('spatial support requires ordered spin-spatial index matches')
        object.__setattr__(self, 'spatial_support', support)
        if self.raw_index is not None and (
                type(self.raw_index) is not int or self.raw_index < 0
                or self.kind not in {"identity", "spin_spatial"}):
            raise ValueError("raw_index requires a nonnegative leaf slab index")
        if self.compute_dtype is not None:
            dtype = np.dtype(self.compute_dtype)
            if dtype not in tuple(map(np.dtype, ('float32','float64','complex64','complex128'))):
                raise TypeError('unsupported preparation computation dtype')
            object.__setattr__(self, 'compute_dtype', dtype.name)
        columns = tuple(self.spin_columns)
        if (type(self.spatial_only) is not bool or self.spatial_only and
                (self.kind != 'spin_spatial' or columns or self.spatial_structure != 'general')):
            raise ValueError('spatial-only output requires an uncompressed general spin-spatial recipe')
        if columns and (self.kind != 'spin_spatial' or len(columns) != 4
                        or any(type(i) is not int for i in columns)
                        or set(columns) != set(range(4))):
            raise ValueError('compressed spin requires an exact four-spin permutation')
        indices = tuple((axis, tuple(order)) for axis, order in self.index_permutations)
        if indices and (self.kind != 'identity' or len({a for a, _ in indices}) != len(indices)
                        or any(type(axis) is not int or axis < 0 or len(order) != 4
                               or any(type(i) is not int for i in order)
                               or set(order) != set(range(4)) for axis, order in indices)):
            raise ValueError('spin index permutations require distinct identity axes and four-spin orders')
        object.__setattr__(self, 'spin_columns', columns)
        object.__setattr__(self, 'index_permutations', indices)
        inputs, constants, permutation = tuple(self.inputs), tuple(self.constants), tuple(self.permutation)
        if self.kind not in {"identity", "spin_spatial", "stack", "constant", "linear_combination"}:
            raise ValueError("unknown tensor preparation kind")
        if self.spatial_structure not in {"general", "diagonal", "point_diagonal"}:
            raise ValueError("unknown elemental spatial structure")
        if self.spatial_structure != "general" and self.kind != "spin_spatial":
            raise ValueError("only spin preparation accepts elemental structure")
        if any(not isinstance(key, str) or not key for key in inputs):
            raise ValueError("preparation inputs must be nonempty IDs")
        if any(not isinstance(value, ArrayConstant) for value in constants):
            raise TypeError("preparation constants must be ArrayConstant records")
        if type(self.conjugate) is not bool:
            raise TypeError("conjugate must be bool")
        if (any(type(axis) is not int for axis in permutation)
                or set(permutation) != set(range(len(permutation)))):
            raise ValueError("preparation permutation must contain each axis once")
        if self.kind == "identity" and (len(inputs) != 1 or constants):
            raise ValueError("identity preparation needs one block and no constants")
        if self.kind == "constant" and (inputs or len(constants) != 1):
            raise ValueError("constant preparation needs one constant and no inputs")
        if self.kind == "stack" and (not inputs or constants):
            raise ValueError("stack preparation needs member tensors and no constants")
        if self.kind == 'linear_combination' and (
                not inputs or len(constants) != 1 or constants[0].shape != (len(inputs),)):
            raise ValueError('linear combination requires members and their ordered weights')
        if self.kind == "spin_spatial" and (len(inputs) != 1 or len(constants) not in (1, 4)):
            raise ValueError("spin preparation needs one block, spin and optional profile constants")
        if self.kind == "spin_spatial" and constants[0].shape != (4, 4):
            raise ValueError("spin preparation requires a 4 by 4 spin matrix")
        if self.conjugate and self.kind not in {"identity", "spin_spatial"}:
            raise ValueError("only identity or spin preparation accepts conjugation")
        if self.kind != "spin_spatial" and (permutation or self.configuration is not None):
            raise ValueError("only spin preparation accepts spatial transforms")
        if columns:
            spin = constants[0].array()
            rows, actual_columns = np.nonzero(spin)
            if (not np.array_equal(rows, np.arange(4))
                    or tuple(map(int, actual_columns)) != columns
                    or not np.all(np.isfinite(spin))):
                raise ValueError('compressed spin columns must describe the exact finite constant support')
        if self.kind == "spin_spatial" and len(constants) == 4:
            left, right, weights = constants[1:]
            if (len(left.shape) != 2 or len(right.shape) != 2
                    or left.shape[0] != right.shape[0] or weights.shape != (left.shape[0],)
                    or not isinstance(self.configuration, str)):
                raise ValueError("profile preparation requires compatible rows, weights and cfg")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "permutation", permutation)

    def constants_at(self, arguments):
        """Select immutable definition rows using an already bound raw read."""
        constants = list(self.constants)
        for position, argument, rows in self.coordinate_constants:
            row = rows[arguments[argument]]
            if row is None:
                raise ValueError('profile does not cover the requested coordinate')
            constants[position] = row
        return tuple(constants)
