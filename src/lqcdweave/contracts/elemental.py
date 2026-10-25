"""Elemental definitions, independent of production and file formats."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol

import numpy as np


TEMPORAL_POINT_PATH_SCHEMA = "lattice.elemental.temporal-point-path/v1"


def integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def normalize_elemental_structure(structure, point_diagonal=False):
    """Normalize the three producer-declared elemental support structures.

    ``point_diagonal`` is the pre-structure compatibility flag.  A true flag
    upgrades the default ``general`` declaration to ``point_diagonal`` while
    an explicit stronger ``diagonal`` declaration remains intact.
    """
    if type(point_diagonal) is not bool:
        raise TypeError("point_diagonal must be bool")
    if not isinstance(structure, str):
        raise TypeError("elemental structure must be a string")
    if structure not in {"general", "diagonal", "point_diagonal"}:
        raise ValueError("elemental structure must be 'general', 'diagonal', or 'point_diagonal'")
    if point_diagonal and structure == "general":
        return "point_diagonal"
    return structure


def normalize_momenta(momenta):
    """Freeze an ordered XYZ sequence or mapping values, never its keys.

    Mapping values may be integer triples or legacy three-integer strings.
    Their iteration positions are the shared production/operator storage indices.
    """
    values = momenta.values() if isinstance(momenta, Mapping) else momenta
    normalized = []
    try:
        for value in values:
            if isinstance(value, str):
                value = tuple(int(n) for n in value.split())
            momentum = tuple(integer(n, "momentum coordinate") for n in value)
            if len(momentum) != 3:
                raise ValueError("momentum requires an integer XYZ triple")
            normalized.append(momentum)
    except TypeError as exc:
        raise ValueError("momenta must contain integer XYZ triples") from exc
    result = tuple(normalized)
    if not result or len(set(result)) != len(result):
        raise ValueError("nonempty distinct XYZ momenta are required")
    return result


@dataclass(frozen=True)
class GaugePath:
    """Ordered steps 0..5 = +x,+y,+z,-x,-y,-z; 6,7 = +t,-t.

    Integer IDs use length-first base eight. Use explicit steps when importing
    a different path numbering scheme.
    """

    steps: tuple = ()
    NAMES = ("+x", "+y", "+z", "-x", "-y", "-z", "+t", "-t")
    AXES = (0, 1, 2, 0, 1, 2, 3, 3)
    SIGNS = (1, 1, 1, -1, -1, -1, 1, -1)
    REVERSE = (3, 4, 5, 0, 1, 2, 7, 6)

    def __post_init__(self):
        steps = tuple(self.NAMES.index(s) if isinstance(s, str) else integer(s, "step", 0)
                      for s in self.steps)
        if any(s >= 8 for s in steps):
            raise ValueError("path directions must lie in 0..7")
        object.__setattr__(self, "steps", steps)

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Integral):
            return cls.from_index(value)
        return cls(tuple(value))

    @classmethod
    def from_index(cls, index):
        index = integer(index, "path index", 0)
        steps = []
        while index:
            index, digit = divmod(index - 1, 8)
            steps.append(digit)
        return cls(tuple(reversed(steps)))

    @property
    def idx(self):
        value = 0
        for step in self.steps:
            value = 8 * value + step + 1
        return value

    @property
    def displacement(self):
        offset = [0] * 4
        for step in self.steps:
            offset[self.AXES[step]] += self.SIGNS[step]
        return tuple(offset)

    @property
    def time_bounds(self):
        t = lo = hi = 0
        for step in self.steps:
            if self.AXES[step] == 3:
                t += self.SIGNS[step]
            lo, hi = min(lo, t), max(hi, t)
        return lo, hi

    def conjugate(self):
        return GaugePath(tuple(self.REVERSE[s] for s in reversed(self.steps)))


@dataclass(frozen=True)
class ElementalComponent:
    """A stored component, with explicit endpoint times relative to its anchor."""

    key: object
    time_offsets: tuple = (0, 0)

    def __post_init__(self):
        hash(self.key)
        offsets = tuple(integer(x, "time offset") for x in self.time_offsets)
        if len(offsets) != 2:
            raise ValueError("two endpoint time offsets are required")
        if isinstance(self.key, GaugePath) and offsets != (0, self.key.displacement[3]):
            raise ValueError("path endpoint times must agree with its displacement; use ElementalComponent.path")
        object.__setattr__(self, "time_offsets", offsets)

    @classmethod
    def path(cls, path):
        path = GaugePath.coerce(path)
        return cls(path, (0, path.displacement[3]))


    @classmethod
    def catalog(cls, name, version, number):
        from .elemental_catalog import catalog_key
        return cls(catalog_key(name, version, number))


def spatial_projection_rows(rows):
    """Parse the existing consecutive-row spatial path/coefficient contract.

    Preserve term order and duplicates; a numerical consumer may accumulate
    duplicate paths. This describes coefficients without producing any data.
    """
    if getattr(rows, "catalog", None) is not None:
        return tuple(terms for _, terms in elemental_operator_rows(rows))
    rows = tuple(rows)
    if not rows or any(integer(row["id"], "projection row", 0) != position
                       for position, row in enumerate(rows)):
        raise ValueError("consecutive nonempty projection rows are required")
    parsed = []
    for row in rows:
        terms = []
        for term in row["terms"]:
            path = GaugePath(tuple(term["path"]))
            coefficient = complex(*term["coefficient"])
            if any(step >= 6 for step in path.steps) or not np.isfinite(coefficient):
                raise ValueError("projection rows require finite coefficients and spatial paths")
            terms.append((path, coefficient))
        if not terms:
            raise ValueError("empty projection row")
        parsed.append(tuple(terms))
    return tuple(parsed)


@dataclass
class SparsePointBlock:
    """Color matrices for valid point pairs; no dense Np x Np allocation.

    Arrays can borrow generator storage. to_dense() explicitly allocates.
    structure is a producer support proof in the ordered spatial basis;
    point_diagonal is its legacy compatibility view.
    """

    rows: object
    columns: object
    values: object
    point_shape: tuple
    point_diagonal: bool = False
    structure: str = "general"

    def __post_init__(self):
        structure = normalize_elemental_structure(self.structure, self.point_diagonal)
        shape = tuple(self.point_shape)
        if len(shape) != 2:
            raise ValueError("sparse point block requires two point extents")
        if structure != "general" and shape[0] != shape[1]:
            raise ValueError("diagonal sparse point blocks require equal point extents")
        self.point_shape = shape
        self.structure = structure
        self.point_diagonal = structure in {"diagonal", "point_diagonal"}

    def to_dense(self, *, array_module=None):
        xp = array_module
        if xp is None:
            xp = np
            if type(self.values).__module__.split(".", 1)[0] == "cupy":
                import cupy as xp
        result = xp.zeros((self.point_shape[0], 3, self.point_shape[1], 3), self.values.dtype)
        pairs = self.rows * self.point_shape[1] + self.columns
        if int(xp.unique(pairs).size) != len(self.rows):
            raise ValueError("duplicate point pairs cannot be densified without an accumulation policy")
        result[self.rows, :, self.columns, :] = self.values
        return result

    def copy(self):
        return SparsePointBlock(self.rows.copy(), self.columns.copy(), self.values.copy(), self.point_shape,
                                self.point_diagonal, self.structure)


@dataclass(frozen=True)
class SparsePointRecordMetadata:
    """Header-only description of one legacy identity/COO P2P record.

    ``record_bytes`` is the exact resident record extent read by the adapter;
    it is zero for the symbolic identity record. ``nnz`` is the exact number
    of COO rows, or the number of implicit diagonal pairs for identity.
    point_diagonal guarantees COO row and column labels are equal. Identity
    storage describes values only; support structure must be declared explicitly.
    """

    kind: str
    point_shape: tuple[int, int]
    nnz: int
    value_dtype: object
    index_dtype: object
    record_bytes: int
    data_identity: str | None
    selection_identity: str
    backend: str = "numpy"
    point_diagonal: bool = False
    structure: str = "general"

    def __post_init__(self):
        if type(self.point_diagonal) is not bool:
            raise TypeError("point_diagonal must be bool")
        structure = normalize_elemental_structure(self.structure, self.point_diagonal)
        if self.kind not in {"identity", "sparse"}:
            raise ValueError("sparse point record kind must be 'identity' or 'sparse'")
        shape = tuple(self.point_shape)
        if len(shape) != 2 or any(
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, Integral) or int(size) < 1
                for size in shape):
            raise ValueError("sparse point record needs two positive point extents")
        nnz = integer(self.nnz, "sparse record nnz", 0)
        value_dtype, index_dtype = np.dtype(self.value_dtype), np.dtype(self.index_dtype)
        if value_dtype.kind not in "fc" or index_dtype.kind not in "iu":
            raise TypeError("sparse record value/index dtypes must be floating/complex and integer")
        record_bytes = integer(self.record_bytes, "sparse record bytes", 0)
        if self.kind == "identity":
            if record_bytes != 0 or nnz != min(shape):
                raise ValueError("identity records have no buffers and cover every diagonal point")
            if structure != "general" and shape[0] != shape[1]:
                raise ValueError("rectangular identity records cannot declare a diagonal structure")
        if structure != "general" and shape[0] != shape[1]:
            raise ValueError("point-diagonal records require equal ordered point extents")
        if self.kind == "sparse" and record_bytes != nnz * (2 * index_dtype.itemsize + 9 * value_dtype.itemsize):
            raise ValueError("COO record byte extent disagrees with its exact nnz/dtypes")
        if self.data_identity is not None and (
                not isinstance(self.data_identity, str) or not self.data_identity):
            raise ValueError("data_identity must be a nonempty string or None")
        if not isinstance(self.selection_identity, str) or not self.selection_identity:
            raise ValueError("selection_identity must be a nonempty string")
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError("sparse point record backend must be 'numpy' or 'cupy'")
        object.__setattr__(self, "point_shape", tuple(int(value) for value in shape))
        object.__setattr__(self, "nnz", nnz)
        object.__setattr__(self, "value_dtype", value_dtype)
        object.__setattr__(self, "index_dtype", index_dtype)
        object.__setattr__(self, "record_bytes", record_bytes)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "point_diagonal",
                           structure in {"diagonal", "point_diagonal"})


__all__ = [
    "TEMPORAL_POINT_PATH_SCHEMA", "GaugePath", "ElementalComponent", "SparsePointBlock",
    "SparsePointRecordMetadata", "ElementalReader", "BaryonComponent",
    "integer", "normalize_elemental_structure",
]


class ElementalReader(Protocol):
    """Return one component/momentum/time block, or raise for missing coverage.

    Readers advertise phase_convention ('midpoint', 'anchor', or 'none') and
    Lt, the execution extent. Explicit compact readers use Lt=Nt and consume
    only slot coordinates; time_selection retains the ordered physical mapping
    for binding and reconstruction. Unmarked full-time readers keep physical
    coordinate semantics. A returned block may borrow reader buffers; consumers must finish or
    copy it before the next read.

    Sparse readers that physically filter their COO records may additionally
    expose ``describe_selected_sparse_record(component, momentum, t, *,
    usedNp)``. Its exact point extents, nnz, bytes and selection identity must
    describe the same record returned by ``block(..., usedNp=usedNp)``.
    Readers retaining full raw records use ``describe_sparse_record`` instead.
    """

    phase_convention: str
    Lt: int

    def block(self, component: ElementalComponent, momentum: int, t: int, sector: str): ...

    def describe_block(self, component: ElementalComponent, momentum: int, t: int,
                       sector: str, *, usedNe=None, mode_indices=None,
                       usedNp=None, point_indices=None):
        """Return axis/shape/dtype/source metadata without reading block values."""
        ...

    def describe_slab(self, component: ElementalComponent, momentum: int,
                      times_selected, sector: str, *, usedNe=None,
                      mode_indices=None, usedNp=None, point_indices=None):
        """Describe an ordered time slab without loading numerical values.

        Metadata names a leading ``coordinate_batch`` axis. Implementations
        preserve requested time order and may expose point-pool identity per
        batch entry when it varies with physical time.
        """
        ...

    def read_slab(self, component: ElementalComponent, momentum: int,
                  times_selected, sector: str, *, usedNe=None,
                  mode_indices=None, usedNp=None, point_indices=None):
        """Return one ordered time slab; the result may borrow reader storage."""
        ...

    def point_pool_id(self, component: ElementalComponent, momentum: int, t: int,
                      side: str) -> str | None:
        """Return an explicitly registered bar/field sampling-pool identity."""
        ...

    def sampling_group(self, group_id: str):
        """Return contracts.spatial_sampling.SamplingPoolSpec or None."""
        ...


@dataclass(frozen=True)
class BaryonComponent:
    """Ordered three-leg spatial component, independent of physical flavor.

    ``flavors`` is accepted for compatibility with legacy producers. It does
    not participate in component equality/hash or the new operator definition;
    physical flavor belongs to the symbolic Wick and propagator bindings.
    """

    key: object
    flavors: tuple = field(default=("q", "q", "q"), compare=False, repr=False)

    def __post_init__(self):
        hash(self.key)
        flavors = tuple(str(f) for f in self.flavors)
        if len(flavors) != 3 or any(not f for f in flavors):
            raise ValueError("three nonempty quark flavor labels are required")
        object.__setattr__(self, "flavors", flavors)

    @classmethod
    def paths(cls, paths=((), (), ()), *, flavors=("q", "q", "q")):
        paths = tuple(GaugePath.coerce(p) for p in paths)
        if len(paths) != 3 or any(any(s >= 6 for s in p.steps) for p in paths):
            raise ValueError("three spatial paths are required")
        return cls(paths, flavors)

    @classmethod
    def catalog(cls, name, version, number):
        from .elemental_catalog import catalog_key
        return cls(catalog_key(name, version, number))


def elemental_operator_rows(table, *, rank=2):
    """Normalize an explicit table into ordered component/coefficient rows.

    Flat records contain component/path/coefficient; catalog rows contain
    id/terms with the same path/coefficient records. No paths are enumerated.
    """
    if rank not in (2, 3):
        raise ValueError("elemental rank must be 2 or 3")
    catalog = getattr(table, "catalog", None)
    if catalog is not None and catalog.rank != rank:
        raise ValueError("elemental catalog rank differs from the generator")
    records = tuple(table)
    if not records:
        raise ValueError("operator_table must be nonempty")
    grouped = {}
    for record in records:
        component = record.get("component", record.get("id"))
        if component is None:
            raise ValueError("operator table needs component or id")
        hash(component)
        entries = record.get("terms", (record,))
        if not entries:
            raise ValueError("operator table rows need terms")
        row = grouped.setdefault(component, [])
        for term in entries:
            path = term.get("path", term.get("paths"))
            if path is None:
                raise ValueError("operator table term needs path")
            if rank == 3:
                path = tuple(GaugePath.coerce(p) for p in path)
                if len(path) != 3:
                    raise ValueError("three leg paths required")
            else:
                path = GaugePath.coerce(path)
            if "coefficient" not in term:
                raise ValueError("operator table term needs coefficient")
            value = term["coefficient"]
            coefficient = complex(*value) if isinstance(value,(list,tuple)) else complex(value)
            if not np.isfinite(coefficient):
                raise ValueError("finite coefficient required")
            row.append((path,coefficient))
    parsed = tuple((component,tuple(terms)) for component,terms in grouped.items())
    if catalog is not None:
        for component, terms in parsed:
            key = getattr(component, "key", ())
            if (not isinstance(key, tuple) or len(key) != 4
                    or component != catalog.component(key[-1]) or terms != catalog.rows[key[-1]]):
                raise ValueError("operator table conflicts with its versioned elemental catalog")
    return parsed
