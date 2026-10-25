"""Immutable scene descriptors; no planner or numerical-kernel imports."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from numbers import Integral
from typing import Any, Mapping
import json
import math
import numpy as np
from .ir import TensorSpec, ContractionTerm
from .equations import parse_term
from .indices import AxisPattern, axis_ids, parse_axis_pattern


def _validate_dense_terms(terms, specs):
    """Reuse only a native equation's already checked ordered dimensions."""
    if any(type(spec) is not TensorSpec for spec in specs.values()):
        for index, term in enumerate(terms):
            parse_term(term, specs, index)
        return
    checked = set()
    for index, term in enumerate(terms):
        if (type(term) is not ContractionTerm or type(term.subscripts) is not AxisPattern
                or any(type(key) is not str for key in term.inputs)):
            parse_term(term, specs, index)
            continue
        try:
            shapes = tuple(specs[key].shape for key in term.inputs)
        except KeyError:
            # Preserve the original error text and current term/operand index.
            parse_term(term, specs, index)
            continue
        # terms holds every immutable equation alive for this construction.
        # Input keys and coefficients are not merged or kept in this proof.
        layout = (id(term.subscripts), shapes)
        if layout not in checked:
            parse_term(term, specs, index)
            if len(checked) < 128:
                checked.add(layout)


@dataclass(frozen=True)
class SourceSpec:
    """A cfg-independent request and its shareable backing region."""

    region: str
    region_bytes: int
    staging_bytes: int = 0
    transfer_bytes: int = 0
    host_region_bytes: int = 0
    snapshot_bytes: int | None = None
    metadata: Any = None
    read_bytes: int | None = None

    def __post_init__(self):
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("source region must be a stable nonempty string")
        for name in ("region_bytes", "staging_bytes", "transfer_bytes", "host_region_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.snapshot_bytes is not None and (
            isinstance(self.snapshot_bytes, bool) or not isinstance(self.snapshot_bytes, int) or self.snapshot_bytes < 0
        ):
            raise ValueError("snapshot_bytes must be non-negative or None")
        if self.read_bytes is not None and (
            isinstance(self.read_bytes, bool) or not isinstance(self.read_bytes, int) or self.read_bytes < 0
        ):
            raise ValueError("read_bytes must be non-negative or None")
        try:
            encoded = json.dumps(_plain_json(self.metadata), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("source metadata must be JSON-compatible and contain no arrays") from exc
        object.__setattr__(self, "metadata", _freeze_json(json.loads(encoded)))


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value):
    if isinstance(value, MappingABC):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ResourceTransition:
    """A complete owned allocation move; no numerical I/O is implied."""
    domain: str
    source: str
    target: str
    nbytes: int
    workspace_host_bytes: int = 0
    workspace_device_bytes: int = 0

    def __post_init__(self):
        if self.domain not in ("tensor", "block"):
            raise ValueError("resource domain must be tensor or block")
        if self.source not in ("host", "device", "disk") or self.target not in ("host", "device", "disk", "absent"):
            raise ValueError("unknown resource tier")
        for name in ("nbytes", "workspace_host_bytes", "workspace_device_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative integer bytes")


@dataclass(frozen=True)
class Action:
    kind: str
    tensor: str
    operands: tuple[str, ...] = ()
    equation: AxisPattern | None = None
    seconds: float = 0.0
    reused: bool = False
    used_np: int = 0
    join_id: str = ""
    left_axes: tuple[int, int, int] = (-3, -2, -1)
    right_axes: tuple[int, int, int] = (0, 1, 2)
    same_point: bool = False
    transpose_bridge: bool = False
    term_index: int = -1
    positions: tuple[int, ...] = ()
    kernel_path: tuple[tuple[int, ...], ...] = ()
    used_np_right: int | None = None
    sparse_spin: tuple[tuple[int, int, int, complex], ...] | None = field(default=None, kw_only=True)
    core: object = field(default=None, kw_only=True)
    resource: "ResourceTransition | None" = field(default=None, kw_only=True)
    wide_accumulation: bool = field(default=False, kw_only=True)
    kernel_description: object = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.wide_accumulation and (type(self.wide_accumulation) is not bool or self.kind != 'contract'):
            raise ValueError('wide accumulation requires a dense contraction')
        if (type(self) is Action and type(self.operands) is tuple
                and (self.equation is None or type(self.equation) is AxisPattern)
                and self.resource is None and self.sparse_spin is None):
            return
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "equation", None if self.equation in (None, "") else parse_axis_pattern(self.equation))
        if self.resource is not None and not isinstance(self.resource, ResourceTransition):
            raise TypeError("resource must be ResourceTransition or None")
        object.__setattr__(self, "sparse_spin", _sparse_spin_entries(self.sparse_spin))


@dataclass(frozen=True)
class Factor:
    tensor: str
    labels: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "labels", axis_ids(self.labels))


@dataclass(frozen=True)
class PendingTerm:
    factors: tuple[Factor, ...]
    output: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "output", axis_ids(self.output))


@dataclass(frozen=True)
class ContractionProblem:
    specs: Mapping[str, TensorSpec]
    terms: tuple[ContractionTerm, ...]
    sources: Mapping[str, SourceSpec]
    sparse_bridges: tuple["SparseBridgeSpec", ...] = ()
    baryon_joins: tuple["BaryonJoinSpec", ...] = ()
    backend: str = "numpy"
    single_read_regions: bool = False
    meson_cores: tuple = field(default=(), kw_only=True)
    # Terms include whole-vertex factors as ordinary operands with retained
    # row axes; the planner seeds each scene with its own pair path.
    factorized_vertices: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        specs = dict(self.specs)
        terms = tuple(self.terms)
        object.__setattr__(self, '_has_sum_barriers', any(getattr(term, 'sum_barriers', ()) for term in terms))
        if not terms:
            raise ValueError("at least one scene term is required")
        if self.backend not in ("numpy", "cupy"):
            raise ValueError("backend must be numpy or cupy")
        if not isinstance(self.single_read_regions, bool):
            raise TypeError("single_read_regions must be bool")
        if type(self.factorized_vertices) is not bool:
            raise TypeError("factorized_vertices must be bool")
        for key, spec in specs.items():
            if not isinstance(key, str) or not key or not isinstance(spec, TensorSpec):
                raise TypeError("contraction input keys and TensorSpec entries are required")
            if spec.dtype.kind not in "biufc":
                raise TypeError("contraction inputs must have numeric dtypes")
        from .meson_core import MesonWholeCore, MesonWholeKernel
        cores = tuple(self.meson_cores)
        if any(type(core) is not MesonWholeCore for core in cores):
            raise TypeError('meson_cores must contain whole meson descriptions')
        core_keys = {core.key for core in cores}
        if len(core_keys) != len(cores) or not core_keys <= specs.keys() or core_keys & self.sources.keys():
            raise ValueError('whole meson cores require distinct unallocated logical tensor identities')
        if any(not set(core.inputs) <= self.sources.keys() for core in cores):
            raise ValueError('whole meson data dependencies must be declared sources')
        if any(len(specs[core.key].shape) not in (5, 6)
               or specs[core.key].shape[-5] != len(core.output_rows) for core in cores):
            raise ValueError('whole meson row identities must match the independent output axis')
        for term in terms:
            if term.kernel is not None and (type(term.kernel) is not MesonWholeKernel
                    or not set(term.kernel.cores) <= core_keys
                    or not set(term.kernel.cores) <= set(term.inputs)
                    or not set(term.kernel.inputs) <= self.sources.keys()):
                raise ValueError('whole meson kernels require their complete logical and numerical bindings')
        if not set(specs) <= set(self.sources) | core_keys:
            raise ValueError("each input TensorSpec requires one SourceSpec")
        for key, spec in specs.items():
            if key in core_keys:
                continue
            declared_snapshot = self.sources[key].snapshot_bytes
            if declared_snapshot is not None and declared_snapshot < math.prod(spec.shape) * spec.dtype.itemsize:
                raise ValueError("dense input snapshot reserve cannot be smaller than its TensorSpec")
        regions = {}
        for source in self.sources.values():
            size = (source.region_bytes, source.host_region_bytes)
            if source.region in regions and regions[source.region] != size:
                raise ValueError("one backing region must have consistent byte extents")
            regions[source.region] = size
        for bridge in self.sparse_bridges:
            if bridge.left not in specs or bridge.right not in specs or bridge.record not in self.sources:
                raise ValueError("sparse bridge inputs and record source must be declared")
            if bridge.scene is not None and (bridge.scene < 0 or bridge.scene >= len(terms)):
                raise ValueError("sparse bridge scene index is invalid")
            left, right = specs[bridge.left], specs[bridge.right]
            left_axes, right_axes = bridge.left_axes, bridge.right_axes
            record_left, record_right = bridge.point_shape
            input_left, input_right = ((record_right, record_left)
                                       if bridge.transpose_bridge
                                       else (record_left, record_right))
            if (
                any(not -len(left.shape) <= axis < len(left.shape) for axis in left_axes)
                or any(not -len(right.shape) <= axis < len(right.shape) for axis in right_axes)
                or left.shape[left_axes[1] % len(left.shape)] != input_left
                or right.shape[right_axes[1] % len(right.shape)] != input_right
                or left.shape[left_axes[2] % len(left.shape)] != right.shape[right_axes[2] % len(right.shape)]
                or (bridge.same_point and record_left != record_right)
            ):
                raise ValueError("sparse bridge axes disagree with declared TensorSpec")
            source_metadata = self.sources[bridge.record].metadata
            if (isinstance(source_metadata, MappingABC)
                    and source_metadata.get("family") == "p2p-coordinate-batch"):
                _validate_coordinate_batch_bridge(source_metadata, bridge, specs)
        marked_sources = {
            key for key, source in self.sources.items()
            if isinstance(source.metadata, MappingABC)
            and source.metadata.get("family") == "p2p-coordinate-batch"
        }
        referenced_marked_sources = {
            bridge.record for bridge in self.sparse_bridges
            if isinstance(self.sources[bridge.record].metadata, MappingABC)
            and self.sources[bridge.record].metadata.get("family") == "p2p-coordinate-batch"
        }
        if marked_sources != referenced_marked_sources:
            raise ValueError("coordinate-batched sparse metadata must be attached to a declared bridge source")
        if self.baryon_joins:
            parsed_terms = tuple(parse_term(term, specs, index) for index, term in enumerate(terms))
        else:
            # Only baryon joins consume ParsedTerm objects. Native dense terms
            # can share a checked layout without retaining any parsed objects.
            parsed_terms = ()
            _validate_dense_terms(terms, specs)
        for join in self.baryon_joins:
            if join.spin not in specs or join.elemental not in specs:
                raise ValueError("baryon join spin and elemental inputs need TensorSpec")
            if join.sparse_spin is not None and specs[join.spin].shape != (4, 4, 4):
                raise ValueError("sparse baryon spin coefficients require shape (4,4,4)")
            if join.scene < 0 or join.scene >= len(terms):
                raise ValueError("baryon join scene index is invalid")
            if join.core is not None:
                if join.core.components[0].inputs != (join.spin, join.elemental):
                    raise ValueError("baryon core first component must match its declared legacy inputs")
                mode_shape = specs[join.elemental].shape[-3:]
                expected = (*specs[join.spin].shape, *mode_shape)
                if join.batch_label is not None:
                    expected = (parsed_terms[join.scene].label_dimensions[join.batch_label], *expected)
                if join.core.spec.shape != expected:
                    raise ValueError("baryon core shape differs from its endpoint slots")
                for component in join.core.components:
                    spin, elemental = component.inputs
                    if spin not in specs or elemental not in specs:
                        raise ValueError("baryon core component sources must be declared")
                    if specs[spin].shape != specs[join.spin].shape or specs[elemental].shape[-3:] != mode_shape:
                        raise ValueError("baryon core components require equal ordered slot extents")
            for labels, spec in ((join.spin_labels, specs[join.spin]), (join.elemental_labels(specs[join.elemental]), specs[join.elemental])):
                if len(spec.shape) != len(labels) or any(
                    parsed_terms[join.scene].label_dimensions.get(label) != size
                    for label, size in zip(labels, spec.shape, strict=True)
                ):
                    raise ValueError("baryon slot labels disagree with tensor dimensions")
        for term in terms:
            if term.coefficient != 1:
                raise ValueError("physical scene coefficients belong to the output assembler")
        object.__setattr__(self, "specs", MappingProxyType(specs))
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "sparse_bridges", tuple(self.sparse_bridges))
        object.__setattr__(self, "baryon_joins", tuple(self.baryon_joins))
        object.__setattr__(self, 'meson_cores', cores)


@dataclass(frozen=True)
class SparseBridgeSpec:
    left: str
    right: str
    record: str
    used_np: int
    nnz_upper: int
    left_axes: tuple[int, int, int] = (-3, -2, -1)
    right_axes: tuple[int, int, int] = (0, 1, 2)
    same_point: bool = False
    scene: int | None = None
    transpose_bridge: bool = False
    part_count: int = 1
    reverse_operands: bool = False
    used_np_right: int | None = None

    def __post_init__(self):
        if (isinstance(self.used_np, bool) or not isinstance(self.used_np, Integral)
                or self.used_np < 1 or self.nnz_upper < 0 or self.part_count < 1):
            raise ValueError("sparse bridge needs positive point/part counts and a non-negative nnz bound")
        right_count = self.used_np if self.used_np_right is None else self.used_np_right
        if (isinstance(right_count, bool) or not isinstance(right_count, Integral)
                or right_count < 1):
            raise ValueError("sparse bridge right point count must be a positive integer")
        left_axes, right_axes = tuple(self.left_axes), tuple(self.right_axes)
        if len(left_axes) != 3 or len(right_axes) != 3 or len(set(left_axes)) != 3 or len(set(right_axes)) != 3:
            raise ValueError("sparse bridge endpoint axes must be distinct")
        if not isinstance(self.reverse_operands, bool):
            raise TypeError("reverse_operands must be bool")
        object.__setattr__(self, "left_axes", left_axes)
        object.__setattr__(self, "right_axes", right_axes)
        object.__setattr__(self, "used_np", int(self.used_np))
        object.__setattr__(self, "used_np_right", int(right_count))

    @property
    def point_shape(self) -> tuple[int, int]:
        """Materialized sparse record extents in its ordered COO index axes."""
        return self.used_np, self.used_np_right


def _validate_coordinate_batch_bridge(metadata, bridge, specs):
    """Validate the JSON-only lane contract carried by a sparse source.

    Scalar bridge descriptors have no marker and keep their historical meaning.
    The aggregate ``nnz_upper`` on a marked bridge is the sum of the per-lane
    per-record bounds; ``part_count`` remains the common number of parts in
    every lane.
    """
    keys = metadata.get("coordinate_keys")
    bounds = metadata.get("lane_nnz_upper")
    lane_sources = metadata.get("lane_sources")
    point_shape = metadata.get("point_shape")
    part_count = metadata.get("part_count")
    if (not isinstance(keys, (tuple, list)) or not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or len(keys) != len(set(keys))):
        raise ValueError("coordinate-batched P2P metadata needs unique ordered coordinate_keys")
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != len(keys)
            or any(type(bound) is not int or bound < 0 for bound in bounds)):
        raise ValueError("coordinate-batched P2P metadata needs one nonnegative lane bound per key")
    if (not isinstance(lane_sources, (tuple, list)) or len(lane_sources) != len(keys)
            or any(not isinstance(item, MappingABC) for item in lane_sources)):
        raise ValueError("coordinate-batched P2P metadata needs one source identity mapping per lane")
    if (not isinstance(point_shape, (tuple, list)) or len(point_shape) != 2
            or any(type(size) is not int or size < 1 for size in point_shape)):
        raise ValueError("coordinate-batched P2P metadata needs two positive point extents")
    if type(part_count) is not int or part_count < 1:
        raise ValueError("coordinate-batched P2P metadata needs a positive part_count")
    for role in (bridge.left, bridge.right):
        _validate_batch_complex128_envelope(specs[role].dtype, f"endpoint {role!r}")
    for lane_index, lane_source in enumerate(lane_sources):
        raw_metadata = lane_source.get("metadata")
        if not isinstance(raw_metadata, MappingABC):
            raise ValueError(f"coordinate-batched lane {lane_index} needs raw source metadata")
        try:
            raw_dtype = np.dtype(raw_metadata["value_dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"coordinate-batched lane {lane_index} needs a valid raw value_dtype"
            ) from exc
        _validate_batch_complex128_envelope(raw_dtype, f"lane {lane_index} raw values")
    if tuple(point_shape) != bridge.point_shape:
        raise ValueError("coordinate-batched P2P point_shape differs from its SparseBridgeSpec")
    if part_count != bridge.part_count:
        raise ValueError("coordinate-batched P2P part_count differs from its SparseBridgeSpec")
    if bridge.nnz_upper != sum(bounds):
        raise ValueError("coordinate-batched P2P nnz_upper must sum all per-lane bounds")


def _validate_batch_complex128_envelope(dtype, label):
    """Bound marked batch arithmetic to the complex128 result-size model."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "biufc":
        raise ValueError(f"coordinate-batched {label} must use a numeric dtype")
    promoted = np.result_type(dtype, np.dtype(np.complex128))
    if promoted.kind != "c" or promoted.itemsize > np.dtype(np.complex128).itemsize:
        raise ValueError(
            f"coordinate-batched {label} exceeds the complex128 result envelope"
        )


@dataclass(frozen=True)
class BaryonJoinSpec:
    scene: int
    name: str
    spin: str
    elemental: str
    spin_labels: tuple[int, ...]
    mode_labels: tuple[int, ...]
    sparse_spin: tuple[tuple[int, int, int, complex], ...] | None = field(default=None, kw_only=True)
    core: object = field(default=None, kw_only=True)
    kernel_description: object = field(default=None, kw_only=True)
    batch_label: int | None = field(default=None, kw_only=True)

    @property
    def core_labels(self):
        labels = self.spin_labels + self.mode_labels
        return labels if self.batch_label is None else (self.batch_label, *labels)

    def elemental_labels(self, spec):
        if len(spec.shape) == 4 and self.batch_label is not None:
            return (self.batch_label, *self.mode_labels)
        return self.mode_labels

    def __post_init__(self):
        object.__setattr__(self, "spin_labels", axis_ids(self.spin_labels))
        object.__setattr__(self, "mode_labels", axis_ids(self.mode_labels))
        if len(self.spin_labels) != 3 or len(self.mode_labels) != 3:
            raise ValueError("baryon join requires three spin and three mode slots")
        if len(set(self.spin_labels + self.mode_labels)) != 6:
            raise ValueError("baryon join slot labels must be distinct")
        if self.batch_label is not None and (type(self.batch_label) is not int
                or self.batch_label < 0 or self.batch_label in self.spin_labels + self.mode_labels):
            raise ValueError("baryon coordinate axis must be a distinct retained label")
        object.__setattr__(self, "sparse_spin", _sparse_spin_entries(self.sparse_spin))
        if self.core is not None:
            from .cores import TensorCoreSpec
            if not isinstance(self.core, TensorCoreSpec) or self.core.kind != "baryon":
                raise TypeError("baryon join core must be TensorCoreSpec(kind='baryon')")


def normalized_scene_relations(terms, joins=()):
    """Rename local IDs consistently in each Term and its declared core slots.

    Array identities, operand order, output order and recipes are preserved.
    This is a metadata comparison, without computation or path selection.
    """
    normalized, maps = [], []
    for term in terms:
        pattern = term.subscripts.normalized()
        maps.append({old: new for before, after in zip(term.subscripts.inputs, pattern.inputs, strict=True)
                     for old, new in zip(before, after, strict=True)})
        axes = maps[-1]
        barriers = tuple((tuple(axes[label] for label in trigger),
                          tuple(axes[label] for label in dependent))
                         for trigger, dependent in term.sum_barriers)
        normalized.append(replace(term, subscripts=pattern, sum_barriers=barriers))
    return tuple(normalized), tuple(replace(join,
        spin_labels=tuple(maps[join.scene][axis] for axis in join.spin_labels),
        mode_labels=tuple(maps[join.scene][axis] for axis in join.mode_labels),
        batch_label=(None if join.batch_label is None else maps[join.scene][join.batch_label])) for join in joins)


def _sparse_spin_entries(entries):
    """Freeze an exact Ns=4 coefficient structure without operator dependencies."""
    if entries is None:
        return None
    normalized = []
    previous = None
    for entry in entries:
        if len(entry) != 4:
            raise ValueError("sparse spin entries require three indices and one coefficient")
        indices = tuple(entry[:3])
        if any(isinstance(index, bool) or not isinstance(index, Integral)
               or not 0 <= index < 4 for index in indices):
            raise ValueError("sparse spin indices must be integers in 0..3")
        indices = tuple(int(index) for index in indices)
        if previous is not None and indices <= previous:
            raise ValueError("sparse spin entries must have unique, lexicographically ordered indices")
        coefficient = complex(entry[3])
        if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)):
            raise ValueError("sparse spin coefficients must be finite")
        if coefficient == 0:
            raise ValueError("sparse spin entries must contain exact nonzero coefficients")
        normalized.append((*indices, coefficient))
        previous = indices
    return tuple(normalized)
