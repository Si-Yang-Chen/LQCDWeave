"""Explicit table frontends over the established elemental kernels."""

from collections.abc import Mapping

import numpy as np
from lqcdweave.contracts.elemental import (
    elemental_operator_rows,
    SparsePointBlock,
)
from lqcdweave.contracts.time_selection import normalize_time_selection
from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from .generator import _ElementalEngine
from .baryon import _BaryonEngine
from .sampling_weights import weight_point_block
from .structure import elemental_structures
from .dilution_weights import DILUTION_WEIGHT_SCHEME, selected_dilution_groups, weight_dilution_pairs


class _ArrayField:
    def __init__(self, array, time_selection=None):
        self.array = array
        self.time_selection = time_selection
        self.Ne = array.shape[1] if len(array.shape) == 6 else None

    def load(self, key):
        return self.array

    def read_modes(self, key, t, start, stop):
        slot = t if self.time_selection is None else self.time_selection.slot(t)
        return self.array[slot, start:stop]


class _MappedField:
    """Attach declared eager-input coordinates; streaming times stay physical."""

    def __init__(self, field, time_selection):
        existing = getattr(field, "time_selection", None)
        if existing is not None and existing != time_selection:
            raise ValueError("explicit input selection disagrees with handle.time_selection")
        self.field, self.time_selection = field, time_selection

    @property
    def Ne(self):
        return self.field.Ne

    @property
    def basis_orthogonal(self):
        return getattr(self.field, "basis_orthogonal", False)

    def load(self, key):
        return self.field.load(key)

    def read_modes(self, key, t, start, stop):
        return self.field.read_modes(key, t, start, stop)

    def __getattr__(self, name):
        return getattr(self.field, name)


def _field(value, selection=None, *, default_selection=None):
    if hasattr(value, "shape"):
        return _ArrayField(value, selection if selection is not None else default_selection)
    return value if selection is None else _MappedField(value, selection)


def _selection(selection, Lt):
    return None if selection is None else normalize_time_selection(selection, Lt=Lt)


def _lattice(gauge, lattice_size):
    if lattice_size is not None:
        return tuple(lattice_size)
    if not hasattr(gauge, "shape"):
        raise ValueError("lazy field handles require explicit lattice_size")
    if len(gauge.shape) != 7:
        raise ValueError("gauge layout must be (t,z,y,x,d,a,b)")
    t, z, y, x = gauge.shape[:4]
    return x, y, z, t


def _momentum_input(momenta, momentum_dict):
    if momentum_dict is not None:
        if momenta is not None:
            raise ValueError("provide only one of momenta and momentum_dict")
        if not isinstance(momentum_dict, Mapping):
            raise ValueError("momentum_dict must be a mapping")
        return momentum_dict
    if momenta is None:
        raise ValueError("momenta or momentum_dict is required")
    return momenta


class ElementalGenerator:
    """Initialize data and one coefficient table; calculate with time only.

    Arrays are loaded immediately. Lazy handles use lattice_size and load(key).
    Dense arrays borrow storage until another calculation or load.
    momentum_dict shares Insertion's ordered XYZ values; keys are not slots.
    Legacy momenta sequences remain accepted. Supply only one momentum input.
    """

    def __init__(
        self,
        gauge_field,
        eigenvectors,
        momenta=None,
        operator_table=None,
        *,
        momentum_dict=None,
        Ne=None,
        blocks=("v2v",),
        points=None,
        array_module=np,
        ne_block_size=32,
        phase_convention="midpoint",
        path_boundary="periodic",
        dilution=None,
        is_blending=False,
        write=None,
        lattice_size=None,
        selection=None,
        eigenvector_selection=None,
        point_selection=None,
        elemental_weighted=False,
        vector_bases=None,
    ):
        if type(elemental_weighted) is not bool:
            raise TypeError("elemental_weighted must be bool")
        self.elemental_weighted = elemental_weighted
        self.vector_bases = dict(vector_bases or {})
        if set(self.vector_bases) - {"V", "D"}:
            raise ValueError("vector_bases keys must be V or D")
        if eigenvectors is not None:
            if "V" in self.vector_bases and self.vector_bases["V"] is not eigenvectors:
                raise ValueError("eigenvectors and vector_bases V must name the same input")
            self.vector_bases["V"] = eigenvectors
        self._structure_basis = self.vector_bases.get("V")
        momenta = _momentum_input(momenta, momentum_dict)
        if operator_table is None:
            raise ValueError("operator_table is required")
        self.rows = elemental_operator_rows(operator_table)
        self.catalog = getattr(operator_table, "catalog", None)
        self.components = tuple(c for c, _ in self.rows)
        self.blocks = tuple(blocks)
        self.dilution_weight_scheme = DILUTION_WEIGHT_SCHEME if any("d" in sector for sector in self.blocks) else None
        if (
            not self.blocks
            or len(set(self.blocks)) != len(self.blocks)
            or set(self.blocks) - {f"{left}2{right}" for left in "vdp" for right in "vdp"}
        ):
            raise ValueError("select distinct known sectors")
        if write is not None and not callable(write):
            raise ValueError("write must be callable")
        self.write = write
        paths = tuple(dict.fromkeys(p for _, terms in self.rows for p, c in terms))
        lattice = _lattice(gauge_field, lattice_size)
        selection = _selection(selection, lattice[3])
        vector_selections = (dict(eigenvector_selection) if isinstance(eigenvector_selection, Mapping)
                             else {kind: eigenvector_selection for kind in self.vector_bases})
        if set(vector_selections) - set(self.vector_bases):
            raise ValueError("vector time selections require the corresponding actual input")
        vector_selections = {kind: _selection(value, lattice[3]) for kind, value in vector_selections.items()}
        point_selection = _selection(point_selection, lattice[3])
        point = points
        if isinstance(points, SpatialPointSet) and point_selection is not None:
            if point_selection.times != points.times or point_selection.Lt != points.lattice_size[3]:
                raise ValueError("point_selection disagrees with SpatialPointSet")
        elif points is not None and not hasattr(points, "shape"):
            point = _field(points, point_selection)
        if hasattr(points, "shape"):
            # Public point arrays use (time,point,xyz), matching SpatialPointSet.
            point = _ArrayField(points,
                                point_selection if point_selection is not None else selection)
        adapted_vectors = {kind: _field(value, vector_selections.get(kind), default_selection=selection)
                           for kind, value in self.vector_bases.items()}
        counts = dict(Ne) if isinstance(Ne, Mapping) else {}
        if set(counts) - set(self.vector_bases):
            raise ValueError("vector counts require the corresponding actual input")
        self._engine = _ElementalEngine(
            lattice,
            _field(gauge_field),
            adapted_vectors.get("V"),
            paths=paths,
            path_version="v2",
            momentum_list=momenta,
            usedNe=None if isinstance(Ne, Mapping) else Ne,
            vector_bases=adapted_vectors,
            vector_counts=counts,
            point=point,
            array_module=array_module,
            ne_block_size=ne_block_size,
            phase_convention=phase_convention,
            path_boundary=path_boundary,
            dilution=dilution,
            is_blending=is_blending,
            selection=selection,
        )
        self.momenta = self._engine.momentum_list
        self.time_selection = self._engine.time_selection
        self.Lt, self.Nt, self.times = self._engine.Lt, self._engine.Nt, self.time_selection.times
        snapshot = points if isinstance(points, SpatialPointSet) else getattr(points, "points", None)
        self._sampling_points = snapshot if isinstance(snapshot, SpatialPointSet) else None
        # A structural promise, valid at every anchor; never inspect values.
        # Different temporal pools can use different ordered coordinate bases.
        shared_temporal_basis = (self._sampling_points is not None
                                 and self._sampling_points.pool_ids is not None
                                 and len(set(self._sampling_points.pool_ids)) == 1
                                 and self._sampling_points.pool_ids[0] is not None)
        self.point_diagonal_components = tuple(
            component for component, terms in self.rows
            if all(all(offset % extent == 0 for offset, extent in
                       zip(path.displacement[:3], lattice[:3], strict=True))
                   and (path.displacement[3] % self.Lt == 0 or shared_temporal_basis)
                   for path, _coefficient in terms)
        )
        self._describe_structures()
        self.sampling_groups = dict(getattr(self._sampling_points, "sampling_groups", {}))
        self.sampling_pool_ids = {}
        if self._sampling_points is not None:
            for component in self.components:
                for mi in range(len(self.momenta)):
                    for time in self._sampling_points.times:
                        pool = self._sampling_points.point_pool_id(time)
                        if pool is not None:
                            for side in ("bar", "field"):
                                self.sampling_pool_ids[component, mi, time, side] = pool
        if elemental_weighted and any("p" in sector for sector in self.blocks):
            if not self.sampling_groups:
                raise ValueError("elemental_weighted requires SpatialPointSet with explicit sampling pool IDs")
            if is_blending:
                raise NotImplementedError("point compensation does not define QR or mode-blending weights")
            if any(len({p.displacement[3] for p, _ in terms}) != 1
                   for _, terms in self.rows):
                raise ValueError("weighted components must have one actual endpoint-time displacement")
        self.dilution_endpoints, self.basis_endpoints = {}, {}
        if hasattr(gauge_field, "shape") and all(hasattr(value, "shape") for value in self.vector_bases.values()) and getattr(points, "cfg", None) is None:
            self.load(None)

    def iter_slices(self):
        """Yield (physical time, borrowed result) in the explicit anchor order."""
        for t in self.times:
            yield t, self.calc_path_blocks(t)

    def load(self, key):
        self._engine.load(key)
        self._describe_structures()
        self._describe_endpoints()
        return self

    def _describe_structures(self):
        self.elemental_structures = elemental_structures(
            self.rows, self.momenta, self.blocks,
            orthogonal_basis=getattr(self._structure_basis, "basis_orthogonal", False) is True,
            orthogonal_bases={kind: getattr(value, "basis_orthogonal", False) is True
                              for kind, value in self.vector_bases.items()},
            point_diagonal_components=self.point_diagonal_components)

    def _describe_endpoints(self):
        from lqcdweave.contracts.elemental_sectors import sector_kinds
        self.dilution_endpoints, self.basis_endpoints = {}, {}
        self._dilution_weights = {}
        self._dilution_groups = {}
        for sector in self.blocks:
            for side, kind in zip(("bar", "field"), (kind.upper() for kind in sector_kinds(sector))):
                if kind == "P":
                    continue
                handle = self.vector_bases.get(kind)
                if handle is None:
                    raise ValueError(f"{kind} sectors require an actual vector basis")
                identity = getattr(handle, "basis_id", None)
                count = self._engine.used_modes[kind]
                labels = tuple(getattr(handle, "mode_labels", range(count)))[:count]
                if identity is not None:
                    self.basis_endpoints[sector, side] = (identity, labels)
                if kind == "D":
                    source = getattr(handle, "dilution_source", getattr(handle, "dilution_basis", None))
                    if source is not None:
                        from lqcdweave.contracts.dilution import DilutionSource
                        if not isinstance(source, DilutionSource):
                            source = DilutionSource.from_document(source)
                        source.validate_selection(labels)
                        if identity is not None and identity != source.data_id:
                            raise ValueError("D source identity differs from its actual vector handle")
                        self.basis_endpoints[sector, side] = (source.data_id, labels)
                        self.dilution_endpoints[sector, side] = source
                        if len(labels) != count:
                            raise ValueError("D endpoint labels differ from the actual retained vector count")
                        self._dilution_weights[sector, side] = self._engine.xp.asarray(
                            tuple(source.first_moment_weights[index] for index in labels))
                        self._dilution_groups[sector, side] = selected_dilution_groups(source, labels)
                    elif self.elemental_weighted:
                        raise ValueError("elemental_weighted D endpoints require explicit dilution source provenance")
        if self.elemental_weighted and any("d" in sector or "p" in sector for sector in self.blocks):
            if any(len({p.displacement[3] for p, _ in terms}) != 1 for _, terms in self.rows):
                raise ValueError("weighted components must have one actual endpoint-time displacement")

    def _weight_dilution(self, value, sector, component, left_time, right_time):
        from lqcdweave.contracts.elemental_sectors import sector_kinds
        left, right = sector_kinds(sector)
        if left == right == "d":
            for mi, block in enumerate(value):
                weight_dilution_pairs(block,
                    self.dilution_endpoints[sector, "bar"], self.dilution_endpoints[sector, "field"],
                    self._dilution_groups[sector, "bar"], self._dilution_groups[sector, "field"],
                    self._dilution_weights[sector, "bar"], self._dilution_weights[sector, "field"],
                    left_time, right_time, self.elemental_structures.get((sector, component, mi), "general"),
                    array_module=self._engine.xp)
            return
        for side, kind in (("bar", left), ("field", right)):
            if kind != "d":
                continue
            count = self._engine.used_modes["D"]
            weights = self._dilution_weights[sector, side]
            axis = 1 if side == "bar" else value.ndim - 1
            shape = [1] * value.ndim
            shape[axis] = count
            value *= weights.reshape(shape)

    def stout_smear(self, nstep, rho, *, matrix_batch_size=4096):
        self._engine.stout_smear(nstep, rho, matrix_batch_size=matrix_batch_size)

    def weight_spatial_block(self, value, sector, component, physical_time, *,
                             point_pool=None, sampling_groups=None):
        """Internal workflow bridge for an unweighted, single spatial slice.

        A compact Lt=1 producer still uses the actual physical-time D/P pools.
        The returned value aliases the supplied producer-owned block. Ordinary
        generator users select elemental_weighted at construction instead.
        """
        from lqcdweave.contracts.elemental_sectors import sector_kinds, sector_layout
        if self.elemental_weighted:
            raise ValueError("cannot weight an already weighted producer block")
        if any(step >= 6 for _, terms in self.rows for path, _ in terms for step in path.steps):
            raise ValueError("physical-time remapping requires spatial paths")
        if "d" in sector_kinds(sector):
            self._weight_dilution(value[None], sector, component, physical_time, physical_time)
        if "p" in sector_kinds(sector):
            if point_pool is None or sampling_groups is None:
                raise ValueError("point weighting requires the actual physical-time pool")
            weight_point_block(value, sector_layout(sector), point_pool, point_pool,
                               sampling_groups, array_module=self._engine.xp)
        return value

    def project_SU3(self, *, matrix_batch_size=4096):
        self._engine.project_SU3(matrix_batch_size=matrix_batch_size)

    def _combine(self, raw):
        xp = self._engine.xp
        index = {p: i for i, p in enumerate(self._engine.paths)}
        identity = all(
            len(terms) == 1 and terms[0][1] == 1 and index[terms[0][0]] == i
            for i, (_, terms) in enumerate(self.rows)
        ) and len(self.rows) == len(index)
        if identity:
            return raw
        result = {}
        for sector, values in raw.items():
            if sector != "p2p":
                out = self._engine._buffer(
                    "combined_" + sector, (len(self.rows), *values.shape[1:])
                )
                out.fill(0)
                for ri, (_, terms) in enumerate(self.rows):
                    for p, c in terms:
                        out[ri] += c * values[index[p]]
                result[sector] = out
            else:
                rows = []
                for _, terms in self.rows:
                    momenta = []
                    for mi in range(self._engine.num_momentum):
                        pieces = [(values[index[p]][mi], c) for p, c in terms]
                        size = pieces[0][0].point_shape
                        keys = xp.concatenate(
                            [b.rows * size[1] + b.columns for b, c in pieces]
                        )
                        payload = xp.concatenate([c * b.values for b, c in pieces])
                        unique, inverse = xp.unique(keys, return_inverse=True)
                        out = xp.zeros((len(unique), 3, 3), dtype=complex)
                        if xp is np:
                            xp.add.at(out, inverse, payload)
                        else:
                            # CuPy scatter-add supports real floating dtypes,
                            # including HIP, but not complex payloads. Preserve
                            # repeated point-pair contributions in both parts.
                            xp.add.at(out.real, inverse, payload.real)
                            xp.add.at(out.imag, inverse, payload.imag)
                        momenta.append(
                            SparsePointBlock(
                                unique // size[1], unique % size[1], out, size
                            )
                        )
                    rows.append(momenta)
                result[sector] = rows
        return result

    def _deliver(self, result, t):
        from lqcdweave.contracts.elemental_sectors import sector_kinds, sector_layout
        for ri, component in enumerate(self.components):
            if "p2p" in result:
                for mi, value in enumerate(result["p2p"][ri]):
                    value.point_diagonal = component in self.point_diagonal_components
                    value.structure = self.elemental_structures.get(("p2p", component, mi), "general")
        if self.elemental_weighted:
            for ri, (component, terms) in enumerate(self.rows):
                offset = terms[0][0].displacement[3]
                endpoint = t + offset
                if self._engine.path_boundary == "open" and not 0 <= endpoint < self.Lt:
                    continue  # The geometrically absent block is already zero.
                endpoint %= self.Lt
                for sector, values in result.items():
                    left_kind, right_kind = (kind.upper() for kind in sector_kinds(sector))
                    if "D" in (left_kind, right_kind):
                        self._weight_dilution(values[ri], sector, component, t, endpoint)
                    if "P" not in (left_kind, right_kind):
                        continue
                    left = (self._sampling_points.point_pool_id(t)
                            if left_kind == "P" else None)
                    right = (self._sampling_points.point_pool_id(endpoint)
                             if right_kind == "P" else None)
                    for mi in range(len(self.momenta)):
                        weight_point_block(values[ri][mi], sector_layout(sector), left, right,
                                           self.sampling_groups, array_module=self._engine.xp)
        if self.write is None:
            return result
        # This constructor callback consumes final combinations, not raw paths.
        for sector, values in result.items():
            for ri in range(len(self.rows)):
                for mi in range(self._engine.num_momentum):
                    value = values[ri][mi]
                    slices = (
                        ()
                        if sector == "p2p"
                        else tuple(slice(None) for _ in value.shape)
                    )
                    self.write(ri, mi, sector, slices, value)
        return None

    def calc_path_blocks(self, t):
        if self.blocks == ("v2v",) and all(
            step < 6 for p in self._engine.paths for step in p.steps
        ):
            projection = [
                {
                    "id": i,
                    "terms": [
                        {"path": p.steps, "coefficient": [c.real, c.imag]}
                        for p, c in terms
                    ],
                }
                for i, (_, terms) in enumerate(self.rows)
            ]
            return self._deliver(self._engine.calc_path_blocks(t, rows=projection), t)
        return self._deliver(
            self._combine(self._engine.calc_path_blocks(t, self.blocks, recipe="paths")), t
        )

    def calc_derivative(self, t):
        if self.blocks != ("v2v",):
            raise ValueError("derivatives support only v2v")
        if any(step > 2 for p in self._engine.paths for step in p.steps):
            raise ValueError("derivative directions must be x/y/z (0/1/2)")
        self._engine.derivative_list = tuple(p.steps for p in self._engine.paths)
        raw = self._engine._calc_recipe(
            t, self._engine.derivative_list, kind="derivative"
        )
        return self._deliver(self._combine({"v2v": raw}), t)


class BaryonElementalGenerator:
    """One shared basis by default; an explicit triple selects three bases.

    momentum_dict shares Insertion's ordered XYZ values; keys are not slots.
    Legacy momenta sequences remain accepted. Supply only one momentum input.
    """

    def __init__(
        self,
        gauge_field,
        eigenvectors,
        momenta=None,
        operator_table=None,
        *,
        momentum_dict=None,
        Ne=None,
        array_module=np,
        ne_block_size=16,
        spatial_block_size=256,
        phase_convention="anchor",
        path_boundary="periodic",
        cache_bytes=64 * 1024**2,
        stream_eigenvectors=False,
        vector_scratch_limit_bytes=None,
        flavors=("q", "q", "q"),
        write=None,
        lattice_size=None,
        selection=None,
        eigenvector_selection=None,
    ):
        momenta = _momentum_input(momenta, momentum_dict)
        if operator_table is None:
            raise ValueError("operator_table is required")
        self.rows = elemental_operator_rows(operator_table, rank=3)
        self.catalog = getattr(operator_table, "catalog", None)
        self.components = tuple(c for c, _ in self.rows)
        if write is not None and not callable(write):
            raise ValueError("write must be callable")
        self.write = write
        shared = (
            hasattr(eigenvectors, "shape")
            or callable(getattr(eigenvectors, "load", None))
            or callable(getattr(eigenvectors, "read_modes", None))
        )
        vectors = (eigenvectors,) * 3 if shared else tuple(eigenvectors)
        if len(vectors) != 3:
            raise ValueError("one basis or three bases required")
        self._projection = tuple(
            {
                "id": i,
                "terms": [
                    {"paths": [p.steps for p in ps], "coefficient": [c.real, c.imag]}
                    for ps, c in terms
                ],
            }
            for i, (_, terms) in enumerate(self.rows)
        )
        paths = tuple(dict.fromkeys(ps for _, terms in self.rows for ps, c in terms))
        lattice = _lattice(gauge_field, lattice_size)
        selection = _selection(selection, lattice[3])
        if eigenvector_selection is None:
            selections = (None,) * 3
        elif isinstance(eigenvector_selection, (tuple, list)):
            if len(eigenvector_selection) != 3:
                raise ValueError("one eigenvector selection or three selections required")
            selections = tuple(_selection(s, lattice[3]) for s in eigenvector_selection)
        else:
            selections = (_selection(eigenvector_selection, lattice[3]),) * 3
        fields = {}
        adapted = tuple(fields.setdefault((id(v), s), _field(v, s, default_selection=selection))
                        for v, s in zip(vectors, selections))
        self._engine = _BaryonEngine(
            lattice,
            _field(gauge_field),
            adapted,
            momentum_list=momenta,
            usedNe=Ne,
            paths=paths,
            array_module=array_module,
            ne_block_size=ne_block_size,
            spatial_block_size=spatial_block_size,
            phase_convention=phase_convention,
            path_boundary=path_boundary,
            cache_bytes=cache_bytes,
            stream_eigenvectors=stream_eigenvectors,
            vector_scratch_limit_bytes=vector_scratch_limit_bytes,
            flavors=flavors,
            selection=selection,
        )
        self.momenta = self._engine.momentum_list
        self.time_selection = self._engine.time_selection
        self.Lt, self.Nt, self.times = self._engine.Lt, self._engine.Nt, self.time_selection.times
        if hasattr(gauge_field, "shape") and all(hasattr(v, "shape") for v in vectors):
            self.load(None)

    def iter_slices(self):
        """Yield (physical time, borrowed result) in the explicit anchor order."""
        for t in self.times:
            yield t, self.calc_path_blocks(t)

    def load(self, key):
        self._engine.load(key)
        return self

    def stout_smear(self, nstep, rho, *, matrix_batch_size=4096):
        self._engine.stout_smear(nstep, rho, matrix_batch_size=matrix_batch_size)

    def project_SU3(self, *, matrix_batch_size=4096):
        self._engine.project_SU3(matrix_batch_size=matrix_batch_size)

    def calc_path_blocks(self, t):
        return self._engine.calc_path_blocks(t, rows=self._projection, write=self.write)
