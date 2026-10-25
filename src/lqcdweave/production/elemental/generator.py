"""Elemental production with explicit inputs and reusable output buffers."""

import numpy as np

from lqcdweave.contracts.elemental import GaugePath, SparsePointBlock, integer, normalize_momenta, spatial_projection_rows
from lqcdweave.contracts.time_selection import TimeSelectionRecord, normalize_time_selection
from .kernels import bilinear, point_pairs, sparse_point_block
from .structures import paths_up_to, derivatives_up_to, derivative_assignments, differentiate, covariant_derivative
from .transport import path_transport
from .workspace import ElementalWorkspace


class _ElementalEngine:
    """Produce one anchor time. Dense results borrow storage until the next call.

    Handles provide load(key); gauge layout is (t,z,y,x,d,a,b), vector layout
    (t,mode,z,y,x,color), point layout (point,t,xyz). A SpatialPointSet may be
    supplied directly. No process-global backend state is modified.
    """

    def __init__(self, latt_size, gauge_field, eigenvector=None, num_nabla=0,
                 momentum_list=((0, 0, 0),), dilution=None, is_blending=False,
                 usedNe=None, usedNp=None, calc_mode="calc_disp", debug=False,
                 paths=None, path_version="legacy", path_boundary="periodic", point=None,
                  *, point_right=None, usedNp_right=None,
                  array_module=np, ne_block_size=32, phase_convention="midpoint", selection=None,
                  vector_bases=None, vector_counts=None):
        self.latt_size = tuple(integer(n, "extent", 1) for n in latt_size)
        if len(self.latt_size) != 4:
            raise ValueError("latt_size must be (Lx,Ly,Lz,Lt)")
        self.Lt = self.latt_size[3]
        self.time_selection = (TimeSelectionRecord(self.Lt, tuple(range(self.Lt)))
                               if selection is None else normalize_time_selection(selection, Lt=self.Lt))
        self.Nt = self.time_selection.Nt
        self.mode_pair_weights = None
        if is_blending:
            from .mode_weights import dilution_pair_weights
            if dilution is None or len(dilution) != 2:
                raise ValueError("blending requires (total_counts, used_counts)")
            self.mode_pair_weights = array_module.asarray(dilution_pair_weights(*dilution))
        elif dilution is not None:
            raise ValueError("dilution requires explicit is_blending=True")
        if calc_mode not in ("calc_disp", "calc_deriv") or path_boundary not in ("periodic", "open"):
            raise ValueError("invalid calculation mode or boundary")
        if phase_convention not in ("midpoint", "anchor", "none"):
            raise ValueError("invalid phase convention")
        if path_version not in ("legacy", "v2") or (paths is not None and path_version != "v2"):
            raise ValueError("explicit paths require path_version='v2'")
        self.xp = array_module
        self.gauge_field, self.eigenvector, self.point = gauge_field, eigenvector, point
        self.vector_bases = dict(vector_bases or {})
        if eigenvector is not None:
            self.vector_bases.setdefault("V", eigenvector)
        self._requested_modes = dict(vector_counts or {})
        if usedNe is not None:
            self._requested_modes.setdefault("V", usedNe)
        self._basis_vectors, self._basis_times, self.used_modes = {}, {}, {}
        self.point_right = point_right
        self.num_nabla = integer(num_nabla, "num_nabla", 0)
        self.paths = tuple(GaugePath.coerce(p) for p in paths) if paths is not None else paths_up_to(num_nabla, path_version)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("paths must be nonempty and unique")
        self.derivative_list = derivatives_up_to(num_nabla)
        self.momentum_list = normalize_momenta(momentum_list)
        self.num_momentum, self.num_disp = len(self.momentum_list), len(self.paths)
        self.calc_mode, self.path_boundary, self.phase_convention = calc_mode, path_boundary, phase_convention
        self.path_version = path_version
        self.ne_block_size = integer(ne_block_size, "ne_block_size", 1)
        self.usedNe = None if usedNe is None else integer(usedNe, "usedNe", 1)
        self.usedNp = None if usedNp is None else integer(usedNp, "usedNp", 1)
        self.usedNp_right = (None if usedNp_right is None
                             else integer(usedNp_right, "usedNp_right", 1))
        self._requestedNe, self._requestedNp = self.usedNe, self.usedNp
        self._requestedNpRight = self.usedNp_right
        self._workspace = ElementalWorkspace(self.latt_size, gauge_field,
            array_module=array_module, phase_convention=phase_convention)
        self._vectors = self._points = self._points_right = None

    def load(self, key):
        self._vectors = self._points = self._points_right = None
        self._basis_vectors = {}
        try:
            return self._load_inputs(key)
        except Exception:
            self._workspace.invalidate()
            self._vectors = self._points = self._points_right = None
            self._basis_vectors = {}
            raise

    def _load_inputs(self, key):
        self._workspace.load(key)
        lx, ly, lz, lt = self.latt_size
        vectors = None
        fulltime = TimeSelectionRecord(lt, tuple(range(lt)))
        vector_times = getattr(self.eigenvector, "time_selection", None)
        self._vector_times = (fulltime if vector_times is None
                              else normalize_time_selection(vector_times, Lt=lt))
        loaded, times, counts = {}, {}, {}
        for kind, handle in self.vector_bases.items():
            declared = getattr(handle, "time_selection", None)
            times[kind] = fulltime if declared is None else normalize_time_selection(declared, Lt=lt)
            payload = handle.load(key)
            shape = tuple(payload.shape)
            if len(shape) != 6 or shape[0] != times[kind].Nt or shape[2:] != (lz, ly, lx, 3):
                raise ValueError("vectors must have shape (input_Nt,Ne,Lz,Ly,Lx,3) matching time_selection")
            requested = self._requested_modes.get(kind)
            count = shape[1] if requested is None else integer(requested, "vector count", 1)
            if not 1 <= count <= shape[1]:
                raise ValueError("usedNe exceeds available modes")
            loaded[kind], counts[kind] = payload, count
            if kind == "V" and self.mode_pair_weights is not None and self.mode_pair_weights.shape != (shape[1],shape[1]):
                raise ValueError("dilution counts must cover the loaded mode basis")
        self._basis_vectors, self._basis_times, self.used_modes = loaded, times, counts
        vectors = loaded.get("V")
        if vectors is not None:
            self.usedNe, self._vector_times = counts["V"], times["V"]
        elif self.usedNe is not None:
            raise ValueError("usedNe requires eigenvector input")
        def load_point_table(handle, name):
            if handle is None:
                return None, None
            if hasattr(handle, "coordinates"):
                if tuple(handle.lattice_size) != self.latt_size:
                    raise ValueError("point set must match the physical lattice")
                time_selection = TimeSelectionRecord(lt, tuple(handle.times))
                values = np.array(handle.coordinates, copy=True)
            else:
                declared = getattr(handle, "time_selection", None)
                time_selection = (fulltime if declared is None
                                  else normalize_time_selection(declared, Lt=lt))
                raw = handle.load(key)[:]
                raw = raw.get() if hasattr(raw, "get") else raw
                values = np.array(raw, copy=True)
            if (values.ndim != 3 or values.shape[0] != time_selection.Nt or values.shape[-1] != 3
                    or values.dtype.kind not in "iu"):
                raise ValueError(f"{name} require integer XYZ coordinates matching input time_selection")
            if np.any(values < 0) or np.any(values >= np.asarray((lx, ly, lz))):
                raise ValueError(f"{name} coordinates are outside the lattice")
            if any(len(np.unique(row, axis=0)) != values.shape[1] for row in values):
                raise ValueError(f"{name} coordinates must be unique at each time")
            return values, time_selection

        def select_point_prefix(values, requested, name):
            if values is None:
                if requested is not None:
                    raise ValueError(f"{name} requires point input")
                return None, None
            count = values.shape[1] if requested is None else requested
            if not 1 <= count <= values.shape[1]:
                raise ValueError(f"{name} count exceeds available points")
            values = values[:, :count]
            return values, count

        point_table, self._point_times = load_point_table(self.point, "points")
        points, left_count = select_point_prefix(point_table, self._requestedNp, "points")
        if self.point_right is None and self._requestedNpRight is None:
            points_right, right_count = points, left_count
            self._point_right_times = self._point_times
        elif self.point_right is None:
            # Retain the historical same-pool/equal-prefix behavior unless an
            # explicit independent right count was requested. Reuse the one
            # loaded table when the same source serves both endpoint prefixes.
            points_right, right_count = select_point_prefix(
                point_table, self._requestedNpRight, "right points")
            self._point_right_times = self._point_times
        else:
            right_table, self._point_right_times = load_point_table(self.point_right, "right points")
            points_right, right_count = select_point_prefix(
                right_table, self._requestedNpRight, "right points")
        self.usedNp, self.usedNp_right = left_count, right_count
        self._vectors, self._points, self._points_right = vectors, points, points_right
        return self

    def _time(self, t):
        t = self._workspace.time(t)
        self.time_selection.slot(t)
        return t

    def _buffer(self, name, shape):
        return self._workspace.buffer(name, shape)

    def _v(self, t, s, kind="V"):
        field = self._basis_vectors[kind][self._basis_times[kind].slot(t), s]
        # Keep CPU loader scratch isolated; already resident backend arrays are
        # immutable views during contraction and need no extra device copy.
        if self.xp is not np and isinstance(field, self.xp.ndarray):
            return self.xp.asarray(field, dtype=complex)
        return self.xp.array(field, dtype=complex, copy=True)

    def _phase(self, momentum, displacement=(0,0,0), convention=None):
        return self._workspace.phase(momentum, displacement, convention)

    def stout_smear(self, nstep, rho, *, matrix_batch_size=4096):
        self._workspace.stout_smear(nstep, rho, matrix_batch_size=matrix_batch_size)

    def project_SU3(self, *, matrix_batch_size=4096):
        self._workspace.project_SU3(matrix_batch_size=matrix_batch_size)

    def calc_path_blocks(self, t, blocks=("v2v",), *, recipe=None, rows=None, write=None):
        """Compute path sectors, derivative recipes, or projected rows.

        Results are sector dictionaries; write callbacks return None. Paths
        and projection can stream output; derivatives retain a full result.

        write(path_index, momentum_index, sector, slices, value) consumes each
        tile before returning. P2P values are SparsePointBlock with slices=().
        Without write, P2P is a tuple[path][momentum] of sparse blocks.
        Open-boundary invalid paths produce zero blocks, not wrapped paths.
        """
        blocks = tuple(blocks)
        if write is not None and not callable(write):
            raise ValueError("write must be callable")
        if recipe not in (None, "paths", "derivatives"):
            raise ValueError("recipe must be paths or derivatives")
        if rows is not None:
            if recipe is not None or blocks != ("v2v",):
                raise ValueError("projected rows require only v2v and no recipe override")
            result = self._calc_projected(rows,t,write=write)
            return None if write is not None else {"v2v": result}
        recipe = recipe or ("derivatives" if self.calc_mode == "calc_deriv" else "paths")
        if recipe == "derivatives":
            if blocks != ("v2v",):
                raise ValueError("derivative recipes support only v2v")
            result = self._calc_deriv(t)
            if write is None:
                return {"v2v": result}
            # The existing derivative kernel retains its full result; callbacks
            # change output ownership, not its working-set bound.
            for ri in range(len(self.derivative_list)):
                for mi in range(self.num_momentum):
                    write(ri,mi,"v2v",(slice(None),slice(None)),result[ri,mi])
            return None
        from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS, sector_kinds
        t = self._time(t)
        if not blocks or len(set(blocks)) != len(blocks) or set(blocks) - set(ELEMENTAL_SECTORS):
            raise ValueError("select distinct known sectors")
        kinds = {sector: tuple(kind.upper() for kind in sector_kinds(sector)) for sector in blocks}
        for left, right in kinds.values():
            for kind in (left, right):
                if kind != "P" and kind not in self._basis_vectors:
                    raise ValueError(f"{kind} sectors require an actual vector basis")
            if left == "P" and self._points is None:
                raise ValueError("left point sectors require left points")
            if right == "P" and self._points_right is None:
                raise ValueError("right point sectors require right points")
        # Resolve actual endpoints before any output is delivered.
        for path in self.paths:
            lo, hi = path.time_bounds
            if self.path_boundary == "open" and (t + lo < 0 or t + hi >= self.Lt):
                continue
            endpoint = (t + path.displacement[3]) % self.Lt
            for left, right in kinds.values():
                (self._point_times if left == "P" else self._basis_times[left]).slot(t)
                (self._point_right_times if right == "P" else self._basis_times[right]).slot(endpoint)
        xp = self.xp
        nleft, nright = self.usedNp or 0, self.usedNp_right or 0
        shapes = {}
        for sector, (left, right) in kinds.items():
            shapes[sector] = ((nleft, 3) if left == "P" else (self.used_modes[left],)) + (
                (nright, 3) if right == "P" else (self.used_modes[right],))
        result = {} if write is None else None
        if result is not None:
            for sector in blocks:
                if sector == "p2p":
                    result[sector] = [[None] * self.num_momentum for _ in self.paths]
                else:
                    result[sector] = self._buffer(sector, (len(self.paths), self.num_momentum, *shapes[sector]))
                    result[sector].fill(0)

        def emit(pi, mi, sector, slices, value):
            if write is not None:
                write(pi, mi, sector, slices, value)
            elif sector == "p2p":
                result[sector][pi][mi] = value
            else:
                result[sector][(pi, mi) + slices] = value

        whole = slice(None)
        ranges = {kind: [slice(i, min(i + self.ne_block_size, count))
                        for i in range(0, count, self.ne_block_size)]
                  for kind, count in self.used_modes.items()}
        for pi, path in enumerate(self.paths):
            dx, dy, dz, dt = path.displacement
            endpoint = (t + dt) % self.Lt
            transport = path_transport(self._workspace.links, path, t, self.path_boundary, array_module=xp)
            if transport is None:
                for sector in blocks:
                    for mi in range(self.num_momentum):
                        if sector == "p2p":
                            value = SparsePointBlock(xp.empty(0, dtype=np.int64), xp.empty(0, dtype=np.int64),
                                xp.empty((0, 3, 3), dtype=complex), (nleft, nright))
                            emit(pi, mi, sector, (), value)
                        else:
                            emit(pi, mi, sector, tuple(whole for _ in shapes[sector]), xp.zeros(shapes[sector], complex))
                continue
            if any(left == "P" for left, right in kinds.values()):
                pl = self._points[self._point_times.slot(t)]
                px, py, pz = xp.asarray(pl).T
            if any(right == "P" for left, right in kinds.values()):
                pr = self._points_right[self._point_right_times.slot(endpoint)]
                bases = (pr - np.asarray((dx, dy, dz))) % np.asarray(self.latt_size[:3])
                bx, by, bz = xp.asarray(bases).T
            if "p2p" in blocks:
                pairs = point_pairs(pl, pr, path.displacement, self.latt_size[:3])
                for mi, mom in enumerate(self.momentum_list):
                    emit(pi, mi, "p2p", (), sparse_point_block(transport, self._phase(mom, path.displacement),
                        pl, pr, path.displacement, self.latt_size[:3], array_module=xp, pair_indices=pairs))
            # Every vector kind uses the same transport and projection kernels.
            for right_kind in self.used_modes:
                right_sectors = [sector for sector, (_, right) in kinds.items() if right == right_kind]
                if not right_sectors:
                    continue
                for rs in ranges[right_kind]:
                    right = self._v(endpoint, rs, right_kind)
                    shifted = xp.roll(right, (-dz, -dy, -dx), axis=(1, 2, 3))
                    transported = xp.einsum("zyxab,fzyxb->fzyxa", transport, shifted)
                    for sector in right_sectors:
                        left_kind, _ = kinds[sector]
                        if left_kind == "P":
                            base = xp.einsum("pab,epb->pae", transport[pz, py, px], shifted[:, pz, py, px], optimize=True)
                            for mi, mom in enumerate(self.momentum_list):
                                emit(pi, mi, sector, (whole, whole, rs), base * self._phase(mom, path.displacement)[pz, py, px][:, None, None])
                            continue
                        for ls in ranges[left_kind]:
                            left = self._v(t, ls, left_kind)
                            for mi, mom in enumerate(self.momentum_list):
                                if (left_kind == right_kind and getattr(self.vector_bases[left_kind], "basis_orthogonal", False) is True
                                        and not path.steps and tuple(mom) == (0, 0, 0)):
                                    value = xp.zeros((ls.stop - ls.start, rs.stop - rs.start), complex)
                                    if ls == rs:
                                        indices = xp.arange(ls.stop - ls.start)
                                        value[indices, indices] = xp.sum(left.conj() * left, axis=(1, 2, 3, 4))
                                else:
                                    value = bilinear(left, transported, self._phase(mom, path.displacement), array_module=xp)
                                if sector == "v2v" and self.mode_pair_weights is not None:
                                    value *= self.mode_pair_weights[ls, rs]
                                emit(pi, mi, sector, (ls, rs), value)
            for sector, (left_kind, right_kind) in kinds.items():
                if right_kind != "P" or left_kind == "P":
                    continue
                for ls in ranges[left_kind]:
                    base = xp.einsum("epa,pab->epb", self._v(t, ls, left_kind)[:, bz, by, bx].conj(), transport[bz, by, bx], optimize=True)
                    for mi, mom in enumerate(self.momentum_list):
                        emit(pi, mi, sector, (ls, whole, whole), base * self._phase(mom, path.displacement)[bz, by, bx][None, :, None])
        if result is not None and "p2p" in result:
            result["p2p"] = tuple(tuple(row) for row in result["p2p"])
        return result

    def _calc_recipe(self, t, recipes, *, kind):
        t = self._time(t)
        if self._vectors is None:
            raise ValueError("spatial recipes require eigenvectors")
        xp, ne = self.xp, self.usedNe
        result = self._buffer(kind, (len(recipes), self.num_momentum, ne, ne))
        result.fill(0)
        ranges = [slice(i, min(i + self.ne_block_size, ne)) for i in range(0, ne, self.ne_block_size)]
        for ls in ranges:
            left = self._v(t, ls)
            for rs in ranges:
                right = self._v(t, rs)
                for ri, recipe in enumerate(recipes):
                    pairs = ((differentiate(left, a, self._workspace.links, t, xp),
                              differentiate(right, b, self._workspace.links, t, xp), c)
                             for a, b, c in derivative_assignments(recipe))
                    for a, b, c in pairs:
                        for mi, mom in enumerate(self.momentum_list):
                            phase = self._phase(mom, convention=("none" if self.phase_convention == "none" else "anchor"))
                            result[ri, mi, ls, rs] += c * bilinear(a, b, phase, array_module=xp)
        if self.mode_pair_weights is not None:
            result *= self.mode_pair_weights[:ne,:ne]
        return result

    def _calc_deriv(self, t):
        t=self._time(t)
        xp,ne=self.xp,self.usedNe
        volume=np.prod(self.latt_size[:3])
        field_bytes=ne*volume*3*np.dtype(np.complex128).itemsize
        # Cache every ordered subderivative only while the bound remains
        # modest. Larger bases retain the streaming tiled evaluation path.
        if field_bytes*len(self.derivative_list)>1024**3:
            return self._calc_recipe(t,self.derivative_list,kind="derivative")
        output=self._buffer("derivative",(len(self.derivative_list),self.num_momentum,ne,ne))
        output.fill(0)
        base=self._v(t,slice(0,ne))
        fields={():base}
        for directions in self.derivative_list[1:]:
            prefix,d=directions[:-1],directions[-1]
            fields[directions]=covariant_derivative(fields[prefix],d,self._workspace.links,t,xp)
        phase_convention="none" if self.phase_convention=="none" else "anchor"
        for di,directions in enumerate(self.derivative_list):
            for left,right,coefficient in derivative_assignments(directions):
                lvalue,rvalue=fields[left],fields[right]
                for mi,momentum in enumerate(self.momentum_list):
                    value=bilinear(lvalue,rvalue,self._phase(momentum,convention=phase_convention),array_module=xp)
                    if self.mode_pair_weights is not None:
                        value*=self.mode_pair_weights[:ne,:ne]
                    output[di,mi]+=coefficient*value
        return output

    def _calc_projected(self, rows, t, *, write=None):
        """Fuse spatial path rows before mode-pair contraction.

        rows: sequence of {id: consecutive integer, terms: [{path: steps,
        coefficient: [real,imag]}]}. One row/momentum/mode tile at a time.
        write(row,momentum,'v2v',(left_slice,right_slice),value) is optional.
        """
        t = self._time(t)
        if self._vectors is None:
            raise ValueError("vectors and consecutive nonempty projection rows are required")
        parsed = []
        for terms in spatial_projection_rows(rows):
            merged = {}
            for path, c in terms:
                merged[path] = merged.get(path, 0) + c
            parsed.append(merged)
        xp,ne=self.xp,self.usedNe
        out=None if write is not None else self._buffer("projected",(len(rows),self.num_momentum,ne,ne))
        paths={path for row in parsed for path in row}
        if (getattr(self.eigenvector, "basis_orthogonal", False) is True
                and all(not path.steps for path in paths)
                and self.momentum_list == ((0, 0, 0),)):
            vectors = self._v(t, slice(0, ne)).reshape(ne, -1)
            norms = xp.sum(vectors.conj() * vectors, axis=1)
            indices = xp.arange(ne)
            for row, recipe in enumerate(parsed):
                value = xp.zeros((ne, ne), dtype=xp.complex128)
                coefficient = sum(recipe.values())
                value[indices, indices] = coefficient * norms
                if self.mode_pair_weights is not None:
                    value *= self.mode_pair_weights[:ne, :ne]
                if write is None:
                    out[row, 0] = value
                else:
                    write(row, 0, "v2v", (slice(None), slice(None)), value)
            return out
        from .projection import project_rows
        return project_rows(self, parsed, t, out, write=write)
