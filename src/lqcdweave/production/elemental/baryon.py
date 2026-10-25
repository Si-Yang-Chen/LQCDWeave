"""Equal-time three-leg elemental production using shared gauge transport."""

from collections import OrderedDict
from itertools import product
import numpy as np

from lqcdweave.contracts.elemental import BaryonComponent, integer, normalize_momenta
from lqcdweave.contracts.time_selection import TimeSelectionRecord, normalize_time_selection
from .workspace import ElementalWorkspace
from .kernels import trilinear
from .structures import displace
from .transport import path_transport, apply_transport


def _triple_counts(value, name):
    if isinstance(value, (int, np.integer)):
        value = (value,) * 3
    value = tuple(integer(n, name, 1) for n in value)
    if len(value) != 3:
        raise ValueError(f"{name} requires one count or three counts")
    return value


class _BaryonEngine:
    """Produce (component,momentum,N0,N1,N2) buffers or stream mode tiles.

    eigenvector is one handle shared by all legs, or three handles. By default,
    input snapshots own selected modes; optional mode reads use one tile per leg.
    Gauge loading, preparation, phases and output allocation use the same
    workspace as meson production. Explicit selections map compact input slots
    to physical times; no selection is inferred from array extents.
    """

    def __init__(self, latt_size, gauge_field, eigenvector,
                 momentum_list=((0, 0, 0),), usedNe=None, paths=None, *,
                 flavors=("q", "q", "q"),
                 array_module=np, ne_block_size=16, spatial_block_size=256,
                 phase_convention="anchor", path_boundary="periodic",
                 cache_bytes=64*1024**2, stream_eigenvectors=False,
                 vector_scratch_limit_bytes=None, selection=None):
        if phase_convention not in ("anchor", "none"):
            raise ValueError("baryon phase must be anchor or none")
        self._workspace = ElementalWorkspace(latt_size, gauge_field,
            array_module=array_module, phase_convention=phase_convention)
        if path_boundary not in ("periodic", "open"):
            raise ValueError("invalid path boundary")
        self.latt_size = self._workspace.latt_size
        self.Lt = self.latt_size[3]
        self.time_selection = (TimeSelectionRecord(self.Lt, tuple(range(self.Lt)))
                               if selection is None else normalize_time_selection(selection, Lt=self.Lt))
        self.Nt = self.time_selection.Nt
        self.momentum_list = normalize_momenta(momentum_list)
        self.num_momentum = len(self.momentum_list)
        self.phase_convention, self.path_boundary = phase_convention, path_boundary
        self.xp = array_module
        if not isinstance(stream_eigenvectors, (bool, np.bool_)):
            raise ValueError("stream_eigenvectors must be boolean")
        self.stream_eigenvectors = bool(stream_eigenvectors)
        method = "read_modes" if stream_eigenvectors else "load"
        self.eigenvectors = ((eigenvector,)*3 if callable(getattr(eigenvector, method, None))
                            else tuple(eigenvector))
        if len(self.eigenvectors) != 3 or any(not callable(getattr(h, method, None)) for h in self.eigenvectors):
            raise ValueError(f"one eigenvector handle or three handles with {method} required")
        self.paths = tuple(BaryonComponent.paths(p).key for p in
                           ((((), (), ()),) if paths is None else paths))
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("nonempty unique path triples required")
        self.flavors = tuple(str(f) for f in flavors)
        if len(self.flavors) != 3 or any(not f for f in self.flavors):
            raise ValueError("three quark flavor labels are required")
        self.components = tuple(BaryonComponent(p, self.flavors) for p in self.paths)
        self.num_disp = len(self.paths)
        self._requestedNe = None if usedNe is None else _triple_counts(usedNe, "usedNe")
        self.ne_block_size = _triple_counts(ne_block_size, "ne_block_size")
        self.spatial_block_size = integer(spatial_block_size, "spatial_block_size", 1)
        self.cache_bytes = integer(cache_bytes, "cache_bytes", 0)
        if vector_scratch_limit_bytes is not None:
            vector_scratch_limit_bytes = integer(vector_scratch_limit_bytes,
                                                 "vector_scratch_limit_bytes", 1)
            if not stream_eigenvectors:
                raise ValueError("vector scratch limit requires stream_eigenvectors=True")
        self.vector_scratch_limit_bytes = vector_scratch_limit_bytes
        self.vector_scratch_bytes = 0
        self._vector_blocks = None
        self._vectors = None
        self.usedNe = None
        self.key = None

    def load(self, key):
        self._vectors, self.usedNe, self.key = None, None, None
        self._vector_blocks = None
        self.vector_scratch_bytes = 0
        try:
            return self._load_inputs(key)
        except Exception:
            self._workspace.invalidate()
            raise

    def _load_inputs(self, key):
        self._workspace.load(key)
        lx, ly, lz, lt = self.latt_size
        fulltime = TimeSelectionRecord(lt, tuple(range(lt)))
        self._vector_times = tuple(
            fulltime if getattr(handle, "time_selection", None) is None else
            normalize_time_selection(handle.time_selection, Lt=lt)
            for handle in self.eigenvectors)
        for selection in self._vector_times:
            selection.require_times(self.time_selection.times)
        if self.stream_eigenvectors:
            counts = []
            for leg, handle in enumerate(self.eigenvectors):
                available = integer(getattr(handle, "Ne"), "Ne", 1)
                count = available if self._requestedNe is None else self._requestedNe[leg]
                if count > available:
                    raise ValueError("usedNe exceeds available modes")
                counts.append(count)
            capacities = tuple(min(n, b) for n, b in zip(counts, self.ne_block_size))
            required = sum(capacities) * lz * ly * lx * 3 * np.dtype(complex).itemsize
            if (self.vector_scratch_limit_bytes is not None and
                    required > self.vector_scratch_limit_bytes):
                raise ValueError("vector staging scratch exceeds vector_scratch_limit_bytes")
            self._vector_blocks = tuple(self.xp.empty((n, lz, ly, lx, 3), dtype=complex)
                                        for n in capacities)
            self.vector_scratch_bytes = required
            self._vectors, self.usedNe, self.key = self.eigenvectors, tuple(counts), key
            return self
        snapshots, counts, groups = {}, [], []
        for leg, handle in enumerate(self.eigenvectors):
            identity = id(handle)
            if identity not in snapshots:
                raw = handle.load(key)
                shape = tuple(raw.shape)
                if len(shape) != 6 or shape[0] != self._vector_times[leg].Nt or shape[2:] != (lz, ly, lx, 3) or shape[1] < 1:
                    raise ValueError("vectors require (input_Nt,Ne,Lz,Ly,Lx,3) matching time_selection")
                legs = [i for i, h in enumerate(self.eigenvectors) if h is handle]
                count = shape[1] if self._requestedNe is None else max(self._requestedNe[i] for i in legs)
                if count > shape[1]:
                    raise ValueError("usedNe exceeds available modes")
                # Own immediately: another handle can use the same scratch memory.
                raw = raw[:, :count]
                owner = self.xp if isinstance(raw, self.xp.ndarray) else np
                snapshots[identity] = owner.array(raw, dtype=complex, copy=True)
            group = snapshots[identity]
            count = group.shape[1] if self._requestedNe is None else self._requestedNe[leg]
            groups.append(group)
            counts.append(count)
        self._vectors, self.usedNe, self.key = tuple(groups), tuple(counts), key
        return self

    def _time(self, t):
        if self._vectors is None:
            raise ValueError("load inputs before calculation")
        t = self._workspace.time(t)
        self.time_selection.slot(t)
        return t

    def stout_smear(self, nstep, rho, *, matrix_batch_size=4096):
        if self._vectors is None:
            raise ValueError("load inputs before transforming gauge")
        self._workspace.stout_smear(nstep, rho, matrix_batch_size=matrix_batch_size)

    def project_SU3(self, *, matrix_batch_size=4096):
        if self._vectors is None:
            raise ValueError("load inputs before transforming gauge")
        self._workspace.project_SU3(matrix_batch_size=matrix_batch_size)

    def calc_path_blocks(self, t, blocks=("vvv",), *, rows=None, distances=None, write=None):
        """Compute path triples, projected rows, or distance triples.

        Return a vvv sector dictionary, or synchronously consume borrowed tiles
        with write(component,momentum,'vvv',three_slices,value).
        """
        if tuple(blocks) != ("vvv",):
            raise ValueError("baryon production supports only vvv")
        if rows is not None and distances is not None:
            raise ValueError("select projected rows or distances, not both")
        if rows is not None:
            result = self._calc_projected(rows,t,write=write)
            return None if write is not None else {"vvv": result}
        if distances is not None:
            result = self._calc_distance(t,distances,write=write)
            return None if write is not None else {"vvv": result}
        rows = [((1, tuple(("path", p) for p in paths)),) for paths in self.paths]
        result = self._calculate(t, rows, "vvv", write)
        return None if result is None else {"vvv": result}

    def _calc_projected(self, rows, t, *, write=None):
        """Rows contain id and terms with paths=[P0,P1,P2], coefficient=[real,imag].

        Each correlated path triple is contracted before the row sum. This
        preserves correlations between legs without generating cross terms.
        Store projected rows with explicit BaryonComponent keys of your choice.
        """
        rows = tuple(rows)
        if not rows or [r["id"] for r in rows] != list(range(len(rows))):
            raise ValueError("nonempty consecutive projection rows required")
        parsed = []
        for row in rows:
            merged = {}
            for term in row["terms"]:
                paths = BaryonComponent.paths(term["paths"], flavors=self.flavors).key
                coefficient = complex(*term["coefficient"])
                if not np.isfinite(coefficient):
                    raise ValueError("finite projection coefficient required")
                merged[paths] = merged.get(paths, 0) + coefficient
            if not merged or any(not np.isfinite(c) for c in merged.values()):
                raise ValueError("nonempty finite projection row required")
            parsed.append(tuple((c, tuple(("path", p) for p in paths))
                                for paths, c in merged.items() if c != 0))
        return self._calculate(t, parsed, "projected", write)

    @staticmethod
    def distance_components(distances, *, flavors=("q", "q", "q")):
        recipes = []
        for triple in distances:
            triple = tuple(integer(d, "distance", 0) for d in triple)
            if len(triple) != 3:
                raise ValueError("three leg distances required")
            recipes.append(BaryonComponent(("distance", triple), flavors))
        if not recipes:
            raise ValueError("nonempty distance components required")
        return tuple(recipes)

    def _calc_distance(self, t, distances, *, write=None):
        """Each nonzero leg distance averages its six straight spatial paths."""
        components = self.distance_components(distances, flavors=self.flavors)
        rows = [((1, tuple(("distance", d) for d in c.key[1])),) for c in components]
        return self._calculate(t, rows, "distance", write)

    def _calculate(self, t, rows, name, write):
        t = self._time(t)
        if write is not None and not callable(write):
            raise ValueError("write must be callable")
        xp, workspace = self.xp, self._workspace
        shape = (len(rows), self.num_momentum, *self.usedNe)
        output = workspace.buffer(name, shape) if write is None else None
        ranges = [[slice(i, min(i+b, n)) for i in range(0, n, b)]
                  for n, b in zip(self.usedNe, self.ne_block_size)]
        # Caches live for one call, so gauge preparation/load cannot leave stale fields.
        cache, retained = OrderedDict(), 0

        def cached(key, make):
            nonlocal retained
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            value = make()
            if value.nbytes <= self.cache_bytes:
                while cache and retained + value.nbytes > self.cache_bytes:
                    _, old = cache.popitem(last=False)
                    retained -= old.nbytes
                cache[key] = value
                retained += value.nbytes
            return value

        def field(leg, recipe, span):
            kind, spec = recipe
            def make():
                if self.stream_eigenvectors:
                    try:
                        borrowed = self.eigenvectors[leg].read_modes(
                            self.key, t, span.start, span.stop)
                        expected = (span.stop-span.start, *self.latt_size[2::-1], 3)
                        if tuple(borrowed.shape) != expected or borrowed.dtype.kind not in "fc":
                            raise ValueError("mode block requires (mode,z,y,x,color) data")
                        raw = self._vector_blocks[leg][:span.stop-span.start]
                        raw[...] = xp.asarray(borrowed, dtype=complex)
                        if not bool(xp.isfinite(raw).all()):
                            raise ValueError("mode block must be finite")
                    except Exception:
                        self._vectors = None
                        self._workspace.invalidate()
                        raise
                else:
                    raw = xp.asarray(self._vectors[leg][self._vector_times[leg].slot(t), span], dtype=complex)
                if kind == "path":
                    if not spec.steps:
                        return raw
                    transport = cached(("transport", spec), lambda: path_transport(
                        workspace.links, spec, t, self.path_boundary, array_module=xp))
                    return apply_transport(transport, raw, spec.displacement, array_module=xp)
                return displace(raw, spec, workspace.links, t, xp)
            # A cached view of stream scratch would become stale on the next tile.
            return make() if self.stream_eigenvectors else cached(
                (id(self._vectors[leg]), recipe, span.start, span.stop), make)

        tile_shape = tuple(min(b, n) for b, n in zip(self.ne_block_size, self.usedNe))
        tile_buffer = workspace.buffer("baryon_tile", tile_shape)
        work_buffer = workspace.buffer("baryon_work", tile_shape)
        for ri, terms in enumerate(rows):
            for spans in product(*ranges):
                extent = tuple(slice(0, s.stop-s.start) for s in spans)
                tile, work = tile_buffer[extent], work_buffer[extent]
                for mi, momentum in enumerate(self.momentum_list):
                    tile.fill(0)
                    phase = workspace.phase(momentum)
                    for coefficient, recipes in terms:
                        vectors = tuple(field(leg, recipe, span) for leg, (recipe, span)
                                        in enumerate(zip(recipes, spans)))
                        trilinear(vectors, phase, block_size=self.spatial_block_size,
                                  out=work, array_module=xp)
                        tile += coefficient * work
                    if write is None:
                        output[(ri, mi, *spans)] = tile
                    else:
                        write(ri, mi, "vvv", spans, tile)
        return output
