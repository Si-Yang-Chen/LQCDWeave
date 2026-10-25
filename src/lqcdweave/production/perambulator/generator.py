"""One solve stream, projected onto an ordered list of actual sink bases."""

from copy import deepcopy
from time import perf_counter

import numpy as np

from lqcdweave.contracts.time_selection import normalize_time_selection
from lqcdweave.contracts.propagator import PropagatorEndpoint
from ._inputs import InputReader, _integer, _count, _snapshot_endpoint
from .basis import VectorBasis, PointBasis
from .highmode import HighmodeBasis, project_point_source, sample_high_points
from .products import product_shape
from .result import PropagatorResult, PropagatorSolution


class PerambulatorGenerator(InputReader):
    """Produce from one actual source and any number of actual sinks.

    Inputs are specified at calc time. V/D identity is carried by VectorBasis,
    point identity by PointBasis. Results own their arrays. Backend instances
    belong to one generator; the generator is not reentrant or thread safe.
    """

    def __init__(self, latt_size, gauge_field, mass, tol, maxiter,
                 xi_0=1.0, nu=1.0, clover_coeff_t=0.0, clover_coeff_r=1.0,
                 t_boundary=1, multigrid=None, contract_prec='<c16', *, backend,
                 MRHS=False, rhs_batch_size=None, stream_eigenvectors=False,
                 sink_ne_block_size=32, sink_time_block_size=1,
                 sink_cache_limit_bytes=0, vector_scratch_limit_bytes=None,
                 source_selection=None, sink_selection=None, measure=False,
                 gamma5=None, gamma5_hermitian=None):
        self.latt_size = tuple(_integer(x, 'lattice extent', 1) for x in latt_size)
        if len(self.latt_size) != 4:
            raise ValueError('latt_size must be (Lx,Ly,Lz,Lt)')
        for name, value in (('mass', mass), ('tol', tol), ('xi_0', xi_0), ('nu', nu),
                            ('clover_coeff_t', clover_coeff_t), ('clover_coeff_r', clover_coeff_r)):
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 < tol < 1 or xi_0 <= 0 or nu <= 0:
            raise ValueError('tol must be in (0,1); xi_0 and nu must be positive')
        if isinstance(t_boundary, (bool, np.bool_)) or t_boundary not in (-1, 1):
            raise ValueError('t_boundary must be +1 or -1')
        for name, value in (('MRHS', MRHS), ('measure', measure),
                            ('stream_eigenvectors', stream_eigenvectors)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{name} must be boolean')
        if not callable(getattr(gauge_field, 'load', None)):
            raise ValueError('gauge_field must provide load(key)')
        if backend is None:
            raise ValueError('pass an explicit propagator backend')
        dtype = np.dtype(contract_prec)
        if dtype.kind != 'c' or dtype.itemsize not in (8, 16):
            raise ValueError('contract_prec must be complex64 or complex128')
        self.contract_prec = dtype.str
        self.backend, self.xp, self.gauge_field = backend, backend.array_module, gauge_field
        self.source_selection = normalize_time_selection(source_selection, Lt=self.latt_size[3])
        self.sink_selection = normalize_time_selection(sink_selection, Lt=self.latt_size[3])
        self.source_times, self.sink_times = self.source_selection.times, self.sink_selection.times
        self.MRHS, self.measure = bool(MRHS), bool(measure)
        self.rhs_batch_size = None if rhs_batch_size is None else _integer(rhs_batch_size, 'rhs_batch_size', 1)
        self.stream_eigenvectors = bool(stream_eigenvectors)
        self.sink_ne_block_size = _integer(sink_ne_block_size, 'sink_ne_block_size', 1)
        self.sink_time_block_size = _integer(sink_time_block_size, 'sink_time_block_size', 1)
        if self.sink_time_block_size > self.latt_size[3]:
            raise ValueError('sink_time_block_size must be at most Lt')
        self.sink_cache_limit_bytes = _integer(sink_cache_limit_bytes, 'sink_cache_limit_bytes')
        self.vector_scratch_limit_bytes = (None if vector_scratch_limit_bytes is None else
            _integer(vector_scratch_limit_bytes, 'vector_scratch_limit_bytes', 1))
        if not self.stream_eigenvectors and (self.sink_cache_limit_bytes or self.vector_scratch_limit_bytes is not None):
            raise ValueError('vector scratch/cache limits require stream_eigenvectors=True')
        self._parameters = dict(latt_size=self.latt_size, mass=float(mass), tol=float(tol),
            maxiter=_integer(maxiter, 'maxiter', 1), xi_0=float(xi_0), nu=float(nu),
            clover_coeff_t=float(clover_coeff_t), clover_coeff_r=float(clover_coeff_r),
            t_boundary=int(t_boundary), multigrid=deepcopy(multigrid))
        self._gamma5_hermitian = (getattr(backend, 'gamma5_hermitian', False)
                                  if gamma5_hermitian is None else gamma5_hermitian)
        self._gamma5 = getattr(backend, 'gamma5', None) if gamma5 is None else gamma5
        if self._gamma5_hermitian is not False and self._gamma5_hermitian is not True:
            raise ValueError('gamma5_hermitian must be boolean')
        if self._gamma5_hermitian:
            gamma = self.xp.asarray(self._gamma5)
            if (gamma.shape != (4, 4) or not bool(self.xp.isfinite(gamma).all())
                    or not bool(self.xp.allclose(gamma, gamma.conj().T))
                    or not bool(self.xp.allclose(gamma @ gamma, self.xp.eye(4)))):
                raise ValueError('gamma5 must be a finite Hermitian 4x4 involution')
            self._gamma5 = self.xp.asarray(gamma).copy()
        ordered = getattr(backend, 'stream_ordered_solve', False) is True
        self._synchronize = getattr(backend, 'synchronize', lambda: None) if self.measure or not ordered else lambda: None
        self._generation = 0
        self._active = False
        self._rhs = self._solution = None
        self._invalidate_loaded_state()

    def _invalidate_loaded_state(self):
        self._generation += 1
        self._loaded, self._key = False, None
        self.gauge_field_smear, self.gauge_field_new = None, False
        self._gauge_id = None
        self._gauge_transforms = []
        self._sink_cache = {}
        self.sink_cache_used_bytes = 0
        self.last_metrics = {}

    def load(self, key):
        """Load the gauge; actual bases are staged when calc receives them."""
        if self._active:
            raise RuntimeError('cannot load during a calculation')
        self._invalidate_loaded_state()
        gauge = self.gauge_field.load(key)
        self.gauge_field_smear, self.gauge_field_new = gauge, True
        self._gauge_id = deepcopy(getattr(self.gauge_field, 'gauge_id', None))
        self._loaded, self._key = True, key

    def stout_smear(self, nstep, rho, dir_ignore=3):
        if self._active:
            raise RuntimeError('cannot smear during a calculation')
        if not self._loaded:
            raise RuntimeError('load a configuration successfully before smearing')
        nstep, dir_ignore = _integer(nstep, 'nstep'), _integer(dir_ignore, 'dir_ignore')
        if dir_ignore > 4 or not np.isfinite(rho):
            raise ValueError('dir_ignore must be 0..4 and rho finite')
        transform = getattr(self.backend, 'stout_smear', None)
        if not callable(transform):
            raise NotImplementedError('backend does not support stout_smear')
        try:
            self.gauge_field_smear = transform(self.gauge_field_smear, nstep=nstep,
                                               rho=float(rho), dir_ignore=dir_ignore)
        except Exception:
            self._invalidate_loaded_state()
            raise
        self._generation += 1
        self.gauge_field_new = True
        self._gauge_transforms.append(dict(nstep=nstep, rho=float(rho), dir_ignore=dir_ignore))

    def _stage(self, basis, selection):
        if not isinstance(basis, (VectorBasis, PointBasis)):
            raise TypeError('src and each contracted sink must be VectorBasis or PointBasis')
        point = isinstance(basis, PointBasis)
        handle = basis.handle
        count = _count(handle, basis.count, 'Np' if point else 'Ne', 'basis.count')
        if count < 1:
            raise ValueError('a basis must contain at least one vector/point')
        self._input_selection(handle, selection)
        if point:
            data = self._read_points(handle, count, self._key, selection)
        elif self.stream_eigenvectors:
            if not callable(getattr(handle, 'read_modes', None)):
                raise ValueError('streamed vector handles must provide read_modes')
            data = None
        else:
            data = self._read_vectors(handle, count, self._key, selection)
        declared = basis.endpoint or getattr(handle, 'endpoint', None)
        if declared is not None and not isinstance(declared, PropagatorEndpoint):
            raise TypeError('basis endpoint must be a PropagatorEndpoint')
        known_kind = ('D' if getattr(handle, 'dilution_source', None) is not None else
                      getattr(handle, 'vector_kind', None))
        kind = declared.kind if declared is not None else 'P' if point else known_kind or 'V'
        if not point and known_kind is not None and kind != known_kind:
            raise ValueError('basis endpoint differs from the input vector representation')
        if (point and kind != 'P') or (not point and kind not in ('V', 'D')):
            raise ValueError('basis type differs from its actual endpoint')
        if basis.endpoint is not None and getattr(handle, 'endpoint', None) is not None and handle.endpoint != declared:
            raise ValueError('explicit basis endpoint differs from the handle')
        endpoint = _snapshot_endpoint(handle, kind, count, endpoint=declared)
        high = basis.highmode_basis if point else None
        high_data = None
        if high is not None:
            if not isinstance(high, HighmodeBasis):
                raise TypeError('point highmode_basis must be HighmodeBasis')
            endpoint = high.point_endpoint(endpoint)
            self._input_selection(high, selection)
            if not self.stream_eigenvectors or not high.supports_mode_reads:
                high_data = self._read_vectors(high, high.Ne, self._key, selection)
        elif 'high_mode_projection' in endpoint.metadata:
            raise ValueError('a high point endpoint requires its actual highmode_basis')
        return dict(basis=basis, count=count, endpoint=endpoint, data=data,
                    selection=selection, high=high, high_data=high_data, high_cache=None)

    def endpoint(self, basis, *, source=False):
        """Resolve a configured basis for storage planning, without solving."""
        if not self._loaded:
            raise RuntimeError('load a configuration before resolving its endpoint')
        return self._stage(basis, self.source_selection if source else self.sink_selection)['endpoint']

    def _high_at(self, state, time):
        if state['high_data'] is not None:
            return state['high_data'][state['selection'].slot(time)]
        cached = state['high_cache']
        if cached is None or cached[0] != time:
            high = state['high']
            lx, ly, lz, _ = self.latt_size
            scratch = self.backend.empty((high.Ne, lz, ly, lx, 3), np.dtype(self.contract_prec))
            cached = (time, self._read_mode_block(high, time, 0, high.Ne, scratch))
            state['high_cache'] = cached
        return cached[1]

    def _sinks(self, sink, staged=None):
        if sink is None:
            return [None]
        if not isinstance(sink, list) or not sink:
            raise TypeError('sink must be a nonempty list of actual bases/None, or None')
        staged = {} if staged is None else staged
        result = []
        for basis in sink:
            if basis is None:
                result.append(None)
                continue
            key = (id(basis), self.sink_times)
            if key not in staged:
                staged[key] = self._stage(basis, self.sink_selection)
            result.append(staged[key])
        return result

    def _scratch(self, sinks, source_vector):
        lx, ly, lz, _ = self.latt_size
        modes = max((min(s['count'], self.sink_ne_block_size)
            for s in sinks if s is not None and s['endpoint'].kind != 'P'), default=0)
        capacity = int(source_vector) + modes
        self.vector_scratch_bytes = capacity*lz*ly*lx*3*np.dtype(self.contract_prec).itemsize if self.stream_eigenvectors else 0
        if self.vector_scratch_limit_bytes is not None and self.vector_scratch_bytes > self.vector_scratch_limit_bytes:
            raise ValueError('vector staging scratch exceeds vector_scratch_limit_bytes')
        self._source_scratch = self._sink_scratch = None
        if self.stream_eigenvectors:
            if source_vector:
                self._source_scratch = self.backend.empty((1, lz, ly, lx, 3), np.dtype(self.contract_prec))
            if modes:
                self._sink_scratch = self.backend.empty((modes, lz, ly, lx, 3), np.dtype(self.contract_prec))

    def _outputs(self, source, t_src, sinks, write, generation):
        outputs = []
        lx, ly, lz, lt = self.latt_size
        for state in sinks:
            if state is None:
                if write is not None:
                    raise ValueError('uncontracted fields require retained output; use sink bases with write')
                shape = (source['count'], 4) + ((3,) if source['endpoint'].kind == 'P' else ())
                shape += (lt, lz, ly, lx, 4, 3)
                def project(solution, sink, write, generation=generation):
                    if not self._loaded or self._generation != generation:
                        raise RuntimeError('load/smear changed the configuration; solution projection requires its original generator state')
                    return self._project_solution(solution, sink, write)
                outputs.append(PropagatorSolution(self.backend.empty(shape, np.complex128),
                    source['endpoint'], t_src, project))
            else:
                endpoint = state['endpoint']
                product = endpoint.kind + 'S' + source['endpoint'].kind
                shape = (self.sink_selection.Nt,) + product_shape(product, state['count'], source['count'])
                values = None if write is not None else self.backend.empty(shape, np.dtype(self.contract_prec))
                outputs.append(PropagatorResult(values, source['endpoint'], endpoint, t_src,
                    self.sink_times, self._gamma5, self._gamma5_hermitian, self.xp))
        return outputs

    def _emit(self, output, index, selection, value, write):
        if write is None:
            output.values[selection] = value
        else:
            write(index, selection, self.xp.asarray(value, dtype=self.contract_prec))

    def _contract(self, solution, source_index, first, source_kind, state, output, index, write):
        point_source = source_kind == 'P'
        point_sink = state['endpoint'].kind == 'P'
        for t0 in range(0, len(self.sink_times), self.sink_time_block_size):
            for slot in range(t0, min(t0 + self.sink_time_block_size, len(self.sink_times))):
                time = self.sink_times[slot]
                if point_sink:
                    points = state['data'][slot]
                    if state['high'] is None:
                        x, y, z = points[:, 0], points[:, 1], points[:, 2]
                        projected = solution[:, time, z, y, x].transpose(2, 0, 1, 3)
                    else:
                        projected = sample_high_points(solution[:, time], points,
                            self._high_at(state, time), array_module=self.xp).transpose(2, 0, 1, 3)
                    blocks = [(0, state['count'], projected)]
                else:
                    blocks = self._vector_projections(solution[:, time], state, slot, time)
                for start, stop, projected in blocks:
                    for r in range(len(solution)):
                        spin, color = divmod(first + r, 3) if point_source else (first + r, None)
                        selection = (slice(slot, slot+1), slice(None), slice(spin, spin+1), slice(start, stop))
                        value = projected[:, r][None, :, None]
                        if point_sink:
                            selection += (slice(None),)
                        selection += (slice(source_index, source_index+1),)
                        value = value[..., None]
                        if point_source:
                            selection += (slice(color, color+1),)
                            value = value[..., None]
                        self._emit(output, index, selection, value, write)

    def _vector_projections(self, solution, state, slot, time):
        for start in range(0, state['count'], self.sink_ne_block_size):
            stop = min(start + self.sink_ne_block_size, state['count'])
            if state['data'] is not None:
                vectors = state['data'][slot, start:stop].conj()
            else:
                key = (id(state), time, start, stop)
                vectors = self._sink_cache.get(key)
                if vectors is None:
                    vectors = self._read_mode_block(state['basis'].handle, time, start, stop, self._sink_scratch)
                    self.xp.conjugate(vectors, out=vectors)
                    if self.sink_cache_used_bytes + vectors.nbytes <= self.sink_cache_limit_bytes:
                        self._sink_cache[key] = vectors.copy()
                        self.sink_cache_used_bytes += vectors.nbytes
            yield start, stop, self.xp.einsum('kzyxc,rzyxsc->srk', vectors, solution, optimize=True)

    def _deliver(self, solution, source_index, first, source_kind, sinks, outputs, write):
        width = 12 if source_kind == 'P' else 4
        for index, (state, output) in enumerate(zip(sinks, outputs)):
            if state is None:
                fields = output.values[source_index].reshape((width,) + tuple(solution.shape[1:]))
                fields[first:first+len(solution)] = solution
            else:
                self._contract(solution, source_index, first, source_kind, state, output, index, write)

    def calc(self, t_src, *, src, sink, write=None):
        """Solve one source and project every requested sink using the same RHS.

        sink is an ordered nonempty list of VectorBasis, PointBasis or None.
        None retains the full solution. Returns results in the same order.
        write(index, selection, value) optionally consumes borrowed tiles and
        suppresses full projected allocations. A failed call returns no result;
        a writer must discard any partial tiles it has already received.
        """
        if self._active:
            raise RuntimeError('calc is not reentrant')
        self._active = True
        try:
            return self._calculate(t_src, src=src, sink=sink, write=write)
        finally:
            self._active = False

    def _calculate(self, t_src, *, src, sink, write):
        if not self._loaded:
            raise RuntimeError('load a configuration successfully before calculation')
        if write is not None and not callable(write):
            raise TypeError('write must be callable')
        t_src = _integer(t_src, 't_src')
        self.source_selection.slot(t_src)
        self.last_metrics = {}
        self._sink_cache.clear()
        self.sink_cache_used_bytes = 0
        source = self._stage(src, self.source_selection)
        sinks = self._sinks(sink, {(id(src), self.source_times): dict(source)})
        source_kind = source['endpoint'].kind
        width = 12 if source_kind == 'P' else 4
        batch = min(width, self.rhs_batch_size or width) if self.MRHS else 1
        lx, ly, lz, lt = self.latt_size
        self._scratch(sinks, source_kind != 'P')
        outputs = self._outputs(source, t_src, sinks, write, self._generation)
        shape = (batch, lt, lz, ly, lx, 4, 3)
        if self._rhs is None or self._rhs.shape != shape:
            self._rhs = self.backend.zeros(shape, np.complex128)
            self._solution = self.backend.empty(shape, np.complex128)
        # Assemble the four spin solves of one spatial vector before projecting.
        # This retains the old bounded workspace and avoids rereading every sink
        # basis four times when the solver only supports single RHS calls.
        vector_solution = (self.backend.empty((4,) + shape[1:], np.complex128)
                           if source_kind != 'P' and batch < 4 else None)
        if self.gauge_field_new:
            try:
                self.backend.load(self.gauge_field_smear, deepcopy(self._parameters))
            except Exception:
                self._invalidate_loaded_state()
                raise
            self.gauge_field_new = False
        started = perf_counter()
        for source_index in range(source['count']):
            if source_kind != 'P':
                if source['data'] is None:
                    mode = self._read_mode_block(src.handle, t_src, source_index, source_index+1, self._source_scratch)[0]
                else:
                    mode = source['data'][self.source_selection.slot(t_src), source_index]
            else:
                x, y, z = (int(c) for c in source['data'][self.source_selection.slot(t_src), source_index])
            for first in range(0, width, batch):
                count = min(batch, width-first)
                rhs, solution = self._rhs[:count], self._solution[:count]
                rhs.fill(0)
                for r in range(count):
                    if source_kind == 'P':
                        spin, color = divmod(first+r, 3)
                        rhs[r, t_src, z, y, x, spin, color] = 1
                    else:
                        rhs[r, t_src, ..., first+r, :] = mode
                if source['high'] is not None:
                    project_point_source(rhs[:, t_src], self._high_at(source, t_src), array_module=self.xp)
                self._synchronize()
                self.backend.solve_into(rhs, solution, tol=self._parameters['tol'], maxiter=self._parameters['maxiter'])
                self._synchronize()
                if vector_solution is not None:
                    vector_solution[first:first+count] = solution
                else:
                    self._deliver(solution, source_index, first, source_kind, sinks, outputs, write)
            if vector_solution is not None:
                self._deliver(vector_solution, source_index, 0, source_kind, sinks, outputs, write)
        self._synchronize()
        self.last_metrics = dict(source_time=t_src, rhs_count=source['count']*width,
            sink_times=self.sink_times, sinks=len(sinks), elapsed_seconds=perf_counter()-started,
            timing_mode='synchronized' if self.measure else 'submission')
        return outputs

    def _project_solution(self, solution, sink, write):
        if self._active:
            raise RuntimeError('projection is not reentrant')
        self._active = True
        try:
            return self._project_fields(solution, sink, write)
        finally:
            self._active = False

    def _project_fields(self, solution, sink, write):
        if write is not None and not callable(write):
            raise TypeError('write must be callable')
        self._sink_cache.clear()
        self.sink_cache_used_bytes = 0
        states = self._sinks(sink)
        if any(state is None for state in states):
            raise ValueError('project requires contracted sink bases')
        self._scratch(states, False)
        source = dict(count=len(solution.source.indices), endpoint=solution.source)
        outputs = self._outputs(source, solution.source_time, states, write, self._generation)
        width = 12 if solution.source.kind == 'P' else 4
        for index in range(source['count']):
            fields = solution.values[index].reshape((width,) + solution.values.shape[-6:])
            for k, (state, output) in enumerate(zip(states, outputs)):
                self._contract(fields, index, 0, solution.source.kind, state, output, k, write)
        return outputs

    def describe(self):
        """Describe the operator; endpoint records belong to each calc result."""
        return deepcopy(dict(loaded=self._loaded, key=self._key, gauge_id=self._gauge_id,
            lattice=self.latt_size, parameters=self._parameters, contract_prec=self.contract_prec,
            source_times=self.source_times, sink_times=self.sink_times,
            gauge_transforms=self._gauge_transforms, gamma5_hermitian=self._gamma5_hermitian))
