"""Managed elemental inputs: describe first, reuse or produce before execution.

Production and persistence are internal workflow dependencies. The public
declaration contains scientific inputs, not user-supplied preparation callbacks.
"""

from collections import defaultdict
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental, ArrayBaryonElemental
from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle
from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec, stable_digest
from lqcdweave.contracts.elemental import BaryonComponent, elemental_operator_rows, normalize_momenta, integer
from lqcdweave.contracts.elemental_sectors import sector_kinds
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec, SpatialSamplingDesign
from lqcdweave.planning.production import validate_elemental_fields, validate_point_field
from lqcdweave.planning.dataset import plan_requirement
from lqcdweave.production.elemental.api import ElementalGenerator, BaryonElementalGenerator, describe_elemental_structures
from lqcdweave.storage.dataset import DatasetIndex
from lqcdweave.runtime.dispatch import AtomicOpen
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.runtime.datasets import run_dataset_requests, open_dataset_view, plan_dataset_request


@dataclass(frozen=True)
class _Header:
    shape: tuple
    dtype: object = np.dtype("complex128")

    @property
    def ndim(self):
        return len(self.shape)


class _SliceBasis:
    def __init__(self, handle, cfg, time, xp):
        self.basis_id, self.mode_labels = handle.basis_id, handle.mode_labels
        self.basis_orthogonal = handle.basis_orthogonal
        self.dilution_source = handle.dilution_source
        self.value = xp.array(handle.read_modes(cfg, time, 0, handle.Ne), copy=True)[None]

    def load(self, _cfg):
        return self.value


class Elemental(ArrayElemental):
    """Declare registered inputs; calculate() owns missing spatial production.

    gauge and bases are open DatasetReader/View objects; bases is either one V
    reader or a mapping with V/D keys. A rank-three catalog accepts one shared
    V reader or three ordered V readers. points is an optional registered point
    reader. The shared directory contains standard Dataset products and index.
    Borrowed parent readers must remain open. Spatial meson/current sectors are
    inferred from the supplied V/D/P inputs; no prepare callback is accepted.
    """

    background_read_safe = False
    requires_preparation = True

    def __new__(cls, *arguments, **options):
        operator_table = options.get("operator_table", arguments[2] if len(arguments) > 2 else None)
        if cls is Elemental and getattr(getattr(operator_table, "catalog", None), "rank", 2) == 3:
            return object.__new__(_BaryonElemental)
        return object.__new__(cls)

    def __init__(self, gauge, bases, operator_table, momentum_dict, *, directory,
                 points=None, stout=(0, 0.12), phase_convention="anchor"):
        self.gauge = gauge
        self.configuration = gauge.spec.document["configuration"]
        self.lattice = tuple(self.configuration["lattice"])
        self.bases = dict(bases) if isinstance(bases, Mapping) else {"V": bases}
        if not self.bases or set(self.bases) - {"V", "D"}:
            raise ValueError("bases requires registered V and/or D fields")
        self.bases = {kind: self.bases[kind] for kind in ("V", "D") if kind in self.bases}
        self.handles = {}
        for kind, dataset in self.bases.items():
            validate_elemental_fields(gauge.spec, dataset.spec)
            handle = self.handles[kind] = DatasetFieldHandle(dataset)
            if kind == "D" and handle.dilution_source is None:
                raise ValueError("D requires actual dilution provenance")
        self.gauge_handle = DatasetFieldHandle(gauge)
        self.points = points
        self.point_handle = None
        if points is not None:
            validate_point_field(gauge.spec, points.spec)
            self.point_handle = DatasetPointHandle(points)
        self.rows = elemental_operator_rows(operator_table)
        self.catalog = getattr(operator_table, "catalog", None)
        if any(step >= 6 for _, terms in self.rows for path, _ in terms for step in path.steps):
            raise NotImplementedError("managed Elemental currently requires spatial paths")
        self.momentum_list = normalize_momenta(momentum_dict)
        self.stout = (integer(stout[0], "stout steps", 0), float(stout[1]))
        if not np.isfinite(self.stout[1]):
            raise ValueError("stout rho must be finite")
        self.directory = Path(directory)
        self.path_boundary = "periodic"
        self._tables = {c: (self.catalog.table((c.key[-1],)) if self.catalog is not None else
                           ({"component": c, "terms": tuple(
                               {"path": p.steps, "coefficient": z} for p, z in terms)},))
                        for c, terms in self.rows}
        kinds = tuple(kind.lower() for kind in self.bases) + (("p",) if points is not None else ())
        counts = {kind.lower(): handle.Ne for kind, handle in self.handles.items()}
        if points is not None:
            counts["p"] = self.point_handle.Np
        sectors, endpoints, dilution = {}, {}, {}
        for left in kinds:
            for right in kinds:
                sector = left + "2" + right
                shape = tuple(3 if axis.startswith("color") else counts[
                    left if axis.endswith("left") else right] for axis in self.sector_tail_axes[sector])
                sectors[sector] = _Header((len(self.rows), len(self.momentum_list), self.lattice[3], *shape))
                for side, kind in (("bar", left), ("field", right)):
                    dataset = points if kind == "p" else self.bases[kind.upper()]
                    labels = tuple(dataset.spec.coordinates("point" if kind == "p" else "mode"))
                    endpoints[sector, side] = dataset.data_id, labels
                    if kind == "d":
                        dilution[sector, side] = self.handles["D"].dilution_source
        point_diagonal = tuple(c for c, terms in self.rows if all(
            all(d % n == 0 for d, n in zip(p.displacement[:3], self.lattice[:3])) for p, _ in terms))
        structures = describe_elemental_structures(self.rows, self.momentum_list, tuple(sectors),
            orthogonal_bases={k: h.basis_orthogonal for k, h in self.handles.items()},
            point_diagonal_components=point_diagonal)
        pools, groups = self._point_pools()
        pool_bindings = {(c, mi, t, side): pool for t, pool in pools.items()
                         for c, _ in self.rows for mi in range(len(self.momentum_list)) for side in ("bar", "field")}
        super().__init__(sectors, tuple(c for c, _ in self.rows), cfg=self.configuration["configuration"],
            phase_convention=phase_convention, momentum_list=self.momentum_list,
            basis_endpoints=endpoints, dilution_endpoints=dilution,
            sampling_pool_ids=pool_bindings, sampling_groups=groups,
            point_diagonal_components=point_diagonal, elemental_structures=structures,
            elemental_weighted=False)
        self._products = {}
        self._pool_by_time = pools
        self._gauge_product = None
        self._provenance = None

    def _for_calculation(self, weighted):
        """Bind the workflow's single weight option without changing a template."""
        if self.elemental_weighted == weighted:
            return self
        if weighted and self.points is not None and not self.sampling_groups:
            raise ValueError("weighted point Elemental requires a registered sampling design")
        selected = copy(self)
        selected.elemental_weighted = weighted
        selected._products = {}
        selected._provenance = None
        return selected

    def _point_pools(self):
        if self.points is None:
            return {}, {}
        definition = self.points.spec.document["definition"]
        realization = definition.get("realization", {})
        sampling = realization.get("sampling", realization)
        design = sampling.get("design")
        if design is None:
            return {}, {}
        design = SpatialSamplingDesign(**design)
        if design.lattice_size != self.lattice or design.num_points != self.point_handle.Np:
            raise ValueError("point sampling design differs from registered geometry/coverage")
        width = design.segment_length
        declared = sampling.get("pool_ids")
        times = self.points.spec.coordinates("time")
        pools = (dict(zip(times, declared, strict=True)) if declared is not None else
                 {t: self.points.data_id + f"/segment-{t // width}" for t in times})
        groups = {pool: SamplingPoolSpec(pool, prod(self.lattice[:3]), self.point_handle.Np)
                  for pool in set(pools.values())}
        return pools, groups

    def block(self, component, momentum, t, sector, **selection):
        self.describe_block(component, momentum, t, sector, **selection)
        product = self._products.get((component, momentum, sector, t))
        if product is None:
            raise RuntimeError("managed Elemental must be executed through calculate()")
        with open_dataset_view(product[0], product[1], subset_axes=("time",), data_id=product[2]) as view:
            value = view.read(time=t).copy()
        from lqcdweave.adapters.selection import select_axes
        return select_axes(value, self.sector_tail_axes[sector], **selection)

    def _spec(self, component, momentum, sector, times):
        tail = self.sector_tail_axes[sector]
        bindings = self.basis_endpoints
        axes = [dict(name="time", coordinates=list(times))]
        bases = {}
        for axis, size in zip(tail, self.sectors[sector].shape[3:], strict=True):
            if axis.startswith("color"):
                labels = list(range(size))
            else:
                identity, labels = bindings[sector, "bar" if axis.endswith("left") else "field"]
                labels = list(labels)
                bases[axis] = dict(data_id=identity, indices=labels)
            axes.append(dict(name=axis, coordinates=labels))
        terms = self.rows[self._index[component]][1]
        kinds = sector_kinds(sector)
        parents = [self.gauge.data_id, *(dataset.data_id for kind, dataset in self.bases.items() if kind.lower() in kinds)]
        if "p" in kinds:
            parents.append(self.points.data_id)
        definition = dict(gauge_id=self.gauge.data_id, product=sector,
                component=repr(component), terms=[dict(path=list(p.steps), coefficient=[z.real, z.imag]) for p, z in terms],
                momentum=list(self.momentum_list[momentum]), stout=list(self.stout),
                dilution={side: source.document for (s, side), source in self.dilution_endpoints.items() if s == sector},
                point_pools=self._pool_by_time if "p" in kinds else {})
        if self.catalog is not None:
            definition["catalog"] = dict(name=self.catalog.name, version=self.catalog.version)
            definition["component"] = component.key[-1]
        return DatasetSpec(dict(schema=1, kind="managed_elemental_" + sector,
            configuration=self.configuration, axes=axes, dtype="complex128", bases=bases,
            definition=definition,
            conventions=dict(phase=self.phase_convention, path_boundary=self.path_boundary,
                elemental_weighted=self.elemental_weighted,
                structure=self.elemental_structure(component, momentum, sector)), parents=parents))

    def read_slab(self, component, momentum, times_selected, sector, **selection):
        selected = ((times_selected,) if isinstance(times_selected, (int, np.integer)) else tuple(times_selected))
        self.describe_slab(component, momentum, selected, sector, **selection)
        return np.stack([self.block(component, momentum, time, sector, **selection) for time in selected])

    def read_slab_workspace_bytes(self, metadata, **_selection):
        # Both individual owned rows and the final stack can be live.
        return int(prod(metadata.shape) * metadata.dtype.itemsize)

    def _workspace_bytes(self, sector):
        # One spatial slice; owned gauge, stout matrices, copied bases, transport,
        # tile operands, phases, sparse records and dense snapshots. Opaque BLAS
        # allocations are covered by the workflow's ordinary runtime reserves.
        volume = prod(self.lattice[:3])
        modes = sum(handle.Ne for handle in self.handles.values())
        paths = max(len(terms) for _, terms in self.rows)
        output = prod(self.sectors[sector].shape[3:]) * 16
        return int(16 * volume * (512 + 24 * modes) + (4 * paths + 8) * output)

    def _produce(self, component, momentum, sector, time, xp):
        gauge = self._gauge_slice(time, xp)
        kinds = set(sector_kinds(sector)) - {"p"}
        vectors = {kind.upper(): _SliceBasis(self.handles[kind.upper()], self.cfg, time, xp) for kind in kinds}
        points = None if "p" not in sector_kinds(sector) else np.array(
            self.point_handle.read_points(self.cfg, time, 0, self.point_handle.Np), copy=True)[None]
        producer = ElementalGenerator(gauge, None, (self.momentum_list[momentum],), self._tables[component],
            vector_bases=vectors, blocks=(sector,), points=points, array_module=xp,
            phase_convention=self.phase_convention).load(self.cfg)
        value = producer.calc_path_blocks(0)[sector][0][0]
        if self.elemental_weighted:
            producer.weight_spatial_block(value, sector, component, time,
                point_pool=self._pool_by_time.get(time), sampling_groups=self.sampling_groups)
        if sector == "p2p":
            value = value.to_dense()
        return np.array(value, copy=True) if xp is np else xp.asnumpy(value)

    def _gauge_slice(self, time, xp):
        if self.stout[0]:
            index, spec, identity = self._gauge_product
            with open_dataset_view(index, spec, subset_axes=("time",), data_id=identity) as view:
                gauge = xp.array(view.read(time=time), dtype=complex, copy=True)[None]
        else:
            gauge = xp.array(self.gauge_handle.load(self.cfg)[time:time+1], dtype=complex, copy=True)
        return gauge

    def _location(self, spec):
        index_path, output = self._paths(spec)
        return DatasetIndex(index_path), output

    def _paths(self, spec):
        family = spec.document
        family["axes"][0]["coordinates"] = []
        # Limit candidate lookup to this scientific family, rather than scanning
        # all cfgs/components in a campaign for every preparation stage.
        return self.directory / "index" / stable_digest(family), self.directory / "products" / spec.definition_id

    def _stages(self, requests):
        grouped = defaultdict(set)
        for request in requests:
            component, momentum, time, sector = request.arguments
            self.describe_block(component, momentum, time, sector)
            grouped[component, momentum, sector].add(time)
        return tuple((c, m, s, tuple(sorted(times)), self._spec(c, m, s, tuple(sorted(times))))
                     for (c, m, s), times in grouped.items())

    def _describe_preparation(self, requests, backend, host_limit, device_limit):
        result = []
        for component, momentum, sector, times, spec in self._stages(requests):
            index_path, _output = self._paths(spec)
            decision = (plan_dataset_request(DatasetIndex(index_path), spec, subset_axes=("time",))
                        if index_path.is_dir() else plan_requirement(spec, (), subset_axes=("time",)))
            if decision["action"] == "blocked":
                raise ValueError(f"elemental preparation is blocked: {decision['reasons']}")
            peak = self._workspace_bytes(sector) if decision["missing"] else 0
            self._check_budget(peak, backend, host_limit, device_limit)
            result.append(dict(product=sector, component=repr(component), momentum=list(self.momentum_list[momentum]),
                times=list(times), missing_times=[times[i] for i in decision["missing"]],
                action=decision["action"], peak_host_bytes=peak,
                peak_device_bytes=peak if backend != "numpy" else 0,
                elemental_weighted=self.elemental_weighted))
        return result

    @staticmethod
    def _check_budget(peak, backend, host_limit, device_limit):
        if host_limit is not None and peak > host_limit:
            raise MemoryError("elemental preparation exceeds host memory budget")
        if backend != "numpy" and device_limit is not None and peak > device_limit:
            raise MemoryError("elemental preparation exceeds device memory budget")

    def _run_product(self, spec, producer, *, extra_provenance=None):
        index, output = self._location(spec)
        provenance = dict(self._provenance, **(extra_provenance or {}))
        result = run_dataset_requests({"elemental": DatasetRequest(spec, subset_axes=("time",))},
            index, {"elemental": output}, {"elemental": provenance}, {"elemental": producer}, resume=True)
        return index, spec, result["elemental"]

    def _prepare_gauge(self, times, xp):
        document = self.gauge.spec.document
        document.update(kind="elemental_gauge", dtype="complex128", bases={},
            parents=[self.gauge.data_id], definition=dict(gauge_id=self.gauge.data_id, stout=list(self.stout)),
            conventions=dict(layout="time,z,y,x,direction,color_row,color_column"))
        document["axes"][0]["coordinates"] = list(times)
        spec = DatasetSpec(document)

        def produce(_spec, _inputs, writer):
            from lqcdweave.production.gauge import stout_smear
            for slot in writer.missing:
                links = xp.array(self.gauge_handle.load(self.cfg)[times[slot]:times[slot]+1], dtype=complex, copy=True)
                smeared = stout_smear(links.transpose(4, 0, 1, 2, 3, 5, 6), *self.stout, array_module=xp)
                value = smeared.transpose(1, 2, 3, 4, 0, 5, 6)[0]
                writer.write(slot, np.array(value, copy=True) if xp is np else xp.asnumpy(value))

        self._gauge_product = self._run_product(spec, produce)

    def _ensure(self, requests, *, backend, host_limit, device_limit):
        if self._closed:
            raise ValueError("elemental reader is closed")
        if self._provenance is None:
            self._provenance = dict(local_provenance(), workflow="managed-spatial-elemental-v1")
        self._provenance["backend"] = backend.__name__
        pending = []
        for stage in self._stages(requests):
            component, momentum, sector, times, spec = stage
            # _products already binds published, scientifically admitted data.
            # Its index directory is the existing full scientific-family hash
            # (only the allowed time coverage is omitted). Recheck that family
            # before accepting each previously bound time; new definitions or
            # missing times still go through ordinary persistent admission.
            index_path, _ = self._paths(spec)
            family = index_path.resolve()
            if not all((product := self._products.get((component, momentum, sector, time)))
                       is not None and product[0].directory == family for time in times):
                pending.append(stage)
        if not pending:
            return
        # Existing file locking serializes one cfg's preparation; publication
        # itself retains storage's short registration locks.
        locks = self.directory / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with AtomicOpen(str(locks / (stable_digest(self.configuration) + ".lock")), "a+"):
            stages = []
            needed_times = set()
            for component, momentum, sector, times, spec in pending:
                index, _output = self._location(spec)
                decision = plan_dataset_request(index, spec, subset_axes=("time",))
                if decision["action"] == "blocked":
                    raise ValueError(f"elemental preparation is blocked: {decision['reasons']}")
                if decision["action"] != "reuse":
                    peak = self._workspace_bytes(sector)
                    self._check_budget(peak, backend.__name__, host_limit, device_limit)
                    needed_times.update(times[slot] for slot in decision["missing"])
                stages.append((component, momentum, sector, times, spec))
            if self.stout[0] and needed_times:
                self._prepare_gauge(tuple(sorted(needed_times)), backend)
            for component, momentum, sector, times, spec in stages:
                def produce(_spec, _inputs, writer):
                    for slot in writer.missing:
                        writer.write(slot, self._produce(component, momentum, sector, times[slot], backend))

                provenance = ({} if not self.stout[0] or self._gauge_product is None else
                              {"prepared_gauge_data_id": self._gauge_product[2]})
                index, _spec, identity = self._run_product(spec, produce, extra_provenance=provenance)
                for time in times:
                    self._products[component, momentum, sector, time] = index, spec, identity

    def close(self):
        self._products.clear()
        self._gauge_product = None
        super().close()


class _BaryonElemental(Elemental):
    """The same managed workflow with three ordered V basis roles."""

    component_type = BaryonComponent
    sector_ranks = ArrayBaryonElemental.sector_ranks
    sector_tail_axes = ArrayBaryonElemental.sector_tail_axes

    def __init__(self, gauge, bases, operator_table, momentum_dict, *, directory,
                 points=None, stout=(0, 0.12), phase_convention="anchor"):
        if points is not None:
            raise NotImplementedError("managed baryon elementals currently support VVV")
        if phase_convention not in ("anchor", "none"):
            raise ValueError("baryon phase must be anchor or none")
        self.gauge = gauge
        self.configuration = gauge.spec.document["configuration"]
        self.lattice = tuple(self.configuration["lattice"])
        self.bases = tuple(bases) if isinstance(bases, (tuple, list)) else (bases,)*3
        if len(self.bases) != 3:
            raise ValueError("baryon requires one shared V basis or three ordered V bases")
        self.handles = tuple(DatasetFieldHandle(dataset) for dataset in self.bases)
        for dataset, handle in zip(self.bases, self.handles):
            validate_elemental_fields(gauge.spec, dataset.spec)
            if handle.dilution_source is not None:
                raise NotImplementedError("managed baryon elementals currently support VVV, not D")
        self.gauge_handle = DatasetFieldHandle(gauge)
        self.points, self.point_handle = None, None
        self.rows = elemental_operator_rows(operator_table, rank=3)
        self.catalog = operator_table.catalog
        self.momentum_list = normalize_momenta(momentum_dict)
        self.stout = (integer(stout[0], "stout steps", 0), float(stout[1]))
        if not np.isfinite(self.stout[1]):
            raise ValueError("stout rho must be finite")
        self.directory = Path(directory)
        self.path_boundary = "periodic"
        self._tables = {c: self.catalog.table((c.key[-1],)) for c, _ in self.rows}
        self._leg_bindings = {f"mode{i}": (dataset.data_id, tuple(dataset.spec.coordinates("mode")))
                              for i, dataset in enumerate(self.bases)}
        shape = (len(self.rows), len(self.momentum_list), self.lattice[3], *(h.Ne for h in self.handles))
        ArrayElemental.__init__(self, {"vvv":_Header(shape)}, tuple(c for c, _ in self.rows),
            phase_convention=phase_convention, cfg=self.configuration["configuration"],
            momentum_list=self.momentum_list, elemental_weighted=False)
        self._products, self._pool_by_time = {}, {}
        self._gauge_product, self._provenance = None, None

    def describe_block(self, component, momentum, t, sector, **selection):
        metadata = super().describe_block(component, momentum, t, sector, **selection)
        from dataclasses import replace
        from lqcdweave.adapters.selection import resolve_selections
        selected = resolve_selections(metadata.axes, tuple(h.Ne for h in self.handles), **selection)
        bindings = {}
        for axis, (identity, labels) in self._leg_bindings.items():
            choice = selected.get(axis, slice(None))
            positions = range(*choice.indices(len(labels))) if isinstance(choice, slice) else choice
            bindings[axis] = (identity, tuple(labels[i] for i in positions))
        return replace(metadata, axis_bindings=bindings)

    def describe_slab(self, component, momentum, times_selected, sector, **selection):
        metadata = super().describe_slab(component, momentum, times_selected, sector, **selection)
        from dataclasses import replace
        # The shared array adapter supplies shape/selection validation; retain
        # the same three actual basis roles as a single physical-time block.
        bindings = self.describe_block(component, momentum, 0, sector, **selection).axis_bindings
        return replace(metadata, axis_bindings=bindings)

    def _spec(self, component, momentum, sector, times):
        axes = [dict(name="time", coordinates=list(times))]
        bases = {}
        for axis, (identity, labels) in self._leg_bindings.items():
            axes.append(dict(name=axis, coordinates=list(labels)))
            bases[axis] = dict(data_id=identity, indices=list(labels))
        terms = self.rows[self._index[component]][1]
        return DatasetSpec(dict(schema=1, kind="managed_elemental_vvv", configuration=self.configuration,
            axes=axes, dtype="complex128", bases=bases,
            definition=dict(gauge_id=self.gauge.data_id, product=sector,
                catalog=dict(name=self.catalog.name, version=self.catalog.version), component=component.key[-1],
                terms=[dict(paths=[list(p.steps) for p in paths], coefficient=[z.real,z.imag]) for paths,z in terms],
                momentum=list(self.momentum_list[momentum]), stout=list(self.stout)),
            conventions=dict(phase=self.phase_convention, path_boundary=self.path_boundary,
                             elemental_weighted=self.elemental_weighted, color_contraction="epsilon", vector_conjugation="none"),
            parents=list(dict.fromkeys((self.gauge.data_id, *(dataset.data_id for dataset in self.bases))))))

    def _workspace_bytes(self, sector):
        volume = prod(self.lattice[:3])
        modes = sum(handle.Ne for handle in self.handles)
        output = prod(self.sectors[sector].shape[3:])*16
        return int(16*volume*(512+48*modes)+16*output)

    def _produce(self, component, momentum, sector, time, xp):
        gauge = self._gauge_slice(time, xp)
        shared = {}
        vectors = []
        for dataset, handle in zip(self.bases, self.handles):
            if id(dataset) not in shared:
                shared[id(dataset)] = _SliceBasis(handle, self.cfg, time, xp)
            vectors.append(shared[id(dataset)])
        producer = BaryonElementalGenerator(gauge, tuple(vectors), (self.momentum_list[momentum],),
            self._tables[component], array_module=xp, phase_convention=self.phase_convention).load(self.cfg)
        value = producer.calc_path_blocks(0)["vvv"][0,0]
        return np.array(value, copy=True) if xp is np else xp.asnumpy(value)


def prepare_elemental_sources(requests, readers, *, backend=np, host_limit=None,
                              device_limit=None):
    """Internal execution dependency, using only registered physical requests."""
    _prepare_sources(((requests, readers),), backend, host_limit, device_limit)


def prepare_elemental_inputs(inputs, *, backend=np, host_limit=None, device_limit=None):
    """Prepare either fixed-coordinate inputs or a coalesced slab task."""
    if inputs is None:
        return
    _prepare_sources(_input_sources(inputs), backend, host_limit, device_limit)


def _input_sources(inputs):
    if inputs is None:
        return ()
    sources = getattr(inputs.loader, "preparation_inputs", ())
    readers = getattr(inputs.loader, "input_readers", None)
    return (*sources, (inputs.read_requests, readers)) if readers and inputs.read_requests else sources


def describe_elemental_preparation(inputs, *, backend="numpy", host_limit=None, device_limit=None):
    """Metadata-only workflow admission; querying absent output creates nothing."""
    stages = [stage for reader, requests in _managed_sources(_input_sources(inputs))
              for stage in reader._describe_preparation(requests, backend, host_limit, device_limit)]
    if not stages:
        return {}
    return dict(stages=stages, peak_host_bytes=max(stage["peak_host_bytes"] for stage in stages),
                peak_device_bytes=max(stage["peak_device_bytes"] for stage in stages),
                cost_model="not_calibrated")


def _prepare_sources(sources, backend, host_limit, device_limit):
    managed = _managed_sources(sources)
    if not managed:
        return
    from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
    limits = resolve_diagram_memory_limits(backend=backend,
        memory_limit=host_limit if backend is np else device_limit,
        host_memory_limit=host_limit, execution_mode="local")
    for reader, selected in managed:
        host_limit = limits.host_bytes
        device_limit = limits.device_bytes
        reader._ensure(selected, backend=backend, host_limit=host_limit, device_limit=device_limit)


def _managed_sources(sources):
    grouped = {}
    for requests, readers in sources:
        for primary in requests.values():
            for template in (primary, *primary.aliases):
                if template.family != "elemental":
                    continue
                reader = readers[template.reader_role]
                from lqcdweave.adapters.dilution_prefix import DilutionPrefixElemental
                while isinstance(reader, DilutionPrefixElemental):
                    reader = reader._reader
                if not isinstance(reader, Elemental):
                    continue
                entry = grouped.setdefault(id(reader), (reader, []))
                entry[1].append(template)
    return tuple(grouped.values())


def bind_managed_elemental(reader, weighted):
    """Apply the single workflow option through native reader wrappers."""
    if isinstance(reader, Elemental):
        return reader._for_calculation(weighted)
    from lqcdweave.adapters.dilution_prefix import DilutionPrefixElemental
    if isinstance(reader, DilutionPrefixElemental):
        base = bind_managed_elemental(reader._reader, weighted)
        if base is not reader._reader:
            reader = copy(reader)
            reader._reader = base
    return reader


__all__ = ["Elemental"]
