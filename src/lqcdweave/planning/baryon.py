"""Scientific descriptions for registered three-leg baryon elementals."""

from collections.abc import Mapping
from numbers import Real
import math

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import bind_time_selections, dataset_time_selection
from lqcdweave.contracts.elemental import GaugePath, integer
from lqcdweave.contracts.elemental_catalog import stored_catalog_components
from lqcdweave.contracts.memory import MemoryEstimate
from lqcdweave.planning.production import validate_elemental_fields


_PRODUCTS = {"raw_paths", "projected", "distance"}
_ENCODINGS = {
    "raw_paths": "ordered_path_triples_v1",
    "projected": "projected_row_ids_v1",
    "distance": "distance_triples_v1",
}
_DISTANCE_RULE = "independent_six_straight_paths_per_nonzero_leg"
_BARYON_DTYPE_BYTES = {
    "complex64": 8,
    "complex128": 16,
}
_BARYON_SPAN_ENTRY_ESTIMATE_BYTES = 512
_BARYON_SEEN_ENTRY_ESTIMATE_BYTES = 384
_BARYON_CACHE_ENTRY_ESTIMATE_BYTES = 256


def _baryon_nonnegative_int(value, name):
    value = integer(value, name, 0)
    return value


def _baryon_positive_triple(value, name):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        count = integer(value, name, 1)
        return (count, count, count)
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be one positive integer or three positive integers")
    try:
        values = tuple(integer(item, name, 1) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be one positive integer or three positive integers") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three positive integers")
    return values


def _baryon_array_bytes(shape, dtype, name):
    if dtype not in _BARYON_DTYPE_BYTES:
        raise ValueError(f"{name} must use complex64 or complex128 storage")
    size = _BARYON_DTYPE_BYTES[dtype]
    for extent in shape:
        if type(extent) is not int or extent < 1:
            raise ValueError(f"{name} has an invalid array extent")
        size *= extent
    return size


def _baryon_cache_recipe_counts(document):
    definition = document["definition"]
    product = definition["product"]
    recipes = [set() for _ in range(3)]
    transports = set()
    if product == "raw_paths":
        for component in document["axes"][1]["coordinates"]:
            for leg, path in enumerate(component["paths"]):
                key = tuple(path)
                recipes[leg].add(key)
                if key:
                    transports.add(key)
    elif product == "projected":
        for row in definition["projected_rows"]:
            for term in row["terms"]:
                for leg, path in enumerate(term["paths"]):
                    key = tuple(path)
                    recipes[leg].add(key)
                    if key:
                        transports.add(key)
    else:
        for component in document["axes"][1]["coordinates"]:
            for leg, distance in enumerate(component["legs"]):
                recipes[leg].add(("distance", distance))
    return tuple(len(items) for items in recipes), len(transports)


def estimate_baryon_vvv_memory(
    input_specs: Mapping[str, DatasetSpec],
    output_spec: DatasetSpec,
    *,
    gauge_id: str,
    basis_ids,
    ne_block_size=16,
    spatial_block_size=256,
    cache_bytes=64 * 1024**2,
    stream_eigenvectors=True,
    target="cpu",
    workspace_reserve_bytes=0,
    host_staging_bytes=0,
) -> tuple[MemoryEstimate, MemoryEstimate]:
    """Estimate one registered VVV producer's logical host working set.

    This CPU model covers the visible full input payloads, the gauge load/copy,
    role-specific eager snapshots or streaming buffers, elemental caches and
    tiles, one output row, row coverage bookkeeping and finite checks.
    Python-container costs use explicit
    interpreter-size estimates (512 bytes per expected mode span, 384 per seen
    tile key, and 256 per retained cache entry), not a portable exact byte
    bound. Caller workspace/staging reserves remain required for the unmodeled
    Python, allocator, BLAS and native workspaces. The pair return value follows
    the existing production estimator convention: host estimate, device
    estimate. This is not RSS or a device-capacity guarantee.
    """
    if not isinstance(input_specs, Mapping):
        raise TypeError("input_specs must map actual dataset IDs to DatasetSpec values")
    if not isinstance(output_spec, DatasetSpec):
        raise TypeError("output_spec must be a DatasetSpec")
    if target != "cpu":
        raise ValueError("baryon VVV memory model supports target='cpu' only")
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("gauge_id must be a nonempty actual dataset ID")
    if isinstance(basis_ids, (str, bytes, Mapping)):
        raise TypeError("basis_ids must contain three ordered actual IDs")
    try:
        basis_ids = tuple(basis_ids)
    except TypeError as exc:
        raise TypeError("basis_ids must contain three ordered actual IDs") from exc
    if (len(basis_ids) != 3 or any(not isinstance(value, str) or not value
                                   for value in basis_ids)):
        raise ValueError("basis_ids must contain three nonempty actual IDs")
    expected_ids = {gauge_id, *basis_ids}
    if set(input_specs) != expected_ids:
        raise ValueError("input_specs must contain exactly the actual gauge and basis IDs")
    if any(not isinstance(value, DatasetSpec) for value in input_specs.values()):
        raise TypeError("input_specs values must be DatasetSpec instances")
    if any(identity == gauge_id for identity in basis_ids):
        raise ValueError("gauge and basis IDs cannot identify the same dataset")

    block_sizes = _baryon_positive_triple(ne_block_size, "ne_block_size")
    spatial_block_size = integer(spatial_block_size, "spatial_block_size", 1)
    cache_bytes = _baryon_nonnegative_int(cache_bytes, "cache_bytes")
    if type(stream_eigenvectors) is not bool:
        raise TypeError("stream_eigenvectors must be bool")
    workspace_reserve_bytes = _baryon_nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _baryon_nonnegative_int(host_staging_bytes, "host_staging_bytes")

    gauge = input_specs[gauge_id]
    basis_specs = tuple(input_specs[identity] for identity in basis_ids)
    counts = tuple(len(output_spec.coordinates(f"mode{leg}")) for leg in range(3))
    basis_views = tuple(baryon_basis_view(spec, count)
                        for spec, count in zip(basis_specs, counts))
    selection = dataset_time_selection(output_spec)
    for basis in basis_specs:
        validate_elemental_fields(gauge, basis, selection=selection)

    document = output_spec.document
    if (document["kind"] != "elemental_baryon_vvv" or
            output_spec.axis_names != (
                "time", "component", "momentum", "mode0", "mode1", "mode2") or
            document["dtype"] != "complex128"):
        raise ValueError("output_spec must be the canonical complex128 baryon VVV product")
    definition = document["definition"]
    product_name = definition.get("product")
    if product_name not in _PRODUCTS:
        raise ValueError("output_spec has an unsupported baryon VVV product")
    components = output_spec.coordinates("component")
    momenta = output_spec.coordinates("momentum")
    expected_document = baryon_dataset_document(
        gauge, basis_views,
        gauge_id=gauge_id,
        basis_ids=basis_ids,
        product=product_name,
        component_coordinates=components,
        momentum_list=momenta,
        phase_convention=document["conventions"].get("phase"),
        path_boundary=document["conventions"].get("path_boundary"),
        projected_rows=definition.get("projected_rows") if product_name == "projected" else None,
        catalog=definition.get("catalog"),
        parent_actual_ids=(gauge_id, *basis_ids), selection=selection,
    )
    if expected_document != document:
        raise ValueError("output_spec differs from the actual baryon input IDs or ordered views")

    cfg = gauge.document["configuration"]
    lx, ly, lz, lt = cfg["lattice"]
    volume = lx * ly * lz
    full_input_bytes = sum(
        _baryon_array_bytes(spec.shape, spec.document["dtype"], "registered input")
        for spec in {gauge_id: gauge, **{identity: input_specs[identity]
                                       for identity in set(basis_ids)}}.values())
    full_basis_bytes = tuple(
        _baryon_array_bytes(spec.shape, spec.document["dtype"], "registered basis")
        for spec in basis_specs)
    owned_links_bytes = 4 * lt * volume * 3 * 3 * 16
    selected_snapshots = sum(spec.shape[0] * count * volume * 3 * 16 for spec, count in zip(basis_specs, counts))
    largest_eager_load = max(full_basis_bytes)
    snapshot_bytes = owned_links_bytes
    if not stream_eigenvectors:
        snapshot_bytes += selected_snapshots + largest_eager_load

    tile_modes = tuple(min(count, block) for count, block in zip(counts, block_sizes))
    tile_elements = math.prod(tile_modes)
    tile_result_bytes = tile_elements * 16
    vector_tile_bytes = tuple(size * volume * 3 * 16 for size in tile_modes)
    stream_buffers_bytes = sum(vector_tile_bytes) if stream_eigenvectors else 0
    max_vector_tile = max(vector_tile_bytes)
    matrix_bytes = volume * 3 * 3 * 16
    phase_bytes = volume * 16
    phase_keys = len(momenta)
    phase_slots = min(64, phase_keys)
    stable_phase_bytes = phase_slots * phase_bytes
    old_phase_slots = min(64, max(phase_keys - 1, 0))
    # momentum_phase creates int64 x/y/z grids before returning the complex
    # ones array for phase='none'; those allocations overlap at the peak.
    phase_build_bytes = volume * (40 if document["conventions"]["phase"] == "none" else 64)
    phase_cache_peak = max(stable_phase_bytes,
                           old_phase_slots * phase_bytes + phase_build_bytes)
    phase_miss_extra = max(0, phase_cache_peak - stable_phase_bytes)

    recipe_counts, transport_count = _baryon_cache_recipe_counts(document)
    mode_span_count = math.prod(
        (count + block - 1) // block for count, block in zip(counts, block_sizes))
    field_cache_keys = (sum(recipe_counts) * mode_span_count
                        if not stream_eigenvectors else 0)
    transport_cache_keys = transport_count
    cache_candidate_count = field_cache_keys + transport_cache_keys
    minimum_cache_item = volume * 3 * 16
    cache_entry_count = (min(cache_candidate_count, cache_bytes // minimum_cache_item)
                         if cache_candidate_count and cache_bytes else 0)
    cache_max_item = max(max_vector_tile, matrix_bytes) if cache_candidate_count else 0
    cache_data_peak = min(cache_bytes, cache_entry_count * cache_max_item)
    cache_python_bytes = cache_entry_count * _BARYON_CACHE_ENTRY_ESTIMATE_BYTES

    row_bytes = _baryon_array_bytes(
        output_spec.shape[1:], "complex128", "baryon output row")
    finite_mask_bytes = row_bytes // 16
    component_count, momentum_count = len(components), len(momenta)
    seen_entry_count = component_count * momentum_count * mode_span_count
    python_bookkeeping_bytes = (
        mode_span_count * _BARYON_SPAN_ENTRY_ESTIMATE_BYTES
        + seen_entry_count * _BARYON_SEEN_ENTRY_ESTIMATE_BYTES
        + cache_python_bytes
    )
    assembly_bytes = finite_mask_bytes + python_bookkeeping_bytes

    gauge_stack_bytes = _baryon_array_bytes(gauge.shape, gauge.document["dtype"], "gauge stack")
    conversion_tile = 0
    if stream_eigenvectors:
        complex64_roles = [vector_tile_bytes[leg] for leg, spec in enumerate(basis_specs)
                           if spec.document["dtype"] == "complex64"]
        conversion_tile = max(complex64_roles, default=0)
    # Both persistent epsilon result buffers coexist with path/cache data.
    tile_buffers_bytes = 2 * tile_result_bytes
    path_work = (sum(vector_tile_bytes) + 2 * max_vector_tile + 4 * matrix_bytes)
    distance_work = (sum(vector_tile_bytes) + 3 * max_vector_tile + 4 * matrix_bytes)
    # Optimized einsum can create a spatially tiled mode-product intermediate;
    # allocator/BLAS extras remain in workspace_reserve_bytes.
    spatial_kernel_work = min(volume, spatial_block_size) * tile_result_bytes
    epsilon_work = 2 * tile_result_bytes + spatial_kernel_work
    transform_peak = distance_work if product_name == "distance" else path_work
    calc_intermediate = (
        stable_phase_bytes + phase_miss_extra + cache_data_peak + transform_peak
        + tile_buffers_bytes + epsilon_work + conversion_tile
    )
    gauge_load_peak = gauge_stack_bytes
    producer_intermediate = max(gauge_load_peak, calc_intermediate)
    intermediate_bytes = producer_intermediate + stream_buffers_bytes

    host_estimate = MemoryEstimate(
        input_bytes=full_input_bytes,
        snapshot_bytes=snapshot_bytes,
        intermediate_bytes=intermediate_bytes,
        output_bytes=row_bytes,
        assembly_extra_bytes=assembly_bytes,
        workspace_bytes=workspace_reserve_bytes + host_staging_bytes,
    )
    return host_estimate, MemoryEstimate()


def normalize_baryon_components(product, components=None):
    """Detach and validate the one product-specific component payload."""
    if product not in _PRODUCTS:
        raise ValueError("product must be raw_paths, projected or distance")
    if components is None:
        if product != "raw_paths":
            raise ValueError(f"components are required for product {product!r}")
        components = (((), (), ()),)
    if getattr(components, "catalog", None) is not None:
        if product != "projected":
            raise ValueError("catalog recipes use product='projected'")
        from lqcdweave.contracts.elemental import elemental_operator_rows
        parsed = elemental_operator_rows(components, rank=3)
        components = tuple(dict(id=i, terms=[dict(
            paths=[list(p.steps) for p in paths], coefficient=[z.real, z.imag])
            for paths, z in terms]) for i, (_, terms) in enumerate(parsed))
    if isinstance(components, (str, bytes, Mapping)):
        raise TypeError("components must be an ordered product payload")
    try:
        values = tuple(components)
    except TypeError as exc:
        raise TypeError("components must be an ordered product payload") from exc
    if not values:
        raise ValueError("components must be nonempty")

    if product == "raw_paths":
        normalized = []
        for triple in values:
            if isinstance(triple, (str, bytes, Mapping)):
                raise TypeError("each raw component must contain three ordered paths")
            try:
                paths = tuple(triple)
            except TypeError as exc:
                raise TypeError("each raw component must contain three ordered paths") from exc
            if len(paths) != 3:
                raise ValueError("each raw component must contain exactly three leg paths")
            steps = [_spatial_steps(path) for path in paths]
            normalized.append(dict(encoding="ordered_path_triple_v1", paths=steps))
        keys = [tuple(tuple(path) for path in item["paths"]) for item in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("raw path triples must be unique")
        return tuple(normalized), None

    if product == "projected":
        rows = []
        coordinates = []
        for position, row in enumerate(values):
            if not isinstance(row, Mapping) or set(row) != {"id", "terms"}:
                raise ValueError("projected rows require exactly id and terms")
            if type(row["id"]) is not int or row["id"] != position:
                raise ValueError("projected row IDs must be consecutive integers from zero")
            terms = row["terms"]
            if isinstance(terms, (str, bytes, Mapping)):
                raise TypeError("projected terms must be a nonempty ordered sequence")
            try:
                terms = tuple(terms)
            except TypeError as exc:
                raise TypeError("projected terms must be a nonempty ordered sequence") from exc
            if not terms:
                raise ValueError("projected rows require at least one original term")
            normalized_terms = []
            for term in terms:
                if not isinstance(term, Mapping) or set(term) != {"paths", "coefficient"}:
                    raise ValueError("projected terms require exactly paths and coefficient")
                paths = term["paths"]
                if isinstance(paths, (str, bytes, Mapping)):
                    raise TypeError("projected term paths must contain three ordered legs")
                try:
                    paths = tuple(paths)
                except TypeError as exc:
                    raise TypeError("projected term paths must contain three ordered legs") from exc
                if len(paths) != 3:
                    raise ValueError("projected terms require exactly three leg paths")
                coefficient = term["coefficient"]
                if (not isinstance(coefficient, (tuple, list)) or len(coefficient) != 2 or
                        any(isinstance(value, (bool, np.bool_)) or
                            not isinstance(value, Real) for value in coefficient)):
                    raise ValueError("projection coefficients require two finite real values")
                real, imag = float(coefficient[0]), float(coefficient[1])
                if not np.isfinite(real) or not np.isfinite(imag):
                    raise ValueError("projection coefficients must be finite")
                normalized_terms.append(dict(
                    paths=[_spatial_steps(path) for path in paths],
                    coefficient=[real, imag],
                ))
            rows.append(dict(id=position, terms=normalized_terms))
            coordinates.append(dict(encoding="projected_row_id_v1", id=position))
        return tuple(coordinates), tuple(rows)

    distances = []
    for triple in values:
        if isinstance(triple, (str, bytes, Mapping)):
            raise TypeError("distance components must be ordered integer triples")
        try:
            triple = tuple(triple)
        except TypeError as exc:
            raise TypeError("distance components must be ordered integer triples") from exc
        if len(triple) != 3:
            raise ValueError("distance components require exactly three leg distances")
        distances.append([integer(value, "distance", 0) for value in triple])
    if len({tuple(triple) for triple in distances}) != len(distances):
        raise ValueError("distance triples must be unique")
    return tuple(dict(encoding="distance_triple_v1", legs=triple)
                 for triple in distances), None


def normalize_baryon_momenta(momentum_list):
    if isinstance(momentum_list, (str, bytes, Mapping)):
        raise TypeError("momentum_list must be an ordered sequence of XYZ triples")
    try:
        momenta = tuple(tuple(integer(value, "momentum") for value in momentum)
                        for momentum in momentum_list)
    except TypeError as exc:
        raise TypeError("momentum_list must be an ordered sequence of XYZ triples") from exc
    if (not momenta or any(len(momentum) != 3 for momentum in momenta) or
            len(set(momenta)) != len(momenta)):
        raise ValueError("momentum_list must contain unique nonempty XYZ triples")
    return tuple(tuple(value for value in momentum) for momentum in momenta)


def normalize_baryon_mode_counts(usedNe, available):
    """Resolve one scalar or three role-specific ordered mode prefixes."""
    if usedNe is None:
        counts = tuple(len(labels) for labels in available)
    elif isinstance(usedNe, (int, np.integer)) and not isinstance(usedNe, (bool, np.bool_)):
        count = integer(usedNe, "usedNe", 1)
        counts = (count, count, count)
    else:
        if isinstance(usedNe, (str, bytes, Mapping)):
            raise TypeError("usedNe must be one count or three leg counts")
        try:
            counts = tuple(integer(value, "usedNe", 1) for value in usedNe)
        except TypeError as exc:
            raise TypeError("usedNe must be one count or three leg counts") from exc
        if len(counts) != 3:
            raise ValueError("usedNe requires one count or three leg counts")
    for leg, (count, labels) in enumerate(zip(counts, available)):
        if count > len(labels):
            raise ValueError(f"usedNe for leg {leg} exceeds its registered basis")
    return tuple(counts)


def baryon_basis_view(spec, count):
    """Return the exact ordered mode-prefix view of one actual basis."""
    if not isinstance(spec, DatasetSpec):
        raise TypeError("basis view requires a DatasetSpec")
    labels = spec.coordinates("mode")
    count = integer(count, "mode count", 1)
    if count > len(labels):
        raise ValueError("mode count exceeds the registered basis")
    selected = list(labels[:count])
    document = spec.document
    for axis in document["axes"]:
        if axis["name"] == "mode":
            axis["coordinates"] = selected
    if "mode" in document["bases"]:
        document["bases"]["mode"]["indices"] = selected
    return DatasetSpec(document)


def baryon_dataset_document(gauge, basis_views, *, gauge_id, basis_ids,
                             product, component_coordinates, momentum_list,
                             phase_convention, path_boundary, projected_rows=None,
                             parent_actual_ids=None, selection=None, catalog=None):
    """Build one resolved or DatasetRequest-template baryon vvv description."""
    if len(basis_views) != 3 or len(basis_ids) != 3:
        raise ValueError("three ordered basis views and IDs are required")
    if product not in _PRODUCTS:
        raise ValueError("product must be raw_paths, projected or distance")
    if phase_convention not in ("anchor", "none"):
        raise ValueError("phase_convention must be anchor or none")
    if path_boundary not in ("periodic", "open"):
        raise ValueError("path_boundary must be periodic or open")
    times = validate_elemental_fields(gauge, basis_views[0], selection=selection)
    for basis in basis_views[1:]:
        if validate_elemental_fields(gauge, basis, selection=selection) != times:
            raise ValueError("all baryon basis legs require complete physical-time coverage")
    modes = [list(view.coordinates("mode")) for view in basis_views]
    roles = {
        "gauge": gauge_id,
        "basis_leg0": basis_ids[0],
        "basis_leg1": basis_ids[1],
        "basis_leg2": basis_ids[2],
    }
    role_refs = (gauge_id, *basis_ids)
    if parent_actual_ids is None:
        parent_actual_ids = role_refs
    if (len(parent_actual_ids) != 4 or
            any(not isinstance(identity, str) or not identity for identity in parent_actual_ids)):
        raise ValueError("four actual input IDs are required to deduplicate parents")
    parents = []
    seen = set()
    for identity, reference in zip(parent_actual_ids, role_refs):
        if identity not in seen:
            seen.add(identity)
            parents.append(reference)
    definition = dict(
        gauge_id=gauge_id,
        gauge_purpose="elemental_path_transport",
        product=product,
        component_encoding=_ENCODINGS[product],
        input_roles=roles,
        input_views={
            "gauge": gauge.document,
            "basis_leg0": basis_views[0].document,
            "basis_leg1": basis_views[1].document,
            "basis_leg2": basis_views[2].document,
        },
    )
    if product == "projected":
        if not projected_rows:
            raise ValueError("projected products require their nonempty row recipes")
        definition["projected_rows"] = list(projected_rows)
    elif projected_rows is not None:
        raise ValueError("only projected products may declare projected_rows")
    if product == "distance":
        definition["distance_rule"] = _DISTANCE_RULE
    if catalog is not None:
        if product != "projected":
            raise ValueError("catalog recipes require the projected product")
        stored_catalog_components(catalog, projected_rows, rank=3)
        definition["catalog"] = catalog
    document = dict(
        schema=1,
        kind="elemental_baryon_vvv",
        configuration=gauge.document["configuration"],
        axes=[
            dict(name="time", coordinates=list(times)),
            dict(name="component", coordinates=list(component_coordinates)),
            dict(name="momentum", coordinates=[list(value) for value in momentum_list]),
            dict(name="mode0", coordinates=modes[0]),
            dict(name="mode1", coordinates=modes[1]),
            dict(name="mode2", coordinates=modes[2]),
        ],
        dtype="complex128",
        bases={f"mode{leg}": dict(data_id=basis_ids[leg], indices=modes[leg])
               for leg in range(3)},
        definition=definition,
        conventions=dict(
            normalization="unscaled",
            phase=phase_convention,
            path_boundary=path_boundary,
            color_contraction="epsilon",
            vector_conjugation="none",
        ),
        parents=parents,
    )
    return document if selection is None else bind_time_selections(document, {"time": selection})


def _spatial_steps(path):
    steps = GaugePath.coerce(path).steps
    if any(step >= 6 for step in steps):
        raise ValueError("baryon elemental paths may use spatial directions 0..5 only")
    return list(steps)


__all__ = [
    "baryon_dataset_document",
    "baryon_basis_view",
    "estimate_baryon_vvv_memory",
    "normalize_baryon_components",
    "normalize_baryon_mode_counts",
    "normalize_baryon_momenta",
]
