"""Plan baryon elemental products with optional explicit anchor selections."""

from collections.abc import Mapping
from contextlib import ExitStack

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.contracts.elemental_catalog import catalog_definition
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.baryon import (
    baryon_dataset_document,
    baryon_basis_view,
    normalize_baryon_components,
    normalize_baryon_mode_counts,
    normalize_baryon_momenta,
)
from lqcdweave.planning.production import validate_elemental_fields
from lqcdweave.runtime.datasets import (
    apply_source_policy,
)


def _actual_ids(gauge_id, basis_ids):
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("gauge_id must be an actual registered dataset ID")
    if isinstance(basis_ids, (str, bytes, Mapping)):
        raise TypeError("basis_ids must contain three ordered actual basis IDs")
    try:
        basis_ids = tuple(basis_ids)
    except TypeError as exc:
        raise TypeError("basis_ids must contain three ordered actual basis IDs") from exc
    if (len(basis_ids) != 3 or any(not isinstance(identity, str) or not identity
                                   for identity in basis_ids)):
        raise ValueError("basis_ids must contain three nonempty actual dataset IDs")
    return basis_ids


def _source_policy(value):
    if value not in ("allow_unknown", "require_known"):
        raise ValueError("source_policy must be allow_unknown or require_known")
    return value










def _prepare_baryon_options(gauge_id, basis_ids, *, product, components,
                            momentum_list, phase_convention, path_boundary, selection=None):
    basis_ids = _actual_ids(gauge_id, basis_ids)
    coordinates, projected_rows = normalize_baryon_components(product, components)
    momenta = normalize_baryon_momenta(momentum_list)
    if phase_convention not in ("anchor", "none"):
        raise ValueError("phase_convention must be anchor or none")
    if path_boundary not in ("periodic", "open"):
        raise ValueError("path_boundary must be periodic or open")
    return dict(
        gauge_id=gauge_id,
        basis_ids=basis_ids,
        product=product,
        coordinates=coordinates,
        projected_rows=projected_rows,
        catalog=catalog_definition(components),
        momenta=momenta,
        phase_convention=phase_convention,
        path_boundary=path_boundary, selection=selection,
    )


def _baryon_requests_from_specs(options, input_specs, *, usedNe):
    gauge_id = options["gauge_id"]
    basis_ids = options["basis_ids"]
    gauge_spec = input_specs[gauge_id]
    parent_specs = tuple(input_specs[identity] for identity in basis_ids)
    for spec in parent_specs:
        validate_elemental_fields(gauge_spec, spec, selection=options["selection"])
    counts = normalize_baryon_mode_counts(
        usedNe, tuple(spec.coordinates("mode") for spec in parent_specs))
    views = tuple(baryon_basis_view(spec, count)
                  for spec, count in zip(parent_specs, counts))

    requests = {
        "gauge": DatasetRequest(gauge_spec, data_id=gauge_id),
    }
    for leg, (identity, view) in enumerate(zip(basis_ids, views)):
        requests[f"basis_leg{leg}"] = DatasetRequest(
            view, subset_axes=("mode",), data_id=identity)
    basis_refs = tuple({"request": f"basis_leg{leg}"} for leg in range(3))
    document = baryon_dataset_document(
        gauge_spec, views,
        gauge_id={"request": "gauge"},
        basis_ids=basis_refs,
        product=options["product"],
        component_coordinates=options["coordinates"],
        momentum_list=options["momenta"],
        phase_convention=options["phase_convention"],
        path_boundary=options["path_boundary"],
        projected_rows=options["projected_rows"],
        catalog=options["catalog"],
        parent_actual_ids=(gauge_id, *basis_ids), selection=options["selection"],
    )
    resolved_document = baryon_dataset_document(
        gauge_spec, views,
        gauge_id=gauge_id,
        basis_ids=basis_ids,
        product=options["product"],
        component_coordinates=options["coordinates"],
        momentum_list=options["momenta"],
        phase_convention=options["phase_convention"],
        path_boundary=options["path_boundary"],
        projected_rows=options["projected_rows"],
        catalog=options["catalog"],
        parent_actual_ids=(gauge_id, *basis_ids), selection=options["selection"],
    )
    requests[options["product"]] = DatasetRequest(
        document,
        subset_axes=("component", "momentum", "mode0", "mode1", "mode2"),
    )
    return (requests, options["coordinates"], options["projected_rows"],
            options["momenta"], counts, DatasetSpec(resolved_document))


def _indexed_baryon_requests(index, gauge_id, basis_ids, *, product, components,
                             momentum_list, usedNe, phase_convention, path_boundary,
                             prepared=None, input_specs=None, selection=None):
    options = (prepared if prepared is not None else _prepare_baryon_options(
        gauge_id, basis_ids, product=product, components=components,
        momentum_list=momentum_list, phase_convention=phase_convention,
        path_boundary=path_boundary, selection=selection))
    if input_specs is not None:
        return _baryon_requests_from_specs(options, input_specs, usedNe=usedNe)

    with ExitStack() as stack:
        gauge = stack.enter_context(index.open(gauge_id))
        opened_basis = {}
        for identity in options["basis_ids"]:
            if identity not in opened_basis:
                opened_basis[identity] = stack.enter_context(index.open(identity))
        specs = {gauge_id: gauge.spec}
        specs.update({identity: reader.spec for identity, reader in opened_basis.items()})
        return _baryon_requests_from_specs(options, specs, usedNe=usedNe)


def plan_indexed_baryon_elementals(index, gauge_id, basis_ids, *,
                                   product="raw_paths", components=None,
                                   momentum_list=((0, 0, 0),), usedNe=None,
                                   phase_convention="anchor", path_boundary="periodic",
                                   data_ids=None, source_policy="allow_unknown", selection=None):
    """Preview raw, projected, or distance baryons; default to all physical times."""
    _source_policy(source_policy)
    requests, _, _, _, _, _ = _indexed_baryon_requests(
        index, gauge_id, basis_ids,
        product=product, components=components, momentum_list=momentum_list,
        usedNe=usedNe, phase_convention=phase_convention,
        path_boundary=path_boundary, selection=selection)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    graph = apply_source_policy(preview, requests, index, source_policy=source_policy)
    stage = graph["stages"][product]
    dataset_spec = stage.get("spec")
    times = (() if dataset_spec is None else tuple(dataset_spec.coordinates("time")))
    return dict(
        requests=requests,
        graph=graph,
        dataset_spec=dataset_spec,
        times=times,
        product=product,
    )




__all__ = [
    "plan_indexed_baryon_elementals",
]
