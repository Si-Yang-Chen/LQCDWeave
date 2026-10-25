"""Read-only Wilson reference planning and product request construction."""

from importlib.metadata import version
from pathlib import Path
import numpy as np

from lqcdweave.adapters.dense_wilson import DenseWilsonBackend
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.contracts.dataset import DatasetSpec, DatasetRequest
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath, integer
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.planning.production import validate_gauge_field, elemental_dataset_document
from lqcdweave.planning.production import (
    validate_elemental_fields, validate_point_field, propagator_dataset_document,
    vsp_dataset_document,
)
from lqcdweave.planning.laplace import (
    laplace_basis_document, laplace_eigenvalues_document,
)
from lqcdweave.planning.meson import meson_measurement_document
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.runtime.dataset_stage import stage_plan
from lqcdweave.storage.dataset import DatasetReader
from lqcdweave.production.elemental.planning import prepare_elemental_stage
from lqcdweave.contraction.meson_measurement import prepare_meson_stage
from lqcdweave.runtime.datasets import apply_source_policy


def _reference_input(gauge_data, max_dimension):
    times = validate_gauge_field(gauge_data.spec)
    if gauge_data.spec.document["conventions"].get("boundary") != "periodic":
        raise ValueError("reference chain requires explicitly periodic gauge boundary")
    cfg = gauge_data.spec.document["configuration"]
    lx, ly, lz, lt = cfg["lattice"]
    if 12 * lx * ly * lz * lt > max_dimension:
        raise ValueError("lattice exceeds bounded Wilson reference dimension")
    return times, cfg


def _reference_provenance():
    provenance = local_provenance()
    provenance["scipy"] = version("scipy")
    provenance["workflow"] = "wilson_reference"
    return provenance


def _basis_spec(cfg, times, gauge_id, ne, tolerance):
    return DatasetSpec(laplace_basis_document(cfg, times, gauge_id, ne, tolerance))


def _values_spec(cfg, times, ne, basis_id, gauge_id, basis_spec):
    return DatasetSpec(laplace_eigenvalues_document(cfg, times, ne, basis_id, gauge_id, basis_spec.document))


def _vsv_document(cfg, times, ne, basis_id, gauge_id, mass, tolerance, t_boundary):
    return propagator_dataset_document(cfg, times, product="VSV", basis_id=basis_id,
        modes=range(ne), gauge_id=gauge_id, solver_tolerance=tolerance,
        dirac=dict(action="wilson", mass=float(mass), r=1, lattice_spacing=1, clover=0),
        temporal_boundary="antiperiodic" if t_boundary == -1 else "periodic")


def _vsv_spec(cfg, times, ne, basis_id, gauge_id, mass, tolerance, t_boundary):
    return DatasetSpec(_vsv_document(cfg, times, ne, basis_id, gauge_id, mass, tolerance, t_boundary))


def _pseudoscalar_operator():
    return Operator.from_terms("pseudoscalar", [
        OperatorTerm(15, ElementalComponent.path(GaugePath()), 0)])


def plan_wilson_reference(gauge, output, *, Ne=2, mass=.4, tolerance=1e-11,
                          t_boundary=-1, momentum=(1,0,0), max_dimension=2048,
                          configuration=None):
    """Inspect explicit stage directories without creating or changing files.

    Downstream stages are blocked until their actual parent dataset IDs exist.
    ``compute`` includes exact missing first-axis block indices; ``reuse`` has a
    published ID. Incompatible metadata or malformed array layout raises.
    """
    output = Path(output)
    backend = DenseWilsonBackend(max_dimension=max_dimension)
    momentum = tuple(integer(p, "momentum") for p in momentum)
    if len(momentum) != 3:
        raise ValueError("XYZ momentum required")
    def blocked(*requires):
        return dict(action="blocked", requires=requires, data_id=None,
                    completed=(), missing=())
    with DatasetReader(gauge, cfg=configuration) as gauge_data:
        times, cfg = _reference_input(gauge_data, backend.max_dimension)
        LaplaceEigenvectorGenerator(cfg["lattice"], DatasetFieldHandle(gauge_data),
                                   Ne, tolerance, backend="numpy")
        provenance = _reference_provenance()
        basis_spec = _basis_spec(cfg, times, gauge_data.data_id, Ne, tolerance)
        stages = {"basis": stage_plan(output / "basis", basis_spec, provenance)}
        if stages["basis"]["action"] != "reuse":
            stages.update(eigenvalues=blocked("basis"), vsv=blocked("basis"),
                          elemental=blocked("basis"),
                          correlator=blocked("vsv", "elemental"))
            return dict(gauge_id=gauge_data.data_id, stages=stages)
        basis_id = stages["basis"]["data_id"]
        values_spec = _values_spec(cfg, times, Ne, basis_id, gauge_data.data_id,
                                   basis_spec)
        vsv_spec = _vsv_spec(cfg, times, Ne, basis_id, gauge_data.data_id,
                             mass, tolerance, t_boundary)
        stages["eigenvalues"] = stage_plan(output / "eigenvalues", values_spec,
                                            provenance)
        stages["vsv"] = stage_plan(output / "vsv", vsv_spec, provenance)
        with DatasetReader(output / "basis", cfg=cfg["configuration"]) as basis_data:
            _, elemental_spec, elemental_provenance, _, _ = prepare_elemental_stage(
                gauge_data, basis_data, momentum_list=[momentum],
                phase_convention="anchor")
            stages["elemental"] = stage_plan(output / "elemental", elemental_spec,
                                              elemental_provenance)
        if (stages["vsv"]["action"] != "reuse" or
                stages["elemental"]["action"] != "reuse"):
            stages["correlator"] = blocked("vsv", "elemental")
        else:
            with DatasetReader(output / "vsv", cfg=cfg["configuration"]) as vsv_data, \
                 DatasetReader(output / "elemental", cfg=cfg["configuration"]) as elemental_data:
                operator = _pseudoscalar_operator()
                correlator_spec, _, _, _ = prepare_meson_stage(
                    (vsv_data, vsv_data, elemental_data, elemental_data),
                    operator, operator)
                stages["correlator"] = stage_plan(output / "correlator",
                    correlator_spec, local_provenance())
        return dict(gauge_id=gauge_data.data_id, stages=stages)


def _checked_wilson_parameters(mass, tolerance, t_boundary, max_dimension):
    limit = integer(max_dimension, "max_dimension", 1)
    if isinstance(mass, (bool, np.bool_)) or not np.isreal(mass) or not np.isfinite(mass):
        raise ValueError("finite real lattice mass required")
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isreal(tolerance)
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("tolerance must be finite and between zero and one")
    if isinstance(t_boundary, (bool, np.bool_)) or t_boundary not in (-1, 1):
        raise ValueError("t_boundary must be +1 or -1")
    return limit


def _indexed_wilson_requests(index, gauge_id, *, Ne, mass, tolerance,
                             t_boundary, momentum, max_dimension, basis_id=None):
    """Build explicit logical dependencies without constructing numeric producers."""
    ne = integer(Ne, "Ne", 1)
    limit = _checked_wilson_parameters(mass, tolerance, t_boundary, max_dimension)
    momentum = tuple(integer(p, "momentum") for p in momentum)
    if len(momentum) != 3:
        raise ValueError("XYZ momentum required")
    with index.open(gauge_id) as gauge:
        times, cfg = _reference_input(gauge, limit)
        if ne > 3 * int(np.prod(cfg["lattice"][:3])):
            raise ValueError("Ne exceeds spatial/color dimension")
        ref_gauge, ref_basis = dict(request="gauge"), dict(request="basis")
        basis_template = laplace_basis_document(cfg, times, ref_gauge, ne, tolerance)
        basis_description = _basis_spec(cfg, times, gauge_id, ne, tolerance)
        requests = dict(gauge=DatasetRequest(gauge.spec, data_id=gauge_id),
            basis=DatasetRequest(basis_template, subset_axes=("time", "mode"), data_id=basis_id),
            eigenvalues=DatasetRequest(laplace_eigenvalues_document(cfg, times, ne, ref_basis,
                ref_gauge, basis_template), subset_axes=("time", "mode")),
            vsv=DatasetRequest(_vsv_document(cfg, times, ne, ref_basis, ref_gauge,
                mass, tolerance, t_boundary),
                subset_axes=("source_time", "sink_mode", "source_mode")),
            elemental=DatasetRequest(elemental_dataset_document(gauge.spec, basis_description,
                gauge_id=ref_gauge, basis_id=ref_basis, components=[dict(path=[])],
                momenta=[list(momentum)], modes=list(range(ne)), phase_convention="anchor"),
                subset_axes=("time", "sink_mode", "source_mode")))
        operator_definition = [dict(gamma=15, path=[], momentum=list(momentum),
                                    coefficient=[1.0, 0.0])]
        ref_vsv, ref_elemental = dict(request="vsv"), dict(request="elemental")
        requests["correlator"] = DatasetRequest(meson_measurement_document(cfg, times,
            [requests["vsv"].template] * 2 + [requests["elemental"].template] * 2,
            [ref_vsv] * 2 + [ref_elemental] * 2, operator_definition, operator_definition),
            subset_axes=("source_time", "sink_time"))
    return requests, tuple(times), cfg, momentum, ne


def plan_indexed_wilson_reference(index, gauge_id, *, Ne=2, mass=.4, tolerance=1e-11,
                                  t_boundary=-1, momentum=(1, 0, 0), max_dimension=2048,
                                  basis_id=None, source_policy="allow_unknown"):
    """Plan the complete reference dependency graph over registered actual inputs.

    All physical times are requested. Missing parent IDs block descendants
    until publication; this preview performs no solve or output creation.
    """
    requests, _, _, _, _ = _indexed_wilson_requests(index, gauge_id, Ne=Ne, mass=mass,
        tolerance=tolerance, t_boundary=t_boundary, momentum=momentum,
        max_dimension=max_dimension, basis_id=basis_id)
    preview = plan_dataset_graph(requests, index.observations())
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def _field_prefix(spec, axis, count):
    coordinates = spec.coordinates(axis)
    count = len(coordinates) if count is None else integer(count, axis + " count", 1)
    if count > len(coordinates):
        raise ValueError(f"requested {axis} count exceeds actual field coverage")
    coordinates = coordinates[:count]
    document = spec.document
    for description in document["axes"]:
        if description["name"] == axis:
            description["coordinates"] = coordinates
    if axis in document["bases"]:
        document["bases"][axis]["indices"] = coordinates
    return DatasetSpec(document), coordinates


def _indexed_wilson_propagator_requests(index, gauge_id, *, basis_id, points_id,
                                       points_snk_id, points_src_id, products,
                                       Ne, Np, Np_snk, Np_src, mass, tolerance,
                                       t_boundary, max_dimension):
    limit = _checked_wilson_parameters(mass, tolerance, t_boundary, max_dimension)
    if isinstance(products, str):
        raise TypeError("products must be a sequence of VSV/PSV/VSP/PSP names")
    products = tuple(products)
    if (not products or any(product not in ("VSV", "PSV", "VSP", "PSP") for product in products)
            or len(set(products)) != len(products)):
        raise ValueError("distinct VSV/PSV/VSP/PSP products required")
    needs_vectors = any(product in ("VSV", "PSV", "VSP") for product in products)
    needs_points_snk = any(product in ("PSV", "PSP") for product in products)
    needs_points_src = any(product in ("VSP", "PSP") for product in products)
    if needs_vectors and basis_id is None:
        raise ValueError("registered actual basis_id required for vector-source products")
    explicit_points = any(value is not None for value in (points_snk_id, points_src_id))
    if points_id is not None and explicit_points:
        raise ValueError("pass points_id or explicit points_snk_id/points_src_id")
    if points_id is not None and (Np_snk is not None or Np_src is not None):
        raise ValueError("pass points_id with Np, or explicit point IDs with endpoint counts")
    if points_id is not None:
        points_snk_id = points_src_id = points_id
    if needs_points_snk and points_snk_id is None:
        raise ValueError("registered actual points_snk_id required for sink-point products")
    if needs_points_src and points_src_id is None:
        raise ValueError("registered actual points_src_id required for source-point products")
    if not needs_points_snk and points_snk_id is not None and points_id is None:
        raise ValueError("points_snk_id is unused by the selected products")
    if not needs_points_src and points_src_id is not None and points_id is None:
        raise ValueError("points_src_id is unused by the selected products")
    if Np is not None and (Np_snk is not None or Np_src is not None):
        raise ValueError("pass Np or the endpoint-specific Np_snk/Np_src counts")
    if points_id is not None:
        Np_snk = Np if needs_points_snk else None
        Np_src = Np if needs_points_src else None
    else:
        Np_snk = Np if needs_points_snk and Np_snk is None else Np_snk
        Np_src = Np if needs_points_src and Np_src is None else Np_src
    if not needs_points_snk and Np_snk is not None:
        raise ValueError("Np_snk is unused by the selected products")
    if not needs_points_src and Np_src is not None:
        raise ValueError("Np_src is unused by the selected products")
    if not needs_vectors and (basis_id is not None or Ne is not None):
        raise ValueError("pure PSP request has no vector-basis dependency")
    if not needs_points_snk and not needs_points_src and (points_id is not None or Np is not None):
        raise ValueError("pure VSV request has no point-set dependency")
    modes = ()
    point_labels = {}
    point_nodes = {}
    producer_stages = {}
    with index.open(gauge_id) as gauge:
        times, cfg = _reference_input(gauge, limit)
        requests = dict(gauge=DatasetRequest(gauge.spec, data_id=gauge_id))
        if needs_vectors:
            with index.open(basis_id) as basis:
                validate_elemental_fields(gauge.spec, basis.spec)
                view, modes = _field_prefix(basis.spec, "mode", Ne)
                requests["basis"] = DatasetRequest(view, subset_axes=("mode",), data_id=basis_id)
        point_roles = []
        if needs_points_snk:
            point_roles.append(("sink", points_snk_id, Np_snk))
        if needs_points_src:
            point_roles.append(("source", points_src_id, Np_src))
        point_specs = {}
        for role, identity, count in point_roles:
            if identity not in point_specs:
                with index.open(identity) as points:
                    validate_point_field(gauge.spec, points.spec)
                    point_specs[identity] = points.spec
            view, labels = _field_prefix(point_specs[identity], "point", count)
            point_labels[role] = tuple(labels)
            point_nodes[role] = identity

        shared_pool = (needs_points_snk and needs_points_src and
                       points_snk_id == points_src_id)
        if shared_pool:
            point_node = "points"
            max_count = max(len(point_labels["sink"]), len(point_labels["source"]))
            view, _ = _field_prefix(point_specs[points_snk_id], "point", max_count)
            requests[point_node] = DatasetRequest(
                view, subset_axes=("point",), data_id=points_snk_id)
            point_nodes["sink"] = point_nodes["source"] = point_node
        else:
            for role, identity, count in point_roles:
                view, labels = _field_prefix(point_specs[identity], "point", count)
                if points_id is not None:
                    node = "points"
                else:
                    node = "points_snk" if role == "sink" else "points_src"
                requests[node] = DatasetRequest(
                    view, subset_axes=("point",), data_id=identity)
                point_nodes[role] = node

        binding_ids = {"gauge": gauge_id}
        if needs_vectors:
            binding_ids["basis"] = basis_id
        for role, identity, _ in point_roles:
            binding_ids[point_nodes[role]] = identity

        common = dict(solver_tolerance=tolerance,
            dirac=dict(action="wilson", mass=float(mass), r=1, lattice_spacing=1, clover=0),
            temporal_boundary="antiperiodic" if t_boundary == -1 else "periodic")
        if "VSV" in products:
            template = propagator_dataset_document(cfg, times, product="VSV",
                gauge_id=dict(request="gauge"), basis_id=dict(request="basis"),
                modes=modes, **common)
            requests["vsv"] = DatasetRequest(
                template, subset_axes=("source_time", "sink_mode", "source_mode"))
            producer_stages["vsv"] = dict(product="VSV")

        if "PSV" in products:
            template = propagator_dataset_document(cfg, times, product="PSV",
                gauge_id=dict(request="gauge"), basis_id=dict(request="basis"), modes=modes,
                points_snk_id=dict(request=point_nodes["sink"]),
                point_snk_labels=point_labels["sink"], **common)
            requests["psv"] = DatasetRequest(
                template, subset_axes=("source_time", "sink_point", "source_mode"))
            producer_stages["psv"] = dict(
                product="PSV", point_node=point_nodes["sink"],
                point_count=len(point_labels["sink"]))

        psv_for_vsp = None
        if "VSP" in products:
            same_requested_points = (
                "PSV" in products and points_snk_id == points_src_id and
                point_labels["sink"] == point_labels["source"])
            if same_requested_points:
                psv_for_vsp = "psv"
            else:
                psv_for_vsp = "psv_vsp" if "PSV" in products else "psv"
                template = propagator_dataset_document(cfg, times, product="PSV",
                    gauge_id=dict(request="gauge"), basis_id=dict(request="basis"), modes=modes,
                    points_snk_id=dict(request=point_nodes["source"]),
                    point_snk_labels=point_labels["source"], **common)
                requests[psv_for_vsp] = DatasetRequest(
                    template, subset_axes=("source_time", "sink_point", "source_mode"))
                producer_stages[psv_for_vsp] = dict(
                    product="PSV", point_node=point_nodes["source"],
                    point_count=len(point_labels["source"]))
            psv_spec = requests[psv_for_vsp].resolve(binding_ids)
            vsp = vsp_dataset_document(psv_spec.document,
                psv_id=dict(request=psv_for_vsp), gauge_id=gauge_id,
                basis_id=basis_id, points_id=points_src_id)
            requests["vsp"] = DatasetRequest(
                vsp, subset_axes=("source_time", "sink_mode", "source_point"))
            producer_stages["vsp"] = dict(product="VSP", psv_node=psv_for_vsp)

        if "PSP" in products:
            template = propagator_dataset_document(cfg, times, product="PSP",
                gauge_id=dict(request="gauge"),
                points_snk_id=dict(request=point_nodes["sink"]),
                point_snk_labels=point_labels["sink"],
                points_src_id=dict(request=point_nodes["source"]),
                point_src_labels=point_labels["source"], **common)
            requests["psp"] = DatasetRequest(
                template, subset_axes=("source_time", "sink_point", "source_point"))
            producer_stages["psp"] = dict(
                product="PSP", point_snk_node=point_nodes["sink"],
                point_src_node=point_nodes["source"],
                point_snk_count=len(point_labels["sink"]),
                point_src_count=len(point_labels["source"]))
    return requests, tuple(times), cfg, producer_stages


def plan_indexed_wilson_propagators(index, gauge_id, *, basis_id=None, points_id=None,
                                    products=("VSV",), Ne=None, Np=None, mass=.4,
                                    tolerance=1e-11, t_boundary=-1, max_dimension=2048,
                                    data_ids=None, points_snk_id=None,
                                    points_src_id=None, Np_snk=None, Np_src=None,
                                    source_policy="allow_unknown"):
    """Plan requested VSV/PSV/VSP/PSP over registered actual vector/point inputs.

    Pure PSP has no Laplace dependency. VSP adds one explicit PSV dependency.
    All products request every physical
    source and sink time; input prefixes remain views over explicit actual IDs.
    No producer or output directory is constructed by this preview.
    """
    requests, _, _, _ = _indexed_wilson_propagator_requests(index, gauge_id,
        basis_id=basis_id, points_id=points_id, points_snk_id=points_snk_id,
        points_src_id=points_src_id, products=products, Ne=Ne, Np=Np,
        Np_snk=Np_snk, Np_src=Np_src,
        mass=mass, tolerance=tolerance, t_boundary=t_boundary, max_dimension=max_dimension)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)
