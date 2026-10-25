"""Small CPU input fixtures for independent contraction/adapter references.

These test helpers publish fresh synthetic inputs through public numerical APIs
and the ordinary writer. They are not production entry points and deliberately
do not implement receipts, admission, reuse or resume policies.
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from lqcdweave.adapters.dense_wilson import DenseWilsonBackend
from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contraction.propagator_views import psv_to_vsp_dagger
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.planning.production import vsp_dataset_document
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, VectorBasis, PointBasis,
)
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.runtime.datasets import open_dataset_view
from lqcdweave.production.perambulator.planning import plan_indexed_perambulators
from fixtures.perambulator_script import produce_source_rows
from lqcdweave.production.elemental.planning import plan_indexed_point_elementals, prepare_elemental_stage, plan_indexed_directed_current
from fixtures.elemental_script import produce_point_elemental, produce_current
from lqcdweave.production.wilson_reference import plan_indexed_wilson_propagators


_PROVENANCE = {"producer": "small-cpu-numerical-test-fixture"}


def publish_directed_current(index, root, gauge_id, basis_id, *, output_name,
                             momentum_list=((0, 0, 0),), usedNe=None,
                             boundary="periodic"):
    plan = plan_indexed_directed_current(index, gauge_id, basis_id,
        momentum_list=momentum_list, usedNe=usedNe, boundary=boundary)
    spec = plan["stages"]["current"]["spec"]
    with index.open(gauge_id) as gauge, open_dataset_view(
            index, plan["stages"]["basis"]["spec"], data_id=basis_id,
            subset_axes=("mode",)) as basis:
        destination = Path(root)/output_name
        with DatasetWriter(destination, spec, provenance=_PROVENANCE) as writer:
            produce_current(gauge, basis, writer)
            identity = writer.publish()
        index.register(destination)
    return identity


def publish_basis(index, root, gauge_id, *, output_name, Ne=2, tolerance=1e-11):
    root = Path(root)/output_name
    with index.open(gauge_id) as gauge:
        cfg = gauge.spec.document["configuration"]
        times = list(range(cfg["lattice"][3]))
        spec = DatasetSpec(laplace_basis_document(cfg, times, gauge_id, Ne, tolerance))
        generator = LaplaceEigenvectorGenerator(cfg["lattice"], DatasetFieldHandle(gauge),
                                                Ne=Ne, tol=tolerance, backend="numpy")
        generator.load(cfg["configuration"])
        values = []
        with DatasetWriter(root/"basis", spec, provenance=_PROVENANCE) as writer:
            for position in writer.missing:
                vectors, eigenvalues = generator.calc(times[position])
                writer.write(position, vectors)
                values.append(np.array(eigenvalues, dtype=np.float64, copy=True))
            basis_id = writer.publish()
        index.register(root/"basis")
        value_spec = DatasetSpec(laplace_eigenvalues_document(
            cfg, times, Ne, basis_id, gauge_id, spec.document))
        with DatasetWriter(root/"eigenvalues", value_spec, provenance=_PROVENANCE) as writer:
            for position in writer.missing:
                writer.write(position, values[position])
            values_id = writer.publish()
        index.register(root/"eigenvalues")
    return {"basis": basis_id, "eigenvalues": values_id}


def publish_elemental_products(index, root, gauge_id, basis_id=None, points_id=None, *,
                               output_name, paths=((),), momentum_list=((0, 0, 0),),
                               sectors=("v2p", "p2v", "p2p"), usedNe=None, usedNp=None,
                               points_right_id=None, usedNp_right=None,
                               point_right_labels=None, phase_convention="anchor",
                               current_schema=None, temporal_path_schema=None,
                               path_boundary="periodic"):
    identities = {}
    root = Path(root)/output_name
    with ExitStack() as stack:
        gauge = stack.enter_context(index.open(gauge_id))
        basis = stack.enter_context(index.open(basis_id)) if basis_id is not None else None
        if "v2v" in sectors:
            generator, spec, provenance, times, cfg = prepare_elemental_stage(
                gauge, basis, paths=paths, momentum_list=momentum_list, usedNe=usedNe,
                phase_convention=phase_convention)
            with DatasetWriter(root/"v2v", spec, provenance=provenance) as writer:
                generator.load(cfg["configuration"])
                for row in writer.missing:
                    writer.write(row, generator.calc_path_blocks(times[row])["v2v"])
                identities["v2v"] = writer.publish()
            index.register(root/"v2v")
        point_sectors = tuple(sector for sector in sectors if sector != "v2v")
        if point_sectors:
            plan = plan_indexed_point_elementals(index, gauge_id,
                basis_id if any(p != "p2p" for p in point_sectors) else None, points_id,
                paths=paths, momentum_list=momentum_list, sectors=point_sectors,
                usedNe=usedNe if any(p != "p2p" for p in point_sectors) else None,
                usedNp=usedNp, points_right_id=points_right_id, usedNp_right=usedNp_right,
                point_right_labels=point_right_labels, phase_convention=phase_convention,
                current_schema=current_schema, temporal_path_schema=temporal_path_schema,
                path_boundary=path_boundary)
            points = stack.enter_context(index.open(points_id))
            right = stack.enter_context(index.open(points_right_id)) if points_right_id else None
            for sector in point_sectors:
                spec = plan["stages"][sector]["spec"]
                with DatasetWriter(root/sector, spec, provenance=_PROVENANCE) as writer:
                    produce_point_elemental(gauge, basis if sector != "p2p" else None,
                                               points, writer, points_right=right)
                    identities[sector] = writer.publish()
                index.register(root/sector)
    return identities


def publish_perambulators(index, root, gauge_id, *, output_name, basis_id=None,
                          points_id=None, points_snk_id=None, points_src_id=None,
                          products=("VSV",), Ne=None, Np=None, Np_snk=None, Np_src=None,
                          mass=.4, tolerance=1e-11, t_boundary=-1, max_dimension=2048,
                          gamma5_vsp=True):
    root = Path(root)/output_name
    sink_id = points_snk_id or points_id
    source_id = points_src_id or points_id
    sink_count = Np_snk if Np_snk is not None else Np
    source_count = Np_src if Np_src is not None else Np
    ids = {"gauge": gauge_id}
    if basis_id is not None:
        ids["basis"] = basis_id
    if points_id is not None:
        ids["points"] = points_id
    else:
        ids.update({name: identity for name, identity in
                    (("points_snk", sink_id), ("points_src", source_id)) if identity})

    def publish_product(product, name, *, point_sink=sink_id, point_source=source_id,
                        count_sink=sink_count, count_source=source_count, direct=False):
        options = {"products": (product,), "Ne": Ne if product != "PSP" else None,
                   "basis_id": basis_id if product != "PSP" else None,
                   "mass": mass, "tolerance": tolerance, "t_boundary": t_boundary,
                   "max_dimension": max_dimension}
        if product in ("PSV", "PSP"):
            options.update(points_snk_id=point_sink, Np_snk=count_sink)
        if product in ("VSP", "PSP"):
            options.update(points_src_id=point_source, Np_src=count_source)
        if direct:
            dirac = dict(action="wilson", mass=mass, xi_0=1., nu=1.,
                         clover_coeff_t=0., clover_coeff_r=0., r=1, lattice_spacing=1)
            options.pop("mass"); options.pop("tolerance"); options.pop("t_boundary")
            options.pop("max_dimension")
            plan = plan_indexed_perambulators(index, gauge_id, **options,
                dirac_definition=dirac, solver_tolerance=tolerance,
                temporal_boundary="antiperiodic" if t_boundary == -1 else "periodic")
        else:
            plan = plan_indexed_wilson_propagators(index, gauge_id, **options)
        spec = plan["stages"][product.lower()]["spec"]
        with ExitStack() as stack:
            gauge = stack.enter_context(index.open(gauge_id))
            basis = stack.enter_context(index.open(basis_id)) if product != "PSP" else None
            sink = stack.enter_context(index.open(point_sink)) if product in ("PSV", "PSP") else None
            source = stack.enter_context(index.open(point_source)) if product in ("VSP", "PSP") else None
            cfg = gauge.spec.document["configuration"]
            src_basis = (VectorBasis(DatasetFieldHandle(basis),count=Ne)
                         if product in ("VSV","PSV") else
                         PointBasis(DatasetPointHandle(source),count=count_source))
            sink_basis = (VectorBasis(DatasetFieldHandle(basis),count=Ne)
                          if product in ("VSV","VSP") else
                          PointBasis(DatasetPointHandle(sink),count=count_sink))
            generator = PerambulatorGenerator(cfg["lattice"], DatasetFieldHandle(gauge),
                mass, tolerance, 100, clover_coeff_t=0., clover_coeff_r=0.,
                t_boundary=t_boundary, contract_prec="complex128",
                backend=DenseWilsonBackend(max_dimension=max_dimension))
            with DatasetWriter(root/name, spec, provenance=_PROVENANCE) as writer:
                produce_source_rows(writer, generator, spec.coordinates("source_time"),
                                        cfg["configuration"], product,src=src_basis,sink=sink_basis)
                identity = writer.publish()
            index.register(root/name)
        return identity

    for product in products:
        if product != "VSP":
            ids[product.lower()] = publish_product(product, product.lower())
    if "VSP" in products:
        if gamma5_vsp:
            if "psv" in ids and sink_id == source_id and sink_count == source_count:
                parent_id = ids["psv"]
            else:
                parent_id = publish_product("PSV", "psv_for_vsp", point_sink=source_id,
                                            count_sink=source_count)
                ids["psv_for_vsp"] = parent_id
            with index.open(parent_id) as parent:
                spec = DatasetSpec(vsp_dataset_document(parent.spec.document,
                    psv_id=parent_id, gauge_id=gauge_id, basis_id=basis_id, points_id=source_id))
                with DatasetWriter(root/"vsp", spec, provenance=_PROVENANCE) as writer:
                    for row in writer.missing:
                        source_time = spec.coordinates("source_time")[row]
                        values = [psv_to_vsp_dagger(np.expand_dims(parent.read(
                            source_time=t, sink_time=source_time), 0), gamma(15), backend=np)[0]
                            for t in spec.coordinates("sink_time")]
                        writer.write(row, np.stack(values))
                    ids["vsp"] = writer.publish()
                index.register(root/"vsp")
        else:
            ids["vsp"] = publish_product("VSP", "vsp", direct=True)
    return ids
