"""Term-level temporal endpoint wiring against the independent Wilson oracle."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

import test_directed_current_point_diagram as fixture
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import ConservedVectorCurrent
from lqcdweave.contraction import diagram_measurement
from lqcdweave.contraction.correlator import calc_diagram_eval
from fixtures.c_group_production import publish_directed_current, publish_perambulators


class TemporalCurrentGraphEndpointMappingTests(unittest.TestCase):
    def test_cross_boundary_mixed_diagram_cell_matches_dense_current_trace(self):
        case = fixture.RegisteredTemporalPointCurrentTests()
        case.setUp()
        try:
            vv_id, elemental_ids, wilson_ids = case._produce_census_diagram_inputs()
            modes = np.stack([case.mode_values[time] for time in range(case.lattice[3])])
            points_a = np.stack([case.left_coordinates[time] for time in range(case.lattice[3])])
            points_b = np.stack([case.right_coordinates[time] for time in range(case.lattice[3])])
            expected = fixture.projected_current_trace_grid(
                case.links, modes, modes, points_a, points_b, 0.4,
                momentum=(0, 0, 0),
            )
            with ExitStack() as stack:
                diagram, bindings, source_time, sink_time, _callers = case._open_census_diagram(
                    stack, vv_id, elemental_ids, wilson_ids,
                )
                elemental_sectors = set()
                propagator_sectors = set()
                elemental_block = DatasetElemental.block
                propagator_block = DatasetPropagator.block

                def observe_elemental(reader, component, momentum, time, sector, **selection):
                    elemental_sectors.add(sector)
                    return elemental_block(reader, component, momentum, time, sector, **selection)

                def observe_propagator(reader, sink, source, sector="vsv", **selection):
                    propagator_sectors.add(sector)
                    return propagator_block(reader, sink, source, sector, **selection)

                with patch.object(DatasetElemental, "block", new=observe_elemental), \
                     patch.object(DatasetPropagator, "block", new=observe_propagator):
                    result = calc_diagram_eval(
                        diagram,
                        propagator_map=bindings.propagator_map,
                        vertex_map=bindings.vertex_map,
                        configuration=bindings.configuration,
                        usedNe=bindings.usedNe,
                        usedNp=bindings.usedNp,
                        groups_independent=bindings.groups_independent,
                        time_axes={source_time: 2, sink_time: 1},
                        batch="auto", optimize="none", backend=np, execute=True,
                    )
            self.assertEqual(np.asarray(result.value).shape, ())
            self.assertEqual(elemental_sectors, {"v2v", "v2p", "p2v", "p2p"})
            self.assertEqual(propagator_sectors, {"vsv", "psv", "vsp", "psp"})
            np.testing.assert_allclose(result.value, expected[2, 1], rtol=3e-10, atol=3e-10)
        finally:
            case.doCleanups()

    def test_open_temporal_current_endpoint_is_rejected_during_metadata_preflight(self):
        case = fixture.RegisteredTemporalPointCurrentTests()
        case.setUp()
        try:
            case._reset_to_census_current_fixture()
            case.momentum_list = ((0, 0, 0),)
            point_elementals = {}
            for name, identity, labels in (
                    ("a", case.left_id, case.left_labels),
                    ("b", case.right_id, case.right_labels)):
                result = case._produce_point_current(
                    boundary="open", name=f"open-point-current-{name}",
                    left_id=identity, right_id=identity,
                    left_labels=labels, right_labels=labels,
                )
                point_elementals[name] = result["datasets"]
            vv_id = publish_directed_current(
                case.index, case.root, case.gauge_id, case.basis_id,
                output_name="open-directed-vv", momentum_list=case.momentum_list,
                usedNe=2, boundary="open",
            )
            wilson_ids = {}
            for orientation, points_sink, points_source in (
                    ("ab", case.right_id, case.left_id),
                    ("ba", case.left_id, case.right_id)):
                wilson_ids[orientation] = publish_perambulators(
                    case.index, case.root, case.gauge_id,
                    output_name=f"open-wilson-{orientation}", basis_id=case.basis_id,
                    points_snk_id=points_sink, points_src_id=points_source,
                    products=("VSV", "PSV", "VSP", "PSP"),
                    Ne=2, Np_snk=3, Np_src=3, mass=0.4, t_boundary=+1,
                )

            with ExitStack() as stack:
                readers = {}

                def open_once(identity):
                    if identity not in readers:
                        readers[identity] = stack.enter_context(case.index.open(identity))
                    return readers[identity]

                vv_document = open_once(vv_id).spec.document
                for axis in vv_document["axes"]:
                    if axis["name"] == "component":
                        axis["coordinates"] = [{"path": [6]}, {"path": [7]}]
                vv_spec = fixture.DatasetSpec(vv_document)
                vv_view = stack.enter_context(fixture.open_dataset_view(
                    case.index, vv_spec, subset_axes=("component",), data_id=vv_id,
                ))

                operator = ConservedVectorCurrent().operator(3)
                vertices = {}
                for name in ("a", "b"):
                    datasets = {
                        "v2v": vv_view,
                        **{sector: open_once(point_elementals[name][sector])
                           for sector in ("v2p", "p2v", "p2p")},
                    }
                    point_reader = open_once(case.left_id if name == "a" else case.right_id)
                    pools, pool_ids = {}, {}
                    for term in operator.terms:
                        for time in range(case.lattice[3]):
                            pool_id = f"open-pool-{name}-{time}"
                            pools[pool_id] = SamplingPoolSpec(
                                pool_id, spatial_sites=3, sampled_points=3,
                            )
                            for side in ("bar", "field"):
                                pool_ids[(term.component, 0, time, side)] = pool_id
                    elemental = DatasetElemental(
                        datasets, point_dataset=point_reader,
                        point_right_dataset=point_reader,
                        sampling_pool_ids=pool_ids, sampling_groups=pools,
                    )
                    vertices[f"current-{name}"] = Meson(
                        elemental, operator, source=(name == "b"),
                        phase_convention="midpoint", boundary="open",
                    )

                symbols = {
                    "ab": Propagator("u", Tag(0, 0), Tag(1, 1)),
                    "ba": Propagator("u", Tag(1, 1), Tag(0, 0)),
                }
                propagators = {}
                for orientation, symbol in symbols.items():
                    products = {
                        sector: open_once(identity)
                        for sector, identity in wilson_ids[orientation].items()
                        if sector in ("vsv", "psv", "vsp", "psp")
                    }
                    propagators[symbol] = DatasetPropagator(products)
                source_time, sink_time = Symbol("open-source-time"), Symbol("open-sink-time")
                diagram = Diagram(
                    QuarkDiagram([[0, 1], [2, 0]]), [source_time, sink_time],
                    ["current-a", "current-b"],
                    [None, symbols["ab"], symbols["ba"]],
                )
                bindings = DiagramBindings(
                    propagators, vertices,
                    case.configuration["configuration"], 2, 3, True,
                )
                numeric_calls = []

                def reject_elemental_read(*args, **kwargs):
                    numeric_calls.append("elemental")
                    raise AssertionError("numeric elemental block read before endpoint preflight")

                def reject_propagator_read(*args, **kwargs):
                    numeric_calls.append("propagator")
                    raise AssertionError("numeric propagator block read before endpoint preflight")

                with patch.object(DatasetElemental, "block", new=reject_elemental_read), \
                     patch.object(DatasetPropagator, "block", new=reject_propagator_read):
                    forward = operator.terms[0]
                    self.assertEqual(forward.current_term.link, "forward")
                    with self.assertRaisesRegex(ValueError, "lies outside open temporal domain"):
                        vertices["current-a"].describe_factors(
                            case.lattice[3] - 1, "v2p", indices=(0,), usedNe=2, usedNp=3,
                        )
                    with self.assertRaisesRegex(ValueError, "lies outside open temporal domain"):
                        diagram_measurement.plan_indexed_diagram_measurement(
                            case.index, diagram, bindings=bindings,
                            source_time_symbol=source_time,
                            sink_time_symbol=sink_time,
                        )
                self.assertEqual(numeric_calls, [])
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
