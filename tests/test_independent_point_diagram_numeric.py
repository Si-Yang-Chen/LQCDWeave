"""Full-population independent point pools against a dense Wilson trace."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

import test_published_double_current_diagram as census
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.storage.dataset import DatasetIndex
from lqcdweave.runtime.datasets import DatasetView
from lqcdweave.contraction.correlator import calc_diagram_eval
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
from fixtures.c_group_production import publish_elemental_products, publish_perambulators


class IndependentPointDiagramNumericTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        helper = census.PublishedDoubleCurrentDiagramTests()
        helper.root = Path(temporary.name)
        helper.index = DatasetIndex(helper.root / "index")
        helper.links = census.random_gauge(helper.lattice, seed=2501)
        helper.configuration = dict(
            ensemble="published-point-reference",
            configuration="cfg-double-current",
            lattice=list(helper.lattice),
        )
        helper.basis_values = helper._make_basis_values()
        helper.gauge_id = helper._publish_gauge()
        helper.basis_id = helper._publish_basis()
        helper.v2v_id = helper._publish_v2v()
        helper.current = census.OperatorPath(
            "complex-x-path-current",
            ((1, census.GaugePath((0,)), 1, helper.coefficient),),
            current_terms=(census.CurrentTerm(
                helper.coefficient, 0, (1, 0, 0, 0), 1,
            ),),
            current_term_ranges=((0,),),
        )
        self.helper = helper

    def _publish_sample(self, side, subset):
        helper = self.helper
        labels_for_site = (9, 3, 17, 1) if side == "left" else (4, 8, 2, 6)
        labels = [labels_for_site[site] for site in subset]
        name = side + "-" + "".join(map(str, subset))
        spec = DatasetSpec(dict(
            schema=1,
            kind="spatial_points",
            configuration=helper.configuration,
            axes=[
                dict(name="time", coordinates=[2, 0, 1]),
                dict(name="point", coordinates=labels),
                dict(name="xyz", coordinates=["x", "y", "z"]),
            ],
            dtype="int64",
            bases={},
            parents=[],
            definition=dict(selection="explicit-SRSWOR-realization", pool=side,
                            population_size=4),
            conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
        ))
        rows = {
            time: np.asarray([[site, 0, 0] for site in subset], dtype=np.int64)
            for time in helper.times
        }
        point_id = helper._publish(name, spec, rows)
        propagator_ids = publish_perambulators(
            helper.index, helper.root, helper.gauge_id,
            output_name=name + "-propagators", basis_id=helper.basis_id,
            points_snk_id=point_id, points_src_id=point_id,
            products=("VSV", "PSV", "VSP", "PSP"), Ne=1,
            Np_snk=len(subset), Np_src=len(subset), mass=helper.mass,
            t_boundary=helper.boundary,
        )
        return point_id, propagator_ids

    def _publish_elementals(self, output_name, left, right, *, used_np_left, used_np_right):
        helper = self.helper
        products = publish_elemental_products(
            helper.index, helper.root, helper.gauge_id, helper.basis_id, left[0],
            output_name=output_name, points_right_id=right[0],
            sectors=("v2p", "p2v", "p2p"),
            usedNe=1, usedNp=used_np_left, usedNp_right=used_np_right,
            paths=((), (0,)), momentum_list=((0, 0, 0), (1, 0, 0)),
            phase_convention="anchor",
        )
        return dict(products, v2v=helper.v2v_id)

    def _bind(self, stack, left, right, elemental_ids, source_time, sink_time,
              *, point_prefix=None):
        helper = self.helper
        opened = {}

        def reader(identity):
            if identity not in opened:
                opened[identity] = stack.enter_context(helper.index.open(identity))
            return opened[identity]

        component = helper.current.terms[0].component
        left_points = reader(left[0])
        left_count = len(left_points.spec.coordinates("point"))
        if point_prefix is not None:
            document = left_points.spec.document
            for axis in document["axes"]:
                if axis["name"] == "point":
                    axis["coordinates"] = axis["coordinates"][:point_prefix]
            left_points = DatasetView(
                left_points, DatasetSpec(document), subset_axes=("point",),
            )
            left_count = point_prefix
        right_points = reader(right[0])
        pools = {
            "left-pool": SamplingPoolSpec("left-pool", 4, left_count),
            "right-pool": SamplingPoolSpec(
                "right-pool", 4, len(right_points.spec.coordinates("point")),
            ),
        }
        elemental = DatasetElemental(
            {sector: reader(elemental_ids[sector])
             for sector in ("v2v", "v2p", "p2v", "p2p")},
            point_dataset=left_points,
            point_right_dataset=right_points,
            sampling_pool_ids={
                (component, 1, time, endpoint): pool
                for time in helper.times
                for endpoint, pool in (("field", "left-pool"), ("bar", "right-pool"))
            },
            sampling_groups=pools,
        )
        quark_left = Propagator("u", Tag(0, 0), Tag(1, 1))
        quark_right = Propagator("u", Tag(1, 1), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]),
            [source_time, sink_time],
            ["current-a", "current-b"],
            [None, quark_left, quark_right],
        )
        propagators = {}
        for quark, sample in ((quark_left, left), (quark_right, right)):
            sectors = {sector: reader(sample[1][sector])
                       for sector in ("vsv", "psv", "vsp", "psp")}
            psp_bases = sectors["psp"].spec.document["bases"]
            self.assertEqual(psp_bases["source_point"]["data_id"], sample[0])
            self.assertEqual(psp_bases["sink_point"]["data_id"], sample[0])
            propagators[quark] = DatasetPropagator(sectors)
        vertices = {
            "current-a": Meson(elemental, helper.current, phase_convention="anchor"),
            "current-b": Meson(elemental, helper.current, source=True,
                               phase_convention="anchor"),
        }
        return diagram, propagators, vertices

    def test_planner_preserves_only_the_consumed_point_coordinate_view(self):
        helper = self.helper
        left = self._publish_sample("left", (0, 1, 2))
        right = self._publish_sample("right", (0, 1))
        elementals = self._publish_elementals(
            "prefix-pair", left, right, used_np_left=3, used_np_right=2,
        )
        source, sink = Symbol("prefix-source"), Symbol("prefix-sink")
        with ExitStack() as stack:
            diagram, propagators, vertices = self._bind(
                stack, left, right, elementals, source, sink, point_prefix=2,
            )
            bindings = DiagramBindings(
                propagators, vertices, "cfg-double-current", 1, 2, True,
            )
            with patch.object(DatasetPropagator, "block",
                              side_effect=AssertionError("preflight read propagator")), \
                 patch.object(DatasetElemental, "block",
                              side_effect=AssertionError("preflight read elemental")):
                plan = plan_indexed_diagram_measurement(
                    helper.index, diagram, bindings=bindings,
                    source_time_symbol=source, sink_time_symbol=sink,
                )
            left_roles = [node for role, node in plan["input_roles"].items()
                          if role.startswith("vertex-") and role.endswith("-points")]
            self.assertTrue(left_roles)
            for node in left_roles:
                view = DatasetSpec(plan["requests"][node].template)
                self.assertEqual(view.coordinates("point"), [9, 3])

    def test_full_population_independent_pools_match_full_time_trace(self):
        helper = self.helper
        expected = helper._independent_trace_values()
        sites = tuple(range(4))
        left = self._publish_sample("left", sites)
        right = self._publish_sample("right", sites)
        elementals = self._publish_elementals(
            "full-population-pair", left, right, used_np_left=4, used_np_right=4,
        )
        source, sink = Symbol("trace-source"), Symbol("trace-sink")
        values = {}
        with ExitStack() as stack:
            diagram, propagators, vertices = self._bind(
                stack, left, right, elementals, source, sink,
            )
            for pair in expected:
                with patch.object(
                    DatasetPropagator, "block", autospec=True,
                    side_effect=DatasetPropagator.block,
                ) as reads:
                    calculation = calc_diagram_eval(
                        diagram, propagator_map=propagators, vertex_map=vertices,
                        configuration="cfg-double-current", usedNe=1, usedNp=None,
                        groups_independent=True, optimize="none",
                        time_axes={source: pair[0], sink: pair[1]},
                    )
                values[pair] = calculation.value
                psp_readers = {
                    id(call.args[0]) for call in reads.call_args_list
                    if call.args[3] == "psp"
                }
                self.assertTrue({id(item) for item in propagators.values()} <= psp_readers)
                descriptions = [spec.description
                                for spec in calculation.inputs.problem.specs.values()]
                propagator_descriptions = [item for item in descriptions
                                           if item.role == "propagator"]
                self.assertTrue(propagator_descriptions)
                self.assertTrue(all(
                    check["status"] == "verified"
                    for description in propagator_descriptions
                    for check in description.bindings["endpoint_axis_bindings"].values()
                ))
                if pair == (0, 0):
                    bridges = calculation.expanded.problem.sparse_bridges
                    self.assertTrue(any(
                        sum(other.scene == bridge.scene for other in bridges) == 2
                        for bridge in bridges
                    ))
        for pair in expected:
            with self.subTest(source_time=pair[0], sink_time=pair[1]):
                np.testing.assert_allclose(values[pair], expected[pair],
                                           rtol=3e-10, atol=3e-10)


if __name__ == "__main__":
    unittest.main()
