"""Module tests split from test_diagram_expansion.py; shared numerical fixtures stay unchanged."""

from test_diagram_expansion import (
    ArrayElemental,
    ArrayPropagator,
    ConservedVectorInsertion,
    Diagram,
    DiagramBindings,
    ElementalComponent,
    LegacyDiagramBinding,
    Meson,
    Operator,
    OperatorTerm,
    Propagator,
    QuarkDiagram,
    SamplingGroup,
    SamplingPoolSpec,
    Tag,
    _DiagramExpansionMetadataFixture,
    expand_correlator,
    expand_state_sampling_variants,
    lower_state_graph,
    np,
)
from lqcdweave.contracts.correlator.api import AxisPattern


class DiagramExpansionMetadataTests(_DiagramExpansionMetadataFixture):
    def test_old_sector_axis_order_for_all_four_vertex_types(self):
        adjacency = [[0, 1], [2, 0]]
        expected = {
            ("v", "v"): "v2v",
            ("v", "p"): "v2p",
            ("p", "v"): "p2v",
            ("p", "p"): "p2p",
        }
        for state, sector in expected.items():
            states = [
                {"left": state[0], "right": state[1]},
                {"left": "v", "right": "v"},
            ]
            point_labels = {}
            if state[0] == "p":
                point_labels[(0, "field")] = "pool-A:field"
            if state[1] == "p":
                point_labels[(0, "bar")] = "pool-A:bar"
            (network,) = lower_state_graph(adjacency, states, point_labels=point_labels)
            vertex = next(
                item for item in network.requests
                if item.kind == "vertex" and item.vertex_index == 0
            )
            self.assertEqual(vertex.sector, sector)
            self.assertIsInstance(network.equation, AxisPattern)
            self.assertEqual(network.equation.inputs,
                             tuple(item.subscripts for item in network.requests))
            self.assertEqual(network.equation.output, ())

        (network,) = lower_state_graph(
            adjacency,
            [{"left": "v", "right": "p"}, {"left": "v", "right": "v"}],
            point_labels={(0, "bar"): "P0"},
        )
        self.assertEqual(
            tuple(item.sector for item in network.requests if item.kind == "propagator"),
            ("vsv", "psv"),
        )

    def test_explicit_shared_point_label_joins_disconnected_components(self):
        adjacency = [
            [0, 1, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 3],
            [0, 0, 4, 0],
        ]
        states = [
            {"left": "p", "right": "v"},
            {"left": "v", "right": "v"},
            {"left": "p", "right": "v"},
            {"left": "v", "right": "v"},
        ]
        separate = lower_state_graph(
            adjacency, states,
            point_labels={(0, "field"): "pool-A:0", (2, "field"): "pool-B:0"},
        )
        shared = lower_state_graph(
            adjacency, states,
            point_labels={(0, "field"): "shared:0", (2, "field"): "shared:0"},
        )
        self.assertEqual(len(separate), 2)
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(shared[0].propagator_edges), 4)

    def test_missing_point_pool_is_not_inferred_from_time_or_np(self):
        with self.assertRaisesRegex(ValueError, "no explicit sampling-scene point label"):
            lower_state_graph(
                [[0, 1], [2, 0]],
                [{"left": "p", "right": "v"}, {"left": "v", "right": "v"}],
            )

    def test_fixed_two_current_shared_pool_has_52_scenes(self):
        pools = {"shared": SamplingGroup("shared", (), 64, 4)}
        point_pool_by_leg = {
            (vertex, side): "shared"
            for vertex in range(2) for side in ("field", "bar")
        }
        variants = tuple(expand_state_sampling_variants(
            2, (0, 1), point_pool_by_leg, pools,
        ))
        self.assertEqual(len(variants), 52)
        by_state = {}
        for variant in variants:
            p2p_count = sum(state == {"left": "p", "right": "p"}
                            for state in variant.vertex_states)
            by_state[p2p_count] = by_state.get(p2p_count, 0) + 1
        self.assertEqual(by_state, {0: 13, 1: 24, 2: 15})

    def test_native_maps_expand_all_four_sectors_without_numeric_reads(self):
        diagram, elemental, propagator, vertices, prop_map = self._two_current_fixture()
        elemental.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("numeric elemental read"))
        propagator.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("numeric propagator read"))
        expanded = expand_correlator(
            diagram, adapter=LegacyDiagramBinding(),
            bindings=DiagramBindings(
                prop_map, vertices, configuration="cfg-0", usedNe=1, usedNp=4,
            ),
            time_axes={0: 0, 1: 2},
        )
        self.assertEqual(len(expanded.terms), 52)
        self.assertEqual(set(expanded.inputs.diagram_recipes), {diagram})
        vertex_sectors = {request[3] for request in expanded.inputs.tensor_requests.values()
                          if request[0] == "vertex"}
        prop_sectors = {request[2] for request in expanded.inputs.tensor_requests.values()
                        if request[0] == "propagator"}
        self.assertEqual(vertex_sectors, {"v2v", "v2p", "p2v", "p2p"})
        self.assertEqual(prop_sectors, {"vsv", "psv", "vsp", "psp"})

    def test_missing_raw_p2p_vertex_data_is_not_skipped(self):
        diagram, _elemental, propagator, vertices, prop_map = self._two_current_fixture(include_p2p=False)
        with self.assertRaises(KeyError):
            expand_correlator(
                diagram, adapter=LegacyDiagramBinding(),
                bindings=DiagramBindings(
                    prop_map, vertices, configuration="cfg-0", usedNe=1, usedNp=4,
                ),
                time_axes={0: 0, 1: 2},
            )

    def test_temporal_current_uses_each_selected_real_endpoint_and_shared_pool(self):
        current = ConservedVectorInsertion(direction=3, wilson_r=1.0).operator()
        normal = Operator.from_terms("normal", [OperatorTerm(0, ElementalComponent(0), 0)])
        components = tuple(dict.fromkeys(term.component for op in (current, normal)
                                         for term in op.terms))
        lt, ne, np_count = 4, 1, 4
        sectors = {
            "v2v": np.zeros((len(components), 1, lt, ne, ne), dtype=np.complex128),
            "v2p": np.zeros((len(components), 1, lt, ne, np_count, 3), dtype=np.complex128),
            "p2v": np.zeros((len(components), 1, lt, np_count, 3, ne), dtype=np.complex128),
            "p2p": np.zeros((len(components), 1, lt, np_count, 3, np_count, 3), dtype=np.complex128),
        }
        pool_ids = {(component, 0, time, side): "same-pool-across-times"
                    for component in components for time in range(lt)
                    for side in ("bar", "field")}
        elemental = ArrayElemental(
            sectors, components, phase_convention="anchor", cfg="cfg-1",
            data_identity="cfg-1/elemental", sampling_pool_ids=pool_ids,
            sampling_groups={"same-pool-across-times": SamplingPoolSpec(
                "same-pool-across-times", 64, np_count)},
        )
        propagator = ArrayPropagator({
            "vsv": np.zeros((lt, lt, 4, 4, ne, ne), dtype=np.complex128),
            "psv": np.zeros((lt, lt, 4, 4, np_count, 3, ne), dtype=np.complex128),
            "vsp": np.zeros((lt, lt, 4, 4, ne, np_count, 3), dtype=np.complex128),
            "psp": np.zeros((lt, lt, 4, 4, np_count, 3, np_count, 3), dtype=np.complex128),
        }, time_convention="absolute", cfg="cfg-1", data_identity="cfg-1/peram")
        vertex_map = {
            "current": Meson(elemental, current, phase_convention="anchor"),
            "normal": Meson(elemental, normal, phase_convention="anchor"),
        }
        p0 = Propagator("u", Tag(0, 0), Tag(1, 2))
        p1 = Propagator("u", Tag(1, 2), Tag(0, 0))
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1],
                          ["current", "normal"], [None, p0, p1])
        calls = []
        original_describe = propagator.describe_block

        def record_describe(sink, source, sector="vsv", **kwargs):
            calls.append((sink, source, sector))
            return original_describe(sink, source, sector, **kwargs)

        propagator.describe_block = record_describe
        expanded = expand_correlator(
            diagram, adapter=LegacyDiagramBinding(),
            bindings=DiagramBindings({p0: propagator, p1: propagator}, vertex_map,
                                     configuration="cfg-1", usedNe=1, usedNp=np_count),
            time_axes={0: 3, 1: 2},
        )
        self.assertEqual(len(expanded.terms), 10)
        self.assertIn((2, 0, "vsv"), calls)  # forward field endpoint wraps 3+1 -> 0
        self.assertIn((0, 2, "vsv"), calls)  # backward bar endpoint wraps 3+1 -> 0
