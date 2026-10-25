"""Single-current sparse P2P adapter and native direct-closure checks."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.adapters.sparse_elemental import SparseElemental
from lqcdweave.contraction.expansion.legacy_diagram import lower_state_graph
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.contracts.correlator.api import AxisPattern, DiagramBindings
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import (
    ElementalComponent, SparsePointBlock, SparsePointRecordMetadata,
)
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Propagator
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorPath, OperatorTerm
from lqcdweave.contraction.correlator import (
    calc_diagram_eval, execute_contraction_path,
)
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


class _RecordReader:
    def __init__(self, records, point_shape):
        self.records = records
        self.point_shape = point_shape
        self.reads = []
        self.closed = []
        self.invalid_override = None
        self.close_error = None

    def describe_record(self, component, momentum, t):
        record = self.records[component, momentum, t]
        kind = record["type"]
        if kind == "identity":
            nnz, value_dtype, index_dtype, nbytes = min(self.point_shape), np.dtype("complex128"), np.dtype("int64"), 0
        else:
            indices, values = record["indices"], record["values"]
            nnz = len(indices)
            value_dtype, index_dtype = values.dtype, indices.dtype
            nbytes = indices.nbytes + values.nbytes
        return SparsePointRecordMetadata(
            kind, self.point_shape, nnz, value_dtype, index_dtype, nbytes,
            data_identity="cfg-A/current-p2p",
            selection_identity=f"component={component.key!r};momentum={momentum};time={t}",
        )

    def read_record(self, component, momentum, t):
        self.reads.append((component, momentum, t))
        if self.invalid_override is not None:
            return self.invalid_override
        return self.records[component, momentum, t]

    def close_record(self, component, momentum, t, record):
        self.closed.append((component, momentum, t, record))
        if self.close_error is not None:
            raise self.close_error


class SparseDiagramBindingTests(unittest.TestCase):
    def _case(self, record, *, source=False, seed=241, point_count=2):
        rng = np.random.default_rng(seed)
        current_base = LocalVectorCurrent().operator(3)
        current_term = current_base.current_terms[0]
        current_weight = 1.25 - 0.6j
        current_operator = OperatorPath(
            "weighted-local-current",
            ((current_base.terms[0].gamma, current_base.terms[0].component.key,
              current_base.terms[0].momentum, current_weight,
              current_base.terms[0].anchor_time_offset),),
            current_terms=(current_term,), current_term_ranges=((0,),),
        )
        normal_component = ElementalComponent(0)
        normal_weight = -0.4 + 0.9j
        normal_operator = Operator.from_terms(
            "normal", (OperatorTerm(2, normal_component, 0, normal_weight),))
        current_component = current_operator.terms[0].component
        components = tuple(dict.fromkeys((current_component, normal_component)))
        nt, ne, np_ = 2, 1, point_count
        dense_sectors = {
            "v2v": _complex(rng, (len(components), 1, nt, ne, ne)),
            "v2p": _complex(rng, (len(components), 1, nt, ne, np_, 3)),
            "p2v": _complex(rng, (len(components), 1, nt, np_, 3, ne)),
        }
        dense_p2p = np.zeros((len(components), 1, nt, np_, 3, np_, 3), dtype=np.complex128)
        component_index = {component: index for index, component in enumerate(components)}
        if record["type"] == "identity":
            for point in range(np_):
                dense_p2p[component_index[current_component], 0, 0, point, :, point, :] = np.eye(3)
        else:
            for (left, right), values in zip(record["indices"], record["values"], strict=True):
                dense_p2p[component_index[current_component], 0, 0, left, :, right, :] += values
        pool_ids = {
            (component, 0, t, side): "ordered-full-sample"
            for component in components for t in range(nt)
            for side in ("bar", "field")
        }
        pool = SamplingPoolSpec("ordered-full-sample", np_, np_)
        dense_reader = ArrayElemental(
            dense_sectors, components, phase_convention="anchor", cfg="cfg-A",
            data_identity="cfg-A/elemental", sampling_pool_ids=pool_ids,
            sampling_groups={pool.group_id: pool},
        )
        reference_reader = ArrayElemental(
            {**dense_sectors, "p2p": dense_p2p}, components,
            phase_convention="anchor", cfg="cfg-A", data_identity="cfg-A/elemental",
            sampling_pool_ids=pool_ids, sampling_groups={pool.group_id: pool},
        )
        record_reader = _RecordReader(
            {(current_component, 0, 0): record}, (np_, np_),
        )
        sparse_reader = SparseElemental(dense_reader, record_reader)
        peram = {
            "vsv": _complex(rng, (nt, nt, 4, 4, ne, ne)),
            "psv": _complex(rng, (nt, nt, 4, 4, np_, 3, ne)),
            "vsp": _complex(rng, (nt, nt, 4, 4, ne, np_, 3)),
            "psp": _complex(rng, (nt, nt, 4, 4, np_, 3, np_, 3)),
        }
        prop = ArrayPropagator(peram, time_convention="absolute", cfg="cfg-A",
                               data_identity="cfg-A/perambulator")
        quark_out = Propagator("u", Tag(0, 0), Tag(1, 0))
        quark_in = Propagator("u", Tag(1, 0), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [0, 0], ["current", "normal"],
            [None, quark_out, quark_in],
        )
        sparse_vertices = {
            "current": Meson(sparse_reader, current_operator, source=source,
                             phase_convention="anchor"),
            "normal": Meson(sparse_reader, normal_operator,
                            phase_convention="anchor"),
        }
        dense_vertices = {
            "current": Meson(reference_reader, current_operator, source=source,
                             phase_convention="anchor"),
            "normal": Meson(reference_reader, normal_operator,
                            phase_convention="anchor"),
        }
        bindings_sparse = DiagramBindings(
            {quark_out: prop, quark_in: prop}, sparse_vertices,
            configuration="cfg-A", usedNe=ne, usedNp=np_,
        )
        bindings_dense = DiagramBindings(
            {quark_out: prop, quark_in: prop}, dense_vertices,
            configuration="cfg-A", usedNe=ne, usedNp=np_,
        )
        return {
            "record": record, "record_reader": record_reader,
            "dense_reader": dense_reader, "reference_reader": reference_reader,
            "sparse_reader": sparse_reader, "prop": prop, "peram": peram,
            "diagram": diagram, "current_operator": current_operator,
            "normal_operator": normal_operator,
            "sparse_vertices": sparse_vertices, "dense_vertices": dense_vertices,
            "bindings_sparse": bindings_sparse, "bindings_dense": bindings_dense,
            "components": component_index, "np": np_,
        }

    def _run(self, case, *, sparse):
        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("optimize='none' must not search")), \
             patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("optimize='none' must not search storage paths")):
            calculation = calc_diagram_eval(
                case["diagram"],
                propagator_map=case["bindings_sparse" if sparse else "bindings_dense"].propagator_map,
                vertex_map=case["sparse_vertices" if sparse else "dense_vertices"],
                configuration="cfg-A", usedNe=1, usedNp=case["np"],
                optimize="none", execute=False,
            )
            executed = execute_contraction_path(calculation.expanded, calculation.path)
        value = assemble_correlator(calculation.expanded, executed.term_values)
        return calculation.expanded, executed, value

    def _dense_p2p_oracle(self, case, *, source, same_point):
        label_right = "point-0" if same_point else "point-1"
        network = lower_state_graph(
            [[0, 1], [2, 0]],
            ({"left": "p", "right": "p"}, {"left": "v", "right": "v"}),
            point_labels={(0, "field"): "point-0", (0, "bar"): label_right},
        )[0]
        current = case["current_operator"].terms[0]
        raw = case["reference_reader"].block(current.component, 0, 0, "p2p", usedNp=case["np"])
        current_spin = gamma(current.gamma)
        if source:
            g5 = gamma(8)
            current_spin = g5 @ current_spin.conj().T @ g5
            raw = raw.conj().transpose((2, 3, 0, 1))
        current_factor = np.einsum("AD,naod->ADnaod", current_spin, raw, optimize=False)
        normal = case["normal_operator"].terms[0]
        normal_raw = case["reference_reader"].block(normal.component, 0, 0, "v2v", usedNe=1)
        normal_factor = np.einsum("CB,PO->CBPO", gamma(normal.gamma), normal_raw, optimize=False)
        operands = []
        for request in network.requests:
            if request.kind == "propagator":
                sink_time = 0
                source_time = 0
                selection = {}
                if request.sector in {"psv", "vsp", "psp"}:
                    selection["usedNp"] = case["np"]
                if request.sector in {"psv", "vsp", "vsv"}:
                    selection["usedNe"] = 1
                operands.append(case["prop"].block(
                    sink_time, source_time, request.sector, **selection,
                ))
            elif request.vertex_index == 0:
                operands.append(current_factor)
            else:
                operands.append(normal_factor)
        return np.einsum(*einsum_arguments(network.equation, operands), optimize=False)

    def test_native_single_current_sparse_bridge_matches_dense_enumeration(self):
        rng = np.random.default_rng(644)
        indices = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1), (0, 1)), dtype=np.int64)
        values = _complex(rng, (len(indices), 3, 3)).astype(np.complex128)
        record = {"type": "sparse", "indices": indices, "values": values}
        case = self._case(record)
        with patch.object(SparsePointBlock, "to_dense",
                          side_effect=AssertionError("native sparse path must not densify COO")):
            sparse_expanded, sparse_execution, sparse_value = self._run(case, sparse=True)
        dense_expanded, _dense_execution, dense_value = self._run(case, sparse=False)

        bridges = sparse_expanded.problem.sparse_bridges
        self.assertEqual(len(bridges), 2)
        self.assertEqual({bridge.same_point for bridge in bridges}, {True, False})
        self.assertEqual(len(case["record_reader"].reads), 1)  # scene terms share one raw record read
        self.assertEqual(len(case["record_reader"].closed), 1)  # one borrowed physical record is released
        self.assertIs(case["record_reader"].closed[0][3], record)
        vertex_sectors = {
            request[3] for request in sparse_expanded.inputs.tensor_requests.values()
            if request[0] == "vertex"
        }
        self.assertEqual(vertex_sectors, {"v2v", "v2p", "p2v"})
        np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-12, atol=2e-10)
        direct_by_id = {value.term_id: value.value for value in sparse_execution.term_values}
        for bridge in bridges:
            expected = self._dense_p2p_oracle(case, source=False,
                                              same_point=bridge.same_point)
            np.testing.assert_allclose(
                direct_by_id[sparse_expanded.inputs.term_ids[bridge.scene]], expected,
                rtol=2e-12, atol=2e-10,
            )
            pattern = sparse_expanded.problem.terms[bridge.scene].subscripts
            sparse_pair = (ord("A"), ord("D"))
            self.assertFalse(any(
                tuple(labels[index:index + 2]) == sparse_pair
                for labels in pattern.inputs
                for index in range(len(labels) - 1)
            ))  # the P2P tensor is represented by the bridge record
        self.assertEqual(len(dense_expanded.problem.sparse_bridges), 0)

    def test_empty_coo_and_identity_records_match_dense_current(self):
        empty_record = {
            "type": "sparse",
            "indices": np.empty((0, 2), dtype=np.int64),
            "values": np.empty((0, 3, 3), dtype=np.complex128),
        }
        empty_case = self._case(empty_record, seed=812)
        _expanded, _execution, sparse_value = self._run(empty_case, sparse=True)
        _expanded, _execution, dense_value = self._run(empty_case, sparse=False)
        np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-12, atol=2e-10)

        identity_case = self._case({"type": "identity"}, seed=912)
        _expanded, _execution, sparse_value = self._run(identity_case, sparse=True)
        _expanded, _execution, dense_value = self._run(identity_case, sparse=False)
        np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-12, atol=2e-10)
        self.assertEqual(identity_case["record_reader"].reads[0][0],
                         identity_case["current_operator"].terms[0].component)

    def test_source_dagger_transposes_coo_conjugates_once_and_maps_raw_pool_sides(self):
        rng = np.random.default_rng(983)
        indices = np.asarray(((0, 1), (1, 0), (1, 1)), dtype=np.int64)
        values = _complex(rng, (len(indices), 3, 3)).astype(np.complex128)
        case = self._case({"type": "sparse", "indices": indices, "values": values},
                          source=True, seed=983)
        sparse_expanded, sparse_execution, sparse_value = self._run(case, sparse=True)
        _dense_expanded, _dense_execution, dense_value = self._run(case, sparse=False)
        np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-12, atol=2e-10)
        direct_by_id = {value.term_id: value.value for value in sparse_execution.term_values}
        for bridge in sparse_expanded.problem.sparse_bridges:
            expected = self._dense_p2p_oracle(case, source=True,
                                              same_point=bridge.same_point)
            np.testing.assert_allclose(
                direct_by_id[sparse_expanded.inputs.term_ids[bridge.scene]], expected,
                rtol=2e-12, atol=2e-10,
            )

        current = case["current_operator"]
        raw_pool_calls = []
        dense_reader = case["dense_reader"]
        original_pool = dense_reader.point_pool_id

        def observe_pool(component, momentum, time, side):
            raw_pool_calls.append(side)
            return original_pool(component, momentum, time, side)

        dense_reader.point_pool_id = observe_pool
        descriptor = case["sparse_vertices"]["current"].describe_factors(
            0, "p2p", usedNe=1, usedNp=case["np"],
        )[0]
        self.assertEqual(raw_pool_calls[:2], ["field", "bar"])
        self.assertEqual(descriptor.point_pool_ids,
                         {"bar": "ordered-full-sample", "field": "ordered-full-sample"})

    def test_native_double_current_all_52_scenes_match_dense_and_p2p_oracles(self):
        rng = np.random.default_rng(1501)
        n = 4
        indices = np.asarray([(i, j) for i in range(n) for j in range(n)], dtype=np.int64)
        values = _complex(rng, (n * n, 3, 3)).astype(np.complex128)
        record = {"type": "sparse", "indices": indices, "values": values}
        case = self._case(record, seed=1501, point_count=n)
        sparse_reader = case["sparse_reader"]
        reference_reader = case["reference_reader"]
        case["sparse_vertices"]["normal"] = Meson(
            sparse_reader, case["current_operator"], phase_convention="anchor",
        )
        case["dense_vertices"]["normal"] = Meson(
            reference_reader, case["current_operator"], phase_convention="anchor",
        )
        case["bindings_sparse"] = DiagramBindings(
            case["bindings_sparse"].propagator_map, case["sparse_vertices"],
            configuration="cfg-A", usedNe=1, usedNp=n,
        )
        case["bindings_dense"] = DiagramBindings(
            case["bindings_dense"].propagator_map, case["dense_vertices"],
            configuration="cfg-A", usedNe=1, usedNp=n,
        )

        sparse_expanded, sparse_execution, sparse_value = self._run(case, sparse=True)
        dense_expanded, _dense_execution, dense_value = self._run(case, sparse=False)
        self.assertEqual(len(sparse_expanded.terms), 52)
        self.assertEqual(len(dense_expanded.terms), 52)
        np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-11, atol=3e-9)

        bridge_counts = {}
        bridges_by_scene = {}
        for bridge in sparse_expanded.problem.sparse_bridges:
            bridge_counts[bridge.scene] = bridge_counts.get(bridge.scene, 0) + 1
            bridges_by_scene.setdefault(bridge.scene, []).append(bridge)
        double_scenes = [scene for scene, count in bridge_counts.items() if count == 2]
        self.assertEqual(len(double_scenes), 15)
        self.assertEqual(len(case["record_reader"].reads), 1)
        self.assertEqual(len(case["record_reader"].closed), 1)
        direct = {value.term_id: value.value for value in sparse_execution.term_values}
        prop_block = case["prop"].block(0, 0, "psp", usedNp=n)
        raw_vertex = reference_reader.block(
            case["current_operator"].terms[0].component, 0, 0, "p2p", usedNp=n,
        )
        vertex_factor = np.einsum(
            "ab,...->ab...", gamma(case["current_operator"].terms[0].gamma),
            raw_vertex, optimize=False,
        )
        for scene in double_scenes:
            with self.subTest(scene=scene):
                pattern = sparse_expanded.problem.terms[scene].subscripts
                prop_labels = pattern.inputs
                self.assertEqual(pattern.output, ())
                vertex_labels = []
                for bridge in bridges_by_scene[scene]:
                    if not bridge.transpose_bridge:
                        field_slot, field_axes = 0, bridge.left_axes
                        bar_slot, bar_axes = 1, bridge.right_axes
                    else:
                        field_slot, field_axes = 1, bridge.right_axes
                        bar_slot, bar_axes = 0, bridge.left_axes
                    field, bar = prop_labels[field_slot], prop_labels[bar_slot]
                    vertex_labels.append((
                        field[field_axes[0]], bar[bar_axes[0]],
                        field[field_axes[1]], field[field_axes[2]],
                        bar[bar_axes[1]], bar[bar_axes[2]],
                    ))
                dense_pattern = AxisPattern((*prop_labels, *vertex_labels), ())
                expected = np.einsum(*einsum_arguments(
                    dense_pattern, (prop_block, prop_block, vertex_factor, vertex_factor),
                ), optimize=False)
                term_id = sparse_expanded.inputs.term_ids[scene]
                np.testing.assert_allclose(direct[term_id], expected,
                                           rtol=2e-11, atol=3e-9)

    def test_source_profile_uses_original_pre_dagger_endpoint_times(self):
        class RecordingProfile:
            def __init__(self):
                self.times = None

            def apply_block(self, value, *, times, cfg, array_module):
                self.times = tuple(times)
                return value

        profile = RecordingProfile()
        component = ElementalComponent(7, (1, 3))
        operator = Operator.from_terms(
            "point-split-profile",
            (OperatorTerm(0, component, 0, 1.0, anchor_time_offset=2, profile=profile),),
        )
        raw = np.ones((1, 1, 7, 2, 2), dtype=np.complex128)
        reader = ArrayElemental({"v2v": raw}, (component,), phase_convention="anchor")
        meson = Meson(reader, operator, source=True, phase_convention="anchor")
        descriptor = meson.describe_factors(0, "v2v")[0]
        meson.materialize_unweighted_factor(descriptor, meson.read_factor_block(descriptor))
        self.assertEqual(profile.times, (3, 5))
        self.assertEqual(descriptor.field_time, 3)  # daggered endpoint differs from raw source time

    def test_inconsistent_coo_metadata_fails_and_releases_the_borrowed_record(self):
        rng = np.random.default_rng(1081)
        indices = np.asarray(((0, 0), (1, 1)), dtype=np.int64)
        values = _complex(rng, (2, 3, 3)).astype(np.complex128)
        case = self._case({"type": "sparse", "indices": indices, "values": values})
        case["record_reader"].invalid_override = {
            "type": "sparse", "indices": indices,
            "values": values.real.astype(np.float64),
        }
        case["record_reader"].close_error = RuntimeError("close transport failed")
        with self.assertRaisesRegex(ValueError, "dtypes differ") as caught:
            calculation = calc_diagram_eval(
                case["diagram"], propagator_map=case["bindings_sparse"].propagator_map,
                vertex_map=case["sparse_vertices"], configuration="cfg-A",
                usedNe=1, usedNp=case["np"], optimize="none", execute=False,
            )
            execute_contraction_path(calculation.expanded, calculation.path)
        self.assertEqual(len(case["record_reader"].closed), 1)
        self.assertIs(case["record_reader"].closed[0][3],
                      case["record_reader"].invalid_override)
        self.assertTrue(any("cleanup also failed" in note
                            for note in getattr(caught.exception, "__notes__", ())))

    def test_sparse_reader_rejects_undeclared_mode_and_point_subselections(self):
        rng = np.random.default_rng(1307)
        indices = np.asarray(((0, 0),), dtype=np.int64)
        values = _complex(rng, (1, 3, 3)).astype(np.complex128)
        case = self._case({"type": "sparse", "indices": indices, "values": values})
        component = case["current_operator"].terms[0].component
        with self.assertRaisesRegex(NotImplementedError, "usedNp prefix"):
            case["sparse_reader"].describe_block(
                component, 0, 0, "p2p", usedNe=1,
            )
        with self.assertRaisesRegex(NotImplementedError, "usedNp prefix"):
            case["sparse_reader"].block(
                component, 0, 0, "p2p", point_indices=(0,),
            )
        self.assertEqual(case["record_reader"].reads, [])


if __name__ == "__main__":
    unittest.main()
