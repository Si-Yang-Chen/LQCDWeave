"""Direct numerical acceptance for the dynamic ContractionSession adapters."""

from __future__ import annotations

from itertools import permutations, product
import unittest

import numpy as np

from lqcdweave.contraction.contraction_session import ContractionSession


class _Diagram:
    def __init__(self, groups=(), *, subscripts=(), propagator_types=None):
        self.operands = tuple(groups)
        self.subscripts = tuple(subscripts)
        if propagator_types is not None:
            self.propagator_types = tuple(tuple(group) for group in propagator_types)


def _group(propagators=(), vertices=()):
    return (tuple(propagators), tuple(vertices))


def _diagram_group(subscripts, propagators=(), vertices=(), *, sector="VSV"):
    prop_refs = tuple(propagators)
    return _Diagram(
        (_group(prop_refs, vertices),),
        subscripts=(subscripts,),
        propagator_types=(tuple(sector for _ in prop_refs),),
    )


class _RecordingVertex:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.calls = []

    def get(self, time):
        self.calls.append((time,))
        return self.value


class _SectorPropagator:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def _read(self, method, *times):
        self.calls.append((method, *times))
        value = self.values[method]
        if method == "get":
            return value
        return value, f"effective:{method}"

    def get(self, *times):
        return self._read("get", *times)

    def get_v2p(self, *times):
        return self._read("get_v2p", *times)

    def get_p2v(self, *times):
        return self._read("get_p2v", *times)

    def get_p2p(self, *times):
        return self._read("get_p2p", *times)


class _ScratchPropagator:
    """Return one mutable array whose contents depend on both time arguments."""

    def __init__(self, shape=(2, 2)):
        self.scratch = np.empty(shape, dtype=np.complex128)
        self.calls = []

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        grid = np.arange(self.scratch.size, dtype=np.float64).reshape(self.scratch.shape)
        self.scratch[...] = (
            (source_time + 1) * (grid + 1)
            + 1j * (sink_time + 2) * (grid[::-1] + 0.5)
        )
        return self.scratch


class _BaryonDiagram:
    def __init__(self, source_slots=(0, 1, 2), *, source=0, sink=1, propagator_ids=(1, 2, 3),
                 propagator_types=None, adjacency=None):
        if adjacency is None:
            block = [[0, 0, 0] for _ in range(3)]
            for sink_slot, source_slot in enumerate(source_slots):
                block[source_slot][sink_slot] = propagator_ids[sink_slot]
            adjacency = [[0, 0], [0, 0]]
            adjacency[source][sink] = block
        self.adjacency_matrix = adjacency
        if propagator_types is not None:
            self.propagator_types = propagator_types


class _BaryonVertex:
    def __init__(self, spin, elemental):
        self.spin = np.asarray(spin)
        self.elemental = np.asarray(elemental)
        self.calls = []

    def get_components(self, time):
        self.calls.append((time,))
        return self.spin, self.elemental


class _BaryonPropagator:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.calls = []

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        return self.value


class _ScratchBaryonPropagator:
    """Reuses one propagator buffer across forward and reverse endpoint requests."""

    def __init__(self, identity):
        self.identity = identity
        self.scratch = np.empty((3, 3, 1, 1), dtype=np.complex128)
        self.calls = []

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        sink = np.arange(3, dtype=np.float64)[:, None]
        source = np.arange(3, dtype=np.float64)[None, :]
        self.scratch[:, :, 0, 0] = (
            self.identity + 0.2 * source_time + 0.07 * sink_time
            + (sink + 1) * (source + 2)
            + 1j * (0.11 * self.identity + 0.03 * source_time - 0.05 * sink_time
                   + 0.5 * sink - 0.2 * source)
        )
        return self.scratch


def _epsilon3():
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for order in permutations(range(3)):
        inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
        epsilon[order] = -1 if inversions % 2 else 1
    return epsilon


def _baryon_epsilon_reference(source, sink, propagators, source_slots, coefficient):
    """Small explicit index sum; does not use the execution kernel/equation builder."""
    total = 0j
    for source_spin_indices in product(range(3), repeat=3):
        source_factor = (
            source.spin[source_spin_indices]
            * source.elemental[0, 0, 0]
        )
        for sink_spin_indices in product(range(3), repeat=3):
            value = coefficient * source_factor * sink.spin[sink_spin_indices] * sink.elemental[0, 0, 0]
            for line, source_slot in enumerate(source_slots):
                value *= propagators[line][
                    sink_spin_indices[line], source_spin_indices[source_slot], 0, 0,
                ]
            total += value
    return total


def _baryon_vertices():
    epsilon = _epsilon3()
    source = _BaryonVertex(
        epsilon * (0.75 - 0.25j),
        np.full((1, 1, 1), 0.8 + 0.1j, dtype=np.complex128),
    )
    sink = _BaryonVertex(
        epsilon * (-0.4 + 0.6j),
        np.full((1, 1, 1), 1.1 - 0.2j, dtype=np.complex128),
    )
    return source, sink


class DynamicContractionSessionDiagramTests(unittest.TestCase):
    def setUp(self):
        # Frozen baseline: this scalar adapter reads both global backend selectors.
        self.session = ContractionSession(backend=np)

    def test_all_four_propagator_getters_match_independent_numpy_reference(self):
        methods = {
            "VSV": "get",
            "VSP": "get_v2p",
            "PSV": "get_p2v",
            "PSP": "get_p2p",
        }
        for sector, method in methods.items():
            with self.subTest(sector=sector):
                propagator_value = np.array(
                    [[1 + 2j, 2 - 1j, -0.5j], [3 - 0.25j, -2 + 1j, 4 + 2j]],
                    dtype=np.complex128,
                )
                vertex_value = np.array([2 - 1j, -0.5 + 0.25j, 3 + 0.75j])
                propagator = _SectorPropagator({method: propagator_value})
                vertex = _RecordingVertex(vertex_value)
                diagram = _diagram_group(
                    "ab,b->a", ((0, 0, 1),), (0,), sector=sector,
                )

                result = self.session.diagrams(
                    [diagram], [4, 9], [vertex], [propagator],
                )

                expected = np.einsum("ab,b->a", propagator_value, vertex_value, optimize=False)
                self.assertEqual(result.shape, (1, 2))
                self.assertEqual(result.dtype, np.dtype(np.complex128))
                np.testing.assert_allclose(result[0], expected, rtol=0, atol=0)
                self.assertEqual(propagator.calls, [(method, 4, 9)])
                self.assertEqual(vertex.calls, [(4,)])

    def test_implicit_output_preserves_non_scalar_axes(self):
        left = np.arange(6, dtype=np.float64).reshape(2, 3) + 1
        right = (np.arange(12, dtype=np.float64).reshape(3, 4) - 2) * 1j
        prop_left = _SectorPropagator({"get": left})
        prop_right = _SectorPropagator({"get": right})
        diagram = _diagram_group(
            "ab,bc", ((0, 0, 1), (1, 0, 1)), (), sector="VSV",
        )

        result = self.session.diagrams(
            [diagram], [2, 6], [], [prop_left, prop_right],
        )

        expected = np.einsum("ab,bc", left, right, optimize=False)
        self.assertEqual(result.shape, (1, 2, 4))
        self.assertEqual(result.dtype, expected.dtype)
        np.testing.assert_allclose(result[0], expected, rtol=0, atol=0)

    def test_empty_diagrams_and_empty_group_keep_neutral_shapes(self):
        empty = self.session.diagrams([], [], [], [])
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.dtype(np.float64))

        empty_group = self.session.diagrams([_Diagram()], [], [], [])
        np.testing.assert_array_equal(empty_group, np.array([1.0]))

    def test_repeated_and_distinct_requests_snapshot_a_mutable_getter_buffer(self):
        prop = _ScratchPropagator()
        vertex_value = np.array(
            [[1 + 0.5j, -0.25 + 1j], [2 - 1j, 0.75 + 0.25j]],
            dtype=np.complex128,
        )
        vertex = _RecordingVertex(vertex_value)
        repeated = _diagram_group("ab,ab->", ((0, 0, 1),), (0,))
        repeated.operands = repeated.operands + (_group(((0, 0, 1),), (0,)),)
        repeated.subscripts = repeated.subscripts + ("ab,ab->",)
        repeated.propagator_types = repeated.propagator_types + (("VSV",),)
        reversed_times = _diagram_group("ab,ab->", ((0, 1, 0),), (0,))

        result = self.session.diagrams(
            [repeated, reversed_times], [2, 7], [vertex], [prop],
        )

        def matrix_for(source_time, sink_time):
            grid = np.arange(4, dtype=np.float64).reshape(2, 2)
            return (
                (source_time + 1) * (grid + 1)
                + 1j * (sink_time + 2) * (grid[::-1] + 0.5)
            )

        first = np.einsum("ab,ab->", matrix_for(2, 7), vertex_value, optimize=False)
        second = np.einsum("ab,ab->", matrix_for(7, 2), vertex_value, optimize=False)
        np.testing.assert_allclose(result, np.array([first * first, second]), rtol=0, atol=0)
        self.assertEqual(prop.calls, [(2, 7), (7, 2)])
        self.assertEqual(vertex.calls, [(2,)])

    def test_neutral_multiply_preserves_dtype_and_ieee_values(self):
        values_by_dtype = (
            (np.float32, np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 2.5], dtype=np.float32)),
            (np.float64, np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 2.5], dtype=np.float64)),
            (np.complex64, np.array(
                [0 + 0j, complex(-0.0, -0.0), complex(np.nan, 1),
                 complex(2, np.inf), complex(-np.inf, -0.0), 2.5 - 3j],
                dtype=np.complex64,
            )),
            (np.complex128, np.array(
                [0 + 0j, complex(-0.0, -0.0), complex(np.nan, 1),
                 complex(2, np.inf), complex(-np.inf, -0.0), 2.5 - 3j],
                dtype=np.complex128,
            )),
        )
        for dtype, values in values_by_dtype:
            with self.subTest(dtype=np.dtype(dtype).name):
                prop = _SectorPropagator({"get": values})
                diagram = _diagram_group("i->i", ((0, 0, 1),), ())
                with np.errstate(invalid="ignore", over="ignore"):
                    result = self.session.diagrams([diagram], [0, 1], [], [prop])
                    # The execution term has coefficient 1, then diagram assembly multiplies by 1.0.
                    contracted = np.einsum("i->i", values, optimize=False)
                    term_value = np.multiply(contracted, 1)
                    expected = np.multiply(1.0, term_value)
                actual = result[0]
                self.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_equal(actual, expected)
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
                np.testing.assert_array_equal(np.signbit(actual.real), np.signbit(expected.real))
                if np.issubdtype(np.dtype(dtype), np.complexfloating):
                    np.testing.assert_array_equal(np.signbit(actual.imag), np.signbit(expected.imag))

    def test_missing_sector_method_is_wrapped(self):
        class VsvOnly:
            def __init__(self):
                self.calls = []

            def get(self, *times):
                self.calls.append(times)
                return np.ones((2, 2), dtype=np.float64)

        prop = VsvOnly()
        vertex = _RecordingVertex(np.ones(2, dtype=np.float64))
        vsp = _diagram_group("ab,b->a", ((0, 0, 1),), (0,), sector="VSP")
        with self.assertRaisesRegex(RuntimeError, r"Error getting propagator\[0\] type VSP"):
            self.session.diagrams([vsp], [0, 1], [vertex], [prop])
        self.assertEqual(prop.calls, [])

        # Frozen baseline-only seam: remove this global mismatch assertion when Session moves to W explicit backend.

    def test_dynamic_plan_cache_reuses_structure_and_rebinds_current_values(self):
        diagram = _diagram_group("ab,b->a", ((0, 0, 1),), (0,))
        first_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        second_matrix = np.array([[5.0, -1.0], [2.5, 7.0]], dtype=np.float64)
        first_vertex_value = np.array([2.0, -0.5], dtype=np.float64)
        second_vertex_value = np.array([-3.0, 4.0], dtype=np.float64)

        first = self.session.diagrams(
            [diagram], [0, 1], [_RecordingVertex(first_vertex_value)],
            [_SectorPropagator({"get": first_matrix})],
        )
        second = self.session.diagrams(
            [diagram], [0, 1], [_RecordingVertex(second_vertex_value)],
            [_SectorPropagator({"get": second_matrix})],
        )

        expected_first = np.einsum("ab,b->a", first_matrix, first_vertex_value, optimize=False)
        expected_second = np.einsum("ab,b->a", second_matrix, second_vertex_value, optimize=False)
        np.testing.assert_array_equal(first[0], expected_first)
        np.testing.assert_array_equal(second[0], expected_second)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(self.session.plan_cache.misses, 1)
        self.assertEqual(self.session.plan_cache.hits, 1)
        self.session.clear()
        self.assertEqual(self.session.plan_cache.hits, 0)
        self.assertEqual(self.session.plan_cache.misses, 0)

    def test_shape_mismatch_and_unsupported_session_backend_raise(self):
        prop = _SectorPropagator({"get": np.ones((2, 3), dtype=np.float64)})
        vertex = _RecordingVertex(np.ones(4, dtype=np.float64))
        diagram = _diagram_group("ab,b->a", ((0, 0, 1),), (0,))
        with self.assertRaises((TypeError, ValueError)):
            self.session.diagrams([diagram], [0, 1], [vertex], [prop])
        with self.assertRaisesRegex(ValueError, "session backend must be NumPy or CuPy"):
            ContractionSession(backend="torch")


class DynamicContractionSessionBaryonTests(unittest.TestCase):
    def setUp(self):
        self.session = ContractionSession(backend=np)

    def test_all_source_slot_permutations_and_complex_coefficients_match_epsilon_sum(self):
        source, sink = _baryon_vertices()
        rng = np.random.default_rng(20261002)
        propagators = {
            index: _BaryonPropagator(
                rng.standard_normal((3, 3, 1, 1))
                + 1j * rng.standard_normal((3, 3, 1, 1))
            )
            for index in (1, 2, 3)
        }
        permutations_to_check = tuple(permutations(range(3)))
        diagrams = [_BaryonDiagram(slots) for slots in permutations_to_check]
        coefficients = tuple(
            complex(0.25 + index * 0.1, -0.5 + index * 0.07)
            for index in range(len(diagrams))
        )

        result = self.session.baryon_diagrams(
            diagrams, [3, 8], [source, sink], propagators,
            coefficients=coefficients,
        )

        for index, (slots, coefficient) in enumerate(zip(permutations_to_check, coefficients, strict=True)):
            expected = _baryon_epsilon_reference(
                source, sink,
                tuple(propagators[prop_id].value for prop_id in (1, 2, 3)),
                slots, coefficient,
            )
            np.testing.assert_allclose(result[index], expected, rtol=2e-13, atol=2e-12)
        self.assertEqual(result.shape, (6,))
        self.assertEqual(result.dtype, np.dtype(np.complex128))
        self.assertEqual(source.calls, [(3,)])
        self.assertEqual(sink.calls, [(8,)])
        self.assertEqual([propagators[key].calls for key in (1, 2, 3)], [[(3, 8)]] * 3)

    def test_empty_input_returns_empty_complex_result_without_getters(self):
        result = self.session.baryon_diagrams([], [], [], {})
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(np.complex128))

    def test_bad_later_graph_is_rejected_before_any_numeric_getter(self):
        source, sink = _baryon_vertices()
        props = {key: _BaryonPropagator(np.ones((3, 3, 1, 1), dtype=np.complex128))
                 for key in (1, 2, 3)}
        valid = _BaryonDiagram()
        malformed_adjacency = [[0, 0], [0, 0]]
        invalid_cases = (
            ("topology", _BaryonDiagram(adjacency=malformed_adjacency), (1 + 0j, 1 + 0j)),
            ("missing-id", _BaryonDiagram(propagator_ids=(1, 2, 4)), (1 + 0j, 1 + 0j)),
            ("coefficient", _BaryonDiagram(), (1 + 0j, object())),
        )
        for label, invalid, coefficients in invalid_cases:
            with self.subTest(failure=label):
                source.calls.clear()
                sink.calls.clear()
                for prop in props.values():
                    prop.calls.clear()
                expected_exception = KeyError if label == "missing-id" else (TypeError if label == "coefficient" else ValueError)
                with self.assertRaises(expected_exception):
                    self.session.baryon_diagrams(
                        [valid, invalid], [2, 5], [source, sink], props,
                        coefficients=coefficients,
                    )
                self.assertEqual(source.calls, [])
                self.assertEqual(sink.calls, [])
                self.assertTrue(all(prop.calls == [] for prop in props.values()))

    def test_unsupported_metadata_and_missing_getter_fail_during_preflight(self):
        source, sink = _baryon_vertices()
        props = {key: _BaryonPropagator(np.ones((3, 3, 1, 1), dtype=np.complex128))
                 for key in (1, 2, 3)}
        unsupported = _BaryonDiagram(propagator_types=("VSP", "VSV", "VSV"))
        with self.assertRaises(NotImplementedError):
            self.session.baryon_diagrams([unsupported], [1, 2], [source, sink], props)
        self.assertEqual(source.calls, [])
        self.assertTrue(all(prop.calls == [] for prop in props.values()))

        class MissingVertexGetter:
            pass

        with self.assertRaisesRegex(TypeError, "must provide get_components"):
            self.session.baryon_diagrams(
                [_BaryonDiagram()], [1, 2], [MissingVertexGetter(), sink], props,
            )
        self.assertTrue(all(prop.calls == [] for prop in props.values()))

    def test_distinct_endpoint_requests_snapshot_reused_propagator_buffers(self):
        source, sink = _baryon_vertices()
        props = {key: _ScratchBaryonPropagator(key) for key in (1, 2, 3)}
        forward = _BaryonDiagram((2, 0, 1), source=0, sink=1)
        reverse = _BaryonDiagram((1, 2, 0), source=1, sink=0)
        diagrams = (forward, reverse)

        result = self.session.baryon_diagrams(
            diagrams, [4, 9], [source, sink], props,
            coefficients=(0.5 + 0.25j, -0.75 + 0.5j),
        )

        snapshots = {
            key: {
                (4, 9): self._scratch_value(key, 4, 9),
                (9, 4): self._scratch_value(key, 9, 4),
            }
            for key in (1, 2, 3)
        }
        expected = (
            _baryon_epsilon_reference(
                source, sink, tuple(snapshots[key][(4, 9)] for key in (1, 2, 3)),
                (2, 0, 1), 0.5 + 0.25j,
            ),
            _baryon_epsilon_reference(
                sink, source, tuple(snapshots[key][(9, 4)] for key in (1, 2, 3)),
                (1, 2, 0), -0.75 + 0.5j,
            ),
        )
        np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-12)
        self.assertEqual([props[key].calls for key in (1, 2, 3)], [[(4, 9), (9, 4)]] * 3)

    @staticmethod
    def _scratch_value(identity, source_time, sink_time):
        scratch = np.empty((3, 3, 1, 1), dtype=np.complex128)
        sink_axis = np.arange(3, dtype=np.float64)[:, None]
        source_axis = np.arange(3, dtype=np.float64)[None, :]
        scratch[:, :, 0, 0] = (
            identity + 0.2 * source_time + 0.07 * sink_time
            + (sink_axis + 1) * (source_axis + 2)
            + 1j * (0.11 * identity + 0.03 * source_time - 0.05 * sink_time
                   + 0.5 * sink_axis - 0.2 * source_axis)
        )
        return scratch

    def test_shape_and_backend_errors_are_visible(self):
        source, sink = _baryon_vertices()
        bad_shape = {1: _BaryonPropagator(np.ones((2, 3, 1, 1), dtype=np.complex128)),
                     2: _BaryonPropagator(np.ones((3, 3, 1, 1), dtype=np.complex128)),
                     3: _BaryonPropagator(np.ones((3, 3, 1, 1), dtype=np.complex128))}
        with self.assertRaises((TypeError, ValueError)):
            self.session.baryon_diagrams([_BaryonDiagram()], [0, 1], [source, sink], bad_shape)
        with self.assertRaisesRegex(ValueError, "session backend must be NumPy or CuPy"):
            ContractionSession(backend="torch")



class DynamicSessionPersistedBoundaryTests(unittest.TestCase):
    def test_persisted_session_rejects_each_dynamic_method(self):
        from lqcdweave.contraction.planning.api import compile_plan
        from lqcdweave.contracts.correlator.api import (
            ContractionTerm,
            ExecutionBudget,
            PlanBatch,
            PlanBundle,
            TensorSpec,
            WorkloadSpec,
        )
        from lqcdweave.storage.contraction_plans.api import validate_bundle

        workload = WorkloadSpec(
            specs={"input": TensorSpec((2,), np.float64)},
            terms=(ContractionTerm(("input",), "i->i"),),
            input_sources={"input": {"family": "dynamic-session-test"}},
        )
        plan = compile_plan(workload.specs, workload.terms)
        bundle = PlanBundle(
            workload,
            (PlanBatch(plan, (0,)),),
            ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0),
        )
        session = ContractionSession.from_plan(
            bundle, backend=np, validate_bundle=validate_bundle,
        )
        dynamic_calls = (
            lambda: session.tensors({}, ()),
            lambda: session.baryon_terms({}, ()),
            lambda: session.baryon_diagrams([], [], [], {}),
            lambda: session.diagrams([], [], [], []),
        )
        for call in dynamic_calls:
            with self.subTest(method=call):
                with self.assertRaisesRegex(ValueError, "persisted sessions use execute_cfg"):
                    call()


if __name__ == "__main__":
    unittest.main()





