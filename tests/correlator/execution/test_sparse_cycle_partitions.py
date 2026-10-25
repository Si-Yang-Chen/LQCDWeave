"""Module tests split from test_sparse_cycle_partitions.py; shared numerical fixtures stay unchanged."""

from test_sparse_cycle_partitions import (
    SparseBridgeSpec,
    _SparseCyclePartitionFixture,
    _bridges,
    _complex,
    _dense_oracle,
    _record_dense,
    _scene_labels,
    bridge_point_equalities_covered,
    contract_p2p_bridge,
    contract_p2p_cycle,
    contract_sparse_p2p_join,
    cycle_point_equalities_covered,
    cycle_point_equivalence,
    labelled_set_partitions,
    lower_state_graph,
    np,
)
from lqcdweave.contracts.correlator.api import AxisPattern


class SparseCyclePartitionTests(_SparseCyclePartitionFixture):
    def test_existing_cycle_and_streamed_join_match_all_fifteen_dense_oracles(self):
        n = 3
        partitions = labelled_set_partitions(4)
        self.assertEqual(len(partitions), 15)
        unsupported = {(0, 1, 1, 0), (0, 1, 1, 2), (0, 1, 2, 0)}
        seen_join = set()
        for index, partition in enumerate(partitions):
            with self.subTest(partition=partition):
                rng = np.random.default_rng(20260928 + index)
                left = _complex(rng, (4, 4, n, 3, n, 3))
                right = _complex(rng, (4, 4, n, 3, n, 3))
                gamma0, gamma1 = _complex(rng, (4, 4)), _complex(rng, (4, 4))
                raw0, raw1 = _complex(rng, (n, n, 3, 3)), _complex(rng, (n, n, 3, 3))
                indices = np.asarray([(i, j) for i in range(n) for j in range(n)], dtype=np.int64)
                record0 = {"type": "sparse", "indices": indices, "values": raw0.reshape(-1, 3, 3)}
                record1 = {"type": "sparse", "indices": indices, "values": raw1.reshape(-1, 3, 3)}
                expected, network = _dense_oracle(
                    partition, left, right, gamma0, gamma1, record0, record1, n,
                )
                prop_requests = [item for item in network.requests if item.kind == "propagator"]
                equation = AxisPattern(
                    (prop_requests[0].subscripts, prop_requests[1].subscripts), (),
                )
                first, second = _bridges(partition, n)
                covered = cycle_point_equalities_covered(equation, first, second)
                self.assertEqual(covered, partition not in unsupported)
                if not covered:
                    seen_join.add(partition)
                actual = contract_p2p_cycle(
                    left, right, ((gamma0, 1, record0),), ((gamma1, 1, record1),),
                    first_bridge=first, second_bridge=second, equation=equation,
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        self.assertEqual(seen_join, unsupported)

    def test_join_handles_duplicate_empty_identity_and_transposed_records(self):
        n = 3
        partition = (0, 1, 2, 0)  # A=D is a cross-current equality.
        rng = np.random.default_rng(773)
        left = _complex(rng, (4, 4, n, 3, n, 3))
        right = _complex(rng, (4, 4, n, 3, n, 3))
        gamma0, gamma1 = _complex(rng, (4, 4)), _complex(rng, (4, 4))
        indices0 = np.asarray(((0, 1), (0, 1), (1, 0), (2, 2)), dtype=np.int64)
        values0 = _complex(rng, (len(indices0), 3, 3))
        indices1 = np.asarray(((0, 1), (2, 0), (2, 0), (1, 1)), dtype=np.int64)
        values1 = _complex(rng, (len(indices1), 3, 3))
        cases = (
            ({"type": "sparse", "indices": indices0, "values": values0},
             {"type": "sparse", "indices": indices1, "values": values1}, True, False),
            ({"type": "sparse", "indices": np.empty((0, 2), np.int64),
              "values": np.empty((0, 3, 3), np.complex128)},
             {"type": "sparse", "indices": indices1, "values": values1}, False, False),
            ({"type": "identity"},
             {"type": "sparse", "indices": indices1, "values": values1}, False, False),
        )
        for record0, record1, transpose0, transpose1 in cases:
            with self.subTest(record0=record0["type"], transpose0=transpose0):
                first, second = _bridges(
                    partition, n, first_transpose=transpose0, second_transpose=transpose1,
                )
                network = lower_state_graph(
                    [[0, 1], [2, 0]],
                    ({"left": "p", "right": "p"}, {"left": "p", "right": "p"}),
                    point_labels=_scene_labels(partition),
                )[0]
                props = [item for item in network.requests if item.kind == "propagator"]
                equation = AxisPattern((props[0].subscripts, props[1].subscripts), ())
                classes = cycle_point_equivalence(equation, first, second)
                actual = contract_sparse_p2p_join(
                    left, right, ((gamma0, 1, record0),), ((gamma1, 1, record1),),
                    first_bridge=first, second_bridge=second, equation=equation,
                    point_equivalence=classes, output_labels="", join_index_block_size=1,
                )
                expected, _ = _dense_oracle(
                    partition, left, right, gamma0, gamma1, record0, record1, n,
                    first_transpose=transpose0, second_transpose=transpose1,
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)

    def test_join_retains_an_explicit_point_output_axis(self):
        n = 3
        partition = (0, 1, 2, 0)  # A=D is joined; external point n remains output.
        rng = np.random.default_rng(1173)
        left = _complex(rng, (n, 4, 4, n, 3, n, 3))
        right = _complex(rng, (4, 4, n, 3, n, 3))
        gamma0, gamma1 = _complex(rng, (4, 4)), _complex(rng, (4, 4))
        indices = np.asarray([(i, j) for i in range(n) for j in range(n)], dtype=np.int64)
        values0, values1 = _complex(rng, (n * n, 3, 3)), _complex(rng, (n * n, 3, 3))
        record0 = {"type": "sparse", "indices": indices, "values": values0}
        record1 = {"type": "sparse", "indices": indices, "values": values1}
        first = SparseBridgeSpec(
            "p0", "p1", "r0", n, n*n,
            left_axes=(2, 5, 6), right_axes=(0, 2, 3),
        )
        second = SparseBridgeSpec(
            "p1", "p0", "r1", n, n*n,
            left_axes=(1, 4, 5), right_axes=(1, 3, 4),
        )
        equation = "nBAobna,DCpdnc->n"
        classes = cycle_point_equivalence(equation, first, second)
        actual = contract_sparse_p2p_join(
            left, right, ((gamma0, 1, record0),), ((gamma1, 1, record1),),
            first_bridge=first, second_bridge=second, equation=equation,
            point_equivalence=classes, output_labels="n", join_index_block_size=2,
        )
        raw0 = values0.reshape(n, n, 3, 3).transpose(0, 2, 1, 3)
        raw1 = values1.reshape(n, n, 3, 3).transpose(0, 2, 1, 3)
        dense0 = np.einsum("AD,naod->ADnaod", gamma0, raw0, optimize=False)
        dense1 = np.einsum("CB,naob->CBnaob", gamma1, raw1, optimize=False)
        expected = np.einsum(
            "nBAobna,DCpdnc,ADnapd,CBncob->n",
            left, right, dense0, dense1, optimize=False,
        )
        self.assertEqual(actual.shape, (n,))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)

    def test_single_bridge_streaming_join_enforces_external_point_labels(self):
        n = 3
        rng = np.random.default_rng(1919)
        left = _complex(rng, (4, 4, n, 3, n, 3))  # sink P and source P endpoints
        right = _complex(rng, (4, 4, n, 3, 1))  # sink P, source V endpoint
        spin = _complex(rng, (4, 4))
        indices = np.asarray([(i, j) for i in range(n) for j in range(n)], dtype=np.int64)
        values = _complex(rng, (n * n, 3, 3))
        record = {"type": "sparse", "indices": indices, "values": values}
        bridge = (1, 4, 5)
        opposite = (0, 2, 3)
        errors = []
        for part in labelled_set_partitions(3):
            with self.subTest(partition=part):
                # Legs are A=current-field, C=current-bar, X=the external P leg.
                symbols = "xyz"
                point_a, point_c, point_x = (symbols[part[index]] for index in range(3))
                left_labels = f"BA{point_x}b{point_a}a"
                right_labels = f"DC{point_c}dP"
                equation = f"{left_labels},{right_labels}->{point_x}"
                same_point = part[0] == part[1]
                self.assertEqual(
                    bridge_point_equalities_covered(
                        equation, bridge, opposite, same_point=same_point,
                    ),
                    part[2] not in {part[0], part[1]},
                )
                current = np.einsum(
                    "AD,naod->ADnaod", spin,
                    _record_dense(record, n), optimize=False,
                )
                expected = np.einsum(
                    f"{left_labels},{right_labels},AD{point_a}a{point_c}d->{point_x}",
                    left, right, current, optimize=False,
                )
                actual = contract_p2p_bridge(
                    left, right, ((spin, 1, record),), used_np=n,
                    left_axes=bridge, right_axes=opposite,
                    same_point=same_point, equation=equation,
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
                errors.append(np.max(np.abs(np.asarray(actual) - expected)))
        self.assertLess(max(errors), 1e-10)
