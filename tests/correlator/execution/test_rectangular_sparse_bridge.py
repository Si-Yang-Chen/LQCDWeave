"""Module tests split from test_rectangular_sparse_bridge.py; shared numerical fixtures stay unchanged."""

from __future__ import annotations
from test_rectangular_sparse_bridge import (
    _complex,
    _dense_record,
    _record,
    _rectangular_cycle_case,
    contract_p2p_bridge,
    contract_p2p_cycle,
    contract_sparse_p2p_bridge_join,
    cycle_point_equalities_covered,
    np,
    unittest,
)


class RectangularSparseBridgeTests(unittest.TestCase):
    def test_one_bridge_rectangles_and_transposed_record_match_dense_oracle(self):
        rng = np.random.default_rng(61002)
        left_count, right_count = 3, 2
        gamma = _complex(rng, (4, 4))
        record = _record(rng, left_count, right_count)
        dense = _dense_record(record, left_count, right_count)
        coefficient = 0.6 + 0.4j

        left = _complex(rng, (4, left_count, 3))
        right = _complex(rng, (4, right_count, 3))
        expected = coefficient * np.einsum(
            "aic,aj,ikcd,jkd->", left, gamma, dense, right, optimize=False,
        )
        actual = contract_p2p_bridge(
            left, right, ((gamma, coefficient, record),),
            used_np=left_count, used_np_right=right_count,
            equation="aic,jkd->",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)

        left_t = _complex(rng, (4, right_count, 3))
        right_t = _complex(rng, (4, left_count, 3))
        dense_t = dense.transpose(1, 0, 3, 2)
        expected_t = coefficient * np.einsum(
            "aic,aj,ikcd,jkd->", left_t, gamma.T, dense_t, right_t,
            optimize=False,
        )
        actual_t = contract_p2p_bridge(
            left_t, right_t, ((gamma, coefficient, record),),
            used_np=left_count, used_np_right=right_count,
            transpose_bridge=True, equation="aic,jkd->",
        )
        np.testing.assert_allclose(actual_t, expected_t, rtol=1e-12, atol=1e-10)

    def test_rectangular_open_point_join_matches_independent_dense_tensor(self):
        rng = np.random.default_rng(61003)
        left_count, right_count = 3, 2
        left = _complex(rng, (left_count, 4, left_count, 3))
        right = _complex(rng, (4, right_count, 3))
        gamma = _complex(rng, (4, 4))
        record = _record(rng, left_count, right_count)
        dense = _dense_record(record, left_count, right_count)
        coefficient = -0.3 + 0.8j
        equation = "iAia,Dkd->i"
        actual = contract_p2p_bridge(
            left, right, ((gamma, coefficient, record),),
            used_np=left_count, used_np_right=right_count,
            left_axes=(1, 2, 3), right_axes=(0, 1, 2), equation=equation,
        )
        dense_current = np.einsum("AD,ikcd->ADikcd", gamma, dense, optimize=False)
        expected = coefficient * np.einsum(
            "iAia,Dkd,ADikad->i", left, right, dense_current, optimize=False,
        )
        self.assertEqual(actual.shape, (left_count,))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)

        direct_join = contract_sparse_p2p_bridge_join(
            left, right, ((gamma, coefficient, record),),
            used_np=left_count, used_np_right=right_count,
            left_axes=(1, 2, 3), right_axes=(0, 1, 2),
            same_point=False, transpose_bridge=False, equation=equation,
            point_equivalence={"left": "left-sample", "right": "right-sample"},
            output_labels="i",
        )
        np.testing.assert_allclose(direct_join, expected, rtol=1e-12, atol=1e-10)

    def test_two_rectangular_currents_use_cross_current_join_and_match_dense_oracle(self):
        (rng, left, right, gamma0, gamma1, record0, record1,
         first, second, equation) = _rectangular_cycle_case()
        self.assertEqual(first.point_shape, (3, 2))
        self.assertEqual(second.point_shape, (2, 3))
        self.assertFalse(cycle_point_equalities_covered(equation, first, second))

        dense0 = _dense_record(record0, 3, 2)
        dense1 = _dense_record(record1, 2, 3)
        current0 = np.einsum("AD,xybd->ADxbyd", gamma0, dense0, optimize=False)
        current1 = np.einsum("BC,yxac->BCyaxc", gamma1, dense1, optimize=False)
        expected = np.einsum(
            "BAxbya,DCydxc,ADxbyd,BCyaxc->",
            left, right, current0, current1, optimize=False,
        )
        actual = contract_p2p_cycle(
            left, right,
            ((gamma0, 1, record0),), ((gamma1, 1, record1),),
            first_bridge=first, second_bridge=second, equation=equation,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)
