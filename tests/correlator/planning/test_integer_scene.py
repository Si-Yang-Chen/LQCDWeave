"""Integer-only Scene search and audit, independent of backend spelling."""
import unittest
from unittest.mock import patch

import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern
from lqcdweave.contraction.execution.api import replay
from correlator.planning.test_integer_search import make_problem, solve


class IntegerSceneTests(unittest.TestCase):
    def test_all_modes_keep_alignment_diagonal_coefficients_and_integer_actions(self):
        problem = make_problem([
            (('A', 'B'), AxisPattern(((900, 700), (900, 700)), (900,))),
            (('D',), AxisPattern(((400, 400),), ())),
        ], {'A': (2, 3), 'B': (2, 3), 'D': (3, 3)})
        arrays = {'A': np.arange(6).reshape(2, 3) + 1j,
                  'B': np.arange(6).reshape(2, 3) - 2j,
                  'D': np.arange(9).reshape(3, 3) + 3j}
        expected = ((arrays['A'] * arrays['B']).sum(axis=1), np.trace(arrays['D']))
        for mode in ('anytime', 'exact', 'dynamic_programming'):
            with self.subTest(mode=mode):
                with (patch('opt_einsum.contract_path', side_effect=AssertionError('string path wrapper')),
                      patch('opt_einsum.get_symbol', side_effect=AssertionError('generated labels'))):
                    result = solve(problem, optimizer=mode)
                for action in result.state.actions:
                    if action.kind == 'contract':
                        self.assertIsInstance(action.equation, AxisPattern)
                values = replay(result.state, lambda key, source: arrays[key])
                for actual, reference in zip(values, expected, strict=True):
                    np.testing.assert_allclose(actual, reference)



if __name__ == '__main__':
    unittest.main()
