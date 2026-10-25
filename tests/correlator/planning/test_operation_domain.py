"""Non-outer operation domain shared by direct, heuristic and exact planning."""
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.path_planning import _first_direct_shared_label_pair, PathSearchExhausted
from lqcdweave.contraction.execution.api import replay


def case(equation, shapes):
    return ContractionProblem({k: TensorSpec(shape, "float64") for k, shape in shapes.items()},
        (ContractionTerm(tuple(shapes), equation),), {k: SourceSpec(k, 0) for k in shapes})


class NonOuterOperationTests(unittest.TestCase):
    def test_ordinary_and_batched_outer_products_are_rejected_before_cost(self):
        for equation, shapes in (("i,j->ij", {"A": (2,), "B": (3,)}),
                                 ("bi,bj->bij", {"A": (4, 2), "B": (4, 3)})):
            with self.subTest(equation=equation):
                problem = case(equation, shapes)
                state = ContractionState.initial(problem)
                with self.assertRaisesRegex(ValueError, "outer product"):
                    state._contract_layout(0, (0, 1))
                with self.assertRaisesRegex(ValueError, "outer product"):
                    _first_direct_shared_label_pair(state.pending[0])
                self.assertEqual(list(IntegerSpace(problem, profile=CostProfile()).candidates(lambda: None)), [])
                preparations = {key: SourcePreparation(()) for key in shapes}
                macro = MacroSearchSpace(problem, {}, preparations, memory_limit=10**8)
                self.assertEqual(list(macro.logical_steps(macro.initial_state())), [])
                with self.assertRaises(PathSearchExhausted):
                    solve_storage(problem, {}, preparations, memory_limit=10**8, optimizer="anytime")
                for mode in ("exact", "dynamic_programming"):
                    result = solve_storage(problem, {}, preparations, memory_limit=10**8, optimizer=mode)
                    self.assertIsNone(result.plan)
                    self.assertFalse(result.limited)

    def test_aligned_products_scaling_and_contractions_still_execute(self):
        for equation, arrays in (
            ("ij,ij->ij", {"A": np.arange(6.).reshape(2, 3), "B": np.arange(6.).reshape(2, 3) + 1}),
            ("ij,j->ij", {"A": np.arange(6.).reshape(2, 3), "B": np.arange(3.) + 1}),
            ("ij,->ij", {"A": np.arange(6.).reshape(2, 3), "B": np.array(2.)}),
            (",->", {"A": np.array(3.), "B": np.array(2.)}),
            ("ij,jk->ik", {"A": np.arange(6.).reshape(2, 3), "B": np.arange(12.).reshape(3, 4)}),
        ):
            with self.subTest(equation=equation):
                problem = case(equation, {key: array.shape for key, array in arrays.items()})
                state = ContractionState.initial(problem)
                for key in arrays:
                    state = state.load(key, CostProfile(), 10**8)
                self.assertEqual(_first_direct_shared_label_pair(state.pending[0]), (0, 1))
                after = state.contract(0, (0, 1), CostProfile(), 10**8)
                np.testing.assert_allclose(replay(after, lambda key, source: arrays[key])[0],
                                           np.einsum(equation, *arrays.values()))
