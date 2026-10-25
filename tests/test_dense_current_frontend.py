"""Independent block-matrix oracle for all four dense current sectors."""

import unittest
from itertools import combinations

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Propagator
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calc_diagram_eval


def _single_current_weighted(spatial, population, sampled_points):
    """Apply the single-current factors to already selected spatial arrays."""
    weighted = {sector: array.copy() for sector, array in spatial.items()}
    one_point_weight = population / sampled_points
    weighted["v2p"] *= one_point_weight
    weighted["p2v"] *= one_point_weight

    p2p_weight = np.full(
        (sampled_points, sampled_points),
        population * (population - 1) / (sampled_points * (sampled_points - 1)),
    )
    np.fill_diagonal(p2p_weight, one_point_weight)
    weighted["p2p"] *= p2p_weight[None, None, None, :, None, :, None]
    return weighted


class DenseCurrentFrontendTests(unittest.TestCase):
    def test_full_point_population_matches_explicit_block_matrix_trace(self):
        # Synthetic dense data, not a production estimator calibration. With
        # every spatial point sampled, the four sectors form an ordinary matrix.
        rng = np.random.default_rng(92731)
        ne, npoint, lt = 1, 3, 2

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        current = LocalVectorCurrent().operator(3)
        component = current.terms[0].component
        normal = Operator.from_terms("normal", [OperatorTerm(0, component, 0, 1)])
        tails = {"v2v": (ne, ne), "v2p": (ne, npoint, 3),
                 "p2v": (npoint, 3, ne), "p2p": (npoint, 3, npoint, 3)}
        spatial = {sector: random((1, 1, lt, *shape)) for sector, shape in tails.items()}
        pools = {(component, 0, t, side): "whole-space"
                 for t in range(lt) for side in ("bar", "field")}
        elemental = ArrayElemental(
            spatial, (component,), phase_convention="anchor", sampling_pool_ids=pools,
            sampling_groups={"whole-space": SamplingPoolSpec("whole-space", npoint, npoint)})
        propagators = {
            "vsv": random((lt, lt, 4, 4, ne, ne)),
            "vsp": random((lt, lt, 4, 4, ne, npoint, 3)),
            "psv": random((lt, lt, 4, 4, npoint, 3, ne)),
            "psp": random((lt, lt, 4, 4, npoint, 3, npoint, 3)),
        }
        reader = ArrayPropagator(propagators, time_convention="absolute")
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                        ["current", "normal"], [None, q])
        actual = calc_diagram_eval(
            graph, propagator_map={q: reader},
            vertex_map={"current": Meson(elemental, current, phase_convention="anchor"),
                        "normal": Meson(elemental, normal, phase_convention="anchor")},
            usedNe=ne, usedNp=npoint, optimize="none").value

        def matrix(tensor, rows, columns):
            return tensor.reshape(4, 4, rows, columns).transpose(0, 2, 1, 3).reshape(
                4 * rows, 4 * columns)

        npcolor = npoint * 3
        current_spatial = np.block([
            [spatial["v2v"][0, 0, 0], spatial["v2p"][0, 0, 0].reshape(ne, npcolor)],
            [spatial["p2v"][0, 0, 0].reshape(npcolor, ne),
             spatial["p2p"][0, 0, 0].reshape(npcolor, npcolor)],
        ])
        # Keep spin-major ordering when combining mode and point subspaces.
        outgoing = np.concatenate((propagators["vsv"][0, 1],
                                   propagators["vsp"][0, 1].reshape(4, 4, ne, npcolor)), axis=3)
        incoming = np.concatenate((propagators["vsv"][1, 0],
                                   propagators["psv"][1, 0].reshape(4, 4, npcolor, ne)), axis=2)
        current_matrix = np.kron(gamma(current.terms[0].gamma), current_spatial)
        normal_matrix = np.kron(gamma(0), spatial["v2v"][0, 0, 1])
        expected = np.trace(matrix(outgoing, ne, ne + npcolor) @ current_matrix
                            @ matrix(incoming, ne + npcolor, ne) @ normal_matrix)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)

        weighted_full_elemental = ArrayElemental(
            spatial, (component,), phase_convention="anchor", sampling_pool_ids=pools,
            sampling_groups={"whole-space": SamplingPoolSpec("whole-space", npoint, npoint)},
            elemental_weighted=True,
        )
        weighted_full = calc_diagram_eval(
            graph, propagator_map={q: reader},
            vertex_map={"current": Meson(weighted_full_elemental, current, phase_convention="anchor"),
                        "normal": Meson(weighted_full_elemental, normal, phase_convention="anchor")},
            usedNe=ne, usedNp=npoint, optimize="none", elemental_weighted=True,
        ).value
        np.testing.assert_allclose(weighted_full, expected, rtol=1e-12, atol=1e-11)

        # Enumerate every size-two point sample independently of the scene
        # implementation. The average estimator must recover the full trace,
        # including off-diagonal and diagonal P2P contributions.
        for elemental_weighted in (False, True):
            estimates = []
            for selected in combinations(range(npoint), 2):
                selected_spatial = {}
                for sector, array in spatial.items():
                    point_axes = {"v2v": (), "v2p": (4,), "p2v": (3,), "p2p": (3, 5)}[sector]
                    for axis in point_axes:
                        array = np.take(array, selected, axis=axis)
                    selected_spatial[sector] = array
                if elemental_weighted:
                    selected_spatial = _single_current_weighted(
                        selected_spatial, npoint, len(selected),
                    )
                selected_props = {}
                for sector, array in propagators.items():
                    point_axes = {"vsv": (), "vsp": (5,), "psv": (4,), "psp": (4, 6)}[sector]
                    for axis in point_axes:
                        array = np.take(array, selected, axis=axis)
                    selected_props[sector] = array
                sampled_elemental = ArrayElemental(
                    selected_spatial, (component,), phase_convention="anchor",
                    sampling_pool_ids=pools,
                    sampling_groups={"whole-space": SamplingPoolSpec("whole-space", npoint, 2)},
                    elemental_weighted=elemental_weighted,
                )
                sampled_reader = ArrayPropagator(selected_props, time_convention="absolute")
                estimates.append(calc_diagram_eval(
                    graph, propagator_map={q: sampled_reader},
                    vertex_map={"current": Meson(sampled_elemental, current, phase_convention="anchor"),
                                "normal": Meson(sampled_elemental, normal, phase_convention="anchor")},
                    usedNe=ne, usedNp=2, optimize="none",
                    elemental_weighted=elemental_weighted,
                ).value)
            np.testing.assert_allclose(np.mean(estimates), expected, rtol=1e-12, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
