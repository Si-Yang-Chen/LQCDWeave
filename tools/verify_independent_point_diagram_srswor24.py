"""Opt-in 24-pair registered SRSWOR point-Diagram numerical oracle.

Run explicitly with ``python tools/verify_independent_point_diagram_srswor24.py``.
The test-only fixtures use public CPU kernels and generic DatasetWriter; this
script does not exercise a fixed measurement runner or persistence lifecycle.
"""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from sympy import Symbol


REPOSITORY = Path(__file__).resolve().parents[1]
_parser = ArgumentParser(description=__doc__)
_parser.add_argument(
    "--package-root", type=Path, default=REPOSITORY / "src",
    help="root containing lqcdweave/ (defaults to this worktree's src)",
)
PACKAGE_ROOT = _parser.parse_args().package_root.resolve()
if not (PACKAGE_ROOT / "lqcdweave").is_dir():
    raise SystemExit(f"package root has no lqcdweave/: {PACKAGE_ROOT}")
sys.path.insert(0, str(REPOSITORY / "tests"))
sys.path.insert(0, str(PACKAGE_ROOT))

from test_independent_point_diagram_numeric import (  # noqa: E402
    IndependentPointDiagramNumericTests,
)
from lqcdweave.adapters.meson_dataset import DatasetPropagator  # noqa: E402
from lqcdweave.contraction.expansion.sampling_binding import (  # noqa: E402
    describe_point_pool_metadata,
)
from lqcdweave.contraction.correlator import calc_diagram_eval  # noqa: E402


def _assert_imports_from_package_root(testcase):
    root = PACKAGE_ROOT.resolve()
    wrong = []
    observed = []
    for name, module in tuple(sys.modules.items()):
        if name != "lqcdweave" and not name.startswith("lqcdweave."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        resolved = Path(origin).resolve()
        observed.append((name, resolved))
        if not resolved.is_relative_to(root):
            wrong.append((name, resolved))
    testcase.assertTrue(observed, "no lqcdweave modules were loaded")
    testcase.assertFalse(wrong, f"package imports escaped selected root: {wrong!r}")
    print(f"PACKAGE_ROOT={root}; loaded_package_modules={len(observed)}", flush=True)


class IndependentPointDiagramSRSWOR24Acceptance(unittest.TestCase):
    def setUp(self):
        self.fixture = IndependentPointDiagramNumericTests(
            "test_full_population_independent_pools_match_full_time_trace"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        _assert_imports_from_package_root(self)

    def _assert_rectangular_source_dagger_bindings(self, ordinary, dagger):
        ordinary_descriptor = ordinary.describe_factors(0, "p2p")[0]
        dagger_descriptor = dagger.describe_factors(0, "p2p")[0]
        self.assertFalse(ordinary_descriptor.dagger)
        self.assertTrue(dagger_descriptor.dagger)
        self.assertEqual(ordinary_descriptor.logical_sector, "p2p")
        self.assertEqual(dagger_descriptor.logical_sector, "p2p")
        ordinary_legs, ordinary_pools = describe_point_pool_metadata(
            ((ordinary_descriptor,),), ({"left": "p", "right": "p"},),
        )
        dagger_legs, dagger_pools = describe_point_pool_metadata(
            ((dagger_descriptor,),), ({"left": "p", "right": "p"},),
        )
        self.assertEqual(ordinary_legs[(0, "field")], "left-pool")
        self.assertEqual(ordinary_legs[(0, "bar")], "right-pool")
        self.assertEqual(dagger_legs[(0, "field")], "right-pool")
        self.assertEqual(dagger_legs[(0, "bar")], "left-pool")
        for pools in (ordinary_pools, dagger_pools):
            self.assertEqual(pools["left-pool"].spatial_sites, 4)
            self.assertEqual(pools["left-pool"].sampled_points, 3)
            self.assertEqual(pools["right-pool"].spatial_sites, 4)
            self.assertEqual(pools["right-pool"].sampled_points, 2)

    def _calculate_full_grid(self, left, right, elemental_ids, oracle_pairs):
        fixture = self.fixture
        helper = fixture.helper
        source, sink = Symbol("srswor-source"), Symbol("srswor-sink")
        with ExitStack() as stack:
            diagram, propagators, vertices = fixture._bind(
                stack, left, right, elemental_ids, source, sink,
            )
            self.assertNotEqual(left[0], right[0])
            self.assertEqual(len(vertices["current-a"].elemental.point_dataset.spec
                                 .coordinates("point")), 3)
            self.assertEqual(len(vertices["current-a"].elemental.point_right_dataset.spec
                                 .coordinates("point")), 2)
            self._assert_rectangular_source_dagger_bindings(
                vertices["current-a"], vertices["current-b"],
            )
            values = {}
            for pair in oracle_pairs:
                with patch.object(
                    DatasetPropagator, "block", autospec=True,
                    side_effect=DatasetPropagator.block,
                ) as reads:
                    calculation = calc_diagram_eval(
                        diagram, propagator_map=propagators,
                        vertex_map=vertices, configuration="cfg-double-current",
                        usedNe=1, usedNp=None, groups_independent=True,
                        optimize="none", time_axes={source: pair[0], sink: pair[1]},
                    )
                values[pair] = calculation.value
                psp_calls = [call for call in reads.call_args_list
                             if call.args[3] == "psp"]
                psp_readers = {id(call.args[0]) for call in psp_calls}
                self.assertTrue(
                    {id(propagator) for propagator in propagators.values()} <= psp_readers
                )
                observed_endpoint_ids = set()
                for call in psp_calls:
                    endpoint_bindings = call.args[0].point_bindings
                    if "psp" in endpoint_bindings:
                        endpoint_bindings = endpoint_bindings["psp"]
                    if {"source_point", "sink_point"} <= set(endpoint_bindings):
                        observed_endpoint_ids.add((
                            endpoint_bindings["source_point"][0],
                            endpoint_bindings["sink_point"][0],
                        ))
                self.assertTrue({(left[0], left[0]), (right[0], right[0])}
                                <= observed_endpoint_ids)
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
        return values

    def test_all_24_real_point_pairs_average_the_full_trace(self):
        fixture = self.fixture
        helper = fixture.helper
        oracle = helper._independent_trace_values()
        self.assertEqual(len(oracle), 9)
        self.assertEqual(set(oracle), {
            (source_time, sink_time)
            for source_time in helper.times for sink_time in helper.times
        })
        left_subsets = tuple(combinations(range(4), 3))
        right_subsets = tuple(combinations(range(4), 2))
        left_samples = {subset: fixture._publish_sample("left", subset)
                        for subset in left_subsets}
        right_samples = {subset: fixture._publish_sample("right", subset)
                         for subset in right_subsets}
        self.assertEqual(len(left_samples), 4)
        self.assertEqual(len(right_samples), 6)
        print("Produced 4 left and 6 right registered point/propagator bundles.", flush=True)

        accumulated = {pair: 0j for pair in oracle}
        completed_pairs = 0
        for number, (left_subset, left) in enumerate(left_samples.items()):
            for right_subset, right in right_samples.items():
                print(f"Pair {completed_pairs + 1}/24: "
                      f"left={left_subset}, right={right_subset}", flush=True)
                elemental_ids = fixture._publish_elementals(
                    f"srswor24-elemental-{number}-{right_subset}", left, right,
                    used_np_left=3, used_np_right=2,
                )
                values = self._calculate_full_grid(left, right, elemental_ids, oracle)
                self.assertEqual(set(values), set(oracle))
                for pair, value in values.items():
                    accumulated[pair] += value
                completed_pairs += 1
        self.assertEqual(completed_pairs, 24)

        _assert_imports_from_package_root(self)
        for pair in oracle:
            average = accumulated[pair] / 24
            with self.subTest(source_time=pair[0], sink_time=pair[1]):
                np.testing.assert_allclose(average, oracle[pair], rtol=3e-10, atol=3e-10)
            print(f"PASS mean[{pair[0]},{pair[1]}] over 24 registered samples", flush=True)


if __name__ == "__main__":
    print(f"Using package root: {PACKAGE_ROOT}", flush=True)
    suite = unittest.TestSuite((
        IndependentPointDiagramSRSWOR24Acceptance(
            "test_all_24_real_point_pairs_average_the_full_trace"
        ),
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
