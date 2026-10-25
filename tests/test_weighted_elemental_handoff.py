"""Weighted data declarations, source adjoints, and changed sample prefixes."""

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate


class WeightedElementalHandoffTests(unittest.TestCase):
    def contexts(self):
        from test_double_current_sampling_frontend import DoubleCurrentSamplingFrontendTests
        fixture = DoubleCurrentSamplingFrontendTests()
        return tuple(fixture._context(
            6, 5, pool_mode="same", sets={"shared-across-times": tuple(range(5))},
            seed=914, elemental_weighted=flag,
        ) for flag in (False, True))

    def test_prefix_reselection_and_source_dagger_match_raw_estimator(self):
        for source in (False, True):
            raw, weighted = self.contexts()
            results = []
            for flag, context in zip((False, True), (raw, weighted)):
                vertices = {key: Meson(vertex.reader, vertex.operator, source=source,
                                       phase_convention="anchor")
                            for key, vertex in context["vertices"].items()}
                result = calculate(
                    context["diagram"], propagator_map=context["prop_map"],
                    vertex_map=vertices, configuration="cfg-double",
                    usedNe=1, usedNp=4, optimize="none", elemental_weighted=flag,
                )
                self.assertEqual(result.options["elemental_weighted"], flag)
                results.append(result.value)
            np.testing.assert_allclose(*results, rtol=2e-11, atol=2e-9)

    def test_mismatched_option_fails_before_any_numerical_read(self):
        for flag, context in zip((True, False), self.contexts()):
            with patch.object(context["provider"], "read_record",
                              side_effect=AssertionError("numeric data was read")):
                with self.assertRaisesRegex(ValueError, "stored elemental weighting"):
                    calculate(context["diagram"], propagator_map=context["prop_map"],
                              vertex_map=context["vertices"], usedNe=1, usedNp=5,
                              optimize="none", execute=False, elemental_weighted=flag)
            self.assertEqual(context["provider"].read_times, [])

    def test_saved_dense_plan_retains_residuals_and_checks_weight_marker(self):
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.contraction.correlator import save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        context = self.contexts()[1]
        sparse = context["vertices"]["early"].reader
        dense = sparse.dense_reader
        p2p = np.zeros((1, 1, 3, 5, 3, 5, 3), dtype=complex)
        for (component, mi, time), record in context["provider"].records.items():
            for (left, right), value in zip(record["indices"], record["values"]):
                p2p[0, mi, time, left, :, right, :] = value
        elemental = ArrayElemental(
            {**dense.sectors, "p2p": p2p}, dense.components,
            phase_convention="anchor", cfg="cfg-double", data_identity="weighted-dense",
            sampling_pool_ids=dense.sampling_pool_ids, sampling_groups=dense.sampling_groups,
            elemental_weighted=True,
        )
        vertices = {key: Meson(elemental, vertex.operator, phase_convention="anchor")
                    for key, vertex in context["vertices"].items()}
        selected = calculate(
            context["diagram"], propagator_map=context["prop_map"], vertex_map=vertices,
            configuration="cfg-double", usedNe=1, usedNp=4, optimize="none",
            elemental_weighted=True, execute=False,
        )
        complete = complete_calculation(selected, readers=selected.inputs.loader.input_readers)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weighted.json.gz"
            save_calculation_plan(path, complete)
            with patch("lqcdweave.contraction.correlator.expand_correlator",
                       side_effect=AssertionError("archive performed expansion")):
                actual = calculate(plan_archive=path, configuration="cfg-double",
                                   propagator_map=context["prop_map"], vertex_map=elemental).value
            np.testing.assert_allclose(actual, selected.run().value, rtol=2e-11, atol=2e-9)
            elemental.elemental_weighted = False
            with patch.object(elemental, "block", side_effect=AssertionError("numeric read")):
                with self.assertRaisesRegex(ValueError, "conventions differ"):
                    calculate(plan_archive=path, configuration="cfg-double",
                              propagator_map=context["prop_map"], vertex_map=elemental)

    def test_registered_dataset_restores_weighting_and_rejects_mixed_markers(self):
        from test_dataset_point_elemental_adapter import _DatasetPointElementalAdapterFixture
        from lqcdweave.adapters.meson_dataset import DatasetElemental
        from lqcdweave.contracts.dataset import DatasetSpec
        fixture = _DatasetPointElementalAdapterFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        registered = fixture.readers["v2p"]
        document = registered.spec.document
        document["conventions"]["elemental_weighted"] = True
        weight = np.prod(fixture.helper.sizes[:3]) / len(fixture.helper.labels)
        values = np.stack([registered.read(time=t) for t in range(3)]) * weight
        identity = fixture.helper.save("weighted-v2p", DatasetSpec(document), values)
        restored = fixture.helper.index.open(identity)
        self.addCleanup(restored.close)
        elemental = DatasetElemental({"v2p": restored})
        self.assertTrue(elemental.elemental_weighted)
        component = elemental.components[0]
        np.testing.assert_allclose(elemental.block(component, 0, 1, "v2p"), values[1, 0, 0])
        with self.assertRaisesRegex(ValueError, "one boolean elemental_weighted"):
            DatasetElemental({"v2p": restored, "p2v": fixture.readers["p2v"]})


if __name__ == "__main__":
    unittest.main()
