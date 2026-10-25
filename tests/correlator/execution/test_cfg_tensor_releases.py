"""Fixed cfg replay releases numerical owners at their global final uses."""
from contextlib import contextmanager
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.planning.api import (
    audit_cfg_plan, build_cfg_plan, solve_storage,
)
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.contraction.execution.cfg_runtime import _CfgBlockExecution
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value


class CfgTensorReleaseTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(318)
        values = {f'{x}{i}': rng.normal(size=(12, 12)).astype(np.complex128)
                  for i in range(8) for x in 'abc'}
        specs = {k: TensorSpec(v.shape, v.dtype) for k, v in values.items()}
        terms = tuple(ContractionTerm(tuple(f'{x}{i}' for x in 'abc'), 'ij,jk,ki->')
                      for i in range(8))
        problem = ContractionProblem(specs, terms,
            {k: SourceSpec(k, 0, read_bytes=0) for k in specs}, backend='numpy')
        preparations = {k: SourcePreparation((), recipe=TensorPreparation(
            'constant', constants=(ArrayConstant.from_array(v),))) for k, v in values.items()}
        selected = solve_storage(problem, {}, preparations, optimizer='anytime',
            memory_limit=1 << 25, deadline_seconds=1, max_states=1000,
            root_width=1, branch_width=1, component_workers=1, kernel_batching=False).plan
        points, _ = describe_coordinates()
        cfg = build_cfg_plan(selected, None, points.points, {}, preparations, specs,
                             prefetch_host_bytes=0, debug=True)
        return cfg, values

    def test_final_use_envelope_fits_when_point_end_retention_does_not(self):
        cfg, values = self.fixture()
        instance = cfg.instances[0]
        self.assertIsNotNone(instance.tensor_releases)
        rows = instance.tensor_releases.array()
        self.assertGreater(len(rows), 20)
        delayed = replace(cfg, instances=(replace(instance, tensor_releases=None,
            releases=instance.releases + tuple(map(int, rows[:, 1]))),),
            peak_host_bytes=1 << 25)
        delayed_peak = audit_cfg_plan(delayed)['peak_host_bytes']
        self.assertGreater(delayed_peak, cfg.peak_host_bytes)
        cap = (delayed_peak + cfg.peak_host_bytes) // 2
        cfg = replace(cfg, statistics={**cfg.statistics, 'host_memory_cap': cap})
        audit_cfg_plan(cfg)
        with self.assertRaisesRegex(ValueError, 'exceeds.*cap|exceeds.*budget'):
            audit_cfg_plan(replace(delayed, statistics={**delayed.statistics,
                                                       'host_memory_cap': cap}))
        restored = decode_plan_value(encode_plan_value(cfg))
        self.assertEqual(restored.instances, cfg.instances)
        expected = [np.einsum('ij,jk,ki->', *(values[f'{x}{i}'] for x in 'abc'))
                    for i in range(8)]
        observed_releases = []
        original = _CfgBlockExecution.end_action
        def observe(execution, index, timed):
            gids = execution.tensor_releases.get(index, ())
            self.assertTrue(all(gid in execution.arrays for gid in gids))
            original(execution, index, timed)
            self.assertTrue(all(gid not in execution.arrays for gid in gids))
            observed_releases.extend(gids)
        @contextmanager
        def factory(configuration):
            yield restore_input_loader(restored.requests, {}, restored.preparations,
                                       restored.specs, configuration=configuration)
        for debug in (False, True):
            results = []
            with patch.object(_CfgBlockExecution, 'end_action', observe):
                report = execute_cfg_plan(restored, ['cfg'], factory,
                    lambda cfg, index, output: results.append(output) if index is not None else None,
                    debug=debug)
            self.assertEqual(report['completed'], ('cfg',))
            np.testing.assert_allclose(np.asarray(results[0]), expected, rtol=2e-12, atol=2e-12)
        self.assertEqual(len(observed_releases), 2 * len(rows))

    def test_early_output_release_is_rejected_by_independent_audit(self):
        cfg, _ = self.fixture()
        instance = cfg.instances[0]
        gid = instance.releases[0]
        rows = np.vstack((instance.tensor_releases.array(), [0, gid]))
        rows = rows[np.argsort(rows[:, 0], kind='stable')].astype('<i8')
        bad = replace(instance, releases=instance.releases[1:],
                      tensor_releases=ArrayConstant.from_array(rows))
        with self.assertRaisesRegex(ValueError, 'output was released|consumer precedes|absent'):
            audit_cfg_plan(replace(cfg, instances=(bad,)))


if __name__ == '__main__':
    unittest.main()
