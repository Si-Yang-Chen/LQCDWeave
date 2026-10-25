"""CPU protocol regression for terminal output ownership in saved cfg replay."""
from contextlib import contextmanager
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from correlator.execution.test_device import _FakeCupy, _Tracker
from correlator.planning.test_gemm_groups import fixture
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, DenseGemmGroup, SourcePreparation, TensorPreparation,
    describe_coordinates, validate_executable_resources,
)
from lqcdweave.contraction.planning.api import (
    build_cfg_plan, lower_independent_dense_batches,
)
from lqcdweave.contraction.execution import cfg_runtime, storage_runtime
from lqcdweave.contraction.execution.operations import gemm_group
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


def saved_cfg(*, releases=False):
    plan, profile = fixture()
    if releases:
        # Source release records are eliminated by cfg's global lifetime pass.
        state = plan.state
        for key in state.problem.specs:
            state = state.release(key)
        # Replan only storage for the same selected contractions.
        from lqcdweave.contraction.planning.api import plan_scene_storage
        from lqcdweave.contracts.correlator.api import dense_batch_memory
        selected = plan_scene_storage(state, {}, plan.preparations, profile=profile,
            memory_limit=1 << 20, host_memory_limit=1 << 20)
        selected = replace(selected, kernel_batches=plan.kernel_batches)
        memory = dense_batch_memory(selected)
        plan = replace(selected, memory=memory,
                       peak_device_bytes=max(point.device_bytes for point in memory))
    plan, _ = lower_independent_dense_batches(plan, profile=profile, target_runtime='cuda')
    rng = np.random.default_rng(8031)
    values = {key: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
              for key, spec in plan.state.problem.specs.items()}
    preps = {key: SourcePreparation((), recipe=TensorPreparation('constant',
        constants=(ArrayConstant.from_array(value),))) for key, value in values.items()}
    points, _ = describe_coordinates(time_pairs=[{0:0, 1:0}, {0:1, 1:1}])
    cfg = build_cfg_plan(plan, None, points.points, {}, preps, plan.state.problem.specs,
                         prefetch_host_bytes=0, prefetch_device_bytes=0)
    return plan, decode_plan_value(encode_plan_value(cfg)), values


class CfgTemplateTerminalMemoryTests(unittest.TestCase):
    def test_saved_cfg_budget_is_still_enforced_before_inputs_are_opened(self):
        _, cfg, _ = saved_cfg()
        backend = _FakeCupy(_Tracker())
        factory = unittest.mock.Mock(side_effect=AssertionError('inputs opened before budget check'))
        for debug in (False, True):
            with self.subTest(debug=debug):
                with self.assertRaisesRegex(MemoryError, 'below the saved cfg envelope'):
                    cfg_runtime.execute_cfg_plan(cfg, ('2000',), factory, lambda *_: None,
                        backend=backend, debug=debug, device_memory_limit=cfg.peak_device_bytes-1)
        factory.assert_not_called()

    def test_terminal_output_point_survives_with_and_without_removed_releases(self):
        for releases in (False, True):
            with self.subTest(releases=releases):
                original, cfg, _ = saved_cfg(releases=releases)
                template = cfg.templates[0].plan
                before = original.memory[-1]
                after = template.memory[-1]
                self.assertEqual(before.phase, 'output_copy')
                self.assertEqual(before.action_index, len(original.state.actions))
                self.assertEqual(after.action_index, len(template.state.actions))
                self.assertEqual(after, replace(before, action_index=len(template.state.actions)))
                self.assertEqual(template.peak_device_bytes, original.peak_device_bytes)
                self.assertEqual(template.peak_host_bytes, original.peak_host_bytes)
                self.assertEqual(template.peak_device_bytes,
                                 max(point.device_bytes for point in template.memory))
                self.assertEqual(len(original.state.actions) > len(template.state.actions), releases)
                validate_executable_resources(template)
                if not releases:
                    self.assertGreater(after.device_bytes,
                                       max(point.device_bytes for point in template.memory[:-1]))
                    damaged = replace(template, memory=template.memory[:-1])
                    with self.assertRaisesRegex(ValueError, 'peaks disagree'):
                        validate_executable_resources(damaged)

    def test_saved_cfg_fake_cupy_replay_validates_and_reuses_second_coordinate(self):
        for releases in (False, True):
            with self.subTest(releases=releases):
                self._assert_replay(releases=releases)

    def _assert_replay(self, *, releases):
        original, cfg, values = saved_cfg(releases=releases)
        self.assertIsInstance(cfg.instances[0].kernel_batches[0], DenseGemmGroup)
        self.assertFalse(cfg.instances[1].kernel_batches)
        expected = tuple(values[term.inputs[0]] @ values[term.inputs[1]]
                         for term in original.state.problem.terms)
        backend = _FakeCupy(_Tracker())
        backend.asnumpy = np.asarray

        @contextmanager
        def factory(_):
            yield bind_prepared_loader(None, cfg.preparations, cfg.specs, backend=backend)

        native_bind = gemm_group.bind_dense_gemm_group
        for debug in (False, True):
            with self.subTest(debug=debug):
                outputs = []
                # The fake device uses NumPy storage and GEMM. Everything from
                # cfg archive/ID binding through real resource audit, releases,
                # output copying and delivery follows the public execution path.
                with patch.object(gemm_group, 'bind_dense_gemm_group',
                        side_effect=lambda group, catalog, _backend: native_bind(group, catalog, np)), \
                     patch.object(storage_runtime, 'validate_executable_resources',
                        wraps=validate_executable_resources) as audit, \
                     patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                        side_effect=AssertionError('saved cfg cannot regroup')):
                    report = cfg_runtime.execute_cfg_plan(cfg, ('2000',), factory,
                        lambda _cfg, index, result: outputs.append(result) if index is not None else None,
                        backend=backend, debug=debug)
                self.assertEqual(report['completed'], ('2000',))
                self.assertEqual(audit.call_count, 2 if debug else 0)
                self.assertEqual(len(outputs), 2)
                for output in outputs:
                    for actual, reference in zip(output, expected, strict=True):
                        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
