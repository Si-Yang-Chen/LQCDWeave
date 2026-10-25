"""Explicit frontend bindings reuse a fixed path without changing physics."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from correlator.plan_archive import test_workflow_import as fixtures
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.storage.contraction_plans.api import (
    PlanArchiveError, load_storage_schedule, save_storage_schedule,
)
from lqcdweave.contraction.correlator import calc_diagram_eval


class ScheduleBindingTests(unittest.TestCase):
    def fixture(self):
        diagram, propagator, elemental, vertices = fixtures.WorkflowPlanImportTests()._fixture()
        options = dict(propagator_map={'q': propagator}, vertex_map=vertices,
                       configuration='cfg', usedNe=1, memory_limit=1 << 20,
                       optimize='none', execute=False)
        selected = calc_diagram_eval(diagram, time_axes={0: 1, 1: 0}, **options)
        current = calc_diagram_eval(diagram, time_axes={0: 2, 1: 0}, **options)
        return selected, current

    def save(self, path, selected, hardware):
        shapes = {key: list(spec.shape) for key, spec in selected.expanded.problem.specs.items()}
        save_storage_schedule(path, (selected.plan,), shapes=shapes, hardware=hardware)

    def test_explicit_frontend_rebinds_time_and_preserves_saved_actions_and_cost(self):
        selected, current = self.fixture()
        hardware = {'backend': 'numpy', 'memory': 1 << 20}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'selected.json.gz'
            self.save(path, selected, hardware)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('loading must not search')):
                (restored,), _ = load_storage_schedule(path, expected=current.expanded,
                                                      hardware=hardware)
        self.assertEqual(restored.state.actions, selected.plan.state.actions)
        self.assertEqual(restored.memory, selected.plan.memory)
        self.assertEqual(restored.seconds, selected.plan.seconds)
        actual = replay_scene_storage(restored, current.inputs.loader)
        reference = replay_scene_storage(current.plan, current.inputs.loader)
        old = replay_scene_storage(selected.plan, selected.inputs.loader)
        for value, expected in zip(actual, reference, strict=True):
            np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(actual, old, strict=True)))

    def test_bare_problem_cannot_implicitly_change_identity_bindings(self):
        selected, current = self.fixture()
        hardware = {'backend': 'numpy', 'memory': 1 << 20}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'selected.json'
            self.save(path, selected, hardware)
            with self.assertRaisesRegex(PlanArchiveError, 'semantic identity'):
                load_storage_schedule(path, expected=current.expanded.problem,
                    hardware=hardware, blocks=current.inputs.blocks,
                    preparations=current.inputs.preparations)

    def test_explicit_bindings_still_reject_changed_physics_and_preparation(self):
        selected, current = self.fixture()
        hardware = {'backend': 'numpy', 'memory': 1 << 20}
        problem = current.expanded.problem
        key, spec = next(iter(problem.specs.items()))
        description = replace(spec.description, semantics={'phase': 'different-physics'})
        changed = replace(problem, specs=dict(problem.specs, **{key: replace(spec, description=description)}))
        wrong_physics = replace(current.expanded, inputs=replace(current.inputs, problem=changed))
        key = next(key for key, prep in current.inputs.preparations.items()
                   if prep.recipe is not None)
        prep = current.inputs.preparations[key]
        wrong_preps = dict(current.inputs.preparations)
        wrong_preps[key] = replace(prep, workspace_host_bytes=prep.workspace_host_bytes + 1)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'selected.json'
            self.save(path, selected, hardware)
            with self.assertRaisesRegex(PlanArchiveError, 'semantic identity'):
                load_storage_schedule(path, expected=wrong_physics, hardware=hardware)
            with self.assertRaisesRegex(PlanArchiveError, 'source preparation changed'):
                load_storage_schedule(path, expected=current.expanded, hardware=hardware,
                    blocks=current.inputs.blocks, preparations=wrong_preps)


if __name__ == '__main__':
    unittest.main()
