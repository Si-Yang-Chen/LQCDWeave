"""Unbatched archives retain their selected native kernel declarations."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator import api as c
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.operators.correlator.api import split_meson_mode_pairs
from lqcdweave.storage.contraction_plans import api
from test_structured_correlator import fixture, task


@pytest.fixture(scope='module')
def unbatched_whole(tmp_path_factory):
    binding, _, propagator = fixture()
    value = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=(0, 1, 3))
    selected = calculate(value, vertex_map=binding,
        propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
        configuration='cfg', usedNe=3, time_axes={0: 1, 1: 0}, memory_limit=1 << 30,
        optimize='anytime', execute=False)
    assert not selected.expanded.inputs.task_batches
    assert selected.expanded.problem.terms[0].kernel is not None
    path = tmp_path_factory.mktemp('native-kernel') / 'fixed.json'
    save_calculation_plan(path, selected)
    hardware = json.loads(path.read_text())['payload']['hardware']
    return selected, path, hardware


def changed_kernel_problem(problem, change):
    term = problem.terms[0]
    kernel = term.kernel
    if change == 'declaration':
        kernel = replace(kernel, kernel_description=c.VertexKernelDescription('meson', 'whole-matrix-v2'))
    elif change == 'inputs':
        kernel = replace(kernel, inputs=(kernel.inputs[1], kernel.inputs[0], *kernel.inputs[2:]))
    elif change == 'output-order':
        pattern = kernel.private_equation
        kernel = replace(kernel, private_equation=c.AxisPattern(pattern.inputs, pattern.output[::-1]))
    else:
        kernel = None
    assert kernel != term.kernel
    return replace(problem, terms=(replace(term, kernel=kernel), *problem.terms[1:]))


def test_unchanged_native_kernel_replays_without_search(unbatched_whole):
    selected, path, hardware = unbatched_whole
    with patch('lqcdweave.contraction.planning.api.find_contraction_path',
               side_effect=AssertionError('fixed archive must not search')):
        loaded = api.load_contraction_plan(path, expected=selected.expanded, hardware=hardware)
    assert loaded.plan.state.actions == selected.path.plan.state.actions
    np.testing.assert_allclose(replace(selected, path=loaded).run().value,
                               selected.run().value, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('change', ('declaration', 'inputs', 'output-order', 'missing'))
def test_unbatched_path_rejects_changed_kernel(unbatched_whole, change):
    selected, path, hardware = unbatched_whole
    problem = changed_kernel_problem(selected.expanded.problem, change)
    expected = replace(selected.expanded, inputs=replace(selected.expanded.inputs, problem=problem))
    with pytest.raises(api.PlanArchiveError, match='term kernels differ'):
        api.load_contraction_plan(path, expected=expected, hardware=hardware)


def test_storage_schedule_checks_the_same_parsed_kernel(unbatched_whole, tmp_path):
    selected, _, _ = unbatched_whole
    plan = selected.path.plan
    problem = plan.state.problem
    shapes = {key: list(spec.shape) for key, spec in problem.specs.items()}
    hardware = {'backend': 'numpy', 'memory': 1 << 30}
    path = tmp_path / 'schedule.json'
    api.save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
    with pytest.raises(api.PlanArchiveError, match='term kernels differ'):
        api.load_storage_schedule(path, shapes=shapes, hardware=hardware,
            expected=(changed_kernel_problem(problem, 'declaration'),),
            blocks=plan.blocks, preparations=plan.preparations)


def test_old_missing_kernel_default_matches_only_the_same_ordinary_term():
    term = c.ContractionTerm(('a',), c.AxisPattern(((0,),), (0,)))
    encoded = api.encode_plan_value(term)
    del encoded['value']['kernel']
    restored = api.decode_plan_value(encoded)
    api._assert_saved_term_kernels(SimpleNamespace(terms=(restored,)), SimpleNamespace(terms=(term,)))
    with pytest.raises(api.PlanArchiveError, match='term kernels differ'):
        api._assert_saved_term_kernels(SimpleNamespace(terms=(restored,)),
            SimpleNamespace(terms=(replace(term, kernel=c.OutputNode('constant', 1)),)))
