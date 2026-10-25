"""Fixed whole routes must preserve the original immutable replay exactly."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lqcdweave.contraction.planning import meson_core, storage
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.performance.api import CostProfile
from correlator.execution.test_meson_whole_core import case


def fixture(kind, dtype, batched, backend, count=12):
    problem, _ = case(kind, dtype, batched)
    specs, sources, terms, cores = {}, {}, [], []
    for index in range(count):
        def name(key):
            return f'{index}:{key}'
        specs.update((name(key), spec) for key, spec in problem.specs.items())
        sources.update((name(key), source) for key, source in problem.sources.items())
        cores.extend(replace(core, key=name(core.key), inputs=tuple(map(name, core.inputs)))
                     for core in problem.meson_cores)
        term = problem.terms[0]
        kernel = replace(term.kernel, cores=tuple(map(name, term.kernel.cores)),
                         inputs=tuple(map(name, term.kernel.inputs)))
        terms.append(replace(term, inputs=tuple(map(name, term.inputs)), kernel=kernel))
    # Reuse after other outputs exist, and release different inputs at
    # different positions. Each distinct numerical value is still computed once.
    order = (*terms, *terms[::2], *reversed(terms), *terms[1::2])
    return replace(problem, specs=specs, sources=sources, terms=order,
                   meson_cores=tuple(cores), backend=backend)


def immutable_reference(problem, profile):
    state = ContractionState.initial(storage.logical_source_problem(problem))
    remaining = Counter(key for term in problem.terms for key in set(term.kernel.inputs))
    for scene, term in enumerate(problem.terms):
        for key in dict.fromkeys(term.kernel.inputs):
            if key not in state.loaded:
                state = state.load(key, profile, 1 << 100)
        action, _, _ = meson_core.describe_meson_whole_action(state, scene, profile=profile)
        state = meson_core.replay_meson_whole_action(state, action, profile)
        for key in dict.fromkeys(term.kernel.inputs):
            remaining[key] -= 1
            if not remaining[key]:
                state = state.release(key)
    return state


@pytest.mark.parametrize('kind', ('connected', 'loop'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('batched', (False, True))
@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
def test_fixed_delivery_matches_original_actions_costs_and_lifetimes(kind, dtype, batched, backend):
    problem = fixture(kind, dtype, batched, backend)
    profile = CostProfile()
    expected = immutable_reference(problem, profile)
    with patch.object(storage, 'plan_scene_storage',
                      side_effect=lambda state, *args, **kwargs: SimpleNamespace(state=state, feasible=True)), \
         patch.object(ContractionState, 'load', side_effect=AssertionError('immutable load loop')), \
         patch.object(ContractionState, 'release', side_effect=AssertionError('immutable release loop')), \
         patch.object(meson_core, 'estimate_meson_whole', wraps=meson_core.estimate_meson_whole) as estimate:
        actual = meson_core.plan_meson_whole(problem, {}, {}, profile=profile,
                                            memory_limit=1 << 40).state
    assert actual == expected
    assert actual.resident_bytes == expected.resident_bytes
    assert actual.resident_host_bytes == expected.resident_host_bytes
    unique = {action.tensor for action in expected.actions if action.kind == 'contract'}
    assert estimate.call_count == len(unique)
    assert all(type(action.seconds) is float for action in actual.actions if action.kind == 'contract')
