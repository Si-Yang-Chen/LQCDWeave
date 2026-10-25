"""Joint task planning and fixed-path replay across independent cfg bindings."""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from test_structured_correlator import task
from tests.correlator.planning.test_meson_matrix import multi_component_fixture
from correlator.execution.test_cfg_runtime import SafeReader
from lqcdweave.contraction.correlator import calculate
from lqcdweave.contraction.correlator_output import load_correlator_output
import lqcdweave.contraction.correlator as planning


def test_joint_channels_plan_once_then_replay_multiple_cfgs_without_e_or_search(tmp_path):
    bind, elemental, propagator = multi_component_fixture()
    requests = {name: task(structured=True, names=(name,)) for name in ('a', 'z')}
    mapping = {'S^q': propagator, r'S^q_\mathrm{local}': propagator}
    pairs = ({0: 1, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 1, 1: 1})
    reference = calculate({name: task(names=(name,)) for name in requests},
        vertex_map=bind, propagator_map=mapping, configuration='cfg', usedNe=3,
        time_pairs=pairs, optimize='none', memory_limit=1 << 30).value
    with patch.object(planning, 'find_contraction_path', wraps=planning.find_contraction_path) as search:
        selected = calculate(requests, vertex_map=bind, propagator_map=mapping,
            configuration='cfg', usedNe=3, time_axes={0: 1, 1: 0},
            optimize='meson_matrix', search={'deadline_seconds': .1, 'max_states': 1000,
                                            'component_workers': 1},
            memory_limit=1 << 30, execute=False)
    assert search.call_count == 1
    frozen = selected.plan
    with ExitStack() as blocked:
        for name in ('lqcdweave.contraction.correlator.expand_correlator',
                     'lqcdweave.contraction.correlator.find_contraction_path',
                     'lqcdweave.contraction.planning.path_planning.solve_storage',
                     'lqcdweave.contraction.correlator_cfg.build_cfg_plan'):
            blocked.enter_context(patch(name, side_effect=AssertionError('replanned task')))
        for cfg, scale in (('a', 1), ('b', 2)):
            element = SafeReader(elemental, cfg, scale, [])
            prop = SafeReader(propagator, cfg, scale, [])
            actual = selected.run(output=tmp_path / cfg, configuration=cfg,
                vertex_map=element,
                propagator_map={'S^q': prop, r'S^q_\mathrm{local}': prop},
                time_pairs=pairs, physical_times=(3, 27), physical_lt=128)
            assert actual.report['completed'] == (cfg,)
            assert selected.plan is frozen
            for channel in requests:
                restored = load_correlator_output(tmp_path / cfg, (channel,), configuration=cfg)
                np.testing.assert_allclose(restored, reference[channel] * scale**4,
                                           rtol=3e-12, atol=3e-12)
    # Replay exposes no optimizer or task-replacement option, even accidentally.
    with pytest.raises(TypeError):
        selected.run(output=tmp_path / 'bad', optimize='anytime')
    with pytest.raises(ValueError, match='require output'):
        selected.run(configuration='b')
