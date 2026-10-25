"""Reused results do not keep their unexecuted inputs resident."""
from dataclasses import replace

import pytest

from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.planning import cfg_plan
from correlator.planning import test_cfg_operand_lifetimes as lifetime_fixtures


@pytest.mark.parametrize('debug', (False, True))
def test_reused_result_survives_after_its_original_inputs_are_released(debug):
    selected, preps, specs, _ = lifetime_fixtures.CfgOperandLifetimeTests.fixture()
    points, _ = describe_coordinates()
    plan = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preps, specs,
                                   output_mode='dataset', debug=debug)
    template, = plan.templates
    instance, = plan.instances
    bindings = dict(zip(template.keys, map(int, instance.node_ids.array()), strict=True))
    source = bindings['a']
    reuse = instance.reused.array()
    paid = [index for index, action in enumerate(template.plan.state.actions)
            if action.kind != 'load' and not reuse[index]
            and source in (bindings[key] for key in action.operands)]
    repeated = [index for index, action in enumerate(template.plan.state.actions)
                if action.kind != 'load' and reuse[index]]
    assert paid and repeated and max(paid) < max(repeated)
    released = [int(index) for index, gid in instance.tensor_releases.array()
                if int(gid) == source]
    assert released == [max(paid)]
    cfg_plan.audit_cfg_plan(plan)
    other = cfg_plan.build_cfg_plan(selected, None, points.points, {}, preps, specs,
                                    output_mode='dataset', debug=not debug)
    assert replace(plan, statistics={}) == replace(other, statistics={})
