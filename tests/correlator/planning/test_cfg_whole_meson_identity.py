"""Repeated whole-meson values have one cfg owner across proof occurrences."""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    AxisPattern, SourcePreparation, SourceReadRequest, StorageBlock,
    TensorAxisMetadata, TensorPreparation, VertexKernelDescription, describe_coordinates,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.meson_core import plan_meson_whole
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan
from correlator.execution.test_meson_whole_core import case


def fixture(kind, dtype):
    problem, values = case(kind, dtype, False)
    problem = replace(problem, terms=problem.terms * 3)
    inputs = problem.terms[0].kernel.inputs
    metadata = {key: TensorAxisMetadata(tuple(f'axis{i}' for i in range(value.ndim)),
        value.shape, value.dtype, 'data:' + key, 'full')
        for key, value in zip(inputs, values, strict=True)}
    blocks = {key: StorageBlock('data:' + key, 'full', value.nbytes, host_bytes=value.nbytes)
              for key, value in zip(inputs, values, strict=True)}
    preparations = {key: SourcePreparation((key,), dtype=dtype,
        recipe=TensorPreparation('identity', inputs=(key,))) for key in inputs}
    requests = {key: SourceReadRequest(key, 'elemental', (0,), (), metadata[key],
        coordinate_arguments=((0, '0', 0, None),)) for key in inputs}
    selected = plan_meson_whole(problem, blocks, preparations, profile=CostProfile(),
        memory_limit=1 << 28, host_memory_limit=1 << 28)
    return selected, preparations, requests, metadata, dict(zip(inputs, values, strict=True))


@pytest.mark.parametrize('kind', ('connected', 'loop'))
def test_native_whole_identity_uses_value_semantics_and_ordered_actual_inputs(kind):
    selected, preparations, _, _, _ = fixture(kind, 'complex64')
    action = next(action for action in selected.state.actions if action.kind == 'contract')
    occurrence = replace(action, term_index=91, positions=tuple(reversed(action.positions)))
    operands = tuple(range(len(action.operands)))
    assert cfg_plan._computation_identity(action, operands) == cfg_plan._computation_identity(occurrence, operands)
    assert cfg_plan._computation_identity(action, operands) != cfg_plan._computation_identity(action, operands[::-1])
    axes = action.core.private_equation
    renamed = AxisPattern(tuple(tuple(axis + 100 for axis in labels) for labels in axes.inputs),
                          tuple(axis + 100 for axis in axes.output))
    same = replace(occurrence, core=replace(action.core, private_equation=renamed))
    variants = (
        same,
        replace(action, core=replace(action.core, representation='dense')),
        replace(action, core=replace(action.core, lane_axis=99)),
        replace(action, core=replace(action.core,
            private_equation=AxisPattern(axes.inputs, axes.output[::-1] if len(axes.output) > 1 else ()))),
        replace(action, core=replace(action.core,
            kernel_description=VertexKernelDescription('meson', 'whole-matrix-v2'))),
        replace(action, kernel_path=((0, 1),)),
        replace(action, wide_accumulation=True),
    )
    base = cfg_plan._computation_identity(action, operands)
    assert cfg_plan._computation_identity(variants[0], operands) == base
    for changed in variants[1:]:
        assert cfg_plan._computation_identity(changed, operands) != base
    actions = (action, occurrence, same, *variants[1:])
    identities = cfg_plan._bind_computation_identities(actions, debug=False)
    assert tuple(bind(operands) for bind in identities) == tuple(
        cfg_plan._computation_identity(item, operands) for item in actions)
    reuses = cfg_plan._bind_local_computation_reuses(actions, identities,
        tuple(cfg_plan._bind_operands(item.operands) for item in actions), preparations, debug=False)
    assert reuses[:3] == (False, True, True)
    assert not any(reuses[3:])


@pytest.mark.parametrize('kind', ('connected', 'loop'))
@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('debug', (False, True))
def test_repeated_whole_cfg_archive_has_no_orphan_releases_and_replays_current_data(tmp_path, kind, dtype, debug):
    selected, preparations, requests, metadata, values = fixture(kind, dtype)
    points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}, {0: 0}])
    with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
        plan = cfg_plan.build_cfg_plan(selected, None, points.points, requests,
            preparations, selected.state.problem.specs, debug=debug)
    bound = {int(gid) for instance in plan.instances for gid in instance.node_ids.array()}
    released = {int(gid) for instance in plan.instances if instance.tensor_releases is not None
                for _, gid in instance.tensor_releases.array()}
    assert released <= bound, f'orphaned whole-value releases: {released - bound}'
    cfg_plan.audit_cfg_plan(plan)
    path = tmp_path / 'whole-cfg.json.gz'
    save_cfg_plan(path, plan)
    with patch.object(cfg_plan, 'build_cfg_plan', side_effect=AssertionError('archive replanning')):
        restored = load_cfg_plan(path)
    collected = {}

    class Reader:
        def __init__(self, key, configuration):
            self.key, self.cfg = key, configuration

        def describe_block(self, time):
            record = metadata[self.key]
            return replace(record, data_identity=record.data_identity + ':' + self.cfg)

        def block(self, time):
            return (values[self.key] * ((1. if self.cfg == 'a' else 1.3) + .1 * time)).astype(dtype)

    @contextmanager
    def factory(configuration):
        yield restore_input_loader(restored.requests,
            {request.reader_role: Reader(request.reader_role, configuration)
             for request in restored.requests.values()},
            restored.preparations, restored.specs, configuration=configuration)

    execute_cfg_plan(restored, ('a', 'b'), factory,
        lambda configuration, index, result: collected.update({(configuration, index): result}),
        debug=debug, prefetch_host_bytes=0)
    kernel = selected.state.problem.terms[0].kernel
    equation = kernel.private_equation
    for configuration in ('a', 'b'):
        for index, time in enumerate((0, 1, 0)):
            current = tuple(Reader(key, configuration).block(time).astype('complex128') for key in kernel.inputs)
            arguments = tuple(item for value, axes in zip(current, equation.inputs, strict=True)
                              for item in (value, axes)) + (equation.output,)
            expected = np.einsum(*arguments, optimize=True)
            assert len(collected[configuration, index]) == 3
            for actual in collected[configuration, index]:
                np.testing.assert_allclose(actual, expected,
                    rtol=5e-5 if dtype == 'complex64' else 5e-12,
                    atol=5e-5 if dtype == 'complex64' else 5e-12)
