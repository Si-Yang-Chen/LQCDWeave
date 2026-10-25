"""Partial producer reuse must not leave downstream GEMMs ahead of inputs."""
from contextlib import contextmanager
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, DenseGemmGroup,
    SourcePreparation, SourceSpec, TensorPreparation, TensorSpec,
    dense_batch_memory, describe_coordinates,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, audit_cfg_plan, build_cfg_plan,
    describe_dense_batch, plan_scene_storage,
)
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value


def fixture(backend='numpy'):
    rng = np.random.default_rng(792)
    values = {f'{letter}{i}': rng.normal(size=(2, 2)).astype('complex128')
              for letter in 'abxyz' for i in range(2)}
    values.update(alias_a=values['a0'], alias_b=values['b0'])
    specs = {key: TensorSpec(value.shape, value.dtype) for key, value in values.items()}
    terms = (ContractionTerm(('alias_a', 'alias_b'), 'ij,jk->ik'),) + tuple(
        ContractionTerm((f'a{i}', f'b{j}'), 'ij,jk->ik')
        for i in range(2) for j in range(2)) + tuple(
        ContractionTerm((f'x{i}', f'y{j}'), 'ij,jk->ik')
        for i in range(2) for j in range(2)) + tuple(
        ContractionTerm(('a1', f'b{i}', f'z{j}'), 'ij,jk,kl->il')
        for i in range(2) for j in range(2))
    problem = ContractionProblem(specs, terms,
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=backend)
    profile = CostProfile()
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 1 << 24)
    state = state.contract(0, (0, 1), profile, 1 << 24)
    producer, first, second = [], [], []
    for term in (1, 2):
        producer.append(len(state.actions))
        state = state.contract(term, (0, 1), profile, 1 << 24)
    for term in range(5, 9):
        first.append(len(state.actions))
        state = state.contract(term, (0, 1), profile, 1 << 24)
    for term in (3, 4):
        producer.append(len(state.actions))
        state = state.contract(term, (0, 1), profile, 1 << 24)
    for term in range(9, 13):
        second.append(len(state.actions))
        state = state.contract(term, (0, 1), profile, 1 << 24)
    batches = tuple(describe_dense_batch(state.actions, tuple(indices), state.catalog,
        profile, backend=backend) for indices in (producer, first, second))
    assert all(batches)
    batches = tuple(replace(batch, packing_dtype='complex128') for batch in batches)
    preps = {key: SourcePreparation((), recipe=TensorPreparation('constant',
        constants=(ArrayConstant.from_array(value),))) for key, value in values.items()}
    plan = plan_scene_storage(state, {}, preps, profile=profile,
        memory_limit=1 << 24, host_memory_limit=1 << 24)
    group = DenseGemmGroup(batches[1:], (4, 2, 4), sum(b.seconds for b in batches[1:]))
    plan = replace(plan, kernel_batches=(batches[0], group))
    memory = dense_batch_memory(plan)
    plan = replace(plan, memory=memory, peak_host_bytes=max(p.host_bytes for p in memory),
                   peak_device_bytes=max(p.device_bytes for p in memory))
    points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 0}])
    return plan, preps, specs, points.points, values, terms


class CfgBatchDependencyTests(unittest.TestCase):
    def test_partial_producer_reuse_splits_only_the_unsafe_downstream_group(self):
        for backend in ('numpy', 'cupy'):
            plan, preps, specs, points, values, terms = fixture(backend)
            cfg = build_cfg_plan(plan, None, points, {}, preps, specs)
            selected = cfg.instances[0].kernel_batches
            self.assertEqual(selected, plan.kernel_batches[1].batches)
            self.assertFalse(cfg.instances[1].kernel_batches)
            audit = audit_cfg_plan(cfg)
            self.assertEqual(audit['peak_host_bytes'], cfg.peak_host_bytes)
            self.assertEqual(audit['peak_device_bytes'], cfg.peak_device_bytes)
            debug = build_cfg_plan(plan, None, points, {}, preps, specs, debug=True)
            self.assertEqual(replace(cfg, statistics={}), replace(debug, statistics={}))
            if backend != 'numpy':
                continue
            restored = decode_plan_value(encode_plan_value(cfg))
            @contextmanager
            def factory(configuration):
                yield restore_input_loader(restored.requests, {}, restored.preparations,
                                           restored.specs, configuration=configuration)
            expected = [np.linalg.multi_dot([values[k] for k in term.inputs])
                        for term in terms]
            results = []
            with patch('lqcdweave.contraction.planning.api.find_contraction_path',
                       side_effect=AssertionError('fixed replay must not search')):
                execute_cfg_plan(restored, ['cfg'], factory,
                    lambda cfg, index, outputs: results.append(outputs)
                        if index is not None else None)
            self.assertEqual(len(results), 2)
            for outputs in results:
                np.testing.assert_allclose(outputs, expected, rtol=2e-12, atol=2e-12)

    def test_audit_rejects_an_early_group_even_if_logical_actions_are_ordered(self):
        plan, preps, specs, points, _, _ = fixture()
        cfg = build_cfg_plan(plan, None, points, {}, preps, specs)
        bad = replace(cfg.instances[0], kernel_batches=(plan.kernel_batches[1],))
        with self.assertRaisesRegex(ValueError, 'batch input precedes its producer'):
            audit_cfg_plan(replace(cfg, instances=(bad, *cfg.instances[1:])))
