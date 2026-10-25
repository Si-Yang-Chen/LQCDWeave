"""Shared dense owners survive reuse while other coordinate owners are freed."""
from dataclasses import replace
from math import prod
import unittest

import numpy as np

from correlator.planning import test_cfg_plan as fixtures
from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, SourcePreparation, SourceReadRequest, StorageBlock, TensorAxisMetadata,
    TensorPreparation, dense_batch_memory, describe_coordinates,
)
from lqcdweave.contraction.planning.api import (
    audit_cfg_plan, build_cfg_plan, plan_scene_storage,
)


class CfgBatchOwnerCountTests(unittest.TestCase):
    def test_chained_views_with_different_consumers_keep_owner_through_last_coordinate(self):
        from correlator.planning.test_gemm_groups import fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches

        selected, profile = fixture(chained=True, count=2)
        selected, _ = lower_independent_dense_batches(selected, profile=profile, target_runtime='cuda')
        owner = selected.kernel_batches[0]
        last_uses = [max(index for index, action in enumerate(selected.state.actions)
                         if result.tensor in action.operands) for result in owner.results]
        self.assertGreater(len(set(last_uses)), 1)
        specs = selected.state.problem.specs
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.full(spec.shape, index+1, dtype=spec.dtype)),)))
            for index, (key, spec) in enumerate(specs.items())}
        points, _ = describe_coordinates(time_pairs=[{0:i} for i in (0, 1, 0, 2)])
        for window in (1, 3):
            with self.subTest(reuse_window=window):
                from lqcdweave.contraction.planning import cfg_plan
                bound = cfg_plan._bind_cfg_template(selected, None, {}, preps, specs,
                                                    output_mode='dataset')
                cfg = bound(points.points, reuse_window=window)
                original = cfg_plan._bind_cfg_template(selected, None, {}, preps, specs,
                                                       output_mode='dataset')
                original.keywords['fine_operand_uses'] = True
                expected = original(points.points, reuse_window=window)
                self.assertEqual(replace(cfg, statistics={}), replace(expected, statistics={}))
                positions = {key:i for i, key in enumerate(cfg.templates[0].keys)}
                ids = cfg.instances[0].node_ids.array()
                owner_ids = {int(ids[positions[result.tensor]]) for result in owner.results}
                release_at = {}
                for index, instance in enumerate(cfg.instances):
                    released = list(instance.releases)
                    if instance.tensor_releases is not None:
                        released.extend(instance.tensor_releases.array()[:, 1].tolist())
                    for gid in released:
                        if gid in owner_ids:
                            self.assertNotIn(gid, release_at)
                            release_at[gid] = index
                self.assertEqual(set(release_at), owner_ids)
                self.assertEqual(set(release_at.values()), {window-1})
                audited = audit_cfg_plan(cfg)
                self.assertEqual(audited['peak_host_bytes'], cfg.peak_host_bytes)
                self.assertEqual(audited['peak_device_bytes'], cfg.peak_device_bytes)

    def test_unproven_output_redefinition_keeps_original_owner_path(self):
        from lqcdweave.contraction.planning import cfg_plan

        class ExtendedAction(Action):
            pass

        selected, preps, specs = fixtures.CfgPlanTests.dense_fixture()
        product = next(action for action in selected.state.actions if action.kind == 'contract')
        selected = replace(selected, state=replace(selected.state,
            actions=(*selected.state.actions, ExtendedAction(**vars(product)))))
        points, _ = describe_coordinates(time_pairs=[{0:i} for i in (0, 1, 0)])
        bound = cfg_plan._bind_cfg_template(selected, None, {}, preps, specs, output_mode='dataset')
        self.assertTrue(bound.keywords['fine_operand_uses'])
        cfg = bound(points.points, reuse_window=3)
        self.assertEqual([len(instance.kernel_batches) for instance in cfg.instances], [1, 0, 0])
        positions = {key:i for i, key in enumerate(cfg.templates[0].keys)}
        ids = cfg.instances[0].node_ids.array()
        owner_ids = {int(ids[positions[result.tensor]]) for result in selected.kernel_batches[0].results}
        self.assertTrue(owner_ids.isdisjoint(cfg.instances[0].releases))
        self.assertTrue(owner_ids <= set(cfg.instances[-1].releases))
        audit_cfg_plan(cfg)

    def test_overlapping_owners_and_repeated_coordinate_match_independent_audit(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                selected, _, specs = fixtures.CfgPlanTests.dense_fixture(backend)
                blocks, preparations, requests = {}, {}, {}
                for key, spec in specs.items():
                    amount = prod(spec.shape) * spec.dtype.itemsize
                    blocks[key] = StorageBlock(key, 'full', amount,
                        host_bytes=amount if backend == 'numpy' else 0,
                        device_bytes=amount if backend == 'cupy' else 0,
                        staging_host_bytes=10*amount,
                        staging_device_bytes=10*amount if backend == 'cupy' else 0)
                    preparations[key] = SourcePreparation((key,),
                        recipe=TensorPreparation('identity', (key,)))
                    requests[key] = SourceReadRequest(key, 'elemental', (0,), (),
                        TensorAxisMetadata(('row', 'column'), spec.shape, spec.dtype,
                                           key, 'full'),
                        coordinate_arguments=((0, '0', 0, None),))
                physical = plan_scene_storage(selected.state, blocks, preparations,
                    profile=selected.cost_profile, memory_limit=10**8, host_memory_limit=10**8)
                physical = replace(physical, kernel_batches=selected.kernel_batches)
                memory = dense_batch_memory(physical)
                physical = replace(physical, memory=memory,
                    peak_host_bytes=max(point.host_bytes for point in memory),
                    peak_device_bytes=max(point.device_bytes for point in memory))
                dense_batch_memory(physical, check=True)
                points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in (0, 1, 0, 2)])
                cfg = build_cfg_plan(physical, None, points.points, requests, preparations,
                                     specs, output_mode='dataset', debug=True)
                ordinary = build_cfg_plan(physical, None, points.points, requests, preparations,
                                          specs, output_mode='dataset')
                self.assertEqual(replace(ordinary, statistics={}), replace(cfg, statistics={}))
                self.assertEqual([len(i.kernel_batches) for i in cfg.instances], [1, 1, 0, 1])
                self.assertTrue(all(cfg.instances[2].reused.array()))
                first_outputs = tuple(int(cfg.instances[0].node_ids.array()[
                    cfg.templates[0].keys.index(key)]) for key in physical.state.results)
                self.assertTrue(set(first_outputs).isdisjoint(cfg.instances[0].releases))
                self.assertTrue(set(first_outputs) <= set(cfg.instances[2].releases))
                audited = audit_cfg_plan(cfg)
                self.assertEqual(audited['peak_host_bytes'], cfg.peak_host_bytes)
                self.assertEqual(audited['peak_device_bytes'], cfg.peak_device_bytes)


if __name__ == '__main__':
    unittest.main()
