"""Cfg value sharing preserves the layouts promised to prepared batch sides."""
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.planning.api import build_cfg_plan, solve_storage
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.contraction.execution.operations.dense_batch import bind_dense_batch, contract_dense_batch


class CfgProducerLayoutTests(unittest.TestCase):
    def fixture(self):
        specs = {f'v{i}': TensorSpec((2, 3, 5), 'complex128') for i in range(2)}
        specs.update({f'c{i}': TensorSpec((4, 2), 'complex128') for i in range(2)})
        specs.update({f'b{i}': TensorSpec((5, 3+i), 'complex128') for i in range(4)})
        # Independent names intentionally bind the same physical inputs.
        terms = (ContractionTerm(('v0', 'c0'), 'klm,nk->nlm'),) + tuple(
            ContractionTerm(('v1', 'c1', f'b{i}'), 'klm,nk,mx->nlx') for i in range(4))
        problem = ContractionProblem(specs, terms,
            {k: SourceSpec(k, 0, read_bytes=0) for k in specs}, backend='cupy')
        selected = solve_storage(problem, {}, {k: SourcePreparation(()) for k in specs},
            optimizer='anytime', kernel_batching=True, memory_limit=10**8,
            deadline_seconds=.1, max_states=1000, root_width=1, branch_width=1,
            component_workers=1).plan
        values = {k: (np.arange(np.prod(s.shape)).reshape(s.shape)+1j).astype(s.dtype)
                  for k, s in specs.items()}
        preparations = {k: SourcePreparation((), recipe=TensorPreparation('constant',
            constants=(ArrayConstant.from_array(v),))) for k, v in values.items()}
        points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 0}])
        cfg = build_cfg_plan(selected, None, points.points, {}, preparations, specs)
        return cfg, values

    def test_layout_identity_keeps_matching_reuse_and_separates_permutations(self):
        from dataclasses import replace
        from lqcdweave.contracts.correlator.api import Action
        first = Action('contract', 'first', ('left', 'right'), 'klm,nk->nlm')
        actions = (first, replace(first, tensor='same'), replace(first, tensor='other'),
                   replace(first, tensor='ordinary'))
        for debug in (False, True):
            binders = cfg_plan._bind_computation_identities(actions, debug=debug,
                output_layouts={'first': (0, 1, 2), 'same': (0, 1, 2), 'other': (1, 2, 0)})
            values = [bind((3, 7)) for bind in binders]
            self.assertEqual(values[0], values[1])
            self.assertEqual(len(set(values)), 3)

    def test_saved_cfg_preserves_layout_owners_without_replanning(self):
        from unittest.mock import patch
        from lqcdweave.storage.contraction_plans.api import load_cfg_plan, save_cfg_plan
        cfg, _ = self.fixture()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'selected.json.gz'
            save_cfg_plan(path, cfg)
            with patch.object(cfg_plan, 'build_cfg_plan', side_effect=AssertionError('replanning')):
                restored = load_cfg_plan(path)
        self.assertEqual(restored, cfg)

    def test_disabled_batch_variant_keeps_the_template_producer_layout(self):
        from dataclasses import replace
        from lqcdweave.contraction.execution.cfg_runtime import _bind_cfg_tables
        cfg, values = self.fixture()
        template = cfg.templates[0]
        batch, = cfg.instances[0].kernel_batches
        prepared = batch.left[0].tensor
        first = replace(cfg.instances[0], kernel_batches=())
        variant = replace(cfg, instances=(first, *cfg.instances[1:]))
        backend = SimpleNamespace(__name__='cupy', array=np.array, empty=np.empty,
            transpose=np.transpose, reshape=np.reshape, concatenate=np.concatenate,
            matmul=np.matmul, einsum=np.einsum)
        _, tables = _bind_cfg_tables(variant, backend)
        table = next(iter(tables.values()))
        self.assertEqual(table.layouts[prepared], batch.left[0].permutation)
        index = next(i for i, action in enumerate(template.plan.state.actions)
                     if action.tensor == prepared and not first.reused.array()[i])
        action = template.plan.state.actions[index]
        actual, _ = table.kernels[index](tuple(values[k] for k in action.operands))
        self.assertTrue(actual.transpose(batch.left[0].permutation).flags.c_contiguous)
        np.testing.assert_allclose(actual,
            np.einsum('klm,nk->nlm', values['v1'], values['c1']), rtol=2e-12, atol=2e-12)

    def test_equal_values_with_different_layouts_keep_distinct_owners(self):
        cfg, values = self.fixture()
        template, instance = cfg.templates[0], cfg.instances[0]
        batch, = instance.kernel_batches
        self.assertEqual(batch.prepared_sides, (True, False))
        prepared = batch.left[0].tensor
        ordinary = template.plan.state.results[0]
        binding = dict(zip(template.keys, map(int, instance.node_ids.array()), strict=True))
        self.assertNotEqual(binding[ordinary], binding[prepared])
        # Matching layout and values remain reusable across coordinates.
        self.assertTrue(all(cfg.instances[1].reused.array()))
        self.assertEqual(sum(len(i.kernel_batches) for i in cfg.instances), 1)

        backend = SimpleNamespace(__name__='cupy', array=np.array, empty=np.empty,
            transpose=np.transpose, reshape=np.reshape, concatenate=np.concatenate,
            matmul=np.matmul, einsum=np.einsum)
        arrays = _NodeView(binding, {binding[k]: v for k, v in values.items()})
        members = set(batch.action_indices)
        layout = {prepared: batch.left[0].permutation}
        for index, action in enumerate(template.plan.state.actions):
            if action.kind != 'contract' or index in members or instance.reused.array()[index]:
                continue
            value, _ = bind_dense_action(action, template.plan.state.catalog, backend,
                layout.get(action.tensor))(tuple(arrays[k] for k in action.operands))
            arrays[action.tensor] = value
        self.assertFalse(arrays[ordinary].flags.c_contiguous)
        self.assertTrue(arrays[prepared].flags.c_contiguous)
        kernel = bind_dense_batch(batch, template.plan.state.catalog, backend)
        current = arrays[prepared]
        for invalid in (arrays[ordinary], current.astype('complex64')):
            arrays[prepared] = invalid
            with self.assertRaisesRegex(ValueError, 'tensor=.*expected_dtype=complex128'):
                contract_dense_batch(batch, arrays, backend=backend, bound_kernel=kernel)
        arrays[prepared] = current
        contract_dense_batch(batch, arrays, backend=backend, bound_kernel=kernel)
        expected = [np.einsum('klm,nk->nlm', values['v0'], values['c0'])] + [
            np.einsum('klm,nk,mx->nlx', values['v1'], values['c1'], values[f'b{i}'])
            for i in range(4)]
        for key, reference in zip(template.plan.state.results, expected, strict=True):
            actual = arrays[key]
            if hasattr(actual, 'array'):
                actual = actual.array()
            np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
