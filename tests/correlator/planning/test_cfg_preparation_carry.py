"""P carries actual terminal owners; X reads/prepares less and replays fixed tables."""
from array import array
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from correlator.planning.test_cfg_plan import selected_template
from correlator.planning import test_cfg_plan as plan_fixtures
from lqcdweave.contracts.correlator.api import ArrayConstant, describe_coordinates
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.expansion.api import describe_input_reads, restore_input_loader
from lqcdweave.contraction.planning import cfg_plan, cfg_preparation_carry as carry
from lqcdweave.contraction.planning.api import CostProfile, solve_storage


PAIRS = [{0: i, 1: j} for j in range(2) for i in range(2)]


def complete_fixture(*, monomial=False):
    binding, elemental, propagator = fixture()
    selected = calculate(task(structured=True, flavor='I0'), vertex_map=binding,
        propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
        configuration='cfg', time_axes={0: 1, 1: 0}, execute=False,
        optimize='meson_matrix' if monomial else 'none',
        search={'deadline_seconds': 2, 'max_states': 10000, 'root_width': 1,
                'branch_width': 1, 'component_workers': 1} if monomial else None,
        memory_limit=1<<28, debug=True)
    points, _ = describe_coordinates(time_pairs=PAIRS)
    with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 3*len(selected.plan.state.actions)):
        with patch.object(carry, 'retain_cfg_preparations', side_effect=lambda p, costs: p):
            reference = complete_calculation(selected, coordinates=points)
        actual = complete_calculation(selected, coordinates=points)
    return reference, actual, elemental, propagator


def replay(plan, elemental, propagator):
    records, outputs = [], {}

    class Reader:
        background_read_safe = False
        def __init__(self, original, cfg, family):
            self.original, self.cfg, self.family = original, cfg, family
        def __getattr__(self, name):
            return getattr(self.original, name)
        def describe_block(self, *args, **options):
            metadata = self.original.describe_block(*args, **options)
            return replace(metadata, data_identity=metadata.data_identity+':'+self.cfg)
        def block(self, *args, **options):
            records.append((self.cfg, self.family, args))
            return self.original.block(*args, **options)*(1 if self.cfg == 'a' else 2)

    @contextmanager
    def factory(cfg):
        readers = {request.reader_role: Reader(
            elemental if request.family == 'elemental' else propagator, cfg, request.family)
            for primary in plan.requests.values() for request in (primary, *primary.aliases)}
        yield restore_input_loader(plan.requests, readers, plan.preparations, plan.specs,
                                   configuration=cfg)
    def consume(cfg, index, values):
        if index is not None:
            outputs[cfg, index] = values
    report = execute_cfg_plan(plan, iter(('a', 'b')), factory, consume,
                              prefetch_host_bytes=0, debug=True)
    return report, outputs, records


class CfgPreparationCarryTests(unittest.TestCase):
    def test_whole_vertex_without_terminal_dense_owners_needs_no_carry(self):
        reference, actual, elemental, propagator = complete_fixture()
        # Whole vertices consume their bound spatial inputs inside the kernel;
        # they do not prepare the dense spin-spatial terminal owners this pass
        # retains. Exercise that route separately from the eligible matrix case.
        self.assertFalse(carry._preparation_groups(actual.plan))
        self.assertEqual(actual.plan.instances, reference.plan.instances)
        self.assertEqual(actual.plan.statistics.get('cross_window_reused_preparations', 0), 0)
        _, expected, before_reads = replay(reference.plan, elemental, propagator)
        _, values, reads = replay(actual.plan, elemental, propagator)
        self.assertEqual(len(reads), len(before_reads))
        for key in values:
            for value, wanted in zip(values[key], expected[key], strict=True):
                np.testing.assert_array_equal(value, wanted)

    def test_compressed_spin_carries_owned_roots_and_excludes_identity_gathers(self):
        reference, actual, elemental, propagator = complete_fixture(monomial=True)
        plan = actual.plan
        self.assertTrue(any(prep.recipe.spin_columns or prep.recipe.spatial_only
                            for prep in plan.preparations.values()))
        # Spatial-only spin recipes no longer need the older spin gathers;
        # every borrowed identity owner (including any gather) stays excluded.
        borrowed = {int(key) for key, prep in plan.preparations.items()
                    if prep.recipe.kind == 'identity'}
        self.assertTrue(borrowed)
        self.assertTrue(borrowed.isdisjoint(gid for group in carry._preparation_groups(plan) for gid in group))
        self.assertGreater(plan.statistics['cross_window_reused_preparations'], 0)
        _, expected, before_reads = replay(reference.plan, elemental, propagator)
        _, values, reads = replay(plan, elemental, propagator)
        self.assertLess(len(reads), len(before_reads))
        for key in values:
            for value, wanted in zip(values[key], expected[key], strict=True):
                np.testing.assert_array_equal(value, wanted)

    def test_single_window_and_no_eligible_recipe_skip_full_cfg_pass(self):
        selected, preparations, specs = plan_fixtures.CfgPlanTests.dense_fixture()
        coordinates, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}])
        with patch.object(carry, 'retain_cfg_preparations', side_effect=AssertionError('single window')):
            cfg_plan.build_cfg_plan(selected, None, coordinates.points, {}, preparations, specs)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1), \
                patch.object(carry, '_preparation_groups', side_effect=AssertionError('no eligible owner')):
            cfg_plan.build_cfg_plan(selected, None, coordinates.points, {}, preparations, specs)

    def test_carried_terminal_recipes_own_results_independently_of_raw_and_members(self):
        reference, _, elemental, propagator = complete_fixture(monomial=True)
        plan = reference.plan
        readers = {request.reader_role: propagator if request.family == 'propagator' else elemental
                   for primary in plan.requests.values() for request in (primary, *primary.aliases)}
        physical = restore_input_loader(plan.requests, readers, plan.preparations, plan.specs,
                                        configuration='cfg')
        loader = bind_prepared_loader(physical, plan.preparations, plan.specs)
        seen = []
        for group in carry._preparation_groups(plan):
            for gid in group:
                key = str(gid)
                prep = plan.preparations[key]
                raw = {block: physical.read_block(block, plan.blocks[block]) for block in prep.blocks}
                value = loader.materialize(key, None, raw)
                self.assertTrue(loader.materializes_owned(key))
                self.assertTrue(value.flags.owndata)
                self.assertEqual(value.nbytes, np.prod(plan.specs[key].shape)*plan.specs[key].dtype.itemsize)
                self.assertTrue(all(not np.shares_memory(value, block) for block in raw.values()))
                self.assertTrue(all(not np.shares_memory(value, previous) for previous in seen))
                seen.append(value)
        self.assertTrue(seen)

    def test_three_window_chains_retain_until_the_final_real_consumer(self):
        expanded, selected, elemental, propagator = selected_template()
        coordinates, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}]*3)
        arguments = (selected, expanded.output_recipe, coordinates.points,
            describe_input_reads(expanded.inputs), expanded.inputs.loader.preparations,
            expanded.inputs.loader.specs)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
            with patch.object(carry, 'retain_cfg_preparations', side_effect=lambda p, costs: p):
                reference = cfg_plan.build_cfg_plan(*arguments)
            actual = cfg_plan.build_cfg_plan(*arguments, debug=True)
        before, expected, before_reads = replay(reference, elemental, propagator)
        after, values, reads = replay(actual, elemental, propagator)
        self.assertGreater(actual.statistics['cross_window_saved_reads'], 0)
        self.assertLess(len(reads), len(before_reads))
        self.assertLess(after['source_materialization_calls'], before['source_materialization_calls'])
        for key in values:
            for value, wanted in zip(values[key], expected[key], strict=True):
                np.testing.assert_array_equal(value, wanted)

    def test_offline_device_owner_rounding_and_dual_budget_are_audited(self):
        expanded, _, *_ = selected_template()
        inputs = expanded.inputs
        blocks = {key: replace(block, device_bytes=block.host_bytes,
                              transfer_bytes=block.host_bytes) for key, block in inputs.blocks.items()}
        preparations = {key: replace(prep, workspace_device_bytes=prep.workspace_host_bytes,
                                     workspace_host_bytes=0) for key, prep in inputs.loader.preparations.items()}
        selected = solve_storage(replace(inputs.problem, backend='cupy'), blocks,
            {key: preparations[key] for key in inputs.preparations},
            profile=CostProfile(), memory_limit=1<<28, optimizer='anytime', deadline_seconds=2,
            max_states=10000, root_width=1, branch_width=1, component_workers=1).plan
        selected = replace(selected, packed_value_cache_device_bytes=0)
        coordinates, _ = describe_coordinates(time_pairs=PAIRS)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 3*len(selected.state.actions)):
            plan = cfg_plan.build_cfg_plan(selected, expanded.output_recipe, coordinates.points,
                describe_input_reads(inputs), preparations, inputs.loader.specs,
                target_summary={'host_memory_limit': 1<<20, 'device_memory_limit': 1<<20},
                debug=True)
        self.assertGreater(plan.statistics['cross_window_carry_bytes'], 0)
        self.assertEqual(plan.statistics['cross_window_carry_bytes'] % 512, 0)
        self.assertLessEqual(plan.peak_device_bytes+plan.prefetch_device_bytes, 1<<20)
        self.assertLessEqual(plan.peak_host_bytes+plan.prefetch_host_bytes, 1<<20)
        cfg_plan.audit_cfg_plan(plan)

    def test_public_complete_keeps_compute_and_reduces_actual_reads_and_prepares(self):
        reference, actual, elemental, propagator = complete_fixture(monomial=True)
        old, new = reference.plan, actual.plan
        self.assertEqual(new.statistics['reuse_window'], old.statistics['reuse_window'])
        self.assertEqual(new.templates, old.templates)
        self.assertEqual(new.requests, old.requests)  # all original input validation remains
        self.assertEqual(new.preparations, old.preparations)
        self.assertGreater(new.statistics['cross_window_reused_preparations'], 0)
        self.assertGreater(new.statistics['cross_window_saved_reads'], 0)
        for before, after in zip(old.instances, new.instances, strict=True):
            template = old.templates[before.template]
            positions = {key: i for i, key in enumerate(template.keys)}
            compute = [positions[a.tensor] for a in template.plan.state.actions if a.kind == 'contract']
            np.testing.assert_array_equal(before.node_ids.array()[compute], after.node_ids.array()[compute])
            self.assertEqual(before.kernel_batches, after.kernel_batches)
        expected_report, expected, expected_reads = replay(old, elemental, propagator)
        report, values, reads = replay(new, elemental, propagator)
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertEqual(len(expected_reads)-len(reads), 2*new.statistics['cross_window_saved_reads'])
        self.assertEqual(expected_report['source_materialization_calls']-report['source_materialization_calls'],
                         2*new.statistics['cross_window_reused_preparations'])
        self.assertEqual(values.keys(), expected.keys())
        for key in values:
            for value, wanted in zip(values[key], expected[key], strict=True):
                np.testing.assert_array_equal(value, wanted)
        self.assertLess(new.statistics['preparation_seconds'], old.statistics['preparation_seconds'])
        cfg_plan.audit_cfg_plan(new)

    def test_budget_rejects_or_partially_admits_carry_and_shrinks_prefetch(self):
        reference, actual, *_ = complete_fixture(monomial=True)
        plan = reference.plan
        costs = array('d', [1.0]*sum(
            action.kind == 'load' and not flag for instance in plan.instances
            for action, flag in zip(plan.templates[instance.template].plan.state.actions,
                                    instance.reused.array(), strict=True)))
        def limited(spare):
            stats = dict(plan.statistics, host_memory_cap=plan.peak_host_bytes+spare,
                         prefetch_host_bytes=spare)
            return replace(plan, statistics=stats, prefetch_host_bytes=spare)
        constrained = limited(0)
        self.assertIs(carry.retain_cfg_preparations(constrained, costs), constrained)
        root_bytes = min(np.prod(plan.specs[str(gid)].shape)*plan.specs[str(gid)].dtype.itemsize
                         for group in carry._preparation_groups(plan) for gid in group)
        partial = carry.retain_cfg_preparations(limited(int(root_bytes)), costs)
        self.assertGreater(partial.statistics['cross_window_reused_preparations'], 0)
        self.assertLess(partial.statistics['cross_window_reused_preparations'],
                        actual.plan.statistics['cross_window_reused_preparations'])
        self.assertLessEqual(partial.peak_host_bytes, partial.statistics['host_memory_cap'])
        self.assertEqual(partial.prefetch_host_bytes+partial.peak_host_bytes,
                         partial.statistics['host_memory_cap'])
        cfg_plan.audit_cfg_plan(partial)

    def test_same_shape_is_not_read_or_recursive_recipe_identity(self):
        expanded, selected, *_ = selected_template()
        coordinates, _ = describe_coordinates(time_pairs=PAIRS)
        plan = cfg_plan._bind_cfg_template(selected, expanded.output_recipe,
            describe_input_reads(expanded.inputs), expanded.inputs.loader.preparations,
            expanded.inputs.loader.specs)(coordinates.points, reuse_window=3)
        first, second = carry._preparation_groups(plan)[0]
        first, second = str(first), str(second)
        raw = plan.preparations[second].blocks[0]
        request = plan.requests[raw]
        variants = (
            replace(request, metadata=replace(request.metadata, data_identity='different-file')),
            replace(request, metadata=replace(request.metadata, selection_identity='other-selection')),
            replace(request, metadata=replace(request.metadata,
                axis_bindings={'mode_left': ('basis', (2, 0, 1))})),
            replace(request, conventions=(('time_selection_identity', ('ordered', (7, 0))),)),
            replace(request, arguments=(*request.arguments[:2], 0, *request.arguments[3:])),
            replace(request, aliases=()),
        )
        def still_shared(changed):
            return any(int(first) in group and int(second) in group
                       for group in carry._preparation_groups(changed))
        for changed in variants:
            with self.subTest(request=changed):
                self.assertFalse(still_shared(replace(plan, requests=dict(plan.requests, **{raw: changed}))))
        child = plan.preparations[second].recipe.inputs[0]
        prep = plan.preparations[child]
        for recipe in (replace(prep.recipe, conjugate=not prep.recipe.conjugate),
                       replace(prep.recipe, permutation=(1, 0)),
                       replace(prep.recipe, compute_dtype='complex64'),
                       replace(prep.recipe, spin_columns=tuple(map(int,
                           np.nonzero(prep.recipe.constants[0].array())[1]))),
                       replace(prep.recipe, constants=(ArrayConstant.from_array(-prep.recipe.constants[0].array()),))):
            changed = dict(plan.preparations, **{child: replace(prep, recipe=recipe)})
            with self.subTest(recipe=recipe):
                self.assertFalse(still_shared(replace(plan, preparations=changed)))

    def test_complete_archive_public_replay_for_two_cfgs_never_plans_or_reloads_skipped_sources(self):
        reference, complete, elemental, propagator = complete_fixture(monomial=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'carried.plan.gz'
            reference_path = Path(directory)/'original.plan.gz'
            save_calculation_plan(path, complete)
            save_calculation_plan(reference_path, reference)
            @contextmanager
            def factory(cfg):
                from correlator.execution.test_cfg_runtime import SafeReader
                yield {request.reader_role: SafeReader(
                    propagator if request.family == 'propagator' else elemental, cfg,
                    1 if cfg == 'a' else 2, [])
                    for primary in complete.plan.requests.values() for request in (primary, *primary.aliases)}
            with ExitStack() as blocked:
                for name in ('lqcdweave.contraction.correlator.expand_correlator',
                             'lqcdweave.contraction.correlator.find_contraction_path',
                             'lqcdweave.contraction.planning.cfg_plan._bind_cfg_coordinates',
                             'lqcdweave.contraction.planning.cfg_preparation_carry.retain_cfg_preparations'):
                    blocked.enter_context(patch(name, side_effect=AssertionError(name)))
                loaded = calculate(plan_archive=path, configuration=iter(('a', 'b')), data=factory,
                                   debug=True)
                values = dict(loaded.value)
                expected = dict(calculate(plan_archive=reference_path,
                    configuration=iter(('a', 'b')), data=factory, debug=True).value)
            for cfg in ('a', 'b'):
                np.testing.assert_array_equal(values[cfg], expected[cfg])


if __name__ == '__main__':
    unittest.main()
