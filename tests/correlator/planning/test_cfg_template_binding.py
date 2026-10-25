"""Canonical identities keep coordinate-dependent ordering and repeated legs."""
from dataclasses import dataclass, fields, FrozenInstanceError, replace
from itertools import product
from functools import partial
from types import MappingProxyType
import unittest
import numpy as np
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, AxisPattern, CfgTemplate, MemoryPoint, SourcePreparation, SourceReadRequest,
    TensorAxisMetadata, TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.expansion.api import describe_input_reads
from lqcdweave.contraction.planning.api import build_cfg_plan
from correlator.planning import test_cfg_plan as cfg_fixtures


class CfgTemplateBindingTests(unittest.TestCase):
    def test_memory_remap_retains_native_unchanged_records_and_all_changed_fields(self):
        values = {field.name: (index + 1) * 17 for index, field in enumerate(fields(MemoryPoint))}
        values.update(action_index=3, phase='contract')
        original = MemoryPoint(**values)
        expected = replace(original, action_index=11)
        with patch.object(cfg_plan, 'replace', side_effect=AssertionError('generic native reconstruction')):
            self.assertIs(cfg_plan._remap_memory_point(original, 3, debug=False), original)
            actual = cfg_plan._remap_memory_point(original, 11, debug=False)
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, original)
        self.assertEqual(original.action_index, 3)

    def test_debug_and_extended_memory_points_keep_original_constructors(self):
        original = MemoryPoint(3, 'contract', 1, 2, 3, 4, 5, 6)
        debug = cfg_plan._remap_memory_point(original, 3, debug=True)
        self.assertEqual(debug, original)
        self.assertIsNot(debug, original)

        @dataclass(frozen=True)
        class ExtendedPoint(MemoryPoint):
            tag: str = 'extension'

            def __post_init__(self):
                if self.action_index == 11:
                    raise ValueError('extension rejected remapped position')

        extended = ExtendedPoint(**vars(original))
        actual = cfg_plan._remap_memory_point(extended, 3, debug=False)
        self.assertIs(type(actual), ExtendedPoint)
        self.assertIsNot(actual, extended)
        self.assertEqual(actual.tag, 'extension')
        with self.assertRaisesRegex(ValueError, 'extension rejected remapped position'):
            cfg_plan._remap_memory_point(extended, 11, debug=False)

    def test_template_memory_matches_the_original_release_index_remap(self):
        selected, preps, specs = cfg_fixtures.CfgPlanTests.dense_fixture()
        positions = {old:new for new,(old,_) in enumerate(
            (i,action) for i,action in enumerate(selected.state.actions) if action.kind!='release')}
        positions[len(selected.state.actions)] = len(positions)
        expected = tuple(replace(point, action_index=positions[point.action_index])
                         for point in selected.memory if point.action_index in positions)
        bound = cfg_plan._bind_cfg_template(selected, None, {}, preps, specs)
        self.assertEqual(bound.keywords['template'].plan.memory, expected)

    def test_preparation_cost_follows_first_load_order_and_excludes_unused_records(self):
        selected, preparations, specs = cfg_fixtures.CfgPlanTests.dense_fixture()
        unused = ArrayConstant.from_array(np.array([-123.5 + 7j]))
        preparations = dict(preparations, unused=SourcePreparation(
            (), recipe=TensorPreparation('constant', constants=(unused,))))
        specs = dict(specs, unused=TensorSpec((1,), 'complex128'))
        coordinates, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}, {0: 0}])
        for debug in (False, True):
            for slots in (4_000_000, len(selected.state.actions)):
                calls = []

                def cost(prep, profile):
                    calls.append(prep)
                    return 1.0 / len(calls)

                with self.subTest(debug=debug, slots=slots), \
                        patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', slots), \
                        patch.object(SourcePreparation, 'seconds', cost):
                    cfg = build_cfg_plan(selected, None, coordinates.points, {},
                                         preparations, specs, output_mode='dataset', debug=debug)
                    expected = []
                    template = cfg.templates[0]
                    positions = {key: index for index, key in enumerate(template.keys)}
                    for instance in cfg.instances:
                        node_ids = instance.node_ids.array()
                        for action, reused in zip(template.plan.state.actions,
                                                  instance.reused.array(), strict=True):
                            if action.kind == 'load' and not reused:
                                expected.append(cfg.preparations[str(node_ids[positions[action.tensor]])])
                    self.assertEqual(calls, expected)
                    self.assertTrue(calls)
                    self.assertTrue(all(unused not in prep.recipe.constants for prep in calls))
                    self.assertEqual(cfg.statistics['preparation_seconds'],
                                     sum(1.0 / (index + 1) for index in range(len(expected))))
                    if slots == 4_000_000:
                        self.assertTrue(all(cfg.instances[1].reused.array()))

    def test_read_alias_merging_preserves_order_and_complete_reader_semantics(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        main = SourceReadRequest('main', 'elemental', (4,), (), metadata)
        first = replace(main, reader_role='first')
        second = replace(main, reader_role='second', conventions=(('time_convention', 'absolute'),))
        alternate = replace(main, reader_role='alternate', arguments=(5,))
        cases = (
            (main, main),
            (replace(main, aliases=(first,)), main),
            (replace(main, aliases=(first, first, main)),
             replace(alternate, aliases=(second, first, main))),
            (replace(main, aliases=(second,)), replace(main, metadata=replace(metadata, data_identity='other'))),
        )
        for previous, actual in cases:
            aliases = []
            for alias in (*previous.aliases, *actual.aliases, replace(actual, aliases=())):
                if alias != replace(previous, aliases=()) and alias not in aliases:
                    aliases.append(alias)
            expected = replace(previous, aliases=tuple(aliases))
            for debug in (False, True):
                with self.subTest(previous=previous.reader_role, actual=actual.reader_role, debug=debug):
                    merged = cfg_plan._merge_read_aliases(previous, actual, debug=debug)
                    self.assertEqual(merged, expected)
                    self.assertEqual(merged.arguments, previous.arguments)

    def test_unchanged_read_aliases_reuse_the_frozen_standard_record(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        main = SourceReadRequest('main', 'elemental', (4,), (), metadata)
        with patch.object(SourceReadRequest, '__post_init__', side_effect=AssertionError('redundant constructor')):
            self.assertIs(cfg_plan._merge_read_aliases(main, main, debug=False), main)
        previous = replace(main, aliases=(replace(main, reader_role='alternate'),))
        self.assertIs(cfg_plan._merge_read_aliases(previous, main, debug=False), previous)

    def test_debug_and_extended_read_aliases_keep_constructor_hooks(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        main = SourceReadRequest('main', 'elemental', (4,), (), metadata)

        @dataclass(frozen=True)
        class ExtendedRead(SourceReadRequest):
            pass

        extended = ExtendedRead('main', 'elemental', (4,), (), metadata)
        for previous, actual, debug in ((main, main, True), (extended, extended, False),
                                        (main, extended, False)):
            with self.subTest(debug=debug, type=type(actual)), \
                    patch.object(SourceReadRequest, '__post_init__', side_effect=ValueError('constructor hook')):
                with self.assertRaisesRegex(ValueError, 'constructor hook'):
                    cfg_plan._merge_read_aliases(previous, actual, debug=debug)

    def test_full_cfg_alias_merging_matches_the_original_constructor_path(self):
        fixture = cfg_fixtures.CfgPlanTests()
        merge = cfg_plan._merge_read_aliases
        with patch.object(cfg_plan, '_merge_read_aliases',
                          side_effect=lambda previous, actual, debug: merge(previous, actual, debug=True)):
            reference = fixture.build(mode='dataset')
        actual = fixture.build(mode='dataset')
        self.assertEqual(replace(actual, statistics={}), replace(reference, statistics={}))

    def test_bound_reads_match_public_coordinate_and_metadata_semantics(self):
        metadata = TensorAxisMetadata(('mode_left', 'mode_right'), (3, 3), 'complex128', 'data', 'all')
        alias = SourceReadRequest('alias', 'propagator', (11, 19, 99), (('usedNe', 3),), metadata,
            144, conventions=(('time_convention', 'absolute'),),
            coordinate_arguments=((0, 'sink', -1, 8),),
            coordinate_tables=((1, 'source', (4, None, 7)),))
        request = replace(alias, reader_role='main', aliases=(alias,))
        overrides = {('main', (7, 7, 99), request.selection): replace(metadata, selection_identity='main-selection'),
                     ('alias', (7, 7, 99), alias.selection): replace(metadata, selection_identity='alias-selection')}
        for debug in (False, True):
            bind = cfg_plan._bind_read_request(request, overrides, debug=debug)
            for values in ({}, {'sink': np.int64(0), 'source': np.int64(2)},
                           {'sink': 9, 'source': 0}, {'sink': -1}):
                expected_alias = alias.at(values)
                expected_alias = replace(expected_alias, metadata=overrides.get(
                    (expected_alias.reader_role, expected_alias.arguments, expected_alias.selection), metadata))
                expected = request.at(values)
                expected = replace(expected, metadata=overrides.get(
                    (expected.reader_role, expected.arguments, expected.selection), metadata), aliases=(expected_alias,))
                with self.subTest(debug=debug, values=values):
                    self.assertEqual(bind(values), expected)
            self.assertEqual(request.arguments, (11, 19, 99))
            self.assertEqual(request.aliases, (alias,))
        bound = cfg_plan._bind_read_request(request, overrides, debug=False)
        with patch.object(SourceReadRequest, '__post_init__', side_effect=AssertionError('fixed request rescan')):
            actual = bound({'sink': 0, 'source': 2})
            self.assertEqual(actual.arguments, (7, 7, 99))
            self.assertEqual(actual.aliases[0].metadata.selection_identity, 'alias-selection')
            with self.assertRaises(FrozenInstanceError):
                actual.arguments = ()

    def test_bound_read_tables_keep_actual_slot_and_missing_endpoint_checks(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        request = SourceReadRequest('main', 'elemental', (4,), (), metadata,
                                    coordinate_tables=((0, 'time', (4, None, 7)),))
        for debug in (False, True):
            bind = cfg_plan._bind_read_request(request, {}, debug=debug)
            for slot in (-1, 3, 1.0, True, np.bool_(False)):
                with self.subTest(debug=debug, slot=slot), self.assertRaisesRegex(ValueError, 'outside saved lookup'):
                    bind({'time': slot})
            with self.subTest(debug=debug), self.assertRaisesRegex(KeyError, 'physical endpoint is missing'):
                bind({'time': 1})
            self.assertEqual(bind({'time': np.int64(2)}).arguments, (7,))

    def test_fixed_reads_and_new_metadata_keep_boundary_validation(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        request = SourceReadRequest('main', 'elemental', (4,), (), metadata)
        bind = cfg_plan._bind_read_request(request, {}, debug=False)
        self.assertIs(bind({'time': 1}), request)
        changed = replace(metadata, selection_identity='changed')
        overridden = cfg_plan._bind_read_request(request, {('main', (4,), ()): changed}, debug=False)
        self.assertIs(overridden({}).metadata, changed)
        for debug in (False, True):
            with self.subTest(debug=debug), self.assertRaisesRegex(TypeError, 'dense TensorAxisMetadata'):
                cfg_plan._bind_read_request(request, {('main', (4,), ()): object()}, debug=debug)({})

    def test_debug_and_extended_reads_keep_at_and_constructor_hooks(self):
        metadata = TensorAxisMetadata(('mode',), (3,), 'complex128', 'data', 'all')
        request = SourceReadRequest('main', 'elemental', (4,), (), metadata)
        debug = cfg_plan._bind_read_request(request, {}, debug=True)
        with patch.object(SourceReadRequest, '__post_init__', side_effect=ValueError('debug constructor')):
            with self.assertRaisesRegex(ValueError, 'debug constructor'):
                debug({})

        @dataclass(frozen=True)
        class ExtendedRead(SourceReadRequest):
            def __post_init__(self):
                super().__post_init__()
                if self.arguments == (6,):
                    raise ValueError('extension rejected bound argument')

            def at(self, values):
                return replace(self, arguments=(int(values['time']) + 1,))

        extended = ExtendedRead('main', 'elemental', (4,), (), metadata)
        bind = cfg_plan._bind_read_request(extended, {}, debug=False)
        self.assertEqual(bind({'time': 1}).arguments, (2,))
        with self.assertRaisesRegex(ValueError, 'extension rejected'):
            bind({'time': 5})

    def test_preparation_discovery_preserves_nested_order_and_distinct_raw_roles(self):
        # The outer record appears first, and shared children appear several
        # times. A raw name also exists as a preparation: stack must use the
        # prepared value, while identity must keep the raw request.
        constant = ArrayConstant.from_array(np.array(1.))
        preparations = {
            'outer': SourcePreparation((), recipe=TensorPreparation('stack', ('inner', 'inner', 'tail'))),
            'tail': SourcePreparation(('raw',), recipe=TensorPreparation('identity', ('raw',))),
            'inner': SourcePreparation((), recipe=TensorPreparation('stack', ('leaf', 'raw'))),
            'leaf': SourcePreparation(('raw',), recipe=TensorPreparation('identity', ('raw',))),
            'raw': SourcePreparation((), recipe=TensorPreparation('constant', constants=(constant,))),
        }
        order = cfg_plan._preparation_order(preparations, {'raw': object()})
        self.assertEqual(order, ('leaf', 'raw', 'inner', 'tail', 'outer'))
        self.assertEqual(preparations['outer'].recipe.inputs, ('inner', 'inner', 'tail'))

    def test_preparation_discovery_rejects_cycles_and_missing_inputs(self):
        cases = (
            ({'a': SourcePreparation((), recipe=TensorPreparation('identity', ('b',))),
              'b': SourcePreparation((), recipe=TensorPreparation('identity', ('a',)))},
             {}, 'cyclic saved preparation'),
            ({'a': SourcePreparation((), recipe=TensorPreparation('stack', ('a',)))},
             {'a': object()}, 'cyclic saved preparation'),
            ({'a': SourcePreparation((), recipe=TensorPreparation('stack', ('missing',)))},
             {'missing': object()}, 'no preparation'),
        )
        for preparations, requests, message in cases:
            with self.subTest(message=message, requests=tuple(requests)), self.assertRaisesRegex(ValueError, message):
                cfg_plan._preparation_order(preparations, requests)

    def test_preparation_id_binding_preserves_all_fixed_semantics_without_revalidation(self):
        from lqcdweave.contracts.correlator.api import ArrayConstant
        spin = ArrayConstant.from_array(np.eye(4, dtype=np.complex128))
        rows = ArrayConstant.from_array(np.ones((2, 3), dtype=np.complex128))
        weights = ArrayConstant.from_array(np.array([1., 2.]))
        recipes = (
            TensorPreparation('identity', ('raw',)),
            TensorPreparation('stack', ('left', 'right')),
            TensorPreparation('constant', constants=(rows,)),
            TensorPreparation('spin_spatial', ('raw',), (spin, rows, rows, weights),
                              True, (1, 0), 'cfg', 'diagonal'),
        )
        for recipe in recipes:
            prep = SourcePreparation(('raw-a', 'raw-b'), 7, 11, 13, 17, 19, 'complex64', recipe)
            binder = cfg_plan._bind_preparation(prep, debug=False)
            inputs = tuple(str(index + 10) for index in range(len(recipe.inputs)))
            original_inputs = recipe.inputs
            blocks = ('20', '21')
            expected = replace(prep, blocks=blocks, recipe=replace(recipe, inputs=inputs))
            with (
                self.subTest(kind=recipe.kind),
                patch.object(TensorPreparation, '__post_init__', side_effect=AssertionError('fixed recipe rescan')),
                patch.object(SourcePreparation, '__post_init__', side_effect=AssertionError('fixed resource rescan')),
            ):
                bound = binder(inputs, blocks)
                self.assertEqual(bound, expected)
                self.assertIs(bound.recipe.constants, recipe.constants)
                self.assertEqual(prep.blocks, ('raw-a', 'raw-b'))
                self.assertEqual(recipe.inputs, original_inputs)
                with self.assertRaises(FrozenInstanceError):
                    bound.flops = 0

    def test_coordinate_aliases_still_cannot_collapse_distinct_block_dependencies(self):
        prep = SourcePreparation(('raw-a', 'raw-b'),
                                 recipe=TensorPreparation('identity', ('raw-a',)))
        for debug in (False, True):
            with self.subTest(debug=debug), self.assertRaisesRegex(ValueError, 'duplicate source block'):
                cfg_plan._bind_preparation(prep, debug=debug)(('10',), ('10', '10'))

    def test_debug_and_extended_preparation_records_keep_constructor_hooks(self):
        @dataclass(frozen=True)
        class ExtendedPreparation(SourcePreparation):
            def __post_init__(self):
                super().__post_init__()
                if self.blocks == ('rejected',):
                    raise ValueError('extension rejected bound block')

        recipe = TensorPreparation('identity', ('raw',))
        prep = SourcePreparation(('raw',), recipe=recipe)
        with patch.object(SourcePreparation, '__post_init__', side_effect=ValueError('debug constructor')):
            with self.assertRaisesRegex(ValueError, 'debug constructor'):
                cfg_plan._bind_preparation(prep, debug=True)(('10',), ('10',))
        extended = ExtendedPreparation(('raw',), recipe=recipe)
        with self.assertRaisesRegex(ValueError, 'extension rejected'):
            cfg_plan._bind_preparation(extended, debug=False)(('10',), ('rejected',))

    def test_full_cfg_default_binding_matches_debug_constructor_and_archive_path(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lqcdweave.storage.contraction_plans.api import load_cfg_plan, save_cfg_plan
        expanded, selected, *_ = cfg_fixtures.selected_template()
        loader = expanded.inputs.loader
        points, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 0}])
        args = (selected, expanded.output_recipe, points.points,
                describe_input_reads(expanded.inputs), loader.preparations, loader.specs)
        default = build_cfg_plan(*args, output_mode='dataset')
        debug = build_cfg_plan(*args, output_mode='dataset', debug=True)
        self.assertEqual(replace(default, statistics={}), replace(debug, statistics={}))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bound.json'
            save_cfg_plan(path, default)
            restored = load_cfg_plan(path)
        self.assertEqual(restored, default)

    def test_rebound_extension_workspace_is_counted_before_resource_admission(self):
        @dataclass(frozen=True)
        class ReboundWorkspace(SourcePreparation):
            def __post_init__(self):
                super().__post_init__()
                if self.blocks and all(block.isdecimal() for block in self.blocks):
                    object.__setattr__(self, 'workspace_host_bytes', 1 << 20)

        expanded, selected, *_ = cfg_fixtures.selected_template()
        loader = expanded.inputs.loader
        key = next(action.tensor for action in selected.state.actions if action.kind == 'load')
        original = loader.preparations[key]
        extended = ReboundWorkspace(**vars(original))
        self.assertLess(extended.workspace_host_bytes, 1 << 20)
        points, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}])
        for shared in (False, True):
            if shared:
                # An extension discovered first can own a standard Action's
                # shared preparation despite identical accepted resource fields.
                preparations = {'extra:extended': extended, **loader.preparations}
                specs = {'extra:extended': loader.specs[key], **loader.specs}
                selected_variant = selected
            else:
                preparations, specs = loader.preparations, loader.specs
                selected_variant = replace(selected, preparations={**selected.preparations, key: extended})
            args = (expanded.output_recipe, points.points, describe_input_reads(expanded.inputs),
                    preparations, specs)
            plans = []
            for debug in (False, True):
                with self.subTest(debug=debug, shared=shared):
                    plan = build_cfg_plan(selected_variant, *args, output_mode='dataset', debug=debug)
                    index = plan.templates[0].keys.index(key)
                    gid = str(plan.instances[0].node_ids.array()[index])
                    bound = plan.preparations[gid]
                    self.assertIsInstance(bound, ReboundWorkspace)
                    self.assertEqual(bound.workspace_host_bytes, 1 << 20)
                    self.assertGreaterEqual(plan.statistics['peak_preparation_workspace_host_bytes'], 1 << 20)
                    with self.assertRaisesRegex(MemoryError, 'host bytes'):
                        build_cfg_plan(replace(selected_variant, memory_limit=plan.peak_host_bytes - 1),
                                       *args, output_mode='dataset', debug=debug)
                    plans.append(plan)
            self.assertEqual(replace(plans[0], statistics={}), replace(plans[1], statistics={}))

    def test_bound_keys_preserve_order_diagonals_and_equal_operands(self):
        patterns = (
            AxisPattern(((0, 1), (1, 2)), (0, 2)),
            AxisPattern(((4, 4, 9), (9, 2)), (2, 4)),
            AxisPattern(((0, 1), (0, 1)), ()),
            AxisPattern(((1,), (2,)), (2, 1)),
            AxisPattern(((7, 7, 3),), (3,)),
            AxisPattern(((0, 1), (1, 2), (2, 3)), (0, 3)),
        )
        for pattern in patterns:
            action = Action('contract', 'result', tuple(f'x{i}' for i in range(len(pattern.inputs))), pattern)
            bound = cfg_plan._bind_computation_identity(action)
            for operands in product((0, 3, 8), repeat=len(pattern.inputs)):
                with self.subTest(pattern=pattern, operands=operands):
                    expected = cfg_plan._computation_identity(action, operands)
                    self.assertEqual(bound(operands), expected)
        # Different dummy labels describe the same operation.
        first = Action('contract', 'r', ('a', 'b'), AxisPattern(((0, 1), (1, 2)), (0, 2)))
        second = replace(first, equation=AxisPattern(((6, 9), (9, 4)), (6, 4)))
        self.assertEqual(cfg_plan._bind_computation_identity(first)((8, 3)),
                         cfg_plan._bind_computation_identity(second)((8, 3)))

    def test_special_parameters_remain_in_identity(self):
        first = Action('contract', 'r', ('a', 'b'), AxisPattern(((0, 1), (1, 2)), (0, 2)), used_np=3)
        second = replace(first, used_np=4)
        for operands in ((2, 1), (1, 1)):
            self.assertEqual(cfg_plan._bind_computation_identity(first)(operands),
                             cfg_plan._computation_identity(first, operands))
            self.assertNotEqual(cfg_plan._bind_computation_identity(first)(operands),
                                cfg_plan._bind_computation_identity(second)(operands))

    def test_interned_keys_preserve_reference_sharing_across_actions_and_coordinates(self):
        patterns = (
            AxisPattern(((0, 1), (1, 2)), (0, 2)),
            AxisPattern(((6, 9), (9, 4)), (6, 4)),
            AxisPattern(((1, 2), (0, 1)), (0, 2)),
            AxisPattern(((0, 1), (1, 2)), (2, 0)),
            AxisPattern(((4, 4, 9), (9, 2)), (2, 4)),
            AxisPattern(((0, 1), (0, 1)), ()),
            AxisPattern(((1,), (2,)), (2, 1)),
            AxisPattern(((7, 7, 3),), (3,)),
            AxisPattern(((8, 8, 5),), (5,)),
            AxisPattern(((0, 1), (1, 2), (2, 3)), (0, 3)),
        )
        actions = [Action('contract', 'result',
                          tuple(f'x{i}' for i in range(len(pattern.inputs))), pattern)
                   for pattern in patterns]
        actions.extend(replace(actions[0], used_np=count) for count in (3, 4))
        axes = {}
        binders = [cfg_plan._bind_computation_identity(action, axes) for action in actions]
        reference_to_bound, bound_to_reference = {}, {}
        for action, bound in zip(actions, binders):
            for operands in product((0, 3, 8), repeat=len(action.operands)):
                expected = cfg_plan._computation_identity(action, operands)
                actual = bound(operands)
                with self.subTest(equation=action.equation, special=action.used_np, operands=operands):
                    # Both directions matter: no lost reuse and no false sharing.
                    self.assertEqual(reference_to_bound.setdefault(expected, actual), actual)
                    self.assertEqual(bound_to_reference.setdefault(actual, expected), expected)

    def test_full_cfg_metadata_matches_unbound_reference(self):
        fixture = cfg_fixtures.CfgPlanTests()
        with patch.object(cfg_plan, '_bind_computation_identity',
                          side_effect=lambda action, axis_ids=None: partial(cfg_plan._computation_identity, action)):
            reference = fixture.build(mode='dataset')
        actual = fixture.build(mode='dataset')
        # Timing observations differ; all identity, event, owner and budget fields must match.
        self.assertEqual(replace(actual, statistics={}), replace(reference, statistics={}))

    def test_template_carries_full_nested_preparation_closure_and_local_roles(self):
        expanded, selected, *_ = cfg_fixtures.selected_template()
        loader = expanded.inputs.loader
        preparations = dict(loader.preparations)
        specs = dict(loader.specs)
        requests = describe_input_reads(expanded.inputs)
        explicit_spec = TensorSpec(specs['tensor:0'].shape, specs['tensor:0'].dtype)
        specs['tensor:0'] = explicit_spec
        fallback_key = 'tensor:1'
        self.assertIn(fallback_key, selected.state.catalog)
        del specs[fallback_key]

        # The selected storage plan has the top-level stack but omits its spin
        # members. Add an unused nested stack closure to exercise every recipe
        # depth without changing the selected Actions or storage trajectory.
        leaf = preparations['tensor:2']
        leaf_spec = specs['tensor:2']
        leaves = ('template:leaf0', 'template:leaf1',
                  'template:leaf2', 'template:leaf3')
        for key in leaves:
            preparations[key] = leaf
            specs[key] = leaf_spec

        inner_spec = TensorSpec((2, *leaf_spec.shape), leaf_spec.dtype)
        inners = ('template:inner0', 'template:inner1')
        for index, key in enumerate(inners):
            members = leaves[2 * index:2 * index + 2]
            preparations[key] = SourcePreparation(
                leaf.blocks,
                recipe=TensorPreparation('stack', members),
            )
            specs[key] = inner_spec
        preparations['template:outer'] = SourcePreparation(
            leaf.blocks,
            recipe=TensorPreparation('stack', inners),
        )
        specs['template:outer'] = TensorSpec((2, *inner_spec.shape), inner_spec.dtype)

        points, _ = describe_coordinates(time_pairs=[{0: 0, 1: 0}, {0: 1, 1: 1}])
        cfg = build_cfg_plan(selected, expanded.output_recipe, points.points,
                             requests, preparations, specs)
        template, = cfg.templates
        self.assertIn('matrix:vertices:0', selected.preparations)
        self.assertNotIn('tensor:2', selected.preparations)
        self.assertEqual(set(template.preparation_closure), set(preparations))
        self.assertEqual(template.preparation_closure['template:outer'].recipe.inputs, inners)
        self.assertEqual(template.preparation_closure['template:inner0'].recipe.inputs, leaves[:2])
        self.assertEqual(set(template.preparation_specs), set(preparations))
        self.assertIs(template.preparation_specs['tensor:0'], explicit_spec)
        self.assertIs(template.preparation_specs[fallback_key],
                      selected.state.catalog[fallback_key])
        self.assertEqual(template.plan.preparations, selected.preparations)
        self.assertIsInstance(template.preparation_closure, MappingProxyType)
        self.assertIsInstance(template.preparation_specs, MappingProxyType)
        with self.assertRaises(TypeError):
            template.preparation_closure['new'] = leaf

        local_keys = (*leaves, *inners, 'template:outer', 'tensor:2', 'tensor:3')
        for instance in cfg.instances:
            binding = dict(zip(template.keys, instance.node_ids.array(), strict=True))
            for key in local_keys:
                self.assertGreaterEqual(int(binding[key]), 0, key)
                record = cfg.preparations[str(int(binding[key]))]
                recipe = record.recipe
                if recipe.kind == 'spin_spatial':
                    self.assertTrue(all(block in cfg.blocks for block in recipe.inputs))
                elif recipe.kind == 'stack':
                    self.assertTrue(all(member in cfg.preparations for member in recipe.inputs))

    def test_template_semantics_include_leaf_constant_configuration_and_dtype(self):
        template = cfg_fixtures.CfgPlanTests().build().templates[0]
        key = next(key for key, prep in template.preparation_closure.items()
                   if prep.recipe.kind == 'spin_spatial')
        prep = template.preparation_closure[key]
        constant = prep.recipe.constants[0]
        changed_array = np.array(constant.array(), copy=True)
        changed_array.flat[0] += 1
        changed_constant = ArrayConstant.from_array(changed_array)
        changed_recipe = replace(prep.recipe,
            constants=(changed_constant, *prep.recipe.constants[1:]))
        closure = dict(template.preparation_closure)
        closure[key] = replace(prep, recipe=changed_recipe)
        self.assertNotEqual(replace(template, preparation_closure=closure), template)

        changed_recipe = replace(prep.recipe, configuration='other-configuration')
        closure = dict(template.preparation_closure)
        closure[key] = replace(prep, recipe=changed_recipe)
        self.assertNotEqual(replace(template, preparation_closure=closure), template)

        changed_specs = dict(template.preparation_specs)
        changed_specs[key] = TensorSpec(changed_specs[key].shape, np.complex64)
        self.assertNotEqual(replace(template, preparation_specs=changed_specs), template)

    def test_legacy_template_copies_mutable_keys(self):
        source = cfg_fixtures.CfgPlanTests().build().templates[0]
        external_keys = list(source.keys)
        template = CfgTemplate(source.plan, external_keys, source.output)

        external_keys.append('external-mutation')

        self.assertIs(type(template.keys), tuple)
        self.assertEqual(template.keys, source.keys)
        self.assertIsNone(template.preparation_closure)
        self.assertIsNone(template.preparation_specs)


if __name__ == '__main__':
    unittest.main()
