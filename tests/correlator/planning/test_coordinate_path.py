"""Fixed dense DAG lifting agrees with independent scalar-coordinate sums."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, ContractionProblem, ContractionTerm, SourceSpec,
    SourcePreparation, StorageBlock, TensorSpec,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, find_dense_batches, lift_coordinate_path,
    plan_scene_storage,
)
from lqcdweave.contraction.planning import canonical_operations as canonical
from lqcdweave.contraction.planning import coordinate_path


LANE = 37
PROFILE = CostProfile()
CAP = 10**9


def problem(shapes, terms, *, backend='numpy', dtype='complex128'):
    specs = {key: TensorSpec(shape, dtype) for key, shape in shapes.items()}
    return ContractionProblem(specs, tuple(terms),
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=backend)


def selected(value, *, canonical=False, release=False):
    state = ContractionState.initial(value)
    for key in value.specs:
        state = state.load(key, PROFILE, CAP)
    while not state.done:
        scene = next(i for i, term in enumerate(state.pending) if term is not None)
        term = state.pending[scene]
        positions = (0,) if len(term.factors) == 1 else (0, 1)
        if canonical:
            action, spec, workspace, output, positions = canonical_operations.describe(
                state, scene, positions, profile=PROFILE)
            state = canonical_operations.apply_description(
                state, action, spec, workspace, output, positions)
        else:
            state = state.contract(scene, positions, PROFILE, CAP)
    if release:
        for key in tuple(state.loaded):
            if key not in state.results:
                state = state.release(key)
    return state


# Keep the fixture option readable without shadowing its module.
canonical_operations = canonical


def batch_problem(old, dynamic, width, *, rename=False):
    names = {key: ('new:' + key if rename else key) for key in old.specs}
    specs = {names[key]: TensorSpec((width, *spec.shape) if key in dynamic else spec.shape, spec.dtype)
             for key, spec in old.specs.items()}
    terms = tuple(ContractionTerm(tuple(names[key] for key in term.inputs),
        AxisPattern(tuple((LANE, *axes) if key in dynamic else axes
                          for key, axes in zip(term.inputs, term.subscripts.inputs)),
                    (LANE, *term.subscripts.output) if set(term.inputs) & set(dynamic)
                    else term.subscripts.output), term.coefficient) for term in old.terms)
    return ContractionProblem(specs, terms,
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=old.backend), names


def arrays_for(value, seed=782):
    rng = np.random.default_rng(seed)
    return {key: (rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)).astype(spec.dtype)
            for key, spec in value.specs.items()}


def einsum(values, pattern, *, optimize):
    # NumPy's integer front end accepts only [0,52); C axis IDs are arbitrary.
    names, arguments = {}, []
    for value, labels in zip(values, pattern.inputs):
        arguments.extend((value, [names.setdefault(label, len(names)) for label in labels]))
    output = [names[label] for label in pattern.output]
    return np.einsum(*arguments, output, optimize=optimize)


def evaluate_dag(state, inputs):
    resident = {}
    for action in state.actions:
        if action.kind == 'load':
            resident[action.tensor] = inputs[action.tensor]
        elif action.kind == 'release':
            del resident[action.tensor]
        elif not action.reused:
            resident[action.tensor] = einsum(
                [resident[key] for key in action.operands], action.equation, optimize=False)
    return tuple(resident[key] for key in state.results)


def scalar_reference(old, inputs, names, dynamic, width):
    result = []
    for term in old.terms:
        lanes = []
        for lane in range(width):
            values = []
            for key in term.inputs:
                value = inputs[names[key]]
                values.append(value[lane] if key in dynamic else value)
            lanes.append(einsum(values, term.subscripts, optimize='greedy'))
        result.append(np.stack(lanes) if set(term.inputs) & set(dynamic) else lanes[0])
    return result


class CoordinatePathTests(unittest.TestCase):
    def assert_numeric(self, old, dynamic, width=5, *, canon=False, release=False, rename=False):
        template = selected(old, canonical=canon, release=release)
        batched, names = batch_problem(old, dynamic, width, rename=rename)
        lifted = lift_coordinate_path(template, batched, names, lane_axis=LANE,
            profile=PROFILE, memory_limit=CAP, host_memory_limit=CAP, debug=True)
        inputs = arrays_for(batched)
        actual = evaluate_dag(lifted, inputs)
        expected = scalar_reference(old, inputs, names, dynamic, width)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
        self.assertEqual(len([a for a in lifted.actions if a.kind == 'contract' and not a.reused]),
                         len([a for a in template.actions if a.kind == 'contract' and not a.reused]))
        return template, lifted

    def test_static_left_right_and_two_dynamic_inputs_never_mix_lanes(self):
        old = problem({'A': (7,), 'B': (7,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        for dynamic in ({'A'}, {'B'}, {'A', 'B'}):
            for canon in (False, True):
                with self.subTest(dynamic=dynamic, canonical=canon):
                    _, lifted = self.assert_numeric(old, dynamic, canon=canon, rename=True)
                    self.assertEqual(lifted.catalog[lifted.results[0]].shape, (5,))
                    equation = next(a.equation for a in lifted.actions if a.kind == 'contract')
                    self.assertEqual(equation.output, (LANE,))

    def test_closed_loop_transposed_output_and_repeated_input(self):
        fixtures = (
            (problem({'A': (2, 3), 'B': (3, 4), 'C': (4, 2)},
                     (ContractionTerm(('A', 'B', 'C'), 'ij,jk,ki->'),)), {'A', 'C'}),
            (problem({'A': (2, 3), 'B': (3, 4)},
                     (ContractionTerm(('A', 'B'), 'ij,jk->ki'),)), {'A', 'B'}),
            (problem({'A': (3, 3)}, (ContractionTerm(('A', 'A'), 'ij,ji->'),)), {'A'}),
            (problem({'A': (3, 3)}, (ContractionTerm(('A',), 'ii->'),)), {'A'}),
        )
        for old, dynamic in fixtures:
            for canon in (False, True):
                with self.subTest(terms=old.terms, canonical=canon):
                    self.assert_numeric(old, dynamic, canon=canon, release=True)

    def test_shared_static_intermediate_and_static_output_are_not_replicated(self):
        old = problem({'A': (2, 3), 'B': (3, 4), 'C': (4, 2), 'D': (4, 2), 'S': (3, 3)}, (
            ContractionTerm(('A', 'B', 'C'), 'ij,jk,ki->'),
            ContractionTerm(('A', 'B', 'D'), 'ij,jk,ki->'),
            ContractionTerm(('S',), 'ii->'),
        ))
        for canon in (False, True):
            template, lifted = self.assert_numeric(old, {'C', 'D'}, canon=canon, release=True)
            self.assertEqual(lifted.catalog[lifted.results[2]].shape, ())
            shared = [a for a in lifted.actions if a.kind == 'contract' and not a.reused
                      and all(LANE not in axes for axes in a.equation.inputs)
                      and lifted.catalog[a.tensor].shape == (2, 4)]
            self.assertEqual(len(shared), 1)
            self.assertEqual(tuple(t.coefficient for t in lifted.problem.terms),
                             tuple(t.coefficient for t in template.problem.terms))

    def test_true_tail_width_one_and_two(self):
        old = problem({'A': (2, 3), 'B': (3, 2)}, (ContractionTerm(('A', 'B'), 'ij,jk->ik'),))
        for width in (1, 2, 5):
            self.assert_numeric(old, {'A', 'B'}, width, canon=True)

    def test_reused_producers_preserve_actions_under_dummy_axis_renaming(self):
        old = problem({'A': (2, 3), 'B': (3, 4), 'C': (4, 2)}, (
            ContractionTerm(('A', 'B', 'C'), 'ij,jk,ki->'),
            ContractionTerm(('A', 'B', 'C'), 'ab,bc,ca->'),
            ContractionTerm(('A', 'B', 'C'), 'xy,yz,zx->'),
        ))
        template = selected(old, canonical=True, release=True)
        self.assertTrue(any(a.reused for a in template.actions))
        batched, names = batch_problem(old, {'A', 'C'}, 3, rename=True)
        options = dict(lane_axis=LANE, profile=PROFILE, memory_limit=CAP)
        normal = lift_coordinate_path(template, batched, names, **options)
        independent = lift_coordinate_path(template, batched, names, debug=True, **options)
        self.assertEqual(normal.actions, independent.actions)
        self.assertEqual(normal.results, independent.results)
        for seed in (51, 91):
            inputs = arrays_for(batched, seed)
            actual = evaluate_dag(normal, inputs)
            expected = scalar_reference(old, inputs, names, {'A', 'C'}, 3)
            for value, reference in zip(actual, expected):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_single_or_repeated_coordinates_can_be_entirely_static(self):
        old = problem({'A': (2, 3), 'B': (3, 2)}, (ContractionTerm(('A', 'B'), 'ij,jk->ik'),))
        for width in (1, 5):
            _, lifted = self.assert_numeric(old, set(), width, canon=True, rename=True)
            self.assertEqual(lifted.catalog[lifted.results[0]].shape, (2, 2))

    def test_no_search_or_pending_replay_in_normal_path(self):
        old = problem({'A': (5,), 'B': (5,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        template = selected(old, canonical=True)
        batched, names = batch_problem(old, {'A', 'B'}, 4)
        with patch.object(ContractionState, 'initial', side_effect=AssertionError('pending rebuilt')), \
             patch.object(coordinate_path, '_independent_replay', side_effect=AssertionError('debug audit')):
            lifted = lift_coordinate_path(template, batched, names, lane_axis=LANE,
                                          profile=PROFILE, memory_limit=CAP)
        self.assertTrue(lifted.done)

    def test_logical_budget_is_strict_and_physical_audit_includes_raw_and_workspace(self):
        old = problem({'A': (7,), 'B': (7,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        template = selected(old, canonical=True)
        batched, names = batch_problem(old, {'A', 'B'}, 3)
        lifted = lift_coordinate_path(template, batched, names, lane_axis=LANE,
                                      profile=PROFILE, memory_limit=CAP)
        for cap in (lifted.peak_bytes - 1,):
            with self.assertRaises(MemoryError):
                lift_coordinate_path(template, batched, names, lane_axis=LANE,
                                     profile=PROFILE, memory_limit=cap)
            with self.assertRaises(MemoryError):
                lift_coordinate_path(template, batched, names, lane_axis=LANE,
                                     profile=PROFILE, memory_limit=CAP, host_memory_limit=cap)
        lift_coordinate_path(template, batched, names, lane_axis=LANE,
                             profile=PROFILE, memory_limit=lifted.peak_bytes)
        blocks = {key: StorageBlock(key, 'selected', 128, host_bytes=128) for key in names.values()}
        preps = {key: SourcePreparation((key,), workspace_host_bytes=4096) for key in names.values()}
        kwargs = dict(profile=PROFILE, memory_limit=CAP, host_memory_limit=CAP)
        normal = plan_scene_storage(lifted, blocks, preps, **kwargs)
        independent = plan_scene_storage(lifted, blocks, preps, debug=True, **kwargs)
        self.assertEqual(normal.memory, independent.memory)
        self.assertEqual(normal.events, independent.events)
        self.assertGreater(normal.peak_host_bytes, lifted.peak_bytes)
        self.assertFalse(plan_scene_storage(lifted, blocks, preps, profile=PROFILE,
                                           memory_limit=normal.peak_host_bytes - 1).feasible)

    def test_cupy_metadata_has_the_same_math_and_separate_logical_host_bound(self):
        old = problem({'A': (2, 3), 'B': (3, 2)},
                      (ContractionTerm(('A', 'B'), 'ij,jk->ik'),), backend='cupy')
        template, lifted = self.assert_numeric(old, {'A', 'B'}, canon=True)
        self.assertEqual(lifted.peak_host_bytes, 0)
        self.assertEqual(lifted.peak_device_bytes, lifted.peak_bytes)
        batched, names = batch_problem(old, {'A'}, 2)
        lift_coordinate_path(template, batched, names, lane_axis=LANE,
                             profile=PROFILE, memory_limit=CAP, host_memory_limit=1)

    def test_input_preloaded_dense_state_is_given_explicit_load_actions(self):
        old = problem({'A': (5,), 'B': (5,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        initial = ContractionState.initial(old)
        template = replace(initial, loaded=frozenset(old.specs)).contract(0, (0, 1), PROFILE, CAP)
        batched, names = batch_problem(old, {'A'}, 2)
        lifted = lift_coordinate_path(template, batched, names, lane_axis=LANE,
            profile=PROFILE, memory_limit=CAP, debug=True)
        self.assertEqual([a.kind for a in lifted.actions[:2]], ['load', 'load'])

    def test_incompatible_axis_rank_dtype_map_or_algebra_is_rejected(self):
        old = problem({'A': (3,), 'B': (3,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        template = selected(old, canonical=True)
        batched, names = batch_problem(old, {'A', 'B'}, 2)
        def lift(value=batched, mapping=names, axis=LANE):
            return lift_coordinate_path(template, value, mapping, lane_axis=axis,
                                        profile=PROFILE, memory_limit=CAP)
        for mapping in ({'A': 'A'}, {'A': 'A', 'B': 'A'}):
            with self.assertRaises(ValueError): lift(mapping=mapping)
        for axis in (True, -1, 0):
            with self.assertRaises(ValueError): lift(axis=axis)
        for shapes in ({'A': (2, 3), 'B': (4, 3)}, {'A': (2, 1, 3), 'B': (2, 3)},
                       {'A': (2, 4), 'B': (2, 4)}):
            # Some malformed dimensions are rejected by the C constructor.
            with self.assertRaises(ValueError):
                specs = {key: TensorSpec(shape, 'complex128') for key, shape in shapes.items()}
                lift(replace(batched, specs=specs))
        with self.assertRaises(ValueError):
            lift(replace(batched, specs={k: TensorSpec(s.shape, 'complex64') for k, s in batched.specs.items()}))
        with self.assertRaises(ValueError):
            lift(replace(batched, terms=(replace(batched.terms[0], coefficient=2),)))
        with self.assertRaises(ValueError):
            lift(replace(batched, backend='cupy'))
        # Dropping the lane axis is a valid all-static lift only if every
        # input and term consistently drop it together.
        with self.assertRaises(ValueError):
            lift(replace(batched, specs=old.specs))

    def test_special_actions_and_opaque_sources_are_refused(self):
        old = problem({'A': (3,), 'B': (3,)}, (ContractionTerm(('A', 'B'), 'i,i->'),))
        template = selected(old, canonical=True)
        batched, names = batch_problem(old, {'A'}, 2)
        compute = next(i for i, a in enumerate(template.actions) if a.kind == 'contract')
        for kind, core in (('sparse_bridge', None), ('baryon_join', None), ('contract', object())):
            actions = list(template.actions)
            actions[compute] = replace(actions[compute], kind=kind, core=core)
            with self.assertRaises(ValueError):
                lift_coordinate_path(replace(template, actions=tuple(actions)), batched, names,
                                     lane_axis=LANE, profile=PROFILE, memory_limit=CAP)
        opaque = replace(batched, sources={**batched.sources, 'opaque': SourceSpec('opaque', 0)})
        with self.assertRaisesRegex(ValueError, 'opaque'):
            lift_coordinate_path(template, opaque, names, lane_axis=LANE, profile=PROFILE, memory_limit=CAP)

    def test_existing_cartesian_lowering_keeps_only_eligible_single_dynamic_side(self):
        old = problem({key: (11,) for key in ('a0', 'a1', 'b0', 'b1')},
            tuple(ContractionTerm((f'a{i}', f'b{j}'), 'i,i->') for i in range(2) for j in range(2)))
        template = selected(old, canonical=True)
        for dynamic, eligible in (({'a0', 'a1'}, True), ({'a0', 'a1', 'b0', 'b1'}, False)):
            batched, names = batch_problem(old, dynamic, 3)
            lifted = lift_coordinate_path(template, batched, names, lane_axis=LANE,
                                          profile=PROFILE, memory_limit=CAP, debug=True)
            self.assertEqual(bool(find_dense_batches(lifted, PROFILE)), eligible)
            actual = evaluate_dag(lifted, arrays_for(batched))
            expected = scalar_reference(old, arrays_for(batched), names, dynamic, 3)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
