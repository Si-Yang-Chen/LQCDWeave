"""P delivers fully planned windows without consuming all future coordinates."""
from dataclasses import replace
import unittest
import weakref
from unittest.mock import patch

from correlator.planning.test_cfg_plan import selected_template
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import describe_input_reads
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.api import iter_cfg_plan_windows, audit_cfg_plan, build_cfg_plan


class CfgPlanGeneratorTests(unittest.TestCase):
    def setUp(self):
        expanded, self.selected, *_ = selected_template()
        self.args = (expanded.output_recipe, describe_input_reads(expanded.inputs),
                     expanded.inputs.loader.preparations, expanded.inputs.loader.specs)
        self.points, _ = describe_coordinates(time_pairs=[{0:i % 2, 1:(i+1) % 2} for i in range(5)])

    def generate(self, points, **options):
        output, requests, preparations, specs = self.args
        return iter_cfg_plan_windows(self.selected, output, points, requests,
                                     preparations, specs, **options)

    def test_lazy_windows_keep_order_and_are_each_feasible(self):
        consumed = []
        def coordinates():
            for point in self.points.points:
                consumed.append(point)
                yield point
        iterator = self.generate(coordinates(), window_size=2)
        self.assertEqual(consumed, [])
        first = next(iterator)
        self.assertEqual(len(consumed), 2)
        self.assertEqual(first.statistics['coordinate_offset'], 0)
        self.assertEqual(first.statistics['window_index'], 0)
        audit_cfg_plan(first)
        rest = list(iterator)
        self.assertEqual([p.statistics['coordinate_offset'] for p in rest], [2, 4])
        self.assertEqual([len(p.instances) for p in (first, *rest)], [2, 2, 1])
        for plan in rest:
            audit_cfg_plan(plan)
        self.assertEqual([i.coordinate for p in (first, *rest) for i in p.instances],
                         [tuple((str(role), value) for role, value in point.values)
                          for point in self.points.points])

    def test_closing_does_not_bind_future_input_or_hide_later_error(self):
        def coordinates():
            yield self.points.points[0]
            raise RuntimeError('later coordinate failure')
        iterator = self.generate(coordinates(), window_size=1)
        next(iterator)
        iterator.close()
        iterator = self.generate(coordinates(), window_size=1)
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, 'later coordinate failure'):
            next(iterator)

    def test_invalid_window_and_empty_coordinates_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(window=value), self.assertRaises(ValueError):
                next(self.generate(self.points.points, window_size=value))
        with self.assertRaisesRegex(ValueError, 'at least one coordinate'):
            next(self.generate([], window_size=1))

    def test_requested_window_cannot_exceed_planner_working_set(self):
        with patch('lqcdweave.contraction.planning.cfg_plan._CFG_BINDING_WINDOW_SLOTS', 1):
            windows = list(self.generate(self.points.points, window_size=1000))
        self.assertEqual(len(windows), 5)
        self.assertTrue(all(len(p.instances) == 1 for p in windows))

    def test_fixed_binding_is_lazy_and_shared_but_actual_windows_are_independent(self):
        with patch.object(cfg_plan, '_bind_cfg_template', wraps=cfg_plan._bind_cfg_template) as bind:
            iterator = self.generate(self.points.points, window_size=2, debug=True)
            self.assertEqual(bind.call_count, 0)
            windows = list(iterator)
            self.assertEqual(bind.call_count, 1)
        self.assertEqual([window.statistics['template_binding_count'] for window in windows], [1, 0, 0])
        self.assertTrue(all(window.templates[0] is windows[0].templates[0] for window in windows))
        output, requests, preparations, specs = self.args
        for index, window in enumerate(windows):
            standalone = build_cfg_plan(self.selected, output, self.points.points[index*2:index*2+2],
                requests, preparations, specs, output_mode='dataset', debug=True)
            self.assertEqual(replace(window, statistics={}), replace(standalone, statistics={}))
        self.assertIsNot(windows[0].requests, windows[1].requests)
        self.assertIsNot(windows[0].preparations, windows[1].preparations)

    def test_accepted_input_tables_are_not_changed_by_later_external_dictionary_edits(self):
        output, requests, preparations, specs = self.args
        tables = [dict(table) for table in (requests, preparations, specs)]
        iterator = iter_cfg_plan_windows(self.selected, output, self.points.points, *tables, window_size=1)
        first = next(iterator)
        for table in tables:
            table.clear()
        remaining = list(iterator)
        for index, window in enumerate((first, *remaining)):
            standalone = build_cfg_plan(self.selected, output, (self.points.points[index],),
                requests, preparations, specs, output_mode='dataset')
            self.assertEqual(replace(window, statistics={}), replace(standalone, statistics={}))

    def test_separate_generators_keep_distinct_preparation_semantics(self):
        output, requests, preparations, specs = self.args
        key = next(key for key, prep in preparations.items() if prep.recipe.kind == 'spin_spatial')
        original = preparations[key]
        changed = dict(preparations)
        changed[key] = replace(original, recipe=replace(original.recipe, conjugate=not original.recipe.conjugate))
        first_iterator = self.generate(self.points.points, window_size=1)
        second_iterator = iter_cfg_plan_windows(self.selected, output, self.points.points,
            requests, changed, specs, window_size=1)
        first, second = next(first_iterator), next(second_iterator)
        self.assertIsNot(first.templates[0], second.templates[0])
        self.assertEqual(first.templates[0].preparation_closure[key].recipe.conjugate, original.recipe.conjugate)
        self.assertEqual(second.templates[0].preparation_closure[key].recipe.conjugate, not original.recipe.conjugate)
        self.assertIs(next(first_iterator).templates[0], first.templates[0])
        self.assertIs(next(second_iterator).templates[0], second.templates[0])
        first_iterator.close()
        second_iterator.close()

    def test_fixed_binding_lifetime_ends_on_close_completion_and_coordinate_error(self):
        bind_template = cfg_plan._bind_cfg_template
        references = []
        def bind(*args, **kwargs):
            binding = bind_template(*args, **kwargs)
            references.append(weakref.ref(binding))
            return binding
        def failing_points():
            yield self.points.points[0]
            raise RuntimeError('later coordinate failure')
        for end in ('close', 'complete', 'error'):
            with self.subTest(end=end), patch.object(cfg_plan, '_bind_cfg_template', side_effect=bind):
                iterator = self.generate(failing_points() if end == 'error' else self.points.points, window_size=1)
                first = next(iterator)
                self.assertIsNotNone(references[-1]())
                if end == 'close':
                    iterator.close()
                elif end == 'complete':
                    list(iterator)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'later coordinate failure'):
                        next(iterator)
                self.assertIsNone(references[-1]())
                audit_cfg_plan(first)


if __name__ == '__main__':
    unittest.main()
