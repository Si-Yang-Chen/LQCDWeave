"""Offline Scene representation conversion retains the complete arrangement."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from correlator.execution import test_cfg_runtime as fixtures
from lqcdweave.storage.contraction_plans.api import (
    encode_plan_value, decode_plan_value, save_cfg_plan, load_cfg_plan,
    migrate_plan_archive, PlanArchiveError,
)
from lqcdweave.contracts.correlator.api import Action, ContractionTerm, AxisPattern


def legacy_labels(axes):
    return "".join(chr(axis) if 65 <= axis <= 122 else chr(97 + axis) for axis in axes)


def legacy_records(value):
    if isinstance(value, list):
        return [legacy_records(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get('type', '').endswith('.AxisPattern'):
        pattern = decode_plan_value(value)
        return ','.join(legacy_labels(axes) for axes in pattern.inputs) + '->' + legacy_labels(pattern.output)
    fields = value.get('value')
    if isinstance(fields, dict):
        name = value.get('type', '').rsplit('.', 1)[-1]
        for field in {'Factor': ('labels',), 'PendingTerm': ('output',),
                      'BaryonJoinSpec': ('spin_labels', 'mode_labels')}.get(name, ()):
            fields[field] = legacy_labels(decode_plan_value(fields[field]))
        if name == 'Action' and fields.get('equation') is None:
            fields['equation'] = ''
    return {key: legacy_records(item) for key, item in value.items()}


class IntegerSceneMigrationTests(unittest.TestCase):
    def test_fixed_paths_migrate_across_axis_renaming_and_batch_tail(self):
        from dataclasses import replace
        from correlator.plan_archive.test_workflow_import import WorkflowPlanImportTests
        from lqcdweave.contraction.correlator import calc_diagram_eval, save_calculation_plan
        from lqcdweave.storage.contraction_plans import api as archive
        fixture = WorkflowPlanImportTests()
        diagram, propagator, _, vertices = fixture._fixture()
        from test_structured_correlator import fixture as matrix_fixture, task
        bind, _, matrix_propagator = matrix_fixture()
        cases = [(diagram, dict(propagator_map={'q': propagator}, vertex_map=vertices,
                    usedNe=1, time_axes={0: 1, 1: 0} if batch is None else {0: range(3), 1: 0},
                    batch='auto' if batch is None else batch))
                 for batch in (None, {'axis': 0, 'size': 2})]
        cases.extend((task(structured=True, flavor='I0'), dict(vertex_map=bind,
            propagator_map={'S^q': matrix_propagator, 'S^q_\\mathrm{local}': matrix_propagator},
            usedNe=3, time_axes=times, batch=batch)) for times, batch in (
                ({0: 1, 1: 0}, 'auto'), ({0: range(2), 1: 0}, {'axis': 0, 'size': 2})))
        for index, (correlator, options) in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                calculation = calc_diagram_eval(correlator, configuration='cfg', memory_limit=1 << 20,
                                               optimize='anytime', execute=False, **options)
                source, target = Path(directory)/'old.json', Path(directory)/'new.json'
                save_calculation_plan(source, calculation)
                document = json.loads(source.read_text())
                document['version'] = 5
                document['payload'] = legacy_records(document['payload'])
                payload = document['payload']
                for task in (payload['task'], *(row['task'] for row in payload['plans'])):
                    for term in task['problem']['terms']:
                        pattern = term['subscripts']
                        term['subscripts'] = ','.join(legacy_labels(axes) for axes in pattern['inputs']) + '->' + legacy_labels(pattern['output'])
                payload['path_signature'] = archive._digest(
                    archive._problem_path_invariants_from_payload(payload['task']['problem'], normalize_axes=False))
                source.write_text(json.dumps(document))
                with patch('lqcdweave.contraction.planning.api.find_contraction_path',
                           side_effect=AssertionError('migration must not search')):
                    migrate_plan_archive(source, target)
                    restored = archive.load_contraction_plan(target, expected=calculation.expanded,
                                                             hardware=payload['hardware'])
                fixture._assert_values_equal(replace(calculation, path=restored).run().value,
                                             calculation.run().value)

    def test_storage_conversion_keeps_shared_block_table(self):
        from correlator.plan_archive.test_storage_schedule_archive import ScheduleArchiveTests
        from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule
        plans, shapes, hardware, expected, blocks, preparations = ScheduleArchiveTests()._schedule_inputs()
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'old.json', Path(directory)/'new.json'
            save_storage_schedule(source, plans, shapes=shapes, hardware=hardware)
            document = json.loads(source.read_text())
            document['version'] = 4
            document['payload'] = legacy_records(document['payload'])
            for problem in document['payload']['semantics']:
                for term in problem['terms']:
                    pattern = term['subscripts']
                    term['subscripts'] = ','.join(legacy_labels(axes) for axes in pattern['inputs']) + '->' + legacy_labels(pattern['output'])
            source.write_text(json.dumps(document))
            migrate_plan_archive(source, target)
            restored, _ = load_storage_schedule(target, shapes=shapes, hardware=hardware,
                                               expected=expected, blocks=blocks, preparations=preparations)
            for before, after in zip(plans, restored, strict=True):
                self.assertEqual(before.state.actions, after.state.actions)
                self.assertEqual(before.events, after.events)
                self.assertEqual(blocks, after.blocks)

    def test_old_matrix_cfg_is_rejected_without_migration_or_file_changes(self):
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'old.json', Path(directory)/'new.json'
            save_cfg_plan(source, plan)
            document = json.loads(source.read_text())
            document['version'] = 5
            document['payload'] = {'plan': legacy_records(encode_plan_value(plan))}
            source.write_text(json.dumps(document))
            before = source.read_bytes()
            with self.assertRaisesRegex(PlanArchiveError, 'generate a new plan'):
                load_cfg_plan(source)
            with self.assertRaisesRegex(PlanArchiveError, 'no migrator'):
                migrate_plan_archive(source, target)
            self.assertEqual(source.read_bytes(),before)
            self.assertFalse(target.exists())

    def test_new_records_do_not_accept_retired_strings(self):
        for record, field, spelling in (
            (Action('contract', 'z', ('x',), AxisPattern(((0,),), (0,))), 'equation', 'i->i'),
            (ContractionTerm(('x',), AxisPattern(((0,),), (0,))), 'subscripts', 'i->i'),
        ):
            encoded = encode_plan_value(record)
            encoded['value'][field] = spelling
            with self.assertRaisesRegex(PlanArchiveError, 'integer AxisPattern'):
                decode_plan_value(encoded)


if __name__ == '__main__':
    unittest.main()
