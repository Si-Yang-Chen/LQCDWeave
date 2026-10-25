"""Complete cfg archive loading is pure decoding, with no planner call."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
from correlator.execution import test_cfg_runtime as fixtures
from lqcdweave.storage.contraction_plans.api import save_cfg_plan,load_cfg_plan


class CfgArchiveTests(unittest.TestCase):
    def test_legacy_cfg_envelope_requires_offline_conversion(self):
        import json
        plan,*_=fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cfg.json'
            save_cfg_plan(path,plan)
            document=json.loads(path.read_text())
            document['version']=4
            with (patch('lqcdweave.contraction.planning.api.audit_cfg_plan',side_effect=AssertionError('audit')),
                  patch('lqcdweave.contraction.planning.api.find_contraction_path',side_effect=AssertionError('search')),
                  patch('lqcdweave.storage.contraction_plans.api.migrate_plan_bundle',side_effect=AssertionError('convert'))):
                from lqcdweave.storage.contraction_plans.api import PlanArchiveError
                with self.assertRaisesRegex(PlanArchiveError, 'generate a new plan'):
                    load_cfg_plan(json.dumps(document))

    def test_archive_restores_immutable_templates_and_packed_instances(self):
        plan,*_=fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cfg.json.gz'
            save_cfg_plan(path,plan)
            with (patch('lqcdweave.contraction.planning.api.audit_cfg_plan',side_effect=AssertionError('audit')),
                  patch('lqcdweave.contraction.planning.api.find_contraction_path',side_effect=AssertionError('search')),
                  patch('lqcdweave.contraction.planning.api.plan_scene_storage',side_effect=AssertionError('storage')),
                  patch('lqcdweave.contraction.planning.api.rebind_scene_path',side_effect=AssertionError('replay')),
                  patch('lqcdweave.contraction.planning.api.estimate_matrix_action',side_effect=AssertionError('cost')),
                  patch('lqcdweave.contraction.expansion.api.expand_correlator',side_effect=AssertionError('expand'))):
                restored=load_cfg_plan(path)
            self.assertEqual(restored.statistics,plan.statistics)
            self.assertEqual(restored.instances,plan.instances)
            self.assertEqual(restored.templates[0].plan.state.actions,plan.templates[0].plan.state.actions)
            with self.assertRaises(TypeError): restored.requests['new']=None

    def test_preparation_closure_and_specs_roundtrip_with_shared_records(self):
        from types import MappingProxyType
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        template, = plan.templates
        local_key = 'tensor:2'
        self.assertNotIn(local_key, template.plan.preparations)
        binding = dict(zip(template.keys, plan.instances[0].node_ids.array(), strict=True))
        global_key = str(int(binding[local_key]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.json'
            save_cfg_plan(path, plan)
            with (patch('lqcdweave.contraction.planning.api.build_cfg_plan',
                        side_effect=AssertionError('replan')),
                  patch('lqcdweave.contraction.planning.api.audit_cfg_plan',
                        side_effect=AssertionError('audit'))):
                restored = load_cfg_plan(path)
        restored_template, = restored.templates
        self.assertIsInstance(restored_template.preparation_closure, MappingProxyType)
        self.assertIsInstance(restored_template.preparation_specs, MappingProxyType)
        self.assertEqual(restored_template.preparation_closure,
                         template.preparation_closure)
        self.assertEqual(restored_template.preparation_specs,
                         template.preparation_specs)
        self.assertNotIn(local_key, restored_template.plan.preparations)
        self.assertIs(restored_template.preparation_specs[local_key],
                      restored.specs[global_key])
        self.assertIs(restored_template.preparation_closure[local_key].recipe.constants[0],
                      restored.preparations[global_key].recipe.constants[0])

    def test_older_template_records_default_new_fields_to_none(self):
        from lqcdweave.contracts.correlator.api import CfgTemplate
        from lqcdweave.storage.contraction_plans.api import _plan_type_maps
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        template_type = _plan_type_maps()[0][CfgTemplate]
        old_template = replace(plan.templates[0],
            preparation_closure=None, preparation_specs=None)
        old_plan = replace(plan, templates=(old_template,))
        found = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.json'
            # Keep a historical generic shared-record fixture independent of
            # the current writer's compact representation.
            from lqcdweave.storage.contraction_plans import api
            api._write_document(path, api._make_envelope('complete-cfg-plan', {
                'encoding': 'unified-device-v2',
                'plan': api._encode_plan_value(old_plan, shared={})}))
            document = json.loads(path.read_text())

            def remove_new_fields(value):
                if isinstance(value, dict):
                    if value.get('type') == 'archive.shared':
                        record = value.get('value', {}).get('record', {})
                        if record.get('type') == template_type:
                            fields = record['value']
                            if 'preparation_closure' in fields:
                                fields.pop('preparation_closure')
                                fields.pop('preparation_specs')
                                found.append(True)
                    for item in value.values():
                        remove_new_fields(item)
                elif isinstance(value, list):
                    for item in value:
                        remove_new_fields(item)

            remove_new_fields(document['payload']['plan'])
            self.assertTrue(found)
            path.write_text(json.dumps(document))
            restored = load_cfg_plan(path)
        self.assertIsNone(restored.templates[0].preparation_closure)
        self.assertIsNone(restored.templates[0].preparation_specs)

    def test_old_payload_requires_explicit_offline_completion(self):
        from lqcdweave.storage.contraction_plans.api import PlanArchiveError
        from lqcdweave.storage.contraction_plans.api import ARCHIVE_FORMAT
        import json
        with self.assertRaisesRegex(PlanArchiveError,'unsupported plan archive version'):
            load_cfg_plan(json.dumps({'format':ARCHIVE_FORMAT,'version':6,
                           'payload_type':'correlator-plan','payload':{}}))

    def test_shared_spec_and_readonly_constants_survive_roundtrip_once(self):
        from lqcdweave.contracts.correlator.api import ArrayConstant
        import numpy as np
        plan,*_=fixtures.CfgReplayTests().setup_plan()
        spec=next(iter(plan.specs.values()))
        constant=ArrayConstant.from_array(np.eye(4,dtype=np.complex128))
        distinct = replace(spec)
        self.assertIsNot(distinct, spec)
        plan=replace(plan,specs=dict(plan.specs,shared_a=spec,shared_b=spec,distinct=distinct),
            output_definition=dict(first=constant,second=constant,task=dict(partitions=[0,1,3])))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shared.json.gz';save_cfg_plan(path,plan)
            restored=load_cfg_plan(path)
        self.assertIs(restored.specs['shared_a'],restored.specs['shared_b'])
        self.assertIsNot(restored.specs['shared_a'],restored.specs['distinct'])
        self.assertEqual(restored.specs['shared_a'],restored.specs['distinct'])
        self.assertIs(restored.output_definition['first'],restored.output_definition['second'])
        self.assertFalse(restored.output_definition['first'].array().flags.writeable)
        with self.assertRaises(TypeError): restored.output_definition['task']['partitions']=(0,3)
        self.assertEqual(restored.output_definition['task']['partitions'],(0,1,3))

    def test_small_cfg_uses_checked_compact_schemas_without_changing_plan(self):
        from lqcdweave.storage.contraction_plans import api
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'small.json'
            save_cfg_plan(path, plan)
            document = json.loads(path.read_text())
            self.assertEqual(document['payload']['encoding'], 'unified-device-v2')
            self.assertEqual(document['payload']['plan']['type'], 'archive.compact-v2')
            self.assertEqual(load_cfg_plan(path), plan)
            # Schema and reference checks are still the existing strict codec.
            schema = document['payload']['plan']['value']['schemas'][0]
            schema[1][0] = 'unrecognized_field'
            with self.assertRaisesRegex(api.PlanArchiveError, 'fields'):
                load_cfg_plan(json.dumps(document))

    def test_unknown_and_cyclic_shared_references_are_rejected(self):
        from lqcdweave.storage.contraction_plans.api import _decode_plan_value,encode_plan_value,PlanArchiveError
        from lqcdweave.contracts.correlator.api import ArrayConstant
        import numpy as np
        with self.assertRaisesRegex(PlanArchiveError,'forward shared'):
            _decode_plan_value(dict(type='archive.reference',value=0),shared={})
        record=encode_plan_value(ArrayConstant.from_array(np.eye(4)))
        record['value']['shape']=dict(type='archive.reference',value=0)
        with self.assertRaisesRegex(PlanArchiveError,'forward shared'):
            _decode_plan_value(dict(type='archive.shared',value=dict(id=0,record=record)),shared={})
        with self.assertRaisesRegex(PlanArchiveError,'immutable contract'):
            _decode_plan_value(dict(type='archive.shared',value=dict(id=0,
                record=dict(type='python.list',value=[]))),shared={})


if __name__=='__main__': unittest.main()
