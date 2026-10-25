"""Validate changed/public cfg bindings, reuse only accepted immutable identity."""
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from correlator.planning import test_cfg_plan as fixtures
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, CfgPlan, CfgOutputField, describe_coordinates,
)
from lqcdweave.contraction.planning.api import audit_cfg_plan, build_cfg_plan, iter_cfg_plan_windows
from lqcdweave.storage.contraction_plans.api import load_cfg_plan, save_cfg_plan


class CfgBindingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected, cls.preparations, cls.specs = fixtures.CfgPlanTests.dense_fixture()
        cls.points, _ = describe_coordinates(time_pairs=[{0: 0, 1: 0}, {0: 1, 1: 0}])
        cls.plan = cls.build()

    @classmethod
    def build(cls, **options):
        return build_cfg_plan(cls.selected, None, cls.points.points, {},
                              cls.preparations, cls.specs, **options)

    def test_build_and_window_statistics_validate_each_binding_only_once(self):
        validator = CfgPlan._validate_bindings
        with patch.object(CfgPlan, '_validate_bindings', autospec=True,
                          side_effect=validator) as check:
            self.build()
        self.assertEqual(check.call_count, 1)
        with patch.object(CfgPlan, '_validate_bindings', autospec=True,
                          side_effect=validator) as check:
            plans = list(iter_cfg_plan_windows(self.selected, None, self.points.points, {},
                self.preparations, self.specs, window_size=1))
        self.assertEqual(check.call_count, len(plans))
        self.assertEqual([p.statistics['window_index'] for p in plans], [0, 1])

    def test_no_batch_instances_never_decode_reused_tables(self):
        instances = tuple(replace(i, kernel_batches=()) for i in self.plan.instances)
        reused_ids = {id(i.reused) for i in instances}
        decode = ArrayConstant.array

        def without_reused_decode(table):
            if id(table) in reused_ids:
                raise AssertionError('unused decode')
            return decode(table)

        # Tensor release rows still require boundary validation even when no
        # batch needs the separate reused-Action table.
        with patch.object(ArrayConstant, 'array', autospec=True, side_effect=without_reused_decode):
            plan = replace(self.plan, instances=instances)
            changed = replace(plan, statistics={'observation': 1})
        self.assertEqual(changed.instances, instances)

    def test_statistics_and_output_changes_preserve_binding_identity(self):
        field = CfgOutputField((0,), (2, 2), '<c16', 0, 8)
        with patch.object(CfgPlan, '_validate_bindings', side_effect=AssertionError('binding rescan')):
            changed = replace(self.plan, statistics={'observation': 1})
            output = replace(changed, output_fields=(field,), output_words=8,
                             output_storage_shape=(1, 8))
            with self.assertRaisesRegex(ValueError, 'packing size'):
                replace(output, output_words=7)
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                replace(changed, peak_host_bytes=-1)
        self.assertIs(output.templates, self.plan.templates)
        self.assertIs(output.instances, self.plan.instances)

    def test_external_construction_and_archive_reload_recheck_bindings(self):
        data = {f.name: getattr(self.plan, f.name) for f in fields(CfgPlan)}
        self.assertNotIn('_binding_validation', data)
        validator = CfgPlan._validate_bindings
        with patch.object(CfgPlan, '_validate_bindings', autospec=True,
                          side_effect=validator) as check:
            CfgPlan(**data)
        self.assertEqual(check.call_count, 1)
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'cfg.json'
            save_cfg_plan(archive, self.plan)
            self.assertNotIn('_binding_validation', archive.read_text())
            with patch.object(CfgPlan, '_validate_bindings', autospec=True,
                              side_effect=validator) as check:
                restored = load_cfg_plan(archive)
            self.assertEqual(check.call_count, 1)
            self.assertEqual(restored, self.plan)

    def test_changed_instances_cannot_reuse_copied_identity_proof(self):
        first = self.plan.instances[0]
        bad = (
            (replace(first, template=len(self.plan.templates)), 'unknown template'),
            (replace(first, node_ids=ArrayConstant.from_array(np.zeros(1, dtype=np.int64))),
             'binding table'),
            (replace(first, reused=ArrayConstant.from_array(np.ones(first.reused.shape, dtype=bool))),
             'reused Action'),
        )
        for instance, message in bad:
            with self.subTest(error=message), self.assertRaisesRegex(ValueError, message):
                replace(self.plan, instances=(instance, *self.plan.instances[1:]),
                        _binding_validation=self.plan._binding_validation)

    def test_changed_templates_cannot_reuse_copied_identity_proof(self):
        template = self.plan.templates[0]
        bad = (
            (replace(template, keys=(*template.keys, 'extra')), 'binding table'),
            (replace(template, plan=replace(template.plan, kernel_batches=())), 'absent'),
        )
        for changed, message in bad:
            with self.subTest(error=message), self.assertRaisesRegex(ValueError, message):
                replace(self.plan, templates=(changed,),
                        _binding_validation=self.plan._binding_validation)

    def test_external_mutable_containers_do_not_create_stale_acceptance(self):
        templates, instances = list(self.plan.templates), list(self.plan.instances)
        accepted = replace(self.plan, templates=templates, instances=instances)
        templates.clear()
        instances.clear()
        self.assertEqual(accepted.templates, self.plan.templates)
        self.assertEqual(accepted.instances, self.plan.instances)
        template = self.plan.templates[0]
        batches = list(template.plan.kernel_batches)
        mutable = replace(self.plan, templates=(replace(template,
            plan=replace(template.plan, kernel_batches=batches)),))
        self.assertIsNone(mutable._binding_validation)
        batches.clear()
        with self.assertRaisesRegex(ValueError, 'absent'):
            replace(mutable, statistics={'observation': 2})

    def test_output_audit_reserves_only_floating_or_complex_field_masks(self):
        integer = CfgOutputField(('integer',), (100,), '<i8', 0, 100)
        for dtype in (None, '<f4', '<c16'):
            with self.subTest(dtype=dtype):
                payload = largest = 800
                selected = (integer,)
                words, finite = 100, 0
                if dtype is not None:
                    nbytes = 2 * np.dtype(dtype).itemsize
                    selected += (CfgOutputField(('numeric',), (2,), dtype, 100, nbytes // 8),)
                    payload += nbytes
                    words += nbytes // 8
                    finite = 2
                minimum = payload + words * 8 + largest + finite
                candidate = replace(self.plan, output_mode='dataset',
                    output_fields=selected, output_words=words,
                    output_storage_shape=(1, words), output_host_bytes=minimum,
                    peak_host_bytes=10**8, prefetch_host_bytes=0)
                audit_cfg_plan(candidate)
                with self.assertRaisesRegex(ValueError, 'packing or reconstruction'):
                    audit_cfg_plan(replace(candidate, output_host_bytes=minimum - 1))


if __name__ == '__main__':
    unittest.main()
