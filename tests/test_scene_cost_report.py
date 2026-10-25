import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_scene_storage_batches import batch
from lqcdweave.contraction.performance.api import CostProfile, predict_plan_cost
from lqcdweave.contraction.planning.api import join_storage_batches
from lqcdweave.storage.scene_cost import (
    save_cost_prediction, load_cost_prediction, save_cost_profile, load_cost_profile,
)


class CostReportTests(unittest.TestCase):
    def report(self, plans):
        return predict_plan_cost(plans, CostProfile(), workload={'cfg': 'test'},
                                    provenance={'script': 'test_scene_cost_report.py'})

    def test_shared_input_is_counted_once_and_reserves_included(self):
        plans = join_storage_batches((batch('a'), batch('b'), batch('a')))
        plans = tuple(p.with_reserves(reserved_host_bytes=16) for p in plans)
        report = self.report(plans)
        self.assertEqual(report['read_bytes'], 64)
        self.assertEqual(report['physical_reads'], 2)
        self.assertAlmostEqual(report['total_seconds'], sum(p.seconds for p in plans))
        self.assertEqual(set(report['parts_seconds']), {
            'source_preparation', 'logical_snapshot', 'dense_contraction',
            'sparse_contraction', 'baryon_contraction', 'release',
        })
        self.assertNotIn('read_seconds', report)
        self.assertNotIn('transfer_seconds', report)
        self.assertNotIn('resource_seconds', report)
        self.assertAlmostEqual(sum(report['parts_fraction'].values()), 1)
        self.assertEqual(report['peak_host_bytes'], max(p.peak_host_bytes for p in plans))
        self.assertEqual(report['calibration_status'], 'unmeasured')
        self.assertFalse(self.report((plans[0].with_reserves(reserved_host_bytes=100000),))['feasible'])
        with self.assertRaisesRegex(ValueError, 'globally joined'):
            self.report((batch('a'), batch('a')))
        with self.assertRaises(TypeError):
            CostProfile(read_call_seconds=1)

    def test_persistence_field_edits_and_failed_write_preserves_old_document(self):
        report = self.report((batch('a'),))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.json'
            save_cost_prediction(path, report)
            self.assertEqual(load_cost_prediction(path), report)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                save_cost_prediction(path, dict(report, total_seconds=float('nan')))
            self.assertEqual(path.read_bytes(), original)
            document = json.loads(path.read_text())
            document['total_seconds'] += 1
            path.write_text(json.dumps(document))
            self.assertEqual(load_cost_prediction(path)['total_seconds'], report['total_seconds'] + 1)
            document['format'] = 'unsupported'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'format'):
                load_cost_prediction(path)
            save_cost_profile(path, CostProfile())
            self.assertEqual(CostProfile(**load_cost_profile(path)), CostProfile())
            old_document = dict(report, format='easy-distillation.scene-cost-prediction')
            path.write_text(json.dumps(old_document))
            with self.assertRaisesRegex(ValueError, 'format'):
                load_cost_prediction(path)


if __name__ == '__main__':
    unittest.main()
