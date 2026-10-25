"""Pure retention graph decisions never inspect files or an index."""
import unittest
from unittest.mock import patch
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.dataset import plan_dataset_retention


def record(identity,parents=(),location=None):
    return dict(data_id=identity,location=location or identity,
        spec=DatasetSpec(dict(schema=1,kind='retention-fixture',
          configuration=dict(ensemble='test',configuration='cfg',lattice=[1,1,1,1]),
          axes=[dict(name='time',coordinates=[0])],dtype='float64',bases={},
          definition={},conventions={},parents=list(parents))).document,
        source_origin={},provenance={})


class PureRetentionTests(unittest.TestCase):
    def plan(self,inventory,*,pins=(),receipts=()):
        candidates=tuple(dict(data_id=x['data_id'],location=x['location']) for x in inventory)
        with patch('builtins.open',side_effect=AssertionError('pure planner opened a file')):
            return plan_dataset_retention(inventory=inventory,revision=4,
                candidate_records=candidates,candidate_reasons={(x['data_id'],x['location']):set() for x in inventory},
                verified_receipts=receipts,expected_project_id='p',pinned_actual_ids=pins)

    def test_child_first_order_and_pinned_parent_closure(self):
        a,b,c='a'*64,'b'*64,'c'*64
        rows=[record(a),record(b,(a,)),record(c,(b,))]
        result=self.plan(rows)
        self.assertEqual([x['data_id'] for x in result['decisions']],[c,b,a])
        self.assertTrue(all(x['action']=='quarantine_and_delete' for x in result['decisions']))
        result=self.plan(rows,pins=(c,))
        self.assertTrue(all(x['action']=='retain' for x in result['decisions']))
        self.assertEqual(result['publication_revision'],4)
        self.assertEqual(result['inventory'],rows)

    def test_cycle_refused_and_receipts_protect_actual_ids(self):
        a,b='a'*64,'b'*64
        with self.assertRaisesRegex(ValueError,'cycle'): self.plan([record(a,(b,)),record(b,(a,))])
        result=self.plan([record(a)],receipts=[dict(records=[dict(data_id=a)],receipt_path='receipt')])
        self.assertEqual(result['decisions'][0]['reasons'],['project_receipt'])
