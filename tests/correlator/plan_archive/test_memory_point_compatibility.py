"""Known MemoryPoint additions keep strict old positional archive schemas."""
import unittest

from lqcdweave.storage.contraction_plans import api, compact


def historical_v1():
    # Fixed historical field names and positional payload, deliberately not
    # generated from the current dataclass or current encoder defaults.
    return {'type': 'archive.compact-v1', 'value': {
        'schemas': [['lqcdweave.contracts.correlator.storage.MemoryPoint', [
            'action_index', 'phase', 'raw_host_bytes', 'raw_device_bytes',
            'logical_host_bytes', 'logical_device_bytes', 'workspace_host_bytes',
            'workspace_device_bytes', 'reserved_host_bytes', 'reserved_device_bytes',
            'retained_host_bytes', 'retained_device_bytes', 'temporary_disk_bytes']]],
        'strings': ['contract'],
        'root': [0, 1, -1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0]}}


class MemoryPointCompatibilityTests(unittest.TestCase):
    def test_fixed_old_v1_supplies_only_known_zero_extensions(self):
        actual = api.decode_plan_value(historical_v1())
        expected = api.contract_api.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.allocation_rounding_device_bytes, 0)
        self.assertEqual(actual.packed_value_cache_device_bytes, 0)

    def test_v1_still_rejects_other_missing_defaults_unknown_order_and_counts(self):
        changes = (
            lambda value: value['schemas'][0][1].remove('reserved_host_bytes'),
            lambda value: value['schemas'][0][1].remove('phase'),
            lambda value: value['schemas'][0][1].__setitem__(2, 'unknown_device_bytes'),
            lambda value: value['schemas'][0][1].__setitem__(slice(2, 4),
                ['raw_device_bytes', 'raw_host_bytes']),
            lambda value: value['root'].pop(),
            lambda value: value['root'].append(0),
            lambda value: value['schemas'][0].append([[8, 0]]),
        )
        for change in changes:
            encoded = historical_v1()
            change(encoded['value'])
            with self.subTest(change=change), self.assertRaises(api.PlanArchiveError):
                api.decode_plan_value(encoded)

    def test_old_v2_keeps_masked_values_and_defaults(self):
        record = api.contract_api.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7,
            reserved_device_bytes=91, scalar_reduction_device_bytes=27)
        encoded = compact.encode(record)
        row = encoded['value']['schemas'][0]
        removed = row[1].index('allocation_rounding_device_bytes')
        self.assertFalse(encoded['value']['root'][1] & (1 << removed))
        row[1].pop(removed)
        row[2] = [[index-(index > removed), value] for index, value in row[2] if index != removed]
        mask = encoded['value']['root'][1]
        encoded['value']['root'][1] = (mask & ((1 << removed)-1)) | ((mask >> (removed+1)) << removed)
        self.assertEqual(api.decode_plan_value(encoded), record)

    def test_current_v2_and_shared_roundtrip_preserve_nonzero_charge(self):
        record = api.contract_api.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7,
            allocation_rounding_device_bytes=1536)
        self.assertEqual(api.decode_plan_value(compact.encode(record)), record)
        schemas, strings, roots = compact.StreamEncoder().chunk((record, record))
        restored = compact.StreamDecoder().chunk(schemas, strings, roots)
        self.assertEqual(tuple(restored), (record, record))
        self.assertIs(restored[0], restored[1])

    def test_other_branch_v1_keeps_scalar_charge_and_defaults_rounding(self):
        encoded = historical_v1()
        encoded['value']['schemas'][0][1].extend([
            'packed_value_cache_device_bytes', 'scalar_reduction_device_bytes'])
        encoded['value']['root'].extend([19, 23])
        point = api.decode_plan_value(encoded)
        self.assertEqual(point.allocation_rounding_device_bytes, 0)
        self.assertEqual(point.packed_value_cache_device_bytes, 19)
        self.assertEqual(point.scalar_reduction_device_bytes, 23)

    def test_both_nonzero_buckets_add_to_device_bytes_and_roundtrip(self):
        point = api.contract_api.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7,
            allocation_rounding_device_bytes=1536, scalar_reduction_device_bytes=2048)
        self.assertEqual(point.device_bytes, 3+5+7+1536+2048)
        self.assertEqual(api.decode_plan_value(compact.encode(point)), point)


if __name__ == '__main__':
    unittest.main()
