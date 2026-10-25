"""Metadata snapshots keep full provenance without repeatedly serializing it."""
import unittest

from lqcdweave.contraction.expansion.physical_inputs import PhysicalInputRegistry
from lqcdweave.contracts.correlator.api import SourcePreparation, TensorAxisMetadata, TensorDescription
from lqcdweave.contracts.dilution import DilutionSource, DilutionGroup
from lqcdweave.contraction.expansion.tensor_description import tensor_recipe_key


class InputMetadataReuseTests(unittest.TestCase):
    def test_description_key_does_not_keep_construction_cache_alive_until_gc(self):
        import gc
        import weakref
        class Cache(dict):
            pass
        description = TensorDescription(data_identity='data', selection_identity='selection', bindings={'parents': (3, 1, 2)})
        enabled = gc.isenabled()
        gc.disable()
        try:
            cache = Cache()
            reference = weakref.ref(cache)
            key = tensor_recipe_key(description, (3,), 'complex128', 0, metadata_cache=cache)
            del cache
            self.assertIsNone(reference())
            self.assertIsNotNone(key)
        finally:
            if enabled:
                gc.enable()

    def test_full_dilution_document_and_canonical_key_keep_order_and_identity(self):
        first = DilutionSource('actual-D', 'parents', (DilutionGroup((0, 3, 1, 7), ((0, 0), (0, 1))),))
        changed = DilutionSource('actual-D', 'parents', (DilutionGroup((7, 1, 3, 0), ((0, 0), (0, 1))),))
        registry = PhysicalInputRegistry(None)
        prototype = registry.describe_dilution(first)
        self.assertIs(prototype, registry.describe_dilution(first))
        other = registry.describe_dilution(changed)
        self.assertIsNot(other, prototype)
        self.assertEqual(prototype.bindings['groups'][0]['parent_modes'], (0, 3, 1, 7))
        self.assertEqual(other.bindings['groups'][0]['parent_modes'], (7, 1, 3, 0))
        cache = {}
        keys = []
        for value in (prototype, other):
            description = TensorDescription(data_identity='data', selection_identity='selected',
                bindings={'source': value.bindings}, _metadata_sources=(value,))
            expected = tensor_recipe_key(description, (2, 2), 'complex128', 0)
            cached = tensor_recipe_key(description, (2, 2), 'complex128', 0, metadata_cache=cache)
            self.assertEqual(cached, expected)
            keys.append(cached)
        self.assertNotEqual(*keys)
        self.assertIsNone(tensor_recipe_key(TensorDescription(), (2, 2), 'complex128', 0, metadata_cache=cache))

    def test_dilution_convention_is_captured_once_per_reader_and_fresh_plan(self):
        class Reader:
            calls = 0
            version = 'actual-D-v1'
            @property
            def dilution_identity(self):
                self.calls += 1
                return self.version

        reader = Reader()
        metadata = TensorAxisMetadata(('mode',), (1,), 'complex128', 'data', 'selection')

        def register():
            registry = PhysicalInputRegistry(None)
            registry.preparations['tensor'] = SourcePreparation(('block',))
            registry.block_metadata['block'] = metadata
            for time in (0, 1):
                registry.register_read('tensor', reader, 'elemental', ('component', 0, time), {})
            return registry.read_requests['block']

        first = register()
        self.assertEqual(reader.calls, 1)
        self.assertEqual(dict(first.conventions)['dilution_identity'], 'actual-D-v1')
        self.assertEqual(dict(first.aliases[0].conventions)['dilution_identity'], 'actual-D-v1')
        reader.version = 'actual-D-v2'
        second = register()
        self.assertEqual(reader.calls, 2)
        self.assertEqual(dict(second.conventions)['dilution_identity'], 'actual-D-v2')
        self.assertEqual(dict(first.conventions)['dilution_identity'], 'actual-D-v1')
