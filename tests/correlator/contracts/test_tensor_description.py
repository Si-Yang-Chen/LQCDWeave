import unittest
from dataclasses import fields, replace
from types import MappingProxyType
import numpy as np

from lqcdweave.contracts.correlator.api import TensorDescription, TensorSpec


class TensorDescriptionTests(unittest.TestCase):
    def test_validated_sources_share_only_their_frozen_metadata(self):
        parents = [0, 3, 1, 7]
        source = TensorDescription(bindings={'parents': parents, 'n': 4, 'weight': np.float64(.5)})
        mutable = {'times': [0, 12]}
        child = TensorDescription(bindings={'D': source.bindings, 'other': MappingProxyType(mutable)},
                                  _metadata_sources=(source,))
        self.assertIs(child.bindings['D'], source.bindings)
        parents.reverse()
        mutable['times'][0] = 71
        expected = {'D': {'parents': (0, 3, 1, 7), 'n': 4, 'weight': .5}, 'other': {'times': (0, 12)}}
        self.assertEqual(child.bindings, expected)
        self.assertEqual(child, replace(child))
        self.assertNotIn('_metadata_sources', {item.name for item in fields(child)})
        self.assertNotIn('_metadata_sources', vars(child))
        with self.assertRaises(TypeError):
            child.bindings['D']['parents'] = (2, 3, 4, 5)
        with self.assertRaises(TypeError):
            TensorDescription(bindings={'bad': MappingProxyType({'value': np.ones(2)})},
                              _metadata_sources=(source,))
        with self.assertRaises(TypeError):
            TensorDescription(_metadata_sources=(MappingProxyType({}),))

    def test_real_binding_is_read_only_and_separate_from_stable_semantics(self):
        actual = {"configuration": "cfg-A", "endpoint_times": [2, 3]}
        description = TensorDescription(
            axes=("spin_bar", "spin_field"), role="current-spin",
            transform=({"relative_time_offsets": [0, 1]},),
            semantics={"basis": "degrand_rossi"}, bindings=actual,
        )
        actual["endpoint_times"][0] = 9
        self.assertEqual(description.bindings["endpoint_times"], (2, 3))
        with self.assertRaises(TypeError):
            description.semantics["basis"] = "other"
        spec = TensorSpec((4, 4), "complex128", description=description)
        self.assertEqual(spec.axes, ("spin_bar", "spin_field"))
        self.assertIsNone(spec.data_identity)

    def test_binding_description_rejects_live_data_and_inconsistent_axes(self):
        for live in (np.ones((2, 2)), lambda: None, object()):
            with self.subTest(live=type(live).__name__), self.assertRaises(TypeError):
                TensorDescription(bindings={"live": live})
        with self.assertRaises(ValueError):
            TensorSpec((4, 4), "complex128", description=TensorDescription(axes=("spin",)))
        with self.assertRaises(ValueError):
            TensorDescription(axes=("point", "point"))

    def test_shape_abi_equality_does_not_prove_same_scientific_identity(self):
        first = TensorSpec((2,), "float64", description=TensorDescription(
            axes=("mode",), data_identity="basis-A", selection_identity="prefix-2"))
        second = TensorSpec((2,), "float64", description=TensorDescription(
            axes=("mode",), data_identity="basis-B", selection_identity="prefix-2"))
        self.assertEqual(first, second)  # Legacy numerical ABI equality only.
        self.assertNotEqual(first.data_identity, second.data_identity)
        self.assertNotEqual(first.description, second.description)


if __name__ == "__main__":
    unittest.main()
