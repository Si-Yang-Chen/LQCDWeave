"""Module tests split from test_diagram_expansion.py; shared numerical fixtures stay unchanged."""

from test_diagram_expansion import (
    ArrayElemental,
    ElementalComponent,
    _DiagramExpansionMetadataFixture,
    np,
)


class DiagramExpansionMetadataTests(_DiagramExpansionMetadataFixture):
    def test_array_elemental_describes_without_calling_block(self):
        component = ElementalComponent(0)
        sectors = {"v2v": np.zeros((1, 1, 2, 3, 3), dtype=np.complex128)}
        reader = ArrayElemental(sectors, (component,), phase_convention="anchor",
                                data_identity="weak-field/basis-7")
        reader.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("metadata read data"))
        metadata = reader.describe_block(component, 0, 1, "v2v", usedNe=2)
        self.assertEqual(metadata.axes, ("mode_left", "mode_right"))
        self.assertEqual(metadata.shape, (2, 2))
        self.assertEqual(metadata.data_identity, "weak-field/basis-7")
