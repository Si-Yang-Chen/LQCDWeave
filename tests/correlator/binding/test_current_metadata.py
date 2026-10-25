"""Module tests split from test_diagram_expansion.py; shared numerical fixtures stay unchanged."""

from test_diagram_expansion import (
    ArrayElemental,
    ConservedVectorInsertion,
    Meson,
    SamplingPoolSpec,
    _DiagramExpansionMetadataFixture,
    np,
)


class DiagramExpansionMetadataTests(_DiagramExpansionMetadataFixture):
    def test_meson_current_metadata_is_header_only_and_unweighted(self):
        current = ConservedVectorInsertion(direction=3, wilson_r=1.25)
        operator = current.operator()
        components = tuple(dict.fromkeys(term.component for term in operator.terms))
        lt, ne = 8, 3
        sectors = {
            "v2v": np.zeros((len(components), 1, lt, ne, ne), dtype=np.complex128),
            "v2p": np.zeros((len(components), 1, lt, ne, 2, 3), dtype=np.complex128),
            "p2v": np.zeros((len(components), 1, lt, 2, 3, ne), dtype=np.complex128),
            "p2p": np.zeros((len(components), 1, lt, 2, 3, 2, 3), dtype=np.complex128),
        }
        pools = {
            (component, 0, time, side): "shared-sample"
            for component in components for time in range(lt)
            for side in ("bar", "field")
        }
        reader = ArrayElemental(sectors, components, phase_convention="anchor",
                                data_identity="cfg-100/basis-ordered",
                                sampling_pool_ids=pools,
                                sampling_groups={"shared-sample": SamplingPoolSpec(
                                    "shared-sample", 64, 2)})
        reader.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("metadata called getter"))
        meson = Meson(reader, operator, False, phase_convention="anchor")
        described = meson.describe_factors(2, "p2p", usedNe=2, usedNp=2)
        self.assertEqual(len(described), len(operator.terms))
        self.assertTrue(all(item.logical_sector == "p2p" for item in described))
        self.assertEqual({item.current_term_index for item in described}, {0, 1})
        self.assertTrue(all(item.current_term is not None and not item.dagger for item in described))
        self.assertTrue(all(item.point_pool_ids == {"bar": "shared-sample",
                                                    "field": "shared-sample"}
                            for item in described))
        self.assertTrue(all(item.point_pool_metadata["shared-sample"].spatial_sites == 64
                            for item in described))
        self.assertEqual({item.tensor.shape for item in described}, {(2, 3, 2, 3)})
        self.assertTrue(all("unweighted_tensor" in item.coefficient_state for item in described))
