"""Module tests split from test_directed_current_contract.py; shared numerical fixtures stay unchanged."""

from test_directed_current_contract import (
    CURRENT_ELEMENTAL_SCHEMA,
    DirectedCurrentBasis,
    build_current_raw_contract,
    current_raw_cache_key,
    np,
    unittest,
    validate_current_raw_contract,
)


class DirectedCurrentContractTests(unittest.TestCase):
    def test_direction_table_and_term_index(self):
        basis = DirectedCurrentBasis.metadata()
        self.assertEqual(tuple(item["name"] for item in basis["directions"]),
                         ("+x", "+y", "+z", "-x", "-y", "-z", "+t", "-t"))
        for index in range(8):
            self.assertEqual(DirectedCurrentBasis.direction(index),
                             DirectedCurrentBasis.DIRECTIONS[index])
        self.assertEqual(DirectedCurrentBasis.index_for_term(3, "backward"), 7)
        with self.assertRaises(ValueError):
            DirectedCurrentBasis.direction(8)

    def test_raw_metadata_and_legacy_rejection(self):
        raw = {"v2v": np.ones((8, 3, 1, 2, 2), dtype=np.complex128)}
        contract = build_current_raw_contract(raw, boundary="periodic",
                                              available_ne=3, used_ne=2,
                                              momentum_count=1)
        self.assertEqual(contract["cache_identity"],
                         "e403b6ffef91dbff2985b68ec530ec206b4d40e42da29d6f29766ecaf2f2b30a")
        self.assertEqual(validate_current_raw_contract(raw, contract,
                                                       require_temporal=True), contract)
        self.assertEqual(current_raw_cache_key("cfg", contract),
                         "cfg:" + contract["cache_identity"])
        with self.assertRaisesRegex(ValueError, "cache identity"):
            validate_current_raw_contract(raw, {**contract, "boundary": "open"})
        with self.assertRaisesRegex(ValueError, "legacy spatial"):
            validate_current_raw_contract(raw, {"schema": CURRENT_ELEMENTAL_SCHEMA},
                                          require_temporal=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            altered = {"v2v": raw["v2v"].copy()}
            altered["v2v"][0, 0, 0, 0, 0] = np.nan
            validate_current_raw_contract(altered, contract)
