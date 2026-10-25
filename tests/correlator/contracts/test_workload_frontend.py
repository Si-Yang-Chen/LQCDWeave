"""Pure saved-frontend metadata checks at the C boundary."""

import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    TensorSpec,
    WorkloadSpec,
    validate_frontend_workload,
)


class WorkloadFrontendContractTests(unittest.TestCase):
    @staticmethod
    def _workload(*, recipes=((0, 1), (2,)), adapter="meson-simple-v1",
                  convention=None, scalar=True, reserve=96):
        keys = ("left", "middle", "right")
        equation = "i->" if scalar else "i->i"
        return WorkloadSpec(
            specs={key: TensorSpec((1,), np.complex128) for key in keys},
            terms=tuple(ContractionTerm((key,), equation) for key in keys),
            convention=adapter if convention is None else convention,
            output_groups=((0, 1), (2,)),
            metadata={
                "adapter": adapter,
                "output_recipes": recipes,
                "assembly_extra_bytes": reserve,
            },
        )

    def test_accepts_known_meson_and_generic_workloads(self):
        self.assertIsNone(validate_frontend_workload(self._workload()))

        generic = WorkloadSpec(
            specs={"input": TensorSpec((1,), np.float64)},
            terms=(ContractionTerm(("input",), "i->"),),
        )
        self.assertIsNone(validate_frontend_workload(generic))

    def test_rejects_unknown_adapter_or_mismatched_convention(self):
        with self.assertRaisesRegex(ValueError, "unsupported saved frontend"):
            validate_frontend_workload(self._workload(adapter="other-v1"))

        with self.assertRaisesRegex(ValueError, "unsupported saved frontend"):
            validate_frontend_workload(
                self._workload(convention="other-v1")
            )

    def test_rejects_recipe_coverage_nonscalar_terms_and_short_reserve(self):
        cases = (
            (self._workload(recipes=((0, 1), (2, 2))),
             "assemble every term exactly once"),
            (self._workload(scalar=False), "term outputs must be scalar"),
            (self._workload(reserve=95), "memory reserve is insufficient"),
        )
        for workload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_frontend_workload(workload)


if __name__ == "__main__":
    unittest.main()
