"""Exact neutral coefficient structures in existing scene/action records."""

from dataclasses import FrozenInstanceError
import unittest

from lqcdweave.contracts.correlator.api import (
    Action, BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
)


class SparseSpinContractTests(unittest.TestCase):
    def records(self, entries):
        return (
            Action("baryon_join", "result", sparse_spin=entries),
            BaryonJoinSpec(0, "source", "g", "phi", "abc", "ijk", sparse_spin=entries),
        )

    def test_exact_small_nonzero_and_immutable_snapshot(self):
        entries = [[0, 1, 2, 1e-200j], [3, 2, 1, 0.5 - 0.25j]]
        action, join = self.records(entries)
        entries[0][3] = 99
        for record in (action, join):
            self.assertEqual(record.sparse_spin, ((0, 1, 2, 1e-200j), (3, 2, 1, 0.5 - 0.25j)))
            with self.assertRaises(FrozenInstanceError):
                record.sparse_spin = ()

    def test_empty_zero_structure_differs_from_unspecified_dense(self):
        for record in self.records(None):
            self.assertIsNone(record.sparse_spin)
        for record in self.records(()):
            self.assertEqual(record.sparse_spin, ())

    def test_declared_structure_requires_four_spin_states(self):
        specs = {
            "g": TensorSpec((2, 2, 2), "complex128"),
            "phi": TensorSpec((1, 1, 1), "complex128"),
            **{key: TensorSpec((2, 1), "complex128") for key in ("p0", "p1", "p2")},
        }
        term = ContractionTerm(("p0", "p1", "p2"), "ai,bj,ck->")
        sources = {key: SourceSpec(key, 0) for key in specs}
        with self.assertRaisesRegex(ValueError, "shape"):
            ContractionProblem(
                specs, (term,), sources,
                baryon_joins=(BaryonJoinSpec(0, "source", "g", "phi", "abc", "ijk", sparse_spin=()),),
            )
        # The pre-existing arbitrary-extent dense ABI remains valid.
        ContractionProblem(
            specs, (term,), sources,
            baryon_joins=(BaryonJoinSpec(0, "source", "g", "phi", "abc", "ijk"),),
        )

    def test_rejects_noncanonical_or_nonfinite_entries(self):
        invalid = (
            ((4, 0, 0, 1),), ((True, 0, 0, 1),), ((0.0, 0, 0, 1),),
            ((0, 0, 0, float("nan")),), ((0, 0, 0, float("inf")),),
            ((0, 0, 0, 0),), ((0, 0, 0, 1), (0, 0, 0, 2)),
            ((3, 0, 0, 1), (0, 0, 0, 2)), ((0, 0, 1),),
        )
        for entries in invalid:
            with self.subTest(entries=entries):
                for constructor in (
                    lambda: Action("baryon_join", "result", sparse_spin=entries),
                    lambda: BaryonJoinSpec(0, "source", "g", "phi", "abc", "ijk", sparse_spin=entries),
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        constructor()
