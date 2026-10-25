"""Sparse baryon vertices checked independently against dense spin algebra."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.operators.single_hadron import (
    Insertion, InsertionTerm, Operator, OperatorTerm, SparseSpinTensor, gamma)
from lqcdweave.contraction.binding.api import Baryon


class _BaryonFixture(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(182)
        self.component = BaryonComponent.paths()
        self.data = self.rng.normal(size=(1, 1, 2, 2, 3, 4)) + 1j*self.rng.normal(size=(1, 1, 2, 2, 3, 4))
        self.reader = ArrayBaryonElemental({"vvv": self.data}, [self.component], cfg="c")
    def operator(self, spin):
        insertion = Insertion.from_rows(
            [[InsertionTerm(self.component, spin=spin)]], {0: (0, 0, 0)},
        )
        return Operator("test", [insertion[0](0, 0, 0)], [1+2j])


class BaryonTests(_BaryonFixture):
    def test_npy_and_owned_factors(self):
        spin = SparseSpinTensor([(0,1,2,1), (2,3,0,1j)])
        with TemporaryDirectory() as directory:
            path = Path(directory)/"b.npy"
            np.save(path, self.data.transpose(2,0,1,3,4,5))
            with ArrayBaryonElemental.from_npy({"vvv":path}, [self.component],
                    axis_order=("time","component","momentum"), phase_convention="anchor", cfg="c") as reader:
                obj = Baryon(reader,self.operator(spin)).load("c")
                factor = next(obj.factors(0))
                expected = factor.dense().copy()
                obj.get(1)
            np.testing.assert_array_equal(factor.dense(),expected)
            with self.assertRaises(ValueError):
                obj.get(0)

    def test_validation_and_exact_cancellation(self):
        spin = SparseSpinTensor([(0,0,0,1), (0,0,0,-1), (1,2,3,1e-30)])
        self.assertEqual(len(spin.entries),1)
        op = self.operator(spin)
        with self.assertRaises(ValueError): Baryon(self.reader,op).load("wrong")
        with self.assertRaises(ValueError): Baryon(self.reader,op).load("c",usedNe=5).get(0)
        with self.assertRaises(ValueError): next(Baryon(self.reader,op).factors(0,"v2v"))
        with self.assertRaises(ValueError): BaryonComponent.paths(((6,),(),()))
        with self.assertRaises(ValueError): SparseSpinTensor([(4,0,0,1)])
        with self.assertRaises(ValueError): Insertion.local_spin("G2", "g", {0: (0, 0, 0)})


if __name__ == "__main__":
    unittest.main()
