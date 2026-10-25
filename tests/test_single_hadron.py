"""Stored elemental consumption, coefficient conventions and endpoint safety."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contracts.elemental import ElementalComponent, SparsePointBlock
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron import (
    Operator, OperatorTerm, OperatorPath, OperatorDisplacement, InsertionRowMom,
    Insertion, InsertionGaugeLink, GammaName, DerivativeName, ProjectionName,
    SpectralProfile, gamma,
    LocalVectorCurrent, LocalAxialCurrent, ConservedVectorCurrent,
)


class _SingleHadronFixture(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.sectors = {}
        for sector, shape in {"v2v":(2,2), "v2p":(2,3,3), "p2v":(3,3,2), "p2p":(3,3,3,3)}.items():
            self.sectors[sector] = rng.normal(size=(2,2,3,*shape)) + 1j*rng.normal(size=(2,2,3,*shape))
        self.components = (ElementalComponent("row0"), ElementalComponent("row1"))
        self.data = ArrayElemental(self.sectors, self.components, phase_convention="midpoint", cfg="cfg")


class SingleHadronTests(_SingleHadronFixture):
    def test_memmap_axis_order_and_missing_coverage(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"elemental.npy"
            np.save(path,self.sectors["v2v"].transpose(2,0,1,3,4))
            data = ArrayElemental.from_npy({"v2v":path},self.components,
                    axis_order=("time","component","momentum"),phase_convention="midpoint")
            self.assertFalse(data.sectors["v2v"].flags.writeable)
            np.testing.assert_array_equal(data.block(self.components[1],1,2,"v2v"),self.sectors["v2v"][1,1,2])
            with self.assertRaises(KeyError):
                data.block(ElementalComponent.path(()),0,0,"v2v")
            data.close()
        op = Operator.from_terms("o",[OperatorTerm(0,self.components[0],0)])
        with self.assertRaisesRegex(ValueError,"phase"):
            Meson(self.data,op,phase_convention="none").get(0)


if __name__ == "__main__":
    unittest.main()
