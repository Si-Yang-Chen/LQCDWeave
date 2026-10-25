"""Independent site-matrix/determinant checks of three-leg production."""
import unittest
from unittest.mock import patch
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from production.elemental.test_elemental import fixture, reference_matrix, Handle
from lqcdweave.production.elemental import BaryonElementalGenerator, generate_baryon_elemental
from lqcdweave.production.gauge import stout_smear, project_su3
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.single_hadron import Insertion, Operator
from lqcdweave.contraction.binding.api import Baryon

from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator


class BorrowedModes:
    """One reusable record buffer; whole-basis reads are forbidden."""

    def __init__(self, values, capacity=3):
        self.values = values
        self.Ne = values.shape[1]
        self.buffer = np.empty((capacity, *values.shape[2:]), values.dtype)
        self.reads = []

    def load(self, key):
        raise AssertionError("whole-basis load is forbidden")

    def read_modes(self, key, time, start, stop):
        self.reads.append((key,time,start,stop))
        self.buffer[:] = np.nan
        self.buffer[:stop-start] = self.values[time,start:stop]
        return self.buffer[:stop-start]


def determinant_reference(gauge, vectors, paths, t, momentum, phase=True):
    """Explicit site transport matrices and batched 3x3 determinants, not epsilon einsum."""
    fields = []
    for vector, path in zip(vectors, paths):
        matrix = reference_matrix(gauge, path, t, phase="none")
        fields.append((matrix @ vector[t].reshape(vector.shape[1], -1).T).T.reshape(vector.shape[1], -1, 3))
    lt, lz, ly, lx = gauge.shape[:4]
    z,y,x = np.indices((lz,ly,lx))
    weight = np.exp(2j*np.pi*(momentum[0]*x/lx+momentum[1]*y/ly+momentum[2]*z/lz)).ravel() if phase else np.ones(lx*ly*lz)
    out = np.empty(tuple(v.shape[0] for v in fields), complex)
    for a,b,c in np.ndindex(out.shape):
        out[a,b,c] = weight @ np.linalg.det(np.stack((fields[0][a],fields[1][b],fields[2][c]),axis=1))
    return out


class _BaryonProductionFixture(unittest.TestCase):
    def setUp(self):
        self.sizes,self.u,v,_ = fixture()
        self.vectors = (v[:,:2], (v*(.7+.2j)+.3)[:,:3], (v*.4-.1j)[:,:2])
        self.paths = (((),(),()), ((0,),(4,),(2,0)), ((0,1,3,4),(),(5,)), ((1,0),(0,1),()))
        self.momenta = ((0,0,0),(1,-1,1))
    def generator(self, **kwargs):
        settings = dict(paths=self.paths, momentum_list=self.momenta, ne_block_size=(1,2,1), spatial_block_size=5)
        settings.update(kwargs)
        return BaryonElementalGenerator(self.sizes,Handle(self.u), tuple(Handle(v) for v in self.vectors), **settings).load("c")


class BaryonProductionTests(_BaryonProductionFixture):
    def test_full_production_npy_insertion_baryon(self):
        gen = self.generator()
        with TemporaryDirectory() as directory:
            path = Path(directory)/"elemental.npy"
            out = np.lib.format.open_memmap(path,mode="w+",dtype=complex,
                shape=(len(self.paths),len(self.momenta),self.sizes[3],*gen.usedNe))
            for t in range(self.sizes[3]):
                gen.calc_path_blocks(t,write=lambda pi,mi,se,sl,v: out.__setitem__((pi,mi,t,*sl),v))
            out.flush(); out._mmap.close()
            insertion = Insertion.local_spin("H", "g", dict(enumerate(self.momenta)), gen.components[1])
            operator = Operator("H2", [insertion[1](*self.momenta[1])], [.3+.4j])
            with ArrayBaryonElemental.from_npy({"vvv":path},gen.components,
                    axis_order=("component","momentum","time"),phase_convention=gen.phase_convention,cfg="c") as reader:
                for source in (False,True):
                    vertex = Baryon(reader,operator,source).load("c")
                    for t in range(self.sizes[3]):
                        factor = next(vertex.factors(t))
                        expected = determinant_reference(self.u,self.vectors,self.paths[1],t,self.momenta[1])
                        np.testing.assert_allclose(factor.spatial,expected.conj() if source else expected,atol=2e-12)
                        self.assertEqual(factor.spin.basis,"degrand_rossi")
                        self.assertEqual(len(factor.spin.entries),6)


if __name__ == "__main__":
    unittest.main()
