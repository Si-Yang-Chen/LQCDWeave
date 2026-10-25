"""Record-level I/O checks at the QDP/production boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

from test_legacy_readers import _qdp
from production.elemental.test_elemental import Handle, fixture, reference_matrix
from lqcdweave.adapters.legacy_fields import QDPTimesliceField
from lqcdweave.production.elemental import ElementalGenerator, BaryonElementalGenerator
from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class _QDPProductionBlockFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sizes, self.gauge, self.vectors, _ = fixture()
        path = Path(self.temp.name) / "basis.mod"
        _qdp(path, {(t,n): self.vectors[t,n].astype(">c16")
                    for t in range(self.sizes[3]) for n in range(3)}, self.sizes)
        self.reader = QDPTimesliceField(str(path.parent)+"/", path.name,
            role="eigenvector", count=3, dtype=">c16").load("")


class QDPProductionBlockTests(_QDPProductionBlockFixture):
    def test_elemental_tiles_all_times_against_site_reference(self):
        paths = ((), (0,1,3), (6,0))
        momentum = (1,-1,1)
        generator = ElementalGenerator(self.sizes, Handle(self.gauge), Handle(self.reader),
            usedNe=2, paths=paths, path_version="v2", ne_block_size=1,
            momentum_list=(momentum,)).load("cfg")
        with patch.object(self.reader, "_record", wraps=self.reader._record) as read:
            for t in range(self.sizes[3]):
                actual = generator.calc_path_blocks(t)["v2v"].copy()
                for i, path in enumerate(paths):
                    endpoint = (t+path.count(6)-path.count(7)) % self.sizes[3]
                    left = self.vectors[t,:2].reshape(2,-1)
                    right = self.vectors[endpoint,:2].reshape(2,-1)
                    matrix = reference_matrix(self.gauge, path, t, momentum)
                    np.testing.assert_allclose(actual[i,0], left.conj() @ matrix @ right.T,
                                               atol=2e-12)
        self.assertEqual({c.args[0] for c in read.call_args_list}, set(range(self.sizes[3])))
        self.assertEqual({c.args[1] for c in read.call_args_list}, {0,1})

    def test_baryon_snapshot_reads_only_selected_prefix(self):
        with patch.object(self.reader, "_record", wraps=self.reader._record) as read:
            generator = BaryonElementalGenerator(self.sizes, Handle(self.gauge),
                Handle(self.reader), usedNe=(1,2,1)).load("cfg")
        self.assertEqual([c.args for c in read.call_args_list],
                         [(t,n) for t in range(self.sizes[3]) for n in range(2)])
        reference = BaryonElementalGenerator(self.sizes, Handle(self.gauge),
            Handle(self.vectors), usedNe=(1,2,1)).load("cfg")
        for t in range(self.sizes[3]):
            np.testing.assert_allclose(generator.calc_path_blocks(t)["vvv"], reference.calc_path_blocks(t)["vvv"])
