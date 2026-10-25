"""Backend boundary and cache lifecycle checks without a CUDA installation."""

import unittest
from unittest.mock import Mock

import numpy as np

from lqcdweave.adapters.laplace_quda import QudaLaplaceBackend
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from tests.fixtures.laplace_reference import GaugeHandle, random_gauge, dense_laplacian


class DenseLaplaceBackend:
    def __init__(self):
        self.invalidations = 0
        self.loads = 0
        self.fail = False

    def invalidate(self):
        self.invalidations += 1

    def load(self, handle, key, sizes):
        self.loads += 1
        return handle.load(key).transpose(4, 0, 1, 2, 3, 5, 6).copy()

    def stout_smear(self, links, sizes, steps, rho):
        if self.fail:
            raise RuntimeError("backend failed")
        return links.copy()

    def solve(self, links, sizes, time, count, tol, degree, cut):
        values, vectors = np.linalg.eigh(dense_laplacian(links[:3, time]))
        return vectors[:, :count].T.reshape(count, sizes[2], sizes[1], sizes[0], 3), values[:count]


class _LaplaceBackendFixture(unittest.TestCase):
    def setUp(self):
        self.sizes = (2, 2, 2, 2)
        self.handle = GaugeHandle(random_gauge(self.sizes))
    def adapter(self):
        # Only object/array plumbing is emulated; this is not a GPU numerical test.
        adapter = QudaLaplaceBackend.__new__(QudaLaplaceBackend)
        adapter.cp = np
        adapter.invalidate()
        return adapter
    def native(self):
        links = self.handle.data.transpose(4, 0, 1, 2, 3, 5, 6).copy()
        native = Mock()
        native.latt_info.size = self.sizes
        native.latt_info.grid_coord = [0]*4
        native.lexico.side_effect = lambda: links
        native.stoutSmear.side_effect = lambda *args, **kwargs: None
        return native
