"""Shared independent small-lattice numerical references for EV and full-chain tests.

These fixtures build gauge fields and a dense Laplacian independently of the
production implementation. Pure eigenvector unit tests live in
``tests/production/eigenvector/``.
"""

import numpy as np

try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None


class GaugeHandle:
    def __init__(self, links):
        self.data = links
        self.gauge_id = "synthetic:gauge"
        self.calls = 0

    def load(self, key):
        self.calls += 1
        return self.data


def random_gauge(latt_size, seed=17):
    lx, ly, lz, lt = latt_size
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(lt, lz, ly, lx, 4, 3, 3)) + 1j * rng.normal(
        size=(lt, lz, ly, lx, 4, 3, 3)
    )
    q, r = np.linalg.qr(raw)
    q *= np.diagonal(r, axis1=-2, axis2=-1).conj()[..., None, :] / np.abs(
        np.diagonal(r, axis1=-2, axis2=-1)
    )[..., None, :]
    q *= np.exp(-1j * np.angle(np.linalg.det(q)) / 3)[..., None, None]
    return q


def dense_laplacian(links):
    """Build the operator by site/color indices, without calling production code."""
    _, lz, ly, lx, _, _ = links.shape
    dimension = 3 * lx * ly * lz
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    def index(z, y, x, color):
        return (((z * ly + y) * lx + x) * 3 + color)

    for z in range(lz):
        for y in range(ly):
            for x in range(lx):
                point = (z, y, x)
                for color in range(3):
                    row = index(z, y, x, color)
                    matrix[row, row] += 6
                    for direction, axis in enumerate((2, 1, 0)):
                        ahead = list(point)
                        behind = list(point)
                        ahead[axis] = (ahead[axis] + 1) % (lz, ly, lx)[axis]
                        behind[axis] = (behind[axis] - 1) % (lz, ly, lx)[axis]
                        for source_color in range(3):
                            matrix[row, index(*ahead, source_color)] -= links[
                                direction, z, y, x, color, source_color
                            ]
                            matrix[row, index(*behind, source_color)] -= links[
                                direction, behind[0], behind[1], behind[2], source_color, color
                            ].conj()
    return matrix
