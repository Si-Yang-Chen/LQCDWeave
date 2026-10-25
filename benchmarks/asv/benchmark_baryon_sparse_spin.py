"""Ne64 complete baryon actions with actual G1/H spin operator structures.

Synthetic elemental/propagators isolate the atomic operation; these are not
physical-ensemble measurements. Both revisions consume the same dense G and
fixed kernel path. The old wrapper ignores sparse_spin metadata.
"""

from types import SimpleNamespace

import numpy as np

from lqcdweave.contraction.execution.operations.baryon_join import contract_baryon_join
from lqcdweave.operators.single_hadron.api import Insertion, SparseSpinTensor


_EQUATION = "ijk,abc,ilad,jmbe,kncf->ldmenf"
_PATH = ((0, 2), (0, 1), (1, 2), (0, 1))
_CASE_NAMES = (
    "H_DP_row0", "G1_DP_row0", "H_DP_row1", "G1_DP_complex_projection",
    "H_DPu_complex_projection_source", "G1_DR_row0", "H_DR_row1_source",
    "G1_DR_parity_row0", "G1_DR_parity_u_row1_source",
    "H_DR_parity_row0", "H_DR_parity_row1_source",
)


def _spin_case(name):
    # Freeze the original DP embeddings so historical ASV names compare
    # identical operators across the change to local_spin's DR recipe.
    g1 = [SparseSpinTensor(((0,1,s,1/np.sqrt(2)), (1,0,s,-1/np.sqrt(2))),
                           "dirac_pauli") for s in (0,1)]
    def h_rows(lo, hi):
        return [SparseSpinTensor(entries, "dirac_pauli") for entries in (
            ((lo,lo,lo,1),),
            ((lo,lo,hi,1/np.sqrt(3)), (lo,hi,lo,1/np.sqrt(3)), (hi,lo,lo,1/np.sqrt(3))),
            ((lo,hi,hi,1/np.sqrt(3)), (hi,lo,hi,1/np.sqrt(3)), (hi,hi,lo,1/np.sqrt(3))),
            ((hi,hi,hi,1),),
        )]
    h, hu = h_rows(0,1), h_rows(2,3)

    def mix(rows, weights):
        return SparseSpinTensor(
            ((*entry[:3], entry[3] * weight)
             for row, weight in zip(rows, weights)
             for entry in row.entries), "dirac_pauli",
        )

    cases = {
        "H_DP_row0": (h[0], "dirac_pauli", False),
        "G1_DP_row0": (g1[0], "dirac_pauli", False),
        "H_DP_row1": (h[1], "dirac_pauli", False),
        "G1_DP_complex_projection": (mix(g1, (0.5 + 0.25j, -0.3 + 0.7j)), "dirac_pauli", False),
        "H_DPu_complex_projection_source": (mix(hu[1:3], (0.2 + 0.6j, -0.8 + 0.1j)), "dirac_pauli", True),
        "G1_DR_row0": (g1[0], "degrand_rossi", False),
        "H_DR_row1_source": (h[1], "degrand_rossi", True),
    }
    new_cases = {
        "G1_DR_parity_row0": ("G1", "g", 0, False),
        "G1_DR_parity_u_row1_source": ("G1", "u", 1, True),
        "H_DR_parity_row0": ("H", "g", 0, False),
        "H_DR_parity_row1_source": ("H", "g", 1, True),
    }
    if name in new_cases:
        irrep, parity, row, source = new_cases[name]
        original = Insertion.local_spin(irrep, parity, {0:(0,0,0)}).rows[row][0].spin
        cases[name] = (original, "degrand_rossi", source)
    original, basis, source = cases[name]
    spin = original.to_basis(basis)
    return (spin.source() if source else spin), original, basis, source


def _independent_spin(original, basis, source):
    eye = np.eye(2)
    rotation = np.block([[eye, eye], [eye, -eye]]) / np.sqrt(2)
    rotation = rotation if basis != original.basis else np.eye(4)
    dense = np.einsum("abc,ai,bj,ck->ijk", original.dense(), rotation, rotation, rotation, optimize=False)
    if source:
        g4 = (np.diag([1, 1, -1, -1]) if basis == "dirac_pauli"
              else np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]))
        dense = np.einsum("abc,ia,jb,kc->ijk", dense.conj(), g4, g4, g4, optimize=False)
    return dense


def _inputs(modes, dtype, name):
    rng = np.random.default_rng(300610)
    spin, original, basis, source = _spin_case(name)
    np.testing.assert_allclose(spin.dense(), _independent_spin(original, basis, source),
                               rtol=1e-14, atol=1e-15)

    def random(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)

    elemental = random((modes,) * 3)
    if source:
        elemental = elemental.conj()
    props = tuple(random((4, 4, modes, modes)) for _ in range(3))
    arrays = dict(zip("sepqr", (spin.dense().astype(dtype), elemental, *props)))
    action = SimpleNamespace(
        reused=False, equation=_EQUATION, operands=tuple("sepqr"), tensor="result",
        kernel_path=_PATH, sparse_spin=spin.entries,
    )
    return action, arrays


def _timed(_phase, calculation, **_options):
    return calculation()


def _sample_reference(arrays, index):
    """Independent complex128 matrix transform at one selected output index."""
    l, d, m, e, n, f = index
    dense, elemental, p, q, r = (arrays[key] for key in "sepqr")
    ne = elemental.shape[0]
    elemental = elemental.astype(np.complex128)
    total = 0j
    for i, j, k in zip(*np.nonzero(dense)):
        first = p[i, l, :, d].astype(np.complex128) @ elemental.reshape(ne, -1)
        second = q[j, m, :, e].astype(np.complex128) @ first.reshape(ne, ne)
        total += complex(dense[i, j, k]) * (second @ r[k, n, :, f].astype(np.complex128))
    return total


def verify_small(modes=2):
    """Full scalar truth for every dtype/spin case; excluded from ASV timing."""
    for name in _CASE_NAMES:
        for dtype in ("complex64", "complex128"):
            action, arrays = _inputs(modes, dtype, name)
            contract_baryon_join(action, arrays, np, timed=_timed)
            result = arrays["result"]
            expected = np.zeros(result.shape, np.complex128)
            dense, elemental, p, q, r = (arrays[key] for key in "sepqr")
            for index in np.ndindex(result.shape):
                l, d, m, e, n, f = index
                total = 0j
                for i, j, k in zip(*np.nonzero(dense)):
                    for a, b, c in np.ndindex(elemental.shape):
                        total += (complex(dense[i, j, k]) * complex(elemental[a, b, c])
                                  * complex(p[i, l, a, d]) * complex(q[j, m, b, e])
                                  * complex(r[k, n, c, f]))
                expected[index] = total
            tolerance = 2e-5 if dtype == "complex64" else 2e-12
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(_sample_reference(arrays, (0, 0, 1, 1, 2, 0)),
                                       expected[0, 0, 1, 1, 2, 0], rtol=2e-12, atol=2e-12)
            assert result.flags.owndata
            assert all(not np.shares_memory(result, arrays[key]) for key in "sepqr")


class BaryonSparseSpinBenchmarks:
    params = ([64], ["complex64", "complex128"], list(_CASE_NAMES))
    param_names = ["modes", "dtype", "spin_structure"]
    repeat = 3
    rounds = 3
    number = 1
    timeout = 300

    def setup(self, modes, dtype, name):
        self.action, self.arrays = _inputs(modes, dtype, name)
        contract_baryon_join(self.action, self.arrays, np, timed=_timed)
        indices = (
            (0, 0, 0, 0, 0, 0),
            (3, modes - 1, 2, modes // 2, 1, modes - 1),
            (1, modes // 3, 3, modes - 1, 2, modes // 4),
            (2, modes // 2, 1, modes // 3, 3, modes // 2),
        )
        expected = [_sample_reference(self.arrays, index) for index in indices]
        tolerance = (3e-4, 3e-2) if dtype == "complex64" else (2e-11, 2e-9)
        np.testing.assert_allclose([self.arrays["result"][index] for index in indices], expected,
                                   rtol=tolerance[0], atol=tolerance[1])

    def time_fixed_path_action_and_owned_snapshot(self, modes, dtype, name):
        contract_baryon_join(self.action, self.arrays, np, timed=_timed)
