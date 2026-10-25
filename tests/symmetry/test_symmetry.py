"""Independent representation and projection identities for the symmetry module."""
import numpy as np
import pytest

from lqcdweave.symmetry import compose, irrep_matrices, row_subduction


@pytest.mark.parametrize("label", ["A1", "T1", "G1"])
def test_rest_representation_obeys_group_law_and_unitarity(label):
    matrices = {k: np.asarray(v, dtype=complex)
                for k, v in irrep_matrices((0, 0, 0), label).items()}
    for left, a in matrices.items():
        np.testing.assert_allclose(a.conj().T @ a, np.eye(len(a)), atol=1e-12)
        for right, b in matrices.items():
            np.testing.assert_allclose(a @ b, matrices[compose(left, right)], atol=1e-12)


def test_row_subduction_resolves_two_equivalent_copies():
    matrices = [np.asarray(v, dtype=complex)
                for v in irrep_matrices((0, 0, 0), "T1").values()]
    representation = [np.kron(m, np.eye(2)) for m in matrices]
    result = row_subduction(representation, matrices)
    basis = result["basis"]
    assert result["multiplicity"] == 2
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(basis @ basis.conj().T, result["projector"], atol=1e-12)
    for u, d in zip(representation, matrices):
        np.testing.assert_allclose(u @ basis, basis @ np.kron(d, np.eye(2)), atol=1e-12)
