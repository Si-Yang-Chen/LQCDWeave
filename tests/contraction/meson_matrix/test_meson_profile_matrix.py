"""Independent contractions for the copied meson profile matrix kernel."""

import unittest
from unittest import mock

import numpy as np

from lqcdweave.contraction import meson_matrix as meson_matrix_module
from lqcdweave.contraction.meson_matrix import (
    contract_meson_matrix_blocked,
    contract_meson_vertices_blocked,
    meson_matrix_from_operator_data,
)


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


class _UnimportableBackendArray(np.ndarray):
    pass


_UnimportableBackendArray.__module__ = '_missing_meson_backend_for_test_9d17'


class _RecordingNumpyModule:
    __name__ = 'numpy'

    def __init__(self):
        self.empty_calls = 0
        self.cumsum_calls = 0
        self.stack_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, *args, **kwargs):
        self.empty_calls += 1
        return np.empty(*args, **kwargs)

    def cumsum(self, *args, **kwargs):
        self.cumsum_calls += 1
        return np.cumsum(*args, **kwargs)

    def stack(self, *args, **kwargs):
        self.stack_calls += 1
        return np.stack(*args, **kwargs)


class MesonProfileMatrixTests(unittest.TestCase):
    def test_rectangular_modes_and_all_cross_profiles(self):
        rng = np.random.default_rng(183)
        backward = random_complex(rng, (4, 4, 4, 4))
        forward = random_complex(rng, (4, 4, 5, 5))
        sink = random_complex(rng, (3, 4, 4, 4, 5))
        source = random_complex(rng, (2, 4, 4, 4, 5))
        cuts_back, cuts_forward = (1, 2, 4), (1, 3, 5)
        for strategy in ("blocks", "weighted", "batched", "auto"):
            with self.subTest(strategy=strategy):
                result = contract_meson_vertices_blocked(
                    forward, backward, sink, source, cuts_back, cuts_forward,
                    strategy=strategy,
                )
                self.assertEqual(result.shape, (3, 2, 3, 3))
                for p, (sink_back, sink_forward) in enumerate(zip(cuts_back, cuts_forward)):
                    for q, (source_back, source_forward) in enumerate(zip(cuts_back, cuts_forward)):
                        expected = -np.einsum(
                            "ijab,sjkbc,klcd,rliad->sr",
                            backward[:, :, :source_back, :sink_back],
                            sink[:, :, :, :sink_back, :sink_forward],
                            forward[:, :, :sink_forward, :source_forward],
                            source[:, :, :, :source_back, :source_forward],
                            optimize=True,
                        )
                        np.testing.assert_allclose(result[:, :, p, q], expected,
                                                   rtol=2e-12, atol=2e-10)

    def test_numpy_subclass_does_not_select_or_import_a_backend(self):
        rng = np.random.default_rng(927)
        backward = random_complex(rng, (4, 4, 2, 2))
        forward = random_complex(rng, (4, 4, 3, 3)).view(_UnimportableBackendArray)
        sink = random_complex(rng, (1, 4, 4, 2, 3))
        source = random_complex(rng, (1, 4, 4, 2, 3))

        result = contract_meson_vertices_blocked(
            forward, backward, sink, source, (1, 2), (1, 3),
            strategy='blocks', array_module=np, backend='numpy',
        )
        expected = -np.einsum(
            'ijab,sjkbc,klcd,rliad->sr', backward, sink, forward, source, optimize=True,
        )
        np.testing.assert_allclose(result[:, :, -1, -1], expected, rtol=2e-12, atol=2e-10)

    def test_wrappers_forward_explicit_array_and_einsum_backends(self):
        rng = np.random.default_rng(611)
        ne = 2
        forward = random_complex(rng, (4, 4, ne, ne))
        backward = random_complex(rng, (4, 4, ne, ne))
        gamma = np.broadcast_to(np.eye(4), (1, 4, 4)).copy()
        elemental = random_complex(rng, (1, ne, ne))
        module = _RecordingNumpyModule()
        calls = []
        original_contract = meson_matrix_module.contract

        def recording_contract(*args, **kwargs):
            calls.append(kwargs.get('backend'))
            return original_contract(*args, **kwargs)

        with mock.patch.object(meson_matrix_module, 'contract', side_effect=recording_contract):
            contract_meson_matrix_blocked(
                forward, backward, elemental, elemental, gamma, gamma,
                (1, ne), (1, ne), strategy='blocks',
                array_module=module, backend='numpy',
            )
        self.assertTrue(calls)
        self.assertEqual(set(calls), {'numpy'})
        self.assertGreater(module.empty_calls, 0)
        self.assertGreater(module.cumsum_calls, 0)

        # The pre-injection CPU call shape remains valid with its defaults.
        legacy_result = contract_meson_matrix_blocked(
            forward, backward, elemental, elemental, gamma, gamma, (1, ne), (1, ne),
        )
        self.assertEqual(legacy_result.shape, (1, 1, 2, 2))

        module = _RecordingNumpyModule()
        calls.clear()
        operator_data = [(np.eye(4)[None], random_complex(rng, (1, 2, ne, ne)))]
        with mock.patch.object(meson_matrix_module, 'contract', side_effect=recording_contract):
            meson_matrix_from_operator_data(
                forward, backward, operator_data, 0, 1, np.eye(4), np.eye(4),
                (1, ne), (1, ne), strategy='weighted',
                array_module=module, backend='numpy',
            )
        self.assertTrue(calls)
        self.assertEqual(set(calls), {'numpy'})
        self.assertGreater(module.stack_calls, 0)
        self.assertGreater(module.empty_calls, 0)
        self.assertGreater(module.cumsum_calls, 0)

    def test_invalid_backend_and_array_module_fail_before_input_access(self):
        self.assertEqual(meson_matrix_module._resolve_strategy('auto', 'numpy'), 'blocks')
        self.assertEqual(meson_matrix_module._resolve_strategy('auto', 'cupy'), 'batched')

        for backend in (None, '', 'auto', '_missing_opt_einsum_backend_9d17'):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    contract_meson_vertices_blocked(
                        None, None, None, None, (), (), backend=backend,
                    )

        for array_module in (None, object()):
            with self.subTest(array_module=array_module):
                with self.assertRaises(ValueError):
                    contract_meson_vertices_blocked(
                        None, None, None, None, (), (), array_module=array_module,
                    )

    def test_multiterm_source_dagger_and_gamma5_reverse(self):
        rng = np.random.default_rng(39)
        ne = 4
        forward = random_complex(rng, (4, 4, ne, ne))
        backward_raw = random_complex(rng, (4, 4, ne, ne))
        gamma4 = np.diag((1, 1, -1, -1))
        gamma5 = np.diag((1, -1, 1, -1))
        operators = [
            (random_complex(rng, (2, 4, 4)), random_complex(rng, (2, 3, ne, ne))),
            (random_complex(rng, (1, 4, 4)), random_complex(rng, (1, 3, ne, ne))),
        ]
        result = meson_matrix_from_operator_data(
            forward, backward_raw, operators, 0, 2, gamma4, gamma5,
            (1, 2, ne), (1, 2, ne), strategy="blocks",
        )
        reverse = np.einsum("ii,jiba,jj->ijab", gamma5, backward_raw.conj(), gamma5)
        for sink_index, (sink_spin, sink_space) in enumerate(operators):
            for source_index, (source_spin, source_space) in enumerate(operators):
                dagger_spin = np.einsum("ij,xkj,kl->xil", gamma4,
                                        source_spin.conj(), gamma4)
                expected = 0j
                for s in range(len(sink_spin)):
                    for r in range(len(source_spin)):
                        expected -= np.einsum(
                            "ijab,jk,bc,klcd,li,ad->",
                            reverse, sink_spin[s], sink_space[s, 2],
                            forward, dagger_spin[r], source_space[r, 0].conj(),
                            optimize=True,
                        )
                np.testing.assert_allclose(result[sink_index, source_index, -1, -1],
                                           expected, rtol=2e-12, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
