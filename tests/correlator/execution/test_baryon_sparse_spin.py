"""Sparse-spin elimination against explicit independent mode/spin sums."""

import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contraction.execution.operations.baryon_join import contract_baryon_join
from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern
from lqcdweave.contraction.execution.operations.baryon_sparse_spin import (
    sparse_spin_pattern_supported, sparse_spin_recipe,
)
from lqcdweave.operators.single_hadron.baryon import SparseSpinTensor
from lqcdweave.operators.single_hadron import Insertion


EQUATION = 'ijk,abc,ilad,jmbe,kncf->ldmenf'
PATH = ((0, 2), (0, 1), (1, 2), (0, 1))


def timed(phase, function, **kwargs):
    return function()


class SparseBaryonSpinTests(unittest.TestCase):
    def spin(self):
        return SparseSpinTensor(((0, 1, 2, 0.7 + 0.2j), (1, 0, 2, -0.3j),
                                 (3, 2, 1, -0.4 + 0.1j)))

    def arrays(self, spin, dtype='complex128', stride=False):
        rng = np.random.default_rng(232)
        values = [spin.dense().astype(dtype)]
        for shape in ((2, 2, 2), (4, 2, 2, 1), (4, 2, 2, 1), (4, 2, 2, 1)):
            value = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
            if stride:
                storage = np.empty(tuple(2 * size for size in shape), dtype=dtype)
                view = storage[tuple(slice(None, None, 2) for _ in shape)]
                view[...] = value
                value = view
            values.append(value)
        return dict(zip('sepqr', values))

    def action(self, entries, equation=EQUATION, path=PATH):
        return SimpleNamespace(reused=False, equation=parse_axis_pattern(equation), operands=tuple('sepqr'),
                               tensor='result', kernel_path=path, sparse_spin=entries)

    def reference(self, arrays):
        # Include every spin value in dense G, independently of sparse metadata.
        s, e, p, q, r = (arrays[key] for key in 'sepqr')
        result = np.zeros((2, 1, 2, 1, 2, 1), dtype=np.complex128)
        for l, d, m, f, n, h in np.ndindex(result.shape):
            result[l, d, m, f, n, h] = sum(
                complex(s[i, j, k]) * complex(e[a, b, c]) * complex(p[i, l, a, d])
                * complex(q[j, m, b, f]) * complex(r[k, n, c, h])
                for i, j, k in np.ndindex(4, 4, 4)
                for a, b, c in np.ndindex(2, 2, 2)
            )
        return result

    def test_complex_reference_noncontiguous_precision_and_snapshot(self):
        spin = self.spin()
        for dtype in ('complex64', 'complex128'):
            for stride in (False, True):
                with self.subTest(dtype=dtype, stride=stride):
                    arrays = self.arrays(spin, dtype, stride)
                    expected = self.reference(arrays)
                    phases = []
                    def record(phase, function, **kwargs):
                        phases.append(phase)
                        return function()
                    with patch('opt_einsum.contract_expression', side_effect=AssertionError('dense G')):
                        contract_baryon_join(self.action(spin.entries), arrays, np, timed=record)
                    np.testing.assert_allclose(arrays['result'], expected,
                                               rtol=3e-6 if dtype == 'complex64' else 3e-13,
                                               atol=2e-6 if dtype == 'complex64' else 3e-13)
                    self.assertEqual(arrays['result'].dtype, np.dtype(dtype))
                    self.assertTrue(arrays['result'].flags.owndata)
                    self.assertEqual(phases, ['baryon_prepare', 'baryon_join_kernel', 'result_snapshot'])
                    stored = arrays['result'].copy()
                    for key in 'sepqr':
                        arrays[key][...] = 0
                    np.testing.assert_array_equal(arrays['result'], stored)

    def test_real_gamma_basis_source_and_zero(self):
        spins = [SparseSpinTensor(())]
        for irrep in ('G1', 'H'):
            insertion = Insertion.local_spin(irrep, 'g', {0: (0, 0, 0)})
            # Use the actual operator construction, then explicit source/basis.
            spin = insertion.rows[0][0].spin
            spins.extend((spin, spin.to_basis('dirac_pauli'), spin.source()))
        for spin in spins:
            with self.subTest(nnz=len(spin.entries), basis=spin.basis):
                arrays = self.arrays(spin)
                expected = self.reference(arrays)
                contract_baryon_join(self.action(spin.entries), arrays, np, timed=timed)
                np.testing.assert_allclose(arrays['result'], expected, rtol=3e-13, atol=3e-13)

    def test_device_sparse_result_adoption_preserves_lifetime_and_measurement(self):
        six = Insertion.local_spin('H', 'u', {0:(0,0,0)}).rows[1][0].spin
        for spin in (SparseSpinTensor(()), self.spin(), six):
            for no_search in (False, True):
                with self.subTest(nnz=len(spin.entries), no_search=no_search):
                    arrays = self.arrays(spin, stride=True)
                    expected = self.reference(arrays).transpose(5,4,3,2,1,0)
                    equation = EQUATION.split('->')[0] + '->fnemdl'
                    phases = []
                    def record(phase, function, **kwargs):
                        phases.append((phase, kwargs))
                        return function()
                    with patch('lqcdweave.contraction.execution.operations.baryon_join.copy_to_backend',
                               side_effect=AssertionError('redundant sparse snapshot')):
                        contract_baryon_join(self.action(spin.entries, equation), arrays, np,
                                            timed=record, device=True, no_search=no_search)
                    result = arrays['result']
                    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-13)
                    self.assertEqual([p for p, _ in phases], ['baryon_prepare', 'baryon_join_kernel', 'result_snapshot'])
                    self.assertEqual(phases[-1][1]['nbytes'], 0)
                    for key in 'sepqr':
                        self.assertFalse(np.shares_memory(result, arrays[key]))
                        arrays[key][...] = 0
                    contract_baryon_join(self.action(spin.entries, equation), arrays, np,
                                        timed=timed, device=True, no_search=no_search)
                    del arrays
                    gc.collect()
                    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-13)

    def test_device_dense_fallback_keeps_snapshot_for_borrowed_result(self):
        arrays = self.arrays(self.spin())
        borrowed = arrays['p'][0]
        expression = lambda *args, **kwargs: borrowed
        with patch('lqcdweave.contraction.execution.operations.baryon_kernel._join_expression',
                   return_value=expression):
            contract_baryon_join(self.action(None), arrays, np, timed=timed, device=True)
        self.assertFalse(np.shares_memory(arrays['result'], borrowed))
        before = arrays['result'].copy()
        borrowed[...] = 0
        np.testing.assert_array_equal(arrays['result'], before)

    def test_alpha_renaming_leg_permutation_and_output_order(self):
        spin = self.spin()
        arrays = self.arrays(spin, stride=True)
        reference = self.reference(arrays)
        arrays['p'] = arrays['p'].transpose(3, 2, 1, 0)
        arrays['q'], arrays['r'] = arrays['r'], arrays['q']
        equation = 'XYZ,ABC,dAlX,ZnCf,YmBe->fnemdl'
        contract_baryon_join(self.action(spin.entries, equation), arrays, np, timed=timed)
        np.testing.assert_allclose(arrays['result'], reference.transpose(5, 4, 3, 2, 1, 0),
                                   rtol=3e-13, atol=3e-13)

    def test_all_dr_parity_rows_and_sources_against_full_spin_sum(self):
        for irrep in ('G1', 'H'):
            for parity in ('g', 'u'):
                insertion = Insertion.local_spin(irrep, parity, {0:(0,0,0)})
                for row, terms in enumerate(insertion.rows):
                    for source in (False, True):
                        spin = terms[0].spin.source() if source else terms[0].spin
                        for dtype in ('complex64', 'complex128'):
                            with self.subTest(irrep=irrep, parity=parity, row=row,
                                              source=source, dtype=dtype):
                                arrays = self.arrays(spin, dtype)
                                expected = self.reference(arrays)
                                self.assertTrue(sparse_spin_pattern_supported(spin.entries))
                                # A numerically correct dense fallback would hide the
                                # missing six-entry route; require the sparse kernel.
                                with patch('opt_einsum.contract_expression',
                                           side_effect=AssertionError('DR parity row took dense path')):
                                    contract_baryon_join(self.action(spin.entries), arrays, np, timed=timed)
                                tolerance = 3e-6 if dtype == 'complex64' else 3e-13
                                np.testing.assert_allclose(arrays['result'], expected,
                                                           rtol=tolerance, atol=tolerance)
                                self.assertTrue(arrays['result'].flags.owndata)

    def test_rectangular_modes_with_permuted_negative_stride_elemental(self):
        rng = np.random.default_rng(443)
        for spin in (self.spin(), Insertion.local_spin('H', 'g', {0:(0,0,0)}).rows[1][0].spin):
            for dtype in ('complex64', 'complex128'):
                with self.subTest(nnz=len(spin.entries), dtype=dtype):
                    shapes = ((2,3,4), (4,2,2,1), (4,3,3,1), (4,2,4,1))
                    values = [spin.dense().astype(dtype)] + [
                        (rng.normal(size=s)+1j*rng.normal(size=s)).astype(dtype) for s in shapes
                    ]
                    values[1] = values[1][:,::-1,::-1].transpose(2,0,1)
                    equation = 'ijk,cab,ilad,jmbe,kncf->fnemdl'
                    expected = np.einsum(equation, *values, optimize=False)
                    arrays = dict(zip('sepqr', values))
                    with patch('opt_einsum.contract_expression', side_effect=AssertionError('dense fallback')):
                        contract_baryon_join(self.action(spin.entries, equation), arrays, np, timed=timed)
                    tolerance = 3e-6 if dtype == 'complex64' else 3e-13
                    np.testing.assert_allclose(arrays['result'], expected, rtol=tolerance, atol=tolerance)

    def test_fallback_dense_mixed_dtype_shared_labels_and_workspace(self):
        spin = self.spin()
        arrays = self.arrays(spin)
        # Explicit None means arbitrary dense G, not a request to infer sparsity.
        with patch('lqcdweave.contraction.execution.operations.baryon_kernel.contract_sparse_spin',
                   side_effect=AssertionError('sparse dispatch')):
            contract_baryon_join(self.action(None), arrays, np, timed=timed)
        np.testing.assert_allclose(arrays['result'], self.reference(arrays), rtol=3e-13)
        shapes = tuple(arrays[key].shape for key in 'sepqr')
        dtypes = ('complex128',) * 5
        self.assertIsNotNone(sparse_spin_recipe(EQUATION, shapes, dtypes, PATH))
        self.assertIsNone(sparse_spin_recipe(EQUATION, shapes, ('complex64',) + dtypes[1:], PATH))
        self.assertIsNone(sparse_spin_recipe('ijk,abc,ilad,jlbe,kncf->ldenf', shapes, dtypes, PATH))
        self.assertIsNone(sparse_spin_recipe(EQUATION, shapes, dtypes, ((0, 1, 2, 3, 4),)))
        # Tiny free extents and large b make our extra accumulators exceed
        # this particular explicit path's reservation: keep its fallback.
        small = ((4, 4, 4), (1, 64, 1), (4, 1), (4, 64), (4, 1))
        self.assertIsNone(sparse_spin_recipe('ijk,abc,ia,jb,kc->', small, dtypes, PATH))

    def test_six_entry_pairing_respects_existing_path_workspace(self):
        spin = Insertion.local_spin('H', 'g', {0:(0,0,0)}).rows[1][0].spin
        # The ordinary Ne64 path fits the two-group pack. A tight scalar-mode
        # case fits sparse execution but cannot reserve the paired final step.
        wide = ((4,4,4), (64,64,64), (4,4,64,64), (4,4,64,64), (4,4,64,64))
        self.assertTrue(sparse_spin_recipe(EQUATION, wide, ('complex128',)*5, PATH).pair_groups)
        shapes = ((4,4,4), (1,1,1), (4,1,1), (4,2,1), (4,4,1))
        equation = 'ijk,abc,ila,jmb,knc->lmn'
        self.assertFalse(sparse_spin_recipe(equation, shapes, ('complex128',)*5, PATH).pair_groups)
        rng = np.random.default_rng(112)
        values = [spin.dense()] + [rng.normal(size=s)+1j*rng.normal(size=s) for s in shapes[1:]]
        arrays = dict(zip('sepqr', values))
        expected = np.einsum(equation, *values, optimize=False)
        with patch('opt_einsum.contract_expression', side_effect=AssertionError('dense fallback')):
            contract_baryon_join(self.action(spin.entries, equation), arrays, np, timed=timed)
        np.testing.assert_allclose(arrays['result'], expected, rtol=3e-13, atol=3e-13)

    def test_direct_no_search_sparse_and_fallback(self):
        spin = self.spin()
        for entries in (spin.entries, None):
            arrays = self.arrays(spin)
            with patch('opt_einsum.contract_path', side_effect=AssertionError('path requested')), \
                 patch('opt_einsum.contract_expression', side_effect=AssertionError('compiled')):
                contract_baryon_join(self.action(entries), arrays, np, timed=timed, no_search=True)
            np.testing.assert_allclose(arrays['result'], self.reference(arrays), rtol=3e-13)

    def test_direct_dense_fallback_uses_only_explicit_binary_steps(self):
        arrays = self.arrays(self.spin())
        rng = np.random.default_rng(3)
        arrays['s'] = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
        expected = self.reference(arrays)
        calls = []
        original = np.einsum
        def einsum(*arguments, **kwargs):
            operand_count = (len(arguments) - 1) // 2
            pattern = AxisPattern(
                tuple(tuple(arguments[2 * index + 1]) for index in range(operand_count)),
                tuple(arguments[-1]),
            )
            calls.append((operand_count, pattern))
            self.assertFalse(kwargs['optimize'])
            return original(*arguments, **kwargs)
        with patch('opt_einsum.contract_path', side_effect=AssertionError('search')), \
             patch.object(np, 'einsum', side_effect=einsum):
            contract_baryon_join(self.action(None), arrays, np, timed=timed, no_search=True)
        self.assertEqual([arity for arity, _ in calls], [2, 2, 2, 2])
        final_step = parse_axis_pattern("kncf,kldcme->ldmenf").normalized()
        self.assertEqual(calls[-1][1].output, final_step.output)
        np.testing.assert_allclose(arrays['result'], expected, rtol=3e-13)

    def test_large_support_dr_sixteen_and_twentyfour_keep_dense_fallback(self):
        # Explicit old DP embeddings remain valid custom operators. Do not
        # derive historical fallback coverage from the current local_spin.
        g1 = SparseSpinTensor(((0,1,0,1/np.sqrt(2)), (1,0,0,-1/np.sqrt(2))), 'dirac_pauli')
        h = SparseSpinTensor(((0,0,1,1/np.sqrt(3)), (0,1,0,1/np.sqrt(3)),
                              (1,0,0,1/np.sqrt(3))), 'dirac_pauli')
        for original, nnz in ((g1, 16), (h, 24)):
            spin = original.to_basis('degrand_rossi')
            if nnz == 24:
                spin = spin.source()
            self.assertEqual(len(spin.entries), nnz)
            self.assertFalse(sparse_spin_pattern_supported(spin.entries))
            arrays = self.arrays(spin)
            with patch('lqcdweave.contraction.execution.operations.baryon_kernel.contract_sparse_spin',
                       side_effect=AssertionError('large-support sparse dispatch')):
                contract_baryon_join(self.action(spin.entries), arrays, np, timed=timed)
            np.testing.assert_allclose(arrays['result'], self.reference(arrays), rtol=3e-13, atol=3e-13)
        # Magnitudes do not choose a route or remove any exact nonzero.
        tiny = SparseSpinTensor(tuple((i, j, k, w * 1e-30) for i, j, k, w in self.spin().entries))
        self.assertTrue(sparse_spin_pattern_supported(tiny.entries))

    def test_structural_cache_and_reused_do_not_retain_inputs(self):
        spin = self.spin()
        arrays = self.arrays(spin)
        refs = [weakref.ref(value) for value in arrays.values()]
        contract_baryon_join(self.action(spin.entries), arrays, np, timed=timed)
        del arrays
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        action = self.action(spin.entries)
        action.reused = True
        with patch('lqcdweave.contraction.execution.operations.baryon_kernel.sparse_spin_recipe',
                   side_effect=AssertionError('prepared')):
            contract_baryon_join(action, {'result': np.array(9)}, np, timed=timed)


if __name__ == '__main__':
    unittest.main()
