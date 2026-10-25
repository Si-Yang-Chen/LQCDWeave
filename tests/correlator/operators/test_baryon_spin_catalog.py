"""Independent finite-group and numerical oracles for the full DR spin space."""

from collections import Counter
from dataclasses import FrozenInstanceError
from importlib.resources import files
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from lqcdweave.symmetry import irrep_matrices, restrict_irrep

from lqcdweave.operators.single_hadron.api import (
    BaryonSpinOperator, Insertion, InsertionGaugeLink, Operator,
    baryon_spin_preset, baryon_spin_operators, baryon_preset,
)
from lqcdweave.operators.correlator.api import QuarkExchange
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.contraction.binding.api import Baryon


def columns(recipe):
    return np.column_stack([spin.dense().reshape(64) for spin in recipe.rows])


def field_rotations():
    """Literal DR spin frame from the fundamental lift, without gamma()/builder."""
    frame = np.array([[0,-1],[1,0]],complex)
    return {g:np.kron(np.eye(2),frame@np.asarray(m,complex)@frame.conj().T)
            for g,m in irrep_matrices((0,0,0),'G1').items()}


class BaryonSpinCatalogTests(unittest.TestCase):
    def test_numbering_metadata_multiplicities_and_complete_orthonormal_basis(self):
        table = baryon_spin_preset()
        recipes = baryon_spin_operators()
        self.assertEqual([row['id'] for row in table],list(range(64)))
        self.assertEqual(len(recipes),24)
        self.assertEqual({(row['catalog'],row['version']) for row in table},{('baryon-spin','v1')})
        expected = Counter({('G1',1,'S'):1,('H',1,'S'):2,('G1',1,'A'):1,
                            ('G1',1,'MS'):3,('G1',1,'MA'):3,('H',1,'MS'):1,('H',1,'MA'):1})
        expected.update({(irrep,-1,exchange):count for (irrep,_,exchange),count in list(expected.items())})
        self.assertEqual(Counter((r.irrep,r.parity,r.exchange_component) for r in recipes),expected)
        matrix = np.column_stack([columns(recipe) for recipe in recipes])
        np.testing.assert_allclose(matrix.conj().T@matrix,np.eye(64),atol=5e-14,rtol=0)
        np.testing.assert_allclose(matrix@matrix.conj().T,np.eye(64),atol=5e-14,rtol=0)
        for recipe in recipes:
            self.assertIsInstance(recipe,BaryonSpinOperator)
            self.assertEqual(np.asarray(recipe.coefficients).shape,(len(recipe.rows),4,4,4))
            self.assertEqual(recipe.basis,'degrand_rossi')
            for row_id,spin in zip(recipe.row_ids,recipe.rows):
                row = table[row_id]
                dense = np.zeros((4,4,4),complex)
                for term in row['terms']:
                    dense[tuple(term['spin'])] += complex(*term['coefficient'])
                np.testing.assert_array_equal(spin.dense(),dense)
            with self.assertRaises(FrozenInstanceError):
                recipe.name = 'changed'
        table[0]['terms'].clear()
        self.assertTrue(baryon_spin_preset()[0]['terms'])
        for version in ('v2','csns',None):
            with self.assertRaises(ValueError): baryon_spin_operators(version=version)
            with self.assertRaises(ValueError): baryon_spin_preset(version=version)

    def test_all_double_cubic_rotations_and_parity_have_canonical_rows(self):
        rotations = field_rotations()
        self.assertEqual(len(rotations),48)
        # DR gamma4 interchanges upper/lower spin blocks.
        g4 = np.block([[np.zeros((2,2)),np.eye(2)],[np.eye(2),np.zeros((2,2))]])
        inversion = np.kron(np.kron(g4,g4),g4)
        for recipe in baryon_spin_operators():
            basis = columns(recipe)
            target = irrep_matrices((0,0,0),recipe.irrep)
            np.testing.assert_allclose(inversion@basis,recipe.parity*basis,atol=3e-14,rtol=0)
            for g,s in rotations.items():
                action = np.kron(np.kron(s.conj(),s.conj()),s.conj())
                np.testing.assert_allclose(action@basis,basis@np.asarray(target[g],complex),atol=4e-14,rtol=0)

    def test_every_exchange_including_mixed_partner_matrices(self):
        recipes = {r.name:r for r in baryon_spin_operators()}
        for recipe in recipes.values():
            if recipe.exchange_component == 'MA': continue
            partners = [recipe] if recipe.partner is None else [recipe,recipes[recipe.partner]]
            kind = {'S':'symmetric','A':'antisymmetric','MS':'mixed'}[recipe.exchange_component]
            representation = QuarkExchange(kind)
            for p in permutations(range(3)):
                inverse = tuple(p.index(i) for i in range(3))
                matrix = np.asarray(representation.matrices[p],complex)
                for r in range(len(recipe.rows)):
                    values = np.array([partner.rows[r].dense() for partner in partners])
                    actual = np.array([value.transpose(inverse) for value in values])
                    np.testing.assert_allclose(actual,np.einsum('ba,bijk->aijk',matrix,values),atol=3e-14,rtol=0)

    def test_character_counts_are_independent_of_projector_construction(self):
        rotations = field_rotations()
        eye = np.eye(64).reshape(4,4,4,64)
        g4 = np.kron(np.array([[0,1],[1,0]]),np.eye(2))
        parity = np.kron(np.kron(g4,g4),g4)
        exchanges = {kind:QuarkExchange(kind).matrices for kind in ('symmetric','antisymmetric','mixed')}
        for eta in (1,-1):
            projector = (np.eye(64)+eta*parity)/2
            for kind,rep in exchanges.items():
                for irrep,expected in (('G1',{'symmetric':1,'antisymmetric':1,'mixed':3}[kind]),
                                       ('H',{'symmetric':2,'antisymmetric':0,'mixed':1}[kind]),('G2',0)):
                    characters = irrep_matrices((0,0,0),irrep)
                    total = 0j
                    for g,s in rotations.items():
                        rotation = np.kron(np.kron(s.conj(),s.conj()),s.conj())
                        for p,d in rep.items():
                            action = eye.transpose(*(p.index(i) for i in range(3)),3).reshape(64,64)
                            total += (complex(characters[g].trace()).conjugate()*complex(d.trace()).conjugate()
                                      *np.trace(projector@rotation@action))
                    self.assertAlmostEqual(total.real/(48*6),expected,places=11)
                    self.assertAlmostEqual(total.imag,0,places=10)

    def test_frozen_resource_rebuild(self):
        root = Path(__file__).resolve().parents[3]
        with TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable,str(root/'tools/build_baryon_spin64.py'),
                '--output-dir',directory],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),dict(rows=64,multiplets=24))
            stored = files('lqcdweave.operators.single_hadron').joinpath('presets','baryon_spin_v1.json')
            self.assertEqual((Path(directory)/'baryon_spin_v1.json').read_bytes(),stored.read_bytes())

    def test_runtime_catalog_needs_no_symbolic_optional_dependencies(self):
        code = """
import sys
sys.modules['scipy'] = None
sys.modules['lazy_object_proxy'] = None
sys.modules['sympy'] = None
from lqcdweave.operators.single_hadron.api import baryon_spin_preset, baryon_spin_operators
assert len(baryon_spin_preset()) == 64
assert sum(len(r.rows) for r in baryon_spin_operators()) == 64
"""
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_all_recipes_use_shared_insertion_operator_and_source_binding(self):
        rng = np.random.default_rng(841)
        table = baryon_preset()[:1]
        component = table[0]['component']
        spatial = rng.normal(size=(2,3,2))+1j*rng.normal(size=(2,3,2))
        reader = ArrayBaryonElemental({'vvv':spatial[None,None,None]},(component,),
            phase_convention='anchor',momentum_list=((0,0,0),))
        g4 = np.kron(np.array([[0,1],[1,0]]),np.eye(2))
        for recipe in baryon_spin_operators():
            insertion = recipe.insertion()
            for r,spin in enumerate(recipe.rows):
                operator = Operator(recipe.name,[insertion[r](0,0,0)],[.3+.7j])
                sink = Baryon(reader,operator).get(0)
                source = Baryon(reader,operator,source=True).get(0)
                expected = np.einsum('abc,ijk->abcijk',spin.dense(),spatial)*(.3+.7j)
                np.testing.assert_allclose(sink,expected,atol=1e-14)
                expected_source = np.einsum('ia,jb,kc,abcxyz->ijkxyz',g4,g4,g4,expected.conj())
                np.testing.assert_allclose(source,expected_source,atol=1e-14)

    def test_nontrivial_spatial_coupling_covers_all_target_copies(self):
        from lqcdweave.contracts.elemental import BaryonComponent
        components = tuple(BaryonComponent.paths(((i,),(),())) for i in range(3))
        spatial = [[(1,c)] for c in components]
        momenta = {0:(0,0,0)}
        rotations = field_rotations()
        t1 = irrep_matrices((0,0,0),'T1')
        selected = [r for r in baryon_spin_operators()
                    if r.name in ('spin_G1g_MS_copy2','spin_Hu_MA_copy0')]
        for recipe in selected:
            target_copies = {'G1':1,'H':1} if recipe.irrep == 'G1' else {'G1':1,'G2':1,'H':2}
            complete = []
            for target,count in target_copies.items():
                target_matrices = irrep_matrices((0,0,0),target)
                for copy in range(count):
                    insertion = Insertion.from_irreps(recipe,spatial,'T1u',target,momenta,copy_index=copy)
                    dense = np.zeros((64,3,len(insertion.rows)),complex)
                    for r,row in enumerate(insertion.rows):
                        for term in row:
                            dense[:,components.index(term.component),r] += term.coefficient*term.spin.dense().reshape(64)
                    basis = dense.reshape(192,-1)
                    complete.append(insertion.coupling_coefficients)
                    self.assertEqual(insertion.parity,-recipe.parity)
                    for g,s in rotations.items():
                        moved = np.einsum('ai,bj,ck,dl,ijklr->abcdr',s.conj(),s.conj(),s.conj(),
                            np.asarray(t1[g],complex),dense.reshape(4,4,4,3,-1)).reshape(192,-1)
                        np.testing.assert_allclose(moved,basis@np.asarray(target_matrices[g],complex),atol=1e-13,rtol=0)
            coefficients = np.column_stack(complete)
            np.testing.assert_allclose(coefficients@coefficients.conj().T,np.eye(len(recipe.rows)*3),atol=1e-13,rtol=0)
