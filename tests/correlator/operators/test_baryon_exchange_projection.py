"""Combined spin x spatial projection against independent coefficient tensors."""

from itertools import permutations
import pickle
import unittest

import numpy as np

from lqcdweave.contracts.elemental import BaryonComponent, elemental_operator_rows
from lqcdweave.contracts.elemental_catalog import ElementalCatalog
from lqcdweave.operators.correlator.api import QuarkExchange
from lqcdweave.operators.single_hadron.api import (
    Insertion, InsertionTerm, Operator, SparseSpinTensor, baryon_preset, baryon_spin_operators,
)
from lqcdweave.production.elemental.api import BaryonElementalGenerator
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.contraction.binding.api import Baryon
from test_baryon_elemental import determinant_reference
from production.elemental.test_elemental import fixture


def dense_rows(insertion, paths):
    result = np.zeros((64,len(paths),len(insertion.rows)),complex)
    definitions = dict(elemental_operator_rows(insertion.operator_table,rank=3))
    for r,row in enumerate(insertion.rows):
        for term in row:
            for triple,z in definitions[term.component]:
                index = paths.index(tuple(p.steps for p in triple))
                result[:,index,r] += term.coefficient*z*term.spin.dense().reshape(64)
    return result


class BaryonExchangeProjectionTests(unittest.TestCase):
    def setUp(self):
        self.table = baryon_preset().select(('paths/0,1,2',))
        self.seed = Insertion.from_rows([[InsertionTerm(self.table[0]['component'],
            spin=SparseSpinTensor(((0,1,2,.7+.3j),)))]],{0:(0,0,0)})

    def expected_tensor(self):
        _,links,vectors,_ = fixture()
        spatial = determinant_reference(links,(vectors,)*3,((),(0,),(1,)),0,(0,0,0))
        spin = self.seed.rows[0][0].spin.dense()
        return np.einsum('abc,ijk->aibjck',spin,spatial).reshape(12,12,12)

    def actual_tensor(self, insertion):
        _,links,vectors,_ = fixture()
        values = BaryonElementalGenerator(links,vectors,{0:(0,0,0)},insertion.operator_table).calc_path_blocks(0)['vvv']
        reader = ArrayBaryonElemental({'vvv':values[:,:,None]},
            tuple(row['component'] for row in insertion.operator_table),phase_convention='anchor',momentum_list=((0,0,0),))
        operator = Operator('projected',[insertion[0](0,0,0)],[1])
        value = Baryon(reader,operator).get(0)
        return value.transpose(0,3,1,4,2,5).reshape(12,12,12)

    def test_shared_matrix_contract_preserves_legacy_symbolic_pickle_path(self):
        from lqcdweave.contracts.quark_exchange import QuarkExchange as Shared
        self.assertIs(QuarkExchange,Shared)
        for rep in (Shared('all'),Shared.pair(slots=(0,2)),Shared().tensor_product(Shared())):
            legacy = pickle.dumps(rep,protocol=2).replace(
                b'lqcdweave.contracts.quark_exchange\nQuarkExchange\n',
                b'lqcdweave.operators.correlator.quark_exchange_matrix\nQuarkExchange\n')
            self.assertEqual(pickle.loads(legacy),rep)

    def test_scalar_projection_matches_literal_full_tensor_and_is_idempotent(self):
        original = self.expected_tensor()
        for target,kind in (('S','symmetric'),('A','antisymmetric')):
            projected = self.seed.project_exchange(target,operator_table=self.table)
            expected = sum(float(matrix[0,0])*original.transpose(tuple(p.index(i) for i in range(3)))
                for p,matrix in QuarkExchange(kind).elements)/6
            actual = self.actual_tensor(projected)
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=3e-13)
            repeated = projected.project_exchange(target,operator_table=projected.operator_table)
            np.testing.assert_allclose(self.actual_tensor(repeated),expected,atol=2e-13,rtol=3e-13)
            self.assertEqual(projected.exchange_multiplicity,1)
            self.assertEqual(projected.exchange_irrep,target)

    def test_mixed_copies_and_all_six_exchange_matrices_match_literal_tensor(self):
        original = self.expected_tensor()
        representation = QuarkExchange()
        for copy in range(2):
            pair = self.seed.project_exchange('mixed',operator_table=self.table,copy_index=copy)
            actual = np.array([self.actual_tensor(partner) for partner in pair])
            for row in range(2):
                expected = sum(complex(matrix[row,copy])*original.transpose(tuple(p.index(i) for i in range(3)))
                    for p,matrix in representation.elements)/3
                np.testing.assert_allclose(actual[row],expected,atol=2e-13,rtol=3e-13)
                self.assertEqual(pair[row].exchange_multiplicity,2)
                self.assertEqual(pair[row].exchange_component,('MS','MA')[row])
            self.assertEqual(pair[0].operator_table,pair[1].operator_table)
            for p,matrix in representation.elements:
                moved = np.array([value.transpose(tuple(p.index(i) for i in range(3))) for value in actual])
                np.testing.assert_allclose(moved,np.einsum('ba,bijk->aijk',np.asarray(matrix,complex),actual),atol=2e-13)

    def test_s2_supports_all_three_pair_positions(self):
        original = self.expected_tensor()
        for slots in ((0,1),(0,2),(1,2)):
            for target,sign in (('S',1),('A',-1)):
                projected = self.seed.project_exchange(target,operator_table=self.table,slots=slots)
                expected = (original+sign*original.swapaxes(*slots))/2
                np.testing.assert_allclose(self.actual_tensor(projected),expected,atol=2e-13)
                self.assertEqual(projected.exchange_slots,slots)

    def test_complex_seed_copies_are_orthogonalized_without_normalizing_amplitudes(self):
        table = baryon_preset().select(('paths/0,1,2','paths/0,2,1'))
        seed = Insertion.from_rows([[
            InsertionTerm(table[0]['component'],spin=SparseSpinTensor(((0,1,2,1),))),
            InsertionTerm(table[1]['component'],spin=SparseSpinTensor(((0,2,1,.3+.4j),))),
        ]],{0:(0,0,0)})
        def formal(insertion,spatial_table):
            result = np.zeros((4,7,4,7,4,7),complex)
            definitions = dict(elemental_operator_rows(spatial_table,rank=3))
            for term in insertion.rows[0]:
                for paths,z in definitions[term.component]:
                    labels = [0 if not path.steps else path.steps[0]+1 for path in paths]
                    result[:,labels[0],:,labels[1],:,labels[2]] += term.coefficient*z*term.spin.dense()
            return result.reshape(28,28,28)
        original = formal(seed,table)
        def unit(a,b):
            return sum(complex(matrix[a,b])*(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))*
                original.transpose(tuple(p.index(i) for i in range(3)))
                for p,matrix in QuarkExchange().elements)/3
        first,second = unit(0,0),unit(0,1)
        ratio = np.vdot(first,second)/np.vdot(first,first)
        self.assertGreater(abs(ratio),1e-3)
        for index in (0,1):
            pair = seed.project_exchange('mixed',operator_table=table,copy_index=index)
            for component,part in enumerate(pair):
                expected = unit(component,index)-(ratio*unit(component,0) if index else 0)
                np.testing.assert_allclose(formal(part,part.operator_table),expected,atol=3e-15,rtol=0)
        first_pair = seed.project_exchange('mixed',operator_table=table)
        repeated = first_pair[0].project_exchange('mixed',operator_table=first_pair[0].operator_table)
        self.assertEqual(repeated[0].exchange_multiplicity,1)

    def test_custom_catalog_closes_paths_and_preserves_small_coefficients(self):
        catalog = ElementalCatalog.from_rows('long-path','v1',[dict(terms=[dict(
            paths=((0,0),(),(1,2)),coefficient=[.3,.2])])],rank=3)
        seed = Insertion.from_rows([[InsertionTerm(catalog.component(0),
            spin=SparseSpinTensor(((0,1,2,1e-90),)))]],{0:(0,0,0)})
        projected = seed.project_exchange('A',operator_table=catalog.table())
        self.assertEqual(len(projected.operator_table),6)
        self.assertEqual(projected.operator_table.catalog.name,'custom')
        self.assertGreater(max(abs(z) for row in projected.rows for t in row for *_,z in t.spin.entries),0)
        again = seed.project_exchange('A',operator_table=catalog.table())
        self.assertEqual(projected.rows,again.rows)
        self.assertEqual(projected.operator_table,again.operator_table)

    def test_color_is_counted_once_and_absent_or_unsupported_targets_are_rejected(self):
        recipes = {r.name:r for r in baryon_spin_operators()}
        local = recipes['spin_Hg_S_copy0'].insertion()
        projected = local.project_exchange('A')
        np.testing.assert_allclose(projected.rows[0][0].spin.dense(),local.rows[0][0].spin.dense(),atol=1e-15)
        with self.assertRaisesRegex(ValueError,'0 independent'):
            local.project_exchange('S')
        for target,options in (('invalid',{}),('mixed',{'slots':(0,1)}),('A',{'copy_index':1})):
            with self.assertRaises(ValueError):
                self.seed.project_exchange(target,operator_table=self.table,**options)
        with self.assertRaisesRegex(ValueError,'operator_table'):
            Insertion.from_rows([[InsertionTerm(BaryonComponent(17),spin=SparseSpinTensor(((0,0,0,1),)))]],
                {0:(0,0,0)}).project_exchange('A')

    def test_projection_after_cubic_coupling_preserves_rotation_and_all_cubic_rows(self):
        from lqcdweave.symmetry import irrep_matrices
        recipes = {r.name:r for r in baryon_spin_operators()}
        table = baryon_preset()
        by_paths = {tuple(p.steps for p in terms[0][0]):component
                    for component,terms in elemental_operator_rows(table,rank=3) if len(terms)==1 and terms[0][1]==1}
        spatial = [[(1/np.sqrt(2),by_paths[((axis,),(),())]),
                    (-1/np.sqrt(2),by_paths[((axis+3,),(),())])] for axis in range(3)]
        coupled = Insertion.from_irreps(recipes['spin_G1g_MS_copy0'],spatial,'T1u','H',
            {0:(0,0,0),1:(0,0,1)})
        pair = coupled.project_exchange('mixed',operator_table=table)
        from correlator.operators.test_baryon_spin_catalog import field_rotations
        t1 = irrep_matrices((0,0,0),'T1')
        target = irrep_matrices((0,0,0),'H')
        for projected in pair:
            self.assertEqual(projected._rest_irrep,'H')
            self.assertEqual(projected.parity,-1)
            paths = tuple(tuple(p.steps for p in terms[0][0])
                          for _,terms in elemental_operator_rows(projected.operator_table,rank=3))
            coefficients = dense_rows(projected,paths)
            directions = np.concatenate((np.eye(3,dtype=int),-np.eye(3,dtype=int)))
            for g,s in field_rotations().items():
                rotation = np.asarray(t1[g],complex).real.round().astype(int)
                mapping = [next(j for j,x in enumerate(directions) if np.array_equal(rotation@v,x)) for v in directions]
                spin_rotation = np.kron(np.kron(s.conj(),s.conj()),s.conj())
                moved = np.zeros_like(coefficients)
                for i,triple in enumerate(paths):
                    transformed = tuple(tuple(mapping[step] for step in path) for path in triple)
                    moved[:,paths.index(transformed)] += spin_rotation@coefficients[:,i]
                expected = np.einsum('air,rj->aij',coefficients,np.asarray(target[g],complex))
                np.testing.assert_allclose(moved,expected,atol=3e-13,rtol=0)
            moving = projected.little_group_projection((0,0,1),'G1',0)
            self.assertIs(moving.operator_table,projected.operator_table)
            self.assertEqual(moving.exchange_component,projected.exchange_component)
            self.assertEqual(moving.exchange_matrices,projected.exchange_matrices)
            self.assertEqual(moving[0](0,0,1).momentum,1)


from test_production_dataset import _ProductionDatasetFixture


class ManagedBaryonExchangeProjectionTests(_ProductionDatasetFixture):
    def test_spin_space_then_flavor_auto_preparation_and_wick_reduction(self):
        from unittest.mock import patch
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.storage.dataset import DatasetReader
        from lqcdweave.contraction.correlator import Elemental, calculate
        from lqcdweave.operators.single_hadron.api import gamma
        from test_baryon_quark_exchange import request
        table = baryon_preset().select(('paths/1,0,0',))
        combined = Insertion.local_spin('H','g',{0:(0,0,0)},table[0]['component'])
        projected = combined.project_exchange('A',operator_table=table)
        self.assertEqual(len(projected.operator_table),3)
        with DatasetReader(self.root/'gauge',cfg='cfg') as gauge, DatasetReader(self.root/'basis',cfg='cfg') as basis:
            with patch.object(gauge,'read',side_effect=AssertionError('numeric read during construction')):
                reader = Elemental(gauge,basis,projected.operator_table,{0:(0,0,0)},directory=self.root/'managed')
                self.addCleanup(reader.close)
                operators = {name:Operator(name,[projected[0](0,0,0)],[weight])
                             for name,weight in (('sink',.7+.2j),('source',-.3+.4j))}
                vertices = {name:Baryon(reader,op,source=name=='source') for name,op in operators.items()}
            rng = np.random.default_rng(7181)
            data = (rng.normal(size=(3,3,4,4,3,3))+1j*rng.normal(size=(3,3,4,4,3,3)))/3
            prop = ArrayPropagator({'vsv':data},time_convention='absolute',cfg='cfg',
                basis_endpoints={('vsv',side):(self.bid,(4,1,9)) for side in ('sink','source')})
            # Independent antisymmetrization of determinant-reference space;
            # the spin row is (000+222)/sqrt(2), including its gamma4 image.
            def phi(time):
                value = determinant_reference(self.gauge,(self.vectors,)*3,((0,),(),()),time,(0,0,0))
                return sum((-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))*
                    value.transpose(tuple(p.index(i) for i in range(3))) for p in permutations(range(3)))/6
            spin_sum = sum(np.linalg.det(np.einsum('b,bmn->mn',gamma(8)[:,source],data[2,0,sink]))/2
                           for source in (0,2) for sink in (0,2))
            expected = (36*spin_sum*phi(0)[0,1,2]*phi(2)[0,1,2].conjugate()
                        *(.7+.2j)*(-.3+.4j).conjugate())
            self.assertGreater(abs(expected),1e-7)
            for structured in (False,True):
                for symmetry in (None,'antisymmetric'):
                    planned = calculate(request(symmetry,structured=structured,flavor='uuu'),
                        vertex_map=vertices,propagator_map={'S^q':prop},configuration='cfg',
                        time_axes={'sink':0,'source':2},execute=False,optimize='none')
                    self.assertIn('input_preparation',planned.prediction)
                    result = planned.run()
                    np.testing.assert_allclose(result.value.item(),expected,atol=3e-10,rtol=3e-12)
