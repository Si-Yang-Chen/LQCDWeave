"""One managed elemental workflow for baryon catalog recipes and contraction."""

from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contracts.correlator.api import SourceReadRequest
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.operators.single_hadron.api import baryon_preset, Operator, OperatorTerm, SparseSpinTensor
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.storage.dataset import DatasetReader
from lqcdweave.contraction.correlator import Elemental, calculate
from lqcdweave.contraction.elemental_preparation import prepare_elemental_sources
from test_production_dataset import _ProductionDatasetFixture
from test_baryon_elemental import determinant_reference
from test_baryon_quark_exchange import request as symbolic_request


class ManagedBaryonElementalTests(_ProductionDatasetFixture):
    def setUp(self):
        super().setUp()
        self.gauge_reader = DatasetReader(self.root/'gauge', cfg='cfg')
        self.basis_reader = DatasetReader(self.root/'basis', cfg='cfg')
        self.addCleanup(self.gauge_reader.close)
        self.addCleanup(self.basis_reader.close)
        self.table = baryon_preset().select(('local','distance/0,0,1','s3/0,1,2/mixed0/MS'))

    def reader(self, table=None, **options):
        reader = Elemental(self.gauge_reader, self.basis_reader, self.table if table is None else table,
                           {0:(0,0,0)}, directory=self.root/'managed', **options)
        self.addCleanup(reader.close)
        return reader

    def ensure(self, reader, times, components=None):
        components = reader.components if components is None else components
        requests = {}
        for c in components:
            for time in times:
                requests[len(requests)] = SourceReadRequest('el','elemental',(c,0,time,'vvv'),(),
                                                            reader.describe_block(c,0,time,'vvv'))
        prepare_elemental_sources(requests, {'el':reader})

    def expected(self, row, time, gauge=None):
        return sum(complex(*term['coefficient'])*determinant_reference(
            self.gauge if gauge is None else gauge, (self.vectors,)*3, term['paths'], time, (0,0,0))
                   for term in row['terms'])

    def test_metadata_only_and_three_recipe_types_match_independent_determinants(self):
        with patch.object(self.gauge_reader,'read',side_effect=AssertionError('numeric read')):
            reader = self.reader()
            for component in reader.components:
                metadata = reader.describe_block(component,0,2,'vvv',usedNe=(2,3,1))
                self.assertEqual(metadata.shape,(2,3,1))
                self.assertEqual(metadata.axis_bindings['mode0'],(self.bid,(4,1)))
        self.assertFalse((self.root/'managed').exists())
        with patch.object(reader,'_produce',wraps=reader._produce) as produce:
            self.ensure(reader,(2,2))
            self.assertEqual(produce.call_count,3)
        for row in self.table:
            value = reader.block(row['component'],0,2,'vvv')
            np.testing.assert_allclose(value,self.expected(row,2),atol=3e-12,rtol=3e-12)
            np.testing.assert_allclose(reader.block(row['component'],0,2,'vvv',usedNe=(2,3,1)),
                                       value[:2,:3,:1],atol=3e-12)

    def test_reuse_extension_recovery_and_memory_rejection(self):
        table = self.table[:1]
        self.ensure(self.reader(table),(0,))
        second = self.reader(table)
        original = second._produce
        def interrupted(c,m,s,t,xp):
            if t == 2:
                raise RuntimeError('interrupted')
            return original(c,m,s,t,xp)
        with patch.object(second,'_produce',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                self.ensure(second,(0,1,2))
        with patch.object(second,'_produce',wraps=original) as produce:
            self.ensure(second,(0,1,2))
            self.assertEqual([call.args[3] for call in produce.call_args_list],[2])
        third = self.reader(table)
        with patch.object(third,'_produce',side_effect=AssertionError('recomputed')):
            self.ensure(third,(1,2))
        unprepared = self.reader(self.table[1:2])
        c = unprepared.components[0]
        request = SourceReadRequest('el','elemental',(c,0,0,'vvv'),(),unprepared.describe_block(c,0,0,'vvv'))
        with patch.object(self.gauge_reader,'read',side_effect=AssertionError('numeric read')):
            with self.assertRaisesRegex(MemoryError,'preparation'):
                prepare_elemental_sources({0:request},{'el':unprepared},host_limit=1)

    def test_three_ordered_bases_and_shared_meson_stout(self):
        from lqcdweave.operators.single_hadron.api import meson55_preset
        from lqcdweave.production.gauge import stout_smear
        import lqcdweave.production.gauge as gauge_module
        doc = self.bspec.document
        doc['axes'][1]['coordinates'] = [11, 7]
        other_vectors = (.6-.2j)*self.vectors[:,:2]+.13
        other_id = self.save('other', other_vectors[[1,2,0]], DatasetSpec(doc))
        other = DatasetReader(self.root/'other',cfg='cfg')
        self.addCleanup(other.close)
        baryon = Elemental(self.gauge_reader,(self.basis_reader,other,self.basis_reader),
            self.table[1:2],{0:(0,0,0)},directory=self.root/'managed',stout=(1,.12))
        self.addCleanup(baryon.close)
        meson_table = meson55_preset()[:1]
        meson = Elemental(self.gauge_reader,self.basis_reader,meson_table,
            {0:(0,0,0)},directory=self.root/'managed',stout=(1,.12))
        self.addCleanup(meson.close)
        with patch.object(gauge_module,'stout_smear',wraps=stout_smear) as smear:
            self.ensure(baryon,(2,))
            c = meson.components[0]
            request = SourceReadRequest('el','elemental',(c,0,2,'v2v'),(),meson.describe_block(c,0,2,'v2v'))
            prepare_elemental_sources({0:request},{'el':meson})
            self.assertEqual(smear.call_count,1)
        c = baryon.components[0]
        metadata = baryon.describe_slab(c,0,(2,),'vvv',usedNe=(2,1,3))
        self.assertEqual(metadata.axis_bindings['mode1'],(other_id,(11,)))
        links = stout_smear(self.gauge.transpose(4,0,1,2,3,5,6),1,.12).transpose(1,2,3,4,0,5,6)
        expected = sum(complex(*term['coefficient'])*determinant_reference(
            links,(self.vectors,other_vectors,self.vectors),term['paths'],2,(0,0,0))
            for term in self.table[1]['terms'])
        np.testing.assert_allclose(baryon.block(c,0,2,'vvv'),expected,atol=3e-12)

    def test_calculate_prepares_both_endpoints_and_matches_independent_fock_determinant(self):
        reader = self.reader(self.table[:1])
        c = reader.components[0]
        weights = {'sink': .7+.2j, 'source': -.3+.4j}
        operators = {name:Operator.from_terms(name,(OperatorTerm.from_baryon(
            c,SparseSpinTensor(((0,0,0,1),)),coefficient=weight),)) for name,weight in weights.items()}
        def bind(row):
            return Baryon(reader,operators[row.hadron_name],source=row.dagger)
        rng = np.random.default_rng(7162)
        data = (rng.normal(size=(3,3,4,4,3,3))+1j*rng.normal(size=(3,3,4,4,3,3)))/3
        prop = ArrayPropagator({'vsv':data},time_convention='absolute',cfg='cfg',
            basis_endpoints={('vsv',side):(self.bid,(4,1,9)) for side in ('sink','source')})
        for structured in (False,True):
            planned = calculate(symbolic_request('antisymmetric',structured=structured),vertex_map=bind,
                propagator_map={'S^q':prop},configuration='cfg',time_axes={'sink':0,'source':2},
                execute=False,optimize='none')
            self.assertIn('input_preparation',planned.prediction)
            if not structured:
                self.assertFalse((self.root/'managed').exists())
            result = planned.run()
            kernel = np.einsum('b,bmn->mn',gamma(8)[:,0],data[2,0,0],optimize=False)
            expected = (36*np.linalg.det(kernel)*self.expected(self.table[0],0)[0,1,2]
                        *self.expected(self.table[0],2)[0,1,2].conjugate()
                        *weights['sink']*weights['source'].conjugate())
            self.assertGreater(abs(expected),1e-7)
            np.testing.assert_allclose(result.value.item(),expected,atol=3e-10,rtol=3e-12)
