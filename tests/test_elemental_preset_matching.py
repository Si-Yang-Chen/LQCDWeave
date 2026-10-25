"""Paired predefined momentum/path tables, from production through saved identity."""
import unittest
from copy import deepcopy, copy
import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.preset import ElementalNpy
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.elemental import ElementalComponent, elemental_operator_rows
from lqcdweave.contraction.binding.single_hadron import Meson
from lqcdweave.operators.single_hadron.api import (
    GammaName, InsertionGaugeLink, Operator, ProjectionName,
    gauge_link_preset, gauge_link_tables, mom_dict_to_list, momentum_recipe,
    momDict_mom2 as api_momDict_mom2, momDict_mom3,
)
from lqcdweave.operators.single_hadron import momDict_mom2 as package_momDict_mom2
from lqcdweave.production.elemental.api import ElementalGenerator
import test_elemental_projection_dataset as projection_fixture
from production.elemental.test_elemental import reference_matrix


class PresetMatchingTests(unittest.TestCase):
    setUp = projection_fixture.ElementalProjectionDatasetTests.setUp
    save = projection_fixture.ElementalProjectionDatasetTests.save
    raw = projection_fixture.ElementalProjectionDatasetTests.raw
    project = projection_fixture.ElementalProjectionDatasetTests.project

    def tables(self):
        preset = gauge_link_preset(1)
        # Production deliberately differs from preset's original irrep ordering.
        return gauge_link_tables({'T_1u-': preset['T_1u-'],
                                  'A_1g+': [preset['A_1g+'][0]]})

    def operator(self, insertions, momenta=None):
        moments = self.momenta if momenta is None else momenta
        ins = InsertionGaugeLink(GammaName.A0, 'T_1u-', 0, ProjectionName.T1,
                                 dict(enumerate(moments)), insertions)
        return Operator('x-link', [ins[0](*self.momenta[0])], [1j])

    def test_preset_counts_and_actual_momenta(self):
        for key, count in (("mom1", 1), ("mom2", 19), ("mom3", 27), ("mom9", 123), ("mom_test", 23)):
            momenta = list(momentum_recipe(key).momenta)
            self.assertEqual(len(momenta), count)
            self.assertEqual(len(set(momenta)), count)
        mom2 = list(momentum_recipe("mom2").momenta)
        mom3 = list(momentum_recipe("mom3").momenta)
        p_squared_at_most_two = {
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if x * x + y * y + z * z <= 2
        }
        self.assertEqual(mom2, mom3[:19])
        self.assertEqual(list(api_momDict_mom2.items()), list(momDict_mom3.items())[:19])
        self.assertEqual(set(mom2), p_squared_at_most_two)
        self.assertTrue(all(sum(n * n for n in momentum) <= 2 for momentum in mom2))
        self.assertEqual(api_momDict_mom2, package_momDict_mom2)
        self.assertEqual(mom_dict_to_list(api_momDict_mom2), mom2)
        self.assertEqual(set(mom3), {(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)})
        custom = {19: "1 -1 0", -6: (np.int64(0), np.int64(0), np.int64(1))}
        self.assertEqual(mom_dict_to_list(custom), [(1, -1, 0), (0, 0, 1)])
        for length,count in ((0,1),(1,7),(2,37),(3,187)):
            table,_=gauge_link_tables(gauge_link_preset(length))
            self.assertEqual(len(table),count)
        with self.assertRaises(TypeError):
            mom_dict_to_list(4)
        with self.assertRaises(ValueError):
            gauge_link_preset(4)
        with self.assertRaises(ValueError):
            gauge_link_preset(4)

    def test_produced_subset_and_reordered_tables_roundtrip(self):
        table, insertions = self.tables()
        self.paths = tuple(dict.fromkeys(path.steps for _,terms in elemental_operator_rows(table)
                                        for path,_ in terms))
        self.rows = table
        raw = self.raw()
        result = self.project(raw, 'selected-preset')
        # The frontend and the raw->projected saved pipeline use exactly one table.
        generator = ElementalGenerator(self.links, self.vectors, self.momenta, table)
        operator = self.operator(insertions)
        with self.index.open(result['datasets']['projected']) as saved:
            missing = copy(saved)
            document = saved.spec.document
            del document['definition']['rows']
            missing.spec = DatasetSpec(document)
            with self.assertRaisesRegex(ValueError, 'explicit rows'):
                DatasetElemental(missing)
            reader = DatasetElemental(saved)
            self.assertEqual(reader.components, generator.components)
            meson = Meson(reader, operator).load('cfg')
            for time in range(3):
                produced = generator.calc_path_blocks(time)['v2v']
                for ci,component in enumerate(generator.components):
                    for mi in range(2):
                        np.testing.assert_allclose(reader.block(component,mi,time,'v2v'),
                                                   produced[ci,mi],atol=2e-12)
                # Independent spatial-path oracle, preserving minus/plus and i factor.
                v=self.vectors[time].reshape(3,-1).T
                spatial=v.conj().T @ (-reference_matrix(self.links,(0,),time,self.momenta[0])
                                      +reference_matrix(self.links,(3,),time,self.momenta[0])) @ v
                np.testing.assert_allclose(meson.get(time),
                    1j*np.einsum('ab,ij->abij',np.eye(4),spatial),atol=3e-12)
            with self.assertRaisesRegex(ValueError,'momentum'):
                Meson(reader,self.operator(insertions,self.momenta[::-1])).load('cfg')
            altered=deepcopy(gauge_link_preset(1))
            altered['T_1u-'][0][0][0]['coefficient']=[1,0]
            _, wrong=gauge_link_tables({'T_1u-':altered['T_1u-']})
            with self.assertRaises(KeyError):
                Meson(reader,self.operator(wrong)).get(0)

    def test_missing_momentum_metadata_and_wrong_component_rejected(self):
        table,insertions=self.tables()
        values=np.zeros((len(table),2,3,3,3),complex)
        components=tuple(row['component'] for row in table)
        reader=ArrayElemental({'v2v':values},components,phase_convention='midpoint')
        with self.assertRaisesRegex(ValueError,'momentum'):
            Meson(reader,self.operator(insertions)).load('cfg')
        labeled=ArrayElemental({'v2v':values},components,phase_convention='midpoint',
                               momentum_list=self.momenta)
        Meson(labeled,self.operator(insertions)).load('cfg')
        with self.assertRaises(KeyError):
            labeled.block(ElementalComponent(0),0,0,'v2v')
        with self.assertRaises(ValueError):
            gauge_link_tables({'A_1g+':[[[dict(path=7,coefficient=[1,0])]]]})
        with self.assertRaises(ValueError):
            ArrayElemental({'v2v':values},components,phase_convention='midpoint',
                           momentum_list=[(0,0,0),(0,0,0)])

    def test_npy_preset_preserves_explicit_table_and_momentum_identity(self):
        table, insertions = self.tables()
        generator = ElementalGenerator(self.links, self.vectors, self.momenta, table)
        values = np.stack([generator.calc_path_blocks(t)['v2v'].copy() for t in range(3)],axis=2)
        np.save(self.root/'preset-cfg.npy', values)
        source = ElementalNpy(str(self.root/'preset-'), components=generator.components,
                             momentum_list=self.momenta)
        reader = source.load('cfg')
        self.addCleanup(reader.close)
        bound = Meson(reader,self.operator(insertions)).load('cfg')
        np.testing.assert_allclose(bound.get(1),
            1j*np.einsum('ab,ij->abij',np.eye(4),values[0,0,1]))
        with self.assertRaisesRegex(ValueError,'phase'):
            Meson(reader,self.operator(insertions),phase_convention='anchor').load('cfg')

    def test_presets_are_independent_and_old_one_link_signs_preserved(self):
        a=gauge_link_preset(1)
        self.assertEqual(a['T_1u-'][0][0],
            [dict(path=[0],coefficient=[-1,0]),dict(path=[3],coefficient=[1,0])])
        a['T_1u-'][0][0][0]['coefficient'][0]=99
        self.assertEqual(gauge_link_preset(1)['T_1u-'][0][0][0]['coefficient'][0],-1)
        # Directory versions, rather than repeated recipe payloads, distinguish rows.
        left=ElementalComponent.catalog('gauge-link','v1',0)
        right=ElementalComponent.catalog('gauge-link','v2',0)
        self.assertNotEqual(left,right)

    def test_mom2_and_mom3_presets_drive_production_and_insertion(self):
        from lqcdweave.operators.single_hadron.api import (
            Insertion, DerivativeName,
        )
        component=ElementalComponent.path(())
        for name, momentum_dict in (("mom2", api_momDict_mom2), ("mom3", momDict_mom3)):
            with self.subTest(preset=name):
                generator=ElementalGenerator(
                    self.links,
                    self.vectors,
                    operator_table=[dict(component=component,path=[],coefficient=1)],
                    momentum_dict=momentum_dict,
                )
                momenta=generator.momenta
                values=generator.calc_path_blocks(0)['v2v'].copy()[:,:,None]
                reader=ArrayElemental({'v2v':values},(component,),phase_convention='midpoint',
                                       momentum_list=momenta)
                insertion=Insertion(GammaName.PI,DerivativeName.IDEN,ProjectionName.A1,
                                    momentum_dict,momentum_identity=True)
                # The original derivative table's identity row is explicitly mapped to the raw path.
                insertion.rows[0][1][0][1]=component
                for slot in (0, len(momenta)//2, len(momenta)-1):
                    operator=Operator('pi',[insertion[0](*momenta[slot])],[1])
                    self.assertEqual(operator.terms[0].momentum,slot)
                    Meson(reader,operator).load('cfg')
                    vectors=self.vectors[0].reshape(3,-1).T
                    expected=vectors.conj().T @ reference_matrix(self.links,(),0,momenta[slot]) @ vectors
                    np.testing.assert_allclose(reader.block(component,slot,0,'v2v'),expected,atol=3e-12)

    def test_mixed_mapping_is_shared_by_production_and_source_insertion(self):
        from lqcdweave.operators.single_hadron.api import (
            DerivativeName,
            Insertion,
            gamma,
        )

        momentum_dict = {31: "1 -1 0", -8: (0, 0, 0)}
        component = ElementalComponent.path(())
        table = [dict(component=component, path=(), coefficient=1)]
        generator = ElementalGenerator(
            self.links,
            self.vectors,
            operator_table=table,
            momentum_dict=momentum_dict,
        )
        momenta = ((1, -1, 0), (0, 0, 0))
        self.assertEqual(generator.momenta, momenta)
        produced = generator.calc_path_blocks(0)["v2v"].copy()[:, :, None]
        reader = ArrayElemental(
            {"v2v": produced},
            (component,),
            phase_convention="midpoint",
            momentum_list=generator.momenta,
        )

        insertion = Insertion(
            GammaName.PI,
            DerivativeName.IDEN,
            ProjectionName.A1,
            momentum_dict,
            momentum_identity=True,
        )
        insertion.rows[0][1][0][1] = component
        coefficient = 0.3 + 0.4j
        operator = Operator(
            "pi-custom-momentum",
            [insertion[0](*momenta[0])],
            [coefficient],
        )
        self.assertEqual(operator.terms[0].momentum, 0)
        self.assertEqual(operator.terms[0].momentum_vector, momenta[0])

        vectors = self.vectors[0].reshape(3, -1).T
        direct = vectors.conj().T @ reference_matrix(
            self.links, (), 0, momenta[0]
        ) @ vectors
        np.testing.assert_allclose(reader.block(component, 0, 0, "v2v"), direct, atol=3e-12)

        for source in (False, True):
            vertex = Meson(reader, operator, source=source).load("cfg")
            spatial = direct
            spin = gamma(operator.terms[0].gamma)
            weight = coefficient
            if source:
                spin = gamma(8) @ spin.conj().T @ gamma(8)
                spatial = spatial.conj().T
                weight = weight.conjugate()
            expected = weight * np.einsum("ab,ij->abij", spin, spatial)
            np.testing.assert_allclose(vertex.get(0), expected, atol=3e-12)
