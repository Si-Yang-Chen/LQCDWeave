"""Direct projections retain real inputs without inventing a raw-path parent."""
from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental_catalog import catalog_definition
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.operators.single_hadron.api import meson55_preset, momDict_meson55, mom_dict_to_list
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter


class DirectProjectedElementalTests(unittest.TestCase):
    def fixture(self):
        table = meson55_preset()
        modes = dict(data_id='actual-ordered-basis', indices=[0, 1])
        document = dict(schema=1, kind='elemental_projected', configuration=dict(
            ensemble='toy', configuration='2000', lattice=[2, 2, 2, 128]),
            axes=[dict(name='time', coordinates=[0, 1]),
                  dict(name='component', coordinates=[dict(projection=i) for i in range(55)]),
                  dict(name='momentum', coordinates=[list(p) for p in mom_dict_to_list(momDict_meson55)]),
                  dict(name='sink_mode', coordinates=[0, 1]),
                  dict(name='source_mode', coordinates=[0, 1])],
            dtype='complex128', bases=dict(sink_mode=modes, source_mode=modes),
            definition=dict(method='direct_calc_projected',
                catalog=catalog_definition(table),
                rows=[{k:v for k,v in row.items() if k != 'component'} for row in table],
                gauge_source_sha256='a'*64, input_sha256=dict(gauge='a'*64, eigenvector='b'*64),
                gauge_transform=dict(steps=20, rho=.12, dimensions='spatial'), path_boundary='periodic'),
            conventions=dict(phase='midpoint', normalization='catalog coefficients; no volume factor',
                time_coordinates=dict(domain='slot', layout='absolute', selections={
                    'time':dict(Lt=128, times=[104, 1])})), parents=['actual-ordered-basis'])
        rng = np.random.default_rng(901)
        values = rng.normal(size=(2,55,33,2,2)) + 1j*rng.normal(size=(2,55,33,2,2))
        return table, document, values

    def test_saved_direct_projection_matches_array_binding_for_new_g_rows(self):
        table, document, values = self.fixture()
        with TemporaryDirectory() as td:
            with DatasetWriter(td, DatasetSpec(document), provenance={'test':'direct projection'}) as w:
                for t in w.missing:
                    w.write(t, values[t])
                w.publish()
            array = ArrayElemental({'v2v':values.transpose(1,2,0,3,4)},
                tuple(row['component'] for row in table),
                momentum_list=mom_dict_to_list(momDict_meson55),phase_convention='midpoint')
            with DatasetReader(td,cfg='2000') as reader:
                elemental = DatasetElemental(reader)
                self.assertEqual(elemental.time_selection.times,(104,1))
                disk, reference = RecipeVertex(elemental), RecipeVertex(array)
                checked = set()
                for recipe in disk.recipes.values():
                    changed = set(recipe.kernel_ids) & {16,22,23,24}
                    if not changed - checked:
                        continue
                    row = disk.symbol(recipe.name, tag=Tag(0,0))[0]
                    actual = disk(row).load('2000')
                    expected = reference(row).load('2000')
                    for t in range(2):
                        np.testing.assert_allclose(actual.get(t),expected.get(t),rtol=1e-13,atol=1e-13)
                    checked.update(changed)
                    if checked == {16,22,23,24}:
                        break
                self.assertEqual(checked,{16,22,23,24})

    def test_direct_projection_rejects_missing_or_conflicting_inputs(self):
        _, document, values = self.fixture()
        cases = []
        for key in ('rows','input_sha256','gauge_transform','gauge_source_sha256','path_boundary'):
            wrong = deepcopy(document)
            del wrong['definition'][key]
            cases.append(wrong)
        wrong = deepcopy(document); wrong['parents'] = []; cases.append(wrong)
        wrong = deepcopy(document); wrong['definition']['raw_id'] = 'invented-raw'; cases.append(wrong)
        wrong = deepcopy(document); wrong['definition']['gauge_source_sha256'] = 'c'*64; cases.append(wrong)
        for wrong in cases:
            with self.subTest(definition=wrong['definition'].keys()), TemporaryDirectory() as td:
                with DatasetWriter(td,DatasetSpec(wrong),provenance={'test':'invalid direct'}) as w:
                    for t in w.missing:
                        w.write(t,values[t])
                    w.publish()
                with DatasetReader(td,cfg='2000') as reader:
                    with self.assertRaisesRegex(ValueError,'direct projected elemental'):
                        DatasetElemental(reader)

    def test_explicit_catalog_bridge_checks_all_stored_coefficients(self):
        table, document, values = self.fixture()
        del document['definition']['catalog']
        for corrupt in (False, True):
            checked = deepcopy(document)
            if corrupt:
                checked['definition']['rows'][22]['terms'][0]['coefficient'][0] += .01
            with self.subTest(corrupt=corrupt), TemporaryDirectory() as td:
                with DatasetWriter(td, DatasetSpec(checked), provenance={'test':'legacy catalog bridge'}) as w:
                    for t in w.missing:
                        w.write(t, values[t])
                    w.publish()
                with DatasetReader(td, cfg='2000') as reader:
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, 'recipe conflicts'):
                            DatasetElemental(reader, catalog=table)
                    else:
                        plain = DatasetElemental(reader)
                        bound = DatasetElemental(reader, catalog=table)
                        self.assertNotEqual(plain.components, bound.components)
                        self.assertEqual(bound.components, tuple(row['component'] for row in table))
                        self.assertEqual(bound.dataset.data_id, reader.data_id)
                        np.testing.assert_array_equal(bound.block(table[22]['component'], 0, 1, 'v2v'),
                                                      values[1,22,0])
                        selection = {'mode_left': (1,), 'mode_right': (0, 1)}
                        metadata = bound.describe_block(table[22]['component'], 0, 1, 'v2v',
                                                        mode_indices=selection)
                        self.assertEqual(metadata.shape, (1, 2))
                        self.assertEqual(metadata.axis_bindings['sink_mode'][1], (1,))
                        np.testing.assert_array_equal(bound.block(table[22]['component'], 0, 1, 'v2v',
                            mode_indices=selection), values[1,22,0,1:2,:])
                        self.assertNotIn('catalog', reader.spec.document['definition'])


if __name__ == '__main__':
    unittest.main()
