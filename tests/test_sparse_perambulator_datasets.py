"""Physical sparse production through existing slot manifests and row recovery."""
from contextlib import ExitStack
import numpy as np

from test_perambulator_dataset import (
    _PerambulatorDatasetFixture, _MatrixBackend, _LATTICE, _CONFIGURATION,
    _MASS, _BOUNDARY, _DIRAC,
)
from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import TimeSelectionRecord, bind_time_selections, dataset_time_selection
from lqcdweave.production.perambulator import (
    PerambulatorGenerator, VectorBasis, PointBasis,
)
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset
from fixtures.perambulator_script import produce_source_rows


class SparsePerambulatorDatasetTests(_PerambulatorDatasetFixture):
    def _selected(self, identity, name, selection):
        with self.index.open(identity) as reader:
            document = reader.spec.document
            document['axes'][0]['coordinates'] = list(selection.times)
            spec = DatasetSpec(bind_time_selections(document,{'time':selection}))
            rows = {slot:reader.read(time=time).copy() for slot,time in enumerate(selection.times)}
        return self._publish(name,spec,rows)

    def _expected(self, product, source_time, sink_time):
        source = (self._basis_columns(self.source_basis_values,source_time,2) if product[2]=='V'
                  else self._point_columns(self.source_point_rows,source_time,1))
        sink = (self._basis_columns(self.sink_basis_values,sink_time,2) if product[0]=='V'
                else self._point_columns(self.sink_point_rows,sink_time,2))
        matrix = sink.conj().T @ self.inverse @ source
        if product == 'VSV': return matrix.reshape(4,2,4,2).transpose(0,2,1,3)
        if product == 'PSV': return matrix.reshape(4,2,3,4,2).transpose(0,3,1,2,4)
        if product == 'VSP': return matrix.reshape(4,2,4,1,3).transpose(0,2,1,3,4)
        return matrix.reshape(4,2,3,4,1,3).transpose(0,3,1,2,4,5)

    def test_compact_inputs_all_products_identity_and_interrupted_row_resume(self):
        source = TimeSelectionRecord(3,(2,0))
        sink = TimeSelectionRecord(3,(1,))
        src_basis = self._selected(self.source_basis_id,'sparse-source',source)
        snk_basis = self._selected(self.sink_basis_id,'sparse-sink',sink)
        src_points = self._selected(self.source_points_id,'sparse-source-points',source)
        snk_points = self._selected(self.sink_points_id,'sparse-sink-points',sink)
        products = ('VSV','PSV','VSP','PSP')
        preview = self._plan_products(products,basis_src_id=src_basis,basis_snk_id=snk_basis,
            points_src_id=src_points,points_snk_id=snk_points,Ne_src=2,Ne_snk=2,
            source_selection=source,sink_selection=sink)
        with self.assertRaises((ValueError,KeyError)):
            self._plan_products(('VSV',),basis_src_id=src_basis,basis_snk_id=snk_basis)
        with ExitStack() as stack:
            gauge = DatasetFieldHandle(stack.enter_context(self.index.open(self.gauge_id)))
            src = DatasetFieldHandle(stack.enter_context(self.index.open(src_basis)))
            snk = DatasetFieldHandle(stack.enter_context(self.index.open(snk_basis)))
            ps = DatasetPointHandle(stack.enter_context(self.index.open(src_points)))
            pk = DatasetPointHandle(stack.enter_context(self.index.open(snk_points)))
            source_bases={'V':VectorBasis(src,count=2),'P':PointBasis(ps,count=1)}
            sink_bases={'V':VectorBasis(snk,count=2),'P':PointBasis(pk,count=2)}
            gen = PerambulatorGenerator(_LATTICE,gauge,_MASS,1e-11,120,
                clover_coeff_r=0,t_boundary=_BOUNDARY,
                source_selection=source,sink_selection=sink,MRHS=True,rhs_batch_size=4,
                backend=_MatrixBackend(self.matrix,self.links,_DIRAC,1e-11))
            for product in products:
                spec = preview['stages'][product.lower()]['spec']
                self.assertEqual(spec.shape[:2],(2,1))
                self.assertEqual(spec.coordinates('source_time'),[0,1])
                self.assertEqual(dataset_time_selection(spec,'source_time'),source)
                self.assertEqual(dataset_time_selection(spec,'sink_time'),sink)
                self.assertEqual(spec.document['configuration']['lattice'][3],3)
                expected_parents = {'source_mode':src_basis,'sink_mode':snk_basis,
                                    'source_point':src_points,'sink_point':snk_points}
                for role,basis in spec.document['bases'].items():
                    self.assertEqual(basis['data_id'],expected_parents[role])
                path = self.root / ('selected-'+product)
                method = gen.calc
                calls = []
                fail = [True]
                def calculate(time,*,src,sink,write=None):
                    calls.append(time)
                    if fail[0] and time == 0:
                        raise RuntimeError('test interrupt')
                    return method(time,src=src,sink=sink,write=write)
                gen.calc=calculate
                with self.assertRaisesRegex(RuntimeError,'test interrupt'):
                    with DatasetWriter(path,spec,provenance={'test':'sparse'}) as writer:
                        produce_source_rows(writer,gen,source.times,'cfg-7',product,
                            src=source_bases[product[2]],sink=sink_bases[product[0]])
                state = inspect_dataset(path, cfg='cfg-7')
                self.assertEqual(state['state'], 'partial')
                self.assertEqual(state['completed'], (0,))
                partial = np.load(path/'.state'/'cfg-7.npy', mmap_mode='r', allow_pickle=False)
                try:
                    completed_bytes = {position: np.asarray(partial[position]).tobytes()
                                       for position in state['completed']}
                finally:
                    partial._mmap.close()
                fail[0] = False
                with DatasetWriter(path,spec,provenance={'test':'sparse'},resume=True) as writer:
                    self.assertEqual(tuple(writer.missing),(1,))
                    produce_source_rows(writer,gen,source.times,'cfg-7',product,
                        src=source_bases[product[2]],sink=sink_bases[product[0]])
                    writer.publish()
                gen.calc=method
                self.assertEqual(calls,[2,0,0])
                with DatasetReader(path, cfg='cfg-7') as reader:
                    for i,t in enumerate(source.times):
                        if i in completed_bytes:
                            self.assertEqual(reader.read(source_time=i,sink_time=0).tobytes(),
                                             completed_bytes[i])
                        np.testing.assert_allclose(reader.read(source_time=i,sink_time=0),
                            self._expected(product,t,1),rtol=3e-12,atol=3e-13)

    def test_writer_rejects_equal_size_wrong_sink_order_before_loading(self):
        source = TimeSelectionRecord(3,(2,0))
        sink = TimeSelectionRecord(3,(1,2))
        preview = self._plan_products(('VSV',),basis_src_id=self.source_basis_id,
            basis_snk_id=self.sink_basis_id,source_selection=source,sink_selection=sink)
        spec = preview['stages']['vsv']['spec']
        class Generator:
            source_selection = source
            sink_selection = TimeSelectionRecord(3,(2,1))
            def load(self,key): raise AssertionError('must reject before input reads')
        with DatasetWriter(self.root/'wrong',spec,provenance={}) as writer:
            with self.assertRaisesRegex(ValueError,'sink selection'):
                produce_source_rows(writer,Generator(),source.times,'cfg-7',src=object(),sink=object())


    def test_writer_checks_physical_geometry_even_when_compact_shape_matches(self):
        source = TimeSelectionRecord(3,(2,0))
        preview = self._plan_products(('VSV',),basis_id=self.source_basis_id,
            source_selection=source,sink_selection=source)
        spec = preview['stages']['vsv']['spec']
        class Generator:
            source_selection = TimeSelectionRecord(4,(2,0))
            sink_selection = TimeSelectionRecord(4,(2,0))
            def load(self,key): raise AssertionError('must reject before input reads')
        with DatasetWriter(self.root/'wrong-lattice',spec,provenance={}) as writer:
            with self.assertRaisesRegex(ValueError,'physical Lt'):
                produce_source_rows(writer,Generator(),source.times,'cfg-7',src=object(),sink=object())
