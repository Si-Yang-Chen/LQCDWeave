"""Actual selected LapH/EL/PR datasets -> native slot contraction -> saved dt.

CPU synthetic SU(3), isotropic unclovered Wilson; no production-scale claims.
The oracle constructs all physical-time low-mode projectors and full inverse.
"""
from contextlib import ExitStack
import unittest
import numpy as np

from tests.fixtures.laplace_reference import dense_laplacian, scipy
from test_perambulator_dataset import (
    _PerambulatorDatasetFixture, _MatrixBackend, _LATTICE, _CONFIGURATION,
    _MASS, _BOUNDARY, _DIRAC,
)
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contracts.time_selection import TimeSelectionRecord, dataset_time_selection
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram, Propagator, Tag
from lqcdweave.contracts.correlator.api import DiagramBindings, BatchSpec
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
from sympy import Symbol
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.production.eigenvector import LaplaceEigenvectorGenerator
from lqcdweave.production.perambulator import PerambulatorGenerator, VectorBasis
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset
from lqcdweave.production.elemental.planning import prepare_elemental_stage
from fixtures.perambulator_script import produce_source_rows
from lqcdweave.contraction.correlator import calculate, save_correlator, load_correlator
from lqcdweave.sampling.temporal.reconstruction import reconstruct_two_point_from_coordinates


@unittest.skipIf(scipy is None, "SciPy is required for actual Laplace production")
class SparsePersistedPipelineTests(_PerambulatorDatasetFixture):
    def _full_reference(self, ne, coefficient):
        """Dense physical low-mode projectors and full space/spin/color trace."""
        vertices = []
        for t in range(3):
            spatial_links = self.links[t,:,:,:,:3].transpose(3,0,1,2,4,5)
            _, columns = np.linalg.eigh(dense_laplacian(spatial_links))
            vectors = columns[:,:ne].T.reshape(ne,1,1,2,3)
            rows = np.zeros((3,ne,1,1,2,3),complex)
            rows[t] = vectors
            embedding = self._basis_columns(rows,t,ne)
            # Momentum (1,0,0) gives phase +1,-1 on the two spatial sites.
            phase = np.repeat([1.,-1.],3)
            vv = vectors.reshape(ne,-1)
            elemental = vv.conj() @ (phase[:,None]*vv.T)
            local = coefficient*np.kron(gamma(1),elemental)
            g4 = np.kron(gamma(8),np.eye(ne))
            source = g4 @ local.conj().T @ g4
            vertices.append((embedding @ local @ embedding.conj().T,
                             embedding @ source @ embedding.conj().T))
        return np.asarray([[-np.trace(vertices[t][0] @ self.inverse @ vertices[s][1] @ self.inverse)
                            for t in range(3)] for s in range(3)])

    def test_actual_compact_production_persists_through_slots_and_physical_dt(self):
        selection = TimeSelectionRecord(3,(2,0))
        ne, coefficient = 2, .7+.2j
        eigenvalues = {}
        with self.index.open(self.gauge_id) as gauge:
            ev = LaplaceEigenvectorGenerator(_LATTICE,DatasetFieldHandle(gauge),ne,1e-11,
                                             selection=selection, backend="numpy")
            ev.load('cfg-7')
            basis_spec = DatasetSpec(laplace_basis_document(_CONFIGURATION,range(3),
                self.gauge_id,ne,1e-11,selection=selection))
            basis_path = self.root/'selected-ev'
            with DatasetWriter(basis_path,basis_spec,provenance={'test':'selected-ev'}) as writer:
                for slot in writer.missing:
                    t = selection.physical_time(slot)
                    vectors, values = ev.calc(t)
                    spatial_links = self.links[t,:,:,:,:3].transpose(3,0,1,2,4,5)
                    np.testing.assert_allclose(values,np.linalg.eigvalsh(dense_laplacian(spatial_links))[:ne],atol=2e-9)
                    eigenvalues[slot] = values.copy()
                    writer.write(slot,vectors)
                basis_id = writer.publish()
        self.index.register(basis_path)
        values_spec = DatasetSpec(laplace_eigenvalues_document(_CONFIGURATION,range(3),ne,
            basis_id,self.gauge_id,basis_spec.document))
        self._publish('selected-eigenvalues',values_spec,eigenvalues)

        with self.index.open(self.gauge_id) as gauge, self.index.open(basis_id) as basis:
            elemental, spec, provenance, times, cfg = prepare_elemental_stage(gauge,basis,
                paths=((),),momentum_list=((1,0,0),),usedNe=ne,phase_convention='anchor',
                selection=selection)
            elemental.load('cfg-7')
            elemental_path = self.root/'selected-elemental'
            with DatasetWriter(elemental_path,spec,provenance=provenance) as writer:
                for slot in writer.missing:
                    writer.write(slot,elemental.calc_path_blocks(times[slot])['v2v'])
                elemental_id = writer.publish()
            preview = self._plan_products(('VSV',),basis_id=basis_id,Ne=ne,
                source_selection=selection,sink_selection=selection)
            peram_spec = preview['stages']['vsv']['spec']
            mode_basis=VectorBasis(DatasetFieldHandle(basis),count=ne)
            propagator = PerambulatorGenerator(_LATTICE,DatasetFieldHandle(gauge),_MASS,1e-11,120,
                clover_coeff_r=0,t_boundary=_BOUNDARY,source_selection=selection,sink_selection=selection,
                MRHS=True,
                backend=_MatrixBackend(self.matrix,self.links,_DIRAC,1e-11))
            propagator_path = self.root/'selected-perambulator'
            with DatasetWriter(propagator_path,peram_spec,provenance={'test':'selected-pr'}) as writer:
                produce_source_rows(writer,propagator,selection.times,'cfg-7',
                                        src=mode_basis,sink=mode_basis)
                propagator_id = writer.publish()
        self.index.register(elemental_path)
        self.index.register(propagator_path)
        self.assertEqual(dataset_time_selection(peram_spec,'source_time'),selection)
        self.assertEqual(peram_spec.document['bases']['source_mode']['data_id'],basis_id)
        self.assertEqual(spec.document['bases']['sink_mode']['data_id'],basis_id)
        full_reference = self._full_reference(ne,coefficient)
        reference = full_reference[np.ix_(selection.times,selection.times)]
        component = ElementalComponent.path(GaugePath())
        operator = Operator.from_terms('local', [OperatorTerm(1,component,0,coefficient)])
        # A raw Diagram is a positive cycle trace; include the connected
        # meson Wick coefficient explicitly, as in the independent oracle.
        diagram = -Diagram(QuarkDiagram([[0,1],[1,0]]),[0,1],['source','sink'],[None,'q'])
        with ExitStack() as stack:
            elemental_reader = stack.enter_context(DatasetReader(elemental_path))
            propagator_reader = stack.enter_context(DatasetReader(propagator_path))
            self.assertEqual(elemental_reader.data_id,elemental_id)
            self.assertEqual(propagator_reader.data_id,propagator_id)
            e = DatasetElemental(elemental_reader)
            p = DatasetPropagator(propagator_reader)
            self.assertEqual((e.Lt,p.Lt),(2,2))
            result = calculate(diagram,propagator_map={'q':p},
                vertex_map={'source':Meson(e,operator,True,phase_convention='anchor'),
                            'sink':Meson(e,operator,phase_convention='anchor')},
                configuration='cfg-7',usedNe=ne,time_axes={0:range(2),1:range(2)},
                optimize='none',memory_limit=32*2**20)
            np.testing.assert_allclose(result.value,reference,atol=3e-13,rtol=3e-11)
            destination = self.root/'selected-correlator'
            save_correlator(destination,result,coordinate_domain='slot',selection=selection,
                source_axis=0,sink_axis=1,boundary='periodic',average_tsrc=True)
            # The older fixed two-axis planner remains read-only. Compose its
            # verified spec with existing numerical calculate and generic writer.
            source_symbol,sink_symbol=Symbol('saved-source'),Symbol('saved-sink')
            forward=Propagator('u',Tag(0,0),Tag(1,1))
            reverse=Propagator('u',Tag(1,1),Tag(0,0))
            raw_diagram=Diagram(QuarkDiagram([[0,1],[2,0]]),[source_symbol,sink_symbol],
                                ['source','sink'],[None,forward,reverse])
            propagators={forward:p,reverse:p}
            vertices={'source':Meson(e,operator,True,phase_convention='anchor'),
                      'sink':Meson(e,operator,phase_convention='anchor')}
            bindings=DiagramBindings(propagators,vertices,'cfg-7',ne)
            plan=plan_indexed_diagram_measurement(self.index,raw_diagram,bindings=bindings,
                source_time_symbol=source_symbol,sink_time_symbol=sink_symbol,sink_batch=BatchSpec(size=2,axis=sink_symbol))
            measurement=plan['measurement_spec']
            self.assertEqual(plan['source_times'],(0,1))
            self.assertEqual(plan['time_selection'],selection)
            self.assertEqual(measurement.shape,(2,2))
            self.assertEqual(measurement.document['conventions']['time_coverage'],'selected_physical')
            self.assertEqual(dataset_time_selection(measurement,'source_time'),selection)
            measurement_path=self.root/'planned-slot-measurement'
            computed=[]
            def row(slot):
                computed.append(slot)
                value=calculate(raw_diagram,propagator_map=propagators,vertex_map=vertices,
                    configuration='cfg-7',usedNe=ne,
                    time_axes={source_symbol:slot,sink_symbol:range(2)},
                    optimize='none',memory_limit=32*2**20).value
                np.testing.assert_allclose(value,-reference[slot],atol=3e-13,rtol=3e-11)
                return np.asarray(value,dtype=np.complex128)
            with self.assertRaisesRegex(RuntimeError,'interrupt'):
                with DatasetWriter(measurement_path,measurement,provenance={'test':'selected-diagram'}) as writer:
                    writer.write(0,row(0))
                    raise RuntimeError('test interrupt')
            state=inspect_dataset(measurement_path,cfg='cfg-7')
            self.assertEqual(state['state'],'partial')
            self.assertEqual(state['completed'],(0,))
            partial=np.load(measurement_path/'.state'/'cfg-7.npy',mmap_mode='r',allow_pickle=False)
            try:
                completed_bytes={slot:np.asarray(partial[slot]).tobytes()
                                 for slot in state['completed']}
            finally:
                partial._mmap.close()
            with DatasetWriter(measurement_path,measurement,provenance={'test':'selected-diagram'},resume=True) as writer:
                self.assertEqual(tuple(writer.missing),(1,))
                for slot in writer.missing: writer.write(slot,row(slot))
                writer.publish()
            self.assertEqual(computed,[0,1])
            with DatasetReader(measurement_path,cfg='cfg-7') as saved:
                actual=np.stack([saved.read(source_time=slot) for slot in range(2)])
                for slot,data in completed_bytes.items():
                    self.assertEqual(saved.read(source_time=slot).tobytes(),data)
                np.testing.assert_allclose(actual,-reference,atol=3e-13,rtol=3e-11)
                self.assertEqual(dataset_time_selection(saved.spec,'sink_time'),selection)
        loaded = load_correlator(destination)
        expected_dt = np.asarray([np.mean([reference[i,j] for i,s in enumerate(selection.times)
                    for j,t in enumerate(selection.times) if (t-s)%3==dt]) for dt in range(3)])
        np.testing.assert_allclose(loaded.value,expected_dt,atol=3e-13,rtol=3e-11)
        np.testing.assert_array_equal(loaded.reduction['counts'],[2,1,1])
        direct = reconstruct_two_point_from_coordinates(reference,selection,tsrc_axis=0,dt_axis=1)
        np.testing.assert_allclose(direct.values,expected_dt,atol=3e-13,rtol=3e-11)
        np.testing.assert_array_equal(direct.counts,[2,1,1])
