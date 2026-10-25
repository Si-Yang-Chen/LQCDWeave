"""Fixed CPU gauge -> EV -> elemental -> PR -> correlator save/load gate.

Synthetic SU(3), seed 94; lattice (Lx,Ly,Lz,Lt)=(2,1,1,3), Ne=2.
Isotropic unclovered Wilson, m=0.4, r=a=1, antiperiodic quarks;
periodic spatial links, unscaled propagator, DR spin basis, complex128.
The independent references are the existing site-index Laplacian and shifted
Wilson stencil, not the production dense-Wilson matrix or contraction kernel.
This tiny 72-dimensional CPU problem is not an ensemble/QUDA/MPI benchmark.
"""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.fixtures.laplace_reference import dense_laplacian, random_gauge, scipy
from test_wilson_pipeline import _WilsonFixture, stencil
from lqcdweave.adapters.dense_wilson import DenseWilsonBackend
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contracts.time_selection import TimeSelectionRecord, dataset_time_selection
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.production.perambulator.api import PerambulatorGenerator, VectorBasis
from lqcdweave.storage.dataset import DatasetIndex, DatasetReader, DatasetWriter
from lqcdweave.storage.time_selection import load_time_selection, save_time_selection
from lqcdweave.contraction.correlator import calculate, load_correlator, save_correlator
from lqcdweave.production.perambulator.planning import plan_indexed_perambulators
from fixtures.perambulator_script import produce_source_rows
from lqcdweave.production.elemental.planning import prepare_elemental_stage


_LATTICE = (2, 1, 1, 3)
_NE = 2
_COEFFICIENT = .7 + .2j
_DIRAC = dict(action="wilson", mass=.4, xi_0=1., nu=1.,
              clover_coeff_t=0., clover_coeff_r=0., r=1, lattice_spacing=1)


@unittest.skipIf(scipy is None, "SciPy required for actual Laplace production")
class SmallScientificPipelineTests(_WilsonFixture):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.links = random_gauge(_LATTICE, seed=94)
        self.index = DatasetIndex(self.root / "index")
        self.gauge_id = self.index.register(self._registered_gauge(self.root, seed=94))
        shape = (3, 1, 1, 2, 4, 3)
        dimension = int(np.prod(shape))
        self.inverse = np.linalg.inv(np.column_stack([
            stencil(self.links, column.reshape(shape), .4, -1).ravel()
            for column in np.eye(dimension, dtype=np.complex128)
        ]))
        self.laplacians = [dense_laplacian(
            self.links[t, ..., :3, :, :].transpose(3, 0, 1, 2, 4, 5)) for t in range(3)]

    @staticmethod
    def _embedding(vectors, time):
        """Explicit full lattice columns, indexed independently of PR layouts."""
        columns = []
        for spin in range(4):
            for mode in range(_NE):
                field = np.zeros((3, 1, 1, 2, 4, 3), dtype=np.complex128)
                field[time, ..., spin, :] = vectors[mode]
                columns.append(field.ravel())
        return np.column_stack(columns)

    def _full_trace_reference(self):
        # Independently diagonalize all physical slices, including unselected
        # times. Eigenvector phase freedom drops out of full-space projectors.
        vertices = []
        for time, laplacian in enumerate(self.laplacians):
            _, modes = np.linalg.eigh(laplacian)
            vectors = modes[:, :_NE].T.reshape(_NE, 1, 1, 2, 3)
            embedding = self._embedding(vectors, time)
            phase = np.repeat([1., -1.], 3)  # p=(1,0,0), anchor x=0,1
            elemental = modes[:, :_NE].conj().T @ (phase[:, None] * modes[:, :_NE])
            local = _COEFFICIENT * np.kron(gamma(1), elemental)
            g4 = np.kron(gamma(8), np.eye(_NE))
            vertices.append((embedding @ local @ embedding.conj().T,
                             embedding @ (g4 @ local.conj().T @ g4) @ embedding.conj().T))
        return np.asarray([[-np.trace(sink @ self.inverse @ source @ self.inverse)
                            for sink, _ in vertices] for _, source in vertices])

    def _run_pipeline(self, selection):
        times = tuple(range(3)) if selection is None else selection.times
        slots = tuple(range(len(times)))
        if selection is not None:
            selection_path = self.root / "time-selection.npy"
            save_time_selection(selection_path, selection)
            self.assertEqual(load_time_selection(selection_path, Lt=3), selection)
            selection = load_time_selection(selection_path, Lt=3)
        with self.index.open(self.gauge_id) as gauge:
            configuration = gauge.spec.document["configuration"]
            basis_spec = DatasetSpec(laplace_basis_document(
                configuration, range(3), self.gauge_id, _NE, 1e-11, selection=selection))
            generator = LaplaceEigenvectorGenerator(_LATTICE, DatasetFieldHandle(gauge),
                _NE, 1e-11, selection=selection, backend="numpy")
            generator.load("cfg")
            eigenvalues = []
            with DatasetWriter(self.root / "ev", basis_spec,
                                provenance={"test": "fixed-small-pipeline"}) as writer:
                for slot in writer.missing:
                    vectors, values = generator.calc(times[slot])
                    writer.write(slot, vectors)
                    eigenvalues.append(values.copy())
                basis_id = writer.publish()
        self.index.register(self.root / "ev")
        value_spec = DatasetSpec(laplace_eigenvalues_document(configuration, range(3),
            _NE, basis_id, self.gauge_id, basis_spec.document))
        with DatasetWriter(self.root / "eigenvalues", value_spec,
                            provenance={"test": "fixed-small-pipeline"}) as writer:
            for slot in writer.missing:
                writer.write(slot, eigenvalues[slot])
            writer.publish()

        with self.index.open(self.gauge_id) as gauge, self.index.open(basis_id) as basis:
            elemental, elemental_spec, provenance, physical_times, _ = prepare_elemental_stage(
                gauge, basis, paths=((),), momentum_list=((1, 0, 0),), usedNe=_NE,
                phase_convention="anchor", selection=selection)
            self.assertEqual(tuple(physical_times), times)
            elemental.load("cfg")
            with DatasetWriter(self.root / "el", elemental_spec, provenance=provenance) as writer:
                for slot in writer.missing:
                    writer.write(slot, elemental.calc_path_blocks(times[slot])["v2v"])
                elemental_id = writer.publish()
            plan = plan_indexed_perambulators(self.index, self.gauge_id,
                products=("VSV",), basis_id=basis_id, Ne=_NE,
                dirac_definition=_DIRAC, solver_tolerance=1e-11,
                temporal_boundary="antiperiodic", source_selection=selection,
                sink_selection=selection)
            peram_spec = plan["stages"]["vsv"]["spec"]
            mode_basis = VectorBasis(DatasetFieldHandle(basis),count=_NE)
            propagator = PerambulatorGenerator(_LATTICE, DatasetFieldHandle(gauge),
                .4, 1e-11, 120, clover_coeff_t=0., clover_coeff_r=0., t_boundary=-1,
                source_selection=selection, sink_selection=selection,
                backend=DenseWilsonBackend(max_dimension=72))
            with DatasetWriter(self.root / "pr", peram_spec,
                                provenance={"test": "fixed-small-pipeline"}) as writer:
                produce_source_rows(writer, propagator, times, "cfg",src=mode_basis,sink=mode_basis)
                propagator_id = writer.publish()
        self.index.register(self.root / "el")
        self.index.register(self.root / "pr")
        self.assertEqual(elemental_spec.document["bases"]["sink_mode"]["data_id"], basis_id)
        self.assertEqual(peram_spec.document["bases"]["source_mode"]["data_id"], basis_id)
        self.assertEqual(dataset_time_selection(peram_spec, "source_time"), selection)
        self.assertEqual(dataset_time_selection(peram_spec, "sink_time"), selection)

        # All production readers are reopened before consumption, so this
        # checks real persisted bytes and identities, not live borrowed arrays.
        with ExitStack() as stack:
            basis = stack.enter_context(DatasetReader(self.root / "ev"))
            values = stack.enter_context(DatasetReader(self.root / "eigenvalues"))
            elementals = stack.enter_context(DatasetReader(self.root / "el"))
            perams = stack.enter_context(DatasetReader(self.root / "pr"))
            self.assertEqual((basis.data_id, elementals.data_id, perams.data_id),
                             (basis_id, elemental_id, propagator_id))
            self.assertEqual(basis.spec.coordinates("time"), list(slots))
            embeddings = []
            for slot, time in enumerate(times):
                vectors = basis.read(time=slot)
                modes = vectors.reshape(_NE, -1).T
                spectrum = values.read(time=slot)
                laplacian = self.laplacians[time]
                np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(laplacian)[:_NE], atol=1e-10)
                np.testing.assert_allclose(laplacian @ modes, modes * spectrum, atol=1e-10)
                np.testing.assert_allclose(modes.conj().T @ modes, np.eye(_NE), atol=1e-10)
                expected_elemental = modes.conj().T @ (np.repeat([1., -1.], 3)[:, None] * modes)
                np.testing.assert_allclose(elementals.read(time=slot)[0, 0], expected_elemental,
                                           atol=1e-11, rtol=1e-11)
                embeddings.append(self._embedding(vectors, time))
            for source, sink in np.ndindex(len(times), len(times)):
                projected = embeddings[sink].conj().T @ self.inverse @ embeddings[source]
                np.testing.assert_allclose(perams.read(source_time=source, sink_time=sink),
                    projected.reshape(4, _NE, 4, _NE).transpose(0, 2, 1, 3),
                    atol=1e-11, rtol=1e-10)
            operator = Operator.from_terms("fixed-local-vector", [OperatorTerm(
                1, ElementalComponent.path(GaugePath()), 0, _COEFFICIENT)])
            diagram = -Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                               ["source", "sink"], [None, "q"])
            e, p = DatasetElemental(elementals), DatasetPropagator(perams)
            calculation = calculate(diagram, propagator_map={"q": p},
                vertex_map={"source": Meson(e, operator, True, phase_convention="anchor"),
                            "sink": Meson(e, operator, phase_convention="anchor")},
                configuration="cfg", usedNe=_NE, time_axes={0: slots, 1: slots},
                optimize="none", profile="default", memory_limit=32 * 2**20)
            expected = self._full_trace_reference()[np.ix_(times, times)]
            np.testing.assert_allclose(calculation.value, expected, atol=1e-11, rtol=1e-10)
            declaration = dict(source_axis=0, sink_axis=1, boundary="periodic")
            declaration.update(dict(coordinate_domain="physical", physical_lt=3)
                               if selection is None else dict(coordinate_domain="slot", selection=selection))
            save_correlator(self.root / "correlator", calculation, **declaration)
            save_correlator(self.root / "dt", calculation, average_tsrc=True, **declaration)

        raw, reduced = load_correlator(self.root / "correlator"), load_correlator(self.root / "dt")
        np.testing.assert_array_equal(raw.value, calculation.value)
        np.testing.assert_allclose(raw.value, expected, atol=1e-11, rtol=1e-10)
        self.assertEqual(raw.axes, (0, 1))
        self.assertEqual(dict(raw.coordinates), {0: list(slots), 1: list(slots)})
        self.assertEqual(raw.metadata["configuration"], "cfg")
        self.assertEqual(raw.metadata["time_declaration"]["coordinate_domain"],
                         "physical" if selection is None else "slot")
        contributions = [[expected[s, t] for s, source in enumerate(times)
                          for t, sink in enumerate(times) if (sink - source) % 3 == dt]
                         for dt in range(3)]
        np.testing.assert_allclose(reduced.value, [np.mean(row) for row in contributions],
                                   atol=1e-11, rtol=1e-10)
        np.testing.assert_array_equal(reduced.reduction["counts"], [len(row) for row in contributions])
        self.assertEqual(reduced.reduction["physical_lt"], 3)
        if selection is not None:
            self.assertEqual(tuple(reduced.reduction["selections"]["source"]["times"]), (2, 0))
            # Four selected pairs cannot be silently promoted to nine full pairs.
            np.testing.assert_array_equal(reduced.reduction["counts"], [2, 1, 1])
        else:
            np.testing.assert_array_equal(reduced.reduction["counts"], [3, 3, 3])

    def test_default_full_time_pipeline_and_save_load(self):
        self._run_pipeline(None)

    def test_explicit_reordered_sparse_time_pipeline_and_save_load(self):
        self._run_pipeline(TimeSelectionRecord(3, (2, 0)))
