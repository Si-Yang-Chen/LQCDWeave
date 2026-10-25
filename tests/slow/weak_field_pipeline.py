"""Opt-in 4^3 x 8 weak_field scientific audit; excluded from normal discovery.

Run with tools/test_tiers.py --tier slow. This recomputes an explicitly defined
unimproved Wilson problem, not the undocumented historical clover propagator.
The test-only LU adapter caches the production Wilson matrix; the oracle has
an independently assembled sparse matrix, checked against a shifted stencil.
Neither sparse adapter is a new supported production backend.
"""

from contextlib import ExitStack
from itertools import product
import hashlib
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np

try:
    from scipy.sparse import coo_matrix, csc_matrix
    from scipy.sparse.linalg import splu
except ImportError:
    splu = None

from tests.fixtures.laplace_reference import dense_laplacian
from test_wilson_pipeline import stencil
from production.elemental.test_elemental import point_basis, reference_matrix
from lqcdweave.adapters.dense_wilson import wilson_matrix
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.adapters.legacy_fields import IldgGauge
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.adapters.propagator import ArrayPropagator, PropagatorPairReader
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contraction.binding.api import Meson, Baryon
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, BaryonComponent
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS
from lqcdweave.contracts.time_selection import TimeSelectionRecord, bind_time_selections
from lqcdweave.contracts.spatial_sampling import SpatialSamplingDesign
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram, Propagator, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion
from lqcdweave.operators.single_hadron.baryon import (
    BaryonOperator, BaryonOperatorTerm, SparseSpinTensor,
)
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.planning.baryon import baryon_dataset_document, normalize_baryon_components
from lqcdweave.planning.production import elemental_dataset_document, point_elemental_dataset_document
from lqcdweave.planning.spatial_points import validate_complete_spatial_point_set
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator, calc_overlap_row
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from lqcdweave.production.elemental.api import ElementalGenerator, BaryonElementalGenerator
from lqcdweave.production.gauge import project_su3, stout_smear
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, ModePartition, HighmodeBasis, PropagatorEndpoint,
    VectorBasis, PointBasis, propagator_pair_spec, PropagatorWriter, PropagatorReader,
)
from lqcdweave.sampling.spatial.api import sample_spatial_points
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader, DatasetIndex
from lqcdweave.storage.time_selection import save_time_selection, load_time_selection
from lqcdweave.contraction.correlator import (
    calculate, save_correlator, load_correlator, save_calculation_plan, load_calculation_plan,
)
from lqcdweave.production.perambulator.pair_planning import plan_indexed_perambulator_pairs


FIXTURE = Path(__file__).resolve().parents[1] / 'data' / 'weak_field'
LATTICE = (4, 4, 4, 8)
FIELD_SHAPE = (8, 4, 4, 4, 4, 3)
MASS = .4
NE = 6
LOW = (0, 1, 3)
MEDIUM = (2, 4, 5)
CFG = 'weak-field-wilson-audit'
CONFIGURATION = dict(ensemble='weak-field-cpu-audit', configuration=CFG, lattice=list(LATTICE))
LOCAL = ElementalComponent.path(())
DIRAC = dict(action='wilson', mass=MASS, xi_0=1., nu=1.,
             clover_coeff_t=0., clover_coeff_r=0., temporal_boundary=-1,
             r=1, lattice_spacing=1)


def announce(message):
    print(f'\nweak_field audit: {message}', flush=True)


def sparse_reference(links):
    """Vectorized site shifts and explicit spin/color axes, independent of PR."""
    shape = links.shape[:4]
    sites = np.arange(prod(shape)).reshape(shape)
    rows, columns, data = [], [], []
    local = np.arange(12)
    for mu, axis in enumerate((3, 2, 1, 0)):
        for sign in (1, -1):
            neighbors = np.roll(sites, -sign, axis=axis).ravel()
            color = (links[..., mu, :, :] if sign == 1 else
                     np.roll(links[..., mu, :, :], 1, axis=axis).conj().swapaxes(-1, -2))
            boundary = np.ones(shape)
            if mu == 3:
                boundary[-1 if sign == 1 else 0] = -1
            spin = np.eye(4) - sign * gamma(1 << mu)
            blocks = -.5 * np.einsum('sr,...ab,...->...sarb', spin, color, boundary).reshape(-1, 12, 12)
            rr = np.broadcast_to(sites.ravel()[:, None, None] * 12 + local[None, :, None], blocks.shape)
            cc = np.broadcast_to(neighbors[:, None, None] * 12 + local[None, None, :], blocks.shape)
            keep = blocks != 0
            rows.append(rr[keep]); columns.append(cc[keep]); data.append(blocks[keep])
    dimension = prod(FIELD_SHAPE)
    rows.append(np.arange(dimension)); columns.append(np.arange(dimension))
    data.append(np.full(dimension, MASS + 4))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                      shape=(dimension, dimension)).tocsc()


class AuditLUBackend:
    """Bounded test backend, using the *production* Wilson assembly and cached LU.

    PR still constructs its own actual RHS and projections. The independent
    reference below never obtains solutions, RHS or projection data here.
    """
    array_module = np
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)

    def __init__(self, fixture):
        self.fixture = fixture
        self.last_relative_residuals = None

    def load(self, gauge, parameters):
        assert parameters['mass'] == MASS and parameters['t_boundary'] == -1
        assert parameters['clover_coeff_t'] == parameters['clover_coeff_r'] == 0
        assert parameters['xi_0'] == parameters['nu'] == 1
        assert parameters['multigrid'] in (None, False)
        np.testing.assert_array_equal(np.asarray(gauge[:]), self.fixture.links)

    def solve_into(self, rhs, solution, *, tol, maxiter):
        values = self.fixture.production_lu.solve(rhs.reshape(len(rhs), -1).T)
        solution[...] = values.T.reshape(rhs.shape)
        errors = self.fixture.production_matrix @ solution.reshape(len(rhs), -1).T - rhs.reshape(len(rhs), -1).T
        norms = np.linalg.norm(rhs.reshape(len(rhs), -1), axis=1)
        residuals = np.linalg.norm(errors, axis=0) / np.where(norms, norms, 1)
        self.last_relative_residuals = tuple(map(float, residuals))
        self.fixture.solve_residuals.extend(self.last_relative_residuals)
        if not np.isfinite(solution).all() or np.max(residuals) > tol:
            raise RuntimeError('weak_field audit solve residual exceeds tolerance')


class Input:
    def __init__(self, values, **metadata):
        self.values = values
        self.__dict__.update(metadata)

    def load(self, cfg):
        if cfg != CFG:
            raise ValueError('wrong audit cfg')
        return self.values


class AuditFixture:
    def __init__(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory(prefix='ed-weak-field-slow-')))
        self.index = DatasetIndex(self.root / 'index')
        self.solve_residuals = []
        self.metrics = dict(lattice=list(LATTICE), Ne=NE, mass=MASS,
                            boundary='antiperiodic', clover=False, stout=False, backend='test-only CPU sparse LU')
        raw = np.load(FIXTURE / 'weak_field.npy', mmap_mode='r')
        self.links = raw.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6).copy()
        self.gauge_spec = self.spec('gauge', self.links, ('time', 'z', 'y', 'x', 'direction', 'color_row', 'color_column'),
                                    conventions=dict(boundary='periodic'))
        self.gauge_id, self.gauge = self.publish('gauge', self.gauge_spec, self.links)
        announce('compute and persist all 8 Laplace slices, Ne=6')
        basis_spec = DatasetSpec(laplace_basis_document(CONFIGURATION, range(8), self.gauge_id, NE, 2e-11))
        ev = LaplaceEigenvectorGenerator(LATTICE, DatasetFieldHandle(self.gauge), NE, 2e-11,
                                         backend="numpy").load(CFG)
        vectors, spectra = [], []
        for t, value, spectrum in ev.iter_slices():
            vectors.append(value.copy()); spectra.append(spectrum.copy())
        self.vectors, self.spectra = np.stack(vectors), np.stack(spectra)
        self.parent_id, self.parent = self.publish('EV', basis_spec, self.vectors)
        view_document = self.parent.spec.document
        view_document['axes'][1]['coordinates'] = list(LOW)
        self.v_spec = DatasetSpec(view_document)
        value_spec = DatasetSpec(laplace_eigenvalues_document(CONFIGURATION, range(8), NE,
                                                             self.parent_id, self.gauge_id, basis_spec.document))
        self.publish('EV-values', value_spec, self.spectra)
        self.parent_handle = DatasetFieldHandle(self.parent)
        self.partition = ModePartition(self.parent_id, LOW, (MEDIUM,))
        self.high = HighmodeBasis(self.parent_handle, self.partition)
        self.v_handle = Input(self.vectors[:, LOW], Ne=len(LOW), mode_labels=LOW,
                              basis_id=self.parent_id, basis_orthogonal=True,
                              endpoint=PropagatorEndpoint('V', self.parent_id, LOW))
        self.exact = self.make_noise('D-exact', 3)
        self.qr = self.make_noise('D-qr', 2)
        self.points = self.make_points('points-segmented', 4, 4)
        announce('assemble independent shifted Wilson operator and production Wilson matrix, dimension=6144')
        self.reference_matrix = sparse_reference(self.links)
        self.production_matrix = csc_matrix(wilson_matrix(self.links, MASS, t_boundary=-1, max_dimension=6144))
        announce('factor independent and production operators; cache only for this audit')
        self.reference_lu = splu(self.reference_matrix)
        self.production_lu = splu(self.production_matrix)
        self._solutions = {}
        self._bundles = {}

    def spec(self, kind, values, names, *, conventions=None, parents=(), definition=None, selection=None):
        doc = dict(schema=1, kind=kind, configuration=CONFIGURATION,
                   axes=[dict(name=n, coordinates=list(range(s))) for n, s in zip(names, values.shape)],
                   dtype=str(values.dtype), bases={}, parents=list(parents), definition=definition or {},
                   conventions=conventions or {})
        if selection is not None:
            doc = bind_time_selections(doc, {'time': selection})
        return DatasetSpec(doc)

    def publish(self, name, spec, values, provenance=None):
        with DatasetWriter(self.root / name, spec, provenance=provenance or dict(test='weak-field-slow-audit')) as writer:
            for slot in writer.missing:
                writer.write(slot, np.asarray(values[slot], dtype=spec.document['dtype']))
            identity = writer.publish()
        self.index.register(self.root / name)
        return identity, self.stack.enter_context(DatasetReader(self.root / name))

    def make_noise(self, name, count):
        gen = NoisevectorGenerator(self.parent_handle, ((3,), (count,)), mode_groups=(MEDIUM,)).load(CFG)
        values = gen.calc(seed=952)
        description = gen.describe()
        spec = self.spec('explicit_basis', values, ('time', 'mode', 'z', 'y', 'x', 'color'),
                         conventions=dict(representation='dilution', normalization='raw',
                                          basis_orthogonal=description['basis_orthogonal']), parents=(self.parent_id,))
        identity, reader = self.publish(name, spec, values, dict(generator=description))
        handle = DatasetFieldHandle(reader)
        source = handle.dilution_source
        handle.endpoint = PropagatorEndpoint('D', identity, range(count), metadata=dict(
            vector_labels=[list(x) for x in source.vector_labels], mode_groups=[list(MEDIUM)],
            parent_basis_id=self.parent_id, realization_id=identity, noise_pool_id=identity,
            normalization='raw', dilution_source=source.document))
        return dict(id=identity, reader=reader, handle=handle, values=values, source=source)

    def make_points(self, name, count, length):
        points = sample_spatial_points(SpatialSamplingDesign(LATTICE, count, length), seed=34)
        spec = validate_complete_spatial_point_set(points, configuration=CONFIGURATION)
        identity, reader = self.publish(name, spec, points.coordinates)
        handle = SpatialPointHandle.from_dataset(self.root / name, cfg=CFG)
        handle.load(CFG)
        point_input = Input(points.coordinates, Np=points.coordinates.shape[1],
                            endpoint=PropagatorEndpoint('P', identity, range(count)))
        return dict(id=identity, reader=reader, handle=handle, input=point_input, points=points, spec=spec)

    def basis(self, kind, t, noise, points, *, high=True):
        if kind == 'v':
            return self.vectors[t, LOW].reshape(len(LOW), -1).T
        if kind == 'd':
            return noise['values'][t].reshape(noise['values'].shape[1], -1).T
        basis = point_basis(points['points'].coordinates[t], LATTICE)
        if high:
            parents = self.vectors[t].reshape(NE, -1).T
            basis = basis - parents @ (parents.conj().T @ basis)
        return basis

    @staticmethod
    def embedding(basis, t):
        # Explicit physical spin/color order, independent of PR source routines.
        count = basis.shape[1]
        columns = np.zeros((6144, 4 * count), dtype=complex)
        for spin in range(4):
            fields = np.zeros((*FIELD_SHAPE, count), dtype=complex)
            fields[t, ..., spin, :, :] = basis.reshape(4, 4, 4, 3, count)
            columns[:, spin * count:(spin + 1) * count] = fields.reshape(6144, count)
        return columns

    def solution(self, kind, t, noise, points):
        identity = self.parent_id if kind == 'v' else noise['id'] if kind == 'd' else points['id']
        key = kind, identity, t
        if key not in self._solutions:
            columns = self.embedding(self.basis(kind, t, noise, points), t)
            self._solutions[key] = self.reference_lu.solve(columns)
        return self._solutions[key]

    def reference_block(self, sector, sink, source, noise, points):
        left = self.basis(sector[0], sink, noise, points)
        right = self.basis(sector[2], source, noise, points)
        field = self.solution(sector[2], source, noise, points).reshape(*FIELD_SHAPE, 4 * right.shape[1])
        # Select physical sink time, then contract color/site separately from spin.
        result = np.einsum('ia,isj->saj', left.conj(), field[sink].transpose(0, 1, 2, 4, 3, 5).reshape(192, 4, -1))
        result = result.reshape(4, left.shape[1], 4, right.shape[1]).transpose(0, 2, 1, 3)
        tail = []
        for kind, value in ((sector[0], left), (sector[2], right)):
            tail.extend((value.shape[1] // 3, 3) if kind == 'p' else (value.shape[1],))
        return result.reshape(4, 4, *tail)

    def bundle(self, name='segmented', *, noise=None, points=None, times=tuple(range(8))):
        if name in self._bundles:
            return self._bundles[name]
        noise, points = noise or self.exact, points or self.points
        announce(f'produce six saved propagators / nine reading views: {name}, times={times}')
        bases = dict(V=VectorBasis(self.v_handle), D=VectorBasis(noise['handle']),
                     P=PointBasis(points['input'], highmode_basis=self.high))
        selection = None if times == tuple(range(8)) else TimeSelectionRecord(8, times)
        saved = []
        for a, b, products in (('V', 'V', ('VSV', 'PSV', 'PSP')),
                               ('D', 'V', ('VSD', 'PSD')), ('D', 'D', ('DSD',))):
            gen = PerambulatorGenerator(LATTICE, DatasetFieldHandle(self.gauge), MASS, 2e-11, 100,
                clover_coeff_r=0, clover_coeff_t=0, t_boundary=-1,
                backend=AuditLUBackend(self),
                source_selection=selection, sink_selection=selection, MRHS=True, rhs_batch_size=24)
            gen.load(CFG)
            endpoints = {(kind, source): gen.endpoint(basis, source=source)
                for kind, basis in bases.items() for source in (False, True)}
            with ExitStack() as writers:
                outputs = {}
                for pr in products:
                    sink = endpoints[pr[0], False]
                    source = endpoints[pr[2], True]
                    spec = propagator_pair_spec(pr, configuration=CONFIGURATION, sink=sink, source=source,
                        time_pairs=tuple(product(times, repeat=2)), gauge_id=self.gauge_id, dirac=DIRAC,
                        gamma5=gamma(15), spin_basis='degrand_rossi', gamma5_hermitian=True)
                    writer = writers.enter_context(DatasetWriter(self.root / name / pr, spec, provenance=dict(test='weak-field-new-Wilson-solves')))
                    outputs[pr] = writer, PropagatorWriter(writer), sink, source
                for s in times:
                    arrays = {}
                    for source_kind in dict.fromkeys(pr[2] for pr in products):
                        sink_kinds = tuple(dict.fromkeys(pr[0] for pr in products if pr[2] == source_kind))
                        results = gen.calc(s, src=bases[source_kind],
                                           sink=[bases[sink_kind] for sink_kind in sink_kinds])
                        arrays.update((result.product, result) for result in results)
                    for pr, (writer, output, sink, source) in outputs.items():
                        for slot, t in enumerate(times):
                            output.write(arrays[pr].at(t))
                for pr, (writer, *_rest) in outputs.items():
                    writer.publish()
                    self.index.register(self.root / name / pr)
                    saved.append(self.stack.enter_context(DatasetReader(self.root / name / pr)))
        result = dict(prop=PropagatorPairReader(PropagatorReader(saved)), saved=saved,
                      noise=noise, points=points, times=times, name=name)
        self._bundles[name] = result
        return result

    def elemental(self, bundle, *, weighted=False, paths=((),), momenta=((0, 0, 0),), phase='anchor', selection=None):
        components = tuple(ElementalComponent.path(path) for path in paths)
        gen = ElementalGenerator(DatasetFieldHandle(self.gauge), None, momenta,
            [dict(component=c, path=p, coefficient=1) for c, p in zip(components, paths)],
            vector_bases=dict(V=self.v_handle, D=bundle['noise']['handle']), points=bundle['points']['handle'],
            blocks=ELEMENTAL_SECTORS, phase_convention=phase, elemental_weighted=weighted,
            lattice_size=LATTICE, selection=selection, ne_block_size=2).load(CFG)
        arrays = {sector: [] for sector in ELEMENTAL_SECTORS}
        times = tuple(range(8)) if selection is None else selection.times
        for t in times:
            for sector, value in gen.calc_path_blocks(t).items():
                arrays[sector].append(np.asarray([[x.to_dense() for x in row] for row in value])
                                      if sector == 'p2p' else value.copy())
        reader = ArrayElemental({s: np.stack(rows, axis=2) for s, rows in arrays.items()}, components,
            cfg=CFG, data_identity=f"{bundle['name']}/EL/{phase}/{weighted}/{paths}/{times}", momentum_list=momenta,
            phase_convention=phase, elemental_weighted=weighted, elemental_structures=gen.elemental_structures,
            sampling_pool_ids=gen.sampling_pool_ids, sampling_groups=gen.sampling_groups,
            dilution_endpoints=gen.dilution_endpoints, basis_endpoints=gen.basis_endpoints,
            selection=selection,
            sampling_selection=None if selection is None else TimeSelectionRecord(8, tuple(range(8))))
        return gen, reader


@unittest.skipIf(splu is None, 'SciPy required for the explicit weak_field slow audit')
class WeakFieldPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = time.perf_counter()
        # Register cleanup before work, including setUpClass failures.
        cls.f = AuditFixture.__new__(AuditFixture)
        try:
            cls.f.__init__()
        except BaseException:
            if hasattr(cls.f, 'stack'):
                cls.f.stack.close()
            raise
        cls.addClassCleanup(cls.f.stack.close)
        cls.f.metrics['fixture_seconds'] = time.perf_counter() - start

    def setUp(self):
        self.audit_metrics = dict(self.f.metrics)
        self.started = time.perf_counter()

    def tearDown(self):
        self.audit_metrics['case_seconds'] = time.perf_counter() - self.started

    def close_enough(self, label, actual, expected, *, atol=2e-10, rtol=2e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertTrue(np.isfinite(actual).all(), label)
        delta = np.linalg.norm(actual - expected)
        relative = float(delta / max(np.linalg.norm(expected), np.finfo(float).tiny))
        record = self.audit_metrics.setdefault('checks', {}).setdefault(label, dict(count=0, max_absolute=0., relative_l2=0.))
        record['count'] += 1
        record['max_absolute'] = max(record['max_absolute'], float(np.max(np.abs(actual - expected), initial=0)))
        record['relative_l2'] = max(record['relative_l2'], relative)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)

    def test_01_gauge_formats_operator_and_boundary(self):
        f = self.f
        raw = np.load(FIXTURE / 'weak_field.npy')
        self.close_enough('raw bin equals NPY', np.fromfile(FIXTURE / 'weak_field.bin', dtype='<c16'), raw.ravel(), atol=0, rtol=0)
        ildg = IldgGauge(str(FIXTURE) + '/', '.lime').load('weak_field')
        self.close_enough('ILDG equals canonical gauge', np.asarray(ildg[:]), f.links, atol=0, rtol=0)
        self.close_enough('SU3 unitarity', f.links @ f.links.conj().swapaxes(-1, -2),
                          np.broadcast_to(np.eye(3), f.links.shape), atol=2e-10, rtol=0)
        self.close_enough('SU3 determinant', np.linalg.det(f.links), np.ones(f.links.shape[:-2]), atol=2e-10, rtol=0)
        self.audit_metrics['fixture_sha256'] = {name: hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
                                               for name in ('weak_field.npy', 'weak_field.bin', 'weak_field.lime')}
        rng = np.random.default_rng(17)
        for _ in range(5):
            field = rng.normal(size=FIELD_SHAPE) + 1j * rng.normal(size=FIELD_SHAPE)
            reference = stencil(f.links, field, MASS, -1)
            self.close_enough('independent CSR versus shifted stencil', (f.reference_matrix @ field.ravel()).reshape(FIELD_SHAPE), reference, atol=2e-13, rtol=2e-13)
            self.close_enough('production Wilson versus shifted stencil', (f.production_matrix @ field.ravel()).reshape(FIELD_SHAPE), reference, atol=2e-13, rtol=2e-13)
        # Boundary-localized probes distinguish antiperiodicity from a periodic operator.
        field = np.zeros(FIELD_SHAPE, complex); field[0, 0, 0, 0, 0, 0] = 1
        anti, periodic = stencil(f.links, field, MASS, -1), stencil(f.links, field, MASS, 1)
        self.assertGreater(np.linalg.norm(anti[-1] - periodic[-1]), 1)
        probe = rng.normal(size=6144) + 1j * rng.normal(size=6144)
        def g5(value):
            return np.einsum('sr,irc->isc', gamma(15), value.reshape(512, 4, 3)).ravel()
        self.close_enough('Dirac gamma5 hermiticity', f.reference_matrix.conj().T @ probe,
                          g5(f.reference_matrix @ g5(probe)), atol=2e-13, rtol=2e-13)

    def test_02_eigenvectors_all_times_and_compact_reordered_production(self):
        f = self.f
        for t in range(8):
            laplacian = dense_laplacian(f.links[t, ..., :3, :, :].transpose(3, 0, 1, 2, 4, 5))
            eig, vec = np.linalg.eigh(laplacian)
            basis = f.vectors[t].reshape(NE, -1).T
            self.close_enough('Laplace eigenvalues', f.spectra[t], eig[:NE], atol=2e-10, rtol=2e-10)
            self.close_enough('Laplace residual', laplacian @ basis, basis * f.spectra[t], atol=2e-10, rtol=2e-10)
            self.close_enough('EV Gram', basis.conj().T @ basis, np.eye(NE), atol=2e-10, rtol=0)
            self.close_enough('EV phase-independent projector', basis @ basis.conj().T, vec[:, :NE] @ vec[:, :NE].conj().T, atol=2e-9, rtol=2e-9)
        selection = TimeSelectionRecord(8, (7, 0, 4, 3))
        file = f.root / 'selected-times.npy'
        save_time_selection(file, selection)
        self.assertEqual(load_time_selection(file, Lt=8), selection)
        gen = LaplaceEigenvectorGenerator(LATTICE, DatasetFieldHandle(f.gauge), NE, 2e-11,
                                          selection=selection, backend="numpy").load(CFG)
        rows = []
        for t, basis, spectrum in gen.iter_slices():
            self.close_enough('compact EV spectrum', spectrum, f.spectra[t])
            rows.append(basis.copy())
        doc = DatasetSpec(laplace_basis_document(CONFIGURATION, range(8), f.gauge_id, NE, 2e-11, selection=selection))
        _, saved = f.publish('compact-EV', doc, np.stack(rows))
        handle = DatasetFieldHandle(saved)
        self.assertEqual(handle.time_selection, selection)
        for t in selection.times:
            basis = handle.read_modes(CFG, t, 0, NE).reshape(NE, -1).T
            original = f.vectors[t].reshape(NE, -1).T
            self.close_enough('compact physical-time projector', basis @ basis.conj().T, original @ original.conj().T, atol=2e-9, rtol=2e-9)

    def test_03_noise_realization_noncontiguous_parents_and_segmented_points(self):
        f = self.f
        rng = np.random.RandomState(952)
        for t in range(8):
            q, _ = np.linalg.qr(rng.randn(3, 3) + 1j * rng.randn(3, 3))
            expected = (f.vectors[t, MEDIUM].reshape(3, -1).T @ q[:, :2]).T.reshape(2, 4, 4, 4, 3)
            self.close_enough('QR independent mixing', f.qr['values'][t], expected, atol=2e-13, rtol=2e-13)
            d = f.qr['values'][t].reshape(2, -1)
            self.close_enough('QR raw Gram', d.conj() @ d.T, np.eye(2), atol=2e-10, rtol=0)
        self.close_enough('exact NV copies actual parents', f.exact['values'], f.vectors[:, MEDIUM], atol=0, rtol=0)
        self.assertEqual(f.qr['source'].groups[0].parent_modes, MEDIUM)
        self.assertEqual(len(set(pool for _, pool in f.qr['source'].time_pools)), 8)
        points = f.points['handle'].points
        self.assertEqual(points.pool_ids, f.points['points'].pool_ids)
        self.assertEqual(len(set(points.pool_ids)), 2)
        for t in range(8):
            self.assertEqual(points.point_pool_id(t), points.point_pool_id(t // 4 * 4))
            xyz = points.coordinates[t]
            expected = f.vectors[t][:, xyz[:, 2], xyz[:, 1], xyz[:, 0], :]
            self.close_enough('EV/P overlap', calc_overlap_row(f.vectors[t], xyz), expected, atol=0, rtol=0)
        selection = TimeSelectionRecord(8, (7, 0, 4, 3))
        selected = sample_spatial_points(SpatialSamplingDesign(LATTICE, 4, 4), seed=34, selection=selection)
        self.close_enough('selected points retain full RNG stream', selected.coordinates, points.coordinates[list(selection.times)], atol=0, rtol=0)
        self.assertEqual(selected.pool_ids[0], selected.pool_ids[2])
        self.assertEqual(selected.pool_ids[1], selected.pool_ids[3])
        self.assertNotEqual(selected.pool_ids[0], selected.pool_ids[1])
        # Rank-one QR is still unit norm, unlike the explicitly legacy U(1)
        # branch. A second realization must not acquire the first UUIDs.
        rank_one = NoisevectorGenerator(f.parent_handle, ((3,), (1,)), mode_groups=(MEDIUM,)).load(CFG)
        first = rank_one.calc(seed=29)
        first_pools = rank_one.describe()['time_pools']
        again = rank_one.calc(seed=29)
        self.close_enough('rank-one deterministic coordinates', first, again, atol=0, rtol=0)
        self.assertNotEqual(first_pools, rank_one.describe()['time_pools'])
        self.close_enough('rank-one raw QR norm', np.sum(abs(first.reshape(8, -1)) ** 2, axis=1), np.ones(8), atol=2e-10, rtol=0)

    def test_04_nine_propagators_every_physical_pair_and_six_saved_views(self):
        f = self.f
        bundle = f.bundle()
        self.assertEqual(set(bundle['prop'].sectors), {'vsv', 'vsd', 'vsp', 'dsv', 'dsd', 'dsp', 'psv', 'psd', 'psp'})
        for sector in sorted(bundle['prop'].sectors):
            announce(f'independent PR projection: {sector}, all 64 time pairs')
            for sink, source in product(range(8), repeat=2):
                actual = bundle['prop'].block(sink, source, sector)
                expected = f.reference_block(sector, sink, source, bundle['noise'], bundle['points'])
                self.close_enough(f'PR {sector}', actual, expected, atol=3e-10, rtol=2e-9)
        for dataset in bundle['saved']:
            with patch.object(DatasetReader, 'read', side_effect=AssertionError('reuse planner performed numeric IO')):
                plan = plan_indexed_perambulator_pairs(f.index, dataset.spec)
            self.assertEqual(plan['action'], 'reuse')
            self.assertEqual(plan['missing_pairs'], ())
        self.audit_metrics['saved_products'] = ['VSV', 'PSV', 'PSP', 'VSD', 'PSD', 'DSD']
        self.audit_metrics['projection_checks'] = 9 * 64
        self.audit_metrics['max_solve_residual'] = max(f.solve_residuals)
        serial = PerambulatorGenerator(LATTICE, DatasetFieldHandle(f.gauge), MASS, 2e-11, 100,
            clover_coeff_r=0, clover_coeff_t=0, t_boundary=-1,
            backend=AuditLUBackend(f), MRHS=False)
        serial.load(CFG)
        vector_basis = VectorBasis(f.v_handle)
        values = serial.calc(7, src=vector_basis, sink=[vector_basis])[0].values
        for sink in range(8):
            self.close_enough('serial versus MRHS actual source', values[sink], bundle['prop'].block(sink, 7, 'vsv'))

    def test_05_nine_elementals_paths_phases_and_local_weights(self):
        f = self.f
        bundle = dict(name='qr-elemental', noise=f.qr, points=f.points)
        paths = ((), (0,), (3,), (6,), (7,), (6, 0, 7), (0, 1, 3, 4))
        momenta = ((0, 0, 0), (1, -1, 1))
        for phase in ('anchor', 'midpoint', 'none'):
            for weighted in (False, True):
                announce(f'all-nine EL site oracle: phase={phase}, weighted={weighted}')
                gen, reader = f.elemental(bundle, weighted=weighted, paths=paths, momenta=momenta, phase=phase)
                for t, path_index, mi, sector in product(range(8), range(len(paths)), range(2), ELEMENTAL_SECTORS):
                    path = paths[path_index]
                    endpoint = (t + sum(1 if x == 6 else -1 if x == 7 else 0 for x in path)) % 8
                    left = f.basis(sector[0], t, f.qr, f.points, high=False)
                    right = f.basis(sector[-1], endpoint, f.qr, f.points, high=False)
                    expected = left.conj().T @ reference_matrix(f.links, path, t, momenta[mi], phase) @ right
                    if weighted:
                        weights_left = np.full(left.shape[1], 1.5 if sector[0] == 'd' else 16 if sector[0] == 'p' else 1.)
                        weights_right = np.full(right.shape[1], 1.5 if sector[-1] == 'd' else 16 if sector[-1] == 'p' else 1.)
                        weight = weights_left[:, None] * weights_right[None, :]
                        if sector == 'p2p' and f.points['points'].point_pool_id(t) == f.points['points'].point_pool_id(endpoint):
                            same = np.arange(12)[:, None] // 3 == np.arange(12)[None, :] // 3
                            weight = np.where(same, 16., 64 * 63 / (4 * 3))
                        if sector == 'd2d' and t == endpoint:
                            weight = np.full((2, 2), 3.)
                            np.fill_diagonal(weight, 1.5)
                        expected *= weight
                    actual = reader.block(gen.components[path_index], mi, t, sector)
                    self.close_enough(f'EL {sector} {phase} weighted={weighted}', actual.reshape(expected.shape), expected, atol=8e-10, rtol=2e-9)

    @staticmethod
    def diagram(names):
        n = len(names)
        adjacency = [[0] * n for _ in range(n)]
        symbols = []
        for i in range(n):
            adjacency[i][(i + 1) % n] = i + 1
            symbols.append(Propagator('u', Tag(i, 0), Tag((i + 1) % n, 0)))
        return Diagram(QuarkDiagram(adjacency), list(range(n)), list(names), [None, *symbols]), symbols

    def test_06_persisted_meson_source_dagger_full_times_dt_and_plan_replay(self):
        f = self.f
        bundle = f.bundle()
        # Registered EL written, closed and reopened before binding.
        gen, original = f.elemental(bundle, paths=((0,),), momenta=((1, -1, 1),))
        doc = elemental_dataset_document(f.gauge.spec, f.v_spec, gauge_id=f.gauge_id, basis_id=f.parent_id,
            components=[dict(path=[0])], momenta=((1, -1, 1),), modes=LOW, phase_convention='anchor')
        values = np.stack([original.block(gen.components[0], 0, t, 'v2v')[None, None] for t in range(8)])
        _, dataset = f.publish('meson-V-EL', DatasetSpec(doc), values)
        elemental = DatasetElemental(dataset)
        component = gen.components[0]
        coefficient = .7 + .2j
        operator = Operator.from_terms('weak-field-vector', (OperatorTerm(1, component, 0, coefficient),))
        vertices = dict(source=Meson(elemental, operator, source=True, phase_convention='anchor', requested_sectors=('v2v',)),
                        sink=Meson(elemental, operator, phase_convention='anchor', requested_sectors=('v2v',)))
        diagram, symbols = self.diagram(('source', 'sink'))
        result = calculate(-diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
            configuration=CFG, time_axes={0: range(8), 1: range(8)}, optimize='none', profile='default', memory_limit=512 * 2**20)
        expected = np.empty((8, 8), complex)
        g4 = np.kron(gamma(8), np.eye(3))
        for s, t in product(range(8), repeat=2):
            es = f.basis('v', s, f.exact, f.points)
            et = f.basis('v', t, f.exact, f.points)
            os = es.conj().T @ reference_matrix(f.links, (0,), s, (1, -1, 1), 'anchor') @ es
            ot = et.conj().T @ reference_matrix(f.links, (0,), t, (1, -1, 1), 'anchor') @ et
            source = coefficient * np.kron(gamma(1), os)
            source = g4 @ source.conj().T @ g4
            sink = coefficient * np.kron(gamma(1), ot)
            st = f.reference_block('vsv', s, t, f.exact, f.points).transpose(0, 2, 1, 3).reshape(12, 12)
            ts = f.reference_block('vsv', t, s, f.exact, f.points).transpose(0, 2, 1, 3).reshape(12, 12)
            expected[s, t] = -np.trace(source @ st @ sink @ ts)
        self.close_enough('full 64-pair meson Wick/source adjoint', result.value, expected)
        save_correlator(f.root / 'meson-result', result, source_axis=0, sink_axis=1, boundary='periodic', coordinate_domain='physical', physical_lt=8)
        save_correlator(f.root / 'meson-dt', result, average_tsrc=True, source_axis=0, sink_axis=1,
                       boundary='periodic', coordinate_domain='physical', physical_lt=8)
        self.close_enough('result roundtrip bytes', load_correlator(f.root / 'meson-result').value, result.value, atol=0, rtol=0)
        dt = load_correlator(f.root / 'meson-dt')
        self.close_enough('source average physical dt', dt.value, np.asarray([np.mean([expected[s, (s + d) % 8] for s in range(8)]) for d in range(8)]))
        np.testing.assert_array_equal(dt.reduction['counts'], np.full(8, 8))
        planfile = f.root / 'meson-plan.json'
        save_calculation_plan(planfile, result)
        replay = load_calculation_plan(planfile, expected=result).run()
        self.close_enough('saved plan replay', replay.value, result.value)

    def persisted_local_elemental(self, bundle, weighted):
        """Nine actual EL datasets, including row-sparse P/P and D provenance."""
        f = self.f
        gen, original = f.elemental(bundle, weighted=weighted)
        d_doc = bundle['noise']['reader'].spec.document
        d_doc['conventions']['dilution_source'] = bundle['noise']['source'].document
        specs = dict(v=f.v_spec, d=DatasetSpec(d_doc))
        identities = dict(v=f.parent_id, d=bundle['noise']['id'])
        datasets = {}
        for sector in ELEMENTAL_SECTORS:
            if 'p' not in sector:
                left, right = sector[0], sector[-1]
                document = elemental_dataset_document(f.gauge.spec, specs[left], gauge_id=f.gauge_id,
                    basis_id=identities[left], product=sector, components=[dict(path=[])], momenta=((0, 0, 0),),
                    modes=LOW if left == 'v' else range(bundle['noise']['values'].shape[1]),
                    right_basis=specs[right], right_basis_id=identities[right],
                    right_modes=LOW if right == 'v' else range(bundle['noise']['values'].shape[1]),
                    phase_convention='anchor', elemental_weighted=weighted)
            else:
                kind = sector[0] if sector[-1] == 'p' else sector[-1]
                vector = 'v' if kind == 'p' else kind
                document = point_elemental_dataset_document(f.gauge.spec, None if sector == 'p2p' else specs[vector],
                    bundle['points']['spec'], product=sector, gauge_id=f.gauge_id,
                    basis_id=None if sector == 'p2p' else identities[vector], points_id=bundle['points']['id'],
                    components=[dict(path=[])], momenta=((0, 0, 0),),
                    modes=() if sector == 'p2p' else LOW if vector == 'v' else range(bundle['noise']['values'].shape[1]),
                    point_labels=range(bundle['points']['points'].num_points), phase_convention='anchor', elemental_weighted=weighted)
            rows = []
            for t in range(8):
                if sector == 'p2p':
                    value = gen.calc_path_blocks(t)['p2p'][0][0].values
                else:
                    value = original.block(LOCAL, 0, t, sector)
                rows.append(value[None, None].copy())
            _, datasets[sector] = f.publish(f"{bundle['name']}/EL-{weighted}-{sector}", DatasetSpec(document), rows)
        return DatasetElemental(datasets, point_dataset=bundle['points']['reader'],
                                sampling_groups=gen.sampling_groups, sampling_pool_ids=gen.sampling_pool_ids)

    def two_local_value(self, bundle, reader, times, weighted):
        operator = Operator.from_terms('local-pseudoscalar', (OperatorTerm(15, LOCAL, 0),))
        vertices = {name: Meson(reader, operator, requested_sectors=ELEMENTAL_SECTORS, phase_convention='anchor')
                    for name in ('early', 'late')}
        diagram, symbols = self.diagram(('early', 'late'))
        return calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
            configuration=CFG, time_axes={0: (times[0],), 1: (times[1],)}, elemental_weighted=weighted,
            groups_independent=True, optimize='none', profile='default', memory_limit=2 * 2**30).value.item()

    def full_site_propagator(self, sink, source):
        f = self.f
        key = 'full-site', source
        if key not in f._solutions:
            f._solutions[key] = f.reference_lu.solve(f.embedding(np.eye(192), source))
        # (site,spin,color; source-spin,source-site,color) -> spin-first site matrix.
        block = f._solutions[key].reshape(*FIELD_SHAPE, 768)[sink]
        return block.transpose(3, 0, 1, 2, 4, 5).reshape(768, 768)

    def test_07_exact_low_medium_high_closes_full_space_after_real_persistence(self):
        f = self.f
        points = f.make_points('full-space-points', 64, 4)
        bundle = f.bundle('full-space', points=points, times=(4, 0))
        # Completeness uses all spatial sites; no single random realization is
        # asserted to equal an ensemble average.
        for t in (0, 4):
            v, d, q = (f.basis(k, t, f.exact, points) for k in ('v', 'd', 'p'))
            self.close_enough('V+D+Q completeness', v @ v.conj().T + d @ d.conj().T + q @ q.conj().T,
                              np.eye(192), atol=2e-9, rtol=2e-9)
        first = self.full_site_propagator(0, 4)
        second = self.full_site_propagator(4, 0)
        g = np.kron(gamma(15), np.eye(192))
        target = np.trace(g @ first @ g @ second)
        estimates = []
        for weighted in (False, True):
            announce(f'full-space nine-sector correlator, persisted EL/PR, weighted={weighted}')
            reader = self.persisted_local_elemental(bundle, weighted)
            estimates.append(self.two_local_value(bundle, reader, (0, 4), weighted))
            self.close_enough('full Wilson trace nine-sector closure', estimates[-1], target, atol=2e-8, rtol=3e-9)
        self.close_enough('full-space raw versus preweighted', estimates[0], estimates[1], atol=2e-8, rtol=3e-9)
        self.audit_metrics['full_trace'] = [float(target.real), float(target.imag)]

    def test_08_actual_qr_and_random_high_points_match_literal_realization_trace(self):
        f = self.f
        bundle = f.bundle('qr-random-high', noise=f.qr, times=(4, 0))
        kernels = []
        for t in (0, 4):
            raw = np.concatenate([f.basis(k, t, f.qr, f.points, high=False) for k in ('v', 'd', 'p')], axis=1)
            projected = np.concatenate([f.basis(k, t, f.qr, f.points) for k in ('v', 'd', 'p')], axis=1)
            weights = np.ones((17, 17))
            weights[3:5, :] *= 1.5; weights[:, 3:5] *= 1.5
            weights[5:, :] *= 16.; weights[:, 5:] *= 16.
            weights[3:5, 3:5] = np.where(np.eye(2, dtype=bool), 1.5, 3.)
            same = np.arange(12)[:, None] // 3 == np.arange(12)[None, :] // 3
            weights[5:, 5:] = np.where(same, 16., 64 * 63 / (4 * 3))
            # Include finite-sample V/P and D/P cross terms. Raw EL uses
            # delta points; PR projects its P legs into Q. Orthogonal
            # subspaces do not make these finite-sample cross terms vanish.
            kernels.append(projected @ ((raw.conj().T @ raw) * weights) @ projected.conj().T)
        for sector in sorted(bundle['prop'].sectors):
            for sink, source in product((0, 4), repeat=2):
                self.close_enough('QR PR ' + sector, bundle['prop'].block(sink, source, sector),
                    f.reference_block(sector, sink, source, f.qr, f.points))
        s04, s40 = self.full_site_propagator(0, 4), self.full_site_propagator(4, 0)
        operators = [np.kron(gamma(15), value) for value in kernels]
        target = np.trace(operators[0] @ s04 @ operators[1] @ s40)
        results = []
        for weighted in (False, True):
            reader = self.persisted_local_elemental(bundle, weighted)
            results.append(self.two_local_value(bundle, reader, (0, 4), weighted))
            self.close_enough('QR plus random high literal realization', results[-1], target, atol=3e-8, rtol=3e-9)
        self.close_enough('random realization raw versus preweighted', *results, atol=3e-8, rtol=3e-9)
        self.audit_metrics['statistical_scope'] = 'One literal weighted realization; no Monte Carlo unbiasedness claim.'

    def point_loop_oracle(self, t, u):
        """Literal four endpoint sums with falling-factorial probabilities.

        No production compensation/scene routines or contraction labels.
        Color can repeat inside one sampled site; distinctness counts XYZ
        point labels within each actual pool, never colors or physical times.
        """
        f = self.f
        endpoints = (t, (t + 1) % 8, u, (u + 1) % 8)
        p = f.points['points']
        pool_ids = [p.point_pool_id(x) for x in endpoints]
        matrices = []
        for start in (t, u):
            left = f.basis('p', start, f.exact, f.points, high=False)
            right = f.basis('p', (start + 1) % 8, f.exact, f.points, high=False)
            matrices.append(left.conj().T @ reference_matrix(f.links, (6,), start, (0, 0, 0), 'anchor') @ right)
        propagators = [f.reference_block('psp', endpoints[1], endpoints[2], f.exact, f.points),
                       f.reference_block('psp', endpoints[3], endpoints[0], f.exact, f.points)]
        spin_blocks = [value.transpose(0, 2, 3, 1, 4, 5).reshape(4, 12, 4, 12) for value in propagators]
        gp = [np.einsum('uv,vbwc->ubwc', gamma(1), value) for value in spin_blocks]
        tensor = np.einsum('ubwc,wdua->abcd', *gp)
        counts = np.zeros((4, 4, 4, 4))
        for labels in np.ndindex(counts.shape):
            weight = 1.
            for pool in dict.fromkeys(pool_ids):
                r = len({label for label, owner in zip(labels, pool_ids) if owner == pool})
                weight *= prod(64 - i for i in range(r)) / prod(4 - i for i in range(r))
            counts[labels] = weight
        for axis in range(4):
            counts = counts.repeat(3, axis=axis)
        return np.einsum('ab,cd,abcd,abcd->', *matrices, tensor, counts)

    def test_09_temporal_gauge_links_match_segment_pools_and_joint_coefficients(self):
        f = self.f
        bundle = f.bundle()
        cases = ((0, 2), (2, 3), (3, 4), (7, 0))
        component = ElementalComponent.path((6,))
        operator = Operator.from_terms('temporal-point-path', (OperatorTerm(1, component, 0),))
        diagram, symbols = self.diagram(('early', 'late'))
        expected = {pair: self.point_loop_oracle(*pair) for pair in cases}
        self.assertTrue(all(abs(value) > 1e-12 for value in expected.values()), 'every pool-boundary case must have a nonzero signal')
        for weighted in (False, True):
            _, reader = f.elemental(bundle, paths=((6,),), weighted=weighted)
            vertices = {name: Meson(reader, operator, requested_sectors=('p2p',), phase_convention='anchor')
                        for name in ('early', 'late')}
            for t, u in cases:
                for time in (t, u):
                    metadata = vertices['early'].describe_factors(time, 'p2p')[0]
                    self.assertEqual(metadata.point_pool_ids['bar'], f.points['points'].point_pool_id(time))
                    self.assertEqual(metadata.point_pool_ids['field'], f.points['points'].point_pool_id((time + 1) % 8))
                announce(f'temporal P/P loop t={t},u={u}, weighted={weighted}')
                result = calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
                    configuration=CFG, time_axes={0: (t,), 1: (u,)}, groups_independent=True,
                    elemental_weighted=weighted, optimize='none', profile='default', memory_limit=512 * 2**20)
                self.close_enough(f'temporal four-label oracle {t},{u}', result.value.item(), expected[t, u], atol=2e-8, rtol=3e-9)
            with self.assertRaisesRegex(ValueError, 'independence'):
                calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
                    configuration=CFG, time_axes={0: (3,), 1: (4,)}, elemental_weighted=weighted,
                    execute=False, optimize='none', profile='default')
        self.audit_metrics['cases'] = ['inside shared segment', 'partial segment crossing', 'segment boundary', 'periodic temporal wrap']

    def test_10_baryon_three_legs_source_adjoint_and_literal_scalar_contraction(self):
        from test_baryon_elemental import determinant_reference
        from test_baryon_production_measurement import _scalar_reference
        f = self.f
        bundle = f.bundle()
        paths = ((0,), (), (3,))
        component = BaryonComponent.paths(paths)
        vectors = f.vectors[:, LOW]
        gen = BaryonElementalGenerator(f.links, vectors, ((1, 0, 0),),
            [dict(component=component, paths=paths, coefficient=.3 + .2j)],
            ne_block_size=2, spatial_block_size=17)
        values, independent = [], []
        for t in range(8):
            row = gen.calc_path_blocks(t)['vvv']
            reference = (.3 + .2j) * determinant_reference(f.links, (vectors,) * 3, paths, t, (1, 0, 0))
            self.close_enough('baryon epsilon versus determinant', row[0, 0], reference)
            values.append(row.copy()); independent.append(reference)
        basis_document = f.parent.spec.document
        basis_document['axes'][1]['coordinates'] = list(LOW)
        basis_view = DatasetSpec(basis_document)
        # The complex spatial coefficient is stored as a projected recipe.
        recipe = ({'id': 0, 'terms': [dict(paths=paths, coefficient=[.3, .2])]},)
        coordinates, projected = normalize_baryon_components('projected', recipe)
        document = baryon_dataset_document(f.gauge.spec, (basis_view,) * 3,
            gauge_id=f.gauge_id, basis_ids=(f.parent_id,) * 3, product='projected',
            component_coordinates=coordinates, momentum_list=((1, 0, 0),),
            phase_convention='anchor', path_boundary='periodic', projected_rows=projected,
            parent_actual_ids=(f.gauge_id, f.parent_id, f.parent_id, f.parent_id))
        _, dataset = f.publish('baryon-EL', DatasetSpec(document), values)
        reader = DatasetBaryonElemental(dataset)
        component = reader.components[0]
        source_spin = SparseSpinTensor(((0, 1, 2, .5 + .2j), (3, 2, 1, -.25 + .4j)))
        sink_spin = SparseSpinTensor(((0, 1, 2, -.3 + .5j), (3, 2, 1, .7 - .1j)))
        cs, ct = .7 - .4j, -.2 + .8j
        vertices = dict(source=Baryon(reader, BaryonOperator.from_terms('source', (BaryonOperatorTerm(source_spin, component, cs, ('u', 'd', 'u'), 0),)), source=True),
                        sink=Baryon(reader, BaryonOperator.from_terms('sink', (BaryonOperatorTerm(sink_spin, component, ct, ('d', 'u', 'u'), 0),))))
        edges = [[0, 0, 0] for _ in range(3)]
        source_slots = (1, 2, 0)
        for i, source in enumerate(source_slots):
            edges[source][i] = i + 1
        zero = [[0, 0, 0] for _ in range(3)]
        symbols = tuple(Propagator(flavor, Tag(source, 0), Tag(3 + i, 1))
                        for i, (flavor, source) in enumerate(zip(('d', 'u', 'u'), source_slots)))
        diagram = Diagram(QuarkDiagram([[zero, edges], [zero, zero]]), [0, 1], ['source', 'sink'], [None, *symbols])
        for source, sink in ((0, 4), (7, 0)):
            result = calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
                               configuration=CFG, time_axes={0: source, 1: sink}, optimize='none', profile='default')
            peram = f.reference_block('vsv', sink, source, f.exact, f.points)
            expected = _scalar_reference(independent[source], independent[sink], [peram] * 3,
                                         source_spin.dense(), sink_spin.dense(), cs, ct)
            self.close_enough('baryon literal spin/mode sum', result.value, expected, atol=2e-10, rtol=3e-9)

    def test_11_compact_temporal_current_keeps_physical_neighbors_and_missing_endpoints(self):
        f = self.f
        bundle = f.bundle()
        insertion = ConservedVectorInsertion(direction=3).operator()
        normal = Operator.from_terms('normal', (OperatorTerm(15, LOCAL, 0),))
        diagram, symbols = self.diagram(('source', 'current', 'sink'))
        # The slot selection is reordered and nonconsecutive; endpoint data
        # remain full-time so physical t+1 is present even if not an anchor.
        selection = TimeSelectionRecord(8, (7, 0, 4, 3))
        _, dense = f.elemental(bundle, paths=((), (6,), (7,)), phase='midpoint')
        vertices = dict(source=Meson(dense, normal, source=True, requested_sectors=('v2v',)),
                        current=Meson(dense, insertion, requested_sectors=ELEMENTAL_SECTORS),
                        sink=Meson(dense, normal, requested_sectors=('v2v',)))
        expected = {}
        for current in (0, 3, 7):
            result = calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
                configuration=CFG, time_axes={0: (4,), 1: (current,), 2: (0,)}, groups_independent=True,
                optimize='none', profile='default', memory_limit=2 * 2**30)
            # Independently sum all four directed conserved-current terms,
            # using their explicit physical endpoints and spin coefficients.
            target = 0j
            v4, v0 = f.basis('v', 4, f.exact, f.points), f.basis('v', 0, f.exact, f.points)
            source_op = np.kron(gamma(8) @ gamma(15).conj().T @ gamma(8), v4 @ v4.conj().T)
            sink_op = np.kron(gamma(15), v0 @ v0.conj().T)
            for term in insertion.terms:
                anchor = (current + term.anchor_time_offset) % 8
                path = term.component.key.steps
                endpoint = (anchor + term.component.time_offsets[1]) % 8
                raw_left = np.concatenate([f.basis(k, anchor, f.exact, f.points, high=False) for k in ('v', 'd', 'p')], axis=1)
                raw_right = np.concatenate([f.basis(k, endpoint, f.exact, f.points, high=False) for k in ('v', 'd', 'p')], axis=1)
                left = np.concatenate([f.basis(k, anchor, f.exact, f.points) for k in ('v', 'd', 'p')], axis=1)
                right = np.concatenate([f.basis(k, endpoint, f.exact, f.points) for k in ('v', 'd', 'p')], axis=1)
                elemental = raw_left.conj().T @ reference_matrix(f.links, path, anchor) @ raw_right
                weight = np.ones(elemental.shape)
                weight[6:, :] *= 16.; weight[:, 6:] *= 16.
                if f.points['points'].point_pool_id(anchor) == f.points['points'].point_pool_id(endpoint):
                    same = np.arange(12)[:, None] // 3 == np.arange(12)[None, :] // 3
                    weight[6:, 6:] = np.where(same, 16., 64 * 63 / (4 * 3))
                spatial = left @ (elemental * weight) @ right.conj().T
                vertex = term.coefficient * np.kron(gamma(term.gamma), spatial)
                matrix = self.full_site_propagator
                # Conserved temporal current reverses its graph-side mapping.
                # Follow this Diagram's directed 0->1->2->0 cycle. A
                # reversed three-vertex trace is a different observable.
                target += np.trace(source_op @ matrix(4, 0) @ sink_op @ matrix(0, anchor) @ vertex @ matrix(endpoint, 4))
            self.close_enough(f'temporal conserved-current trace t={current}', result.value.item(), target, atol=2e-10, rtol=3e-9)
            expected[current] = target
        _, compact = f.elemental(bundle, paths=((), (6,), (7,)), phase='midpoint', selection=selection)
        for sector, component, slot in product(ELEMENTAL_SECTORS, compact.components, range(selection.Nt)):
            self.close_enough('compact EL physical values ' + sector, compact.block(component, 0, slot, sector),
                dense.block(component, 0, selection.times[slot], sector), atol=0, rtol=0)
        compact_vertices = dict(source=Meson(compact, normal, source=True, requested_sectors=('v2v',)),
                                current=Meson(compact, insertion, requested_sectors=ELEMENTAL_SECTORS),
                                sink=Meson(compact, normal, requested_sectors=('v2v',)))
        # Mark the full physical propagator coverage explicitly when coupling
        # it to compact anchor slots. Retain actual endpoint IDs and labels.
        sectors, bindings = {}, {}
        for sector in (a + 's' + b for a in ('v', 'd', 'p') for b in ('v', 'd', 'p')):
            sectors[sector] = np.stack([np.stack([bundle['prop'].block(t, s, sector) for t in range(8)]) for s in range(8)])
            metadata = bundle['prop'].describe_block(0, 0, sector)
            for side, kind in (('sink', sector[0]), ('source', sector[-1])):
                bindings[sector, side] = metadata.axis_bindings[side + ('_point' if kind == 'p' else '_mode')]
        full_selection = TimeSelectionRecord(8, tuple(range(8)))
        marked = ArrayPropagator(sectors, time_convention='absolute', cfg=CFG,
            source_selection=full_selection, sink_selection=full_selection, basis_endpoints=bindings)
        for slot, physical in ((3, 3), (0, 7)):
            compact_result = calculate(diagram, vertex_map=compact_vertices, propagator_map={s: marked for s in symbols},
                configuration=CFG, time_axes={0: (2,), 1: (slot,), 2: (1,)}, groups_independent=True,
                optimize='none', profile='default', memory_limit=2 * 2**30)
            self.close_enough('compact physical current matches full-time trace', compact_result.value.item(), expected[physical], atol=2e-10, rtol=3e-9)
        # Physical current at 0 needs anchor 1, absent from compact EL coverage.
        with self.assertRaises((KeyError, ValueError)):
            calculate(diagram, vertex_map=compact_vertices, propagator_map={s: marked for s in symbols},
                configuration=CFG, time_axes={0: (2,), 1: (1,), 2: (1,)}, groups_independent=True,
                optimize='none', profile='default', execute=False)
        self.audit_metrics['compact_selection'] = list(selection.times)

    def test_12_gauge_preparation_covariance_and_open_path_boundary(self):
        from scipy.linalg import expm
        f = self.f
        canonical = f.links.transpose(4, 0, 1, 2, 3, 5, 6)
        expected = canonical.copy()
        # One spatial stout step, independent point loops + scipy expm.
        rho = .08
        for t, z, y, x, mu in product(range(8), range(4), range(4), range(4), range(3)):
            pos = np.array((x, y, z))
            def link(position, direction):
                px, py, pz = position % 4
                return f.links[t, pz, py, px, direction]
            emu = np.eye(3, dtype=int)[mu]
            staple = np.zeros((3, 3), complex)
            for nu in range(3):
                if nu == mu:
                    continue
                enu = np.eye(3, dtype=int)[nu]
                staple += link(pos, nu) @ link(pos + enu, mu) @ link(pos + emu, nu).conj().T
                staple += link(pos - enu, nu).conj().T @ link(pos - enu, mu) @ link(pos - enu + emu, nu)
            omega = rho * staple @ link(pos, mu).conj().T
            q = .5j * (omega.conj().T - omega)
            q -= np.trace(q) / 3 * np.eye(3)
            expected[mu, t, z, y, x] = expm(1j * q) @ link(pos, mu)
        self.close_enough('spatial stout point/expm oracle', stout_smear(canonical, 1, rho), expected, atol=3e-12, rtol=3e-12)
        self.close_enough('SU3 projection of actual SU3', project_su3(canonical), canonical, atol=3e-12, rtol=3e-12)
        rng = np.random.default_rng(119)
        raw = rng.normal(size=(8, 4, 4, 4, 3, 3)) + 1j * rng.normal(size=(8, 4, 4, 4, 3, 3))
        rotations = np.linalg.qr(raw)[0]
        rotations[..., :, 0] /= np.linalg.det(rotations)[..., None]
        links = f.links.copy()
        for mu, axis in enumerate((3, 2, 1, 0)):
            neighbor = np.roll(rotations, -1, axis=axis)
            links[..., mu, :, :] = rotations @ f.links[..., mu, :, :] @ neighbor.conj().swapaxes(-1, -2)
        vectors = np.einsum('tzyxab,tizyxb->tizyxa', rotations, f.vectors)
        components = tuple(ElementalComponent.path(path) for path in ((0, 1, 3), (6,), (7,), (6, 7)))
        table = [dict(component=c, path=c.key.steps, coefficient=.3 + .7j) for c in components]
        original = ElementalGenerator(f.links, f.vectors, ((1, -1, 1),), table, path_boundary='open')
        transformed = ElementalGenerator(links, vectors, ((1, -1, 1),), table, path_boundary='open')
        for t in (0, 3, 7):
            first = original.calc_path_blocks(t)['v2v'].copy()
            second = transformed.calc_path_blocks(t)['v2v'].copy()
            self.close_enough('gauge-covariant path elemental', first, second, atol=3e-10, rtol=3e-9)
            for i, c in enumerate(components):
                endpoint = t + c.time_offsets[1]
                if 0 <= endpoint < 8:
                    expected_matrix = reference_matrix(f.links, c.key.steps, t, (1, -1, 1), boundary='open')
                    left = f.vectors[t].reshape(NE, -1).T
                    right = f.vectors[endpoint].reshape(NE, -1).T
                    self.close_enough('open path physical endpoints', first[i, 0], (.3 + .7j) * left.conj().T @ expected_matrix @ right)
                else:
                    self.close_enough('open boundary absent endpoint is zero', first[i, 0], np.zeros((NE, NE)), atol=0, rtol=0)

    def test_13_ordered_derivatives_complex_combinations_and_momentum_order(self):
        f = self.f
        terms = (((0, 1), .2 + .5j), ((1, 0), -.3), ((), .7))
        table = [dict(component='ordered-derivative', path=path, coefficient=c) for path, c in terms]
        momenta = {'nonzero': '1 -1 1', 17: (0, 0, 0)}
        gen = ElementalGenerator(f.links, f.vectors, operator_table=table, momentum_dict=momenta, ne_block_size=2)
        self.assertEqual(gen.momenta, ((1, -1, 1), (0, 0, 0)))
        for t in range(8):
            value = gen.calc_derivative(t)['v2v'].copy()
            vectors = f.vectors[t].reshape(NE, -1).T
            ds = [reference_matrix(f.links, (mu,), t) - reference_matrix(f.links, (mu + 3,), t) for mu in range(3)]
            for mi, momentum in enumerate(gen.momenta):
                phase = reference_matrix(f.links, (), t, momentum)
                expected = np.zeros((NE, NE), complex)
                for directions, coefficient in terms:
                    for assignment in product((0, 1), repeat=len(directions)):
                        left, right = vectors.copy(), vectors.copy()
                        for mu in [d for d, side in zip(directions, assignment) if side == 0][::-1]:
                            left = ds[mu] @ left
                        for mu in [d for d, side in zip(directions, assignment) if side == 1]:
                            right = ds[mu] @ right
                        expected += coefficient * (-1) ** sum(assignment) * left.conj().T @ phase @ right
                self.close_enough('ordered two-sided derivative', value[0, mi], expected, atol=3e-10, rtol=3e-9)
        self.audit_metrics['combination'] = '0.2+0.5j D_xD_y -0.3 D_yD_x +0.7 local'

    def test_14_persistence_missing_coverage_resume_and_wrong_scientific_identity(self):
        f = self.f
        # Resume must use actual completed coverage; a directory alone does
        # not certify the complete gauge. Values are from the full fixture.
        destination = f.root / 'resume-gauge'
        with DatasetWriter(destination, f.gauge_spec, provenance=dict(test='coverage-audit')) as writer:
            writer.write(0, f.links[0])
        with DatasetWriter(destination, f.gauge_spec, provenance=dict(test='coverage-audit'), resume=True) as writer:
            self.assertNotIn(0, writer.missing)
            self.assertEqual(tuple(writer.missing), tuple(range(1, 8)))
            for t in writer.missing:
                writer.write(t, f.links[t])
            writer.publish()
        with DatasetReader(destination) as reader:
            self.close_enough('resumed gauge bytes', reader.read(time=7), f.links[7], atol=0, rtol=0)
        bundle = f.bundle()
        gen, reader = f.elemental(bundle)
        operator = Operator.from_terms('local', (OperatorTerm(15, LOCAL, 0),))
        vertices = {name: Meson(reader, operator, requested_sectors=ELEMENTAL_SECTORS, phase_convention='anchor')
                    for name in ('a', 'b')}
        diagram, symbols = self.diagram(('a', 'b'))
        # A stored preweight mismatch must be rejected in metadata planning,
        # before any contraction data is requested.
        with patch.object(ArrayElemental, 'block', side_effect=AssertionError('invalid request read EL values')):
            with self.assertRaisesRegex(ValueError, 'elemental_weighted'):
                calculate(diagram, vertex_map=vertices, propagator_map={s: bundle['prop'] for s in symbols},
                    configuration=CFG, time_axes={0: (0,), 1: (4,)}, elemental_weighted=True,
                    groups_independent=True, execute=False, optimize='none', profile='default')
        wrong_partition = ModePartition('unrelated-parent-basis', LOW, (MEDIUM,))
        with self.assertRaisesRegex(ValueError, 'identity'):
            HighmodeBasis(f.parent_handle, wrong_partition)
