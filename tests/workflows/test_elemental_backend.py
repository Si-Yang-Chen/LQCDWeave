"""Explicit elemental backend routing; NumPy protocol checks are not GPU evidence."""

from collections import Counter
import unittest
from unittest import mock

import numpy as np

from test_production_dataset import _ProductionDatasetFixture
from production.elemental.test_elemental import reference_matrix
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from lqcdweave.production.elemental.planning import prepare_elemental_stage


class _ArrayModule:
    """Count selected-backend calls without pretending NumPy arrays are on GPU."""
    __name__ = "cupy"
    ndarray = np.ndarray

    def __init__(self):
        self.calls = Counter()

    def __getattr__(self, name):
        value = getattr(np, name)
        if isinstance(value, type) or not callable(value):
            return value
        def invoke(*args, **kwargs):
            self.calls[name] += 1
            return value(*args, **kwargs)
        return invoke


class ElementalBackendTests(_ProductionDatasetFixture):
    def _check_stage(self, backend, to_host):
        paths, momenta = ((), (0,), (3, 1)), ((0, 0, 0), (1, -1, 1))
        selection = TimeSelectionRecord(self.sizes[3], (2, 0))
        with DatasetReader(self.root / "gauge") as gauge, DatasetReader(self.root / "basis") as basis:
            generator, spec, provenance, times, cfg = prepare_elemental_stage(
                gauge, basis, paths=paths, momentum_list=momenta, usedNe=2,
                ne_block_size=1, selection=selection, array_module=backend)
            self.assertIs(generator.xp, backend)
            self.assertEqual(provenance["production"]["array_module"], backend.__name__)
            self.assertEqual(tuple(times), selection.times)
            self.assertEqual(spec.coordinates("time"), [0, 1])
            generator.load(cfg["configuration"])
            with DatasetWriter(self.root / "backend-out", spec, provenance=provenance) as writer:
                for row, time in enumerate(times):
                    value = generator.calc_path_blocks(time)["v2v"]
                    self.assertIsInstance(value, backend.ndarray)
                    actual = to_host(value)
                    vectors = self.vectors[time, :2].reshape(2, -1)
                    expected = np.empty((len(paths), len(momenta), 2, 2), complex)
                    for pi, path in enumerate(paths):
                        for mi, momentum in enumerate(momenta):
                            expected[pi, mi] = vectors.conj() @ reference_matrix(
                                self.gauge, path, time, momentum, phase="anchor") @ vectors.T
                    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
                    # DatasetWriter is host-only: the transfer is explicit here.
                    writer.write(row, actual)
                writer.publish()
        with DatasetReader(self.root / "backend-out") as output:
            self.assertEqual(output.spec.document, spec.document)
            self.assertEqual(output.read(time=0).dtype, np.dtype("complex128"))

    def test_default_keeps_numpy_and_its_provenance(self):
        with DatasetReader(self.root / "gauge") as gauge, DatasetReader(self.root / "basis") as basis:
            generator, _, provenance, _, _ = prepare_elemental_stage(gauge, basis)
            self.assertIs(generator.xp, np)
            self.assertEqual(provenance["production"]["array_module"], "numpy")

    def test_selected_backend_reaches_kernel_and_explicit_dataset_handoff(self):
        backend = _ArrayModule()
        self._check_stage(backend, np.asarray)
        for operation in ("array", "asarray", "empty", "exp"):
            self.assertGreater(backend.calls[operation], 0, operation)

    def test_public_table_frontend_keeps_complex_projection_on_selected_backend(self):
        backend = _ArrayModule()
        rows = [dict(component="mixed", path=(0,), coefficient=0.3 + 0.7j),
                dict(component="mixed", path=(3,), coefficient=-0.5 + 0.2j)]
        momenta = ((1, -1, 1),)
        generator = ElementalGenerator(self.gauge, self.vectors, momenta, rows,
            Ne=2, array_module=backend, ne_block_size=1, phase_convention="anchor")
        for time, blocks in generator.iter_slices():
            vectors = self.vectors[time, :2].reshape(2, -1)
            spatial = sum(row["coefficient"] * reference_matrix(
                self.gauge, row["path"], time, momenta[0], phase="anchor") for row in rows)
            expected = vectors.conj() @ spatial @ vectors.T
            np.testing.assert_allclose(blocks["v2v"][0, 0], expected, rtol=3e-12, atol=3e-12)
        self.assertGreater(backend.calls["einsum"], 0)

    def _check_grouped_projection(self, backend, to_host):
        from lqcdweave.production.elemental import projection
        rows = [dict(id=0,terms=[dict(path=[0,1],coefficient=[.3,.7]),
                                 dict(path=[1,0],coefficient=[-.2,.4])]),
                dict(id=1,terms=[dict(path=[0],coefficient=[.6,-.1]),
                                 dict(path=[0,0,0,0],coefficient=[-.3,.2])]),
                dict(id=2,terms=[dict(path=[],coefficient=[.5,.2]),
                                 dict(path=[],coefficient=[-.5,-.2])])]
        momenta = [(0,0,0),(1,-1,0),(-1,0,1)]
        for phase in ('midpoint','anchor','none'):
            for cache_bytes in (0,512*1024**2):
                with self.subTest(phase=phase,cache_bytes=cache_bytes):
                    generator = ElementalGenerator(self.gauge,self.vectors,momenta,rows,
                        array_module=backend,phase_convention=phase)
                    with mock.patch.object(projection,'_TRANSPORT_CACHE_BYTES',cache_bytes), \
                         mock.patch.object(projection,'_WEIGHT_BYTES',2*self.vectors[0].nbytes), \
                         mock.patch.object(projection,'path_transport',wraps=projection.path_transport) as transport:
                        actual = to_host(generator.calc_path_blocks(1)['v2v']).copy()
                    self.assertTrue(transport.call_args_list)
                    self.assertTrue(all(call.kwargs['array_module'] is backend
                                        for call in transport.call_args_list))
                    vectors = self.vectors[1].reshape(3,-1)
                    for ri,row in enumerate(rows):
                        for mi,momentum in enumerate(momenta):
                            matrix = sum(complex(*term['coefficient'])*reference_matrix(
                                self.gauge,term['path'],1,momentum,phase=phase)
                                for term in row['terms'])
                            np.testing.assert_allclose(actual[ri,mi],
                                vectors.conj() @ matrix @ vectors.T,atol=3e-12,rtol=3e-12)
                    written = np.empty_like(actual)

                    def write(ri,mi,sector,slices,value):
                        self.assertIsInstance(value,backend.ndarray)
                        written[ri,mi] = to_host(value)

                    # Exercise the numerical callback path without frontend copying.
                    normalized = [dict(id=i,terms=row['terms']) for i,row in enumerate(rows)]
                    self.assertIsNone(generator._engine.calc_path_blocks(1,rows=normalized,write=write))
                    np.testing.assert_allclose(written,actual,atol=3e-12,rtol=3e-12)

    def test_grouped_projection_uses_selected_array_module_on_cpu_protocol(self):
        backend = _ArrayModule()
        self._check_grouped_projection(backend,np.asarray)
        for name in ('stack','zeros','einsum'):
            self.assertGreater(backend.calls[name],0,name)


class ElementalBackendCuPyTests(_ProductionDatasetFixture):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as error:
            raise unittest.SkipTest(f"requires working CuPy/CUDA: {error}")
        cls.cp = cupy

    def test_stage_device_output_and_explicit_host_writer(self):
        ElementalBackendTests._check_stage(self, self.cp, self.cp.asnumpy)

    def test_grouped_projection_device_output_and_streamed_values(self):
        ElementalBackendTests._check_grouped_projection(self,self.cp,self.cp.asnumpy)
