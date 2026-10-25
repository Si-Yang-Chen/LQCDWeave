"""Pure perambulator memory estimates and row-copy lifecycle checks."""

from pathlib import Path
import unittest
import weakref
from unittest.mock import patch

import numpy as np

import fixtures.perambulator_script as perambulator_workflow
from lqcdweave.contracts.memory import MemoryEstimate
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.planning.production import estimate_perambulator_memory
from lqcdweave.production.perambulator import PerambulatorGenerator, VectorBasis
from lqcdweave.production.perambulator.planning import plan_indexed_perambulators
from fixtures.perambulator_script import produce_source_rows
from test_perambulator_dataset import (
    _CONFIGURATION,
    _DIRAC,
    _PerambulatorDatasetFixture,
)


class PerambulatorBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _PerambulatorDatasetFixture("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)


    def test_hand_calculated_lifecycle_uses_single_shared_basis_snapshot(self):
        fixture = self.fixture
        two_mode_id = fixture._publish_basis(
            "budget-shared-basis", [8, 3], fixture.source_basis_values[:, :2])
        with fixture.index.open(fixture.gauge_id) as gauge, \
                fixture.index.open(two_mode_id) as basis:
            input_specs = {fixture.gauge_id: gauge.spec, two_mode_id: basis.spec}

        host, device = estimate_perambulator_memory(
            input_specs, _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
            basis_src_id=two_mode_id, basis_snk_id=two_mode_id,
            mode_src_count=2, mode_snk_count=2, same_eigenvector=True,
            dtype="complex128",
            execution_options=dict(MRHS=False,
                rhs_batch_size=None, sink_ne_block_size=32,
                stream_eigenvectors=False, sink_time_block_size=1,
                sink_cache_limit_bytes=0),
            target="cpu")

        # Parent mappings: gauge 3456B + two-mode basis 576B. The identical
        # source/sink basis object and time map need one staged 576B copy.
        # During projection, RHS/solution use 2304B, the four-spin assembly
        # uses 4608B, and the previous/current sink tile peak is 1216B.
        # Output is 3072B; the row-write phase is smaller than projection.
        vector_snapshot = 3 * 2 * 2 * 3 * 16
        rhs_solution = 2 * (1 * 3 * 2 * 4 * 3 * 16)
        four_spin_assembly = 4 * 3 * 2 * 4 * 3 * 16
        sink_tile = 4 * 4 * 2 * 16
        sink_mode_block = 2 * 2 * 3 * 16
        projection = max(2 * sink_tile + sink_mode_block,
                         sink_tile + 2 * sink_mode_block)
        intermediate = rhs_solution + four_spin_assembly + projection
        output = 3 * 4 * 4 * 2 * 2 * 16
        expected = 4032 + vector_snapshot + intermediate + output
        self.assertEqual(host.logical_peak_bytes, expected)
        self.assertEqual(host.input_bytes, 4032)
        self.assertEqual(host.snapshot_bytes, vector_snapshot)
        self.assertEqual(host.intermediate_bytes, intermediate)
        self.assertEqual(host.output_bytes, output)
        self.assertEqual(host.assembly_extra_bytes, 0)
        self.assertEqual(device, MemoryEstimate())

        gpu_host, gpu_device = estimate_perambulator_memory(
            input_specs, _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
            basis_src_id=two_mode_id, basis_snk_id=two_mode_id,
            mode_src_count=2, mode_snk_count=2, same_eigenvector=True,
            dtype="complex128",
            execution_options=dict(MRHS=False,
                rhs_batch_size=None, sink_ne_block_size=32,
                stream_eigenvectors=False, sink_time_block_size=1,
                sink_cache_limit_bytes=0),
            target="gpu", workspace_reserve_bytes=7,
            host_staging_bytes=11)
        self.assertEqual(gpu_host.workspace_bytes, 18)
        self.assertEqual(gpu_device.workspace_bytes, 7)
        gauge_bytes = (np.prod(input_specs[fixture.gauge_id].shape) *
                       np.dtype(input_specs[fixture.gauge_id].document["dtype"]).itemsize)
        self.assertEqual(gpu_device.snapshot_bytes, vector_snapshot + gauge_bytes)
        self.assertEqual(gpu_device.intermediate_bytes, intermediate)
        self.assertEqual(gpu_device.output_bytes, output)
        self.assertGreater(gpu_host.logical_peak_bytes, 0)
        self.assertGreater(gpu_device.logical_peak_bytes, 0)

    def test_known_c64_mrhs_and_streaming_temporaries_are_counted(self):
        fixture = self.fixture
        with (fixture.index.open(fixture.gauge_id) as gauge,
              fixture.index.open(fixture.source_basis_id) as source_basis,
              fixture.index.open(fixture.sink_basis_id) as sink_basis,
              fixture.index.open(fixture.sink_points_id) as sink_points,
              fixture.index.open(fixture.source_points_id) as source_points):
            gauge_spec = gauge.spec
            source_basis_spec = source_basis.spec
            sink_basis_spec = sink_basis.spec
            sink_points_spec = sink_points.spec
            source_points_spec = source_points.spec
        gauge_bytes = (np.prod(gauge_spec.shape) *
                       np.dtype(gauge_spec.document["dtype"]).itemsize)

        common_options = dict(MRHS=True,
            rhs_batch_size=2, sink_ne_block_size=32,
            stream_eigenvectors=False, sink_time_block_size=1,
            sink_cache_limit_bytes=0)
        vector_specs = {fixture.gauge_id: gauge_spec,
                        fixture.source_basis_id: source_basis_spec,
                        fixture.sink_basis_id: sink_basis_spec}
        vsv_host, _ = estimate_perambulator_memory(
            vector_specs, _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
            basis_src_id=fixture.source_basis_id,
            basis_snk_id=fixture.sink_basis_id,
            mode_src_count=1, mode_snk_count=2, dtype="complex64",
            execution_options=common_options, target="cpu")
        # Two RHS/solution batches use 4608B; a source-vector call with a
        # batch below four also assembles four spin fields (4608B). The
        # non-streamed sink projection peak is 1120B for its mode block and
        # adjacent tile, while the c64 output row is 768B.
        rhs_solution = 2 * (2 * 3 * 2 * 4 * 3 * 16)
        four_spin_assembly = 4 * 3 * 2 * 4 * 3 * 16
        sink_tile = 4 * 4 * 2 * 16
        sink_mode_block = 2 * 2 * 3 * 8
        projection = max(2 * sink_tile + sink_mode_block,
                         sink_tile + 2 * sink_mode_block)
        self.assertEqual(vsv_host.output_bytes, 3 * 4 * 4 * 2 * 1 * 8)
        self.assertEqual(vsv_host.snapshot_bytes, 144 + 288)
        self.assertEqual(vsv_host.intermediate_bytes,
                         rhs_solution + four_spin_assembly + projection)
        self.assertEqual(vsv_host.assembly_extra_bytes, 0)

        streamed = dict(common_options, stream_eigenvectors=True,
                        sink_ne_block_size=1, sink_time_block_size=1)
        _, streamed_device = estimate_perambulator_memory(
            vector_specs, _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
            basis_src_id=fixture.source_basis_id,
            basis_snk_id=fixture.sink_basis_id,
            mode_src_count=1, mode_snk_count=2, dtype="complex64",
            execution_options=streamed, target="gpu")
        # Streaming owns one source and one sink mode scratch (96B). Its sink
        # projection peak is 512B; the basis has no eager snapshot, while the
        # GPU estimate still counts the 3456B declared gauge payload.
        self.assertEqual(streamed_device.snapshot_bytes, gauge_bytes)
        streamed_rhs_solution = rhs_solution
        streamed_four_spin_assembly = four_spin_assembly
        streamed_sink_tile = 4 * 4 * 1 * 16
        streamed_sink_mode_block = 1 * 2 * 3 * 8
        streamed_projection = max(2 * streamed_sink_tile,
                                  streamed_sink_tile + streamed_sink_mode_block)
        self.assertEqual(streamed_device.intermediate_bytes,
                         streamed_rhs_solution + streamed_four_spin_assembly
                         + 96 + streamed_projection)

        psv_specs = {fixture.gauge_id: gauge_spec,
                     fixture.source_basis_id: source_basis_spec,
                     fixture.sink_points_id: sink_points_spec}
        psv_host, _ = estimate_perambulator_memory(
            psv_specs, _CONFIGURATION, "PSV", gauge_id=fixture.gauge_id,
            basis_src_id=fixture.source_basis_id, points_snk_id=fixture.sink_points_id,
            mode_src_count=1, point_snk_count=2, dtype="complex64",
            execution_options=common_options, target="cpu")
        # PSV projects points at one selected physical time. The known peak is
        # the two RHS/solution batches, four-spin assembly, and two live point
        # gather/projection tiles; no selected-time full-field gather exists.
        point_projection = 2 * 4 * 2 * 4 * 3 * 16
        self.assertEqual(psv_host.intermediate_bytes,
                         rhs_solution + four_spin_assembly + point_projection)

        psp_specs = {fixture.gauge_id: gauge_spec,
                     fixture.source_points_id: source_points_spec,
                     fixture.sink_points_id: sink_points_spec}
        psp_options = dict(common_options, rhs_batch_size=12)
        psp_host, _ = estimate_perambulator_memory(
            psp_specs, _CONFIGURATION, "PSP", gauge_id=fixture.gauge_id,
            points_src_id=fixture.source_points_id,
            points_snk_id=fixture.sink_points_id,
            point_src_count=1, point_snk_count=2, dtype="complex64",
            execution_options=psp_options, target="cpu")
        # Point sources use a batch of twelve: RHS/solution are 27648B and the
        # selected-time point projection peak is 9216B.
        point_rhs_solution = 2 * (12 * 3 * 2 * 4 * 3 * 16)
        point_projection = 2 * 12 * 2 * 4 * 3 * 16
        self.assertEqual(psp_host.intermediate_bytes,
                         point_rhs_solution + point_projection)






    def test_row_copy_is_released_before_next_host_row_allocation(self):
        fixture = self.fixture
        preview = plan_indexed_perambulators(
            fixture.index, fixture.gauge_id, products=("VSV",),
            dirac_definition=_DIRAC, solver_tolerance=1e-11,
            temporal_boundary="antiperiodic", basis_id=fixture.source_basis_id,
            Ne=1)
        spec = preview["stages"]["vsv"]["spec"]

        class Writer:
            def __init__(self):
                self.spec = spec
                self.missing = (0, 1, 2)
                self.rows = []

            def write(self, position, value):
                self.rows.append((position, np.array(value, copy=True)))

        class Generator:
            class Result:
                def __init__(self, product, values):
                    self.product = product
                    self.values = values

            def __init__(self):
                self.previous = None
                self.calls = 0

            def load(self, key):
                self.key = key

            def calc(self, source_time, *, src, sink):
                if self.previous is not None:
                    result_ref, value_ref = self.previous
                    self.assertIsNone(result_ref(),
                                      "previous owned result survived into next calc")
                    self.assertIsNone(value_ref(),
                                      "previous owned result values survived into next calc")
                values = np.full(spec.shape[1:], source_time + 1,
                                 dtype=np.dtype(spec.document["dtype"]))
                result = self.Result("VSV", values)
                self.previous = weakref.ref(result), weakref.ref(values)
                self.calls += 1
                return [result]

            @staticmethod
            def assertIsNone(value, message):
                if value is not None:
                    raise AssertionError(message)

        writer = Writer()
        references = []
        original_host_row = perambulator_workflow._host_row

        def observe_host_row(*args, **kwargs):
            if references:
                self.assertIsNone(references[-1](),
                                  "previous owned host row survived into next allocation")
            row = original_host_row(*args, **kwargs)
            self.assertEqual(row.shape, spec.shape[1:])
            references.append(weakref.ref(row))
            return row

        generator = Generator()
        with patch.object(perambulator_workflow, "_host_row", new=observe_host_row):
            produce_source_rows(writer, generator, [0, 1, 2],
                                    _CONFIGURATION, "VSV", src=object(), sink=object(),
                                    to_host=lambda value: np.array(value, copy=True))
        self.assertIsNone(references[-1]())
        self.assertIsNone(generator.previous[0]())
        self.assertIsNone(generator.previous[1]())
        self.assertEqual(generator.calls, 3)
        self.assertEqual([position for position, _ in writer.rows], [0, 1, 2])
        self.assertTrue(all(value.shape == spec.shape[1:] for _, value in writer.rows))


    def test_noncontiguous_sink_projects_by_physical_time_without_lt_gather(self):
        from lqcdweave.adapters.production_dataset import DatasetFieldHandle

        fixture = self.fixture
        source = TimeSelectionRecord(3, (1, 0))
        sink = TimeSelectionRecord(3, (2, 0))
        with fixture.index.open(fixture.gauge_id) as gauge, \
                fixture.index.open(fixture.source_basis_id) as basis:
            handle = DatasetFieldHandle(basis)
            vector = VectorBasis(handle, 1)
            staged_refs = []
            staged_rows = []
            testcase = self

            class Backend:
                array_module = np

                def __init__(self):
                    self.allocations = []
                    self.stage_ids_during_solve = []

                def empty(self, shape, dtype):
                    value = np.empty(shape, dtype=dtype)
                    self.allocations.append((tuple(shape), weakref.ref(value)))
                    return value

                def zeros(self, shape, dtype):
                    value = np.zeros(shape, dtype=dtype)
                    self.allocations.append((tuple(shape), weakref.ref(value)))
                    return value

                def load(self, gauge_field, parameters):
                    pass

                def solve_into(self, rhs, solution, *, tol, maxiter):
                    live = [reference() for reference in staged_refs]
                    testcase.assertEqual(len(live), 2)
                    testcase.assertIsNotNone(live[0])
                    testcase.assertIsNotNone(live[1])
                    testcase.assertIsNot(live[0], live[1])
                    self.stage_ids_during_solve.append(tuple(map(id, live)))
                    solution[...] = rhs

            backend = Backend()
            generator = PerambulatorGenerator(
                _CONFIGURATION["lattice"], DatasetFieldHandle(gauge),
                .4, 1e-11, 100, source_selection=source,
                sink_selection=sink, backend=backend)
            generator.load("cfg-7")
            read_vectors = generator._read_vectors

            def record_stage(handle, count, key, selection):
                value = read_vectors(handle, count, key, selection)
                staged_refs.append(weakref.ref(value))
                staged_rows.append(value.copy())
                return value

            generator._read_vectors = record_stage
            result = generator.calc(0, src=vector, sink=[vector])[0]
            self.assertEqual(result.sink_times, sink.times)
            self.assertEqual(result.values.shape, (sink.Nt, 4, 4, 1, 1))
            expected = np.zeros(result.values.shape, dtype=np.complex128)
            expected[1, np.arange(4), np.arange(4), 0, 0] = 1
            np.testing.assert_allclose(result.values, expected, atol=1e-14, rtol=0)
            np.testing.assert_array_equal(
                staged_rows[0], fixture.source_basis_values[[1, 0], :1])
            np.testing.assert_array_equal(
                staged_rows[1], fixture.source_basis_values[[2, 0], :1])
            self.assertEqual(len(backend.stage_ids_during_solve), 4)
            self.assertTrue(all(source_id != sink_id
                                for source_id, sink_id in backend.stage_ids_during_solve))

            allocation_shapes = [shape for shape, _ in backend.allocations]
            # The solve retains a physical-Lt four-spin assembly. There is no
            # separate gather over the two selected sink times.
            self.assertIn((4, 3, 1, 1, 2, 4, 3), allocation_shapes)
            self.assertNotIn((4, sink.Nt, 1, 1, 2, 4, 3), allocation_shapes)
            staged_shape = (source.Nt, 1, 1, 1, 2, 3)
            self.assertEqual(allocation_shapes.count(staged_shape), 2)
            self.assertTrue(all(reference() is None for reference in staged_refs),
                            "source/sink staged arrays survived calc return")

            estimate, _ = estimate_perambulator_memory(
                {fixture.gauge_id: gauge.spec,
                 fixture.source_basis_id: basis.spec},
                _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
                basis_src_id=fixture.source_basis_id,
                basis_snk_id=fixture.source_basis_id,
                mode_src_count=1, mode_snk_count=1, same_eigenvector=True,
                source_selection=source, sink_selection=sink, target="cpu")
            # The same basis ID is not enough to share its staged copy when
            # source and sink time maps differ.
            self.assertEqual(estimate.snapshot_bytes,
                             2 * (2 * 1 * 2 * 3 * 16))

            unequal_counts, _ = estimate_perambulator_memory(
                {fixture.gauge_id: gauge.spec,
                 fixture.source_basis_id: basis.spec},
                _CONFIGURATION, "VSV", gauge_id=fixture.gauge_id,
                basis_src_id=fixture.source_basis_id,
                basis_snk_id=fixture.source_basis_id,
                mode_src_count=2, mode_snk_count=1, same_eigenvector=True,
                source_selection=source, sink_selection=source, target="cpu")
            # An explicitly shared object with unequal selected counts also
            # needs independent source and sink staged arrays.
            self.assertEqual(unequal_counts.snapshot_bytes,
                             (2 * 2 * 2 * 3 * 16) + (2 * 1 * 2 * 3 * 16))


_CONFIGURATION = dict(ensemble="perambulator-test", configuration="cfg-7",
                      lattice=[2, 1, 1, 3])


if __name__ == "__main__":
    unittest.main()
