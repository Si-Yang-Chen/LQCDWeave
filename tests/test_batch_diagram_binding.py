"""Numerical checks for the native dense coordinate-batch lowering."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.expansion.batching import (
    _BatchTaskLoader, _build_lane_contract, _collect_slab_catalog,
)
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.correlator.api import (
    BatchSpec, BatchTensorBinding, CoordinatePoint, CoordinateSpec,
    ContractionProblem, ContractionTerm, DiagramInputs, PhysicalSlabBinding,
    SourcePreparation, SourceSpec, StorageBlock, TensorAxisMetadata, TensorSpec,
    describe_coordinates, group_coordinate_batches,
)
from lqcdweave.contraction.execution import execute_contraction_path as execute_domain
from lqcdweave.contraction.planning.path_planning import _best_predicted_batch_width
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Propagator
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.correlator.api import Tag
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calc_diagram_eval


class BatchDiagramBindingTests(unittest.TestCase):
    def test_slab_reads_keep_physical_identity_when_allocation_is_retargeted(self):
        raw = np.arange(2, dtype=np.complex128).reshape(2, 1)
        reads = []
        def read(times):
            reads.append(times)
            return raw
        metadata = TensorAxisMetadata(('coordinate_batch', 'mode'), raw.shape,
                                      raw.dtype, 'dataset', 'selection')
        slab = PhysicalSlabBinding(('reader', 'selection'), lambda times: metadata,
                                   read, read_slab_in_background=read)
        block = StorageBlock('dataset', 'selection', raw.nbytes, host_bytes=raw.nbytes)
        loader = _BatchTaskLoader({
            'blocks': {'slab:0': block}, 'block_by_identity': {slab.identity: 'slab:0'},
            'metadata_by_block': {'slab:0': metadata},
            'templates': {slab.identity: slab}, 'times': {slab.identity: (0, 1)},
        })
        placed = replace(block, device_bytes=raw.nbytes, transfer_bytes=raw.nbytes,
                         staging_host_bytes=raw.nbytes)
        np.testing.assert_array_equal(loader.read_block('slab:0', placed), raw)
        np.testing.assert_array_equal(loader.prefetch_block('slab:0', placed), raw)
        self.assertEqual(reads, [(0, 1), (0, 1)])
        for method in (loader.read_block, loader.prefetch_block):
            with self.assertRaisesRegex(ValueError, 'differs from the full-task catalog'):
                method('slab:0', replace(placed, selection='another-selection'))
        self.assertEqual(len(reads), 2)

    def test_whole_vertex_multiple_slabs_and_shared_leaf_keep_coordinate_lanes(self):
        rng = np.random.default_rng(261022)
        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)
        components = tuple(ElementalComponent(i) for i in (0, 1))
        elemental = ArrayElemental({'v2v': random((2, 1, 3, 2, 2))}, components,
            phase_convention='anchor', data_identity='whole-batch/E')
        propagator = ArrayPropagator({'vsv': random((3, 3, 4, 4, 2, 2))},
            time_convention='absolute', data_identity='whole-batch/P')
        operator = Operator.from_terms('whole-batch', [
            OperatorTerm(0, components[0], 0, 1 + .2j),
            OperatorTerm(1, components[1], 0, -.3 + .7j),
            OperatorTerm(2, components[0], 0, 1e-16j),
        ])
        q = Propagator('u', Tag(0, 0), Tag(1, 1))
        graph = Diagram(QuarkDiagram([[0, 1], [1, 0]]), ['sink', 'source'],
                        ['a', 'b'], [None, q])
        arguments = dict(propagator_map={q: propagator}, vertex_map={
            'a': Meson(elemental, operator, phase_convention='anchor'),
            'b': Meson(elemental, operator, True, phase_convention='anchor')},
            usedNe=2, optimize='none')
        expected = np.asarray([calc_diagram_eval(graph, **arguments,
            time_axes={'sink': t, 'source': 1}).value for t in range(3)])
        reads = []
        read_slab = elemental.read_slab
        def record(*args, **kwargs):
            reads.append(args)
            return read_slab(*args, **kwargs)
        elemental.read_slab = record
        elemental.block = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('batch preparation must use shared slabs'))
        for width in (1, 2, 'full', 'auto'):
            with self.subTest(width=width):
                reads.clear()
                calculation = calc_diagram_eval(graph, **arguments,
                    time_axes={'sink': range(3), 'source': 1}, batch=width, execute=False)
                self.assertEqual(reads, [])
                for item in calculation.inputs.task_batches:
                    self.assertEqual(len(item.problem.terms), 1)
                    self.assertEqual(len(item.problem.terms[0].inputs), 4)
                np.testing.assert_allclose(calculation.run().value, expected,
                                           rtol=1e-12, atol=1e-12)
                self.assertEqual(len(reads), 2)

    def test_candidate_comparison_accepts_small_or_large_profile_winner(self):
        self.assertEqual(_best_predicted_batch_width((
            {"batch_width": 1, "predicted_seconds": 1.0, "feasible": True},
            {"batch_width": 2, "predicted_seconds": 2.0, "feasible": True},
        )), 1)
        self.assertEqual(_best_predicted_batch_width((
            {"batch_width": 1, "predicted_seconds": 3.0, "feasible": True},
            {"batch_width": 2, "predicted_seconds": 1.0, "feasible": True},
        )), 2)

    def test_conflicting_metadata_for_shared_slab_identity_fails_before_reads(self):
        coordinates, batch_spec = describe_coordinates(
            time_axes={"t": range(2)}, batch={"axis": "t", "size": 2},
        )
        coordinate_batch, = group_coordinate_batches(coordinates, batch_spec)
        reads = []
        lanes = []

        for index, point in enumerate(coordinate_batch.points):
            lane_coordinates = CoordinateSpec(
                "bound", (), (), (), (CoordinatePoint(point.key, point.values),), (),
            )
            problem = ContractionProblem(
                {"tensor:0": TensorSpec((1,), np.complex128)},
                (ContractionTerm(("tensor:0",), "a->"),),
                {"tensor:0": SourceSpec("logical", 16, host_region_bytes=16,
                                        snapshot_bytes=16)},
            )

            def describe_slab(times, tail=1 if index == 0 else 2):
                return TensorAxisMetadata(
                    ("coordinate_batch", "mode"), (len(times), tail),
                    np.complex128, "same-data", "same-selection",
                )

            slab = PhysicalSlabBinding(
                ("shared-reader", "fixed-selection"), describe_slab,
                lambda times: reads.append(tuple(times)),
            )
            lanes.append(DiagramInputs(
                problem,
                {"block": StorageBlock("dataset", "selection", 16, host_bytes=16)},
                {"tensor:0": SourcePreparation(("block",))}, object(),
                lane_coordinates, BatchSpec(), ("T",),
                batch_bindings={"tensor:0": BatchTensorBinding(
                    "tensor:0", (slab,), (index,), (lambda raw: raw,),
                )},
                batch_topology=("same-topology",),
            ))

        contract = _build_lane_contract(lanes)
        with self.assertRaisesRegex(ValueError, "shared physical slab identity"):
            _collect_slab_catalog(lanes, contract)
        self.assertEqual(reads, [])

    def test_shared_slab_read_workspace_uses_maximum_in_read_peak(self):
        coordinates, batch_spec = describe_coordinates(
            time_axes={"t": range(2)}, batch={"axis": "t", "size": 2},
        )
        reads = []
        lanes = []
        for index, point in enumerate(coordinates.points):
            lane_coordinates = CoordinateSpec(
                "bound", (), (), (), (CoordinatePoint(point.key, point.values),), (),
            )
            problem = ContractionProblem(
                {"tensor:0": TensorSpec((1,), np.complex128)},
                (ContractionTerm(("tensor:0",), "a->"),),
                {"tensor:0": SourceSpec("logical", 16, host_region_bytes=16,
                                        snapshot_bytes=16)},
            )
            slab = PhysicalSlabBinding(
                ("shared-reader", "fixed-selection"),
                lambda times: TensorAxisMetadata(
                    ("coordinate_batch", "mode"), (len(times), 1),
                    np.complex128, "same-data", "same-selection",
                ),
                lambda times: reads.append(tuple(times)),
                lambda metadata, workspace=64 * (index + 1): workspace,
            )
            lanes.append(DiagramInputs(
                problem,
                {"block": StorageBlock("dataset", "selection", 16, host_bytes=16)},
                {"tensor:0": SourcePreparation(("block",))}, object(),
                lane_coordinates, BatchSpec(), ("T",),
                batch_bindings={"tensor:0": BatchTensorBinding(
                    "tensor:0", (slab,), (index,), (lambda raw: raw,),
                )},
                batch_topology=("same-topology",),
            ))
        catalog = _collect_slab_catalog(lanes, _build_lane_contract(lanes))
        (block,) = catalog["blocks"].values()
        self.assertEqual(block.staging_host_bytes, 128)
        self.assertEqual(reads, [])

    def test_two_coordinate_dense_batch_matches_independent_scalar_runs(self):
        rng = np.random.default_rng(260928)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        component = ElementalComponent(0)
        elemental = ArrayElemental(
            {"v2v": random((1, 1, 2, 1, 1))},
            (component,), phase_convention="anchor", data_identity="batch-test/el",
        )
        propagator = ArrayPropagator(
            {"vsv": random((2, 2, 4, 4, 1, 1))},
            time_convention="absolute", data_identity="batch-test/peram",
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, q],
        )
        operator = Operator.from_terms("weighted-local", [
            OperatorTerm(0, component, 0, 1.5 + 0.25j),
        ])
        vertices = {
            name: Meson(elemental, operator, phase_convention="anchor")
            for name in ("a", "b")
        }

        expected = np.asarray([
            calc_diagram_eval(
                graph, propagator_map={q: propagator}, vertex_map=vertices,
                usedNe=1, time_axes={"sink": sink, "source": 1},
                optimize="none",
            ).value
            for sink in (0, 1)
        ])

        slab_reads = []
        original_elemental_read_slab = elemental.read_slab
        original_propagator_read_slab = propagator.read_slab

        def read_elemental(*args, **kwargs):
            slab_reads.append(("elemental", args[2]))
            return original_elemental_read_slab(*args, **kwargs)

        def read_propagator(*args, **kwargs):
            slab_reads.append(("propagator", args[0], args[1]))
            return original_propagator_read_slab(*args, **kwargs)

        elemental.read_slab = read_elemental
        propagator.read_slab = read_propagator
        elemental.block = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("batched execution must read one time slab, not scalar blocks"))
        propagator.block = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("batched execution must read one source slab, not scalar blocks"))

        calculation = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes={"sink": range(2), "source": 1},
            batch={"axis": "sink", "size": 2}, optimize="none", execute=False,
        )
        self.assertEqual(slab_reads, [])
        self.assertEqual(calculation.inputs.task_batches[0].coordinate_batch.actual_size, 2)
        self.assertTrue(any(
            shape.shape[0] == 2 for shape in calculation.expanded.tensor_specs.values()
        ))
        batched_specs = [spec for spec in calculation.expanded.tensor_specs.values()
                         if spec.shape[0] == 2 and spec.description is not None]
        self.assertTrue(batched_specs)
        for spec in batched_specs:
            self.assertEqual(spec.description.axes[0], "coordinate_batch")
            self.assertTrue(spec.description.source_bindings)
            self.assertEqual(
                tuple(dict(values)["sink"] for values in
                      spec.description.bindings["coordinate_values"]),
                (0, 1),
            )
        self.assertTrue(all(
            term.subscripts.output
            for term in calculation.expanded.terms
            if any(calculation.expanded.tensor_specs[key].shape[0] == 2
                   for key in term.inputs)
        ))

        automatic = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes={"sink": range(2), "source": 1},
            batch="auto", optimize="none", execute=False,
        )
        self.assertEqual(len(automatic.inputs.task_batches), 1)
        self.assertEqual([item.coordinate_batch.actual_size
                          for item in automatic.inputs.task_batches], [2])
        self.assertEqual(slab_reads, [])

        domain_result = execute_domain(calculation.expanded, calculation.path)
        domain_value = assemble_correlator(calculation.expanded, domain_result.term_values)
        self.assertEqual(domain_value.shape, (2,))
        np.testing.assert_allclose(domain_value, expected, rtol=1e-12, atol=1e-12)
        # Two unique propagator source slabs and one shared elemental slab.
        self.assertEqual(len(slab_reads), 3)
        self.assertEqual(sum(item[0] == "elemental" for item in slab_reads), 1)
        self.assertEqual(sum(item[0] == "propagator" for item in slab_reads), 2)

        slab_reads.clear()
        actual = calculation.run().value
        self.assertEqual(actual.shape, (2,))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(slab_reads), 3)

        slab_reads.clear()
        full = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes={"sink": range(2), "source": 1},
            batch="full", optimize="none",
        ).value
        np.testing.assert_allclose(full, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(slab_reads), 3)

        slab_reads.clear()
        singleton = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes={"sink": (0,), "source": 1},
            batch=1, optimize="none",
        ).value
        self.assertEqual(singleton.shape, (1,))
        np.testing.assert_allclose(singleton, expected[:1], rtol=1e-12, atol=1e-12)

        slab_reads.clear()
        pairs = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1,
            time_pairs=({"sink": 0, "source": 1}, {"sink": 1, "source": 1}),
            batch={"axis": "pair", "size": 2}, optimize="none",
        ).value
        np.testing.assert_allclose(pairs, expected, rtol=1e-12, atol=1e-12)

    def test_one_dense_current_batch_matches_scalar_recipe_and_all_sectors(self):
        rng = np.random.default_rng(260929)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        ne, npoint, lt = 1, 2, 2
        current = LocalVectorCurrent().operator(3)
        component = current.terms[0].component
        normal = Operator.from_terms("normal", [OperatorTerm(0, component, 0, 1)])
        tails = {
            "v2v": (ne, ne), "v2p": (ne, npoint, 3),
            "p2v": (npoint, 3, ne), "p2p": (npoint, 3, npoint, 3),
        }
        elemental = ArrayElemental(
            {sector: random((1, 1, lt, *shape)) for sector, shape in tails.items()},
            (component,), phase_convention="anchor",
            sampling_pool_ids={
                (component, 0, time, side): "whole-space"
                for time in range(lt) for side in ("bar", "field")
            },
            sampling_groups={"whole-space": SamplingPoolSpec("whole-space", npoint, npoint)},
        )
        prop_tails = {
            "vsv": (ne, ne), "vsp": (ne, npoint, 3),
            "psv": (npoint, 3, ne), "psp": (npoint, 3, npoint, 3),
        }
        propagator = ArrayPropagator(
            {sector: random((lt, lt, 4, 4, *shape))
             for sector, shape in prop_tails.items()},
            time_convention="absolute",
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["current", "normal"], [None, q],
        )
        vertices = {
            "current": Meson(elemental, current, phase_convention="anchor"),
            "normal": Meson(elemental, normal, phase_convention="anchor"),
        }
        expected = np.asarray([
            calc_diagram_eval(
                graph, propagator_map={q: propagator}, vertex_map=vertices,
                usedNe=ne, usedNp=npoint,
                time_axes={"sink": sink, "source": 1}, optimize="none",
            ).value
            for sink in range(lt)
        ])
        actual = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=ne, usedNp=npoint,
            time_axes={"sink": range(lt), "source": 1},
            batch={"axis": "sink", "size": 2}, optimize="none",
        ).value
        self.assertEqual(actual.shape, (lt,))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)

    def test_multibatch_two_sources_union_reads_once_and_planned_paths_reuse_by_shape(self):
        rng = np.random.default_rng(260930)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        lt = 5
        component = ElementalComponent(0)
        elemental = ArrayElemental(
            {"v2v": random((1, 1, lt, 1, 1))},
            (component,), phase_convention="anchor", data_identity="multi/el",
        )
        propagator = ArrayPropagator(
            {"vsv": random((lt, lt, 4, 4, 1, 1))},
            time_convention="absolute", data_identity="multi/peram",
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, q],
        )
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0, 1)])
        vertices = {name: Meson(elemental, operator, phase_convention="anchor")
                    for name in ("a", "b")}
        expected = np.empty((lt, 2), dtype=np.complex128)
        for sink in range(lt):
            for source in range(2):
                expected[sink, source] = calc_diagram_eval(
                    graph, propagator_map={q: propagator}, vertex_map=vertices,
                    usedNe=1, time_axes={"sink": sink, "source": source},
                    optimize="none",
                ).value

        reads = []
        elemental_read = elemental.read_slab
        propagator_read = propagator.read_slab

        def record_elemental(*args, **kwargs):
            reads.append(("elemental", tuple(args[2]), args[3]))
            return elemental_read(*args, **kwargs)

        def record_propagator(*args, **kwargs):
            reads.append(("propagator", args[1], tuple(args[0]), args[2]))
            return propagator_read(*args, **kwargs)

        elemental.read_slab = record_elemental
        propagator.read_slab = record_propagator
        elemental.block = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("multi-batch execution must not use scalar elemental reads"))
        propagator.block = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("multi-batch execution must not use scalar propagator reads"))
        coordinates = {"sink": range(lt), "source": range(2)}
        batch = {"axis": "sink", "size": 2}

        automatic_narrow = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes=coordinates,
            batch={"axis": "sink", "size": "auto"}, optimize="none",
            memory_limit=7_500, host_memory_limit=7_500, execute=False,
        )
        self.assertEqual(
            [item.coordinate_batch.actual_size
             for item in automatic_narrow.inputs.task_batches],
            [1] * 10,
        )
        narrow_global_peak = max(
            item.plan.peak_host_bytes for item in automatic_narrow.path.batch_paths
        )
        self.assertEqual(narrow_global_peak, 6_800)
        self.assertEqual(automatic_narrow.path.prediction["batch_selection"]["selected_width"], 1)
        self.assertEqual(reads, [])

        automatic = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes=coordinates,
            batch={"axis": "sink", "size": "auto"}, optimize="none",
            memory_limit=40_000, host_memory_limit=40_000, execute=False,
        )
        self.assertEqual([item.coordinate_batch.actual_size
                          for item in automatic.inputs.task_batches], [5, 5])
        wide_global_peak = max(
            item.plan.peak_host_bytes for item in automatic.path.batch_paths
        )
        self.assertEqual(wide_global_peak, 16_016)
        self.assertEqual(automatic.path.prediction["batch_selection"]["selected_width"], 5)
        self.assertEqual(reads, [])

        with self.assertRaises(MemoryError):
            calc_diagram_eval(
                graph, propagator_map={q: propagator}, vertex_map=vertices,
                usedNe=1, time_axes=coordinates,
                batch={"axis": "sink", "size": "auto"}, optimize="none",
                memory_limit=6_000, host_memory_limit=6_000, execute=False,
            )
        self.assertEqual(reads, [])

        direct = calc_diagram_eval(
            graph, propagator_map={q: propagator}, vertex_map=vertices,
            usedNe=1, time_axes=coordinates, batch=batch, optimize="none",
        )
        self.assertEqual(len(direct.inputs.task_batches), 6)
        np.testing.assert_allclose(direct.value, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(reads), len(direct.inputs.blocks))
        self.assertEqual(len(set(reads)), len(reads))

        reads.clear()
        from lqcdweave.contraction.performance.api import CostProfile
        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   wraps=__import__(
                       "lqcdweave.contraction.planning.path_planning",
                       fromlist=["solve_storage"],
                   ).solve_storage) as matrix_plan:
            planned = calc_diagram_eval(
                graph, propagator_map={q: propagator}, vertex_map=vertices,
                usedNe=1, time_axes=coordinates, batch=batch,
                optimize="anytime", memory_limit=1 << 28,
                host_memory_limit=1 << 28, execute=False, profile=CostProfile(),
            )
        self.assertEqual(matrix_plan.call_count, 2)
        self.assertEqual(len(planned.path.batch_paths), 6)
        self.assertEqual(len(set(planned.path.batch_shape_keys)), 2)
        self.assertEqual(reads, [])

        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   wraps=__import__(
                       "lqcdweave.contraction.planning.path_planning",
                       fromlist=["solve_storage"],
                   ).solve_storage) as auto_matrix_plan:
            automatic_planned = calc_diagram_eval(
                graph, propagator_map={q: propagator}, vertex_map=vertices,
                usedNe=1, time_axes=coordinates,
                batch={"axis": "sink", "size": "auto"},
                optimize="anytime", memory_limit=1 << 26,
                host_memory_limit=1 << 26, execute=False, profile=CostProfile(),
            )
        selection = automatic_planned.path.prediction["batch_selection"]
        candidate_predictions = selection["candidate_predictions"]
        distinct_sizes = {
            size for candidate in candidate_predictions
            for size in candidate["batch_sizes"]
        }
        self.assertEqual(auto_matrix_plan.call_count, len(distinct_sizes))
        self.assertEqual(reads, [])
        feasible_predictions = [item for item in candidate_predictions if item["feasible"]]
        fastest = min(feasible_predictions,
                      key=lambda item: (item["predicted_seconds"], item["batch_width"]))
        self.assertEqual(selection["selected_width"], fastest["batch_width"])
        self.assertEqual(selection["calibration_status"], "unmeasured")
        domain_result = execute_domain(planned.expanded, planned.path)
        domain_value = assemble_correlator(planned.expanded, domain_result.term_values)
        np.testing.assert_allclose(domain_value, expected, rtol=1e-12, atol=1e-11)
        self.assertEqual(len(reads), len(planned.inputs.blocks))
        self.assertEqual(len(set(reads)), len(reads))

        reads.clear()
        actual = planned.run().value
        np.testing.assert_allclose(actual, domain_value, rtol=1e-12, atol=1e-11)
        self.assertEqual(len(reads), len(planned.inputs.blocks))

    def test_nested_repeated_diagrams_reserve_task_terms_and_assembly_output(self):
        rng = np.random.default_rng(260931)
        lt = 3
        component = ElementalComponent(0)
        elemental = ArrayElemental(
            {"v2v": rng.normal(size=(1, 1, lt, 1, 1)).astype(np.complex128)},
            (component,), phase_convention="anchor",
        )
        propagator = ArrayPropagator(
            {"vsv": rng.normal(size=(lt, lt, 4, 4, 1, 1)).astype(np.complex128)},
            time_convention="absolute",
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, q],
        )
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0, 1)])
        vertices = {name: Meson(elemental, operator, phase_convention="anchor")
                    for name in ("a", "b")}
        options = dict(
            propagator_map={q: propagator}, vertex_map=vertices, usedNe=1,
            time_axes={"sink": range(lt), "source": 1},
            batch={"axis": "sink", "size": 2},
            memory_limit=1 << 26, host_memory_limit=1 << 26,
            execute=False,
        )
        single = calc_diagram_eval(graph, optimize="none", **options)
        nested_expression = {
            "first": [graph, graph],
            "nested": {"again": (graph, graph, graph)},
        }
        nested = calc_diagram_eval(nested_expression, optimize="none", **options)
        self.assertGreater(nested.inputs.assembly_host_bytes,
                           single.inputs.assembly_host_bytes)
        self.assertTrue(all(
            batch_inputs.assembly_host_bytes == nested.inputs.assembly_host_bytes
            for batch_inputs in nested.inputs.task_batches
        ))

        planned = calc_diagram_eval(
            nested_expression, optimize="anytime", **options,
        )
        self.assertTrue(all(
            path.plan.reserved_host_bytes >= nested.inputs.assembly_host_bytes
            for path in planned.path.batch_paths
        ))

    def test_fixed_coordinate_nested_repeat_also_reserves_recipe_payload(self):
        rng = np.random.default_rng(260932)
        component = ElementalComponent(0)
        elemental = ArrayElemental(
            {"v2v": rng.normal(size=(1, 1, 2, 1, 1)).astype(np.complex128)},
            (component,), phase_convention="anchor",
        )
        propagator = ArrayPropagator(
            {"vsv": rng.normal(size=(2, 2, 4, 4, 1, 1)).astype(np.complex128)},
            time_convention="absolute",
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, q],
        )
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0, 1)])
        vertices = {name: Meson(elemental, operator, phase_convention="anchor")
                    for name in ("a", "b")}
        shared = dict(
            propagator_map={q: propagator}, vertex_map=vertices, usedNe=1,
            time_axes={"sink": 0, "source": 1}, memory_limit=1 << 25,
            host_memory_limit=1 << 25, execute=False,
        )
        single = calc_diagram_eval(graph, optimize="anytime", **shared)
        repeated = calc_diagram_eval(
            {"one": [graph, graph], "two": {"deep": (graph, graph, graph)}},
            optimize="anytime", **shared,
        )
        self.assertFalse(repeated.inputs.task_batches)
        self.assertGreater(repeated.inputs.assembly_host_bytes,
                           single.inputs.assembly_host_bytes)
        self.assertGreaterEqual(repeated.path.reserved_host_bytes,
                               repeated.inputs.assembly_host_bytes)


if __name__ == "__main__":
    unittest.main()
