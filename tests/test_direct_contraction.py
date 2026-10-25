"""Fixed-order execution tests with independent dense numerical references."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern
from lqcdweave.contracts.correlator.api import (
    BatchSpec,
    ContractionProblem,
    DiagramBindings,
    DiagramInputs,
    ExpandedCorrelator,
    OutputNode,
    OutputRecipe,
    SourcePreparation,
    SourceSpec,
    SparseBridgeSpec,
    StorageBlock,
    TermContribution,
    describe_coordinates,
)
from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.contraction.results.api import (
    TermValue,
    assemble_correlator,
)
from lqcdweave.contracts.correlator.api import ContractionTerm, TensorSpec
from lqcdweave.contraction.execution.domain import _validate_term_values
from lqcdweave.contracts.execution import DirectExecution, PathExecution
from lqcdweave.runtime.scheduler import (
    DirectExecution as RuntimeDirectExecution,
    PathExecution as RuntimePathExecution,
)
from lqcdweave.contraction.correlator import execute_contraction_path


class _MemoryLoader:
    def __init__(self, payloads, *, transforms=None, fail_materialize=None):
        self.payloads = payloads
        self.transforms = transforms or {}
        self.fail_materialize = fail_materialize
        self.reads = []
        self.materializations = []
        self.closed = []

    def read_block(self, key, block):
        self.reads.append(key)
        return self.payloads[key]

    def materialize(self, key, source, blocks):
        self.materializations.append(key)
        if key == self.fail_materialize:
            raise FileNotFoundError(f"missing materialized data for {key}")
        transform = self.transforms.get(key)
        if transform is not None:
            return transform(key, source, blocks)
        raw = blocks[source.region]
        return raw[key] if isinstance(raw, dict) and key in raw else raw

    def close_block(self, key, value):
        self.closed.append(key)


def _make_expanded(problem, term_ids, payloads, *, coordinates=None, recipe=None,
                   shared_regions=None, preparations_override=None, loader=None,
                   adapter=None):
    coordinates, batch = describe_coordinates() if coordinates is None else (coordinates, BatchSpec())
    shared_regions = shared_regions or {}
    sources = problem.sources
    blocks = {}
    preparations = {}
    block_payloads = {}
    for key, source in sources.items():
        block_id = shared_regions.get(key, source.region)
        blocks.setdefault(
            block_id,
            StorageBlock("fixture", block_id, max(source.region_bytes, 1),
                         host_bytes=max(source.region_bytes, 1)),
        )
        preparations[key] = SourcePreparation((block_id,))
        if block_id in payloads:
            block_payloads[block_id] = payloads[block_id]
        elif key in payloads:
            block_payloads.setdefault(block_id, {})[key] = payloads[key]
    if preparations_override is not None:
        preparations.update(preparations_override)
    loader = _MemoryLoader(block_payloads) if loader is None else loader
    inputs = DiagramInputs(
        problem, blocks, preparations, loader, coordinates, batch, tuple(term_ids),
    )
    if recipe is None:
        recipe = OutputRecipe(OutputNode("constant", 0), tuple(term_ids), coordinates)
    return ExpandedCorrelator(
        object(), inputs, recipe, DiagramBindings(), coordinates, batch, adapter,
    )


def _sources_for(specs, *, record_keys=(), shared=()):
    shared = set(shared)
    sources = {}
    shared_size = sum(
        int(np.prod(specs[key].shape)) * specs[key].dtype.itemsize
        for key in shared
    )
    for key, spec in specs.items():
        size = int(np.prod(spec.shape)) * spec.dtype.itemsize
        sources[key] = SourceSpec("shared" if key in shared else key,
                                  shared_size if key in shared else size)
    for key in record_keys:
        sources[key] = SourceSpec(key, 0)
    return sources


class DirectContractionTests(unittest.TestCase):
    def test_runtime_execution_dtos_are_contract_compatibility_reexports(self):
        self.assertIs(RuntimeDirectExecution, DirectExecution)
        self.assertIs(RuntimePathExecution, PathExecution)

    def test_dense_general_terms_repeat_inputs_and_assemble_coefficients_once(self):
        rng = np.random.default_rng(712)
        arrays = {
            "A": rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)),
            "B": rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)),
            "C": rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)),
        }
        specs = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        terms = (
            ContractionTerm(("A", "B", "C"), "ij,jk,kl->il"),
            ContractionTerm(("A", "A"), "ij,ij->"),
        )
        problem = ContractionProblem(specs, terms, _sources_for(specs, shared=("A", "B")))
        coords, _ = describe_coordinates()
        recipe = OutputRecipe(
            OutputNode("tuple", children=(
                OutputNode("term", TermContribution("chain", 2 + 3j)),
                OutputNode("term", TermContribution("repeat", -0.5j)),
            )),
            ("chain", "repeat"), coords,
        )
        loader = _MemoryLoader({"shared": arrays, "C": arrays["C"]})
        expanded = _make_expanded(
            problem, ("chain", "repeat"), {"shared": arrays, "C": arrays["C"]},
            recipe=recipe, loader=loader,
        )
        chain_reference = np.einsum("ij,jk,kl->il", arrays["A"], arrays["B"], arrays["C"], optimize=False)
        repeat_reference = np.einsum("ij,ij->", arrays["A"], arrays["A"], optimize=False)

        original_einsum = np.einsum
        calls = []

        def no_search_einsum(*arguments, **kwargs):
            self.assertIs(kwargs.get("optimize"), False)
            self.assertFalse(isinstance(arguments[0], str),
                             "direct execution must use integer einsum arguments")
            operand_count = (len(arguments) - 1) // 2
            calls.append(AxisPattern(
                tuple(tuple(arguments[2 * index + 1]) for index in range(operand_count)),
                tuple(arguments[-1]),
            ))
            return original_einsum(*arguments, **kwargs)

        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("direct must not search")), \
             patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("direct must not search")), \
             patch("opt_einsum.contract_path",
                   side_effect=AssertionError("direct must not ask opt_einsum for a path")), \
             patch("numpy.einsum", new=no_search_einsum):
            path = find_contraction_path(expanded, memory_limit=1 << 30, optimize="none")
            execution = execute_contraction_path(expanded, path)

        self.assertEqual(set(calls), {
            parse_axis_pattern(equation).normalized()
            for equation in ("ij,jk->ik", "kl,ik->il", "ij,ij->")
        })
        self.assertEqual(loader.reads, ["shared", "C"])
        self.assertCountEqual(loader.materializations, ["A", "B", "C"])
        self.assertEqual(loader.closed, ["C", "shared"])
        self.assertEqual([value.term_id for value in execution.term_values], ["chain", "repeat"])
        np.testing.assert_allclose(execution.term_values[0].value, chain_reference)
        self.assertEqual(execution.term_values[1].value.shape, ())
        np.testing.assert_allclose(execution.term_values[1].value, repeat_reference)
        assembled = assemble_correlator(expanded, execution.term_values)
        np.testing.assert_allclose(assembled[0], (2 + 3j) * chain_reference)
        np.testing.assert_allclose(assembled[1], (-0.5j) * repeat_reference)
        self.assertFalse(np.shares_memory(execution.term_values[0].value, arrays["A"]))

    def test_shape_error_and_materialize_failure_close_borrowed_blocks(self):
        expected = TensorSpec((2, 3), "float64")
        problem = ContractionProblem(
            {"A": expected},
            (ContractionTerm(("A",), "ij->i"),),
            {"A": SourceSpec("A", 48)},
        )
        malformed = _MemoryLoader({"A": np.ones((2, 4), dtype=np.float64)})
        expanded = _make_expanded(problem, ("T",), {"A": np.ones((2, 4))}, loader=malformed)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        with self.assertRaisesRegex(ValueError, "mismatched tensor"):
            execute_contraction_path(expanded, path)
        self.assertEqual(malformed.closed, ["A"])

        failing = _MemoryLoader({"A": np.ones((2, 3))}, fail_materialize="A")
        expanded = _make_expanded(problem, ("T",), {"A": np.ones((2, 3))}, loader=failing)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        with self.assertRaisesRegex(FileNotFoundError, "missing materialized data"):
            execute_contraction_path(expanded, path)
        self.assertEqual(failing.closed, ["A"])

    def test_missing_preparation_fails_before_reads_and_real_oom_propagates(self):
        spec = TensorSpec((3,), "float64")
        problem = ContractionProblem(
            {"A": spec},
            (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("A", 24)},
        )
        loader = _MemoryLoader({"A": np.arange(3, dtype=np.float64)})
        expanded = _make_expanded(
            problem, ("T",), {"A": np.arange(3, dtype=np.float64)}, loader=loader,
            preparations_override={"A": SourcePreparation(())},
        )
        # Replace the empty preparation with a missing block identity.
        bad_preparations = {"A": SourcePreparation(("absent",))}
        inputs = replace(expanded.inputs, preparations=bad_preparations)
        expanded = replace(expanded, inputs=inputs)
        with self.assertRaisesRegex(ValueError, "missing block"):
            find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        self.assertEqual(loader.reads, [])

        valid_loader = _MemoryLoader({"A": np.arange(3, dtype=np.float64)})
        expanded = _make_expanded(problem, ("T",), {"A": np.arange(3, dtype=np.float64)}, loader=valid_loader)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        with patch("numpy.einsum", side_effect=MemoryError("simulated allocator failure")):
            with self.assertRaisesRegex(MemoryError, "simulated allocator failure"):
                execute_contraction_path(expanded, path)
        self.assertEqual(valid_loader.closed, ["A"])

    def test_scan_coordinates_are_explicitly_unsupported_before_loader_io(self):
        coordinates, _ = describe_coordinates(time_pairs=({0: 1},))
        spec = TensorSpec((2,), "float64")
        problem = ContractionProblem(
            {"A": spec},
            (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("A", 16)},
        )
        loader = _MemoryLoader({"A": np.ones(2)})
        expanded = _make_expanded(problem, ("T",), {"A": np.ones(2)},
                                  coordinates=coordinates, loader=loader)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        with self.assertRaisesRegex(NotImplementedError, "explicit CoordinateBatch metadata"):
            execute_contraction_path(expanded, path)
        self.assertEqual(loader.reads, [])

    def test_result_coverage_uses_term_coordinate_pairs(self):
        coordinates, _ = describe_coordinates(time_pairs=({0: 1}, {0: 2}))
        spec = TensorSpec((1,), "float64")
        problem = ContractionProblem(
            {"A": spec},
            (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("A", 8)},
        )
        expanded = _make_expanded(problem, ("T",), {"A": np.ones(1)}, coordinates=coordinates)
        complete = tuple(
            TermValue("T", point.key, np.asarray(index))
            for index, point in enumerate(coordinates.points)
        )
        _validate_term_values(expanded, complete)
        with self.assertRaisesRegex(ValueError, "term_id, coordinate_key"):
            _validate_term_values(expanded, complete[:1])
        with self.assertRaisesRegex(ValueError, "term_id, coordinate_key"):
            _validate_term_values(expanded, complete + (complete[0],))

    def test_single_sparse_bridge_preserves_diagonal_point_axis_and_duplicate_coo(self):
        rng = np.random.default_rng(901)
        used_np, ne, nc = 3, 2, 3
        left = rng.normal(size=(ne, 4, used_np, nc)) + 1j * rng.normal(size=(ne, 4, used_np, nc))
        right = rng.normal(size=(4, used_np, nc, ne)) + 1j * rng.normal(size=(4, used_np, nc, ne))
        gamma = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        indices = np.array([[0, 0], [1, 1], [1, 1], [2, 2], [0, 2]])
        values = rng.normal(size=(len(indices), nc, nc)) + 1j * rng.normal(size=(len(indices), nc, nc))
        empty = {"type": "sparse", "indices": np.empty((0, 2), dtype=np.int64),
                 "values": np.empty((0, nc, nc), dtype=np.complex128)}
        record = {"type": "sparse", "indices": indices, "values": values}
        parts = ((gamma, 0.75 - 0.2j, record), (gamma, 2.0, empty))

        dense_record = np.zeros((used_np, used_np, nc, nc), dtype=np.complex128)
        for (left_point, right_point), block in zip(indices, values, strict=True):
            if left_point == right_point:
                dense_record[left_point, right_point] += block
        dense_reference = (0.75 - 0.2j) * np.einsum(
            "aipc,ij,pqcd,jqdb->abp", left, gamma, dense_record, right, optimize=False,
        )

        specs = {
            "left": TensorSpec(left.shape, left.dtype),
            "right": TensorSpec(right.shape, right.dtype),
        }
        sources = _sources_for(specs, record_keys=("record",))
        problem = ContractionProblem(
            specs,
            (ContractionTerm(("left", "right"), "aipc,jqdb->abp"),),
            sources,
            sparse_bridges=(SparseBridgeSpec(
                "left", "right", "record", used_np, len(indices), same_point=True,
                part_count=len(parts),
            ),),
        )
        loader = _MemoryLoader({"left": left, "right": right, "record": parts})
        expanded = _make_expanded(problem, ("sparse",),
                                  {"left": left, "right": right, "record": parts}, loader=loader)
        path = find_contraction_path(expanded, memory_limit=1 << 30, optimize="none")
        original_einsum = np.einsum

        def no_search_einsum(*arguments, **kwargs):
            self.assertIs(kwargs.get("optimize"), False)
            return original_einsum(*arguments, **kwargs)

        with patch("opt_einsum.contract_path",
                   side_effect=AssertionError("sparse direct execution cannot search")), \
             patch("numpy.einsum", new=no_search_einsum):
            execution = execute_contraction_path(expanded, path)
        np.testing.assert_allclose(execution.term_values[0].value, dense_reference,
                                   rtol=1e-12, atol=1e-10)
        self.assertEqual(execution.term_values[0].value.shape, (ne, ne, used_np))
        self.assertEqual(loader.reads, ["left", "right", "record"])
        self.assertEqual(loader.closed, ["record", "right", "left"])

    def test_double_sparse_cycle_keeps_diagonal_and_repeated_records(self):
        rng = np.random.default_rng(928)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        propagator = random((4, 4, 3, 3, 3, 3))
        original = propagator.copy()
        spin1, spin2 = random((4, 4)), random((4, 4))
        indices = np.array([[0, 0], [1, 1], [1, 1], [2, 2], [0, 2]])
        values1, values2 = random((5, 3, 3)), random((5, 3, 3))
        diagonal1, diagonal2 = np.zeros((3, 3, 3), complex), np.zeros((3, 3, 3), complex)
        for (left_point, right_point), first, second in zip(indices, values1, values2, strict=True):
            if left_point == right_point:
                diagonal1[left_point] += first
                diagonal2[left_point] += second
        dense_reference = np.einsum(
            "abpdpf,ghpjpl,bg,pfj,ah,pdl->",
            propagator, propagator, spin1, diagonal1, spin2, diagonal2,
            optimize=False,
        )
        record1 = {"type": "sparse", "indices": indices, "values": values1}
        record2 = {"type": "sparse", "indices": indices, "values": values2}
        parts1, parts2 = ((spin1, 1, record1),), ((spin2, 1, record2),)
        spec = TensorSpec(propagator.shape, propagator.dtype)
        problem = ContractionProblem(
            {"same": spec},
            (ContractionTerm(("same", "same"), "abedef,ghejel->"),),
            {
                "same": SourceSpec("same", propagator.nbytes),
                "r1": SourceSpec("r1", 0),
                "r2": SourceSpec("r2", 0),
            },
            sparse_bridges=(
                SparseBridgeSpec("same", "same", "r1", 3, 5,
                                 left_axes=(1, 4, 5), right_axes=(0, 2, 3), same_point=True),
                SparseBridgeSpec("same", "same", "r2", 3, 5,
                                 left_axes=(0, 2, 3), right_axes=(1, 4, 5), same_point=True),
            ),
        )
        loader = _MemoryLoader({"same": propagator, "r1": parts1, "r2": parts2})
        expanded = _make_expanded(problem, ("cycle",),
                                  {"same": propagator, "r1": parts1, "r2": parts2}, loader=loader)
        path = find_contraction_path(expanded, memory_limit=1 << 32, optimize="none")
        original_einsum = np.einsum

        def no_search_einsum(*arguments, **kwargs):
            self.assertIs(kwargs.get("optimize"), False)
            return original_einsum(*arguments, **kwargs)

        with patch("opt_einsum.contract_path",
                   side_effect=AssertionError("cycle direct execution cannot search")), \
             patch("numpy.einsum", new=no_search_einsum):
            execution = execute_contraction_path(expanded, path)
        self.assertEqual(execution.term_values[0].value.shape, ())
        np.testing.assert_allclose(execution.term_values[0].value, dense_reference,
                                   rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(propagator, original)
        self.assertCountEqual(loader.reads, ["same", "r1", "r2"])


if __name__ == "__main__":
    unittest.main()
