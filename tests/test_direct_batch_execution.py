"""Direct X execution preserves globally joined physical block ownership."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    BatchSpec,
    ContractionProblem,
    ContractionTerm,
    DiagramBindings,
    DiagramInputs,
    ExpandedCorrelator,
    OutputNode,
    OutputRecipe,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
    TermContribution,
    describe_coordinates,
    group_coordinate_batches,
)
from lqcdweave.contraction.execution.api import BoundContractionExecution
from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.runtime.scheduler import execute_request


class _TaskLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.active = None
        self.reads = []
        self.materializations = []
        self.closed = []
        self.close_all_calls = 0

    @contextmanager
    def activate(self, batch_index):
        old = self.active
        self.active = batch_index
        try:
            yield
        finally:
            self.active = old

    def read_block(self, key, block):
        self.reads.append(key)
        return {"all_coordinates": self.arrays}

    def materialize(self, key, source, blocks):
        self.materializations.append((key, self.active))
        return blocks["shared"]["all_coordinates"][self.active]

    def close_block(self, key, value):
        self.closed.append(key)

    def close_all(self):
        self.close_all_calls += 1


class DirectTaskBatchTests(unittest.TestCase):
    def test_shared_block_reads_once_until_final_coordinate_batch(self):
        coordinates, batch = describe_coordinates(
            time_pairs=({"source": 0, "sink": 4}, {"source": 1, "sink": 4}),
            batch={"axis": "pair", "size": 1},
        )
        coordinate_batches = group_coordinate_batches(coordinates, batch)
        arrays = (np.array([1.0, 2.0]), np.array([10.0, 20.0]))
        loader = _TaskLoader(arrays)
        problem = ContractionProblem(
            {"A": TensorSpec((2,), np.float64)},
            (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("shared", 32, snapshot_bytes=16)},
        )
        blocks = {"shared": StorageBlock("fixture", "both coordinates", 32, host_bytes=32)}
        preparations = {"A": SourcePreparation(("shared",))}
        term_ids = ("sum",)
        children = tuple(
            DiagramInputs(
                problem,
                blocks,
                preparations,
                loader,
                coordinates,
                batch,
                term_ids,
                coordinate_batch=coordinate_batch,
                batch_output_axes=(None,),
            )
            for coordinate_batch in coordinate_batches
        )
        inputs = DiagramInputs(
            problem,
            blocks,
            preparations,
            loader,
            coordinates,
            batch,
            term_ids,
            task_batches=children,
        )
        expanded = ExpandedCorrelator(
            object(),
            inputs,
            OutputRecipe(OutputNode("term", TermContribution("sum")), term_ids, coordinates),
            DiagramBindings(),
            coordinates,
            batch,
        )
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")

        from lqcdweave.contraction.execution import domain
        with patch.object(domain, '_validate_term_values', wraps=domain._validate_term_values) as audit:
            result = execute_request(
                BoundContractionExecution(expanded, path), backend=np, measure=False,
            )
            audit.assert_not_called()

        self.assertEqual(loader.reads, ["shared"])
        self.assertEqual(loader.materializations, [("A", 0), ("A", 1)])
        self.assertEqual(loader.closed, ["shared"])
        self.assertEqual(loader.close_all_calls, 1)
        self.assertEqual(
            [(item.term_id, item.coordinate_key) for item in result.term_values],
            [("sum", point.key) for point in coordinates.points],
        )
        np.testing.assert_array_equal(
            assemble_correlator(expanded, result.term_values),
            np.array([3.0, 30.0]),
        )

        for optimize in ('none', 'anytime'):
            checked_path = find_contraction_path(expanded, memory_limit=1 << 20, optimize=optimize)
            with patch.object(domain, '_validate_term_values', wraps=domain._validate_term_values) as audit:
                checked = execute_request(
                    BoundContractionExecution(expanded, checked_path, debug=True), backend=np,
                )
                self.assertEqual(audit.call_count, 1)
            np.testing.assert_array_equal(
                assemble_correlator(expanded, checked.term_values), np.array([3.0, 30.0]))


if __name__ == "__main__":
    unittest.main()
