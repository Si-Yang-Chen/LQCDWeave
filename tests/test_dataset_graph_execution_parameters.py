"""Runtime execution settings bind only to real producer attempts."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import test_dataset_graph_runtime as graph_fixture
from lqcdweave.runtime import dataset_stage
from lqcdweave.storage.dataset import DatasetWriter, inspect_dataset
from lqcdweave.runtime import datasets as dataset_workflow
from lqcdweave.runtime.datasets import run_dataset_requests


class DatasetGraphExecutionParameterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = graph_fixture.DatasetGraphRuntimeTests(
            "test_only_published_parents_bind_and_repeated_graph_does_no_work"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_graph(self, *, resume=False, data_ids=None,
                  producers=None, execution_parameters=None,
                  pass_execution_parameters=False,
                  materialize_reused_outputs=(), outputs=None):
        kwargs = dict(resume=resume, data_ids=data_ids,
                      materialize_reused_outputs=materialize_reused_outputs)
        if pass_execution_parameters:
            kwargs["execution_parameters"] = execution_parameters
        return run_dataset_requests(
            self.fixture.requests, self.fixture.index,
            self.fixture.outputs if outputs is None else outputs,
            self.fixture.provenance,
            dict(basis=self.fixture.basis, child=self.fixture.child)
            if producers is None else producers,
            **kwargs,
        )

    def bind_spy(self, calls):
        original = DatasetWriter.bind_producer_attempt

        def bind(writer, producer_attempt_id, **kwargs):
            calls.append(deepcopy(kwargs))
            return original(writer, producer_attempt_id, **kwargs)

        return patch.object(DatasetWriter, "bind_producer_attempt", new=bind)

    def test_invalid_execution_map_is_rejected_before_index_or_output_io(self):
        invalid = (
            ({"unknown": {"batch": 2}}, ValueError),
            ({"basis": None}, TypeError),
            ({"child": {"batch": float("nan")}}, ValueError),
            ({"child": {"batch": (1, 2)}}, TypeError),
            ({1: {"batch": 2}}, TypeError),
        )
        with patch.object(self.fixture.index, "observations",
                          side_effect=AssertionError("index observations reached")), \
             patch.object(self.fixture.index, "open",
                          side_effect=AssertionError("index open reached")):
            for parameters, error in invalid:
                with self.subTest(parameters=parameters):
                    with self.assertRaises(error):
                        self.run_graph(execution_parameters=parameters,
                                       pass_execution_parameters=True)
                    self.assertTrue(all(not path.exists()
                                        for path in self.fixture.outputs.values()))

    def test_explicit_parameters_are_detached_and_default_bind_shape_is_unchanged(self):
        parameters = {
            "basis": {"solver": {"tolerance": 1e-8, "labels": ["a", "b"]}},
            "child": {"diagram_sink_batch": {
                "axis_role": "sink_time", "requested_size": 2,
            }},
        }
        calls = []
        with self.bind_spy(calls):
            identities = self.run_graph(execution_parameters=parameters,
                                        pass_execution_parameters=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], {"execution_parameters": parameters["basis"]})
        self.assertEqual(calls[1], {"execution_parameters": parameters["child"]})
        self.assertIsNot(calls[0]["execution_parameters"], parameters["basis"])
        self.assertIsNot(calls[0]["execution_parameters"]["solver"],
                         parameters["basis"]["solver"])
        parameters["basis"]["solver"]["labels"].append("caller-mutation")
        self.assertEqual(calls[0]["execution_parameters"]["solver"]["labels"],
                         ["a", "b"])

        self.assertEqual(identities.keys(), {"basis", "child"})
        before = {name: graph_fixture.registration_snapshot(path)
                  for name, path in self.fixture.outputs.items()}
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "materialized-child"
            reuse_calls = []
            with self.bind_spy(reuse_calls):
                materialized = self.run_graph(
                    resume=True, data_ids=identities,
                    producers=dict(
                        basis=lambda *args: self.fail("basis materialization ran producer"),
                        child=lambda *args: self.fail("child materialization ran producer"),
                    ),
                    execution_parameters={"basis": {"solver": {"batch": 9}},
                                          "child": {"diagram_sink_batch": {
                                              "axis_role": "sink_time",
                                              "requested_size": "full",
                                          }}},
                    pass_execution_parameters=True,
                    materialize_reused_outputs=("child",),
                    outputs={"child": destination},
                )
            self.assertEqual(materialized, identities)
            self.assertEqual(reuse_calls, [])
            self.assertEqual(graph_fixture.registration_snapshot(destination),
                             graph_fixture.registration_snapshot(
                                 self.fixture.outputs["child"]))
        self.assertEqual({name: graph_fixture.registration_snapshot(path)
                          for name, path in self.fixture.outputs.items()}, before)

        fresh = graph_fixture.DatasetGraphRuntimeTests(
            "test_only_published_parents_bind_and_repeated_graph_does_no_work"
        )
        fresh.setUp()
        self.addCleanup(fresh.doCleanups)
        default_calls = []
        original = DatasetWriter.bind_producer_attempt

        def default_bind(writer, producer_attempt_id, **kwargs):
            default_calls.append(dict(kwargs))
            return original(writer, producer_attempt_id, **kwargs)

        with patch.object(DatasetWriter, "bind_producer_attempt", new=default_bind):
            run_dataset_requests(
                fresh.requests, fresh.index, fresh.outputs, fresh.provenance,
                dict(basis=fresh.basis, child=fresh.child),
            )
        self.assertEqual(default_calls, [{}, {}])
        self.assertIsNone(inspect_dataset(fresh.outputs["basis"])["execution"][
                          "attempts"][0].get("execution_parameters"))
        self.assertIsNone(inspect_dataset(fresh.outputs["child"])["execution"][
                          "attempts"][0].get("execution_parameters"))

    def test_partial_resume_changes_only_missing_rows_and_reuse_adds_no_attempt(self):
        events = []
        original_child = self.fixture.child

        def interrupt_after_first_row(spec, inputs, writer):
            events.append(("producer", tuple(writer.missing)))
            writer.write(0, 2 * graph_fixture.row(1, (4, 1)))
            raise RuntimeError("controlled parameterized interruption")

        first_parameters = {
            "basis": {"basis_batch": 3},
            "child": {"diagram_sink_batch": {"axis_role": "sink_time",
                                               "requested_size": 2}},
        }
        with self.bind_spy(events):
            with self.assertRaisesRegex(RuntimeError, "controlled parameterized interruption"):
                self.run_graph(producers=dict(
                    basis=self.fixture.basis, child=interrupt_after_first_row,
                ), execution_parameters=first_parameters,
                   pass_execution_parameters=True)
        child_path = self.fixture.outputs["child"]
        partial_row = graph_fixture.cfg_rows(child_path, (0,))
        partial = inspect_dataset(child_path)
        self.assertEqual(partial["state"], "partial")
        first_attempt = partial["execution"]["attempts"][-1]
        self.assertEqual(first_attempt["execution_parameters"], first_parameters["child"])
        basis_attempt = inspect_dataset(self.fixture.outputs["basis"])["execution"]["attempts"][-1]
        self.assertEqual(basis_attempt["execution_parameters"], first_parameters["basis"])

        events.clear()
        resumed_parameters = {"child": {"diagram_sink_batch": {
            "axis_role": "sink_time", "requested_size": 4,
        }}}

        def record_resumed_producer(spec, inputs, writer):
            events.append(("producer", tuple(writer.missing)))
            return original_child(spec, inputs, writer)

        with self.bind_spy(events):
            identities = self.run_graph(resume=True, producers=dict(
                basis=lambda *args: self.fail("basis reuse ran producer"),
                child=record_resumed_producer,
            ), execution_parameters=resumed_parameters,
               pass_execution_parameters=True)
        self.assertEqual(events[0], {"execution_parameters": resumed_parameters["child"]})
        self.assertEqual(events[1], ("producer", (1,)))
        self.assertEqual(graph_fixture.cfg_rows(child_path, (0,)), partial_row)
        completed = inspect_dataset(child_path)
        self.assertEqual(identities["child"], completed["data_id"])
        self.assertEqual(
            [attempt["execution_parameters"] for attempt in completed["execution"]["attempts"]],
            [first_parameters["child"], resumed_parameters["child"]],
        )

        registrations = {name: graph_fixture.registration_snapshot(path)
                         for name, path in self.fixture.outputs.items()}
        calls_before_reuse = list(events)
        with self.bind_spy(events):
            reused = self.run_graph(
                resume=True, data_ids=identities,
                producers=dict(
                    basis=lambda *args: self.fail("complete basis reuse ran producer"),
                    child=lambda *args: self.fail("complete child reuse ran producer"),
                ), execution_parameters={"basis": {"basis_batch": 9},
                                         "child": {"diagram_sink_batch": {
                                             "axis_role": "sink_time",
                                             "requested_size": "full",
                                         }}},
                pass_execution_parameters=True,
            )
        self.assertEqual(reused, identities)
        self.assertEqual(events, calls_before_reuse)
        self.assertEqual({name: graph_fixture.registration_snapshot(path)
                          for name, path in self.fixture.outputs.items()}, registrations)

if __name__ == "__main__":
    unittest.main()
