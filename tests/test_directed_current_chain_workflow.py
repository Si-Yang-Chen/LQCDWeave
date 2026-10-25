"""V/V one-current open tensor chains against a dense oracle."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import test_project_source_provenance as source_fixture

from production.elemental.test_elemental import fixture, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.operators.single_hadron.api import ConservedVectorInsertion, gamma
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.directed_current import directed_current_chain_document
from lqcdweave.planning.production import directed_current_dataset_document
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader, DatasetWriter, register_npy,
)
from lqcdweave.contraction.directed_current_measurement import (
    plan_indexed_directed_current_chain, write_directed_current_chain_rows,
)
from lqcdweave.contraction import directed_current_measurement as workflow
from lqcdweave.runtime.datasets import open_dataset_view


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class DirectedCurrentChainWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, self.links, vectors, _ = fixture()
        self.lt = self.sizes[3]
        self.labels = [2, 0, 1]
        self.vectors = vectors[:, self.labels]
        self.momenta = [(1, -1, 0), (0, 0, 0)]
        self.selected_momentum = 1
        self.cfg = dict(ensemble="synthetic-chain", configuration="cfg-chain",
                        lattice=list(self.sizes))

        gauge_axes = [axis("time", [2, 0, 1]), axis("z", range(self.sizes[2])),
            axis("y", range(self.sizes[1])), axis("x", range(self.sizes[0])),
            axis("direction", range(4)), axis("color_row", range(3)),
            axis("color_column", range(3))]
        gauge_spec = self.spec("gauge", gauge_axes)
        self.gauge_id = self.save("gauge", gauge_spec, self.links[[2, 0, 1]])
        other_links = np.roll(self.links, 1, axis=0).copy()
        third_links = np.roll(self.links, 2, axis=1).copy()
        self.other_gauge_id = self.save("incoming-gauge", gauge_spec, other_links[[2, 0, 1]])
        self.third_gauge_id = self.save("outgoing-gauge", gauge_spec, third_links[[2, 0, 1]])

        basis_axes = [axis("time", [1, 2, 0]), axis("mode", self.labels),
            axis("z", range(self.sizes[2])), axis("y", range(self.sizes[1])),
            axis("x", range(self.sizes[0])), axis("color", range(3))]
        self.basis_spec = self.spec("explicit_basis", basis_axes)
        self.basis_id = self.save("basis", self.basis_spec, self.vectors[[1, 2, 0]])
        with self.index.open(self.gauge_id) as gauge, self.index.open(self.basis_id) as basis:
            _, current_document = directed_current_dataset_document(
                gauge.spec, basis.spec, gauge_id=self.gauge_id, basis_id=self.basis_id,
                momentum_list=self.momenta, boundary="periodic")
        current_values = np.empty((self.lt, 8, len(self.momenta), len(self.labels), len(self.labels)),
                                  dtype=np.complex128)
        for time in range(self.lt):
            for direction in range(8):
                field_time = (time + (1 if direction == 6 else -1 if direction == 7 else 0)) % self.lt
                for momentum_index, momentum in enumerate(self.momenta):
                    matrix = reference_matrix(self.links, (direction,), time, momentum,
                                              phase="midpoint", boundary="periodic")
                    current_values[time, direction, momentum_index] = (
                        self.vectors[time].reshape(len(self.labels), -1).conj() @ matrix @
                        self.vectors[field_time].reshape(len(self.labels), -1).T)
        self.current_id = self.save("current", DatasetSpec(current_document), current_values)
        self.current_values = current_values

        dimension = int(np.prod(self.sizes[:3])) * 3 * 4
        rng = np.random.default_rng(407)
        self.incoming_dense = (rng.normal(size=(self.lt, self.lt, dimension, dimension)) +
                               1j * rng.normal(size=(self.lt, self.lt, dimension, dimension)))
        self.outgoing_dense = (rng.normal(size=(self.lt, self.lt, dimension, dimension)) +
                               1j * rng.normal(size=(self.lt, self.lt, dimension, dimension)))
        self.incoming_tau = self.project_dense(self.incoming_dense)
        self.outgoing_tau = self.project_dense(self.outgoing_dense)
        self.incoming_id = self.save_vsv("incoming-vsv", self.incoming_tau,
            self.other_gauge_id, mass=0.13)
        self.outgoing_id = self.save_vsv("outgoing-vsv", self.outgoing_tau,
            self.third_gauge_id, mass=0.31)

        terms = ConservedVectorInsertion(3, wilson_r=0.9).terms
        self.terms = (
            replace(terms[0], coefficient=0.37 + 0.21j, normalization=0.8 - 0.14j,
                bar_offset=(0, 0, 0, 1), field_offset=(0, 0, 0, 2),
                link_origin_offset=(0, 0, 0, 1)),
            replace(terms[1], coefficient=-0.16 + 0.42j, normalization=0.55 + 0.19j,
                bar_offset=(0, 0, 0, -1), field_offset=(0, 0, 0, -2),
                link_origin_offset=(0, 0, 0, -2)),
        )

    def spec(self, kind, axes, **extra):
        document = dict(schema=1, kind=kind, configuration=self.cfg, axes=axes,
            dtype="complex128", bases={}, definition={}, conventions={}, parents=[])
        document.update(extra)
        return DatasetSpec(document)

    def save(self, name, spec, values, *, provenance=None, provenance_mode=None):
        source = self.root / f"{name}.npy"
        np.save(source, values)
        options = dict(provenance=(dict(test="directed_current_chain_oracle")
                                   if provenance is None else provenance))
        if provenance_mode is not None:
            options["provenance_mode"] = provenance_mode
        identity = register_npy(self.root / name, source, spec, **options)
        self.index.register(self.root / name)
        return identity

    def project_dense(self, dense):
        result = np.empty((self.lt, self.lt, 4, 4, len(self.labels), len(self.labels)),
                          dtype=np.complex128)
        for source_time in range(self.lt):
            for sink_time in range(self.lt):
                left = np.kron(np.eye(4), self.vectors[sink_time].reshape(len(self.labels), -1).conj())
                right = np.kron(np.eye(4), self.vectors[source_time].reshape(len(self.labels), -1).T)
                result[source_time, sink_time] = (
                    left @ dense[source_time, sink_time] @ right
                ).reshape(4, len(self.labels), 4, len(self.labels)).transpose(0, 2, 1, 3)
        return result

    def save_vsv(self, name, tau, gauge_id, *, mass):
        binding = dict(data_id=self.basis_id, indices=list(self.labels))
        dirac = dict(action="wilson", mass=mass, r=1.0,
                     lattice_spacing=1.0, clover=0.0)
        spec = self.spec("vsv", [axis("source_time", [2, 0, 1]),
            axis("sink_time", [1, 2, 0]), axis("sink_spin", range(4)),
            axis("source_spin", range(4)), axis("sink_mode", self.labels),
            axis("source_mode", self.labels)],
            bases=dict(sink_mode=binding, source_mode=binding),
            definition=dict(gauge_id=gauge_id, solver_tolerance=1e-8, dirac=dirac),
            conventions=dict(normalization="unscaled", spin_basis="degrand_rossi",
                             temporal_boundary="periodic"),
            parents=[gauge_id, self.basis_id])
        stored = tau[[2, 0, 1]][:,[1, 2, 0]]
        return self.save(name, spec, stored)

    def write_chain(self, output, **options):
        preview = plan_indexed_directed_current_chain(self.index, self.current_id,
            self.incoming_id, self.outgoing_id, self.terms,
            momentum=self.selected_momentum, usedNe=2, **options)
        if preview.get("action") == "blocked":
            raise ValueError("strict source policy blocks current chain inputs")
        stages = preview["stages"]
        with ExitStack() as stack:
            views = []
            for role, identity in zip(
                    ("current", "incoming_vsv", "outgoing_vsv"),
                    (self.current_id, self.incoming_id, self.outgoing_id), strict=True):
                subset_axes = (("sink_mode", "source_mode", "momentum")
                               if role == "current" else ("sink_mode", "source_mode"))
                views.append(stack.enter_context(open_dataset_view(
                    self.index, stages[role]["spec"], data_id=identity,
                    subset_axes=subset_axes)))
            destination = self.root / output / "measurement"
            writer = stack.enter_context(DatasetWriter(
                destination, stages["measurement"]["spec"],
                provenance={"test": "directed-current-chain-dense-oracle"},
            ))
            write_directed_current_chain_rows(*views, writer)
            identity = writer.publish()
        self.index.register(destination)
        return {"datasets": {"measurement": identity}}

    def expected(self, source_time, sink_time, anchor_time, *, ne=2):
        result = np.zeros((4, 4, ne, ne), dtype=np.complex128)
        for term in self.terms:
            bar_time = (anchor_time + term.bar_offset[3]) % self.lt
            field_time = (anchor_time + term.field_offset[3]) % self.lt
            direction = 6 if term.link == "forward" else 7
            raw = self.current_values[bar_time, direction, self.selected_momentum, :ne, :ne]
            g = np.asarray(gamma(term.gamma_index))
            spin = term.wilson_r * np.eye(4) + (-1 if term.link == "forward" else 1) * g
            outgoing = self.outgoing_tau[field_time, sink_time, :, :, :ne, :ne]
            incoming = self.incoming_tau[source_time, bar_time, :, :, :ne, :ne]
            for external_sink_spin in range(4):
                for external_source_spin in range(4):
                    for external_sink_mode in range(ne):
                        for external_source_mode in range(ne):
                            total = 0j
                            for current_sink_spin in range(4):
                                for current_source_spin in range(4):
                                    for current_sink_mode in range(ne):
                                        for current_source_mode in range(ne):
                                            total += (
                                                outgoing[external_sink_spin, current_source_spin,
                                                         external_sink_mode, current_source_mode] *
                                                spin[current_sink_spin, current_source_spin] *
                                                raw[current_sink_mode, current_source_mode] *
                                                incoming[current_sink_spin, external_source_spin,
                                                         current_sink_mode, external_source_mode]
                                            )
                            result[external_sink_spin, external_source_spin,
                                   external_sink_mode, external_source_mode] += (
                                term.coefficient * term.normalization * total)
        return result

    def assert_matches_oracle(self, data_id):
        with self.index.open(data_id) as output:
            for source_time in output.spec.coordinates("source_time"):
                for sink_time in output.spec.coordinates("sink_time"):
                    for anchor_time in output.spec.coordinates("anchor_time"):
                        np.testing.assert_allclose(
                            output.read(source_time=source_time, sink_time=sink_time,
                                        anchor_time=anchor_time),
                            self.expected(source_time, sink_time, anchor_time),
                            rtol=2e-12, atol=2e-9)

    def test_open_tensor_rows_match_independent_matrix_oracle(self):
        with patch.object(DatasetReader, "read",
                          side_effect=AssertionError("preview read numeric data")), \
                patch.object(workflow, "load_registered_directed_current",
                             side_effect=AssertionError("preview adapted numeric data")), \
                patch.object(workflow, "contract_directed_current_v2v",
                             side_effect=AssertionError("preview contracted numeric data")):
            preview = plan_indexed_directed_current_chain(self.index, self.current_id,
                self.incoming_id, self.outgoing_id, self.terms, momentum=self.selected_momentum,
                usedNe=2, source_times=[2, 0], sink_times=[1, 0], anchor_times=[0, 2])
        self.assertNotEqual(preview.get("action"), "blocked")
        planned = preview["stages"]["measurement"]["spec"]
        self.assertEqual(planned.coordinates("source_time"), [2, 0])
        self.assertEqual(planned.coordinates("sink_time"), [1, 0])
        self.assertEqual(planned.coordinates("anchor_time"), [0, 2])

        result = self.write_chain("open-chain", source_times=[2, 0], sink_times=[1, 0],
                          anchor_times=[0, 2])
        chain_id = result["datasets"]["measurement"]
        self.assert_matches_oracle(chain_id)
        with self.index.open(chain_id) as output:
            self.assertEqual(output.spec.axis_names, (
                "source_time", "sink_time", "anchor_time", "external_sink_spin",
                "external_source_spin", "external_sink_mode", "external_source_mode"))
            self.assertEqual(output.spec.coordinates("source_time"), [2, 0])
            self.assertEqual(output.spec.coordinates("sink_time"), [1, 0])
            self.assertEqual(output.spec.coordinates("anchor_time"), [0, 2])
            self.assertEqual(output.spec.coordinates("external_sink_mode"), [2, 0])
            self.assertEqual(output.spec.coordinates("external_source_mode"), [2, 0])
            document = output.spec.document
            self.assertEqual(document["parents"], [self.current_id, self.incoming_id, self.outgoing_id])
            self.assertEqual(document["bases"]["external_sink_mode"],
                             dict(data_id=self.basis_id, indices=[2, 0]))
            self.assertEqual(document["bases"]["external_source_mode"],
                             dict(data_id=self.basis_id, indices=[2, 0]))
            definition = document["definition"]
            self.assertEqual(definition["input_roles"], dict(current=self.current_id,
                incoming_vsv=self.incoming_id, outgoing_vsv=self.outgoing_id))
            current_view = definition["input_views"]["current"]
            incoming_view = definition["input_views"]["incoming_vsv"]
            outgoing_view = definition["input_views"]["outgoing_vsv"]
            self.assertEqual(current_view["axes"][3]["coordinates"], [2, 0])
            self.assertEqual(current_view["axes"][2]["coordinates"], [list(self.momenta[1])])
            self.assertEqual(incoming_view["definition"]["gauge_id"], self.other_gauge_id)
            self.assertEqual(outgoing_view["definition"]["gauge_id"], self.third_gauge_id)
            self.assertEqual(incoming_view["definition"]["dirac"]["mass"], 0.13)
            self.assertEqual(outgoing_view["definition"]["dirac"]["mass"], 0.31)
            self.assertEqual(document["conventions"]["interpretation"],
                             "open_tensor_chain_no_ward_claim")
        default_preview = plan_indexed_directed_current_chain(self.index, self.current_id,
            self.incoming_id, self.outgoing_id, self.terms, momentum=self.selected_momentum,
            usedNe=2)
        default_spec = default_preview["stages"]["measurement"]["spec"]
        self.assertEqual(default_spec.coordinates("source_time"), [0, 1, 2])
        self.assertEqual(default_spec.coordinates("sink_time"), [0, 1, 2])
        self.assertEqual(default_spec.coordinates("anchor_time"), [0, 1, 2])

    def test_roles_keep_same_actual_vsv_and_unknown_source_policy_blocks(self):
        preview = plan_indexed_directed_current_chain(self.index, self.current_id,
            self.incoming_id, self.incoming_id, self.terms, usedNe=2)
        spec = preview["stages"]["measurement"]["spec"]
        roles = spec.document["definition"]["input_roles"]
        views = spec.document["definition"]["input_views"]
        self.assertEqual(roles["incoming_vsv"], self.incoming_id)
        self.assertEqual(roles["outgoing_vsv"], self.incoming_id)
        self.assertIn("incoming_vsv", views)
        self.assertIn("outgoing_vsv", views)

        strict = plan_indexed_directed_current_chain(self.index, self.current_id,
            self.incoming_id, self.outgoing_id, self.terms, usedNe=2,
            source_policy="require_known")
        self.assertEqual(strict["action"], "blocked")

    def test_known_source_policy_binds_actual_chain_inputs(self):
        known_gauge = source_fixture.register_known_copy(
            self.index, self.gauge_id, self.root / "known-gauge")
        known_incoming_gauge = source_fixture.register_known_copy(
            self.index, self.other_gauge_id, self.root / "known-incoming-gauge")
        known_outgoing_gauge = source_fixture.register_known_copy(
            self.index, self.third_gauge_id, self.root / "known-outgoing-gauge")
        known_basis = source_fixture.register_known_copy(
            self.index, self.basis_id, self.root / "known-basis")

        with self.index.open(known_gauge) as gauge, self.index.open(known_basis) as basis:
            _, current_document = directed_current_dataset_document(
                gauge.spec, basis.spec, gauge_id=known_gauge, basis_id=known_basis,
                momentum_list=self.momenta, boundary="periodic")
        known_current = self.save("known-current", DatasetSpec(current_document),
            self.current_values, provenance=source_fixture.import_provenance(
                dataset_id="current-chain/current"), provenance_mode="external_import")

        def rebind_vsv(gauge_id):
            def transform(document):
                document["parents"] = [gauge_id, known_basis]
                for binding in document["bases"].values():
                    binding["data_id"] = known_basis
                document["definition"]["gauge_id"] = gauge_id
            return transform

        known_incoming = source_fixture.register_known_copy(self.index, self.incoming_id,
            self.root / "known-incoming-vsv", parents=(known_incoming_gauge, known_basis),
            transform=rebind_vsv(known_incoming_gauge))
        known_outgoing = source_fixture.register_known_copy(self.index, self.outgoing_id,
            self.root / "known-outgoing-vsv", parents=(known_outgoing_gauge, known_basis),
            transform=rebind_vsv(known_outgoing_gauge))
        preview = plan_indexed_directed_current_chain(self.index, known_current,
            known_incoming, known_outgoing, self.terms, momentum=self.selected_momentum,
            usedNe=2, source_policy="require_known")
        self.assertEqual(preview["source_status"], "known")
        self.assertTrue({known_current, known_incoming, known_outgoing,
                         known_gauge, known_incoming_gauge,
                         known_outgoing_gauge, known_basis}.issubset(
                             set(preview["source_origins"])))

        self.assertEqual(preview["stages"]["measurement"]["spec"].document[
            "definition"]["input_roles"], dict(current=known_current,
            incoming_vsv=known_incoming, outgoing_vsv=known_outgoing))

    def test_invalid_time_selection_and_open_endpoints_fail_before_output_creation(self):
        for name, options in (("duplicate-time", dict(source_times=[0, 0])),
                              ("outside-time", dict(sink_times=[self.lt]))):
            with self.assertRaises(ValueError):
                plan_indexed_directed_current_chain(self.index, self.current_id,
                    self.incoming_id, self.outgoing_id, self.terms,
                    momentum=self.selected_momentum, usedNe=2, **options)

        with self.index.open(self.current_id) as current:
            open_current_document = current.spec.document
        open_current_document["definition"]["path_boundary"] = "open"
        open_current = self.save("open-current", DatasetSpec(open_current_document),
                                 self.current_values)
        with self.index.open(self.incoming_id) as incoming:
            open_vsv_document = incoming.spec.document
        open_vsv_document["conventions"]["temporal_boundary"] = "open"
        open_vsv_values = self.incoming_tau[[2, 0, 1]][:,[1, 2, 0]]
        open_vsv = self.save("open-vsv", DatasetSpec(open_vsv_document), open_vsv_values)
        with self.assertRaisesRegex(IndexError, "open temporal boundary"):
            plan_indexed_directed_current_chain(self.index, open_current, open_vsv,
                open_vsv, self.terms, momentum=self.selected_momentum,
                usedNe=2, source_times=[0], sink_times=[0], anchor_times=[0])

    def test_actual_basis_and_dirac_requirements_fail_before_output(self):
        other_basis_document = self.basis_spec.document
        other_basis_document["definition"] = dict(realization="same-labels-different-actual-basis")
        other_basis_id = self.save("other-basis", DatasetSpec(other_basis_document),
                                   self.vectors[[1, 2, 0]])
        with self.index.open(self.outgoing_id) as outgoing:
            wrong_basis_document = outgoing.spec.document
        wrong_basis_document["bases"]["sink_mode"]["data_id"] = other_basis_id
        wrong_basis_document["bases"]["source_mode"]["data_id"] = other_basis_id
        wrong_basis_document["parents"] = [self.third_gauge_id, other_basis_id]
        wrong_basis_id = self.save("wrong-basis-vsv", DatasetSpec(wrong_basis_document),
            self.outgoing_tau[[2, 0, 1]][:,[1, 2, 0]])
        with self.assertRaisesRegex(ValueError, "basis identity or ordered modes"):
            plan_indexed_directed_current_chain(self.index, self.current_id, self.incoming_id,
                wrong_basis_id, self.terms, momentum=self.selected_momentum, usedNe=2)

        with self.index.open(self.outgoing_id) as outgoing:
            missing_dirac = outgoing.spec.document
        del missing_dirac["definition"]["dirac"]
        missing_dirac_id = self.save("missing-dirac-vsv", DatasetSpec(missing_dirac),
            self.outgoing_tau[[2, 0, 1]][:,[1, 2, 0]])
        with self.assertRaisesRegex(ValueError, "explicit Dirac definition"):
            plan_indexed_directed_current_chain(self.index, self.current_id, self.incoming_id,
                missing_dirac_id, self.terms, momentum=self.selected_momentum, usedNe=2)

    def test_wrong_actual_parent_is_rejected_before_numeric_adaptation(self):
        with self.index.open(self.incoming_id) as incoming:
            changed = incoming.spec.document
        changed["definition"]["dirac"]["mass"] = 0.17
        wrong_incoming_id = self.save("wrong-incoming-parent", DatasetSpec(changed),
            self.incoming_tau[[2, 0, 1]][:,[1, 2, 0]])
        requests = workflow._chain_requests(self.index, self.current_id, self.incoming_id,
            self.outgoing_id, self.terms, self.selected_momentum, 2, None, None, None)
        planned = plan_dataset_graph(requests, self.index.observations())
        resolved_spec = planned["stages"]["measurement"]["spec"]
        with self.index.open(self.current_id) as current, \
                self.index.open(wrong_incoming_id) as incoming, \
                self.index.open(self.outgoing_id) as outgoing, \
                DatasetWriter(self.root / "wrong-actual-parent", resolved_spec,
                               provenance={}) as writer:
            with patch.object(workflow, "load_registered_directed_current",
                              side_effect=AssertionError("numeric adapter ran")):
                with self.assertRaisesRegex(ValueError, "actual input views"):
                    workflow.write_directed_current_chain_rows(current, incoming, outgoing, writer)

    def test_full_reader_writer_uses_selected_nonleading_momentum(self):
        output_path = self.root / "full-current-reader-writer"
        with self.index.open(self.current_id) as current, \
                self.index.open(self.incoming_id) as incoming, \
                self.index.open(self.outgoing_id) as outgoing:
            operator, _ = workflow._operator_snapshot(self.terms, self.selected_momentum,
                                                       current.spec, range(self.lt))
            document = directed_current_chain_document(current.spec, incoming.spec,
                outgoing.spec, [self.current_id, self.incoming_id, self.outgoing_id], operator)
            with DatasetWriter(output_path, DatasetSpec(document), provenance={}) as writer:
                workflow.write_directed_current_chain_rows(current, incoming, outgoing, writer)
                writer.publish()
        with DatasetReader(output_path) as result:
            for source_time in result.spec.coordinates("source_time"):
                for sink_time in result.spec.coordinates("sink_time"):
                    for anchor_time in result.spec.coordinates("anchor_time"):
                        np.testing.assert_allclose(
                            result.read(source_time=source_time, sink_time=sink_time,
                                        anchor_time=anchor_time),
                            self.expected(source_time, sink_time, anchor_time, ne=3),
                            rtol=2e-12, atol=2e-9)

if __name__ == "__main__":
    unittest.main()
