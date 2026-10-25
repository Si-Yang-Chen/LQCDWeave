"""Automatic production, coverage reuse and contraction against site oracles."""

from unittest.mock import patch
from dataclasses import replace

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import SourceReadRequest
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.production.gauge import stout_smear
from lqcdweave.storage.dataset import DatasetReader, register_npy
from lqcdweave.contraction.correlator import calculate, save_calculation_plan, load_calculation_plan
from lqcdweave.contraction.elemental_preparation import Elemental, prepare_elemental_sources
from production.elemental.test_elemental import reference_matrix, point_basis
from test_production_dataset import _ProductionDatasetFixture, axis


class ManagedElementalTests(_ProductionDatasetFixture):
    def setUp(self):
        super().setUp()
        self.gauge_reader = DatasetReader(self.root / "gauge", cfg="cfg")
        self.basis_reader = DatasetReader(self.root / "basis", cfg="cfg")
        self.addCleanup(self.gauge_reader.close)
        self.addCleanup(self.basis_reader.close)
        self.component = ElementalComponent.path((0,))
        self.table = ({"component": self.component, "path": (0,), "coefficient": .7+.2j},)

    def reader(self, **options):
        reader = Elemental(self.gauge_reader, self.basis_reader, self.table,
            {0: (1, -1, 0)}, directory=self.root / "managed", **options)
        self.addCleanup(reader.close)
        return reader

    def request(self, reader, time, sector="v2v"):
        return SourceReadRequest("el", "elemental", (self.component, 0, time, sector), (),
            reader.describe_block(self.component, 0, time, sector))

    def ensure(self, reader, times, sector="v2v"):
        prepare_elemental_sources({i: self.request(reader, t, sector) for i, t in enumerate(times)}, {"el": reader})

    def expected(self, time, *, gauge=None, left=None, right=None):
        vectors = self.vectors[time].reshape(3, -1)
        left = vectors if left is None else left
        right = vectors if right is None else right
        matrix = reference_matrix(self.gauge if gauge is None else gauge, (0,), time, (1, -1, 0), phase="anchor")
        return (.7+.2j) * (left.conj() @ matrix @ right.T)

    def test_metadata_does_not_load_or_create_outputs_and_only_requested_times_produced(self):
        with patch.object(self.gauge_reader, "read", side_effect=AssertionError("numeric read")):
            reader = self.reader()
            reader.describe_block(self.component, 0, 2, "v2v")
        self.assertFalse((self.root / "managed").exists())
        with patch.object(reader, "_produce", wraps=reader._produce) as produce:
            self.ensure(reader, (2, 2))
            self.assertEqual(produce.call_count, 1)
            self.assertEqual(produce.call_args.args[3], 2)
        np.testing.assert_allclose(reader.block(self.component, 0, 2, "v2v"), self.expected(2), atol=2e-12)
        with self.assertRaises(RuntimeError):
            reader.block(self.component, 0, 0, "v2v")

    def test_other_consumer_reuses_coverage_and_only_extends_missing_times(self):
        self.ensure(self.reader(), (0,))
        second = self.reader()
        with patch.object(second, "_produce", wraps=second._produce) as produce:
            self.ensure(second, (0, 2))
            self.assertEqual([c.args[3] for c in produce.call_args_list], [2])
        for time in (0, 2):
            np.testing.assert_allclose(second.block(self.component, 0, time, "v2v"), self.expected(time), atol=2e-12)
        third = self.reader()
        with patch.object(third, "_produce", side_effect=AssertionError("recomputed")):
            self.ensure(third, (2,))

    def test_accepted_coverage_is_reused_in_memory_but_missing_times_and_recipe_changes_are_admitted(self):
        reader = self.reader()
        self.ensure(reader, (0,))
        with patch('lqcdweave.contraction.elemental_preparation.plan_dataset_request',
                   side_effect=AssertionError('repeated persistent admission')):
            self.ensure(reader, (0,))
        with patch.object(reader, '_produce', wraps=reader._produce) as produce:
            self.ensure(reader, (0, 2))
            self.assertEqual([c.args[3] for c in produce.call_args_list], [2])
        with patch('lqcdweave.contraction.elemental_preparation.plan_dataset_request',
                   side_effect=AssertionError('repeated persistent admission')):
            self.ensure(reader, (2,))
            self.ensure(reader, (0, 2))
        for time in (0, 2):
            np.testing.assert_allclose(reader.block(self.component, 0, time, 'v2v'), self.expected(time), atol=2e-12)
        reader.stout = (1, .12)
        with patch.object(reader, '_produce', wraps=reader._produce) as produce:
            self.ensure(reader, (2,))
            self.assertEqual([c.args[3] for c in produce.call_args_list], [2])
        links = stout_smear(self.gauge.transpose(4, 0, 1, 2, 3, 5, 6), 1, .12).transpose(1, 2, 3, 4, 0, 5, 6)
        np.testing.assert_allclose(reader.block(self.component, 0, 2, 'v2v'), self.expected(2, gauge=links), atol=2e-12)

    def test_stout_recipe_cannot_reuse_unsmeared_product(self):
        self.ensure(self.reader(), (0,))
        reader = self.reader(stout=(1, .12))
        with patch.object(reader, "_produce", wraps=reader._produce) as produce:
            self.ensure(reader, (0,))
            self.assertEqual(produce.call_count, 1)
        links = stout_smear(self.gauge.transpose(4, 0, 1, 2, 3, 5, 6), 1, .12).transpose(1, 2, 3, 4, 0, 5, 6)
        np.testing.assert_allclose(reader.block(self.component, 0, 0, "v2v"), self.expected(0, gauge=links), atol=2e-12)

    def test_stout_is_shared_across_components_and_other_consumers(self):
        import lqcdweave.production.gauge as gauge_module
        first, second = ElementalComponent.path((0,)), ElementalComponent.path((3,))
        table = ({"component": first, "path": (0,), "coefficient": 1},
                 {"component": second, "path": (3,), "coefficient": 1})
        reader = Elemental(self.gauge_reader, self.basis_reader, table, {0: (0, 0, 0)},
            directory=self.root / "managed", stout=(1, .12))
        self.addCleanup(reader.close)
        requests = {i: SourceReadRequest("el", "elemental", (c, 0, 2, "v2v"), (),
                    reader.describe_block(c, 0, 2, "v2v")) for i, c in enumerate((first, second))}
        with patch.object(gauge_module, "stout_smear", wraps=gauge_module.stout_smear) as smear:
            prepare_elemental_sources(requests, {"el": reader})
            self.assertEqual(smear.call_count, 1)
        other = Elemental(self.gauge_reader, self.basis_reader, table, {0: (1, 0, 0)},
            directory=self.root / "managed", stout=(1, .12))
        self.addCleanup(other.close)
        with patch.object(gauge_module, "stout_smear", side_effect=AssertionError("re-smeared")):
            prepare_elemental_sources(requests, {"el": other})

    def test_interrupted_stage_resumes_saved_rows(self):
        reader = self.reader()
        original = reader._produce
        def interrupt(component, momentum, sector, time, backend):
            if time == 2:
                raise RuntimeError("interrupted")
            return original(component, momentum, sector, time, backend)
        with patch.object(reader, "_produce", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.ensure(reader, (0, 2))
        with patch.object(reader, "_produce", wraps=original) as produce:
            self.ensure(reader, (0, 2))
            self.assertEqual([c.args[3] for c in produce.call_args_list], [2])

    def test_missing_preparation_budget_stops_before_field_read(self):
        reader = self.reader()
        with patch.object(self.gauge_reader, "read", side_effect=AssertionError("numeric read")):
            with self.assertRaisesRegex(MemoryError, "preparation"):
                prepare_elemental_sources({0: self.request(reader, 0)}, {"el": reader}, host_limit=1)

    def test_all_nine_sectors_against_independent_site_matrices(self):
        # Register an actual D realization, preserving its complete group/pool
        # description through the native imported-noise provenance contract.
        ddoc = self.bspec.document
        ddoc["axes"][1]["coordinates"] = [0, 1, 2]
        ddoc["conventions"] = dict(representation="dilution")
        d = .3 * self.vectors.conj()
        np.save(self.root / "noise.npy", d[[1, 2, 0]])
        desc = dict(complete=True, normalization="raw", parent_basis_id="actual-full-complement", mode_groups=[list(range(6))],
            vector_counts=[3], branches=["qr"], covariance_factors=[.5],
            vector_labels=[[0, i] for i in range(3)], parent_counts=[6],
            time_pools=[[t, "D-pool-"+str(t)] for t in range(3)], algorithm="actual-test-qr")
        register_npy(self.root / "D", self.root / "noise.npy", DatasetSpec(ddoc), provenance={"generator": desc})
        dr = DatasetReader(self.root / "D", cfg="cfg")
        self.addCleanup(dr.close)
        coordinates = np.array([[[0, 0, 0], [1, 0, 0]]] * 3, dtype=np.int64)
        point_spec = self.spec("spatial_points", [axis("time", range(3)), axis("point", [0, 1]), axis("xyz", ["x", "y", "z"])],
            dtype="int64", conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
            definition=dict(realization=dict(design=dict(temporal_policy=1, lattice_size=list(self.sizes), num_points=2))))
        self.save("points", coordinates, point_spec)
        pr = DatasetReader(self.root / "points", cfg="cfg")
        self.addCleanup(pr.close)
        for weighted in (False, True):
            reader = Elemental(self.gauge_reader, {"V": self.basis_reader, "D": dr}, self.table, {0: (1, -1, 0)},
                points=pr, directory=self.root / "managed")._for_calculation(weighted)
            self.addCleanup(reader.close)
            request = {s: self.request(reader, 2, s) for s in reader.sectors}
            prepare_elemental_sources(request, {"el": reader})
            matrices = {"v": self.vectors[2].reshape(3, -1), "d": d[2].reshape(3, -1),
                        "p": point_basis(coordinates[2], self.sizes[:3]).T}
            for sector in reader.sectors:
                with self.subTest(weighted=weighted, sector=sector):
                    expected = self.expected(2, left=matrices[sector[0]], right=matrices[sector[2]])
                    if weighted:
                        if sector == "d2d":
                            weights = np.full((3, 3), 5.)
                            np.fill_diagonal(weights, 2.)
                            expected *= weights
                        else:
                            expected *= (2. if "d" in sector else 1.)
                        if "p" in sector:
                            n = np.prod(self.sizes[:3])
                            if sector == "p2p":
                                weights = np.full((2, 2), n*(n-1)/2)
                                np.fill_diagonal(weights, n/2)
                                expected = (expected.reshape(2, 3, 2, 3) * weights[:, None, :, None]).reshape(6, 6)
                            else:
                                expected *= n/2
                    actual = reader.block(self.component, 0, 2, sector)
                    np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=2e-12)
            if weighted:
                from lqcdweave.adapters.dilution_prefix import DilutionPrefix, DilutionPrefixElemental
                wrapped = DilutionPrefixElemental(reader, DilutionPrefix(2))
                prepare_elemental_sources({0: self.request(wrapped, 2, "d2d")}, {"el": wrapped})
                expected = self.expected(2, left=d[2, :2].reshape(2, -1), right=d[2, :2].reshape(2, -1))
                expected *= np.array([[3., 15.], [15., 3.]])
                np.testing.assert_allclose(wrapped.block(self.component, 0, 2, "d2d"), expected, atol=2e-12)

    def test_calculate_automatically_prepares_and_complete_archive_reuses(self):
        reader = self.reader()
        operator = Operator.from_terms("test", (OperatorTerm(15, self.component, 0, momentum_vector=(1, -1, 0)),))
        vertices = {"source": Meson(reader, operator, source=True, phase_convention="anchor", requested_sectors=("v2v",)),
                    "sink": Meson(reader, operator, phase_convention="anchor", requested_sectors=("v2v",))}
        rng = np.random.default_rng(731)
        prop = ArrayPropagator({"vsv": rng.normal(size=(3, 3, 4, 4, 3, 3)).astype(complex)}, time_convention="absolute", cfg="cfg",
            basis_endpoints={("vsv", side): (self.bid, (4, 1, 9)) for side in ("sink", "source")})
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ["source", "sink"], [None, "forward", "backward"])
        plan = calculate(diagram, vertex_map=vertices, propagator_map={"forward": prop, "backward": prop},
            configuration="cfg", time_axes={0: (0,), 1: (2,)}, execute=False, optimize="none")
        self.assertFalse((self.root / "managed").exists())
        preparation = plan.prediction["input_preparation"]
        self.assertEqual(preparation["stages"][0]["missing_times"], [0, 2])
        self.assertGreater(preparation["peak_host_bytes"], 0)
        with self.assertRaisesRegex(MemoryError, "elemental preparation"):
            calculate(diagram, vertex_map=vertices, propagator_map={"forward": prop, "backward": prop},
                configuration="cfg", time_axes={0: 0, 1: 2}, execute=False, optimize="none", host_memory_limit=100000)
        self.assertFalse((self.root / "managed").exists())
        with patch.object(reader, "_produce", wraps=reader._produce) as produce:
            result = plan.run()
            self.assertEqual({c.args[3] for c in produce.call_args_list}, {0, 2})
        # Literal spin/site contraction including the native source adjoint.
        from lqcdweave.operators.single_hadron.gamma import gamma
        g = gamma(15)
        source = gamma(8) @ g.conj().T @ gamma(8)
        a = self.expected(0).conj().T
        b = self.expected(2)
        forward = prop.block(2, 0, "vsv")
        backward = prop.block(0, 2, "vsv")
        expected = np.einsum("ab,ij,bcjk,cd,kl,dali->", g, b, forward, source, a, backward)
        np.testing.assert_allclose(np.asarray(result.value).squeeze(), expected, atol=2e-9)
        reused = calculate(diagram, vertex_map=vertices, propagator_map={"forward": prop, "backward": prop},
            configuration="cfg", time_axes={0: 0, 1: 2}, execute=False, optimize="none")
        self.assertEqual(reused.prediction["input_preparation"]["stages"][0]["action"], "reuse")
        self.assertEqual(reused.prediction["input_preparation"]["peak_host_bytes"], 0)
        weighted = calculate(diagram, vertex_map=vertices, propagator_map={"forward": prop, "backward": prop},
            configuration="cfg", time_axes={0: 0, 1: 2}, execute=False, optimize="none", elemental_weighted=True)
        from lqcdweave.contraction.expansion.api import input_readers
        managed = [r for r in input_readers(weighted.inputs).values() if isinstance(r, Elemental)]
        self.assertTrue(managed)
        self.assertTrue(all(r.elemental_weighted for r in managed))
        self.assertFalse(reader.elemental_weighted)
        np.testing.assert_allclose(np.asarray(weighted.run().value).squeeze(), expected, atol=2e-9)
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        coordinates, _ = describe_coordinates(time_axes={0: (0, 1), 1: (2,)})
        fixed = calculate(diagram, vertex_map=vertices, propagator_map={"forward": prop, "backward": prop},
            configuration="cfg", time_axes={0: 0, 1: 2}, execute=False, optimize="none")
        complete = complete_calculation(fixed, coordinates=coordinates, readers=input_readers(fixed.inputs))
        with patch.object(reader, "_produce", wraps=reader._produce) as produce:
            complete.run()
            self.assertEqual([c.args[3] for c in produce.call_args_list], [1])
        archive = self.root / "complete.json"
        save_calculation_plan(archive, complete)
        restored = load_calculation_plan(archive, configuration="cfg", readers=input_readers(fixed.inputs))
        with patch.object(reader, "_produce", side_effect=AssertionError("recomputed")):
            restored.run()
        fresh = self.reader()
        fresh.directory = self.root / "wrong-configuration"
        bindings = {role: fresh if isinstance(r, Elemental) else r
                    for role, r in input_readers(fixed.inputs).items()}
        wrong = replace(restored, options=dict(restored.options, configuration="other", readers=bindings))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            wrong.run()
        self.assertFalse(fresh.directory.exists())
