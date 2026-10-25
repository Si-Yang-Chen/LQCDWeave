"""End-to-end D estimators against literal parent-space traces."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator, PropagatorPairReader
from lqcdweave.contracts.dilution import DilutionGroup, DilutionSource
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.propagator import PropagatorEndpoint
from lqcdweave.production.perambulator.io import propagator_pair_spec, PropagatorReader
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation


def complex_array(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def component_pair_weights(source):
    """Literal same/distinct column probabilities, independent of expansion."""
    members = [(g, label) for g in source.groups for label in g.vector_labels]
    result = np.empty((len(members), len(members)), float)
    for a, (ga, la) in enumerate(members):
        for b, (gb, lb) in enumerate(members):
            if ga is not gb:
                result[a, b] = ga.first_moment_weight * gb.first_moment_weight
            elif ga.m == ga.n:
                result[a, b] = 1
            elif la == lb:
                result[a, b] = ga.n / ga.m
            elif ga.m > 1:
                result[a, b] = ga.n * (ga.n - 1) / (ga.m * (ga.m - 1))
            else:
                raise ValueError("distinct components are not estimable")
    return result


def literal_component_trace(a, c, d, source):
    """Sum individual saved-column projectors with their tuple probabilities."""
    weights = component_pair_weights(source)
    result = 0j
    for i in range(d.shape[1]):
        for j in range(d.shape[1]):
            result += weights[i, j] * (d[:, i].conj() @ a @ d[:, j]) * (d[:, j].conj() @ c @ d[:, i])
    return result


class DilutionCorrelatorTests(unittest.TestCase):
    cfg = "actual-dilution-cfg"

    def fixture(self, *, weighted=False, dagger=False, groups=((0, 1, 2, 3),), ranks=(2,),
                seed=827, pools=True, branch="qr", v_count=3):
        rng = np.random.default_rng(seed)
        parent_count = max(max(group) for group in groups) + 1
        columns, records = [], []
        for gi, (parents, rank) in enumerate(zip(groups, ranks)):
            q, _ = np.linalg.qr(complex_array(rng, (len(parents), len(parents))))
            block = np.zeros((parent_count, rank), complex)
            block[np.asarray(parents)] = q[:, :rank]
            columns.append(block)
            records.append(DilutionGroup(tuple(parents), tuple((gi, i) for i in range(rank)), branch))
        d = np.concatenate(columns, axis=1)
        nd = d.shape[1]
        source = DilutionSource("actual-D", "actual-parent", tuple(records),
            ((0, "actual-Q-pool"), (1, "other-time-Q-pool")) if pools else ())
        v = np.eye(parent_count, dtype=complex)[:, :v_count]
        a = complex_array(rng, (parent_count, parent_count))
        b = complex_array(rng, (v_count, v_count))
        r = complex_array(rng, (parent_count, parent_count))
        t = complex_array(rng, (parent_count, parent_count))
        spin_r, spin_t = complex_array(rng, (4, 4)), complex_array(rng, (4, 4))
        source_component, sink_component = ElementalComponent(("source", seed)), ElementalComponent(("sink", seed))
        components = (source_component, sink_component)
        dd = np.zeros((2, 1, 2, nd, nd), complex)
        vv = np.zeros((2, 1, 2, v_count, v_count), complex)
        dd[0, 0, 0], vv[1, 0, 1] = d.conj().T @ a @ d, b
        if weighted:
            dd *= component_pair_weights(source)[None, None, None, :, :]
        reader = ArrayElemental({"d2d": dd, "v2v": vv}, components, phase_convention="anchor",
            cfg=self.cfg, data_identity=f"EL-{seed}", elemental_weighted=weighted,
            basis_endpoints={("d2d", side): (source.data_id, tuple(range(nd))) for side in ("bar", "field")}
                | {("v2v", side): ("actual-V", tuple(range(v_count))) for side in ("bar", "field")},
            dilution_endpoints={("d2d", side): source for side in ("bar", "field")})
        dsv, vsd = np.zeros((2, 2, 4, 4, nd, v_count), complex), np.zeros((2, 2, 4, 4, v_count, nd), complex)
        dsv[1, 0] = np.einsum("ab,ij->abij", spin_r, d.conj().T @ r @ v)
        vsd[0, 1] = np.einsum("ab,ij->abij", spin_t, v.conj().T @ t @ d)
        propagator = ArrayPropagator({"dsv": dsv, "vsd": vsd}, time_convention="absolute",
            cfg=self.cfg, data_identity=f"PR-{seed}", basis_endpoints={
                ("dsv", "sink"): (source.data_id, tuple(range(nd))),
                ("dsv", "source"): ("actual-V", tuple(range(v_count))),
                ("vsd", "sink"): ("actual-V", tuple(range(v_count))),
                ("vsd", "source"): (source.data_id, tuple(range(nd)))})
        source_coefficient, sink_coefficient = 0.31 + 0.82j, -0.17 + 0.43j
        source_operator = Operator.from_terms("D source", (OperatorTerm(3, source_component, 0, source_coefficient),))
        sink_operator = Operator.from_terms("V sink", (OperatorTerm(14, sink_component, 0, sink_coefficient),))
        vertices = {"source": Meson(reader, source_operator, source=dagger, phase_convention="anchor", requested_sectors=("d2d",)),
                    "sink": Meson(reader, sink_operator, phase_convention="anchor", requested_sectors=("v2v",))}
        outgoing = Propagator("u", Tag(0, 0), Tag(1, 0))
        incoming = Propagator("d", Tag(1, 0), Tag(0, 0))
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ["source", "sink"], [None, outgoing, incoming])
        if dagger:
            a = a.conj().T
            source_gamma = gamma(8) @ gamma(3).conj().T @ gamma(8)
            source_coefficient = source_coefficient.conjugate()
        else:
            source_gamma = gamma(3)
        spin = np.trace(source_gamma @ spin_r @ gamma(14) @ spin_t)
        c = r @ v @ b @ v.conj().T @ t
        expected = literal_component_trace(a, c, d, source)
        expected *= source_coefficient * sink_coefficient * spin
        return dict(reader=reader, source=source, d=d, v=v, a=a, c=c, vertices=vertices,
                    r=r, t=t, b=b, spin_r=spin_r, spin_t=spin_t,
                    prefactor=source_coefficient * sink_coefficient * spin,
                    propagator=propagator, prop_map={outgoing: propagator, incoming: propagator},
                    diagram=diagram, expected=expected, weighted=weighted)

    def evaluate(self, fixture, execute=True):
        return calculate(fixture["diagram"], propagator_map=fixture["prop_map"],
            vertex_map=fixture["vertices"], configuration=self.cfg, optimize="none",
            groups_independent=fixture.get("groups_independent"),
            elemental_weighted=fixture["weighted"], execute=execute)

    def test_same_pool_second_moment_complex_spin_and_source_dagger(self):
        for dagger in (False, True):
            with self.subTest(dagger=dagger):
                raw, weighted = self.fixture(dagger=dagger), self.fixture(dagger=dagger, weighted=True)
                actual, preweighted = self.evaluate(raw).value, self.evaluate(weighted).value
                np.testing.assert_allclose(actual, raw["expected"], rtol=3e-12, atol=2e-10)
                np.testing.assert_allclose(preweighted, actual, rtol=3e-12, atol=2e-10)

    def test_noncontiguous_independent_groups_are_summed_with_cross_group_terms(self):
        groups = ((0, 2, 4, 6), (1, 3, 5, 7))
        raw = self.fixture(groups=groups, ranks=(2, 2), v_count=3)
        weighted = self.fixture(groups=groups, ranks=(2, 2), v_count=3, weighted=True)
        np.testing.assert_allclose(self.evaluate(raw).value, raw["expected"], rtol=4e-12, atol=5e-9)
        np.testing.assert_allclose(self.evaluate(weighted).value, raw["expected"], rtol=4e-12, atol=5e-9)

    def test_unequal_group_ratios_keep_each_component_probability(self):
        groups = ((0, 2, 4, 6), (1, 3, 5, 7, 8, 9))
        for rank in (3, 4):
            with self.subTest(second_group_rank=rank):
                raw = self.fixture(groups=groups, ranks=(2, rank))
                weighted = self.fixture(groups=groups, ranks=(2, rank), weighted=True)
                for context in (raw, weighted):
                    np.testing.assert_allclose(self.evaluate(context).value, raw["expected"], rtol=4e-12, atol=5e-9)
                # Distinct probabilities matter even when first moments match.
                self.assertEqual(component_pair_weights(raw["source"])[0, 1], 6)
                self.assertEqual(component_pair_weights(raw["source"])[2, 3], 30 / (rank * (rank - 1)))

    def test_rank_one_proven_diagonal_is_estimable_raw_and_preweighted(self):
        coefficient = 0.2 - 0.8j
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.fixture(ranks=(1,), weighted=weighted)
                reader = context["reader"]
                reader.sectors["d2d"][0, 0, 0] = coefficient * (4 if weighted else 1)
                reader.elemental_structures["d2d", reader.components[0], 0] = "diagonal"
                p = context["d"] @ context["d"].conj().T
                expected = coefficient * 4 * np.trace(context["c"] @ p) * context["prefactor"]
                np.testing.assert_allclose(self.evaluate(context).value, expected, rtol=3e-12, atol=2e-10)

    def test_previous_weight_scheme_is_rejected_before_numeric_read(self):
        context = self.fixture(weighted=True)
        context["reader"].dilution_weight_scheme = "inverse_projector-v1"
        with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")):
            with self.assertRaisesRegex(ValueError, "component_inclusion-v1"):
                self.evaluate(context)

    def test_complete_diagonal_d_factor_uses_same_component_weight(self):
        coefficient = 0.4 + 0.7j
        results = []
        for diagonal in (False, True):
            with self.subTest(diagonal=diagonal):
                context = self.fixture()
                context["reader"].sectors["d2d"][0, 0, 0] = coefficient * np.eye(2)
                if diagonal:
                    context["reader"].elemental_structures["d2d", context["reader"].components[0], 0] = "diagonal"
                p = context["d"] @ context["d"].conj().T
                expected = coefficient * 2 * np.trace(context["c"] @ p) * context["prefactor"]
                value = self.evaluate(context).value
                results.append(value)
                np.testing.assert_allclose(value, expected, rtol=3e-12, atol=2e-10)
        np.testing.assert_allclose(*results, rtol=3e-12, atol=2e-10)

    def test_mixed_vd_dv_vertices_use_independent_time_projectors(self):
        rng = np.random.default_rng(719)
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.fixture(weighted=weighted)
                d0, v, source = context["d"], context["v"], context["source"]
                d1, _ = np.linalg.qr(complex_array(rng, (4, 4)))
                d1 = d1[:, :2]
                b = complex_array(rng, (4, 4))
                vd, dv = np.zeros((2, 1, 2, 3, 2), complex), np.zeros((2, 1, 2, 2, 3), complex)
                vd[0, 0, 0] = v.conj().T @ context["a"] @ d0 * (2 if weighted else 1)
                dv[1, 0, 1] = d1.conj().T @ b @ v * (2 if weighted else 1)
                reader = ArrayElemental({"v2d": vd, "d2v": dv}, context["reader"].components,
                    phase_convention="anchor", cfg=self.cfg, data_identity="mixedVD-EL", elemental_weighted=weighted,
                    basis_endpoints={("v2d", "bar"): ("actual-V", (0, 1, 2)), ("v2d", "field"): (source.data_id, (0, 1)),
                        ("d2v", "bar"): (source.data_id, (0, 1)), ("d2v", "field"): ("actual-V", (0, 1, 2))},
                    dilution_endpoints={("v2d", "field"): source, ("d2v", "bar"): source})
                context["vertices"]["source"].reader = reader
                context["vertices"]["source"].requested_sectors = ("v2d",)
                context["vertices"]["sink"].reader = reader
                context["vertices"]["sink"].requested_sectors = ("d2v",)
                dsd, vsv = np.zeros((2, 2, 4, 4, 2, 2), complex), np.zeros((2, 2, 4, 4, 3, 3), complex)
                dsd[1, 0] = np.einsum("ab,ij->abij", context["spin_r"], d0.conj().T @ context["r"] @ d1)
                vsv[0, 1] = np.einsum("ab,ij->abij", context["spin_t"], v.conj().T @ context["t"] @ v)
                prop = ArrayPropagator({"dsd": dsd, "vsv": vsv}, time_convention="absolute", cfg=self.cfg,
                    data_identity="mixedVD-PR", basis_endpoints={
                        (sector, side): (source.data_id if sector == "dsd" else "actual-V", (0, 1) if sector == "dsd" else (0, 1, 2))
                        for sector in ("dsd", "vsv") for side in ("sink", "source")})
                context["reader"], context["propagator"] = reader, prop
                context["prop_map"] = {key: prop for key in context["prop_map"]}
                p0, p1 = d0 @ d0.conj().T, d1 @ d1.conj().T
                expected = 4 * context["prefactor"] * np.trace(
                    context["a"] @ p0 @ context["r"] @ p1 @ b @ v @ v.conj().T @ context["t"] @ v @ v.conj().T)
                np.testing.assert_allclose(self.evaluate(context).value, expected, rtol=3e-12, atol=2e-10)

    def test_invalid_same_pool_and_unknown_pool_fail_before_numeric_reads(self):
        for options, message in ((dict(ranks=(1,)), "not estimable"),
                                 (dict(ranks=(1,), branch="u1"), "legacy u1"),
                                 (dict(pools=False), "pool identity")):
            with self.subTest(options=options):
                context = self.fixture(**options)
                with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")), \
                     patch.object(context["propagator"], "block", side_effect=AssertionError("numeric propagator read")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.evaluate(context)

    def test_weight_tag_identity_and_partial_group_fail_before_numeric_reads(self):
        context = self.fixture(weighted=True)
        context["weighted"] = False
        with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")):
            with self.assertRaisesRegex(ValueError, "elemental_weighted"):
                self.evaluate(context)
        context = self.fixture()
        metadata = context["propagator"].describe_block
        def wrong_identity(*args, **kwargs):
            description = metadata(*args, **kwargs)
            bindings = dict(description.axis_bindings)
            for axis, binding in tuple(bindings.items()):
                if binding[0] == "actual-D":
                    bindings[axis] = ("different-actual-D", binding[1])
            return replace(description, axis_bindings=bindings)
        with patch.object(context["propagator"], "describe_block", side_effect=wrong_identity), \
             patch.object(context["propagator"], "block", side_effect=AssertionError("numeric propagator read")):
            with self.assertRaisesRegex(ValueError, "identity|binding|basis"):
                self.evaluate(context)
        context = self.fixture()
        reader = context["reader"]
        with patch.object(reader, "block", side_effect=AssertionError("numeric elemental read")):
            with self.assertRaisesRegex(ValueError, "partial dilution group"):
                calculate(context["diagram"], propagator_map=context["prop_map"], vertex_map=context["vertices"],
                    configuration=self.cfg, usedNe={"D": 1}, optimize="none")
        with self.assertRaisesRegex(ValueError, "partial dilution group|vector-axis extent"):
            ArrayElemental({"d2d": reader.sectors["d2d"][..., :1, :1]}, reader.components,
                phase_convention="anchor", cfg=self.cfg,
                basis_endpoints={("d2d", side): ("actual-D", (0,)) for side in ("bar", "field")},
                dilution_endpoints=reader.dilution_endpoints)

    def test_complete_plan_replay_and_changed_pool_rejected_before_numeric_read(self):
        context = self.fixture(dagger=True)
        selected = self.evaluate(context, execute=False)
        complete = complete_calculation(selected, readers=selected.inputs.loader.input_readers)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actual-D-plan.json.gz"
            save_calculation_plan(path, complete)
            replayed = calculate(plan_archive=path, configuration=self.cfg,
                propagator_map=context["prop_map"], vertex_map=context["reader"])
            np.testing.assert_allclose(replayed.value, context["expected"], rtol=3e-12, atol=2e-10)
            scheme = context["reader"].dilution_weight_scheme
            context["reader"].dilution_weight_scheme = "inverse_projector-v1"
            with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")):
                with self.assertRaisesRegex(ValueError, "dilution|conventions"):
                    calculate(plan_archive=path, configuration=self.cfg,
                        propagator_map=context["prop_map"], vertex_map=context["reader"])
            context["reader"].dilution_weight_scheme = scheme
            changed = replace(context["source"], time_pools=((0, "changed-actual-pool"), (1, "other-time-Q-pool")))
            context["reader"].dilution_endpoints = {(sector, side): changed
                for sector, side in context["reader"].dilution_endpoints}
            with patch.object(context["reader"], "block", side_effect=AssertionError("numeric elemental read")), \
                 patch.object(context["propagator"], "block", side_effect=AssertionError("numeric propagator read")):
                with self.assertRaisesRegex(ValueError, "dilution|pool|source|conventions"):
                    calculate(plan_archive=path, configuration=self.cfg,
                        propagator_map=context["prop_map"], vertex_map=context["reader"])

    def test_independent_endpoint_pools_use_first_moments_without_reconnection(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.fixture(weighted=weighted)
                rng = np.random.default_rng(953)
                right, _ = np.linalg.qr(complex_array(rng, (4, 4)))
                right = right[:, :2]
                source = replace(context["source"], data_id="actual-independent-D",
                    time_pools=((0, "independent-Q-pool"), (1, "independent-other-time")))
                reader, propagator = context["reader"], context["propagator"]
                reader.dilution_endpoints["d2d", "field"] = source
                reader.basis_endpoints["d2d", "field"] = (source.data_id, (0, 1))
                reader.sectors["d2d"][0, 0, 0] = context["d"].conj().T @ context["a"] @ right
                if weighted:
                    reader.sectors["d2d"][0, 0, 0] *= 4.0
                propagator.basis_endpoints["dsv", "sink"] = (source.data_id, (0, 1))
                propagator.sectors["dsv"][1, 0] = np.einsum("ab,ij->abij", context["spin_r"],
                    right.conj().T @ context["r"] @ context["v"])
                pl, pr = context["d"] @ context["d"].conj().T, right @ right.conj().T
                expected = 4 * context["prefactor"] * np.trace(context["a"] @ pr @ context["c"] @ pl)
                with self.assertRaisesRegex(ValueError, "groups_independent"):
                    self.evaluate(context)
                context["groups_independent"] = True
                np.testing.assert_allclose(self.evaluate(context).value, expected, rtol=3e-12, atol=2e-10)

    def test_d_only_two_time_pools_match_literal_component_sum(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = self.fixture(weighted=weighted, v_count=4)
                rng = np.random.default_rng(1183)
                d1, _ = np.linalg.qr(complex_array(rng, (4, 4)))
                d1 = d1[:, :2]
                source, d0 = context["source"], context["d"]
                reader = context["reader"]
                dd = reader.sectors["d2d"].copy()
                dd[1, 0, 1] = d1.conj().T @ context["b"] @ d1
                if weighted:
                    dd[1, 0, 1] *= component_pair_weights(source)
                new_reader = ArrayElemental({"d2d": dd}, reader.components, cfg=self.cfg,
                    phase_convention="anchor", elemental_weighted=weighted, data_identity="DonlyEL",
                    basis_endpoints={("d2d", side): (source.data_id, (0, 1)) for side in ("bar", "field")},
                    dilution_endpoints={("d2d", side): source for side in ("bar", "field")})
                for vertex in context["vertices"].values():
                    vertex.reader = new_reader
                    vertex.requested_sectors = ("d2d",)
                dsd = np.zeros((2, 2, 4, 4, 2, 2), complex)
                dsd[1, 0] = np.einsum("ab,ij->abij", context["spin_r"], d0.conj().T @ context["r"] @ d1)
                dsd[0, 1] = np.einsum("ab,ij->abij", context["spin_t"], d1.conj().T @ context["t"] @ d0)
                prop = ArrayPropagator({"dsd": dsd}, time_convention="absolute", cfg=self.cfg,
                    data_identity="DonlyPR", basis_endpoints={("dsd", side): (source.data_id, (0, 1))
                        for side in ("sink", "source")})
                context["reader"], context["propagator"] = new_reader, prop
                context["prop_map"] = {key: prop for key in context["prop_map"]}
                a, b, r, t = (context[key] for key in ("a", "b", "r", "t"))
                ad, bd, rd, td = d0.conj().T @ a @ d0, d1.conj().T @ b @ d1, d0.conj().T @ r @ d1, d1.conj().T @ t @ d0
                weights = component_pair_weights(source)
                expected = sum(ad[i,j] * rd[j,k] * bd[k,l] * td[l,i] * weights[i,j] * weights[k,l]
                    for i in range(2) for j in range(2) for k in range(2) for l in range(2)) * context["prefactor"]
                np.testing.assert_allclose(self.evaluate(context).value, expected, rtol=4e-12, atol=5e-10)

    def test_same_pool_disconnected_loops_are_joined_only_in_spatial_indices(self):
        rng = np.random.default_rng(217)
        base = self.fixture()
        d, v, source = (base[key] for key in ("d", "v", "source"))
        operators = base["vertices"]
        components = base["reader"].components
        aa = (base["a"], complex_array(rng, (4, 4)))
        tt = (base["t"], complex_array(rng, (4, 4)))
        spins = (base["spin_t"], complex_array(rng, (4, 4)))
        coefficients = (0.31 + 0.82j, -0.17 + 0.43j)
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                dv = np.zeros((2, 1, 2, 2, 3), complex)
                for i, a in enumerate(aa):
                    dv[i, 0, 0] = d.conj().T @ a @ v * (2 if weighted else 1)
                reader = ArrayElemental({"d2v": dv}, components, cfg=self.cfg, phase_convention="anchor",
                    elemental_weighted=weighted, data_identity="disconnectedEL",
                    basis_endpoints={("d2v", "bar"): (source.data_id, (0, 1)),
                                     ("d2v", "field"): ("actual-V", (0, 1, 2))},
                    dilution_endpoints={("d2v", "bar"): source})
                vertices = {name: Meson(reader, old.operator, phase_convention="anchor", requested_sectors=("d2v",))
                            for name, old in operators.items()}
                loops, readers = [], []
                for i in range(2):
                    values = np.zeros((2, 2, 4, 4, 3, 2), complex)
                    values[0, 0] = np.einsum("ab,ij->abij", spins[i], v.conj().T @ tt[i] @ d)
                    readers.append(ArrayPropagator({"vsd": values}, time_convention="absolute", cfg=self.cfg,
                        data_identity=f"loopPR-{i}", basis_endpoints={("vsd", "sink"): ("actual-V", (0, 1, 2)),
                        ("vsd", "source"): (source.data_id, (0, 1))}))
                    loops.append(Propagator("u" if i == 0 else "d", Tag(i, 0), Tag(i, 0)))
                diagram = Diagram(QuarkDiagram([[1, 0], [0, 2]]), [0, 0], ["source", "sink"], [None, *loops])
                value = calculate(diagram, propagator_map=dict(zip(loops, readers)), vertex_map=vertices,
                    configuration=self.cfg, optimize="none", elemental_weighted=weighted).value
                c0, c1 = (a @ v @ v.conj().T @ t for a, t in zip(aa, tt))
                weights = component_pair_weights(source)
                expected = sum(weights[i,j] * (d[:,i].conj() @ c0 @ d[:,i]) * (d[:,j].conj() @ c1 @ d[:,j])
                               for i in range(2) for j in range(2))
                expected *= coefficients[0] * coefficients[1] * np.trace(gamma(3) @ spins[0]) * np.trace(gamma(14) @ spins[1])
                np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-10)

    def test_saved_representative_pair_reader_feeds_calculate_and_gamma5_reverse(self):
        context = self.fixture()
        source, d, v = (context[key] for key in ("source", "d", "v"))
        v_endpoint = PropagatorEndpoint("V", "actual-V", tuple(range(3)))
        d_endpoint = PropagatorEndpoint("D", source.data_id, tuple(range(2)), metadata=dict(
            vector_labels=[list(label) for label in source.vector_labels],
            mode_groups=[list(group.parent_modes) for group in source.groups],
            parent_basis_id=source.parent_basis_id, realization_id="actual-realization",
            noise_pool_id="actual-Q-pool", normalization="raw"))
        spec = propagator_pair_spec("VSD", configuration=dict(ensemble="bounded", configuration=self.cfg, lattice=[1, 1, 1, 2]),
            sink=v_endpoint, source=d_endpoint, time_pairs=((1, 0),), gauge_id="actual-gauge", dirac={"action": "test"},
            gamma5=gamma(8), spin_basis="degrand_rossi", gamma5_hermitian=True)
        with TemporaryDirectory() as directory:
            with DatasetWriter(directory, spec, provenance={"test": "PairReader calculate"}) as writer:
                writer.write(0, context["propagator"].sectors["vsd"][0, 1])
                writer.publish()
            with DatasetReader(directory) as saved:
                pair_reader = PropagatorPairReader(PropagatorReader([saved]))
                context["prop_map"] = {key: pair_reader for key in context["prop_map"]}
                c = context["t"].conj().T @ v @ context["b"] @ v.conj().T @ context["t"]
                spin_r = gamma(8) @ context["spin_t"].conj().T @ gamma(8)
                spin = np.trace(gamma(3) @ spin_r @ gamma(14) @ context["spin_t"])
                expected = literal_component_trace(context["a"], c, d, source) * (0.31 + 0.82j) * (-0.17 + 0.43j) * spin
                np.testing.assert_allclose(self.evaluate(context).value, expected, rtol=4e-12, atol=5e-10)

    def test_registered_point_diagonal_current_combines_d_and_point_weight_switch(self):
        rng = np.random.default_rng(347)
        spatial_s, spatial_t = complex_array(rng, (3, 6)), complex_array(rng, (6, 4))
        spin_s, spin_t = complex_array(rng, (4, 4)), complex_array(rng, (4, 4))
        k = complex_array(rng, (3, 3))
        point_basis = np.eye(6, dtype=complex)[:, :3]
        configuration = dict(ensemble="bounded", configuration=self.cfg, lattice=[2, 1, 1, 2])
        def axis(name, values):
            return dict(name=name, coordinates=list(values))
        points_spec = DatasetSpec(dict(schema=1, kind="spatial_points", configuration=configuration,
            axes=[axis("time", range(2)), axis("point", (0,)), axis("xyz", ("x", "y", "z"))],
            dtype="int64", bases={}, parents=[], definition={},
            conventions=dict(representation="coordinate_delta", normalization="unit_delta")))
        for weighted in (False, True):
            with self.subTest(weighted=weighted), TemporaryDirectory() as directory:
                context = self.fixture(weighted=weighted)
                root = Path(directory)
                with DatasetWriter(root / "points", points_spec, provenance={"test": "actual point pool"}) as writer:
                    for time in range(2):
                        writer.write(time, np.asarray([[0, 0, 0]], dtype=np.int64))
                    writer.publish()
                with DatasetReader(root / "points") as points:
                    pp_spec = DatasetSpec(dict(schema=1, kind="elemental_p2p_row_sparse", configuration=configuration,
                        axes=[axis("time", range(2)), axis("component", (dict(path=[]),)), axis("momentum", ((0, 0, 0),)),
                              axis("point_left", (0,)), axis("color_left", range(3)), axis("color_right", range(3))],
                        dtype="complex128", bases=dict(point_left=dict(data_id=points.data_id, indices=[0])),
                        parents=[points.data_id, "actual-gauge"], definition=dict(gauge_id="actual-gauge", path_boundary="periodic",
                            point_right_id=points.data_id, point_right_selection="same_ordered_prefix_as_point_left"),
                        conventions=dict(normalization="unscaled", phase="anchor", point_pair_layout="left-point-row-slots-v1",
                            elemental_weighted=weighted, elemental_structures=[dict(component=0, momentum=0, structure="point_diagonal")])))
                    with DatasetWriter(root / "PP", pp_spec, provenance={"test": "P diagonal current"}) as writer:
                        for time in range(2):
                            writer.write(time, (k * (2 if weighted else 1)).reshape(1, 1, 1, 3, 3))
                        writer.publish()
                    with DatasetReader(root / "PP") as pp_stored:
                        component = ElementalComponent.path(())
                        pool = SamplingPoolSpec("actual-P-pool", 2, 1)
                        pp_reader = DatasetElemental({"p2p": pp_stored}, point_dataset=points,
                            sampling_groups={pool.group_id: pool}, sampling_pool_ids={
                                (component, 0, time, side): pool.group_id for time in range(2) for side in ("bar", "field")})
                        current_operator = Operator.from_terms("P current", (OperatorTerm(2, component, 0, 0.7 - 0.2j),))
                        context["vertices"]["current"] = Meson(pp_reader, current_operator,
                            phase_convention="anchor", requested_sectors=("p2p",))
                        d, v = context["d"], context["v"]
                        vsp = np.zeros((2, 2, 4, 4, 3, 1, 3), complex)
                        psd = np.zeros((2, 2, 4, 4, 1, 3, 2), complex)
                        vsp[1, 1] = np.einsum("ab,ij->abij", spin_s, spatial_s @ point_basis).reshape(4, 4, 3, 1, 3)
                        psd[0, 1] = np.einsum("ab,ij->abij", spin_t, point_basis.conj().T @ spatial_t @ d).reshape(4, 4, 1, 3, 2)
                        dsv = context["propagator"].sectors["dsv"]
                        bindings = {("dsv", "sink"): ("actual-D", (0, 1)), ("dsv", "source"): ("actual-V", (0, 1, 2)),
                            ("vsp", "sink"): ("actual-V", (0, 1, 2)), ("vsp", "source"): (points.data_id, (0,)),
                            ("psd", "sink"): (points.data_id, (0,)), ("psd", "source"): ("actual-D", (0, 1))}
                        prop = ArrayPropagator({"dsv": dsv, "vsp": vsp, "psd": psd}, time_convention="absolute",
                            cfg=self.cfg, data_identity="D-P-combinedPR", basis_endpoints=bindings)
                        forward = Propagator("u", Tag(0, 0), Tag(1, 0))
                        middle = Propagator("u", Tag(1, 0), Tag(2, 0))
                        backward = Propagator("d", Tag(2, 0), Tag(0, 0))
                        diagram = Diagram(QuarkDiagram([[0, 1, 0], [0, 0, 2], [3, 0, 0]]), [0, 1, 1],
                            ["source", "current", "sink"], [None, forward, middle, backward])
                        result = calculate(diagram, propagator_map={symbol: prop for symbol in (forward, middle, backward)},
                            vertex_map=context["vertices"], configuration=self.cfg, optimize="none", elemental_weighted=weighted,
                            groups_independent=True)
                        c = context["r"] @ v @ context["b"] @ spatial_s @ point_basis @ k @ point_basis.conj().T @ spatial_t
                        expected = 2 * literal_component_trace(context["a"], c, d, context["source"])
                        expected *= (0.31 + 0.82j) * (-0.17 + 0.43j) * (0.7 - 0.2j)
                        expected *= np.trace(gamma(3) @ context["spin_r"] @ gamma(14) @ spin_s @ gamma(2) @ spin_t)
                        np.testing.assert_allclose(result.value, expected, rtol=4e-12, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
