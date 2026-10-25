"""Finite dilution-subset estimates using the archived weak-field inputs.

The archive is source-time first, then cyclic relative sink time, followed
by sink/source spin and sink/source eigenmode.  Its 20-mode perambulator is
projected by selecting the same actual input eigenvector columns used below;
the stored physical clover/stout solves are not recomputed.  The fixture has
no complete historical production manifest, so this is an archive-level
integration/reference check, not a new validation of those Dirac solves.

The random experiments uniformly select two of three or four of five columns
from a fixed complete basis. All possibilities are enumerated, with no Monte
Carlo approximation and no production weight function in the reference.
"""

from dataclasses import replace
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.dilution import DilutionGroup, DilutionSource
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.contraction.correlator import calculate
from production.elemental.test_elemental import reference_matrix


FIXTURE = Path(__file__).resolve().parent / "data" / "weak_field"
CFG = "archived-weak-field"
PARENT_ID = "archived-weak-field/input-eigenvectors/ordered-20"
PARENT_MODES = (4, 5, 6)
FIXED_V_MODES = (1, 4, 8, 13)
FULL = (0, 1, 2)
SUBSETS = tuple(combinations(FULL, 2))
LOCAL = ElementalComponent.path(())
SPATIAL = ElementalComponent.path((0,))


class ActualBasis:
    basis_orthogonal = True

    def __init__(self, values, identity, labels, source=None):
        self.values, self.basis_id, self.mode_labels = values, identity, tuple(labels)
        self.dilution_source = source
        self.Ne = values.shape[1]

    def load(self, key):
        return self.values


@lru_cache(maxsize=1)
def actual_inputs():
    raw = np.load(FIXTURE / "weak_field.npy", mmap_mode="r")
    gauge = raw.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6)
    vectors = np.load(FIXTURE / "weak_field.eigenvector.input.npy", mmap_mode="r")
    propagator = np.load(FIXTURE / "weak_field.perambulator.npy", mmap_mode="r")
    return gauge, vectors, propagator


@lru_cache(maxsize=2)
def fixed_rotation(parent_modes=PARENT_MODES):
    # One complete nontrivial unitary mixing, shared at every physical time.
    # Randomness only constructs this fixed coordinate transformation; the
    # finite column-selection reference enumerates every possible subset.
    rng = np.random.default_rng(5137)
    n = len(parent_modes)
    complete, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return complete


def actual_coordinates(columns, rotated, parent_modes=PARENT_MODES):
    if rotated:
        return parent_modes, fixed_rotation(parent_modes)[:, columns]
    return tuple(parent_modes[i] for i in columns), np.eye(len(columns), dtype=complex)


@lru_cache(maxsize=64)
def produced(columns, weighted, kind, role="shared", rotated=False, parent_modes=PARENT_MODES):
    gauge, vectors, _ = actual_inputs()
    mode_ids, mixing = actual_coordinates(columns, rotated, parent_modes)
    identity = f"{PARENT_ID}/parents-{parent_modes}/columns-{columns}/{role}/rotation-{rotated}"
    pool = f"fixed-column-selection-{parent_modes}/{columns}/{role}/rotation-{rotated}"
    source = DilutionSource(identity, PARENT_ID,
        (DilutionGroup(parent_modes, tuple((0, i) for i in columns),
                       "exact" if len(columns) == len(parent_modes) and not rotated else "qr"),),
        tuple((time, pool) for time in range(8)))
    values = np.einsum("ti...,ia->ta...", np.take(vectors, mode_ids, axis=1), mixing)
    d = ActualBasis(values, identity, source.ordered_indices, source)
    fixed = ActualBasis(np.take(vectors, FIXED_V_MODES, axis=1), PARENT_ID, FIXED_V_MODES)
    component, path, momentum = ((SPATIAL, (0,), (1, 0, 0))
                                 if kind == "general" else (LOCAL, (), (0, 0, 0)))
    generator = ElementalGenerator(gauge, None, (momentum,),
        [dict(component=component, path=path, coefficient=1)],
        vector_bases={"D": d, "V": fixed}, blocks=("d2d", "v2v"),
        lattice_size=(4, 4, 4, 8), phase_convention="anchor",
        elemental_weighted=weighted, ne_block_size=2).load(CFG)
    # Copy borrowed numerical buffers at every physical time.
    values = {sector: [] for sector in generator.blocks}
    for time in range(8):
        for sector, value in generator.calc_path_blocks(time).items():
            values[sector].append(value.copy())
    reader = ArrayElemental({sector: np.stack(value, axis=2) for sector, value in values.items()},
        generator.components, momentum_list=generator.momenta, cfg=CFG,
        data_identity=f"{identity}/elemental/{kind}/{weighted}", phase_convention="anchor",
        elemental_weighted=weighted, elemental_structures=generator.elemental_structures,
        basis_endpoints=generator.basis_endpoints, dilution_endpoints=generator.dilution_endpoints)
    return reader, source, mode_ids


@lru_cache(maxsize=64)
def projected_propagator(source_columns, sink_columns, both_d, independent=False, rotated=False):
    _, _, archive = actual_inputs()
    source_role, sink_role = (("source", "sink") if independent else ("shared", "shared"))
    _, source, source_modes = produced(source_columns, False, "local", source_role, rotated)
    _, sink, sink_modes = produced(sink_columns, False, "local", sink_role, rotated)
    _, source_mix = actual_coordinates(source_columns, rotated)
    _, sink_mix = actual_coordinates(sink_columns, rotated)
    v_endpoint = (PARENT_ID, FIXED_V_MODES)
    source_endpoint = (source.data_id, source.ordered_indices)
    sink_endpoint = (sink.data_id, sink.ordered_indices)

    def select(sink_indices, source_indices, sink_transform, source_transform):
        selected = np.take(np.take(archive, sink_indices, axis=4), source_indices, axis=5)
        return np.einsum("ia,tsuvij,jb->tsuvab", sink_transform.conj(), selected, source_transform)

    if both_d:
        # Separate readers retain which actual sample owns each endpoint.
        outgoing = ArrayPropagator({"dsd": select(sink_modes, source_modes, sink_mix, source_mix)},
            time_convention="relative", cfg=CFG, data_identity=f"actual-archive/out/{source.data_id}/{sink.data_id}",
            basis_endpoints={("dsd", "sink"): sink_endpoint, ("dsd", "source"): source_endpoint})
        incoming = ArrayPropagator({"dsd": select(source_modes, sink_modes, source_mix, sink_mix)},
            time_convention="relative", cfg=CFG, data_identity=f"actual-archive/in/{source.data_id}/{sink.data_id}",
            basis_endpoints={("dsd", "sink"): source_endpoint, ("dsd", "source"): sink_endpoint})
    else:
        fixed_mix = np.eye(len(FIXED_V_MODES), dtype=complex)
        outgoing = ArrayPropagator({"vsd": select(FIXED_V_MODES, source_modes, fixed_mix, source_mix)},
            time_convention="relative", cfg=CFG, data_identity=f"actual-archive/out-V/{source.data_id}",
            basis_endpoints={("vsd", "sink"): v_endpoint, ("vsd", "source"): source_endpoint})
        incoming = ArrayPropagator({"dsv": select(source_modes, FIXED_V_MODES, source_mix, fixed_mix)},
            time_convention="relative", cfg=CFG, data_identity=f"actual-archive/in-V/{source.data_id}",
            basis_endpoints={("dsv", "sink"): source_endpoint, ("dsv", "source"): v_endpoint})
    return outgoing, incoming


def evaluate(source_columns, sink_columns=FULL, *, weighted=False, both_d=False, independent=False, rotated=False):
    source_role, sink_role = (("source", "sink") if independent else ("shared", "shared"))
    source_reader, _, _ = produced(source_columns, weighted, "local" if both_d else "general", source_role, rotated)
    sink_reader, _, _ = produced(sink_columns, weighted, "local", sink_role, rotated)
    source_component = LOCAL if both_d else SPATIAL
    source_operator = Operator.from_terms("weak-field source", (OperatorTerm(0, source_component, 0, 0.37 + 0.19j),))
    sink_operator = Operator.from_terms("weak-field sink", (OperatorTerm(0, LOCAL, 0, -0.41 + 0.23j),))
    vertices = {
        "source": Meson(source_reader, source_operator, phase_convention="anchor", requested_sectors=("d2d",)),
        "sink": Meson(sink_reader, sink_operator, phase_convention="anchor",
                      requested_sectors=("d2d" if both_d else "v2v",))}
    outgoing, incoming = Propagator("u", Tag(0, 0), Tag(1, 0)), Propagator("d", Tag(1, 0), Tag(0, 0))
    readers = projected_propagator(source_columns, sink_columns, both_d, independent, rotated)
    diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ["source", "sink"], [None, outgoing, incoming])
    result = calculate(diagram, vertex_map=vertices, propagator_map=dict(zip((outgoing, incoming), readers)),
        configuration=CFG, time_axes={1: tuple(range(1, 8))}, optimize="none",
        elemental_weighted=weighted, groups_independent=True if independent else None)
    return np.asarray(result.value).reshape(7)


def report(label, full, samples):
    average = np.mean(samples, axis=0)
    error = np.linalg.norm(average - full) / np.linalg.norm(full)
    print(f"\nweak_field {label}; source=0, sinks=1..7; actual parent modes={PARENT_MODES}")
    print(f"full three columns: {full.tolist()}")
    for subset, value in zip(SUBSETS, samples):
        print(f"subset {subset}: {np.asarray(value).tolist()}")
    print(f"mean: {average.tolist()}; relative error: {error:.17g}")
    return average


def two_current_context(columns, *, weighted=False, general=False, rotated=False,
                        parent_modes=PARENT_MODES):
    """One connected four-point loop with both D/D currents at physical t=1."""
    normal, _, _ = produced(FULL, weighted, "local")
    current, source, _ = produced(columns, weighted, "general" if general else "local",
                                 rotated=rotated, parent_modes=parent_modes)
    component = SPATIAL if general else LOCAL
    base = LocalVectorCurrent().operator(3)
    operator = Operator.from_terms("weak-field temporal-vector insertion",
                                  (replace(base.terms[0], component=component),))
    vertices = {
        "source": Meson(normal, Operator.from_terms("source", (OperatorTerm(15, LOCAL, 0),)),
                        source=True, phase_convention="anchor", requested_sectors=("v2v",)),
        "current-1": Meson(current, operator, phase_convention="anchor", requested_sectors=("d2d",)),
        "current-2": Meson(current, operator, phase_convention="anchor", requested_sectors=("d2d",)),
        "sink": Meson(normal, Operator.from_terms("sink", (OperatorTerm(15, LOCAL, 0),)),
                      phase_convention="anchor", requested_sectors=("v2v",)),
    }
    _, _, archive = actual_inputs()
    coordinates = {"v": (FIXED_V_MODES, np.eye(len(FIXED_V_MODES))),
                   "d": actual_coordinates(columns, rotated, parent_modes)}
    identities = {"v": (PARENT_ID, FIXED_V_MODES),
                  "d": (source.data_id, source.ordered_indices)}
    sectors, endpoints = {}, {}
    for sector in ("vsv", "vsd", "dsv", "dsd"):
        left, left_mix = coordinates[sector[0]]
        right, right_mix = coordinates[sector[2]]
        block = np.take(np.take(archive, left, axis=4), right, axis=5)
        sectors[sector] = np.einsum("ia,tsuvij,jb->tsuvab", left_mix.conj(), block, right_mix)
        endpoints[sector, "sink"], endpoints[sector, "source"] = (
            identities[sector[0]], identities[sector[2]])
    propagator = ArrayPropagator(sectors, time_convention="relative", cfg=CFG,
        data_identity=f"actual-archive/two-currents/{source.data_id}", basis_endpoints=endpoints)
    symbols = tuple(Propagator("u", Tag(i, 0), Tag((i + 1) % 4, 0)) for i in range(4))
    diagram = Diagram(QuarkDiagram([[0, 1, 0, 0], [0, 0, 2, 0],
                                   [0, 0, 0, 3], [4, 0, 0, 0]]),
                      [0, 1, 1, 2], list(vertices), [None, *symbols])
    return dict(diagram=diagram, vertex_map=vertices,
                propagator_map={symbol: propagator for symbol in symbols},
                current_reader=current, normal_reader=normal, propagator=propagator,
                elemental_weighted=weighted)


def evaluate_two_currents(context):
    return np.asarray(calculate(context["diagram"], vertex_map=context["vertex_map"],
        propagator_map=context["propagator_map"], configuration=CFG,
        time_axes={2: tuple(range(2, 8))}, optimize="none",
        elemental_weighted=context["elemental_weighted"]).value).reshape(6)


def complete_two_current_trace(parent_modes=PARENT_MODES, *, general=False, rotated=False,
                               include_four_distinct=False):
    """Independent full spin-mode matrix trace from the archived inputs."""
    gauge, vectors, archive = actual_inputs()
    mixing = fixed_rotation(parent_modes) if rotated else np.eye(len(parent_modes))
    transport = reference_matrix(gauge, (0,), 1, (1, 0, 0), phase="anchor") if general else None

    def vertex(time, modes, spin, spatial=None):
        values = np.take(vectors[time], modes, axis=0).reshape(len(modes), -1)
        if modes == parent_modes:
            values = mixing.T @ values
        matrix = values.conj() @ (values.T if spatial is None else spatial @ values.T)
        return np.kron(spin, matrix), matrix

    def prop(sink_time, sink_modes, source_time, source_modes):
        value = archive[source_time, (sink_time - source_time) % 8]
        value = np.take(np.take(value, sink_modes, axis=2), source_modes, axis=3)
        left = mixing if sink_modes == parent_modes else np.eye(len(sink_modes))
        right = mixing if source_modes == parent_modes else np.eye(len(source_modes))
        value = np.einsum("ia,uvij,jb->uvab", left.conj(), value, right)
        return value.transpose(0, 2, 1, 3).reshape(4 * len(sink_modes), 4 * len(source_modes))

    source, _ = vertex(0, FIXED_V_MODES, gamma(8) @ gamma(15).conj().T @ gamma(8))
    current, spatial = vertex(1, parent_modes, gamma(8), transport)
    middle = prop(1, parent_modes, 1, parent_modes)
    forward = prop(1, parent_modes, 0, FIXED_V_MODES)
    full, distinct = [], []
    n = len(parent_modes)
    for time in range(2, 8):
        sink, _ = vertex(time, FIXED_V_MODES, gamma(15))
        bridge = forward @ source @ prop(0, FIXED_V_MODES, time, FIXED_V_MODES) @ sink \
            @ prop(time, FIXED_V_MODES, 1, parent_modes)
        full.append(np.trace(current @ middle @ current @ bridge))
        if include_four_distinct:
            # Keep all four mode labels explicit; this reference does not use
            # any inclusion coefficient or graph/equality expansion helper.
            terms = np.einsum("ab,cd,pq,qbrc,rs,sdpa->abcd", spatial, spatial,
                gamma(8), middle.reshape(4, n, 4, n), gamma(8), bridge.reshape(4, n, 4, n))
            np.testing.assert_allclose(terms.sum(), full[-1], rtol=5e-12, atol=1e-15)
            distinct.append(sum(terms[indices] for indices in product(range(n), repeat=4)
                                if len(set(indices)) == 4))
    return (np.asarray(full), np.asarray(distinct)) if include_four_distinct else np.asarray(full)


class WeakFieldDilutionSubsetTests(unittest.TestCase):
    def test_actual_two_general_dd_currents_five_choose_four_subset_mean(self):
        parents = (4, 5, 6, 7, 8)
        columns = tuple(range(5))
        subsets = tuple(combinations(columns, 4))
        full = evaluate_two_currents(two_current_context(columns, general=True, parent_modes=parents))
        for rotated in (False, True):
            reference, four_distinct = complete_two_current_trace(
                parents, general=True, rotated=rotated, include_four_distinct=True)
            np.testing.assert_allclose(full, reference, rtol=5e-12, atol=1e-15)
            self.assertGreater(np.linalg.norm(four_distinct), 1e-12)
            samples = []
            for selected in subsets:
                with self.subTest(columns=selected, rotated=rotated):
                    context = two_current_context(selected, general=True,
                                                  rotated=rotated, parent_modes=parents)
                    self.assertEqual(context["current_reader"].elemental_structures.get(
                        ("d2d", SPATIAL, 0), "general"), "general")
                    raw = evaluate_two_currents(context)
                    preweighted = evaluate_two_currents(two_current_context(selected, weighted=True,
                        general=True, rotated=rotated, parent_modes=parents))
                    np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-15)
                    samples.append(raw)
            average = np.mean(samples, axis=0)
            error = np.linalg.norm(average - full) / np.linalg.norm(full)
            print(f"\nweak_field general same-time double-current 5 choose 4; parents={parents}; "
                  f"rotation={rotated}; source=0, currents=1, sinks=2..7; "
                  f"full={full.tolist()}; subsets={list(zip(subsets, [v.tolist() for v in samples]))}; "
                  f"mean={average.tolist()}; relative error={error:.17g}; "
                  f"four-distinct contribution={four_distinct.tolist()}")
            np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-15)

    def test_actual_two_diagonal_dd_currents_at_same_time_subset_mean(self):
        full = evaluate_two_currents(two_current_context(FULL))
        np.testing.assert_allclose(full, complete_two_current_trace(), rtol=5e-12, atol=1e-15)
        for rotated in (False, True):
            samples = []
            for columns in SUBSETS:
                with self.subTest(columns=columns, rotated=rotated):
                    raw = evaluate_two_currents(two_current_context(columns, rotated=rotated))
                    preweighted = evaluate_two_currents(two_current_context(
                        columns, weighted=True, rotated=rotated))
                    np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-15)
                    samples.append(raw)
            average = np.mean(samples, axis=0)
            error = np.linalg.norm(average - full) / np.linalg.norm(full)
            print(f"\nweak_field two D/D currents at t=1; source=0, sinks=2..7; rotation={rotated}; "
                  f"full={full.tolist()}; subsets={[value.tolist() for value in samples]}; "
                  f"mean={average.tolist()}; relative error={error:.17g}")
            np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-15)

    def test_actual_two_general_dd_currents_at_same_time_reject_rank_two(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                context = two_current_context(SUBSETS[0], weighted=weighted, general=True)
                # Two unconstrained D/D insertions can need three distinct
                # labels from this population of three. Rank two has zero
                # inclusion probability; reject before reading numerical data.
                with patch.object(context["current_reader"], "block",
                                  side_effect=AssertionError("numeric current read")), \
                     patch.object(context["normal_reader"], "block",
                                  side_effect=AssertionError("numeric normal read")), \
                     patch.object(context["propagator"], "block",
                                  side_effect=AssertionError("numeric propagator read")):
                    with self.assertRaisesRegex(ValueError, "support|sample|points|rank|distinct"):
                        evaluate_two_currents(context)

    def test_actual_nonzero_momentum_source_dd_fixed_vv_subset_mean(self):
        full = evaluate(FULL)
        reader, _, _ = produced(SUBSETS[0], False, "general")
        self.assertEqual(reader.elemental_structures.get(("d2d", SPATIAL, 0), "general"), "general")
        # Nonzero off-diagonal source terms prevent a diagonal-only test from
        # accidentally passing an incorrect equal/distinct component weight.
        source = reader.sectors["d2d"][0, 0, 0]
        self.assertGreater(abs(source[0, 1]) + abs(source[1, 0]), 1e-8)
        samples = []
        for columns in SUBSETS:
            raw, preweighted = evaluate(columns), evaluate(columns, weighted=True)
            np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-18)
            samples.append(raw)
        average = report("nonzero momentum/path DD + fixed VV", full, samples)
        np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-18)

    def test_actual_fixed_complex_qr_mixture_source_subset_mean(self):
        # The baseline is still the actual original three Laplace columns;
        # complete Q is merely another orthonormal basis of that same subspace.
        full = evaluate(FULL)
        np.testing.assert_allclose(evaluate(FULL, rotated=True), full, rtol=5e-12, atol=1e-18)
        samples = []
        for columns in SUBSETS:
            raw = evaluate(columns, rotated=True)
            preweighted = evaluate(columns, weighted=True, rotated=True)
            np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-18)
            samples.append(raw)
        average = report("fixed complex QR mixture DD + fixed VV", full, samples)
        np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-18)

    def test_actual_diagonal_dd_reused_subset_across_physical_times(self):
        full = evaluate(FULL, FULL, both_d=True)
        samples = []
        for columns in SUBSETS:
            reader, source, _ = produced(columns, False, "local")
            self.assertEqual(reader.elemental_structures["d2d", LOCAL, 0], "diagonal")
            self.assertEqual(len({pool for _, pool in source.time_pools}), 1)
            raw = evaluate(columns, columns, both_d=True)
            preweighted = evaluate(columns, columns, weighted=True, both_d=True)
            np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-18)
            samples.append(raw)
        average = report("diagonal DD + DD, one subset/pool across time", full, samples)
        np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-18)

    def test_actual_diagonal_dd_independent_endpoint_subset_control(self):
        full = evaluate(FULL, FULL, both_d=True, independent=True)
        samples = []
        for left, right in product(SUBSETS, repeat=2):
            raw = evaluate(left, right, both_d=True, independent=True)
            preweighted = evaluate(left, right, both_d=True, independent=True, weighted=True)
            np.testing.assert_allclose(preweighted, raw, rtol=5e-12, atol=1e-18)
            samples.append(raw)
        average = np.mean(samples, axis=0)
        error = np.linalg.norm(average - full) / np.linalg.norm(full)
        print(f"\nweak_field independent endpoint selections: full={full.tolist()}; "
              f"nine-subset mean={average.tolist()}; relative error={error:.17g}")
        np.testing.assert_allclose(average, full, rtol=5e-12, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
