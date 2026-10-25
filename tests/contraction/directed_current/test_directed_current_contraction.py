"""Adapted from old test/test_current_v2v_contraction.py."""
import numpy as np
import unittest

from lqcdweave.contraction.directed_current import (
    CURRENT_V2V_CONTRACTION_SCHEMA,
    CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
    contract_directed_current_pair_v2v,
    contract_directed_current_v2v,
)
from lqcdweave.operators.single_hadron.current import (
    ConservedVectorCurrent,
    ConservedVectorInsertion,
)
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contracts.current import build_current_raw_contract


_checker = unittest.TestCase()


class ExistingVSV:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []
        self.load_calls = []

    def get(self, source_time, sink_time):
        self.calls.append((source_time, sink_time))
        return self.blocks[(source_time, sink_time)]

    def load(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))
        raise AssertionError("the current bridge must not load or generate propagators")


def _fixture(boundary="periodic"):
    values = np.zeros((8, 3, 1, 2, 2), dtype=np.complex128)
    values[6, 2, 0] = np.array([[1 + 1j, 2], [3, 4 - 2j]])
    values[7, 0, 0] = np.array([[5, 6 + 1j], [7 - 3j, 8]])
    raw = {"v2v": values}
    contract = build_current_raw_contract(
        raw,
        boundary=boundary,
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )
    return raw, contract


def _block(seed, sink_ne, source_ne):
    values = np.arange(4 * 4 * sink_ne * source_ne, dtype=np.float64)
    values = values.reshape(4, 4, sink_ne, source_ne)
    return (seed + values + 1j * (seed * 2 - values)).astype(np.complex128)


def _check_termwise_temporal_vsv_contraction_preserves_endpoints_axes_and_weights():
    raw, contract = _fixture()
    terms = ConservedVectorCurrent(wilson_r=1.25).terms[6:8]
    outgoing_blocks = {
        (0, 1): _block(2, 3, 2),
        (2, 1): _block(3, 3, 2),
    }
    incoming_blocks = {
        (0, 2): _block(5, 2, 4),
        (0, 0): _block(7, 2, 4),
    }
    outgoing = ExistingVSV(outgoing_blocks)
    incoming = ExistingVSV(incoming_blocks)

    result = contract_directed_current_v2v(
        terms,
        raw,
        contract,
        incoming,
        outgoing,
        source_time=0,
        sink_time=1,
        anchor_time=2,
        current_source_ne=1,
        current_sink_ne=2,
    )

    forward_spin = 1.25 * np.eye(4) - np.asarray(gamma(8))
    backward_spin = 1.25 * np.eye(4) + np.asarray(gamma(8))
    forward_vertex = forward_spin[:, :, None, None] * raw["v2v"][6, 2, 0, :2, :1]
    backward_vertex = backward_spin[:, :, None, None] * raw["v2v"][7, 0, 0, :2, :1]
    expected = -0.5 * np.einsum(
        "afAi,bfji,bcjC->acAC",
        outgoing_blocks[(0, 1)][..., :1],
        forward_vertex,
        incoming_blocks[(0, 2)][..., :2, :],
    ) + 0.5 * np.einsum(
        "afAi,bfji,bcjC->acAC",
        outgoing_blocks[(2, 1)][..., :1],
        backward_vertex,
        incoming_blocks[(0, 0)][..., :2, :],
    )

    assert result["schema"] == CURRENT_V2V_CONTRACTION_SCHEMA
    assert result["axes"] == (
        "external_sink_spin",
        "external_source_spin",
        "external_sink_ne",
        "external_source_ne",
    )
    assert result["value"].shape == (4, 4, 3, 4)
    np.testing.assert_allclose(result["value"], expected, rtol=1e-13, atol=1e-13)
    assert outgoing.calls == [(0, 1), (2, 1)]
    assert incoming.calls == [(0, 2), (0, 0)]
    assert outgoing.load_calls == incoming.load_calls == []
    assert result["terms"][0]["raw"]["direction"] == 6
    assert result["terms"][1]["raw"]["direction"] == 7
    assert result["terms"][1]["raw"]["time"] == 0


def _check_pair_contraction_matches_two_vertex_loop_and_preserves_term_endpoints():
    raw, contract = _fixture()
    terms = ConservedVectorCurrent(wilson_r=1.25).terms[6:8]
    blocks = {}
    for source_time in range(3):
        for sink_time in range(3):
            blocks[(source_time, sink_time)] = _block(10 * source_time + sink_time + 1, 2, 2)
    vsv = ExistingVSV(blocks)

    result = contract_directed_current_pair_v2v(
        terms,
        raw,
        contract,
        terms,
        raw,
        contract,
        vsv,
        first_anchor_time=2,
        second_anchor_time=0,
        first_field_ne=2,
        first_bar_ne=2,
        second_field_ne=2,
        second_bar_ne=2,
    )

    expected = 0j
    expected_calls = []
    for first_term in terms:
        first_endpoints = {
            "bar_time": 2 if first_term.link == "forward" else 0,
            "field_time": 0 if first_term.link == "forward" else 2,
        }
        first_spin = (
            1.25 * np.eye(4) - np.asarray(gamma(8))
            if first_term.link == "forward"
            else 1.25 * np.eye(4) + np.asarray(gamma(8))
        )
        first_direction = 6 if first_term.link == "forward" else 7
        first_raw_time = first_endpoints["bar_time"]
        first_vertex = first_spin[:, :, None, None] * raw["v2v"][first_direction, first_raw_time, 0][None, None]
        for second_term in terms:
            second_endpoints = {
                "bar_time": 0 if second_term.link == "forward" else 1,
                "field_time": 1 if second_term.link == "forward" else 0,
            }
            second_spin = (
                1.25 * np.eye(4) - np.asarray(gamma(8))
                if second_term.link == "forward"
                else 1.25 * np.eye(4) + np.asarray(gamma(8))
            )
            second_direction = 6 if second_term.link == "forward" else 7
            second_raw_time = second_endpoints["bar_time"]
            second_vertex = second_spin[:, :, None, None] * raw["v2v"][second_direction, second_raw_time, 0][None, None]
            first_to_second_key = (
                first_endpoints["field_time"],
                second_endpoints["bar_time"],
            )
            second_to_first_key = (
                second_endpoints["field_time"],
                first_endpoints["bar_time"],
            )
            expected_calls.extend([first_to_second_key, second_to_first_key])
            pair = np.einsum(
                "bfji,ackl,afki,bcjl->",
                blocks[first_to_second_key],
                blocks[second_to_first_key],
                first_vertex,
                second_vertex,
            )
            expected += (
                first_term.coefficient
                * first_term.normalization
                * second_term.coefficient
                * second_term.normalization
                * pair
            )

    assert result["schema"] == CURRENT_V2V_PAIR_CONTRACTION_SCHEMA
    assert result["axes"] == ()
    np.testing.assert_allclose(result["value"], expected, rtol=1e-13, atol=1e-13)
    assert vsv.calls == expected_calls
    assert vsv.load_calls == []
    assert len(result["term_pairs"]) == 4
    assert result["term_pairs"][0]["first_raw"]["direction"] == 6
    assert result["term_pairs"][3]["second_raw"]["direction"] == 7
    assert "no implicit Wick sign" in result["operation"]


def _check_pair_contraction_accepts_distinct_raw_inputs_and_rejects_incompatibility():
    first_raw, first_contract = _fixture()
    second_raw, second_contract = _fixture()
    second_raw = {"v2v": second_raw["v2v"] * (2 - 1j)}
    second_contract = build_current_raw_contract(
        second_raw,
        boundary="periodic",
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )
    terms = ConservedVectorCurrent().terms[6:8]
    blocks = {(source, sink): _block(source * 4 + sink + 1, 2, 2) for source in range(3) for sink in range(3)}
    result = contract_directed_current_pair_v2v(
        terms,
        first_raw,
        first_contract,
        terms,
        second_raw,
        second_contract,
        ExistingVSV(blocks),
        first_anchor_time=2,
        second_anchor_time=2,
        first_field_ne=1,
        first_bar_ne=2,
        second_field_ne=2,
        second_bar_ne=1,
    )
    baseline = contract_directed_current_pair_v2v(
        terms,
        first_raw,
        first_contract,
        terms,
        first_raw,
        first_contract,
        ExistingVSV(blocks),
        first_anchor_time=2,
        second_anchor_time=2,
        first_field_ne=1,
        first_bar_ne=2,
        second_field_ne=2,
        second_bar_ne=1,
    )
    assert result["raw_cache_identities"]["first"] == result["raw_cache_identities"]["second"]
    assert result["value"] != baseline["value"]
    assert np.isfinite(result["value"])

    open_contract = build_current_raw_contract(
        second_raw,
        boundary="open",
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )
    with _checker.assertRaisesRegex(ValueError, expected_regex="boundaries"):
        contract_directed_current_pair_v2v(
            terms,
            first_raw,
            first_contract,
            terms,
            second_raw,
            open_contract,
            ExistingVSV({}),
            first_anchor_time=0,
            second_anchor_time=1,
            first_field_ne=1,
            first_bar_ne=1,
            second_field_ne=1,
            second_bar_ne=1,
        )


def _check_pair_contraction_rejects_short_vsv_ne_and_empty_terms():
    raw, contract = _fixture()
    terms = ConservedVectorCurrent().terms[6:8]
    with _checker.assertRaisesRegex(ValueError, expected_regex="both Current term collections"):
        contract_directed_current_pair_v2v(
            (),
            raw,
            contract,
            terms,
            raw,
            contract,
            ExistingVSV({}),
            first_anchor_time=0,
            second_anchor_time=1,
            first_field_ne=1,
            first_bar_ne=1,
            second_field_ne=1,
            second_bar_ne=1,
        )

    blocks = {
        (1, 1): _block(1, 1, 1),
        (2, 0): _block(2, 1, 1),
    }
    with _checker.assertRaisesRegex(ValueError, expected_regex="Ne extents"):
        contract_directed_current_pair_v2v(
            terms[:1],
            raw,
            contract,
            terms[:1],
            raw,
            contract,
            ExistingVSV(blocks),
            first_anchor_time=0,
            second_anchor_time=1,
            first_field_ne=2,
            first_bar_ne=1,
            second_field_ne=1,
            second_bar_ne=1,
        )


def _check_open_boundary_rejects_crossing_before_vsv_access():
    raw, contract = _fixture("open")
    incoming = ExistingVSV({})
    outgoing = ExistingVSV({})
    with _checker.assertRaisesRegex(IndexError, expected_regex="open temporal boundary"):
        contract_directed_current_v2v(
            ConservedVectorCurrent().terms[6:8],
            raw,
            contract,
            incoming,
            outgoing,
            source_time=0,
            sink_time=1,
            anchor_time=2,
            current_source_ne=1,
            current_sink_ne=1,
        )
    assert incoming.calls == outgoing.calls == []


def _check_vsv_bridge_rejects_short_ne_noncomplex_blocks_and_bad_times():
    raw, contract = _fixture()
    forward = ConservedVectorCurrent().terms[6:7]
    incoming = ExistingVSV({(0, 1): _block(1, 1, 2)})
    outgoing = ExistingVSV({(2, 0): np.ones((4, 4, 1, 1), dtype=float)})
    with _checker.assertRaisesRegex(TypeError, expected_regex="complex"):
        contract_directed_current_v2v(
            forward,
            raw,
            contract,
            incoming,
            outgoing,
            source_time=0,
            sink_time=0,
            anchor_time=1,
            current_source_ne=1,
            current_sink_ne=1,
        )

    with _checker.assertRaisesRegex(ValueError, expected_regex="source_time"):
        contract_directed_current_v2v(
            forward,
            raw,
            contract,
            ExistingVSV({}),
            ExistingVSV({}),
            source_time=-1,
            sink_time=0,
            anchor_time=1,
            current_source_ne=1,
            current_sink_ne=1,
        )

    outgoing = ExistingVSV({(2, 0): _block(1, 1, 1)})
    with _checker.assertRaisesRegex(ValueError, expected_regex="bar Ne"):
        contract_directed_current_v2v(
            forward,
            raw,
            contract,
            ExistingVSV({(0, 1): _block(1, 1, 2)}),
            outgoing,
            source_time=0,
            sink_time=0,
            anchor_time=1,
            current_source_ne=1,
            current_sink_ne=2,
        )




class DirectedCurrentContractionTests(unittest.TestCase):
    def test_dedicated_insertion_binds_momentum_for_both_contractions(self):
        base, _ = _fixture()
        raw = {"v2v": np.concatenate((base["v2v"], (2 + 1j) * base["v2v"]), axis=2)}
        contract = build_current_raw_contract(raw, boundary="periodic",
                                              available_ne=2, used_ne=2,
                                              momentum_count=2)
        first = ConservedVectorInsertion(3, momentum_index=1, wilson_r=1.25)
        second = ConservedVectorInsertion(3, momentum_index=0, wilson_r=1.25)
        blocks = {(source, sink): _block(source * 4 + sink + 1, 2, 2)
                  for source in range(3) for sink in range(3)}
        kwargs = dict(first_anchor_time=2, second_anchor_time=0,
                      first_field_ne=2, first_bar_ne=2,
                      second_field_ne=2, second_bar_ne=2)
        actual = contract_directed_current_pair_v2v(
            first, raw, contract, second, raw, contract,
            ExistingVSV(blocks), **kwargs)
        explicit = contract_directed_current_pair_v2v(
            first.terms, raw, contract, second.terms, raw, contract,
            ExistingVSV(blocks), first_momentum=1, second_momentum=0, **kwargs)
        np.testing.assert_allclose(actual["value"], explicit["value"])
        self.assertEqual(actual["momenta"], {"first": 1, "second": 0})

        single_kwargs = dict(source_time=0, sink_time=1, anchor_time=2,
                             current_source_ne=2, current_sink_ne=2)
        actual_single = contract_directed_current_v2v(
            first, raw, contract, ExistingVSV(blocks), ExistingVSV(blocks),
            **single_kwargs)
        explicit_single = contract_directed_current_v2v(
            first.terms, raw, contract, ExistingVSV(blocks), ExistingVSV(blocks),
            momentum=1, **single_kwargs)
        np.testing.assert_allclose(actual_single["value"], explicit_single["value"])
        self.assertEqual(actual_single["momentum"], 1)

        untouched = ExistingVSV(blocks)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            contract_directed_current_pair_v2v(
                first, raw, contract, second, raw, contract, untouched,
                first_momentum=0, **kwargs)
        self.assertEqual(untouched.calls, [])

    def test_termwise_temporal_vsv_contraction_preserves_endpoints_axes_and_weights(self):
        _check_termwise_temporal_vsv_contraction_preserves_endpoints_axes_and_weights()

    def test_pair_contraction_matches_two_vertex_loop_and_preserves_term_endpoints(self):
        _check_pair_contraction_matches_two_vertex_loop_and_preserves_term_endpoints()

    def test_pair_contraction_accepts_distinct_raw_inputs_and_rejects_incompatibility(self):
        _check_pair_contraction_accepts_distinct_raw_inputs_and_rejects_incompatibility()

    def test_pair_contraction_rejects_short_vsv_ne_and_empty_terms(self):
        _check_pair_contraction_rejects_short_vsv_ne_and_empty_terms()

    def test_open_boundary_rejects_crossing_before_vsv_access(self):
        _check_open_boundary_rejects_crossing_before_vsv_access()

    def test_vsv_bridge_rejects_short_ne_noncomplex_blocks_and_bad_times(self):
        _check_vsv_bridge_rejects_short_ne_noncomplex_blocks_and_bad_times()

