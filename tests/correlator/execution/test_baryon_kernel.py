"""Independent numerical checks for the rank-three baryon contraction kernel.

The oracle below deliberately uses explicit loops.  It therefore checks the
kernel's source-slot permutation and axis convention without reproducing its
einsum construction.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np
import pytest

from lqcdweave.contraction import ContractionPlanCache, TensorSpec, execute_plan
from lqcdweave.contraction import (
    BaryonTerm,
    BaryonVertex,
    compile_baryon_terms,
    execute_baryon_terms,
)


_PERMUTATIONS = tuple(permutations(range(3)))
_SOURCE_SPIN = (1, 2, 3)
_SOURCE_MODE = (2, 3, 1)
_SINK_SPIN = (3, 1, 2)
_SINK_MODE = (1, 2, 3)


def _noncontiguous_complex(shape: tuple[int, ...], offset: float) -> np.ndarray:
    """Make a complex array view with a deliberately non-contiguous last stride."""

    size = int(np.prod(shape, dtype=int))
    storage = (
        np.arange(2 * size, dtype=np.float64).reshape(shape + (2,))
        + 1j * (offset + np.arange(2 * size, dtype=np.float64)).reshape(shape + (2,))
    ).astype(np.complex128)
    value = storage[..., 0]
    assert not value.flags.c_contiguous
    return value


def _bindings(offset: float = 0.0, source_slots=(0, 1, 2)) -> dict[str, np.ndarray]:
    return {
        "S": _noncontiguous_complex(_SOURCE_SPIN, offset + 1),
        "E": _noncontiguous_complex(_SOURCE_MODE, offset + 11),
        "R": _noncontiguous_complex(_SINK_SPIN, offset + 21),
        "F": _noncontiguous_complex(_SINK_MODE, offset + 31),
        "P0": _noncontiguous_complex(
            (_SINK_SPIN[0], _SOURCE_SPIN[source_slots[0]], _SINK_MODE[0], _SOURCE_MODE[source_slots[0]]),
            offset + 41,
        ),
        "P1": _noncontiguous_complex(
            (_SINK_SPIN[1], _SOURCE_SPIN[source_slots[1]], _SINK_MODE[1], _SOURCE_MODE[source_slots[1]]),
            offset + 51,
        ),
        "P2": _noncontiguous_complex(
            (_SINK_SPIN[2], _SOURCE_SPIN[source_slots[2]], _SINK_MODE[2], _SOURCE_MODE[source_slots[2]]),
            offset + 61,
        ),
    }


def _term(source_slots=(0, 1, 2), coefficient=1) -> BaryonTerm:
    return BaryonTerm(
        source=BaryonVertex("S", "E"),
        sink=BaryonVertex("R", "F"),
        propagators=("P0", "P1", "P2"),
        source_slots=source_slots,
        coefficient=coefficient,
    )


def _oracle(bindings: dict[str, np.ndarray], source_slots, coefficient=1) -> complex:
    """Evaluate c*S[a,b,c]*E[i,j,k]*R[d,e,f]*F[l,m,n]*P0*P1*P2."""

    source_spin = bindings["S"]
    source_mode = bindings["E"]
    sink_spin = bindings["R"]
    sink_mode = bindings["F"]
    propagators = (bindings["P0"], bindings["P1"], bindings["P2"])
    total = 0j
    for source_indices in product(*(range(size) for size in _SOURCE_SPIN)):
        for source_modes in product(*(range(size) for size in _SOURCE_MODE)):
            source_value = source_spin[source_indices] * source_mode[source_modes]
            for sink_indices in product(*(range(size) for size in _SINK_SPIN)):
                for sink_modes in product(*(range(size) for size in _SINK_MODE)):
                    value = source_value * sink_spin[sink_indices] * sink_mode[sink_modes]
                    for q, slot in enumerate(source_slots):
                        value *= propagators[q][
                            sink_indices[q],
                            source_indices[slot],
                            sink_modes[q],
                            source_modes[slot],
                        ]
                    total += value
    return coefficient * total


def _specs(bindings: dict[str, np.ndarray]) -> dict[str, TensorSpec]:
    return {key: TensorSpec(value.shape, value.dtype) for key, value in bindings.items()}


@pytest.mark.parametrize("source_slots", _PERMUTATIONS)
def test_all_source_slot_permutations_match_independent_loop_oracle(source_slots) -> None:
    bindings = _bindings(source_slots=source_slots)
    result = execute_baryon_terms(bindings, (_term(source_slots),))

    assert len(result.values) == 1
    np.testing.assert_allclose(result.values[0], _oracle(bindings, source_slots), rtol=1e-12, atol=1e-10)


def test_heterogeneous_noncontiguous_complex_axes_and_complex_coefficient() -> None:
    bindings = _bindings(7.0, source_slots=(2, 0, 1))
    coefficient = -0.75 + 1.25j
    term = _term((2, 0, 1), coefficient)
    result = execute_baryon_terms(bindings, (term,))

    expected = _oracle(bindings, (2, 0, 1), coefficient)
    np.testing.assert_allclose(result.values[0], expected, rtol=1e-12, atol=1e-10)
    assert np.asarray(result.values[0]).shape == ()


def test_explicit_terms_remain_independent_outputs_without_implicit_sum_or_conjugation() -> None:
    bindings = _bindings(13.0, source_slots=(1, 2, 0))
    terms = (
        _term((1, 2, 0), coefficient=1.0 + 0.5j),
        _term((1, 2, 0), coefficient=-2.0 + 0.25j),
    )
    result = execute_baryon_terms(bindings, terms)

    assert len(result.values) == 2
    for value, term in zip(result.values, terms):
        np.testing.assert_allclose(
            value,
            _oracle(bindings, term.source_slots, term.coefficient),
            rtol=1e-12,
            atol=1e-10,
        )
    assert not np.allclose(result.values[0], result.values[1])
    assert result.stats.shared_operations > 0
    assert not np.shares_memory(result.values[0], result.values[1])
    second_before = np.array(result.values[1], copy=True)
    result.values[0][...] = 0.0
    np.testing.assert_array_equal(result.values[1], second_before)
    bindings["S"][...] = 0.0
    np.testing.assert_array_equal(result.values[1], second_before)


def test_compile_and_direct_execution_share_the_same_scalar_contract() -> None:
    bindings = _bindings(19.0, source_slots=(2, 0, 1))
    terms = (_term((2, 0, 1), coefficient=0.25 - 0.5j),)
    plan = compile_baryon_terms(_specs(bindings), terms)
    compiled_result = execute_plan(plan, bindings)
    direct_result = execute_baryon_terms(bindings, terms)

    np.testing.assert_allclose(compiled_result.values[0], direct_result.values[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        direct_result.values[0],
        _oracle(bindings, (2, 0, 1), 0.25 - 0.5j),
        rtol=1e-12,
        atol=1e-10,
    )


def test_plan_cache_rebinds_fresh_values_and_outputs_do_not_alias_inputs() -> None:
    first = _bindings(23.0, source_slots=(1, 0, 2))
    second = _bindings(127.0, source_slots=(1, 0, 2))
    terms = (_term((1, 0, 2)),)
    cache = ContractionPlanCache(max_entries=2)

    first_result = execute_baryon_terms(first, terms, plan_cache=cache)
    second_result = execute_baryon_terms(second, terms, plan_cache=cache)

    np.testing.assert_allclose(first_result.values[0], _oracle(first, (1, 0, 2)), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(second_result.values[0], _oracle(second, (1, 0, 2)), rtol=1e-12, atol=1e-10)
    assert cache.hits == 1
    assert not np.allclose(first_result.values[0], second_result.values[0])
    for output in (first_result.values[0], second_result.values[0]):
        output_array = np.asarray(output)
        for input_array in (*first.values(), *second.values()):
            assert not np.shares_memory(output_array, input_array)


@pytest.mark.parametrize(
    "bad_source_slots",
    ((0, 0, 1), (0, 1), (0, 1, 3), (0, 1, -1), (True, 1, 2), (0.0, 1, 2)),
)
def test_invalid_source_slot_permutations_are_rejected(bad_source_slots) -> None:
    with pytest.raises((TypeError, ValueError)):
        execute_baryon_terms(_bindings(), (_term(bad_source_slots),))


@pytest.mark.parametrize("bad_key", ("missing", "extra"))
def test_missing_or_extra_binding_keys_are_rejected(bad_key) -> None:
    bindings = _bindings()
    if bad_key == "missing":
        bindings.pop("P1")
    else:
        bindings[bad_key] = np.ones(1, dtype=np.complex128)

    with pytest.raises((KeyError, TypeError, ValueError)):
        execute_baryon_terms(bindings, (_term(),))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda values: values.update(S=np.ones((1, 2), dtype=np.complex128)),
        lambda values: values.update(
            P0=np.ones(
                (
                    _SINK_SPIN[0],
                    _SOURCE_SPIN[0] + 1,
                    _SINK_MODE[0],
                    _SOURCE_MODE[0],
                ),
                dtype=np.complex128,
            )
        ),
    ),
)
def test_rank_and_endpoint_shape_errors_are_rejected(mutator) -> None:
    bindings = _bindings()
    mutator(bindings)

    with pytest.raises((TypeError, ValueError)):
        execute_baryon_terms(bindings, (_term(),))
