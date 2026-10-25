"""Independent numerical checks for the generic contraction executor.

The arrays in this module are deliberately small and nonsymmetric.  The
expected values come from direct Python loops so that these tests do not merely
restate the executor's use of an einsum implementation.  The final rank-three
case is a baryon-shaped tensor network, but it validates only generic tensor
topology and index mapping; it is not a production Wick-contraction test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from lqcdweave.contraction import (
    ContractionTerm,
    TensorSpec,
    compile_plan,
    execute_plan,
)


def _array(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Return deterministic complex data with distinct values on every axis."""

    size = int(np.prod(shape, dtype=int))
    values = np.arange(seed, seed + size, dtype=np.float64).reshape(shape)
    # The imaginary part is different from the real part to catch accidental
    # conjugation and makes axis-order bugs visible in the expected result.
    return (values + 1j * (2.0 * values + 0.25)).astype(np.complex128)


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Explicit matrix product used as a numerical oracle."""

    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.result_type(left, right))
    for i in range(left.shape[0]):
        for k in range(right.shape[1]):
            for j in range(left.shape[1]):
                result[i, k] += left[i, j] * right[j, k]
    return result


def _chain3(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    return _matmul(_matmul(first, second), third)


def _assert_values(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]) -> None:
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-12, atol=1e-12)


def test_tensor_spec_normalizes_dtype_and_shape() -> None:
    spec = TensorSpec((2, 3), "complex128")

    assert spec.shape == (2, 3)
    assert spec.dtype == np.dtype("complex128")


def test_complex_nonsymmetric_network_reuses_outputs_and_keeps_equal_shape_keys_distinct() -> None:
    """Common dependencies may be shared, while equal shapes do not imply equal data."""

    first = _array((2, 3), 1)
    second = _array((3, 4), 11)
    third = _array((4, 5), 31)
    first_alias = _array((2, 3), 101)
    specs = {
        "first": TensorSpec(first.shape, str(first.dtype)),
        "second": TensorSpec(second.shape, str(second.dtype)),
        "third": TensorSpec(third.shape, str(third.dtype)),
        # Same shape and dtype as ``first``, but a separate input identity.
        "first_alias": TensorSpec(first_alias.shape, str(first_alias.dtype)),
    }
    terms = (
        ContractionTerm(
            ("first", "second", "third"),
            "ij,jk,kl->il",
            coefficient=1.0 + 0.5j,
        ),
        # The ordered dependencies are identical; only output axes are
        # permuted.  This must preserve the explicit output order.
        ContractionTerm(
            ("first", "second", "third"),
            "ij,jk,kl->li",
            coefficient=-2.0j,
        ),
        # Equal shape is not an identity relation: this output must use the
        # independently bound ``first_alias`` data.
        ContractionTerm(
            ("first_alias", "second", "third"),
            "ij,jk,kl->il",
            coefficient=0.75 - 0.25j,
        ),
    )

    plan = compile_plan(specs, terms)
    result = execute_plan(plan, {"first": first, "second": second, "third": third, "first_alias": first_alias})

    common = _chain3(first, second, third)
    alias = _chain3(first_alias, second, third)
    expected = (
        (1.0 + 0.5j) * common,
        (-2.0j) * common.T,
        (0.75 - 0.25j) * alias,
    )
    _assert_values(result.values, expected)
    assert result.values[0].shape == (2, 5)
    assert result.values[1].shape == (5, 2)
    assert result.values[2].shape == (2, 5)


def test_rebinding_same_plan_uses_new_values_and_coefficients_do_not_alias_outputs() -> None:
    x = _array((2, 3), 3)
    y = _array((3, 4), 17)
    specs = {
        "x": TensorSpec(x.shape, str(x.dtype)),
        "y": TensorSpec(y.shape, str(y.dtype)),
    }
    terms = (
        ContractionTerm(("x", "y"), "ij,jk->ik", coefficient=2.0),
        ContractionTerm(("x", "y"), "ij,jk->ik", coefficient=-3.0),
    )
    plan = compile_plan(specs, terms)
    x_before = x.copy()
    y_before = y.copy()

    first = execute_plan(plan, {"x": x, "y": y})
    reference = _matmul(x, y)
    _assert_values(first.values, (2.0 * reference, -3.0 * reference))
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)
    assert not np.shares_memory(first.values[0], first.values[1])

    # Mutating a coefficient-applied result must not mutate the other output
    # or either borrowed input.
    first.values[0][0, 0] += 123.0
    np.testing.assert_allclose(first.values[1], -3.0 * reference)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)

    x_new = _array((2, 3), 203)
    y_new = _array((3, 4), 307)
    second = execute_plan(plan, {"x": x_new, "y": y_new})
    new_reference = _matmul(x_new, y_new)
    _assert_values(second.values, (2.0 * new_reference, -3.0 * new_reference))
    assert not np.allclose(second.values[0], first.values[0])


def test_plan_summary_and_execution_stats_report_exact_reuse_and_logical_memory() -> None:
    x = _array((2, 3), 13)
    y = _array((3, 4), 29)
    specs = {
        "x": TensorSpec(x.shape, "complex128"),
        "y": TensorSpec(y.shape, "complex128"),
    }
    terms = (
        ContractionTerm(("x", "y"), "ij,jk->ik", coefficient=2),
        ContractionTerm(("x", "y"), "ij,jk->ik", coefficient=-1j),
    )
    plan = compile_plan(specs, terms)
    summary = plan.summary

    assert summary["terms"] == 2
    assert summary["total_operations"] == plan.total_operations
    assert summary["unique_operations"] == plan.unique_operations
    assert summary["shared_operations"] == plan.shared_operations
    assert summary["eliminated_operations"] == plan.eliminated_operations
    assert plan.total_operations > plan.unique_operations
    assert plan.shared_operations >= 1
    assert summary() == dict(summary)

    result = execute_plan(plan, {"x": x, "y": y})
    stats = result.stats
    assert stats.executed_operations == plan.unique_operations
    assert stats.total_operations == plan.total_operations
    assert stats.unique_operations == plan.unique_operations
    assert stats.shared_operations == plan.shared_operations
    assert stats.input_bytes == x.nbytes + y.nbytes
    assert stats.output_bytes == sum(value.nbytes for value in result.values)
    assert stats.peak_live_bytes >= stats.peak_intermediate_bytes >= 0
    assert stats.peak_logical_live_bytes == stats.peak_live_bytes
    assert stats.operation_count == stats.executed_operations
    assert stats.scratch_bytes is None or stats.scratch_bytes >= 0


def test_unary_trace_and_scalar_output_match_explicit_loops() -> None:
    matrix = _array((3, 3), 7)
    vector = _array((3,), 19)
    specs = {
        "matrix": TensorSpec(matrix.shape, str(matrix.dtype)),
        "vector": TensorSpec(vector.shape, str(vector.dtype)),
    }
    terms = (
        ContractionTerm(("matrix",), "ii->", coefficient=1.5 - 0.25j),
        ContractionTerm(("vector", "vector"), "i,i->", coefficient=-2.0j),
    )

    result = execute_plan(
        compile_plan(specs, terms),
        {"matrix": matrix, "vector": vector},
    )
    trace = sum((matrix[i, i] for i in range(matrix.shape[0])), 0j)
    dot = sum((vector[i] * vector[i] for i in range(vector.shape[0])), 0j)
    _assert_values(result.values, ((1.5 - 0.25j) * trace, (-2.0j) * dot))
    assert np.ndim(result.values[0]) == 0
    assert np.ndim(result.values[1]) == 0


def test_disconnected_product_keeps_both_output_axes() -> None:
    left = _array((2,), 5)
    right = _array((3,), 23)
    specs = {
        "left": TensorSpec(left.shape, str(left.dtype)),
        "right": TensorSpec(right.shape, str(right.dtype)),
    }
    result = execute_plan(
        compile_plan(
            specs,
            (ContractionTerm(("left", "right"), "i,j->ij"),),
        ),
        {"left": left, "right": right},
    )

    expected = np.empty((2, 3), dtype=np.complex128)
    for i in range(left.shape[0]):
        for j in range(right.shape[0]):
            expected[i, j] = left[i] * right[j]
    _assert_values(result.values, (expected,))


def test_rank_three_three_quark_shaped_network_matches_literal_nested_loops() -> None:
    """This checks numerical topology, not production baryon Wick physics."""

    source = _array((2, 3, 2), 1)
    sink = _array((4, 2, 3), 41)
    propagator_0 = _array((4, 2), 81)
    propagator_1 = _array((2, 3), 91)
    propagator_2 = _array((3, 2), 101)
    specs = {
        "source": TensorSpec(source.shape, str(source.dtype)),
        "sink": TensorSpec(sink.shape, str(sink.dtype)),
        "p0": TensorSpec(propagator_0.shape, str(propagator_0.dtype)),
        "p1": TensorSpec(propagator_1.shape, str(propagator_1.dtype)),
        "p2": TensorSpec(propagator_2.shape, str(propagator_2.dtype)),
    }
    term = ContractionTerm(("source", "sink", "p0", "p1", "p2"), "abc,xyz,xa,yb,zc->")
    result = execute_plan(
        compile_plan(specs, (term,)),
        {
            "source": source,
            "sink": sink,
            "p0": propagator_0,
            "p1": propagator_1,
            "p2": propagator_2,
        },
    )

    expected = 0j
    for a in range(source.shape[0]):
        for b in range(source.shape[1]):
            for c in range(source.shape[2]):
                for x in range(sink.shape[0]):
                    for y in range(sink.shape[1]):
                        for z in range(sink.shape[2]):
                            expected += (
                                source[a, b, c]
                                * sink[x, y, z]
                                * propagator_0[x, a]
                                * propagator_1[y, b]
                                * propagator_2[z, c]
                            )
    _assert_values(result.values, (np.asarray(expected),))


@pytest.mark.parametrize(
    "make_plan",
    [
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128"), "b": TensorSpec((4, 2), "complex128")},
            (ContractionTerm(("a", "b"), "ij,jk->ik"),),
        ),
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128")},
            (ContractionTerm(("missing",), "ij->ij"),),
        ),
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128")},
            (ContractionTerm(("a",), "i->i"),),
        ),
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128")},
            (ContractionTerm(("a",), "ij->ik"),),
        ),
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128")},
            (ContractionTerm(("a",), "i...->i"),),
        ),
        lambda: compile_plan(
            {"a": TensorSpec((2, 3), "complex128")},
            (ContractionTerm(("a",), "ij"),),
        ),
    ],
    ids=[
        "dimension_mismatch",
        "unknown_input_key",
        "rank_mismatch",
        "unknown_output_label",
        "ellipsis_not_supported",
        "implicit_output_not_allowed",
    ],
)
def test_compile_rejects_invalid_shapes_keys_and_index_syntax(make_plan: Callable[[], object]) -> None:
    with pytest.raises((KeyError, TypeError, ValueError, NotImplementedError)):
        make_plan()


def test_execute_rejects_missing_wrong_shape_and_wrong_dtype_bindings() -> None:
    specs = {
        "a": TensorSpec((2, 3), "complex128"),
        "b": TensorSpec((3, 4), "complex128"),
    }
    plan = compile_plan(specs, (ContractionTerm(("a", "b"), "ij,jk->ik"),))
    a = _array((2, 3), 1)
    b = _array((3, 4), 13)

    with pytest.raises((KeyError, TypeError, ValueError)):
        execute_plan(plan, {"a": a})
    with pytest.raises((TypeError, ValueError)):
        execute_plan(plan, {"a": np.zeros((2, 2), dtype=np.complex128), "b": b})
    with pytest.raises((TypeError, ValueError)):
        execute_plan(plan, {"a": a.astype(np.complex64), "b": b})


def test_tensor_spec_rejects_unknown_dtype() -> None:
    with pytest.raises((TypeError, ValueError)):
        TensorSpec((2,), "definitely-not-a-numpy-dtype")
