"""Directed-current V2V term and pair contractions.

Copied from lattice/insertion/current.py and lattice/current_elemental.py.
The numeric formulas and provenance records retain the legacy convention.
"""
from numbers import Integral
from typing import Mapping
import numpy as np
from lqcdweave.contracts.current import DirectedCurrentBasis, validate_current_raw_contract
from lqcdweave.operators.single_hadron.api import (
    ConservedVectorInsertion, coerce_current_term,
    validate_one_link_current_term,
)
from lqcdweave.operators.single_hadron.api import gamma

CURRENT_V2V_CONTRACTION_SCHEMA = "lattice.current.v2v-termwise-contraction/v1"
CURRENT_V2V_PAIR_CONTRACTION_SCHEMA = "lattice.current.v2v-term-pair-contraction/v1"

_coerce_term = coerce_current_term


def _momentum_index(current_or_terms, requested, name):
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, Integral)):
        raise TypeError(f"{name} must be an integer")
    if isinstance(current_or_terms, ConservedVectorInsertion):
        bound = current_or_terms.momentum_index
        if requested is not None and int(requested) != bound:
            raise ValueError(f"{name} disagrees with the conserved-current insertion")
        return bound
    return 0 if requested is None else int(requested)

def _finite_complex_array(value, name):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} dtype must be complex")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array



def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _ne_count(value, name):
    count = _integer(value, name)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _validated_raw_endpoints(endpoints, term, boundary, extent):
    if not isinstance(endpoints, Mapping) or set(endpoints) != {
        "bar_time", "field_time", "link_origin_time", "temporal_point_split", "boundary"
    }:
        raise ValueError("resolved current endpoints are missing or unknown")
    if endpoints["boundary"] != boundary:
        raise ValueError("resolved endpoint boundary must match the raw current contract")
    if endpoints["temporal_point_split"] is not term.temporal_point_split:
        raise ValueError("resolved endpoint temporal flag does not match the current term")
    for name in ("bar_time", "field_time", "link_origin_time"):
        _integer(endpoints[name], name)
        if not 0 <= endpoints[name] < extent:
            raise ValueError("raw link endpoints must lie inside the physical time coverage")
    field_time = endpoints["bar_time"] + term.displacement[3]
    if boundary == "periodic":
        field_time %= extent
    origin_time = endpoints["bar_time"] if term.link == "forward" else field_time
    if endpoints["field_time"] != field_time or endpoints["link_origin_time"] != origin_time:
        raise ValueError("raw link endpoints differ from its directed one-link orientation")
    return endpoints


def resolve_directed_current_raw(raw, contract, term, *, endpoints, source_ne, sink_ne, momentum=0):
    """Resolve raw orientation/anchor only; Current weighting remains in the assembler."""
    current_term = validate_one_link_current_term(term)
    temporal = current_term.direction == 3 and current_term.link != "none"
    validated = validate_current_raw_contract(
        raw, contract, require_temporal=temporal, validate_values=False,
    )
    if current_term.link == "none":
        raise ValueError("directed raw resolver requires a forward or backward linked current term")
    endpoints = _validated_raw_endpoints(endpoints, current_term, validated["boundary"],
                                         validated["shapes"]["v2v"][1])
    source_ne = _ne_count(source_ne, "source_ne")
    sink_ne = _ne_count(sink_ne, "sink_ne")
    used_ne = validated["ne"]["used"]
    if source_ne > used_ne or sink_ne > used_ne:
        raise ValueError("requested source_ne/sink_ne exceeds symmetric raw generator used_ne")
    if not isinstance(momentum, Integral) or isinstance(momentum, (bool, np.bool_)):
        raise TypeError("momentum must be an integer")
    momentum = int(momentum)
    if not 0 <= momentum < validated["shapes"]["v2v"][2]:
        raise ValueError("momentum is outside the raw current momentum axis")
    direction = DirectedCurrentBasis.index_for_term(current_term.direction, current_term.link)
    raw_anchor_time = endpoints["bar_time"]
    extent = validated["shapes"]["v2v"][1]
    if not 0 <= raw_anchor_time < extent:
        raise ValueError("resolved raw-basis bar anchor is outside raw current data")
    value = raw["v2v"][direction, raw_anchor_time, momentum, :sink_ne, :source_ne]
    value = _finite_complex_array(value, "directed current raw contribution")
    return {
        "value": value,
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {
            "raw_schema": validated["schema"], "cache_identity": validated["cache_identity"],
            "direction": direction, "time": raw_anchor_time, "momentum": momentum,
            "raw_anchor": "bar_endpoint",
        },
    }


def _boundary_inputs(anchor_time, temporal_extent, boundary):
    anchor_time = _integer(anchor_time, "anchor_time")
    if boundary not in {"periodic", "open", "unbounded"}:
        raise ValueError("boundary must be one of: periodic, open, unbounded")
    if boundary == "unbounded":
        return anchor_time, temporal_extent
    temporal_extent = _integer(temporal_extent, "temporal_extent")
    if temporal_extent <= 0:
        raise ValueError("temporal_extent must be a positive integer")
    return anchor_time, temporal_extent


def resolve_current_term_endpoints(
    term, *, anchor_time, temporal_extent=None, boundary,
):
    """Resolve a v1 current term's temporal endpoints under caller policy."""
    current_term = _coerce_term(term)
    anchor_time, temporal_extent = _boundary_inputs(
        anchor_time, temporal_extent, boundary,
    )
    raw_times = {
        "bar_time": anchor_time + current_term.bar_offset[3],
        "field_time": anchor_time + current_term.field_offset[3],
        "link_origin_time": anchor_time + current_term.link_origin_offset[3],
    }
    if boundary == "periodic":
        times = {
            name: value % temporal_extent for name, value in raw_times.items()
        }
    elif boundary == "open":
        if any(value < 0 or value >= temporal_extent for value in raw_times.values()):
            raise IndexError("current term endpoint is outside the open temporal boundary")
        times = raw_times
    else:
        times = raw_times
    return {
        **times,
        "temporal_point_split": current_term.temporal_point_split,
        "boundary": boundary,
    }


def _spin_matrix_for_term(term):
    """Return the one explicit 4x4 spin action encoded by a current term."""
    current_term = _coerce_term(term)
    gamma_matrix = np.asarray(gamma(current_term.gamma_index))
    if gamma_matrix.shape != (4, 4):
        raise ValueError("current gamma matrix must have shape (4, 4)")
    if not np.issubdtype(gamma_matrix.dtype, np.complexfloating):
        gamma_matrix = gamma_matrix.astype(np.complex128)
    if not np.all(np.isfinite(gamma_matrix)):
        raise ValueError("current gamma matrix must contain only finite values")
    if current_term.link == "none":
        return gamma_matrix
    if current_term.wilson_r is None:
        raise ValueError("linked current term requires wilson_r")
    identity = np.eye(4, dtype=gamma_matrix.dtype)
    if current_term.link == "forward":
        return current_term.wilson_r * identity - gamma_matrix
    return current_term.wilson_r * identity + gamma_matrix


def _validate_spin_raw_result(resolved, *, source_ne, sink_ne):
    """Validate one unweighted, spinless elemental contribution from a callback."""
    if not isinstance(resolved, Mapping):
        raise TypeError("raw elemental resolver result must be a mapping")
    missing = {"value", "source_ne", "sink_ne"}.difference(resolved)
    if missing:
        raise TypeError(
            "raw elemental resolver result is missing: " + ", ".join(sorted(missing))
        )
    resolved_source = _ne_count(resolved["source_ne"], "raw resolver source_ne")
    resolved_sink = _ne_count(resolved["sink_ne"], "raw resolver sink_ne")
    if resolved_source != source_ne or resolved_sink != sink_ne:
        raise ValueError("raw resolver source_ne/sink_ne must exactly match requested values")
    value = _finite_complex_array(resolved["value"], "raw elemental contribution")
    if value.ndim != 2 or value.shape != (sink_ne, source_ne):
        raise ValueError(
            "raw elemental contribution must have shape "
            f"(sink_ne, source_ne) = ({sink_ne}, {source_ne})"
        )
    if value.size == 0:
        raise ValueError("raw elemental contribution must be non-empty")
    return value


def resolve_current_term_spin(
    term, raw_resolver, *, endpoints, source_ne, sink_ne,
):
    """Resolve one term into `(sink_spin, source_spin, sink_ne, source_ne)`.

    ``raw_resolver`` owns raw link/displacement selection and returns a complex
    spinless V2V contribution.  This bridge applies exactly one explicit spin
    matrix; it never applies a term coefficient or normalization.
    """
    if not callable(raw_resolver):
        raise TypeError("raw_resolver must be callable")
    current_term = _coerce_term(term)
    source_ne = _ne_count(source_ne, "source_ne")
    sink_ne = _ne_count(sink_ne, "sink_ne")
    raw_result = raw_resolver(
        current_term.as_dict(), endpoints=endpoints,
        source_ne=source_ne, sink_ne=sink_ne,
    )
    raw_value = _validate_spin_raw_result(
        raw_result, source_ne=source_ne, sink_ne=sink_ne,
    )
    spin_matrix = _spin_matrix_for_term(current_term)
    value = spin_matrix[:, :, None, None] * raw_value[None, None, :, :]
    return {
        "value": value,
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {
            "spin_structure": current_term.spin_structure,
            "spin_axes": ("sink_spin", "source_spin"),
            "raw": raw_result.get("provenance"),
        },
    }



def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _vsv_block(value, name):
    if hasattr(value, "get") and callable(value.get):
        value = value.get()
    try:
        value = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if value.ndim != 4 or value.shape[:2] != (4, 4):
        raise ValueError(f"{name} must have axes (sink_spin, source_spin, sink_ne, source_ne)")
    if not np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a complex dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def contract_directed_current_v2v(
    current_or_terms,
    raw,
    raw_contract,
    incoming_vsv,
    outgoing_vsv,
    *,
    source_time,
    sink_time,
    anchor_time,
    current_source_ne,
    current_sink_ne,
    momentum=None,
):
    """Contract point-split Current terms with already-loaded VSV accessors.

    For every term this computes ``S(field->sink) J(bar<-field)
    S(source->bar)`` and only then applies the term coefficient and sums. The
    accessors must already be loaded and expose ``get(t_source, t_sink)``;
    this function never calls ``load`` or any propagator generator.
    """
    momentum = _momentum_index(current_or_terms, momentum, "momentum")
    validated = validate_current_raw_contract(
        raw, raw_contract, require_temporal=True, validate_values=False,
    )
    terms = current_or_terms.terms if hasattr(current_or_terms, "terms") else tuple(current_or_terms)
    if not terms:
        raise ValueError("current terms must be non-empty")
    for name, accessor in (("incoming_vsv", incoming_vsv), ("outgoing_vsv", outgoing_vsv)):
        if not callable(getattr(accessor, "get", None)):
            raise TypeError(f"{name} must provide get(t_source, t_sink)")
    source_time = _time(source_time, "source_time")
    sink_time = _time(sink_time, "sink_time")
    anchor_time = _time(anchor_time, "anchor_time")
    extent = validated["shapes"]["v2v"][1]
    for name, time in (("source_time", source_time), ("sink_time", sink_time)):
        if not 0 <= time < extent:
            raise ValueError(f"{name} must be inside the raw temporal extent")
    boundary = validated["boundary"]

    result = None
    expected_shape = None
    records = []
    for index, term in enumerate(terms):
        endpoints = resolve_current_term_endpoints(
            term,
            anchor_time=anchor_time,
            temporal_extent=extent,
            boundary=boundary,
        )

        def raw_resolver(term_mapping, **kwargs):
            return resolve_directed_current_raw(
                raw,
                validated,
                term_mapping,
                momentum=momentum,
                **kwargs,
            )

        resolved = resolve_current_term_spin(
            term,
            raw_resolver,
            endpoints=endpoints,
            source_ne=current_source_ne,
            sink_ne=current_sink_ne,
        )
        vertex = resolved["value"]
        outgoing = _vsv_block(
            outgoing_vsv.get(endpoints["field_time"], sink_time),
            "outgoing VSV block",
        )
        incoming = _vsv_block(
            incoming_vsv.get(source_time, endpoints["bar_time"]),
            "incoming VSV block",
        )
        if outgoing.shape[3] < current_source_ne:
            raise ValueError("outgoing VSV field Ne is smaller than current_source_ne")
        if incoming.shape[2] < current_sink_ne:
            raise ValueError("incoming VSV bar Ne is smaller than current_sink_ne")
        outgoing = outgoing[..., :current_source_ne]
        incoming = incoming[..., :current_sink_ne, :]
        term_value = np.einsum(
            "afAi,bfji,bcjC->acAC",
            outgoing,
            vertex,
            incoming,
            optimize=True,
        )
        if expected_shape is None:
            expected_shape = term_value.shape
        elif term_value.shape != expected_shape:
            raise ValueError("term-wise VSV contractions must have identical external shapes")
        term_mapping = term.as_dict() if hasattr(term, "as_dict") else term
        coefficient = term_mapping["coefficient"]
        normalization = term_mapping.get("normalization", 1)
        weighted = coefficient * normalization * term_value
        result = weighted.copy() if result is None else result + weighted
        records.append(
            {
                "term_index": index,
                "endpoints": endpoints,
                "coefficient": coefficient,
                "normalization": normalization,
                "incoming_vsv_get": [source_time, endpoints["bar_time"]],
                "outgoing_vsv_get": [endpoints["field_time"], sink_time],
                "raw": resolved["provenance"]["raw"],
            }
        )

    return {
        "schema": CURRENT_V2V_CONTRACTION_SCHEMA,
        "value": result,
        "axes": (
            "external_sink_spin",
            "external_source_spin",
            "external_sink_ne",
            "external_source_ne",
        ),
        "source_time": source_time,
        "sink_time": sink_time,
        "anchor_time": anchor_time,
        "boundary": boundary,
        "momentum": int(momentum),
        "terms": tuple(records),
        "raw_cache_identity": validated["cache_identity"],
        "uses_preloaded_vsv_only": True,
    }


def contract_directed_current_pair_v2v(
    first_current_or_terms,
    first_raw,
    first_raw_contract,
    second_current_or_terms,
    second_raw,
    second_raw_contract,
    vsv,
    *,
    first_anchor_time,
    second_anchor_time,
    first_field_ne,
    first_bar_ne,
    second_field_ne,
    second_bar_ne,
    first_momentum=None,
    second_momentum=None,
    array_backend=None,
):
    """Close two point-split Current vertices with preloaded VSV blocks.

    This computes the ordered, unflavored and unsigned connected trace for
    every pair of Current terms.  For first term ``A`` and second term ``B``
    the two propagators are ``S(field_A -> bar_B)`` and
    ``S(field_B -> bar_A)``.  The exact contraction is
    ``bfji,ackl,afki,bcjl->`` before multiplying the two terms' coefficients
    and normalizations.  No Wick sign, flavor factor, volume normalization,
    conjugation, real-part selection, or source averaging is implicit.

    ``vsv`` must already be loaded and expose ``get(t_source, t_sink)``.  This
    function never loads or generates propagators.
    """
    first_momentum = _momentum_index(first_current_or_terms, first_momentum, "first_momentum")
    second_momentum = _momentum_index(second_current_or_terms, second_momentum, "second_momentum")
    backend = np if array_backend is None else array_backend
    if not callable(getattr(backend, "asarray", None)) or not callable(getattr(backend, "einsum", None)):
        raise TypeError("array_backend must provide callable asarray and einsum")
    first_validated = validate_current_raw_contract(
        first_raw, first_raw_contract, require_temporal=True, validate_values=False,
    )
    second_validated = validate_current_raw_contract(
        second_raw, second_raw_contract, require_temporal=True, validate_values=False,
    )
    first_shape = first_validated["shapes"]["v2v"]
    second_shape = second_validated["shapes"]["v2v"]
    if first_shape[1] != second_shape[1]:
        raise ValueError("paired Current raw temporal extents must match")
    if first_validated["boundary"] != second_validated["boundary"]:
        raise ValueError("paired Current raw boundaries must match")
    if not callable(getattr(vsv, "get", None)):
        raise TypeError("vsv must provide get(t_source, t_sink)")

    first_terms = (
        first_current_or_terms.terms if hasattr(first_current_or_terms, "terms") else tuple(first_current_or_terms)
    )
    second_terms = (
        second_current_or_terms.terms
        if hasattr(second_current_or_terms, "terms")
        else tuple(second_current_or_terms)
    )
    if not first_terms or not second_terms:
        raise ValueError("both Current term collections must be non-empty")

    extent = first_shape[1]
    boundary = first_validated["boundary"]
    first_anchor_time = _time(first_anchor_time, "first_anchor_time")
    second_anchor_time = _time(second_anchor_time, "second_anchor_time")
    for name, time in (
        ("first_anchor_time", first_anchor_time),
        ("second_anchor_time", second_anchor_time),
    ):
        if not 0 <= time < extent:
            raise ValueError(f"{name} must be inside the raw temporal extent")
    ne_counts = {
        "first_field_ne": _time(first_field_ne, "first_field_ne"),
        "first_bar_ne": _time(first_bar_ne, "first_bar_ne"),
        "second_field_ne": _time(second_field_ne, "second_field_ne"),
        "second_bar_ne": _time(second_bar_ne, "second_bar_ne"),
    }
    if any(value <= 0 for value in ne_counts.values()):
        raise ValueError("paired Current Ne counts must be positive")

    result = None
    records = []
    for first_index, first_term in enumerate(first_terms):
        first_endpoints = resolve_current_term_endpoints(
            first_term,
            anchor_time=first_anchor_time,
            temporal_extent=extent,
            boundary=boundary,
        )

        def first_raw_resolver(term_mapping, **kwargs):
            return resolve_directed_current_raw(
                first_raw,
                first_validated,
                term_mapping,
                momentum=first_momentum,
                **kwargs,
            )

        first_resolved = resolve_current_term_spin(
            first_term,
            first_raw_resolver,
            endpoints=first_endpoints,
            source_ne=ne_counts["first_field_ne"],
            sink_ne=ne_counts["first_bar_ne"],
        )
        for second_index, second_term in enumerate(second_terms):
            second_endpoints = resolve_current_term_endpoints(
                second_term,
                anchor_time=second_anchor_time,
                temporal_extent=extent,
                boundary=boundary,
            )

            def second_raw_resolver(term_mapping, **kwargs):
                return resolve_directed_current_raw(
                    second_raw,
                    second_validated,
                    term_mapping,
                    momentum=second_momentum,
                    **kwargs,
                )

            second_resolved = resolve_current_term_spin(
                second_term,
                second_raw_resolver,
                endpoints=second_endpoints,
                source_ne=ne_counts["second_field_ne"],
                sink_ne=ne_counts["second_bar_ne"],
            )
            first_to_second = backend.asarray(
                _vsv_block(
                    vsv.get(
                        first_endpoints["field_time"],
                        second_endpoints["bar_time"],
                    ),
                    "first-to-second VSV block",
                )
            )
            second_to_first = backend.asarray(
                _vsv_block(
                    vsv.get(
                        second_endpoints["field_time"],
                        first_endpoints["bar_time"],
                    ),
                    "second-to-first VSV block",
                )
            )
            if (
                first_to_second.shape[2] < ne_counts["second_bar_ne"]
                or first_to_second.shape[3] < ne_counts["first_field_ne"]
            ):
                raise ValueError("first-to-second VSV Ne extents are smaller than requested")
            if (
                second_to_first.shape[2] < ne_counts["first_bar_ne"]
                or second_to_first.shape[3] < ne_counts["second_field_ne"]
            ):
                raise ValueError("second-to-first VSV Ne extents are smaller than requested")
            first_to_second = first_to_second[
                ...,
                : ne_counts["second_bar_ne"],
                : ne_counts["first_field_ne"],
            ]
            second_to_first = second_to_first[
                ...,
                : ne_counts["first_bar_ne"],
                : ne_counts["second_field_ne"],
            ]
            first_vertex = backend.asarray(first_resolved["value"])
            second_vertex = backend.asarray(second_resolved["value"])
            pair_value = backend.einsum(
                "bfji,ackl,afki,bcjl->",
                first_to_second,
                second_to_first,
                first_vertex,
                second_vertex,
                optimize=True,
            )
            first_mapping = first_term.as_dict() if hasattr(first_term, "as_dict") else first_term
            second_mapping = second_term.as_dict() if hasattr(second_term, "as_dict") else second_term
            first_weight = first_mapping["coefficient"] * first_mapping.get("normalization", 1)
            second_weight = second_mapping["coefficient"] * second_mapping.get("normalization", 1)
            weighted = backend.asarray(first_weight * second_weight * pair_value)
            result = weighted.copy() if result is None else result + weighted
            records.append(
                {
                    "first_term_index": first_index,
                    "second_term_index": second_index,
                    "first_endpoints": first_endpoints,
                    "second_endpoints": second_endpoints,
                    "first_coefficient": first_mapping["coefficient"],
                    "first_normalization": first_mapping.get("normalization", 1),
                    "second_coefficient": second_mapping["coefficient"],
                    "second_normalization": second_mapping.get("normalization", 1),
                    "first_to_second_vsv_get": [
                        first_endpoints["field_time"],
                        second_endpoints["bar_time"],
                    ],
                    "second_to_first_vsv_get": [
                        second_endpoints["field_time"],
                        first_endpoints["bar_time"],
                    ],
                    "first_raw": first_resolved["provenance"]["raw"],
                    "second_raw": second_resolved["provenance"]["raw"],
                }
            )

    return {
        "schema": CURRENT_V2V_PAIR_CONTRACTION_SCHEMA,
        "value": result,
        "axes": (),
        "operation": (
            "sum_terms bfji,ackl,afki,bcjl->; ordered connected trace; "
            "no implicit Wick sign, flavor factor, normalization, conjugation, "
            "real-part selection, or source averaging"
        ),
        "first_anchor_time": first_anchor_time,
        "second_anchor_time": second_anchor_time,
        "boundary": boundary,
        "momenta": {
            "first": int(first_momentum),
            "second": int(second_momentum),
        },
        "ne": ne_counts,
        "term_pairs": tuple(records),
        "raw_cache_identities": {
            "first": first_validated["cache_identity"],
            "second": second_validated["cache_identity"],
        },
        "uses_preloaded_vsv_only": True,
    }
