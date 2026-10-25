"""Legacy-compatible directed one-link current data contract.

Copied from lattice/insertion/gauge_link.py::DirectedCurrentBasis and
lattice/insertion/current.py raw-contract helpers; file I/O stays in storage.
"""
from numbers import Integral
from typing import Mapping
from lqcdweave.contracts.time_selection import dataset_time_selection

import numpy as np

from .dataset import DatasetSpec, stable_digest

class DirectedCurrentBasis:
    """Stable one-link basis for raw conserved-current elementals."""

    SCHEMA = "lattice.current.directed-one-link-basis/v1"
    VERSION = 1
    DIRECTIONS = (
        (0, "+x", (1, 0, 0, 0), 0, False),
        (1, "+y", (0, 1, 0, 0), 1, False),
        (2, "+z", (0, 0, 1, 0), 2, False),
        (3, "-x", (-1, 0, 0, 0), 0, True),
        (4, "-y", (0, -1, 0, 0), 1, True),
        (5, "-z", (0, 0, -1, 0), 2, True),
        (6, "+t", (0, 0, 0, 1), 3, False),
        (7, "-t", (0, 0, 0, -1), 3, True),
    )

    @classmethod
    def metadata(cls):
        return {
            "schema": cls.SCHEMA,
            "version": cls.VERSION,
            "axis_order": [
                "direction", "time", "z", "y", "x", "color_row", "color_column"
            ],
            "directions": [
                {
                    "index": index,
                    "name": name,
                    "vector": list(vector),
                    "gauge_axis": gauge_axis,
                    "dagger": dagger,
                }
                for index, name, vector, gauge_axis, dagger in cls.DIRECTIONS
            ],
        }

    @classmethod
    def direction(cls, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("current direction index must be an integer")
        index = int(index)
        if not 0 <= index < len(cls.DIRECTIONS):
            raise ValueError("current direction index must be between 0 and 7")
        return cls.DIRECTIONS[index]

    @classmethod
    def index_for_term(cls, direction, link):
        if isinstance(direction, bool) or not isinstance(direction, (int, np.integer)):
            raise TypeError("current term direction must be an integer")
        direction = int(direction)
        if not 0 <= direction <= 3:
            raise ValueError("current term direction must be between 0 and 3")
        if link == "forward":
            return direction if direction < 3 else 6
        if link == "backward":
            return direction + 3 if direction < 3 else 7
        raise ValueError("current link must be 'forward' or 'backward'")


CURRENT_ELEMENTAL_SCHEMA = "lattice.current.raw-spatial-displacement-basis/v1"
CURRENT_DIRECTED_RAW_SCHEMA = "lattice.current.raw-directed-one-link-basis/v1"
CURRENT_DIRECTED_POINT_SCHEMA = "lattice.current.point-ended-directed-one-link/v1"
CURRENT_TERM_SCHEMA = "lattice.current.term/v1"
CURRENT_ASSEMBLER_SCHEMA = "lattice.current.assembler/v1"


def validate_directed_point_current_spec(spec):
    """Validate one registered temporal one-link point-ended current product.

    This is a metadata-only contract. It deliberately does not accept or read
    array values, and it does not change the older V/V raw-current contract.
    The DatasetSpec remains the sole owner of axes, ordered endpoint views,
    parent identities and the complete scientific definition.
    """
    if not isinstance(spec, DatasetSpec):
        raise TypeError("directed point-current validation requires a DatasetSpec")
    document = spec.document
    definition = document["definition"]
    if definition.get("directed_current_schema") != CURRENT_DIRECTED_POINT_SCHEMA:
        raise ValueError("dataset does not declare the supported point-current schema")
    kind = document["kind"]
    if kind not in {"elemental_v2p", "elemental_p2v", "elemental_p2p_row_sparse"}:
        raise ValueError("point-current products must be V/P, P/V or sparse P/P")
    if document["dtype"] != "complex128":
        raise ValueError("point-current products require complex128 storage")
    if document["conventions"].get("normalization") != "unscaled":
        raise ValueError("point-current products require unscaled normalization")
    if document["conventions"].get("phase") != "midpoint":
        raise ValueError("point-current products require midpoint phase")
    if definition.get("path_boundary") not in {"periodic", "open"}:
        raise ValueError("point-current path boundary must be periodic or open")
    expected_axes = {
        "elemental_v2p": ("time", "component", "momentum", "mode_left",
                          "point_right", "color_right"),
        "elemental_p2v": ("time", "component", "momentum", "point_left",
                          "color_left", "mode_right"),
        "elemental_p2p_row_sparse": ("time", "component", "momentum",
                                     "point_left", "color_left", "color_right"),
    }[kind]
    if spec.axis_names != expected_axes:
        raise ValueError("point-current axes do not match their ordered endpoint contract")
    extent = document["configuration"]["lattice"][3]
    selection = dataset_time_selection(spec, 'time')
    if spec.coordinates("time") != list(range(extent if selection is None else selection.Nt)):
        raise ValueError("point-current products must cover every physical anchor time in order")
    components = spec.coordinates("component")
    if (not components or any(
            not isinstance(component, dict) or set(component) != {"path"}
            or not isinstance(component["path"], list)
            or len(component["path"]) != 1
            or type(component["path"][0]) is not int
            or component["path"][0] not in (6, 7)
            for component in components)):
        raise ValueError("point-current components must be distinct one-step +t/-t paths")
    roles = ({"mode_left", "point_right"} if kind == "elemental_v2p" else
             {"point_left", "mode_right"} if kind == "elemental_p2v" else
             {"point_left"})
    if set(document["bases"]) != roles:
        raise ValueError("point-current basis bindings do not match their endpoint roles")
    gauge_id = definition.get("gauge_id")
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("point-current definition requires its actual gauge ID")
    parent_ids = {gauge_id}
    for role in roles:
        binding = document["bases"][role]
        data_id = binding["data_id"]
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(f"point-current {role} endpoint requires its actual dataset ID")
        parent_ids.add(data_id)
    right_point_labels = None
    if kind == "elemental_p2p_row_sparse":
        right_id = definition.get("point_right_id")
        selection = definition.get("point_right_selection")
        if not isinstance(right_id, str) or not right_id:
            raise ValueError("point-current P/P requires its actual right point dataset ID")
        if selection == "same_ordered_prefix_as_point_left":
            if right_id != document["bases"]["point_left"]["data_id"]:
                raise ValueError("same-prefix point-current P/P must reuse its left point dataset")
            if "point_right_labels" in definition:
                raise ValueError("same-prefix point-current P/P must not override its right labels")
            right_point_labels = list(spec.coordinates("point_left"))
        elif selection == "explicit_ordered_labels":
            labels = definition.get("point_right_labels")
            if (not isinstance(labels, list) or not labels
                    or any(type(label) is not int or label < 0 for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError("point-current P/P right labels must be unique ordered nonnegative integers")
            right_point_labels = labels
        else:
            raise ValueError("point-current P/P requires an explicit right endpoint selection")
        parent_ids.add(right_id)
    if (len(document["parents"]) != len(parent_ids)
            or set(document["parents"]) != parent_ids):
        raise ValueError("point-current parents must bind its actual gauge and endpoint datasets")
    for role in roles:
        labels = spec.coordinates(role)
        if not labels or document["bases"][role]["indices"] != labels:
            raise ValueError(f"point-current {role} must retain its actual ordered endpoint view")
    color_axes = (("color_right",) if kind == "elemental_v2p" else
                  ("color_left",) if kind == "elemental_p2v" else
                  ("color_left", "color_right"))
    for color_axis in color_axes:
        if spec.coordinates(color_axis) != [0, 1, 2]:
            raise ValueError(f"point-current {color_axis} must explicitly order all three colors")
    if kind == "elemental_p2p_row_sparse":
        if document["conventions"].get("point_pair_layout") != "left-point-row-slots-v1":
            raise ValueError("point-current P/P requires left-point row-slot storage")
        if not right_point_labels:
            raise ValueError("point-current P/P must retain its actual ordered right endpoint view")


def _current_basis_metadata():
    return DirectedCurrentBasis.metadata()


def _current_raw_fingerprint(contract):
    semantic = {key: value for key, value in contract.items() if key != "cache_identity"}
    return stable_digest(semantic)


def _current_raw_contract_keys():
    return {
        "schema", "version", "basis", "boundary", "combined_with_current_terms",
        "supports_temporal_point_split", "term_application", "term_schema",
        "assembler_schema", "channels", "shapes", "dtypes", "axes", "ne",
        "cache_identity",
    }


def _json_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_current_raw_contract_metadata(contract):
    if not isinstance(contract, Mapping):
        raise TypeError("raw current contract must be a mapping")
    if set(contract) != _current_raw_contract_keys():
        raise ValueError("raw current contract has missing or unknown metadata")
    if contract["schema"] != CURRENT_DIRECTED_RAW_SCHEMA or contract["version"] != 1:
        raise ValueError("raw current contract schema/version is unsupported")
    if contract["basis"] != _current_basis_metadata():
        raise ValueError("raw current contract directions do not match the v1 basis")
    if contract["boundary"] not in {"periodic", "open"}:
        raise ValueError("raw current contract boundary is invalid")
    if contract["combined_with_current_terms"] is not False:
        raise ValueError("raw current contract must not combine Current terms")
    if contract["supports_temporal_point_split"] is not True:
        raise ValueError("raw current contract must support temporal point splitting")
    if contract["term_application"] != "external-resolver-required":
        raise ValueError("raw current contract term application is invalid")
    if contract["term_schema"] != CURRENT_TERM_SCHEMA:
        raise ValueError("raw current contract term schema is invalid")
    if contract["assembler_schema"] != CURRENT_ASSEMBLER_SCHEMA:
        raise ValueError("raw current contract assembler schema is invalid")
    if contract["channels"] != ["v2v-one-link"]:
        raise ValueError("raw current contract is v2v-one-link only")
    if contract["axes"] != {
        "v2v": ["direction", "time", "momentum", "sink_ne", "source_ne"]
    }:
        raise ValueError("raw current contract axes are invalid")
    if set(contract["shapes"]) != {"v2v"} or set(contract["dtypes"]) != {"v2v"}:
        raise ValueError("raw current contract channel metadata is invalid")
    shape = contract["shapes"]["v2v"]
    if not isinstance(shape, list) or len(shape) != 5:
        raise ValueError("raw current contract v2v shape is invalid")
    shape = [_json_integer(value, "raw current shape") for value in shape]
    if shape[0] != 8 or any(value < 0 for value in shape):
        raise ValueError("raw current contract v2v shape is invalid")
    try:
        dtype = np.dtype(contract["dtypes"]["v2v"])
    except TypeError as exc:
        raise ValueError("raw current contract dtype is invalid") from exc
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("raw current contract dtype must be complex")
    ne = contract["ne"]
    if not isinstance(ne, Mapping) or set(ne) != {
        "available", "used", "source", "sink", "raw_generator_used_ne_is_symmetric"
    }:
        raise ValueError("raw current contract Ne metadata is invalid")
    available = _json_integer(ne["available"], "available raw Ne")
    used = _json_integer(ne["used"], "used raw Ne")
    if available < 0 or not 0 <= used <= available:
        raise ValueError("raw current contract Ne bounds are invalid")
    if ne["source"] != used or ne["sink"] != used or ne["raw_generator_used_ne_is_symmetric"] is not True:
        raise ValueError("raw current contract must have symmetric raw Ne")
    if shape[3:] != [used, used]:
        raise ValueError("raw current contract shape/Ne metadata disagrees")
    identity = contract["cache_identity"]
    if not isinstance(identity, str) or identity != _current_raw_fingerprint(contract):
        raise ValueError("raw current contract cache identity is stale or tampered")
    return {key: contract[key] for key in contract}


def build_current_raw_contract(raw, *, boundary, available_ne, used_ne, momentum_count):
    """Describe generated v2v-only eight-one-link raw data without writing it."""
    if not isinstance(raw, Mapping) or set(raw) != {"v2v"}:
        raise ValueError("directed current raw data must contain exactly v2v")
    if boundary not in {"periodic", "open"}:
        raise ValueError("raw current boundary must be 'periodic' or 'open'")
    available_ne, used_ne = _requested_ne(available_ne, used_ne, "raw")
    momentum_count = _ne_count(momentum_count, "momentum_count")
    value = raw["v2v"]
    shape = _shape(value)
    if len(shape) != 5 or shape[0] != 8 or shape[2:] != (momentum_count, used_ne, used_ne):
        raise ValueError("directed current v2v shape must be (8, Lt, momentum, sink_ne, source_ne)")
    dtype = np.dtype(value.dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError("directed current v2v dtype must be complex")
    contract = {
        "schema": CURRENT_DIRECTED_RAW_SCHEMA,
        "version": 1,
        "basis": _current_basis_metadata(),
        "boundary": boundary,
        "combined_with_current_terms": False,
        "supports_temporal_point_split": True,
        "term_application": "external-resolver-required",
        "term_schema": CURRENT_TERM_SCHEMA,
        "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
        "channels": ["v2v-one-link"],
        "shapes": {"v2v": list(shape)},
        "dtypes": {"v2v": dtype.str},
        "axes": {"v2v": ["direction", "time", "momentum", "sink_ne", "source_ne"]},
        "ne": {
            "available": available_ne,
            "used": used_ne,
            "source": used_ne,
            "sink": used_ne,
            "raw_generator_used_ne_is_symmetric": True,
        },
    }
    contract["cache_identity"] = _current_raw_fingerprint(contract)
    return contract

def _shape(value):
    return tuple(int(size) for size in np.shape(value))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _ne_count(value, name):
    count = _integer(value, name)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _requested_ne(available, used, axis):
    available = _ne_count(available, f"available_{axis}_ne")
    if used is None:
        used = available
    used = _ne_count(used, f"used_{axis}_ne")
    if used > available:
        raise ValueError(
            f"used_{axis}_ne must not exceed available_{axis}_ne ({available})"
        )
    return available, used



def validate_current_raw_contract(raw, contract, *, require_temporal=False,
                                 validate_values=True):
    """Validate raw data before consuming it; legacy data can never imply time links."""
    if not isinstance(contract, Mapping):
        raise TypeError("raw current contract must be a mapping")
    schema = contract.get("schema")
    if schema == CURRENT_ELEMENTAL_SCHEMA:
        raise ValueError("legacy spatial raw contract cannot supply directed current links")
    if schema != CURRENT_DIRECTED_RAW_SCHEMA:
        raise ValueError("unrecognized raw current schema")
    metadata = _validate_current_raw_contract_metadata(contract)
    expected = build_current_raw_contract(
        raw, boundary=metadata["boundary"],
        available_ne=metadata["ne"]["available"],
        used_ne=metadata["ne"]["used"],
        momentum_count=metadata["shapes"]["v2v"][2],
    )
    if metadata != expected:
        raise ValueError("raw current contract does not match its data")
    if type(validate_values) is not bool:
        raise TypeError("validate_values must be bool")
    if validate_values:
        values = np.asarray(raw["v2v"])
        if not np.all(np.isfinite(values)):
            raise ValueError("directed current raw v2v must contain only finite values")
    return expected


def current_raw_cache_key(key, contract):
    if not isinstance(key, str) or not key:
        raise ValueError("raw current cache key must be a non-empty string")
    validated = _validate_current_raw_contract_metadata(contract)
    return f"{key}:{validated['cache_identity']}"
