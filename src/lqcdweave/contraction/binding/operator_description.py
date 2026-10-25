"""Pure, array-free identity for a canonical OperatorTerm used by E."""

from __future__ import annotations

import hashlib

import numpy as np

from lqcdweave.contracts.elemental import (
    BaryonComponent,
    ElementalComponent,
    GaugePath,
)
from lqcdweave.operators.single_hadron.api import SpectralProfile


def operator_term_semantics(term):
    """Return stable term semantics without exposing live arrays or callbacks."""
    component = term.component
    if isinstance(component, ElementalComponent):
        component_record = {
            "rank": 2,
            "key": _component_key(component.key),
            "time_offsets": tuple(component.time_offsets),
        }
        spin = None
    elif isinstance(component, BaryonComponent):
        component_record = {
            "rank": 3,
            "ordered_spatial_slots": _component_key(component.key),
        }
        spin = _spin_record(term.spin)
    else:
        raise TypeError("operator term contains an unsupported component type")

    profile = None if term.profile is None else _profile_record(term.profile)
    current = None
    if term.current_term is not None:
        current = dict(term.current_term.as_dict())
        # Current/Operator coefficients are represented once by OutputRecipe.
        current.pop("coefficient", None)
    result = {
        "component": component_record,
        "momentum": int(term.momentum),
        "gamma": term.gamma,
        "spin": spin,
        "profile": profile,
        "current_term": current,
    }
    if getattr(term, 'momentum_vector', None) is not None:
        result['momentum_vector'] = tuple(term.momentum_vector)
    return result


def _component_key(value):
    if isinstance(value, GaugePath):
        return {
            "kind": "gauge-path",
            "steps": tuple(value.steps),
            "displacement": tuple(value.displacement),
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_component_key(item) for item in value)
    return repr(value)


def _spin_record(spin):
    return {
        "basis": spin.basis,
        "entries": tuple(
            (int(a), int(b), int(c), complex(value))
            for a, b, c, value in spin.entries
        ),
    }


def _profile_record(profile):
    if not isinstance(profile, SpectralProfile):
        return {"type": type(profile).__qualname__, "identity": repr(profile)}
    digest = hashlib.sha256()
    arrays = []
    for name in ("left", "right", "coefficients"):
        value = np.asarray(getattr(profile, name))
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
        arrays.append((name, value.dtype.str, tuple(int(size) for size in value.shape)))
    return {
        "type": "SpectralProfile",
        "cfg": profile.cfg,
        "name": profile.name,
        "arrays": tuple(arrays),
        "sha256": digest.hexdigest(),
    }


__all__ = ["operator_term_semantics"]
