"""Local and Wilson point-split currents as coefficient recipes.

CurrentTerm and its validation were copied from lattice/insertion/current.py.
"""
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Mapping, Optional
import numpy as np

CURRENT_TERM_SCHEMA = "lattice.current.term/v1"
_ZERO_OFFSET = (0, 0, 0, 0)

def _finite_numeric_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite numeric scalar")
    arr = np.asarray(value)
    if arr.ndim != 0 or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be a finite numeric scalar")
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite numeric scalar") from exc
    if not bool(finite):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _offset(value, name):
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain four integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain four integers") from exc
    if len(values) != 4:
        raise ValueError(f"{name} must contain four integers")
    return tuple(_integer(component, f"{name}[{index}]")
                 for index, component in enumerate(values))


@dataclass(frozen=True)
class CurrentTerm:
    coefficient: complex
    direction: int
    displacement: tuple[int, int, int, int]
    gamma_index: int
    link: str = "none"
    wilson_r: Optional[float] = None
    spin_structure: str = "gamma"
    normalization: complex = 1
    bar_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    field_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    link_origin_offset: tuple[int, int, int, int] = _ZERO_OFFSET
    link_dagger: bool = False
    boundary_policy: str = "caller-supplied"
    temporal_point_split: bool = False

    def __post_init__(self):
        _finite_numeric_scalar(self.coefficient, "coefficient")
        _finite_numeric_scalar(self.normalization, "normalization")
        direction = _integer(self.direction, "direction")
        if not -1 <= direction <= 3:
            raise ValueError("direction must be between -1 and 3")
        gamma_index = _integer(self.gamma_index, "gamma_index")
        if not 0 <= gamma_index <= 15:
            raise ValueError("gamma_index must be between 0 and 15")
        displacement = _offset(self.displacement, "displacement")
        bar_offset = _offset(self.bar_offset, "bar_offset")
        field_offset = _offset(self.field_offset, "field_offset")
        link_origin_offset = _offset(self.link_origin_offset, "link_origin_offset")
        if self.link not in {"none", "forward", "backward"}:
            raise ValueError("link must be one of: none, forward, backward")
        if self.wilson_r is not None:
            _finite_numeric_scalar(self.wilson_r, "wilson_r")
        if not isinstance(self.spin_structure, str) or not self.spin_structure:
            raise TypeError("spin_structure must be a non-empty string")
        if not isinstance(self.link_dagger, (bool, np.bool_)):
            raise TypeError("link_dagger must be a bool")
        if self.boundary_policy != "caller-supplied":
            raise ValueError("boundary_policy must be 'caller-supplied'")
        if not isinstance(self.temporal_point_split, (bool, np.bool_)):
            raise TypeError("temporal_point_split must be a bool")
        has_temporal_endpoint = any(
            offset[3] != 0
            for offset in (bar_offset, field_offset, link_origin_offset)
        )
        if has_temporal_endpoint and direction != 3:
            raise ValueError("temporal endpoint offsets require direction=3")
        expected_temporal = direction == 3 and has_temporal_endpoint
        if bool(self.temporal_point_split) != expected_temporal:
            raise ValueError(
                "temporal_point_split must match direction=3 temporal endpoints"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "gamma_index", gamma_index)
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "bar_offset", bar_offset)
        object.__setattr__(self, "field_offset", field_offset)
        object.__setattr__(self, "link_origin_offset", link_origin_offset)
        object.__setattr__(self, "link_dagger", bool(self.link_dagger))
        object.__setattr__(self, "temporal_point_split", bool(self.temporal_point_split))

    def term_time_offsets(self):
        """Return ``(left_delta, right_delta)`` for this term.

        This is the vertex protocol's per-term leg placement (02 SS2): the left
        (sink-facing) leg sits at ``anchor + bar_offset[3]`` and the right
        (source-facing) leg at ``anchor + field_offset[3]``.  A plain meson has no
        such method, which is exactly how the graph decides whether a vertex is
        handled term by term -- the predicate is whether this method exists.

        The deltas are unwrapped integers, so a backward boundary term reports -1
        and the propagator's own modular time handling performs the wrap.
        """
        return int(self.bar_offset[3]), int(self.field_offset[3])

    def as_dict(self):
        return {
            "schema": CURRENT_TERM_SCHEMA,
            "coefficient": self.coefficient,
            "direction": self.direction,
            "displacement": self.displacement,
            "gamma_index": self.gamma_index,
            "link": self.link,
            "wilson_r": self.wilson_r,
            "spin_structure": self.spin_structure,
            "normalization": self.normalization,
            "bar_offset": self.bar_offset,
            "field_offset": self.field_offset,
            "link_origin_offset": self.link_origin_offset,
            "link_dagger": self.link_dagger,
            "boundary_policy": self.boundary_policy,
            "temporal_point_split": self.temporal_point_split,
        }


_TERM_FIELD_NAMES = frozenset(field.name for field in fields(CurrentTerm))



def _coerce_term(term):
    if isinstance(term, CurrentTerm):
        return term
    if not isinstance(term, Mapping):
        raise TypeError("term must be a CurrentTerm or v1 mapping")
    if "schema" not in term:
        raise ValueError("term mapping must include its v1 schema")
    schema = term["schema"]
    if schema != CURRENT_TERM_SCHEMA:
        raise ValueError(f"term schema must be {CURRENT_TERM_SCHEMA}")
    unknown = set(term).difference(_TERM_FIELD_NAMES | {"schema"})
    if unknown:
        raise ValueError("term contains unknown fields: " + ", ".join(sorted(unknown)))
    try:
        values = {name: term[name] for name in _TERM_FIELD_NAMES if name in term}
        return CurrentTerm(**values)
    except KeyError as exc:
        raise ValueError(f"term is missing required field: {exc.args[0]}") from exc
    except TypeError as exc:
        if "required positional argument" in str(exc):
            raise ValueError("term is missing required fields") from exc
        raise



coerce_current_term = _coerce_term


def validate_one_link_current_term(term):
    """Require a linked recipe represented by the stored eight-one-link basis.

    This validates relative endpoints and orientation, not a Ward identity or
    agreement with a particular Dirac action. Common endpoint shifts are kept.
    """
    term = coerce_current_term(term)
    if term.direction not in range(4) or term.link not in ("forward", "backward"):
        raise ValueError("eight-link raw requires a directed linked current term")
    if term.wilson_r is None:
        raise ValueError("linked current term requires wilson_r")
    sign = 1 if term.link == "forward" else -1
    displacement = tuple(sign if i == term.direction else 0 for i in range(4))
    if (term.displacement != displacement or
            tuple(f-b for f,b in zip(term.field_offset, term.bar_offset)) != displacement):
        raise ValueError("current endpoints/displacement must match one directed link")
    origin = term.bar_offset if sign == 1 else term.field_offset
    if term.link_origin_offset != origin or term.link_dagger != (sign == -1):
        raise ValueError("current link origin/dagger differs from its orientation")
    return term


class LocalVectorCurrent:
    name = "local_vector"

    @property
    def terms(self):
        return tuple(CurrentTerm(1, d, (0,0,0,0), 1 << d) for d in range(4))

    def operator(self, direction, momentum_index=0):
        from .operator import OperatorPath
        return OperatorPath.from_current(self, direction, momentum_index)


class LocalAxialCurrent(LocalVectorCurrent):
    name = "local_axial"

    @property
    def terms(self):
        return tuple(CurrentTerm(c, d, (0,0,0,0), g)
                     for d, (g, c) in enumerate(zip((14,13,11,7), (1,-1,1,-1))))


class PseudoScalarDensity(LocalVectorCurrent):
    name = "pseudoscalar_density"

    @property
    def terms(self):
        return (CurrentTerm(1, -1, (0,0,0,0), 15),)


class ConservedVectorCurrent(LocalVectorCurrent):
    """Wilson point-split coefficient definition; no automatic Z or a factors."""
    name = "conserved_vector"

    def __init__(self, *, wilson_r=1.0):
        if not np.isreal(wilson_r) or not np.isfinite(wilson_r):
            raise ValueError("wilson_r must be finite and real")
        self.wilson_r = float(wilson_r)

    def operator(self, direction, momentum_index=0):
        from .operator import ConservedVectorOperator

        return ConservedVectorOperator(direction, momentum_index, wilson_r=self.wilson_r)

    @property
    def terms(self):
        terms = []
        for d in range(4):
            forward = tuple(int(a == d) for a in range(4))
            backward = tuple(-v for v in forward)
            terms.extend((
                CurrentTerm(
                    -.5, d, forward, 1 << d, "forward", self.wilson_r,
                    f"r-gamma_{d}", field_offset=forward,
                    temporal_point_split=d == 3,
                ),
                CurrentTerm(
                    .5, d, backward, 1 << d, "backward", self.wilson_r,
                    f"r+gamma_{d}", bar_offset=forward, link_dagger=True,
                    temporal_point_split=d == 3,
                ),
            ))
        return tuple(terms)


@dataclass(frozen=True)
class ConservedVectorInsertion:
    """One conserved-current component at one stored momentum index.

    Keep the two CurrentTerm endpoints intact for term-wise contractions.
    ``operator()`` supplies the equivalent path factors to single-hadron
    consumers without discarding this insertion's term representation.
    """

    direction: int
    momentum_index: int = 0
    wilson_r: float = 1.0

    name = "conserved_vector"

    def __post_init__(self):
        direction = _integer(self.direction, "current direction")
        momentum_index = _integer(self.momentum_index, "momentum index")
        if direction not in range(4) or momentum_index < 0:
            raise ValueError("current direction must be 0..3 and momentum index nonnegative")
        wilson_r = _finite_numeric_scalar(self.wilson_r, "wilson_r")
        if not bool(np.isreal(wilson_r)):
            raise ValueError("wilson_r must be finite and real")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "momentum_index", momentum_index)
        object.__setattr__(self, "wilson_r", float(np.real(wilson_r)))

    @property
    def terms(self):
        start = 2 * self.direction
        return ConservedVectorCurrent(wilson_r=self.wilson_r).terms[start:start + 2]

    def operator(self):
        from .operator import ConservedVectorOperator

        return ConservedVectorOperator(
            self.direction,
            self.momentum_index,
            wilson_r=self.wilson_r,
        )
