"""Single-hadron coefficient definitions, independent of elemental production."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from lqcdweave.contracts.elemental import (
    BaryonComponent,
    ElementalComponent,
    integer,
    normalize_momenta,
)

if TYPE_CHECKING:
    from .baryon import SparseSpinTensor
    from .current import CurrentTerm


@dataclass(frozen=True)
class OperatorTerm:
    """One weighted meson or baryon insertion term.

    A meson term uses ``gamma`` and an ``ElementalComponent``.  A baryon term
    uses ``gamma=None``, a three-slot ``BaryonComponent`` and rank-3 ``spin``.
    ``current_term`` retains the full directed-current endpoint recipe when a
    point-split current expands into more than one meson term.
    """

    gamma: int | None
    component: ElementalComponent | BaryonComponent
    momentum: int
    momentum_vector: tuple | None = field(default=None, kw_only=True)
    coefficient: complex = 1
    anchor_time_offset: int = 0
    profile: object = None
    spin: SparseSpinTensor | None = field(default=None, kw_only=True)
    endpoint_time_offsets: tuple[int, ...] | None = field(default=None, kw_only=True)
    current_term: CurrentTerm | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.momentum_vector is not None:
            object.__setattr__(self, "momentum_vector", _integer_offsets(
                self.momentum_vector, 3, "momentum_vector"))
        if isinstance(self.component, ElementalComponent):
            if self.gamma is None:
                raise ValueError("meson terms require a gamma index")
            g = integer(self.gamma, "gamma", 0)
            if g > 15 or self.spin is not None:
                raise ValueError("meson terms require gamma 0..15 and no rank-3 spin")
            if self.endpoint_time_offsets is not None:
                raise ValueError(
                    "two-leg endpoints come from the component path or CurrentTerm metadata"
                )
        elif isinstance(self.component, BaryonComponent):
            if self.gamma is not None:
                raise ValueError("baryon terms use spin, not a gamma index")
            from .baryon import SparseSpinTensor

            if not isinstance(self.spin, SparseSpinTensor):
                raise TypeError("baryon terms require a SparseSpinTensor")
            if self.profile is not None:
                raise ValueError("baryon spectral profiles are not supported")
            offsets = ((0, 0, 0) if self.endpoint_time_offsets is None else
                       _integer_offsets(self.endpoint_time_offsets, 3, "endpoint_time_offsets"))
            object.__setattr__(self, "endpoint_time_offsets", offsets)
        else:
            raise TypeError("component must be ElementalComponent or BaryonComponent")
        c = complex(self.coefficient)
        if not np.isfinite(c):
            raise ValueError("coefficient must be finite")
        object.__setattr__(self, "coefficient", c)
        integer(self.momentum, "momentum index", 0)
        integer(self.anchor_time_offset, "anchor offset")
        if self.current_term is not None:
            from .current import CurrentTerm

            if not isinstance(self.current_term, CurrentTerm):
                raise TypeError("current_term must be a CurrentTerm")
            if not isinstance(self.component, ElementalComponent):
                raise ValueError("directed-current metadata applies to two-leg terms")

    @classmethod
    def from_baryon(
        cls,
        component: BaryonComponent,
        spin: SparseSpinTensor,
        momentum: int = 0,
        coefficient: complex = 1,
        *,
        anchor_time_offset: int = 0,
        endpoint_time_offsets=(0, 0, 0),
    ):
        """Create one explicit ordered three-leg term."""
        return cls(
            None,
            component,
            momentum,
            coefficient,
            anchor_time_offset,
            spin=spin,
            endpoint_time_offsets=endpoint_time_offsets,
        )

    @property
    def rank(self) -> int:
        return 3 if isinstance(self.component, BaryonComponent) else 2


def _integer_offsets(values, expected, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain {expected} integers")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain {expected} integers") from exc
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} integers")
    return tuple(integer(value, f"{name}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True)
class InsertionTerm:
    """Unbound row term; momentum is selected when an insertion row is called."""

    component: ElementalComponent | BaryonComponent
    gamma: int | None = None
    coefficient: complex = 1
    anchor_time_offset: int = 0
    profile: object = None
    spin: SparseSpinTensor | None = field(default=None, kw_only=True)
    endpoint_time_offsets: tuple[int, ...] | None = field(default=None, kw_only=True)
    current_term: CurrentTerm | None = field(default=None, kw_only=True)

    def __post_init__(self):
        # Reuse the canonical term's validation so insertion and operator
        # construction cannot disagree about rank, spin or endpoint fields.
        term = OperatorTerm(
            self.gamma,
            self.component,
            0,
            self.coefficient,
            self.anchor_time_offset,
            self.profile,
            spin=self.spin,
            endpoint_time_offsets=self.endpoint_time_offsets,
            current_term=self.current_term,
        )
        object.__setattr__(self, "coefficient", term.coefficient)
        object.__setattr__(self, "anchor_time_offset", term.anchor_time_offset)
        object.__setattr__(self, "endpoint_time_offsets", term.endpoint_time_offsets)

    def at_momentum(self, momentum: int, *, coefficient=None, profile=None):
        return OperatorTerm(
            self.gamma,
            self.component,
            momentum,
            self.coefficient if coefficient is None else coefficient,
            self.anchor_time_offset,
            self.profile if profile is None else profile,
            spin=self.spin,
            endpoint_time_offsets=self.endpoint_time_offsets,
            current_term=self.current_term,
        )


@dataclass
class InsertionRowMom:
    row: object
    momentum: int
    profile: object = None
    momentum_vector: tuple | None = None


class InsertionRow:
    def __init__(self, row, momentum_dict, profile=None, *, momentum_identity=False):
        self.row, self.momentum_dict, self.profile = row, momentum_dict, profile
        self.momentum_identity = momentum_identity

    def __call__(self, npx, npy, npz):
        momenta = normalize_momenta(self.momentum_dict)
        requested = (integer(npx, "momentum x"), integer(npy, "momentum y"),
                     integer(npz, "momentum z"))
        try:
            momentum = momenta.index(requested)
        except ValueError as exc:
            raise ValueError(f"momentum {requested} is not present in the insertion dictionary") from exc
        return InsertionRowMom(self.row, momentum, self.profile,
                               requested if self.momentum_identity else None)


class Operator:
    """Build from insertion rows, or from explicit bitmask terms via from_terms.

    Insertion rows use the cyclic Gamma naming convention; masks 5/13 have
    their sign converted once here. Explicit OperatorTerm uses bitmask Gamma.
    """

    def __init__(self, name, insertion_rows, coefficients):
        _validate_operator_name(name)
        if len(insertion_rows) != len(coefficients):
            raise ValueError("one coefficient per insertion row is required")
        from .baryon import BaryonOperatorTerm
        from .current import ConservedVectorInsertion
        if insertion_rows and all(isinstance(row, ConservedVectorInsertion) for row in insertion_rows):
            terms = []
            for insertion, weight in zip(insertion_rows, coefficients):
                definition = insertion.operator()
                terms.extend(_scale_operator_term(term, weight) for term in definition.terms)
            self._set_explicit(name, terms)
            return

        if insertion_rows and all(_is_generic_insertion_row(row, BaryonOperatorTerm)
                                  for row in insertion_rows):
            terms = []
            for row, weight in zip(insertion_rows, coefficients):
                for item in row.row:
                    if isinstance(item, InsertionTerm):
                        term = item.at_momentum(row.momentum, coefficient=item.coefficient * weight,
                                                profile=row.profile)
                        term = _scale_current_term_metadata(term, weight)
                    elif isinstance(item, OperatorTerm):
                        term = _scale_operator_term(replace(item, momentum=row.momentum), weight)
                    elif isinstance(item, BaryonOperatorTerm):
                        term = OperatorTerm.from_baryon(
                            item.component, item.spin, row.momentum,
                            item.coefficient * weight,
                            endpoint_time_offsets=getattr(item, "endpoint_time_offsets", (0, 0, 0)),
                        )
                    terms.append(replace(term, momentum_vector=row.momentum_vector))
            self._set_explicit(name, terms)
            return

        self.name, self.parts = name, []
        for row, weight in zip(insertion_rows, coefficients):
            if len(row.row) % 2:
                raise ValueError("row requires alternating gamma and spatial terms")
            for i in range(0, len(row.row), 2):
                g = row.row[i]
                sign = -1 if g in (5, 13) else 1
                self.parts.extend([g, [[sign * weight * c, key, row.momentum, row.profile]
                                       for c, key in row.row[i + 1]]])
        self._explicit = None
        if any(row.momentum_vector is not None for row in insertion_rows):
            # Preserve each selected physical label alongside its positional slot.
            explicit = []
            for row, weight in zip(insertion_rows, coefficients):
                for g, entries in zip(row.row[::2], row.row[1::2]):
                    for c, key in entries:
                        explicit.append(OperatorTerm(
                            g, key if isinstance(key, ElementalComponent) else ElementalComponent(key),
                            row.momentum, (-1 if g in (5, 13) else 1) * weight * c,
                            profile=row.profile, momentum_vector=row.momentum_vector))
            self._explicit = tuple(explicit)
        self.current_terms, self.current_term_ranges = (), ()
        tuple(self.terms)  # validate before exposing a definition

    def _set_explicit(self, name, terms):
        _validate_operator_name(name)
        self.name, self.parts = name, []
        self._explicit = tuple(terms)
        if not self._explicit or any(not isinstance(term, OperatorTerm) for term in self._explicit):
            raise ValueError("nonempty OperatorTerm definitions are required")
        ranks = {term.rank for term in self._explicit}
        if len(ranks) != 1:
            raise ValueError("one Operator cannot mix two-leg and three-leg terms")
        if ranks == {3} and len({term.spin.basis for term in self._explicit}) != 1:
            raise ValueError("three-leg Operator terms must use one spin basis")
        self.current_terms, self.current_term_ranges = _current_term_views(self._explicit)

    @classmethod
    def from_terms(cls, name, terms):
        obj = cls.__new__(cls)
        from .baryon import BaryonOperatorTerm

        converted = []
        for term in terms:
            if isinstance(term, OperatorTerm):
                converted.append(term)
            elif isinstance(term, BaryonOperatorTerm):
                converted.append(OperatorTerm.from_baryon(
                    term.component, term.spin, term.momentum, term.coefficient,
                    endpoint_time_offsets=getattr(term, "endpoint_time_offsets", (0, 0, 0)),
                ))
            else:
                raise TypeError("terms must be OperatorTerm records")
        obj._set_explicit(name, converted)
        return obj

    @property
    def terms(self):
        if self._explicit is not None:
            return self._explicit
        return tuple(
            OperatorTerm(
                g,
                key if isinstance(key, ElementalComponent) else ElementalComponent(key),
                m,
                c,
                profile=p,
            )
            for g, entries in zip(self.parts[::2], self.parts[1::2])
            for c, key, m, p in entries
        )

    def set_gamma(self, i_row, gamma_idx):
        if self._explicit is not None:
            raise ValueError("explicit terms are immutable; construct a new operator")
        gamma_idx = integer(gamma_idx, "gamma", 0)
        if gamma_idx > 15:
            raise ValueError("gamma must be 0..15")
        self.parts[2 * i_row] = gamma_idx

    def set_derivative(self, i_row, i_term, deriv_idx):
        if self._explicit is not None:
            raise ValueError("explicit terms are immutable; construct a new operator")
        self.parts[2 * i_row + 1][i_term][1] = deriv_idx


def _validate_operator_name(name):
    if not isinstance(name, str):
        raise TypeError("Operator.name must be a nonempty string")
    if not name.strip():
        raise ValueError("Operator.name must be a nonempty string")


def _scale_operator_term(term, coefficient):
    coefficient = complex(coefficient)
    return _scale_current_term_metadata(
        replace(term, coefficient=term.coefficient * coefficient), coefficient,
    )


def _scale_current_term_metadata(term, coefficient):
    current_term = term.current_term
    if current_term is not None:
        current_term = replace(current_term, coefficient=current_term.coefficient * coefficient)
    return replace(term, current_term=current_term)


def _current_term_views(terms):
    current_terms = []
    ranges = []
    index_by_term = {}
    for index, term in enumerate(terms):
        current = term.current_term
        if current is None:
            continue
        try:
            current_index = index_by_term[current]
        except KeyError:
            current_index = len(current_terms)
            index_by_term[current] = current_index
            current_terms.append(current)
            ranges.append([])
        ranges[current_index].append(index)
    return tuple(current_terms), tuple(tuple(group) for group in ranges)


def _is_generic_insertion_row(row, baryon_operator_term):
    return (
        isinstance(row, InsertionRowMom)
        and isinstance(row.row, (tuple, list))
        and bool(row.row)
        and all(isinstance(term, (InsertionTerm, OperatorTerm, baryon_operator_term))
                for term in row.row)
    )


class OperatorDisplacement(Operator):
    def __init__(self, name, insertion_rows, coefficients, distances):
        if len(insertion_rows) != len(distances):
            raise ValueError("one distance per insertion row is required")
        converted = []
        for entry, distance in zip(insertion_rows, distances):
            distance = integer(distance, "distance", 0)
            row = []
            for g, terms in zip(entry.row[::2], entry.row[1::2]):
                if any(key != 0 for c, key in terms):
                    raise ValueError("distance insertion requires identity derivative")
                row.extend([g, [[c, distance] for c, key in terms]])
            converted.append(InsertionRowMom(row, entry.momentum, entry.profile, entry.momentum_vector))
        super().__init__(name, converted, coefficients)


class OperatorPath(Operator):
    path_version = "v2"

    def __init__(self, name, terms, *, current_terms=(), current_term_ranges=()):
        self.name, self.parts = name, []
        parsed = []
        for term in terms:
            if len(term) not in (4, 5):
                raise ValueError("path terms require gamma,path,momentum,coefficient[,anchor offset]")
            g, path, m, c = term[:4]
            offset = term[4] if len(term) == 5 else 0
            parsed.append(OperatorTerm(g, ElementalComponent.path(path), m, c, offset))
        if not parsed:
            raise ValueError("path operator must have terms")
        current_terms = tuple(current_terms)
        current_term_ranges = tuple(tuple(index for index in group) for group in current_term_ranges)
        if current_terms and len(current_terms) != len(current_term_ranges):
            raise ValueError("one operator-term range is required per CurrentTerm")
        if any(not group or any(type(index) is not int or not 0 <= index < len(parsed)
                                for index in group)
               for group in current_term_ranges):
            raise ValueError("current term ranges must index the path operator terms")
        terms_with_current = list(parsed)
        for current, indices in zip(current_terms, current_term_ranges):
            for index in indices:
                if terms_with_current[index].current_term is not None:
                    raise ValueError("a path term cannot refer to two CurrentTerm records")
                terms_with_current[index] = replace(terms_with_current[index], current_term=current)
        self._set_explicit(name, terms_with_current)

    @property
    def path_terms(self):
        return tuple((t.gamma, t.component.key, t.momentum, t.coefficient, t.anchor_time_offset) for t in self.terms)

    @classmethod
    def from_operator(cls, operator, paths):
        paths = tuple(paths)
        terms = []
        for term in operator.terms:
            if (term.rank != 2 or term.component != ElementalComponent(0)
                    or term.profile is not None):
                raise ValueError("path expansion requires identity spatial component and no profile")
            for path, coefficient in paths:
                current_term = term.current_term
                if current_term is not None:
                    current_term = replace(
                        current_term,
                        coefficient=current_term.coefficient * coefficient,
                    )
                terms.append(OperatorTerm(
                    term.gamma,
                    ElementalComponent.path(path),
                    term.momentum,
                    term.coefficient * coefficient,
                    term.anchor_time_offset,
                    spin=None,
                    current_term=current_term,
                ))
        return cls.from_terms(operator.name, terms)

    @classmethod
    def from_current(cls, current, direction, momentum_index=0):
        """Consume a current recipe exposing terms with explicit Wilson factors."""
        integer(direction, "current direction", 0)
        if direction > 3:
            raise ValueError("current direction must be 0..3")
        terms = []
        for term in current.terms:
            if term.direction not in (direction, -1):
                continue
            weight = term.coefficient * term.normalization
            offset = term.bar_offset[3]
            if term.link == "none":
                terms.append(OperatorTerm(
                    term.gamma_index,
                    ElementalComponent.path(()),
                    momentum_index,
                    weight,
                    offset,
                    current_term=term,
                ))
            else:
                if term.link not in ("forward", "backward"):
                    raise ValueError("unknown current link")
                d = term.direction
                digit = (d if d < 3 else 6) if term.link == "forward" else (d + 3 if d < 3 else 7)
                terms.extend((
                    OperatorTerm(
                        0,
                        ElementalComponent.path((digit,)),
                        momentum_index,
                        weight * term.wilson_r,
                        offset,
                        current_term=term,
                    ),
                    OperatorTerm(
                        term.gamma_index,
                        ElementalComponent.path((digit,)),
                        momentum_index,
                        weight * (-1 if term.link == "forward" else 1),
                        offset,
                        current_term=term,
                    ),
                ))
        return cls.from_terms(current.name, terms)


class ConservedVectorOperator(Operator):
    """Ready-to-bind Wilson conserved vector current as ordinary Operator terms."""

    def __init__(self, direction: int, momentum_index: int = 0, *, wilson_r: float = 1.0):
        from .current import ConservedVectorInsertion

        insertion = ConservedVectorInsertion(direction, momentum_index, wilson_r)
        definition = OperatorPath.from_current(insertion, direction, momentum_index)
        self.direction = insertion.direction
        self.momentum_index = insertion.momentum_index
        self.wilson_r = insertion.wilson_r
        self._set_explicit(insertion.name, definition.terms)
