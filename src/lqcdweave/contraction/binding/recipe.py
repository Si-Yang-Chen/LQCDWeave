"""Bind one gauge recipe plus one momentum recipe to symbolic rows."""

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType

import numpy as np

from lqcdweave.operators.single_hadron.api import (
    MomentumRecipe,
    Operator,
    mom_dict_to_list,
    meson55_operators,
)
from .single_hadron import Meson


_ZERO_MOMENTUM = (0, 0, 0)


def _momentum_vector(momentum):
    if isinstance(momentum, (str, bytes)):
        raise TypeError("momentum must contain three integers")
    try:
        values = tuple(momentum)
    except TypeError as exc:
        raise TypeError("momentum must contain three integers") from exc
    if len(values) != 3:
        raise ValueError("momentum must contain exactly three integers")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise TypeError("momentum entries must be integers")
    return tuple(int(value) for value in values)


def _irrep_name(irrep):
    if not isinstance(irrep, str) or not irrep:
        raise TypeError("irrep must be a nonempty string")
    if "/" in irrep:
        raise ValueError("irrep must not contain '/'")
    normalized = irrep.replace("_", "")
    if not normalized:
        raise ValueError("irrep must be a nonempty string")
    return normalized


def _copy_index(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("copy_index must be an integer")
    if value < 0:
        raise ValueError("copy_index must be nonnegative")
    return int(value)


class RecipeVertex:
    """Turn recipe names and symbolic rows into bound Meson vertices."""

    def __init__(
        self, elemental, *, momenta=None, momentum_dict=None, array_module=np, version="v2"
    ):
        """Bind one gauge recipe to one ordered momentum axis.

        The axis comes from the data being consumed (``elemental.momenta`` or
        ``elemental.momentum_list``) unless supplied explicitly as
        ``momenta`` (ordered XYZ triples) or ``momentum_dict`` (legacy
        slot -> ``"x y z"`` mapping). There is no packaged default: two
        recipes, one gauge and one momentum, define every vertex.
        """
        if momenta is not None and momentum_dict is not None:
            raise ValueError("provide only one of momenta and momentum_dict")
        data_axis = (getattr(elemental, "momenta", None)
                     or getattr(elemental, "momentum_list", None))
        data_axis = None if data_axis is None else tuple(tuple(momentum) for momentum in data_axis)
        if momentum_dict is not None or momenta is not None:
            if momenta is not None:
                values = tuple(tuple(momentum) for momentum in momenta)
            else:
                values = tuple(mom_dict_to_list(momentum_dict))
            if data_axis is not None and data_axis != values:
                raise ValueError(
                    "momentum recipe does not match the momentum axis of the consumed data")
        else:
            if data_axis is None:
                raise ValueError(
                    "the consumed data carries no momentum axis; pass the ordered "
                    "momentum recipe explicitly (momenta= or momentum_dict=)")
            values = data_axis
        self.momentum_recipe = MomentumRecipe.from_values("explicit", values)
        self.elemental = elemental
        self.version = version
        self.momentum_dict = MappingProxyType(self.momentum_recipe.as_mapping())
        self.array_module = array_module
        self.recipes = MappingProxyType({
            recipe.name: recipe for recipe in meson55_operators(version=version)
        })
        self._insertions = {}
        self._operator_terms = {}
        self._momenta = frozenset(self.momentum_recipe.momenta)

    def _requested_momentum(self, momentum):
        vector = _momentum_vector(momentum)
        if vector not in self._momenta:
            raise ValueError(
                f"momentum {vector} is not present in the momentum recipe"
            )
        return vector

    def _recipe(self, name):
        try:
            return self.recipes[name]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown gauge recipe {name!r}") from exc

    def _projection(self, recipe, momentum, irrep, occurrence):
        key = (recipe.name, momentum, irrep, occurrence)
        insertion = self._insertions.get(key)
        if insertion is None:
            insertion = recipe.insertion(self.momentum_dict).little_group_projection(
                momentum, irrep, occurrence
            )
            self._insertions[key] = insertion
        return insertion

    def symbol(self, name, momentum=(0, 0, 0), *, tag, irrep=None, copy_index=0):
        """Create one symbolic hadron, checking its momentum and irrep copy."""
        from lqcdweave.operators.correlator.api import HadronIrrep, Tag

        recipe = self._recipe(name)
        vector = self._requested_momentum(momentum)
        occurrence = _copy_index(copy_index)
        if not isinstance(tag, Tag):
            raise TypeError("tag must be a Tag")
        if vector == _ZERO_MOMENTUM:
            if occurrence != 0:
                raise ValueError("rest-frame symbols only have copy_index=0")
            requested_irrep = _irrep_name(recipe.irrep if irrep is None else irrep)
            hadron_name = recipe.name
        else:
            if irrep is None:
                raise ValueError("a moving symbol requires an explicit irrep")
            requested_irrep = _irrep_name(irrep)
            hadron_name = f"{recipe.name}/{requested_irrep}/{occurrence}"

        self._projection(recipe, vector, requested_irrep, occurrence)
        symbol_irrep = _irrep_name(recipe.irrep) if vector == _ZERO_MOMENTUM else requested_irrep
        if vector == _ZERO_MOMENTUM and requested_irrep != symbol_irrep:
            self._projection(recipe, vector, symbol_irrep, occurrence)
        return HadronIrrep(
            hadron_name,
            vector,
            symbol_irrep,
            recipe.parity,
            tag,
        )

    def _resolve_hadron_name(self, hadron_name):
        if hadron_name in self.recipes:
            return self.recipes[hadron_name], None, 0, False

        parts = hadron_name.rsplit("/", 2) if isinstance(hadron_name, str) else ()
        if len(parts) == 3:
            recipe_name, encoded_irrep, encoded_copy = parts
            recipe = self.recipes.get(recipe_name)
            if recipe is not None:
                irrep = _irrep_name(encoded_irrep)
                occurrence = _copy_index(int(encoded_copy)) if encoded_copy.isdecimal() else None
                if occurrence is not None and str(occurrence) == encoded_copy:
                    return recipe, irrep, occurrence, True
        raise KeyError(f"unknown recipe hadron name {hadron_name!r}")

    def __call__(self, row):
        from lqcdweave.operators.correlator.api import HadronIrrepRow

        if not isinstance(row, HadronIrrepRow):
            raise TypeError("RecipeVertex requires a HadronIrrepRow")

        recipe, encoded_irrep, occurrence, moving_name = self._resolve_hadron_name(
            row.hadron_name
        )
        momentum = self._requested_momentum(row.momentum)
        if row.parity != recipe.parity:
            raise ValueError(
                f"symbol parity {row.parity!r} differs from recipe parity {recipe.parity}"
            )
        if moving_name:
            if momentum == _ZERO_MOMENTUM:
                raise ValueError("moving symbol names require nonzero momentum")
            if _irrep_name(row.irrep_name) != encoded_irrep:
                raise ValueError("row irrep differs from the moving symbol irrep")
        elif momentum != _ZERO_MOMENTUM:
            raise ValueError(
                "a nonzero momentum row requires RecipeVertex.symbol(..., momentum=..., irrep=...)"
            )

        insertion = self._projection(recipe, momentum, _irrep_name(row.irrep_name), occurrence)
        # Only immutable definition terms are reused. Every call still owns a
        # fresh mutable Operator/Meson, and mode/reader views are never cached.
        key = (row.hadron_name, momentum, row.irrep_name, occurrence,
               row.row_idx, row.dagger)
        terms = self._operator_terms.get(key)
        if terms is None:
            operator = Operator(row.hadron_name, [insertion[row.row_idx](*momentum)], [1])
            if not operator.parts and all(term.profile is None for term in operator.terms):
                self._operator_terms[key] = tuple(operator.terms)
        else:
            operator = Operator.from_terms(row.hadron_name, terms)

        return Meson(
            self.elemental,
            operator,
            source=row.dagger,
            phase_convention="midpoint",
            boundary="periodic",
            array_module=self.array_module,
        )


__all__ = ["RecipeVertex"]
