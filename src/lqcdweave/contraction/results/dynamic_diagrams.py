"""Output assembly for dynamic simple and three-line baryon diagrams.

The grouping and dtype-preserving fast path were migrated from the now-retired
``contraction.diagram_adapter``. These functions consume completed
term values; they do not plan or execute contractions.
"""

from __future__ import annotations

import importlib
import operator
from typing import Any

import numpy as np


def _backend_module(backend: Any) -> tuple[Any, str]:
    if isinstance(backend, str):
        name = backend.lower()
        if name not in {"numpy", "cupy"}:
            raise NotImplementedError("dynamic diagram results support NumPy or CuPy")
        try:
            backend = importlib.import_module(name)
        except ImportError as exc:
            raise NotImplementedError(f"{name} is not installed") from exc
    if backend is np:
        return backend, "numpy"
    if getattr(backend, "__name__", "").rsplit(".", 1)[-1].lower() == "cupy":
        try:
            actual_cupy = importlib.import_module("cupy")
        except ImportError as exc:
            raise NotImplementedError("dynamic diagram results support CuPy only when installed") from exc
        if backend is actual_cupy:
            return backend, "cupy"
    raise NotImplementedError("dynamic diagram results support the NumPy backend or CuPy only")


def _sequence(value: Any, role: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{role} must be a sequence, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be a sequence") from exc


def _output_recipes(value: Any, term_count: int) -> tuple[tuple[int, ...], ...]:
    raw_recipes = _sequence(value, "output_recipes")
    recipes = []
    seen = set()
    for recipe_index, raw_recipe in enumerate(raw_recipes):
        raw_terms = _sequence(raw_recipe, f"output_recipes[{recipe_index}]")
        recipe = []
        for raw_index in raw_terms:
            if isinstance(raw_index, (bool, np.bool_)):
                raise TypeError("output recipe term indexes must be integers, not bool")
            try:
                term_index = operator.index(raw_index)
            except TypeError as exc:
                raise TypeError("output recipe term indexes must be integers") from exc
            if not 0 <= term_index < term_count:
                raise ValueError(f"output recipe term index {term_index} is outside the term values")
            if term_index in seen:
                raise ValueError(f"output recipe term index {term_index} occurs more than once")
            seen.add(term_index)
            recipe.append(term_index)
        recipes.append(tuple(recipe))
    if seen != set(range(term_count)):
        missing = sorted(set(range(term_count)) - seen)
        raise ValueError(f"output recipes must cover every term value exactly once (missing={missing})")
    return tuple(recipes)


def _single_term_outputs(values, recipes):
    if not recipes or len(values) != len(recipes) or any(len(recipe) != 1 for recipe in recipes):
        return None
    arranged = [None] * len(recipes)
    reference_shape = None
    reference_dtype = None
    for recipe_index, recipe in enumerate(recipes):
        index = recipe[0]
        if not isinstance(index, int) or not 0 <= index < len(values):
            return None
        value = values[index]
        dtype = getattr(value, "dtype", None)
        if dtype is None or str(dtype) not in {"float64", "complex128"}:
            return None
        if reference_dtype is None:
            reference_dtype = str(dtype)
        elif str(dtype) != reference_dtype:
            return None
        shape = tuple(value.shape)
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            return None
        arranged[recipe_index] = value
    if any(value is None for value in arranged):
        return None
    return arranged


def _synchronise(xp: Any, backend_name: str) -> None:
    if backend_name == "cupy":
        xp.cuda.get_current_stream().synchronize()


def assemble_simple_diagram_outputs(values, output_recipes, backend):
    """Multiply each diagram's term outputs, then stack diagrams in order."""
    xp, backend_name = _backend_module(backend)
    term_values = _sequence(values, "values")
    recipes = _output_recipes(output_recipes, len(term_values))

    batched = _single_term_outputs(term_values, recipes)
    if batched is not None:
        result = xp.multiply(1.0, xp.stack([xp.asarray(value) for value in batched]))
    else:
        diagram_values = [1.0 for _ in recipes]
        for recipe_index, recipe in enumerate(recipes):
            for term_index in recipe:
                diagram_values[recipe_index] = diagram_values[recipe_index] * term_values[term_index]
        result = (
            xp.stack([xp.asarray(value) for value in diagram_values])
            if diagram_values else xp.asarray([])
        )
    _synchronise(xp, backend_name)
    return result


def assemble_baryon_diagram_outputs(values, backend):
    """Stack one scalar per baryon term; preserve the legacy empty result."""
    xp, backend_name = _backend_module(backend)
    term_values = _sequence(values, "values")
    result = (
        xp.stack([xp.asarray(value) for value in term_values])
        if term_values
        else xp.empty((0,), dtype=np.complex128)
    )
    _synchronise(xp, backend_name)
    return result


__all__ = ["assemble_simple_diagram_outputs", "assemble_baryon_diagram_outputs"]
