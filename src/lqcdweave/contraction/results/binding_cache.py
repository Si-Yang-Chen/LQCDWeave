"""Process-local output callables, shared by fixed and complete calculations."""
from dataclasses import fields, is_dataclass
import numpy as np

from .core import bind_output_assembly


def _immutable(value, seen):
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return True
    if isinstance(value, np.number):
        return True
    if type(value).__module__.startswith('sympy.'):
        from sympy import Basic
        return isinstance(value, Basic)
    if id(value) in seen:
        return True
    seen.add(id(value))
    if type(value) is tuple:
        return all(_immutable(item, seen) for item in value)
    if (is_dataclass(value) and type(value).__module__.startswith('lqcdweave.contracts.')
            and type(value).__dataclass_params__.frozen):
        return all(_immutable(getattr(value, field.name), seen) for field in fields(value))
    return False


class OutputAssemblyCache:
    """One plan's fixed-coordinate assemblers; evaluations own their results."""
    def __init__(self):
        self._binding = None

    def bind(self, owner, recipes, *, debug=False):
        previous = None if debug else self._binding
        if previous is not None and previous[0] is owner:
            return previous[1]
        result = tuple(bind_output_assembly(recipe, debug=debug) for recipe in recipes)
        if not debug and _immutable(recipes, set()):
            self._binding = (owner, result)
        return result
