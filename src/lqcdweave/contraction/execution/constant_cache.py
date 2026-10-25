"""Budgeted immutable recipe constants, separate from numerical input caches."""
from lqcdweave.contracts.correlator.api import ArrayConstant, TensorPreparation


MAX_CONSTANT_BYTES = 8 * 1024**2
MAX_CONSTANTS = 128


def describe_constants(*tables):
    """Bound the catalog; return saved bytes only, never reader-owned arrays."""
    values = {}
    for table in tables:
        for prep in table.values():
            recipe = prep.recipe
            if type(recipe) is not TensorPreparation:
                continue
            for item in recipe.constants:
                if type(item) is ArrayConstant and item not in values:
                    values[item] = None
                    if len(values) == MAX_CONSTANTS:
                        return frozenset(values)
    return frozenset(values)


class DeviceConstantCache:
    """A leased private pool's constants; allocation extents count toward capacity."""
    def __init__(self):
        self.values = {}
        self.items = frozenset()
        self.capacity = self.nbytes = 0

    def clear(self):
        self.values.clear()
        self.items = frozenset()
        self.capacity = self.nbytes = 0

    def configure(self, items, capacity):
        if items != self.items or self.nbytes > capacity:
            self.clear()
        self.items, self.capacity = items, capacity

    def get(self, item, create):
        previous = self.values.get(item)
        if previous is not None:
            return previous
        value = create()
        if item not in self.items or len(self.values) >= MAX_CONSTANTS:
            return value
        try:
            # CuPy's pooled allocation can be larger than logical ndarray bytes.
            # Unknown allocation protocols retain the original run-local path.
            extent = int(value.data.mem.size)
        except (AttributeError, TypeError, ValueError):
            return value
        if value.nbytes <= extent <= self.capacity - self.nbytes:
            self.values[item] = value
            self.nbytes += extent
        return value
