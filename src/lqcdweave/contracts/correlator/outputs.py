"""Immutable expression/coordinate output descriptors."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .inputs import CoordinateSpec

@dataclass(frozen=True)
class OutputNode:
    kind: str
    value: object = None
    children: tuple["OutputNode", ...] = ()
    keys: tuple = ()
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class OutputRecipe:
    root: OutputNode
    term_ids: tuple[str, ...]
    coordinates: CoordinateSpec
    dtype: str | None = None
    # None preserves legacy per-node dtype semantics. New narrow tasks keep
    # output arithmetic wide and cast numeric fields only at their boundary.
    intermediate_dtype: str | None = None

    def __post_init__(self):
        if self.dtype is not None:
            dtype = np.dtype(self.dtype)
            if dtype not in tuple(map(np.dtype, ('float32','float64','complex64','complex128'))):
                raise TypeError('unsupported output computation dtype')
            object.__setattr__(self, 'dtype', dtype.name)
        if self.intermediate_dtype is not None:
            intermediate = np.dtype(self.intermediate_dtype)
            if self.dtype is None or (np.dtype(self.dtype).kind, intermediate.kind) not in {
                    ('f', 'f'), ('c', 'c')} or intermediate not in {
                    np.dtype('float64'), np.dtype('complex128')}:
                raise TypeError('output intermediate dtype must be a wide dtype of the output kind')
            object.__setattr__(self, 'intermediate_dtype', intermediate.name)

