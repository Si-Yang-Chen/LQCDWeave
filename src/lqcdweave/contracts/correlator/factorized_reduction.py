"""An explicit thin, wide reduction of an integer factor graph.

The two large inputs are reduced to internal wide partials. Remaining small
factors are combined in wide precision before the declared owned output cast.
Input and optional output precision are explicit; no numerical constant is stored.
"""
from dataclasses import dataclass, field
import numpy as np

from .factorized_geometry import bind_geometry
from .indices import AxisPattern

MAX_FACTORIZED_TILE_ELEMENTS = 65536


@dataclass(frozen=True)
class FactorizedWideReduction:
    equation: AxisPattern
    shapes: tuple[tuple[int, ...], ...]
    large_operands: tuple[int, int]
    dtype: str = 'complex64'
    # None retains the original un-tiled archive/core identity and arithmetic.
    tile_elements: int | None = field(default=None, repr=False)
    # None preserves existing archives' logical output cast. Newly planned
    # terminal values can remain wide until OutputRecipe publishes its field.
    output_dtype: str | None = field(default=None, repr=False)
    # An independently budgeted numerical implementation, not an X toggle.
    gram_strategy: str = field(default='reduction', repr=False)

    def __post_init__(self):
        shapes = tuple(tuple(shape) for shape in self.shapes)
        large = tuple(self.large_operands)
        object.__setattr__(self, 'shapes', shapes)
        object.__setattr__(self, 'large_operands', large)
        object.__setattr__(self, 'dtype', np.dtype(self.dtype).name)
        if self.output_dtype is not None:
            output_dtype = np.dtype(self.output_dtype).name
            if output_dtype not in ('complex64', 'complex128'):
                raise ValueError('thin wide outputs require complex64 or complex128')
            object.__setattr__(self, 'output_dtype', None if output_dtype == self.dtype else output_dtype)
        if self.tile_elements is not None and (type(self.tile_elements) is not int
                                               or not 1 <= self.tile_elements <= MAX_FACTORIZED_TILE_ELEMENTS):
            raise ValueError('thin wide epilogue tiles must contain 1..65536 elements')
        if self.gram_strategy not in ('reduction','gemm'):
            raise ValueError('thin wide Gram strategy must be reduction or gemm')
        if (type(self.equation) is not AxisPattern or len(shapes) not in ((4,) if self.tile_elements is None else (3,4))
                or len(self.equation.inputs) != len(shapes) or len(large) != 2
                or any(type(i) is not int or not 0 <= i < len(shapes) for i in large)
                or large[0] >= large[1] or self.dtype != 'complex64'):
            raise ValueError('thin wide reduction requires four complex64 factors and two ordered large operands')
        object.__setattr__(self, '_geometry', bind_geometry(
            self.equation, shapes, large, self.tile_elements, self.result_dtype,self.gram_strategy))

    def __getattr__(self, name):
        if name == '_geometry':
            # Public JSON archives construct normally. Older Python/spawn
            # pickle state can omit this non-field derived metadata.
            self.__post_init__()
            return object.__getattribute__(self, name)
        raise AttributeError(name)

    @property
    def result_dtype(self):
        return self.output_dtype or self.dtype

    @property
    def dimensions(self):
        # Callers receive a fresh dictionary, never a mutable shared cache.
        return dict(self._geometry.dimensions)

    @property
    def small_operands(self):
        return self._geometry.small_operands

    @property
    def small_reduced_axes(self):
        return self._geometry.small_reduced_axes

    @property
    def small_reduction_elements(self):
        return self._geometry.small_reduction_elements

    @property
    def gram_axes(self):
        return self._geometry.gram_axes

    @property
    def reduced_axes(self):
        return self._geometry.reduced_axes

    @property
    def gram_elements(self):
        return self._geometry.gram_elements

    @property
    def epilogue_elements(self):
        return self._geometry.epilogue_elements

    @property
    def reduction_elements(self):
        return self._geometry.reduction_elements

    @property
    def output_shape(self):
        return self._geometry.output_shape

    @property
    def output_elements(self):
        return self._geometry.output_elements

    @property
    def blocks(self):
        return self._geometry.blocks

    @property
    def partial_bytes(self):
        return self._geometry.partial_bytes

    @property
    def tile_width(self):
        return self._geometry.tile_width

    @property
    def tiles(self):
        return self._geometry.tiles

    @property
    def partial_elements(self):
        return self._geometry.partial_elements

    @property
    def accumulator_bytes(self):
        return self._geometry.accumulator_bytes

    @property
    def kernel_calls(self):
        return self._geometry.kernel_calls

    @property
    def workspace_bytes(self):
        return self._geometry.workspace_bytes

    @property
    def flops(self):
        return self._geometry.flops

    @property
    def traffic_bytes(self):
        return self._geometry.traffic_bytes
