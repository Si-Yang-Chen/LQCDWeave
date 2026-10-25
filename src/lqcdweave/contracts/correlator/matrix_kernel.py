"""Array layouts for the fixed meson K/H template; no execution or IO."""
from dataclasses import dataclass
from math import prod
import numpy as np
from .indices import AxisPattern


@dataclass(frozen=True, slots=True)
class MatrixKernelSpec:
    kind: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    partitions: tuple[int, ...] | None = None
    row_block: int = 1
    mode_block: bool = False

    def __post_init__(self):
        shapes = tuple(tuple(shape) for shape in self.shapes)
        dtypes = tuple(np.dtype(dtype).name for dtype in self.dtypes)
        count = {'connected': 6, 'loop': 3}.get(self.kind)
        if count is None or len(shapes) != count or len(dtypes) != count:
            raise ValueError('matrix template requires six connected or three loop inputs')
        if any(type(n) is not int or n < 0 for shape in shapes for n in shape):
            raise ValueError('matrix dimensions must be nonnegative integers')
        if any(np.dtype(dtype).kind not in 'fc' for dtype in dtypes):
            raise TypeError('matrix inputs require floating or complex dtypes')
        if type(self.row_block) is not int or self.row_block <= 0:
            raise ValueError('row_block must be a positive integer')
        p = shapes[0]
        if type(self.mode_block) is not bool:
            raise TypeError('mode_block must be bool')
        if self.mode_block and self.partitions is not None:
            raise ValueError('a selected mode block cannot be partitioned again')
        if len(p) != 4 or p[:2] != (4, 4) or min(p[2:]) <= 0:
            raise ValueError('matrix propagators require [4,4,sink_modes,source_modes]')
        ne = p[2]
        if self.kind == 'connected':
            if len(shapes[1])!=4 or shapes[1][:2]!=(4,4) or min(shapes[1][2:])<=0:
                raise ValueError('invalid second matrix propagator layout')
            pairs = ((shapes[2], shapes[4]), (shapes[3], shapes[5]))
            tails = ((4,4,p[3],shapes[1][2]), (4,4,shapes[1][3],p[2]))
        else:
            pairs = ((shapes[1], shapes[2]),)
            tails = ((4,4,p[3],p[2]),)
        if not self.mode_block and any(shape[-2:]!=(ne,ne) for shape in shapes[:2 if self.kind=='connected' else 1]):
            raise ValueError('unselected matrix templates require equal square mode axes')
        for (vertex, weight), tail in zip(pairs,tails,strict=True):
            if (len(vertex) != 5 or vertex[1:] != tail or len(weight) != 2
                    or weight[1] != vertex[0]):
                raise ValueError('invalid matrix vertex or coefficient layout')
        boundaries = (0, ne) if self.partitions is None else tuple(self.partitions)
        if (len(boundaries) < 2 or any(type(n) is not int for n in boundaries)
                or boundaries[0] != 0 or boundaries[-1] != ne
                or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError('partitions must increase from zero to Ne')
        object.__setattr__(self, 'shapes', shapes)
        object.__setattr__(self, 'dtypes', dtypes)
        if self.partitions is not None:
            object.__setattr__(self, 'partitions', boundaries)

    @property
    def ne(self):
        return self.shapes[0][-1]

    @property
    def widths(self):
        boundaries = (0, self.ne) if self.partitions is None else self.partitions
        return tuple(b - a for a, b in zip(boundaries, boundaries[1:]))

    @property
    def dtype(self):
        return np.result_type(*(np.dtype(dtype) for dtype in self.dtypes))

    @property
    def output_shape(self):
        q = len(self.widths)
        pc = q * (q + 1) // 2
        if self.kind == 'connected':
            shape = (self.shapes[4][0], self.shapes[5][0])
            return shape if self.partitions is None else (*shape, pc, pc)
        shape = (self.shapes[2][0],)
        return shape if self.partitions is None else (*shape, pc)

    @property
    def input_bytes(self):
        return sum(prod(shape) * np.dtype(dtype).itemsize
                   for shape, dtype in zip(self.shapes, self.dtypes))

    @property
    def promotion_bytes(self):
        return sum(prod(shape) * self.dtype.itemsize
                   for shape, dtype in zip(self.shapes, self.dtypes)
                   if np.dtype(dtype) != self.dtype)

    @property
    def equation(self):
        if self.kind == 'connected':
            # Ordered spin, mode, component, linear-output and partition axes.
            inputs = ((0,1,2,3), (4,5,6,7), (8,1,4,3,6),
                      (9,5,0,7,2), (10,8), (11,9))
            return AxisPattern(inputs, (10,11)) if self.partitions is None else AxisPattern(
                (*inputs, (12,3,6), (13,7,2)), (10,11,12,13))
        inputs = ((0,1,2,3), (4,1,0,3,2), (5,4))
        return AxisPattern(inputs, (5,)) if self.partitions is None else AxisPattern(
            (*inputs, (6,3,2)), (5,6))

    @property
    def projection_shape(self):
        q = len(self.widths)
        return (q*(q+1)//2, self.ne, self.ne)

    @property
    def action_shapes(self):
        count = 0 if self.partitions is None else (2 if self.kind == 'connected' else 1)
        return self.shapes + (self.projection_shape,)*count

    @property
    def action_dtypes(self):
        return self.dtypes + (self.dtype.name,)*(len(self.action_shapes)-len(self.shapes))

    @property
    def projection_blocks(self):
        """(PC, row start/stop, column start/stop, coefficient) declarations."""
        boundaries = (0,self.ne) if self.partitions is None else self.partitions
        result, pc = [], 0
        for a in range(len(boundaries)-1):
            for b in range(a,len(boundaries)-1):
                result.append((pc,boundaries[a],boundaries[a+1],boundaries[b],boundaries[b+1],
                               1.0 if a==b else float(1/np.sqrt(2))))
                if a != b:
                    result.append((pc,boundaries[b],boundaries[b+1],boundaries[a],boundaries[a+1],float(1/np.sqrt(2))))
                pc += 1
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MatrixKernelStageEstimate:
    name: str
    flops: int
    traffic_bytes: int
    kernel_calls: int
    seconds: float


@dataclass(frozen=True, slots=True)
class MatrixKernelEstimate:
    spec: MatrixKernelSpec
    stages: tuple[MatrixKernelStageEstimate, ...]
    input_bytes: int
    promotion_bytes: int
    output_bytes: int
    retained_product_bytes: int
    temporary_bytes: int
    workspace_reserve_bytes: int

    @property
    def peak_array_bytes(self):
        return (self.input_bytes + self.promotion_bytes + self.output_bytes
                + self.retained_product_bytes + self.temporary_bytes
                + self.workspace_reserve_bytes)

    @property
    def seconds(self):
        return sum(stage.seconds for stage in self.stages)


__all__ = ['MatrixKernelSpec', 'MatrixKernelStageEstimate', 'MatrixKernelEstimate']
