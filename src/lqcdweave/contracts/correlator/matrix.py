"""Matrix-level task records, without expanded cells or executable callbacks."""
from dataclasses import dataclass
import numpy as np

from .vertices import VertexFactorMetadata
from .outputs import OutputNode


@dataclass(frozen=True, slots=True, eq=False)
class MatrixLinearAxis:
    labels: tuple[str, ...]
    offsets: np.ndarray
    component_ids: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        labels = tuple(self.labels)
        offsets = np.array(self.offsets, dtype=np.int64, copy=True)
        components = np.array(self.component_ids, dtype=np.int64, copy=True)
        coefficients = np.array(self.coefficients, dtype=np.complex128, copy=True)
        if any(array.ndim != 1 for array in (offsets, components, coefficients)):
            raise ValueError('matrix coefficient tables must be one-dimensional')
        if (len(offsets) != len(labels) + 1 or offsets[0] != 0
                or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(components)
                or len(coefficients) != len(components) or np.any(components < 0)):
            raise ValueError('invalid matrix coefficient row table')
        if not np.all(np.isfinite(coefficients)):
            raise ValueError('matrix coefficients must be finite')
        for array in (offsets, components, coefficients):
            array.flags.writeable = False
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'offsets', offsets)
        object.__setattr__(self, 'component_ids', components)
        object.__setattr__(self, 'coefficients', coefficients)

    @property
    def nbytes(self):
        return self.offsets.nbytes + self.component_ids.nbytes + self.coefficients.nbytes


@dataclass(frozen=True, slots=True)
class MatrixComponent:
    time_label: int | str
    factor: VertexFactorMetadata


@dataclass(frozen=True, slots=True)
class MatrixNetwork:
    adjacency: tuple
    time_labels: tuple[int | str, ...]
    vertex_slots: tuple[int, ...]
    propagators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatrixWickProduct:
    coefficient: complex
    networks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MatrixWickBlock:
    axis_positions: tuple[tuple[int, ...], ...]
    products: tuple[MatrixWickProduct, ...]


@dataclass(frozen=True, slots=True)
class MatrixGroup:
    axes: tuple[int, ...]
    wick_blocks: tuple[MatrixWickBlock, ...]
    partitions: tuple[int, ...] | None
    partition_mode: str = 'radial'

    def __post_init__(self):
        if self.partition_mode not in ('radial', 'memory'):
            raise ValueError("partition_mode must be 'radial' or 'memory'")
        if self.partition_mode == 'memory' and self.partitions is None:
            raise ValueError('memory partition_mode requires partitions')


@dataclass(frozen=True, slots=True)
class MatrixCatalog:
    components: tuple[MatrixComponent, ...]
    axes: tuple[MatrixLinearAxis, ...]
    networks: tuple[MatrixNetwork, ...]
    groups: tuple[MatrixGroup, ...]
    output: OutputNode


@dataclass(frozen=True, slots=True)
class BoundMatrixCatalog:
    """Ephemeral bindings; not an executable plan or a portable archive."""
    catalog: MatrixCatalog
    component_bindings: tuple[object, ...]
    representative_times: tuple[tuple[int | str, int], ...]


__all__ = ['MatrixLinearAxis', 'MatrixComponent', 'MatrixNetwork',
           'MatrixWickProduct', 'MatrixWickBlock', 'MatrixGroup', 'MatrixCatalog',
           'BoundMatrixCatalog']
