"""Coordinate writes must not create one HDF5 chunk per matrix element."""
from math import prod
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from lqcdweave.storage.correlator_results import (
    CfgOutputWriter, PackedDtOutputWriter, load_correlator_output,
)


@pytest.mark.parametrize('shape,dtype', [((400, 400), 'complex64'), ((9, 5, 4), 'complex128'),
                                      ((), 'int64'), ((0, 3), 'float32')])
def test_coordinate_tiling_retains_native_values_without_element_chunks(tmp_path, shape, dtype):
    coordinates = ({'sink': 0, 'source': 0}, {'sink': 1, 'source': 0})
    field = SimpleNamespace(path=('matrix',), shape=shape, dtype=np.dtype(dtype).str,
                            time_role=None, dt_offset=-1)
    plan = SimpleNamespace(output_fields=(field,), coordinate_shape=(2, 1),
        instances=tuple(SimpleNamespace(coordinate=c) for c in coordinates),
        physical_lt=None, output_definition={}, time_roles=('sink', 'source'),
        dt_coordinates=(), output_layout='pairs')
    binding = SimpleNamespace(ensemble=None, lattice=None, physical_lt=None,
        physical_times=(), parents=(), provenance={})
    first = np.arange(prod(shape), dtype=np.float64).reshape(shape).astype(dtype)
    second = (first + 3).astype(dtype)
    writer = CfgOutputWriter(plan, 'a', tmp_path, binding)
    writer.consume(0, {'matrix': first})
    writer.consume(1, {'matrix': second})
    writer.publish()
    actual = load_correlator_output(tmp_path, ('matrix',), configuration='a')
    np.testing.assert_array_equal(actual, np.stack((first, second), axis=-1)[..., None])
    assert actual.dtype == np.dtype(dtype)
    with h5py.File(Path(tmp_path) / 'a.h5', 'r') as handle:
        dataset = handle['pairs/matrix']
        if all(shape):
            assert dataset.chunks[-2:] == (1, 1)
            assert prod(dataset.chunks) * dataset.dtype.itemsize <= 1024 * 1024
        if shape == (400, 400):
            # The old leading-time heuristic creates 320,000 eight-byte chunks.
            assert prod(dataset.chunks[:2]) > 1
            assert (Path(tmp_path) / 'a.h5').stat().st_size < actual.nbytes + 256 * 1024


@pytest.mark.parametrize('writer_type', [CfgOutputWriter, PackedDtOutputWriter])
def test_large_dt_fields_keep_counts_gaps_and_coordinate_tiles(tmp_path, writer_type):
    shape = (400, 400)
    field = SimpleNamespace(path=('matrix',), shape=shape, dtype=np.dtype('complex64').str,
                            time_role=None, dt_offset=0, words=prod(shape))
    coordinates = ({'sink': 1, 'source': 0}, {'sink': 2, 'source': 0})
    plan = SimpleNamespace(output_fields=(field,), coordinate_shape=(2, 1),
        instances=tuple(SimpleNamespace(coordinate=c) for c in coordinates),
        physical_lt=4, output_definition={}, time_roles=('sink', 'source'),
        dt_coordinates=(0, 1, 2, 3), output_layout='dt_packed' if writer_type is PackedDtOutputWriter else 'pairs',
        dt_words=prod(shape), one_point_slots=0)
    binding = SimpleNamespace(ensemble=None, lattice=None, physical_lt=4,
        physical_times=(0, 1, 3), parents=(), provenance={})
    first = np.arange(prod(shape), dtype=np.float32).reshape(shape).astype('complex64')
    second = first * np.complex64(1 + 2j)
    writer = writer_type(plan, 'a', tmp_path, binding)
    writer.consume(0, {'matrix': first})
    writer.consume(1, {'matrix': second})
    writer.publish()
    actual = load_correlator_output(tmp_path, ('matrix',), configuration='a', dt=True)
    np.testing.assert_array_equal(actual[..., 1], first)
    np.testing.assert_array_equal(actual[..., 3], second)
    assert np.isnan(actual[..., (0, 2)]).all()
    assert actual.dtype == np.dtype('complex64')
    with h5py.File(Path(tmp_path) / 'a.h5', 'r') as handle:
        dataset = handle['dt/matrix']
        assert dataset.chunks[-1] == 1
        assert prod(dataset.chunks[:-1]) > 1
        assert prod(dataset.chunks) * dataset.dtype.itemsize <= 1024 * 1024
