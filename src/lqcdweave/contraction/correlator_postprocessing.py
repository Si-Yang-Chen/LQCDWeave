"""Compose output planning, X device numerics and TS temporal reduction.

All allocation stays within the existing worker device scope. Offline reserves
use the selected target's neutral metadata and never inspect a planning GPU.
"""
from math import prod
from lqcdweave.contracts.correlator.api import device_allocation_bytes
from types import SimpleNamespace
import numpy as np
import sys
from lqcdweave.contraction.execution.api import (
    DeviceOutputProcessor, active_device_execution,
    describe_device_output_reservation, describe_device_download_layout,
)
from lqcdweave.sampling.temporal.api import (
    DeviceTimePairAccumulator, describe_device_accumulator_allocations,
)


def validate_postprocess_backend(value, target_backend, *, output_mode='dataset'):
    if value not in ('cpu', 'gpu'):
        raise ValueError("postprocess_backend must be 'cpu' or 'gpu'")
    if value == 'gpu':
        if target_backend != 'cupy':
            raise ValueError('GPU output postprocessing requires a CuPy execution target')
        if output_mode != 'dataset':
            raise ValueError('GPU output postprocessing currently requires dataset output')
    return value


def describe_gpu_postprocessing_reservation(selected, recipe, layout):
    """Pure owner-aligned device requirements, independent of saved reserves."""
    problem = selected.state.problem
    description = describe_device_output_reservation(recipe, problem)
    actual = description['fields']
    expected = tuple((f.path, f.shape, f.dtype) for f in layout.output_fields)
    if actual != expected:
        raise ValueError('GPU output dtype/shape differs from the saved CPU packing layout; '
                         'use a compatible explicit numeric dtype before selecting the plan')
    # Cover native grouped snapshots and diagnostic individual copies. The
    # latter can allocate separately even for repeated result occurrences.
    term_bytes = sum(device_allocation_bytes(prod(selected.state.catalog[k].shape)
                     * selected.state.catalog[k].dtype.itemsize) for k in selected.state.results)
    assembly = sum(map(device_allocation_bytes, description['device_owner_bytes']))
    packed = layout.output_layout == 'dt_packed'
    fields = tuple(f for f in layout.output_fields if packed or f.dt_offset >= 0)
    dts = layout.dt_coordinates if packed else tuple(range(layout.physical_lt or 0))
    accumulator = (sum(map(device_allocation_bytes,
                   describe_device_accumulator_allocations(fields, layout.physical_lt,
                       layout.one_point_slots if packed else 0, dts))) if layout.physical_lt else 0)
    # Two snapshots cover current replay plus one previous event owner. The
    # shared output queue separately admits at most two packed pair downloads
    # from remaining headroom; it cannot steal this fixed reservation.
    final_packing = ((prod(layout.dt_packed_storage_shape) + prod(layout.one_point_storage_shape)) * 8
                     if packed else (prod(layout.output_storage_shape) + prod(layout.dt_storage_shape)) * 8)
    packing_shapes = ((layout.dt_packed_storage_shape, layout.one_point_storage_shape)
                      if packed else (layout.output_storage_shape, layout.dt_storage_shape))
    packing_device = sum(device_allocation_bytes(prod(shape) * 8) for shape in packing_shapes)
    largest = max((device_allocation_bytes(prod(f.shape) * np.dtype(f.dtype).itemsize)
                   for f in fields), default=0)
    device_bytes = 2 * (term_bytes + assembly) + accumulator + packing_device + largest
    host_bytes = (final_packing + (layout.physical_lt or 0) * 8 + layout.one_point_slots * len(fields)
                  + 2 * description['metadata_bytes'])
    return dict(postprocess_backend='gpu', postprocess_device_bytes=device_bytes,
                postprocess_host_bytes=host_bytes)


def reserve_gpu_postprocessing(selected, recipe, layout):
    """Account before P chooses cfg/window residency; no backend allocation."""
    required = describe_gpu_postprocessing_reservation(selected, recipe, layout)
    device_bytes, host_bytes = required['postprocess_device_bytes'], required['postprocess_host_bytes']
    selected = selected.with_reserves(
        reserved_device_bytes=selected.reserved_device_bytes + device_bytes,
        reserved_host_bytes=selected.reserved_host_bytes + host_bytes)
    if not selected.feasible:
        raise MemoryError('GPU postprocessing buffers exceed the selected execution target budget')
    return selected, required


class GpuDatasetPostprocessing:
    """W lifecycle owner shared by complete-cfg and incremental window replay."""
    # Private X/W handoff; arbitrary callbacks retain the public tuple contract.
    _accepts_owner_snapshots = True
    def __init__(self, layout, bindings, *, debug=False):
        self.layout, self.bindings, self.debug = layout, bindings, debug
        for template in getattr(layout, 'templates', ()):
            required = describe_gpu_postprocessing_reservation(template.plan, template.output, layout)
            saved = layout.statistics
            if any(saved.get(key, -1) < required[key] for key in
                   ('postprocess_device_bytes', 'postprocess_host_bytes')):
                raise MemoryError('saved GPU postprocessing reserve is too small for this implementation; '
                                  'complete the output plan again from the selected template')
        self.active = {}
        self._download_layout = describe_device_download_layout(layout)
        self.output_bytes = 0 if layout.output_layout == 'dt_packed' else prod(layout.output_storage_shape) * 8

    def __call__(self, cfg, index, values, template):
        processor = self.active.get(cfg)
        if processor is None:
            device = active_device_execution()
            if device is None:
                raise ValueError('GPU output postprocessing requires the active execution device')
            packed = self.layout.output_layout == 'dt_packed'
            fields = tuple(f for f in self.layout.output_fields if packed or f.dt_offset >= 0)
            accumulator = (DeviceTimePairAccumulator(fields, Lt=self.layout.physical_lt,
                backend=device.backend, one_point_slots=self.layout.one_point_slots if packed else 0,
                debug=self.debug) if self.layout.physical_lt else None)
            processor = DeviceOutputProcessor(self.layout, self.bindings[cfg].physical_times,
                device, accumulator, debug=self.debug)
            self.active[cfg] = processor
        return processor.process(index, values, template)

    def publish(self, cfg, writer):
        processor = self.active[cfg]
        try:
            if self.layout.output_layout == 'dt_packed':
                return writer.publish_prepacked(*processor.packed_records())
            return writer.publish_prepacked(processor.counts, processor.pair_records())
        finally:
            primary = sys.exc_info()[1]
            try:
                processor.close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                if callable(getattr(primary, 'add_note', None)):
                    primary.add_note(f'GPU output cleanup also failed: {cleanup!r}')
            finally:
                del self.active[cfg]

    def close(self, primary=None):
        error = None
        for processor in tuple(self.active.values()):
            try:
                processor.close()
            except BaseException as failure:
                if error is None:
                    error = failure
        self.active.clear()
        if error is not None:
            if primary is None:
                raise error
            if callable(getattr(primary, 'add_note', None)):
                primary.add_note(f'GPU output cleanup also failed: {error!r}')


__all__ = ['validate_postprocess_backend', 'describe_gpu_postprocessing_reservation',
           'reserve_gpu_postprocessing', 'GpuDatasetPostprocessing']
