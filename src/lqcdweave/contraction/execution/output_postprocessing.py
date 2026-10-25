"""Device result assembly, byte packing and bounded snapshot ownership.

W injects the temporal numerical service. X never selects physical time mappings,
opens storage, or changes the output recipe's scientific meaning.
"""
from math import prod
from collections import OrderedDict
from weakref import ref
import numpy as np
from .output_program import bind_device_output_assembly


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def describe_device_download_layout(layout):
    """Describe the actual pair-download owner emitted by DeviceOutputProcessor.

    Assembly fields may have different dtypes, but pack_device_fields places
    them in one int64-word owner. Packed-dt execution downloads no pair slots.
    """
    from .output_queue import OutputDownloadLayout
    return OutputDownloadLayout(() if layout.output_layout == 'dt_packed' else
                                (prod(layout.output_storage_shape) * 8,))


def pack_device_fields(backend, fields, values, shape, *, dt=False, columns=None):
    """Pack numeric device fields using the existing int64-word byte ABI."""
    packed = backend.zeros(shape, dtype=np.int64)
    destination = packed.reshape(-1)
    offset = 0
    for field in fields:
        array = values[field.path]
        if columns is None:
            start = field.dt_offset if dt else field.offset
            words = field.words
        else:
            start, words = offset, field.words * columns
            offset += words
        if start < 0:
            continue
        payload = backend.ascontiguousarray(array).reshape(-1).view(np.uint8)
        destination[start:start + words].view(np.uint8)[:payload.size] = payload
    return packed


class DeviceOutputProcessor:
    """One cfg's device numerical outputs, inside its existing device scope.

    At most one previous input/field snapshot remains pending. Its event is
    joined before replacing it; a second current replay result can coexist and
    is included in W's pre-planning reservation. Downloads retain packed owners
    separately through DeviceExecution and the existing two-slot output queue.
    """
    def __init__(self, layout, physical_times, device, accumulator, *, debug=False):
        self.layout, self.times, self.device = layout, tuple(physical_times), device
        self.backend, self.accumulator = device.backend, accumulator
        self.debug = debug
        self.coordinates = tuple(dict(p) for p in layout.coordinates) if hasattr(layout, 'coordinates') else tuple(
            dict(i.coordinate) for i in layout.instances)
        self.programs = []
        self._template_programs = OrderedDict()
        self.pending = None
        self.consumed = 0
        self.closed = False

    def _drain(self):
        if self.pending is not None:
            event, owners = self.pending
            event.synchronize()
            self.pending = None

    def process(self, index, values, template):
        try:
            return self._process(index, values, template)
        except BaseException as error:
            # Keep this call's operands and the failed frame's local fields
            # alive until all queued arithmetic/copies have finished.
            try:
                self.backend.cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                if callable(getattr(error, 'add_note', None)):
                    error.add_note(f'GPU output cleanup also failed: {cleanup!r}')
            raise

    def _process(self, index, values, template):
        if self.closed or index != self.consumed:
            raise ValueError('device output stream repeats or omits a coordinate')
        self._drain()
        recipe, problem = template.output, template.plan.state.problem
        # Consecutive windows rebind identical recipes; keep only the current
        # template set instead of accumulating an unbounded history of plans.
        cached = self._template_programs.get(id(template))
        entry = cached[1] if cached is not None and cached[0]() is template else next((p for r, q, p in self.programs
                      if (r is recipe and q is problem) or (r == recipe and q == problem)), None)
        if entry is None:
            entry = bind_device_output_assembly(recipe, problem, self.backend, debug=self.debug)
            self._template_programs.clear()
            self.programs = [(recipe, problem, entry)]
            statistics = self.device.statistics
            statistics['output_postprocess_bindings'] = statistics.get('output_postprocess_bindings', 0) + 1
        # Do not retain old windows' large action tables. Weak identity also
        # detects id reuse without recursive comparisons at each coordinate.
        try:
            identity = ref(template)
        except TypeError:  # injected lightweight protocol objects
            identity = lambda: template
        self._template_programs[id(template)] = (identity, entry)
        self._template_programs.move_to_end(id(template))
        if len(self._template_programs) > 16:
            self._template_programs.popitem(last=False)
        assembled = entry(values)
        statistics = self.device.statistics
        statistics['output_postprocess_calls'] = statistics.get('output_postprocess_calls', 0) + 1
        statistics['output_postprocess_stage_launches'] = statistics.get('output_postprocess_stage_launches', 0) + getattr(entry, 'last_stage_launches', 0)
        statistics['output_postprocess_max_scratch_bytes'] = max(statistics.get('output_postprocess_max_scratch_bytes', 0), getattr(entry, 'scratch_bytes', 0))
        metadata_bytes = getattr(entry, 'metadata_bytes', 0)
        if metadata_bytes:
            statistics = self.device.statistics
            statistics['output_metadata_upload_calls'] = statistics.get('output_metadata_upload_calls', 0) + 1
            statistics['output_metadata_upload_bytes'] = statistics.get('output_metadata_upload_bytes', 0) + metadata_bytes
        fields = {f.path: _at(assembled, f.path) for f in self.layout.output_fields}
        if self.debug and any(np.dtype(value.dtype).kind in 'fc' and
                              not bool(self.backend.isfinite(value).all()) for value in fields.values()):
            raise ValueError('nonfinite correlator output')
        if self.accumulator is not None:
            point = self.coordinates[index]
            sink, source = self.layout.time_roles
            self.accumulator.add(self.times[point[source]], self.times[point[sink]],
                {f.path: fields[f.path] for f in self.accumulator.fields},
                slots=point, reject_equal=self.layout.output_layout == 'dt_packed')
        packed = (() if self.layout.output_layout == 'dt_packed' else
                  (pack_device_fields(self.backend, self.layout.output_fields, fields,
                                      self.layout.output_storage_shape),))
        event = self.backend.cuda.Event()
        try:
            event.record(self.backend.cuda.get_current_stream())
        except BaseException:
            self.backend.cuda.get_current_stream().synchronize()
            raise
        self.pending = event, (values, assembled, getattr(entry, "call_owners", ()))
        self.consumed += 1
        return packed

    def _complete(self):
        if self.closed or self.consumed != len(self.coordinates):
            raise ValueError('device output stream has incomplete cfg coverage')
        self._drain()

    def packed_records(self):
        """Finish mean/layout/packing on device, then download final records."""
        try:
            return self._packed_records()
        except BaseException as error:
            try:
                self.backend.cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                if callable(getattr(error, 'add_note', None)):
                    error.add_note(f'GPU finalization cleanup also failed: {cleanup!r}')
            raise

    def _packed_records(self):
        self._complete()
        plan, backend = self.layout, self.backend
        values = self.accumulator.finalize(plan.dt_coordinates)
        dt_fields = tuple(f for f in plan.output_fields if f.dt_offset >= 0)
        auxiliary = tuple(f for f in plan.output_fields if f.dt_offset < 0)
        packed = [pack_device_fields(backend, dt_fields, values, plan.dt_packed_storage_shape,
                                     columns=len(plan.dt_coordinates))]
        if auxiliary:
            packed.append(pack_device_fields(backend, auxiliary, values, plan.one_point_storage_shape,
                                              columns=plan.one_point_slots))
        downloaded = self.device.download(tuple(packed))
        return self.accumulator.counts.copy(), downloaded[0], downloaded[1] if auxiliary else None

    def pair_records(self):
        """Yield owned host dt payloads, one at a time, for pair-preserving IO."""
        records = self._pair_records()
        try:
            yield from records
        finally:
            # Also fence on generator close or a packing/download failure while
            # its frame still owns the final numerical arrays.
            try:
                self.backend.cuda.get_current_stream().synchronize()
            finally:
                records.close()

    def _pair_records(self):
        self._complete()
        if self.accumulator is None:
            return
        fields = tuple(f for f in self.layout.output_fields if f.dt_offset >= 0)
        dts = tuple(map(int, np.flatnonzero(self.accumulator.counts)))
        values = self.accumulator.finalize(dts)
        for column, dt in enumerate(dts):
            current = {f.path: values[f.path][..., column] for f in fields}
            # Match the ordinary pair writer's unconditional final finite gate.
            if any(not bool(self.backend.isfinite(v).all()) for v in current.values()):
                raise ValueError('nonfinite reconstructed correlator output')
            packed = pack_device_fields(self.backend, fields, current, self.layout.dt_storage_shape, dt=True)
            yield self.device.download((packed,))[0]

    @property
    def counts(self):
        return None if self.accumulator is None else self.accumulator.counts.copy()

    def close(self):
        if not self.closed:
            try:
                self._drain()
            finally:
                # Even a failed event query cannot release queued operands
                # before the compute stream has drained.
                if self.pending is not None:
                    self.backend.cuda.get_current_stream().synchronize()
                    self.pending = None
                if self.accumulator is not None:
                    self.accumulator.close()
                self.programs.clear()
                self._template_programs.clear()
                self.closed = True


__all__ = ['DeviceOutputProcessor', 'pack_device_fields', 'describe_device_download_layout']
