"""Compose incremental P/X with the existing complete-cfg dataset consumers."""
from contextlib import ExitStack
from dataclasses import replace
from math import prod
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

from lqcdweave.contracts.correlator.api import CoordinateSpec, device_allocator_headroom
from lqcdweave.contraction.expansion.api import (
    describe_input_preparations, describe_input_reads, describe_coordinate_reads,
    unknown_reader_aliases, input_readers, input_sampling_signature,
)
from lqcdweave.contraction.results.api import freeze_output_recipe, bind_output_assembly
from lqcdweave.contraction.execution.api import execute_cfg_windows
from lqcdweave.storage.contraction_plans.api import encode_plan_value
from lqcdweave.adapters.execution_target import check_execution_target
from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
from lqcdweave.storage.correlator_results import CfgOutputWriter, PackedDtOutputWriter
from lqcdweave.contraction.correlator_cfg import CfgSources, source_loader_factory, is_cfg_iterator, check_coordinate_sampling
from lqcdweave.contraction.correlator_stream import plan_cfg_windows
from lqcdweave.contraction.correlator_window_output import describe_window_output
from lqcdweave.contraction.correlator_result_metadata import operator_coordinates


from lqcdweave.contraction.calculation_compat import CfgWindowResult


class _ObservedWindows:
    def __init__(self, windows, progress):
        self.windows, self.progress = windows, progress
        self.peak_host_bytes = self.peak_device_bytes = 0
        self.windows_delivered = 0

    def __iter__(self):
        return self

    def observe(self, plan):
        if plan is not None:
            self.peak_host_bytes = max(self.peak_host_bytes, plan.peak_host_bytes)
            self.peak_device_bytes = max(self.peak_device_bytes, plan.peak_device_bytes)
            self.windows_delivered += 1
            if self.progress is not None:
                self.progress('window', dict(index=plan.statistics['window_index'],
                    coordinate_offset=plan.statistics['coordinate_offset'], count=len(plan.instances),
                    peak_host_bytes=plan.peak_host_bytes, peak_device_bytes=plan.peak_device_bytes,
                    planning_seconds=plan.statistics.get('lifecycle_seconds')))
        return plan

    def __next__(self):
        return self.observe(next(self.windows))

    def next_ready(self):
        return self.observe(self.windows.next_ready())

    def close(self):
        self.windows.close()


def run_cfg_window_calculation(calculation, *, coordinates, output, configuration=None,
                               source_factory=None, readers=None, backend=None,
                               physical_lt=None, output_definition=None,
                               time_roles=('0', '1'),
                               dt_fields=('two_point', 'connected', 'disconnected'),
                               output_layout='pairs_and_dt', dt_coordinates=None,
                               window_size=None, debug=None, progress=None, postprocess_backend='cpu'):
    """Execute one full cfg while a private CPU process binds future windows.

    The selected path is reused verbatim. Whole-cfg output storage is reserved
    before starting P; no full coordinate binding or archive save is required.
    Readers stay open for this cfg, and the root dataset publishes only after
    every requested coordinate succeeds. Existing complete archives use their
    normal run() path instead of this explicitly incremental entry point.
    """
    started = perf_counter()
    if calculation.plan is None or calculation.path.batch_paths:
        raise ValueError('window execution requires a selected single-coordinate template')
    expanded = calculation.expanded
    if len(expanded.coordinates.points) != 1:
        raise ValueError('window execution requires a single-coordinate selected template')
    cfg = expanded.bindings.configuration if configuration is None else configuration
    if is_cfg_iterator(cfg):
        raise ValueError('window execution currently accepts one cfg at a time')
    debug = calculation.options.get('debug', False) if debug is None else debug
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    if source_factory is None and readers is None:
        raise ValueError('window execution requires reader bindings')
    target = calculation.options.get('execution_target')
    backend = check_execution_target(target, backend or calculation.plan.state.problem.backend)
    if backend.__name__ != calculation.plan.state.problem.backend:
        raise ValueError('execution backend differs from the selected template')
    from lqcdweave.contraction.correlator_postprocessing import (
        validate_postprocess_backend, reserve_gpu_postprocessing, GpuDatasetPostprocessing,
    )
    validate_postprocess_backend(postprocess_backend, backend.__name__)
    gpu = postprocess_backend == 'gpu'
    selected = calculation.plan
    host_cap = selected.host_memory_limit
    if backend.__name__ == 'numpy':
        host_cap = selected.memory_limit if host_cap is None else min(host_cap, selected.memory_limit)
    limits = resolve_diagram_memory_limits(backend=backend,
        memory_limit=selected.memory_limit, host_memory_limit=host_cap, execution_mode='local')
    preparations, specs = describe_input_preparations(expanded.inputs)
    point_coordinates = CoordinateSpec('bound', (), (), (), expanded.coordinates.points, ())
    recipe = freeze_output_recipe(replace(expanded.output_recipe, coordinates=point_coordinates))
    definition = dict(output_definition or {})
    definition['frontend_operator_coordinates'] = encode_plan_value(operator_coordinates(expanded.correlator))
    definition['frontend_coordinates'] = dict(
        mode=coordinates.mode, axes=[[axis, list(values)] for axis, values in coordinates.axes],
        scan_axes=list(coordinates.scan_axes), scan_shape=list(coordinates.scan_shape),
        points=[list(point.values) for point in coordinates.points],
    )
    definition["input_sampling_signature"] = input_sampling_signature(
        describe_input_reads(expanded.inputs), input_readers(expanded.inputs))
    layout = describe_window_output(recipe, selected.state.problem, preparations, specs, coordinates,
        physical_lt=physical_lt, time_roles=time_roles, dt_fields=dt_fields,
        output_definition=definition, output_layout=output_layout, dt_coordinates=dt_coordinates)
    result_bytes = sum(prod(selected.state.catalog[key].shape) * selected.state.catalog[key].dtype.itemsize
                       for key in set(selected.state.results))
    reserved = max(selected.reserved_host_bytes, result_bytes) + layout.extra_host_bytes
    allocator = device_allocator_headroom(
        limits.device_bytes if backend.__name__ == 'cupy' else limits.host_bytes,
        backend=backend.__name__, device_name=getattr(target, 'name', None))
    previous = calculation.options.get('coordinate_allocator_headroom', {}).get('reserve_bytes', 0)
    selected = selected.with_reserves(reserved_host_bytes=reserved,
        reserved_device_bytes=selected.reserved_device_bytes-previous+allocator['reserve_bytes'])
    postprocess_statistics = {}
    if gpu:
        selected, postprocess_statistics = reserve_gpu_postprocessing(selected, recipe, layout)
    if (not selected.feasible or selected.peak_host_bytes > limits.host_bytes
            or (limits.device_bytes is not None and selected.peak_device_bytes > limits.device_bytes)):
        raise MemoryError('complete output buffers exceed the worker memory budget')
    # Observed worker limits can reduce the declared task allocation, never
    # expand it. P sees those limits before choosing each residency window.
    summary = dict(calculation.options.get('execution_target_summary') or {},
                   host_memory_limit=limits.host_bytes, device_memory_limit=limits.device_bytes)
    requests = describe_input_reads(expanded.inputs)
    read_metadata = describe_coordinate_reads(expanded.inputs, coordinates.points)
    aliases = unknown_reader_aliases(expanded.inputs)
    assemble = None if gpu else bind_output_assembly(recipe, debug=debug)
    writer = None
    published = {}
    consumed = 0
    with ExitStack() as scope:
        binding = scope.enter_context(source_factory(cfg)) if source_factory is not None else readers
        binding = binding if isinstance(binding, CfgSources) else CfgSources(binding)
        check_coordinate_sampling(expanded, coordinates, binding.readers)
        bindings = {cfg: binding}
        postprocessor = GpuDatasetPostprocessing(layout, bindings, debug=debug) if gpu else None
        def factory(window, current_cfg):
            nonlocal writer
            from contextlib import contextmanager
            @contextmanager
            def bind():
                nonlocal writer
                holder = SimpleNamespace(plan=window, options=dict(readers=binding, debug=debug))
                with source_loader_factory(holder)(current_cfg) as loader:
                    actual = loader.output_binding
                    if writer is None:
                        writer_type = PackedDtOutputWriter if output_layout == 'dt_packed' else CfgOutputWriter
                        writer = writer_type(layout, cfg, Path(output), actual,
                                             debug=debug, coordinates=layout.coordinates, prepacked=gpu)
                    elif actual.parents != writer.binding.parents:
                        raise ValueError('input identities changed between cfg windows')
                    yield loader
            return bind()
        def consume(current_cfg, index, values):
            nonlocal consumed
            if index is None:
                if consumed != len(layout.coordinates):
                    raise ValueError('window stream did not cover all requested coordinates')
                published[current_cfg] = dict(directory=str(writer.directory), configuration=str(current_cfg), data_id=(postprocessor.publish(current_cfg, writer) if gpu else writer.publish()))
            else:
                if index != consumed:
                    raise ValueError('window stream repeats or omits an output coordinate')
                if gpu:
                    if output_layout != 'dt_packed':
                        writer.consume_packed(index, values[0])
                else:
                    writer.consume(index, assemble(values))
                consumed += 1
            if progress is not None:
                progress('published' if index is None else 'point', dict(cfg=str(cfg), count=consumed,
                    seconds=perf_counter()-started))
        producer = scope.enter_context(plan_cfg_windows(selected, recipe, coordinates.points,
            requests, preparations, specs, output_mode='dataset', read_metadata=read_metadata,
            reader_aliases=aliases, execution_target=target, target_summary=summary,
            window_size=window_size, debug=debug))
        windows = _ObservedWindows(producer, progress)
        try:
            with calculation._execution_bindings.factory_scope():
                report = execute_cfg_windows(windows, cfg, factory, consume, backend=backend,
                    host_memory_limit=limits.host_bytes, device_memory_limit=limits.device_bytes,
                    prefetch_host_bytes=calculation.options.get('prefetch_host_bytes'),
                    prefetch_device_bytes=calculation.options.get('prefetch_device_bytes'), debug=debug,
                    device_transform=postprocessor)
        except BaseException as error:
            if writer is not None:
                writer.abort(error)
            raise
        finally:
            if postprocessor is not None: postprocessor.close(primary=sys.exc_info()[1])
    report.update(postprocess_statistics)
    report.update(producer.statistics)
    report.update(total_seconds=perf_counter()-started, coordinate_count=consumed,
        peak_planned_host_bytes=windows.peak_host_bytes, peak_planned_device_bytes=windows.peak_device_bytes,
        whole_cfg_output_extra_host_bytes=layout.extra_host_bytes,
        output_coverage='complete', execution_mode='cfg_windows', allocator_headroom=allocator)
    # Retain the whole request, never just the first produced window. Reader
    # handles belong to this W handle, not the structural execution cache.
    request = dict(coordinates=coordinates, output=output, configuration=cfg,
        source_factory=source_factory, readers=readers, backend=backend,
        physical_lt=physical_lt, output_definition=output_definition,
        time_roles=time_roles, dt_fields=dt_fields, output_layout=output_layout,
        dt_coordinates=dt_coordinates, window_size=window_size, debug=debug,
        progress=progress, postprocess_backend=postprocess_backend)
    return replace(calculation, value=published, report=report, _window_request=request)


__all__ = ['run_cfg_window_calculation', 'CfgWindowResult']
