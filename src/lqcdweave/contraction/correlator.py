"""Public Diagram calculator with explicit bindings and staged execution."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from os import PathLike
import numbers
import platform

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ExpandedCorrelator, DiagramBindings, DiagramData, PlannedContraction, PlanBundle,
    validate_frontend_workload, ExecutionBudget,
)
from lqcdweave.contraction.expansion.api import expand_correlator, describe_meson_diagrams
from lqcdweave.contraction.binding.api import bind_meson_getters
from lqcdweave.contraction.planning.api import plan_workload
from lqcdweave.contracts.memory import BudgetExceededError
from lqcdweave.runtime.memory import resolve_memory_limits as _resolve_workload_limits
from lqcdweave.storage.contraction_plans.api import (
    PlanArchiveError as _PlanArchiveError,
    load_contraction_plan as _load_contraction_plan,
    load_plan as _load_plan,
    save_plan as _save_plan,
    save_contraction_plan as _save_contraction_plan,
    validate_bundle as _validate_bundle,
)
from lqcdweave.contraction.execution.api import BoundContractionExecution, ExecutionBindingCache
from lqcdweave.contraction.planning.api import find_contraction_path, retarget_expanded
from lqcdweave.contraction.results.api import assemble_correlator, OutputAssemblyCache
from lqcdweave.contraction.performance.api import load_cost_profile as _resolve_cost_profile
from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
from lqcdweave.runtime.scheduler import execute_request
from lqcdweave.contraction.correlator_inputs import bind_native_configuration
from lqcdweave.contraction.elemental_preparation import Elemental
from lqcdweave.contraction.calculation import Calculation
from lqcdweave.contraction.contraction_task import ContractionTask


def execute_contraction_path(expanded, path, *, backend=np, measure=False, debug=False, execution_cache=None,
                             runtime_device_limit=None):
    """Build the domain execution request, then hand it to generic runtime dispatch."""
    request = BoundContractionExecution(expanded, path, debug=debug,
        execution_cache=execution_cache if execution_cache is not None else ExecutionBindingCache(),
        runtime_device_limit=runtime_device_limit)
    from lqcdweave.contraction.elemental_preparation import prepare_elemental_inputs
    prepare_elemental_inputs(expanded.inputs, backend=backend,
        host_limit=(path.host_memory_limit - path.reserved_host_bytes) if path.host_memory_limit is not None else None,
        device_limit=(path.memory_limit - path.reserved_device_bytes) if backend is not np and path.memory_limit is not None else None)
    return execute_request(request, backend=backend, measure=measure)


def _run_fixed_calculation(self, *, backend=None, measure=False, debug=None, output=None,
        configuration=None, propagator_map=None, vertex_map=None,
        time_axes=None, time_pairs=None, physical_times=(), physical_lt=None,
        time_roles=("0", "1"), output_definition=None, batch=None):
    """Replay the selected task; planning options and new channels are fixed.

    With output, W streams all requested coordinates through its bounded
    cfg service. Current cfg reader maps replace runtime handles only.
    """
    if output is not None:
        if batch is not None:
            raise ValueError('explicit coordinate batching currently uses in-memory replay; bounded Dataset replay uses its saved windows')
        if measure:
            raise ValueError('cfg window replay does not support measure=True')
        from lqcdweave.contraction.correlator_replay import run_selected_cfg
        return run_selected_cfg(self, output=output, configuration=configuration,
            propagator_map=propagator_map, vertex_map=vertex_map,
            time_axes=time_axes, time_pairs=time_pairs, physical_times=physical_times,
            physical_lt=physical_lt, time_roles=time_roles, output_definition=output_definition,
            backend=backend, debug=debug)
    if time_axes is not None or time_pairs is not None or batch is not None:
        if physical_times or physical_lt is not None or time_roles != ("0", "1") or output_definition is not None:
            raise ValueError('physical reconstruction and output definitions require Dataset output')
        from lqcdweave.contraction.correlator_coordinates import select_coordinate_replay
        selected = select_coordinate_replay(self, time_axes=time_axes, time_pairs=time_pairs,
            batch='auto' if batch is None else batch, configuration=configuration,
            propagator_map=propagator_map, vertex_map=vertex_map, backend=backend, debug=debug,
            retain_raw_results=measure)
        result = selected.run(backend=backend, measure=measure, debug=debug)
        result.report['coordinate_batching'] = selected.prediction['coordinate_batching']
        return result
    if any(value is not None for value in (configuration, propagator_map, vertex_map,
            time_axes, time_pairs, physical_lt, output_definition)) or physical_times or time_roles != ("0", "1"):
        raise ValueError('current cfg bindings and coordinates require output for bounded replay')
    from lqcdweave.adapters.execution_target import check_execution_target
    target = self.options.get('execution_target')
    backend = check_execution_target(target, backend or (target.backend if target else 'numpy'))
    debug = self.options.get('debug', False) if debug is None else debug
    limits = resolve_diagram_memory_limits(backend=backend, memory_limit=self.path.memory_limit,
        host_memory_limit=self.path.host_memory_limit, execution_mode='local')
    if self.plan is not None and (self.plan.peak_host_bytes > limits.host_bytes
            or (limits.device_bytes is not None and self.plan.peak_device_bytes > limits.device_bytes)):
        raise MemoryError('execution resources are below the saved plan requirements')
    compact_output = self.prediction.get('coordinate_batching', {}).get('output_retention') == 'assembled-coordinates'
    if compact_output and not measure:
        from lqcdweave.contraction.correlator_batch_output import run_coordinate_output
        value, report = run_coordinate_output(self, backend=backend, debug=debug,
                                              runtime_device_limit=limits.device_bytes)
    else:
        execution_path = self.path
        if compact_output:
            from lqcdweave.contraction.correlator_batch_output import measured_coordinate_path
            execution_path = measured_coordinate_path(self)
        executed = execute_contraction_path(self.expanded, execution_path,
                                            backend=backend, measure=measure, debug=debug,
                                            execution_cache=self._execution_bindings,
                                            runtime_device_limit=limits.device_bytes)
        recipe = self.expanded.output_recipe
        if not debug and not recipe.coordinates.scan_shape and len(recipe.coordinates.points) == 1:
            # X returns the ordered fixed binding. Keep a cheap boundary check
            # before using R's shared compiled assembler.
            if tuple(item.term_id for item in executed.term_values) != recipe.term_ids:
                raise ValueError('execution term order differs from the selected output recipe')
            point_key = recipe.coordinates.points[0].key
            if any(item.coordinate_key != point_key for item in executed.term_values):
                raise ValueError('execution coordinates differ from the selected output recipe')
            assemble, = self._output_bindings.bind(recipe, (recipe,))
            value = assemble(tuple(item.value for item in executed.term_values))
        else:
            value = assemble_correlator(self.expanded, executed.term_values)
        report = executed.report
    options = dict(self.options, execute=True, measure=measure, debug=debug,
                   backend=getattr(backend, "__name__", str(backend)))
    if 'coordinate_batching' in self.options:
        from lqcdweave.contraction.calculation import _Report
        report = (_Report(report) if isinstance(report, Mapping) else
                  _Report() if report is None else _Report(execution=report))
        report['coordinate_batching'] = self.options['coordinate_batching']
    return replace(self, value=value, report=report, options=options)



def save_calculation_plan(
    path: str | PathLike,
    calculation: Calculation,
    *,
    hardware=None,
    profile_source=None,
    metadata=None,
) -> Path:
    """Save a Calculation through W, translating its options to A's pure API."""
    from lqcdweave.contracts.correlator.api import CfgPlan
    if getattr(calculation, '_window_request', None) is not None:
        raise ValueError("window results carry a selected template, not a complete archive; save the original selection")
    if isinstance(getattr(calculation, 'plan', None), CfgPlan):
        from lqcdweave.storage.contraction_plans.api import save_cfg_plan
        return save_cfg_plan(path,calculation.plan,debug=calculation.options.get('debug',False))

    planned = getattr(calculation, "path", None)
    expanded = getattr(calculation, "expanded", None)
    options = getattr(calculation, "options", {})
    if not isinstance(planned, PlannedContraction) or not isinstance(
        expanded, ExpandedCorrelator
    ):
        raise TypeError(
            "calculation must expose a PlannedContraction path and "
            "ExpandedCorrelator expanded"
        )
    if planned.expanded is not expanded:
        raise _PlanArchiveError("Calculation path and expanded task do not match")
    if not isinstance(options, Mapping):
        options = {}

    target = hardware if hardware is not None else options.get("target_hardware")
    source = profile_source if profile_source is not None else options.get("profile_source")
    metadata = dict(metadata or {})
    if options.get('execution_target') is not None:
        metadata['execution_target'] = options['execution_target'].to_dict()
        metadata['execution_target_summary'] = options.get('execution_target_summary')
    if options.get('planning_host') is not None:
        metadata['planning_host'] = options['planning_host']
    return _save_contraction_plan(
        path,
        planned,
        hardware=target,
        profile_source=source,
        metadata=metadata,
        storage_policy=options.get("storage"),
        optimize=options.get("optimize"),
        reserved_host_bytes=options.get("reserved_host_bytes"),
        reserved_device_bytes=options.get("reserved_device_bytes"),
    )



def load_calculation_plan(
    path, *, expected=None, hardware=None, configuration=None,
    propagator_map=None, vertex_map=None, source_factory=None, readers=None,
    output=None, host_memory_limit=None, memory_limit=None, debug=False,
):
    """Load the matching saved calculation without searching or executing.

    Complete cfg archives need only current reader roles (or a source factory).
    Ordinary fixed-binding archives require ``expected``: a currently bound
    Calculation or ExpandedCorrelator. For the latter, also pass ``hardware``.
    Use ``complete_calculation`` before saving for independent cross-cfg replay.
    """
    if type(debug) is not bool:
        raise TypeError("debug must be bool")
    if expected is not None:
        if any(value is not None for value in
               (configuration, propagator_map, vertex_map, source_factory, readers, output)):
            raise ValueError("fixed-binding loading takes its current bindings from expected")
        options = dict(expected.options) if isinstance(expected, Calculation) else {}
        expanded = expected.expanded if isinstance(expected, Calculation) else expected
        target = hardware if hardware is not None else options.get("target_hardware")
        if target is None:
            raise ValueError("fixed-binding loading requires hardware or a Calculation with target_hardware")
        selected = _load_contraction_plan(
            path, expected=expanded, hardware=target,
            host_memory_limit=host_memory_limit, memory_limit=memory_limit,
        )
        options.update(execute=False, debug=debug, plan_archive=str(path),
                       target_hardware=target,
                       kernel_batching=selected.prediction.get('kernel_batching',True))
        return Calculation(selected.expanded, selected, options=options)
    if hardware is not None:
        raise ValueError("complete archives carry their target hardware; hardware requires expected")
    if source_factory is not None and any(value is not None for value in
                                         (readers, propagator_map, vertex_map)):
        raise ValueError("pass source_factory or current readers, not both")
    if readers is not None and (propagator_map is not None or vertex_map is not None):
        raise ValueError("pass readers or propagator_map/vertex_map, not both")
    from lqcdweave.contraction.correlator_cfg import load_calculation, archive_readers
    calculation = load_calculation(
        path, configuration=configuration, source_factory=source_factory,
        readers=readers, output=output, host_memory_limit=host_memory_limit,
        device_memory_limit=memory_limit, debug=debug,
    )
    if source_factory is None and readers is None:
        current = archive_readers(calculation.plan, propagator_map, vertex_map)
        calculation = replace(calculation, options=dict(calculation.options, readers=current))
    return calculation



def save_correlator(path, result, **options):
    """Save numerical results, preserving containers and declared coordinates.

    A completed Calculation provides its trailing scan axes automatically.
    Raw values use the storage API's explicit axes/coordinates options.
    Source-time reduction is opt-in and requires explicit physical roles.
    """
    from lqcdweave.storage.correlator_results import save_correlator as save_result
    from lqcdweave.contracts.correlator.api import CfgPlan
    if isinstance(result, Calculation):
        if result._window_request is not None:
            raise ValueError("window calculations already publish output; save an in-memory result")
        if result.value is None:
            raise ValueError("run the calculation before saving its numerical result")
        if result.expanded is None and not isinstance(result.plan, CfgPlan):
            # Old session values carried no frontend coordinate declaration.
            # Preserve raw-value saving with the caller's explicit metadata.
            return save_result(path, result.value, **options)
        if isinstance(result.plan, CfgPlan):
            if result.plan.output_mode != 'memory':
                raise ValueError("dataset calculations already publish output; save an in-memory result")
            from lqcdweave.contraction.correlator_cfg import is_cfg_iterator
            if is_cfg_iterator(result.options.get('configuration')):
                raise ValueError("save each (cfg, value) yielded by the complete calculation separately")
            from lqcdweave.contracts.correlator.api import thaw_json
            description = result.plan.output_definition.get('frontend_coordinates')
            if description is not None:
                description = thaw_json(description)
            if description is None:
                raise ValueError("this older complete plan lacks frontend coordinate metadata; pass its value and explicit axes")
        else:
            coordinates = result.expanded.coordinates
            description = dict(mode=coordinates.mode, axes=coordinates.axes,
                               scan_axes=coordinates.scan_axes, scan_shape=coordinates.scan_shape,
                               points=[point.values for point in coordinates.points])
        from lqcdweave.contraction.correlator_result_metadata import operator_coordinates
        if result.expanded is not None:
            operator_labels = operator_coordinates(result.expanded.correlator)
        else:
            from lqcdweave.storage.contraction_plans.api import decode_plan_value
            from lqcdweave.contracts.correlator.api import thaw_json
            encoded = result.plan.output_definition.get('frontend_operator_coordinates')
            operator_labels = {} if encoded is None else decode_plan_value(thaw_json(encoded))
        if operator_labels:
            supplied = options.get('field_coordinates')
            combined = {path: dict(coords) for path, coords in operator_labels.items()}
            for field_path, coords in (supplied or {}).items():
                combined.setdefault(field_path, {}).update(coords)
            options['field_coordinates'] = combined
        names = tuple(description['scan_axes'])
        axis_coordinates = dict(description['axes'])
        if description['mode'] == 'pairs':
            axis_coordinates = {'pair': tuple(range(len(description['points'])))}
        options.setdefault('axes', names)
        options.setdefault('coordinates', {axis: axis_coordinates[axis] for axis in names})
        metadata = dict(options.get('metadata') or {})
        metadata['measurement_coordinates'] = description
        configuration = (result.expanded.bindings.configuration if result.expanded is not None
                         else result.options.get('configuration'))
        metadata['configuration'] = None if configuration is None else str(configuration)
        options.setdefault('configuration', configuration)
        options['metadata'] = metadata
        provenance = dict(options.get('provenance') or {})
        provenance.setdefault('configuration', metadata['configuration'])
        provenance.setdefault('execution_mode', result.execution_mode)
        options['provenance'] = provenance
        result = result.value
    return save_result(path, result, **options)


def load_correlator(path, *, configuration=None):
    """Restore a CorrelatorResult, including axis labels and reduction provenance."""
    from lqcdweave.storage.correlator_results import load_correlator as load_result
    return load_result(path, configuration=configuration)


def calc_diagram_eval(
    correlator=None,
    *,
    propagator_map=None,
    vertex_map=None,
    configuration=None,
    usedNe=None,
    usedNp=None,
    groups_independent=None,
    elemental_weighted=False,
    time_axes=None,
    time_pairs=None,
    batch="auto",
    kernel_batching=False,
    factorized_gram_tiling=None,
    factorized_gram_strategy=None,
    factorized_retile=None,
    baryon_vertices=None,
    partitions=None,
    partition_mode='radial',
    optimize="anytime",
    storage=None,
    execute=True,
    profile=None,
    search=None,
    host_memory_limit=None,
    memory_limit=None,
    reserved_host_bytes=0,
    reserved_device_bytes=0,
    backend=None,
    dtype=None,
    measure=False,
    plan_archive=None,
    diagram_adapter: DiagramData | None = None,
    data: DiagramData | None = None,
    target_hardware=None,
    output=None,
    debug=False,
):
    """Expand, select, execute, and assemble a symbolic correlator.

    The regular script supplies propagator/vertex readers or preset declarations
    and the task's configuration/Ne. Declarations and unbound vertex templates
    are resolved for this call without mutating the templates.
    ``diagram_adapter`` is an optional format/backend bridge.
    When both legacy maps are supplied, native metadata-only binding is selected
    automatically; ``data`` remains a compatibility alias for explicit adapters.
    With ``partitions``, ``partition_mode='radial'`` keeps PC pair results;
    ``'memory'`` sums real blocks into the original correlator/output axes.
    Contraction-kernel batching is disabled by default; ``kernel_batching=True``
    enables it. Joint planning and reuse remain enabled; ``batch`` controls coordinates.
    ``factorized_gram_strategy`` selects reduction,
    tiled register reuse, or resource-audited wide GEMM (default for new CUDA P).
    None follows the saved
    numerical strategy on archive replay. The older boolean tiling option is
    a compatibility spelling for reduction/tiled, not for GEMM.
    ``factorized_retile=False`` disables budgeted tile selection for new plans.
    None enables it for new CUDA plans and preserves saved archive tiles;
    explicit overrides are invalid when replaying a complete archive.
    ``baryon_vertices`` lowers whole Baryon vertices of new plans: ``"auto"``
    (default) plans the per-family rule and the uniform pair and nonzero
    forms and keeps the least predicted time, ``"pair"`` always splits spin
    constants into two parts, ``"nonzero"`` into nonzero spin entries,
    ``"dense"`` keeps the whole constant; every request opens, also with
    Meson/current or glueball vertices, exchange labels, mode blocks or
    several Phi per row. Only a declared ``"joined"`` keeps the closed
    kernel. Archives keep theirs.

    dtype fixes numerical input and output precision before planning. Required
    raw blocks are converted before preparation; omitting dtype retains the
    minimum-input-precision policy. Saved plans retain their declared precision.
    """
    from lqcdweave.contracts.correlator.precision import declared_contraction_dtype
    dtype = declared_contraction_dtype(dtype)
    if any(type(value) is not bool for value in (execute, measure, debug, elemental_weighted,kernel_batching)):
        raise TypeError("execute, measure, debug, elemental_weighted and kernel_batching must be bool")
    from .gram_strategy import resolve_gram_strategy,new_gram_strategy,archived_gram_strategy
    gram_strategy=resolve_gram_strategy(factorized_gram_strategy,factorized_gram_tiling)
    if factorized_retile is not None and type(factorized_retile) is not bool:
        raise TypeError('factorized_retile must be None or bool')
    if baryon_vertices is not None and diagram_adapter is not None:
        raise ValueError('baryon_vertices selects the native Baryon lowering; an explicit adapter owns its own')
    if partition_mode not in ('radial', 'memory'):
        raise ValueError("partition_mode must be 'radial' or 'memory'")
    if partition_mode == 'memory' and partitions is None and plan_archive is None:
        raise ValueError('memory partition_mode requires partitions; pre-split requests carry their own mode')
    if diagram_adapter is not None and data is not None:
        raise ValueError("pass only one of diagram_adapter or compatibility alias data")
    adapter = diagram_adapter if diagram_adapter is not None else data
    from lqcdweave.contraction.correlator_cfg import (
        CompleteCalculation,complete_calculation,is_cfg_iterator,
    )
    if plan_archive is not None:
        overrides={name:value for name,value in dict(time_axes=time_axes,time_pairs=time_pairs,
            partitions=partitions,usedNe=usedNe,usedNp=usedNp,
            partition_mode=None if partition_mode == 'radial' else partition_mode,
            groups_independent=groups_independent,search=search,storage=storage,profile=profile,
            factorized_retile=factorized_retile,baryon_vertices=baryon_vertices,dtype=dtype).items()
            if value is not None}
        if elemental_weighted:
            overrides['elemental_weighted'] = True
        if kernel_batching:
            overrides['kernel_batching'] = True
        if overrides:
            raise ValueError('complete archive fixes planning options; complete a new plan offline to change '
                             +', '.join(overrides))
        calculation=load_calculation_plan(plan_archive,configuration=configuration,
            source_factory=adapter if callable(adapter) else None,output=output,
            propagator_map=None if callable(adapter) else propagator_map,
            vertex_map=None if callable(adapter) else vertex_map,
            host_memory_limit=host_memory_limit,memory_limit=memory_limit,debug=debug)
        saved_strategy=archived_gram_strategy(calculation,gram_strategy)
        calculation=replace(calculation,options=dict(calculation.options,
            factorized_gram_strategy=saved_strategy,
            factorized_gram_tiling=saved_strategy in ('tiled','gemm')))
        return calculation.run(backend=backend,measure=measure,debug=debug) if execute else calculation
    if correlator is None:
        raise ValueError('calculate requires a correlator or a complete plan_archive')
    if is_cfg_iterator(configuration):
        raise ValueError('plan one representative cfg, save it, then pass the cfg iterator with plan_archive')
    from lqcdweave.contraction.correlator_target import resolve_planning_target, target_summary, calibrate_if_needed
    target, limits, profile_document, summary = resolve_planning_target(
        target_hardware, backend, memory_limit=memory_limit,
        host_memory_limit=host_memory_limit, profile=profile)
    backend_name = target.backend
    gram_strategy=new_gram_strategy(gram_strategy,target.runtime)
    memory_limit = limits.host_bytes if backend_name == 'numpy' else limits.device_bytes
    host_memory_limit = limits.host_bytes
    if output is not None or (optimize=='meson_matrix' and (time_axes is not None or time_pairs is not None)):
        from lqcdweave.contracts.correlator.api import describe_coordinates
        coordinates,_=describe_coordinates(time_axes=time_axes,time_pairs=time_pairs,batch='auto')
        if output is not None or bool(coordinates.scan_shape):
            representative=max(coordinates.points,key=lambda point:len(set(v for _,v in point.values)))
            selected=calc_diagram_eval(correlator,propagator_map=propagator_map,
                vertex_map=vertex_map,configuration=configuration,usedNe=usedNe,usedNp=usedNp,
                groups_independent=groups_independent,elemental_weighted=elemental_weighted,
                time_axes=dict(representative.values) or None,
                batch='auto',kernel_batching=kernel_batching,factorized_gram_strategy=gram_strategy,
                factorized_retile=factorized_retile,baryon_vertices=baryon_vertices,
                partitions=partitions,partition_mode=partition_mode,
                optimize=optimize,storage=storage,execute=False,
                profile=profile,search=search,host_memory_limit=host_memory_limit,memory_limit=memory_limit,
                reserved_host_bytes=reserved_host_bytes,reserved_device_bytes=reserved_device_bytes,
                backend=backend,dtype=dtype,measure=False,diagram_adapter=diagram_adapter,
                data=data,target_hardware=target,debug=debug)
            if execute and profile is None:
                document = calibrate_if_needed(target, limits, profile_document,
                    backend=backend or target.backend,
                    dtypes=tuple(sorted({s.dtype.name for s in selected.expanded.tensor_specs.values()})),
                    profile=profile,reserved_host_bytes=reserved_host_bytes,reserved_device_bytes=reserved_device_bytes)
                if document['profile'] != selected.path.profile:
                    path = find_contraction_path(selected.expanded, memory_limit=memory_limit,
                        host_memory_limit=host_memory_limit, optimize=optimize, storage=storage,
                        search=search, profile=document['profile'], kernel_batching=kernel_batching,
                        reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes)
                    from lqcdweave.contraction.planning.api import lower_independent_gemm_path
                    path = lower_independent_gemm_path(path, target_runtime=target.runtime,
                        factorized_retile=factorized_retile is not False)
                    if gram_strategy=='gemm':
                        from lqcdweave.contraction.planning.api import lower_factorized_gemm_path
                        path=lower_factorized_gemm_path(path,target_runtime=target.runtime)
                    summary = target_summary(target, limits, document)
                    path = replace(path, prediction=dict(path.prediction, execution_target=summary))
                    selected = replace(selected, path=path, expanded=path.expanded,
                        options=dict(selected.options, execution_target_summary=summary))
            complete=complete_calculation(selected,coordinates=coordinates,
                output_mode='dataset' if output is not None else 'memory',output=output)
            # Pure source roles can be restored from existing reader handles.
            # E owns the original binding handles; obtain them via its public
            # descriptor interface rather than carrying closures into archives.
            from lqcdweave.contraction.expansion.api import input_readers
            readers=input_readers(selected.inputs)
            complete=replace(complete,options=dict(complete.options,readers=readers))
            return complete.run(backend=backend,measure=measure,debug=debug) if execute else complete
    if partitions is not None:
        from lqcdweave.operators.correlator.api import split_meson_mode_pairs
        correlator = split_meson_mode_pairs(correlator, partitions=partitions, partition_mode=partition_mode)
    bindings = DiagramBindings(propagator_map, vertex_map, configuration, usedNe,
                               usedNp, groups_independent, elemental_weighted)
    native = adapter is None and (bindings.propagator_map is not None or bindings.vertex_map is not None)
    if native:
        from lqcdweave.contraction.expansion.api import NativeDiagramBinding
        bindings = bind_native_configuration(correlator, bindings)
    calibrated = []
    from lqcdweave.contraction.planning.api import PathSearchExhausted

    def plan_lowering(lowering, tolerate=False):
        lowered = adapter
        if native:
            lowered = NativeDiagramBinding(baryon_vertices=lowering, memory_limit=memory_limit,
                host_memory_limit=host_memory_limit, reserved_host_bytes=reserved_host_bytes)
        expanded = None
        if optimize in ('anytime', 'meson_matrix'):
            from lqcdweave.contraction.expansion.api import try_expand_search_matrices
            expanded = try_expand_search_matrices(
                correlator, adapter=lowered, bindings=bindings,
                time_axes=time_axes, time_pairs=time_pairs, batch=batch,
                _factorize_spin=target.runtime != 'hip')
        if expanded is None:
            expanded = expand_correlator(correlator, adapter=lowered, bindings=bindings,
                time_axes=time_axes, time_pairs=time_pairs, batch=batch)
        from lqcdweave.contraction.expansion.api import normalize_contraction_precision
        expanded = normalize_contraction_precision(expanded, dtype=dtype)
        from .kernel_descriptions import attach_vertex_kernel_descriptions
        expanded = attach_vertex_kernel_descriptions(expanded)
        if expanded.problem is not None and expanded.problem.backend != backend_name:
            expanded = retarget_expanded(expanded, backend_name)
        if execute and expanded.problem is not None and not calibrated:
            document = calibrate_if_needed(target, limits, profile_document,
                backend=backend or target.backend,
                dtypes=tuple(sorted({s.dtype.name for s in expanded.tensor_specs.values()})), profile=profile,
                reserved_host_bytes=reserved_host_bytes,reserved_device_bytes=reserved_device_bytes)
            calibrated.append((document, target_summary(target, limits, document)))
        document = calibrated[0][0] if calibrated else profile_document
        try:
            return lowered, find_contraction_path(expanded, memory_limit=memory_limit, optimize=optimize,
                storage=storage, profile=document['profile'], search=search, kernel_batching=kernel_batching,
                host_memory_limit=host_memory_limit, reserved_host_bytes=reserved_host_bytes,
                reserved_device_bytes=reserved_device_bytes)
        except (PathSearchExhausted, MemoryError) as error:
            if not tolerate:
                raise
            return lowered, error

    lowering = baryon_vertices or "auto"
    selecting = native and lowering == "auto"
    lowered, path = plan_lowering(lowering, selecting)
    # No static rule picks the faster opened Baryon form for every request,
    # so "auto" plans each distinct form and keeps the least predicted time.
    # Dense constants are predicted like pair splits but measured slower.
    choices = set(lowered.row_choices) if selecting else set()
    if choices:
        candidates = {"auto": path}
        for candidate in ("pair", "nonzero"):
            if choices != {candidate}:
                candidates[candidate] = plan_lowering(candidate, True)[1]
        feasible = {name: item for name, item in candidates.items() if not isinstance(item, Exception)}
        if not feasible:
            raise path
        predicted = {name: _predicted_seconds(item) for name, item in feasible.items()}
        selected_lowering = min(predicted, key=predicted.get)
        path = feasible[selected_lowering]
        path = replace(path, prediction=dict(path.prediction, baryon_vertex_selection=dict(
            selected=selected_lowering, row_choices=tuple(sorted(choices)), predicted_seconds=predicted,
            infeasible={name: f"{type(item).__name__}: {item}" for name, item in candidates.items()
                        if name not in feasible},
            planning_seconds=sum(item.planning_seconds for item in feasible.values()))))
    elif isinstance(path, Exception):
        raise path
    if calibrated:
        profile_document, summary = calibrated[0]
    resolved_profile = profile_document['profile']
    expanded = path.expanded
    from lqcdweave.contraction.elemental_preparation import describe_elemental_preparation
    input_preparation = describe_elemental_preparation(expanded.inputs, backend=backend_name,
        host_limit=host_memory_limit - reserved_host_bytes,
        device_limit=(memory_limit - reserved_device_bytes) if backend_name != 'numpy' else None)
    if input_preparation:
        path = replace(path, prediction=dict(path.prediction, input_preparation=input_preparation))
    from lqcdweave.contraction.planning.api import lower_independent_gemm_path
    path = lower_independent_gemm_path(path, target_runtime=target.runtime,
        factorized_retile=factorized_retile is not False)
    if gram_strategy=='gemm':
        from lqcdweave.contraction.planning.api import lower_factorized_gemm_path
        path=lower_factorized_gemm_path(path,target_runtime=target.runtime)
    from lqcdweave.contraction.performance.api import target_implementation_coverage
    coverage = target_implementation_coverage(target,
        (action.kind for item in (path.batch_paths or (path,)) if item.plan is not None
         for action in item.plan.state.actions if action.kind in {'contract','baryon_join','sparse_bridge'}))
    if coverage['missing']:
        raise NotImplementedError(f"execution target {target.name!r} lacks implementations: {coverage['missing']}")
    summary = dict(summary, implementations=coverage)
    path = replace(path, prediction=dict(path.prediction, execution_target=summary,
                                        planning_host=platform.node()))
    options = dict(
        optimize=optimize, storage=storage, batch=batch, kernel_batching=kernel_batching, partitions=partitions,
        factorized_gram_tiling=gram_strategy in ('tiled','gemm'),factorized_gram_strategy=gram_strategy,
        factorized_retile=factorized_retile is not False,
        partition_mode=partition_mode, search=dict(search or {}),
        elemental_weighted=elemental_weighted,
        backend=backend_name,
        dtype=expanded.output_recipe.dtype,
        memory_limit=memory_limit, host_memory_limit=host_memory_limit,
        reserved_host_bytes=reserved_host_bytes,
        reserved_device_bytes=reserved_device_bytes,
        profile_version=getattr(resolved_profile, "version", None),
        profile_hardware=getattr(resolved_profile, "hardware_fingerprint", None),
        profile_source=profile_document['source'],
        target_hardware=target.memory, execution_target=target,
        execution_target_summary=summary, execute=execute, measure=measure, debug=debug,
        planning_host=platform.node(),
        plan_archive=None if plan_archive is None else str(plan_archive),
    )
    calculation = Calculation(expanded, path, options=options)
    return calculation.run(backend=backend, measure=measure, debug=debug) if execute else calculation


def _predicted_seconds(path):
    selection = path.prediction.get("batch_selection")
    if selection is not None:
        return float(selection["selected_predicted_seconds"])
    return float(path.prediction["total_seconds"])


def _validate_persisted_meson_bundle(bundle, *, _validated_bundle=None):
    """Validate the Meson W envelope before save or session preparation."""
    if not isinstance(bundle, PlanBundle):
        raise TypeError("persisted Meson operations require a PlanBundle")
    # An archive load already checked this exact immutable DAG. A caller's
    # Python bundle, including a replacement, still needs the full boundary.
    if bundle is not _validated_bundle:
        _validate_bundle(bundle)
    return _validate_persisted_meson_frontend(bundle)


def _validate_persisted_meson_frontend(bundle):
    """Meson semantics are separate from A's generic archive DAG boundary."""
    if not isinstance(bundle, PlanBundle):
        raise TypeError("persisted Meson operations require a PlanBundle")
    workload = bundle.workload
    if (workload.convention != "meson-simple-v1"
            or workload.metadata.get("adapter") != "meson-simple-v1"):
        raise ValueError("persisted Meson bundle requires the meson-simple-v1 adapter and convention")
    validate_frontend_workload(workload)
    recipes = workload.metadata.get("output_recipes")
    if not isinstance(recipes, (tuple, list)):
        raise ValueError("persisted Meson bundle requires ordered output recipes")
    expected_groups = tuple(tuple(recipe) for recipe in recipes if recipe)
    actual_groups = tuple(tuple(group) for group in workload.output_groups)
    if actual_groups != expected_groups:
        raise ValueError("saved output_groups must match the non-empty Meson output recipes in order")

    if set(workload.specs) != set(workload.input_sources):
        raise ValueError("persisted Meson input specs and source declarations must have matching keys")
    for key, source in workload.input_sources.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"persisted Meson source {key!r} must be a mapping")
        kind = source.get("kind")
        handle = source.get("handle")
        method = source.get("method")
        times = source.get("times")
        expected_count = 1 if kind == "vertex" else 2 if kind == "propagator" else None
        if expected_count is None or method != "get":
            raise ValueError(f"persisted Meson source {key!r} has an unsupported kind or method")
        if type(handle) is not int or handle < 0:
            raise ValueError(f"persisted Meson source {key!r} has an invalid handle index")
        if (not isinstance(times, (tuple, list)) or len(times) != expected_count
                or any(type(value) is not int for value in times)):
            raise ValueError(f"persisted Meson source {key!r} has invalid integer times")
        expected_key = f"{kind}:{handle}:" + ":".join(str(value) for value in times)
        if key != expected_key:
            raise ValueError(f"persisted Meson source key {key!r} disagrees with its declared route")
        declared_identity = source.get("source")
        if not isinstance(declared_identity, Mapping) or not declared_identity:
            raise ValueError(f"persisted Meson source {key!r} requires non-empty caller metadata")
    return None


def plan_persisted_meson(
    diagrams,
    times,
    *,
    vertex_specs,
    propagator_specs,
    vertex_sources,
    propagator_sources,
    budget=None,
    max_candidates=4,
    backend=None,
) -> PlanBundle:
    """Describe and finite-plan a persisted scalar VSV Meson workload."""
    workload = describe_meson_diagrams(
        diagrams, times,
        vertex_specs=vertex_specs, propagator_specs=propagator_specs,
        vertex_sources=vertex_sources, propagator_sources=propagator_sources,
    )
    target = None
    if backend is not None:
        # Validate the finite request before observing the process resources.
        budget = ExecutionBudget() if budget is None else budget
        if not isinstance(budget, ExecutionBudget):
            raise TypeError("budget must be an ExecutionBudget")
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, numbers.Integral):
            raise TypeError("max_candidates must be an integer from 1 to 4")
        if not 1 <= max_candidates <= 4:
            raise ValueError("max_candidates must be from 1 to 4")
        resolved = _resolve_workload_limits(budget, backend)
        if resolved.host_bytes == 0 or resolved.device_bytes == 0:
            raise BudgetExceededError("No memory remains after startup reserves")
        budget = replace(budget, host_bytes=resolved.host_bytes, device_bytes=resolved.device_bytes)
        target = "gpu" if str(getattr(backend, "__name__", backend)).lower() == "cupy" else "cpu"
    bundle = plan_workload(
        workload, budget, optimize="greedy", max_candidates=max_candidates, target=target,
    )
    _validate_persisted_meson_bundle(bundle)
    return bundle


def save_persisted_meson_plan(bundle: PlanBundle, path) -> Path:
    """Validate and save a Meson PlanBundle using the existing A archive."""
    _validate_persisted_meson_frontend(bundle)
    return _save_plan(bundle, path)


def open_persisted_meson_session(path, *, backend=None, expected_workload=None) -> "ContractionSession":
    """Open and prepare one saved Meson plan for repeated cfg execution."""
    from lqcdweave.contraction.contraction_session import ContractionSession

    bundle = _load_plan(Path(path), expected_workload=expected_workload)
    _validate_persisted_meson_bundle(bundle, _validated_bundle=bundle)
    return ContractionSession._from_validated_plan(
        bundle, backend=backend, validated_bundle=bundle,
    )


def execute_persisted_meson_cfg(session, cfg, *, vertices, propagators) -> Calculation:
    """Execute one cfg with getter handles already bound to that cfg."""
    loader = bind_meson_getters(cfg, vertices=vertices, propagators=propagators)
    return session.execute_cfg(cfg, loader)


calculate = calc_diagram_eval
from lqcdweave.contraction.correlator_cfg import CompleteCalculation,complete_calculation


__all__ = [
    "Elemental",
    "CompleteCalculation", "complete_calculation",
    "Calculation", "ExpandedCorrelator", "assemble_correlator",
    "calc_diagram_eval", "calculate", "execute_contraction_path",
    "expand_correlator", "find_contraction_path",
    "plan_persisted_meson", "save_persisted_meson_plan",
    "open_persisted_meson_session", "execute_persisted_meson_cfg",
    "save_calculation_plan", "load_calculation_plan", "save_correlator", "load_correlator",
]
