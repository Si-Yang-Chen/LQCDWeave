"""Lift an accepted dense DAG to an explicit retained coordinate axis.

E owns coordinate identities and which inputs vary.  This pass only changes
the rank of those declared inputs and propagates that axis through the same
paid operations.  It does not search, read data, or duplicate static subgraphs.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import math
import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, ContractionProblem, TensorSpec, FactorizedWideReduction, MesonWholeKernel,
)
from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
from .canonical_operations import PREFIX, predict_dense, result_name
from .state import ContractionState, _nbytes
from .storage import logical_source_problem
from lqcdweave.contracts.correlator.cores import (
    TensorCoreSpec, baryon_component_pattern,
)


def _lift_core_action(original, names, dynamic, catalog, lane_axis, width):
    core = original.core
    labels, *legs = original.equation.inputs
    core_dynamic = any(dynamic[key] for key in core.inputs)
    components = tuple(replace(part, inputs=tuple(names[key] for key in part.inputs))
                       for part in core.components)
    lifted = replace(core, components=components,
        axes=('coordinate_batch', *core.axes) if core_dynamic else core.axes,
        spec=TensorSpec((width, *core.spec.shape), core.spec.dtype) if core_dynamic else core.spec)
    leg_keys = original.operands[len(core.inputs):]
    output = ((lane_axis, *original.equation.output)
              if any(dynamic[key] for key in original.operands) else original.equation.output)
    equation = AxisPattern(((lane_axis, *labels) if core_dynamic else labels,
        *((lane_axis, *axes) if dynamic[key] else axes
          for key, axes in zip(leg_keys, legs, strict=True))), output)
    return lifted, equation


def _predict_join(action, catalog, profile):
    """Price the retained kernel recipe, without choosing another path."""
    from .state import _join_path_cost, _join_path_workspace
    from lqcdweave.contraction.performance.api import estimate_vertex_kernel
    core = action.core
    if core is None:
        components = ((action.operands, action.equation, 1),)
    else:
        labels, *legs = action.equation.inputs
        leg_keys = action.operands[len(core.inputs):]
        components = tuple((part.inputs + leg_keys, baryon_component_pattern(
            labels, legs, action.equation.output, len(catalog[part.inputs[1]].shape)), part.coefficient)
            for part in core.components)
    total, workspace, result = 0., 0, None
    for index, (keys, equation, coefficient) in enumerate(components):
        specs = tuple(catalog[key] for key in keys)
        dimensions = {axis: size for axes, spec in zip(equation.inputs, specs, strict=True)
                      for axis, size in zip(axes, spec.shape, strict=True)}
        part_dtype = np.result_type(*(spec.dtype for spec in specs))
        dtype = np.result_type(part_dtype,
            *(() if core is None else (core.spec.dtype,)))
        result = TensorSpec(tuple(dimensions[axis] for axis in equation.output), dtype)
        traffic = sum(_nbytes(spec) for spec in specs) + _nbytes(result)
        if action.kernel_description is not None:
            flops, traffic, scratch = estimate_vertex_kernel(action.kernel_description, equation, specs)
        else:
            shapes = tuple(spec.shape for spec in specs)
            flops, largest = _join_path_cost(equation, shapes, action.kernel_path)
            scratch = max(largest, _join_path_workspace(equation, shapes, action.kernel_path)) * dtype.itemsize + traffic
        total += profile.contract_seconds(flops, traffic, kind='baryon_join', dtype=part_dtype)
        workspace = max(workspace, scratch)
        if core is not None:
            elements = math.prod(result.shape)
            size, part_size = _nbytes(result), elements * part_dtype.itemsize
            workspace = max(workspace, part_size)
            total += profile.contract_seconds(0, part_size + size, kind='baryon_join', dtype=dtype)
            if coefficient != 1:
                total += profile.contract_seconds(elements, 2 * size, kind='baryon_join', dtype=dtype)
            if index:
                total += profile.contract_seconds(elements, 3 * size, kind='baryon_join', dtype=dtype)
    return result, workspace + (2 * _nbytes(result) if core is not None else 0), total


def _validate_problem(template, problem, source_map, lane_axis):
    old = template.problem
    if not isinstance(problem, ContractionProblem):
        raise TypeError('problem must be a ContractionProblem')
    if (old.backend != problem.backend or old.sparse_bridges or problem.sparse_bridges):
        raise ValueError('coordinate path lifting requires one ordinary dense backend')
    if (set(old.sources) | {core.key for core in old.meson_cores} != set(old.specs)
            or set(problem.sources) | {core.key for core in problem.meson_cores} != set(problem.specs)):
        raise ValueError('coordinate path lifting does not support opaque sources')
    names = dict(source_map)
    if (set(names) != set(old.specs) or set(names.values()) != set(problem.specs)
            or len(set(names.values())) != len(names)):
        raise ValueError('source_map must be a complete input bijection')
    if type(lane_axis) is not int or lane_axis < 0:
        raise ValueError('lane_axis must be a fresh nonnegative integer axis')
    require_uniform_contraction_dtype(problem.specs)
    dynamic, width = {}, None
    for key, before in old.specs.items():
        after = problem.specs[names[key]]
        if before.dtype != after.dtype:
            raise ValueError('coordinate lifting cannot change input dtype')
        if before.shape == after.shape:
            dynamic[key] = False
        elif len(after.shape) == len(before.shape) + 1 and after.shape[1:] == before.shape:
            size = after.shape[0]
            if size <= 0 or width is not None and width != size:
                raise ValueError('dynamic inputs require one common positive lane extent')
            width = size
            dynamic[key] = True
        else:
            raise ValueError('inputs may only acquire one leading coordinate axis')
    if len(old.terms) != len(problem.terms):
        raise ValueError('coordinate lifting cannot change the term count')
    for before, after in zip(old.terms, problem.terms, strict=True):
        if any(lane_axis in axes for axes in (*before.subscripts.inputs, before.subscripts.output)):
            raise ValueError('lane_axis must not occur in the original algebra')
        inputs = tuple((lane_axis, *axes) if dynamic[key] else axes
                       for key, axes in zip(before.inputs, before.subscripts.inputs, strict=True))
        output = ((lane_axis, *before.subscripts.output)
                  if any(dynamic[key] for key in before.inputs) else before.subscripts.output)
        if (after.inputs != tuple(names[key] for key in before.inputs)
                or after.subscripts != AxisPattern(inputs, output)
                or after.coefficient != before.coefficient or after.sum_barriers != before.sum_barriers):
            raise ValueError('problem is not the declared coordinate lift of the original terms')
        if before.kernel is not None:
            from .meson_coordinates import lift_meson_kernel
            dynamic_names = {names[key] for key, flag in dynamic.items() if flag}
            if lift_meson_kernel(before.kernel, names, dynamic_names, lane_axis) != after.kernel:
                raise ValueError('coordinate changes the whole meson local binding')
    expected_cores = tuple(replace(core, key=names[core.key],
        inputs=tuple(names[key] for key in core.inputs)) for core in old.meson_cores)
    if expected_cores != problem.meson_cores:
        raise ValueError('coordinate changes the independent whole meson definitions')
    return names, dynamic, width


def _name(action, original):
    if action.kind == 'baryon_join':
        return result_name(action)
    if type(action.core) is FactorizedWideReduction:
        from .factorized_reduction import factorized_result_name
        return factorized_result_name(action.operands, action.core)
    if original.tensor.startswith(PREFIX):
        return result_name(action)
    # Same identity used by ContractionState._contract_layout for ordinary
    # (noncanonical) dense operations.  Dummy axes/order remain explicit.
    signature = (tuple(zip(action.operands, action.equation.inputs)),
                 action.equation.output, None, None)
    return 'node:' + hashlib.sha256(repr(signature).encode()).hexdigest()[:24]


def _independent_replay(expected, profile, memory_limit, host_memory_limit):
    """Expensive original pending-expression replay, only on explicit debug."""
    from .canonical_operations import replay
    state = ContractionState.initial(expected.problem).with_indexed_rows()
    for action in expected.actions:
        state = state.with_action_history(())
        if action.kind == 'load':
            state = state.load(action.tensor, profile, memory_limit,
                               host_limit_bytes=host_memory_limit)
        elif action.kind == 'release':
            state = state.release(action.tensor)
        elif type(action.core) is FactorizedWideReduction:
            from .factorized_reduction import replay_factorized_action
            state = replay_factorized_action(state, action, profile, memory_limit)
        elif action.kind == 'baryon_join':
            join = next(join for join in state.problem.baryon_joins
                        if join.scene == action.term_index and join.name == action.join_id)
            if action.tensor.startswith(PREFIX):
                state = replay(state, action, profile, memory_limit)
            else:
                state = state.contract_join(join, profile, memory_limit,
                    kernel_path_override=action.kernel_path, search_kernel_path=bool(action.kernel_path))
        elif type(action.core) is MesonWholeKernel:
            from .meson_core import replay_meson_whole_action
            state = replay_meson_whole_action(state, action, profile, memory_limit)
        elif action.tensor.startswith(PREFIX):
            state = replay(state, action, profile, memory_limit)
        else:
            state = state.contract(action.term_index, action.positions, profile, memory_limit)
        if state.actions[-1] != action:
            raise ValueError('lifted Action differs from independent dense replay')
    if (not state.done or tuple(state.results) != expected.results
            or state.loaded != expected.loaded
            or not math.isclose(state.seconds, expected.seconds, rel_tol=1e-11, abs_tol=1e-14)):
        raise ValueError('lifted DAG disagrees with independent output replay')


def lift_coordinate_path(template_state, problem, source_map, *, lane_axis,
                         profile, memory_limit, host_memory_limit=None, debug=False):
    """Return the same dense operation order with one retained leading lane axis.

    ``problem`` must be exactly ``template_state.problem`` under the explicit
    source bijection, with only declared input ranks and term equations lifted.
    All dynamic inputs share one positive leading extent.  A scalar/repeated
    coordinate batch may have no dynamic input and only rename this same DAG.
    Dynamic dependence propagates through the paid DAG; static inputs and
    intermediates remain shared.

    The returned state has logical sources and updated costs/residency bounds.
    Call ``plan_scene_storage`` with the actual batched blocks/preparations,
    then the existing optional dense lowering, before execution.  Raw storage,
    preparation workspace, output owners and kernel packing need that audit.
    Thin wide cores retain their explicit lane geometry and internal precision.
    Whole meson and baryon cores keep their fixed internal algorithms.
    Other cores, sparse bridges and explicit resource Actions are refused.
    No numerical backend is imported, including for CuPy metadata.
    """
    if not isinstance(template_state, ContractionState) or not template_state.done:
        raise ValueError('coordinate path lifting requires a complete ContractionState')
    if (len(template_state.results) != len(template_state.problem.terms)
            or any(key is None for key in template_state.results)):
        raise ValueError('template outputs do not match its terms')
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    for label, value in (('memory_limit', memory_limit), ('host_memory_limit', host_memory_limit)):
        if value is None and label == 'host_memory_limit':
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(label + ' must be a positive integer byte limit')
    names, dynamic, width = _validate_problem(template_state, problem, source_map, lane_axis)
    problem = logical_source_problem(problem)
    catalog = dict(problem.specs)
    sizes = {key: _nbytes(spec) for key, spec in catalog.items()}
    allocations = {key: source.snapshot_bytes if source.snapshot_bytes is not None else sizes[key]
                   for key, source in problem.sources.items()}
    old_sources = template_state.problem.sources
    remaining = Counter(key for action in template_state.actions if action.kind in ('contract', 'baryon_join')
                        for key in action.operands)
    output_keys = frozenset(template_state.results)
    explicit_loads = {action.tensor for action in template_state.actions if action.kind == 'load'}
    implicit = tuple(key for key in template_state.problem.specs
                     if key not in explicit_loads and (remaining[key] or key in output_keys))
    loaded, actions, seconds = set(), [], 0.
    equations = {}
    resident, peak = 0, 0
    gpu = problem.backend == 'cupy'

    def bound(required):
        nonlocal peak
        peak = max(peak, required)
        if peak > memory_limit or (not gpu and host_memory_limit is not None and peak > host_memory_limit):
            raise MemoryError('lifted coordinate path exceeds its logical memory budget')

    def load(key, original=None):
        nonlocal resident, seconds
        if key in loaded or key not in allocations:
            raise ValueError('lifted load requires an absent original input')
        resident += allocations[key]
        bound(resident)
        cost = profile.snapshot_seconds(allocations[key])
        action = Action('load', key, seconds=cost)
        if original is not None and original.reused:
            raise ValueError('retained/reused source loads require a separate retention plan')
        actions.append(action)
        loaded.add(key)
        seconds += cost

    # Pure compute states can begin with their inputs resident.  Give those
    # implicit loads an explicit lifecycle for the downstream storage audit.
    for key in implicit:
        load(names[key])
    for original in template_state.actions:
        if (type(original) is not Action or original.resource is not None
                or original.core is not None and type(original.core) not in
                    (FactorizedWideReduction, TensorCoreSpec, MesonWholeKernel)
                or original.kind not in ('load', 'release', 'contract', 'baryon_join')):
            raise ValueError('coordinate lifting only supports ordinary dense Actions')
        if original.kind == 'load':
            load(names[original.tensor], original)
            continue
        if original.kind == 'release':
            if original.tensor not in names:
                raise ValueError('release precedes the lifted tensor definition')
            key = names[original.tensor]
            if (key not in loaded or original.tensor in output_keys
                    or original.tensor not in old_sources and remaining[original.tensor]):
                raise ValueError('invalid release in coordinate path')
            resident -= allocations.get(key, sizes[key])
            loaded.remove(key)
            actions.append(Action('release', key))
            continue
        if original.equation is None or not original.operands:
            raise ValueError('dense coordinate Actions require explicit equations and operands')
        if any(lane_axis in axes for axes in (*original.equation.inputs, original.equation.output)):
            raise ValueError('lane_axis must not occur in an original Action')
        if any(key not in names or names[key] not in loaded for key in original.operands):
            raise ValueError('lifted contraction operands must be resident')
        operands = tuple(names[key] for key in original.operands)
        flags = tuple(dynamic[key] for key in original.operands)
        varies = any(flags)
        if type(original.core) is MesonWholeKernel:
            term = problem.terms[original.term_index]
            equation, core = term.subscripts, term.kernel
            if core.inputs != operands:
                raise ValueError('whole meson actual input binding differs after coordinate lift')
        elif original.kind == 'baryon_join' and original.core is not None:
            core, equation = _lift_core_action(original, names, dynamic, catalog, lane_axis, width)
        else:
            equation_key = original.equation, flags
            equation = equations.get(equation_key)
            if equation is None:
                equation = AxisPattern(tuple((lane_axis, *axes) if flag else axes
                    for flag, axes in zip(flags, original.equation.inputs, strict=True)),
                    (lane_axis, *original.equation.output) if varies else original.equation.output)
                equations[equation_key] = equation
            core = (replace(original.core, equation=equation,
                            shapes=tuple(catalog[key].shape for key in operands),
                            gram_strategy='reduction' if varies else original.core.gram_strategy)
                    if type(original.core) is FactorizedWideReduction else None)
        spec = template_state.catalog[original.tensor]
        shape = (width, *spec.shape) if varies else spec.shape
        # A reused Action names an already lifted producer. Its canonical
        # identity was established there; do not normalize/hash it per term.
        if original.reused and not debug:
            key = names.get(original.tensor)
        else:
            if type(core) is MesonWholeKernel:
                from .meson_core import meson_whole_result_name
                key = meson_whole_result_name(operands, core, shape, spec.dtype)
            else:
                key = _name(replace(original, operands=operands, equation=equation, core=core), original)
            previous = names.setdefault(original.tensor, key)
            if previous != key:
                raise ValueError('coordinate lift changes the selected intermediate sharing relation')
        if original.reused:
            actual = catalog.get(key)
            if (key not in loaded or actual is None
                    or actual.shape != shape or actual.dtype != spec.dtype):
                raise ValueError('reused lifted output is not resident with its declared shape')
            cost, workspace = 0., 0
        else:
            if key in catalog or key in loaded:
                raise ValueError('lifted paid intermediate must have one producer')
            if original.kind == 'baryon_join':
                actual, workspace, cost = _predict_join(replace(original, operands=operands,
                    equation=equation, core=core), catalog, profile)
            elif type(core) is MesonWholeKernel:
                from ..performance.api import estimate_meson_whole
                flops, traffic, workspace = estimate_meson_whole(core, catalog, shape)
                actual = TensorSpec(shape, spec.dtype)
                cost = profile.contract_seconds(flops, traffic, kind='contract', dtype=spec.dtype)
            elif core is not None:
                actual, workspace = TensorSpec(core.output_shape, core.result_dtype), core.workspace_bytes
                cost = profile.contract_seconds(core.flops, core.traffic_bytes,
                    kind='contract', dtype=core.dtype, kernel_calls=core.kernel_calls)
            else:
                actual, workspace, cost = predict_dense(equation, tuple(catalog[name] for name in operands), profile,
                    wide_accumulation=original.wide_accumulation)
            if actual.shape != shape or actual.dtype != spec.dtype:
                raise ValueError('lifted equation differs from its declared output shape/dtype')
            catalog[key] = actual
            sizes[key] = _nbytes(actual)
            # predict_dense returns input+output traffic.  For ordinary dense
            # Actions that is also the existing copy/workspace allowance.
            bound(resident + sizes[key] + workspace)
            resident += sizes[key]
            loaded.add(key)
        dynamic[original.tensor] = varies
        actions.append(replace(original, operands=operands, equation=equation, tensor=key, seconds=cost, core=core))
        seconds += cost
        for operand in original.operands:
            remaining[operand] -= 1
    results = tuple(names[key] for key in template_state.results)
    if any(key not in loaded for key in results):
        raise ValueError('coordinate path failed to retain all term outputs')
    state = ContractionState(problem, catalog, (None,) * len(results), frozenset(loaded), results,
        actions=tuple(actions), seconds=seconds, peak_bytes=peak,
        peak_host_bytes=0 if gpu else peak, peak_device_bytes=peak if gpu else 0)
    if debug:
        _independent_replay(state, profile, memory_limit, host_memory_limit)
    return state


__all__ = ['lift_coordinate_path']
