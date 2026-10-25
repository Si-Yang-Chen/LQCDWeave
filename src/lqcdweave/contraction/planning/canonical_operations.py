"""Canonical selected computations shared by integer search and final audit.

Exact enumeration keeps its established identities. Canonical actions are
explicitly named and replayed independently through the ordinary Action IR.
"""
from dataclasses import replace
import hashlib
import math
import numpy as np

from lqcdweave.contracts.correlator.api import AxisPattern, Action, Factor, PendingTerm, TensorSpec
from .state import _replace_row, _outer_product_axes


PREFIX = "node:canonical:"


def axis_key(inputs, labels, output, *, ordered=False):
    """Preserve axis equality, diagonals and ordered result axes after renaming."""
    order = tuple(range(len(inputs))) if ordered else tuple(sorted(range(len(inputs)), key=lambda i: inputs[i]))
    # A binary repeated input has two possible occurrence orders. Pick a stable
    # normalized relation rather than using the caller's dummy label spelling.
    orders = (order,)
    if not ordered and len(order) == 2 and inputs[0] == inputs[1]:
        orders = (order, order[::-1])
    keys = []
    for positions in orders:
        numbers = {}
        def number(label):
            return numbers.setdefault(label, len(numbers))
        axes = tuple(tuple(number(c) for c in labels[i]) for i in positions)
        result = tuple(number(c) for c in output)
        keys.append((tuple(inputs[i] for i in positions), axes, result))
    return min(keys)


def output_axes(inputs, labels, retained, *, ordered=False):
    positions = range(len(inputs)) if ordered else sorted(range(len(inputs)), key=lambda i: inputs[i])
    return tuple(dict.fromkeys(c for i in positions for c in labels[i] if c in retained))


def computation_key(action):
    labels, output = action.equation.inputs, action.equation.output
    if action.kind == "contract":
        key = (action.kind, axis_key(action.operands, labels, output))
        return (*key, True) if action.wide_accumulation else key
    if action.kind == "baryon_join":
        core = getattr(action, "core", None)
        operands = action.operands if core is None else ("logical-core", *action.operands[len(core.inputs):])
        return (action.kind, axis_key(operands, labels, output, ordered=True),
                action.sparse_spin, None if core is None else core.identity)
    return (action.kind, axis_key(action.operands[:2], labels, output, ordered=True),
            action.operands[2:], action.left_axes, action.right_axes, action.same_point,
            action.transpose_bridge, action.used_np, action.used_np_right)


def result_name(action, extras=()):
    return _result_name(computation_key(action), extras)


def _result_name(key, extras=()):
    return PREFIX + hashlib.blake2b(repr((key, extras)).encode(), digest_size=32).hexdigest()


def predict_dense(equation, specs, profile, *, wide_accumulation=False):
    """Shape/cost facts shared by dense operations with different data names."""
    labels, output = equation.inputs, equation.output
    if len(labels) == 2 and _outer_product_axes(labels[0], labels[1], output):
        raise ValueError("outer products are not planning operations")
    dimensions = {}
    for axes, spec in zip(labels, specs, strict=True):
        for axis, size in zip(axes, spec.shape, strict=True):
            dimensions.setdefault(axis, size)
    dtype = np.result_type(*(spec.dtype for spec in specs))
    spec = TensorSpec(tuple(dimensions[axis] for axis in output), dtype)
    reduction = math.prod(dimensions[axis] for axis in dimensions.keys() - set(output))
    flops = math.prod(spec.shape) * max(reduction, 1) * max(len(labels) - 1, 1)
    traffic = (sum(math.prod(value.shape) * value.dtype.itemsize for value in specs)
               + math.prod(spec.shape) * spec.dtype.itemsize)
    seconds = profile.contract_seconds(flops, traffic, kind="contract", dtype=spec.dtype,
                                       kernel_calls=2 if wide_accumulation else 1)
    return spec, traffic, seconds


def describe_dense(inputs, labels, output, specs, *, positions, profile,
                   equation=None, prediction=None, canonical_key=None, wide_accumulation=False):
    """Bind immutable compiled facts without rebuilding a temporary state.

    The integer workspace owns input definitions and residency. Optional facts
    come from its compiled axis template and fixed-profile prediction cache.
    Sparse/core operations and debug retain the independent describe() path.
    """
    equation = AxisPattern(labels, output) if equation is None else equation
    spec, traffic, seconds = predict_dense(equation, specs, profile,
        wide_accumulation=wide_accumulation) if prediction is None else prediction
    key = ("contract", axis_key(inputs, labels, output)) if canonical_key is None else canonical_key
    if wide_accumulation:
        key = (*key, True)
    name = _result_name(key)
    action = Action("contract", name, inputs, equation, seconds,
                    term_index=0, positions=positions, wide_accumulation=wide_accumulation)
    return action, spec, traffic


def _canonical_contract_layout(state, scene, positions, layout=None, *, output_order=None,
                               scene_joins=None, scoped_bridges=None):
    if layout is None:
        # Static scopes belong to the plan. Only this row's live bindings need
        # filtering; completed joins and consumed bridge legs change per step.
        active_joins = (None if scene_joins is None else {scene: tuple(
            j for j in scene_joins.get(scene, ())
            if (scene, j.name) not in state.completed_joins)})
        active_bridges = None
        if scoped_bridges is not None:
            names = {f.tensor for f in state.pending[scene].factors}
            active_bridges = tuple(b for b in scoped_bridges
                                   if b.scene in (None, scene) and {b.left, b.right} <= names)
        layout = state._contract_layout(scene, positions, scene_joins=active_joins,
                                        active_bridges=active_bridges)
    layout = list(layout)
    selected, retained = layout[:2]
    inputs, labels = tuple(f.tensor for f in selected), tuple(f.labels for f in selected)
    if output_order is not None:
        output = tuple(output_order)
        if len(output) != len(set(output)) or set(output) != set(retained):
            raise ValueError("output layout must permute the retained axes")
    else:
        output = (state.pending[scene].output if len(selected) == 1 else
                  output_axes(inputs, labels, retained, ordered=layout[5] is not None))
    layout[1] = output
    layout[3] = tuple(layout[2][c] for c in output)
    extras = tuple((b.record, tuple(a % len(selected[0].labels) for a in b.left_axes),
                    tuple(a % len(selected[1].labels) for a in b.right_axes), b.point_shape,
                    b.same_point, b.transpose_bridge) for b in layout[5:7] if b is not None)
    return tuple(layout), extras


def logical_result_name(state, scene, positions=(), *, join=None, layout=None):
    """Identify a legal computation without costs, residency or shared effects."""
    if join is None:
        layout, extras = _canonical_contract_layout(state, scene, positions, layout)
        selected, output, _, _, _, bridge, cycle, join_operation, _, _ = layout
        wide = False
        barriers = state.problem.terms[scene].sum_barriers
        if barriers and layout[4] in (np.dtype('float32'), np.dtype('complex64')):
            pending = state.pending[scene]
            before = {a for factor in pending.factors for a in factor.labels}
            after = set(output) | {a for i, factor in enumerate(pending.factors)
                                   if i not in positions for a in factor.labels}
            wide = any(before.intersection(trigger) and not after.intersection(trigger)
                       for trigger, _ in barriers)
        action = Action("sparse_bridge" if bridge else "contract", "",
            tuple(f.tensor for f in selected) + ((bridge.record,) if bridge else ())
            + ((cycle.record,) if cycle else ()), AxisPattern(tuple(f.labels for f in selected), output),
            used_np=bridge.used_np if bridge else 0,
            left_axes=bridge.left_axes if bridge else (-3, -2, -1),
            right_axes=bridge.right_axes if bridge else (0, 1, 2),
            same_point=bridge.same_point if bridge else False,
            transpose_bridge=bridge.transpose_bridge if bridge else False,
            used_np_right=bridge.used_np_right if bridge else None,
            join_id="bounded_coo_join" if join_operation else "", wide_accumulation=wide)
        return result_name(action, extras)
    selected, _, retained, _ = state._join_layout(join) if layout is None else layout
    inputs = (join.spin, join.elemental, *(f.tensor for f in selected))
    labels = (join.spin_labels, join.elemental_labels(state.catalog[join.elemental]), *(f.labels for f in selected))
    output = output_axes(inputs, labels, retained, ordered=True)
    core = join.core
    if core is not None and (len(core.components) != 1 or core.components[0].coefficient != 1 or join.batch_label is not None):
        inputs = core.inputs + tuple(f.tensor for f in selected)
        labels = (join.core_labels, *(f.labels for f in selected))
        action = Action("baryon_join", "", inputs, AxisPattern(tuple(labels), output), core=core)
    else:
        action = Action("baryon_join", "", inputs, AxisPattern(tuple(labels), output), sparse_spin=join.sparse_spin)
    return result_name(action)


def describe(state, scene, positions=(), *, join=None, profile, kernel_path=None, output_order=None,
             scene_joins=None, scoped_bridges=None):
    """Describe one computation; optional static scopes avoid whole-task scans."""
    if join is None:
        layout, extras = _canonical_contract_layout(state, scene, positions, output_order=output_order,
            scene_joins=scene_joins, scoped_bridges=scoped_bridges)
        output = layout[1]
        provisional, spec, workspace = state._describe_contract(scene, positions, layout, profile)
        key = result_name(provisional, extras)
        reused = key in state.loaded
        if key in state.catalog and not reused:
            raise ValueError("released canonical intermediate cannot be recomputed")
        action = replace(provisional, tensor=key, reused=reused, seconds=0 if reused else provisional.seconds)
        return action, spec, 0 if reused else workspace, output, tuple(positions)
    selected, rest, retained, key = state._join_layout(join, scene_joins=scene_joins,
                                                     scoped_bridges=scoped_bridges)
    # A recipe's workspace is independent of peaks reached by earlier steps.
    prediction_state = state._seed_residency(
        replace(state, peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0))
    equation_inputs = (join.spin, join.elemental, *(f.tensor for f in selected))
    equation_labels = (join.spin_labels, join.elemental_labels(state.catalog[join.elemental]), *(f.labels for f in selected))
    output = output_axes(equation_inputs, equation_labels, retained, ordered=True)
    core = join.core
    if core is not None and (len(core.components) != 1 or core.components[0].coefficient != 1 or join.batch_label is not None):
        seconds, workspace, path = 0.0, 0, kernel_path
        spec = None
        dtype = np.result_type(core.spec.dtype, *(state.catalog[k].dtype for k in core.inputs),
                               *(state.catalog[f.tensor].dtype for f in selected))
        if any(part.coefficient.imag for part in core.components):
            dtype = np.result_type(dtype, np.complex64)
        for index, component in enumerate(core.components):
            if any(key not in state.loaded for key in component.inputs):
                raise ValueError("baryon core inputs must be resident")
            leaf = replace(join, spin=component.inputs[0], elemental=component.inputs[1],
                           sparse_spin=component.sparse_spin, core=None)
            after = prediction_state._join_from_layout(leaf, (selected, rest, output, "component:prediction"),
                profile, 1 << 100, kernel_path_override=path, search_kernel_path=path is not None,
                scene_joins=scene_joins)
            part = after.actions[-1]
            path = part.kernel_path
            part_spec = after.catalog[part.tensor]
            spec = TensorSpec(part_spec.shape, dtype)
            size = math.prod(spec.shape) * spec.dtype.itemsize
            part_size = math.prod(part_spec.shape) * part_spec.dtype.itemsize
            workspace = max(workspace, after.peak_bytes - state.resident_bytes - part_size, part_size)
            seconds += part.seconds
            seconds += profile.contract_seconds(0, part_size + size, kind="baryon_join", dtype=dtype)
            if component.coefficient != 1:
                seconds += profile.contract_seconds(math.prod(spec.shape), 2 * size, kind="baryon_join", dtype=spec.dtype)
            if index:
                seconds += profile.contract_seconds(math.prod(spec.shape), 3 * size, kind="baryon_join", dtype=spec.dtype)
        operands = core.inputs + tuple(f.tensor for f in selected)
        equation = AxisPattern((join.core_labels, *(f.labels for f in selected)), output)
        action = Action("baryon_join", "", operands, equation, seconds, join_id=join.name,
                        term_index=scene, kernel_path=path, core=core,
                        kernel_description=join.kernel_description)
        key = result_name(action)
        reused = key in state.loaded
        local = set(join.spin_labels + join.mode_labels)
        positions = tuple(i for i, f in enumerate(state.pending[scene].factors) if local.intersection(f.labels))
        return replace(action, tensor=key, reused=reused, seconds=0 if reused else seconds), spec, \
            0 if reused else workspace + 2 * size, output, positions
    provisional = prediction_state._join_from_layout(join, (selected, rest, output, key), profile, 1 << 100,
        kernel_path_override=kernel_path, search_kernel_path=kernel_path is not None,
        scene_joins=scene_joins)
    action = provisional.actions[-1]
    key = result_name(action)
    reused = key in state.loaded
    if key in state.catalog and not reused:
        raise ValueError("released canonical intermediate cannot be recomputed")
    spec = provisional.catalog[action.tensor]
    workspace = 0 if reused else max(0, provisional.peak_bytes - state.resident_bytes - math.prod(spec.shape) * spec.dtype.itemsize)
    action = replace(action, tensor=key, reused=reused, seconds=0 if reused else action.seconds)
    local = set(join.spin_labels + join.mode_labels)
    positions = tuple(i for i, f in enumerate(state.pending[scene].factors) if local.intersection(f.labels))
    return action, spec, workspace, output, positions


def replay(state, action, profile, memory_limit=1 << 100):
    """Validate and update exactly the occurrence named by a canonical Action."""
    scene = action.term_index
    if not 0 <= scene < len(state.pending) or state.pending[scene] is None:
        raise ValueError("canonical action requires a live Term")
    join = None
    if action.kind == "baryon_join":
        join = next((j for j in state.problem.baryon_joins
                     if j.scene == scene and j.name == action.join_id), None)
        if join is None:
            raise ValueError("canonical join has no declared core binding")
    actual, spec, workspace, output, positions = describe(state, scene, action.positions,
        join=join, profile=profile, kernel_path=action.kernel_path if join else None,
        output_order=action.equation.output if join is None else None)
    if actual != action:
        raise ValueError("canonical action differs from its inputs, axes or independently computed cost")
    return apply_description(state, action, spec, workspace, output, positions,
                             join=join, memory_limit=memory_limit)


def apply_description(state, action, spec, workspace, output, positions, *,
                      join=None, memory_limit=1 << 100, scene_joins=None):
    """Apply the description just accepted by this module's fixed-path builder.

    External Actions use replay(), which independently describes and compares
    their contents. Internal producers already own that exact description.
    """
    scene = action.term_index
    term = state.pending[scene]
    updated = PendingTerm(tuple(f for i, f in enumerate(term.factors) if i not in positions)
                          + (Factor(action.tensor, output),), term.output)
    completed = state.completed_joins
    if join is not None:
        completed |= {(scene, join.name)}
    results = state.results
    joins_left = any(j.scene == scene and (scene, j.name) not in completed for j in
                     (state.problem.baryon_joins if scene_joins is None else scene_joins.get(scene, ())))
    if len(updated.factors) == 1 and updated.factors[0].labels == updated.output and not joins_left:
        pending = _replace_row(state.pending, scene, None)
        results = _replace_row(results, scene, action.tensor)
    else:
        pending = _replace_row(state.pending, scene, updated)
    size = math.prod(spec.shape) * spec.dtype.itemsize
    peak = max(state.peak_bytes, state.resident_bytes + (0 if action.reused else size + workspace))
    if peak > memory_limit:
        raise MemoryError("canonical complete computation exceeds memory cap")
    return state._seed_residency(replace(state, pending=pending, results=results,
        completed_joins=frozenset(completed), catalog=state.catalog.updated(((action.tensor, spec),)),
        loaded=state.loaded if action.reused else state.loaded | {action.tensor},
        seconds=state.seconds + action.seconds,
        peak_bytes=peak, peak_host_bytes=max(state.peak_host_bytes, peak if state.problem.backend == "numpy" else 0),
        peak_device_bytes=max(state.peak_device_bytes, peak if state.problem.backend == "cupy" else 0),
        actions=state.actions + (action,)), state._result_residency_delta(action.tensor, spec))
