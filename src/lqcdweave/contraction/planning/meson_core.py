"""Plan whole connected/trace operations, without private-factor search."""
from collections import Counter
from dataclasses import replace
import hashlib
from math import prod
import numpy as np

from lqcdweave.contracts.correlator.api import Action, TensorSpec
from ..performance.api import estimate_meson_whole


def meson_whole_result_name(inputs, kernel, shape, dtype):
    return 'meson-whole:' + hashlib.sha256(repr((inputs, kernel.identity,
        tuple(shape), np.dtype(dtype).str)).encode()).hexdigest()[:32]


def describe_meson_whole_action(state, scene, *, profile):
    term = state.problem.terms[scene]
    kernel = term.kernel
    if kernel is None or state.pending[scene] is None:
        raise ValueError('whole meson operation needs its live external Term')
    if any(key not in state.loaded for key in kernel.inputs):
        raise ValueError('whole meson numerical dependencies must be resident')
    spec, key = _whole_result(term, state.catalog)
    flops, traffic, workspace = estimate_meson_whole(kernel, state.catalog, spec.shape)
    reused = key in state.loaded
    action = Action('contract', key, kernel.inputs, term.subscripts,
        seconds=0. if reused else profile.contract_seconds(flops, traffic,
            kind='contract', dtype=spec.dtype), reused=reused, term_index=scene,
        positions=tuple(range(len(term.inputs))), core=kernel)
    return action, spec, 0 if reused else workspace


def _whole_result(term, catalog):
    """Describe the same whole value for immutable replay and linear delivery."""
    kernel = term.kernel
    dimensions = {label: size for key, axes in zip(term.inputs, term.subscripts.inputs, strict=True)
                  for label, size in zip(axes, catalog[key].shape, strict=True)}
    shape = tuple(dimensions[label] for label in term.subscripts.output)
    dtype = catalog[kernel.inputs[0]].dtype
    if kernel.kind == 'loop' and kernel.representation == 'unit-spin' and dtype == np.dtype('complex64'):
        dtype = np.dtype('complex128')
    spec = TensorSpec(shape, dtype)
    key = meson_whole_result_name(kernel.inputs, kernel, shape, dtype)
    return spec, key


def replay_meson_whole_action(state, action, profile, memory_limit=1 << 100):
    from .state import _replace_row
    actual, spec, workspace = describe_meson_whole_action(state, action.term_index, profile=profile)
    if action != actual:
        raise ValueError('whole meson operation differs from its external binding')
    size = 0 if action.reused else prod(spec.shape) * spec.dtype.itemsize
    peak = max(state.peak_bytes, state.resident_bytes + size + workspace)
    if peak > memory_limit:
        raise MemoryError('whole meson operation exceeds the execution budget')
    pending = _replace_row(state.pending, action.term_index, None)
    results = _replace_row(state.results, action.term_index, action.tensor)
    return state._seed_residency(replace(state,
        catalog=state.catalog.updated(((action.tensor, spec),)), pending=pending,
        results=results, loaded=state.loaded | {action.tensor},
        actions=state.actions + (action,), seconds=state.seconds + action.seconds,
        peak_bytes=peak,
        peak_host_bytes=max(state.peak_host_bytes, peak) if state.problem.backend == 'numpy' else state.peak_host_bytes,
        peak_device_bytes=max(state.peak_device_bytes, peak) if state.problem.backend == 'cupy' else state.peak_device_bytes), size)


def plan_meson_whole(problem, blocks, preparations, *, profile, memory_limit,
                     host_memory_limit=None, reserved_host_bytes=0, reserved_device_bytes=0):
    """Choose complete-result reuse/order, then use the original storage pass."""
    from .state import ContractionState
    from .storage import logical_source_problem, logical_cost_profile, plan_scene_storage
    if not problem.meson_cores or any(term.kernel is None for term in problem.terms):
        raise ValueError('whole matrix planning requires closed meson Terms')
    logical = logical_source_problem(problem)
    model = logical_cost_profile(profile)
    # This fixed route has no branches or backtracking. Keep construction
    # local, rather than copying a growing immutable residency set for every
    # load/result/release and scanning all earlier results at each release.
    catalog, loaded, output_keys = dict(logical.specs), set(), set()
    history, results = [], []
    resident = peak = 0
    seconds = 0.0
    source_sizes = {}
    remaining = Counter(key for term in problem.terms for key in set(term.kernel.inputs))
    for scene, term in enumerate(problem.terms):
        inputs = tuple(dict.fromkeys(term.kernel.inputs))
        for key in inputs:
            if key not in loaded:
                if key not in logical.sources:
                    raise ValueError('load requires an absent original input')
                source = logical.sources[key]
                amount = source.snapshot_bytes
                if amount is None:
                    amount = prod(catalog[key].shape) * catalog[key].dtype.itemsize if key in catalog else 0
                source_sizes[key] = amount
                cost = model.snapshot_seconds(amount)
                history.append(Action('load', key, seconds=cost))
                seconds += cost
                resident += amount
                peak = max(peak, resident)
                loaded.add(key)
        spec, key = _whole_result(term, catalog)
        reused = key in loaded
        cost, workspace = 0.0, 0
        if not reused:
            flops, traffic, workspace = estimate_meson_whole(term.kernel, catalog, spec.shape)
            cost = model.contract_seconds(flops, traffic, kind='contract', dtype=spec.dtype)
            resident += prod(spec.shape) * spec.dtype.itemsize
            loaded.add(key)
        history.append(Action('contract', key, term.kernel.inputs, term.subscripts,
            seconds=cost, reused=reused, term_index=scene,
            positions=tuple(range(len(term.inputs))), core=term.kernel))
        seconds += cost
        peak = max(peak, resident + workspace)
        catalog[key] = spec
        results.append(key)
        output_keys.add(key)
        for key in inputs:
            remaining[key] -= 1
            if not remaining[key]:
                if key in output_keys:
                    raise ValueError('release requires a resident non-output tensor')
                loaded.remove(key)
                resident -= source_sizes[key]
                history.append(Action('release', key))
    state = ContractionState(logical, catalog, (None,) * len(results), frozenset(loaded),
        tuple(results), actions=tuple(history), seconds=seconds, peak_bytes=peak,
        peak_host_bytes=peak if logical.backend == 'numpy' else 0,
        peak_device_bytes=peak if logical.backend == 'cupy' else 0)
    plan = plan_scene_storage(state, blocks, preparations, profile=profile,
        memory_limit=memory_limit, host_memory_limit=host_memory_limit,
        reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes)
    if not plan.feasible:
        raise MemoryError('whole meson plan exceeds the physical storage budget before IO')
    return plan
