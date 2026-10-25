"""Execute audited scalar groups without replacing ordinary fallback kernels."""
from dataclasses import dataclass

from lqcdweave.contracts.correlator.api import ScalarReductionGroup, scalar_reduction_geometry
from .diagnostics import debug_enabled
from .reduction_group_runtime import _native_output_rounding
from .reduction import bind_precise_reduction
from .scalar_reduction_group import bind_scalar_reduction_group
from .operations.result_views import _DeferredResultView


@dataclass(frozen=True)
class BoundScalarGroup:
    record: ScalarReductionGroup
    inputs: tuple
    results: tuple
    kernel: object

    def tasks(self, arrays):
        tasks = []
        for keys in self.inputs:
            values = []
            for key in keys:
                value = arrays[key]
                if type(value) is _DeferredResultView:
                    value = arrays[key] = value.array()
                values.append(value)
            tasks.append(tuple(values))
        return tuple(tasks)

    def execute(self, arrays, *, timed=None, device=True):
        tasks = self.tasks(arrays)
        values = (self.kernel(tasks) if timed is None else
                  timed('contract_kernel', lambda: self.kernel(tasks), device=device))
        for key, value in zip(self.results, values, strict=True):
            arrays[key] = value

    def bind_call(self, *, shared_residency=False):
        if not shared_residency:
            def call(context):self.execute(context[0])
            return call
        inputs, results, kernel = self.inputs, self.results, self.kernel

        def call(context):
            if len(context) != 4:
                return self.execute(context[0])
            arrays, binding = context[2], context[3]
            tasks = []
            for keys in inputs:
                values = []
                for key in keys:
                    index = binding[key]
                    value = arrays[index]
                    if type(value) is _DeferredResultView:
                        value = arrays[index] = value.array()
                    values.append(value)
                tasks.append(tuple(values))
            values = kernel(tuple(tasks))
            for key, value in zip(results, values, strict=True):
                arrays[binding[key]] = value
        return call


def bind_scalar_groups(state, groups, backend, kernel_cache, dense_members):
    """Only fuse members already eligible for the public wide raw kernel."""
    if not groups or debug_enabled() or getattr(backend, '__name__', '') != 'cupy':
        return {}, {}
    starts, members = {}, {}
    share_small_outputs = _native_output_rounding(state)
    for group in groups:
        if type(group) is not ScalarReductionGroup:
            raise TypeError('scalar execution requires immutable ScalarReductionGroup records')
        if any(index in dense_members or index in members for index in group.action_indices):
            raise ValueError('scalar group overlaps another physical operation')
        actions = tuple(state.actions[index] for index in group.action_indices)
        # A cfg instance may keep only some of the logical members. It must
        # execute every remaining member at its original position, not publish
        # early values over an already retained owner.
        if any(action.reused for action in actions):
            continue
        geometries = tuple(scalar_reduction_geometry(action, state.catalog) for action in actions)
        geometry = geometries[0]
        if any(other != geometry for other in geometries[1:]):
            raise ValueError('scalar group members differ from their common geometry')
        equation = actions[0].equation.normalized()
        # Share compiled sources globally, but keep combined stride/ABI layouts
        # with their stable member set. Many equal-geometry groups otherwise
        # evict each other's combined stride layouts on every warm pass.
        capacity = next(n for n in (8, 16, 32) if len(actions) <= n)
        key = ('scalar-group', group.action_indices, capacity, equation, geometry.shapes,
               geometry.dtype, len(actions), geometry.allowance, share_small_outputs)
        if key not in kernel_cache:
            # C/P describes prospective raw geometry. Tiny reductions can fail
            # the original public bounded-wide admission and must stay narrow.
            original = bind_precise_reduction(equation, geometry.shapes, (geometry.dtype,)*2, backend)
            kernel_cache[key] = (bind_scalar_reduction_group(equation, geometry.shapes, geometry.dtype,
                (geometry.allowance,)*len(actions), backend, share_small_outputs=share_small_outputs)
                if original is not None and getattr(original, 'raw_reduction', False) else None)
        kernel = kernel_cache[key]
        if kernel is None:
            continue
        if kernel.partial_bytes > group.workspace_bytes:
            raise ValueError('scalar group raw partials exceed its saved workspace')
        bound = BoundScalarGroup(group, tuple(action.operands for action in actions),
                                 tuple(action.tensor for action in actions), kernel)
        starts[group.action_indices[0]] = bound
        members.update((index, group) for index in group.action_indices)
    return starts, members
