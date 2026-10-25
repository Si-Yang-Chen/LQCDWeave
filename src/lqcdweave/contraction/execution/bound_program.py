"""Bind saved Actions to calls once, without retaining input arrays or devices.

This is an X-local callable view of the existing plan, not a second schedule.
Numerical operations and their ownership rules remain in operations/.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np

from .operations.output import copy_output, bind_output_layout
from .operations.sparse_bridge import validate_sparse_bridge_parts
from .operations.transfer import copy_to_backend
from .operations.contract import contract_dense as builtin_contract_dense, bind_dense_call
from .plan_cursor import plan_operations, drive_plan_cursors


def unmeasured(name, operation, **metadata):
    return operation()


def _load(context, *, key, spec, problem, backend, device):
    arrays, execution = context[0], context[1]
    source = execution.source(key)
    value = execution.materialize(key, source)
    if spec is None:
        arrays[key] = validate_sparse_bridge_parts(problem, key, value)
    else:
        if tuple(value.shape) != spec.shape or np.dtype(value.dtype) != spec.dtype:
            raise ValueError(f"loader returned a mismatched tensor for {key}")
        arrays[key] = (value if execution.materializes_owned(key) else
                       copy_to_backend(value, backend, phase="snapshot",
                                       timed=unmeasured, device=device))


def _reuse(context, *, key):
    arrays, execution = context[0], context[1]
    if key not in arrays:
        if not execution.has_retained(key):
            raise RuntimeError('retained source missing from execution cache')
        arrays[key] = execution.retained_value(key)


def _release(context, *, key):
    # BlockExecution owns physical regions; this table contains owned tensors.
    del context[0][key]


def _retain(context, *, key, snapshot, backend, device):
    arrays, execution = context[0], context[1]
    if snapshot:
        arrays[key] = copy_to_backend(arrays[key], backend, phase="retention_snapshot",
                                      timed=unmeasured, device=device)
    execution.retain_tensor(key, arrays[key])


@dataclass(frozen=True)
class _ReadCall:
    index: int

    def __call__(self, context):
        context[1].begin_action(self.index, unmeasured)


def _after(context, *, index):
    context[1].end_action(index, unmeasured)


def _move(context, *, action):
    context[1].move_tensor(action, context[0], unmeasured)


def _numerical(context, *, operation):
    operation(arrays=context[0])


@dataclass(frozen=True)
class BoundProgram:
    calls: tuple
    results: tuple
    backend: object
    device: bool
    final_index: int | None
    output_layout: object = None
    preparation_keys: tuple = ()
    shared_residency: bool = False

    def operations(self):
        """Read this existing bound view; keep current arrays out of the cursor.

        Legacy calls already own their waits and events. Multi-resource cursors
        can use plan_operations' readiness protocol when those dependencies are
        exposed; this entry does not invent or move asynchronous operations.
        """
        return plan_operations(self.calls)

    def run(self, execution, arrays, *, owner_snapshot=False):
        context = (arrays, execution)
        if self.shared_residency:
            # The native cfg view exposes this run's owners once. Kernels
            # still resolve current values at every action, never cache them.
            direct = getattr(arrays, '_dense_call_context', None)
            if direct is not None:
                context += direct()
        try:
            queue = getattr(execution, 'read_ahead', None)
            if queue is not None and getattr(queue, 'split_preparation', False):
                def ready(operation):
                    return (not isinstance(operation, _ReadCall)
                            or execution.reads_ready(operation.index))

                def submit(resource, current):
                    nonlocal operation
                    operation = current
                    current(context)

                operation = None
                # IO and CPU preparation cursors are driven by the existing
                # bounded read service's workers. This cursor submits the saved
                # compute-stream calls only after their host dependency exists;
                # DeviceSourceLoader preserves actual H2D event ordering.
                drive_plan_cursors({'compute': plan_operations(self.calls, ready=ready)},
                                   submit=submit, wait=execution.wait_for_reads)
            else:
                for operation in self.operations():
                    operation(context)
        except MemoryError as error:
            if self.device:
                from .allocation_failure import annotate_allocation_failure
                annotate_allocation_failure(error,execution,arrays,operation)
            raise
        values = (self.output_layout.copy(arrays, self.backend, owner_snapshot=owner_snapshot)
                  if self.output_layout is not None else
                  tuple(copy_output(key, arrays, self.backend,
                                    timed=unmeasured, device=self.device)
                        for key in self.results))
        # Events after the last Action follow output snapshots, as in replay.
        if self.final_index is not None:
            execution.end_action(self.final_index, unmeasured)
        return values


def bind_program(state, backend, starts, members, layouts, kernels, events, retained, operations,
                 *, shared_residency=False, _active_actions=None, tensor_release_indices=(),
                 scalar_starts=None, scalar_members=None,
                 cartesian_starts=None, cartesian_members=None):
    """Use P's saved order/batches/events; do not select or move operations.

    A cfg caller shares P's global array owners between instances. Its reused
    Actions already address resident values, so only their saved events need
    calls. Ordinary scene callers still restore separate retained caches.
    """
    before = {event.action_index for event in events if event.kind == 'read'}
    after = {event.action_index for event in events if event.kind != 'read'}
    after.update(tensor_release_indices)
    keep = frozenset(retained)
    contract_dense, contract_dense_batch, contract_sparse_bridge, contract_baryon_join = operations
    device = getattr(backend, '__name__', '') == 'cupy'
    calls = []
    scalar_starts = {} if scalar_starts is None else scalar_starts
    scalar_members = {} if scalar_members is None else scalar_members
    cartesian_starts = {} if cartesian_starts is None else cartesian_starts
    cartesian_members = {} if cartesian_members is None else cartesian_members
    preparation_keys = []
    indexed_actions = enumerate(state.actions)
    if _active_actions is not None and shared_residency and not keep:
        # X collected non-reused calls during the kernel-binding walk. Add
        # every in-range event position and the initial queue kick, preserving
        # the same order. Final events remain after output snapshots below.
        indices = set(_active_actions)
        indices.update(index for index in before | after if 0 <= index < len(state.actions))
        if state.actions:
            indices.add(0)
        indexed_actions = ((index, state.actions[index]) for index in sorted(indices))
    for index, action in indexed_actions:
        # Kick the continuous queue even when the first Action has no read.
        if index == 0 or index in before:
            calls.append(_ReadCall(index))
        kind, key = action.kind, action.tensor
        if action.reused:
            if not shared_residency:
                calls.append(partial(_reuse, key=key))
        elif kind == 'load':
            preparation_keys.append(key)
            calls.append(partial(_load, key=key, spec=state.catalog.get(key),
                                 problem=state.problem, backend=backend, device=device))
        elif kind == 'contract':
            if index in cartesian_starts:
                calls.append(cartesian_starts[index].bind_call(shared_residency=shared_residency))
            elif index in cartesian_members:
                pass  # Each original child cube was published at the group start.
            elif index in scalar_starts:
                calls.append(scalar_starts[index].bind_call(shared_residency=shared_residency))
            elif index in scalar_members:
                pass  # The first member already published this independent owner.
            elif index in starts:
                operation = partial(contract_dense_batch, starts[index], backend=backend,
                                    bound_kernel=kernels[index])
                calls.append(partial(_numerical, operation=operation))
            elif index not in members:
                if contract_dense is builtin_contract_dense:
                    calls.append(bind_dense_call(action, kernels[index], backend,
                                                 shared_residency=shared_residency))
                else:
                    operation = partial(contract_dense, action, backend=backend, timed=unmeasured,
                                        device=device, bound_kernel=kernels[index])
                    calls.append(partial(_numerical, operation=operation))
        elif kind in ('sparse_bridge', 'baryon_join'):
            fn = contract_sparse_bridge if kind == 'sparse_bridge' else contract_baryon_join
            operation = (partial(fn, state, action) if kind == 'sparse_bridge'
                         else partial(fn, action))
            calls.append(partial(_numerical, operation=partial(
                operation, backend=backend, timed=unmeasured, device=device)))
        elif kind == 'release':
            calls.append(partial(_release, key=key))
        elif kind in ('spill_disk', 'restore_host', 'discard'):
            calls.append(partial(_move, action=action))
        elif kind not in ('read_block', 'close_block'):
            raise ValueError(f"unsupported archived action kind {kind!r}")
        if kind in ('load', 'contract', 'sparse_bridge', 'baryon_join') and key in keep:
            calls.append(partial(_retain, key=key,
                                 snapshot=kind == 'sparse_bridge' and not action.reused,
                                 backend=backend, device=device))
        if index in after:
            calls.append(partial(_after, index=index))
    return BoundProgram(tuple(calls), state.results, backend, device,
                        len(state.actions) if len(state.actions) in after else None,
                        bind_output_layout(
                            state.results, state.catalog, backend=backend,
                            validate=False,
                        ) if device and callable(getattr(backend, 'empty', None)) else None,
                        tuple(preparation_keys), shared_residency)
