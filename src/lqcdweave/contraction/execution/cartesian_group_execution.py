"""Combine qualified original Cartesian layouts, preserving their owners/events."""
from dataclasses import dataclass
from functools import partial
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import CartesianReductionGroup, DenseKernelBatch
from .diagnostics import debug_enabled
from .matrix_reduction_group import bind_matrix_reduction_group
from .reduction_group_runtime import _native_output_rounding
from .operations.dense_batch import _BatchLayout, _bind_contiguous_concatenation
from .operations.packing_kernel import (
    bind_packing, _bind_packing_values, _MAX_RAW_INPUTS, _MAX_ARGUMENT_BYTES,
)


# Failed synchronization must retain intermediate packing allocations and the
# private allocation pool as well as the original asynchronous exception frames.
_FAILED_SUBMISSIONS = []


@dataclass(frozen=True)
class _CombinedPackingSide:
    members: tuple
    shape: tuple
    shapes: tuple
    dtype: object
    values: object
    borrow: object
    pack_original: object

    def pack(self, arrays, backend, owners=None):
        values = self.values(arrays)
        # Packing consumes original pointers, before the reduction can validate
        # its packed inputs. Reject changed source metadata before raw access.
        for value, shape in zip(values, self.shapes, strict=True):
            if value.shape != shape or value.dtype != self.dtype:
                raise ValueError('Cartesian group packing input differs from saved shape/dtype')
        ready = self.borrow(values, self.dtype)
        if ready is not None:
            if owners is not None:
                owners.append(ready)
            return ready
        destination = backend.empty(self.shape, dtype=self.dtype, order='C')
        if owners is not None:
            owners.append(destination)  # Protect even a first-pack launch failure.
        return self.pack_original(values, destination)


def _bind_combined_packing(layouts, catalog, backend):
    """Use the original equal-stack copier for bounded compatible child sides.

    Each child's full inner shape/order is preserved, not just its flattened K.
    A side uses the existing bounded 32-pointer chunks. Each chunk's worst
    mixed-stride ABI must fit the existing argument budget.
    Unsupported sides keep their original child packing, chosen at bind time.
    """
    result = []
    for axis in range(2):
        members = tuple(index for index, layout in enumerate(layouts)
                        if not layout.sides[axis].prepared)
        if len(members) < 2:
            continue
        sides = tuple(layouts[index].sides[axis] for index in members)
        first = sides[0]
        if (any(not side.stacked or not side.prepend or side.axis != 0
                or side.original_packer is None or side.input_layouts is None
                or side.shape != first.shape for side in sides)):
            continue
        # The existing mixed-stride kernel may retain one stride per non-unit
        # inner dimension; do not rely on current contiguity to shrink its ABI.
        rank = sum(dimension != 1 for dimension in first.shape[1:])
        if 8 * (_MAX_RAW_INPUTS * (1 + rank) + 3) > _MAX_ARGUMENT_BYTES:
            continue
        operands = tuple(operand for side in sides for operand in side.operands)
        input_layouts = tuple(layout for side in sides for layout in side.input_layouts)
        shapes = tuple(catalog[key].shape for key, _ in operands)
        logical = tuple(tuple(1 if index is None else shape[index] for index in layout)
                        for shape, layout in zip(shapes, input_layouts, strict=True))
        expected = (1, *first.shape[1:])
        if (any(shape != expected for shape in logical)
                or any(len(side.operands) != first.shape[0] for side in sides)):
            continue
        dtype = layouts[0].dtype
        packer = bind_packing(backend, logical, (dtype,) * len(logical), 0, dtype,
                              input_layouts=input_layouts)
        original = getattr(packer, 'pack_original', None)
        if original is None:
            continue
        shape = (sum(side.shape[0] for side in sides), *first.shape[1:])
        # ceil(sum bytes / 512) <= sum ceil(child bytes / 512): this shared
        # allocation fits the existing independently rounded packing envelope.
        combined = _CombinedPackingSide(members, shape, shapes, dtype,
            _bind_packing_values(operands),
            _bind_contiguous_concatenation(0, shape, input_layouts=input_layouts), original)
        result.append((axis, combined))
    return tuple(result)


@dataclass(frozen=True)
class BoundCartesianGroup:
    record: CartesianReductionGroup
    batches: tuple
    layouts: tuple
    kernel: object
    backend: object
    deferred: bool
    packing: tuple = ()

    def execute(self, arrays, *, timed=None, device=True):
        packed, tasks, outputs = [], [], None
        try:
            combined_values = [[None, None] for _ in self.layouts] if self.packing else None
            for axis, combined in self.packing:
                value = (combined.pack(arrays, self.backend, packed) if timed is None else
                         timed('dense_batch_pack', partial(combined.pack, arrays, self.backend, packed), device=device))
                rows = self.layouts[combined.members[0]].matrix_shapes[axis][0]
                for slot, child in enumerate(combined.members):
                    combined_values[child][axis] = value[slot*rows:(slot+1)*rows]
            for child, layout in enumerate(self.layouts):
                values = []
                for axis, (side, shape) in enumerate(zip(layout.sides, layout.matrix_shapes, strict=True)):
                    value = None if combined_values is None else combined_values[child][axis]
                    if value is None:
                        if (timed is not None and not side.prepared and
                                (layout.dtype is not None or len(side.operands)>1)):
                            value = timed('dense_batch_pack',
                                partial(side.pack, arrays, self.backend, layout.dtype), device=device)
                        else:
                            value = side.pack(arrays, self.backend, layout.dtype)
                        packed.append(value)  # Retain before reshape/later packing can fail.
                    values.append(value if value.shape == shape else value.reshape(shape))
                tasks.append(tuple(values))
            tasks = tuple(tasks)
            outputs = (self.kernel(tasks) if timed is None else
                       timed('contract_kernel', lambda: self.kernel(tasks), device=device))
            publish = getattr(arrays, '_publish_result_views', None)
            deferred = self.deferred and timed is None and not debug_enabled() and callable(publish)
            if deferred:
                for layout, output in zip(self.layouts, outputs, strict=True):
                    cube = output if output.shape == layout.output_shape else output.reshape(layout.output_shape)
                    publish(cube, layout.deferred_layouts)
            else:
                results = {}
                for layout, output in zip(self.layouts, outputs, strict=True):
                    cube = output if output.shape == layout.output_shape else output.reshape(layout.output_shape)
                    results.update(layout.views(cube))
                arrays.update(results)  # Build all ordinary views before publication.
        except BaseException as error:
            try:
                self.backend.cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                from .device import active_device_execution
                active = active_device_execution()
                pool = getattr(active, 'pool', None) if active is not None else None
                _FAILED_SUBMISSIONS.append((self.backend, arrays, packed, tasks, outputs, pool, error))
                if callable(getattr(error, 'add_note', None)):
                    error.add_note(f'Cartesian group cleanup also failed: {cleanup!r}')
            raise

    def bind_call(self, *, shared_residency=False):
        if not shared_residency:
            def call(context):self.execute(context[0])
            return call
        # Reuse the same cfg mapping protocol, including unmaterialized owner
        # views and zero-copy packing. No alternate preparation path is needed.
        from .cfg_runtime import _NodeView
        def call(context):
            arrays = context[0]
            if len(context) == 4 and type(arrays) is not _NodeView:
                arrays = _NodeView(context[3], context[2])
            self.execute(arrays)
        return call


def bind_cartesian_groups(state, groups, backend, kernel_cache, starts, kernels, scalar_members):
    """Fuse only existing child raw kernels; unsupported bindings stay original."""
    if not groups or debug_enabled() or getattr(backend, '__name__', '') != 'cupy':
        return {}, {}
    actual = {batch.action_indices:batch for batch in starts.values() if type(batch) is DenseKernelBatch}
    group_starts, members = {}, {}
    share_small_outputs = _native_output_rounding(state)
    for group in groups:
        if type(group) is not CartesianReductionGroup:
            raise TypeError('Cartesian execution requires immutable CartesianReductionGroup records')
        indices = tuple(index for child in group.batch_action_indices for index in child)
        if any(index in members or index in scalar_members for index in indices):
            raise ValueError('Cartesian reduction group overlaps another physical operation')
        if (any(child not in actual for child in group.batch_action_indices)
                or any(state.actions[index].reused for index in indices)):
            continue
        batches = tuple(actual[child] for child in group.batch_action_indices)
        layouts = []
        for batch in batches:
            bound = kernels[batch.action_indices[0]]
            layout = getattr(getattr(bound, 'func', None), '__self__', None)
            if (type(bound) is not partial or type(layout) is not _BatchLayout
                    or not all(batch.stacked_sides)):
                break
            raw = layout.precise_product
            (m,k),(n,rk) = layout.matrix_shapes
            if (raw is None or not getattr(raw, 'raw_reduction', False)
                    or getattr(raw, 'raw_output_tile', None) != (2,2)
                    or layout.dtype not in (np.dtype('float32'),np.dtype('complex64'))
                    or rk != k or not layout.sides[1].prepend
                    or prod(layout.output_shape) != m*n):
                break
            layouts.append(layout)
        if len(layouts) != len(batches):
            continue
        shapes, dtype = layouts[0].matrix_shapes, layouts[0].dtype
        if any(layout.matrix_shapes != shapes or layout.dtype != dtype for layout in layouts[1:]):
            raise ValueError('Cartesian group members differ from their bound matrix geometry')
        allowances = tuple(layout.precise_product.workspace_allowance for layout in layouts)
        # Cache by stable full child records as well as geometry. Shape-only
        # sharing would thrash the combined stride LRU between unrelated groups.
        key = ('cartesian-reduction-group', group.batch_action_indices, batches, shapes, dtype, allowances, share_small_outputs)
        if key not in kernel_cache:
            kernel_cache[key] = bind_matrix_reduction_group(shapes,dtype,allowances,backend,
                share_small_outputs=share_small_outputs)
        kernel = kernel_cache[key]
        if kernel is None:
            continue
        wide = 16 if dtype.kind == 'c' else 8
        outputs = shapes[0][0]*shapes[1][0]
        if any(layout.precise_product.workspace_bytes != outputs*(kernel.blocks*wide+dtype.itemsize)
               for layout in layouts):
            raise ValueError('Cartesian group changed the original wide reduction block count')
        if kernel.partial_bytes > group.workspace_bytes:
            raise ValueError('Cartesian group raw partials exceed its saved workspace')
        bound = BoundCartesianGroup(group,batches,tuple(layouts),kernel,backend,
                                    all(layout.deferred_layouts is not None for layout in layouts),
                                    _bind_combined_packing(layouts, state.catalog, backend))
        group_starts[group.batch_action_indices[0][0]] = bound
        members.update((index,bound) for index in indices)
    return group_starts, members
