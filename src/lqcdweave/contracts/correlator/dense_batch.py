"""Kernel lowering annotations for existing dense contraction Actions."""
from dataclasses import dataclass
import math
from dataclasses import replace
from collections import defaultdict
import numpy as np


@dataclass(frozen=True)
class DenseBatchOperand:
    tensor: str
    permutation: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "permutation", tuple(self.permutation))
        if (not self.tensor or any(type(i) is not int for i in self.permutation)
                or sorted(self.permutation) != list(range(len(self.permutation)))):
            raise ValueError("batch operands require a tensor and an axis permutation")


@dataclass(frozen=True)
class DenseBatchResult:
    tensor: str
    bounds: tuple[tuple[int, int], ...]
    squeeze_axes: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))
        object.__setattr__(self, "squeeze_axes", tuple(self.squeeze_axes))
        if (not self.tensor or any(len(b) != 2 or any(type(v) is not int for v in b)
                or b[0] < 0 or b[1] <= b[0] for b in self.bounds)):
            raise ValueError("batch results require nonempty slices")
        if (any(type(i) is not int or not 0 <= i < len(self.bounds)
                or self.bounds[i][1] - self.bounds[i][0] != 1 for i in self.squeeze_axes)
                or tuple(sorted(set(self.squeeze_axes))) != self.squeeze_axes):
            raise ValueError("batch result may remove only distinct singleton axes")


@dataclass(frozen=True)
class DenseKernelBatch:
    """One Cartesian contraction, with output views of one owned allocation.

    Indices reference existing paid Actions; no second calculation IR is held.
    Each side concatenates along at most one free axis. Explicit packing dtype
    selects C-contiguous matrix inputs on NumPy or CuPy; a proven producer
    layout may avoid the copy. Legacy NumPy records without packing dtype
    retain their original transpose/einsum behavior. Results own one allocation.
    A side without physical free axes uses a saved singleton stack axis;
    result squeeze_axes removes that batch axis without changing scientific axes.
    """
    action_indices: tuple[int, ...]
    left: tuple[DenseBatchOperand, ...]
    right: tuple[DenseBatchOperand, ...]
    left_free_rank: int
    reduction_rank: int
    right_free_rank: int
    left_axis: int
    right_axis: int
    results: tuple[DenseBatchResult, ...]
    seconds: float
    packing_bytes: int
    output_bytes: int
    packing_dtype: str | None = None
    prepared_sides: tuple[bool, bool] = (False, False)
    stacked_sides: tuple[bool, bool] = (False, False)

    def __post_init__(self):
        for name in ("action_indices", "left", "right", "results"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, 'prepared_sides', tuple(self.prepared_sides))
        object.__setattr__(self, 'stacked_sides', tuple(self.stacked_sides))
        if (len(self.stacked_sides) != 2 or any(type(v) is not bool for v in self.stacked_sides)
                or any(stacked and rank != 1 for stacked, rank in
                       zip(self.stacked_sides, (self.left_free_rank, self.right_free_rank)))):
            raise ValueError("stacked sides require one virtual free axis")
        if len(self.prepared_sides) != 2 or any(type(v) is not bool for v in self.prepared_sides):
            raise ValueError("batch preparation flags must be two booleans")
        if any(self.prepared_sides) and self.packing_dtype is None:
            raise ValueError("producer layout requires an explicit matrix packing dtype")
        if (len(self.action_indices) != len(self.results) or len(self.results) < 2
                or any(type(i) is not int or i < 0 for i in self.action_indices)
                or tuple(sorted(set(self.action_indices))) != self.action_indices):
            raise ValueError("batch indices must identify distinct ordered paid Actions")
        if (not self.left or not self.right
                or len(self.left) * len(self.right) != len(self.results)
                or any(not isinstance(o, DenseBatchOperand) for o in (*self.left, *self.right))
                or any(not isinstance(r, DenseBatchResult) for r in self.results)
                or len({r.tensor for r in self.results}) != len(self.results)):
            raise ValueError("a batch must contain one complete Cartesian product")
        if any(type(v) is not int or v <= 0 for v in (
                self.left_free_rank, self.reduction_rank, self.right_free_rank)):
            raise ValueError("batch axes must include free axes on both sides and a reduction")
        if (type(self.left_axis) is not int or not 0 <= self.left_axis < self.left_free_rank
                or type(self.right_axis) is not int
                or not self.reduction_rank <= self.right_axis < self.reduction_rank + self.right_free_rank):
            raise ValueError("concatenation axes must be free axes")
        if any(type(v) is not int or v < 0 for v in (self.packing_bytes, self.output_bytes)):
            raise ValueError("batch allocations must be nonnegative integer bytes")
        if not isinstance(self.seconds, (int, float)) or not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("batch cost must be finite and nonnegative")
        if self.packing_dtype is not None:
            dtype = np.dtype(self.packing_dtype)
            if dtype.name not in ('float32', 'float64', 'complex64', 'complex128'):
                raise ValueError("matrix batch packing requires a supported floating dtype")
            object.__setattr__(self, 'packing_dtype', dtype.name)


def dense_batch_producer_eligible(operand, actions, catalog):
    """Prove that a paid binary producer can directly emit the desired layout.

    Same-dtype, unaligned reductions lower to GEMM. Swapping operand order
    places the two free-axis groups in the requested order without a new
    numerical operation; the original logical tensor becomes a transpose view.
    """
    producers = [a for a in actions if a.kind == 'contract' and not a.reused and a.tensor == operand.tensor]
    if len(producers) != 1:
        return False
    action = producers[0]
    if action.core is not None or action.resource is not None or len(action.operands) != 2:
        return False
    axes, output = action.equation.inputs, action.equation.output
    if any(len(set(a)) != len(a) for a in (*axes, output)) or len(output) != len(operand.permutation):
        return False
    dtypes = {catalog[k].dtype for k in (*action.operands, action.tensor)}
    if len(dtypes) != 1 or next(iter(dtypes)).name not in ('float32','float64','complex64','complex128'):
        return False
    reduced = set(axes[0]) & set(axes[1])
    if not reduced or reduced & set(output) or (set(axes[0]) | set(axes[1])) - reduced != set(output):
        return False
    desired = tuple(output[i] for i in operand.permutation)
    left = tuple(i for i in desired if i in set(axes[0]) - reduced)
    right = tuple(i for i in desired if i in set(axes[1]) - reduced)
    return desired in (left + right, right + left)


def dense_batch_memory(plan, *, check=False, subtract=False):
    """Account for the full allocation until its last result view is released.

    This scans saved actions/points only; it never predicts computation costs
    or changes releases. ``check`` validates already annotated memory points.
    """
    gpu = plan.state.problem.backend == 'cupy'
    starts = {b.action_indices[0]: b for b in plan.kernel_batches}
    output_batches = {r.tensor: b for b in plan.kernel_batches for r in b.results}
    sizes = {k:math.prod(plan.state.catalog[k].shape)*plan.state.catalog[k].dtype.itemsize for k in output_batches}
    live, active, additions, counts = set(), set(), {}, defaultdict(int)
    owned_bytes = live_bytes = 0
    for index, action in enumerate(plan.state.actions):
        batch = starts.get(index)
        if batch is not None:
            active.add(id(batch))
            owned_bytes += batch.output_bytes
            counts[id(batch)] = len(batch.results)
        if action.kind == "contract" and not action.reused and action.tensor in output_batches:
            live.add(action.tensor)
            live_bytes += sizes[action.tensor]
        elif action.kind == "release" and action.tensor in live:
            live.remove(action.tensor)
            live_bytes -= sizes[action.tensor]
            owner = output_batches[action.tensor]
            counts[id(owner)] -= 1
            if counts[id(owner)] == 0:
                active.remove(id(owner))
                owned_bytes -= owner.output_bytes
        extra = owned_bytes - live_bytes
        additions[index] = extra, 0 if batch is None else batch.packing_bytes
    additions[len(plan.state.actions)] = owned_bytes - live_bytes, 0
    points = []
    for point in plan.memory:
        extra, packing = additions[point.action_index]
        if point.phase.startswith("read:"):
            # Batch starts are compute Actions preceded by all input loads,
            # so they cannot host a first-demand raw read.
            packing = 0
        if point.phase != "contract":
            packing = 0
        if check:
            retained = point.retained_device_bytes if gpu else point.retained_host_bytes
            workspace = point.workspace_device_bytes if gpu else point.workspace_host_bytes
            if retained != extra or workspace < packing:
                raise ValueError("batch memory points omit owned result or packing allocations")
            points.append(point)
        else:
            scale = -1 if subtract else 1
            if gpu:
                points.append(replace(point, retained_device_bytes=point.retained_device_bytes + scale*extra,
                                      workspace_device_bytes=point.workspace_device_bytes + scale*packing))
            else:
                points.append(replace(point, retained_host_bytes=point.retained_host_bytes + scale*extra,
                                      workspace_host_bytes=point.workspace_host_bytes + scale*packing))
    return tuple(points)


def validate_dense_batches(plan):
    """Check saved structure and ownership before IO, without replanning."""
    if plan.state.problem.backend not in ('numpy', 'cupy'):
        raise NotImplementedError("Cartesian dense kernel batches require NumPy or CuPy")
    if (plan.explicit_resources or plan.retention_planned or plan.retained_before
            or plan.retained_after or plan.retained_releases):
        raise NotImplementedError("kernel batches require a single fixed tensor lifetime")
    from .gemm_group import DenseGemmGroup, dense_batch_leaves, dense_batch_matrix_shape
    actions, catalog = plan.state.actions, plan.state.catalog
    batch_outputs = {r.tensor for b in plan.kernel_batches for r in b.results}
    producer_layouts = {}
    paid_producers = defaultdict(list)
    for action in actions:
        if action.kind == 'contract' and not action.reused:
            paid_producers[action.tensor].append(action)
    indices, results_seen, starts = set(), set(), {}
    for batch in dense_batch_leaves(plan.kernel_batches):
        if not isinstance(batch, DenseKernelBatch):
            raise TypeError("kernel batches require DenseKernelBatch records")
        if indices.intersection(batch.action_indices):
            raise ValueError("one paid Action cannot belong to multiple batches")
        indices.update(batch.action_indices)
        starts[batch.action_indices[0]] = batch
        if any(i >= len(actions) for i in batch.action_indices):
            raise ValueError("batch references an unknown Action")
        side_shapes, side_bounds = [], []
        packing, operand_dtypes, inner_shape = 0, [], None
        for side, axis, free_rank, prefix, prepared, stacked in (
                (batch.left, batch.left_axis,batch.left_free_rank,0,batch.prepared_sides[0],batch.stacked_sides[0]),
                (batch.right,batch.right_axis,batch.right_free_rank,batch.reduction_rank,batch.prepared_sides[1],batch.stacked_sides[1])):
            if prepared:
                if (len(side) != 1 or side[0].tensor in batch_outputs
                        or catalog[side[0].tensor].dtype != np.dtype(batch.packing_dtype)
                        or not dense_batch_producer_eligible(side[0], paid_producers[side[0].tensor], catalog)):
                    raise ValueError("batch input cannot be generated in the saved producer layout")
                old = producer_layouts.setdefault(side[0].tensor, side[0].permutation)
                if old != side[0].permutation:
                    raise ValueError("one producer cannot emit conflicting batch layouts")
            if len(set(side)) != len(side):
                raise ValueError("duplicate batch operands")
            if (batch.packing_dtype is None
                    and len({catalog[o.tensor].dtype for o in side if o.tensor in catalog}) != 1):
                raise ValueError("mixed input dtypes require saved result-dtype packing")
            shapes, bounds, offset = [], {}, 0
            for operand in side:
                spec = catalog.get(operand.tensor)
                if spec is None or len(operand.permutation) != len(spec.shape):
                    raise ValueError("batch operand differs from the saved tensor description")
                shape = tuple(spec.shape[i] for i in operand.permutation)
                if stacked:
                    shape = (1,) + shape if prefix == 0 else shape + (1,)
                if len(shape) != free_rank + batch.reduction_rank or not all(shape):
                    raise ValueError("batch operand ranks or dimensions disagree")
                current_inner = shape[free_rank:] if prefix == 0 else shape[:prefix]
                if inner_shape is not None and current_inner != inner_shape:
                    raise ValueError("batch reduction shapes disagree")
                inner_shape = current_inner
                if shapes and any(a != b for i,(a,b) in enumerate(zip(shape,shapes[0])) if i != axis):
                    raise ValueError("batch shapes require more than one concatenation axis")
                shapes.append(shape)
                free = shape[:free_rank] if prefix == 0 else shape[prefix:]
                freeaxis = axis - prefix
                bounds[operand] = tuple((offset,offset+n) if i == freeaxis else (0,n)
                                        for i,n in enumerate(free))
                offset += shape[axis]
                operand_dtypes.append(spec.dtype)
                if not prepared and (batch.packing_dtype is not None or len(side)>1):
                    packing += math.prod(shape)*(np.dtype(batch.packing_dtype).itemsize
                                                 if batch.packing_dtype is not None else spec.dtype.itemsize)
            merged = list(shapes[0])
            merged[axis] = sum(s[axis] for s in shapes)
            side_shapes.append(tuple(merged[:free_rank] if prefix == 0 else merged[prefix:]))
            side_bounds.append(bounds)
        dtype = np.result_type(*operand_dtypes)
        if ((batch.packing_dtype is not None and np.dtype(batch.packing_dtype) != dtype)
                or (plan.state.problem.backend == 'cupy' and batch.packing_dtype is None)):
            raise ValueError("CuPy matrix batches require explicit result-dtype packing")
        pairs, names = set(), {r.tensor for r in batch.results}
        if names & results_seen:
            raise ValueError("a result cannot have two batch owners")
        results_seen |= names
        if (batch.packing_bytes != packing or batch.output_bytes != math.prod(side_shapes[0]+side_shapes[1])*dtype.itemsize
                or any(o.tensor in names for o in (*batch.left,*batch.right))):
            raise ValueError("batch allocation envelope or independent inputs disagree")
        left_by_tensor, right_by_tensor = defaultdict(list), defaultdict(list)
        for operand in batch.left:
            left_by_tensor[operand.tensor].append(operand)
        for operand in batch.right:
            right_by_tensor[operand.tensor].append(operand)
        for index, result in zip(batch.action_indices,batch.results):
            action = actions[index]
            if (action.kind != 'contract' or action.reused or action.core is not None
                    or action.resource is not None or len(action.operands)!=2 or action.tensor!=result.tensor):
                raise ValueError("batch member is not an ordinary paid binary Action")
            matches=[]
            for left in (o for k in action.operands for o in left_by_tensor[k]):
                for right in (o for k in action.operands for o in right_by_tensor[k]):
                    if {left.tensor,right.tensor} != set(action.operands) or left.tensor==right.tensor:
                        continue
                    la=action.equation.inputs[action.operands.index(left.tensor)]
                    ra=action.equation.inputs[action.operands.index(right.tensor)]
                    la=tuple(la[i] for i in left.permutation)
                    ra=tuple(ra[i] for i in right.permutation)
                    lf,kr,rf=batch.left_free_rank-int(batch.stacked_sides[0]),batch.reduction_rank,batch.right_free_rank
                    if (len(set(la))!=len(la) or len(set(ra))!=len(ra)
                            or la[lf:]!=ra[:kr] or set(la[lf:]) & set(action.equation.output)
                            or set(la) & set(ra) != set(la[lf:])
                            or action.equation.output!=la[:lf]+ra[kr:]):
                        continue
                    if result.bounds != side_bounds[0][left]+side_bounds[1][right]:
                        continue
                    matches.append((left,right))
            if len(matches)!=1 or matches[0] in pairs:
                raise ValueError("batch slices do not implement the saved axis relation")
            pairs.add(matches[0])
            spec=catalog[result.tensor]
            squeeze = ((0,) if batch.stacked_sides[0] else ()) + (
                (batch.left_free_rank,) if batch.stacked_sides[1] else ())
            if (result.squeeze_axes != squeeze or spec.dtype != dtype
                    or spec.shape != tuple(b-a for i,(a,b) in enumerate(result.bounds) if i not in squeeze)):
                raise ValueError("batch result differs from its saved TensorSpec")
        if any(a.kind!='contract' or a.tensor not in names
               for a in actions[batch.action_indices[0]:batch.action_indices[-1]+1]):
            raise ValueError("batch members must be one uninterrupted compute block")
    # Physical starts may replace several future Cartesian starts. Their inputs
    # must already be live; this also rules out dependencies inside the group.
    starts = {batch.action_indices[0]: batch for batch in plan.kernel_batches}
    for group in plan.kernel_batches:
        if isinstance(group, DenseGemmGroup):
            if plan.state.problem.backend != 'cupy':
                raise NotImplementedError('saved independent GEMM groups require a CuPy CUDA target')
            if any(dense_batch_matrix_shape(child, catalog) != group.matrix_shape for child in group.batches):
                raise ValueError('independent GEMM dimensions differ from their saved Cartesian batches')
            names = {result.tensor for result in group.results}
            if any(operand.tensor in names for operand in (*group.left, *group.right)):
                raise ValueError('independent GEMM group has an internal data dependency')
    live = set()
    for index, action in enumerate(actions):
        batch=starts.get(index)
        if batch is not None:
            if any(o.tensor not in live for o in (*batch.left,*batch.right)):
                raise ValueError("batch inputs must be prepared before its first member")
            live.update(r.tensor for r in batch.results)
        if action.kind=='load':
            live.add(action.tensor)
        elif action.kind=='release':
            live.discard(action.tensor)
        elif action.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            # A non-member reused Action hands off its existing result at this
            # exact position. X checks the owner; historical operands are not
            # recomputed. Paid batch members were already required above.
            if not action.reused and any(k not in live for k in action.operands):
                raise ValueError("batch schedule consumes an absent allocation")
            live.add(action.tensor)
    dense_batch_memory(plan,check=True)
    if (not plan.memory or plan.peak_host_bytes!=max(p.host_bytes for p in plan.memory)
            or plan.peak_device_bytes!=max(p.device_bytes for p in plan.memory)):
        raise ValueError("batch resource peaks disagree with their memory points")
