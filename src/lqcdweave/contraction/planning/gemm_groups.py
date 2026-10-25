"""Coalesce ready resident Cartesian kernels without changing saved Actions."""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseGemmGroup, dense_batch_matrix_shape, dense_batch_memory,
    validate_executable_resources, GEMM_GROUP_MAX_BATCHES,
)
from .dense_batch_schedule import _nodes


def lower_independent_kernel_groups(plan, *, profile, target_runtime, factorized_retile=True):
    """One plan-level physical grouping service, after dense/storage binding.

    Caller owns the kernel_batching switch. Runtime is an explicit target
    declaration; unknown/CPU/HIP targets retain their existing dense batches.
    """
    from .factorized_tiles import retile_factorized_reductions
    selected, tiles = (retile_factorized_reductions(plan,profile=profile,target_runtime=target_runtime)
        if factorized_retile else (plan,dict(applied=False,reason='disabled')))
    from .factorized_reduction import lower_factorized_reductions
    selected, factorized = lower_factorized_reductions(selected, profile=profile, target_runtime=target_runtime)
    selected, gemm = lower_independent_dense_batches(selected, profile=profile,
                                                    target_runtime=target_runtime)
    from .scalar_reduction_groups import lower_scalar_reduction_groups
    selected, scalar = lower_scalar_reduction_groups(selected, target_runtime=target_runtime)
    from .cartesian_reduction_groups import lower_cartesian_reduction_groups
    selected, cartesian = lower_cartesian_reduction_groups(selected, target_runtime=target_runtime)
    return selected, dict(factorized_tile_lowering=tiles,factorized_wide_lowering=factorized, independent_gemm_lowering=gemm,
        scalar_reduction_lowering=scalar, cartesian_reduction_lowering=cartesian)


def lower_independent_gemm_path(path, *, target_runtime, factorized_retile=True):
    """W handoff for a newly selected ordinary path, never an archive replay."""
    if target_runtime != 'cuda':
        return path
    if path.batch_paths:
        paths = tuple(lower_independent_gemm_path(child, target_runtime=target_runtime,factorized_retile=factorized_retile)
                      for child in path.batch_paths)
        if all(new is old for new, old in zip(paths, path.batch_paths)):
            return path
        from ..performance.api import predict_plan_cost
        prediction = dict(path.prediction)
        prediction.update(predict_plan_cost(tuple(child.plan for child in paths), path.profile,
            workload=prediction.get('workload', {}), provenance=prediction.get('provenance', {})))
        if 'predicted_seconds' in prediction:
            prediction['predicted_seconds'] += sum(new.plan.seconds-old.plan.seconds
                for new, old in zip(paths, path.batch_paths))
        return replace(path, plan=paths[0].plan, plan_identity=id(paths[0].plan),
                       batch_paths=paths, prediction=prediction)
    if path.plan is None:
        return path
    if path.prediction.get('kernel_batching', True):
        selected, lowering = lower_independent_kernel_groups(path.plan, profile=path.profile,
            target_runtime=target_runtime,factorized_retile=factorized_retile)
    else:
        # This is an explicit numerical reduction of one selected factor
        # graph, independent of optional grouping of separate submissions.
        from .factorized_tiles import retile_factorized_reductions
        selected, tiles = (retile_factorized_reductions(path.plan,profile=path.profile,target_runtime=target_runtime)
            if factorized_retile else (path.plan,dict(applied=False,reason='disabled')))
        from .factorized_reduction import lower_factorized_reductions
        selected, factorized = lower_factorized_reductions(selected, profile=path.profile,
                                                          target_runtime=target_runtime)
        lowering = dict(factorized_tile_lowering=tiles,factorized_wide_lowering=factorized)
    if not any(stats['applied'] for stats in lowering.values()):
        return path
    from ..performance.api import predict_plan_cost
    prediction = dict(path.prediction)
    prediction.update(predict_plan_cost((selected,), path.profile,
        workload=prediction.get('workload', {}), provenance=prediction.get('provenance', {})))
    if 'predicted_seconds' in prediction:
        prediction['predicted_seconds'] += selected.seconds-path.plan.seconds
    prediction.update(lowering)
    search = path.search
    if search is not None:
        search = replace(search, plan=selected,
            statistics=dict(search.statistics, **lowering))
    return replace(path, plan=selected, plan_identity=id(selected), prediction=prediction, search=search)


class _RangePeak:
    """Incremental range additions over the existing memory trajectory."""
    def __init__(self, values):
        self.n = len(values)
        self.peak = [0] * (4*self.n)
        self.lazy = [0] * (4*self.n)
        def build(node, begin, end):
            if end-begin == 1:
                self.peak[node] = values[begin]
            else:
                mid = (begin+end)//2
                build(2*node, begin, mid); build(2*node+1, mid, end)
                self.peak[node] = max(self.peak[2*node], self.peak[2*node+1])
        build(1, 0, self.n)

    def add(self, begin, end, amount):
        def update(node, left, right):
            if right <= begin or left >= end:
                return
            if begin <= left and right <= end:
                self.peak[node] += amount
                self.lazy[node] += amount
                return
            mid = (left+right)//2
            update(2*node, left, mid); update(2*node+1, mid, right)
            self.peak[node] = self.lazy[node] + max(self.peak[2*node], self.peak[2*node+1])
        if begin < end and amount:
            update(1, 0, self.n)

    @property
    def maximum(self):
        return self.peak[1]


def lower_independent_dense_batches(plan, *, profile, target_runtime):
    """Select estimated profitable CUDA groups within both saved memory caps.

    This P pass is explicit about the execution runtime, so a planning host
    never probes a GPU or mistakes a HIP target for CUDA. Candidate feasibility
    uses range updates; only the accepted result gets a full final audit.
    """
    batches = plan.kernel_batches
    if (target_runtime != 'cuda' or plan.state.problem.backend != 'cupy'
            or not batches or any(isinstance(b, DenseGemmGroup) for b in batches)
            or not plan.memory or not plan.feasible or plan.explicit_resources
            or plan.retention_planned or plan.retained_before or plan.retained_after
            or plan.retained_releases
            or any(a.kind not in ('load', 'release', 'contract', 'sparse_bridge', 'baryon_join')
                   for a in plan.state.actions)
            or not callable(getattr(profile, 'gemm_group_seconds', None))):
        return plan, dict(applied=False, reason='unsupported_domain_or_model')
    graph = _nodes(plan.state, batches)
    if graph is None:
        return plan, dict(applied=False, reason='cyclic_batch_dag')
    nodes, parents, _ = graph
    actions, catalog = plan.state.actions, plan.state.catalog
    node_for_tensor = {a.tensor: i for i, group in enumerate(nodes) for a in group}
    firsts = {b.action_indices[0]: i for i, b in enumerate(batches)}
    region, regions = 0, {}
    for index, action in enumerate(actions):
        regions[index] = region
        if action.kind in ('sparse_bridge', 'baryon_join') or action.core is not None:
            region += 1
    geometries = [((b.packing_dtype, *dense_batch_matrix_shape(b, catalog))
                   if b.packing_dtype is not None and not (
                       any(b.stacked_sides) and b.packing_dtype in ('float32', 'complex64'))
                   else None) for b in batches]
    by_geometry = defaultdict(list)
    inputs, batch_nodes = [], []
    for i, batch in enumerate(batches):
        by_geometry[geometries[i]].append(i)
        inputs.append(frozenset(o.tensor for o in (*batch.left, *batch.right)))
        batch_nodes.append(node_for_tensor[batch.results[0].tensor])

    ends = {a.tensor: i for i, a in enumerate(actions) if a.kind == 'release'}
    owner_ends = [max(ends.get(r.tensor, len(actions)+1) for r in b.results) for b in batches]
    positions = [p.action_index for p in plan.memory]
    if positions != sorted(positions):
        raise ValueError('memory trajectory must follow saved Action order')
    peak = _RangePeak([p.device_bytes for p in plan.memory])
    contract_points = defaultdict(list)
    for i, point in enumerate(plan.memory):
        if point.phase == 'contract':
            contract_points[point.action_index].append(i)

    live, done, consumed = set(), set(), set()
    chosen = {}
    rejected = attempts = 0
    for index, action in enumerate(actions):
        member = firsts.get(index)
        if member is not None and member not in consumed:
            geometry = geometries[member]
            ready = sorted((i for i in by_geometry[geometry]
                     if geometry is not None and i not in consumed
                     and batches[i].action_indices[0] >= index
                     and regions[batches[i].action_indices[0]] == regions[index]
                     and parents[batch_nodes[i]] <= done and inputs[i] <= live),
                     key=lambda i: batches[i].action_indices[0])
            # Keep this saved first kernel as the first lane. No new load,
            # preparation, operation order or canonical expression is selected.
            ready = [member] + [i for i in ready if i != member] if member in ready else []
            for count in range(min(GEMM_GROUP_MAX_BATCHES, len(ready)), 2, -1):
                selected = ready[:count]
                children = tuple(batches[i] for i in selected)
                m, k, n = geometry[1:]
                if max(m, k, n) >= (1 << 31):
                    continue
                itemsize = np.dtype(geometry[0]).itemsize
                matrix_seconds = profile.contract_seconds(m*k*n, (m*k+k*n+m*n)*itemsize,
                                                           dtype=geometry[0])
                packing_seconds = sum(b.seconds-matrix_seconds for b in children)
                if packing_seconds < -1e-12:
                    continue  # A changed model cannot reinterpret saved costs.
                seconds = profile.gemm_group_seconds((m, k, n), geometry[0],
                    batch_count=count, packing_seconds=max(0.0, packing_seconds))
                if seconds >= sum(b.seconds for b in children):
                    continue
                group = DenseGemmGroup(children, (m, k, n), seconds)
                changes = [(bisect_left(positions, index),
                            bisect_left(positions, max(owner_ends[i] for i in selected)),
                            group.output_bytes)]
                for i in selected:
                    child = batches[i]
                    changes.append((bisect_left(positions, child.action_indices[0]),
                                    bisect_left(positions, owner_ends[i]), -child.output_bytes))
                    changes.extend((p, p+1, -child.packing_bytes)
                                   for p in contract_points[child.action_indices[0]])
                changes.extend((p, p+1, group.packing_bytes) for p in contract_points[index])
                attempts += 1
                for begin, end, amount in changes:
                    peak.add(begin, end, amount)
                if peak.maximum > plan.memory_limit:
                    for begin, end, amount in reversed(changes):
                        peak.add(begin, end, -amount)
                    rejected += 1
                    continue
                chosen[member] = group
                consumed.update(selected)
                for i in selected:
                    done.add(batch_nodes[i])
                    live.update(r.tensor for r in batches[i].results)
                break
            if member not in consumed:
                consumed.add(member)
                done.add(batch_nodes[member])
                live.update(r.tensor for r in batches[member].results)
        if action.kind == 'load':
            live.add(action.tensor)
        elif action.kind == 'release':
            live.discard(action.tensor)
        elif action.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            live.add(action.tensor)
            done.add(node_for_tensor[action.tensor])
    if not chosen:
        return plan, dict(applied=False, reason='no_profitable_feasible_resident_group',
                          resource_attempts=attempts, resource_rejections=rejected)
    grouped_members = {id(child) for group in chosen.values() for child in group.batches}
    selected_batches = tuple(chosen[i] if i in chosen else batch
        for i, batch in enumerate(batches) if i in chosen or id(batch) not in grouped_members)
    base_memory = dense_batch_memory(plan, subtract=True)
    candidate = replace(plan, kernel_batches=selected_batches, memory=base_memory)
    memory = dense_batch_memory(candidate)
    host, device = max(p.host_bytes for p in memory), max(p.device_bytes for p in memory)
    if device != peak.maximum:
        raise RuntimeError('incremental GEMM resource bound differs from final owner audit')
    candidate = replace(candidate, memory=memory, peak_host_bytes=host, peak_device_bytes=device,
        feasible=device <= plan.memory_limit and (plan.host_memory_limit is None or host <= plan.host_memory_limit))
    validate_executable_resources(candidate)
    return candidate, dict(applied=True, groups=len(chosen),
        grouped_batches=sum(len(group.batches) for group in chosen.values()),
        implementation='cuda-cublas-gemm-batched-v1', cost_status='estimated-from-dense-v1',
        original_seconds=plan.seconds, seconds=candidate.seconds,
        peak_host_bytes=host, peak_device_bytes=device, resource_audited=True,
        resource_attempts=attempts, resource_rejections=rejected)
