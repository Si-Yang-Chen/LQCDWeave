"""Analytical estimate of X's fixed K/H block traversal, without array reads.

Counts follow execution.matrix_kernels (origin: the migrated meson matrix block
kernel). Traffic counts array operands/results of arithmetic calls, excluding
packing, dtype promotion and IO. Peak bounds include worst-case transpose /
reshape packs. BLAS internals need an explicit additional workspace reserve.
This estimate is not a physical slab audit or a complete execution plan.
"""
from math import prod
from dataclasses import replace
from collections import Counter
import hashlib
from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    MatrixKernelSpec, MatrixKernelEstimate, MatrixKernelStageEstimate, Action, TensorSpec,
)


def estimate_matrix_kernel(spec: MatrixKernelSpec, *, profile, workspace_reserve_bytes=0):
    """Estimate a connected/loop template using the existing CostModel protocol.

    Input bytes are a conservative sum: aliases are not deducted. The caller
    must additionally account for physical backing slabs, prepared vertices,
    pending IO, other live groups and output retention. CPU array bound only.
    Arithmetic flops use two operations per multiply-add, including complex
    values; the existing profile's dtype-specific coefficients remain in force.
    """
    if not isinstance(spec, MatrixKernelSpec):
        raise TypeError('spec must be MatrixKernelSpec')
    if type(workspace_reserve_bytes) is not int or workspace_reserve_bytes < 0:
        raise ValueError('workspace reserve must be nonnegative integer bytes')
    n, widths, item = spec.ne, spec.widths, spec.dtype.itemsize
    counts = {}

    def add(name, flops, sizes, calls=1):
        previous = counts.setdefault(name, [0, 0, 0])
        previous[0] += flops
        previous[1] += sum(sizes) * item
        previous[2] += calls

    retained = temporary = 0
    empty = (any(not spec.shapes[i][0] for i in (2,3,4,5)) if spec.kind=='connected'
             else any(not spec.shapes[i][0] for i in (1,2)))
    if spec.mode_block and empty:
        pass
    elif spec.mode_block:
        o,a = spec.shapes[0][-2:]
        if spec.kind=='connected':
            b,c = spec.shapes[1][-2:]
            cl,cr = spec.shapes[2][0],spec.shapes[3][0]
            nl,nr = spec.shapes[4][0],spec.shapes[5][0]
            tiles = (nl+spec.row_block-1)//spec.row_block
            add('right_product',128*cr*b*c*o,(16*b*c,cr*16*c*o,cr*16*b*o))
            add('right_map',32*nr*cr*b*o,(nr*cr,16*(cr+nr)*b*o))
            add('left_product',128*cl*o*a*b,(16*o*a,cl*16*a*b,cl*16*o*b))
            add('left_map',32*nl*cl*o*b,(nl*cl,16*(cl*tiles+nl)*o*b),tiles)
            add('dot',32*nl*nr*b*o,(16*(nl+nr*tiles)*b*o,nl*nr),tiles)
            retained = nr*16*b*o*item
            temporary = item*(2*max(cl,cr)*16*b*o+min(nl,spec.row_block)*16*b*o+nl*nr)
        else:
            comp,rows = spec.shapes[1][0],spec.shapes[2][0]
            add('loop_product',32*comp*o*a,(16*(1+comp)*o*a,comp))
            add('loop_map',2*rows*comp,(rows*comp,comp,rows))
            temporary = item*((1+comp)*16*o*a+comp+rows)
    elif spec.kind == 'connected':
        cl, cr = spec.shapes[2][0], spec.shapes[3][0]
        nl, nr = spec.shapes[4][0], spec.shapes[5][0]
        if cl and cr and nl and nr:
            q, w = len(widths), max(widths)
            retained = q * nr * 16 * n * n * item
            tiles = (nl + spec.row_block - 1) // spec.row_block
            # Sum widths = Ne. Count the loop template algebraically, without
            # materializing q^4 * row_tiles individual operation records.
            add('right_product', 128*cr*n**3,
                (16*n*n*(q+cr+cr*q),), q*q)
            add('right_map', 32*nr*cr*q*n*n,
                (nr*cr*q*q, 16*(cr+nr)*q*n*n), q*q)
            add('left_product', 128*cl*n**3,
                (16*n*n*(q+cl+cl*q),), q*q)
            add('left_map', 32*nl*cl*q*n*n,
                (nl*cl*q*q, 16*(cl*tiles+nl)*q*n*n), q*q*tiles)
            add('dot', 32*nl*nr*q*q*n*n,
                (16*(nl+nr*tiles)*q*q*n*n, nl*nr*q**4), q**4*tiles)
            # Scalar weighting + in-place addition are two ufunc calls.
            add('accumulate', 2*nl*nr*q**4, (5*nl*nr*q**4,), 2*q**4*tiles)
            r = min(spec.row_block, nl)
            hp, kp, kw = cr*16*n*w, cl*16*n*w, r*16*n*w
            temporary = item * max(
                hp + 16*n*w + cr*16*w*w, 2*hp,
                kp + 16*n*w + cl*16*w*w, 2*kp + kw,
                kp + kw + (r+nr)*16*w*w + 2*r*nr)
    else:
        comp, rows = spec.shapes[1][0], spec.shapes[2][0]
        if comp and rows:
            q = len(widths)
            add('loop_product', 32*comp*n*n,
                (16*(1+comp)*n*n, comp*q*q), q*q)
            add('loop_map', 2*rows*comp*q*q, ((rows*comp+comp+rows)*q*q,), q*q)
            add('accumulate', 2*rows*q*q, (5*rows*q*q,), 2*q*q)
            w = max(widths)
            temporary = item * max((1+comp)*16*w*w + comp, comp+2*rows)
    stages = tuple(MatrixKernelStageEstimate(
        name, flops, traffic, calls,
        profile.contract_seconds(flops, traffic, dtype=spec.dtype, kernel_calls=calls))
        for name, (flops, traffic, calls) in counts.items())
    return MatrixKernelEstimate(spec, stages, spec.input_bytes, spec.promotion_bytes,
        prod(spec.output_shape)*item, retained, temporary, workspace_reserve_bytes)


def estimate_matrix_action(spec, *, profile):
    """Kernel estimate plus explicit PC operands and their runtime validation."""
    estimate = estimate_matrix_kernel(spec,profile=profile)
    count = len(spec.action_shapes)-len(spec.shapes)
    if not count:
        return estimate
    elements = prod(spec.projection_shape)
    traffic = (3*spec.dtype.itemsize+2)*elements*count
    stage = MatrixKernelStageEstimate('projection_validation',2*elements*count,traffic,2*count,
        profile.contract_seconds(2*elements*count,traffic,dtype=spec.dtype,kernel_calls=2*count))
    return replace(estimate, stages=estimate.stages+(stage,),
        input_bytes=estimate.input_bytes+count*elements*spec.dtype.itemsize,
        temporary_bytes=estimate.temporary_bytes+elements*(spec.dtype.itemsize+1))


def describe_matrix_action(state, term_index, spec, *, profile):
    """Describe a complete fixed template as the existing contract Action.

    Projected templates have explicit PC coefficient tensors in their equation.
    X validates their values against the declared uniform block projection.
    Exact/search candidates do not generate this domain automatically.
    """
    if state.problem.backend != 'numpy':
        raise NotImplementedError('matrix K/H Actions currently require NumPy')
    if not 0 <= term_index < len(state.pending) or state.pending[term_index] is None:
        raise ValueError('matrix Action requires a live Term')
    term = state.pending[term_index]
    inputs = tuple(f.tensor for f in term.factors)
    equation = AxisPattern(tuple(f.labels for f in term.factors), term.output)
    if equation.normalized() != spec.equation.normalized():
        raise ValueError('matrix Action equation differs from the fixed template')
    if any(key not in state.loaded for key in inputs):
        raise ValueError('matrix Action inputs must be resident')
    shapes = tuple(state.catalog[key].shape for key in inputs)
    dtypes = tuple(state.catalog[key].dtype.name for key in inputs)
    if shapes != spec.action_shapes or dtypes != spec.action_dtypes:
        raise ValueError('matrix Action layouts differ from the fixed template')
    if any(join.scene == term_index for join in state.problem.baryon_joins):
        raise ValueError('matrix Actions cannot consume baryon joins')
    if any(bridge.scene in (None,term_index) for bridge in state.problem.sparse_bridges):
        raise ValueError('matrix Actions cannot consume sparse bridges')
    key = 'matrix:' + hashlib.sha256(repr((inputs,spec)).encode()).hexdigest()[:24]
    reused = key in state.loaded
    output = TensorSpec(spec.output_shape,spec.dtype)
    if key in state.catalog and (not reused or state.catalog[key] != output):
        raise ValueError('matrix output was released or has an inconsistent layout')
    estimate = estimate_matrix_action(spec,profile=profile)
    workspace = (estimate.promotion_bytes + estimate.retained_product_bytes
                 + estimate.temporary_bytes)
    action = Action('contract',key,inputs,spec.equation,
        seconds=0.0 if reused else estimate.seconds, reused=reused,
        term_index=term_index, positions=tuple(range(len(inputs))),core=spec)
    return action, output, 0 if reused else workspace


def replay_matrix_action(state, action, profile, memory_limit=1 << 100):
    """Independently re-describe, then apply a selected fixed matrix Action."""
    if action.kind != 'contract' or not isinstance(action.core,MatrixKernelSpec):
        raise ValueError('not a fixed matrix contract Action')
    actual, output, workspace = describe_matrix_action(state,action.term_index,action.core,profile=profile)
    if action != actual:
        raise ValueError('matrix Action differs from its inputs, axes or independently computed cost')
    size = prod(output.shape)*output.dtype.itemsize
    peak = max(state.peak_bytes,state.resident_bytes+(0 if action.reused else size+workspace))
    if peak > memory_limit:
        raise MemoryError('fixed matrix computation exceeds declared memory cap')
    pending, results = list(state.pending),list(state.results)
    pending[action.term_index],results[action.term_index] = None,action.tensor
    return state._seed_residency(replace(state, pending=tuple(pending),results=tuple(results),
        catalog=state.catalog.updated(((action.tensor,output),)), loaded=state.loaded|{action.tensor},
        seconds=state.seconds+action.seconds, peak_bytes=peak,
        peak_host_bytes=max(state.peak_host_bytes,peak), actions=state.actions+(action,)),
        state._result_residency_delta(action.tensor,output))


def plan_matrix_actions(problem, kernels, blocks, preparations, *, profile, memory_limit,
                        host_memory_limit=None, reserved_host_bytes=0):
    """Compile complete matrix Terms and audit them with the existing IO plan.

    kernels follows problem.terms order. Only demanded input snapshots are
    prepared; they survive through their last use. No search, numeric IO, batch
    coordinate copying or backend probing occurs. Output snapshots remain live
    until the ordinary storage replay hands them to its consumer.
    """
    from .state import ContractionState
    from .storage import logical_source_problem, logical_cost_profile, plan_scene_storage
    kernels = tuple(kernels)
    if len(kernels) != len(problem.terms) or any(not isinstance(k,MatrixKernelSpec) for k in kernels):
        raise ValueError('one MatrixKernelSpec is required per complete Term')
    if problem.backend != 'numpy':
        raise NotImplementedError('matrix K/H Actions currently require NumPy')
    if any(core.mode_block for core in kernels):
        from .matrix_blocks import plan_matrix_blocks
        return plan_matrix_blocks(problem,kernels,blocks,preparations,profile=profile,
            memory_limit=memory_limit,host_memory_limit=host_memory_limit,
            reserved_host_bytes=reserved_host_bytes)
    state = ContractionState.initial(logical_source_problem(problem))
    model = logical_cost_profile(profile)
    remaining = Counter(key for term in problem.terms for key in set(term.inputs))
    for index,(term,core) in enumerate(zip(problem.terms,kernels,strict=True)):
        for key in dict.fromkeys(term.inputs):
            if key not in state.loaded:
                state = state.load(key,model,1 << 100)
        action,_,_ = describe_matrix_action(state,index,core,profile=model)
        state = replay_matrix_action(state,action,model)
        for key in dict.fromkeys(term.inputs):
            remaining[key] -= 1
            if not remaining[key]:
                state = state.release(key)
    plan = plan_scene_storage(state,blocks,preparations,profile=profile,
        memory_limit=memory_limit,host_memory_limit=host_memory_limit,
        reserved_host_bytes=reserved_host_bytes)
    if not plan.feasible:
        raise MemoryError('fixed matrix plan exceeds the physical storage budget before IO')
    return plan


__all__ = ['estimate_matrix_kernel','estimate_matrix_action','describe_matrix_action',
           'replay_matrix_action','plan_matrix_actions']
