"""Linear lifetime accounting for a fixed contraction stream.

Only allocation sizes and selected kernel scratch are needed here. Pending
terms, axis matching, shared-operation classification and costs belong to P.
"""
from collections import Counter
import math
import numpy as np
from lqcdweave.contracts.correlator.api import Action, FactorizedWideReduction, MesonWholeKernel
from lqcdweave.contracts.correlator.cores import TensorCoreSpec, baryon_component_pattern

from .state import _nbytes


def supports_storage_lifetimes(state):
    return (not state.problem.sparse_bridges
            and all(type(a) is Action and (a.kind in ('load','release') or
                    a.kind == 'contract' and (a.core is None or type(a.core) in
                        (FactorizedWideReduction, MesonWholeKernel)) or
                    a.kind == 'baryon_join' and (a.core is None or type(a.core) is TensorCoreSpec))
                    for a in state.actions))


def _join_workspace(action, catalog, sizes):
    """Consume the chosen local path; never search or price it again."""
    from .state import _join_path_cost, _join_path_workspace
    from ..performance.api import estimate_vertex_kernel
    if action.core is None:
        parts = ((action.operands, action.equation),)
    else:
        labels, *legs = action.equation.inputs
        leg_keys = action.operands[len(action.core.inputs):]
        parts = tuple((part.inputs + leg_keys, baryon_component_pattern(
            labels, legs, action.equation.output, len(catalog[part.inputs[1]].shape)))
            for part in action.core.components)
    workspace = 0
    for keys, equation in parts:
        specs = tuple(catalog[key] for key in keys)
        itemsize = np.result_type(*(spec.dtype for spec in specs)).itemsize
        if action.kernel_description is not None:
            scratch = estimate_vertex_kernel(action.kernel_description, equation, specs)[2]
        else:
            shapes = tuple(spec.shape for spec in specs)
            largest = _join_path_cost(equation, shapes, action.kernel_path)[1]
            scratch = max(largest * itemsize + sum(sizes[key] for key in keys),
                          _join_path_workspace(equation, shapes, action.kernel_path) * itemsize)
        if action.core is not None:
            scratch = max(scratch, math.prod(catalog[action.tensor].shape) * itemsize)
        workspace = max(workspace, scratch)
    return workspace + (2 * sizes[action.tensor] if action.core is not None else 0)


class StorageLifetimes:
    """Track actual owners/workspace without rebuilding pending algebra.

    This consumes the selected catalog/order, never infers a new equation,
    sharing relation, dtype or cost. Full independent algebra replay is debug.
    """
    def __init__(self,state):
        self.state=state
        self.sizes={k:_nbytes(s) for k,s in state.catalog.items()}
        self.allocations={k:(s.snapshot_bytes if s.snapshot_bytes is not None else self.sizes.get(k,0))
                          for k,s in state.problem.sources.items()}
        self.loaded=set()
        self.remaining=Counter(k for a in state.actions if a.kind in ('contract', 'baryon_join') for k in a.operands)
        self.outputs=frozenset(state.results)
        self.resident_bytes=0

    @property
    def resident_host_bytes(self):
        return self.resident_bytes if self.state.problem.backend=='numpy' else 0

    def advance(self,action):
        key=action.tensor
        if action.kind=='load':
            if key not in self.allocations or key in self.loaded:
                raise ValueError('load requires an absent original input')
            self.loaded.add(key)
            self.resident_bytes+=self.allocations[key]
            return 0
        if action.kind=='release':
            if key not in self.loaded or key in self.outputs:
                raise ValueError('release requires a resident non-output tensor')
            if key not in self.allocations and self.remaining[key]:
                raise ValueError('intermediate recomputation is unsupported')
            self.loaded.remove(key)
            self.resident_bytes-=self.allocations.get(key,self.sizes.get(key,0))
            return 0
        if any(k not in self.loaded for k in action.operands):
            raise ValueError('contract operands must be resident')
        for operand in action.operands:
            self.remaining[operand]-=1
        if action.reused:
            if key not in self.loaded:
                raise ValueError('reused output must be resident')
            return 0
        if key in self.loaded or key not in self.sizes:
            raise ValueError('new dense result requires an absent declared tensor')
        size=self.sizes[key]
        self.loaded.add(key)
        self.resident_bytes+=size
        if action.kind == 'baryon_join':
            return _join_workspace(action, self.state.catalog, self.sizes)
        if type(action.core) is MesonWholeKernel:
            from ..performance.api import estimate_meson_whole
            return estimate_meson_whole(action.core, self.state.catalog,
                                       self.state.catalog[key].shape)[2]
        if type(action.core) is FactorizedWideReduction:
            return action.core.workspace_bytes
        # Same ordinary dense workspace allowance as _describe_contract:
        # possible input layout copies plus one output-sized temporary.
        return sum(self.sizes[k] for k in action.operands)+size

    def finish(self):
        if not self.outputs.issubset(self.loaded):
            raise ValueError('selected path did not retain every output')
