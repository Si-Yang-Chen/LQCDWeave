"""Conservative batch planning with audited physical storage lifetimes."""
from dataclasses import replace

from .state import CostProfile
from .storage import logical_source_problem, logical_cost_profile, plan_scene_storage
from .optimizers.results import StorageSearchResult
from .optimizers.session import solve_storage


def solve_storage_batch(problem, blocks, preparations, *, resident_blocks=(),
                        memory_limit, host_memory_limit=None, profile=None,
                        reserved_host_bytes=0, reserved_device_bytes=0,
                        template=None, source_map=None, allow_research=True,
                        **search_options):
    """Plan within headroom left by all raw storage spanning this batch.

    Reserve the union of incoming live blocks and every block this batch needs
    throughout the search. Zero-persistent-size shadow blocks are used ONLY for
    metadata search; staging and transfer costs remain explicit. Returned plans
    contain the real blocks. Join all batches before execution to audit exact
    first/last demand and remove duplicate read costs. This conservative search
    may fail even if a finer interleaving would fit; UNSOLVED is not a proof.
    """
    from .paths import rebind_scene_path
    blocks, preparations = dict(blocks), dict(preparations)
    profile = profile or CostProfile()
    if set(preparations) != set(problem.sources):
        raise ValueError("batch preparations must cover exactly its logical sources")
    required = set(resident_blocks)
    for prep in preparations.values():
        required.update(prep.blocks)
    if required - blocks.keys():
        raise ValueError("unknown resident or required raw block")
    raw_host = sum(blocks[k].host_bytes for k in required)
    raw_device = sum(blocks[k].device_bytes for k in required)
    shadow = {k:replace(b,host_bytes=0,device_bytes=0) for k,b in blocks.items()}
    host_reserve = reserved_host_bytes + raw_host
    device_reserve = reserved_device_bytes + raw_device
    search = None
    if template is not None:
        if source_map is None:
            raise ValueError("path reuse requires an explicit source map")
        headroom = memory_limit - (device_reserve if problem.backend=='cupy' else host_reserve)
        host_headroom = None if host_memory_limit is None else host_memory_limit-host_reserve
        if headroom > 0 and (host_headroom is None or host_headroom > 0):
            try:
                state = rebind_scene_path(template,logical_source_problem(problem),source_map,
                    profile=logical_cost_profile(profile),memory_limit=headroom,
                    host_memory_limit=host_headroom)
                candidate = plan_scene_storage(state,shadow,preparations,profile=profile,
                    memory_limit=memory_limit,host_memory_limit=host_memory_limit,
                    reserved_host_bytes=host_reserve,reserved_device_bytes=device_reserve)
                if candidate.feasible:
                    search = StorageSearchResult('FEASIBLE',candidate,0,False,0,1,0.,0,
                                                 {'reused_operation_order':True})
            except MemoryError:
                pass  # Search a different order under the same raw reservation.
    if search is None and template is not None and not allow_research:
        return StorageSearchResult('UNSOLVED',None,0,False,0,0,None,1,
            dict(reused_operation_order=False,research_disabled=True,
                 reason='fixed path does not fit even without retained logical tensors',
                 raw_reservation_host_bytes=raw_host,raw_reservation_device_bytes=raw_device))
    if search is None:
        search = solve_storage(problem,shadow,preparations,
            memory_limit=memory_limit,host_memory_limit=host_memory_limit,profile=profile,
            reserved_host_bytes=host_reserve,reserved_device_bytes=device_reserve,
            **search_options)
    diagnostic = dict(search.diagnostic, raw_reservation_host_bytes=raw_host,
                      raw_reservation_device_bytes=raw_device)
    plan = None
    if search.plan is not None:
        plan = plan_scene_storage(search.plan.state,blocks,preparations,profile=profile,
            memory_limit=memory_limit,host_memory_limit=host_memory_limit,
            reserved_host_bytes=reserved_host_bytes,reserved_device_bytes=reserved_device_bytes)
        if not plan.feasible:
            raise RuntimeError("real-block audit exceeds conservative raw reservation")
    # The shadow reservation restricts the domain, and joining batches changes
    # raw-read accounting. A local proof is not a certificate for the real task.
    diagnostic.update(proof_scope="conservative_batch", local_search_status=search.status)
    return replace(search, plan=plan, diagnostic=diagnostic,
                   status="FEASIBLE" if plan is not None else "UNSOLVED",
                   lower_bound_seconds=None)
