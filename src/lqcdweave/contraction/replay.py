"""Convenience preparation followed by strict storage-plan execution."""
from __future__ import annotations

from collections import deque

import numpy as np

from lqcdweave.contracts.correlator.api import SceneStoragePlan, SourceLoader
from lqcdweave.contraction.planning.api import (
    join_storage_batches, prepare_coordinate_batch_workspace,
)
from lqcdweave.contraction.execution.api import (
    replay_scene_storage as _replay_scene_storage,
    replay_storage_batches as _replay_storage_batches,
)


def replay_scene_storage(plan: SceneStoragePlan, loader: SourceLoader, *, backend=np,
                         measure=False, source_descriptions=None, io_overlap=True):
    """Complete the known resource envelope, then replay the selected path."""
    prepared, _ = prepare_coordinate_batch_workspace(plan)
    return _replay_scene_storage(prepared, loader, backend=backend, measure=measure,
                                source_descriptions=source_descriptions, io_overlap=io_overlap)


def replay_storage_batches(plans, loader: SourceLoader, consume, *, backend=np, measure=False,
                           prejoined=False, streaming=False, batch_context=None,
                           source_descriptions=None, io_overlap=True, lookahead=2):
    """Join/prepare convenience inputs before handing frozen plans to execution.

    Streaming requires prejoined inputs and prepares each yielded descriptor
    before X admits its bounded window. Batch activation receives the original
    joined descriptor, preserving caller identity checks and loader ownership.
    """
    if streaming and not prejoined:
        raise ValueError('streaming requires explicitly joined per-batch lifetimes')
    if type(lookahead) is not int or lookahead < 1:
        raise ValueError('lookahead must be a positive integer')
    if type(io_overlap) is not bool:
        raise TypeError('io_overlap must be bool')
    selected = iter(plans) if streaming else tuple(plans) if prejoined else join_storage_batches(plans)
    contexts = deque()

    def prepared_plans():
        for original in selected:
            prepared, _ = prepare_coordinate_batch_workspace(original)
            if batch_context is not None:
                contexts.append((prepared, original))
            yield prepared

    def activate(prepared):
        expected, original = contexts.popleft()
        if expected is not prepared:
            raise ValueError("prepared batch context order differs from execution")
        return batch_context(original)

    return _replay_storage_batches(
        prepared_plans(), loader, consume, backend=backend, measure=measure,
        prejoined=True, streaming=streaming,
        batch_context=None if batch_context is None else activate,
        source_descriptions=source_descriptions, io_overlap=io_overlap, lookahead=lookahead,
    )


__all__ = ["replay_scene_storage", "replay_storage_batches"]
