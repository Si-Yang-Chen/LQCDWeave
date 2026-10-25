"""Fixed elements account shared raw owners through remaining source uses."""
from collections import Counter
from dataclasses import replace

import pytest

from lqcdweave.contracts.correlator.api import SourcePreparation
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.storage_schedule import _Dag, _Frontier
from .test_storage_schedule import _fixed_path, _physical, _problem


@pytest.mark.parametrize('shared', (False, True))
@pytest.mark.parametrize('backend', ('numpy', 'cupy'))
def test_raw_use_counts_match_source_scan_through_every_order_and_undo(shared, backend):
    problem, _, blocks, preparations = _problem(shared=shared)
    problem = replace(problem, backend=backend)
    # Multiple logical source preparations share one complete physical owner.
    blocks = {'all': replace(blocks['A'], dataset='shared',
        device_bytes=128 if backend == 'cupy' else 0)}
    preparations = {key: SourcePreparation(('all',)) for key in preparations}
    steps = (((0, ('A', 'B')), (0, None), (1, None)) if shared else
             ((0, ('A', 'B')), (1, ('D', 'E')), (0, None), (1, None)))
    state, spans, workspaces = _fixed_path(problem, CostProfile(), steps)
    dag = _Dag(state, spans, workspaces, _physical(problem, blocks, preparations))
    frontier = _Frontier(dag)

    def snapshot():
        return (frontier.remaining.copy(), frontier.raw_remaining.copy(),
                frontier.live.copy(), frontier.raw.copy(), frontier.prepared.copy(),
                frontier.host, frontier.device, frontier.peak_host, frontier.peak_device)

    def visit():
        expected = Counter({raw: sum(frontier.remaining[key] for key in keys)
                            for raw, keys in dag.raw_sources.items()})
        assert frontier.raw_remaining == expected
        for raw in frontier.opened - frontier.raw:
            assert all(frontier.remaining[key] == 0 and
                (key not in frontier.live or dag.sizes.get(key, 0) > 0)
                for key in dag.raw_sources[raw])
        for index in sorted(frontier.ready):
            before = snapshot()
            change = frontier.advance(index, reversible=True)
            visit()
            frontier.undo(change)
            assert snapshot() == before

    visit()


def test_advance_and_undo_do_not_rescan_shared_owner_sources():
    problem, _, blocks, preparations = _problem(shared=True)
    state, spans, workspaces = _fixed_path(problem, CostProfile(),
        ((0, ('A', 'B')), (0, None), (1, None)))
    preparations = {key: SourcePreparation(('A',)) for key in preparations}
    dag = _Dag(state, spans, workspaces,
        _physical(problem, {'A': blocks['A']}, preparations))

    class NoSourceScan(dict):
        def __getitem__(self, key):
            raise AssertionError('shared sources were rescanned during scheduling')

    dag.raw_sources = NoSourceScan(dag.raw_sources)
    frontier = _Frontier(dag)
    changes = []
    while frontier.ready:
        changes.append(frontier.advance(min(frontier.ready), reversible=True))
    for change in reversed(changes):
        frontier.undo(change)
    assert frontier.raw_remaining == dag.raw_references
