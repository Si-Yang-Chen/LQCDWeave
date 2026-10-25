"""The unchanged release rule builds only the metadata it consumes."""
from collections import defaultdict
from unittest.mock import patch

from lqcdweave.contraction.planning.optimizers import storage_schedule as s
from .test_storage_peak_search import _chain_fixture


def test_resource_emission_does_not_build_release_target_metadata():
    state, spans, workspace, physical, _ = _chain_fixture()
    with patch.object(s._Dag, '_prepare_release_ranking', side_effect=AssertionError('replay masks')):
        dag = s._Dag(state, spans, workspace, physical)
        s._plan(dag, range(len(spans)), audited_computations=True, debug=True)


def test_lazy_target_masks_match_independent_dependency_walk_and_are_shared():
    state, spans, workspace, physical, _ = _chain_fixture()
    dag = s._Dag(state, spans, workspace, physical)
    assert not {'release_masks', 'raw_release_masks', 'computation_mask'} & vars(dag).keys()
    expected = defaultdict(int)
    for index, node in enumerate(dag.nodes):
        required, pending = set(), [index]
        while pending:
            current = pending.pop()
            if current not in required:
                required.add(current)
                pending.extend(dag.nodes[current].parents)
        mask = sum(1 << current for current in required)
        for key, _ in node.references:
            expected[key] |= mask
    prepare = s._Dag._prepare_release_ranking
    with patch.object(s._Dag, '_prepare_release_ranking', autospec=True, side_effect=prepare) as build:
        assert dag.release_masks == dict(expected)
        raw = {}
        for key, sources in dag.raw_sources.items():
            if not sources & dag.outputs:
                raw[key] = 0
                for source in sources:
                    raw[key] |= expected[source]
        assert dag.raw_release_masks == raw
        assert dag.computation_mask == sum(1 << index for index, node in enumerate(dag.nodes)
                                           if not node.actions[0].reused)
        assert build.call_count == 1


def test_default_priorities_reuse_target_metadata_across_undo():
    state, spans, workspace, physical, _ = _chain_fixture()
    prepare = s._Dag._prepare_release_ranking
    with patch.object(s._Dag, '_prepare_release_ranking', autospec=True, side_effect=prepare) as build:
        frontier = s._Frontier(s._Dag(state, spans, workspace, physical))
        assert tuple(frontier.choices(1, lambda: True)) == ((0, None),)
        assert build.call_count == 0
        first = frontier.advance(0, reversible=True)
        before = tuple(frontier.choices(4, lambda: True))
        assert any(goal is not None for _, goal in before)
        change = frontier.advance(before[0][0], reversible=True)
        frontier.undo(change)
        assert tuple(frontier.choices(4, lambda: True)) == before
        frontier.undo(first)
        assert build.call_count == 1
