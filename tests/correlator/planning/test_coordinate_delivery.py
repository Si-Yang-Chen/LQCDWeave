"""Auto delivery compares actual input sharing on one selected DAG."""
from types import SimpleNamespace
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    ContractionTerm, describe_coordinates,
)
from lqcdweave.contraction.expansion.api import describe_coordinate_reads
from lqcdweave.contraction.planning import coordinate_schedule
from lqcdweave.contraction.planning.coordinate_schedule import _coordinate_work_candidates
from correlator.planning.test_coordinate_path import PROFILE, problem, selected
from correlator.execution.test_coordinate_replay import selected as replay_fixture


def test_grid_and_flat_pairs_keep_source_shared_inside_five_lane_groups():
    graph = problem({'A': (3, 4), 'B': (4, 3), 'C': (3, 3)},
        (ContractionTerm(('A', 'B', 'C'), 'ij,jk,kl->il'),))
    template = SimpleNamespace(state=selected(graph), cost_profile=PROFILE)
    pairs = tuple({'source': source, 'sink': sink}
                  for source in range(5) for sink in range(5))
    identities = tuple({'A': f'a:{point["source"]}',
                        'B': f'b:{point["source"]}',
                        'C': f'c:{point["sink"]}'} for point in pairs)
    binding = SimpleNamespace(source_names=identities)
    coordinates = (
        describe_coordinates(time_axes={'source': range(5), 'sink': range(5)}),
        describe_coordinates(time_pairs=pairs),
    )
    for points, grouping in coordinates:
        candidates = _coordinate_work_candidates(template, binding, points, grouping, PROFILE)
        selected_candidate = candidates[0]
        assert selected_candidate['mode'] == 'batched'
        assert selected_candidate['batch_width'] == 5
        assert selected_candidate['batch_sizes'] == (5, 5, 5, 5, 5)
        scalar = next(item for item in candidates if item['mode'] == 'scalar')
        assert selected_candidate['variable_seconds'] == scalar['variable_seconds']
        if points.mode == 'pairs':
            full = next(item for item in candidates if item['batch_width'] == 25)
            assert full['variable_seconds'] > scalar['variable_seconds']


def test_all_static_coordinates_select_scalar_replay():
    graph = problem({'A': (3,), 'B': (3,)},
        (ContractionTerm(('A', 'B'), 'i,i->'),))
    template = SimpleNamespace(state=selected(graph), cost_profile=PROFILE)
    points, grouping = describe_coordinates(time_pairs=({'source': 0, 'sink': 0},) * 5)
    binding = SimpleNamespace(source_names=({'A': 'same:a', 'B': 'same:b'},) * 5)
    candidates = _coordinate_work_candidates(template, binding, points, grouping, PROFILE)
    assert candidates[0]['mode'] == 'scalar'
    assert candidates[0]['batch_width'] == 1


def test_auto_batched_delivery_binds_physical_inputs_only_once():
    calculation, _ = replay_fixture()
    inputs = calculation.inputs
    coordinates, grouping = describe_coordinates(
        time_axes={'source': 3, 'sink': range(7)}, batch='auto')
    metadata = describe_coordinate_reads(inputs, coordinates.points)
    actual_bind = coordinate_schedule.bind_coordinate_inputs
    with patch.object(coordinate_schedule, 'bind_coordinate_inputs', wraps=actual_bind) as binding, \
            patch.object(coordinate_schedule, '_coordinate_work_candidates',
                         return_value=({'mode': 'batched', 'batch_width': 7},)):
        delivery = coordinate_schedule.plan_coordinate_delivery(
            calculation.plan, calculation.expanded.output_recipe,
            coordinates, inputs.read_requests, inputs.loader.preparations,
            inputs.loader.specs, batch=grouping, read_metadata=metadata,
            memory_limit=calculation.plan.memory_limit)
    assert delivery.mode == 'batched'
    assert delivery.replay.statistics['batch_sizes'] == (7,)
    assert binding.call_count == 1
