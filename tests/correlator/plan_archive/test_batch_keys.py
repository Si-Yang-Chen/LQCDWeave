"""Archive and selected-path batch keys retain identical term declarations."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator import api as c
from lqcdweave.contraction.correlator import (
    calc_diagram_eval, save_calculation_plan, load_calculation_plan,
)
from lqcdweave.contraction.planning.path_planning import _batch_shape_signature
from lqcdweave.storage.contraction_plans import api
from correlator.plan_archive import test_workflow_import as workflow_fixtures


def test_archive_and_planner_keys_keep_the_complete_kernel():
    kernel = c.MesonWholeKernel('loop', ('M',), ('P', 'E', 'G'),
                               c.AxisPattern(((0,), (0,), (0,)), ()), 'unit-spin')
    term = c.ContractionTerm(('a',), c.AxisPattern(((0,),), (0,)),
                             np.complex64(.7 + .2j), kernel=kernel)
    problem = SimpleNamespace(backend='numpy', specs={'a': c.TensorSpec((2,), 'complex64')},
                              terms=(term,), sources={'a': c.SourceSpec('a', 16)})
    inputs = SimpleNamespace(problem=problem, batch_topology=('test',),
                             coordinate_batch=SimpleNamespace(actual_size=2))
    key = api._batch_shape_key(inputs)
    assert key == _batch_shape_signature(inputs)
    assert key[4][0][3] is kernel
    assert api._batch_key_without_kernels(key) is None
    problem.terms = (replace(term, kernel=None),)
    ordinary = api._batch_shape_key(inputs)
    assert ordinary != key
    assert api._batch_key_without_kernels(ordinary)[4] == (ordinary[4][0][:3],)


@pytest.fixture
def batched_archive(tmp_path):
    fixture = workflow_fixtures.WorkflowPlanImportTests()
    diagram, propagator, _, vertices = fixture._fixture()
    result = calc_diagram_eval(diagram, vertex_map=vertices, propagator_map={'q': propagator},
        configuration='cfg', usedNe=1, time_axes={0: range(3), 1: 0},
        batch={'axis': 0, 'size': 2}, memory_limit=1 << 20,
        optimize='none', execute=False, profile='default')
    path = tmp_path / 'batched.json'
    save_calculation_plan(path, result)
    document = json.loads(path.read_text())
    keys = api.decode_plan_value(document['payload']['batch_shape_keys'])
    assert len(keys) == 2 and keys[0][1] == 2 and keys[1][1] == 1
    assert all(row[3] is None for key in keys for row in key[4])
    return result, path, document, keys


def test_previous_three_field_keys_replay_fixed_batches(batched_archive):
    result, path, document, keys = batched_archive
    older = tuple(api._batch_key_without_kernels(key) for key in keys)
    document['payload']['batch_shape_keys'] = api.encode_plan_value(older)
    for row, key in zip(document['payload']['plans'], older, strict=True):
        row['batch_shape_key'] = api.encode_plan_value(key)
    path.write_text(json.dumps(document))
    with patch('lqcdweave.contraction.correlator.find_contraction_path',
               side_effect=AssertionError('archive replay must not search')):
        restored = load_calculation_plan(path, expected=result)
    assert restored.path.batch_shape_keys == result.path.batch_shape_keys
    assert tuple(p.plan.state.actions for p in restored.path.batch_paths) == tuple(
        p.plan.state.actions for p in result.path.batch_paths)
    np.testing.assert_allclose(restored.run().value, result.run().value, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('change', ('kernel', 'coefficient', 'shape', 'order'))
def test_archive_keeps_strict_batch_structure_checks(batched_archive, change):
    result, path, document, keys = batched_archive
    if change == 'order':
        changed = tuple(reversed(keys))
    else:
        first = list(keys[0])
        if change == 'shape':
            specs = list(first[3])
            name, shape, dtype = specs[0]
            specs[0] = (name, (*shape, 7), dtype)
            first[3] = tuple(specs)
        else:
            terms = list(first[4])
            term = list(terms[0])
            term[3 if change == 'kernel' else 2] = (
                c.OutputNode('constant', 3) if change == 'kernel' else term[2] + 1)
            terms[0] = tuple(term)
            first[4] = tuple(terms)
        changed = (tuple(first), *keys[1:])
    document['payload']['batch_shape_keys'] = api.encode_plan_value(changed)
    path.write_text(json.dumps(document))
    with pytest.raises(api.PlanArchiveError, match='coordinate batch structure changed'):
        load_calculation_plan(path, expected=result)


def test_legacy_defaults_do_not_skip_each_saved_record_check(batched_archive):
    result, path, document, keys = batched_archive
    older = tuple(api._batch_key_without_kernels(key) for key in keys)
    document['payload']['batch_shape_keys'] = api.encode_plan_value(older)
    for row, key in zip(document['payload']['plans'], older, strict=True):
        row['batch_shape_key'] = api.encode_plan_value(key)
    document['payload']['plans'][0]['batch_shape_key'] = api.encode_plan_value(older[1])
    path.write_text(json.dumps(document))
    with pytest.raises(api.PlanArchiveError, match='coordinate batch order or shape changed'):
        load_calculation_plan(path, expected=result)
