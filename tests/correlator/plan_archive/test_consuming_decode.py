"""Private file decoding releases JSON nodes without changing restored values."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.storage.contraction_plans import api, compact
from correlator.execution import test_cfg_runtime as fixtures


def test_consuming_decode_preserves_values_and_record_identity():
    c = api.contract_api
    array = c.ArrayConstant.from_array(np.asarray([1 + 2j, -3j], dtype='complex64'))
    distinct = replace(array)
    pattern = c.AxisPattern(((0, 1), (1, 2)), (0, 2))
    value = (array, distinct, array, pattern, pattern,
             {'cfg': ('a', 3, -2, np.float64(-0.0)), 'values': [True, None]})
    encoded = compact.encode(value, preserve_identity=True)
    untouched = deepcopy(encoded)
    ordinary = api.decode_plan_value(encoded)
    assert encoded == untouched
    restored = compact.decode(encoded['value'], _consume=True)
    assert encoded['value']['root'] == []
    assert api._json_bytes(compact.encode(restored, preserve_identity=True)) == api._json_bytes(untouched)
    for decoded in (ordinary, restored):
        assert decoded[:5] == value[:5]
        assert decoded[0] is decoded[2]
        assert decoded[0] is not decoded[1]
        assert decoded[3] is decoded[4]
        assert decoded[0].array().dtype == np.dtype('complex64')
        np.testing.assert_array_equal(decoded[0].array(), array.array())
        assert decoded[-1]['cfg'][3].tobytes() == np.float64(-0.0).tobytes()


@pytest.mark.parametrize('consume', (False, True))
def test_consuming_decoder_keeps_invalid_reference_and_depth_checks(consume):
    payload = {'schemas': [], 'strings': [], 'root': [-10, 0]}
    with pytest.raises(api.PlanArchiveError, match='forward'):
        compact.decode(payload, _consume=consume)
    root = 1
    for _ in range(129):
        root = [-2, root]
    with pytest.raises(api.PlanArchiveError, match='nesting'):
        compact.decode({'schemas': [], 'strings': [], 'root': root}, _consume=consume)


def test_file_load_consumes_owned_tree_and_replays_without_planning():
    fixture = fixtures.CfgReplayTests()
    original, expanded, template, elemental, propagator = fixture.setup_plan()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'cfg.json.gz'
        api.save_cfg_plan(path, original)
        document = api._read_document(path)
        root = document['payload']['plan']['value']['root']
        assert root
        with patch('lqcdweave.storage.contraction_plans.cfg_archive._read_document',
                   return_value=document):
            restored = api.load_cfg_plan(path)
        assert root == []
        assert restored == original
        expected = fixture.run_plan(capacity=0)[1]
        with patch.object(fixture, 'setup_plan', return_value=(
                restored, expanded, template, elemental, propagator)):
            actual = fixture.run_plan(capacity=0)[1]
        for key, values in expected.items():
            for reference, value in zip(values, actual[key], strict=True):
                np.testing.assert_array_equal(value, reference)
