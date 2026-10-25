"""Declared diagonal support removes work without a numerical zero threshold."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/campaigns'))
import light_charm_spd as campaign
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contraction.expansion.api import describe_meson_matrix
from lqcdweave.operators.correlator.api import Tag, Hadron, Flavor
from lqcdweave.operators.single_hadron.api import meson55_preset, mom_dict_to_list, momDict_meson55
from lqcdweave.contraction.correlator import calculate


def test_explicit_legacy_declaration_keeps_dataset_and_basis_checks():
    from types import SimpleNamespace
    from copy import deepcopy
    from test_direct_projected_elemental import DirectProjectedElementalTests
    from lqcdweave.adapters.meson_dataset import DatasetElemental
    from lqcdweave.contracts.dataset import DatasetSpec
    table, document, _ = DirectProjectedElementalTests().fixture()
    stored = SimpleNamespace(spec=DatasetSpec(document), data_id='published-fixture')
    original = stored.spec.document
    component = table[0]['component']
    before = DatasetElemental(stored, catalog=table)
    declared = campaign.elemental_reader(stored)
    assert before.describe_block(component, 0, 0, 'v2v').structure == 'general'
    assert declared.describe_block(component, 0, 0, 'v2v').structure == 'diagonal'
    assert before.describe_block(component, 0, 0, 'v2v').selection_identity != declared.describe_block(
        component, 0, 0, 'v2v').selection_identity
    assert stored.spec.document == original
    mesons = RecipeVertex(declared)
    row = mesons.symbol('op_00_gamma5_A1', tag=Tag(0, 'sink'))[0]
    assert mesons(row).with_mode_indices(((0,), (1,))).is_structurally_zero()
    different = deepcopy(document)
    different['bases']['source_mode'] = dict(different['bases']['source_mode'])
    different['bases']['source_mode']['data_id'] = 'different-basis'
    different['parents'].append('different-basis')
    unsafe = campaign.elemental_reader(SimpleNamespace(spec=DatasetSpec(different), data_id='other'))
    assert unsafe.describe_block(component, 0, 0, 'v2v').structure == 'general'
    assert not RecipeVertex(unsafe)(row).with_mode_indices(((0,), (1,))).is_structurally_zero()
    for declarations in ({('v2v', component, 99): 'diagonal'},
                         {('v2v', component, 0): 'point_diagonal'}):
        with pytest.raises(ValueError):
            DatasetElemental(stored, catalog=table, elemental_structures=declarations)
    conflicting = deepcopy(document)
    conflicting['conventions']['elemental_structures'] = [dict(component=0, momentum=0, structure='diagonal')]
    with pytest.raises(ValueError, match='conflicts'):
        DatasetElemental(SimpleNamespace(spec=DatasetSpec(conflicting), data_id='declared'), catalog=table,
            elemental_structures={('v2v', component, 0): 'general'})


def reader(diagonal):
    components = tuple(r['component'] for r in meson55_preset(version='v2'))
    data = np.broadcast_to(np.eye(3, dtype=complex), (55, 33, 2, 3, 3))
    return ArrayElemental({'v2v': data}, components, cfg='toy',
        momentum_list=mom_dict_to_list(momDict_meson55), phase_convention='midpoint',
        elemental_structures={('v2v', components[0], 0): 'diagonal'} if diagonal else {})


def test_zero_proof_uses_headers_and_keeps_unknown_overlap_and_dagger():
    e = reader(True)
    mesons = RecipeVertex(e)
    row = mesons.symbol('op_00_gamma5_A1', tag=Tag(0, 'sink'))[0]
    source = Hadron(row, Flavor('uc')).conjugate().irrep_row
    with patch.object(e, 'block', side_effect=AssertionError('numeric IO')):
        assert mesons(row).with_mode_indices(((0,), (1, 2))).is_structurally_zero()
        assert mesons(source).with_mode_indices(((0,), (1, 2))).is_structurally_zero()
        assert not mesons(row).with_mode_indices(((0, 1), (1, 2))).is_structurally_zero()
        assert not mesons(row).with_mode_indices(((0, 1), (1, 0))).is_structurally_zero()
        moving = mesons.symbol('op_00_gamma5_A1', (0, 0, 1), irrep='A2', tag=Tag(0, 'sink'))[0]
        assert not mesons(moving).with_mode_indices(((0,), (1, 2))).is_structurally_zero()
        with pytest.raises((ValueError, IndexError)):
            mesons(row).with_mode_indices(((0,), (3,))).is_structurally_zero()
    assert not RecipeVertex(reader(False))(row).with_mode_indices(((0,), (1,))).is_structurally_zero()


def test_native_zero_elimination_and_pruned_campaign_match_full_dense_reference():
    diagonal, general = reader(True), reader(False)
    channel = dict(next(c for c in campaign.channels() if c['name'] == 'p2_0_A1_Pm'))
    channel['operators'] = [o for o in channel['operators'] if o['name'] == 'op_00_gamma5_A1']
    options = dict(light_bins=((0,), (1,)), charm_bins=((0,), (1, 2)))
    full, fields, tags = campaign.build_task([channel], **options)
    small, reduced, small_tags = campaign.build_task([channel], elemental=diagonal, **options)
    assert {p[0]: len(f['operators']) for p, f in reduced.items()} == {
        'I0_light_Cp': 2, 'charmonium_Cp': 2, 'I1_light_Cp': 2, 'D': 2}
    assert len(fields) == 7
    rng = np.random.default_rng(475)
    def prop(ne):
        shape = (2, 2, 4, 4, ne, ne)
        return ArrayPropagator({'vsv': rng.normal(size=shape)+1j*rng.normal(size=shape)},
            cfg='toy', time_convention='absolute')
    light, charm = prop(2), prop(3)
    props = {f'S^{f}': p for f, p in (('u', light), ('d', light), ('q', light), ('c', charm))}
    props.update({rf'S^{f}_\mathrm{{local}}': p for f, p in (('u', light), ('d', light), ('q', light), ('c', charm))})
    with patch.object(diagonal, 'block', side_effect=AssertionError('catalog must not read')):
        catalog = describe_meson_matrix(full, bindings=DiagramBindings(
            vertex_map=campaign.vertex_binding(diagonal, tags), propagator_map=props,
            configuration='toy'), representative_times={'sink': 1, 'source': 0})
    for component in catalog.catalog.components:
        a, b = component.factor.mode_indices
        assert set(a).intersection(b)
    def run(task, e, t):
        return calculate(task, vertex_map=campaign.vertex_binding(e, t), propagator_map=props,
            configuration='toy', time_axes={'sink': 1, 'source': 0}, optimize='none',
            backend='numpy', memory_limit=1 << 29, host_memory_limit=1 << 29).value
    expected, actual, compressed = run(full, general, tags), run(full, diagonal, tags), run(small, diagonal, small_tags)
    def get(values, path):
        for key in path:
            values = values[key]
        return values['two_point']
    for path in fields:
        np.testing.assert_allclose(get(actual, path), get(expected, path), atol=1e-10, rtol=1e-12)
    for path, field in reduced.items():
        keys = [(r['flavor'], r['radial']) for r in fields[path]['operators']]
        indices = [keys.index((r['flavor'], r['radial'])) for r in field['operators']]
        np.testing.assert_allclose(get(compressed, path), get(expected, path)[np.ix_(indices, indices)],
            atol=1e-10, rtol=1e-12)
