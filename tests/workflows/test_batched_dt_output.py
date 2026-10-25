"""Packed output retains time slots that occur after the first batch lane."""
from dataclasses import replace

import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate, save_calculation_plan, load_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation, CfgSources
from lqcdweave.contraction.expansion.api import (
    input_readers, describe_input_reads, describe_input_preparations,
    describe_coordinate_reads, unknown_reader_aliases,
)
from lqcdweave.contraction.planning.api import plan_coordinate_batches
from lqcdweave.contracts.correlator.api import OutputNode, describe_coordinates
from lqcdweave.storage.correlator_results import load_correlator_output
from test_structured_correlator import fixture, task


def test_packed_output_includes_later_batch_time_slots(tmp_path):
    bind, elemental, propagator = fixture()
    e = ArrayElemental({'v2v': np.pad(elemental.sectors['v2v'],
        ((0,0),(0,0),(0,1),(0,0),(0,0)), mode='wrap')}, elemental.components,
        phase_convention='anchor', cfg='cfg', data_identity='three-time-elemental')
    p = ArrayPropagator({'vsv': np.pad(propagator.sectors['vsv'],
        ((0,1),(0,1),(0,0),(0,0),(0,0),(0,0)), mode='wrap')},
        time_convention='absolute', cfg='cfg', data_identity='three-time-propagator')
    def vertices(row):
        original = bind(row)
        return Meson(e, original.operator, row.dagger, phase_convention='anchor')
    selected = calculate({'two_point': task(flavor='I0')}, vertex_map=vertices,
        propagator_map={'S^q':p, r'S^q_\mathrm{local}':p}, configuration='cfg', usedNe=3,
        time_axes={0:1,1:0}, optimize='none', execute=False,
        memory_limit=1<<28, host_memory_limit=1<<28)
    # A constant one-point field isolates the output time-slot contract.
    root = selected.expanded.output_recipe.root
    loop = np.array([1+2j, 3+4j])
    recipe = replace(selected.expanded.output_recipe, root=replace(root,
        keys=(*root.keys,'vacuum_sink'), children=(*root.children,
        OutputNode('constant_array', loop, shape=loop.shape))))
    expanded = replace(selected.expanded, output_recipe=recipe)
    selected = replace(selected, expanded=expanded, path=replace(selected.path, expanded=expanded))
    coordinates, grouping = describe_coordinates(
        time_pairs=[{0:1,1:0}, {0:2,1:0}, {0:0,1:2}], batch='full')
    readers = input_readers(selected.inputs)
    expected = complete_calculation(selected, coordinates=coordinates, readers=readers).run().value
    preparations, specs = describe_input_preparations(selected.inputs)
    batches = plan_coordinate_batches(selected.plan, recipe, coordinates,
        describe_input_reads(selected.inputs), preparations, specs, batch=grouping,
        read_metadata=describe_coordinate_reads(selected.inputs, coordinates.points),
        reader_aliases=unknown_reader_aliases(selected.inputs))
    complete = complete_calculation(selected, coordinates=coordinates,
        coordinate_batches=batches, output_mode='dataset', physical_lt=3,
        output_layout='dt_packed', output=tmp_path/'result')
    assert complete.plan.one_point_slots == 3
    with pytest.raises(ValueError, match='omits requested time slots'):
        replace(complete.plan, one_point_slots=2)
    archive = tmp_path/'plan.json.gz'
    save_calculation_plan(archive, complete)
    load_calculation_plan(archive, configuration='cfg',
        readers=CfgSources(readers, (0,1,2), 3), output=tmp_path/'result').run()
    actual = load_correlator_output(tmp_path/'result', ('two_point',), dt=True, configuration='cfg')
    np.testing.assert_allclose(actual[...,0],
        (expected['two_point'][...,0]+expected['two_point'][...,2])/2, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual[...,1], expected['two_point'][...,1], rtol=3e-12, atol=3e-12)
    np.testing.assert_array_equal(load_correlator_output(tmp_path/'result',
        ('vacuum_sink',), configuration='cfg'), np.repeat(loop[:,None], 3, axis=1))
