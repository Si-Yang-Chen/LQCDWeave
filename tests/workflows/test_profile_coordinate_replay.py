"""Time-varying spectral definitions follow each saved input coordinate."""
from dataclasses import replace

import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import ContractionTask, calculate, save_calculation_plan, load_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.expansion.api import input_readers
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.operators.single_hadron.api import Operator
from lqcdweave.operators.single_hadron.profile import SpectralProfile
from workflows.test_contraction_task_plan import declarations, no_planning


def profiled(*, dtype, batch=2, backend='numpy'):
    request, options, original_elemental = declarations(batch=batch, backend=backend)
    bind = options['vertex_map']
    elemental = ArrayElemental({key:value.astype(dtype) for key,value in original_elemental.sectors.items()},
        original_elemental.components, phase_convention='anchor', cfg='cfg', data_identity='profile-coordinates')
    rng = np.random.default_rng(261022)
    left = rng.normal(size=(2,2,3)) + 1j*rng.normal(size=(2,2,3))
    right = rng.normal(size=(2,2,3)) + 1j*rng.normal(size=(2,2,3))
    profile = SpectralProfile(left, right, (.3+.2j, -.1j), cfg='cfg')
    def vertices(row):
        original = bind(row)
        operator = Operator.from_terms(original.operator.name,
            tuple(replace(term, profile=profile) for term in original.operator.terms))
        return Meson(elemental, operator, row.dagger, phase_convention='anchor')
    options['vertex_map'] = vertices
    fixed = dict(options)
    fixed.pop('time_pairs'); fixed.pop('batch'); fixed['backend'] = 'numpy'
    # Each reference binds its own real coordinate before selecting rows.
    expected = np.stack([calculate(request, time_axes=point, **fixed).value
                         for point in options['time_pairs']], axis=-1)
    return request, options, elemental, expected


@pytest.mark.parametrize('dtype', ('complex64', 'complex128'))
@pytest.mark.parametrize('batch', (1, 2, 'full'))
def test_profile_rows_follow_complete_task_coordinates_and_archive(dtype, batch, tmp_path):
    request, options, elemental, expected = profiled(dtype=dtype, batch=batch)
    tolerance = 5e-5 if dtype == 'complex64' else 6e-12
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path/'profile.json.gz'
        task.save_plan(archive)
        assert all(not prep.recipe.coordinate_constants for prep in task._selected.plan.preparations.values())
        with no_planning(), ContractionTask.load_plan(archive, configuration='cfg', vertex_map=elemental,
                propagator_map=options['propagator_map']) as loaded:
            np.testing.assert_allclose(loaded.run().value, expected, rtol=tolerance, atol=tolerance)
            for value in elemental.sectors.values():
                value *= .5
            np.testing.assert_allclose(loaded.run().value, expected*.25, rtol=tolerance, atol=tolerance)


def test_profile_rows_follow_scalar_complete_cfg_binding(tmp_path):
    request, options, elemental, expected = profiled(dtype='complex128')
    points = options.pop('time_pairs'); options.pop('batch')
    selected = calculate(request, time_axes=points[0], execute=False, **options)
    coordinates, _ = describe_coordinates(time_pairs=points)
    complete = complete_calculation(selected, coordinates=coordinates, readers=input_readers(selected.inputs))
    archive = tmp_path/'scalar-profile.json.gz'
    save_calculation_plan(archive, complete)
    with no_planning():
        loaded = load_calculation_plan(archive, configuration='cfg', vertex_map=elemental,
                                      propagator_map=options['propagator_map'])
        np.testing.assert_allclose(loaded.run().value, expected, rtol=6e-12, atol=6e-12)


def test_profile_rows_gpu_task_replay(tmp_path):
    pytest.importorskip('cupy')
    request, options, elemental, expected = profiled(dtype='complex128', backend='cupy')
    with ContractionTask(request, **options) as task:
        task.plan()
        archive = tmp_path/'gpu-profile.json.gz'
        task.save_plan(archive)
        with no_planning(), ContractionTask.load_plan(archive, configuration='cfg', vertex_map=elemental,
                propagator_map=options['propagator_map']) as loaded:
            np.testing.assert_allclose(loaded.run().value, expected, rtol=6e-12, atol=6e-12)


def test_fixed_time_profile_does_not_require_unrequested_time_coverage():
    request, options, elemental = declarations()
    original = options['vertex_map']
    profile = SpectralProfile(np.ones((1,3)), cfg='cfg')
    def vertices(row):
        bound = original(row)
        operator = Operator.from_terms(bound.operator.name,
            tuple(replace(term, profile=profile) for term in bound.operator.terms))
        return Meson(elemental, operator, row.dagger, phase_convention='anchor')
    options.pop('time_pairs'); options.pop('batch')
    options['vertex_map'] = vertices
    with ContractionTask(request, time_axes={0:0,1:0}, **options) as task:
        task.plan()
        assert np.isfinite(task.run().value).all()
