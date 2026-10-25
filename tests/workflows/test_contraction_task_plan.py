"""One full task boundary for memory, Dataset and fresh-process replay."""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.correlator import ContractionTask, calculate
from lqcdweave.contraction.planning.api import audit_cfg_plan
from lqcdweave.storage.correlator_results import load_correlator_output
from test_structured_correlator import fixture, task as correlator


def declarations(*, batch=2, output=None, backend='numpy'):
    bind,elemental,propagator=fixture()
    options=dict(vertex_map=bind,propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator},
        configuration='cfg',usedNe=3,optimize='none',memory_limit=1<<28,host_memory_limit=1<<28,
        time_pairs=[{0:1,1:0},{0:0,1:1},{0:1,1:1}],batch=batch,backend=backend)
    if output is not None:
        options['output']=output
    return correlator(flavor='I0'),options,elemental


def no_planning():
    stack=ExitStack()
    for name in ('lqcdweave.contraction.correlator.calculate',
                 'lqcdweave.contraction.correlator.expand_correlator',
                 'lqcdweave.contraction.correlator.find_contraction_path',
                 'lqcdweave.contraction.planning.api.plan_coordinate_batches',
                 'lqcdweave.contraction.planning.api.plan_coordinate_delivery',
                 'lqcdweave.contraction.correlator_task_plan.plan_coordinate_delivery',
                 'lqcdweave.contraction.planning.api.build_cfg_plan',
                 'lqcdweave.contraction.planning.api.build_coordinate_cfg_plan',
                 'lqcdweave.contraction.planning.cfg_plan._bind_cfg_template'):
        stack.enter_context(patch(name,side_effect=AssertionError(name)))
    return stack


def test_scalar_coordinate_choice_closes_once_and_saves_fixed_delivery(tmp_path):
    from lqcdweave.contraction import correlator as frontend
    from lqcdweave.contraction import correlator_task_plan as workflow
    from lqcdweave.contraction.planning.api import CoordinateDeliverySelection
    request,options,elemental=declarations(batch='auto')
    expected=calculate(request,**dict(options,batch=1)).value
    selection=CoordinateDeliverySelection('scalar',None,
        dict(mode='scalar',selected_width=1,selection='coarse-work-and-complete-reuse'))
    with ContractionTask(request,**options) as task:
        with patch.object(workflow,'plan_coordinate_delivery',return_value=selection), \
                patch.object(frontend,'find_contraction_path',wraps=frontend.find_contraction_path) as search:
            task.plan()
        assert search.call_count==1
        plan=task._selected.plan
        assert len(plan.coordinates)==3
        assert plan.statistics['coordinate_batch_sizes']==(1,1,1)
        assert plan.statistics['coordinate_selection']==selection.statistics
        assert dict(plan.output_definition['coordinate_selection'])==selection.statistics
        audit_cfg_plan(plan)
        archive=tmp_path/'scalar-choice.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value,expected,rtol=3e-12,atol=3e-12)
            with ContractionTask.load_plan(archive,configuration='cfg',vertex_map=elemental,
                    propagator_map=options['propagator_map']) as restored:
                assert dict(restored._selected.plan.output_definition['coordinate_selection'])==selection.statistics
                np.testing.assert_allclose(restored.run().value,expected,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('batch',[1,2,'full','auto'])
def test_full_task_tail_and_archive_replay_without_replanning(batch,tmp_path):
    request,options,elemental=declarations(batch=batch)
    expected=calculate(request,**dict(options,batch=1)).value
    with ContractionTask(request,**options) as task:
        task.plan()
        plan=task._selected.plan
        assert len(plan.coordinates)==3
        assert sum(plan.statistics['coordinate_batch_sizes'])==3
        if batch==2:
            assert plan.statistics['coordinate_batch_sizes']==(2,1)
        audit_cfg_plan(plan)
        archive=tmp_path/'task.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value,expected,rtol=3e-12,atol=3e-12)
            with ContractionTask.load_plan(archive,configuration='cfg',vertex_map=elemental,
                                          propagator_map=options['propagator_map']) as loaded:
                first=loaded.run().value.copy()
                np.testing.assert_allclose(first,expected,rtol=3e-12,atol=3e-12)
                # Mutate actual input arrays without touching the saved plan.
                for array in elemental.sectors.values():
                    array *= .5
                second=loaded.run().value
                np.testing.assert_allclose(second,.25*first,rtol=3e-12,atol=3e-12)
            with ContractionTask.load_plan(archive,configuration=iter(('cfg',)),vertex_map=elemental,
                    propagator_map=options['propagator_map']) as streamed:
                values=list(streamed.run().value)
                assert len(values)==1 and values[0][0]=='cfg'
                np.testing.assert_allclose(values[0][1],.25*expected,rtol=3e-12,atol=3e-12)


def test_memory_and_dataset_use_same_complete_batch_plan(tmp_path):
    request,options,elemental=declarations(output=tmp_path/'dataset')
    expected=calculate(request,**dict(options,output=None,batch=1)).value
    with ContractionTask(request,**options) as task:
        task.plan()
        assert task._selected.plan.statistics['coordinate_batch_sizes']==(2,1)
        archive=tmp_path/'dataset-plan.json.gz'
        task.save_plan(archive)
        with no_planning():
            result=task.run()
            assert result.report['status']=='completed'
            with ContractionTask.load_plan(archive,configuration='cfg',vertex_map=elemental,
                    propagator_map=options['propagator_map'],output=tmp_path/'again') as loaded:
                loaded.run()
    for directory in ('dataset','again'):
        actual=load_correlator_output(tmp_path/directory,configuration='cfg')
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)


def test_recipe_vertex_binding_is_the_same_before_and_after_loading(tmp_path):
    from lqcdweave.contraction.binding.api import RecipeVertex
    from lqcdweave.adapters.elemental import ArrayElemental
    from lqcdweave.operators.correlator.api import Flavor,Hadron,Tag,gen_correlator
    from lqcdweave.operators.single_hadron.api import meson55_preset
    _,options,elemental=declarations()
    elemental=ArrayElemental(elemental.sectors,(meson55_preset()[0]['component'],),phase_convention='midpoint',
        cfg='cfg',data_identity=elemental.data_identity,momentum_list=((0,0,0),))
    vertices=RecipeVertex(elemental,momenta=((0,0,0),))
    pion=Hadron(vertices.symbol('op_00_gamma5_A1',tag=Tag(0,0))[0],Flavor('ud'))
    request=gen_correlator([[pion],[pion]],[0,1],[False,True])
    options['vertex_map']=vertices
    with ContractionTask(request,**options) as task:
        task.plan()
        expected=task.run().value
        archive=tmp_path/'recipe.json.gz'
        task.save_plan(archive)
    with no_planning(), ContractionTask.load_plan(archive,configuration='cfg',vertex_map=vertices,
            propagator_map=options['propagator_map']) as loaded:
        np.testing.assert_allclose(loaded.run().value,expected,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('optimize',['anytime','meson_matrix'])
@pytest.mark.parametrize('backend',['numpy',pytest.param('cupy',id='gpu')])
def test_standard_optimizer_and_dedicated_matrix_keep_their_selected_route(optimize,backend,tmp_path):
    if backend=='cupy':
        pytest.importorskip('cupy')
    request,options,elemental=declarations(batch='auto',backend=backend)
    request=correlator(flavor='I0',structured=True)
    expected=calculate(request,**dict(options,optimize='none',backend='numpy')).value
    options['optimize']=optimize
    with ContractionTask(request,**options) as task:
        task.plan()
        archive=tmp_path/'matrix.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value,expected,rtol=3e-12,atol=3e-12)
            with ContractionTask.load_plan(archive,configuration='cfg',vertex_map=elemental,
                    propagator_map=options['propagator_map']) as loaded:
                np.testing.assert_allclose(loaded.run().value,expected,rtol=3e-12,atol=3e-12)


def test_task_gpu_batches_archive_and_tail(tmp_path):
    pytest.importorskip('cupy')
    request,options,elemental=declarations(backend='cupy')
    expected=calculate(request,**dict(options,backend='numpy',batch=1)).value
    with ContractionTask(request,**options) as task:
        task.plan()
        audit_cfg_plan(task._selected.plan)
        archive=tmp_path/'gpu-plan.json.gz'
        with no_planning():
            task.save_plan(archive)
            np.testing.assert_allclose(task.run().value,expected,rtol=5e-12,atol=5e-12)
            with ContractionTask.load_plan(archive,configuration='cfg',vertex_map=elemental,
                    propagator_map=options['propagator_map']) as loaded:
                np.testing.assert_allclose(loaded.run().value,expected,rtol=5e-12,atol=5e-12)
