"""Output-only wide arithmetic preserves products before cancellation."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ContractionTerm, OutputNode, OutputRecipe, TensorSpec, TermContribution,
    describe_coordinates,
)
from lqcdweave.contraction.results.api import bind_output_assembly, describe_output_layout
from lqcdweave.contraction.execution import output_program
from lqcdweave.storage.contraction_plans import api as archive, compact


def cancellation_fixture():
    # Actual loop partial traces exported from cfg3740, source78/sink105.
    # This reproduces output-expression rounding, not the original GPU GEMMs.
    values = tuple(np.array(v, dtype='complex64') for v in (
        -4.758253743468904-.041272753033141915j,
        4.7582545470953335-.04127296107605177j,
        -4.632686522676167+.12049131727753953j,
        4.632687018350304+.12049213693573868j))
    term = lambda i: OutputNode('term', TermContribution(f't{i}'))
    root = OutputNode('add', children=tuple(OutputNode('multiply', children=(
        OutputNode('constant', 2), term(i), term(j))) for i in (0, 1) for j in (2, 3)))
    coords, _ = describe_coordinates()
    recipe = OutputRecipe(root, tuple(f't{i}' for i in range(4)), coords, 'complex64')
    problem = SimpleNamespace(terms=tuple(ContractionTerm((f'v{i}',), '->') for i in range(4)),
        specs={f'v{i}': TensorSpec((), 'complex64') for i in range(4)})
    return recipe, problem, values


def test_real_partial_loops_need_wide_products_not_only_wide_sum():
    recipe, problem, values = cancellation_fixture()
    reference = .03978430263386715+3.054897944525081e-7j
    original_limit = 2.489635009781809e-6
    old = bind_output_assembly(recipe)(values)
    rounded_products = [np.complex64(2*complex(values[i])*complex(values[j]))
                        for i in (0, 1) for j in (2, 3)]
    sum_only = sum(map(complex, rounded_products))
    assert abs(old-reference) > original_limit
    assert abs(sum_only-reference) > original_limit
    wide = replace(recipe, intermediate_dtype='complex128')
    actual = bind_output_assembly(wide)(values)
    debug = bind_output_assembly(wide, debug=True)(values)
    assert np.asarray(actual).dtype == np.dtype('complex64')
    assert abs(actual-reference) < original_limit
    assert abs(actual-reference) < 8e-8
    np.testing.assert_array_equal(actual, debug)
    assert describe_output_layout(wide, problem)[0] > describe_output_layout(recipe, problem)[0]


def test_actual_archived_order_interleaves_connected_terms_with_large_products():
    # The four nonzero scatter blocks for the original [0,0] field, in saved
    # order. Each adds -connected_trace before its disconnected product.
    recipe, _, loops = cancellation_fixture()
    connected = (.0016029612206866922+.0017156527800539435j,
        .0031994224084608723-.002275417229776612j,
        .0031994501873743426+.0022754238774935435j,
        .0016029393881044495-.0017156672696816055j)
    terms = tuple(OutputNode('term', TermContribution(f'c{i}')) for i in range(4))
    children = []
    for term, product in zip(terms, recipe.root.children):
        block = OutputNode('add', children=(OutputNode('multiply', children=(
            OutputNode('constant', -1), term)), product))
        children.append(OutputNode('scatter', ((0,), (0,)), (OutputNode('reshape',
            children=(block,), shape=(1, 1)),), shape=(10, 10)))
    root = OutputNode('select', (0, 0), (OutputNode('add', children=tuple(children)),))
    recipe = replace(recipe, root=root, term_ids=(*recipe.term_ids, *(f'c{i}' for i in range(4))))
    values = (*loops, *(np.array(-v, dtype='complex64') for v in connected))
    reference = .0493890758384935+2.9764788372213875e-7j
    limit = 2.489635009781809e-6
    old = bind_output_assembly(recipe)(values)
    wide = bind_output_assembly(replace(recipe, intermediate_dtype='complex128'))(values)
    assert abs(old-reference) > limit
    assert abs(wide-reference) < 8e-8 < limit


@pytest.mark.parametrize('dtype,wide', [('float32', 'float64'), ('complex64', 'complex128')])
def test_all_output_operators_keep_wide_intermediates_and_narrow_fields(dtype, wide):
    coords, _ = describe_coordinates()
    term = OutputNode('term', TermContribution('x'))
    root = OutputNode('add', children=(OutputNode('constant', 2**24), term,
                                    OutputNode('constant', -(2**24))))
    expression = OutputNode('multiply', children=(root, root))
    matrix = OutputNode('reshape', children=(OutputNode('array', shape=(2,), children=(
        expression, OutputNode('power', 2, (root,)))),), shape=(1, 2))
    nested = OutputNode('dict', keys=('a', 'b'), children=(matrix,
        OutputNode('tuple', children=(root, OutputNode('constant_array', np.array([.25]), shape=(1,))))))
    recipe = OutputRecipe(nested, ('x',), coords, dtype, wide)
    for value in (1., 3.):
        for debug in (False, True):
            result = bind_output_assembly(recipe, debug=debug)((np.array(value, dtype=dtype),))
            assert result['a'].dtype == np.dtype(dtype)
            np.testing.assert_array_equal(result['a'], [[value**2, value**2]])
            assert np.asarray(result['b'][0]).dtype == np.dtype(dtype)
            assert result['b'][0] == value
            assert result['b'][1].dtype == np.dtype(dtype)


def test_object_container_boundary_cast_and_array_owners_are_call_local():
    recipe, _, values = cancellation_fixture()
    leaf = recipe.root
    root = OutputNode('array', shape=(2,), children=(leaf,
        OutputNode('constant_array', np.ones(3), shape=(3,))))
    wide = replace(recipe, root=root, intermediate_dtype='complex128')
    assemble = bind_output_assembly(wide)
    left, right = assemble(values), assemble(tuple(v*2 for v in values))
    assert left.dtype == object
    assert np.asarray(left[0]).dtype == np.dtype('complex64')
    assert left[1].dtype == np.dtype('complex64')
    np.testing.assert_array_equal(right[0], left[0]*4)
    assert not np.shares_memory(left[1], right[1])


def test_device_metadata_keeps_narrow_inputs_fields_and_wide_segmented_scratch():
    recipe, problem, _ = cancellation_fixture()
    wide = replace(recipe, intermediate_dtype='complex128')
    # Force small stages to check precisely the scratch that P will reserve.
    with patch.object(output_program, '_MAX_STAGE_NODES', 12):
        description = output_program._Description(wide, problem)
        assert description.scratch_slots
        assert all(dtype == np.dtype('complex128') for _, dtype in description.scratch_slots)
        assert all(dtype == np.dtype('complex64') for _, dtype in description.term_specs)
        assert description.fields[0][1].dtype == np.dtype('complex64')
        reserved = output_program.describe_device_output_reservation(wide, problem)
        assert reserved['fields'] == (((), (), '<c8'),)
        assert reserved['device_bytes'] == sum(reserved['device_owner_bytes'])
        assert reserved['device_bytes'] == 8 + description.scratch_nbytes + description.table_nbytes
        sources = []
        for stage in description.stages:
            emitter = output_program._Emitter(description)
            emitter.emit(stage.expression, 'i')
            sources.append('\n'.join(emitter.lines))
        assert any('complex<double>' in source for source in sources)
        assert 'complex<float>' in sources[-1]


def test_old_archive_field_defaults_preserve_old_rounding_new_records_preserve_wide():
    recipe, _, values = cancellation_fixture()
    encoded = archive._encode_plan_value(recipe)
    assert encoded['type'].endswith('.OutputRecipe')
    del encoded['value']['intermediate_dtype']
    old = archive.decode_plan_value(encoded)
    assert old.intermediate_dtype is None
    np.testing.assert_array_equal(bind_output_assembly(old)(values),
                                  bind_output_assembly(recipe)(values))
    historical = compact.encode(recipe)
    for schema in historical['value']['schemas']:
        if schema[0].endswith('.OutputRecipe'):
            removed = schema[1].index('intermediate_dtype')
            assert removed == len(schema[1])-1
            schema[1].pop()
            schema[2] = [pair for pair in schema[2] if pair[0] != removed]
    old_compact = archive.decode_plan_value(historical)
    assert old_compact.intermediate_dtype is None
    np.testing.assert_array_equal(bind_output_assembly(old_compact)(values),
                                  bind_output_assembly(recipe)(values))
    wide = replace(recipe, intermediate_dtype='complex128')
    for encoded in (archive._encode_plan_value(wide), compact.encode(wide)):
        restored = archive.decode_plan_value(encoded)
        assert restored == wide
        np.testing.assert_array_equal(bind_output_assembly(restored)(values),
                                      bind_output_assembly(wide)(values))
    with pytest.raises(TypeError):
        replace(recipe, intermediate_dtype='float64')


def test_public_planning_budgets_output_precision_and_two_cfg_fixed_archive():
    from test_structured_correlator import fixture, task
    from lqcdweave.adapters.elemental import ArrayElemental
    from lqcdweave.adapters.propagator import ArrayPropagator
    from lqcdweave.contraction.correlator import calculate, save_calculation_plan
    from lqcdweave.contraction.correlator_cfg import complete_calculation
    from lqcdweave.operators.correlator.api import split_meson_mode_pairs
    bind, e128, p128 = fixture()
    e = ArrayElemental({key: value.astype('complex64') for key, value in e128.sectors.items()},
        e128.components, phase_convention='anchor', cfg='cfg', data_identity='precision-e')
    p = ArrayPropagator({key: value.astype('complex64') for key, value in p128.sectors.items()},
        time_convention='absolute', cfg='cfg', data_identity='precision-p')
    request = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=(0, 1, 3))
    options = dict(vertex_map=lambda row: bind(row).with_reader(e), configuration='cfg',
        propagator_map={'S^q': p, r'S^q_\mathrm{local}': p}, usedNe=3, optimize='none',
        memory_limit=1 << 28, host_memory_limit=1 << 28)
    selected = calculate(request, execute=False, time_axes={0: 1, 1: 0}, **options)
    recipe = selected.expanded.output_recipe
    assert (recipe.dtype, recipe.intermediate_dtype) == ('complex64', 'complex128')
    assert all(spec.dtype == np.dtype('complex64') for spec in selected.expanded.problem.specs.values())
    reserve, fields = describe_output_layout(recipe, selected.expanded.problem)
    assert selected.expanded.inputs.assembly_host_bytes >= reserve
    assert selected.plan.reserved_host_bytes >= reserve
    assert all(dtype == '<c8' for _, _, dtype in fields)
    coords, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}, {0: 0, 1: 1}])
    complete = complete_calculation(selected, coordinates=coords)
    assert complete.plan.templates[0].output.intermediate_dtype == 'complex128'
    expected, current = {}, {}
    for cfg, scale in (('a', 1), ('b', 2)):
        readers = (ArrayElemental({key: value*scale for key, value in e.sectors.items()},
            e.components, phase_convention='anchor', cfg=cfg, data_identity='precision-e:'+cfg),
            ArrayPropagator({key: value*scale for key, value in p.sectors.items()},
            time_convention='absolute', cfg=cfg, data_identity='precision-p:'+cfg))
        current[cfg] = readers
        expected[cfg] = calculate(request, **dict(options, configuration=cfg,
            vertex_map=lambda row: bind(row).with_reader(readers[0], configuration=cfg),
            propagator_map={'S^q': readers[1], r'S^q_\mathrm{local}': readers[1]}),
            time_pairs=[dict(point.values) for point in coords.points]).value
    @contextmanager
    def factory(cfg):
        yield {r.reader_role: current[cfg][0 if r.family == 'elemental' else 1]
            for primary in complete.plan.requests.values()
            for r in (primary, *primary.aliases)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'complete.json.gz'
        save_calculation_plan(path, complete)
        with ExitStack() as blocked:
            for name in ('lqcdweave.contraction.correlator.expand_correlator',
                         'lqcdweave.contraction.expansion.matrix_inputs._bind_matrix_base',
                         'lqcdweave.contraction.correlator.find_contraction_path',
                         'lqcdweave.contraction.planning.cfg_plan._bind_cfg_coordinates'):
                blocked.enter_context(patch(name, side_effect=AssertionError(name)))
            actual = dict(calculate(plan_archive=path, configuration=iter(('a', 'b')), data=factory).value)
    for cfg in expected:
        np.testing.assert_allclose(actual[cfg], expected[cfg], rtol=3e-6, atol=2e-5)
        assert actual[cfg].dtype == np.dtype('complex64')
