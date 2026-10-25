"""Independent scalar references for fixed-path coordinate input descriptions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceReadRequest, SourceSpec, StorageBlock, TensorAxisMetadata,
    TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.expansion.saved_inputs import restore_input_loader
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.planning.coordinate_inputs import (
    bind_coordinate_inputs, describe_coordinate_batch,
)


class Reader:
    def __init__(self, arrays, identity):
        self.arrays, self.identity, self.reads = arrays, identity, 0

    def describe_block(self, *arguments):
        array = self.arrays[arguments]
        return TensorAxisMetadata(tuple(f'a{i}' for i in range(array.ndim)),
            array.shape, array.dtype, self.identity, repr(arguments))

    def block(self, *arguments):
        self.reads += 1
        return self.arrays[arguments]


def fixture(backend='numpy'):
    random = np.random.default_rng(1098)

    def matrix(shape):
        return (random.standard_normal(shape) + 1j * random.standard_normal(shape)).astype('complex64')

    # All source/sink directions are actual absolute coordinates, including
    # equal endpoints. Rectangular modes prevent accidental transposition.
    readers = {
        'forward': Reader({(sink, 3): matrix((2, 3)) for sink in range(40)}, 'forward'),
        'reverse': Reader({(3, sink): matrix((3, 2)) for sink in range(40)}, 'reverse'),
        'sink': Reader({(sink,): matrix((3, 3)) for sink in range(40)}, 'sink'),
        'source': Reader({(3,): matrix((2, 2))}, 'source'),
    }
    requests, blocks, preparations, specs, sources, metadata = {}, {}, {}, {}, {}, {}
    roles = {
        'a': ('forward', (0, 3), ((0, 'sink', 0, None), (1, 'source', 0, None))),
        'r': ('reverse', (3, 0), ((0, 'source', 0, None), (1, 'sink', 0, None))),
        'v': ('sink', (0,), ((0, 'sink', 0, None),)),
        's': ('source', (3,), ((0, 'source', 0, None),)),
    }
    for key, (role, arguments, coordinates) in roles.items():
        raw = 'raw-' + key
        description = readers[role].describe_block(*arguments)
        request = SourceReadRequest(role, 'propagator' if len(arguments) == 2 else 'elemental',
            arguments, (), description, coordinate_arguments=coordinates)
        requests[raw] = request
        size = int(np.prod(description.shape)) * description.dtype.itemsize
        blocks[raw] = StorageBlock(role, repr(arguments), size, host_bytes=size,
            device_bytes=size if backend == 'cupy' else 0,
            transfer_bytes=size if backend == 'cupy' else 0)
        specs[key] = TensorSpec(description.shape, description.dtype)
        preparations[key] = SourcePreparation((raw,), dtype='complex64',
            recipe=TensorPreparation('identity', (raw,), compute_dtype='complex64'))
        sources[key] = SourceSpec(key, size, host_region_bytes=size, snapshot_bytes=size)
        for args in readers[role].arrays:
            metadata[(role, args, ())] = readers[role].describe_block(*args)
    constant = matrix((2, 2))
    specs['constant'] = TensorSpec(constant.shape, constant.dtype)
    preparations['constant'] = SourcePreparation((), dtype='complex64',
        recipe=TensorPreparation('constant', constants=(ArrayConstant.from_array(constant),)))
    sources['constant'] = SourceSpec('constant', 0, snapshot_bytes=constant.nbytes)
    problem = ContractionProblem(specs, (
        ContractionTerm(('a', 'v', 'r', 's'), 'ij,jk,kl,li->'),
        ContractionTerm(('s', 'constant'), 'ij,ji->'),
    ), sources, backend=backend)
    plan = SimpleNamespace(state=SimpleNamespace(problem=problem, catalog=specs),
        preparations=preparations, blocks=blocks)
    coordinates, _ = describe_coordinates(time_axes={'source': 3, 'sink': tuple(range(40))})
    return plan, requests, preparations, specs, coordinates.points, metadata, readers, constant


def describe_fixture(backend='numpy'):
    plan, requests, preparations, specs, points, metadata, readers, constant = fixture(backend)
    binding = bind_coordinate_inputs(plan, requests, preparations, specs, points, read_metadata=metadata)
    assert all(reader.reads == 0 for reader in readers.values())
    return binding, readers, constant


def materialize(description, readers):
    physical = restore_input_loader(description.requests, readers,
        description.preparation_closure, description.specs, debug=True)
    loader = bind_prepared_loader(physical, description.preparation_closure, description.specs)
    raw = {key: loader.read_block(key, block) for key, block in description.blocks.items()}
    return {key: loader.materialize(key, source, raw)
            for key, source in description.problem.sources.items()}


def evaluate(problem, arrays):
    outputs = []
    for term in problem.terms:
        labels_seen = tuple(dict.fromkeys(label for labels in
            (*term.subscripts.inputs, term.subscripts.output) for label in labels))
        axis_map = {label: index for index, label in enumerate(labels_seen)}
        arguments = []
        for key, labels in zip(term.inputs, term.subscripts.inputs):
            arguments.extend((arrays[key], [axis_map[label] for label in labels]))
        outputs.append(np.einsum(*arguments, [axis_map[label] for label in term.subscripts.output]))
    return outputs


@pytest.mark.parametrize('width', [1, 7, 40])
def test_forty_sinks_rectangular_complex_matches_scalar_reference(width):
    binding, readers, constant = describe_fixture()
    values = []
    shared_source_ids = {point['s'] for point in binding.source_names}
    assert len(shared_source_ids) == 1
    assert len(binding.requests) == 121  # 3 sink-dependent reads, one shared source.
    for start in range(0, 40, width):
        description = describe_coordinate_batch(binding, range(start, min(start + width, 40)), lane_axis=4)
        arrays = materialize(description, readers)
        outputs = evaluate(description.problem, arrays)
        assert description.inputs.batch_output_axes[1] is None
        assert description.problem.terms[0].coefficient == 1
        if min(width, 40 - start) > 1:
            assert description.dynamic_sources == frozenset({'a', 'r', 'v'})
            assert arrays['a'].shape[0] == min(width, 40 - start)
            values.extend(outputs[0])
        else:
            assert not description.dynamic_sources
            values.append(outputs[0])
        np.testing.assert_allclose(outputs[1],
            np.einsum('ij,ji->', readers['source'].arrays[(3,)], constant), rtol=2e-6)
    reference = [np.einsum('ij,jk,kl,li->', readers['forward'].arrays[(sink, 3)],
        readers['sink'].arrays[(sink,)], readers['reverse'].arrays[(3, sink)],
        readers['source'].arrays[(3,)]) for sink in range(40)]
    np.testing.assert_allclose(values, reference, rtol=3e-6, atol=1e-5)


def test_prefixed_batch_closures_merge_and_shared_identity_stays_one_owner():
    binding, readers, _ = describe_fixture()
    first = describe_coordinate_batch(binding, range(0, 4), lane_axis=4, key_prefix='batch:0:')
    second = describe_coordinate_batch(binding, range(4, 7), lane_axis=4, key_prefix='batch:1:')
    assert first.source_map['a'] == 'batch:0:a'
    assert not set(first.problem.sources) & set(second.problem.sources)
    shared = set(first.preparation_closure) & set(second.preparation_closure)
    assert all(first.preparation_closure[key] == second.preparation_closure[key] for key in shared)
    assert len(first.blocks) == 13
    assert len(second.blocks) == 10
    # Restore once against the merged closure, exactly as joint .run() does.
    closure = dict(first.preparation_closure) | dict(second.preparation_closure)
    specs = dict(first.specs) | dict(second.specs)
    requests = dict(first.requests) | dict(second.requests)
    loader = bind_prepared_loader(restore_input_loader(requests, readers, closure, specs), closure, specs)
    raw = {key: loader.read_block(key, block)
           for key, block in (dict(first.blocks) | dict(second.blocks)).items()}
    for description in (first, second):
        arrays = {key: loader.materialize(key, source, raw)
                  for key, source in description.problem.sources.items()}
        assert arrays[description.source_map['a']].shape[0] == len(description.points)
    assert readers['source'].reads == 1


def test_recipe_identity_not_shape_controls_sharing_and_repeated_coordinates():
    plan, requests, preparations, specs, points, metadata, _, _ = fixture()
    repeated, _ = describe_coordinates(time_pairs=[{'source': 3, 'sink': 4}, {'source': 3, 'sink': 4}])
    binding = bind_coordinate_inputs(plan, requests, preparations, specs, repeated.points, read_metadata=metadata)
    result = describe_coordinate_batch(binding, (0, 1), lane_axis=4)
    assert not result.dynamic_sources
    assert len(result.blocks) == 4
    # Same shape, different endpoint request: no invariant-source shortcut.
    binding = bind_coordinate_inputs(plan, requests, preparations, specs, points[:2], read_metadata=metadata)
    assert binding.source_names[0]['a'] != binding.source_names[1]['a']


def test_metadata_only_cupy_placement_and_no_precision_change():
    binding, readers, _ = describe_fixture('cupy')
    description = describe_coordinate_batch(binding, (0, 1, 2), lane_axis=4)
    assert description.problem.backend == 'cupy'
    assert all(spec.dtype == np.dtype('complex64') for spec in description.specs.values())
    assert all(prep.recipe is not None for prep in description.preparation_closure.values())
    assert description.preparations['a'].workspace_device_bytes > 0
    assert description.preparations['a'].workspace_host_bytes == 0
    assert all(reader.reads == 0 for reader in readers.values())


def test_nested_stack_and_complex_source_dagger_use_native_preparations():
    random = np.random.default_rng(312)
    gamma = (random.normal(size=(4, 4)) + 1j * random.normal(size=(4, 4))).astype('complex64')
    raw = { (sink,): (random.normal(size=(3, 2)) + 1j * random.normal(size=(3, 2))).astype('complex64')
           for sink in range(4)}
    reader = Reader(raw, 'elemental')
    metadata = reader.describe_block(0)
    request = SourceReadRequest('elemental', 'elemental', (0,), (), metadata,
        coordinate_arguments=((0, 'sink', 0, None),))
    shape = (4, 4, 2, 3)
    scalar_spec = TensorSpec(shape, 'complex64')
    preps = {'part': SourcePreparation(('raw',), workspace_host_bytes=4 * 4 * 2 * 3 * 8,
        dtype='complex64', recipe=TensorPreparation('spin_spatial', ('raw',),
            (ArrayConstant.from_array(gamma),), conjugate=True, permutation=(1, 0), compute_dtype='complex64')),
        'packed': SourcePreparation(('raw',), dtype='complex64',
            recipe=TensorPreparation('stack', ('part', 'part'), compute_dtype='complex64'))}
    specs = {'part': scalar_spec, 'packed': TensorSpec((2, *shape), 'complex64')}
    problem = ContractionProblem({'packed': specs['packed']},
        (ContractionTerm(('packed',), 'xijab->x'),),
        {'packed': SourceSpec('packed', raw[(0,)].nbytes, snapshot_bytes=2 * 4 * 4 * 2 * 3 * 8)})
    plan = SimpleNamespace(state=SimpleNamespace(problem=problem, catalog=specs), preparations={'packed': preps['packed']},
        blocks={'raw': StorageBlock('elemental', '0', raw[(0,)].nbytes, host_bytes=raw[(0,)].nbytes)})
    points, _ = describe_coordinates(time_axes={'sink': range(4)})
    overrides = {('elemental', args, ()): reader.describe_block(*args) for args in raw}
    binding = bind_coordinate_inputs(plan, {'raw': request}, preps, specs, points.points, read_metadata=overrides)
    description = describe_coordinate_batch(binding, range(4), lane_axis=5)
    arrays = materialize(description, {'elemental': reader})
    expected = np.stack([np.stack([np.einsum('ij,ab->ijab', gamma, raw[(sink,)].conj().T)] * 2)
                         for sink in range(4)])
    np.testing.assert_allclose(arrays['packed'], expected, rtol=2e-6, atol=2e-6)
    assert arrays['packed'].shape == (4, 2, *shape)


def test_rejects_changed_metadata_aliases_opaque_and_invalid_lanes():
    plan, requests, preparations, specs, points, metadata, _, _ = fixture()
    changed = dict(metadata)
    key = ('forward', (1, 3), ())
    changed[key] = replace(metadata[key], dtype='complex128')
    with pytest.raises(ValueError, match='dtype'):
        bind_coordinate_inputs(plan, requests, preparations, specs, points[:2], read_metadata=changed)
    opaque = dict(preparations, a=replace(preparations['a'], recipe=None))
    plan.preparations = opaque
    with pytest.raises(ValueError, match='opaque'):
        bind_coordinate_inputs(plan, requests, opaque, specs, points[:2], read_metadata=metadata)
    plan.preparations = preparations
    aliases = dict(requests)
    aliases['raw-a'] = replace(requests['raw-a'], aliases=(requests['raw-r'],))
    with pytest.raises(ValueError, match='sharing relation'):
        bind_coordinate_inputs(plan, aliases, preparations, specs, points[:2], read_metadata=metadata)
    binding = bind_coordinate_inputs(plan, requests, preparations, specs, points[:2], read_metadata=metadata)
    for indices, axis in (((1, 0), 4), ((0, 0), 4), ((0, 1), ord('i'))):
        with pytest.raises(ValueError):
            describe_coordinate_batch(binding, indices, lane_axis=axis)


def test_list_valued_conventions_are_compared_by_content():
    plan, requests, preparations, specs, points, metadata, _, _ = fixture()
    requests = {key: replace(request, conventions=(('momenta', [[0, 0, 0], [1, 0, 0]]),))
                for key, request in requests.items()}
    binding = bind_coordinate_inputs(plan, requests, preparations, specs, points[:2], read_metadata=metadata)
    assert len(binding.requests) == 7


def test_existing_storage_and_replay_execute_described_batch_without_new_kernel():
    from lqcdweave.contraction.execution.api import replay_scene_storage
    from lqcdweave.contraction.performance.api import CostProfile
    from lqcdweave.contraction.planning.api import solve_storage, plan_scene_storage
    from lqcdweave.contraction.planning.coordinate_path import lift_coordinate_path
    plan, requests, preps, specs, points, metadata, readers, _ = fixture()
    profile = CostProfile()
    selected = solve_storage(plan.state.problem, plan.blocks, preps, optimizer='anytime',
        memory_limit=1 << 20, deadline_seconds=1, max_states=100,
        root_width=1, branch_width=1, component_workers=1, kernel_batching=False).plan
    binding = bind_coordinate_inputs(selected, requests, preps, specs, points[:7], read_metadata=metadata)
    description = describe_coordinate_batch(binding, range(7), lane_axis=200, key_prefix='batch:0:')
    lifted = lift_coordinate_path(selected.state, description.problem, description.source_map,
        lane_axis=200, profile=profile, memory_limit=1 << 20)
    scheduled = plan_scene_storage(lifted, description.blocks, description.preparations,
        profile=profile, memory_limit=1 << 20, debug=True)
    loader = bind_prepared_loader(restore_input_loader(description.requests, readers,
        description.preparation_closure, description.specs), description.preparation_closure, description.specs)
    actual = replay_scene_storage(scheduled, loader, backend=np, io_overlap=False)
    expected = evaluate(description.problem, materialize(description, readers))
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=3e-6, atol=1e-5)
    assert scheduled.feasible


def test_raw_double_to_preselected_single_precision_uses_existing_identity_recipe():
    plan, requests, preps, specs, points, metadata, readers, _ = fixture()
    readers['forward'].arrays = {args: value.astype('complex128') for args, value in readers['forward'].arrays.items()}
    requests['raw-a'] = replace(requests['raw-a'], metadata=readers['forward'].describe_block(0, 3))
    for args in readers['forward'].arrays:
        metadata[('forward', args, ())] = readers['forward'].describe_block(*args)
    block = plan.blocks['raw-a']
    plan.blocks['raw-a'] = replace(block, read_bytes=block.read_bytes * 2, host_bytes=block.host_bytes * 2)
    binding = bind_coordinate_inputs(plan, requests, preps, specs, points[:3], read_metadata=metadata)
    description = describe_coordinate_batch(binding, range(3), lane_axis=200)
    values = materialize(description, readers)
    assert values['a'].dtype == np.dtype('complex64')
    np.testing.assert_array_equal(values['a'], np.stack([readers['forward'].arrays[(sink, 3)].astype('complex64')
                                                       for sink in range(3)]))
