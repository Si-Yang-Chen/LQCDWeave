"""Whole structured sums against explicit support-index references."""
import os
from dataclasses import replace

import numpy as np
import pytest

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contracts.correlator.api import ArrayConstant, SourcePreparation, TensorSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.operators.single_hadron.profile import SpectralProfile


def array_backend():
    if os.environ.get('ED_TEST_BACKEND') == 'cupy':
        import cupy
        assert int(cupy.sum(cupy.arange(5)).get()) == 10
        return cupy
    return np


def fixture(structure='diagonal', *, points=False, dtype=np.complex128, source=False,
            selected=False, profile=False, ne=3, times=3):
    rng = np.random.default_rng(261023)
    shape = (2, 3, 2, 3) if points else (ne, ne)
    raw = rng.normal(size=(2, 1, times, *shape))
    if np.dtype(dtype).kind == 'c':
        raw = raw + 1j * rng.normal(size=raw.shape)
    raw = raw.astype(dtype)
    components = tuple(ElementalComponent(i) for i in (0, 1))
    sector = 'p2p' if points else 'v2v'
    reader = ArrayElemental({sector: raw}, components, phase_convention='anchor',
        cfg='cfg', data_identity='mixed/E',
        elemental_structures={(sector, components[0], 0): structure})
    spectral = (SpectralProfile(rng.normal(size=(2, times, ne)) + .2j,
        rng.normal(size=(2, times, ne)) - .3j, coefficients=(.4+.2j, -.1j), cfg='cfg')
        if profile else None)
    terms = (OperatorTerm(3, components[0], 0, .7+.2j, profile=spectral),
             OperatorTerm(14, components[1], 0, -.3+.1j),
             OperatorTerm(0, components[0], 0, 1e-16j, profile=spectral))
    operator = Operator.from_terms('mixed', terms)
    modes = ((2, 0), (0, 2)) if selected else None
    vertex = Meson(reader, operator, source, phase_convention='anchor')
    if modes is not None:
        vertex = vertex.with_mode_indices(modes)
    return vertex, raw, terms, vertex.mode_indices


def explicit_vertex(raw, terms, time, *, source=False, structure='diagonal', modes=None):
    """No production embedding/preparation code participates in this oracle."""
    shape = raw.shape[3:]
    spatial_shape = ((len(modes[0]), len(modes[1])) if modes is not None else shape)
    result = np.zeros((4, 4, *spatial_shape), dtype=np.complex128)
    for term in terms:
        spatial = np.zeros(spatial_shape, dtype=np.complex128)
        for local in np.ndindex(spatial_shape):
            actual = (modes[0][local[0]], modes[1][local[1]]) if modes is not None else local
            supported = (structure == 'general' or term.component.key != 0 or
                (actual[0] == actual[1] if len(shape) == 2 else
                 actual[0] == actual[2] and (structure == 'point_diagonal' or actual[1] == actual[3])))
            if supported:
                value = raw[term.component.key, 0, time][actual]
                if term.profile is not None:
                    value = sum(c * left[time, local[0]].conjugate() * value * right[time, local[1]]
                        for c, left, right in zip(term.profile.coefficients, term.profile.left,
                                                term.profile.right))
                spatial[local] = value
        spin = gamma(term.gamma)
        coefficient = term.coefficient
        if source:
            spin = gamma(8) @ spin.conj().T @ gamma(8)
            spatial = spatial.conj().transpose((1, 0) if len(shape) == 2 else (2, 3, 0, 1))
            coefficient = coefficient.conjugate()
        result += coefficient * np.einsum('ab,...->ab...', spin, spatial, optimize=False)
    return result


@pytest.mark.parametrize('source', (False, True))
@pytest.mark.parametrize('kind', ('v', 'p-complete', 'p-point', 'selected', 'profile'))
@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.complex64, np.complex128))
def test_support_embedding_ignores_off_support_garbage_and_retains_order(kind, source, dtype):
    points = kind.startswith('p-')
    structure = 'point_diagonal' if kind == 'p-point' else 'diagonal'
    vertex, raw, terms, modes = fixture(structure, points=points, dtype=dtype, source=source,
                                      selected=kind == 'selected', profile=kind == 'profile')
    descriptors = vertex.describe_core_factors(1, 'p2p' if points else 'v2v')
    assert len(descriptors) == 1 and descriptors[0].mixed_structure
    assert descriptors[0].tensor.structure == 'general'
    backend = array_backend()
    target_shape = explicit_vertex(raw, terms, 1, source=source, structure=structure, modes=modes).shape
    specifications, preparations, blocks = {}, {}, {}
    for i, item in enumerate(descriptors[0].factors):
        raw_key, key = f'raw:{i}', f'prepared:{i}'
        blocks[raw_key] = backend.asarray(vertex.read_factor_block(item))
        specifications[key] = TensorSpec(target_shape, np.complex128)
        preparations[key] = SourcePreparation((raw_key,), recipe=vertex.describe_factor_preparation(
            item, raw_key, embed_structure=True))
    loader = bind_prepared_loader(None, preparations, specifications, backend=backend)
    actual = sum((term.coefficient.conjugate() if source else term.coefficient)
                 * loader.materialize(f'prepared:{i}', None, blocks)
                 for i, term in enumerate(terms))
    if backend is not np:
        actual = backend.asnumpy(actual)
    expected = explicit_vertex(raw, terms, 1, source=source, structure=structure, modes=modes)
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=4e-12)


def test_all_diagonal_dense_and_scalar_groups_keep_the_existing_representation():
    vertex, _, _, _ = fixture()
    first = vertex.operator.terms[0]
    vertex.operator = Operator.from_terms('diagonal', (first, replace(first, coefficient=.2j)))
    descriptor, = vertex.describe_core_factors(0)
    assert not descriptor.mixed_structure and descriptor.tensor.structure == 'diagonal'
    vertex.operator = Operator.from_terms('single', (first,))
    assert not hasattr(vertex.describe_core_factors(0)[0], 'factors')


@pytest.mark.parametrize('source', (False, True))
@pytest.mark.parametrize('structure', ('diagonal', 'point_diagonal'))
def test_point_embedding_matches_actual_labels_and_preserves_color_blocks(source, structure):
    vertex, raw, terms, _ = fixture(structure, points=True, source=source)
    vertex.reader.basis_endpoints = {
        ('p2p', 'bar'): ('actual-pool', (31, 7)),
        ('p2p', 'field'): ('actual-pool', (7, 31)),
    }
    descriptor, = vertex.describe_core_factors(1, 'p2p')
    assert descriptor.mixed_structure
    expected_raw = raw.copy()
    for i, j, k, l in np.ndindex(raw.shape[3:]):
        if (31, 7)[i] != (7, 31)[k] or (structure == 'diagonal' and j != l):
            expected_raw[0, 0, 1, i, j, k, l] = 0
    # The oracle now contains physically supported values only; treat it as
    # general so its reference does not use the production support machinery.
    expected = explicit_vertex(expected_raw, terms, 1, source=source, structure='general')
    backend = array_backend()
    total = np.zeros(expected.shape, dtype=complex)
    for i, item in enumerate(descriptor.factors):
        recipe = vertex.describe_factor_preparation(item, 'raw', embed_structure=True)
        loader = bind_prepared_loader(None, {'out': SourcePreparation(('raw',), recipe=recipe)},
            {'out': TensorSpec(expected.shape, np.complex128)}, backend=backend)
        result = loader.materialize('out', None, {'raw': backend.asarray(vertex.read_factor_block(item))})
        result = result if backend is np else backend.asnumpy(result)
        total += (item.operator_term.coefficient.conjugate() if source else item.operator_term.coefficient) * result
    np.testing.assert_allclose(total, expected, rtol=5e-12, atol=5e-12)


def test_components_from_different_point_pools_remain_distinct_vertices():
    vertex, _, terms, _ = fixture(points=True)
    vertex.reader.sampling_pool_ids = {
        (term.component, 0, time, side): f'pool-{term.component.key}'
        for term in terms for time in range(3) for side in ('bar', 'field')
    }
    descriptions = vertex.describe_core_factors(1, 'p2p')
    assert len(descriptions) == 2
    assert all(len(set(item.point_pool_ids.values())) == 1 for item in descriptions)


@pytest.mark.parametrize('dtype', (np.complex64, np.complex128))
def test_support_survives_exact_unit_spin_spatial_preparation(dtype):
    vertex, raw, _, _ = fixture(source=True, dtype=dtype)
    description = vertex.describe_factors(1, indices=(0,))[0]
    recipe = vertex.describe_factor_preparation(description, 'raw', embed_structure=True)
    recipe = replace(recipe, spatial_only=True, compute_dtype=np.dtype(dtype).name,
                     constants=(ArrayConstant.from_array(np.eye(4, dtype=dtype)),))
    backend = array_backend()
    loader = bind_prepared_loader(None, {'spatial': SourcePreparation(('raw',), recipe=recipe)},
        {'spatial': TensorSpec((3, 3), dtype)}, backend=backend)
    actual = loader.materialize('spatial', None, {'raw': backend.asarray(vertex.read_factor_block(description))})
    if backend is not np:
        actual = backend.asnumpy(actual)
    expected = raw[0, 0, 1].conj().T * np.eye(3, dtype=dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('source', (False, True))
def test_singleton_support_retains_scalar_monomial_preparation(source):
    from lqcdweave.contraction.expansion.monomial_spin import monomial_columns
    vertex, raw, terms, _ = fixture(ne=1, source=source)
    descriptor = vertex.describe_core_factors(1)[0]
    assert descriptor.mixed_structure
    recipe = vertex.describe_factor_preparation(descriptor.factors[0], 'raw', embed_structure=True)
    assert not recipe.spatial_support
    assert monomial_columns(recipe) is not None
    # A one-by-one diagonal has no off-support values to embed.
    np.testing.assert_array_equal(explicit_vertex(raw, terms, 1, source=source),
                                  explicit_vertex(raw, terms, 1, source=source, structure='general'))


@pytest.mark.parametrize('difference', ('support', 'raw_index'))
def test_shared_fused_program_keeps_support_and_current_lane_identity(difference):
    from lqcdweave.contracts.correlator.api import TensorPreparation
    backend = array_backend()
    spin = ArrayConstant.from_array(gamma(0))
    support = ((0, 1, (0, 1, 2), (0, 1, 2)),)
    base = TensorPreparation('spin_spatial', ('raw',), (spin,), spatial_support=support)
    other = replace(base, spatial_support=()) if difference == 'support' else replace(base, raw_index=1)
    if difference == 'raw_index':
        base = replace(base, raw_index=0)
    values = np.arange(1, 19 if difference == 'raw_index' else 10, dtype=float)
    values = values.reshape((2, 3, 3) if difference == 'raw_index' else (3, 3)).astype(complex)
    specs = {key: TensorSpec((4, 4, 3, 3), np.complex128) for key in ('a', 'b')}
    preparations = {key: SourcePreparation(('raw',), recipe=recipe)
                    for key, recipe in (('a', base), ('b', other))}
    loader = bind_prepared_loader(None, preparations, specs, backend=backend)
    raw = backend.asarray(values)
    for key, recipe in (('a', base), ('b', other), ('a', base)):
        actual = loader.materialize(key, None, {'raw': raw})
        spatial = values if recipe.raw_index is None else values[recipe.raw_index]
        expected = np.einsum('ab,ij->abij', gamma(0),
                            spatial * np.eye(3) if recipe.spatial_support else spatial)
        np.testing.assert_array_equal(actual.get() if backend is not np else actual, expected)


@pytest.mark.parametrize('reverse_axes', (False, True))
def test_diagonal_embedding_direct_fallback_supports_source_axis_order(reverse_axes):
    from lqcdweave.contraction.execution.structured_preparation import bind_spatial_embedding
    backend = array_backend()
    axes = (1, 0) if reverse_axes else (0, 1)
    support = ((*axes, (0, 1, 2), (0, 1, 2)),)
    raw = np.arange(9).reshape(3, 3).astype(complex) + .3j
    spin = gamma(3)
    for spatial_only in (False, True):
        shape = (3, 3) if spatial_only else (4, 4, 3, 3)
        operation = bind_spatial_embedding(support, shape, spatial_only=spatial_only)
        actual = operation(backend, backend.asarray(spin), backend.asarray(raw), None, np.dtype('complex128'))
        expected = raw * np.eye(3)
        if not spatial_only:
            expected = np.einsum('ab,ij->abij', spin, expected)
        np.testing.assert_array_equal(actual.get() if backend is not np else actual, expected)
