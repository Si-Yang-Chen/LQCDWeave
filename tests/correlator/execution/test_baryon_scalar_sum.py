"""Scalar whole cores read current values and return independent owners."""
from dataclasses import replace
import gc
from itertools import product
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref
from unittest.mock import patch

import numpy as np
import pytest

import lqcdweave
from lqcdweave.contracts.correlator.api import (
    AxisPattern, CoreComponent, TensorCoreSpec, TensorSpec, VertexKernelDescription,
)
from lqcdweave.contracts.correlator.cores import baryon_component_pattern
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contraction.execution.operations import baryon_scalar_sum


def fixture(backend, dtype, layout, batch, parity='g', row=1, source=False, seed=745, modes=(2,3,2)):
    rng = np.random.default_rng(seed)
    catalog = Path(lqcdweave.__file__).parent / 'operators/single_hadron/presets/baryon_spin_v1.json'
    rows = {item['name']: item for item in json.loads(catalog.read_text())['rows']}
    lane = 30
    physical, labels = (6, 2, 10, 7, 3, 11), (10, 11, 2, 3, 6, 7)
    sizes = {6:4, 2:4, 10:4, 7:modes[0], 3:modes[1], 11:modes[2], lane:2}
    if layout == 'permuted':
        labels = tuple(labels[index] for index in (3, 5, 0, 2, 4, 1))
    if batch is not None:
        labels = (lane, *labels)
        if batch != 'static':
            physical = (lane, *physical)
    def array(shape):
        storage = tuple(2 * size for size in shape) if layout == 'strided' else shape
        value = backend.asarray((rng.normal(size=storage) + 1j*rng.normal(size=storage)).astype(dtype))
        if layout == 'strided':
            value = value[tuple(slice(None, None, -2) for _ in shape)]
        return value
    arrays, components = {}, []
    for index, (name, coefficient) in enumerate(zip(('A_copy0','MA_copy0','MA_copy2'),
                                                  (.7+.2j, -.4j, -.3+.1j))):
        terms = rows[f'spin_G1{parity}_{name}/row{row}']['terms']
        support = tuple((*item['spin'], complex(*item['coefficient'])) for item in terms)
        if source:
            support = tuple((i,j,k,weight.conjugate()) for i,j,k,weight in support)
            coefficient = coefficient.conjugate()
        spin = np.zeros((4,4,4), dtype=dtype)
        for i,j,k,weight in support:
            spin[i,j,k] = weight
        if layout == 'strided':
            storage = backend.zeros((8,8,8), dtype=dtype)
            arrays[f'S{index}'] = storage[::-2, ::-2, ::-2]
            arrays[f'S{index}'][...] = backend.asarray(spin)
        else:
            arrays[f'S{index}'] = backend.asarray(spin)
        dynamic = batch == 'dynamic' or (batch == 'mixed' and index % 2 == 0)
        spatial = array((2,*modes) if dynamic else modes)
        if source:
            # Assignment preserves the deliberate negative strides.
            spatial[...] = backend.conj(spatial)
        arrays[f'E{index}'] = spatial
        components.append(CoreComponent((f'S{index}',f'E{index}'),coefficient,support))
    arrays['P'] = array(tuple(sizes[axis] for axis in labels))
    axes, shape = ('s0','s1','s2','v0','v1','v2'), (4,4,4,*modes)
    if batch is not None and batch != 'static':
        axes, shape = ('lane',*axes), (2,*shape)
    core = TensorCoreSpec('baryon', axes, TensorSpec(shape, dtype), tuple(components))
    action = SimpleNamespace(reused=False,equation=AxisPattern((physical,labels), () if batch is None else (lane,)),
        core=core,operands=core.inputs+('P',),tensor='result',kernel_path=(),
        kernel_description=VertexKernelDescription('baryon','ordered-join-v1'))
    return action, arrays


def reference(action, arrays, host):
    total = None
    for part in action.core.components:
        core_labels = action.equation.inputs[0]
        modes = core_labels[-3:]
        if arrays[part.inputs[1]].ndim == 4:
            modes = (core_labels[0], *modes)
        present = set(core_labels[-6:-3]).union(modes,*action.equation.inputs[1:])
        # A static component may broadcast across a lane supplied by another
        # component; sum its dense scalar before NumPy broadcasts the sum.
        output = tuple(axis for axis in action.equation.output if axis in present)
        equation = baryon_component_pattern(action.equation.inputs[0], action.equation.inputs[1:],
            output, arrays[part.inputs[1]].ndim)
        values = tuple(host(arrays[key]).astype('complex128') for key in part.inputs+('P',))
        value = part.coefficient * np.einsum(*einsum_arguments(equation, values), optimize=False)
        total = value if total is None else total+value
    return total


def backend_for(name):
    if name == 'cupy':
        if os.environ.get('LQCDWEAVE_TEST_GPU') != '1':
            pytest.skip('requires an explicitly allocated GPU')
        import cupy
        return cupy, cupy.asnumpy
    return np, np.asarray


def mathematical_preparation(action, arrays, backend, dtype):
    """Exercise the small CPU arithmetic independently of its size selector."""
    if backend.__name__ != 'numpy':
        return baryon_scalar_sum.prepare_scalar_sum(action,arrays,backend,dtype)
    parts = action.core.components
    values = tuple(arrays[key] for part in parts for key in part.inputs)+(arrays['P'],)
    supports = tuple(tuple((i,j,k) for i,j,k,_ in part.sparse_spin) for part in parts)
    recipe = baryon_scalar_sum._recipe(action.equation,
        tuple(tuple(value.shape) for value in values),supports)
    assert recipe is not None
    components = tuple((arrays[part.inputs[0]],arrays[part.inputs[1]],dtype.type(part.coefficient))
                       for part in parts)
    return baryon_scalar_sum.PreparedScalarSum(recipe,components,arrays['P'],backend,dtype)


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
@pytest.mark.parametrize('dtype', ('complex64','complex128'))
@pytest.mark.parametrize('layout', ('contiguous','strided','permuted'))
@pytest.mark.parametrize('batch', (None,'static','dynamic','mixed'))
def test_scalar_current_catalog_inputs_lane_and_independent_owner(backend_name,dtype,layout,batch):
    backend, host = backend_for(backend_name)
    # All twelve selected catalog rows, both physical endpoint conventions.
    for parity, row, source in product(('g','u'),(0,1),(False,True)):
        previous = None
        for current in range(2):
            action, arrays = fixture(backend,dtype,layout,batch,parity,row,source,seed=745+current)
            prepared = mathematical_preparation(action,arrays,backend,np.dtype(dtype))
            assert prepared is not None
            # Values, including spin, may change inside the same declared support.
            # Metadata still carries the original catalog coefficients.
            for value in arrays.values():
                value *= np.dtype(dtype).type(.8+.13j)
            expected = reference(action,arrays,host)
            snapshots = {key:host(value).copy() for key,value in arrays.items()}
            result = prepared.compute()
            actual = host(result).copy()
            prepared.close()
            prepared.close()
            assert prepared.components == () and prepared.leg is None
            assert prepared.metadata is prepared.coefficients is None
            with pytest.raises(RuntimeError,match='released'):
                prepared.compute()
            assert result.dtype == np.dtype(dtype)
            assert result.shape == (() if batch is None else (2,))
            if dtype == 'complex128':
                np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
            else:
                # Existing isolated c64 baryon unit scope, not a full LL3 tolerance.
                np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-6*float(np.max(np.abs(expected))))
            for key,value in snapshots.items():
                np.testing.assert_array_equal(host(arrays[key]),value)
            if previous is not None:
                assert not np.array_equal(actual,previous)
            previous = actual
            for value in arrays.values():
                value[...] = 0
            np.testing.assert_array_equal(host(result),actual)


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
@pytest.mark.parametrize('dtype', ('complex64','complex128'))
def test_compute_reads_spin_again_and_zero_leg(backend_name,dtype):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,dtype,'strided','mixed')
    prepared = mathematical_preparation(action,arrays,backend,np.dtype(dtype))
    first = prepared.compute()
    saved = host(first).copy()
    for part in action.core.components:
        arrays[part.inputs[0]] *= np.dtype(dtype).type(-.3+.4j)
    second = prepared.compute()
    expected = reference(action,arrays,host)
    tolerance = 3e-6 if dtype == 'complex64' else 2e-11
    np.testing.assert_allclose(host(second),expected,rtol=tolerance,
                               atol=tolerance*max(float(np.max(np.abs(expected))),1))
    assert not np.array_equal(saved,host(second))
    np.testing.assert_array_equal(host(first),saved)
    arrays['P'][...] = 0
    np.testing.assert_array_equal(host(prepared.compute()),np.zeros(2))
    prepared.close()


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
def test_close_releases_all_current_input_owners(backend_name):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,'complex128','strided','mixed')
    refs = [weakref.ref(value) for value in arrays.values()]
    prepared = mathematical_preparation(action,arrays,backend,np.dtype('complex128'))
    value = prepared.compute()
    prepared.close()
    del arrays
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert np.isfinite(host(value)).all()


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
def test_no_path_discovery_and_foreign_gpu_inputs_keep_fallback(backend_name):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,'complex128','permuted','mixed')
    expected = reference(action,arrays,host)
    with patch('opt_einsum.paths.get_path_fn',side_effect=AssertionError('path search')):
        prepared = mathematical_preparation(action,arrays,backend,np.dtype('complex128'))
        try:
            value = prepared.compute()
        finally:
            prepared.close()
    np.testing.assert_allclose(host(value),expected,rtol=2e-11,atol=2e-12)
    if backend_name == 'cupy':
        arrays['E0'] = host(arrays['E0'])
        assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,backend,np.dtype('complex128')) is None


def test_unknown_legacy_dtype_and_non_scalar_ports_keep_old_route():
    action, arrays = fixture(np,'complex128','contiguous',None,modes=(8,8,8))
    original = action.kernel_description
    action.kernel_description = None
    assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,np,np.dtype('complex128')) is None
    action.kernel_description = original
    assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,np,np.dtype('complex64')) is None
    missing = replace(action.core,components=tuple(replace(part,sparse_spin=None) for part in action.core.components))
    action.core = missing
    assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,np,np.dtype('complex128')) is None
    action, arrays = fixture(np,'complex128','contiguous',None,modes=(8,8,8))
    action.equation = AxisPattern(action.equation.inputs, (2,))
    assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,np,np.dtype('complex128')) is None
    action, arrays = fixture(np,'complex128','contiguous',None,modes=(8,8,8))
    arrays['E0'] = arrays['E0'].astype('complex64')
    assert baryon_scalar_sum.prepare_scalar_sum(action,arrays,np,np.dtype('complex128')) is None


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
@pytest.mark.parametrize('dtype', ('complex64','complex128'))
def test_dynamic_phi_broadcasts_static_leg_and_preserves_width_one(backend_name,dtype):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,dtype,'permuted','mixed')
    arrays['P'] = arrays['P'][0]
    action.equation = AxisPattern((action.equation.inputs[0],action.equation.inputs[1][1:]),action.equation.output)
    for width in (2,1):
        if width == 1:
            for key in ('E0','E2'):
                arrays[key] = arrays[key][:1]
            action.core = replace(action.core,spec=replace(action.core.spec,
                shape=(1,*action.core.spec.shape[1:])))
        expected = reference(action,arrays,host)
        prepared = mathematical_preparation(action,arrays,backend,np.dtype(dtype))
        assert prepared is not None
        try:
            value = prepared.compute()
        finally:
            prepared.close()
        assert value.shape == (width,)
        tolerance = 3e-6 if dtype == 'complex64' else 2e-11
        np.testing.assert_allclose(host(value),expected,rtol=tolerance,
                                   atol=tolerance*max(float(np.max(np.abs(expected))),1))


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
@pytest.mark.parametrize('parts', (1,2,3))
def test_component_count_has_no_previous_numeric_binding(backend_name,parts):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,'complex128','strided','mixed')
    action.core = replace(action.core,components=action.core.components[:parts])
    action.operands = action.core.inputs+('P',)
    expected = reference(action,arrays,host)
    prepared = mathematical_preparation(action,arrays,backend,np.dtype('complex128'))
    assert prepared is not None
    try:
        value = prepared.compute()
    finally:
        prepared.close()
    np.testing.assert_allclose(host(value),expected,rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('backend_name', ('numpy','cupy'))
@pytest.mark.parametrize('dtype', ('complex64','complex128'))
@pytest.mark.parametrize('modes', ((1,1,1),(2,3,2),(7,8,8),(8,8,8),(8,9,10)))
def test_cpu_size_selector_uses_actual_permuted_leg_modes(backend_name,dtype,modes):
    backend, host = backend_for(backend_name)
    action, arrays = fixture(backend,dtype,'permuted','mixed',modes=modes)
    prepared = baryon_scalar_sum.prepare_scalar_sum(action,arrays,backend,np.dtype(dtype))
    if backend_name == 'numpy' and min(modes) < 8:
        assert prepared is None
        return
    assert prepared is not None
    try:
        actual = host(prepared.compute())
    finally:
        prepared.close()
    expected = reference(action,arrays,host)
    tolerance = 3e-6 if dtype == 'complex64' else 2e-11
    np.testing.assert_allclose(actual,expected,rtol=tolerance,
                               atol=tolerance*max(float(np.max(np.abs(expected))),1))
