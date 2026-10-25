"""Exact finite spin support and physical preparation variants for E.

This consumes actual ArrayConstant coefficients, never a gamma name. Variants
retain the original raw request, endpoint identity, profile and physical modes.
"""
from dataclasses import replace
from math import prod
import numpy as np
from lqcdweave.contracts.correlator.api import ArrayConstant


def monomial_columns(recipe):
    if (recipe is None or recipe.kind != 'spin_spatial'
            or recipe.spatial_structure != 'general' or recipe.spatial_support or recipe.raw_index is not None):
        return None
    spin = recipe.constants[0].array()
    rows, columns = np.nonzero(spin)
    if (not np.array_equal(rows, np.arange(4)) or len(set(columns)) != 4
            or not np.all(np.isfinite(spin))):
        return None
    return tuple(map(int, columns))


def unit_spin(recipe):
    # Spatial support is independent of this exact spin factor. It survives
    # the spatial-only preparation; compressed-spin lowering still declines it.
    spin_recipe = (replace(recipe, spatial_support=())
                   if recipe is not None and recipe.spatial_support else recipe)
    columns = monomial_columns(spin_recipe)
    if columns is None:
        return False
    values = recipe.constants[0].array()[np.arange(4), columns]
    return bool(np.all(np.isin(values, (1, -1, 1j, -1j))))


class MonomialPreparations:
    """One binding's pure layout variants; no raw arrays or pointer cache."""
    def __init__(self, registry):
        self.registry = registry
        self.compressed, self.gathered = {}, {}
        self.spatial, self.separated = {}, {}

    def separate(self, key):
        """A pure spatial source with the full original profile/cast boundary."""
        # Registry tensor contracts are fixed throughout this binding. Reusing
        # this exact input key avoids reconstructing and hashing its full
        # spatial recipe for each component-pair job; equal shapes are not keys.
        if key in self.separated:
            return self.separated[key]
        base = self.registry
        spec, prep, source = base.specs[key], base.preparations[key], base.sources[key]
        if not unit_spin(prep.recipe):
            raise ValueError('spin separation requires actual unit-phase monomial coefficients')
        spin = prep.recipe.constants[0].array()
        canonical = ArrayConstant.from_array(np.eye(4, dtype=spin.dtype))
        recipe = replace(prep.recipe, constants=(canonical, *prep.recipe.constants[1:]),
                         spatial_only=True)
        identity = (prep.blocks, recipe, spec.shape[2:], spec.dtype.str)
        if identity in self.spatial:
            self.separated[key] = self.spatial[identity]
            return self.separated[key]
        name = f'spin-spatial:space:{len(self.spatial)}'
        description = spec.description
        if description is not None:
            description = replace(description, axes=description.axes[2:],
                transform=(*description.transform, ('separate_unit_spin',)))
        compact = replace(spec, shape=spec.shape[2:], description=description)
        base.specs[name] = compact
        base.sources[name] = replace(source, snapshot_bytes=prod(compact.shape)*compact.dtype.itemsize)
        # Retain the original conservative preparation workspace. Profile
        # products retain their ordered arithmetic; only spin copies disappear.
        base.preparations[name] = replace(prep, recipe=recipe)
        self.spatial[identity] = self.separated[key] = name
        return name

    def compress(self, key, columns):
        identity = key, columns
        if identity in self.compressed:
            return self.compressed[identity]
        base = self.registry
        spec, prep, source = base.specs[key], base.preparations[key], base.sources[key]
        if monomial_columns(prep.recipe) != columns:
            raise ValueError('compressed spin layout differs from actual finite coefficients')
        name = f'monomial:vertex:{len(self.compressed)}'
        description = spec.description
        if description is not None:
            description = replace(description, axes=(description.axes[0], *description.axes[2:]),
                transform=(*description.transform, ('monomial_spin_columns', columns)))
        compact = replace(spec, shape=(4, *spec.shape[2:]), description=description)
        base.specs[name] = compact
        base.sources[name] = replace(source, snapshot_bytes=prod(compact.shape)*compact.dtype.itemsize)
        # Keep the established profile/dagger workspace allowance. This is a
        # conservative bound, independent of whether X selects fusion/debug.
        base.preparations[name] = replace(prep, recipe=replace(prep.recipe, spin_columns=columns))
        self.compressed[identity] = name
        return name

    def gather(self, key, permutations):
        permutations = tuple((axis, order) for axis, order in permutations if order != (0,1,2,3))
        if not permutations:
            return key
        identity = key, permutations
        if identity in self.gathered:
            return self.gathered[identity]
        base = self.registry
        spec, prep, source = base.specs[key], base.preparations[key], base.sources[key]
        if prep.recipe is None or prep.recipe.kind != 'identity':
            raise NotImplementedError('monomial spin needs described identity propagator preparation')
        name = f'monomial:propagator:{len(self.gathered)}'
        description = spec.description
        if description is not None:
            description = replace(description,
                transform=(*description.transform, ('spin_index_permutations', permutations)))
        base.specs[name] = replace(spec, description=description)
        base.sources[name] = source
        size = prod(spec.shape)*spec.dtype.itemsize
        base.preparations[name] = replace(prep,
            workspace_host_bytes=prep.workspace_host_bytes+(len(permutations)+1)*size,
            kernel_calls=prep.kernel_calls+len(permutations),
            traffic_bytes=prep.traffic_bytes+2*len(permutations)*size,
            recipe=replace(prep.recipe, index_permutations=permutations))
        self.gathered[identity] = name
        return name
