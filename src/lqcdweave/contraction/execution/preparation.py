"""Apply saved input preparations to raw blocks supplied by execution."""
from dataclasses import dataclass, field
from collections import OrderedDict
from operator import itemgetter
from types import MappingProxyType

import numpy as np
from lqcdweave.contracts.correlator.api import (
    CfgTemplate, SourcePreparation, TensorPreparation, TensorSpec, ArrayConstant,
)
from .diagnostics import debug_enabled
from .preparation_kernel import FusedSpinPreparation, supports_fused_preparation
from .binding_cache import bind_cached, binding_cache_active


@dataclass(frozen=True)
class _BoundPreparation:
    operation: object
    dependencies: tuple
    fused: object = None


@dataclass(frozen=True)
class _TemplatePreparationBinding:
    """One loader-local binding of a frozen template's local preparation roles."""
    template: object
    descriptor: object
    roles: object
    scope: object
    repeated_inputs: object = field(default_factory=dict, compare=False, repr=False)
    input_layouts: object = field(default_factory=OrderedDict, compare=False, repr=False)
    input_getters: object = field(default_factory=dict, compare=False, repr=False)


class _NumericNodeBinding(dict):
    """X's current IDs constructed from an accepted C IntArrayRecord.

    This marker never proves anything about external mappings. Its creator
    converts every node ID to int before delivery, as in the common cfg path.
    """
    __slots__ = ()


@dataclass(frozen=True)
class _InstancePreparationBinding:
    """Current-instance raw identities, never numerical owners or pointers."""
    roles: object
    raw_keys: tuple

    def materialize(self, local_key, raw_blocks):
        role = self.roles.get(local_key)
        if role is None:
            return None
        program, select, single, expand = role
        keys = select(self.raw_keys)
        inputs = (raw_blocks[keys],) if single else itemgetter(*keys)(raw_blocks)
        if expand is not None:
            inputs = expand(inputs)
        return program.materialize(raw_blocks, inputs=inputs)


_UNBOUND_INPUT_LAYOUT = object()


def _repeated_input_layout(keys):
    """Bind repeated explicit roles; never infer identity from pooled programs."""
    if len(keys) < 8:
        return None
    unique = tuple(dict.fromkeys(keys))
    if len(keys) < 2 * len(unique):
        return None
    positions = {key: index for index, key in enumerate(unique)}
    return unique, itemgetter(*(positions[key] for key in keys))


def _bind_repeated_inputs(roles):
    return MappingProxyType({key: layout for key, role in roles.items()
        if role is not None and (layout := _repeated_input_layout(role[1])) is not None})


def _convert_precision(value, dtype, *, copy=False):
    """Convert one required input block; matching precision needs no copy."""
    return value.astype(dtype, copy=copy)


def _store_prepared(value, destination, spec, compute_dtype=None):
    if compute_dtype is not None:
        value = _convert_precision(value, compute_dtype)
    if destination is None:
        return value
    if debug_enabled() and spec is not None and (
            tuple(value.shape) != spec.shape or value.dtype != spec.dtype):
        raise ValueError("materialized component differs from its physical metadata")
    destination[...] = value
    return destination


def _raw_value(blocks, recipe):
    value = blocks[recipe.inputs[0]]
    return value if recipe.raw_index is None else value[recipe.raw_index]


def _prepare_identity(backend, recipe, blocks, destination, spec):
    value = _raw_value(blocks, recipe)
    if recipe.compute_dtype is not None and (recipe.conjugate or recipe.index_permutations):
        value = _convert_precision(value, recipe.compute_dtype)
    if recipe.index_permutations:
        for axis, order in recipe.index_permutations:
            value = backend.take(value, order, axis=axis)
        if recipe.conjugate:
            value = value.conj()
        return _store_prepared(value, destination, spec, recipe.compute_dtype)
    if recipe.conjugate:
        if destination is None:
            destination = backend.empty(spec.shape, dtype=recipe.compute_dtype or spec.dtype)
        backend.conjugate(value, out=destination)
        return destination
    if recipe.compute_dtype is not None:
        if destination is None:
            return _convert_precision(value, recipe.compute_dtype, copy=True)
        destination[...] = value
        return destination
    return _store_prepared(value, destination, spec)


def _bind_leaf_operation(backend, recipe, shape, dtype):
    """Compile a recipe without capturing a reader, input or device constant."""
    spec = TensorSpec(shape, dtype)
    if recipe.kind == 'identity':
        def operation(loader, key, source, blocks, destination=None):
            return _prepare_identity(backend, recipe, blocks, destination, spec)
    elif recipe.kind == 'constant':
        def operation(loader, key, source, blocks, destination=None):
            return _store_prepared(loader._constants(key, recipe)[0], destination, spec)
    else:
        fallback = _bind_spin_operation(backend, recipe, spec)

        def operation(loader, key, source, blocks, destination=None):
            runtime = loader._operation_runtime.get(key)
            if runtime is None:
                constants = loader._constants(key, recipe)
                fused = FusedSpinPreparation.spin(backend, recipe, constants, spec)
                runtime = constants, loader._share_fused(fused)
                loader._operation_runtime[key] = runtime
            constants, fused_run = runtime
            if fused_run is not None:
                result = fused_run(blocks, destination)
                if result is not None:
                    return result
            return fallback(constants, blocks, destination)
    return operation


def _bind_spin_operation(xp, recipe, spec):
    """Bind backend/recipe control once; receive this run's constants at use."""
    conjugate, permutation = recipe.conjugate, recipe.permutation
    multiply = getattr(xp, "multiply", None)
    result_type = getattr(xp, "result_type", np.result_type)
    if recipe.spatial_support:
        from .structured_preparation import bind_spatial_embedding
        embedding = bind_spatial_embedding(recipe.spatial_support, spec.shape,
                                           spatial_only=recipe.spatial_only)
    else:
        embedding = None
    def operation(constants, blocks, destination=None):
        spin, value = constants[0], _raw_value(blocks, recipe)
        if recipe.compute_dtype is not None:
            value = _convert_precision(value, recipe.compute_dtype)
        if len(constants) == 4:
            output = xp.zeros_like(value, dtype=recipe.compute_dtype or complex)
            for coefficient, left, right in zip(constants[3], constants[1], constants[2], strict=True):
                output += coefficient * left.conj()[:, None] * value * right[None, :]
            value = output
        if conjugate:
            value = value.conj()
        if permutation:
            value = value.transpose(permutation)
        if embedding is not None:
            return embedding(xp, spin, value, destination, spec.dtype)
        if recipe.spatial_structure == "diagonal":
            if value.ndim == 2:
                value = xp.diagonal(value)
            else:
                point_shape = value.shape[:2]
                size = point_shape[0] * point_shape[1]
                value = xp.diagonal(value.reshape(size, size)).reshape(point_shape)
        if recipe.spatial_only:
            # Keep the original spin multiplication's promotion and every
            # subsequent stored cast. The unit spin phase remains a small
            # explicit factor in E instead of occupying spatial entries.
            value = value.astype(result_type(spin, value), copy=False)
            if destination is None:
                return xp.array(value, copy=True)
            destination[...] = value
            return destination
        if recipe.spin_columns:
            spin = xp.stack(tuple(spin[row, column] for row, column in
                                  enumerate(recipe.spin_columns)))
            outer_spin = spin.reshape((4,) + (1,) * value.ndim)
            dtype = result_type(spin, value)
            return xp.multiply(outer_spin, value, out=destination, dtype=dtype, casting='unsafe')
        if multiply is None:
            result = xp.einsum("ab,...->ab...", spin, value, optimize=False)
            return _store_prepared(result, destination, spec)
        dtype = result_type(spin, value)
        if destination is not None and debug_enabled() and spec is not None and (
                (4, 4, *value.shape) != spec.shape or dtype != spec.dtype):
            raise ValueError("materialized component differs from its physical metadata")
        outer_spin = spin.reshape((4, 4) + (1,) * value.ndim)
        return multiply(outer_spin, value, out=destination,
                        dtype=dtype, casting="unsafe")
    return operation


class PreparedSourceLoader:
    """Compose a physical reader capability with pure preparation records."""

    def __init__(self, physical, preparations, specs, *, backend=np, constant_values=None):
        self.physical = physical
        self.preparations, self.specs, self.backend = preparations, specs, backend
        self.constants = {}
        self._device_constants = None
        # Equal immutable recipe constants share one backend allocation even
        # when their prepared result receives another residency ID.
        self._constant_pool = {}
        # At most one callable per preparation in this execution's table. No
        # callable captures raw blocks, materialized outputs, or another backend.
        self._bound_preparations = {}
        self._shared_operations = {}
        self._operation_runtime = {}
        self._fused_programs = OrderedDict()
        self._fused_bindings = {}
        self._fused_structures = OrderedDict()
        # Template identities are held strongly while cached so Python cannot
        # recycle an id into a different immutable description. This cache is
        # scoped to this prepared loader/backend and contains programs only.
        self._template_preparation_bindings = OrderedDict()
        # Frozen global recipes need only their raw input identity at replay.
        # Built on first template bind, scoped to this loader, and never arrays.
        self._template_raw_inputs = None
        self._template_numeric_raw_inputs = None
        # A token distinguishes backend/device loader scopes without making a
        # cached binding retain its owning PreparedSourceLoader.
        self._template_preparation_scope = object()
        self._structural_fusion = supports_fused_preparation(backend)
        self._preparation_records = dict(preparations)
        self._frozen = isinstance(preparations, MappingProxyType) and isinstance(specs, MappingProxyType)
        for key, values in (constant_values or {}).items():
            prep = preparations.get(key)
            recipe = None if prep is None else prep.recipe
            if recipe is not None:
                # A supplied cache may be mutable or belong to an earlier
                # execution scope. Its immutable saved records are authoritative.
                cached = values if len(recipe.constants) == len(values) else (None,) * len(recipe.constants)
                self.constants[key] = tuple(self._constant(item, value)
                    for item, value in zip(recipe.constants, cached, strict=True))
            else:
                self.constants[key] = tuple(backend.asarray(value) for value in values)

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def with_backend(self, backend):
        """Rebind the same physical reader and recipes to an independent backend.

        Constants are rebuilt from the saved recipes, so a cfg's device cache
        cannot retain arrays owned by another backend or execution scope.
        """
        if backend is self.backend:
            return self
        if any(prep.recipe is None for prep in self.preparations.values()):
            raise ValueError('device preparation requires saved recipes')
        return bind_prepared_loader(self, self.preparations, self.specs, backend=backend)

    def read_block(self, key, block):
        return self.physical.read_block(key, block)

    @property
    def input_readers(self):
        return self.physical.input_readers

    def close_block(self, key, value):
        close = getattr(self.physical, 'close_block', None)
        if close is not None:
            return close(key, value)

    def can_prefetch_block(self, key, block):
        capable = getattr(self.physical, 'can_prefetch_block', None)
        return bool(capable is not None and capable(key, block))

    def prefetch_host_bytes(self, key, block):
        return self.physical.prefetch_host_bytes(key, block)

    def prefetch_block(self, key, block):
        return self.physical.prefetch_block(key, block)

    def finish_prefetch(self, key, block, value):
        return self.physical.finish_prefetch(key, block, value)

    def close_prefetch(self, key, value):
        return self.physical.close_prefetch(key, value)

    def _constants(self, key, recipe):
        if key not in self.constants:
            self.constants[key] = tuple(self._constant(item) for item in recipe.constants)
        return self.constants[key]

    def _constant(self, item, value=None):
        if item not in self._constant_pool:
            create = lambda: self._create_constant(item, value)
            self._constant_pool[item] = (create() if self._device_constants is None else
                self._device_constants.recipe_constant(item, create))
        return self._constant_pool[item]

    def _create_constant(self, item, value):
        # Supplied caches never override the immutable saved recipe.
        reusable = (isinstance(value, np.ndarray) and not value.flags.writeable
                    and tuple(value.shape) == item.shape and value.dtype.str == item.dtype
                    and value.tobytes(order="C").hex() == item.data)
        return self.backend.asarray(value if reusable else item.array())

    def materializes_owned(self, key):
        """Whether this preparation returns a fresh, independent result.

        X may adopt these outputs instead of snapshotting them again. Borrowed
        identities, shared constants, and opaque callbacks still need snapshots.
        Consult the current record, including after a mutable table is changed.
        """
        recipe = self.preparations[key].recipe
        return recipe is not None and (recipe.kind in {"spin_spatial", "stack", "linear_combination"}
            or recipe.kind == 'identity' and (recipe.compute_dtype is not None or recipe.conjugate or recipe.index_permutations))

    def materialize(self, key, source, raw_blocks):
        if self._frozen and self._structural_fusion and not debug_enabled():
            binding = self._bind_fused_structure(key, set())
            if binding is not None:
                program, inputs = binding
                value = program.materialize(raw_blocks,
                    inputs=tuple(raw_blocks[block] for block in inputs))
                if value is not None:
                    return value
        if binding_cache_active():
            operation = self._shared_operation(key)
            if operation is not None:
                return operation(self, key, source, raw_blocks)
        bound = self._cached_preparation(key)
        if bound is None:
            bound = self._bind_preparation(key, set())
        return bound.operation(source, raw_blocks)

    def _shared_operation(self, key):
        """Reuse pure native leaf operations; owners/constants remain run-local."""
        prep, spec = self.preparations[key], self.specs.get(key)
        previous = self._shared_operations.get(key)
        if previous is not None:
            if previous[0] is prep and previous[1] is spec:
                return previous[2]
            # Mutable tables can change between uses in a single run too.
            self.constants.clear()
            self._constant_pool.clear()
            self._operation_runtime.clear()
            self._shared_operations.clear()
        recipe = prep.recipe
        if (type(prep) is not SourcePreparation or type(recipe) is not TensorPreparation
                or type(spec) is not TensorSpec
                or any(type(item) is not ArrayConstant for item in recipe.constants)
                or recipe.kind in ('stack', 'linear_combination')):
            return None
        # Capture only numerical geometry, never a spec's extension metadata.
        shape, dtype = spec.shape, spec.dtype
        operation = bind_cached('preparation', prep, self.backend,
            lambda: _bind_leaf_operation(self.backend, recipe, shape, dtype), lambda: True,
            slot=key, signature=(shape, dtype.str))
        self._shared_operations[key] = prep, spec, operation
        return operation

    def _bind_fused_structure(self, key, visiting):
        """Bind immutable recipes to a shared program and current raw IDs.

        Structure keys contain numerical semantics; residency IDs only address
        this call's inputs. Existing fusion owns indexing, casts and fallback.
        """
        if key in self._fused_bindings:
            return self._fused_bindings[key]
        if key in visiting:
            raise ValueError("cyclic tensor preparation")
        recipe, spec = self.preparations[key].recipe, self.specs.get(key)
        binding = None
        if recipe is not None and spec is not None and recipe.raw_index is None:
            if recipe.kind == "spin_spatial":
                signature = (recipe.kind, spec.shape, spec.dtype, recipe.constants,
                             recipe.conjugate, recipe.permutation, recipe.configuration,
                             recipe.spatial_structure, recipe.compute_dtype, recipe.spin_columns, recipe.spatial_only,
                             recipe.spatial_support)
                program = self._fused_structures.get(signature)
                if program is None:
                    constants = self._constants(key, recipe)
                    program = FusedSpinPreparation.spin(
                        self.backend, recipe, constants, spec, prefer_raw=self._frozen)
                inputs = recipe.inputs
            elif recipe.kind == "stack":
                visiting.add(key)
                try:
                    members = tuple(self._bind_fused_structure(member, visiting)
                                    for member in recipe.inputs)
                finally:
                    visiting.remove(key)
                if any(member is None for member in members):
                    self._fused_bindings[key] = None
                    return None
                programs = tuple(member[0] for member in members)
                signature = (recipe.kind, spec.shape, spec.dtype, programs)
                program = self._fused_structures.get(signature)
                if program is None:
                    program = FusedSpinPreparation.stack(
                        self.backend, programs, spec,
                        prefer_raw=all(program.prefer_raw for program in programs))
                inputs = tuple(block for member in members for block in member[1])
            else:
                self._fused_bindings[key] = None
                return None
            if program is not None:
                self._fused_structures[signature] = program
                self._fused_structures.move_to_end(signature)
                if len(self._fused_structures) > 512:
                    self._fused_structures.popitem(last=False)
                binding = program, inputs
        self._fused_bindings[key] = binding
        return binding

    def bind_template_preparations(self, template):
        """Bind complete frozen template recipes once for this loader scope.

        The returned roles address the template's local preparation tree. They
        contain fused programs and local leaf keys, never instance IDs or
        numerical inputs. Mutable, incomplete, CPU, and diagnostic execution
        keeps using the ordinary preparation binder.
        """
        if not self._frozen or not self._structural_fusion or debug_enabled():
            return None
        identity = id(template)
        cached = self._template_preparation_bindings.get(identity)
        native = isinstance(template, CfgTemplate)
        if native and cached is not None and cached.template is template:
            # Native CfgTemplate freezes its keys, closure, and specs in
            # __post_init__; avoid walking those maps for every instance.
            self._template_preparation_bindings.move_to_end(identity)
            return cached
        closure = getattr(template, 'preparation_closure', None)
        specs = getattr(template, 'preparation_specs', None)
        if not isinstance(closure, MappingProxyType) or not isinstance(specs, MappingProxyType):
            return None
        # Native templates are frozen by C; lightweight protocol objects retain
        # a full descriptor check because their identity alone proves nothing.
        descriptor = (None if native else
                      (tuple(getattr(template, 'keys', ())),
                       tuple(closure.items()), tuple(specs.items())))
        if (not native and cached is not None and cached.template is template
                and cached.descriptor == descriptor):
            self._template_preparation_bindings.move_to_end(identity)
            return cached

        roles = {}
        visiting = set()
        for key in closure:
            self._bind_template_fused_structure(key, visiting, closure, specs, roles)
        if self._template_raw_inputs is None:
            raw_inputs = {}
            for key, preparation in self.preparations.items():
                recipe = preparation.recipe
                raw_inputs[key] = (recipe.inputs[0]
                    if recipe is not None and recipe.kind == 'spin_spatial' else None)
            self._template_raw_inputs = MappingProxyType(raw_inputs)
            # Keep the original text table for extension/diagnostic mappings.
            # Native cfg node IDs address canonical numeric names directly.
            numeric = {}
            for key, raw in raw_inputs.items():
                if type(key) is str and key.isdecimal() and raw is not None:
                    node = int(key)
                    if str(node) == key:
                        numeric[node] = raw
            self._template_numeric_raw_inputs = MappingProxyType(numeric)
        repeated_inputs = _bind_repeated_inputs(roles)
        input_getters = {}
        for key, role in roles.items():
            if role is not None and len(role[1]) >= 4:
                selected = repeated_inputs.get(key)
                leaves = selected[0] if selected is not None else role[1]
                input_getters[key] = itemgetter(*leaves), len(leaves) == 1
        binding = _TemplatePreparationBinding(
            template, descriptor, MappingProxyType(roles), self._template_preparation_scope,
            repeated_inputs, input_getters=MappingProxyType(input_getters))
        self._template_preparation_bindings[identity] = binding
        self._template_preparation_bindings.move_to_end(identity)
        if len(self._template_preparation_bindings) > 16:
            self._template_preparation_bindings.popitem(last=False)
        return binding

    def _bind_template_fused_structure(self, key, visiting, closure, specs, roles):
        if key in roles:
            return roles[key]
        if key in visiting:
            raise ValueError('cyclic tensor preparation')
        # Index directly so a missing member has the same first-bind failure
        # boundary as the existing recursive preparation binder.
        prep = closure[key]
        recipe, spec = prep.recipe, specs.get(key)
        binding = None
        if recipe is not None and spec is not None and recipe.raw_index is None:
            if recipe.kind == 'spin_spatial':
                signature = (recipe.kind, spec.shape, spec.dtype, recipe.constants,
                             recipe.conjugate, recipe.permutation, recipe.configuration,
                             recipe.spatial_structure, recipe.compute_dtype, recipe.spin_columns, recipe.spatial_only,
                             recipe.spatial_support)
                program = self._fused_structures.get(signature)
                if program is None:
                    constants = tuple(self._constant(item) for item in recipe.constants)
                    program = FusedSpinPreparation.spin(
                        self.backend, recipe, constants, spec, prefer_raw=self._frozen)
                leaf_roles = (key,)
            elif recipe.kind == 'stack':
                visiting.add(key)
                try:
                    members = tuple(self._bind_template_fused_structure(
                        member, visiting, closure, specs, roles)
                        for member in recipe.inputs)
                finally:
                    visiting.remove(key)
                if any(member is None for member in members):
                    roles[key] = None
                    return None
                programs = tuple(member[0] for member in members)
                signature = (recipe.kind, spec.shape, spec.dtype, programs)
                program = self._fused_structures.get(signature)
                if program is None:
                    program = FusedSpinPreparation.stack(
                        self.backend, programs, spec,
                        prefer_raw=all(item.prefer_raw for item in programs))
                leaf_roles = tuple(role for member in members for role in member[1])
            else:
                roles[key] = None
                return None
            if program is not None:
                self._fused_structures[signature] = program
                self._fused_structures.move_to_end(signature)
                if len(self._fused_structures) > 512:
                    self._fused_structures.popitem(last=False)
                binding = program, leaf_roles
        roles[key] = binding
        return binding

    def materialize_template_preparation(self, binding, local_key, node_binding, raw_blocks):
        """Run one prebound local role using this instance's current raw arrays."""
        if binding is None:
            return None
        if debug_enabled():
            return None
        if binding.scope is not self._template_preparation_scope:
            raise ValueError('template preparation binding belongs to another loader scope')
        role = binding.roles.get(local_key)
        if role is None:
            return None
        program, leaf_roles = role
        if (len(leaf_roles) == 1 and type(node_binding) is _NumericNodeBinding
                and self._template_numeric_raw_inputs):
            node = node_binding[leaf_roles[0]]
            try:
                raw_key = self._template_numeric_raw_inputs[node]
            except KeyError:
                # Preserve the text-ID diagnostic at first consumption.
                pass
            else:
                return program.materialize(raw_blocks, inputs=(raw_blocks[raw_key],))
        repeated = binding.repeated_inputs.get(local_key)
        gather = binding.input_getters.get(local_key)
        if (gather is not None and type(node_binding) is _NumericNodeBinding
                and self._template_numeric_raw_inputs):
            select, single = gather
            nodes = select(node_binding)
            try:
                raw_keys = ((self._template_numeric_raw_inputs[nodes],) if single else
                            itemgetter(*nodes)(self._template_numeric_raw_inputs))
            except KeyError:
                # Preserve the original first-consumption boundary and text
                # key diagnostics for incomplete/non-spin native bindings.
                pass
            else:
                values = ((raw_blocks[raw_keys[0]],) if single else
                          itemgetter(*raw_keys)(raw_blocks))
                return program.materialize(raw_blocks,
                    inputs=repeated[1](values) if repeated is not None else values)
        if repeated is not None:
            leaf_roles, expand = repeated
        inputs = []
        for leaf_key in leaf_roles:
            raw_key = self._template_raw_inputs[str(node_binding[leaf_key])]
            if raw_key is None:
                return None
            # C's spin_spatial contract already guarantees exactly one raw input.
            inputs.append(raw_blocks[raw_key])
        values = tuple(inputs)
        return program.materialize(raw_blocks, inputs=expand(values) if repeated is not None else values)

    def bind_instance_preparations(self, binding, node_binding, preparation_keys):
        """Resolve repeated leaf identities once for an actually used variant.

        Low reuse keeps the existing per-role lookup. A failed eager lookup
        also keeps that path, so an unused/incomplete binding cannot move the
        first-consumption error boundary. Neither cache stores current IDs.
        """
        if binding is None or debug_enabled() or not self._frozen or not self._structural_fusion:
            return None
        if binding.scope is not self._template_preparation_scope:
            raise ValueError('template preparation binding belongs to another loader scope')
        keys = tuple(preparation_keys)
        layout = binding.input_layouts.get(keys, _UNBOUND_INPUT_LAYOUT)
        if layout is _UNBOUND_INPUT_LAYOUT:
            roles = {key: binding.roles[key] for key in keys
                     if binding.roles.get(key) is not None}
            references = sum(len(role[1]) for role in roles.values())
            leaf_keys = tuple(dict.fromkeys(leaf for _, leaves in roles.values() for leaf in leaves))
            layout = None
            if references >= 64 and references >= 4 * len(leaf_keys):
                positions = {leaf: index for index, leaf in enumerate(leaf_keys)}
                indexed_roles = {}
                for key, (program, leaves) in roles.items():
                    indices = tuple(positions[leaf] for leaf in leaves)
                    repeated = _repeated_input_layout(indices)
                    indices, expand = (indices, None) if repeated is None else repeated
                    indexed_roles[key] = program, itemgetter(*indices), len(indices) == 1, expand
                layout = leaf_keys, MappingProxyType(indexed_roles)
            binding.input_layouts[keys] = layout
            if len(binding.input_layouts) > 8:
                binding.input_layouts.popitem(last=False)
        else:
            binding.input_layouts.move_to_end(keys)
        if layout is None:
            return None
        leaf_keys, roles = layout
        try:
            raw_keys = tuple(self._template_raw_inputs[str(node_binding[key])] for key in leaf_keys)
        except KeyError:
            return None
        if any(key is None for key in raw_keys):
            return None
        return _InstancePreparationBinding(roles, raw_keys)

    def _cached_preparation(self, key):
        bound = self._bound_preparations.get(key)
        if self._frozen and not debug_enabled():
            return bound
        changed = self.preparations[key] is not self._preparation_records.get(key)
        if changed or (bound is not None and any(
                self.preparations.get(member) is not prep
                or self.specs.get(member) is not spec
                for member, prep, spec in bound.dependencies)):
            # Fixed plans normally never enter this branch. A caller replacing
            # a record must not reuse its old constants or any dependent stack.
            self._bound_preparations.clear()
            self._shared_operations.clear()
            self._operation_runtime.clear()
            self._fused_programs.clear()
            self.constants.clear()
            self._constant_pool.clear()
            self._preparation_records = dict(self.preparations)
            bound = None
        return bound

    def _bind_preparation(self, key, visiting):
        if key in visiting:
            raise ValueError("cyclic tensor preparation")
        cached = self._cached_preparation(key)
        if cached is not None:
            return cached
        prep, spec = self.preparations[key], self.specs.get(key)
        recipe = prep.recipe
        dependencies = {key: (key, prep, spec)}
        fused = None
        if recipe is None:
            materialize = self.physical.materialize

            def operation(source, blocks, destination=None):
                return _store_prepared(materialize(key, source, blocks), destination, spec)
        elif recipe.kind == "constant":
            constant = self._constants(key, recipe)[0]

            def operation(source, blocks, destination=None):
                return _store_prepared(constant, destination, spec)
        elif recipe.kind == "identity":
            def operation(source, blocks, destination=None):
                return _prepare_identity(self.backend, recipe, blocks, destination, spec)
        elif recipe.kind == "linear_combination":
            from .linear_combination import bind_linear_combination
            visiting.add(key)
            try:
                members = tuple(self._bind_preparation(member, visiting)
                                for member in recipe.inputs)
            finally:
                visiting.remove(key)
            for member in members:
                dependencies.update((item[0], item) for item in member.dependencies)
            operation = bind_linear_combination(
                self.backend, tuple(member.operation for member in members),
                recipe.constants[0].array(), spec)
        elif recipe.kind == "stack":
            visiting.add(key)
            try:
                members = tuple(self._bind_preparation(member, visiting)
                                for member in recipe.inputs)
            finally:
                visiting.remove(key)
            for member in members:
                dependencies.update((item[0], item) for item in member.dependencies)
            operations = tuple(member.operation for member in members)
            member_specs = tuple(self.specs[member] for member in recipe.inputs)
            empty = self.backend.empty
            scalar_members = len(spec.shape) == 1
            fused = FusedSpinPreparation.stack(
                self.backend, tuple(member.fused for member in members), spec)
            fused_run = self._share_fused(fused)

            def operation(source, blocks, destination=None):
                if fused is not None:
                    result = fused_run(blocks, destination)
                    if result is not None:
                        return result
                if debug_enabled() and any(item.shape != spec.shape[1:] for item in member_specs):
                    raise ValueError("materialized component differs from its physical metadata")
                # Reuse a parent stack's slice only at this stack's own dtype.
                # Otherwise preserve the intermediate rounding before casting.
                direct = destination is not None and destination.dtype == spec.dtype
                result = destination if direct else empty(spec.shape, dtype=spec.dtype)
                for index, member in enumerate(operations):
                    target = result[index:index + 1].reshape(()) if scalar_members else result[index]
                    member(None, blocks, target)
                return result if direct else _store_prepared(result, destination, spec)
        elif recipe.kind == "spin_spatial":
            constants = self._constants(key, recipe)
            fallback = self._bind_spin(recipe, constants, spec)
            fused = FusedSpinPreparation.spin(self.backend, recipe, constants, spec)
            fused_run = self._share_fused(fused)

            def operation(source, blocks, destination=None):
                if fused is not None:
                    result = fused_run(blocks, destination)
                    if result is not None:
                        return result
                return fallback(source, blocks, destination)
        else:
            raise ValueError("unknown tensor preparation kind")
        bound = _BoundPreparation(operation, tuple(dependencies.values()), fused)
        self._bound_preparations[key] = bound
        return bound

    def _share_fused(self, fused):
        if fused is None:
            return None
        # Residency IDs vary across coordinates, but numerical preparation
        # programs do not. Constants are already interned within this loader.
        key = (fused.stacked, fused.spec.shape, fused.spec.dtype,
               tuple((tuple(id(v) for v in leaf.constants), leaf.spec.shape,
                      leaf.spec.dtype, leaf.casts, leaf.recipe.conjugate,
                      leaf.recipe.permutation, leaf.recipe.compute_dtype, leaf.recipe.spin_columns, leaf.recipe.spatial_only,
                      leaf.recipe.spatial_support, leaf.recipe.raw_index) for leaf in fused.leaves))
        program = self._fused_programs.setdefault(key, fused)
        self._fused_programs.move_to_end(key)
        if len(self._fused_programs) > 512:
            self._fused_programs.popitem(last=False)
        inputs = tuple(leaf.recipe.inputs[0] for leaf in fused.leaves)

        def run(blocks, destination):
            return program.materialize(blocks, destination,
                                       inputs=tuple(blocks[k] for k in inputs))
        return run

    def _bind_spin(self, recipe, constants, spec):
        run = _bind_spin_operation(self.backend, recipe, spec)

        def operation(source, blocks, destination=None):
            return run(constants, blocks, destination)

        return operation


def bind_prepared_loader(loader, preparations, specs, *, backend=np, constant_values=None):
    """Bind pure preparation tables to a physical loader for one execution.

    Rebinding a prepared loader to its current backend and the same tables is
    free. A different backend receives a fresh constant cache. Recipe-less
    inputs remain supported through the physical loader's original callback.
    """
    if isinstance(loader, PreparedSourceLoader):
        same_binding = (
            backend is loader.backend
            and preparations is loader.preparations
            and specs is loader.specs
            and constant_values is None
        )
        if same_binding:
            return loader
        physical = loader.physical
        if backend is loader.backend and constant_values is None:
            constant_values = loader.constants
        elif backend is not loader.backend:
            # Cached arrays belong to the previous backend and execution scope.
            constant_values = None
    else:
        physical = loader
        if constant_values is None:
            constant_values = getattr(physical, 'constant_values', None)

    # A logical source may be a stack of component recipes which do not
    # appear as top-level Actions. Preserve that full preparation closure.
    existing = getattr(loader, 'preparations', None)
    if existing is not None and existing is not preparations:
        merged = dict(existing) | dict(preparations)
        preparations = (MappingProxyType(merged)
                        if isinstance(existing, MappingProxyType)
                        and isinstance(preparations, MappingProxyType)
                        else merged)
    existing = getattr(loader, 'specs', None)
    if existing is not None and existing is not specs:
        merged = dict(existing) | dict(specs)
        specs = (MappingProxyType(merged)
                 if isinstance(existing, MappingProxyType)
                 and isinstance(specs, MappingProxyType)
                 else merged)

    return PreparedSourceLoader(
        physical, preparations, specs, backend=backend,
        constant_values=constant_values,
    )


__all__ = ["PreparedSourceLoader", "bind_prepared_loader"]
