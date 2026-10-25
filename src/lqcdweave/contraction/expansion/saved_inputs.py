"""Restore described reads and preparations without symbolic construction.

Readers are injected by workflow/adapters under the saved logical reader roles.
This loader opens no files and never infers a scientific identity from a path.
"""
import numpy as np
import weakref
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import (
    SourceReadRequest, TensorAxisMetadata, canonical_axis_binding_identity,
    canonical_coordinate_identity,
)


def _selection(request):
    return {name: dict(value) if name in {'mode_indices', 'point_indices'} else value
            for name, value in request.selection}


class SavedPhysicalLoader:
    """Runtime reader handles for an already described set of input requests."""

    def __init__(self, requests, readers, *, configuration=None,
                 preparations=None, specs=None, constant_values=None, convention_cache=None):
        self.requests, self.readers = dict(requests), dict(readers)
        self.input_readers = self.readers
        self.preparations = (preparations if isinstance(preparations, MappingProxyType)
                             else dict(preparations or {}))
        self.specs = (specs if isinstance(specs, MappingProxyType)
                      else dict(specs or {}))
        self.constant_values = dict(constant_values or {})
        self.metadata = {}
        dataset_bindings, source_bindings = {}, {}
        conventions = _ConventionChecks(convention_cache, configuration)
        # All compatibility checks happen before any numeric read.
        for key, request in self.requests.items():
            if not isinstance(request, SourceReadRequest):
                raise TypeError('saved input requests require SourceReadRequest records')
            reader = self.readers[request.reader_role]
            conventions.check(request, reader)
            if configuration is not None and str(getattr(reader, 'cfg', configuration)) != str(configuration):
                raise ValueError('reader configuration differs from requested cfg')
            actual = reader.describe_block(*request.arguments, **_selection(request))
            expected = request.metadata
            if (not isinstance(actual, TensorAxisMetadata) or actual.axes != expected.axes
                    or actual.shape != expected.shape or actual.dtype != expected.dtype
                    or actual.sparse_layout != expected.sparse_layout):
                raise ValueError('saved input axes, shape, dtype or layout differ from reader')
            if actual.point_diagonal != expected.point_diagonal:
                raise ValueError('saved input point support differs from reader')
            if actual.structure != expected.structure:
                raise ValueError('saved input elemental structure differs from reader')
            if set(actual.axis_bindings) != set(expected.axis_bindings):
                raise ValueError('saved input endpoint binding knowledge differs from reader')
            for axis, (expected_id, labels) in expected.axis_bindings.items():
                actual_id, actual_labels = actual.axis_bindings[axis]
                if (tuple(map(canonical_coordinate_identity, labels))
                        != tuple(map(canonical_coordinate_identity, actual_labels))):
                    raise ValueError('saved input ordered endpoint coordinates differ from reader')
                if dataset_bindings.setdefault(expected_id, actual_id) != actual_id:
                    raise ValueError('saved input shared endpoint basis differs between readers')
            if expected.data_identity is not None:
                if (actual.data_identity is None
                        or source_bindings.setdefault(expected.data_identity, actual.data_identity)
                        != actual.data_identity):
                    raise ValueError('saved input source sharing differs between readers')
            self.metadata[key] = actual
            for alias in request.aliases:
                alias_reader = self.readers[alias.reader_role]
                conventions.check(alias, alias_reader)
                if configuration is not None and str(getattr(alias_reader, 'cfg', configuration)) != str(configuration):
                    raise ValueError('alias reader configuration differs from requested cfg')
                metadata = alias_reader.describe_block(*alias.arguments, **_selection(alias))
                if (not isinstance(metadata, TensorAxisMetadata) or metadata.axes != actual.axes
                        or metadata.shape != actual.shape or metadata.dtype != actual.dtype
                        or canonical_axis_binding_identity(metadata.axis_bindings)
                        != canonical_axis_binding_identity(actual.axis_bindings)
                        or metadata.sparse_layout != actual.sparse_layout
                        or metadata.point_diagonal != actual.point_diagonal
                        or metadata.structure != actual.structure
                        or metadata.selection_identity != actual.selection_identity
                        or metadata.data_identity != actual.data_identity
                        or (metadata.data_identity is None and alias_reader is not reader)):
                    raise ValueError('saved physical aliases no longer describe the same input')

    def read_block(self, key, block):
        request = self.requests[key]
        value = self.readers[request.reader_role].block(*request.arguments, **_selection(request))
        expected = self.metadata[key]
        if tuple(value.shape) != expected.shape or np.dtype(value.dtype) != expected.dtype:
            raise ValueError('reader block differs from saved input metadata')
        return value

    def close_block(self, key, value):
        pass

    def can_prefetch_block(self, key, block):
        request = self.requests[key]
        return (request.prefetch_host_bytes is not None
                and getattr(self.readers[request.reader_role], 'background_read_safe', False) is True
                and callable(getattr(self.readers[request.reader_role], 'read_slab_in_background', None)))

    def prefetch_host_bytes(self, key, block):
        return self.requests[key].prefetch_host_bytes

    def prefetch_block(self, key, block):
        request = self.requests[key]
        reader = self.readers[request.reader_role]
        args = request.arguments
        # Use the existing independent slab read, with exactly one lane.
        if request.family == 'propagator':
            arguments = ((args[0],), *args[1:])
        else:
            arguments = (*args[:2], (args[2],), *args[3:])
        value = reader.read_slab_in_background(*arguments, **_selection(request))
        expected = self.metadata[key]
        if tuple(value.shape) != (1,*expected.shape) or np.dtype(value.dtype) != expected.dtype:
            raise ValueError('background slab differs from saved input metadata')
        return np.array(value[0], copy=True)

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        pass


def restore_input_loader(requests, readers, preparations, specs, *, configuration=None,
                         constants=None, reader_aliases=(), debug=False, convention_cache=None):
    """Bind capabilities and saved pure preparation tables.

    A workflow may share a ``convention_cache`` dict across window bindings.
    Its bounded entries use weak reader references and immutable declarations
    only; cfg, selection, axes, shape, dtype and identity headers are always
    checked for each binding. Do not use it as a numerical result cache.
    """
    if type(debug) is not bool:
        raise TypeError('debug must be a bool')
    if any(prep.recipe is None for prep in preparations.values()):
        raise ValueError('saved inputs contain an opaque preparation callback')
    for roles in reader_aliases:
        if any(readers[role] is not readers[roles[0]] for role in roles[1:]):
            raise ValueError('unknown saved source aliases require the same reader handle')
    if debug:
        _check_preparations(requests, preparations, specs)
    for prep in preparations.values():
        if (prep.recipe.configuration is not None and configuration is not None
                and prep.recipe.configuration != str(configuration)):
            raise ValueError('saved spectral profile is bound to another configuration')
    return SavedPhysicalLoader(
        requests, readers, configuration=configuration,
        preparations=preparations, specs=specs, constant_values=constants,
        convention_cache=convention_cache,
    )


def _immutable_convention(value):
    """Prove a snapshot cannot change in place; unknown objects stay uncached."""
    return (type(value) in (str, bytes, int, float, complex, bool, type(None))
            or (isinstance(value, np.generic) and value.dtype.kind in 'biufcSU')
            or (type(value) is tuple and all(_immutable_convention(v) for v in value)))


def _same_immutable_convention(previous, current):
    """Compare against a proven immutable tree without invoking unknown equality.

    Shared immutable subtrees need no new traversal. Lists/arrays/custom
    declarations take the ordinary validation path, including if a caller
    has replaced a formerly immutable request declaration.
    """
    if previous is current:
        return True
    if type(previous) is tuple:
        return (type(current) is tuple and len(previous) == len(current)
                and all(_same_immutable_convention(old, new)
                        for old, new in zip(previous, current)))
    if type(current) is tuple:
        return False
    return _immutable_convention(current) and bool(previous == current)


class _ConventionChecks:
    # Metadata only, with no retained reader handles or numerical arrays.
    limit = 128

    def __init__(self, cache, configuration):
        if cache is not None and not isinstance(cache, dict):
            raise TypeError('convention_cache must be a dict or None')
        self.cache = {} if cache is None else cache
        self.configuration = None if configuration is None else str(configuration)

    def check(self, request, reader):
        current = tuple(getattr(reader, name, None) for name, _ in request.conventions)
        cfg = (self.configuration, str(getattr(reader, 'cfg', None)))
        key = id(reader)
        previous = self.cache.get(key)
        if (previous is not None and previous[0]() is reader
                and previous[1] == cfg
                and _same_immutable_convention(previous[2], request.conventions)
                and len(previous[3]) == len(current)
                and all(old is new for old, new in zip(previous[3], current))):
            return
        # Mutable readers are supported, but a mutable convention is never
        # trusted merely because its object identity has stayed the same.
        self.cache.pop(key, None)
        _check_conventions(request, reader, values=current)
        if not (_immutable_convention(current) and _immutable_convention(request.conventions)):
            return
        try:
            reference = weakref.ref(reader)
        except TypeError:
            return
        if len(self.cache) >= self.limit:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (reference, cfg, request.conventions, current)


def _check_conventions(request, reader, *, values=None):
    values = (tuple(getattr(reader, name, None) for name, _ in request.conventions)
              if values is None else values)
    if any(name == 'dilution_identity' and actual != value
           for (name, value), actual in zip(request.conventions, values, strict=True)):
        raise ValueError('saved input dilution source, pool or weight scheme differs from reader')
    if any((actual != value if name in {'time_selection_identity', 'propagator_time_view'}
            else not np.array_equal(actual, value))
           for (name,value),actual in zip(request.conventions, values, strict=True)):
        raise ValueError('saved input time, phase or path conventions differ from reader')


def _check_preparations(requests, preparations, specs):
    done, active = set(), set()
    for root in preparations:
        pending = [(root, False)]
        while pending:
            key, leaving = pending.pop()
            if leaving:
                active.remove(key)
                done.add(key)
                continue
            if key in done:
                continue
            if key in active:
                raise ValueError('saved preparation graph contains a cycle')
            if key not in specs or key not in preparations:
                raise ValueError('saved preparation references an unknown tensor')
            prep, spec = preparations[key], specs[key]
            recipe = prep.recipe
            if recipe.kind in {'identity','spin_spatial'}:
                if recipe.inputs != prep.blocks or any(block not in requests for block in prep.blocks):
                    raise ValueError('saved preparation dependencies differ from input requests')
            elif recipe.kind == 'constant':
                constant, = recipe.constants
                if prep.blocks or constant.shape != spec.shape or np.dtype(constant.dtype) != spec.dtype:
                    raise ValueError('saved preparation constant differs from its tensor spec')
            else:
                if any(member not in specs or member not in preparations for member in recipe.inputs):
                    raise ValueError('saved preparation references an unknown stack member')
                if recipe.kind == 'linear_combination':
                    if any(specs[member].shape != spec.shape for member in recipe.inputs):
                        raise ValueError('saved vertex members differ from its tensor shape')
                elif not spec.shape or spec.shape[0] != len(recipe.inputs) or any(
                        specs[member].shape != spec.shape[1:] for member in recipe.inputs):
                    raise ValueError('saved stack members differ from its tensor shape')
                blocks = tuple(dict.fromkeys(block for member in recipe.inputs
                                            for block in preparations[member].blocks))
                if blocks != prep.blocks:
                    raise ValueError('saved stack dependencies differ from its members')
                active.add(key)
                pending.append((key, True))
                pending.extend((member, False) for member in reversed(recipe.inputs))
                continue
            done.add(key)


def describe_input_reads(inputs):
    """Freeze optional background-read bounds during complete plan delivery."""
    from dataclasses import replace
    if set(inputs.read_requests) != set(inputs.blocks):
        raise ValueError('complete plans require described reads for every physical block')
    loader = inputs.loader
    result = {}
    for key, request in inputs.read_requests.items():
        block = inputs.blocks[key]
        eligible = loader.can_prefetch_block(key, block)
        if type(eligible) is not bool:
            raise TypeError('can_prefetch_block must return bool')
        size = loader.prefetch_host_bytes(key, block) if eligible else None
        if size is not None and (type(size) is not int or size < block.host_bytes + block.staging_host_bytes):
            raise ValueError('prefetch bound must include host storage and read workspace')
        result[key] = replace(request, prefetch_host_bytes=size)
    return result


def describe_input_preparations(inputs):
    """Freeze E's full pure table, including matrix stack member descriptors."""
    preparations=getattr(inputs.loader,'preparations',inputs.preparations)
    specs=getattr(inputs.loader,'specs',inputs.problem.specs)
    if any(prep.recipe is None for prep in preparations.values()):
        raise ValueError('complete plans require described preparations')
    _check_preparations(inputs.read_requests,preparations,specs)
    return dict(preparations),dict(specs)


def input_readers(inputs):
    """Return borrowed cfg reader handles under their explicit saved roles."""
    if not hasattr(inputs.loader,'input_readers'):
        raise ValueError('input binding does not expose restorable reader roles')
    return dict(inputs.loader.input_readers)


def describe_coordinate_reads(inputs,points):
    """Describe each unique physical read once during offline cfg delivery.

    Actual endpoint basis IDs can vary with time. Capture these headers now,
    keeping the representative graph's endpoint equality checks for each time.
    Runtime only maps these saved relations to the current cfg's actual IDs.
    """
    readers=input_readers(inputs)
    metadata={}; relations={}
    for point in points:
        values=dict(point.values)
        for primary in inputs.read_requests.values():
            for template in (primary,*primary.aliases):
                request=template.at(values)
                key=(request.reader_role,request.arguments,request.selection)
                if key not in metadata:
                    reader=readers[request.reader_role]
                    _check_conventions(request,reader)
                    actual=reader.describe_block(*request.arguments,**_selection(request))
                    expected=template.metadata
                    if (actual.axes!=expected.axes or actual.shape!=expected.shape
                            or actual.dtype!=expected.dtype or actual.sparse_layout!=expected.sparse_layout
                            or actual.point_diagonal!=expected.point_diagonal
                            or actual.structure!=expected.structure
                            or set(actual.axis_bindings)!=set(expected.axis_bindings)):
                        raise ValueError('coordinate input differs from template layout')
                    metadata[key]=actual
                actual=metadata[key]
                for axis,(expected_id,labels) in template.metadata.axis_bindings.items():
                    actual_id,actual_labels=actual.axis_bindings[axis]
                    if (tuple(map(canonical_coordinate_identity,labels))
                            != tuple(map(canonical_coordinate_identity,actual_labels))):
                        raise ValueError('coordinate input changes ordered endpoint labels')
                    if request.family=='propagator':
                        time=request.arguments[0 if axis.startswith('sink_') else 1]
                    else: time=request.arguments[2]
                    relation=(expected_id,time)
                    if relations.setdefault(relation,actual_id)!=actual_id:
                        raise ValueError('coordinate input changes a shared endpoint basis')
    return metadata


def unknown_reader_aliases(inputs):
    """Explicit shared handles remain shared when a data identity is unknown."""
    readers=input_readers(inputs); groups={}; unknown=set()
    for primary in inputs.read_requests.values():
        for request in (primary,*primary.aliases):
            if request.metadata.data_identity is None: unknown.add(request.reader_role)
    for role in sorted(unknown): groups.setdefault(id(readers[role]),[]).append(role)
    return tuple(tuple(roles) for roles in groups.values() if len(roles)>1)


__all__ = ['restore_input_loader', 'describe_input_reads','describe_input_preparations',
           'input_readers','describe_coordinate_reads','unknown_reader_aliases']


def bound_source_identities(loader):
    """Actual header identities, detached from runtime reader handles."""
    physical=getattr(loader,'physical',loader)
    values=tuple(m.data_identity for m in physical.metadata.values())
    return tuple(sorted({v for v in values if v is not None})),all(v is not None for v in values)
