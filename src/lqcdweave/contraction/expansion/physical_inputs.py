"""Physical input descriptions and delayed loaders shared by E constructors.

Extracted from LegacyDiagramBinding.describe in legacy_diagram.py. This service
registers existing C records; endpoint rules, costs, raw reads and preparation
callbacks retain the original implementation. It performs no numeric reads.
"""
import math
from dataclasses import replace
import numpy as np
from lqcdweave.contracts.correlator.api import (
    BatchTensorBinding, PhysicalSlabBinding, TensorAxisMetadata, TensorSpec,
    SourceSpec, SourcePreparation, StorageBlock,
    canonical_axis_binding_identity, canonical_coordinate_identity,
    TensorPreparation, SourceReadRequest, TensorDescription,
)
from lqcdweave.contracts.elemental import SparsePointRecordMetadata
from lqcdweave.contracts.correlator.vertices import uses_temporal_endpoint_order
from lqcdweave.contracts.elemental_sectors import sector_layout, sector_kinds, propagator_axes
from lqcdweave.contraction.binding.api import operator_term_semantics
from .sampling_binding import logical_vertex_tail_shape
from .tensor_description import make_tensor_description, point_pool_bindings, stable_binding_value, tensor_recipe_key


class PhysicalInputRegistry:
    """Call-local metadata registration; no graph, term or path construction."""
    def __init__(self, bindings, *, coordinate_values=None):
        self.bindings = bindings
        self.coordinate_values = {str(role): value for role, value in (coordinate_values or {}).items()}
        self.specs = {}
        self.sources = {}
        self.blocks = {}
        self.preparations = {}
        self.tensor_requests = {}
        self.physical_blocks = {}
        self.logical_tensors = {}
        self.tensor_recipes = {}
        self.batch_bindings = {}
        self.sparse_logical = {}
        self.block_readers = {}
        self.block_metadata = {}
        self.block_closers = {}
        self.materializers = {}
        self.problem_record_metadata = {}
        self.propagator_metadata = {}
        self.endpoint_checks_cache = {}
        self.term_semantics = {}
        self.registered_propagators = {}
        self.registered_vertices = {}
        self.element_counter = 0
        self.record_counter = 0
        self.read_requests = {}
        self.reader_roles = {}
        self.reader_bindings = {}
        self.reader_dilution_identities = {}
        self.dilution_descriptions = {}
        self.description_keys = {}

    def describe_dilution(self, source):
        key = id(source)
        if key not in self.dilution_descriptions:
            self.dilution_descriptions[key] = (source, TensorDescription(bindings=source.document))
        return self.dilution_descriptions[key][1]

    def register_read(self, key, reader, family, arguments, selection, *, role=None, coordinates=(), coordinate_tables=()):
        block_id = self.preparations[key].blocks[0]
        if role is None:
            role = self.reader_roles.setdefault(id(reader), f"elemental:{len(self.reader_roles)}")
        self.reader_bindings[role] = reader
        # mode selections are immutable sequences, never mutable dicts in an
        # archive. The restored adapter expands these explicit keyword pairs.
        selection = tuple((name, tuple(value.items()) if isinstance(value, dict) else value)
                          for name, value in sorted(selection.items()))
        names = ('time_convention',) if family == 'propagator' else ('phase_convention','path_boundary','momenta','momentum_list')
        names = (*names, 'time_selection_identity')
        if family == 'propagator' and getattr(reader, 'read_context_axes', ()):
            names = (*names, 'read_context_axes')
        if family == 'propagator' and hasattr(reader, 'propagator_time_view'):
            names = (*names, 'propagator_time_view')
        # Reader-wide conventions belong to this metadata snapshot. Serializing
        # a full D frame for each tensor repeats every parent-mode label.
        identity_key = id(reader)
        if identity_key not in self.reader_dilution_identities:
            self.reader_dilution_identities[identity_key] = (reader, getattr(reader, 'dilution_identity', None))
        dilution_identity = self.reader_dilution_identities[identity_key][1]
        if dilution_identity is not None:
            names = (*names, 'dilution_identity')
        if family != 'propagator':
            names = (*names, 'elemental_weighted')
        if hasattr(reader, 'glueball_definition'):
            names = (*names, 'glueball_definition')
        request = SourceReadRequest(role, family, tuple(arguments), selection, self.block_metadata[block_id],
            conventions=tuple((name, dilution_identity if name == 'dilution_identity' else getattr(reader,name,None)) for name in names),
            coordinate_arguments=coordinates, coordinate_tables=coordinate_tables)
        previous = self.read_requests.setdefault(block_id, request)
        if replace(previous, aliases=()) != request and request not in previous.aliases:
            self.read_requests[block_id] = replace(previous, aliases=(*previous.aliases, request))

    def loader(self):
        return PhysicalSourceLoader(self)

    def propagator_context(self, reader):
        """Additional saved read arguments, never extra Wick endpoints."""
        axes = getattr(reader, 'read_context_axes', ())
        if not axes:
            return ()
        if (type(axes) is not tuple or any(type(role) is not str or not role for role in axes)
                or len(set(axes)) != len(axes)):
            raise ValueError('read_context_axes must be distinct coordinate role strings')
        try:
            return tuple(self.coordinate_values[role] for role in axes)
        except KeyError as exc:
            raise ValueError(f'propagator read context lacks coordinate role {exc.args[0]!r}') from exc

    def describe_propagator(self, reader, sink, source, sector, selection):
        choices = tuple((name, tuple(sorted(value.items()))
                         if name == "mode_indices" else value)
                        for name, value in sorted(selection.items()))
        context = self.propagator_context(reader)
        key = (id(reader), sink, source, sector, context, choices)
        if key not in self.propagator_metadata:
            metadata = reader.describe_block(sink, source, sector, *context, **selection)
            _check_axes(metadata, sector)
            self.propagator_metadata[key] = metadata
        return self.propagator_metadata[key]

    def endpoint_check(self, metadata, sector, endpoint, factor, side):
        key = (id(metadata), sector, endpoint, factor.logical_sector,
               factor.dagger, factor.tensor.data_identity,
               factor.tensor.selection_identity, side)
        if key not in self.endpoint_checks_cache:
            check = _compare_propagator_endpoint(metadata, sector, endpoint, factor, side)
            self.endpoint_checks_cache[key] = check, _endpoint_check_identity(check)
        return self.endpoint_checks_cache[key]

    def semantics(self, term):
        key = id(term)
        if key not in self.term_semantics:
            self.term_semantics[key] = operator_term_semantics(term)
        return self.term_semantics[key]

    def get_block(self, metadata, reader, family, read_callback, close_callback=None):
        if isinstance(metadata, SparsePointRecordMetadata):
            if family != "p2p-record":
                raise TypeError("SparsePointRecordMetadata is valid only for P2P record blocks")
            raw_bytes = metadata.record_bytes
        elif isinstance(metadata, TensorAxisMetadata):
            if metadata.sparse_layout != "dense":
                raise NotImplementedError(
                    f"{family} sparse layout {metadata.sparse_layout!r} needs an explicit record provider"
                )
            raw_bytes = math.prod(metadata.shape) * metadata.dtype.itemsize
        else:
            raise TypeError(f"{family} metadata provider returned an unsupported metadata object")
        source_identity = metadata.data_identity
        if source_identity is None:
            source_identity = f"runtime-object:{id(reader)}"
        identity = (family, source_identity, metadata.selection_identity)
        block_key = self.physical_blocks.get(identity)
        if block_key is None:
            block_key = f"block:{len(self.blocks)}"
            block = StorageBlock(
                dataset=f"{family}:{source_identity}",
                selection=metadata.selection_identity,
                read_bytes=raw_bytes,
                host_bytes=raw_bytes,
                read_kind="p2p" if family == "p2p-record" else "dense",
            )
            self.physical_blocks[identity] = block_key
            self.blocks[block_key] = block
            self.block_readers[block_key] = read_callback
            self.block_metadata[block_key] = metadata
            self.block_closers[block_key] = close_callback
        elif self.blocks[block_key].read_bytes != raw_bytes:
            raise ValueError("shared block identity has inconsistent shape/dtype byte extent")
        else:
            previous = self.block_metadata[block_key]
            consistent = (
                previous == metadata
                if isinstance(metadata, SparsePointRecordMetadata)
                else isinstance(previous, TensorAxisMetadata)
                and previous.shape == metadata.shape
                and previous.dtype == metadata.dtype
                and previous.sparse_layout == metadata.sparse_layout
                and previous.point_diagonal == metadata.point_diagonal
                and previous.structure == metadata.structure
                and canonical_axis_binding_identity(previous.axis_bindings)
                == canonical_axis_binding_identity(metadata.axis_bindings)
            )
            if not consistent:
                raise ValueError(
                    "shared block identity/selection has inconsistent shape, dtype, or record layout"
                )
        return block_key

    def add_tensor(self, logical_identity, metadata, reader, family, read_callback,
                  materialize_callback, *, output_shape=None, output_dtype=None,
                  workspace_bytes=0, output_axes=None, transform=(),
                  description_semantics=None, description_bindings=None,
                  metadata_sources=()):
        key = self.logical_tensors.get(logical_identity)
        if key is not None:
            expected = self.specs[key]
            requested_shape = metadata.shape if output_shape is None else tuple(output_shape)
            requested_dtype = metadata.dtype if output_dtype is None else np.dtype(output_dtype)
            if expected.shape != requested_shape or expected.dtype != requested_dtype:
                raise ValueError("shared logical tensor identity has inconsistent TensorSpec")
            if expected.description.semantics.get("point_diagonal", False) != metadata.point_diagonal:
                raise ValueError("shared logical tensor identity has inconsistent point support")
            if expected.description.semantics.get("structure", "general") != metadata.structure:
                raise ValueError("shared logical tensor identity has inconsistent elemental structure")
            expected_bindings = expected.description.bindings.get("axis_bindings", {})
            if (canonical_axis_binding_identity(expected_bindings)
                    != canonical_axis_binding_identity(metadata.axis_bindings)):
                raise ValueError("shared logical tensor identity has inconsistent axis bindings")
            return key
        block_key = self.get_block(metadata, reader, family, read_callback)
        shape = metadata.shape if output_shape is None else tuple(output_shape)
        dtype = metadata.dtype if output_dtype is None else np.dtype(output_dtype)
        actual_bindings = dict(description_bindings or {})
        if actual_bindings.get("configuration") is None:
            cfg = self.bindings.configuration
            if cfg is None:
                cfg = getattr(reader, "cfg", None)
            actual_bindings["configuration"] = stable_binding_value(cfg)
        description = make_tensor_description(
            metadata,
            role=family,
            source_bindings=(block_key,),
            axes=output_axes,
            transform=transform,
            semantics=description_semantics,
            bindings=actual_bindings,
            metadata_sources=metadata_sources,
        )
        spec = TensorSpec(shape, dtype, description=description)
        recipe = tensor_recipe_key(description, shape, dtype, workspace_bytes,
                                   metadata_cache=self.description_keys)
        if recipe is not None and recipe in self.tensor_recipes:
            key = self.tensor_recipes[recipe]
            self.logical_tensors[logical_identity] = key
            return key
        key = f"tensor:{self.element_counter}"
        self.element_counter += 1
        snapshot = math.prod(spec.shape) * spec.dtype.itemsize
        block = self.blocks[block_key]
        self.specs[key] = spec
        self.sources[key] = SourceSpec(
            region=block_key,
            region_bytes=block.read_bytes,
            host_region_bytes=block.host_bytes,
            snapshot_bytes=snapshot,
            metadata={"family": family, "block": block_key},
        )
        self.preparations[key] = SourcePreparation(
            (block_key,), workspace_host_bytes=workspace_bytes,
            kernel_calls=1 if workspace_bytes else 0, dtype=spec.dtype.name,
        )
        self.tensor_requests[key] = logical_identity
        self.materializers[key] = materialize_callback
        self.logical_tensors[logical_identity] = key
        if recipe is not None:
            self.tensor_recipes[recipe] = key
        return key

    def add_sparse_record(self, logical_identity, metadata, reader, read_callback,
                          close_callback, materialize_callback, description):
        if not isinstance(metadata, SparsePointRecordMetadata):
            raise TypeError("sparse P2P vertex must describe a SparsePointRecordMetadata record")
        if metadata.backend != "numpy":
            raise NotImplementedError(
                "native sparse P2P bridging currently requires a CPU NumPy record provider"
            )
        key = self.sparse_logical.get(logical_identity)
        if key is not None:
            if self.problem_record_metadata[key] != metadata:
                raise ValueError("shared sparse logical record has inconsistent metadata")
            return key
        block_key = self.get_block(
            metadata, reader, "p2p-record", read_callback,
            close_callback=close_callback,
        )
        key = f"sparse-record:{self.record_counter}"
        self.record_counter += 1
        self.sources[key] = SourceSpec(
            region=block_key,
            region_bytes=self.blocks[block_key].read_bytes,
            host_region_bytes=self.blocks[block_key].host_bytes,
            snapshot_bytes=None,
            metadata={
                "family": "p2p-record", "block": block_key,
                "kind": metadata.kind, "point_shape": metadata.point_shape,
                "nnz": metadata.nnz, "value_dtype": metadata.value_dtype.name,
                "index_dtype": metadata.index_dtype.name,
                "record_bytes": metadata.record_bytes, "backend": metadata.backend,
                "configuration": stable_binding_value(self.bindings.configuration),
                "anchor_time": int(description.anchor_time),
                "bar_time": int(description.bar_time),
                "field_time": int(description.field_time),
                "point_pool_ids": dict(description.point_pool_ids),
                "elemental_weighted": description.elemental_weighted,
                "point_diagonal": metadata.point_diagonal,
                "structure": metadata.structure,
                "axis_bindings": dict(description.tensor.axis_bindings),
                "logical_endpoint_axis_bindings":
                    _logical_vertex_endpoint_bindings(description),
                "coefficient_owner": "OutputRecipe",
            },
        )
        self.preparations[key] = SourcePreparation(
            (block_key,),
            workspace_host_bytes=metadata.record_bytes if logical_identity[-1] else 0,
            dtype=metadata.value_dtype.name,
        )
        self.materializers[key] = materialize_callback
        self.sparse_logical[logical_identity] = key
        self.problem_record_metadata[key] = metadata
        return key

    def register_propagator(self, prop_reader, propagator_symbol, source_factor,
                            sink_factor, sector, *, source_role=None, sink_role=None):
        if _has_temporal_point_split(source_factor):
            source_time = source_factor.bar_time
        else:
            source_time = source_factor.field_time
        if _has_temporal_point_split(sink_factor):
            sink_time = sink_factor.field_time
        else:
            sink_time = sink_factor.bar_time
        roles = ((0, sink_role, sink_time, sink_factor, 2 if _has_temporal_point_split(sink_factor) else 1, 'sink_time'),
                 (1, source_role, source_time, source_factor, 1 if _has_temporal_point_split(source_factor) else 2, 'source_time'))
        coordinate_tables, coordinates, mapped_times = [], [], {}
        for index, role, time, factor, endpoint, axis in roles:
            if factor.coordinate_slot_maps:
                source_record = factor.endpoint_time_selection
                target_record = prop_reader.time_selections[axis]
                if not factor.physical_offset_mapping:
                    if source_record != target_record:
                        raise ValueError('connected same-time edge requires identical ordered time selections for its endpoint role')
                    remap = tuple(range(source_record.Nt))
                else:
                    if source_record.Lt != target_record.Lt:
                        raise ValueError('propagator endpoint physical Lt differs from bound vertex')
                    lookup = {physical: slot for slot, physical in enumerate(target_record.times)}
                    remap = tuple(lookup.get(physical) for physical in source_record.times)
                actual = remap[time]
                if actual is None:
                    raise KeyError('physical vertex endpoint is absent from the connected propagator selection')
                mapped_times[index] = actual
                if role is not None:
                    table = tuple(None if value is None else remap[value]
                                  for value in factor.coordinate_slot_maps[endpoint])
                    coordinate_tables.append((index, str(role), table))
            else:
                mapped_times[index] = time
                if role is not None:
                    coordinates.append((index, str(role), int(time-factor.anchor_time), int(prop_reader.Lt)))
        sink_time, source_time = mapped_times[0], mapped_times[1]
        context = self.propagator_context(prop_reader)
        arguments = (sink_time, source_time, sector, *context)
        coordinates.extend((3 + index, role, 0, None)
                           for index, role in enumerate(getattr(prop_reader, 'read_context_axes', ())))
        coordinates, coordinate_tables = tuple(coordinates), tuple(coordinate_tables)
        selection = _propagator_selection(sector, self.bindings)
        if source_factor.mode_indices is not None or sink_factor.mode_indices is not None:
            if sector != "vsv" or source_factor.mode_indices is None or sink_factor.mode_indices is None:
                raise NotImplementedError("partitioned propagators require both V/V endpoints")
            source_slot = 1 if source_factor.dagger else 0
            sink_slot = 0 if sink_factor.dagger else 1
            selection.pop("usedNe", None)
            selection["mode_indices"] = {
                "sink_mode": sink_factor.mode_indices[sink_slot],
                "source_mode": source_factor.mode_indices[source_slot],
            }
        if "d" in (sector[0], sector[2]):
            selection.pop("usedNe", None)
            full_metadata = self.describe_propagator(prop_reader, sink_time, source_time, sector, {})
            mode_selection = selection.setdefault("mode_indices", {})
            for endpoint, factor, side in (("source", source_factor, "field"),
                                           ("sink", sink_factor, "bar")):
                axis, kind = _vertex_endpoint_axis_name(factor, side)
                if kind == "p":
                    continue
                binding = factor.tensor.axis_bindings.get(axis)
                prop_axis = endpoint + "_mode"
                full_binding = full_metadata.axis_bindings.get(prop_axis)
                if kind == "d" and (binding is None or full_binding is None):
                    raise ValueError("D elemental and propagator require actual basis identities and ordered indices")
                if binding is not None and full_binding is not None:
                    if binding[0] != full_binding[0]:
                        raise ValueError("D/V propagator basis differs from the connected elemental")
                    lookup = {canonical_coordinate_identity(label): i for i, label in enumerate(full_binding[1])}
                    try:
                        mode_selection[prop_axis] = tuple(lookup[canonical_coordinate_identity(label)] for label in binding[1])
                    except KeyError as exc:
                        raise ValueError("D/V elemental labels are absent from the connected propagator") from exc
                elif self.bindings.usedNe is not None:
                    mode_selection[prop_axis] = tuple(range(factor.tensor.shape[factor.tensor.axes.index(axis)]))
        metadata = self.describe_propagator(
            prop_reader, sink_time, source_time, sector, selection,
        )
        source_endpoint_check, source_identity = self.endpoint_check(
            metadata, sector, "source",
            source_factor, "field",
        )
        sink_endpoint_check, sink_identity = self.endpoint_check(
            metadata, sector, "sink",
            sink_factor, "bar",
        )
        endpoint_checks = {
            "source": source_endpoint_check,
            "sink": sink_endpoint_check,
        }
        endpoint_check_identity = source_identity, sink_identity
        logical_id = ("propagator", id(prop_reader), sector,
                      metadata.selection_identity,
                      endpoint_check_identity)
        # Validate both endpoints before sharing. The metadata object also
        # distinguishes headers that claim the same logical identity.
        request_key = id(metadata), logical_id
        cached = self.registered_propagators.get(request_key)
        if cached is not None:
            self.register_read(cached, prop_reader, 'propagator',
                               arguments, selection,
                               role='propagator:' + str(propagator_symbol), coordinates=coordinates,
                               coordinate_tables=coordinate_tables)
            return cached
        read = lambda reader=prop_reader, args=arguments, sel=selection: reader.block(*args, **sel)
        key = self.add_tensor(
            logical_id, metadata, prop_reader, "propagator", read,
            lambda raw: raw,
            description_semantics={
                "sector": sector,
                "symbol": str(propagator_symbol),
                "time_convention": getattr(prop_reader, "time_convention", None),
            },
            description_bindings={
                "configuration": stable_binding_value(self.bindings.configuration),
                "sink_time": int(sink_time),
                "source_time": int(source_time),
                "usedNe": stable_binding_value(selection.get("usedNe")),
                "usedNp": stable_binding_value(selection.get("usedNp")),
                "endpoint_axis_bindings": endpoint_checks,
            },
        )
        self.preparations[key] = replace(self.preparations[key],
            recipe=TensorPreparation("identity", self.preparations[key].blocks))
        self.materializers.pop(key, None)
        self.register_read(key, prop_reader, 'propagator',
                           arguments, selection,
                           role='propagator:' + str(propagator_symbol), coordinates=coordinates,
                               coordinate_tables=coordinate_tables)
        describe_slab = getattr(prop_reader, "describe_slab", None)
        read_slab = getattr(prop_reader, "read_slab", None)
        if key not in self.batch_bindings and callable(describe_slab) and callable(read_slab):
            data_identity = (metadata.data_identity if metadata.data_identity is not None
                             else id(prop_reader))
            fixed_selection = tuple(sorted(
                (name, repr(value)) for name, value in selection.items()))
            slab_identity = (
                "propagator", data_identity,
                source_time, sector, context, fixed_selection,
            )
            slab = PhysicalSlabBinding(
                slab_identity,
                lambda times, reader=prop_reader, t=source_time,
                sec=sector, ctx=context, sel=dict(selection): reader.describe_slab(
                    times, t, sec, *ctx, **sel),
                lambda times, reader=prop_reader, t=source_time,
                sec=sector, ctx=context, sel=dict(selection): reader.read_slab(
                    times, t, sec, *ctx, **sel),
                (lambda metadata, reader=prop_reader,
                 sel=dict(selection): reader.read_slab_workspace_bytes(
                     metadata, **sel))
                if callable(getattr(prop_reader,
                                    "read_slab_workspace_bytes", None))
                else None,
                (lambda times, reader=prop_reader, t=source_time,
                 sec=sector, ctx=context, sel=dict(selection):
                 reader.read_slab_in_background(times, t, sec, *ctx, **sel))
                if (getattr(prop_reader, "background_read_safe", False) is True
                    and callable(getattr(prop_reader, "read_slab_in_background", None)))
                else None,
            )
            binding = BatchTensorBinding(
                key, (slab,), (sink_time,), (lambda raw: raw,),
                workspace_host_bytes=0,
            )
            previous_binding = self.batch_bindings.setdefault(key, binding)
            if previous_binding != binding:
                raise ValueError(
                    "one logical propagator tensor has inconsistent batch slab metadata"
                )
        self.registered_propagators[request_key] = key
        return key

    def register_dense_vertex(self, meson, descriptor, *, time_role=None, compact_structure=True,
                              embed_structure=False):
        from lqcdweave.contracts.correlator.api import MesonCoreMetadata
        if isinstance(descriptor, MesonCoreMetadata):
            from .vertex_sum import register_vertex_sum
            members = tuple(self.register_dense_vertex(meson, item, time_role=time_role,
                            compact_structure=compact_structure and not descriptor.mixed_structure,
                            embed_structure=descriptor.mixed_structure) for item in descriptor.factors)
            return register_vertex_sum(members, descriptor.factors, specs=self.specs,
                sources=self.sources, blocks=self.blocks, preparations=self.preparations,
                tensor_requests=self.tensor_requests, tensor_metadata=descriptor.tensor)
        request_key = (id(meson), descriptor.term_index, descriptor.logical_sector,
                       descriptor.raw_sector, descriptor.tensor.selection_identity,
                       compact_structure, embed_structure)
        cached = self.registered_vertices.get(request_key)
        if cached is not None:
            return cached
        tail_shape = logical_vertex_tail_shape(descriptor)
        if compact_structure and descriptor.tensor.structure == "diagonal":
            tail_shape = tail_shape[:1] if sector_layout(descriptor.logical_sector) == "v2v" else tail_shape[:2]
        output_shape = (4, 4, *tail_shape)
        output_dtype = np.result_type(descriptor.tensor.dtype, np.complex128)
        workspace_bytes = math.prod(output_shape) * np.dtype(output_dtype).itemsize
        if descriptor.dagger:
            # NumPy/CuPy conj() owns a raw-shaped copy;
            # transpose() only returns a view of it.
            workspace_bytes += (
                math.prod(descriptor.tensor.shape)
                * np.dtype(descriptor.tensor.dtype).itemsize
            )
        axis_permutations = {
            "v2v": (1, 0), "v2p": (1, 2, 0),
            "p2v": (2, 0, 1), "p2p": (2, 3, 0, 1),
        }
        permutation = (axis_permutations[sector_layout(descriptor.raw_sector)]
                       if descriptor.dagger else
                       tuple(range(len(descriptor.tensor.axes))))
        logical_tail_axes = tuple(
            descriptor.tensor.axes[index] for index in permutation
        )
        if compact_structure and descriptor.tensor.structure == "diagonal":
            logical_tail_axes = logical_tail_axes[:len(tail_shape)]
        logical_axes = ("spin_field", "spin_bar", *logical_tail_axes)
        term = descriptor.operator_term
        profile = (None if term.profile is None else
                   self.semantics(term)["profile"])
        logical_id = ("vertex", id(meson), descriptor.term_index,
                      descriptor.logical_sector, descriptor.tensor.selection_identity, compact_structure,
                      embed_structure)
        read = lambda m=meson, d=descriptor: m.read_factor_block(d)
        dilution = {side: self.describe_dilution(source)
                    for side, source in descriptor.dilution_endpoints.items()}
        key = self.add_tensor(
            logical_id, descriptor.tensor, meson.reader,
            "elemental", read,
            lambda raw, m=meson, d=descriptor: m.materialize_unweighted_factor(
                d, raw, compact_structure=compact_structure),
            output_shape=output_shape, output_dtype=output_dtype,
            workspace_bytes=workspace_bytes,
            output_axes=logical_axes,
            transform=(
                ("operation", "gamma-spin-spatial-product"),
                ("dagger", descriptor.dagger),
                ("axis_permutation", permutation),
                ("profile_sha256", None if profile is None else profile["sha256"]),
                ("spatial_structure", descriptor.tensor.structure),
                ("embed_structure", embed_structure),
            ),
            description_semantics={
                "logical_sector": descriptor.logical_sector,
                "raw_sector": descriptor.raw_sector,
                "operator_term": self.semantics(term),
                "coefficient_owner": "OutputRecipe",
            },
            description_bindings={
                "configuration": stable_binding_value(self.bindings.configuration),
                "anchor_time": int(descriptor.anchor_time % meson.reader.Lt),
                "bar_time": int(descriptor.bar_time),
                "field_time": int(descriptor.field_time),
                "bar_time_unwrapped": int(descriptor.bar_time_unwrapped),
                "field_time_unwrapped": int(descriptor.field_time_unwrapped),
                "endpoint_offsets": tuple(descriptor.endpoint_offsets),
                "point_pool_ids": dict(descriptor.point_pool_ids),
                "dilution_endpoints": {side: description.bindings for side, description in dilution.items()},
                "dilution_weight_scheme": descriptor.dilution_weight_scheme,
                "logical_endpoint_axis_bindings":
                    _logical_vertex_endpoint_bindings(descriptor),
                "point_pool_metadata": point_pool_bindings(descriptor),
                "elemental_weighted": descriptor.elemental_weighted,
                "usedNe": stable_binding_value(descriptor.usedNe),
                "usedNp": stable_binding_value(descriptor.usedNp),
                "phase_convention": meson.phase_convention,
                "boundary": meson.boundary,
            },
            metadata_sources=tuple(dilution.values()),
        )
        self.preparations[key] = replace(self.preparations[key],
            recipe=meson.describe_factor_preparation(descriptor, self.preparations[key].blocks[0],
                                                    compact_structure=compact_structure,
                                                    embed_structure=embed_structure))
        self.materializers.pop(key, None)
        reader = meson.reader
        selection = dict(usedNe=descriptor.usedNe if sector_layout(descriptor.raw_sector) in {'v2v','v2p','p2v'} else None,
                         usedNp=descriptor.usedNp if 'p' in sector_kinds(descriptor.raw_sector) else None)
        if descriptor.mode_indices is not None:
            selection['mode_indices'] = dict(zip(('mode_left','mode_right'),descriptor.mode_indices))
        if descriptor.basis_selections:
            selection['mode_indices'] = dict(descriptor.basis_selections)
        self.register_read(key, reader, 'elemental',
            (term.component,term.momentum,descriptor.anchor_time % reader.Lt,descriptor.raw_sector),selection,
            coordinates=() if time_role is None or descriptor.coordinate_slot_maps else ((2,str(time_role),0,int(reader.Lt)),),
            coordinate_tables=() if time_role is None or not descriptor.coordinate_slot_maps else
                ((2, str(time_role), descriptor.coordinate_slot_maps[0]),))
        describe_slab = getattr(reader, "describe_slab", None)
        read_slab = getattr(reader, "read_slab", None)
        if key not in self.batch_bindings and callable(describe_slab) and callable(read_slab):
            term = descriptor.operator_term
            anchor_time = descriptor.anchor_time % reader.Lt
            selection = {}
            if sector_layout(descriptor.raw_sector) in {"v2v", "v2p", "p2v"}:
                selection["usedNe"] = descriptor.usedNe
            if descriptor.mode_indices is not None:
                selection.pop("usedNe", None)
                selection["mode_indices"] = dict(zip(
                    ("mode_left", "mode_right"), descriptor.mode_indices))
            if descriptor.basis_selections:
                selection["mode_indices"] = dict(descriptor.basis_selections)
            if "p" in sector_kinds(descriptor.raw_sector):
                selection["usedNp"] = descriptor.usedNp
            data_identity = (descriptor.tensor.data_identity
                             if descriptor.tensor.data_identity is not None
                             else id(reader))
            slab_identity = (
                "elemental", data_identity,
                repr(term.component), term.momentum,
                descriptor.raw_sector,
                tuple(sorted((name, repr(value))
                             for name, value in selection.items())),
            )
            slab = PhysicalSlabBinding(
                slab_identity,
                lambda times, r=reader, c=term.component,
                m=term.momentum, sec=descriptor.raw_sector,
                sel=dict(selection): r.describe_slab(
                    c, m, times, sec, **sel),
                lambda times, r=reader, c=term.component,
                m=term.momentum, sec=descriptor.raw_sector,
                sel=dict(selection): r.read_slab(
                    c, m, times, sec, **sel),
                (lambda metadata, r=reader,
                 sel=dict(selection): r.read_slab_workspace_bytes(
                     metadata, **sel))
                if callable(getattr(reader,
                                    "read_slab_workspace_bytes", None))
                else None,
                (lambda times, r=reader, c=term.component,
                 m=term.momentum, sec=descriptor.raw_sector,
                 sel=dict(selection):
                 r.read_slab_in_background(c, m, times, sec, **sel))
                if (getattr(reader, "background_read_safe", False) is True
                    and callable(getattr(reader, "read_slab_in_background", None)))
                else None,
            )
            binding = BatchTensorBinding(
                key, (slab,), (anchor_time,),
                (lambda raw, m=meson, d=descriptor:
                    m.materialize_unweighted_factor(d, raw, compact_structure=compact_structure),),
                workspace_host_bytes=workspace_bytes,
            )
            previous_binding = self.batch_bindings.setdefault(key, binding)
            if previous_binding != binding:
                raise ValueError(
                    "one logical elemental tensor has inconsistent batch slab metadata"
                )
        self.registered_vertices[request_key] = key
        return key


class PhysicalSourceLoader:
    def __init__(self, registry):
        # Retain only runtime callbacks/descriptions, not construction caches.
        self._batch_bindings = registry.batch_bindings
        self._preparations = registry.preparations
        self._block_metadata = registry.block_metadata
        self._block_readers = registry.block_readers
        self._block_closers = registry.block_closers
        self._materializers = registry.materializers
        self._background_slabs = None
        self.read_requests = registry.read_requests
        self.reader_roles = registry.reader_roles
        self.input_readers = dict(registry.reader_bindings)
        self.preparations, self.specs = registry.preparations, registry.specs
        self.constant_values = {}

    def background_slabs(self):
        # Batched E only uses the slab bindings, so do not construct
        # a second catalog for every fixed-coordinate lane it describes.
        if self._background_slabs is None:
            slabs = {}
            for key, binding in self._batch_bindings.items():
                if len(binding.physical_slabs) != 1 or len(binding.times) != 1:
                    continue
                slab = binding.physical_slabs[0]
                if slab.read_slab_in_background is not None:
                    block_key = self._preparations[key].blocks[0]
                    slabs.setdefault(block_key, {})[(id(slab), binding.times)] = (
                        slab, binding.times,
                    )
            self._background_slabs = slabs
        return self._background_slabs

    def can_prefetch_block(self, key, block):
        return key in self.background_slabs()

    def prefetch_host_bytes(self, key, block):
        workspace = 0
        expected = self._block_metadata[key]
        for slab, times in self.background_slabs()[key].values():
            metadata = slab.describe_slab(times)
            if (not isinstance(metadata, TensorAxisMetadata)
                    or metadata.shape != (1, *expected.shape)
                    or metadata.axes != ("coordinate_batch", *expected.axes)
                    or metadata.dtype != expected.dtype):
                raise ValueError("background slab differs from its fixed block metadata")
            if slab.read_workspace_bytes is not None:
                value = slab.read_workspace_bytes(metadata)
                if type(value) is not int or value < 0:
                    raise ValueError("read_slab_workspace_bytes must return nonnegative integer bytes")
                workspace = max(workspace, value)
        return 2 * block.host_bytes + block.staging_host_bytes + workspace

    def prefetch_block(self, key, block):
        # The slab path combines exact-axis gathers into one selected
        # extent, unlike sequential block() gathers on large raw axes.
        slab, times = next(iter(self.background_slabs()[key].values()))
        value = slab.read_slab_in_background(times)
        expected = self._block_metadata[key]
        if (tuple(value.shape) != (1, *expected.shape)
                or np.dtype(value.dtype) != expected.dtype):
            raise ValueError("background slab shape/dtype differs from metadata")
        return np.array(value[0], copy=True)

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        pass

    def read_block(self, key, block):
        try:
            value = self._block_readers[key]()
        except KeyError as exc:
            raise ValueError(f"no runtime reader was registered for block {key}") from exc
        metadata = self._block_metadata[key]
        if isinstance(metadata, SparsePointRecordMetadata):
            return value
        try:
            actual_shape = tuple(value.shape)
            actual_dtype = np.dtype(value.dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"reader block {key} must be an array with declared metadata") from exc
        if actual_shape != metadata.shape or actual_dtype != metadata.dtype:
            raise ValueError(
                f"reader block {key} shape/dtype differs from metadata: "
                f"actual={(actual_shape, actual_dtype)}, "
                f"expected={(metadata.shape, metadata.dtype)}"
            )
        return value

    def close_block(self, key, value):
        release = self._block_closers.get(key)
        if release is not None:
            release(value)

    def materialize(self, key, source, raw_blocks):
        block_ids = self._preparations[key].blocks
        if len(block_ids) != 1:
            raise ValueError("legacy native materializer expects one raw block per logical tensor")
        try:
            materialize = self._materializers[key]
        except KeyError as exc:
            raise ValueError(f"no opaque materializer was registered for tensor {key}") from exc
        return materialize(raw_blocks[block_ids[0]])


def _vertex_endpoint_axis_name(factor, endpoint):
    kind = factor.logical_sector[0 if endpoint == "field" else 2]
    numerical_kind = "point" if kind == "p" else "mode"
    if sector_layout(factor.logical_sector) == "v2v":
        axes = factor.tensor.axes
        if len(axes) != 2:
            raise ValueError("vector elemental metadata must expose its two ordered mode axes")
        index = (1 if endpoint == "field" else 0) if factor.dagger else (0 if endpoint == "field" else 1)
        return axes[index], kind
    side = ("right" if endpoint == "field" else "left") if factor.dagger else ("left" if endpoint == "field" else "right")
    return f"{numerical_kind}_{side}", kind


def _has_temporal_point_split(factor):
    return uses_temporal_endpoint_order(factor.operator_term, factor.current_term)


def _propagator_endpoint_axis_name(sector, endpoint):
    kind = sector[0 if endpoint == "sink" else 2]
    suffix = {"v": "mode", "d": "mode", "p": "point"}[kind]
    return f"{endpoint}_{suffix}", kind


def _binding_value_identity(binding):
    if binding is None:
        return None
    data_id, coordinates = binding
    return data_id, tuple(canonical_coordinate_identity(value) for value in coordinates)


def _compare_propagator_endpoint(propagator_metadata, sector, endpoint,
                                 vertex_factor, vertex_endpoint):
    propagator_axis, propagator_kind = _propagator_endpoint_axis_name(sector, endpoint)
    vertex_axis, vertex_kind = _vertex_endpoint_axis_name(vertex_factor, vertex_endpoint)
    if propagator_kind != vertex_kind:
        raise ValueError(
            f"propagator {endpoint} endpoint and Meson {vertex_endpoint} endpoint use different axis kinds"
        )
    propagator_binding = propagator_metadata.axis_bindings.get(propagator_axis)
    vertex_binding = vertex_factor.tensor.axis_bindings.get(vertex_axis)
    if propagator_binding is not None and vertex_binding is not None:
        if _binding_value_identity(propagator_binding) != _binding_value_identity(vertex_binding):
            raise ValueError(
                f"propagator {endpoint} endpoint actual dataset ID or ordered coordinates "
                f"disagree with Meson {vertex_endpoint} endpoint"
            )
        status = "verified"
        knowledge = "both-bound"
    else:
        if vertex_kind == "d":
            raise ValueError("D propagator and elemental endpoints require matching actual basis identities")
        status = "unknown"
        knowledge = "both-unbound" if propagator_binding is None and vertex_binding is None else "one-unbound"
    return {
        "status": status,
        "knowledge": knowledge,
        "propagator_axis": propagator_axis,
        "meson_axis": vertex_axis,
        "propagator": propagator_binding,
        "meson": vertex_binding,
    }


def _endpoint_check_identity(check):
    return (
        check["status"], check["knowledge"],
        check["propagator_axis"], check["meson_axis"],
        _binding_value_identity(check["propagator"]),
        _binding_value_identity(check["meson"]),
    )


def _logical_vertex_endpoint_bindings(factor):
    return {
        endpoint: {
            "raw_axis": axis,
            "binding": factor.tensor.axis_bindings.get(axis),
        }
        for endpoint in ("field", "bar")
        for axis, _kind in (_vertex_endpoint_axis_name(factor, endpoint),)
    }


def _check_axes(metadata, sector):
    if not isinstance(metadata, TensorAxisMetadata):
        raise TypeError("propagator describe_block must return TensorAxisMetadata")
    expected = tuple(axis.replace("_dilution_vector", "_mode") for axis in propagator_axes(sector))
    if metadata.axes != expected:
        raise ValueError(f"propagator {sector} axes {metadata.axes} do not match required {expected}")
    if metadata.sparse_layout != "dense":
        raise NotImplementedError("sparse propagator sectors need an explicit SparseBridgeSpec provider")


def _propagator_selection(sector, bindings):
    selection = {}
    if "v" in (sector[0], sector[2]) and bindings.usedNe is not None:
        selection["usedNe"] = (bindings.usedNe.get("V") if isinstance(bindings.usedNe, dict) else bindings.usedNe)
    if "p" in (sector[0], sector[2]) and bindings.usedNp is not None:
        selection["usedNp"] = bindings.usedNp
    return selection
