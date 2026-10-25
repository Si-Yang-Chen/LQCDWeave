"""Test-owned production script composition; not an installed production API."""

import numpy as np
from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import dataset_time_selection
from lqcdweave.contracts.current import validate_directed_point_current_spec
from lqcdweave.contracts.elemental import TEMPORAL_POINT_PATH_SCHEMA, GaugePath
from lqcdweave.planning.production import elemental_projection_documents, directed_current_dataset_document, point_elemental_dataset_document
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator, re_combine

def _physical_time_coordinate(spec, time):
    selection = dataset_time_selection(spec)
    return time if selection is None else selection.slot(time)

class _OrderedDatasetPointHandle:
    """Select actual point labels from one graph-owned DatasetView."""

    def __init__(self, dataset, labels):
        self._handle = DatasetPointHandle(dataset)
        self.time_selection = self._handle.time_selection
        available = dataset.spec.coordinates("point")
        positions = {label: position for position, label in enumerate(available)}
        self._positions = tuple(positions[label] for label in labels if label in positions)
        if len(self._positions) != len(labels) or not self._positions:
            raise ValueError("point selection is not covered by its runtime input view")
        self.Np = len(self._positions)

    def load(self, key):
        return self._handle.load(key)[:, list(self._positions), :]

def _dense_point_pair_row(sparse_blocks, *, time, points, right_reader, left_labels,
                          right_labels, axes, cfg, path_boundary):
    """Validate sparse P/P blocks and build the fixed left-slot output row."""
    row_values = np.zeros((len(axes["component"]), len(axes["momentum"]),
                           len(axes["point_left"]), 3, 3), dtype=np.complex128)
    left_coordinates = np.asarray(points.read(time=_physical_time_coordinate(points.spec, time), select={"point": left_labels}))
    spatial_size = np.asarray(cfg["lattice"][:3])
    for component, component_record in enumerate(axes["component"]):
        component_path = GaugePath(component_record["path"])
        endpoint_unwrapped = time + component_path.displacement[3]
        lo, hi = component_path.time_bounds
        if path_boundary == "open" and (
                time + lo < 0 or time + hi >= cfg["lattice"][3]):
            expected_pairs = []
        else:
            right_time = endpoint_unwrapped % cfg["lattice"][3]
            right_coordinates = np.asarray(right_reader.read(
                time=_physical_time_coordinate(right_reader.spec, right_time), select={"point": right_labels}))
            right_lookup = {tuple(coordinate): index
                            for index, coordinate in enumerate(right_coordinates)}
            displacement = component_path.displacement[:3]
            translated = (left_coordinates + np.asarray(displacement)) % spatial_size
            expected_pairs = [(left, right_lookup[tuple(coordinate)])
                              for left, coordinate in enumerate(translated)
                              if tuple(coordinate) in right_lookup]
        for momentum in range(len(axes["momentum"])):
            block = sparse_blocks[component][momentum]
            if block.point_shape != (len(left_labels), len(right_labels)):
                raise ValueError("P/P numeric block has an incompatible point-pair shape")
            rows = np.asarray(block.rows)
            columns = np.asarray(block.columns)
            if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
                    or columns.shape != rows.shape
                    or np.any(rows < 0) or np.any(rows >= len(axes["point_left"]))
                    or np.any(columns < 0) or np.any(columns >= len(right_labels))
                    or list(zip(rows.tolist(), columns.tolist())) != expected_pairs):
                raise ValueError("P/P kernel pairs differ from registered coordinates and path")
            values = np.asarray(block.values)
            if values.shape != (len(rows), 3, 3):
                raise ValueError("P/P pair values have an invalid color shape")
            row_values[component, momentum, rows] = values
    return row_values

def produce_point_elemental(gauge, basis, points, writer, *, points_right=None,
                               ne_block_size=32):
    """Write missing time rows from the existing V/P or rectangular P/P kernel."""
    output = writer.spec
    selection = dataset_time_selection(output)
    product_by_kind = {"elemental_v2p": "v2p", "elemental_p2v": "p2v",
                       "elemental_p2p_row_sparse": "p2p"}
    try:
        product = product_by_kind[output.document["kind"]]
    except KeyError as exc:
        raise ValueError("writer is not a registered point elemental product") from exc
    current_schema = output.document["definition"].get("directed_current_schema")
    temporal_path_schema = output.document["definition"].get("temporal_path_schema")
    path_boundary = output.document["definition"].get("path_boundary")
    if current_schema is not None and temporal_path_schema is not None:
        raise ValueError("point elemental output cannot combine current and temporal-path schemas")
    if temporal_path_schema is not None and temporal_path_schema != TEMPORAL_POINT_PATH_SCHEMA:
        raise ValueError("unsupported temporal point-path schema")
    if current_schema is not None:
        validate_directed_point_current_spec(output)
    axes = {name: output.coordinates(name) for name in output.axis_names}
    mode_axis = "mode_left" if product == "v2p" else "mode_right"
    point_axis = {"v2p": "point_right", "p2v": "point_left",
                  "p2p": "point_left"}[product]
    basis_required = product in {"v2p", "p2v"}
    if basis_required != (basis is not None):
        raise ValueError("writer basis input does not match its point elemental sector")
    if product == "v2p":
        right_reader = points_right or points
        right_labels = axes["point_right"]
        point_document_spec = points.spec
        point_document_id = points.data_id
        distinct_right = right_reader.data_id != point_document_id
        expected = point_elemental_dataset_document(
            gauge.spec, basis.spec if basis_required else None, point_document_spec,
            product=product, gauge_id=gauge.data_id,
            basis_id=basis.data_id if basis_required else None,
            points_id=point_document_id,
            components=axes["component"], momenta=axes["momentum"],
            modes=axes[mode_axis] if basis_required else (),
            point_labels=right_labels,
            points_right=right_reader.spec if distinct_right else None,
            points_right_id=right_reader.data_id if distinct_right else None,
            point_right_labels=right_labels,
            phase_convention=output.document["conventions"].get("phase"),
            current_schema=current_schema, temporal_path_schema=temporal_path_schema,
            path_boundary=path_boundary, selection=selection)
    elif product == "p2v":
        right_reader = None
        left_labels = axes["point_left"]
        expected = point_elemental_dataset_document(
            gauge.spec, basis.spec if basis_required else None, points.spec,
            product=product, gauge_id=gauge.data_id,
            basis_id=basis.data_id if basis_required else None,
            points_id=points.data_id,
            components=axes["component"], momenta=axes["momentum"],
            modes=axes[mode_axis], point_labels=left_labels,
            phase_convention=output.document["conventions"].get("phase"),
            current_schema=current_schema, temporal_path_schema=temporal_path_schema,
            path_boundary=path_boundary, selection=selection)
    else:
        right_reader = points_right or points
        left_labels = axes["point_left"]
        right_labels = output.document["definition"].get(
            "point_right_labels", left_labels)
        distinct_right = right_reader.data_id != points.data_id
        expected = point_elemental_dataset_document(
            gauge.spec, None, points.spec, product=product,
            gauge_id=gauge.data_id, basis_id=None,
            points_id=points.data_id,
            components=axes["component"], momenta=axes["momentum"],
            point_labels=left_labels,
            points_right=right_reader.spec if distinct_right else None,
            points_right_id=right_reader.data_id if distinct_right else None,
            point_right_labels=right_labels,
            phase_convention=output.document["conventions"].get("phase"),
            current_schema=current_schema, temporal_path_schema=temporal_path_schema,
            path_boundary=path_boundary, selection=selection)
    expected_spec = DatasetSpec(expected)
    if current_schema is not None:
        validate_directed_point_current_spec(expected_spec)
    if expected_spec.definition_id != output.definition_id:
        raise ValueError("point elemental writer differs from actual input IDs, views or conventions")
    if not writer.missing:
        return

    cfg = gauge.spec.document["configuration"]
    if product == "v2p":
        right_handle = _OrderedDatasetPointHandle(right_reader, right_labels)
        left_handle = None
        used_np_left = None
        used_np_right = len(right_labels)
    elif product == "p2v":
        left_handle = _OrderedDatasetPointHandle(points, left_labels)
        right_handle = None
        used_np_left = len(left_labels)
        used_np_right = None
    else:
        left_handle = _OrderedDatasetPointHandle(points, left_labels)
        right_handle = _OrderedDatasetPointHandle(right_reader, right_labels)
        used_np_left = len(left_labels)
        used_np_right = len(right_labels)
    generator = ElementalGenerator(cfg["lattice"], DatasetFieldHandle(gauge),
        DatasetFieldHandle(basis) if basis_required else None,
        momentum_list=axes["momentum"], usedNe=len(axes[mode_axis]) if basis_required else None,
        usedNp=used_np_left, usedNp_right=used_np_right,
        paths=[GaugePath(c["path"]) for c in axes["component"]],
        path_version="v2", point=left_handle, point_right=right_handle,
        path_boundary=path_boundary,
        phase_convention=output.document["conventions"]["phase"],
        ne_block_size=ne_block_size, selection=selection).load(cfg["configuration"])
    for position in writer.missing:
        time = (output.coordinates("time")[position] if selection is None
                else selection.physical_time(position))
        value = generator.calc_path_blocks(time, (product,))[product]
        if product == "p2p":
            value = _dense_point_pair_row(
                value, time=time, points=points, right_reader=right_reader,
                left_labels=left_labels, right_labels=right_labels,
                axes=axes, cfg=cfg, path_boundary=path_boundary)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("nonfinite point elemental values cannot be published")
        writer.write(position, value)
        # V/P and P/V refer to the generator's reusable workspace buffer; P/P
        # owns a newly assembled dense row. The synchronous writer has consumed
        # either value before the next physical-time kernel call.
        del value

def project_elemental(raw, writer):
    """Recombine one stored-time row using the existing public elemental kernel.

    Input is borrowed. Only writer.missing blocks are written, synchronously;
    publication and registration belong to the calling runtime. One requested
    time of raw paths plus the projected result/coefficient matrix is materialized.
    """
    output = writer.spec
    definition = output.document["definition"]
    rows = definition["rows"]
    if definition.get("catalog") is not None:
        from lqcdweave.contracts.elemental_catalog import stored_catalog_table
        rows = stored_catalog_table(definition["catalog"], rows)
    requested, expected = elemental_projection_documents(raw.spec, rows,
                                                         raw_id=raw.data_id)
    if DatasetSpec(expected).definition_id != output.definition_id:
        raise ValueError("projected writer definition differs from actual raw input view")
    components = requested.coordinates("component")
    positions = {tuple(component["path"]): i for i, component in enumerate(components)}
    combinations = [[(complex(*term["coefficient"]), positions[tuple(term["path"])])
                     for term in row["terms"]] for row in definition["rows"]]
    for position in writer.missing:
        time = output.coordinates("time")[position]
        values = np.stack([raw.read(time=time, component=component) for component in components])
        projected = re_combine(values, combinations, 0)
        del values
        if not np.all(np.isfinite(projected)):
            raise FloatingPointError("nonfinite projected elemental values cannot be published")
        writer.write(position, projected)
        del projected

def produce_current(gauge, basis, writer):
    """Borrow registered fields, calculate and synchronously save missing times.

    The generator owns a prepared all-time gauge and one-time output/field
    scratch. Storage/runtime owns publication, registration and recovery.
    The complete writer definition must match the actual parent input views.
    """
    output = writer.spec
    selection = dataset_time_selection(output)
    _, document = directed_current_dataset_document(gauge.spec, basis.spec,
        gauge_id=gauge.data_id, basis_id=basis.data_id,
        momentum_list=output.coordinates("momentum"),
        boundary=output.document["definition"]["path_boundary"], selection=selection)
    if DatasetSpec(document).definition_id != output.definition_id:
        raise ValueError("directed current writer definition differs from actual input views")
    if not writer.missing:
        return
    cfg = gauge.spec.document["configuration"]
    generator = ElementalGenerator(cfg["lattice"], DatasetFieldHandle(gauge),
        DatasetFieldHandle(basis), momentum_list=output.coordinates("momentum"),
        paths=[(direction,) for direction in range(8)], path_version="v2",
        path_boundary=document["definition"]["path_boundary"],
        phase_convention="midpoint", array_module=np, selection=selection).load(cfg["configuration"])
    for position in writer.missing:
        time = (output.coordinates("time")[position] if selection is None
                else selection.physical_time(position))
        result = generator.calc_path_blocks(time, ("v2v",))["v2v"]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite directed current values cannot be published")
        writer.write(position, result)
        # DatasetWriter.write consumes the row synchronously. Do not retain a
        # completed time row while the next full row is allocated.
        del result
