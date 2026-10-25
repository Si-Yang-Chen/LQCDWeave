"""Focused contract tests for sparse coordinate-batch source ownership."""

from dataclasses import replace
from collections.abc import Mapping
from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, BatchSpec, BatchTensorBinding, ContractionProblem, ContractionTerm,
    DiagramBindings, DiagramInputs, ExpandedCorrelator, OutputNode, OutputRecipe,
    PhysicalSlabBinding, PlannedContraction, SourcePreparation, SourceSpec, SparseBridgeSpec,
    StorageBlock, TensorAxisMetadata, TensorSpec, TermContribution,
    describe_coordinates, group_coordinate_batches,
)
from lqcdweave.contraction.execution.operations.sparse_bridge import (
    validate_sparse_bridge_parts,
)
from lqcdweave.contraction.execution.domain import BoundContractionExecution
from lqcdweave.contraction.execution.api import audit_execution_memory
from lqcdweave.contraction.expansion.batching import (
    BatchDiagramBinding, _BatchTaskLoader, _collect_sparse_catalog,
    _auto_batch_widths, _sparse_group_estimate,
)
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge_batch,
)
from lqcdweave.contraction.execution.scene_runtime import (
    _coordinate_batch_sparse_workspace_bytes, _model_features,
)
from lqcdweave.contraction.planning.api import (
    prepare_coordinate_batch_workspace, coordinate_batch_sparse_workspace_bytes,
)
from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.runtime.scheduler import execute_request


class _SparseLaneLoader:
    def __init__(self, record, gamma, coefficient):
        self.record = record
        self.gamma = gamma
        self.coefficient = coefficient
        self.reads = []
        self.materializations = []
        self.closes = []

    def read_block(self, key, block):
        self.reads.append((key, block))
        return self.record

    def materialize(self, key, source, blocks):
        record, = blocks.values()
        self.materializations.append(record)
        return ((self.gamma, self.coefficient, record),)

    def close_block(self, key, block):
        self.closes.append((key, block))

    def close_all(self):
        return None


class _FixedSparseBatchProvider:
    def __init__(self, left_by_time, right_by_time, lane_data):
        self.left_by_time = left_by_time
        self.right_by_time = right_by_time
        self.lane_data = lane_data
        self.slabs = {
            "left": self._slab("left", left_by_time),
            "right": self._slab("right", right_by_time),
        }

    @staticmethod
    def _slab(role, by_time):
        identity = ("sparse-batch-test", role)

        def describe(times):
            return TensorAxisMetadata(
                ("coordinate_batch", "spin", "point", "color"),
                (len(times), 2, 4, 3), np.complex128,
                f"{role}-physical", f"{role}-ordered-view",
            )

        def read(times):
            return np.stack([by_time[time] for time in times], axis=0)

        return PhysicalSlabBinding(identity, describe, read)

    def describe(self, correlator, *, bindings, coordinates, batch):
        time = coordinates.points[0].values[0][1]
        (left_key, right_key, record_key, block, raw_meta, _raw,
         _gamma, _coefficient, loader) = self.lane_data[time]
        left = self.left_by_time[time]
        right = self.right_by_time[time]
        dense_sources = {
            left_key: SourceSpec(f"left-{time}", left.nbytes,
                                 host_region_bytes=left.nbytes,
                                 snapshot_bytes=left.nbytes),
            right_key: SourceSpec(f"right-{time}", right.nbytes,
                                  host_region_bytes=right.nbytes,
                                  snapshot_bytes=right.nbytes),
        }
        problem = ContractionProblem(
            {
                left_key: TensorSpec((2, 4, 3), np.complex128),
                right_key: TensorSpec((2, 4, 3), np.complex128),
            },
            (ContractionTerm((left_key, right_key), "spc,qrd->"),),
            {
                **dense_sources,
                record_key: SourceSpec(
                    f"raw-record-{time}", block.read_bytes,
                    host_region_bytes=block.host_bytes, metadata=raw_meta,
                ),
            },
            sparse_bridges=(SparseBridgeSpec(
                left_key, right_key, record_key, 4, raw_meta["nnz"],
                left_axes=(0, 1, 2), right_axes=(0, 1, 2), scene=0,
                reverse_operands=True, used_np_right=4,
            ),),
        )
        return DiagramInputs(
            problem, {f"sparse-{time}": block},
            {
                left_key: SourcePreparation(()),
                right_key: SourcePreparation(()),
                record_key: SourcePreparation((f"sparse-{time}",), dtype="complex128"),
            },
            loader, coordinates, batch, ("p2p-term",),
            tensor_requests={left_key: None, right_key: None},
            batch_bindings={
                left_key: BatchTensorBinding(
                    ("left",), (self.slabs["left"],), (time,), (lambda value: value,),
                    topology_key=("P/P",),
                ),
                right_key: BatchTensorBinding(
                    ("right",), (self.slabs["right"],), (time,), (lambda value: value,),
                    topology_key=("P/P",),
                ),
            },
            batch_topology=("sparse-P/P", "reverse-bridge"),
        )


class _ThreeBridgeMetadataProvider(_FixedSparseBatchProvider):
    """Give each fixed lane three records in one scene, without reading them."""

    def describe(self, correlator, *, bindings, coordinates, batch):
        item = super().describe(
            correlator, bindings=bindings, coordinates=coordinates, batch=batch,
        )
        bridge, = item.problem.sparse_bridges
        source_key = bridge.record
        source = item.problem.sources[source_key]
        source_map = dict(item.problem.sources)
        preparation_map = dict(item.preparations)
        block_map = dict(item.blocks)
        bridges = [bridge]
        for ordinal in (1, 2):
            record_key = f"{source_key}-{ordinal}"
            block_key = f"sparse-{coordinates.points[0].key}-{ordinal}"
            block = StorageBlock(
                f"record-dataset-{coordinates.points[0].values[0][1]}-{ordinal}",
                "full", source.region_bytes,
                host_bytes=source.host_region_bytes, read_kind="p2p",
            )
            source_map[record_key] = replace(source, region=f"{source.region}-{ordinal}")
            preparation_map[record_key] = SourcePreparation(
                (block_key,), dtype="complex128",
            )
            block_map[block_key] = block
            bridges.append(replace(bridge, record=record_key))
        problem = replace(
            item.problem, sources=source_map, sparse_bridges=tuple(bridges),
        )
        return replace(
            item, problem=problem, blocks=block_map,
            preparations=preparation_map,
        )


class _FourPartBudgetProvider:
    """Two-lane fixture for comparing P admission with X known array scratch."""

    def __init__(self, *, same_point=False):
        self.same_point = bool(same_point)
        rng = np.random.default_rng(204)
        self.left_by_time = {
            t: rng.normal(size=(4, 1, 3)) + 1j * rng.normal(size=(4, 1, 3))
            for t in range(2)
        }
        self.right_by_time = {
            t: rng.normal(size=(4, 1, 3)) + 1j * rng.normal(size=(4, 1, 3))
            for t in range(2)
        }
        self.records = {}
        self.gammas = {}
        self.coefficients = {}
        self.loaders = {}
        self.slab_reads = {"left": 0, "right": 0}
        self.slabs = {}
        for role, table in (("left", self.left_by_time), ("right", self.right_by_time)):
            def describe(times, role=role):
                return TensorAxisMetadata(
                    ("coordinate_batch", "spin", "point", "color"),
                    (len(times), 4, 1, 3), np.complex128,
                    f"{role}-physical", f"{role}-ordered-view",
                )

            def read(times, table=table, role=role):
                self.slab_reads[role] += 1
                return np.stack([table[time] for time in times], axis=0)

            self.slabs[role] = PhysicalSlabBinding(
                ("p4-budget-physical-slab", role), describe, read,
            )
        for time in range(2):
            indices = np.asarray(((0, 0),), dtype=np.int64)
            values = (rng.normal(size=(1, 3, 3))
                      + 1j * rng.normal(size=(1, 3, 3))).astype(np.complex128)
            record = {"type": "sparse", "indices": indices, "values": values}
            record_bytes = int(indices.nbytes + values.nbytes)
            self.assert_record_size(record_bytes)
            self.records[time] = record
            self.gammas[time] = tuple(
                rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
                for _ in range(4)
            )
            self.coefficients[time] = tuple(
                complex(0.75 + part, -0.25 * time) for part in range(4)
            )
            self.loaders[time] = _FourPartLaneLoader(
                record, self.gammas[time], self.coefficients[time],
            )

    @staticmethod
    def assert_record_size(record_bytes):
        if record_bytes != 160:
            raise AssertionError(f"unexpected p4 fixture record size: {record_bytes}")

    def describe(self, correlator, *, bindings, coordinates, batch):
        time = coordinates.points[0].values[0][1]
        left_key, right_key, record_key = (
            f"left-{time}", f"right-{time}", f"record-{time}",
        )
        left = self.left_by_time[time]
        right = self.right_by_time[time]
        record = self.records[time]
        raw_bytes = int(record["indices"].nbytes + record["values"].nbytes)
        raw_metadata = {
            "family": "p2p-record", "kind": "sparse", "point_shape": (1, 1),
            "nnz": 1, "value_dtype": "complex128", "index_dtype": "int64",
            "record_bytes": raw_bytes, "backend": "numpy", "anchor_time": time,
        }
        block = StorageBlock(
            f"p4-record-{time}", "full", raw_bytes,
            host_bytes=raw_bytes, read_kind="p2p",
        )
        problem = ContractionProblem(
            {
                left_key: TensorSpec((4, 1, 3), np.complex128),
                right_key: TensorSpec((4, 1, 3), np.complex128),
            },
            (ContractionTerm(
                (left_key, right_key),
                "spc,qpc->" if self.same_point else "spc,qrd->",
            ),),
            {
                left_key: SourceSpec(
                    f"left-source-{time}", left.nbytes,
                    host_region_bytes=left.nbytes, snapshot_bytes=left.nbytes,
                ),
                right_key: SourceSpec(
                    f"right-source-{time}", right.nbytes,
                    host_region_bytes=right.nbytes, snapshot_bytes=right.nbytes,
                ),
                record_key: SourceSpec(
                    f"p4-raw-{time}", raw_bytes, host_region_bytes=raw_bytes,
                    metadata=raw_metadata,
                ),
            },
            sparse_bridges=(SparseBridgeSpec(
                left_key, right_key, record_key, 1, 1,
                left_axes=(0, 1, 2), right_axes=(0, 1, 2), scene=0,
                same_point=self.same_point, part_count=4, used_np_right=1,
            ),),
            single_read_regions=True,
        )
        return DiagramInputs(
            problem, {f"raw-block-{time}": block},
            {
                left_key: SourcePreparation(()),
                right_key: SourcePreparation(()),
                record_key: SourcePreparation(
                    (f"raw-block-{time}",), dtype="complex128",
                ),
            },
            self.loaders[time], coordinates, batch, ("p4-term",),
            tensor_requests={left_key: None, right_key: None},
            batch_bindings={
                left_key: BatchTensorBinding(
                    ("role", "left"), (self.slabs["left"],), (time,),
                    (lambda value: value,), topology_key=("P/P",),
                ),
                right_key: BatchTensorBinding(
                    ("role", "right"), (self.slabs["right"],), (time,),
                    (lambda value: value,), topology_key=("P/P",),
                ),
            },
            batch_topology=("p4-P/P",),
        )


class _FourPartLaneLoader:
    def __init__(self, record, gammas, coefficients):
        self.record = record
        self.gammas = gammas
        self.coefficients = coefficients
        self.reads = []
        self.materializations = []
        self.closes = []

    def read_block(self, key, block):
        self.reads.append((key, block))
        return self.record

    def materialize(self, key, source, blocks):
        record, = blocks.values()
        self.materializations.append(record)
        return tuple(
            (gamma, coefficient, record)
            for gamma, coefficient in zip(self.gammas, self.coefficients, strict=True)
        )

    def close_block(self, key, block):
        self.closes.append((key, block))

    def close_all(self):
        return None


class _RecordingLoader:
    def __init__(self, name):
        self.name = name
        self.reads = []
        self.materialized = []
        self.closed = []

    def read_block(self, key, block):
        self.reads.append((key, block))
        return {"physical": (block.dataset, block.selection)}

    def materialize(self, key, source, blocks):
        value, = blocks.values()
        self.materialized.append((key, value))
        return (self.name, value)

    def close_block(self, key, value):
        self.closed.append((key, value))


def _sparse_lane(index, *, block, raw_nnz=3):
    source_key = f"record-{index}"
    block_key = f"source-block-{index}"
    loader = _RecordingLoader(f"lane-{index}")
    metadata = {
        "family": "p2p-record",
        "kind": "sparse",
        "point_shape": (4, 5),
        "nnz": raw_nnz,
        "value_dtype": "complex128",
        "index_dtype": "int64",
        "record_bytes": block.read_bytes,
        "backend": "numpy",
        # Lane-specific provenance is not part of the raw physical block.
        "anchor_time": index,
        "point_pool_ids": {"point_left": f"left-{index}"},
    }
    source = SourceSpec(
        f"region-{index}", block.read_bytes,
        host_region_bytes=block.host_bytes, snapshot_bytes=None,
        metadata=metadata,
    )
    bridge = SparseBridgeSpec(
        "left", "right", source_key, 4, 3,
        scene=0, used_np_right=5,
    )
    lane = SimpleNamespace(
        problem=SimpleNamespace(sources={source_key: source}),
        preparations={source_key: SourcePreparation((block_key,), workspace_host_bytes=32 + index)},
        blocks={block_key: block},
        loader=loader,
    )
    return lane, bridge, source_key


def _collect(lanes_and_bridges):
    lanes = tuple(item[0] for item in lanes_and_bridges)
    group = {
        "scene": 0,
        "lanes": tuple((index, item[1]) for index, item in enumerate(lanes_and_bridges)),
    }
    catalog = {"blocks": {}, "block_by_identity": {}}
    _collect_sparse_catalog(lanes, {"sparse_bridges": (group,)}, catalog)
    return lanes, catalog


class SparsePointSinkBatchCatalogTests(unittest.TestCase):
    @staticmethod
    def _four_part_path(memory_limit, *, same_point=False):
        coordinates, batch = describe_coordinates(
            time_axes={"scan": (0, 1)}, batch={"axis": "scan", "size": 2},
        )
        provider = _FourPartBudgetProvider(same_point=same_point)
        inputs = BatchDiagramBinding(provider).describe(
            1.0, bindings=DiagramBindings(), coordinates=coordinates, batch=batch,
        )
        expanded = ExpandedCorrelator(
            1.0, inputs,
            OutputRecipe(
                OutputNode("term", TermContribution("p4-term")),
                ("p4-term",), coordinates,
            ),
            DiagramBindings(), coordinates, batch,
        )
        path = find_contraction_path(
            expanded, memory_limit=memory_limit, optimize="none",
        )
        return provider, expanded, path

    def test_p_known_workspace_rejects_low_budget_before_any_source_io(self):
        provider, expanded, path = self._four_part_path(1 << 28)
        self.assertTrue(path.plan.feasible)
        adjusted, reserve = prepare_coordinate_batch_workspace(path.plan)
        self.assertIs(adjusted, path.plan)
        self.assertEqual(reserve, 0)
        self.assertGreater(path.plan.peak_host_bytes, 10800)
        with self.assertRaises(MemoryError):
            find_contraction_path(expanded, memory_limit=10800, optimize="none")
        self.assertEqual(provider.slab_reads, {"left": 0, "right": 0})
        for loader in provider.loaders.values():
            self.assertEqual(loader.reads, [])
            self.assertEqual(loader.materializations, [])
            self.assertEqual(loader.closes, [])

    def test_public_execution_memory_audit_rejects_incomplete_saved_envelope_without_io(self):
        provider, _expanded, path = self._four_part_path(1 << 28)
        # Model an older/incomplete descriptor by removing only its sparse scratch.
        memory = tuple(replace(point, workspace_host_bytes=0)
                       if point.phase == "sparse_bridge" else point
                       for point in path.plan.memory)
        incomplete = replace(path.plan, memory=memory)
        path = replace(path, plan=incomplete, plan_identity=id(incomplete), batch_paths=())
        original_reserve = incomplete.reserved_host_bytes
        audit = audit_execution_memory(path, backend=np)
        self.assertFalse(audit["feasible"])
        self.assertEqual(audit["peak_host_bytes"], incomplete.peak_host_bytes)
        self.assertGreater(audit["batches"][0]["x_workspace_correction_host_bytes"], 0)
        self.assertIs(path.plan, incomplete)
        self.assertIs(path.plan.memory, memory)
        self.assertEqual(path.plan.reserved_host_bytes, original_reserve)
        self.assertEqual(provider.slab_reads, {"left": 0, "right": 0})
        for loader in provider.loaders.values():
            self.assertEqual(loader.reads, [])
            self.assertEqual(loader.materializations, [])
            self.assertEqual(loader.closes, [])

    def test_public_execution_memory_audit_uses_maximum_across_batch_paths(self):
        provider, _expanded, path = self._four_part_path(1 << 28)
        first_plan = path.plan.with_reserves(
            reserved_host_bytes=path.plan.reserved_host_bytes + 123,
        )
        second_plan = first_plan.with_reserves(
            reserved_host_bytes=first_plan.reserved_host_bytes + 4096,
        )
        first = replace(path, plan=first_plan, plan_identity=id(first_plan), batch_paths=())
        second = replace(path, plan=second_plan, plan_identity=id(second_plan), batch_paths=())
        multi_batch = replace(
            path, plan=first_plan, plan_identity=id(first_plan),
            batch_paths=(first, second),
        )
        original = (
            path.plan, path.plan_identity, first_plan.reserved_host_bytes,
            first_plan.peak_host_bytes, second_plan.reserved_host_bytes,
            second_plan.peak_host_bytes,
        )

        audit = audit_execution_memory(multi_batch, backend=np)

        self.assertTrue(audit["feasible"])
        self.assertEqual(len(audit["batches"]), 2)
        first_record, second_record = audit["batches"]
        self.assertEqual(first_record["batch_index"], 0)
        self.assertEqual(second_record["batch_index"], 1)
        self.assertEqual(
            first_record["reserved_host_bytes"],
            first_plan.reserved_host_bytes
            + first_record["x_workspace_correction_host_bytes"],
        )
        self.assertEqual(
            second_record["reserved_host_bytes"],
            second_plan.reserved_host_bytes
            + second_record["x_workspace_correction_host_bytes"],
        )
        self.assertEqual(
            second_record["corrected_peak_host_bytes"],
            first_record["corrected_peak_host_bytes"] + 4096,
        )
        self.assertEqual(
            audit["peak_host_bytes"], second_record["corrected_peak_host_bytes"],
        )
        self.assertEqual(audit["peak_device_bytes"], 0)
        with self.assertRaises(TypeError):
            audit["peak_host_bytes"] = 0
        with self.assertRaises(TypeError):
            first_record["corrected_peak_host_bytes"] = 0
        self.assertEqual(
            original,
            (path.plan, path.plan_identity, first_plan.reserved_host_bytes,
             first_plan.peak_host_bytes, second_plan.reserved_host_bytes,
             second_plan.peak_host_bytes),
        )
        self.assertEqual(provider.slab_reads, {"left": 0, "right": 0})
        for loader in provider.loaders.values():
            self.assertEqual(loader.reads, [])
            self.assertEqual(loader.materializations, [])
            self.assertEqual(loader.closes, [])

    def test_public_execution_memory_audit_preserves_empty_assembly_peak(self):
        expanded = SimpleNamespace(inputs=SimpleNamespace(
            assembly_host_bytes=123, assembly_device_bytes=45,
        ))
        empty = PlannedContraction(
            expanded, None, None, {"operations": 0}, 0.0, "empty",
            memory_limit=133, host_memory_limit=133,
            reserved_host_bytes=10, reserved_device_bytes=5,
        )

        audit = audit_execution_memory(empty, backend=np)

        self.assertEqual(audit["peak_host_bytes"], 133)
        self.assertEqual(audit["peak_device_bytes"], 50)
        self.assertEqual(audit["reserved_host_bytes"], 10)
        self.assertEqual(audit["reserved_device_bytes"], 5)
        self.assertTrue(audit["feasible"])
        self.assertEqual(audit["batches"], ())

        below_cap = replace(empty, memory_limit=132, host_memory_limit=132)
        self.assertFalse(audit_execution_memory(below_cap, backend=np)["feasible"])

    def test_public_execution_memory_audit_leaves_unmarked_plan_unchanged(self):
        _provider, _expanded, path = self._four_part_path(1 << 28)
        sources = dict(path.plan.state.problem.sources)
        for key, source in tuple(sources.items()):
            metadata = source.metadata
            if (isinstance(metadata, Mapping)
                    and metadata.get("family") == "p2p-coordinate-batch"):
                sources[key] = replace(
                    source, metadata=metadata["lane_sources"][0]["metadata"],
                )
        ordinary_problem = replace(path.plan.state.problem, sources=sources)
        ordinary_state = replace(path.plan.state, problem=ordinary_problem)
        ordinary_plan = replace(path.plan, state=ordinary_state)
        ordinary_path = replace(
            path, plan=ordinary_plan, plan_identity=id(ordinary_plan), batch_paths=(),
        )

        audit = audit_execution_memory(ordinary_path, backend=np)

        self.assertEqual(audit["batches"][0]["x_workspace_correction_host_bytes"], 0)
        self.assertEqual(audit["peak_host_bytes"], ordinary_plan.peak_host_bytes)
        self.assertEqual(audit["peak_device_bytes"], ordinary_plan.peak_device_bytes)
        self.assertTrue(audit["feasible"])

    def test_prepared_workspace_keeps_four_part_cpu_result(self):
        provider, expanded, path = self._four_part_path(1 << 28)
        adjusted, reserve = prepare_coordinate_batch_workspace(path.plan)
        self.assertEqual(reserve, 0)
        self.assertIs(adjusted, path.plan)
        self.assertTrue(adjusted.feasible)
        report = execute_request(
            BoundContractionExecution(expanded, path), backend=np, measure=False,
        )
        actual = {
            item.coordinate_key: complex(np.asarray(item.value).item())
            for item in report.term_values
        }
        expected = {}
        for time in range(2):
            total = 0j
            record = provider.records[time]
            for gamma, coefficient in zip(
                provider.gammas[time], provider.coefficients[time], strict=True,
            ):
                total += coefficient * np.einsum(
                    "ic,iq,cd,qd->",
                    provider.left_by_time[time][:, 0, :], gamma,
                    record["values"][0], provider.right_by_time[time][:, 0, :],
                    optimize=False,
                )
            expected[str(time)] = total
        np.testing.assert_allclose(
            np.asarray([actual[str(time)] for time in range(2)]),
            np.asarray([expected[str(time)] for time in range(2)]),
            rtol=2e-13, atol=2e-13,
        )
        self.assertEqual(provider.slab_reads, {"left": 1, "right": 1})
        for loader in provider.loaders.values():
            self.assertEqual(len(loader.reads), 1)
            self.assertEqual(len(loader.materializations), 1)
            self.assertEqual(len(loader.closes), 1)

    def test_x_same_point_reserve_counts_filtered_chunk_owners(self):
        provider, expanded, path = self._four_part_path(
            1 << 28, same_point=True,
        )
        action = next(
            action for action in path.plan.state.actions
            if action.kind == "sparse_bridge" and not action.reused
        )
        actual_workspace = _coordinate_batch_sparse_workspace_bytes(
            path.plan.state, action,
        )
        # Non-filtered p4 scratch is 7488 bytes. C=8 and F=1 retain eight
        # complete 3x3 c128 value/pair bases plus two one-byte masks.
        self.assertEqual(actual_workspace, 7488 + 8 * (16 + 9 * 16) + 2)
        self.assertEqual(provider.slab_reads, {"left": 0, "right": 0})

    def test_same_physical_view_is_read_once_across_lane_loaders(self):
        physical = StorageBlock(
            "dataset:one", "ordered-view:0,2,3", 256,
            host_bytes=256, read_kind="p2p",
        )
        lane0 = _sparse_lane(0, block=physical)
        lane1 = _sparse_lane(1, block=physical)
        lanes, catalog = _collect((lane0, lane1))

        entries = catalog["sparse_by_scene"][0]
        self.assertEqual(len(catalog["sparse_blocks"]), 1)
        self.assertEqual(len({entry["block_key"] for entry in entries}), 1)
        self.assertIsNot(entries[0]["loader"], entries[1]["loader"])
        self.assertEqual(
            tuple(entry["preparation"].workspace_host_bytes for entry in entries),
            (32, 33),
        )

        loader = _BatchTaskLoader(catalog)
        loader.register(
            0, specs={}, lane_specs=(), role_entries={}, dynamic_roles=(),
            sparse_records={"combined-record": {
                "point_keys": ("point-0", "point-1"), "lanes": entries,
            }},
        )
        loader.register_demands(0, (entries[0]["block_key"],))
        with loader.activate(0):
            block_key = entries[0]["block_key"]
            raw = loader.read_block(block_key, physical)
            self.assertIs(loader.read_block(block_key, physical), raw)
            materialized = loader.materialize(
                "combined-record", None, {block_key: raw},
            )
            self.assertEqual(materialized["layout"], "coordinate_batch_coo")
            self.assertIs(materialized["lanes"][0][1], raw)
            self.assertIs(materialized["lanes"][1][1], raw)
            loader.close_block(block_key, raw)

        self.assertEqual(len(lanes[0].loader.reads), 1)
        self.assertEqual(lanes[1].loader.reads, [])
        self.assertEqual(len(lanes[0].loader.closed), 1)
        self.assertEqual(lanes[1].loader.closed, [])
        self.assertEqual(len(lanes[0].loader.materialized), 1)
        self.assertEqual(len(lanes[1].loader.materialized), 1)

    def test_three_sparse_bridges_in_one_scene_fail_before_source_reads(self):
        rng = np.random.default_rng(20261004)
        left_by_time = {time: rng.normal(size=(2, 4, 3)).astype(np.complex128)
                        for time in range(2)}
        right_by_time = {time: rng.normal(size=(2, 4, 3)).astype(np.complex128)
                         for time in range(2)}
        lane_data = {}
        for time in range(2):
            indices = np.asarray(((0, 1),), dtype=np.int64)
            values = np.ones((1, 3, 3), dtype=np.complex128)
            record = {"type": "sparse", "indices": indices, "values": values}
            raw_bytes = int(indices.nbytes + values.nbytes)
            raw_meta = {
                "family": "p2p-record", "kind": "sparse", "point_shape": (4, 4),
                "nnz": 1, "value_dtype": "complex128", "index_dtype": "int64",
                "record_bytes": raw_bytes, "backend": "numpy",
            }
            block = StorageBlock(f"record-dataset-{time}", "full", raw_bytes,
                                 host_bytes=raw_bytes, read_kind="p2p")
            loader = _SparseLaneLoader(record, np.eye(2), 1 + 0j)
            lane_data[time] = (
                f"left-{time}", f"right-{time}", f"record-{time}",
                block, raw_meta, record, np.eye(2), 1 + 0j, loader,
            )
        coordinates, batch = describe_coordinates(
            time_axes={"scan": (0, 1)}, batch={"axis": "scan", "size": 2},
        )
        provider = _ThreeBridgeMetadataProvider(
            left_by_time, right_by_time, lane_data,
        )
        with self.assertRaisesRegex(NotImplementedError, "at most two"):
            BatchDiagramBinding(provider).describe(
                object(), bindings=DiagramBindings(), coordinates=coordinates,
                batch=batch,
            )
        for entry in lane_data.values():
            loader = entry[-1]
            self.assertEqual(loader.reads, [])
            self.assertEqual(loader.materializations, [])
            self.assertEqual(loader.closes, [])

    def test_same_identity_with_conflicting_raw_descriptor_fails_before_read(self):
        base = StorageBlock(
            "dataset:one", "ordered-view:0,2,3", 256,
            host_bytes=256, read_kind="p2p",
        )
        conflicts = (
            StorageBlock(
                "dataset:one", "ordered-view:0,2,3", 512,
                host_bytes=512, read_kind="p2p",
            ),
            base,
        )
        for index, conflict in enumerate(conflicts, start=1):
            with self.subTest(conflict=index):
                first = _sparse_lane(0, block=base)
                second = _sparse_lane(
                    1, block=conflict,
                    raw_nnz=(4 if index == 2 else 3),
                )
                with self.assertRaisesRegex(ValueError, "physical sparse block identity"):
                    _collect((first, second))
                self.assertEqual(first[0].loader.reads, [])
                self.assertEqual(second[0].loader.reads, [])

    def test_two_lane_kernel_matches_independent_coo_loop_without_cross_lane_terms(self):
        import numpy as np

        rng = np.random.default_rng(260930)
        left = rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2))
        right = rng.normal(size=(2, 2, 3, 2)) + 1j * rng.normal(size=(2, 2, 3, 2))
        lanes = []
        expected = np.zeros(2, dtype=np.complex128)
        for lane_index, pairs in enumerate((
            ((0, 1), (1, 2)),
            ((1, 0), (0, 2)),
        )):
            gamma = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            values = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
            coefficient = complex(0.4 + lane_index, -0.3 + 0.2 * lane_index)
            indices = np.asarray(pairs, dtype=np.int64)
            lanes.append(((gamma, coefficient, {
                "layout": "normalized-coo", "indices": indices, "values": values,
            }),))
            for row, (left_point, right_point) in enumerate(pairs):
                expected[lane_index] += coefficient * np.einsum(
                    "ic,ij,cd,jd->", left[lane_index, :, left_point, :],
                    gamma, values[row], right[lane_index, :, right_point, :],
                    optimize=False,
                )

        bridge = SparseBridgeSpec(
            "left", "right", "batch-record", 2, 4,
            left_axes=(-3, -2, -1), right_axes=(1, 2, 3),
            used_np_right=3,
        )
        actual = contract_p2p_bridge_batch(
            left, right,
            {"layout": "coordinate_batch_coo", "point_keys": ("a", "b"),
             "lanes": tuple(lanes)},
            bridge=bridge, equation="xspc,xqrd->x",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

        crossed = np.zeros(2, dtype=np.complex128)
        for lane_index in range(2):
            gamma, coefficient, record = lanes[1 - lane_index][0]
            for row, (left_point, right_point) in enumerate(record["indices"]):
                crossed[lane_index] += coefficient * np.einsum(
                    "ic,ij,cd,jd->", left[lane_index, :, left_point, :],
                    gamma, record["values"][row],
                    right[1 - lane_index, :, right_point, :], optimize=False,
                )
        self.assertGreater(float(np.max(np.abs(expected - crossed))), 1e-6)

    def test_multi_part_normalization_applies_bound_per_part(self):
        import numpy as np

        point_shape = (4, 4)
        record_bytes = 2 * (2 * np.dtype("int64").itemsize + 9 * 16)
        raw_metadata = {
            "family": "p2p-record", "kind": "sparse",
            "point_shape": point_shape, "nnz": 2,
            "value_dtype": "complex128", "index_dtype": "int64",
            "record_bytes": record_bytes, "backend": "numpy",
        }
        lane_sources = tuple({
            "metadata": raw_metadata,
            "materialized_point_shape": point_shape,
        } for _ in range(2))
        marker = {
            "family": "p2p-coordinate-batch",
            "coordinate_keys": ("lane-a", "lane-b"),
            "lane_nnz_upper": (2, 2),
            "point_shape": point_shape,
            "part_count": 2,
            "lane_sources": lane_sources,
        }
        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, 4,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
            scene=0, part_count=2, used_np_right=4,
        )
        problem = ContractionProblem(
            {
                "left": TensorSpec((2, 2, 4, 3), np.complex128),
                "right": TensorSpec((2, 2, 4, 3), np.complex128),
            },
            (ContractionTerm(("left", "right"), "xspc,xqpc->x"),),
            {
                "left": SourceSpec("left", 0), "right": SourceSpec("right", 0),
                "record": SourceSpec("sparse", 2 * record_bytes, metadata=marker),
            },
            sparse_bridges=(bridge,),
        )

        def part(offset):
            indices = np.asarray(((offset, 0), (offset + 1, 1)), dtype=np.int64)
            values = np.arange(18, dtype=np.float64).astype(np.complex128).reshape(2, 3, 3)
            gamma = np.asarray(((1, 1j), (-0.5j, 2)), dtype=np.complex128)
            return (gamma, 0.75 + 0.125j, {
                "type": "sparse", "indices": indices, "values": values,
            })

        materialized = {
            "layout": "coordinate_batch_coo",
            "point_keys": ("lane-a", "lane-b"),
            "lanes": ((part(0), part(2)), (part(0), part(2))),
        }
        normalized = validate_sparse_bridge_parts(problem, "record", materialized)
        self.assertEqual(tuple(len(lane) for lane in normalized["lanes"]), (2, 2))
        for lane in normalized["lanes"]:
            self.assertEqual(tuple(len(part[2]["indices"]) for part in lane), (2, 2))
            self.assertEqual(sum(len(part[2]["indices"]) for part in lane), 4)

        lower = part(0)
        lower = (np.asarray(lower[0], dtype=np.complex64),
                 np.complex64(0.75 + 0.125j), lower[2])
        lower_precision = validate_sparse_bridge_parts(problem, "record", {
            "layout": "coordinate_batch_coo",
            "point_keys": ("lane-a", "lane-b"),
            "lanes": ((lower,), (lower,)),
        })
        self.assertEqual(lower_precision["lanes"][0][0][0].dtype, np.dtype("complex64"))

        object_metadata = dict(raw_metadata, value_dtype="object")
        object_marker = dict(marker, lane_sources=tuple({
            "metadata": object_metadata,
            "materialized_point_shape": point_shape,
        } for _ in range(2)))
        object_sources = dict(problem.sources)
        object_sources["record"] = SourceSpec(
            "sparse", 2 * record_bytes, metadata=object_marker,
        )
        with self.assertRaisesRegex(ValueError, "raw values must use a numeric dtype"):
            ContractionProblem(
                problem.specs, problem.terms, object_sources,
                sparse_bridges=problem.sparse_bridges,
            )

        extended_dtype = np.dtype(np.longdouble)
        extended_result = np.result_type(extended_dtype, np.dtype(np.complex128))
        if extended_result.itemsize > np.dtype(np.complex128).itemsize:
            extended_specs = dict(problem.specs)
            extended_specs["left"] = TensorSpec((2, 2, 4, 3), extended_dtype)
            with self.assertRaisesRegex(ValueError, "endpoint 'left'.*complex128 result envelope"):
                ContractionProblem(
                    extended_specs, problem.terms, problem.sources,
                    sparse_bridges=problem.sparse_bridges,
                )
            extended_metadata = dict(raw_metadata, value_dtype=extended_dtype.name)
            extended_marker = dict(marker, lane_sources=tuple({
                "metadata": extended_metadata,
                "materialized_point_shape": point_shape,
            } for _ in range(2)))
            extended_sources = dict(problem.sources)
            extended_sources["record"] = SourceSpec(
                "sparse", 2 * record_bytes, metadata=extended_marker,
            )
            with self.assertRaisesRegex(ValueError, "raw values.*complex128 result envelope"):
                ContractionProblem(
                    problem.specs, problem.terms, extended_sources,
                    sparse_bridges=problem.sparse_bridges,
                )

            extended_gamma = (np.ones((2, 2), dtype=extended_dtype), 0.75 + 0.125j,
                              part(0)[2])
            with self.assertRaisesRegex(TypeError, "gamma exceeds the complex128 result envelope"):
                validate_sparse_bridge_parts(problem, "record", {
                    "layout": "coordinate_batch_coo",
                    "point_keys": ("lane-a", "lane-b"),
                    "lanes": ((extended_gamma,), (part(0),)),
                })

        too_many = part(0)
        bad_indices = np.asarray(((0, 0), (1, 1), (2, 2)), dtype=np.int64)
        bad_values = np.ones((3, 3, 3), dtype=np.complex128)
        bad_raw_metadata = dict(raw_metadata, nnz=3,
                                record_bytes=3 * (2 * 8 + 9 * 16))
        bad_marker = dict(marker, lane_sources=tuple({
            "metadata": bad_raw_metadata,
            "materialized_point_shape": point_shape,
        } for _ in range(2)))
        bad_problem = ContractionProblem(
            problem.specs, problem.terms,
            {**problem.sources, "record": SourceSpec(
                "sparse", 2 * bad_raw_metadata["record_bytes"], metadata=bad_marker,
            )},
            sparse_bridges=(bridge,),
        )
        with self.assertRaisesRegex(ValueError, "per-part descriptor bound"):
            validate_sparse_bridge_parts(bad_problem, "record", {
                "layout": "coordinate_batch_coo",
                "point_keys": ("lane-a", "lane-b"),
                "lanes": (((too_many[0], too_many[1], {
                    "type": "sparse", "indices": bad_indices, "values": bad_values,
                }),), (part(0),)),
            })

    def test_estimate_counts_all_lane_parts_and_zero_row_metadata(self):
        import numpy as np

        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, 2,
            left_axes=(0, 1, 2), right_axes=(0, 1, 2),
            scene=0, part_count=2, used_np_right=4,
        )
        block = StorageBlock("dataset", "selection", 100, host_bytes=100)
        raw = {
            "family": "p2p-record", "kind": "sparse", "point_shape": (4, 4),
            "nnz": 2, "value_dtype": "complex64", "index_dtype": "int64",
            "record_bytes": 176, "backend": "numpy",
        }
        entries = tuple({
            "lane_index": index, "bridge": bridge,
            "source": SourceSpec("record", 176, host_region_bytes=100, metadata=raw),
            "preparation": SourcePreparation(("shared",)), "block_key": "shared",
        } for index in range(2))
        contract = {
            "sparse_bridges": ({"scene": 0, "roles": (0, 1), "template": bridge},),
            "lane_specs": (
                TensorSpec((2, 4, 3), np.complex64),
                TensorSpec((2, 4, 3), np.complex64),
            ),
        }
        catalog = {
            "blocks": {"shared": block},
            "sparse_by_scene": {0: entries},
        }
        group = SimpleNamespace(point_indices=(0, 1))
        contract["sparse_bridges"] = (
            {**contract["sparse_bridges"][0], "filter_points": True},
        )

        # Manual phase bound: raw 100 + normalized lane/part COO (2*4*(16+72))
        # plus two gamma/scalar pairs per lane (2*2*(2*2*16+16)); then
        # two possible endpoint reshape copies (768), 8 packed rows (1408),
        # bounded retained filtered bases (884), gathers (768), contributions (128),
        # and output (32).  The 8 rows cross both lanes and both parts.
        self.assertEqual(_sparse_group_estimate(group, contract, catalog), 5112)

        empty_bridge = replace(bridge, nnz_upper=0)
        empty_raw = dict(raw, nnz=0, record_bytes=0)
        empty_entries = tuple({
            "lane_index": index, "bridge": empty_bridge,
            "source": SourceSpec("empty-record", 0, host_region_bytes=0, metadata=empty_raw),
            "preparation": SourcePreparation(("empty",)), "block_key": "empty",
        } for index in range(2))
        empty_contract = {
            "sparse_bridges": ({"scene": 0, "roles": (0, 1), "template": empty_bridge},),
            "lane_specs": contract["lane_specs"],
        }
        empty_catalog = {
            "blocks": {"empty": StorageBlock("empty", "selection", 0, host_bytes=0)},
            "sparse_by_scene": {0: empty_entries},
        }
        # No COO rows still retain four 2x2 gamma/scalar records, endpoint
        # reshape copies, and the lane result.
        self.assertEqual(_sparse_group_estimate(group, empty_contract, empty_catalog), 1120)

        two_color_contract = {
            **contract,
            "lane_specs": (
                TensorSpec((2, 4, 2), np.complex64),
                TensorSpec((2, 4, 2), np.complex64),
            ),
        }
        # The raw COO color block is 2x2 here; the estimate must use that
        # extent instead of assuming the physical QCD color size is always 3.
        self.assertEqual(_sparse_group_estimate(group, two_color_contract, catalog), 3560)

    def test_mixed_lane_value_promotion_limits_automatic_batch_width(self):
        coordinates, batch = describe_coordinates(
            time_axes={"scan": (0, 1)},
            batch={"axis": "scan", "size": "auto"},
        )
        raw_row_bytes = 2 * (2 * np.dtype("int64").itemsize + 9 * 8)
        lane_bounds = (2, 2)
        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, sum(lane_bounds),
            left_axes=(0, 1, 2), right_axes=(0, 1, 2),
            scene=0, part_count=2, used_np_right=4,
        )
        entries = []
        blocks = {}
        for lane_index, value_dtype in enumerate(("float64", "complex64")):
            raw = {
                "family": "p2p-record", "kind": "sparse",
                "point_shape": (4, 4), "nnz": 2,
                "value_dtype": value_dtype, "index_dtype": "int64",
                "record_bytes": raw_row_bytes, "backend": "numpy",
            }
            block_key = f"record-block-{lane_index}"
            blocks[block_key] = StorageBlock(
                f"dataset-{lane_index}", "full", raw_row_bytes,
                host_bytes=raw_row_bytes, read_kind="p2p",
            )
            entries.append({
                "lane_index": lane_index,
                "bridge": replace(bridge, nnz_upper=lane_bounds[lane_index]),
                "source": SourceSpec(
                    f"record-region-{lane_index}", raw_row_bytes,
                    host_region_bytes=raw_row_bytes, metadata=raw,
                ),
                "preparation": SourcePreparation((block_key,)),
                "block_key": block_key,
            })

        lane_specs = (
            TensorSpec((4, 4, 3), np.complex128),
            TensorSpec((4, 4, 3), np.complex128),
        )
        bridge_group = {
            "scene": 0, "roles": (0, 1), "template": bridge,
        }
        contract = {
            "sparse_bridges": (bridge_group,),
            "lane_specs": lane_specs,
            "role_signatures": (
                ("left-lane-0", "left-lane-1"),
                ("right-lane-0", "right-lane-1"),
            ),
            "first": SimpleNamespace(problem=SimpleNamespace(terms=(
                ContractionTerm(("left", "right"), "spc,qpc->"),
            ),)),
            "term_roles": ((0, 1),),
        }
        lanes = tuple(SimpleNamespace(batch_bindings={
            f"{role}-lane-{lane_index}": SimpleNamespace(workspace_host_bytes=0)
            for role in ("left", "right")
        }) for lane_index in range(2))
        catalog = {
            "blocks": blocks,
            "sparse_blocks": tuple(blocks),
            "read_workspace_by_block": {},
            "sparse_by_scene": {0: tuple(entries)},
        }
        groups_one = group_coordinate_batches(coordinates, BatchSpec(1, "scan"))
        groups_two = group_coordinate_batches(coordinates, BatchSpec(2, "scan"))
        one_lane_peak = max(
            _sparse_group_estimate(group, contract, catalog) for group in groups_one
        )
        two_lane_peak = _sparse_group_estimate(groups_two[0], contract, catalog)

        # The eight packed COO rows promote float64 + complex64 to complex128.
        # Relative to max(input itemsize), this adds 8*3*3*8 = 576 bytes.
        self.assertEqual(one_lane_peak, 5696)
        self.assertEqual(two_lane_peak, 11968)
        self.assertEqual(two_lane_peak - 11392, 576)

        # The old max(itemsize) estimate would fit width 2 under this cap;
        # the promoted pack correctly selects width 1.
        selected = _auto_batch_widths(
            coordinates, batch, lanes, contract, catalog,
            assembly_bytes=0, memory_limit=13200, host_memory_limit=None,
            reserved_host_bytes=0,
        )
        self.assertEqual(selected, (1,))

    def test_x_workspace_counts_mixed_lane_value_stack_promotion(self):
        lane_sources = []
        total_raw_bytes = 0
        for value_dtype in ("float64", "complex64"):
            raw_bytes = 32 * (
                2 * np.dtype("int64").itemsize + 3 * 3 * np.dtype(value_dtype).itemsize
            )
            total_raw_bytes += raw_bytes
            lane_sources.append({
                "metadata": {
                    "family": "p2p-record", "kind": "sparse",
                    "point_shape": (1, 1), "nnz": 32,
                    "value_dtype": value_dtype, "index_dtype": "int64",
                    "record_bytes": raw_bytes, "backend": "numpy",
                },
                "materialized_point_shape": (1, 1),
            })
        marker = {
            "family": "p2p-coordinate-batch",
            "coordinate_keys": ("lane-a", "lane-b"),
            "lane_nnz_upper": (32, 32),
            "lane_sources": tuple(lane_sources),
            "point_shape": (1, 1), "part_count": 1,
        }
        bridge = SparseBridgeSpec(
            "left", "right", "record", 1, 64,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
            scene=0, part_count=1, used_np_right=1,
        )
        state = SimpleNamespace(
            problem=SimpleNamespace(
                backend="numpy", sparse_bridges=(bridge,),
                sources={"record": SourceSpec(
                    "mixed-batch-record", total_raw_bytes, metadata=marker,
                )},
            ),
            catalog={
                "left": TensorSpec((2, 1, 1, 3), np.complex128),
                "right": TensorSpec((2, 1, 1, 3), np.complex128),
            },
        )
        action = Action(
            "sparse_bridge", "out", ("left", "right", "record"),
            "aipc,ajqd->a", used_np=1, used_np_right=1,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3), term_index=0,
        )
        # Metadata predicts the same c128 value stack as the real kernel:
        # 64 rows x 3 x 3 x 16 bytes, even though each raw lane is 8 bytes.
        lane_values = (
            np.ones((32, 3, 3), dtype=np.float64),
            np.full((32, 3, 3), 1 + 0.25j, dtype=np.complex64),
        )
        normalized = {
            "layout": "coordinate_batch_coo",
            "point_keys": ("lane-a", "lane-b"),
            "lanes": tuple((
                (np.ones((1, 1), dtype=np.complex128), 1 + 0j, {
                    "layout": "normalized-coo",
                    "indices": np.zeros((32, 2), dtype=np.int64),
                    "values": values,
                }),
            ) for values in lane_values),
        }
        left = np.ones((2, 1, 1, 3), dtype=np.complex128)
        right = np.ones((2, 1, 1, 3), dtype=np.complex128)
        observed_packs = []
        original_stack = np.stack

        def recording_stack(arrays, *args, **kwargs):
            packed = original_stack(arrays, *args, **kwargs)
            if packed.shape == (64, 3, 3):
                observed_packs.append((packed.dtype, packed.nbytes))
            return packed

        from unittest.mock import patch
        with patch(
                "lqcdweave.contraction.execution.operations.sparse_join.np.stack",
                side_effect=recording_stack):
            actual = contract_p2p_bridge_batch(
                left, right, normalized, bridge=bridge, equation="aipc,ajqd->a",
            )
        self.assertIn((np.dtype(np.complex128), 9216), observed_packs)
        np.testing.assert_allclose(
            actual, np.asarray((288, 288 + 72j), dtype=np.complex128),
        )
        self.assertEqual(_coordinate_batch_sparse_workspace_bytes(state, action), 20160)
        self.assertEqual(coordinate_batch_sparse_workspace_bytes(state, action), 20160)

    def test_mixed_dtype_public_plan_preflights_x_workspace_before_reads(self):
        class Provider:
            def __init__(self):
                self.left = {}
                self.right = {}
                self.data = {}
                self.reads = {"left": 0, "right": 0}
                self.slabs = {}
                rng = np.random.default_rng(20261005)
                for time in range(2):
                    self.left[time] = np.ones((1, 1, 3), dtype=np.complex128)
                    self.right[time] = np.ones((1, 1, 3), dtype=np.complex128)
                    dtype = np.float64 if time == 0 else np.complex64
                    indices = np.zeros((32, 2), dtype=np.int64)
                    values = np.ones((32, 3, 3), dtype=dtype)
                    if np.issubdtype(dtype, np.complexfloating):
                        values = values + 1j * rng.normal(size=values.shape).astype(dtype)
                    record = {"type": "sparse", "indices": indices, "values": values}
                    raw_bytes = int(indices.nbytes + values.nbytes)
                    raw = {
                        "family": "p2p-record", "kind": "sparse",
                        "point_shape": (1, 1), "nnz": 32,
                        "value_dtype": np.dtype(dtype).name,
                        "index_dtype": "int64", "record_bytes": raw_bytes,
                        "backend": "numpy",
                    }
                    block = StorageBlock(
                        f"actual-mixed-{time}", "full", raw_bytes,
                        host_bytes=raw_bytes, read_kind="p2p",
                    )
                    gamma = np.ones((1, 1), dtype=np.complex128)
                    coefficient = 1.0 + 0j
                    loader = _SparseLaneLoader(record, gamma, coefficient)
                    self.data[time] = (
                        record, raw, block, gamma, coefficient, loader,
                    )
                for role, table in (("left", self.left), ("right", self.right)):
                    def describe(times, role=role):
                        return TensorAxisMetadata(
                            ("coordinate_batch", "spin", "point", "color"),
                            (len(times), 1, 1, 3), np.complex128,
                            f"mixed-{role}-physical", f"mixed-{role}-ordered",
                        )

                    def read(times, role=role, table=table):
                        self.reads[role] += 1
                        return np.stack([table[time] for time in times], axis=0)

                    self.slabs[role] = PhysicalSlabBinding(
                        ("mixed-promotion", role), describe, read,
                    )

            def describe(self, correlator, *, bindings, coordinates, batch):
                time = coordinates.points[0].values[0][1]
                record, raw, block, gamma, coefficient, loader = self.data[time]
                left_key, right_key, record_key = (
                    f"left-{time}", f"right-{time}", f"record-{time}",
                )
                left, right = self.left[time], self.right[time]
                problem = ContractionProblem(
                    {
                        left_key: TensorSpec((1, 1, 3), np.complex128),
                        right_key: TensorSpec((1, 1, 3), np.complex128),
                    },
                    (ContractionTerm((left_key, right_key), "spc,qrd->"),),
                    {
                        left_key: SourceSpec(
                            f"left-source-{time}", left.nbytes,
                            host_region_bytes=left.nbytes, snapshot_bytes=left.nbytes,
                        ),
                        right_key: SourceSpec(
                            f"right-source-{time}", right.nbytes,
                            host_region_bytes=right.nbytes, snapshot_bytes=right.nbytes,
                        ),
                        record_key: SourceSpec(
                            f"raw-record-{time}", block.read_bytes,
                            host_region_bytes=block.host_bytes, metadata=raw,
                        ),
                    },
                    sparse_bridges=(SparseBridgeSpec(
                        left_key, right_key, record_key, 1, 32,
                        left_axes=(0, 1, 2), right_axes=(0, 1, 2), scene=0,
                        used_np_right=1,
                    ),),
                )
                return DiagramInputs(
                    problem, {f"mixed-block-{time}": block},
                    {
                        left_key: SourcePreparation(()),
                        right_key: SourcePreparation(()),
                        record_key: SourcePreparation(
                            (f"mixed-block-{time}",), dtype="complex128",
                        ),
                    },
                    loader, coordinates, batch, ("p2p-term",),
                    tensor_requests={left_key: None, right_key: None},
                    batch_bindings={
                        left_key: BatchTensorBinding(
                            ("left",), (self.slabs["left"],), (time,),
                            (lambda value: value,), topology_key=("P/P",),
                        ),
                        right_key: BatchTensorBinding(
                            ("right",), (self.slabs["right"],), (time,),
                            (lambda value: value,), topology_key=("P/P",),
                        ),
                    },
                    batch_topology=("mixed-float64-complex64",),
                )

        provider = Provider()
        coordinates, batch = describe_coordinates(
            time_axes={"scan": (0, 1)}, batch={"axis": "scan", "size": 2},
        )
        inputs = BatchDiagramBinding(provider).describe(
            object(), bindings=DiagramBindings(), coordinates=coordinates, batch=batch,
        )
        expanded = ExpandedCorrelator(
            object(), inputs,
            OutputRecipe(OutputNode("term", TermContribution("p2p-term")),
                         ("p2p-term",), coordinates),
            DiagramBindings(), coordinates, batch,
        )
        wide = find_contraction_path(expanded, memory_limit=1 << 28, optimize="none")
        p_peak = wide.plan.peak_host_bytes
        adjusted, reserve = prepare_coordinate_batch_workspace(wide.plan)
        self.assertEqual((p_peak, reserve, adjusted.peak_host_bytes), (31712, 0, 31712))
        self.assertIs(adjusted, wide.plan)
        # The old low P estimate was 29824, before X added 1888 at admission.
        # P now freezes that same full envelope and rejects this cap offline.
        cap = (29824 + p_peak) // 2
        with self.assertRaises(MemoryError):
            find_contraction_path(expanded, memory_limit=cap, optimize="none")
        self.assertEqual(provider.reads, {"left": 0, "right": 0})
        for *_, loader in provider.data.values():
            self.assertEqual(loader.reads, [])
            self.assertEqual(loader.materializations, [])
            self.assertEqual(loader.closes, [])

    def test_coordinate_cycle_known_workspace_counts_lane_local_pair_packs(self):
        def marker(value_dtypes):
            lane_sources = tuple({
                "metadata": {
                    "kind": "sparse", "value_dtype": dtype,
                "point_shape": (2, 2), "nnz": 4,
                    "index_dtype": "int64", "record_bytes": 4 * (
                        16 + 9 * np.dtype(dtype).itemsize
                    ),
                    "backend": "numpy",
                },
                "materialized_point_shape": (2, 2),
            } for dtype in value_dtypes)
            lane_bounds = (4, 4)
            return {
                "family": "p2p-coordinate-batch",
                "coordinate_keys": ("lane-0", "lane-1"),
                "lane_nnz_upper": lane_bounds,
                "lane_sources": lane_sources,
                "point_shape": (2, 2),
                "part_count": 4,
            }

        first = SparseBridgeSpec(
            "left", "right", "record-first", 2, 8,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
            scene=0, part_count=4, used_np_right=2,
        )
        second = SparseBridgeSpec(
            "left", "right", "record-second", 2, 8,
            left_axes=(4, 5, 6), right_axes=(4, 5, 6),
            scene=0, part_count=4, used_np_right=2,
        )
        # The first and second left point axes share p.  All point axes are
        # contracted, while the lane is the only output; the lane/part
        # cross-product reaches the J=64 cap without an unsupported point
        # scatter in a two-bridge cycle.
        equation = "aipxjpd,aescgzu->a"
        action = Action(
            "sparse_bridge", "out",
            ("left", "right", "record-first", "record-second"),
            equation, used_np=2, used_np_right=2,
            left_axes=first.left_axes, right_axes=first.right_axes,
            term_index=0,
        )
        state = SimpleNamespace(
            problem=SimpleNamespace(
                backend="numpy",
                sparse_bridges=(first, second),
                sources={
                    "record-first": SourceSpec(
                        "first", 2 * (16 + 9 * 8),
                        metadata=marker(("float64", "complex64")),
                    ),
                    "record-second": SourceSpec(
                        "second", 2 * (16 + 9 * 16),
                        metadata=marker(("complex128", "complex128")),
                    ),
                },
            ),
            catalog={
                "left": TensorSpec((2, 1, 2, 3, 1, 2, 3), np.complex128),
                "right": TensorSpec((2, 1, 2, 3, 1, 2, 3), np.complex128),
                "out": TensorSpec((2,), np.complex128),
            },
        )
        # The 2 lanes x 4 parts x 4 rows on each side bound 512 candidate
        # pairs, capped to J=64.  Its owners are four J-row int64 selectors
        # (lane + three unique point labels),
        # two gamma packs, one coefficient pack, two promoted 3x3 value packs,
        # both gathered 9-element propagator rows, and the J-element result
        # contribution pack.
        # This hand sum is independent of the workspace helper.
        expected = (
            4 * 64 * 8
            + 2 * 64 * 16
            + 64 * 16
            + 2 * 64 * 9 * 16
            + 2 * 64 * 9 * 16
            + 64 * 16
        )
        self.assertEqual(expected, 43008)
        self.assertEqual(
            _coordinate_batch_sparse_workspace_bytes(state, action), expected,
        )
        self.assertEqual(coordinate_batch_sparse_workspace_bytes(state, action), expected)
        metrics = _model_features(state, action)
        self.assertEqual(metrics["sparse_operation"], "coordinate_batch_lane_pair_join")
        self.assertEqual(metrics["coordinate_batch_join_chunk_rows"], 64)
        self.assertEqual(metrics["coordinate_batch_pair_work_upper_bound"], 512)
        self.assertEqual(metrics["coordinate_batch_numeric_flush_upper_bound"], 8)
        self.assertEqual(metrics["coordinate_batch_known_workspace_host_bytes"], expected)

    def test_filtered_base_lifetime_estimate_covers_full_flush_boundary(self):
        import numpy as np

        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, 64,
            left_axes=(0, 1, 2), right_axes=(0, 1, 2),
            scene=0, part_count=1, used_np_right=4,
        )
        raw_nbytes = 64 * (2 * np.dtype("int64").itemsize + 9 * 16)
        raw = {
            "family": "p2p-record", "kind": "sparse", "point_shape": (4, 4),
            "nnz": 64, "value_dtype": "complex128", "index_dtype": "int64",
            "record_bytes": raw_nbytes, "backend": "numpy",
        }
        block = StorageBlock("dataset", "selection", raw_nbytes, host_bytes=raw_nbytes)
        entry = {
            "lane_index": 0, "bridge": bridge,
            "source": SourceSpec("record", raw_nbytes, host_region_bytes=raw_nbytes,
                                 metadata=raw),
            "preparation": SourcePreparation(("shared",)), "block_key": "shared",
        }
        contract = {
            "sparse_bridges": ({
                "scene": 0, "roles": (0, 1), "template": bridge,
                "filter_points": True,
            },),
            "lane_specs": (
                TensorSpec((4, 4, 3), np.complex128),
                TensorSpec((4, 4, 3), np.complex128),
            ),
        }
        catalog = {
            "blocks": {"shared": block},
            "sparse_by_scene": {0: (entry,)},
        }
        # At C=F=64, X can retain 190 full filtered bases across a flush
        # (the independent real-kernel probe observed 190 for nnz=[1,64]+62*[1]+[64]).
        # Each base row owns two int64 selectors plus a 3x3 complex128 block;
        # two boolean masks can overlap. The estimate also counts the raw and
        # normalized records, endpoint copies/gathers, packed rows, and output.
        self.assertEqual(_sparse_group_estimate(
            SimpleNamespace(point_indices=(0,)), contract, catalog,
        ), 106592)

    def test_batch_diagram_expansion_reaches_direct_x_with_reverse_bridge(self):
        rng = np.random.default_rng(20261003)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        left_by_time = {time: random((2, 4, 3)) for time in range(2)}
        right_by_time = {time: random((2, 4, 3)) for time in range(2)}
        lane_data, records, gammas, coefficients, loaders = {}, {}, {}, {}, {}
        for time in range(2):
            indices = np.asarray(((0, 1), (2, 0)), dtype=np.int64)
            values = random((2, 3, 3))
            record = {"type": "sparse", "indices": indices, "values": values}
            gamma = random((2, 2))
            coefficient = complex(0.4 + time, -0.2 + 0.3 * time)
            raw_bytes = int(indices.nbytes + values.nbytes)
            raw_meta = {
                "family": "p2p-record", "kind": "sparse", "point_shape": (4, 4),
                "nnz": 2, "value_dtype": "complex128", "index_dtype": "int64",
                "record_bytes": raw_bytes, "backend": "numpy", "anchor_time": time,
            }
            block = StorageBlock(f"record-dataset-{time}", "full", raw_bytes,
                                 host_bytes=raw_bytes, read_kind="p2p")
            record_key = f"record-{time}"
            loader = _SparseLaneLoader(record, gamma, coefficient)
            lane_data[time] = (
                f"left-{time}", f"right-{time}", record_key,
                block, raw_meta, record, gamma, coefficient, loader,
            )
            records[time], gammas[time] = record, gamma
            coefficients[time], loaders[time] = coefficient, loader

        coordinates, batch = describe_coordinates(
            time_axes={"scan": (0, 1)}, batch={"axis": "scan", "size": 2},
        )
        provider = _FixedSparseBatchProvider(left_by_time, right_by_time, lane_data)
        inputs = BatchDiagramBinding(provider).describe(
            object(), bindings=DiagramBindings(), coordinates=coordinates, batch=batch,
        )
        self.assertEqual(len(inputs.task_batches), 1)
        combined_problem = inputs.task_batches[0].problem
        self.assertEqual(combined_problem.sources["sparse-record:0"].metadata["family"],
                         "p2p-coordinate-batch")
        bridge = combined_problem.sparse_bridges[0]
        self.assertTrue(bridge.reverse_operands)
        self.assertEqual(bridge.nnz_upper, 4)

        expanded = ExpandedCorrelator(
            object(), inputs,
            OutputRecipe(OutputNode("term", TermContribution("p2p-term")),
                         ("p2p-term",), coordinates),
            DiagramBindings(), coordinates, batch,
        )
        path = find_contraction_path(expanded, memory_limit=1 << 24, optimize="none")
        for batch_path in path.batch_paths:
            self.assertTrue(any(action.kind == "sparse_bridge"
                                for action in batch_path.plan.state.actions))
        result = execute_request(
            BoundContractionExecution(expanded, path), backend=np, measure=False,
        )
        actual = {item.coordinate_key: complex(np.asarray(item.value).item())
                  for item in result.term_values}

        expected = {}
        for time in range(2):
            total = 0j
            for row, (raw_left, raw_right) in enumerate(records[time]["indices"]):
                total += np.einsum(
                    "ic,iq,cd,qd->",
                    left_by_time[time][:, raw_right, :], gammas[time].T,
                    records[time]["values"][row].T,
                    right_by_time[time][:, raw_left, :], optimize=False,
                )
            expected[str(time)] = coefficients[time] * total
        np.testing.assert_allclose(
            np.asarray([actual[str(time)] for time in range(2)]),
            np.asarray([expected[str(time)] for time in range(2)]),
            rtol=2e-13, atol=2e-13,
        )
        for loader in loaders.values():
            self.assertEqual(len(loader.reads), 1)
            self.assertEqual(len(loader.materializations), 1)
            self.assertEqual(len(loader.closes), 1)


if __name__ == "__main__":
    unittest.main()
