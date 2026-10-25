"""Rectangular COO bridges agree with independent dense point-pair sums."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    SparseBridgeSpec,
    StorageBlock,
    TensorSpec,
    cycle_point_equalities_covered,
)
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge,
    contract_p2p_cycle,
)
from lqcdweave.contraction.execution.operations.sparse_join import (
    contract_sparse_p2p_bridge_join,
)
from lqcdweave.contraction.execution.storage_runtime import replay_storage_batches
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    logical_cost_profile,
    logical_source_problem,
    plan_scene_storage,
    rebind_scene_path,
)
from lqcdweave.storage.contraction_plans.api import (
    decode_plan_value,
    encode_plan_value,
    load_storage_schedule,
    save_storage_schedule,
)


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _record(rng, left_count, right_count):
    pairs = [(left, right)
             for left in range(left_count)
             for right in range(right_count)]
    if pairs:
        pairs.append(pairs[1 % len(pairs)])  # duplicate rows remain separate summands
    return {
        "type": "sparse",
        "indices": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        "values": _complex(rng, (len(pairs), 3, 3)),
    }


def _dense_record(record, left_count, right_count):
    dense = np.zeros((left_count, right_count, 3, 3), dtype=np.complex128)
    if record.get("type") == "identity":
        if left_count != right_count:
            raise ValueError("identity record cannot represent a rectangular point shape")
        for point in range(left_count):
            dense[point, point] = np.eye(3)
        return dense
    for (left, right), value in zip(record["indices"], record["values"], strict=True):
        if 0 <= left < left_count and 0 <= right < right_count:
            dense[left, right] += value
    return dense


def _sources_for(specs, records):
    sources = {
        key: SourceSpec(
            key, int(np.prod(spec.shape)) * spec.dtype.itemsize,
            snapshot_bytes=int(np.prod(spec.shape)) * spec.dtype.itemsize,
        )
        for key, spec in specs.items()
    }
    for key, record in records.items():
        size = record["indices"].nbytes + record["values"].nbytes
        sources[key] = SourceSpec(
            key, size, host_region_bytes=size,
            metadata={"family": "p2p-record", "backend": "numpy"},
        )
    return sources


def _rectangular_cycle_case(seed=20261002, left_count=3, right_count=2):
    rng = np.random.default_rng(seed)
    left = _complex(rng, (4, 4, left_count, 3, right_count, 3))
    right = _complex(rng, (4, 4, right_count, 3, left_count, 3))
    gamma0, gamma1 = _complex(rng, (4, 4)), _complex(rng, (4, 4))
    record0 = _record(rng, left_count, right_count)
    # This reversed pair layout is the materialized source-dagger current.
    record1 = _record(rng, right_count, left_count)
    first = SparseBridgeSpec(
        "left", "right", "record0", left_count, len(record0["indices"]),
        left_axes=(1, 2, 3), right_axes=(0, 2, 3),
        used_np_right=right_count,
    )
    second = SparseBridgeSpec(
        "left", "right", "record1", right_count, len(record1["indices"]),
        left_axes=(0, 4, 5), right_axes=(1, 4, 5),
        used_np_right=left_count,
    )
    equation = "BAxbya,DCydxc->"
    return rng, left, right, gamma0, gamma1, record0, record1, first, second, equation


class _ArrayLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []
        self.closed = []

    def read_block(self, key, block):
        self.reads.append((key, block.dataset))
        return self.arrays[key]

    def materialize(self, key, source, blocks):
        return blocks[key]

    def close_block(self, key, _value):
        self.closed.append(key)


def _cycle_problem(*, left_count=3, right_count=2, seed=20261002):
    values = _rectangular_cycle_case(seed, left_count, right_count)
    rng, left, right, gamma0, gamma1, record0, record1, first, second, equation = values
    specs = {"left": TensorSpec(left.shape, left.dtype),
             "right": TensorSpec(right.shape, right.dtype)}
    records = {"record0": record0, "record1": record1}
    problem = ContractionProblem(
        specs,
        (ContractionTerm(("left", "right"), equation),),
        _sources_for(specs, records),
        sparse_bridges=(first, second),
    )
    arrays = {"left": left, "right": right,
              "record0": ((gamma0, 0.6 + 0.4j, record0),),
              "record1": ((gamma1, 0.6 + 0.4j, record1),)}
    blocks = {}
    preparations = {}
    for key, source in problem.sources.items():
        nbytes = max(source.region_bytes, 1)
        blocks[key] = StorageBlock(key, "all", nbytes, host_bytes=nbytes)
        preparations[key] = SourcePreparation((key,))
    return problem, arrays, blocks, preparations, values


def _planned_state(problem):
    profile = CostProfile()
    state = ContractionState.initial(logical_source_problem(problem))
    model = logical_cost_profile(profile)
    for key in problem.sources:
        state = state.load(key, model, 1 << 50)
    state = state.contract(0, (0, 1), model, 1 << 50)
    return state, profile


def _remove_right_extent_fields(value):
    if isinstance(value, dict):
        if (value.get("type", "").endswith((".Action", ".SparseBridgeSpec"))
                and isinstance(value.get("value"), dict)):
            value["value"].pop("used_np_right", None)
        for item in value.values():
            _remove_right_extent_fields(item)
    elif isinstance(value, list):
        for item in value:
            _remove_right_extent_fields(item)


class RectangularSparseBridgeTests(unittest.TestCase):
    def test_contract_planning_rebind_and_legacy_action_shape(self):
        problem, _arrays, _blocks, _preparations, _case = _cycle_problem()
        state, profile = _planned_state(problem)
        action = state.actions[-1]
        self.assertEqual(action.kind, "sparse_bridge")
        self.assertEqual((action.used_np, action.used_np_right), (3, 2))
        rebound = rebind_scene_path(
            state, problem, {key: key for key in problem.sources},
            profile=profile, memory_limit=1 << 50,
        )
        self.assertEqual(rebound.actions[-1].used_np_right, 2)

        saved = encode_plan_value(action)
        old_named = json.loads(json.dumps(saved))
        old_named["type"] = old_named["type"].rsplit(".", 1)[-1]
        old_named["value"].pop("used_np_right")
        decoded_old = decode_plan_value(old_named)
        self.assertIsNone(decoded_old.used_np_right)
        self.assertEqual(
            decode_plan_value(saved).used_np_right,
            2,
        )

    def test_rectangular_schedule_roundtrip_rebind_and_square_field_default(self):
        problem, arrays, blocks, preparations, _case = _cycle_problem()
        state, profile = _planned_state(problem)
        schedule = plan_scene_storage(
            state, blocks, preparations, profile=profile, memory_limit=1 << 50,
        )
        shapes = {key: spec.shape for key, spec in problem.specs.items()}
        hardware = {"backend": "numpy", "memory": 1 << 50}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rectangular-schedule.json"
            save_storage_schedule(
                path, (schedule,), shapes=shapes, hardware=hardware,
                metadata={"case": "3x2-cycle"},
            )
            rebound_blocks = {
                key: replace(value, dataset="fresh:" + key)
                for key, value in blocks.items()
            }
            loaded, metadata = load_storage_schedule(
                path, shapes=shapes, hardware=hardware, expected=problem,
                blocks=rebound_blocks, preparations=preparations,
            )
            self.assertEqual(metadata, {"case": "3x2-cycle"})
            self.assertEqual(loaded[0].state.problem.sparse_bridges[0].point_shape, (3, 2))
            self.assertEqual(loaded[0].state.problem.sparse_bridges[1].point_shape, (2, 3))
            self.assertEqual(loaded[0].blocks["record0"].dataset, "fresh:record0")
            loader = _ArrayLoader(arrays)
            reports = []
            replay_storage_batches(
                loaded, loader, lambda _index, report: reports.append(report),
                prejoined=True,
            )
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0][0].shape, ())

            self.assertEqual(len(loader.reads), len(problem.sources))

        # The current envelope may omit this optional square-shape field.
        square_problem, square_arrays, square_blocks, square_preparations, _ = _cycle_problem(
            seed=61004, left_count=2, right_count=2,
        )
        square_state, square_profile = _planned_state(square_problem)
        square_schedule = plan_scene_storage(
            square_state, square_blocks, square_preparations,
            profile=square_profile, memory_limit=1 << 50,
        )
        square_shapes = {key: spec.shape for key, spec in square_problem.specs.items()}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "square-default.json"
            save_storage_schedule(
                path, (square_schedule,), shapes=square_shapes, hardware=hardware,
                metadata={"archive": "square-default"},
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            payload = document["payload"]
            _remove_right_extent_fields(payload["semantics"])
            _remove_right_extent_fields(payload["plans"])
            path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")),
                            encoding="utf-8")
            square_loaded, metadata = load_storage_schedule(
                path, shapes=square_shapes, hardware=hardware, expected=square_problem,
                blocks=square_blocks, preparations=square_preparations,
            )
            self.assertEqual(metadata, {"archive": "square-default"})
            self.assertEqual(square_loaded[0].state.problem.sparse_bridges[0].point_shape, (2, 2))
            loader = _ArrayLoader(square_arrays)
            reports = []
            replay_storage_batches(
                square_loaded, loader, lambda _index, report: reports.append(report),
                prejoined=True,
            )
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0][0].shape, ())


if __name__ == "__main__":
    unittest.main()
