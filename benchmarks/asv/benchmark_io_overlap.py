"""End-to-end CPU scene replay with whole-file NPY selections and dot kernels.

Each synthetic source file is 32 MiB; the replay selects one 512 x 512
matrix. Setup validates the numerical result and warms the files into the local
page cache. Timings therefore measure repeated warm-cache file reads, selection
copies, contractions, and result consumption, not cold storage throughput.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lqcdweave.adapters.scene_storage import NpyBlockBinding, NpySceneBlockLoader
from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
)
from lqcdweave.contraction.execution.api import replay_storage_batches
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    join_storage_batches,
    logical_cost_profile,
    logical_source_problem,
    plan_scene_storage,
)


_FILE_SHAPE = (2048, 2048)
_MATRIX_SIZE = 512
_DTYPE = np.dtype("float64")
_PAYLOAD_BYTES = int(np.prod(_FILE_SHAPE)) * _DTYPE.itemsize
_MATRIX_BYTES = _MATRIX_SIZE * _MATRIX_SIZE * _DTYPE.itemsize
_MEMORY_LIMIT = 512 * 1024**2


def _make_plan(pair):
    block_a, block_b = f"{pair}:A", f"{pair}:B"
    shape = (_MATRIX_SIZE, _MATRIX_SIZE)
    problem = logical_source_problem(ContractionProblem(
        {"A": TensorSpec(shape, _DTYPE), "B": TensorSpec(shape, _DTYPE)},
        (ContractionTerm(("A", "B"), "ik,kj->ij"),),
        {"A": SourceSpec(block_a, 0, metadata={"block_key": block_a}),
         "B": SourceSpec(block_b, 0, metadata={"block_key": block_b})},
    ))
    profile = CostProfile()
    model = logical_cost_profile(profile)
    state = ContractionState.initial(problem)
    state = state.load("A", model, _MEMORY_LIMIT).load("B", model, _MEMORY_LIMIT)
    state = state.contract(0, (0, 1), model, _MEMORY_LIMIT).release("A").release("B")

    blocks = {
        block_a: StorageBlock(
            f"synthetic-{block_a}", "whole-file-prefix", _PAYLOAD_BYTES,
            host_bytes=_MATRIX_BYTES, staging_host_bytes=_PAYLOAD_BYTES,
        ),
        block_b: StorageBlock(
            f"synthetic-{block_b}", "whole-file-prefix", _PAYLOAD_BYTES,
            host_bytes=_MATRIX_BYTES, staging_host_bytes=_PAYLOAD_BYTES,
        ),
    }
    preparations = {"A": SourcePreparation((block_a,)), "B": SourcePreparation((block_b,))}
    return plan_scene_storage(
        state, blocks, preparations, profile=profile,
        memory_limit=_MEMORY_LIMIT, host_memory_limit=_MEMORY_LIMIT,
    )


class WholeFileReadAheadReplay:
    repeat = (3, 5, 7)
    number = 1
    warmup_time = 0.1
    timeout = 300

    def setup(self):
        self._temporary = TemporaryDirectory(prefix="lqcdweave-asv-io-")
        root = Path(self._temporary.name)
        values = {"pair0:A": 1.0, "pair0:B": 2.0,
                  "pair1:A": 3.0, "pair1:B": 4.0}
        bindings, materializers = {}, {}
        selected = (slice(0, _MATRIX_SIZE), slice(0, _MATRIX_SIZE))
        for key, value in values.items():
            path = root / f"{key.replace(':', '-')}.npy"
            mapped = np.lib.format.open_memmap(
                path, mode="w+", dtype=_DTYPE, shape=_FILE_SHAPE,
            )
            mapped[...] = value
            mapped.flush()
            del mapped
            bindings[key] = NpyBlockBinding(
                path, selected, (_MATRIX_SIZE, _MATRIX_SIZE), _DTYPE.str,
                read_mode="whole_file", file_shape=_FILE_SHAPE,
            )

        materializers = {
            tensor: lambda metadata, blocks: blocks[metadata["block_key"]]
            for tensor in ("A", "B")
        }

        self._loader = NpySceneBlockLoader(bindings, materializers, backend=np)
        self._plans = join_storage_batches((_make_plan("pair0"), _make_plan("pair1")))
        actual = self._replay()
        expected = (
            _MATRIX_SIZE**3 * values["pair0:A"] * values["pair0:B"],
            _MATRIX_SIZE**3 * values["pair1:A"] * values["pair1:B"],
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def _replay(self):
        totals = []

        def consume(index, report):
            totals.append(float(np.asarray(report[0]).sum()))

        replay_storage_batches(
            self._plans, self._loader, consume, prejoined=True,
        )
        return tuple(totals)

    def time_two_independent_whole_file_matmuls(self):
        self._replay()

    def teardown(self):
        self._temporary.cleanup()
