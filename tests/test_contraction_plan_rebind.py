import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction import correlator as workflow_api
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    audit_resource_trajectory, plan_direct_execution,
)
from lqcdweave.contracts.correlator import api as contracts
from lqcdweave.storage.contraction_plans.api import (
    PlanArchiveError,
    load_contraction_plan,
    save_contraction_plan,
)
from lqcdweave.contraction.correlator import (
    Calculation,
    assemble_correlator,
    execute_contraction_path,
    save_calculation_plan,
)


class DatasetLoader:
    def __init__(self):
        self.reads = []
        self.datasets = []
        self.selections = []
        self.closed = []

    def read_block(self, key, block):
        self.reads.append(key)
        self.datasets.append(block.dataset)
        self.selections.append(block.selection)
        offset = 10 if block.selection.endswith("t1") else 0
        return np.arange(4, dtype=np.float64) + offset

    def materialize(self, key, source, blocks):
        return blocks["raw"]

    def close_block(self, key, value):
        self.closed.append(key)

    def close_all(self):
        pass


def expanded(identity, loader, time=0):
    coordinates, batch = contracts.describe_coordinates(time_axes={"sink": time})
    description = contracts.TensorDescription(
        axes=("mode",),
        data_identity=f"ensemble-{identity}",
        selection_identity=f"selection-{identity}-t{time}",
        role="perambulator",
        source_bindings=(f"raw-{identity}",),
        transform=("gamma5", "dagger"),
        semantics={"boundary": "periodic", "basis": "eigenvectors-7"},
        bindings={"configuration": identity, "source_time": 0, "sink_time": time},
    )
    problem = contracts.ContractionProblem(
        {"prop": contracts.TensorSpec((4,), np.float64, description=description)},
        (contracts.ContractionTerm(("prop",), "i->", 1),),
        {
            "prop": contracts.SourceSpec(
                region=f"logical-{identity}",
                region_bytes=32,
                snapshot_bytes=32,
                metadata={"data_identity": f"ensemble-{identity}", "selection": identity, "sink_time": time},
            ),
        },
    )
    block = contracts.StorageBlock(
        dataset=f"dataset-{identity}",
        selection=f"selection-{identity}-t{time}",
        read_bytes=32,
        host_bytes=32,
    )
    inputs = contracts.DiagramInputs(
        problem=problem,
        blocks={"raw": block},
        preparations={"prop": contracts.SourcePreparation(("raw",))},
        loader=loader,
        coordinates=coordinates,
        batch=batch,
        term_ids=("T",),
    )
    output = contracts.OutputRecipe(
        contracts.OutputNode("term", contracts.TermContribution("T")),
        ("T",),
        coordinates,
    )
    return contracts.ExpandedCorrelator(
        correlator="test-correlator",
        inputs=inputs,
        output_recipe=output,
        bindings=contracts.DiagramBindings(configuration=identity),
        coordinates=coordinates,
        batch=batch,
    )


def with_temporary_cpu_spill(path, *, disk_limit_bytes):
    """Build an explicit resource plan on the public test Calculation fixture."""
    profile = path.profile
    storage = path.plan
    state = storage.state
    blocks = dict(storage.blocks)
    preparations = dict(storage.preparations)
    actions = []
    read = set()
    for action in state.actions:
        if action.kind == "load":
            for block_key in preparations[action.tensor].blocks:
                if block_key not in read:
                    actions.append(contracts.Action(
                        "read_block", block_key,
                    ))
                    read.add(block_key)
            actions.append(action)
            size = int(np.prod(state.catalog[action.tensor].shape)) * np.dtype(
                state.catalog[action.tensor].dtype
            ).itemsize
            actions.append(contracts.Action(
                "spill_disk", action.tensor,
                resource=contracts.ResourceTransition("tensor", "host", "disk", size),
            ))
            actions.append(contracts.Action(
                "restore_host", action.tensor,
                resource=contracts.ResourceTransition("tensor", "disk", "host", size),
            ))
        else:
            actions.append(action)
    actions.extend(contracts.Action("close_block", key) for key in sorted(read))
    state = replace(state, actions=tuple(actions), seconds=sum(item.seconds for item in actions))
    resource_plan = audit_resource_trajectory(
        state, blocks, preparations, profile=profile,
        memory_limit=path.memory_limit,
        host_memory_limit=path.host_memory_limit,
        disk_limit_bytes=disk_limit_bytes,
        reserved_host_bytes=storage.reserved_host_bytes,
        reserved_device_bytes=storage.reserved_device_bytes,
    )
    return replace(
        path, plan=resource_plan, plan_identity=id(resource_plan),
        execution_mode="planned",
        memory_limit=resource_plan.memory_limit,
        host_memory_limit=resource_plan.host_memory_limit,
        reserved_host_bytes=resource_plan.reserved_host_bytes,
        reserved_device_bytes=resource_plan.reserved_device_bytes,
    )


class ContractionPlanRebindTests(TestCase):
    def test_rebind_uses_current_identity_and_loader_without_search(self):
        self.assertIn("save_calculation_plan", workflow_api.__all__)
        old_loader = DatasetLoader()
        old_expanded = expanded("old", old_loader, time=1)
        profile = CostProfile(hardware_fingerprint="cpu-test")
        path = plan_direct_execution(
            old_expanded,
            memory_limit=10000,
            host_memory_limit=10000,
            profile=profile,
            reserved_host_bytes=11,
        )
        hardware = {"backend": "numpy", "fingerprint": "cpu-test", "memory_limit": 10000}
        options_hardware = {
            "backend": "numpy", "fingerprint": "options-cpu", "memory_limit": 10000
        }
        profile_source = {"kind": "test-model", "revision": "cpu-test-1"}
        options_profile_source = {"kind": "options-model", "revision": "options-1"}
        options = {
            "target_hardware": options_hardware,
            "profile_source": options_profile_source,
            "reserved_host_bytes": 11,
            "reserved_device_bytes": 0,
            "storage": "bounded",
            "optimize": "greedy",
        }
        calculation = Calculation(old_expanded, path, options=options)

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "path.json"
            save_calculation_plan(
                archive,
                calculation,
                hardware=hardware,
                profile_source=profile_source,
            )
            saved_payload = json.loads(archive.read_text(encoding="utf-8"))["payload"]
            self.assertNotIn("signature", saved_payload)
            self.assertIsInstance(saved_payload["path_signature"], str)
            self.assertEqual(saved_payload["prediction"], path.prediction)
            direct_archive = Path(directory) / "direct-path.json"
            save_contraction_plan(
                direct_archive,
                path,
                hardware=hardware,
                profile_source=profile_source,
                storage_policy="bounded",
                optimize="greedy",
                reserved_host_bytes=11,
                reserved_device_bytes=0,
            )
            self.assertEqual(archive.read_bytes(), direct_archive.read_bytes())
            option_archive = Path(directory) / "option-path.json"
            save_calculation_plan(option_archive, calculation)
            direct_option_archive = Path(directory) / "direct-option-path.json"
            save_contraction_plan(
                direct_option_archive,
                path,
                hardware=options_hardware,
                profile_source=options_profile_source,
                storage_policy="bounded",
                optimize="greedy",
                reserved_host_bytes=11,
                reserved_device_bytes=0,
            )
            self.assertEqual(option_archive.read_bytes(), direct_option_archive.read_bytes())
            with self.assertRaisesRegex(TypeError, "PlannedContraction"):
                save_contraction_plan(Path(directory) / "invalid.json", calculation)
            invalid_options = SimpleNamespace(path=path, expanded=old_expanded, options=[])
            fallback_archive = Path(directory) / "fallback-options.json"
            save_calculation_plan(fallback_archive, invalid_options)
            direct_fallback_archive = Path(directory) / "direct-fallback-options.json"
            save_contraction_plan(direct_fallback_archive, path)
            self.assertEqual(fallback_archive.read_bytes(), direct_fallback_archive.read_bytes())
            current_loader = DatasetLoader()
            current_expanded = expanded("new", current_loader, time=1)
            changed_coordinates = expanded("new", current_loader, time=0)
            mismatched_calculation = SimpleNamespace(
                path=path, expanded=current_expanded, options={}
            )
            with self.assertRaisesRegex(PlanArchiveError, "do not match"):
                save_calculation_plan(
                    Path(directory) / "mismatched.json", mismatched_calculation
                )
            with self.assertRaisesRegex(PlanArchiveError, "hardware"):
                load_contraction_plan(
                    archive,
                    expected=current_expanded,
                    hardware={"backend": "numpy", "fingerprint": "other-cpu", "memory_limit": 10000},
                    profile=profile,
                )
            with self.assertRaisesRegex(PlanArchiveError, "cost profile"):
                load_contraction_plan(
                    archive,
                    expected=current_expanded,
                    hardware=hardware,
                    profile=CostProfile(hardware_fingerprint="different-cpu"),
                )
            with self.assertRaisesRegex(PlanArchiveError, "coordinates or batch structure changed"):
                load_contraction_plan(
                    archive,
                    expected=changed_coordinates,
                    hardware=hardware,
                    profile=profile,
                )
            with patch(
                "lqcdweave.contraction.planning.api.find_contraction_path",
                side_effect=AssertionError("loading must not search for an operation order"),
            ), patch(
                "lqcdweave.contraction.planning.api.plan_scene_storage",
                side_effect=AssertionError("loading must not rebuild a storage schedule"),
            ), patch(
                "lqcdweave.contraction.performance.api.predict_plan_cost",
                side_effect=AssertionError("loading must not recompute a prediction"),
            ):
                rebound = load_contraction_plan(
                    archive,
                    expected=current_expanded,
                    hardware=hardware,
                    profile=profile,
                    memory_limit=10000,
                    host_memory_limit=10000,
                )
                executed = execute_contraction_path(current_expanded, rebound, backend=np)
                value = assemble_correlator(current_expanded, executed.term_values)

        self.assertIs(rebound.expanded, current_expanded)
        self.assertEqual(float(value), 46.0)
        self.assertEqual(current_loader.reads, ["raw"])
        self.assertEqual(current_loader.datasets, ["dataset-new"])
        self.assertEqual(current_loader.selections, ["selection-new-t1"])
        self.assertEqual(old_loader.reads, [])
        self.assertEqual(current_expanded.coordinates.points[0].values, (("sink", 1),))
        self.assertEqual(rebound.prediction, path.prediction)
        self.assertEqual(rebound.plan.state.actions, path.plan.state.actions)
        self.assertEqual(rebound.plan.kernel_batches, path.plan.kernel_batches)
        self.assertEqual(rebound.plan.memory, path.plan.memory)

    def _save_temporary_resource_archive(self, directory):
        current_loader = DatasetLoader()
        current_expanded = expanded("resource-old", current_loader)
        profile = CostProfile(
            hardware_fingerprint="cpu-resource",
        )
        planned = plan_direct_execution(
            current_expanded, memory_limit=10000, host_memory_limit=10000,
            profile=profile,
        )
        planned = with_temporary_cpu_spill(planned, disk_limit_bytes=32)
        calculation = Calculation(current_expanded, planned)
        hardware = {"backend": "numpy", "fingerprint": "cpu-resource", "memory_limit": 10000}
        archive = Path(directory) / "temporary-resource-plan.json"
        save_calculation_plan(archive, calculation, hardware=hardware)
        return archive, current_expanded, planned, profile, hardware

    def test_temporary_resource_plan_roundtrips_and_rejects_unsupported_rebinds(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, current_expanded, original, profile, hardware = (
                self._save_temporary_resource_archive(directory)
            )
            with patch(
                "lqcdweave.contraction.planning.api.plan_scene_storage",
                side_effect=AssertionError("loading must not rebuild a storage schedule"),
            ), patch(
                "lqcdweave.contraction.performance.api.predict_plan_cost",
                side_effect=AssertionError("loading must not recompute a prediction"),
            ):
                loaded = load_contraction_plan(
                    archive, expected=current_expanded, hardware=hardware, profile=profile,
                )
            self.assertTrue(loaded.plan.explicit_resources)
            self.assertEqual(loaded.plan.disk_limit_bytes, 32)
            self.assertEqual(loaded.plan.peak_disk_bytes, 32)
            self.assertEqual(loaded.plan.state.actions, original.plan.state.actions)
            self.assertEqual(
                tuple(point.temporary_disk_bytes for point in loaded.plan.memory),
                tuple(point.temporary_disk_bytes for point in original.plan.memory),
            )
            fitting_budget = load_contraction_plan(
                archive,
                expected=current_expanded,
                hardware=hardware,
                profile=profile,
                memory_limit=original.plan.peak_host_bytes + 100,
            )
            self.assertEqual(fitting_budget.plan.state.actions, original.plan.state.actions)
            self.assertEqual(fitting_budget.plan.memory, original.plan.memory)
            self.assertEqual(fitting_budget.prediction, original.prediction)
            self.assertEqual(fitting_budget.plan.memory_limit, original.plan.peak_host_bytes + 100)

            direct_resource = replace(original, execution_mode="direct")
            with self.assertRaisesRegex(NotImplementedError, "one planned non-batched path"):
                save_contraction_plan(
                    Path(directory) / "direct-resource-plan.json",
                    direct_resource,
                    hardware=hardware,
                )

            too_small = original.plan.peak_host_bytes - 1
            with patch(
                "lqcdweave.contraction.planning.api.find_contraction_path",
                side_effect=AssertionError("loading a fixed plan must not search"),
            ):
                with self.assertRaisesRegex(PlanArchiveError, "current memory budget"):
                    load_contraction_plan(
                        archive, expected=current_expanded, hardware=hardware, profile=profile,
                        memory_limit=too_small,
                    )
                with self.assertRaisesRegex(PlanArchiveError, "current memory budget"):
                    load_contraction_plan(
                        archive, expected=current_expanded, hardware=hardware, profile=profile,
                        host_memory_limit=original.plan.peak_host_bytes - 1,
                    )

                with self.assertRaisesRegex(PlanArchiveError, "memory reserve changed"):
                    load_contraction_plan(
                        archive, expected=current_expanded, hardware=hardware, profile=profile,
                        memory_limit=original.plan.peak_host_bytes + 100,
                        reserved_host_bytes=7,
                    )

            changed_profile = replace(profile, snapshot_call_seconds=0.8)
            with self.assertRaisesRegex(PlanArchiveError, "cost profile changed"):
                load_contraction_plan(
                    archive, expected=current_expanded, hardware=hardware,
                    profile=changed_profile,
                )

            changed_task = expanded("resource-new", DatasetLoader(), time=1)
            with self.assertRaisesRegex(PlanArchiveError, "coordinates or batch structure changed"):
                load_contraction_plan(
                    archive, expected=changed_task, hardware=hardware, profile=profile,
                )
            larger_block = replace(current_expanded.inputs.blocks["raw"], host_bytes=64)
            changed_inputs = replace(
                current_expanded.inputs, blocks={"raw": larger_block},
            )
            changed_layout = replace(current_expanded, inputs=changed_inputs)
            with self.assertRaisesRegex(PlanArchiveError, "physical storage identity or preparation layout"):
                load_contraction_plan(
                    archive, expected=changed_layout, hardware=hardware, profile=profile,
                )

            ordinary = plan_direct_execution(
                current_expanded, memory_limit=10000, host_memory_limit=10000,
                profile=profile,
            )
            ordinary_archive = Path(directory) / "ordinary-path.json"
            save_contraction_plan(ordinary_archive, ordinary, hardware=hardware)
            with self.assertRaisesRegex(PlanArchiveError, "current memory budget"):
                load_contraction_plan(
                    ordinary_archive, expected=current_expanded, hardware=hardware,
                    profile=profile, memory_limit=ordinary.plan.peak_host_bytes - 1,
                )
