"""Indexed VVV budget admission, numeric identity, and row ownership checks."""

import importlib
import math
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.planning.baryon import estimate_baryon_vvv_memory
from lqcdweave.production.elemental.transport import momentum_phase
from lqcdweave.production.elemental import baryon_planning as workflow
import test_baryon_production_dataset as fixture_module



class BaryonProductionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixture_module.BaryonDatasetProductionTests("runTest")
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)

    def _metadata_specs(self):
        identities = dict.fromkeys((self.fx.gauge_id, *self.fx.basis_ids))
        return {identity: self.fx.index.observe_metadata(identity)["spec"]
                for identity in identities}

    def _estimate(self, product="raw_paths", *, components=None, momenta=None,
                  phase="anchor", stream=True, blocks=(1, 2, 1), cache=0,
                  workspace=0, staging=0):
        options = self.fx._options(product=product, components=components)
        if momenta is not None:
            options["momentum_list"] = momenta
        options["phase_convention"] = phase
        preview = workflow.plan_indexed_baryon_elementals(
            self.fx.index, self.fx.gauge_id, self.fx.basis_ids, **options)
        host, device = estimate_baryon_vvv_memory(
            self._metadata_specs(), preview["dataset_spec"],
            gauge_id=self.fx.gauge_id, basis_ids=self.fx.basis_ids,
            ne_block_size=blocks, spatial_block_size=5,
            cache_bytes=cache, stream_eigenvectors=stream,
            target="cpu", workspace_reserve_bytes=workspace,
            host_staging_bytes=staging,
        )
        return preview, host, device

    def test_estimator_models_inputs_rows_mode_bookkeeping_and_phase_lru(self):
        cases = (
            ("raw_paths", self.fx.raw_components),
            ("projected", [{"id": 0, "terms": [
                {"paths": self.fx.raw_components[1], "coefficient": [.2, .7]},
                {"paths": self.fx.raw_components[2], "coefficient": [-.4, .1]},
            ]}]),
            ("distance", ((0, 0, 0), (1, 0, 2))),
        )
        for product, components in cases:
            with self.subTest(product=product):
                preview, host, device = self._estimate(product, components=components)
                self.assertEqual(host.output_bytes,
                                 math.prod(preview["dataset_spec"].shape[1:]) * 16)
                self.assertEqual(device.logical_peak_bytes, 0)
                self.assertGreater(host.intermediate_bytes, 0)

        preview, streamed, _ = self._estimate("raw_paths", stream=True)
        _, eager, _ = self._estimate("raw_paths", stream=False)
        specs = self._metadata_specs()
        gauge = specs[self.fx.gauge_id]
        basis = tuple(specs[identity] for identity in self.fx.basis_ids)
        lx, ly, lz, lt = gauge.document["configuration"]["lattice"]
        volume = lx * ly * lz
        links_copy = 4 * lt * volume * 3 * 3 * 16
        selected_snapshots = sum(
            lt * count * volume * 3 * 16 for count in self.fx.counts)
        largest_full_basis = max(
            math.prod(spec.shape) * np.dtype(spec.document["dtype"]).itemsize
            for spec in basis)
        self.assertEqual(streamed.snapshot_bytes, links_copy)
        self.assertEqual(eager.snapshot_bytes,
                         links_copy + selected_snapshots + largest_full_basis)

        # The estimator counts one storage input per distinct actual ID, while
        # the three basis roles retain their independent selected snapshots.
        unique_specs = {self.fx.gauge_id: gauge,
                        self.fx.basis_a_id: specs[self.fx.basis_a_id],
                        self.fx.basis_b_id: specs[self.fx.basis_b_id]}
        expected_inputs = sum(math.prod(spec.shape) * np.dtype(spec.document["dtype"]).itemsize
                              for spec in unique_specs.values())
        self.assertEqual(streamed.input_bytes, expected_inputs)

        no_cache = streamed
        _, cached, _ = self._estimate("raw_paths", stream=True, cache=1024 * 1024)
        self.assertGreater(cached.intermediate_bytes, no_cache.intermediate_bytes)
        self.assertGreater(cached.assembly_extra_bytes, no_cache.assembly_extra_bytes)

        _, coarse, _ = self._estimate("raw_paths", blocks=(2, 3, 1))
        _, fine, _ = self._estimate("raw_paths", blocks=(1, 1, 1))
        spans_coarse = math.prod(math.ceil(count / block)
                                 for count, block in zip(self.fx.counts, (2, 3, 1)))
        spans_fine = math.prod(math.ceil(count / block)
                               for count, block in zip(self.fx.counts, (1, 1, 1)))
        component_count = len(self.fx.raw_components)
        momentum_count = len(self.fx.momenta)
        seen_coarse = component_count * momentum_count * spans_coarse
        seen_fine = component_count * momentum_count * spans_fine
        expected_bookkeeping_delta = (
            (spans_fine - spans_coarse) * 512 + (seen_fine - seen_coarse) * 384)
        self.assertEqual(fine.assembly_extra_bytes - coarse.assembly_extra_bytes,
                         expected_bookkeeping_delta)

        # The 65th unique phase evicts only after the new phase is built. The
        # phase='none' path still overlaps three int64 index grids and ones.
        momenta = tuple((value, 0, 0) for value in range(65))
        none_preview, none, _ = self._estimate(
            "raw_paths", components=(((), (), ()),), momenta=momenta, phase="none")
        anchor_preview, anchor, _ = self._estimate(
            "raw_paths", components=(((), (), ()),), momenta=momenta, phase="anchor")
        self.assertEqual(none_preview["dataset_spec"].coordinates("momentum"),
                         anchor_preview["dataset_spec"].coordinates("momentum"))
        self.assertEqual(anchor.intermediate_bytes - none.intermediate_bytes,
                         24 * volume)

    def test_phase_none_grid_overlap_and_64_entry_lru_lifetime(self):
        class PhaseProbe:
            def __init__(self):
                self.grid_ref = None
                self.grid_bytes = 0
                self.grid_live_when_ones_allocates = False

            def indices(self, shape):
                result = np.indices(shape)
                self.grid_ref = weakref.ref(result)
                self.grid_bytes = result.nbytes
                return result

            def ones(self, shape, dtype):
                self.grid_live_when_ones_allocates = self.grid_ref() is not None
                return np.ones(shape, dtype=dtype)

        sizes = (*self.fx.sizes[:3], self.fx.sizes[3])
        volume = math.prod(sizes[:3])
        probe = PhaseProbe()
        result = momentum_phase(sizes, (0, 0, 0), convention="none",
                                array_module=probe)
        self.assertEqual(probe.grid_bytes, 24 * volume)
        self.assertTrue(probe.grid_live_when_ones_allocates)
        self.assertEqual(result.nbytes, 16 * volume)
        self.assertIsNone(probe.grid_ref())

        workspace_module = importlib.import_module(
            "lqcdweave.production.elemental.workspace")
        workspace = workspace_module.ElementalWorkspace(
            sizes, None, phase_convention="anchor")
        phase_refs = []
        build_observations = []
        original = workspace_module.momentum_phase

        def track_build(latt_size, momentum, displacement=(0, 0, 0),
                        convention="midpoint", *, array_module=np):
            if len(phase_refs) == 64:
                build_observations.append(
                    all(reference() is not None for reference in phase_refs[-64:]))
            value = original(latt_size, momentum, displacement, convention,
                             array_module=array_module)
            phase_refs.append(weakref.ref(value))
            return value

        with patch.object(workspace_module, "momentum_phase", track_build):
            for value in range(65):
                workspace.phase((value, 0, 0), convention="anchor")
        self.assertEqual(build_observations, [True])
        self.assertIsNone(phase_refs[0]())
        self.assertIsNotNone(phase_refs[1]())
        self.assertIsNotNone(phase_refs[-1]())


if __name__ == "__main__":
    unittest.main()
