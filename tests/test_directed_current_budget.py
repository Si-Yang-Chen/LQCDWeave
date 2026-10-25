"""Logical admission and row ownership for indexed eight-link V/V current."""
import math
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import lqcdweave.production.elemental.workspace as workspace_module
from production.elemental.test_elemental import fixture, Handle, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.production import (
    directed_current_dataset_document,
    estimate_directed_current_memory,
)
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class DirectedCurrentBudgetTests(unittest.TestCase):
    def setUp(self):
        sizes, links, vectors, _ = fixture()
        cfg = dict(ensemble="synthetic", configuration="cfg", lattice=list(sizes))
        axis = lambda name, values: dict(name=name, coordinates=list(values))
        gauge_axes = [axis(name, range(extent)) for name, extent in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"),
            links.shape)]
        basis_axes = [axis(name, values) for name, values in zip(
            ("time", "mode", "z", "y", "x", "color"),
            (range(sizes[3]), (4, 1, 9), range(sizes[2]), range(sizes[1]),
             range(sizes[0]), range(3)))]
        gauge_id, basis_id = "budget-gauge", "budget-basis"
        gauge_spec = DatasetSpec(dict(schema=1, kind="gauge", configuration=cfg,
            axes=gauge_axes, dtype="complex128", bases={}, definition={},
            conventions={"boundary": "periodic"}, parents=[]))
        basis_spec = DatasetSpec(dict(schema=1, kind="explicit_basis", configuration=cfg,
            axes=basis_axes, dtype="complex128", bases={}, definition={},
            conventions={}, parents=[gauge_id]))
        self.fixture = SimpleNamespace(
            sizes=sizes, links=links, vectors=vectors, gspec=gauge_spec,
            bspec=basis_spec, gid=gauge_id, bid=basis_id,
            momenta=((1, -1, 0), (0, 0, 0)))

    def _make_spec(self, gauge_spec=None, basis_spec=None, *,
                   gauge_id=None, basis_id=None, momenta=None, used_ne=2,
                   boundary="periodic"):
        gauge_spec = gauge_spec or self.fixture.gspec
        basis_spec = basis_spec or self.fixture.bspec
        gauge_id = gauge_id or self.fixture.gid
        basis_id = basis_id or self.fixture.bid
        momenta = self.fixture.momenta if momenta is None else momenta
        _, document = directed_current_dataset_document(
            gauge_spec, basis_spec, gauge_id=gauge_id, basis_id=basis_id,
            momentum_list=momenta, usedNe=used_ne, boundary=boundary)
        return DatasetSpec(document)

    def _reference(self, links, vectors, time, direction, momentum, *,
                   ne, boundary):
        field_time = (time + (1 if direction == 6 else
                              -1 if direction == 7 else 0)) % links.shape[0]
        left = vectors[time, :ne].reshape(ne, -1)
        right = vectors[field_time, :ne].reshape(ne, -1)
        matrix = reference_matrix(links, (direction,), time, momentum,
                                  boundary=boundary)
        return left.conj() @ matrix @ right.T

    def test_estimator_models_dtype_phase_lru_path_kernel_and_writer_bytes(self):
        f = self.fixture
        gauge64_doc = f.gspec.document
        gauge64_doc["dtype"] = "complex64"
        basis64_doc = f.bspec.document
        basis64_doc["dtype"] = "complex64"
        gauge64, basis64 = DatasetSpec(gauge64_doc), DatasetSpec(basis64_doc)

        # Ten momenta generate 70 distinct current phase keys (six spatial
        # signed displacements plus one temporal zero displacement per key).
        momenta = tuple((value, 0, 0) for value in range(10))
        output128 = self._make_spec(momenta=momenta)
        output64 = self._make_spec(
            gauge_spec=gauge64, basis_spec=basis64, momenta=momenta)
        reserve = 37
        staging = 53
        estimate128, device128 = estimate_directed_current_memory(
            {f.gid: f.gspec, f.bid: f.bspec}, output128,
            gauge_id=f.gid, basis_id=f.bid,
            workspace_reserve_bytes=reserve, host_staging_bytes=staging)
        estimate64, device64 = estimate_directed_current_memory(
            {f.gid: gauge64, f.bid: basis64}, output64,
            gauge_id=f.gid, basis_id=f.bid,
            workspace_reserve_bytes=reserve, host_staging_bytes=staging)

        parent128 = (math.prod(f.gspec.shape) * 16 + math.prod(f.bspec.shape) * 16)
        parent64 = (math.prod(gauge64.shape) * 8 + math.prod(basis64.shape) * 8)
        self.assertEqual(estimate128.input_bytes, parent128)
        self.assertEqual(estimate64.input_bytes, parent64)
        self.assertEqual(estimate128.workspace_bytes, reserve + staging)
        self.assertEqual(estimate64.workspace_bytes, reserve + staging)
        self.assertEqual(device128.logical_peak_bytes, 0)
        self.assertEqual(device64.logical_peak_bytes, 0)

        lx, ly, lz, _lt = f.sizes
        volume = lx * ly * lz
        ne = 2
        vector_bytes = ne * volume * 3 * 16
        link_bytes = volume * 3 * 3 * 16
        phase_bytes = 64 * volume * 16 + 64 * volume
        self.assertEqual(phase_bytes, 1088 * volume)
        pair_bytes = ne * ne * 16
        expected_calculation_intermediate = max(
            64 * volume * 16 + 6 * vector_bytes + 2 * pair_bytes + link_bytes,
            64 * volume * 16 + 3 * vector_bytes + pair_bytes + 5 * link_bytes,
            phase_bytes + 3 * vector_bytes + pair_bytes + link_bytes,
        )
        self.assertEqual(estimate128.intermediate_bytes,
                         expected_calculation_intermediate)
        self.assertEqual(estimate128.assembly_extra_bytes, 0)
        self.assertEqual(estimate128.output_bytes,
                         8 * len(momenta) * ne * ne * 16)

        # Observe the bounded phase LRU in the shared path calculation.
        phase_references = []
        original_phase = workspace_module.momentum_phase
        def tracked_phase(*args, **kwargs):
            value = original_phase(*args, **kwargs)
            phase_references.append(weakref.ref(value))
            return value

        generator = ElementalGenerator(
            f.sizes, Handle(f.links),
            Handle(f.vectors), momentum_list=momenta,
            usedNe=2,paths=[(d,) for d in range(8)],path_version="v2",
            phase_convention="midpoint").load("cfg")
        with patch.object(workspace_module, "momentum_phase", tracked_phase):
            row = generator.calc_path_blocks(0)["v2v"]
        self.assertEqual(len(phase_references), 70)
        self.assertEqual(sum(reference() is not None for reference in phase_references), 64)
        for direction in range(8):
            for momentum_index, momentum in enumerate(momenta):
                np.testing.assert_allclose(
                    row[direction, momentum_index],
                    self._reference(f.links, f.vectors, 0, direction, momentum,
                                    ne=2, boundary="periodic"),
                    atol=4e-12, rtol=4e-12)
        del row, generator
        self.assertTrue(all(reference() is None for reference in phase_references))


if __name__ == "__main__":
    unittest.main()
