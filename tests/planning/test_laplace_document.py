"""Module tests split from test_laplace_dataset.py; shared numerical fixtures stay unchanged."""

from test_laplace_dataset import _LaplaceDatasetFixture, _axis

import numpy as np

from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.planning.laplace import (
    estimate_laplace_memory,
    laplace_basis_document,
    laplace_eigenvalues_document,
)


class LaplaceDatasetTests(_LaplaceDatasetFixture):
    def test_laplace_memory_model_matches_independent_hand_accounting(self):
        gauge_spec = self.index.observe_metadata(self.gauge_id)["spec"]
        host, device = estimate_laplace_memory(
            gauge_spec, 2, target="cpu", workspace_reserve_bytes=17,
            host_staging_bytes=19)
        # Lt=3, V=2, Ne=2: input=3456, owned links=3456 and the
        # raw host stack=3456. These inputs determine the 10404-byte CPU
        # loading phase, including the two caller declarations.
        self.assertEqual(host.input_bytes, 3456)
        self.assertEqual(host.snapshot_bytes, 6912)
        self.assertEqual(host.assembly_extra_bytes, 19)
        self.assertEqual(host.workspace_bytes, 17)
        self.assertEqual(host.logical_peak_bytes, 10404)
        self.assertEqual(device.logical_peak_bytes, 0)

        gpu_host, gpu_device = estimate_laplace_memory(
            gauge_spec, 2, target="gpu", workspace_reserve_bytes=17,
            host_staging_bytes=19)
        self.assertEqual(gpu_host.logical_peak_bytes, 6948)
        # On the declared GPU route, the device raw array and c128 links
        # overlap during loading; the opaque reserve is charged there too.
        self.assertEqual(gpu_device.input_bytes, 3456)
        self.assertEqual(gpu_device.snapshot_bytes, 6912)
        self.assertEqual(gpu_device.workspace_bytes, 17)
        self.assertEqual(gpu_device.logical_peak_bytes, 10385)

    def test_pure_builders_keep_the_wilson_default_documents(self):
        cfg = dict(ensemble="laplace-test", configuration="cfg-laplace",
                   lattice=[2, 1, 1, 3])
        times = (0, 1, 2)
        ref_gauge, ref_basis = dict(request="gauge"), dict(request="basis")
        basis = laplace_basis_document(cfg, times, ref_gauge, 2, 1e-11)
        expected_basis = dict(schema=1, kind="laplace_basis", configuration=cfg,
            axes=[_axis("time", times), _axis("mode", range(2)),
                  _axis("z", range(1)), _axis("y", range(1)),
                  _axis("x", range(2)), _axis("color", range(3))],
            dtype="complex128", bases={},
            definition=dict(gauge_id=ref_gauge,
                operator="positive_spatial_laplacian", solver_tolerance=1e-11),
            conventions=dict(normalization="orthonormal",
                spatial_boundary="periodic", phase="first_component_real"),
            parents=[ref_gauge])
        self.assertEqual(canonical_json(basis), canonical_json(expected_basis))
        values = laplace_eigenvalues_document(cfg, times, 2, ref_basis,
                                              ref_gauge, basis)
        expected_values = dict(schema=1, kind="laplace_eigenvalues",
            configuration=cfg,
            axes=[_axis("time", times), _axis("mode", range(2))],
            dtype="float64", bases=dict(mode=dict(data_id=ref_basis,
                                                       indices=[0, 1])),
            definition=expected_basis["definition"],
            conventions={}, parents=[ref_basis, ref_gauge])
        self.assertEqual(canonical_json(values), canonical_json(expected_values))
