"""Metadata-first memory admission for indexed point Elementals."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from production.elemental.test_elemental import fixture
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.planning.production import (
    _point_path_transport_build_bytes,
    estimate_point_elemental_memory,
)
from lqcdweave.production.elemental.transport import path_transport
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader, register_npy,
)
from lqcdweave.production.elemental.planning import (
    _normalize_point_elemental_request_options,
)
from lqcdweave.production.elemental.planning import _point_elemental_requests_from_specs


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class PointElementalProductionBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.specs_by_id = {}
        self.sizes, self.links, self.vectors, self.left_values = fixture()
        self.left_values = self.left_values.copy()
        self.left_labels = [8, 2, 5]
        self.cfg = dict(ensemble="synthetic", configuration="cfg",
                        lattice=list(self.sizes))
        self.input_directories = {}
        self.gauge_id = self._save("gauge", self._spec("gauge", (
            ("time", range(3)), ("z", range(2)), ("y", range(2)),
            ("x", range(3)), ("direction", range(4)),
            ("color_row", range(3)), ("color_column", range(3))), "complex128"),
            self.links)
        self.basis_id = self._save("basis", self._spec("explicit_basis", (
            ("time", range(3)), ("mode", [4, 1, 9]), ("z", range(2)),
            ("y", range(2)), ("x", range(3)), ("color", range(3))),
            "complex128"), self.vectors)
        self.left_id = self._save_points("points-left", self.left_values,
                                         self.left_labels)
        self.right_labels = [7, 2, 9, 4]
        self.right_values = np.asarray([
            [(0, 0, 0), (2, 0, 1), (1, 1, 1), (1, 0, 0)],
            [(2, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)],
            [(1, 0, 1), (0, 1, 0), (2, 1, 1), (2, 0, 0)],
        ], dtype=np.int64)
        self.right_id = self._save_points("points-right", self.right_values,
                                          self.right_labels)
        self.paths = ((), (0,), (3,))
        self.momenta = ((1, -1, 0), (0, 0, 0))

    def _spec(self, kind, axes, dtype, conventions=None):
        return DatasetSpec(dict(
            schema=1, kind=kind, configuration=self.cfg,
            axes=[axis(name, coordinates) for name, coordinates in axes],
            dtype=dtype, bases={}, parents=[], definition={},
            conventions={} if conventions is None else conventions,
        ))

    def _save(self, name, spec, values):
        source = self.root / f"{name}.npy"
        np.save(source, values)
        directory = self.root / name
        identity = register_npy(directory, source, spec,
                                provenance={"test": "point elemental budget"})
        self.index.register(directory)
        self.input_directories[name] = directory
        self.specs_by_id[identity] = spec
        return identity

    def _save_points(self, name, values, labels):
        spec = self._spec("spatial_points", (
            ("time", range(self.sizes[3])), ("point", labels),
            ("xyz", ("x", "y", "z"))), "int64",
            conventions=dict(representation="coordinate_delta",
                             normalization="unit_delta"))
        return self._save(name, spec, np.asarray(values, dtype=np.int64))

    def _options(self, *, products=("v2p", "p2v", "p2p"),
                 basis_id=None, points_id=None, paths=None, momenta=None,
                 usedNp=2, usedNp_right=3, point_right_labels=None,
                 points_right_id=None, phase="midpoint", current_schema=None,
                 path_boundary="periodic"):
        vector_products = set(products) & {"v2p", "p2v"}
        if basis_id is None and vector_products:
            basis_id = self.basis_id
        if points_id is None and set(products) & {"p2v", "p2p"}:
            points_id = self.left_id
        if points_right_id is None and set(products) & {"v2p", "p2p"}:
            points_right_id = self.right_id
        return _normalize_point_elemental_request_options(
            self.gauge_id, basis_id, points_id,
            paths=(self.paths if paths is None else paths),
            momentum_list=(self.momenta if momenta is None else momenta),
            sectors=products, usedNe=2 if vector_products else None, usedNp=usedNp,
            phase_convention=phase, points_right_id=points_right_id,
            usedNp_right=usedNp_right if set(products) & {"v2p", "p2p"} else None,
            point_right_labels=point_right_labels, current_schema=current_schema,
            path_boundary=path_boundary,
        )

    def _metadata_requests(self, **kwargs):
        options = self._options(**kwargs)
        requests = _point_elemental_requests_from_specs(
            self.specs_by_id[self.gauge_id],
            self.specs_by_id[options["basis_id"]] if options["vector_products"] else None,
            self.specs_by_id[options["points_id"]] if options["uses_left"] else None,
            self.specs_by_id[options["effective_right_id"]] if options["uses_right"] else None,
            options)
        actual_ids = {name: request.data_id for name, request in requests.items()
                      if request.data_id is not None}
        input_ids = {self.gauge_id}
        if options["vector_products"]:
            input_ids.add(options["basis_id"])
        if options["uses_left"]:
            input_ids.add(options["points_id"])
        if options["uses_right"]:
            input_ids.add(options["effective_right_id"])
        input_specs = {identity: self.specs_by_id[identity] for identity in input_ids}
        output_specs = {product: requests[product].resolve(actual_ids)
                        for product in options["products"]}
        return input_specs, requests, output_specs
    def test_pure_model_matches_hand_calculated_rectangular_workspace_lifetimes(self):
        # One empty path and one zero momentum make the source arrays explicit:
        # gauge 20,736 B; full left/right point tables 216/288 B; left/right
        # selected copies 144/216 B; owned gauge links 20,736 B.
        with (patch.object(self.index, "open", side_effect=AssertionError("payload open")),
              patch.object(DatasetReader, "read", side_effect=AssertionError("payload read"))):
            input_specs, _requests, output_specs = self._metadata_requests(
                products=("p2p",), paths=((),), momenta=((0, 0, 0),),
                usedNp=2, usedNp_right=3, phase="none")
            estimates = estimate_point_elemental_memory(
                input_specs, output_specs, target="cpu", workspace_reserve_bytes=4096,
                host_staging_bytes=2048)
        host, device = estimates["p2p"]
        self.assertEqual(host.input_bytes, 20_736 + 216 + 288)
        self.assertEqual(host.snapshot_bytes, 4 * 3 * 12 * 9 * 16)
        self.assertEqual(host.intermediate_bytes, 4 * 3 * 12 * 9 * 16)
        self.assertEqual(host.workspace_bytes, 6144)
        self.assertEqual(host.logical_peak_bytes, 68_856)
        self.assertEqual(device.logical_peak_bytes, 0)

        # Both roles bind one actual parent here. Its full payload is counted
        # once, while the runtime still creates and owns left/right views.
        same_inputs, _requests, same_outputs = self._metadata_requests(
            products=("p2p",), paths=((),), momenta=((0, 0, 0),),
            usedNp=2, usedNp_right=3,
            points_right_id=self.left_id, phase="none")
        same = estimate_point_elemental_memory(
            same_inputs, same_outputs, target="cpu", workspace_reserve_bytes=4096,
            host_staging_bytes=2048)["p2p"][0]
        self.assertEqual(same.input_bytes, 20_736 + 216)
        self.assertEqual(host.input_bytes - same.input_bytes, 288)
        self.assertEqual(same.snapshot_bytes, host.snapshot_bytes)

        # A substantially rectangular metadata-only model exercises the
        # generator's complete right-endpoint translation table, even when the
        # left side has one point. A one-slice lattice makes the two/three
        # right-table allocations exceed the ordinary load phase, so the
        # returned peak proves the coordinate scratch is represented.
        small_cfg = dict(ensemble="synthetic", configuration="one-slice",
                         lattice=[1, 1, 1, 1])
        gauge = DatasetSpec(dict(
            schema=1, kind="gauge", configuration=small_cfg,
            axes=[axis("time", [0]), axis("z", [0]), axis("y", [0]),
                  axis("x", [0]), axis("direction", range(4)),
                  axis("color_row", range(3)), axis("color_column", range(3))],
            dtype="complex128", bases={}, parents=[], definition={}, conventions={}))
        left = DatasetSpec(dict(
            schema=1, kind="spatial_points", configuration=small_cfg,
            axes=[axis("time", [0]), axis("point", [3]),
                  axis("xyz", ("x", "y", "z"))],
            dtype="int64", bases={}, parents=[], definition={},
            conventions=dict(representation="coordinate_delta",
                             normalization="unit_delta")))
        large_right = DatasetSpec(dict(
            schema=1, kind="spatial_points", configuration=small_cfg,
            axes=[axis("time", [0]), axis("point", range(512)),
                  axis("xyz", ("x", "y", "z"))],
            dtype="int64", bases={}, parents=[], definition={},
            conventions=dict(representation="coordinate_delta",
                             normalization="unit_delta")))
        options = _normalize_point_elemental_request_options(
            "small-gauge", None, "small-left", paths=((), (4,)),
            momentum_list=((0, 0, 0),), sectors=("p2p",), usedNe=None,
            usedNp=1, phase_convention="none", points_right_id="large-right",
            usedNp_right=512)
        from lqcdweave.production.elemental.planning import _point_elemental_requests_from_specs
        requests = _point_elemental_requests_from_specs(
            gauge, None, left, large_right, options)
        resolved = {name: request.data_id for name, request in requests.items()
                    if name in {"gauge", "points", "points_right"}}
        outputs = {"p2p": requests["p2p"].resolve(resolved)}
        large_inputs = {"small-gauge": gauge, "small-left": left,
                        "large-right": large_right}
        large = estimate_point_elemental_memory(
            large_inputs, outputs, target="cpu")["p2p"][0]
        matrix_bytes = 9 * 16
        self.assertEqual(large.input_bytes, 576 + 24 + 12_288)
        self.assertEqual(large.snapshot_bytes, 576 + 24 + 12_288)
        self.assertEqual(large.intermediate_bytes, 37_776)
        self.assertGreater(large.intermediate_bytes, 2 * 512 * 3 * 8)
        self.assertEqual(_point_path_transport_build_bytes((GaugePath((1,)),),
                                                           matrix_bytes),
                         4 * matrix_bytes)
        self.assertEqual(_point_path_transport_build_bytes((GaugePath((4,)),),
                                                           matrix_bytes),
                         5 * matrix_bytes)
        workspace_links = self.links.transpose(4, 0, 1, 2, 3, 5, 6)
        positive_y = path_transport(workspace_links, GaugePath((1,)), 0)
        negative_y = path_transport(workspace_links, GaugePath((4,)), 0)
        np.testing.assert_array_equal(positive_y, workspace_links[1, 0])
        expected_negative = np.roll(workspace_links[1, 0], 1, axis=1)
        expected_negative = expected_negative.conj().swapaxes(-1, -2)
        np.testing.assert_array_equal(negative_y, expected_negative)


if __name__ == "__main__":
    unittest.main()
