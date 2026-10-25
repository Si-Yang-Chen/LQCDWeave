"""Shared perambulator fixtures with independent inverse oracles."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.fixtures.laplace_reference import random_gauge
from test_wilson_pipeline import stencil
from lqcdweave.adapters.production_dataset import (
    DatasetFieldHandle,
    DatasetPointHandle,
)
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.production import propagator_dataset_document
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, VectorBasis, PointBasis,
)
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.production.perambulator.planning import plan_indexed_perambulators
from fixtures.perambulator_script import produce_source_rows


_LATTICE = (2, 1, 1, 3)
_CONFIGURATION = dict(ensemble="perambulator-test", configuration="cfg-7",
                      lattice=list(_LATTICE))
_MASS = 0.4
_BOUNDARY = -1
_DIRAC = dict(action="wilson", mass=_MASS, xi_0=1.0, nu=1.0,
              clover_coeff_t=0.0, clover_coeff_r=0.0,
              r=1, lattice_spacing=1)


class _MatrixBackend:
    array_module = np

    def __init__(self, matrix, links, expected_dirac, expected_tolerance):
        self.matrix = matrix
        self.links = links
        self.expected_dirac = expected_dirac
        self.expected_tolerance = expected_tolerance
        self.parameters = None

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def load(self, gauge, parameters):
        np.testing.assert_array_equal(np.asarray(gauge[:]), self.links)
        for name in ("mass", "xi_0", "nu", "clover_coeff_t", "clover_coeff_r"):
            if parameters[name] != self.expected_dirac[name]:
                raise AssertionError(f"constructor {name} differs from the planned Dirac spec")
        if parameters["tol"] != self.expected_tolerance:
            raise AssertionError("constructor tolerance differs from the planned spec")
        self.parameters = dict(parameters)

    def solve_into(self, rhs, solution, *, tol, maxiter):
        if tol != self.expected_tolerance or maxiter != self.parameters["maxiter"]:
            raise AssertionError("solve controls differ from the planned numerical input")
        flat = np.asarray(rhs).reshape(len(rhs), -1)
        solution[...] = (np.linalg.solve(self.matrix, flat.T).T).reshape(rhs.shape)


class _PerambulatorDatasetFixture(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.links = random_gauge(_LATTICE, seed=173)
        self.gauge_id = self._publish_gauge()
        self.source_basis_values = self._basis_values(seed=13, count=3)
        self.sink_basis_values = self._basis_values(seed=29, count=2)
        self.source_basis_id = self._publish_basis(
            "basis-source", (8, 3, 5), self.source_basis_values)
        self.sink_basis_id = self._publish_basis(
            "basis-sink", (11, 2), self.sink_basis_values)
        self.source_point_rows = {
            0: [[1, 0, 0]], 1: [[0, 0, 0]], 2: [[1, 0, 0]],
        }
        self.sink_point_rows = {
            0: [[0, 0, 0], [1, 0, 0]],
            1: [[1, 0, 0], [0, 0, 0]],
            2: [[0, 0, 0], [1, 0, 0]],
        }
        self.source_points_id = self._publish_points(
            "points-source", (9,), self.source_point_rows)
        self.sink_points_id = self._publish_points(
            "points-sink", (6, 4), self.sink_point_rows)
        self.matrix = self._independent_matrix()
        self.inverse = np.linalg.inv(self.matrix)

    def _publish(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance=dict(test="fixture")) as writer:
            for position, coordinate in enumerate(spec.coordinates(spec.axis_names[0])):
                writer.write(position, np.asarray(rows[coordinate], dtype=spec.document["dtype"]))
            identity = writer.publish()
        self.index.register(directory)
        return identity

    def _publish_gauge(self):
        lt, lz, ly, lx = self.links.shape[:4]
        axes = [dict(name=name, coordinates=list(range(size)))
                for name, size in zip(("time", "z", "y", "x", "direction",
                                       "color_row", "color_column"), self.links.shape)]
        spec = DatasetSpec(dict(schema=1, kind="gauge", configuration=_CONFIGURATION,
            axes=axes, dtype="complex128", bases={}, parents=[], definition={},
            conventions=dict(boundary="periodic")))
        return self._publish("gauge", spec, {time: self.links[time] for time in range(lt)})

    def _basis_values(self, *, seed, count):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=(3, count, 1, 1, 2, 3)) + 1j*rng.normal(
            size=(3, count, 1, 1, 2, 3))
        for time in range(3):
            for mode in range(count):
                values[time, mode] /= np.linalg.norm(values[time, mode])
        return values

    def _publish_basis(self, name, mode_labels, values):
        spec = DatasetSpec(dict(schema=1, kind="explicit_basis",
            configuration=_CONFIGURATION,
            axes=[dict(name="time", coordinates=[0, 1, 2]),
                  dict(name="mode", coordinates=list(mode_labels)),
                  dict(name="z", coordinates=[0]), dict(name="y", coordinates=[0]),
                  dict(name="x", coordinates=[0, 1]),
                  dict(name="color", coordinates=[0, 1, 2])],
            dtype="complex128", bases={}, parents=[self.gauge_id],
            definition=dict(realization=name), conventions=dict(normalization="orthonormal")))
        return self._publish(name, spec, {time: values[time] for time in range(3)})

    def _publish_points(self, name, labels, rows, *, bad=None):
        if bad is not None:
            rows = dict(rows)
            rows[bad[0]] = bad[1]
        spec = DatasetSpec(dict(schema=1, kind="spatial_points",
            configuration=_CONFIGURATION,
            axes=[dict(name="time", coordinates=[0, 1, 2]),
                  dict(name="point", coordinates=list(labels)),
                  dict(name="xyz", coordinates=["x", "y", "z"])],
            dtype="int64", bases={}, parents=[], definition=dict(selection="explicit"),
            conventions=dict(representation="coordinate_delta", normalization="unit_delta")))
        return self._publish(name, spec, rows)

    def _independent_matrix(self):
        shape = (3, 1, 1, 2, 4, 3)
        dimension = int(np.prod(shape))
        columns = np.eye(dimension, dtype=np.complex128)
        return np.column_stack([
            stencil(self.links, columns[:, column].reshape(shape), _MASS, _BOUNDARY).ravel()
            for column in range(dimension)
        ])

    @staticmethod
    def _basis_columns(rows, time, count):
        shape = (3, 1, 1, 2, 4, 3)
        result = np.zeros((int(np.prod(shape)), 4 * count), dtype=np.complex128)
        for spin in range(4):
            for mode in range(count):
                field = np.zeros(shape, dtype=np.complex128)
                field[time, 0, 0, :, spin, :] = rows[time, mode]
                result[:, spin * count + mode] = field.ravel()
        return result

    @staticmethod
    def _point_columns(rows, time, count):
        shape = (3, 1, 1, 2, 4, 3)
        result = np.zeros((int(np.prod(shape)), 4 * count * 3), dtype=np.complex128)
        for spin in range(4):
            for point in range(count):
                x, y, z = rows[time][point]
                for color in range(3):
                    column = (spin * count + point) * 3 + color
                    result[np.ravel_multi_index((time, z, y, x, spin, color), shape), column] = 1
        return result

    def _plan_products(self, products, **options):
        values = dict(
            dirac_definition=_DIRAC,
            solver_tolerance=1e-11,
            temporal_boundary="antiperiodic",
        )
        values.update(options)
        return plan_indexed_perambulators(
            self.index, self.gauge_id, products=products, **values,
        )

    def _produce_products(self, preview, products):
        identities = {}
        for product in products:
            name = product.lower()
            spec = preview["stages"][name]["spec"]
            bindings = spec.document["bases"]
            source_mode = bindings.get("source_mode")
            sink_mode = bindings.get("sink_mode")
            source_point = bindings.get("source_point")
            sink_point = bindings.get("sink_point")
            with ExitStack() as stack:
                gauge = DatasetFieldHandle(stack.enter_context(self.index.open(self.gauge_id)))

                def field_handle(identity):
                    if identity is None:
                        return None
                    return DatasetFieldHandle(stack.enter_context(self.index.open(identity)))

                def point_handle(identity):
                    if identity is None:
                        return None
                    return DatasetPointHandle(stack.enter_context(self.index.open(identity)))

                basis_src_id = source_mode["data_id"] if source_mode else None
                basis_snk_id = sink_mode["data_id"] if sink_mode else None
                basis_src = field_handle(basis_src_id)
                basis_snk = field_handle(basis_snk_id)
                points_snk = point_handle(
                    sink_point["data_id"] if sink_point else None)
                points_src = point_handle(
                    source_point["data_id"] if source_point else None)
                src_basis = (VectorBasis(basis_src,count=len(source_mode["indices"]))
                             if source_mode else
                             PointBasis(points_src,count=len(source_point["indices"])))
                sink_basis = (VectorBasis(basis_snk,count=len(sink_mode["indices"]))
                              if sink_mode else
                              PointBasis(points_snk,count=len(sink_point["indices"])))
                backend = _MatrixBackend(self.matrix, self.links, _DIRAC, 1e-11)
                generator = PerambulatorGenerator(
                    _LATTICE, gauge, _MASS, 1e-11, 120,
                    xi_0=_DIRAC["xi_0"], nu=_DIRAC["nu"],
                    clover_coeff_t=_DIRAC["clover_coeff_t"],
                    clover_coeff_r=_DIRAC["clover_coeff_r"],
                    t_boundary=_BOUNDARY,
                    contract_prec=np.dtype(spec.document["dtype"]).str,
                    MRHS=True,
                    rhs_batch_size=4,
                    sink_ne_block_size=1,
                    backend=backend,
                )
                path = self.root / "numerical-products" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with DatasetWriter(path, spec, provenance={"test": "independent-inverse"}) as writer:
                    produce_source_rows(
                        writer, generator, spec.coordinates("source_time"),
                        _CONFIGURATION["configuration"], product,
                        src=src_basis,sink=sink_basis,
                    )
                    writer.publish()
            identities[name] = self.index.register(path)
        return identities

    def _assert_products_match_inverse(self, identities, products):
        for product in products:
            name = product.lower()
            with self.index.open(identities[name]) as reader:
                self.assertEqual(reader.spec.coordinates("source_time"), [0, 1, 2])
                self.assertEqual(reader.spec.coordinates("sink_time"), [0, 1, 2])
                bases = reader.spec.document["bases"]

                def point_rows(role):
                    identity = bases[role]["data_id"]
                    if identity == self.source_points_id:
                        return self.source_point_rows
                    if identity == self.sink_points_id:
                        return self.sink_point_rows
                    self.fail(f"unexpected point binding for {role}: {identity}")

                for source_time in range(3):
                    for sink_time in range(3):
                        if name == "vsv":
                            source = self._basis_columns(
                                self.source_basis_values, source_time,
                                len(reader.spec.coordinates("source_mode")))
                            sink = self._basis_columns(
                                self.sink_basis_values, sink_time,
                                len(reader.spec.coordinates("sink_mode")))
                            expected = (sink.conj().T @ self.inverse @ source).reshape(
                                4, len(reader.spec.coordinates("sink_mode")),
                                4, len(reader.spec.coordinates("source_mode")),
                            ).transpose(0, 2, 1, 3)
                        elif name == "psv":
                            source = self._basis_columns(
                                self.source_basis_values, source_time,
                                len(reader.spec.coordinates("source_mode")))
                            sink = self._point_columns(
                                self.sink_point_rows, sink_time,
                                len(reader.spec.coordinates("sink_point")))
                            expected = (sink.conj().T @ self.inverse @ source).reshape(
                                4, len(reader.spec.coordinates("sink_point")), 3,
                                4, len(reader.spec.coordinates("source_mode")),
                            ).transpose(0, 3, 1, 2, 4)
                        elif name == "vsp":
                            source = self._point_columns(
                                self.source_point_rows, source_time,
                                len(reader.spec.coordinates("source_point")))
                            sink = self._basis_columns(
                                self.sink_basis_values, sink_time,
                                len(reader.spec.coordinates("sink_mode")))
                            expected = (sink.conj().T @ self.inverse @ source).reshape(
                                4, len(reader.spec.coordinates("sink_mode")),
                                4, len(reader.spec.coordinates("source_point")), 3,
                            ).transpose(0, 2, 1, 3, 4)
                        else:
                            source = self._point_columns(
                                point_rows("source_point"), source_time,
                                len(reader.spec.coordinates("source_point")))
                            sink = self._point_columns(
                                point_rows("sink_point"), sink_time,
                                len(reader.spec.coordinates("sink_point")))
                            expected = (sink.conj().T @ self.inverse @ source).reshape(
                                4, len(reader.spec.coordinates("sink_point")), 3,
                                4, len(reader.spec.coordinates("source_point")), 3,
                            ).transpose(0, 3, 1, 2, 4, 5)
                        tolerance = 3e-5 if reader.spec.document["dtype"] == "complex64" else 3e-12
                        with self.subTest(product=name, source=source_time, sink=sink_time):
                            np.testing.assert_allclose(
                                reader.read(source_time=source_time, sink_time=sink_time),
                                expected, atol=tolerance, rtol=tolerance,
                            )


class PerambulatorNumericsTests(_PerambulatorDatasetFixture):
    def test_distinct_endpoints_all_products_match_independent_inverse(self):
        products = ("VSV", "PSV", "VSP", "PSP")
        preview = self._plan_products(products,
            basis_src_id=self.source_basis_id, basis_snk_id=self.sink_basis_id,
            points_src_id=self.source_points_id, points_snk_id=self.sink_points_id,
            Ne_src=1, Ne_snk=2, Np_src=1, Np_snk=2)
        self.assertEqual(set(preview["stages"]),
                         {"gauge", "basis_src", "basis_snk", "points_src",
                          "points_snk", "vsv", "psv", "vsp", "psp"})
        identities = self._produce_products(preview, products)
        self.assertEqual(set(identities), {product.lower() for product in products})
        self._assert_products_match_inverse(identities, products)

    def test_pure_psp_complex64_has_no_basis_dependency_and_matches_reference(self):
        products = ("PSP",)
        preview = self._plan_products(products, points_id=self.source_points_id,
            Np=1, dtype="complex64")
        self.assertEqual(set(preview["stages"]), {"gauge", "points", "psp"})
        identities = self._produce_products(preview, products)
        with self.index.open(identities["psp"]) as reader:
            self.assertEqual(reader.spec.document["dtype"], "complex64")
            self.assertEqual(reader.spec.document["parents"],
                             [self.gauge_id, self.source_points_id])
        self._assert_products_match_inverse(identities, products)











if __name__ == "__main__":
    unittest.main()
