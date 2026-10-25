"""Independent full-space VVV/VSV -> registered full-time Diagram checks.

Sources: test_baryon_elemental.determinant_reference and test_wilson_pipeline
stencil are independent numerical oracles. Basis-column assembly adapts
test_perambulator_dataset._basis_columns; scalar spin/mode loops adapt
test_baryon_diagram_frontend._independent_oracle, without E einsum labels.
"""

from contextlib import ExitStack
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

from test_baryon_elemental import determinant_reference
from production.elemental.test_elemental import Handle, reference_matrix
from tests.fixtures.laplace_reference import random_gauge
from test_wilson_pipeline import stencil

from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.adapters.meson_dataset import DatasetPropagator
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.baryon import (
    BaryonOperator, BaryonOperatorTerm, SparseSpinTensor,
)
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.storage.dataset import DatasetIndex, DatasetReader, DatasetWriter
from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator
from lqcdweave.production.elemental.baryon_planning import plan_indexed_baryon_elementals
from lqcdweave.contraction.correlator import calc_diagram_eval
from lqcdweave.contraction.diagram_measurement import (
    plan_indexed_diagram_measurement,
)
from lqcdweave.production.perambulator.planning import plan_indexed_perambulators


_LATTICE = (3, 1, 1, 3)
_CONFIGURATION = dict(ensemble="baryon-full-space-test", configuration="cfg-b",
                      lattice=list(_LATTICE))
_MASS = 0.4
_DIRAC = dict(action="wilson", mass=_MASS, xi_0=1.0, nu=1.0,
              clover_coeff_t=0.0, clover_coeff_r=0.0, r=1, lattice_spacing=1)
_COUNTS_SOURCE = (2, 1, 2)
_COUNTS_SINK = (1, 2, 2)
_SOURCE_SLOTS = (1, 2, 0)
_MOMENTUM = (1, 0, 0)
_PATHS = ((0,), (), (3,))
_PROJECTED = ({"id": 0, "terms": [
    {"paths": _PATHS, "coefficient": [0.5, 0.25]},
    {"paths": ((), (), ()), "coefficient": [-0.2, 0.4]},
]},)


def _columns(vectors, time):
    """Explicit physical-site/spin columns in the independent full-space basis."""
    shape = (*vectors.shape[:1], *vectors.shape[2:5], 4, 3)
    count = vectors.shape[1]
    columns = np.zeros((int(np.prod(shape)), 4 * count), dtype=np.complex128)
    for spin, mode in product(range(4), range(count)):
        field = np.zeros(shape, dtype=np.complex128)
        field[time, ..., spin, :] = vectors[time, mode]
        columns[:, spin * count + mode] = field.ravel()
    return columns


def _scalar_reference(source_spatial, sink_spatial, propagators,
                      source_spin, sink_spin, source_coefficient, sink_coefficient):
    # Independent source adjoint: apply gamma4 explicitly to each sparse spin.
    g4 = gamma(8)
    source_adjoint = np.zeros((4, 4, 4), dtype=np.complex128)
    for target in np.ndindex(source_adjoint.shape):
        for original in np.ndindex(source_spin.shape):
            source_adjoint[target] += source_spin[original].conjugate() * (
                g4[target[0], original[0]] * g4[target[1], original[1]]
                * g4[target[2], original[2]])
    value = 0j
    source_order = (2, 1, 0)
    for source_spin_index in np.ndindex(source_adjoint.shape):
        if source_adjoint[source_spin_index] == 0:
            continue
        for sink_spin_index in np.ndindex(sink_spin.shape):
            if sink_spin[sink_spin_index] == 0:
                continue
            for source_modes in np.ndindex(source_spatial.shape):
                for sink_modes in np.ndindex(sink_spatial.shape):
                    term = (source_adjoint[source_spin_index] * sink_spin[sink_spin_index]
                            * source_spatial[source_modes].conjugate()
                            * sink_spatial[sink_modes])
                    for sink_slot, source_slot in enumerate(_SOURCE_SLOTS):
                        raw_axis = source_order[source_slot]
                        term *= propagators[sink_slot][
                            sink_spin_index[sink_slot], source_spin_index[raw_axis],
                            sink_modes[sink_slot], source_modes[raw_axis]]
                    value += term
    return source_coefficient.conjugate() * sink_coefficient * value


class BaryonProductionMeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.links = random_gauge(_LATTICE, seed=719)
        self.gauge_id = self._register("gauge", "gauge", self.links,
            ("time", "z", "y", "x", "direction", "color_row", "color_column"))
        self.source_vectors, self.sink_vectors = [], []
        self.source_ids, self.sink_ids = [], []
        self.source_labels, self.sink_labels = [], []
        for name, counts, vectors, identities, labels in (
            ("source", _COUNTS_SOURCE, self.source_vectors, self.source_ids, self.source_labels),
            ("sink", _COUNTS_SINK, self.sink_vectors, self.sink_ids, self.sink_labels),
        ):
            for leg, count in enumerate(counts):
                rng = np.random.default_rng(37 + leg + (0 if name == "source" else 91))
                values = np.empty((3, count, 1, 1, 3, 3), dtype=np.complex128)
                for time in range(3):
                    raw = rng.normal(size=(9, count)) + 1j * rng.normal(size=(9, count))
                    values[time] = np.linalg.qr(raw)[0].T.reshape(count, 1, 1, 3, 3)
                mode_labels = [11 + leg * 17, 2 + leg * 19][:count]
                vectors.append(values)
                labels.append(mode_labels)
                identities.append(self._register(f"{name}-{leg}", "explicit_basis", values,
                    ("time", "mode", "z", "y", "x", "color"), mode_labels=mode_labels))
        field_shape = (3, 1, 1, 3, 4, 3)
        identity = np.eye(int(np.prod(field_shape)), dtype=np.complex128)
        self.matrix = np.stack([
            stencil(self.links, column.reshape(field_shape), _MASS, -1).ravel()
            for column in identity.T], axis=1)
        self.inverse = np.linalg.inv(self.matrix)
        self.propagator_ids = []
        self.propagator_reference = []
        for sink_leg, source_slot in enumerate(_SOURCE_SLOTS):
            source_leg = (2, 1, 0)[source_slot]
            table = {}
            for source_time, sink_time in product(range(3), repeat=2):
                source = _columns(self.source_vectors[source_leg], source_time)
                sink = _columns(self.sink_vectors[sink_leg], sink_time)
                projected = sink.conj().T @ self.inverse @ source
                table[source_time, sink_time] = projected.reshape(
                    4, _COUNTS_SINK[sink_leg], 4, _COUNTS_SOURCE[source_leg]).transpose(0, 2, 1, 3)
            self.propagator_reference.append(table)
            self.propagator_ids.append(self._publish_vsv(sink_leg, source_leg, table))

    def _register(self, name, kind, data, names, *, mode_labels=None):
        axes = [dict(name=axis, coordinates=(list(mode_labels) if axis == "mode" else
                                            list(range(size))))
                for axis, size in zip(names, data.shape, strict=True)]
        spec = DatasetSpec(dict(schema=1, kind=kind, configuration=_CONFIGURATION,
            axes=axes, dtype="complex128", bases={},
            parents=[] if kind == "gauge" else [self.gauge_id],
            definition=dict(realization=name),
            conventions=(dict(boundary="periodic") if kind == "gauge" else
                         dict(normalization="orthonormal"))))
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance=dict(test="synthetic-cpu-reference")) as writer:
            for time in range(3):
                writer.write(time, data[time])
            actual_id = writer.publish()
        self.index.register(directory)
        return actual_id

    def _publish_vsv(self, sink_leg, source_leg, reference):
        preview = plan_indexed_perambulators(
            self.index, self.gauge_id, products=("VSV",),
            basis_src_id=self.source_ids[source_leg],
            basis_snk_id=self.sink_ids[sink_leg],
            Ne_src=_COUNTS_SOURCE[source_leg], Ne_snk=_COUNTS_SINK[sink_leg],
            dirac_definition=_DIRAC, solver_tolerance=1e-11,
            temporal_boundary="antiperiodic",
        )
        spec = preview["stages"]["vsv"]["spec"]
        directory = self.root / f"vsv-{sink_leg}-{source_leg}"
        with DatasetWriter(directory, spec,
                            provenance={"test": "independent-explicit-stencil-inverse"}) as writer:
            for position in writer.missing:
                source_time = spec.coordinates("source_time")[position]
                row = np.stack([
                    reference[source_time, sink_time]
                    for sink_time in spec.coordinates("sink_time")
                ])
                writer.write(position, np.array(row, copy=True))
            identity = writer.publish()
        self.assertEqual(self.index.register(directory), identity)
        return identity

    def _elemental_reference(self, vectors, product_name, time):
        if product_name == "raw_paths":
            return determinant_reference(self.links, vectors, _PATHS, time, _MOMENTUM)
        if product_name == "projected":
            return sum(complex(*term["coefficient"]) * determinant_reference(
                self.links, vectors, term["paths"], time, _MOMENTUM)
                for term in _PROJECTED[0]["terms"])
        transformed = [values.copy() for values in vectors]
        for leg, distance in enumerate((1, 0, 1)):
            if distance:
                transport = sum(reference_matrix(self.links, (direction,) * distance,
                    time, phase="none") for direction in range(6)) / 6
                values = vectors[leg][time]
                transformed[leg][time] = (transport @ values.reshape(values.shape[0], -1).T).T.reshape(values.shape)
        return determinant_reference(self.links, transformed, ((), (), ()), time, _MOMENTUM)

    def _vertices_and_diagram(self, stack, source_id, sink_id, component, propagator_ids=None):
        source_reader = stack.enter_context(self.index.open(source_id))
        sink_reader = stack.enter_context(self.index.open(sink_id))
        source_adapter, sink_adapter = DatasetBaryonElemental(source_reader), DatasetBaryonElemental(sink_reader)
        source_spin = SparseSpinTensor(((0, 1, 2, 0.5 + 0.2j), (3, 2, 1, -0.25 + 0.4j)))
        sink_spin = SparseSpinTensor(((0, 1, 2, -0.3 + 0.5j), (3, 2, 1, 0.7 - 0.1j)))
        source_coefficient, sink_coefficient = 0.7 - 0.4j, -0.2 + 0.8j
        source_term = BaryonOperatorTerm(source_spin, component, source_coefficient, ("u", "d", "u"), 0)
        sink_term = BaryonOperatorTerm(sink_spin, component, sink_coefficient, ("d", "u", "u"), 0)
        vertices = dict(source=Baryon(source_adapter, BaryonOperator.from_terms("source", (source_term,)), source=True),
                        sink=Baryon(sink_adapter, BaryonOperator.from_terms("sink", (sink_term,))))
        source_symbol, sink_symbol = Symbol("ts"), Symbol("tf")
        edges = [[0, 0, 0] for _ in range(3)]
        for sink_slot, source_slot in enumerate(_SOURCE_SLOTS):
            edges[source_slot][sink_slot] = sink_slot + 1
        zero = [[0, 0, 0] for _ in range(3)]
        symbols = tuple(Propagator(flavor, Tag(source_slot, 0), Tag(3 + sink_slot, 1))
            for sink_slot, (flavor, source_slot) in enumerate(zip(("d", "u", "u"), _SOURCE_SLOTS)))
        diagram = Diagram(QuarkDiagram([[zero, edges], [zero, zero]]),
            [source_symbol, sink_symbol], ["source", "sink"], [None, *symbols])
        propagators = {symbol: DatasetPropagator({"vsv": stack.enter_context(self.index.open(actual_id))})
                       for symbol, actual_id in zip(symbols, propagator_ids or self.propagator_ids)}
        bindings = DiagramBindings(propagators, vertices, _CONFIGURATION["configuration"])
        return (diagram, bindings, source_symbol, sink_symbol, source_adapter,
                source_spin.dense(), sink_spin.dense(), source_coefficient, sink_coefficient)

    def _produce_elementals(self, product_name):
        components, component = {
            "raw_paths": ((_PATHS,), BaryonComponent.paths(_PATHS)),
            "projected": (_PROJECTED, BaryonComponent(("projected", 0))),
            "distance": (((1, 0, 1),), BaryonComponent(("distance", (1, 0, 1)))),
        }[product_name]
        identities = []
        for name, actual_bases, mode_labels in (
                ("source", self.source_ids, self.source_labels),
                ("sink", self.sink_ids, self.sink_labels)):
            counts = tuple(len(labels) for labels in mode_labels)
            preview = plan_indexed_baryon_elementals(
                self.index, self.gauge_id, actual_bases, product=product_name,
                components=components, momentum_list=(_MOMENTUM,), usedNe=counts,
                phase_convention="anchor", path_boundary="periodic",
            )
            spec = preview["dataset_spec"]
            selected = tuple(
                values[:, :len(spec.coordinates(f"mode{leg}"))]
                for leg, values in enumerate(self.source_vectors if name == "source"
                                             else self.sink_vectors)
            )
            generator = BaryonElementalGenerator(
                _LATTICE, Handle(self.links), tuple(Handle(values) for values in selected),
                momentum_list=(_MOMENTUM,), usedNe=counts, paths=(_PATHS,),
                phase_convention="anchor", path_boundary="periodic",
            ).load(_CONFIGURATION["configuration"])
            directory = self.root / f"{name}-{product_name}"
            with DatasetWriter(directory, spec,
                                provenance={"test": "public-baryon-kernel-reference"}) as writer:
                for position in writer.missing:
                    time = spec.coordinates("time")[position]
                    if product_name == "raw_paths":
                        row = generator.calc_path_blocks(time)["vvv"]
                    elif product_name == "projected":
                        row = generator.calc_path_blocks(time, rows=components)["vvv"]
                    else:
                        row = generator.calc_path_blocks(time, distances=components)["vvv"]
                    writer.write(position, np.array(row, copy=True))
                identity = writer.publish()
            self.assertEqual(self.index.register(directory), identity)
            identities.append(identity)
        return identities, component

    def test_all_elemental_recipes_match_full_space_diagram_oracle(self):
        for product_name in ("raw_paths", "projected", "distance"):
            with self.subTest(product=product_name), ExitStack() as stack:
                identities, component = self._produce_elementals(product_name)
                values = self._vertices_and_diagram(stack, *identities, component)
                diagram, bindings, source_symbol, sink_symbol, source_adapter = values[:5]
                source_spin, sink_spin, source_coefficient, sink_coefficient = values[5:]
                for reader, actual_bases, labels in ((source_adapter, self.source_ids, self.source_labels),
                        (bindings.vertex_map["sink"].elemental, self.sink_ids, self.sink_labels)):
                    metadata = reader.describe_block(component, 0, 0, "vvv")
                    for leg in range(3):
                        self.assertEqual(metadata.axis_bindings[f"mode{leg}"],
                                         (actual_bases[leg], tuple(labels[leg])))
                expected = np.empty((3, 3), dtype=np.complex128)
                for source_time, sink_time in product(range(3), repeat=2):
                    source_spatial = self._elemental_reference(self.source_vectors, product_name, source_time)
                    sink_spatial = self._elemental_reference(self.sink_vectors, product_name, sink_time)
                    expected[source_time, sink_time] = _scalar_reference(source_spatial, sink_spatial,
                        [table[source_time, sink_time] for table in self.propagator_reference],
                        source_spin, sink_spin, source_coefficient, sink_coefficient)
                plan = plan_indexed_diagram_measurement(
                    self.index, diagram, bindings=bindings,
                    source_time_symbol=source_symbol, sink_time_symbol=sink_symbol,
                )
                self.assertEqual(plan["measurement_spec"].coordinates("source_time"), [0, 1, 2])
                self.assertEqual(plan["measurement_spec"].coordinates("sink_time"), [0, 1, 2])
                actual = np.empty((3, 3), dtype=np.complex128)
                for source_time, sink_time in product(range(3), repeat=2):
                    actual[source_time, sink_time] = calc_diagram_eval(
                        diagram,
                        propagator_map=bindings.propagator_map,
                        vertex_map=bindings.vertex_map,
                        configuration=bindings.configuration,
                        usedNe=bindings.usedNe,
                        time_axes={source_symbol: source_time, sink_symbol: sink_time},
                        optimize="none", backend=np,
                    ).value
                np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
                # Caller-owned readers remain usable after read-only planning.
                np.testing.assert_allclose(source_adapter.block(component, 0, 0, "vvv"),
                    self._elemental_reference(self.source_vectors, product_name, 0), atol=3e-12)

    def test_same_shape_wrong_actual_source_leg_rejected_before_numeric_read(self):
        identities, component = self._produce_elementals("raw_paths")
        # Edge 1 should use source leg0; leg2 has the same extent and different actual ID.
        wrong = list(self.propagator_ids)
        table = {}
        for source_time, sink_time in product(range(3), repeat=2):
            source = _columns(self.source_vectors[2], source_time)
            sink = _columns(self.sink_vectors[1], sink_time)
            projected = sink.conj().T @ self.inverse @ source
            table[source_time, sink_time] = projected.reshape(
                4, _COUNTS_SINK[1], 4, _COUNTS_SOURCE[2]).transpose(0, 2, 1, 3)
        wrong[1] = self._publish_vsv(1, 2, table)
        with ExitStack() as stack:
            values = self._vertices_and_diagram(stack, *identities, component, propagator_ids=wrong)
            diagram, bindings, source_symbol, sink_symbol = values[:4]
            with patch.object(DatasetReader, "read", side_effect=AssertionError("metadata preflight read values")):
                with self.assertRaisesRegex(ValueError, "binding|identity|dataset ID|coordinates"):
                    plan_indexed_diagram_measurement(self.index, diagram, bindings=bindings,
                        source_time_symbol=source_symbol, sink_time_symbol=sink_symbol)


if __name__ == "__main__":
    unittest.main()
