"""Published Baryon datasets retain tagged components and real mode identities."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import BaryonComponent, GaugePath
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.planning.diagram_measurement import validate_diagram_input_specs
from lqcdweave.runtime.datasets import DatasetView


_CONFIG = dict(ensemble="baryon-binding-fixture", configuration="cfg-baryon-binding",
               lattice=[1, 1, 1, 2])


def _axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


def _publish(root, index, name, spec, values):
    directory = Path(root) / name
    with DatasetWriter(directory, spec, provenance={"test": "published-baryon-binding"}) as writer:
        for position in writer.missing:
            writer.write(position, values[position])
        data_id = writer.publish()
    index.register(directory)
    return data_id


def _base_specs(root, index):
    gauge = DatasetSpec(dict(
        schema=1, kind="gauge", configuration=_CONFIG,
        axes=[_axis("time", (1, 0)), _axis("z", [0]), _axis("y", [0]),
              _axis("x", [0]), _axis("direction", range(4)),
              _axis("color_row", range(3)), _axis("color_column", range(3))],
        dtype="complex128", bases={}, definition={},
        conventions={"boundary": "periodic"}, parents=[],
    ))
    rng = np.random.default_rng(120)
    gauge_values = rng.normal(size=gauge.shape) + 1j * rng.normal(size=gauge.shape)
    gauge_id = _publish(root, index, "gauge", gauge, gauge_values.astype(np.complex128))

    basis = DatasetSpec(dict(
        schema=1, kind="explicit_basis", configuration=_CONFIG,
        axes=[_axis("time", (1, 0)), _axis("mode", [7, 2, 9]),
              _axis("z", [0]), _axis("y", [0]), _axis("x", [0]),
              _axis("color", range(3))],
        dtype="complex128", bases={}, definition={"basis_source": "fixture"},
        conventions={}, parents=[],
    ))
    basis_values = rng.normal(size=basis.shape) + 1j * rng.normal(size=basis.shape)
    basis_id = _publish(root, index, "basis", basis, basis_values.astype(np.complex128))
    return gauge, gauge_id, basis, basis_id


def _basis_view(spec, labels):
    document = spec.document
    for axis in document["axes"]:
        if axis["name"] == "mode":
            axis["coordinates"] = list(labels)
    return document


def _baryon_spec(gauge, gauge_id, basis, basis_id, *, product="raw_paths"):
    labels = ([7, 2], [7, 2, 9], [7])
    coordinates = {
        "raw_paths": [dict(encoding="ordered_path_triple_v1",
                           paths=[[0], [3], []])],
        "projected": [dict(encoding="projected_row_id_v1", id=0),
                      dict(encoding="projected_row_id_v1", id=1)],
        "distance": [dict(encoding="distance_triple_v1", legs=[0, 1, 2]),
                     dict(encoding="distance_triple_v1", legs=[2, 0, 1])],
    }[product]
    encoding = {
        "raw_paths": "ordered_path_triples_v1",
        "projected": "projected_row_ids_v1",
        "distance": "distance_triples_v1",
    }[product]
    definition = dict(
        gauge_id=gauge_id,
        gauge_purpose="elemental_path_transport",
        product=product,
        component_encoding=encoding,
        input_roles={"gauge": gauge_id, "basis_leg0": basis_id,
                     "basis_leg1": basis_id, "basis_leg2": basis_id},
        input_views={"gauge": gauge.document,
                     "basis_leg0": _basis_view(basis, labels[0]),
                     "basis_leg1": _basis_view(basis, labels[1]),
                     "basis_leg2": _basis_view(basis, labels[2])},
    )
    if product == "projected":
        definition["projected_rows"] = [
            dict(id=0, terms=[dict(paths=[[], [], []], coefficient=[0.5, 0.25])]),
            dict(id=1, terms=[dict(paths=[[0], [], []], coefficient=[1.0, 0.0]),
                              dict(paths=[[0], [], []], coefficient=[-1.0, 0.0])]),
        ]
    elif product == "distance":
        definition["distance_rule"] = "independent_six_straight_paths_per_nonzero_leg"
    return DatasetSpec(dict(
        schema=1, kind="elemental_baryon_vvv", configuration=_CONFIG,
        axes=[_axis("time", range(2)), _axis("component", coordinates),
              _axis("momentum", [[1, 0, 0]]), _axis("mode0", labels[0]),
              _axis("mode1", labels[1]), _axis("mode2", labels[2])],
        dtype="complex128",
        bases={f"mode{leg}": dict(data_id=basis_id, indices=list(labels[leg]))
               for leg in range(3)},
        definition=definition,
        conventions=dict(normalization="unscaled", phase="anchor",
                         path_boundary="periodic", color_contraction="epsilon",
                         vector_conjugation="none"),
        parents=[gauge_id, basis_id],
    ))


class _BaryonDatasetBindingFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.gauge, self.gauge_id, self.basis, self.basis_id = _base_specs(
            self.root, self.index,
        )
    def _publish_baryon(self, product="raw_paths"):
        spec = _baryon_spec(self.gauge, self.gauge_id, self.basis,
                            self.basis_id, product=product)
        values = (np.arange(np.prod(spec.shape)).reshape(spec.shape) +
                  0.25j * np.arange(np.prod(spec.shape)).reshape(spec.shape))
        values = values.astype(np.complex128)
        data_id = _publish(self.root, self.index, f"baryon-{product}", spec, values)
        return spec, values, data_id


class BaryonDatasetBindingTests(_BaryonDatasetBindingFixture):
    def test_planning_accepts_full_permuted_input_times_but_not_incomplete_views(self):
        spec = _baryon_spec(self.gauge, self.gauge_id, self.basis, self.basis_id)
        self.assertEqual(tuple(DatasetSpec(spec.document["definition"]["input_views"]["gauge"])
                               .coordinates("time")), (1, 0))
        validate_diagram_input_specs({"baryon": spec})

        missing_gauge_time = spec.document
        missing_gauge_time["definition"]["input_views"]["gauge"]["axes"][0]["coordinates"] = [1]
        with self.assertRaisesRegex(ValueError, "time coverage"):
            validate_diagram_input_specs({"baryon": DatasetSpec(missing_gauge_time)})

        missing_basis_time = spec.document
        missing_basis_time["definition"]["input_views"]["basis_leg1"]["axes"][0]["coordinates"] = [0]
        with self.assertRaisesRegex(ValueError, "time coverage"):
            validate_diagram_input_specs({"baryon": DatasetSpec(missing_basis_time)})

        noncanonical_output = spec.document
        noncanonical_output["axes"][0]["coordinates"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "canonical physical order"):
            validate_diagram_input_specs({"baryon": DatasetSpec(noncanonical_output)})


    def test_planning_accepts_explicit_baryon_slots_and_checks_physical_basis_coverage(self):
        from lqcdweave.contracts.time_selection import TimeSelectionRecord, bind_time_selections
        document=_baryon_spec(self.gauge,self.gauge_id,self.basis,self.basis_id).document
        selection=TimeSelectionRecord(2,(1,))
        document['axes'][0]['coordinates']=[1]
        spec=DatasetSpec(bind_time_selections(document,{'time':selection}))
        self.assertEqual(validate_diagram_input_specs({'baryon':spec})[1],(0,))
        broken=spec.document
        broken['definition']['input_views']['basis_leg1']['axes'][0]['coordinates']=[0]
        with self.assertRaises(KeyError):
            validate_diagram_input_specs({'baryon':DatasetSpec(broken)})


if __name__ == "__main__":
    unittest.main()
