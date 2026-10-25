"""Published point-sector propagators through the current Diagram consumer."""

from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.meson_dataset import DatasetPropagator
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.storage.dataset import DatasetIndex, DatasetReader, DatasetWriter
from lqcdweave.planning.dataset import plan_requirement
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calc_diagram_eval
import test_indexed_point_production as point_production


_TAILS = {
    "vsv": ("sink_spin", "source_spin", "sink_mode", "source_mode"),
    "psv": ("sink_spin", "source_spin", "sink_point", "sink_color", "source_mode"),
    "vsp": ("sink_spin", "source_spin", "sink_mode", "source_point", "source_color"),
    "psp": ("sink_spin", "source_spin", "sink_point", "sink_color",
            "source_point", "source_color"),
}


def _axis(name, coordinates):
    return {"name": name, "coordinates": list(coordinates)}


def _point_spec(kind, *, lt=3, relative=False, order=None,
                sink_labels=(8, 3), source_labels=(11, 5), time_convention=None):
    coordinates = {
        "source_time": (2, 0, 1) if lt == 3 else tuple(reversed(range(lt))),
        ("relative_sink_time" if relative else "sink_time"):
            (1, 2, 0) if lt == 3 else tuple(reversed(range(lt))),
        "sink_spin": (0, 1, 2, 3),
        "source_spin": (0, 1, 2, 3),
        "sink_mode": (7, 2),
        "source_mode": (9, 4),
        "sink_point": tuple(sink_labels),
        "source_point": tuple(source_labels),
        "sink_color": (0, 1, 2),
        "source_color": (0, 1, 2),
    }
    tail = _TAILS[kind]
    basis = {}
    if "sink_mode" in tail:
        basis["sink_mode"] = {"data_id": "actual-mode-basis", "indices": list(coordinates["sink_mode"])}
    if "source_mode" in tail:
        basis["source_mode"] = {"data_id": "actual-mode-basis", "indices": list(coordinates["source_mode"])}
    if "sink_point" in tail:
        basis["sink_point"] = {"data_id": f"actual-{kind}-sink-points",
                               "indices": list(coordinates["sink_point"])}
    if "source_point" in tail:
        basis["source_point"] = {"data_id": f"actual-{kind}-source-points",
                                 "indices": list(coordinates["source_point"])}
    axis_order = tuple(order or ("source_time", "sink_time" if not relative else "relative_sink_time", *tail))
    if not relative and "sink_time" not in axis_order:
        coordinates["sink_time"] = coordinates.pop("relative_sink_time", tuple(range(lt)))
    sink_axis = "relative_sink_time" if relative else "sink_time"
    return DatasetSpec({
        "schema": 1,
        "kind": kind,
        "configuration": {"ensemble": "point-binding-fixture", "configuration": "cfg-point-7",
                          "lattice": [2, 2, 2, lt]},
        "axes": [_axis(name, coordinates[name]) for name in axis_order],
        "dtype": "complex128",
        "bases": basis,
        "definition": {"gauge_id": "actual-gauge-7", "solver_tolerance": 1e-8},
        "conventions": {"time_convention": time_convention or ("relative" if relative else "absolute"),
                        "spin_basis": "degrand_rossi", "temporal_boundary": "antiperiodic",
                        "normalization": "unscaled"},
        "parents": sorted({binding["data_id"] for binding in basis.values()} | {"actual-gauge-7"}),
    })


def _random(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _file_values(spec, logical):
    """Lay physical-time and canonical-tail data out in the declared file axes."""
    kind = spec.document["kind"]
    sink_axis = "relative_sink_time" if "relative_sink_time" in spec.axis_names else "sink_time"
    result = np.empty(spec.shape, dtype=logical.dtype)
    axis_coordinates = [spec.coordinates(name) for name in spec.axis_names]
    for file_index in product(*(range(len(values)) for values in axis_coordinates)):
        labels = {name: axis_coordinates[axis][index]
                  for axis, (name, index) in enumerate(zip(spec.axis_names, file_index, strict=True))}
        source = labels["source_time"]
        stored_sink = labels[sink_axis]
        relative = spec.document["conventions"].get("time_convention") == "relative"
        sink = ((stored_sink + source) % spec.document["configuration"]["lattice"][3]
                if relative else stored_sink)
        tail_index = tuple(spec.coordinates(name).index(labels[name]) for name in _TAILS[kind])
        result[file_index] = logical[(source, sink, *tail_index)]
    return result


def _publish(root, name, spec, values):
    directory = Path(root) / name
    with DatasetWriter(directory, spec, provenance={"test": "point-propagator-adapter"}) as writer:
        for index in range(spec.shape[0]):
            writer.write(index, values[index])
        data_id = writer.publish()
    reader = DatasetReader(directory)
    if reader.data_id != data_id:
        reader.close()
        raise AssertionError("published dataset identity changed")
    return reader


def _logical_shape(spec):
    kind = spec.document["kind"]
    return (spec.document["configuration"]["lattice"][3],
            spec.document["configuration"]["lattice"][3],
            *(len(spec.coordinates(name)) for name in _TAILS[kind]))


def _direct_block(spec, values, sink, source, *, usedNe=None, mode_indices=None,
                  usedNp=None, point_indices=None):
    """Independent axis/coordinate oracle over the published file-shaped array."""
    sink_axis = "relative_sink_time" if "relative_sink_time" in spec.axis_names else "sink_time"
    stored_sink = ((sink - source) % spec.document["configuration"]["lattice"][3]
                   if spec.document["conventions"].get("time_convention") == "relative" else sink)
    coordinates = {"source_time": source, sink_axis: stored_sink}
    selection = tuple(
        spec.coordinates(name).index(coordinates[name]) if name in coordinates else slice(None)
        for name in spec.axis_names
    )
    result = values[selection]
    raw_tail = tuple(name for name in spec.axis_names if name not in ("source_time", sink_axis))
    if raw_tail != _TAILS[spec.document["kind"]]:
        result = np.transpose(result, tuple(raw_tail.index(name) for name in _TAILS[spec.document["kind"]]))
    axes = _TAILS[spec.document["kind"]]
    selectors = {}
    mode_axes = tuple(name for name in axes if name.endswith("_mode"))
    point_axes = tuple(name for name in axes if name.endswith("_point"))
    if usedNe is not None:
        counts = usedNe if isinstance(usedNe, (tuple, list)) else (usedNe,) * len(mode_axes)
        selectors.update({name: tuple(range(count)) for name, count in zip(mode_axes, counts, strict=True)})
    if usedNp is not None:
        counts = usedNp if isinstance(usedNp, (tuple, list)) else (usedNp,) * len(point_axes)
        selectors.update({name: tuple(range(count)) for name, count in zip(point_axes, counts, strict=True)})
    selectors.update(mode_indices or {})
    selectors.update(point_indices or {})
    for name in axes:
        if name in selectors:
            result = np.take(result, selectors[name], axis=axes.index(name))
    return result


class _SectorReader:
    """Test-only composition of the existing per-sector reader contracts."""

    def __init__(self, readers, cfg, calls):
        self.readers, self.cfg, self.calls = readers, cfg, calls
        self.Lt = next(iter(readers.values())).Lt
        self.time_convention = "absolute"

    def describe_block(self, sink_time, source_time, sector, **selection):
        return self.readers[sector].describe_block(
            sink_time, source_time, sector=sector, **selection)

    def block(self, sink_time, source_time, sector, **selection):
        self.calls.append((sector, sink_time, source_time))
        return self.readers[sector].block(
            sink_time, source_time, sector=sector, **selection)


class _PointPropagatorDatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.readers = []
    def publish_sector(self, kind, *, lt=3, relative=False, order=None, seed=11,
                       sink_labels=(8, 3), source_labels=(11, 5)):
        spec = _point_spec(kind, lt=lt, relative=relative, order=order,
                           sink_labels=sink_labels, source_labels=source_labels)
        logical = _random(_logical_shape(spec), seed)
        values = _file_values(spec, logical)
        reader = _publish(self.temp.name, f"{kind}-{len(self.readers)}", spec, values)
        self.readers.append(reader)
        self.addCleanup(reader.close)
        return spec, logical, values, DatasetPropagator(reader), reader


class PointPropagatorDatasetTests(_PointPropagatorDatasetFixture):
    def test_index_request_does_not_reuse_psv_or_psp_with_wrong_point_identity(self):
        index = DatasetIndex(Path(self.temp.name) / "index")
        for kind in ("psv", "psp"):
            spec = _point_spec(kind, lt=2)
            values = _file_values(spec, _random(_logical_shape(spec), 103 + len(self.readers)))
            reader = _publish(self.temp.name, f"indexed-{kind}", spec, values)
            self.readers.append(reader)
            self.addCleanup(reader.close)
            self.assertEqual(index.register(reader.directory), reader.data_id)
            observations = index.observations()
            self.assertEqual(plan_requirement(spec, observations)["action"], "reuse")
            changed = spec.document
            point_axis = "sink_point"
            changed["bases"][point_axis]["data_id"] = "different-actual-point-realization"
            requested = DatasetSpec(changed)
            decision = plan_requirement(requested, observations)
            self.assertNotEqual(decision["action"], "reuse")
            self.assertTrue(any("bases.sink_point.data_id" in reason
                                for reason in decision["reasons"]))

    def test_unknown_configuration_is_readable_but_not_strictly_reusable(self):
        doc = _point_spec("psv", lt=2).document
        doc["configuration"]["ensemble"] = None
        doc["configuration"]["configuration"] = None
        spec = DatasetSpec(doc)
        adapter = DatasetPropagator(type("Header", (), {
            "spec": spec, "data_id": "unknown-configuration"
        })())
        self.assertIsNone(adapter.cfg)
        self.assertIsNone(adapter.ensemble)
        self.assertEqual(plan_requirement(spec, ())["action"], "blocked")

    def test_published_psv_vsp_are_consumed_by_public_diagram_contract(self):
        rng = np.random.default_rng(92731)
        ne, npoint, lt = 2, 2, 2
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        current = LocalVectorCurrent().operator(3)
        component = current.terms[0].component
        normal = Operator.from_terms("normal", [OperatorTerm(0, component, 0, 1)])
        tails = {"v2v": (ne, ne), "v2p": (ne, npoint, 3), "p2v": (npoint, 3, ne),
                 "p2p": (npoint, 3, npoint, 3)}
        spatial = {sector: random((1, 1, lt, *shape)) for sector, shape in tails.items()}
        pools = {(component, 0, t, "bar"): "bar-sampling-design"
                 for t in range(lt)}
        pools.update({(component, 0, t, "field"): "field-sampling-design"
                      for t in range(lt)})
        elemental = ArrayElemental(
            spatial, (component,), phase_convention="anchor", cfg="cfg-point-7",
            data_identity="explicit-array-elemental",
            sampling_pool_ids=pools,
            sampling_groups={
                "bar-sampling-design": SamplingPoolSpec("bar-sampling-design", npoint, npoint),
                "field-sampling-design": SamplingPoolSpec("field-sampling-design", npoint, npoint),
            },
        )
        propagators = {
            "vsv": random((lt, lt, 4, 4, ne, ne)),
            "vsp": random((lt, lt, 4, 4, ne, npoint, 3)),
            "psv": random((lt, lt, 4, 4, npoint, 3, ne)),
            "psp": random((lt, lt, 4, 4, npoint, 3, npoint, 3)),
        }
        ordinary = ArrayPropagator({"vsv": propagators["vsv"], "psp": propagators["psp"]},
                                  time_convention="absolute", cfg="cfg-point-7")
        self.addCleanup(ordinary.close)
        persistent = {}
        for kind, seed, point_labels in (
                ("psv", 201, ((17, 4), (9, 2))),
                ("vsp", 202, ((31, 6), (23, 5)))):
            if kind == "psv":
                logical = propagators["psv"]
                sink_labels, source_labels = point_labels
            else:
                logical = propagators["vsp"]
                sink_labels, source_labels = point_labels
            # Distinct IDs and coordinate labels are explicit for both point ends.
            spec = _point_spec(kind, lt=lt, order=(
                "source_time", "sink_time", *_TAILS[kind]),
                sink_labels=sink_labels, source_labels=source_labels)
            values = _file_values(spec, logical)
            reader = _publish(self.temp.name, f"public-{kind}", spec, values)
            self.readers.append(reader)
            self.addCleanup(reader.close)
            persistent[kind] = DatasetPropagator(reader)
        self.assertNotEqual(persistent["psv"].point_bindings["sink_point"][0],
                            persistent["vsp"].point_bindings["source_point"][0])
        self.assertEqual(persistent["psv"].point_bindings["sink_point"][1], (17, 4))
        self.assertEqual(persistent["vsp"].point_bindings["source_point"][1], (23, 5))
        calls = []
        reader = _SectorReader({"vsv": ordinary, "psp": ordinary, **persistent},
                              "cfg-point-7", calls)
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                        ["current", "normal"], [None, q])
        actual = calc_diagram_eval(
            graph,
            propagator_map={q: reader},
            vertex_map={"current": Meson(elemental, current, phase_convention="anchor"),
                        "normal": Meson(elemental, normal, phase_convention="anchor")},
            configuration="cfg-point-7", usedNe=ne, usedNp=npoint,
            groups_independent=True, optimize="none",
        ).value

        def matrix(tensor, rows, columns):
            return tensor.reshape(4, 4, rows, columns).transpose(0, 2, 1, 3).reshape(
                4 * rows, 4 * columns)

        ncolor = npoint * 3
        current_spatial = np.block([
            [spatial["v2v"][0, 0, 0], spatial["v2p"][0, 0, 0].reshape(ne, ncolor)],
            [spatial["p2v"][0, 0, 0].reshape(ncolor, ne),
             spatial["p2p"][0, 0, 0].reshape(ncolor, ncolor)],
        ])
        outgoing = np.concatenate((propagators["vsv"][0, 1],
                                   propagators["vsp"][0, 1].reshape(4, 4, ne, ncolor)), axis=3)
        incoming = np.concatenate((propagators["vsv"][1, 0],
                                   propagators["psv"][1, 0].reshape(4, 4, ncolor, ne)), axis=2)
        expected = np.trace(
            matrix(outgoing, ne, ne + ncolor)
            @ np.kron(gamma(current.terms[0].gamma), current_spatial)
            @ matrix(incoming, ne + ncolor, ne)
            @ np.kron(gamma(0), spatial["v2v"][0, 0, 1])
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
        self.assertIn("psv", {sector for sector, _sink, _source in calls})
        self.assertIn("vsp", {sector for sector, _sink, _source in calls})


class _CompositeDatasetPropagatorFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = point_production.IndexedPointProductionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
    def _open_products(self, result):
        readers = {
            name: self.fixture.index.open(identity)
            for name, identity in result["datasets"].items()
        }
        for reader in readers.values():
            self.addCleanup(reader.close)
        return readers


if __name__ == "__main__":
    unittest.main()
