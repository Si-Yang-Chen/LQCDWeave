"""Actual endpoint-identity checks for the dense-V Baryon E binding."""

from dataclasses import replace
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.correlator.api import Diagram

from test_baryon_diagram_frontend import _case, _mixed_case


class _AxisBoundReader:
    """Attach independently declared axis identities to an array reader."""

    def __init__(self, reader, binder):
        self._reader = reader
        self._binder = binder
        self.describe_calls = 0
        self.block_calls = 0

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def describe_block(self, *args, **kwargs):
        metadata = self._reader.describe_block(*args, **kwargs)
        self.describe_calls += 1
        bindings, identity = self._binder(metadata, args, kwargs, self.describe_calls)
        changes = {"axis_bindings": bindings}
        if identity is not None:
            changes["data_identity"] = identity
        return replace(metadata, **changes)

    def block(self, *args, **kwargs):
        self.block_calls += 1
        return self._reader.block(*args, **kwargs)


def _labels(prefix, count):
    return tuple(f"{prefix}:{index}" for index in range(count))


def _baryon_reader(vertex, prefix, mode_counts, *, ids=None, dynamic=None):
    reader = vertex.reader
    ids = tuple(ids or (f"{prefix}-basis-{axis}" for axis in range(3)))
    full_labels = tuple(_labels(f"{prefix}-mode-{axis}", count)
                        for axis, count in enumerate(mode_counts))

    def bind(metadata, _args, _kwargs, call):
        bindings = {
            f"mode{axis}": (ids[axis], full_labels[axis][:metadata.shape[axis]])
            for axis in range(3)
        }
        identity = None if dynamic is None else dynamic(call)
        return bindings, identity

    bound = _AxisBoundReader(reader, bind)
    vertex.reader = bound
    vertex.elemental = bound
    return bound, ids, full_labels


def _propagator_reader(reader, source_binding, sink_binding, *, spin_call_ids=False,
                       dynamic_source=None):
    def bind(metadata, _args, _kwargs, call):
        source_id, source_labels = source_binding(call) if callable(source_binding) else source_binding
        sink_id, sink_labels = sink_binding(call) if callable(sink_binding) else sink_binding
        bindings = {
            "source_mode": (source_id, tuple(source_labels[:metadata.shape[3]])),
            "sink_mode": (sink_id, tuple(sink_labels[:metadata.shape[2]])),
        }
        if spin_call_ids:
            bindings["source_spin"] = (f"spin-view-{call}", (0, 1, 2, 3))
        identity = None
        if dynamic_source is not None:
            identity = dynamic_source(call)
        return bindings, identity

    return _AxisBoundReader(reader, bind)


def _bound_case(case, *, mismatch=None):
    source_shape = case["source_shape"]
    sink_shape = case["sink_shape"]
    source, source_ids, source_labels = _baryon_reader(
        case["vertices"]["source"], "source", source_shape,
    )
    sink, sink_ids, sink_labels = _baryon_reader(
        case["vertices"]["sink"], "sink", sink_shape,
    )
    del source, sink

    permutation = (2, 1, 0)
    for position, source_slot in enumerate(case["source_slots"]):
        symbol = f"p{position}"
        source_axis = permutation[source_slot]
        sink_axis = position
        source_binding = (source_ids[source_axis], source_labels[source_axis])
        sink_binding = (sink_ids[sink_axis], sink_labels[sink_axis])
        if mismatch is not None and mismatch[0] == symbol:
            side, mode = mismatch[1], mismatch[2]
            current = source_binding if side == "source" else sink_binding
            if mode == "wrong-id":
                current = (f"wrong-{current[0]}", current[1])
            elif mode == "reordered":
                current = (current[0], current[1][::-1])
            else:
                raise AssertionError(mode)
            if side == "source":
                source_binding = current
            else:
                sink_binding = current
        case["propagators"][symbol] = _propagator_reader(
            case["propagators"][symbol], source_binding, sink_binding,
        )
    return case


def _expand(case, *, correlator=None, vertices=None, propagators=None,
            configuration="cfg-a"):
    return expand_correlator(
        case["diagram"] if correlator is None else correlator,
        adapter=BaryonDiagramBinding(sector_scope="dense-v"),
        bindings=DiagramBindings(
            case["propagators"] if propagators is None else propagators,
            case["vertices"] if vertices is None else vertices,
            configuration,
        ),
        time_axes=None,
        time_pairs=None,
        batch="auto",
    )


def _propagator_descriptions(expanded):
    return tuple(
        spec.description for spec in expanded.tensor_specs.values()
        if spec.description is not None and spec.description.role == "propagator"
    )


def _shared_equal_extent_case(*, spin_call_ids=False):
    case = _case()
    for side in ("source", "sink"):
        vertex = case["vertices"][side]
        component = vertex.operator.terms[0].component
        base = ArrayBaryonElemental(
            {"vvv": np.zeros((1, 1, 2, 2, 2, 2), dtype=np.complex128)},
            (component,), cfg="cfg-a", data_identity=f"{side}-shared-elemental",
        )
        _baryon_reader(
            vertex, side, (2, 2, 2),
            ids=(f"{side}-common-basis",) * 3,
        )
        vertex.reader = _AxisBoundReader(
            base,
            lambda metadata, _args, _kwargs, _call, side=side: (
                {f"mode{axis}": (f"{side}-common-basis", (0, 1))
                 for axis in range(3)}, None,
            ),
        )
        vertex.elemental = vertex.reader

    base_prop = ArrayPropagator(
        {"vsv": np.zeros((2, 2, 4, 4, 2, 2), dtype=np.complex128)},
        time_convention="absolute", cfg="cfg-a", data_identity="shared-propagator",
    )
    shared_prop = _propagator_reader(
        base_prop,
        ("source-common-basis", (0, 1)),
        ("sink-common-basis", (0, 1)),
        spin_call_ids=spin_call_ids,
    )
    case["propagators"] = {f"p{index}": shared_prop for index in range(3)}
    return case, shared_prop


def _mixed_bound_case(dagger):
    case = _mixed_case((0, 1, 2))
    source, source_ids, source_labels = _baryon_reader(
        case["vertices"]["source"], "mixed-source", (2, 2, 2),
    )
    sink, sink_ids, sink_labels = _baryon_reader(
        case["vertices"]["sink"], "mixed-sink", (2, 2, 2),
    )
    del source, sink
    current = case["vertices"]["current"]
    current.dagger = dagger
    left_id, right_id = "current-left-basis", "current-right-basis"
    left_labels, right_labels = ("left-0", "left-1"), ("right-0", "right-1")
    current_reader = _AxisBoundReader(
        current.reader,
        lambda metadata, _args, _kwargs, _call: (
            {"mode_left": (left_id, left_labels[:metadata.shape[0]]),
             "mode_right": (right_id, right_labels[:metadata.shape[1]])}, None,
        ),
    )
    current.reader = current_reader
    current.elemental = current_reader
    bar_axis = "mode_left" if dagger else "mode_right"
    field_axis = "mode_right" if dagger else "mode_left"
    meson_axes = {"mode_left": (left_id, left_labels),
                  "mode_right": (right_id, right_labels)}
    endpoints = {
        "p-in": (source_ids[0], source_labels[0], *meson_axes[bar_axis]),
        "p-direct0": (source_ids[2], source_labels[2], sink_ids[0], sink_labels[0]),
        "p-direct1": (source_ids[1], source_labels[1], sink_ids[1], sink_labels[1]),
        "p-out": (*meson_axes[field_axis], sink_ids[2], sink_labels[2]),
    }
    for symbol, (source_id, source_coords, sink_id, sink_coords) in endpoints.items():
        case["propagators"][symbol] = _propagator_reader(
            case["propagators"][symbol],
            (source_id, source_coords), (sink_id, sink_coords),
        )
    return case


class BaryonDiagramEndpointIdentityTests(unittest.TestCase):
    def test_source_dagger_uses_raw_mode_axis_and_every_edge_is_described(self):
        case = _bound_case(_case((2, 0, 1)))
        expanded = _expand(case)
        descriptions = _propagator_descriptions(expanded)
        self.assertEqual(len(descriptions), 3)
        by_symbol = {description.semantics["symbol"]: description
                     for description in descriptions}
        for position, source_slot in enumerate(case["source_slots"]):
            description = by_symbol[f"p{position}"]
            checks = description.bindings["endpoint_axis_bindings"]
            source = checks["source"]
            sink = checks["sink"]
            source_axis = (2, 1, 0)[source_slot]
            self.assertEqual(source["vertex_kind"], "baryon")
            self.assertEqual(source["vertex_slot"], source_slot)
            self.assertEqual(source["vertex_axis"], f"mode{source_axis}")
            self.assertEqual(source["status"], "verified")
            self.assertEqual(source["knowledge"], "both-bound")
            self.assertEqual(source["propagator"], source["vertex"])
            self.assertEqual(sink["vertex_kind"], "baryon")
            self.assertEqual(sink["vertex_slot"], position)
            self.assertEqual(sink["vertex_axis"], f"mode{position}")
            self.assertEqual(sink["status"], "verified")
            self.assertEqual(sink["propagator"], sink["vertex"])

    def test_same_shape_actual_id_and_order_mismatches_are_rejected(self):
        for mutation in (
            ("p0", "source", "wrong-id"),
            ("p0", "sink", "reordered"),
        ):
            with self.subTest(mutation=mutation):
                case = _bound_case(_case((0, 1, 2)), mismatch=mutation)
                with self.assertRaisesRegex(ValueError, "actual dataset ID or ordered coordinates"):
                    _expand(case)
                self.assertTrue(all(reader.block_calls == 0
                                    for reader in case["propagators"].values()))

    def test_unbound_in_memory_axes_remain_unknown_and_usable(self):
        expanded = _expand(_case())
        descriptions = _propagator_descriptions(expanded)
        self.assertEqual(len(descriptions), 3)
        for description in descriptions:
            checks = description.bindings["endpoint_axis_bindings"]
            self.assertEqual(set(checks), {"source", "sink"})
            for check in checks.values():
                self.assertEqual(check["status"], "unknown")
                self.assertEqual(check["knowledge"], "both-unbound")
                self.assertIsNone(check["propagator"])
                self.assertIsNone(check["vertex"])

    def test_shared_physical_propagator_retains_each_edge_check(self):
        case, _reader = _shared_equal_extent_case()
        expanded = _expand(case)
        descriptions = _propagator_descriptions(expanded)
        self.assertEqual(len(descriptions), 3)
        self.assertEqual(len({description.source_bindings[0]
                              for description in descriptions}), 1)
        source_slots = {
            description.bindings["endpoint_axis_bindings"]["source"]["vertex_slot"]
            for description in descriptions
        }
        sink_slots = {
            description.bindings["endpoint_axis_bindings"]["sink"]["vertex_slot"]
            for description in descriptions
        }
        self.assertEqual(source_slots, {0, 1, 2})
        self.assertEqual(sink_slots, {0, 1, 2})

    def test_mixed_mesons_keep_dagger_dependent_raw_field_bar_axes(self):
        for dagger, expected in (
            (False, ("mode_right", "mode_left")),
            (True, ("mode_left", "mode_right")),
        ):
            with self.subTest(dagger=dagger):
                case = _mixed_bound_case(dagger)
                expanded = _expand(case, configuration="cfg-mixed")
                descriptions = _propagator_descriptions(expanded)
                by_symbol = {description.semantics["symbol"]: description
                             for description in descriptions}
                incoming = by_symbol["p-in"].bindings["endpoint_axis_bindings"]["sink"]
                outgoing = by_symbol["p-out"].bindings["endpoint_axis_bindings"]["source"]
                self.assertEqual(incoming["vertex_kind"], "meson")
                self.assertEqual(incoming["vertex_axis"], expected[0])
                self.assertEqual(outgoing["vertex_kind"], "meson")
                self.assertEqual(outgoing["vertex_axis"], expected[1])
                self.assertEqual(incoming["status"], "verified")
                self.assertEqual(outgoing["status"], "verified")

    def test_shared_physical_block_rejects_axis_binding_conflict(self):
        case, reader = _shared_equal_extent_case(spin_call_ids=True)
        # Conflicting views have stable but different metadata. Re-describing
        # one reader is no longer required after metadata caching.
        views = tuple(_AxisBoundReader(reader._reader,
            lambda metadata, args, kwargs, call, fixed=fixed:
                reader._binder(metadata, args, kwargs, fixed)) for fixed in (1, 2))
        case["propagators"] = {"p0": views[0], "p1": views[1], "p2": views[0]}
        with self.assertRaisesRegex(ValueError, "shared baryon block identity.*axis bindings"):
            _expand(case)
        self.assertTrue(all(view.block_calls == 0 for view in views))

    def test_shared_logical_tensor_rejects_changed_actual_mode_binding(self):
        case = _case((0, 1, 2))
        source_vertex = case["vertices"]["source"]
        source_base = source_vertex.reader
        source_shape = case["source_shape"]

        def changing_source(metadata, _args, _kwargs, call):
            identity = f"source-basis-{call}"
            bindings = {
                f"mode{axis}": (identity, _labels(f"source-{call}-{axis}", source_shape[axis])
                                 [:metadata.shape[axis]])
                for axis in range(3)
            }
            return bindings, f"source-storage-{call}"

        changing_reader = _AxisBoundReader(source_base, changing_source)
        source_vertex.reader = changing_reader
        source_vertex.elemental = changing_reader
        second = Diagram(
            case["diagram"].diagram,
            list(case["diagram"].time_list),
            ["source-alias", "sink-alias"],
            list(case["diagram"].propagator_list),
        )
        vertices = dict(case["vertices"])
        vertices.update({"source-alias": source_vertex,
                         "sink-alias": case["vertices"]["sink"]})
        with self.assertRaisesRegex(ValueError, "shared baryon logical tensor.*axis bindings"):
            _expand(case, correlator=[case["diagram"], second], vertices=vertices)
        self.assertEqual(changing_reader.block_calls, 0)


if __name__ == "__main__":
    unittest.main()
