"""Metadata-only validation for lane sparse bridge topology."""

from dataclasses import replace
import unittest

import test_sparse_point_sink_batch as sparse_batch_fixture

from lqcdweave.contraction.expansion.batching import BatchDiagramBinding
from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    ContractionTerm,
    DiagramBindings,
    SourcePreparation,
    StorageBlock,
    describe_coordinates,
    parse_axis_pattern,
)


class _MetadataProvider:
    """Reuse the established lane fixture, changing only returned descriptors."""

    def __init__(self, mutate):
        self.delegate = sparse_batch_fixture._FourPartBudgetProvider()
        self.mutate = mutate

    def describe(self, correlator, *, bindings, coordinates, batch):
        time = coordinates.points[0].values[0][1]
        item = self.delegate.describe(
            correlator, bindings=bindings, coordinates=coordinates, batch=batch,
        )
        return self.mutate(item, time)

    def assert_no_io(self, testcase):
        testcase.assertEqual(self.delegate.slab_reads, {"left": 0, "right": 0})
        for loader in self.delegate.loaders.values():
            testcase.assertEqual(loader.reads, [])
            testcase.assertEqual(loader.materializations, [])
            testcase.assertEqual(loader.closes, [])


def _time_scan():
    return describe_coordinates(
        time_axes={"scan": (0, 1)},
        batch={"axis": "scan", "size": 2},
    )


def _with_endpoint_axes(item, *, field_axis=0, bar_axis=1):
    bridge, = item.problem.sparse_bridges
    sources = dict(item.problem.sources)
    source = sources[bridge.record]
    metadata = dict(source.metadata)
    metadata["logical_endpoint_axis_bindings"] = {
        "field": {"raw_axis": field_axis},
        "bar": {"raw_axis": bar_axis},
    }
    sources[bridge.record] = replace(source, metadata=metadata)
    return replace(item, problem=replace(item.problem, sources=sources))


def _with_second_record(item, *, time):
    bridge, = item.problem.sparse_bridges
    record_key = f"record-{time}-secondary"
    block_key = f"record-block-{time}-secondary"
    original_source = item.problem.sources[bridge.record]
    secondary_source = replace(
        original_source,
        region=f"{original_source.region}-secondary",
    )
    secondary_block = StorageBlock(
        f"record-dataset-{time}-secondary", "full",
        original_source.region_bytes,
        host_bytes=original_source.host_region_bytes,
        read_kind="p2p",
    )
    sources = dict(item.problem.sources)
    sources[record_key] = secondary_source
    preparations = dict(item.preparations)
    preparations[record_key] = SourcePreparation(
        (block_key,), dtype="complex128",
    )
    blocks = dict(item.blocks)
    blocks[block_key] = secondary_block
    problem = replace(
        item.problem,
        sources=sources,
        sparse_bridges=(*item.problem.sparse_bridges,
                        replace(bridge, record=record_key)),
    )
    return replace(item, problem=problem, preparations=preparations, blocks=blocks)


class SparseBatchPreflightTests(unittest.TestCase):
    def _describe(self, provider):
        coordinates, batch = _time_scan()
        result = BatchDiagramBinding(provider).describe(
            object(), bindings=DiagramBindings(), coordinates=coordinates,
            batch=batch,
        )
        return result

    def test_distinct_record_keys_keep_lane_order_and_bridge_ordinals(self):
        def mutate(item, time):
            item = _with_endpoint_axes(item)
            return _with_second_record(item, time=time)

        provider = _MetadataProvider(mutate)
        inputs = self._describe(provider)
        problem = inputs.task_batches[0].problem
        self.assertEqual(len(problem.sparse_bridges), 2)
        expected_lane_keys = (
            ("record-0", "record-1"),
            ("record-0-secondary", "record-1-secondary"),
        )
        for ordinal, (bridge, lane_keys) in enumerate(
                zip(problem.sparse_bridges, expected_lane_keys, strict=True)):
            source = problem.sources[bridge.record]
            self.assertEqual(source.metadata["bridge_ordinal"], ordinal)
            self.assertEqual(
                tuple(item["source_key"] for item in source.metadata["lane_sources"]),
                lane_keys,
            )
        provider.assert_no_io(self)

    def test_lane_bridge_count_mismatch_rejects_before_source_io(self):
        def mutate(item, time):
            item = _with_endpoint_axes(item)
            if time == 0:
                item = _with_second_record(item, time=time)
            return item

        provider = _MetadataProvider(mutate)
        with self.assertRaisesRegex(ValueError, "same ordered sparse bridge groups"):
            self._describe(provider)
        provider.assert_no_io(self)

    def test_field_bar_metadata_shapes_and_raw_axes_are_preflighted(self):
        malformed = (
            ("top-level nonmapping", ["field", "bar"],
             "logical endpoint axis bindings must be a mapping"),
            ("field nonmapping", {"field": [], "bar": {"raw_axis": 1}},
             "incomplete field/bar raw-axis topology"),
            ("bar nonmapping", {"field": {"raw_axis": 0}, "bar": "raw"},
             "incomplete field/bar raw-axis topology"),
            ("missing raw axis", {"field": {}, "bar": {"raw_axis": 1}},
             "incomplete field/bar raw-axis topology"),
        )
        for name, endpoint_metadata, message in malformed:
            with self.subTest(case=name):
                def mutate(item, time):
                    bridge, = item.problem.sparse_bridges
                    sources = dict(item.problem.sources)
                    source = sources[bridge.record]
                    metadata = dict(source.metadata)
                    metadata["logical_endpoint_axis_bindings"] = endpoint_metadata
                    sources[bridge.record] = replace(source, metadata=metadata)
                    return replace(item, problem=replace(item.problem, sources=sources))

                provider = _MetadataProvider(mutate)
                with self.assertRaisesRegex(ValueError, message):
                    self._describe(provider)
                provider.assert_no_io(self)

    def test_source_dagger_raw_axis_topology_drift_rejects_before_source_io(self):
        def mutate(item, time):
            return _with_endpoint_axes(
                item, field_axis=(0 if time == 0 else 1),
                bar_axis=(1 if time == 0 else 0),
            )

        provider = _MetadataProvider(mutate)
        with self.assertRaisesRegex(
                ValueError, "axis topology or source-dagger orientation"):
            self._describe(provider)
        provider.assert_no_io(self)

    def test_repeated_tensor_operand_makes_bridge_pair_ambiguous(self):
        def mutate(item, time):
            bridge, = item.problem.sparse_bridges
            sources = dict(item.problem.sources)
            source = sources[bridge.record]
            metadata = dict(source.metadata)
            metadata["logical_endpoint_axis_bindings"] = {
                "field": {"raw_axis": 0}, "bar": {"raw_axis": 1},
            }
            sources[bridge.record] = replace(source, metadata=metadata)
            original = item.problem.terms[0]
            repeated = ContractionTerm(
                (original.inputs[0], original.inputs[1], original.inputs[0]),
                "spc,qrd,spc->", coefficient=original.coefficient,
            )
            problem = replace(
                item.problem, sources=sources, terms=(repeated,),
            )
            return replace(item, problem=problem)

        provider = _MetadataProvider(mutate)
        with self.assertRaisesRegex(
                NotImplementedError, "one unambiguous ordered operand pair"):
            self._describe(provider)
        provider.assert_no_io(self)

    def test_reversed_bridge_endpoint_order_is_valid_metadata(self):
        def mutate(item, time):
            item = _with_endpoint_axes(item)
            bridge, = item.problem.sparse_bridges
            reversed_bridge = replace(
                bridge,
                left=bridge.right,
                right=bridge.left,
                left_axes=bridge.right_axes,
                right_axes=bridge.left_axes,
            )
            return replace(
                item,
                problem=replace(item.problem, sparse_bridges=(reversed_bridge,)),
            )

        provider = _MetadataProvider(mutate)
        inputs = self._describe(provider)
        bridge, = inputs.task_batches[0].problem.sparse_bridges
        self.assertEqual((bridge.left, bridge.right), ("tensor:1", "tensor:0"))
        provider.assert_no_io(self)

    def test_reversed_term_operand_order_is_valid_metadata(self):
        def mutate(item, time):
            item = _with_endpoint_axes(item)
            term, = item.problem.terms
            reversed_term = ContractionTerm(
                tuple(reversed(term.inputs)),
                AxisPattern(tuple(reversed(term.subscripts.inputs)),
                            term.subscripts.output),
                coefficient=term.coefficient,
            )
            return replace(item, problem=replace(item.problem, terms=(reversed_term,)))

        provider = _MetadataProvider(mutate)
        inputs = self._describe(provider)
        term, = inputs.task_batches[0].problem.terms
        bridge, = inputs.task_batches[0].problem.sparse_bridges
        self.assertEqual(term.inputs, ("tensor:0", "tensor:1"))
        self.assertEqual(term.subscripts.normalized(),
                         parse_axis_pattern("aqrd,aspc->a").normalized())
        self.assertEqual((bridge.left, bridge.right), ("tensor:1", "tensor:0"))
        provider.assert_no_io(self)
