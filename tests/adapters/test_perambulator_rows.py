"""Explicit source-time row adaptation for canonical perambulator pair reads."""

import unittest

import numpy as np

from lqcdweave.adapters.perambulator_rows import SourceTimePropagatorReader
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.production.perambulator import (
    PropagatorEndpoint,
    PropagatorReader,
    propagator_pair_spec,
)


class SourceTimePropagatorReaderTests(unittest.TestCase):
    def setUp(self):
        self.configuration = dict(
            ensemble="test-ensemble",
            configuration="cfg-1",
            lattice=[2, 1, 1, 3],
        )
        self.gamma5 = np.array(
            [[0, -1j, 0, 0], [1j, 0, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
        self.endpoints = {
            "V2": PropagatorEndpoint("V", "basis-v2", [0, 1]),
            "V3": PropagatorEndpoint("V", "basis-v3", [1, 4, 7]),
            "D2": PropagatorEndpoint(
                "D", "noise-d2", [0, 1], metadata=dict(
                    vector_labels=[[0, 0], [1, 0]],
                    mode_groups=[[0, 1], [2, 3]],
                    parent_basis_id="basis-v2",
                    realization_id="realization-1",
                    noise_pool_id="pool-1",
                    normalization="raw",
                ),
            ),
        }

    def make_spec(self, product, pairs, sink, source, *, dtype="complex128"):
        return propagator_pair_spec(
            product,
            configuration=self.configuration,
            sink=sink,
            source=source,
            time_pairs=pairs,
            gauge_id="gauge-cfg-1",
            dirac={"action": "test-action", "mass": 0.1},
            gamma5=self.gamma5,
            spin_basis="test-spin-basis",
            gamma5_hermitian=True,
            dtype=dtype,
        )

    def test_relative_wrap_uses_explicit_unsorted_row_slot_without_loading_at_init(self):
        spec = self.make_spec(
            "VSV", ((0, 2),), self.endpoints["V2"], self.endpoints["V2"]
        )
        row = (np.arange(3 * 4 * 4 * 2 * 2).reshape(3, 4, 4, 2, 2)
               + 1j * np.arange(3 * 4 * 4 * 2 * 2).reshape(3, 4, 4, 2, 2))
        calls = []

        def load_row(source_time):
            calls.append(source_time)
            return row

        reader = SourceTimePropagatorReader(
            spec,
            load_row,
            product="VSV",
            source_times=(0, 2),
            row_times=(2, 1, 0),
            time_convention="relative",
            data_id="legacy-row-data",
        )
        self.assertEqual(calls, [])
        access = PropagatorReader([reader])
        self.assertEqual(access.coverage('VSV', sink=self.endpoints['V2'],
            source=self.endpoints['V2']), ((0, 2), (2, 0)))
        self.assertEqual(access.missing('VSV', ((0, 2), (1, 2)),
            sink=self.endpoints['V2'], source=self.endpoints['V2']), ((1, 2),))
        self.assertEqual(calls, [])

        # (sink-source) mod Lt = (0-2) mod 3 = 1, at unsorted slot one.
        value = reader.read(time_pair=[0, 2])
        np.testing.assert_array_equal(value, row[1])
        self.assertTrue(np.shares_memory(value, row))
        self.assertEqual(calls, [2])

    def test_absolute_reads_borrow_with_unequal_endpoint_extents(self):
        sink, source = self.endpoints["V2"], self.endpoints["V3"]
        spec = self.make_spec("VSV", ((1, 2),), sink, source)
        row = np.arange(3 * 4 * 4 * 2 * 3, dtype=np.float64).reshape(3, 4, 4, 2, 3)
        row = row.astype(np.complex128) * (1 + 0.25j)
        reader = SourceTimePropagatorReader(
            spec,
            lambda source_time: row,
            product="VSV",
            source_times=(2,),
            row_times=(2, 1, 0),
            time_convention="absolute",
            data_id="absolute-row-data",
        )

        value = reader.read(time_pair=(1, 2))
        self.assertEqual(value.shape, (4, 4, 2, 3))
        self.assertEqual(value.dtype, np.dtype("complex128"))
        np.testing.assert_array_equal(value, row[1])
        self.assertTrue(np.shares_memory(value, row))

    def test_noncanonical_product_reverses_times_endpoints_and_gamma5_values(self):
        raw_sink, raw_source = self.endpoints["D2"], self.endpoints["V3"]
        # DSV raw pair (0,2) is stored canonically as VSD pair (2,0).
        spec = self.make_spec(
            "VSD", ((2, 0),), raw_source, raw_sink
        )
        rng = np.random.default_rng(20261011)
        raw_row = (rng.normal(size=(3, 4, 4, 2, 3))
                   + 1j * rng.normal(size=(3, 4, 4, 2, 3)))
        calls = []

        def load_row(source_time):
            calls.append(source_time)
            return raw_row

        reader = SourceTimePropagatorReader(
            spec,
            load_row,
            product="DSV",
            source_times=(2,),
            row_times=(0, 1, 2),
            time_convention="absolute",
            data_id="raw-dsv-data",
        )

        # Independent matrix reference: dagger swaps spin and endpoint axes,
        # then gamma5 multiplies the two spin axes.
        transposed = raw_row[0].conj().transpose(1, 0, 3, 2)
        expected = np.einsum("aA,ABij,Bb->abij", self.gamma5, transposed, self.gamma5)
        value = reader.read(time_pair=(2, 0))
        self.assertEqual(value.shape, (4, 4, 3, 2))
        self.assertFalse(np.shares_memory(value, raw_row))
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(calls, [2])

        # The adapter satisfies the public pair-reader protocol, including the
        # canonical physical time and endpoint records.
        block = PropagatorReader([reader]).read(
            "VSD", 2, 0, sink=raw_source, source=raw_sink
        )
        self.assertEqual((block.product, block.sink_time, block.source_time), ("VSD", 2, 0))
        self.assertEqual((block.sink, block.source), (raw_source, raw_sink))
        self.assertEqual(block.parent_id, "raw-dsv-data")
        np.testing.assert_allclose(block.values, expected, rtol=1e-13, atol=1e-13)

    def test_constructor_rejects_uncovered_sources_rows_and_product_context(self):
        spec = self.make_spec(
            "VSV", ((0, 2),), self.endpoints["V2"], self.endpoints["V2"]
        )
        common = dict(
            spec=spec,
            load_row=lambda source_time: self.fail("metadata validation loaded a row"),
            product="VSV",
            time_convention="relative",
            data_id="coverage-data",
        )
        with self.assertRaisesRegex(ValueError, "source_times"):
            SourceTimePropagatorReader(
                **common, source_times=(0,), row_times=(1,)
            )
        with self.assertRaisesRegex(ValueError, "row_times"):
            SourceTimePropagatorReader(
                **common, source_times=(2,), row_times=(0,)
            )
        with self.assertRaisesRegex(ValueError, "product"):
            SourceTimePropagatorReader(
                **{**common, "product": "VSP"},
                source_times=(2,), row_times=(1,),
            )

    def test_read_rejects_out_of_coverage_bad_shape_and_bad_dtype(self):
        spec = self.make_spec(
            "VSV", ((0, 2),), self.endpoints["V2"], self.endpoints["V2"]
        )
        good_shape = (3, 4, 4, 2, 2)
        for row, error in (
            (np.zeros((2, 4, 4, 2, 2), dtype=np.complex128), "shape"),
            (np.zeros(good_shape, dtype=np.complex64), "dtype"),
        ):
            reader = SourceTimePropagatorReader(
                spec,
                lambda source_time, row=row: row,
                product="VSV",
                source_times=(2,),
                row_times=(0, 1, 2),
                time_convention="absolute",
                data_id="bad-row-data",
            )
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    reader.read(time_pair=(0, 2))

        valid = SourceTimePropagatorReader(
            spec,
            lambda source_time: np.zeros(good_shape, dtype=np.complex128),
            product="VSV",
            source_times=(2,),
            row_times=(0, 1, 2),
            time_convention="absolute",
            data_id="valid-row-data",
        )
        with self.assertRaisesRegex(KeyError, "coverage"):
            valid.read(time_pair=(1, 2))

    def test_rejects_noncanonical_pair_context(self):
        good = self.make_spec(
            "VSV", ((0, 2),), self.endpoints["V2"], self.endpoints["V2"]
        )
        document = good.document
        document["definition"]["encoding"] = "legacy-source-time-grid"
        malformed = DatasetSpec(document)
        with self.assertRaisesRegex(ValueError, "canonical pr-six-pairs-v1"):
            SourceTimePropagatorReader(
                malformed,
                lambda source_time: self.fail("invalid spec loaded a row"),
                product="VSV",
                source_times=(2,),
                row_times=(0,),
                time_convention="absolute",
                data_id="bad-context-data",
            )

    def test_all_noncanonical_products_match_independent_full_spin_matrix(self):
        endpoints = {'V': self.endpoints['V3'], 'D': self.endpoints['D2'],
                     'P': PropagatorEndpoint('P', 'actual-points', [0, 1])}
        rng = np.random.default_rng(87)
        for product in ('VSP', 'DSV', 'DSP'):
            sink, source = endpoints[product[0]], endpoints[product[2]]
            tail_a = (len(sink.indices), 3) if sink.kind == 'P' else (len(sink.indices),)
            tail_b = (len(source.indices), 3) if source.kind == 'P' else (len(source.indices),)
            shape = (1, 4, 4) + tail_a + tail_b
            row = rng.normal(size=shape) + 1j*rng.normal(size=shape)
            wa, wb = int(np.prod(tail_a)), int(np.prod(tail_b))
            matrix = row[0].reshape(4, 4, wa, wb).transpose(0, 2, 1, 3).reshape(4*wa, 4*wb)
            expected = (np.kron(self.gamma5, np.eye(wb)) @ matrix.conj().T @
                        np.kron(self.gamma5, np.eye(wa)))
            expected = expected.reshape(4, wb, 4, wa).transpose(0, 2, 1, 3).reshape(
                (4, 4) + tail_b + tail_a)
            spec = self.make_spec(product[2]+'S'+product[0], ((2, 0),), source, sink)
            adapter = SourceTimePropagatorReader(spec, lambda t, row=row: row,
                product=product, source_times=(2,), row_times=(0,),
                time_convention='absolute', data_id='actual-'+product)
            with self.subTest(product=product):
                np.testing.assert_allclose(adapter.read(time_pair=(2, 0)), expected,
                                           atol=2e-14, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
