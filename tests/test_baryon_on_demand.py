"""Integration coverage from on-demand baryon VVV fields into contractions."""

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.baryon_on_demand import OnDemandBaryonElemental
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.production.elemental.api import BaryonElementalGenerator
from lqcdweave.operators.correlator.api import split_baryon_eigenmodes

from test_baryon_mode_blocks import (
    _PARTITIONS,
    _calculate,
    _case,
    _independent_einsum,
    _selections_for_block_index,
)


_LATTICE = (3, 1, 1, 2)
_MOMENTUM = ((1, 0, 0),)


def _fields(seed):
    rng = np.random.default_rng(seed)
    lx, ly, lz, lt = _LATTICE
    gauge = np.broadcast_to(
        np.eye(3, dtype=np.complex128), (lt, lz, ly, lx, 4, 3, 3),
    ).copy()
    vectors = tuple(
        rng.normal(size=(lt, 3, lz, ly, lx, 3))
        + 1j * rng.normal(size=(lt, 3, lz, ly, lx, 3))
        for _ in range(3)
    )
    return gauge, vectors


def _make_on_demand(component, gauge, vectors):
    table = ({"component": component, "path": ((), (), ()), "coefficient": 1},)
    return OnDemandBaryonElemental(
        gauge, vectors, _MOMENTUM, table, lattice_size=_LATTICE,
        mode_counts=(3, 3, 3), cfg="cfg-a", ne_block_size=2,
        spatial_block_size=2,
    )


def _full_vvv(component, gauge, vectors, time=0):
    table = ({"component": component, "path": ((), (), ()), "coefficient": 1},)
    producer = BaryonElementalGenerator(
        gauge, vectors, _MOMENTUM, table, Ne=(3, 3, 3),
        ne_block_size=2, spatial_block_size=2,
    ).load("cfg-a")
    return producer.calc_path_blocks(time)["vvv"][0, 0].copy()


def _replace_elemental_readers(case, source_reader, sink_reader,
                               source_spatial, sink_spatial):
    component = case["vertices"]["source"].operator.terms[0].component
    source_array = np.zeros((1, 1, 2, 3, 3, 3), dtype=np.complex128)
    sink_array = np.zeros_like(source_array)
    source_array[0, 0, 1] = source_spatial
    sink_array[0, 0, 0] = sink_spatial
    source_reference = ArrayBaryonElemental(
        {"vvv": source_array}, (component,), cfg="cfg-a",
        data_identity="cfg-a/source-vvv-public-generator",
    )
    sink_reference = ArrayBaryonElemental(
        {"vvv": sink_array}, (component,), cfg="cfg-a",
        data_identity="cfg-a/sink-vvv-public-generator",
    )

    source_operator = case["vertices"]["source"].operator
    sink_operator = case["vertices"]["sink"].operator
    reference_case = dict(case)
    reference_case["source_spatial"] = source_spatial
    reference_case["sink_spatial"] = sink_spatial
    reference_case["source_reader"] = source_reference
    reference_case["sink_reader"] = sink_reference
    reference_case["vertices"] = {
        "source": Baryon(source_reference, source_operator, source=True),
        "sink": Baryon(sink_reference, sink_operator),
    }

    on_demand_case = dict(reference_case)
    on_demand_case["source_reader"] = source_reader
    on_demand_case["sink_reader"] = sink_reader
    on_demand_case["vertices"] = {
        "source": Baryon(source_reader, source_operator, source=True),
        "sink": Baryon(sink_reader, sink_operator),
    }
    return reference_case, on_demand_case


class OnDemandBaryonContractionTests(unittest.TestCase):
    def test_source_adjoint_exchanged_slots_and_every_two_bin_block(self):
        # Exchanging source slots checks that the on-demand raw VVV data still
        # follows the existing source-adjoint and Wick-axis mapping.
        case = _case(source_slots=(1, 0, 2), seed=891)
        component = case["vertices"]["source"].operator.terms[0].component
        source_gauge, source_vectors = _fields(910)
        sink_gauge, sink_vectors = _fields(911)
        source_reader = _make_on_demand(component, source_gauge, source_vectors)
        sink_reader = _make_on_demand(component, sink_gauge, sink_vectors)
        source_spatial = _full_vvv(component, source_gauge, source_vectors, time=1)
        sink_spatial = _full_vvv(component, sink_gauge, sink_vectors)
        np.testing.assert_allclose(source_reader.block(component, 0, 1), source_spatial,
                                   rtol=3e-12, atol=2e-10)
        np.testing.assert_allclose(sink_reader.block(component, 0, 0), sink_spatial,
                                   rtol=3e-12, atol=2e-10)
        reference_case, on_demand_case = _replace_elemental_readers(
            case, source_reader, sink_reader, source_spatial, sink_spatial,
        )

        expected_whole = _calculate(reference_case, case["diagram"]).value
        np.testing.assert_allclose(
            expected_whole, _independent_einsum(reference_case),
            rtol=3e-12, atol=1e-9,
        )
        actual_whole = _calculate(on_demand_case, case["diagram"]).value
        np.testing.assert_allclose(actual_whole, expected_whole,
                                   rtol=3e-12, atol=1e-9)

        blocks = split_baryon_eigenmodes(case["diagram"], partitions=_PARTITIONS)
        self.assertEqual(blocks.shape, (2, 2, 2, 2, 2, 2))
        values = np.empty(blocks.shape, dtype=np.complex128)
        for index in np.ndindex(blocks.shape):
            source_selection, sink_selection = _selections_for_block_index(index)
            reference = _calculate(reference_case, blocks[index]).value
            actual = _calculate(on_demand_case, blocks[index]).value
            np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=1e-9)
            np.testing.assert_allclose(
                actual,
                _independent_einsum(reference_case, source_selection, sink_selection),
                rtol=3e-12, atol=1e-9,
            )
            values[index] = actual
        np.testing.assert_allclose(values.sum(), actual_whole,
                                   rtol=3e-12, atol=1e-9)

    def test_planning_does_not_read_blocks_and_prefix_counts_reach_reader(self):
        case = _case(seed=971)
        component = case["vertices"]["source"].operator.terms[0].component
        source_gauge, source_vectors = _fields(972)
        sink_gauge, sink_vectors = _fields(973)
        source_reader = _make_on_demand(component, source_gauge, source_vectors)
        sink_reader = _make_on_demand(component, sink_gauge, sink_vectors)
        source_spatial = _full_vvv(component, source_gauge, source_vectors, time=1)
        sink_spatial = _full_vvv(component, sink_gauge, sink_vectors)
        _, on_demand_case = _replace_elemental_readers(
            case, source_reader, sink_reader, source_spatial, sink_spatial,
        )
        selected = split_baryon_eigenmodes(
            case["diagram"], partitions=_PARTITIONS,
        )[(1, 0, 1, 0, 1, 0)]
        with (
            patch.object(source_reader, "block",
                         side_effect=AssertionError("planning read source VVV")) as source_read,
            patch.object(sink_reader, "block",
                         side_effect=AssertionError("planning read sink VVV")) as sink_read,
        ):
            calculation = _calculate(on_demand_case, selected, execute=False)
            source_read.assert_not_called()
            sink_read.assert_not_called()
        calculation.run()

        source = Baryon(source_reader, case["vertices"]["source"].operator,
                        source=True).load("cfg-a", usedNe=(1, 2, 1))
        block_calls = []
        original_block = source_reader.block

        def record(*args, **kwargs):
            block_calls.append((args, kwargs))
            return original_block(*args, **kwargs)

        source_reader.block = record
        tuple(source.factors(1))
        self.assertEqual(len(block_calls), 1)
        self.assertEqual(block_calls[0][1]["usedNe"], (1, 2, 1))
        self.assertEqual(block_calls[0][1].get("mode_indices"), None)


if __name__ == "__main__":
    unittest.main()
