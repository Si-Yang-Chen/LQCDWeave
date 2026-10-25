"""Contract and numerical tests for selected, runtime baryon elementals."""

from itertools import product
import unittest

import numpy as np

from lqcdweave.adapters.baryon_on_demand import OnDemandBaryonElemental
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.production.elemental.api import BaryonElementalGenerator


_COUNTS = (3, 4, 2)
_LATTICE = (3, 2, 1, 2)
_MOMENTA = ((1, 0, 0), (0, 1, 0))
_SELECTED = {"mode0": (2, 0), "mode1": (3, 1), "mode2": (1, 0)}
_LOCAL = BaryonComponent.paths(((), (), ()), flavors=("u", "d", "u"))
_PROJECTED = BaryonComponent(("projected", "complex-path-sum"), ("u", "d", "u"))
_PATH_A = ((0,), (1,), ())
_PATH_B = ((3,), (2,), (0,))
_TABLE = (
    {"component": _LOCAL, "path": ((), (), ()), "coefficient": 1},
    {"component": _PROJECTED, "terms": (
        {"paths": _PATH_A, "coefficient": (0.35, -0.6)},
        {"paths": _PATH_B, "coefficient": (-0.2, 0.45)},
    )},
)


class _GaugeHandle:
    def __init__(self, values, *, shaped=False, borrowed=False):
        self.values = values
        self.reads = []
        self.shape = values.shape if shaped else None
        self.dtype = values.dtype if shaped else None
        self.buffer = np.empty_like(values[0]) if borrowed else None

    def __getitem__(self, key):
        raise AssertionError("gauge handle must use read_time")

    def read_time(self, cfg, t):
        self.reads.append((cfg, t))
        if self.buffer is None:
            return self.values[t]
        self.buffer[...] = self.values[t]
        return self.buffer


class _BasisHandle:
    def __init__(self, values, *, shaped=False, borrowed=False, fail_once=False):
        self.values = values
        self.Ne = values.shape[1]
        self.shape = values.shape if shaped else None
        self.dtype = values.dtype if shaped else None
        self.reads = []
        self.buffer = np.empty((self.Ne, *values.shape[2:]), dtype=values.dtype) if borrowed else None
        self.fail_once = fail_once

    def __getitem__(self, key):
        raise AssertionError("basis handle must use read_modes")

    def read_modes(self, cfg, t, start, stop):
        self.reads.append((cfg, t, start, stop))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("basis read failed")
        if self.buffer is None:
            return self.values[t, start:stop]
        self.buffer[:stop - start] = self.values[t, start:stop]
        return self.buffer[:stop - start]


def _fixture(*, handles=False, shaped_handles=False, borrowed=False, failing_leg=None):
    rng = np.random.default_rng(811)
    lx, ly, lz, lt = _LATTICE
    gauge = np.empty((lt, lz, ly, lx, 4, 3, 3), dtype=np.complex128)
    for index in np.ndindex(gauge.shape[:5]):
        matrix = np.eye(3, dtype=np.complex128)
        matrix += 0.012 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        gauge[index] = matrix
    vectors = tuple(
        rng.normal(size=(lt, count, lz, ly, lx, 3))
        + 1j * rng.normal(size=(lt, count, lz, ly, lx, 3))
        for count in _COUNTS
    )
    if handles:
        gauge_input = _GaugeHandle(gauge, shaped=shaped_handles, borrowed=borrowed)
        basis_inputs = tuple(
            _BasisHandle(value, shaped=shaped_handles, borrowed=borrowed,
                         fail_once=(leg == failing_leg))
            for leg, value in enumerate(vectors)
        )
    else:
        gauge_input, basis_inputs = gauge, vectors
    bindings = (
        ("basis-left", ("left-0", "left-1", "left-2")),
        ("basis-middle", ("mid-0", "mid-1", "mid-2", "mid-3")),
        None,
    )
    reader = OnDemandBaryonElemental(
        gauge_input, basis_inputs, _MOMENTA, _TABLE, lattice_size=_LATTICE,
        mode_counts=_COUNTS, basis_bindings=bindings, cfg="cfg-a",
        ne_block_size=(2, 2, 2), spatial_block_size=3,
    )
    return reader, gauge, vectors, gauge_input, basis_inputs


def _public_full(gauge, vectors):
    generator = BaryonElementalGenerator(
        gauge, vectors, _MOMENTA, _TABLE, Ne=_COUNTS,
        ne_block_size=(2, 2, 2), spatial_block_size=3,
    ).load("cfg-a")
    return {t: generator.calc_path_blocks(t)["vvv"].copy()
            for t in range(_LATTICE[3])}


def _local_determinant(vectors, t, momentum, indices=None):
    """Independent epsilon determinant with the declared positive Fourier phase."""
    indices = tuple(tuple(range(count)) for count in _COUNTS) if indices is None else indices
    lx, ly, lz, _ = _LATTICE
    result = np.zeros(tuple(map(len, indices)), dtype=np.complex128)
    for selected_modes in product(*(range(len(axis)) for axis in indices)):
        for z, y, x in product(range(lz), range(ly), range(lx)):
            colors = np.stack(tuple(
                vectors[leg][t, indices[leg][selected_modes[leg]], z, y, x]
                for leg in range(3)
            ))
            phase = np.exp(2j * np.pi * sum(
                p * coordinate / extent
                for p, coordinate, extent in zip(momentum, (x, y, z), (lx, ly, lz))
            ))
            result[selected_modes] += phase * np.linalg.det(colors)
    return result


class OnDemandBaryonElementalTests(unittest.TestCase):
    def test_shared_basis_inferred_headers_and_zero_cache(self):
        _, gauge, vectors, _, _ = _fixture()
        shared = vectors[0]
        with OnDemandBaryonElemental(
            gauge, shared, _MOMENTA, _TABLE, cache_bytes=0,
        ) as reader:
            self.assertEqual(reader.mode_counts, (3, 3, 3))
            self.assertEqual(reader.describe_block(_LOCAL, 0, 0).axis_bindings, {})
            reference = BaryonElementalGenerator(gauge, shared, _MOMENTA, _TABLE)
            expected = reference.calc_path_blocks(0)["vvv"][0, 0].copy()
            first = reader.block(_LOCAL, 0, 0)
            np.testing.assert_allclose(first, expected, rtol=3e-12, atol=2e-10)
            self.assertIsNot(first, reader.block(_LOCAL, 0, 0))
            self.assertIsNone(reader._cached_value)
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.describe_block(_LOCAL, 0, 0)
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.block(_LOCAL, 0, 0)

    def test_reload_clears_cache_and_nonfinite_gauge_fails(self):
        reader, gauge, _, _, _ = _fixture()
        first = reader.block(_LOCAL, 0, 0)
        reader.load("cfg-a")
        self.assertIsNone(reader._cached_value)
        self.assertIsNot(first, reader.block(_LOCAL, 0, 0))
        gauge[1, 0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "gauge field must be finite"):
            reader.block(_LOCAL, 0, 1)
        self.assertIsNone(reader._cached_value)

    def test_full_and_scattered_blocks_match_public_generator_and_local_epsilon(self):
        reader, gauge, vectors, _, _ = _fixture()
        reference = _public_full(gauge, vectors)
        local_row = reader.components.index(_LOCAL)
        projected_row = reader.components.index(_PROJECTED)

        for t in range(_LATTICE[3]):
            for momentum in range(len(_MOMENTA)):
                for component, row in ((_LOCAL, local_row), (_PROJECTED, projected_row)):
                    np.testing.assert_allclose(
                        reader.block(component, momentum, t),
                        reference[t][row, momentum], rtol=3e-12, atol=2e-10,
                    )
                    scattered = reader.block(
                        component, momentum, t, mode_indices=_SELECTED,
                    )
                    expected = reference[t][row, momentum][np.ix_(
                        *_SELECTED.values(),
                    )]
                    np.testing.assert_allclose(scattered, expected, rtol=3e-12, atol=2e-10)
                np.testing.assert_allclose(
                    reader.block(_LOCAL, momentum, t),
                    _local_determinant(vectors, t, _MOMENTA[momentum]),
                    rtol=3e-12, atol=2e-10,
                )
                np.testing.assert_allclose(
                    reader.block(_LOCAL, momentum, t, mode_indices=_SELECTED),
                    _local_determinant(
                        vectors, t, _MOMENTA[momentum],
                        tuple(_SELECTED[axis] for axis in ("mode0", "mode1", "mode2")),
                    ),
                    rtol=3e-12, atol=2e-10,
                )

    def test_description_is_header_only_and_bindings_keep_scatter_order(self):
        reader, _, _, gauge, bases = _fixture(
            handles=True, shaped_handles=True,
        )
        metadata = reader.describe_block(_LOCAL, 0, 1, mode_indices=_SELECTED)
        self.assertEqual(metadata.axes, ("mode0", "mode1", "mode2"))
        self.assertEqual(metadata.shape, (2, 2, 2))
        self.assertEqual(tuple(metadata.axis_bindings), ("mode0", "mode1"))
        self.assertEqual(metadata.axis_bindings["mode0"],
                         ("basis-left", ("left-2", "left-0")))
        self.assertEqual(metadata.axis_bindings["mode1"],
                         ("basis-middle", ("mid-3", "mid-1")))
        self.assertIsNone(metadata.data_identity)
        self.assertIsNone(reader.data_identity)
        self.assertEqual(gauge.reads, [])
        self.assertTrue(all(handle.reads == [] for handle in bases))

        value = reader.block(_LOCAL, 0, 1, mode_indices=_SELECTED)
        self.assertEqual(gauge.reads, [("cfg-a", 1)])
        self.assertEqual(
            [handle.reads for handle in bases],
            [[("cfg-a", 1, 2, 3), ("cfg-a", 1, 0, 1)],
             [("cfg-a", 1, 3, 4), ("cfg-a", 1, 1, 2)],
             [("cfg-a", 1, 1, 2), ("cfg-a", 1, 0, 1)]],
        )
        self.assertEqual(value.shape, (2, 2, 2))

    def test_borrowed_handle_buffers_do_not_alias_cached_results_or_other_legs(self):
        reader, gauge_values, vectors, _, _ = _fixture(
            handles=True, borrowed=True,
        )
        reference = _public_full(gauge_values, vectors)
        first = reader.block(_PROJECTED, 0, 0, mode_indices=_SELECTED)
        saved = first.copy()
        self.assertFalse(first.flags.writeable)
        reader.block(_PROJECTED, 1, 1, mode_indices=_SELECTED)
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_allclose(
            first, reference[0][reader.components.index(_PROJECTED), 0][np.ix_(*_SELECTED.values())],
            rtol=3e-12, atol=2e-10,
        )

    def test_cache_load_failure_invalidation_and_stable_owned_results(self):
        reader, gauge, vectors, gauge_handle, basis_handles = _fixture(handles=True)
        reference = _public_full(gauge, vectors)
        first = reader.block(_LOCAL, 0, 0)
        saved = first.copy()
        reads_after_first = (len(gauge_handle.reads), tuple(len(h.reads) for h in basis_handles))
        self.assertIs(reader.block(_LOCAL, 0, 0), first)
        self.assertEqual(
            reads_after_first,
            (len(gauge_handle.reads), tuple(len(h.reads) for h in basis_handles)),
        )
        reader.clear_cache()
        refreshed = reader.block(_LOCAL, 0, 0)
        self.assertIsNot(refreshed, first)
        reader.block(_LOCAL, 1, 1)
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_allclose(first, reference[0][0, 0], rtol=3e-12, atol=2e-10)

        reader.clear_cache()
        reader.block(_LOCAL, 0, 0)
        basis_handles[1].fail_once = True
        with self.assertRaisesRegex(RuntimeError, "basis read failed"):
            reader.block(_LOCAL, 0, 1)
        self.assertIsNone(reader._cached_key)
        count_before = len(gauge_handle.reads)
        reread = reader.block(_LOCAL, 0, 0)
        self.assertEqual(len(gauge_handle.reads), count_before + 1)
        np.testing.assert_allclose(reread, reference[0][0, 0], rtol=3e-12, atol=2e-10)
        with self.assertRaisesRegex(ValueError, "fresh on-demand reader"):
            reader.load("cfg-b")

    def test_unknown_bindings_and_invalid_coverage_or_selections(self):
        reader, _, _, _, _ = _fixture()
        description = reader.describe_block(_PROJECTED, 0, 0)
        self.assertEqual(tuple(description.axis_bindings), ("mode0", "mode1"))
        self.assertIsNone(description.data_identity)
        for args in (
            (_PROJECTED, 2, 0), (_PROJECTED, 0, 2),
            (BaryonComponent(("unknown",)), 0, 0),
        ):
            with self.subTest(args=args), self.assertRaises(KeyError):
                reader.block(*args)
        with self.assertRaises(KeyError):
            reader.block(_LOCAL, 0, 0, "v2v")
        for selection in (
            {"mode0": ()}, {"mode0": (2, 2)}, {"mode1": (4,)},
            {"mode2": (-1,)}, {"mode3": (0,)},
        ):
            with self.subTest(selection=selection), self.assertRaises((TypeError, ValueError)):
                reader.describe_block(_LOCAL, 0, 0, mode_indices=selection)
        with self.assertRaisesRegex(ValueError, "choose usedNe or mode_indices"):
            reader.describe_block(_LOCAL, 0, 0, usedNe=(2, 2, 2), mode_indices=_SELECTED)
        lx, ly, lz, lt = _LATTICE
        with self.assertRaisesRegex(ValueError, "coverage|usedNe exceeds|out of bounds|invalid"):
            OnDemandBaryonElemental(
                np.zeros((lt, lz, ly, lx, 4, 3, 3), complex),
                tuple(np.zeros((lt, n, lz, ly, lx, 3), complex) for n in (2, 3, 1)),
                _MOMENTA, _TABLE, lattice_size=_LATTICE, mode_counts=_COUNTS,
            )


if __name__ == "__main__":
    unittest.main()
