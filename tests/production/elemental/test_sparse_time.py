"""Compact inputs retain physical-time physics and independent input ordering."""

import unittest
import numpy as np

from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from lqcdweave.production.elemental.api import ElementalGenerator, BaryonElementalGenerator
from production.elemental.test_elemental import fixture, reference_matrix, point_basis, Handle
from test_baryon_elemental import determinant_reference


class MappedModes(Handle):
    def __init__(self, values, selection):
        super().__init__(values)
        self.time_selection = selection
        self.Ne = values.shape[1]
        self.reads = []

    def read_modes(self, key, t, start, stop):
        self.reads.append(t)
        return self.array[self.time_selection.slot(t), start:stop]


class SparseElementalTests(unittest.TestCase):
    def setUp(self):
        self.sizes, self.gauge, self.vectors, self.points = fixture()
        self.selection = TimeSelectionRecord(3, (2, 0))
        self.momenta = ((0, 0, 0), (1, -1, 0))

    def test_same_time_four_sectors_match_independent_fulltime_selection(self):
        table = [dict(component="mixed", path=(), coefficient=0.3 + 0.4j),
                 dict(component="mixed", path=(0,), coefficient=-0.7j)]
        selected = list(self.selection.times)
        generator = ElementalGenerator(
            self.gauge, self.vectors[selected], self.momenta, table,
            points=self.points[selected],
            blocks=("v2v", "v2p", "p2v", "p2p"), selection=self.selection)
        self.assertEqual((generator.Lt, generator.Nt), (3, 2))
        references = []
        for t in range(3):
            bases = {"v": self.vectors[t].reshape(3, -1).T,
                     "p": point_basis(self.points[t], self.sizes)}
            reference = {}
            for mi, momentum in enumerate(self.momenta):
                matrix = (0.3 + 0.4j) * reference_matrix(self.gauge, (), t, momentum)
                matrix -= 0.7j * reference_matrix(self.gauge, (0,), t, momentum)
                for sector in generator.blocks:
                    reference[mi, sector] = bases[sector[0]].conj().T @ matrix @ bases[sector[-1]]
            references.append(reference)
        seen = []
        for t, result in generator.iter_slices():
            seen.append(t)
            for mi in range(2):
                for sector in generator.blocks:
                    value = result[sector][0][mi]
                    if sector == "p2p":
                        value = value.to_dense()
                    np.testing.assert_allclose(value.reshape(references[t][mi, sector].shape),
                                               references[t][mi, sector], atol=2e-12)
        self.assertEqual(seen, selected)
        with self.assertRaises(KeyError):
            generator.calc_path_blocks(1)

    def test_sparse_derivative_and_closed_temporal_excursion(self):
        selected = list(self.selection.times)
        derivative = ElementalGenerator(
            self.gauge, self.vectors[selected], self.momenta,
            [dict(component="d", path=(0,), coefficient=0.2 + 0.4j)],
            selection=self.selection)
        excursion = ElementalGenerator(
            self.gauge, self.vectors[selected], self.momenta,
            [dict(component="loop", path=(6, 0, 7), coefficient=1)],
            selection=self.selection)
        # The intermediate gauge time needs no extra eigenvector endpoint.
        for t in self.selection.times:
            vectors = self.vectors[t].reshape(3, -1).T
            d = reference_matrix(self.gauge, (0,), t) - reference_matrix(self.gauge, (3,), t)
            out = derivative.calc_derivative(t)["v2v"]
            loop = excursion.calc_path_blocks(t)["v2v"]
            for mi, momentum in enumerate(self.momenta):
                phase = reference_matrix(self.gauge, (), t, momentum)
                expected = (0.2 + 0.4j) * ((d @ vectors).conj().T @ phase @ vectors
                                          - vectors.conj().T @ phase @ (d @ vectors))
                np.testing.assert_allclose(out[0, mi], expected, atol=2e-12)
                np.testing.assert_allclose(loop[0, mi], vectors.conj().T @ reference_matrix(
                    self.gauge, (6, 0, 7), t, momentum) @ vectors, atol=2e-12)

    def test_temporal_endpoints_use_independent_basis_and_point_mappings(self):
        basis_selection = TimeSelectionRecord(3, (1, 2, 0))
        point_selection = TimeSelectionRecord(3, (0, 2, 1))
        table = [dict(component="forward", path=(6, 0), coefficient=0.3j),
                 dict(component="backward", path=(7,), coefficient=-0.4)]
        generator = ElementalGenerator(
            self.gauge, self.vectors[list(basis_selection.times)], self.momenta, table,
            points=self.points[list(point_selection.times)],
            blocks=("v2v", "v2p", "p2v", "p2p"), selection=self.selection,
            eigenvector_selection=basis_selection, point_selection=point_selection)
        for t in self.selection.times:
            out = generator.calc_path_blocks(t)
            for ci, (path, coefficient, dt) in enumerate([((6, 0), 0.3j, 1), ((7,), -0.4, -1)]):
                endpoint = (t + dt) % 3
                left = {"v": self.vectors[t].reshape(3, -1).T,
                        "p": point_basis(self.points[t], self.sizes)}
                right = {"v": self.vectors[endpoint].reshape(3, -1).T,
                         "p": point_basis(self.points[endpoint], self.sizes)}
                for mi, momentum in enumerate(self.momenta):
                    matrix = coefficient * reference_matrix(self.gauge, path, t, momentum)
                    for sector in generator.blocks:
                        expected = left[sector[0]].conj().T @ matrix @ right[sector[-1]]
                        value = out[sector][ci][mi]
                        if sector == "p2p":
                            value = value.to_dense()
                        np.testing.assert_allclose(value.reshape(expected.shape), expected, atol=2e-12)

    def test_missing_temporal_endpoint_fails_before_delivery(self):
        delivered = []
        table = [dict(component="local", path=(), coefficient=1),
                 dict(component="later", path=(6,), coefficient=1)]
        generator = ElementalGenerator(
            self.gauge, self.vectors[list(self.selection.times)], self.momenta, table,
            selection=self.selection, write=lambda *args: delivered.append(args))
        with self.assertRaises(KeyError):
            generator.calc_path_blocks(0)  # Physical endpoint 1 is absent; slot 1 is physical 0.
        self.assertEqual(delivered, [])
        generator = ElementalGenerator(
            self.gauge, None, self.momenta, table, blocks=("p2p",),
            points=self.points[list(self.selection.times)],
            selection=self.selection)
        with self.assertRaises(KeyError):
            generator.calc_path_blocks(0)

    def test_sparse_spatial_point_set_keeps_its_own_order(self):
        times = (0, 2)
        points = SpatialPointSet(self.sizes, times, self.points[list(times)])
        generator = ElementalGenerator(
            self.gauge, None, self.momenta,
            [dict(component="local", path=(), coefficient=1)], blocks=("p2p",),
            points=points, selection=self.selection)
        for t, result in generator.iter_slices():
            point = point_basis(self.points[t], self.sizes)
            for mi, momentum in enumerate(self.momenta):
                expected = point.conj().T @ reference_matrix(self.gauge, (), t, momentum) @ point
                np.testing.assert_allclose(result["p2p"][0][mi].to_dense().reshape(expected.shape),
                                           expected, atol=2e-12)

    def test_open_excluded_path_needs_no_wrapped_endpoint_input(self):
        selection = TimeSelectionRecord(3, (0,))
        generator = ElementalGenerator(
            self.gauge, self.vectors[[0]], self.momenta,
            [dict(component="past-edge", path=(7,), coefficient=1)],
            selection=selection, path_boundary="open",
            points=self.points[[0]],
            blocks=("v2v", "v2p", "p2v", "p2p"))
        result = generator.calc_path_blocks(0)
        for sector in ("v2v", "v2p", "p2v"):
            np.testing.assert_array_equal(result[sector], 0)
        self.assertEqual(len(result["p2p"][0][0].rows), 0)

    def test_metadata_less_compact_handles_are_not_guessed(self):
        table = [dict(component="local", path=(), coefficient=1)]
        with self.assertRaises(ValueError):
            ElementalGenerator(self.gauge, Handle(self.vectors[[2, 0]]), self.momenta,
                               table, selection=self.selection).load("cfg")
        handle = MappedModes(self.vectors[[2, 0]], self.selection)
        generator = ElementalGenerator(self.gauge, handle, self.momenta, table,
                                       selection=self.selection).load("cfg")
        value = generator.calc_path_blocks(2)["v2v"][0, 0]
        v = self.vectors[2].reshape(3, -1)
        np.testing.assert_allclose(value, v.conj() @ v.T, atol=2e-12)
        with self.assertRaises(ValueError):
            ElementalGenerator(self.gauge, handle, self.momenta, table,
                               selection=self.selection,
                               eigenvector_selection=TimeSelectionRecord(3, (0, 2)))

    def test_three_baryon_legs_keep_independent_orders_eager_and_streaming(self):
        selections = (TimeSelectionRecord(3, (2, 0)), TimeSelectionRecord(3, (0, 2)),
                      TimeSelectionRecord(3, (1, 0, 2)))
        full = (self.vectors[:, :1], (0.4j * self.vectors + 0.3)[:, :2], self.vectors + 0.2j)
        compact = tuple(v[list(s.times)] for v, s in zip(full, selections))
        paths = ((0,), (1, 3), ())
        table = [dict(component="b", path=paths, coefficient=0.2 + 0.7j)]
        references = [[(0.2 + 0.7j) * determinant_reference(self.gauge, full, paths, t, p)
                       for p in self.momenta] for t in range(3)]
        for streaming in (False, True):
            handles = tuple(MappedModes(v, s) for v, s in zip(compact, selections))
            generator = BaryonElementalGenerator(
                self.gauge, handles if streaming else compact, self.momenta, table,
                selection=self.selection, eigenvector_selection=None if streaming else selections,
                stream_eigenvectors=streaming, ne_block_size=1).load("cfg")
            self.assertEqual((generator.Lt, generator.Nt), (3, 2))
            for t, out in generator.iter_slices():
                for mi in range(2):
                    np.testing.assert_allclose(out["vvv"][0, mi], references[t][mi], atol=2e-12)
            if streaming:
                for handle in handles:
                    self.assertEqual(set(handle.reads), {2, 0})
            generator.project_SU3()  # Selection can omit physical time zero in general.

    def test_baryon_default_shared_compact_array_and_missing_leg(self):
        table = [dict(component="b", path=((), (), ()), coefficient=1)]
        selection = TimeSelectionRecord(3, (2,))
        for streaming in (False, True):
            generator = BaryonElementalGenerator(
                self.gauge, self.vectors[[2]], self.momenta, table,
                selection=selection, stream_eigenvectors=streaming)
            generator.project_SU3()
            self.assertEqual(generator.calc_path_blocks(2)["vvv"].shape, (1, 2, 3, 3, 3))
        with self.assertRaises(KeyError):
            BaryonElementalGenerator(self.gauge, self.vectors[[2]], self.momenta, table,
                                    selection=self.selection, eigenvector_selection=selection)
