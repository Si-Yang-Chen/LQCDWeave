"""Nine logical sectors against literal site/link matrices, with raw D norms."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from lqcdweave.contracts.dilution import DilutionSource
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS
from lqcdweave.contracts.spatial_sampling import SpatialPointSet, SpatialSamplingDesign
from lqcdweave.contracts.time_selection import TimeSelectionRecord, bind_time_selections
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.production.noisevector.api import NoisevectorGenerator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.production import elemental_dataset_document, point_elemental_dataset_document
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader
from lqcdweave.adapters.meson_dataset import DatasetElemental
from production.elemental.test_elemental import fixture, reference_matrix, point_basis


class Basis:
    basis_orthogonal = True
    basis_id = "actual-parent"

    def __init__(self, values, source=None, times=None):
        self.values, self.dilution_source, self.time_selection = values, source, times
        self.Ne = values.shape[1]
        self.mode_labels = tuple(range(self.Ne))
        if source is not None:
            self.basis_id = source.data_id

    def load(self, key):
        return self.values


class DilutionSectorProductionTests(unittest.TestCase):
    def setUp(self):
        self.sizes, self.gauge, raw, points = fixture()
        rng = np.random.default_rng(630)
        values = []
        for _ in range(self.sizes[3]):
            q, _r = np.linalg.qr(rng.normal(size=(raw[0, 0].size, 5)) + 1j * rng.normal(size=(raw[0, 0].size, 5)))
            values.append(q.T.reshape(5, *raw.shape[2:]))
        self.parent = Basis(np.stack(values))
        self.vectors = Basis(self.parent.values[:, :2])
        self.points = SpatialPointSet(self.sizes, tuple(range(self.sizes[3])), points,
            design=SpatialSamplingDesign(self.sizes, points.shape[0], "independent"),
            pool_ids=tuple(f"P-{t}" for t in range(self.sizes[3])))
        self.local, self.directed = ElementalComponent.path(()), ElementalComponent.path((0, 6))
        self.coefficient = 0.3 + 0.7j
        self.table = [dict(component=self.local, path=(), coefficient=self.coefficient),
                      dict(component=self.directed, path=(0, 6), coefficient=1)]
        self.momenta = ((0, 0, 0), (1, -1, 0))

    def noise(self, mixing="qr", dilution=((2, 3), (1, 2))):
        generator = NoisevectorGenerator(self.parent, dilution, mixing=mixing).load(None)
        values = generator.calc(seed=527)
        source = DilutionSource.from_description(generator.describe(), data_id=f"actual-D-{mixing}",
            time_pools=generator.describe()["time_pools"])
        return Basis(values, source), generator

    def producer(self, d, weighted=False, *, blocks=ELEMENTAL_SECTORS, table=None, momenta=None, **kwargs):
        return ElementalGenerator(self.gauge, None, self.momenta if momenta is None else momenta,
            self.table if table is None else table,
            vector_bases={"V": self.vectors, "D": d}, blocks=blocks,
            points=self.points, lattice_size=self.sizes, ne_block_size=1,
            elemental_weighted=weighted, **kwargs).load(None)

    def test_nine_sectors_independent_spatial_temporal_transport_reference(self):
        d, _ = self.noise()
        producer = self.producer(d)
        self.assertEqual(producer.elemental_structures["d2d", self.local, 0], "diagonal")
        self.assertNotIn(("v2d", self.local, 0), producer.elemental_structures)
        for time in range(self.sizes[3]):
            result = producer.calc_path_blocks(time)
            for ri, path in enumerate(((), (0, 6))):
                endpoint = (time + (1 if path else 0)) % self.sizes[3]
                left = {"v": self.vectors.values[time].reshape(2, -1).T,
                        "d": d.values[time].reshape(3, -1).T,
                        "p": point_basis(self.points.coordinates[time], self.sizes)}
                right = {"v": self.vectors.values[endpoint].reshape(2, -1).T,
                         "d": d.values[endpoint].reshape(3, -1).T,
                         "p": point_basis(self.points.coordinates[endpoint], self.sizes)}
                for mi, momentum in enumerate(self.momenta):
                    matrix = reference_matrix(self.gauge, path, time, momentum)
                    for sector in ELEMENTAL_SECTORS:
                        with self.subTest(time=time, row=ri, momentum=mi, sector=sector):
                            expected = left[sector[0]].conj().T @ matrix @ right[sector[-1]]
                            expected *= self.coefficient if ri == 0 else 1
                            value = result[sector][ri][mi]
                            if sector == "p2p":
                                value = value.to_dense()
                            np.testing.assert_allclose(value.reshape(expected.shape), expected, atol=3e-14)

    def test_one_weight_switch_combines_dilution_and_point_factors(self):
        for mixing in ("qr", "legacy"):
            d, _ = self.noise(mixing, ((5,), (3,)) if mixing == "qr" else ((2, 3), (1, 2)))
            blocks = ELEMENTAL_SECTORS if mixing == "qr" else tuple(s for s in ELEMENTAL_SECTORS if s != "d2d")
            raw, weighted = self.producer(d, blocks=blocks), self.producer(d, True, blocks=blocks)
            a, b = raw.calc_path_blocks(0), weighted.calc_path_blocks(0)
            factors = np.asarray(d.dilution_source.first_moment_weights)
            point_weight = np.prod(self.sizes[:3]) / self.points.num_points
            for sector in blocks:
                if sector == "p2p":
                    np.testing.assert_allclose(b[sector][0][0].values, a[sector][0][0].values * point_weight)
                    continue
                value = a[sector].copy()
                if sector == "d2d":
                    # The local row is same-pool; temporal row has independent actual-time pools.
                    for mi in range(len(self.momenta)):
                        mask = np.full((3, 3), 5 * 4 / (3 * 2))
                        np.fill_diagonal(mask, 5 / 3)
                        value[0, mi] *= mask
                        value[1, mi] *= (5 / 3) ** 2
                elif sector[0] == "d":
                    value *= factors.reshape((1, 1, 3) + (1,) * (value.ndim - 3))
                if sector[-1] == "d" and sector != "d2d":
                    value *= factors
                if "p" in sector:
                    value *= point_weight
                np.testing.assert_allclose(b[sector], value, atol=3e-14)

    def test_independent_actual_vector_time_orders_and_d_only(self):
        d, _ = self.noise()
        order = (2, 0, 1)
        reordered = Basis(d.values[list(order)], d.dilution_source, TimeSelectionRecord(3, order))
        result = self.producer(reordered).calc_path_blocks(2)["v2d"]
        reference = self.producer(d).calc_path_blocks(2)["v2d"]
        np.testing.assert_allclose(result, reference, atol=3e-14)
        only = ElementalGenerator(self.gauge, None, ((0, 0, 0),), self.table[:1],
            vector_bases={"D": d}, blocks=("d2d",), elemental_weighted=True, lattice_size=self.sizes).load(None)
        weights = np.asarray(d.dilution_source.first_moment_weights)
        np.testing.assert_allclose(only.calc_path_blocks(0)["d2d"][0, 0],
            self.coefficient * np.diag(weights), atol=3e-14)

    def test_unknown_d_weight_provenance_rejected_and_pools_identify_realizations(self):
        d, generator = self.noise()
        unknown = Basis(d.values)
        self.producer(unknown)
        with self.assertRaisesRegex(ValueError, "explicit dilution source"):
            self.producer(unknown, True)
        before = generator.describe()["time_pools"]
        values = generator.calc(seed=527)
        np.testing.assert_array_equal(values, d.values)
        after = generator.describe()["time_pools"]
        self.assertNotEqual(before, after)
        self.assertEqual(len(set(pool for time, pool in after)), self.sizes[3])

    def test_component_masks_for_noncontiguous_groups_with_distinct_n_over_m(self):
        rng = np.random.default_rng(930)
        parent_values = []
        for time in range(self.sizes[3]):
            q, _ = np.linalg.qr(rng.normal(size=(self.parent.values[0, 0].size, 7))
                                + 1j * rng.normal(size=(self.parent.values[0, 0].size, 7)))
            parent_values.append(q.T.reshape(7, *self.parent.values.shape[2:]))
        parent = Basis(np.stack(parent_values))
        groups = ((0, 2, 5), (1, 3, 4, 6))
        generator = NoisevectorGenerator(parent, ((3, 4), (2, 3)), mode_groups=groups).load(None)
        values = generator.calc(seed=139)
        source = DilutionSource.from_description(generator.describe(), data_id="actual-D-multi-group")
        d = Basis(values, source)
        component = ElementalComponent.path((0, 1, 3, 4))
        table = [dict(component=component, path=(0, 1, 3, 4), coefficient=0.31 - 0.4j)]
        weighted = self.producer(d, True, blocks=("d2d",), table=table)
        self.assertEqual(weighted.dilution_weight_scheme, "component_inclusion-v1")
        self.assertEqual(self.producer(d, blocks=("d2d",), table=table).dilution_weight_scheme,
                         "component_inclusion-v1")
        # This oracle uses scientific component labels, not production's grouping helper.
        mask = np.empty((5, 5), float)
        counts = ((3, 2), (4, 3))
        for left, label_left in enumerate(source.vector_labels):
            for right, label_right in enumerate(source.vector_labels):
                gl, gr = label_left[0], label_right[0]
                nl, ml = counts[gl]
                nr, mr = counts[gr]
                if gl != gr:
                    mask[left, right] = (nl / ml) * (nr / mr)
                elif label_left == label_right:
                    mask[left, right] = nl / ml
                else:
                    mask[left, right] = nl * (nl - 1) / (ml * (ml - 1))
        for time in range(self.sizes[3]):
            actual = weighted.calc_path_blocks(time)["d2d"][0]
            flat = values[time].reshape(5, -1).T
            for mi, momentum in enumerate(self.momenta):
                raw = (0.31 - 0.4j) * flat.conj().T @ reference_matrix(
                    self.gauge, (0, 1, 3, 4), time, momentum) @ flat
                np.testing.assert_allclose(actual[mi], raw * mask, atol=3e-14)
                self.assertGreater(np.linalg.norm(raw - np.diag(np.diag(raw))), 1e-3)

    def test_rank_one_needs_structural_diagonal_and_legacy_same_pool_is_rejected(self):
        d, _ = self.noise()
        diagonal = self.producer(d, True, blocks=("d2d",), table=self.table[:1], momenta=((0, 0, 0),))
        np.testing.assert_allclose(diagonal.calc_path_blocks(0)["d2d"][0, 0],
            self.coefficient * np.diag((2.0, 1.5, 1.5)), atol=3e-14)
        for options in (dict(momenta=((1, 0, 0),), table=self.table[:1]),
                        dict(momenta=((0, 0, 0),), table=[dict(component=ElementalComponent.path((0,)), path=(0,), coefficient=1)])):
            with self.subTest(options=options):
                producer = self.producer(d, True, blocks=("d2d",), **options)
                with self.assertRaisesRegex(ValueError, "rank-one.*proven complete diagonal"):
                    producer.calc_path_blocks(0)
        unknown_orthogonal = Basis(d.values, d.dilution_source)
        unknown_orthogonal.basis_orthogonal = False
        producer = self.producer(unknown_orthogonal, True, blocks=("d2d",), table=self.table[:1], momenta=((0, 0, 0),))
        with self.assertRaisesRegex(ValueError, "rank-one"):
            producer.calc_path_blocks(0)
        legacy, _ = self.noise("legacy")
        producer = self.producer(legacy, True, blocks=("d2d",), table=self.table[:1], momenta=((0, 0, 0),))
        with self.assertRaisesRegex(ValueError, "legacy u1 same-pool"):
            producer.calc_path_blocks(0)

    def test_temporal_endpoints_wrap_to_actual_pools_and_exact_groups_need_no_pool(self):
        d, _ = self.noise(dilution=((5,), (3,)))
        winding = ElementalComponent.path((6, 6, 6))
        table = [self.table[1], dict(component=winding, path=(6, 6, 6), coefficient=1)]
        raw = self.producer(d, blocks=("d2d",), table=table)
        weighted = self.producer(d, True, blocks=("d2d",), table=table)
        for time in range(self.sizes[3]):
            a, b = raw.calc_path_blocks(time)["d2d"], weighted.calc_path_blocks(time)["d2d"]
            mask = np.full((3, 3), 5 * 4 / (3 * 2))
            np.fill_diagonal(mask, 5 / 3)
            np.testing.assert_allclose(b[0], a[0] * (5 / 3) ** 2, atol=3e-14)
            np.testing.assert_allclose(b[1], a[1] * mask, atol=3e-14)
        exact, _ = self.noise(dilution=((5,), (5,)))
        from dataclasses import replace
        exact = Basis(exact.values, replace(exact.dilution_source, time_pools=()))
        raw = self.producer(exact, blocks=("d2d",))
        weighted = self.producer(exact, True, blocks=("d2d",))
        np.testing.assert_array_equal(weighted.calc_path_blocks(0)["d2d"], raw.calc_path_blocks(0)["d2d"])
        missing_pool = Basis(d.values, replace(d.dilution_source, time_pools=()))
        producer = self.producer(missing_pool, True, blocks=("d2d",))
        with self.assertRaisesRegex(ValueError, "pool identities"):
            producer.calc_path_blocks(0)

    def test_temporal_dd_shared_column_selection_uses_joint_pair_weights(self):
        from dataclasses import replace

        d, _ = self.noise(dilution=((5,), (3,)))
        source = replace(d.dilution_source, data_id="shared-time-column-selection",
                         time_pools=tuple((time, "common-selected-columns")
                                          for time in range(self.sizes[3])))
        d = Basis(d.values, source)
        table = [self.table[1]]  # A spatial and temporal path, with general support.
        raw = self.producer(d, blocks=("d2d",), table=table)
        weighted = self.producer(d, True, blocks=("d2d",), table=table)
        pair_weights = np.full((3, 3), 5 * 4 / (3 * 2))
        np.fill_diagonal(pair_weights, 5 / 3)
        for time in range(self.sizes[3]):
            with self.subTest(time=time):
                a = raw.calc_path_blocks(time)["d2d"].copy()
                b = weighted.calc_path_blocks(time)["d2d"].copy()
                np.testing.assert_allclose(b, a * pair_weights, atol=3e-14)

    def test_registration_retains_logical_sectors_and_d_endpoint_provenance(self):
        d, _ = self.noise()
        lx, ly, lz, lt = self.sizes
        configuration = dict(ensemble="bounded", configuration="cfg", lattice=list(self.sizes))
        def spec(kind, axes, conventions, dtype="complex128"):
            return DatasetSpec(dict(schema=1, kind=kind, configuration=configuration,
                axes=[dict(name=name, coordinates=list(values)) for name, values in axes],
                dtype=dtype, bases={}, parents=[], definition={}, conventions=conventions))
        gauge = spec("gauge", (("time", range(lt)), ("z", range(lz)), ("y", range(ly)),
            ("x", range(lx)), ("direction", range(4)), ("color_row", range(3)), ("color_column", range(3))), {})
        def vector(count, conventions):
            return spec("explicit_basis", (("time", range(lt)), ("mode", range(count)),
                ("z", range(lz)), ("y", range(ly)), ("x", range(lx)), ("color", range(3))), conventions)
        v_spec = vector(2, dict(normalization="orthonormal"))
        d_spec = vector(3, dict(basis_orthogonal=True, dilution_source=d.dilution_source.document))
        def compact(basis, time):
            document = basis.document
            document["axes"][0]["coordinates"] = [time]
            return DatasetSpec(bind_time_selections(document, {"time": TimeSelectionRecord(lt, (time,))}))
        temporal_document = elemental_dataset_document(gauge, compact(v_spec, 0), gauge_id="actual-gauge",
            basis_id=self.vectors.basis_id, product="v2d", right_basis=compact(d_spec, 1),
            right_basis_id=d.basis_id, components=[dict(path=[6])], momenta=self.momenta, modes=range(2),
            right_modes=range(3), phase_convention="midpoint", selection=TimeSelectionRecord(lt, (0,)))
        self.assertEqual(temporal_document["axes"][0]["coordinates"], [0])
        points = spec("spatial_points", (("time", range(lt)), ("point", range(3)),
            ("xyz", ("x", "y", "z"))), dict(representation="coordinate_delta", normalization="unit_delta"), "int64")
        components = [dict(path=[])]
        producer = self.producer(d, True, table=self.table[:1], momenta=((0, 0, 0),))
        for sector in ("v2d", "d2v", "d2d", "d2p", "p2d"):
            with self.subTest(sector=sector):
                if "p" not in sector:
                    left_spec = d_spec if sector[0] == "d" else v_spec
                    left_id = d.basis_id if sector[0] == "d" else self.vectors.basis_id
                    right_spec = d_spec if sector[-1] == "d" else v_spec
                    right_id = d.basis_id if sector[-1] == "d" else self.vectors.basis_id
                    document = elemental_dataset_document(gauge, left_spec, gauge_id="actual-gauge", basis_id=left_id,
                        components=components, momenta=((0, 0, 0),), modes=left_spec.coordinates("mode"),
                        right_basis=right_spec, right_basis_id=right_id, right_modes=right_spec.coordinates("mode"),
                        phase_convention="midpoint", product=sector, elemental_weighted=True)
                else:
                    document = point_elemental_dataset_document(gauge, d_spec, points, product=sector,
                        gauge_id="actual-gauge", basis_id=d.basis_id, points_id="actual-P", components=components,
                        momenta=((0, 0, 0),), modes=range(3), point_labels=range(3), phase_convention="midpoint",
                        elemental_weighted=True)
                self.assertEqual(document["definition"]["logical_sector"], sector)
                expected_sides = {side for side, kind in (("bar", sector[0]), ("field", sector[-1])) if kind == "d"}
                self.assertEqual(set(document["conventions"]["dilution_endpoints"]), expected_sides)
                with TemporaryDirectory() as directory:
                    with DatasetWriter(Path(directory), DatasetSpec(document), provenance={"test": "nine sectors"}) as writer:
                        for time in range(lt):
                            writer.write(time, producer.calc_path_blocks(time)[sector][:1].copy())
                        writer.publish()
                    with DatasetReader(Path(directory), cfg="cfg") as stored:
                        reader = DatasetElemental(stored)
                        self.assertEqual(tuple(reader.sectors), (sector,))
                        component = reader.components[0]
                        np.testing.assert_allclose(reader.block(component, 0, 0, sector),
                            producer.calc_path_blocks(0)[sector][0, 0], atol=3e-14)


if __name__ == "__main__":
    unittest.main()
