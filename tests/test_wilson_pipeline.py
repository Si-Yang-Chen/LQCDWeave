"""Independent Wilson stencil, free dispersion and complete trace checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.fixtures.laplace_reference import dense_laplacian, random_gauge, scipy
from lqcdweave.adapters.dense_wilson import DenseWilsonBackend, wilson_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows
from lqcdweave.production.wilson_reference import plan_wilson_reference
from fixtures.c_group_production import (
    publish_basis,
    publish_elemental_products,
    publish_perambulators,
)


def stencil(links, field, mass, boundary):
    """Independent array shifts and spin/color contractions, no dense assembly."""
    result = np.asarray((mass + 4) * field, dtype=complex)
    for mu, axis in enumerate((3, 2, 1, 0)):
        ahead = np.roll(field, -1, axis=axis)
        behind = np.roll(field, 1, axis=axis)
        if mu == 3:
            ahead[-1] *= boundary
            behind[0] *= boundary
        forward = np.einsum("...ab,...sb->...sa", links[..., mu, :, :], ahead)
        back_link = np.roll(links[..., mu, :, :], 1, axis=axis).conj().swapaxes(-1, -2)
        backward = np.einsum("...ab,...sb->...sa", back_link, behind)
        result -= .5 * (
            forward - np.einsum("sr,...rc->...sc", gamma(1 << mu), forward)
            + backward + np.einsum("sr,...rc->...sc", gamma(1 << mu), backward)
        )
    return result


class _WilsonFixture(unittest.TestCase):
    def _registered_gauge(self, root, seed=94, *, cfg="cfg"):
        sizes = (2, 1, 1, 3)
        links = random_gauge(sizes, seed=seed)
        axes = [dict(name=name, coordinates=list(range(size))) for name, size in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"),
            links.shape,
            strict=True,
        )]
        spec = DatasetSpec(dict(
            schema=1,
            kind="gauge",
            configuration=dict(ensemble="test", configuration=cfg, lattice=list(sizes)),
            axes=axes,
            dtype="complex128",
            bases={},
            parents=[],
            definition={},
            conventions=dict(boundary="periodic"),
        ))
        directory = root / f"gauge-{seed}"
        with DatasetWriter(directory, spec, provenance={"test": "independent"}) as writer:
            for time in writer.missing:
                writer.write(time, links[time])
            writer.publish()
        return directory


class WilsonTests(_WilsonFixture):
    def test_reference_preview_selects_cfg_from_shared_gauge_registration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gauge = self._registered_gauge(root, cfg="2000")
            self._registered_gauge(root, cfg="2004")
            output = root / "production"
            with self.assertRaises(KeyError):
                plan_wilson_reference(gauge, output)
            for cfg in ("2000", "2004"):
                plan = plan_wilson_reference(gauge, output, configuration=cfg)
                self.assertEqual(plan["stages"]["basis"]["action"], "compute")
                self.assertEqual(plan["stages"]["correlator"]["action"], "blocked")
            self.assertFalse(output.exists())

    @unittest.skipIf(scipy is None, "SciPy required for Laplace production")
    def test_full_wilson_products_and_meson_match_independent_stencil_trace(self):
        sizes = (2, 1, 1, 3)
        links = random_gauge(sizes, seed=94)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gauge_directory = self._registered_gauge(root, seed=94)
            index = DatasetIndex(root / "index")
            gauge_id = index.register(gauge_directory)
            ids = publish_basis(index, root, gauge_id, output_name="production", Ne=2)
            vsv_ids = publish_perambulators(
                index, root, gauge_id, output_name="production", basis_id=ids["basis"],
                products=("VSV",), Ne=2, mass=.4, t_boundary=-1,
            )
            elemental_ids = publish_elemental_products(
                index, root, gauge_id, ids["basis"], output_name="production",
                sectors=("v2v",), paths=((),), momentum_list=((1, 0, 0),),
                usedNe=2, phase_convention="anchor",
            )
            sink_operator = Operator.from_terms("sink-pseudoscalar", [
                OperatorTerm(15, ElementalComponent.path(GaugePath()), 0),
            ])
            source_operator = Operator.from_terms("source-pseudoscalar", [
                OperatorTerm(15, ElementalComponent.path(GaugePath()), 0),
            ])
            with (index.open(vsv_ids["vsv"]) as forward,
                  index.open(vsv_ids["vsv"]) as reverse,
                  index.open(elemental_ids["v2v"]) as sink,
                  index.open(elemental_ids["v2v"]) as source):
                inputs = (forward, reverse, sink, source)
                measurement_spec, _, _, times = prepare_meson_stage(
                    inputs, sink_operator, source_operator,
                )
                with DatasetWriter(root / "measurement", measurement_spec,
                                    provenance={"test": "independent-wilson-trace"}) as writer:
                    write_meson_rows(inputs, sink_operator, source_operator, writer)
                    measurement_id = writer.publish()
                index.register(root / "measurement")

            shape = (3, 1, 1, 2, 4, 3)
            dimension = int(np.prod(shape))
            matrix = np.column_stack([
                stencil(links, np.eye(dimension)[:, column].reshape(shape), .4, -1).ravel()
                for column in range(dimension)
            ])
            inverse = np.linalg.inv(matrix)
            embeddings, sinks, sources = [], [], []
            with index.open(ids["basis"]) as basis, index.open(ids["eigenvalues"]) as values:
                self.assertEqual(basis.spec.document["parents"], [gauge_id])
                self.assertEqual(values.spec.document["bases"]["mode"]["data_id"],
                                 ids["basis"])
                for time in times:
                    vectors = basis.read(time=time)
                    small = vectors.reshape(2, -1).T
                    laplacian = dense_laplacian(
                        links[time, :, :, :, :3].transpose(3, 0, 1, 2, 4, 5)
                    )
                    eigenvalues = values.read(time=time)
                    np.testing.assert_allclose(
                        eigenvalues, np.linalg.eigvalsh(laplacian)[:2], atol=1e-10,
                    )
                    np.testing.assert_allclose(laplacian @ small,
                                               small * eigenvalues, atol=1e-10)
                    np.testing.assert_allclose(small.conj().T @ small,
                                               np.eye(2), atol=1e-10)
                    embedding = np.zeros((dimension, 8), dtype=complex)
                    for spin in range(4):
                        for mode in range(2):
                            field = np.zeros(shape, dtype=complex)
                            field[time, ..., spin, :] = vectors[mode]
                            embedding[:, spin * 2 + mode] = field.ravel()
                    embeddings.append(embedding)
                    phase = np.repeat([1, -1], 3)
                    elemental = small.conj().T @ (phase[:, None] * small)
                    sinks.append(embedding @ np.kron(gamma(15), elemental)
                                 @ embedding.conj().T)
                    source_spin = gamma(8) @ gamma(15).conj().T @ gamma(8)
                    sources.append(embedding @ np.kron(source_spin, elemental.conj().T)
                                   @ embedding.conj().T)
            with index.open(vsv_ids["vsv"]) as vsv, index.open(measurement_id) as correlator:
                self.assertEqual(vsv.spec.coordinates("source_time"), [0, 1, 2])
                self.assertEqual(correlator.spec.coordinates("sink_time"), [0, 1, 2])
                for source_time in times:
                    for sink_time in times:
                        projected = embeddings[sink_time].conj().T @ inverse @ embeddings[source_time]
                        np.testing.assert_allclose(
                            vsv.read(source_time=source_time, sink_time=sink_time),
                            projected.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3),
                            atol=1e-11,
                        )
                        reference = -np.trace(
                            sinks[sink_time] @ inverse @ sources[source_time] @ inverse
                        )
                        np.testing.assert_allclose(
                            correlator.read(source_time=source_time, sink_time=sink_time),
                            reference,
                            atol=1e-10,
                        )
