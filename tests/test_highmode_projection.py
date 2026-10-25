"""Dense projector oracles for the one-pair high-mode block kernels."""

import unittest

import numpy as np

from lqcdweave.contraction.propagator_views import (
    project_psp_highmode,
    project_psv_highmode,
    project_vsp_highmode,
)


def _random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _point_embedding(site_indices, spatial_site_count=4, colors=3):
    result = np.zeros((spatial_site_count * colors, len(site_indices) * colors),
                      dtype=np.complex128)
    for point, site in enumerate(site_indices):
        for color in range(colors):
            result[site * colors + color, point * colors + color] = 1
    return result


def _fixture():
    rng = np.random.default_rng(719204)
    colors = 3
    spatial_sites = 4
    space_color = spatial_sites * colors
    nv_sink, nv_source = 2, 3
    sink_sites, source_sites = (0, 2), (3, 1, 0, 2)
    es = _point_embedding(sink_sites, spatial_sites, colors)
    er = _point_embedding(source_sites, spatial_sites, colors)
    vs, _ = np.linalg.qr(_random_complex(rng, (space_color, nv_sink)))
    vr, _ = np.linalg.qr(_random_complex(rng, (space_color, nv_source)))
    dense = _random_complex(rng, (4, 4, space_color, space_color))

    vsv = np.empty((4, 4, nv_sink, nv_source), dtype=np.complex128)
    vsp = np.empty((4, 4, nv_sink, len(source_sites), colors), dtype=np.complex128)
    psv = np.empty((4, 4, len(sink_sites), colors, nv_source), dtype=np.complex128)
    psp = np.empty((4, 4, len(sink_sites), colors, len(source_sites), colors),
                   dtype=np.complex128)
    expected_vsp = np.empty_like(vsp)
    expected_psv = np.empty_like(psv)
    expected_psp = np.empty_like(psp)
    qs = np.eye(space_color) - vs @ vs.conj().T
    qr = np.eye(space_color) - vr @ vr.conj().T

    # M[i,p,c] = xi_i(x_p,c): the point basis is a real coordinate unit column.
    sink_overlap = (vs.T @ es).reshape(nv_sink, len(sink_sites), colors)
    source_overlap = (vr.T @ er).reshape(nv_source, len(source_sites), colors)
    for spin_sink in range(4):
        for spin_source in range(4):
            block = dense[spin_sink, spin_source]
            vsv[spin_sink, spin_source] = vs.conj().T @ block @ vr
            vsp[spin_sink, spin_source] = (
                vs.conj().T @ block @ er
            ).reshape(nv_sink, len(source_sites), colors)
            psv[spin_sink, spin_source] = (
                es.conj().T @ block @ vr
            ).reshape(len(sink_sites), colors, nv_source)
            psp[spin_sink, spin_source] = (
                es.conj().T @ block @ er
            ).reshape(len(sink_sites), colors, len(source_sites), colors)

            # Independent full-space Q_s S Q_r, Q_s S, and S Q_r oracles.
            expected_vsp[spin_sink, spin_source] = (
                vs.conj().T @ block @ qr @ er
            ).reshape(nv_sink, len(source_sites), colors)
            expected_psv[spin_sink, spin_source] = (
                es.conj().T @ qs @ block @ vr
            ).reshape(len(sink_sites), colors, nv_source)
            expected_psp[spin_sink, spin_source] = (
                es.conj().T @ qs @ block @ qr @ er
            ).reshape(len(sink_sites), colors, len(source_sites), colors)

    return dict(
        dense=dense, vs=vs, vr=vr, es=es, er=er, qs=qs, qr=qr,
        vsv=vsv, vsp=vsp, psv=psv, psp=psp,
        sink_overlap=sink_overlap, source_overlap=source_overlap,
        expected_vsp=expected_vsp, expected_psv=expected_psv,
        expected_psp=expected_psp,
    )


def _case_calls(values, *, array_module=np):
    return {
        "vsp": project_vsp_highmode(
            values["vsv"], values["vsp"], values["source_overlap"],
            array_module=array_module,
        ),
        "psv": project_psv_highmode(
            values["vsv"], values["psv"], values["sink_overlap"],
            array_module=array_module,
        ),
        "psp": project_psp_highmode(
            values["vsv"], values["psv"], values["vsp"], values["psp"],
            values["sink_overlap"], values["source_overlap"],
            array_module=array_module,
        ),
    }


def _raw_inputs(fixture, dtype_by_name=None):
    names = ("vsv", "vsp", "psv", "psp", "sink_overlap", "source_overlap")
    dtype_by_name = {} if dtype_by_name is None else dtype_by_name
    return {
        name: np.asarray(fixture[name], dtype=dtype_by_name.get(name, np.complex128)).copy()
        for name in names
    }


class HighmodeProjectionTests(unittest.TestCase):
    def test_all_three_sectors_match_full_space_projector_oracle(self):
        fixture = _fixture()
        expected = {
            "vsp": fixture["expected_vsp"],
            "psv": fixture["expected_psv"],
            "psp": fixture["expected_psp"],
        }
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                actual = _case_calls(_raw_inputs(fixture, {
                    name: dtype for name in (
                        "vsv", "vsp", "psv", "psp", "sink_overlap", "source_overlap",
                    )
                }))
                tolerance = 3e-5 if dtype is np.complex64 else 2e-12
                for sector in ("vsp", "psv", "psp"):
                    self.assertEqual(actual[sector].dtype, np.dtype(dtype))
                    self.assertEqual(actual[sector].shape, expected[sector].shape)
                    np.testing.assert_allclose(
                        actual[sector], expected[sector], rtol=tolerance, atol=tolerance,
                    )
                    self.assertGreater(float(np.max(np.abs(actual[sector].imag))), 1e-5)

        # This complex fixture distinguishes the required source conjugation
        # and PSP cross term from the plausible but incorrect alternatives.
        wrong_qr = np.eye(fixture["vr"].shape[0]) - (
            fixture["vr"].conj() @ fixture["vr"].T
        )
        wrong_vsp = np.empty_like(fixture["expected_vsp"])
        missing_cross_psp = np.empty_like(fixture["expected_psp"])
        for spin_sink in range(4):
            for spin_source in range(4):
                block = fixture["dense"][spin_sink, spin_source]
                wrong_vsp[spin_sink, spin_source] = (
                    fixture["vs"].conj().T @ block @ wrong_qr @ fixture["er"]
                ).reshape(fixture["expected_vsp"].shape[2:])
                missing_cross_psp[spin_sink, spin_source] = (
                    fixture["es"].conj().T @ block @ fixture["er"]
                    - fixture["es"].conj().T @ fixture["vs"]
                    @ fixture["vs"].conj().T @ block @ fixture["er"]
                    - fixture["es"].conj().T @ block @ fixture["vr"]
                    @ fixture["vr"].conj().T @ fixture["er"]
                ).reshape(fixture["expected_psp"].shape[2:])
        self.assertGreater(float(np.max(np.abs(wrong_vsp - expected["vsp"]))), 1e-2)
        self.assertGreater(
            float(np.max(np.abs(missing_cross_psp - expected["psp"]))), 1e-2,
        )

    def test_mixed_complex_precision_promotes_all_math_to_result_type(self):
        fixture = _fixture()
        mixed = _raw_inputs(fixture, {
            "vsv": np.complex64, "vsp": np.complex128,
            "psv": np.complex64, "psp": np.complex128,
            "sink_overlap": np.complex64, "source_overlap": np.complex128,
        })
        actual = _case_calls(mixed)
        expected = {
            "vsp": fixture["expected_vsp"],
            "psv": fixture["expected_psv"],
            "psp": fixture["expected_psp"],
        }
        inputs_per_sector = {
            "vsp": ("vsv", "vsp", "source_overlap"),
            "psv": ("vsv", "psv", "sink_overlap"),
            "psp": (
                "vsv", "psv", "vsp", "psp", "sink_overlap", "source_overlap",
            ),
        }
        for sector, result in actual.items():
            self.assertEqual(result.dtype, np.dtype(np.result_type(*(
                mixed[name].dtype for name in inputs_per_sector[sector]
            ))))
            np.testing.assert_allclose(result, expected[sector], rtol=3e-6, atol=3e-6)

    def test_results_are_owned_and_inputs_are_unchanged(self):
        fixture = _fixture()
        values = _raw_inputs(fixture)
        before = {name: value.copy() for name, value in values.items()}
        results = _case_calls(values)
        for result in results.values():
            for value in values.values():
                self.assertFalse(np.shares_memory(result, value))
        for name in values:
            np.testing.assert_array_equal(values[name], before[name])
        results["psp"][0, 0, 0, 0, 0, 0] += 17
        for name in values:
            np.testing.assert_array_equal(values[name], before[name])

    def test_rejects_invalid_ranks_shapes_dtypes_and_nonfinite_inputs(self):
        fixture = _fixture()
        values = _raw_inputs(fixture)
        with self.assertRaisesRegex(ValueError, "rank"):
            project_vsp_highmode(values["vsv"][0], values["vsp"],
                                 values["source_overlap"])
        with self.assertRaisesRegex(ValueError, "spin axes"):
            project_psv_highmode(values["vsv"][:, :3], values["psv"],
                                 values["sink_overlap"])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            project_psp_highmode(
                values["vsv"], values["psv"], values["vsp"],
                values["psp"], values["sink_overlap"],
                values["source_overlap"][:, :-1],
            )
        for bad in (values["vsv"].real, values["vsv"].astype(object),
                    np.zeros(values["vsv"].shape, dtype=bool)):
            with self.subTest(dtype=bad.dtype):
                with self.assertRaisesRegex(TypeError, "complex64 or complex128"):
                    project_vsp_highmode(bad, values["vsp"], values["source_overlap"])

        bad_source = values["source_overlap"].copy()
        bad_source[0, 0, 0] = np.nan + 0j
        with self.assertRaisesRegex(ValueError, "finite"):
            project_vsp_highmode(values["vsv"], values["vsp"], bad_source)
        bad_sink = values["sink_overlap"].copy()
        bad_sink[0, 0, 0] = np.inf + 0j
        with self.assertRaisesRegex(ValueError, "finite"):
            project_psv_highmode(values["vsv"], values["psv"], bad_sink)
        bad_psp = values["psp"].copy()
        bad_psp[0, 0, 0, 0, 0, 0] = np.nan + 0j
        with self.assertRaisesRegex(ValueError, "finite"):
            project_psp_highmode(
                values["vsv"], values["psv"], values["vsp"], bad_psp,
                values["sink_overlap"], values["source_overlap"],
            )


if __name__ == "__main__":
    unittest.main()
