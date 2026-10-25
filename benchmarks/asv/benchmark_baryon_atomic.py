"""Complete fixed-path baryon actions, including their owned snapshot."""

from types import SimpleNamespace

import numpy as np
import opt_einsum

from lqcdweave.contraction.execution.operations.baryon_join import contract_baryon_join


def _timed(phase, function, **kwargs):
    return function()


class BaryonAtomicBenchmarks:
    """Synthetic dense-V endpoints; no physical ensemble or I/O is timed."""

    params = ([64], ["complex64", "complex128"])
    param_names = ["modes", "dtype"]
    repeat = 3
    rounds = 3
    number = 1

    def setup(self, modes, dtype):
        rng = np.random.default_rng(300513)
        shapes = ((4, 4, 4), (modes, modes, modes)) + ((4, 4, modes, modes),) * 3
        operands = [
            (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
            for shape in shapes
        ]
        equation = "ijk,abc,ilad,jmbe,kncf->ldmenf"
        path, _ = opt_einsum.contract_path(equation, *shapes, shapes=True, optimize="greedy")
        keys = tuple("sepqr")
        self.arrays = dict(zip(keys, operands))
        self.action = SimpleNamespace(
            reused=False, equation=equation, operands=keys, tensor="result",
            kernel_path=tuple(tuple(step) for step in path),
        )
        # Warm the structural cache before measuring steady repeated actions.
        contract_baryon_join(self.action, self.arrays, np, timed=_timed)
        spin, elemental, p, q, r = operands
        indices = (
            (0, 0, 0, 0, 0, 0),
            (3, modes - 1, 2, modes // 2, 1, modes - 1),
            (1, modes // 3, 3, modes - 1, 2, modes // 4),
            (2, modes // 2, 1, modes // 3, 3, modes // 2),
        )
        reference = [
            np.einsum(
                "ijk,abc,ia,jb,kc->", spin, elemental,
                p[:, l, :, d], q[:, m, :, e], r[:, n, :, f], optimize=True,
            )
            for l, d, m, e, n, f in indices
        ]
        np.testing.assert_allclose(
            [self.arrays["result"][index] for index in indices], reference,
            rtol=3e-4 if dtype == "complex64" else 2e-11,
            atol=3e-2 if dtype == "complex64" else 2e-9,
        )

    def time_fixed_path_action_and_snapshot(self, modes, dtype):
        contract_baryon_join(self.action, self.arrays, np, timed=_timed)
