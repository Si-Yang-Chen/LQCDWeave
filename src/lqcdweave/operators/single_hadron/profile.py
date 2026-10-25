"""Mode-space profiles applied to previously computed spatial blocks."""

import numpy as np


class SpectralProfile:
    def __init__(self, left, right=None, coefficients=None, *, cfg, name="profile"):
        left = np.array(left, dtype=complex, copy=True)
        if left.ndim == 2:
            left = left[None]
        right = np.array(left if right is None else right, dtype=complex, copy=True)
        if right.ndim == 2:
            right = right[None]
        if left.ndim != 3 or right.ndim != 3 or left.shape[:2] != right.shape[:2] or min(*left.shape, *right.shape) < 1:
            raise ValueError("profile arrays require (basis,time,mode) with matching basis/time")
        coeff = np.asarray(np.ones(len(left)) if coefficients is None else coefficients, dtype=complex)
        if coeff.shape != (len(left),) or not all(np.isfinite(v).all() for v in (left, right, coeff)):
            raise ValueError("finite weights and one coefficient per basis are required")
        self.left, self.right, self.coefficients = (
            np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape) for v in (left, right, coeff))
        self.cfg, self.name = str(cfg), str(name)

    @classmethod
    def gaussian(cls, eigenvalues, sigma, *, cfg, name="gaussian"):
        values = np.asarray(eigenvalues)
        if values.ndim != 2 or not np.isrealobj(values) or not np.isfinite(values).all() or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("real finite (time,mode) eigenvalues and positive sigma are required")
        return cls(np.exp(-0.5 * (values / sigma) ** 2), cfg=cfg, name=name)

    def apply_block(self, elemental, *, times, cfg, array_module=np):
        xp = array_module
        if str(cfg) != self.cfg:
            raise ValueError("profile cfg differs from elemental cfg")
        a, b = times
        if (elemental.ndim != 2 or not 0 <= a < self.left.shape[1] or not 0 <= b < self.right.shape[1]
                or elemental.shape[0] > self.left.shape[2] or elemental.shape[1] > self.right.shape[2]):
            raise ValueError("profile does not cover the requested block")
        out = xp.zeros_like(elemental, dtype=complex)
        for c, left, right in zip(self.coefficients, self.left, self.right):
            out += c * xp.asarray(left[a, :elemental.shape[0]]).conj()[:, None] * elemental * xp.asarray(right[b, :elemental.shape[1]])[None, :]
        return out
