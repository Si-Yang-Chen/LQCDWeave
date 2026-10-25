"""Phase/QR dilution from localized_blending's lattice/generator/noisevector.py.

Preserves raw amplitudes and QR draw order, with an explicit legacy phase
option. Uses a local RNG rather than reseeding the process. No Dirac solver,
estimator weights or file IO lives here.
"""

from copy import deepcopy
from numbers import Integral
from uuid import uuid4

import numpy as np

from lqcdweave.contracts.time_selection import TimeSelection, TimeSelectionRecord


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class NoisevectorGenerator:
    """Generate owned complex128 arrays shaped (Nt,Nd,Lz,Ly,Lx,3).

    ``dilution=(parent_counts, vector_counts)`` defines ordered groups.
    A scalar vector count is repeated for all groups. By default parents are
    consecutive, starting at mode_offset; explicit mode_groups supports
    non-contiguous importance-balanced groups without choosing their design.
    The input handle supplies load(key), just as an existing vector handle does.
    Default mixing="qr" also uses QR when m=1<n. The "legacy" option uses
    U(1) phases for that case, preserving the old covariance convention.
    """

    def __init__(self, eigenvector, dilution, *, array_module=np,
                 mode_offset=0, mode_groups=None, mixing="qr"):
        if not callable(getattr(eigenvector, "load", None)):
            raise ValueError("eigenvector must provide load(key)")
        if getattr(array_module, "__name__", None) not in ("numpy", "cupy"):
            raise ValueError("array_module must be numpy or cupy")
        if mixing not in ("qr", "legacy"):
            raise ValueError("mixing must be 'qr' or 'legacy'")
        if len(dilution) != 2:
            raise ValueError("dilution must be (parent_counts, vector_counts)")
        counts = tuple(_integer(n, "parent count", 1) for n in dilution[0])
        if not counts:
            raise ValueError("dilution must contain at least one group")
        used = dilution[1]
        if isinstance(used, Integral):
            used = (used,) * len(counts)
        used = tuple(_integer(m, "vector count", 1) for m in used)
        if len(counts) != len(used) or any(m > n for n, m in zip(counts, used)):
            raise ValueError("each group requires 1 <= vector count <= parent count")
        offset = _integer(mode_offset, "mode_offset")
        if mode_groups is None:
            groups = []
            for n in counts:
                groups.append(tuple(range(offset, offset + n)))
                offset += n
        else:
            if offset:
                raise ValueError("mode_offset cannot be combined with mode_groups")
            groups = [tuple(_integer(i, "mode index") for i in group) for group in mode_groups]
            if len(groups) != len(counts) or any(len(g) != n for g, n in zip(groups, counts)):
                raise ValueError("mode_groups must match parent_counts")
        flattened = [i for group in groups for i in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("dilution groups must have disjoint parent indices")
        self.eigenvector, self.xp = eigenvector, array_module
        self.mixing = mixing
        self.eigv_list, self.noisev_list = counts, used
        self.mode_groups = tuple(groups)
        self.Nd = sum(used)
        self.vector_labels = tuple((block, vector) for block, m in enumerate(used)
                                   for vector in range(m))
        self._eigenvector_data = None
        self._key = self._seed = None
        self._basis_id = self._time_selection = None
        self._parent_mode_groups = None
        self._complete = False
        self.basis_orthogonal = False
        self._time_pools = ()

    def load(self, key):
        """Snapshot the actual ordered basis; a failed load invalidates state."""
        self._eigenvector_data = None
        self._key = self._seed = None
        self._basis_id = self._time_selection = None
        self._parent_mode_groups = None
        self._complete = False
        self.basis_orthogonal = False
        self._time_pools = ()
        payload = self.eigenvector.load(key)
        raw = self.xp.asarray(payload[:])
        if raw.ndim != 6 or raw.shape[-1] != 3 or any(n < 1 for n in raw.shape):
            raise ValueError("basis must have shape (Nt,Ne,Lz,Ly,Lx,3)")
        if raw.dtype.kind != "c":
            raise ValueError("basis must have a complex dtype")
        if max(max(group) for group in self.mode_groups) >= raw.shape[1]:
            raise ValueError("dilution parent indices exceed the loaded basis")
        labels = tuple(_integer(label, "parent mode label") for label in
                       getattr(self.eigenvector, "mode_labels", range(raw.shape[1])))
        if len(labels) != raw.shape[1] or len(set(labels)) != len(labels):
            raise ValueError("parent mode labels must cover the actual ordered vector axis")
        basis_id = deepcopy(getattr(self.eigenvector, "basis_id", None))
        selection = deepcopy(getattr(self.eigenvector, "time_selection", None))
        if selection is not None:
            if not isinstance(selection, (TimeSelection, TimeSelectionRecord)):
                raise TypeError("time_selection must be a TimeSelection or TimeSelectionRecord")
            if selection.Nt != raw.shape[0]:
                raise ValueError("basis time extent differs from declared time_selection")
        owned = raw.astype(self.xp.complex128, copy=True)
        self._basis_id, self._time_selection = basis_id, selection
        self._eigenvector_data, self._key = owned, key
        self._parent_mode_groups = tuple(tuple(labels[index] for index in group) for group in self.mode_groups)
        self.basis_orthogonal = getattr(self.eigenvector, "basis_orthogonal", False) is True
        return self

    def calc(self, seed=None):
        """Produce one realization without normalizing it or changing global RNG."""
        self._complete = False
        self._seed = None
        self._time_pools = ()
        if self._eigenvector_data is None:
            raise RuntimeError("load a configuration before generating noise vectors")
        if seed is not None:
            seed = _integer(seed, "seed")
        xp, basis = self.xp, self._eigenvector_data
        rng = xp.random.RandomState(seed)
        nt = basis.shape[0]
        result = xp.empty((nt, self.Nd, *basis.shape[2:]), dtype=xp.complex128)
        start = 0
        for group, n, m in zip(self.mode_groups, self.eigv_list, self.noisev_list):
            if group == tuple(range(group[0], group[0] + n)):
                parents = basis[:, group[0]:group[0] + n]
            else:
                parents = basis[:, xp.asarray(group)]
            if m == n:
                result[:, start:start + m] = parents
            elif m == 1 and self.mixing == "legacy":
                phases = xp.exp(1j * rng.uniform(0, 2 * xp.pi, (nt, n)))
                result[:, start] = xp.einsum("tizyxa,ti->tzyxa", parents, phases)
            else:
                for t in range(nt):
                    random = rng.randn(n, n) + 1j * rng.randn(n, n)
                    q, _ = xp.linalg.qr(random)
                    result[t, start:start + m] = xp.einsum("izyxa,ij->jzyxa", parents[t], q[:, :m])
            start += m
        times = tuple(range(nt)) if self._time_selection is None else self._time_selection.times
        self._time_pools = tuple((time, str(uuid4())) for time in times)
        self._seed, self._complete = seed, True
        return result

    def describe(self):
        """Return provenance hints, never a claimed persistent realization ID."""
        branches = tuple("exact" if m == n else "u1" if m == 1 and self.mixing == "legacy" else "qr"
                         for n, m in zip(self.eigv_list, self.noisev_list))
        return deepcopy({
            "representation": "D", "algorithm": f"{self.mixing}-dilution-v1",
            "key": self._key, "parent_basis_id": self._basis_id,
            "time_selection": self._time_selection,
            "time_pools": self._time_pools,
            "mode_groups": self.mode_groups if self._parent_mode_groups is None else self._parent_mode_groups,
            "parent_counts": self.eigv_list,
            "vector_counts": self.noisev_list, "vector_labels": self.vector_labels,
            "branches": branches, "normalization": "raw",
            "basis_orthogonal": self.basis_orthogonal,
            "covariance_factors": tuple(1.0 if branch != "qr" else m / n
                                        for branch, n, m in zip(branches, self.eigv_list, self.noisev_list)),
            "seed": self._seed, "complete": self._complete,
        })
