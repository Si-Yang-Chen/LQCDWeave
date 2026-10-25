"""Shared cubic coupling for derivative and supplied spatial insertion rows."""

from collections.abc import Mapping
from math import sqrt

from .operator import InsertionRow, InsertionTerm, OperatorTerm


class GammaName:
    A0, B0 = r"$a_0$", r"$b_0$"
    PI, PI_2 = r"$\pi$", r"$\pi(2)$"
    RHO, RHO_2 = r"$\rho$", r"$\rho(2)$"
    A1, B1 = r"$a_1$", r"$b_1$"


class DerivativeName:
    IDEN, NABLA = "", r"$\nabla$"
    B, D, E = r"$\mathbb{B}$", r"$\mathbb{D}$", r"$\mathbb{E}$"


class ProjectionName:
    A1, A2, E, T1, T2 = "A_1", "A_2", "E", "T_1", "T_2"
    G1, G2, H = "G_1", "G_2", "H"


# (components, cubic irrep, parity, charge conjugation, hermiticity)
_GAMMA = {
    GammaName.A0: ([0], "A_1", 1, 1, 1),
    GammaName.B0: ([8], "A_1", 1, -1, 1),
    GammaName.PI: ([15], "A_1", -1, 1, -1),
    GammaName.PI_2: ([7], "A_1", -1, 1, 1),
    GammaName.RHO: ([1, 2, 4], "T_1", -1, -1, -1),
    GammaName.RHO_2: ([9, 10, 12], "T_1", -1, -1, 1),
    GammaName.A1: ([14, 13, 11], "T_1", 1, 1, -1),
    GammaName.B1: ([6, 5, 3], "T_1", 1, -1, -1),
}
_DERIVATIVE = {
    DerivativeName.IDEN: ([[[1, 0]]], "A_1", 1, 1, 1),
    DerivativeName.NABLA: ([[[1, 1]], [[1, 2]], [[1, 3]]], "T_1", -1, -1, -1),
    DerivativeName.B: ([[[1, 11], [-1, 9]], [[1, 6], [-1, 10]], [[1, 7], [-1, 5]]], "T_1", 1, -1, -1),
    DerivativeName.D: ([[[1, 11], [1, 9]], [[1, 6], [1, 10]], [[1, 7], [1, 5]]], "T_2", 1, 1, 1),
    DerivativeName.E: ([[[1, 4], [-1, 8]], [[-1, 4], [-1, 8], [2, 12]]], "E", 1, 1, 1),
}


class Row(list):
    def simplify(self):
        merged = {}
        for g, terms in zip(self[::2], self[1::2]):
            for c, key in terms:
                merged[g, key] = merged.get((g, key), 0) + c
        groups = {}
        for (g, key), c in merged.items():
            if c != 0:
                groups.setdefault(g, []).append([c, key])
        self[:] = [v for g, entries in groups.items() for v in (g, entries)]
        return self

    def __add__(self, other):
        return Row(list(self) + list(other)).simplify()

    def __mul__(self, scalar):
        return Row([v for g, terms in zip(self[::2], self[1::2])
                    for v in (g, [[scalar * c, key] for c, key in terms])])

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __iadd__(self, other):
        self[:] = self + other
        return self

    def __imul__(self, scalar):
        self[:] = self * scalar
        return self


def couple_rows(gamma, spatial, left, right, projection):
    """One coupling implementation for every spatial coefficient dictionary."""
    dimensions = {"A_1": 1, "A_2": 1, "E": 2, "T_1": 3, "T_2": 3}
    allowed = {"A_1": ("T_1",), "A_2": ("T_2",), "E": ("T_1", "T_2"),
               "T_1": ("A_1", "T_1", "T_2", "E"), "T_2": ("A_2", "T_1", "T_2", "E")}
    if left not in ("A_1", "T_1") or right not in dimensions or len(spatial) != dimensions[right]:
        raise ValueError("invalid cubic coupling input")
    if len(gamma) != dimensions[left]:
        raise ValueError("gamma component count differs from irrep dimension")
    if (left == "A_1" and right != projection) or (left == "T_1" and projection not in allowed[right]):
        raise ValueError(f"{left} x {right} has no {projection}")

    def make(entries):
        result = Row()
        for gi, si, weight in entries:
            result.extend([gamma[gi], [[weight * c, key] for c, key in spatial[si]]])
        return result.simplify()

    rows = []
    for i in range(dimensions[projection]):
        if left == "A_1":
            entries = [(0, i, 1)]
        elif right in ("A_1", "A_2"):
            entries = [(i, 0, 1)]
        elif right == "E":
            weights = ([(1, -1/sqrt(3)), (-1, -1/sqrt(3)), (2/sqrt(3), 0)]
                       if projection == "T_1" else [(1, sqrt(3)), (1, -sqrt(3)), (-2, 0)])
            entries = [(i, j, w) for j, w in enumerate(weights[i]) if w]
        elif projection in ("A_1", "A_2"):
            entries = [(j, j, 1) for j in range(3)]
        elif projection == "E":
            weights = ((1, -1, 0), (-1/sqrt(3), -1/sqrt(3), 2/sqrt(3))) if right == "T_1" else (
                (1/sqrt(3), 1/sqrt(3), -2/sqrt(3)), (1, -1, 0))
            entries = [(j, j, w) for j, w in enumerate(weights[i]) if w]
        else:
            j, k = (i + 1) % 3, (i + 2) % 3
            entries = [(j, k, 1), (k, j, -1 if right == projection else 1)]
        rows.append(make(entries))
    return rows


class Insertion:
    def __init__(self, gamma, derivative, projection, momentum_dict, profile=None, *,
                 momentum_identity=False):
        self.momentum_identity = momentum_identity
        g, left, gp, gc, gh = _GAMMA[gamma]
        d, right, dp, dc, dh = _DERIVATIVE[derivative]
        self._initialize(g, d, left, right, projection, momentum_dict, profile, gp*dp, gc*dc, gh*dh)

    def _initialize(self, g, d, left, right, projection, momenta, profile, p, c, h):
        self.gamma, self.derivative = g, d
        self.projection = [left, right, projection]
        self.momentum_dict, self.profile = momenta, profile
        self.parity, self.charge_conjugation, self.hermiticity = p, c, h
        self.rows = couple_rows(g, d, left, right, projection)
        self.little_group_irreps_dict = {}
        self._generic_rows = False

    @classmethod
    def local_spin(cls, irrep, parity, momentum_dict, component=None):
        """Normalized local three-quark rows in the DR basis."""
        from lqcdweave.contracts.elemental import BaryonComponent
        from .baryon import baryon_local_spin_rows
        component = BaryonComponent.paths() if component is None else component
        if irrep == ProjectionName.G1:
            irrep = "G1"
        return cls.from_rows([[InsertionTerm(component, spin=row)]
                              for row in baryon_local_spin_rows(irrep, parity)], momentum_dict)

    @classmethod
    def from_irreps(cls, spin, spatial_rows, spatial_irrep, projection,
                    momentum_dict, *, copy_index=0, spatial_matrices=None):
        """Three-quark spin irrep times spatial irrep, with automatic row coupling.

        ``spatial_rows`` use (coefficient, BaryonComponent) terms. Their row
        basis is canonical lattice_symmetry unless ``spatial_matrices`` is
        supplied explicitly. Copy indices resolve repeated target irreps.
        """
        from .irrep_coupling import coupled_baryon_rows
        spatial_rows = tuple(tuple(row) for row in spatial_rows)
        rows, coefficients, multiplicity, parity, target = coupled_baryon_rows(
            spin, spatial_rows, spatial_irrep, projection,
            copy_index=copy_index, spatial_matrices=spatial_matrices)
        obj = cls.from_rows(rows, momentum_dict)
        obj.gamma, obj.derivative = spin, spatial_rows
        from .baryon_spin import BaryonSpinOperator
        obj.projection = [spin.irrep if isinstance(spin,BaryonSpinOperator) else spin[:-1],
                          spatial_irrep[:-1], target]
        obj.parity = parity
        obj.coupling_coefficients = coefficients
        obj.multiplicity, obj.copy_index = multiplicity, copy_index
        obj._rest_irrep = target
        return obj

    @classmethod
    def from_rows(cls, rows, momentum_dict, profile=None):
        """Build a rank-2 or rank-3 insertion from explicit ordered terms.

        A row is a sequence of ``InsertionTerm`` records.  Rank-3 rows use a
        ``BaryonComponent`` and rank-3 ``SparseSpinTensor`` in each record;
        callers provide their projected spin/spatial combination directly.
        """
        if not isinstance(momentum_dict, Mapping) or not momentum_dict:
            raise ValueError("a nonempty momentum dictionary is required")
        normalized = []
        for row_index, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                raise TypeError("insertion rows must be sequences of term records")
            try:
                row = tuple(row)
            except TypeError as exc:
                raise TypeError("insertion rows must be sequences of term records") from exc
            if not row:
                raise ValueError(f"insertion row {row_index} is empty")
            terms = []
            for term in row:
                if isinstance(term, InsertionTerm):
                    terms.append(term)
                elif isinstance(term, OperatorTerm):
                    terms.append(InsertionTerm(
                        term.component,
                        gamma=term.gamma,
                        coefficient=term.coefficient,
                        anchor_time_offset=term.anchor_time_offset,
                        profile=term.profile,
                        spin=term.spin,
                        endpoint_time_offsets=term.endpoint_time_offsets,
                        current_term=term.current_term,
                    ))
                else:
                    raise TypeError("insertion rows require InsertionTerm records")
            normalized.append(tuple(terms))
        if not normalized:
            raise ValueError("at least one insertion row is required")
        ranks = {3 if row[0].spin is not None else 2 for row in normalized}
        if len(ranks) != 1:
            raise ValueError("an Insertion cannot mix two-leg and three-leg rows")
        if any((3 if term.spin is not None else 2) not in ranks
               for row in normalized for term in row):
            raise ValueError("an Insertion row cannot mix two-leg and three-leg terms")
        if ranks == {3} and len({term.spin.basis for row in normalized for term in row}) != 1:
            raise ValueError("three-leg Insertion terms must use one spin basis")
        obj = cls.__new__(cls)
        obj.gamma, obj.derivative, obj.projection = (), (), None
        obj.momentum_dict, obj.profile = dict(momentum_dict), profile
        obj.parity = obj.charge_conjugation = obj.hermiticity = None
        obj.rows = tuple(normalized)
        obj.little_group_irreps_dict = {}
        obj._generic_rows = True
        return obj

    def __getitem__(self, index):
        return InsertionRow(self.rows[index], self.momentum_dict, self.profile,
                            momentum_identity=getattr(self, "momentum_identity", False))

    def project_exchange(self, target, *, operator_table=None, copy_index=0, slots=None):
        """Project the combined baryon spin x spatial rows before adding flavor.

        S/A return an Insertion; mixed returns its aligned MS/MA pair. Spatial
        elementals already include color epsilon. The returned operator_table
        contains the required path components for the shared Elemental input.
        """
        from .exchange_coupling import project_exchange
        return project_exchange(self,target,operator_table=operator_table,
                                copy_index=copy_index,slots=slots)

    def little_group_projection(self, momentum, irrep_name, idx=0):
        if getattr(self, "_generic_rows", False) and not hasattr(self, "_rest_irrep"):
            raise ValueError("generic rows must be projected before Insertion.from_rows")
        if tuple(momentum) == (0, 0, 0):
            if self._generic_rows:
                target = irrep_name.replace("_", "")
                expected = self._rest_irrep + ("g" if self.parity == 1 else "u")
                if target not in (self._rest_irrep, expected) or idx != 0:
                    raise ValueError("rest projection must retain the existing irrep and occurrence")
            return self
        try:
            from lqcdweave.symmetry import restrict_irrep
        except ModuleNotFoundError as exc:
            if exc.name not in {"sympy", "scipy", "lazy_object_proxy"}:
                raise
            raise ModuleNotFoundError(
                f"Missing optional dependency {exc.name!r}. Moving little-group projection "
                "requires Python 3.11+ and the [correlator] extra. "
                "Run python -m pip install -e '.[correlator]'. "
                "See docs/modules/correlator/correlator_frontend.md.",
                name=exc.name,
            ) from exc
        reduction = restrict_irrep(momentum, self.projection[-1], irrep_name, self.parity)
        if not 0 <= idx < reduction["multiplicity"]:
            raise IndexError("requested irrep occurrence is absent")
        if self._generic_rows:
            rows = []
            for basis in reduction["row_bases"]:
                terms = []
                for weight, original in zip(basis[:, idx], self.rows):
                    if weight != 0:
                        terms.extend(InsertionTerm(
                            term.component, gamma=term.gamma,
                            coefficient=complex(weight) * term.coefficient,
                            anchor_time_offset=term.anchor_time_offset, profile=term.profile,
                            spin=term.spin, endpoint_time_offsets=term.endpoint_time_offsets,
                            current_term=term.current_term)
                                     for term in original)
                rows.append(terms)
            obj = Insertion.from_rows(rows, self.momentum_dict)
            obj.momentum_identity = getattr(self, "momentum_identity", False)
            obj.parity = self.parity
            obj.projection = [self.projection[-1], tuple(momentum), irrep_name]
            obj.little_group_irreps_dict[str(momentum)] = obj.rows
            if hasattr(self, "exchange_matrices"):
                for name in ("operator_table", "exchange_irrep", "exchange_component",
                             "exchange_copy_index", "exchange_multiplicity",
                             "exchange_slots", "exchange_matrices"):
                    setattr(obj, name, getattr(self, name))
            return obj
        rows = []
        for basis in reduction["row_bases"]:
            row = Row()
            for weight, original in zip(basis[:, idx], self.rows):
                row = row + original * weight
            rows.append(row)
        self.rows = rows
        self.little_group_irreps_dict[str(momentum)] = rows
        return self


class InsertionGaugeLink(Insertion):
    def __init__(self, gamma, gauge_link_irrep_name, gauge_link_idx, projection,
                 momentum_dict, insertion_dict, profile=None, *, copy_index=0,
                 spatial_matrices=None, momentum_identity=True):
        self.momentum_identity = momentum_identity
        from .irrep_coupling import SpinName
        from .baryon_spin import BaryonSpinOperator
        if isinstance(gamma,BaryonSpinOperator) or gamma in (SpinName.G1G, SpinName.G1U, SpinName.HG, SpinName.HU):
            if profile is not None:
                raise ValueError("rank-3 insertions do not accept a rank-2 spectral profile")
            name = gauge_link_irrep_name
            spatial_irrep = name[:-1] if name[-1:] in ("+", "-") else name
            obj = Insertion.from_irreps(
                gamma, insertion_dict[name][gauge_link_idx], spatial_irrep,
                projection, momentum_dict, copy_index=copy_index,
                spatial_matrices=spatial_matrices)
            self.__dict__.update(obj.__dict__)
            self.gauge_link_irrep_name, self.gauge_link_idx = name, gauge_link_idx
            return
        if copy_index != 0 or spatial_matrices is not None:
            raise ValueError("copy_index and spatial_matrices apply to baryon irrep coupling")
        g, left, gp, gc, gh = _GAMMA[gamma]
        name = gauge_link_irrep_name
        # Names such as T1g-, A2u+, Eg+ carry irrep, parity and C.
        group = {"A1": "A_1", "A2": "A_2", "E": "E", "T1": "T_1", "T2": "T_2"}
        group.update({"A_1": "A_1", "A_2": "A_2", "T_1": "T_1", "T_2": "T_2"})
        if len(name) < 3 or name[-2] not in "gu" or name[-1] not in "+-" or name[:-2] not in group:
            raise ValueError("spatial irrep must use a name such as T1g-")
        p, c = (1 if name[-2] == "g" else -1), (1 if name[-1] == "+" else -1)
        self.gauge_link_irrep_name, self.gauge_link_idx = name, gauge_link_idx
        self._initialize(g, insertion_dict[name][gauge_link_idx], left, group[name[:-2]], projection,
                         momentum_dict, profile, gp*p, gc*c, gh*c)
