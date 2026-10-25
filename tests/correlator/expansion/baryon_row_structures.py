"""uuu row families of chosen coefficient structure, for every opened lowering.

A family's coefficients ``C[r, s0, s1, s2]`` decide how each opened form
lowers: the pair split has the slot rank of ``C`` (rank one with one spin per
row folds into the output selection), nonzero entries cost one operand entry
per nonzero spin position, dense keeps ``C``. Structures are named specs:

* synthetic ``fold-pN``, ``rankK-pN`` (rank K at slot 0, N entries per pair
  matrix) and ``generic-eN`` (N random entries per row, no low rank);
* catalog recipes ``local-*``, ``spin-*`` and ``diquark-*``;
* ``NAME@dp`` binds the same operators to Dirac-Pauli data: the split rank is
  basis invariant, the nonzero pattern and the fold are not;
* ``A+B`` gives every row two parts, A's row on the first Phi and B's row on a
  second Phi (as a displaced or derivative term would): one constant per part;
* ``SINK|SOURCE`` puts two structures into one request (one family each);
* ``single:SPEC`` requests only the first row at both ends.

Shared by ``test_baryon_row_structures.py`` and
``benchmarks/baryon_vertex_structure.py``.
"""
from hashlib import sha256

import numpy as np

from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contraction.expansion.baryon_factor_network import (
    auto_row_lowering, entry_row_coefficients, factor_row_coefficients,
)
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.correlator.api import Flavor, Hadron, HadronIrrep, Tag, gen_correlator
from lqcdweave.operators.single_hadron.api import (
    Operator, OperatorTerm, baryon_diquark_operators, baryon_spin_operators,
)
from lqcdweave.operators.single_hadron.baryon import SparseSpinTensor, baryon_local_spin_rows

COMPONENT = BaryonComponent.paths(((), (), ()))
COMPONENTS = (COMPONENT, BaryonComponent.paths(((0,), (), ())))
IRREPS = {1: "G1", 2: "G1", 4: "H"}


def _rng(name):
    return np.random.default_rng(int.from_bytes(sha256(name.encode()).digest()[:8], "little"))


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _mask(rng, shape, count):
    mask = np.zeros(int(np.prod(shape)), bool)
    mask[rng.choice(mask.size, count, replace=False)] = True
    return mask.reshape(shape)


def _rows(values):
    return tuple(SparseSpinTensor.from_dense(value) for value in values)


def low_rank(name, *, rank, pair_entries, row_entries, rows=4):
    """``C[r, s0, s1, s2] = sum_k W[r, s0, k] P[k, s1, s2]``; ``row_entries`` spins of s0 per (r, k)."""
    rng = _rng(name)
    pair = np.stack([_complex(rng, (4, 4)) * _mask(rng, (4, 4), pair_entries) for _ in range(rank)])
    weight = np.stack([np.stack([_complex(rng, 4) * _mask(rng, 4, row_entries) for _ in range(rank)], axis=-1)
                       for _ in range(rows)])
    if row_entries == 1 and rank == 1:  # one spin per row, all rows distinct: the fold
        weight = np.eye(4)[rng.permutation(4)[:rows], :, None] * _complex(rng, (rows, 1, 1))
    return _rows(np.einsum("rsk,kab->rsab", weight, pair))


def generic(name, *, entries, rows=4):
    rng = _rng(name)
    return _rows(_complex(rng, (rows, 4, 4, 4)) * np.stack([_mask(rng, (4, 4, 4), entries) for _ in range(rows)]))


def _catalog(source, name):
    return tuple({recipe.name: recipe for recipe in source()}[name].rows)


SYNTHETIC = {
    **{f"fold-p{n}": dict(rank=1, pair_entries=n, row_entries=1) for n in (4, 16)},
    "fold-p16-r2": dict(rank=1, pair_entries=16, row_entries=1, rows=2),
    **{f"rank{k}-p{n}": dict(rank=k, pair_entries=n, row_entries=4)
       for k, n in ((1, 4), (1, 16), (2, 4), (2, 16), (3, 4), (3, 16), (4, 16), (5, 16), (6, 16),
                    (8, 16), (12, 16))},
}
GENERIC = {f"generic-e{n}": n for n in (1, 2, 4, 8, 24, 64)}
CATALOG = {
    "local-G1g": lambda: baryon_local_spin_rows("G1", "g"),
    "local-Hg": lambda: baryon_local_spin_rows("H", "g"),
    "spin-G1g-S0": lambda: _catalog(baryon_spin_operators, "spin_G1g_S_copy0"),
    "spin-Hg-S1": lambda: _catalog(baryon_spin_operators, "spin_Hg_S_copy1"),
    "diquark-Cg5-G1g": lambda: _catalog(baryon_diquark_operators, "diquark_Cg5_G1g"),
    "diquark-Cg4g5-G1g": lambda: _catalog(baryon_diquark_operators, "diquark_Cg4g5_G1g"),
    # Two scalar diquarks in one family: an exact rank-2 split.
    "diquark-Cg5-Cg4g5-G1g": lambda: (_catalog(baryon_diquark_operators, "diquark_Cg5_G1g")
                                      + _catalog(baryon_diquark_operators, "diquark_Cg4g5_G1g")),
    "diquark-Cgi-Hg": lambda: _catalog(baryon_diquark_operators, "diquark_Cgi_Hg"),
}
BASES = ("local-G1g", "local-Hg", "spin-Hg-S1", "diquark-Cg5-G1g", "diquark-Cgi-Hg")
MIXED = ("fold-p16|generic-e64", "rank2-p16|rank6-p16", "diquark-Cgi-Hg|spin-Hg-S1",
         "fold-p16-r2|spin-G1g-S0", "rank6-p16|generic-e1")
# Two Phi per row; every part has the row count of its partner.
TWO_PHI = ("fold-p16+fold-p4", "rank2-p16+rank2-p4", "rank2-p16+rank6-p16", "generic-e64+generic-e24",
           "diquark-Cgi-Hg+spin-Hg-S1", "fold-p16-r2+spin-G1g-S0", "rank2-p16+generic-e1")
# Their uuu full-Wick sums cancel exactly (an antisymmetric spin pair, like a
# uu scalar diquark), also on Dirac-Pauli data: timings only, values are rounding.
VANISHING = ("local-G1g", "diquark-Cg5-G1g", "diquark-Cg4g5-G1g", "diquark-Cg5-Cg4g5-G1g")
SINGLE = ("fold-p16", "rank6-p16", "generic-e64", "diquark-Cgi-Hg", "generic-e2")
STRUCTURES = (*SYNTHETIC, *GENERIC, *CATALOG, *(f"{name}@dp" for name in BASES), *MIXED,
              *(f"single:{name}" for name in SINGLE), *TWO_PHI, "single:fold-p16+rank6-p16",
              "single:generic-e4", "single:fold-p16|generic-e2")


def _rows_of(name):
    if name in SYNTHETIC:
        return low_rank(name, **SYNTHETIC[name])
    if name in GENERIC:
        return generic(name, entries=GENERIC[name])
    return tuple(CATALOG[name]())


def family(spec):
    """``(parts, spin_basis)`` of one structure without ``|`` or ``single:``; one row tuple per Phi."""
    names, _, basis = spec.partition("@")
    basis = {"": "degrand_rossi", "dp": "dirac_pauli"}[basis]
    parts = tuple(_rows_of(name) for name in names.split("+"))
    if len({len(rows) for rows in parts}) != 1:
        raise ValueError(f"the parts of {spec} have different row counts")
    return parts, basis


def ends(spec):
    """``(sink, source, single)`` families of a structure spec."""
    single = spec.startswith("single:")
    spec = spec.removeprefix("single:")
    sink, _, source = spec.partition("|")
    return family(sink), family(source or sink), single


def coefficients(rows, basis, *, source):
    """The bound ``(R, 4, 4, 4)`` constant one family lowers, as B describes it."""
    values = []
    for spin in rows:
        spin = spin.to_basis(basis)
        values.append((spin.source() if source else spin).dense())
    return np.stack(values)


def describe(values):
    """Structure of one family: split rank and fold, nonzero positions, the auto choice."""
    split = factor_row_coefficients(values, bounded=False)
    return dict(rows=len(values), rank=split.rank, fold=split.row is None,
                bounded_split=factor_row_coefficients(values) is not None,
                nonzero=int(entry_row_coefficients(values).weight.shape[1]), auto=auto_row_lowering(values))


def request(spec, *, sink_reader, source_reader):
    """Full-Wick uuu matrix of a structure spec and its vertex map.

    The readers hold ``COMPONENTS`` when a family has two parts.
    """
    (sink_parts, sink_basis), (source_parts, source_basis), single = ends(spec)
    vertices, sides = {}, {}
    for side, parts, basis, source, reader in (("sink", sink_parts, sink_basis, False, sink_reader),
                                               ("source", source_parts, source_basis, True, source_reader)):
        sides[side] = []
        count = len(parts[0])
        for r in range(1 if single else count):
            name = f"{side}_r{r}"
            row = HadronIrrep(name, (0, 0, 0), IRREPS[count], 1, Tag(0, 0))[r]
            sides[side].append(Hadron(row, Flavor("uuu")))
            vertices[name] = Baryon(reader, Operator.from_terms(name, tuple(
                OperatorTerm.from_baryon(component, rows[r], coefficient=1.0)
                for component, rows in zip(COMPONENTS, parts))), source=source,
                phase_convention="anchor", spin_basis=basis)
    value = np.asarray(gen_correlator([sides["sink"], sides["source"]], ["sink", "source"], [False, True],
                                      degenerate=True), dtype=object)
    return value, vertices


def structure(spec):
    """Structures of every family part: ``sink``/``source``, or ``sink0``, ``sink1``, ... with two Phi."""
    result = {}
    for side, (parts, basis) in zip(("sink", "source"), ends(spec)[:2]):
        for index, rows in enumerate(parts):
            rows = rows[:1] if ends(spec)[2] else rows
            key = side if len(parts) == 1 else f"{side}{index}"
            result[key] = describe(coefficients(rows, basis, source=side == "source"))
    return result


def two_phi_reader(reader, seed):
    """``reader``'s Phi as the first component and a seeded random second Phi."""
    from lqcdweave.adapters.elemental import ArrayBaryonElemental
    first = reader.sectors["vvv"]
    second = _complex(_rng(f"phi2-{seed}"), first.shape) / 3
    return ArrayBaryonElemental({"vvv": np.concatenate((first, second.astype(first.dtype)))}, COMPONENTS,
                                cfg=reader.cfg, data_identity=f"{reader.data_identity}+phi2")
