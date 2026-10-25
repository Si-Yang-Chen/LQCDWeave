"""Versioned 64-row three-Dirac-spin catalog and immutable cubic multiplets.

The coefficient-only runtime reads a frozen JSON resource, like meson55.
S3 labels apply to spin alone, without color, space, flavor or Wick signs.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json

from .baryon import SparseSpinTensor


def _version(version):
    if version != 'v1':
        raise ValueError("baryon spin version must be 'v1'")
    return version


@lru_cache(maxsize=None)
def _document(version, name='baryon_spin'):
    _version(version)
    return json.loads(files(__package__).joinpath('presets',f'{name}_{version}.json').read_text(encoding='utf-8'))


def baryon_spin_preset(*, version='v1'):
    """Return independent numbered spin rows, in the meson preset table format.

    Each row has id/name/irrep/parity/copy/row and sparse spin/coefficient
    terms. All 64 rows are orthonormal in the ordered DR (4,4,4) space.
    MS and MA partners share irrep, parity and copy, with fixed exchange phase.
    """
    return deepcopy(_document(_version(version))['rows'])


@dataclass(frozen=True)
class BaryonSpinOperator:
    """One canonical cubic multiplet, parallel to GaugeOperator.

    rows are immutable SparseSpinTensor values. Mixed exchange partners are
    separate cubic multiplets, paired by partner and a shared copy_index.
    """

    name: str
    irrep: str
    parity: int
    exchange_irrep: str
    exchange_component: str
    copy_index: int
    row_ids: tuple
    rows: tuple
    partner: str | None
    version: str = 'v1'
    # Catalog combinations (diquark recipes) record ((multiplet, weight), ...).
    weights: tuple = ()

    @property
    def catalog(self):
        return 'baryon-spin'

    @property
    def basis(self):
        return 'degrand_rossi'

    @property
    def coefficients(self):
        """Coefficients with axes (cubic row, spin0, spin1, spin2)."""
        return tuple(tuple(tuple(tuple(complex(z) for z in line) for line in plane)
                           for plane in spin.dense()) for spin in self.rows)

    def insertion(self, momentum_dict=None, *, component=None):
        """Bind a spatial component to all cubic rows, using the meson recipe API.

        The default is the v1 local baryon spatial component. No complete
        baryon exchange declaration is inferred from the spin label alone.
        """
        from .insertion import Insertion
        from .operator import InsertionTerm
        if component is None:
            from lqcdweave.contracts.elemental import BaryonComponent
            component = BaryonComponent.catalog('baryon-vvv','v1',0)
        if momentum_dict is None:
            momentum_dict = {0:(0,0,0)}
        result = Insertion.from_rows([[InsertionTerm(component,spin=spin)] for spin in self.rows],momentum_dict)
        result.momentum_identity = True
        result.parity = self.parity
        result.projection = [self.irrep,'A1',self.irrep]
        result._rest_irrep = self.irrep
        return result


def baryon_diquark_operators(*, version='v1'):
    """Return 24 low-rank recipes ``(q0^T C Gamma q1) q2`` in canonical catalog rows.

    ``Gamma`` runs over the 16 Clifford elements (``C = gamma(2) gamma(8)`` in
    DR); the free quark is slot 2, as in the reference contraction programs.
    Scalar diquarks ``C, Cg5, Cg4, Cg4g5`` give one G1 recipe per parity (each
    row is one diquark matrix times one free spinor); vector diquarks
    ``Cgi, Cgig5, Cgig4, Cgig4g5`` give G1 and H recipes per parity (rank 3).
    The 64 rows are a complete orthonormal basis. Every recipe is an exact
    combination of same-irrep ``baryon_spin_operators`` multiplets
    (``weights``), so it has no single exchange label.
    """
    return _diquark_operators(_version(version))


@lru_cache(maxsize=1)
def _diquark_operators(version):
    document = _document(version, 'baryon_diquark')
    return tuple(BaryonSpinOperator(
        name=recipe['name'],irrep=recipe['irrep'],parity=recipe['parity'],
        exchange_irrep=None,exchange_component=None,copy_index=0,row_ids=(),
        rows=tuple(SparseSpinTensor(((*term['spin'],complex(*term['coefficient'])) for term in row),
                                    document['basis']) for row in recipe['rows']),
        partner=None,version=version,
        weights=tuple((item['multiplet'],complex(*item['coefficient'])) for item in recipe['weights']),
    ) for recipe in document['recipes'])


@lru_cache(maxsize=1)
def _operators(version):
    document = _document(version)
    spins = tuple(SparseSpinTensor(((*term['spin'],complex(*term['coefficient']))
                                   for term in row['terms']),document['basis']) for row in document['rows'])
    return tuple(BaryonSpinOperator(
        name=row['name'],irrep=row['irrep'],parity=row['parity'],
        exchange_irrep=row['exchange_irrep'],exchange_component=row['exchange_component'],
        copy_index=row['copy'],row_ids=tuple(row['row_ids']),
        rows=tuple(spins[i] for i in row['row_ids']),partner=row['partner'],version=version,
    ) for row in document['multiplets'])


def baryon_spin_operators(*, version='v1'):
    """Return 24 immutable cubic multiplets spanning all 64 spin components."""
    return _operators(_version(version))
