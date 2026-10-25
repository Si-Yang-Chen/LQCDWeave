from sympy import Matrix, Symbol, S
from typing import List
from numbers import Integral
from copy import deepcopy
from types import MappingProxyType
from typing import Sequence


from lqcdweave.symmetry import irrep_matrices, wigner_rotation
from .base_types import Tag
from lqcdweave.contracts.correlator.api import normalize_time_key, time_key_order
from .quark_exchange import normalize_quark_symmetry, adjoint_quark_symmetry, quark_symmetry_name


def _spatial_inputs(hadron_name, momentum, irrep_name, parity, tag, dagger):
    if not isinstance(hadron_name, str) or not hadron_name:
        raise TypeError("hadron_name must be a nonempty string")
    if not isinstance(irrep_name, str) or not irrep_name:
        raise TypeError("irrep_name must be a nonempty string")
    if isinstance(momentum, (str, bytes)):
        raise TypeError("momentum must contain three integers")
    try:
        momentum = tuple(momentum)
    except TypeError as exc:
        raise TypeError("momentum must contain three integers") from exc
    if len(momentum) != 3:
        raise ValueError("momentum must contain exactly three integers")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in momentum):
        raise TypeError("momentum entries must be integers")
    momentum = [int(value) for value in momentum]
    if parity is not None:
        if isinstance(parity, bool) or not isinstance(parity, Integral):
            raise TypeError("parity must be -1, 1, or None")
        if int(parity) not in (-1, 1):
            raise ValueError("parity must be -1, 1, or None")
        parity = int(parity)
    if not isinstance(tag, Tag):
        raise TypeError("tag must be a Tag")
    for name, value in (("tag.tag", tag.tag),):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    normalize_time_key(tag.time, name="tag.time")
    if not isinstance(dagger, bool):
        raise TypeError("dagger must be a bool")
    return momentum, parity


def _row_index(row_idx):
    if isinstance(row_idx, bool) or not isinstance(row_idx, Integral):
        raise TypeError("row_idx must be an integer")
    if row_idx < 0:
        raise ValueError("row_idx must be non-negative")
    return int(row_idx)


class HadronIrrep(Symbol):
    def __new__(
        cls,
        hadron_name: str,
        momentum: Sequence[int],
        irrep_name: str,
        parity: int | None,
        tag: Tag,
        dagger: bool = False,
    ):
        momentum, parity = _spatial_inputs(hadron_name, momentum, irrep_name, parity, tag, dagger)
        if parity is None:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}({tag.tag}),t={tag.time!r},{tuple(momentum)},{dagger})",
                commutative=False,
            )
        elif parity == -1:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}u({tag.tag}),t={tag.time!r},{tuple(momentum)},{parity},{dagger})",
                commutative=False,
            )
        else:
            obj = super().__new__(
                cls,
                f"{hadron_name}({irrep_name}g({tag.tag}),t={tag.time!r},{tuple(momentum)},{parity},{dagger})",
                commutative=False,
            )
        return obj

    def __init__(
        self,
        hadron_name: str,
        momentum: Sequence[int],
        irrep_name: str,
        parity: int | None,
        tag: Tag,
        dagger: bool = False,
    ):
        """
        Initialize a HadronIrrep object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            parity: The parity
            tag: The tag
        """
        momentum, parity = _spatial_inputs(hadron_name, momentum, irrep_name, parity, tag, dagger)
        self._hadron_name = hadron_name
        self._momentum = momentum
        self._irrep_name = irrep_name
        self._parity = parity
        self._tag = tag
        self._dagger = dagger

        if irrep_name.startswith("T"):
            self._lenth = 3
        elif irrep_name.startswith("G") or irrep_name.startswith("E"):
            self._lenth = 2
        elif irrep_name.startswith("H"):
            self._lenth = 4
        else:
            self._lenth = 1

    @property
    def hadron_name(self) -> str:
        return self._hadron_name

    @property
    def momentum(self) -> List[int]:
        return self._momentum.copy()

    @property
    def irrep_name(self) -> str:
        return self._irrep_name

    @property
    def parity(self) -> int | None:
        return self._parity

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def dagger(self) -> bool:
        return self._dagger

    @property
    def lenth(self) -> int:
        return self._lenth

    def __getitem__(self, row_idx: int) -> "HadronIrrepRow":
        row_idx = _row_index(row_idx)
        if not 0 <= row_idx < self._lenth:
            raise IndexError("irrep row index is outside the representation")
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            row_idx,
            self._parity,
            self._tag,
            self._dagger,
        )

    def __eq__(self, other):
        if not isinstance(other, HadronIrrep):
            return False
        else:
            return (
                self._hadron_name == other._hadron_name
                and self._momentum == other._momentum
                and self._irrep_name == other._irrep_name
                and self._parity == other._parity
                and self._tag == other._tag
                and self._dagger == other._dagger
            )

    def __hash__(self):
        return hash(
            (
                self._hadron_name,
                tuple(self._momentum),
                self._irrep_name,
                self._parity,
                self._tag,
                self._dagger,
            )
        )

    def copy(self):
        return HadronIrrep(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._parity,
            self._tag,
            self._dagger,
        )

    def __deepcopy__(self, memo):
        clone = self.__xnew__(type(self), self.name, **self.assumptions0)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, deepcopy(value, memo))
        return clone


class HadronIrrepRow(Symbol):
    def __new__(
        cls,
        hadron_name: str,
        momentum: Sequence[int],
        irrep_name: str,
        row_idx: int,
        parity: int | None,
        tag: Tag,
        dagger: bool = False,
        *, quark_symmetry=None, quark_exchange=None,
    ):
        momentum, parity = _spatial_inputs(hadron_name, momentum, irrep_name, parity, tag, dagger)
        row_idx = _row_index(row_idx)
        symmetry = normalize_quark_symmetry(quark_symmetry)
        suffix = f"|quark_symmetry={quark_symmetry_name(symmetry)}" if symmetry else ""
        if quark_exchange is not None:
            from .quark_exchange_matrix import ExchangeComponent
            if not isinstance(quark_exchange, ExchangeComponent):
                raise TypeError("quark_exchange must describe a declared multiplet component")
            if symmetry:
                raise ValueError("declare scalar or matrix exchange, not both on the same row")
            suffix += f"|quark_exchange={quark_exchange.identity}"
        if parity is None:
            obj = Symbol.__xnew__(
                cls,
                rf"{hadron_name}({irrep_name},{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]" + suffix,
                commutative=False,
            )
        elif parity == -1:
            obj = Symbol.__xnew__(
                cls,
                rf"{hadron_name}({irrep_name}u,{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]" + suffix,
                commutative=False,
            )
        else:
            obj = Symbol.__xnew__(
                cls,
                rf"{hadron_name}({irrep_name}g,{tuple(momentum)}){'†' if dagger else ''}[{row_idx}]" + suffix,
                commutative=False,
            )
        return obj

    def __init__(
        self,
        hadron_name: str,
        momentum: Sequence[int],
        irrep_name: str,
        row_idx: int,
        parity: int | None,
        tag: Tag,
        dagger: bool = False,
        *, quark_symmetry=None, quark_exchange=None,
    ):
        """
        Initialize a HadronIrrepRow object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            row_idx: The row index
            parity: The parity
            tag: The tag
        """
        momentum, parity = _spatial_inputs(hadron_name, momentum, irrep_name, parity, tag, dagger)
        row_idx = _row_index(row_idx)
        self._hadron_name = hadron_name
        self._tag = tag
        self._momentum = momentum
        self._irrep_name = irrep_name
        self._row_idx = row_idx
        self._parity = parity
        self._dagger = dagger
        self._quark_symmetry = normalize_quark_symmetry(quark_symmetry)
        self._quark_exchange = quark_exchange
        self._rotate = irrep_matrices((0, 0, 0), "T1u")
        self._little_group_matrix = irrep_matrices(momentum, irrep_name, parity, reference=True)
        dimension = next(iter(self._little_group_matrix.values())).shape[0]
        if row_idx >= dimension:
            raise IndexError("irrep row index is outside the representation")

    @property
    def hadron_name(self) -> str:
        return self._hadron_name

    @property
    def quark_symmetry(self):
        return getattr(self, "_quark_symmetry", ())

    @property
    def quark_exchange(self):
        return getattr(self, "_quark_exchange", None)

    def __getnewargs_ex__(self):
        return ((self._hadron_name, self._momentum, self._irrep_name,
                 self._row_idx, self._parity, self._tag, self._dagger),
                {"quark_symmetry": dict(self.quark_symmetry), "quark_exchange": self.quark_exchange})

    def __reduce_ex__(self, protocol):
        return _restore_hadron_irrep_row, self.__getnewargs_ex__()

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def momentum(self) -> List[int]:
        return self._momentum.copy()

    @property
    def irrep_name(self) -> str:
        return self._irrep_name

    @property
    def row_idx(self) -> int:
        return self._row_idx

    @property
    def parity(self) -> int | None:
        return self._parity

    @property
    def dagger(self) -> bool:
        return self._dagger

    @property
    def rotate(self):
        return self._rotate

    @property
    def little_group_matrix(self):
        return self._little_group_matrix

    def __eq__(self, other):
        if not isinstance(other, HadronIrrepRow):
            return False
        else:
            return (
                self._hadron_name == other._hadron_name
                and self._momentum == other._momentum
                and self._irrep_name == other._irrep_name
                and self._row_idx == other._row_idx
                and self._parity == other._parity
                and self._tag == other._tag
                and self._dagger == other._dagger
                and self.quark_symmetry == other.quark_symmetry
                and self.quark_exchange == other.quark_exchange
            )

    def __hash__(self):
        return hash(
            (
                self._hadron_name,
                tuple(self._momentum),
                self._irrep_name,
                self._row_idx,
                self._parity,
                self._tag,
                self._dagger,
                self.quark_symmetry,
                self.quark_exchange,
            )
        )

    def transform(self, group_element,time=None):
        momentum_final = list(self._rotate[group_element] @ Matrix(self._momentum))
        transform_matrix = self._little_group_matrix[wigner_rotation(self._momentum, group_element)]
        result = S(0)
        for i in range(transform_matrix.shape[0]):
            result += transform_matrix[i, self._row_idx] * HadronIrrepRow(
                self._hadron_name,
                momentum_final,
                self._irrep_name,
                i,
                self._parity,
                self._tag,
                self._dagger,
                quark_symmetry=dict(self.quark_symmetry),
                quark_exchange=self.quark_exchange,
            )
        return result

    def __lt__(self, other):
        if self._tag.time != other._tag.time:
            return time_key_order(self._tag.time) < time_key_order(other._tag.time)
        if self._irrep_name != other._irrep_name:
            return self._irrep_name < other._irrep_name
        if self._row_idx != other._row_idx:
            return self._row_idx < other._row_idx
        return self._tag.tag < other._tag.tag

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def copy(self):
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._row_idx,
            self._parity,
            self._tag,
            self._dagger,
            quark_symmetry=dict(self.quark_symmetry),
            quark_exchange=self.quark_exchange,
        )

    def __deepcopy__(self, memo):
        clone = self.__xnew__(type(self), self.name, **self.assumptions0)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in ('_rotate', '_little_group_matrix') and isinstance(value, MappingProxyType):
                copied = MappingProxyType(deepcopy(dict(value), memo))
            else:
                copied = deepcopy(value, memo)
            setattr(clone, name, copied)
        return clone

    def conjugate(self):
        return HadronIrrepRow(
            self._hadron_name,
            self._momentum,
            self._irrep_name,
            self._row_idx,
            self._parity,
            self._tag,
            not self._dagger,
            quark_symmetry=adjoint_quark_symmetry(self.quark_symmetry),
            quark_exchange=self.quark_exchange.adjoint() if self.quark_exchange else None,
        )


def _restore_hadron_irrep_row(arguments, keywords):
    return HadronIrrepRow(*arguments, **keywords)
