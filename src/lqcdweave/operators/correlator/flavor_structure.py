from typing import List
from copy import deepcopy

from sympy import Integer, Symbol, Tuple
from sympy.physics.quantum import Operator
from .base_types import Tag, QuarkFlavor
from lqcdweave.contracts.correlator.api import normalize_time_key
from .quark_exchange import normalize_quark_symmetry, adjoint_quark_symmetry, quark_symmetry_name


class Qurak(Symbol):
    def __new__(cls, flavor: QuarkFlavor, tag: Tag, anti: bool, **assumptions) -> None:
        obj = super().__new__(
            cls, Rf"\bar{{{flavor}}}({tag.tag})" if anti else Rf"{flavor}({tag.tag})", commutative=False, **assumptions
        )
        return obj

    def __init__(self, flavor: QuarkFlavor, tag: Tag, anti: bool, **assumptions) -> None:
        """
        Initialize a Qurak object.

        Args:
            flavor: The flavor of the quark
            tag: The tag of the quark
            anti: Whether the quark is an anti-quark
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.tag = tag
        self.anti = anti


class Propagator(Symbol):
    """
    Symbol representing quark propagator
    """

    def __new__(cls, flavor: QuarkFlavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        obj = super().__new__(cls, Rf"S^{flavor}({sink_tag.tag}, {source_tag.tag})", **assumptions)
        return obj

    def __init__(self, flavor: QuarkFlavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        """
        Initialize a Propagator object.

        Args:
            flavor: The flavor of the propagator
            source_tag: The source tag of the propagator
            sink_tag: The sink tag of the propagator
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.source_tag = source_tag
        self.sink_tag = sink_tag
        self.tag = Rf"S^{flavor}_\mathrm{{local}}" if source_tag.time == sink_tag.time else Rf"S^{flavor}"

    def __deepcopy__(self, memo):
        # Symbol's normal constructor is cached and its pickle arguments omit
        # this subclass's flavor/tags. The inherited uncached constructor keeps
        # that detail inside S and preserves the original symbol assumptions.
        clone = self.__xnew__(type(self), self.name, **self.assumptions0)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, deepcopy(value, memo))
        return clone


class HadronFlavorStructure(Operator):
    def __new__(cls, flavor_str: str, time: int | str = 0, quark_symmetry=None) -> None:
        flavor_str, time, quark_symmetry = _semantic_flavor_args(
            flavor_str, time, quark_symmetry)
        baryon_num, _, _ = _validate_flavor_structure(flavor_str, time)
        symmetry = normalize_quark_symmetry(quark_symmetry)
        if symmetry and baryon_num == 0:
            raise ValueError("quark exchange symmetry requires a three-quark baryon")
        time = normalize_time_key(time, name="time")
        encoded_time = Symbol(time) if isinstance(time, str) else Integer(time)
        encoded_symmetry = Tuple(*(Tuple(Tuple(*permutation), sign)
                                   for permutation, sign in symmetry))
        return super().__new__(cls, Symbol(flavor_str), encoded_time, encoded_symmetry)

    def __init__(self, flavor_str: str, time: int | str = 0, quark_symmetry=None) -> None:
        """
        Initialize a HadronFlavorStructure object.

        Args:
            flavor_str: The flavor string
            time: The time value
        """
        flavor_str, time, quark_symmetry = _semantic_flavor_args(
            flavor_str, time, quark_symmetry)
        baryon_num, quarks, antiquarks = _validate_flavor_structure(flavor_str, time)
        self._flavor_str = flavor_str
        self._time = normalize_time_key(time, name="time")
        self._baryon_num = baryon_num
        self._quark_list = quarks
        self._anti_quark_list = antiquarks
        self._quark_symmetry = normalize_quark_symmetry(quark_symmetry)
        visible = (rf"bar{{{flavor_str[4:-1]}}}"
                   if flavor_str.startswith("bar{") else flavor_str)
        suffix = (f"|quark_symmetry={quark_symmetry_name(self._quark_symmetry)}"
                  if self._quark_symmetry else "")
        self._display_label = Symbol(rf"{visible}({self._time!r}){suffix}")

    @property
    def label(self):
        # Keep Operator's established display while args carry reconstruction
        # data. SymPy rebuilds expressions with leaf.func(*leaf.args).
        return (self._display_label,)

    @property
    def quark_symmetry(self):
        return self._quark_symmetry

    def __getnewargs__(self):
        return self._flavor_str, self._time, dict(self._quark_symmetry)

    def __reduce_ex__(self, protocol):
        return type(self), self.__getnewargs__()

    def __deepcopy__(self, memo):
        clone = type(self)(*self.__getnewargs__())
        memo[id(self)] = clone
        return clone

    @property
    def flavor_str(self) -> str:
        return self._flavor_str

    @property
    def time(self) -> int | str:
        return self._time

    @property
    def baryon_num(self) -> int:
        return self._baryon_num

    @property
    def quark_list(self) -> List[str]:
        return self._quark_list.copy()

    @property
    def anti_quark_list(self) -> List[str]:
        return self._anti_quark_list.copy()

    def conjugate(self) -> "HadronFlavorStructure":
        """
        Create a new HadronFlavorStructure instance with conjugated values.
        The original instance remains unchanged.
        """
        if not self._quark_list and not self._anti_quark_list:
            new_flavor_str = ""
        elif self._baryon_num == 0:
            new_flavor_str = self._quark_list[0] + self._anti_quark_list[0]
        elif self._baryon_num == 1:
            new_flavor_str = f'bar{{{"".join(reversed(self._quark_list))}}}'
        elif self._baryon_num == -1:
            new_flavor_str = "".join(reversed(self._anti_quark_list))
        return HadronFlavorStructure(new_flavor_str, self._time,
                                    adjoint_quark_symmetry(self._quark_symmetry))

    def _eval_is_zero(self):
        """Return True if this expression is zero, False otherwise."""
        return False

    def _eval_is_infinite(self):
        """Return True if this expression is infinite, False otherwise."""
        return False

    def _eval_is_extended_real(self):
        """Return True if this expression is extended real, False otherwise."""
        return True

    def _eval_is_finite(self):
        """Return True if this expression is finite, False otherwise."""
        return True


def _semantic_flavor_args(flavor_str, time, quark_symmetry):
    """Decode the semantic SymPy args without parsing the display label."""
    if isinstance(flavor_str, Symbol):
        if not isinstance(time, (Integer, Symbol)) or not isinstance(quark_symmetry, Tuple):
            raise TypeError("a SymPy flavor rebuild requires its time and symmetry args")
        return (flavor_str.name,
                time.name if isinstance(time, Symbol) else int(time),
                {tuple(map(int, permutation)): int(sign)
                 for permutation, sign in quark_symmetry})
    return flavor_str, time, quark_symmetry


def _validate_flavor_structure(flavor_str, time):
    if not isinstance(flavor_str, str):
        raise TypeError("flavor_str must be a string")
    normalize_time_key(time, name="time")
    if flavor_str == "":
        return 0, [], []
    flavors = "udscbt"
    if flavor_str.startswith("bar{") and flavor_str.endswith("}"):
        body = flavor_str[4:-1]
        if len(body) != 3 or any(flavor not in flavors for flavor in body):
            raise ValueError("antibaryon flavor must contain three u/d/s/c/b/t quarks")
        return -1, [], list(body)
    if "bar" in flavor_str:
        raise ValueError("antibaryon flavor must use the form bar{uds}")
    if len(flavor_str) == 3 and all(flavor in flavors for flavor in flavor_str):
        return 1, list(flavor_str), []
    if len(flavor_str) == 2 and all(flavor in flavors for flavor in flavor_str):
        # Meson strings are ordered anti-quark then quark, e.g. ``ud`` is
        # anti-u times d.
        return 0, [flavor_str[1]], [flavor_str[0]]
    if len(flavor_str) not in (2, 3):
        raise ValueError("flavor structure must be empty or contain two or three quark labels")
    raise ValueError("quark flavor labels must be one of u/d/s/c/b/t")
