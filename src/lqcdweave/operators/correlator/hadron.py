import sympy as sp
import numpy as np
from sympy import Add, Expr, S, Pow, Mul
from typing import List, Sequence

from itertools import product

from ._expressions import word_factors
from .base_types import Tag
from lqcdweave.contracts.correlator.api import normalize_time_key
from .flavor_structure import HadronFlavorStructure
from .spatial_structure import HadronIrrepRow
from .symbolic_graph import Diagram, diagram_simplify, diagram_vertice_replace, quark_contract
from .quark_exchange import normalize_quark_symmetry


class Hadron:
    """Symbolic operator with optional full three-quark coefficient symmetry."""

    def __init__(self, irep_row: Expr, flavor_structure: Expr, *, quark_symmetry=None):
        symmetry = normalize_quark_symmetry(quark_symmetry)
        if symmetry and irep_row != 0 and flavor_structure != 0:
            flavors = flavor_structure.atoms(HadronFlavorStructure)
            rows = irep_row.atoms(HadronIrrepRow)
            if not flavors or not rows or any(factor.baryon_num == 0 for factor in flavors):
                raise ValueError("quark exchange symmetry requires baryon rows and flavors")
            if any(row.quark_exchange is not None for row in rows):
                raise ValueError("declare scalar or matrix exchange, not both on the same row")
            for expression, kind in ((irep_row, HadronIrrepRow),
                                     (flavor_structure, HadronFlavorStructure)):
                if any(sum(isinstance(factor, kind) for factor in word_factors(term)) != 1
                       for term in Add.make_args(expression.expand())):
                    raise ValueError("declare quark exchange symmetry on each single baryon before forming products")
            if len(rows) != 1:
                raise ValueError("declare quark exchange symmetry on one complete baryon row before forming linear combinations")
            flavor_structure = flavor_structure.xreplace({factor: HadronFlavorStructure(
                factor.flavor_str, factor.time, dict(symmetry)) for factor in flavors})
            irep_row = irep_row.xreplace({row: HadronIrrepRow(
                row.hadron_name, row.momentum, row.irrep_name, row.row_idx,
                row.parity, row.tag, row.dagger, quark_symmetry=dict(symmetry),
            ) for row in rows})
        rows = irep_row.atoms(HadronIrrepRow) if irep_row != 0 else ()
        flavors = flavor_structure.atoms(HadronFlavorStructure) if flavor_structure != 0 else ()
        if any(row.quark_exchange is not None for row in rows) and flavor_structure != 0:
            position_words = [tuple(f for f in word_factors(term) if isinstance(f, HadronIrrepRow))
                              for term in Add.make_args(irep_row.expand())]
            flavor_words = [tuple(f for f in word_factors(term) if isinstance(f, HadronFlavorStructure))
                            for term in Add.make_args(flavor_structure.expand())]
            for position_word, flavor_word in product(position_words, flavor_words):
                if len(position_word) != len(flavor_word) or any(
                        row.quark_exchange is not None and row.quark_exchange.flavors != tuple(
                            f.quark_list if f.baryon_num == 1 else f.anti_quark_list)
                        for row, f in zip(position_word, flavor_word)):
                    raise ValueError("matrix exchange components must retain their ordered flavor words")
        if irep_row != 0 and flavor_structure != 0 and (
                any(row.quark_symmetry for row in rows) or any(f.quark_symmetry for f in flavors)):
            declarations = set()
            for expression, kind in ((irep_row, HadronIrrepRow),
                                     (flavor_structure, HadronFlavorStructure)):
                for term in Add.make_args(expression.expand()):
                    declarations.add(tuple(factor.quark_symmetry for factor in word_factors(term)
                                           if isinstance(factor, kind)))
            if len(declarations) != 1:
                raise ValueError("position and flavor must carry the same ordered quark symmetry declarations")
        self._irrep_row = irep_row
        self._flavor_structure = flavor_structure

    @classmethod
    def multiplet(cls, hadrons, *, quark_symmetry="mixed"):
        """Declare coupled complete baryon coefficients in one exchange basis.

        The standard two components are ordered MS, MA. Component names must
        differ; quantum numbers, insertion tags and the fixed flavor word
        agree. Numerical bindings must provide every named partner used.
        """
        from .quark_exchange_matrix import QuarkExchange, ExchangeComponent
        representation = QuarkExchange(quark_symmetry)
        hadrons = tuple(hadrons)
        if len(hadrons) != representation.dimension:
            raise ValueError("one baryon is required per exchange matrix component")
        if any(not isinstance(h, cls) or not isinstance(h.irrep_row, HadronIrrepRow)
               or not isinstance(h.flavor_structure, HadronFlavorStructure)
               or h.flavor_structure.baryon_num == 0 for h in hadrons):
            raise ValueError("exchange multiplets require single baryon rows and fixed flavor words")
        rows = tuple(h.irrep_row for h in hadrons)
        if any(row.quark_symmetry or row.quark_exchange for row in rows):
            raise ValueError("exchange multiplets require unlabelled component rows")
        names = tuple(row.hadron_name for row in rows)
        if len(set(names)) != len(names):
            raise ValueError("exchange multiplet components require distinct hadron names")
        signatures = {(tuple(r.momentum), r.irrep_name, r.row_idx, r.parity, r.tag, r.dagger,
                       h.flavor_structure) for r, h in zip(rows, hadrons)}
        if len(signatures) != 1:
            raise ValueError("exchange multiplet components must share quantum numbers, tags and flavor")
        flavor = hadrons[0].flavor_structure
        flavors = tuple(flavor.quark_list if flavor.baryon_num == 1 else flavor.anti_quark_list)
        result = []
        for i, h in enumerate(hadrons):
            component = ExchangeComponent(representation, names, flavors, i)
            result.append(cls(component.row(h.irrep_row, i), h.flavor_structure))
        return tuple(result)

    @property
    def quark_symmetry(self):
        if self._irrep_row == 0:
            return ()
        groups = {row.quark_symmetry for row in self._irrep_row.atoms(HadronIrrepRow)}
        return next(iter(groups)) if len(groups) == 1 else ()

    @property
    def irrep_row(self) -> Expr:
        return self._irrep_row

    @property
    def flavor_structure(self) -> Expr:
        return self._flavor_structure

    def set_time(self, time: int | str) -> "Hadron":
        """
        Create a new Hadron instance with modified time value.
        The original instance remains unchanged.

        Args:
            time: The new time value to set

        Returns:
            A new Hadron instance
        """
        # Modify the tag in irrep_row
        new_irrep_row = set_time_in_expr(self._irrep_row, time)

        # Modify the tag in flavor_structure
        new_flavor_structure = set_time_in_expr(self._flavor_structure, time)

        # Return a new Hadron instance
        return Hadron(new_irrep_row, new_flavor_structure)

    def conjugate(self) -> "Hadron":
        """
        Create a new Hadron instance with conjugated values.
        The original instance remains unchanged.
        """
        # 修改irrep_row中的dagger值
        new_irrep_row = operator_conjugate(self._irrep_row.expand())
        new_flavor_structure = operator_conjugate(self._flavor_structure)
        return Hadron(new_irrep_row, new_flavor_structure)

    def copy(self) -> "Hadron":
        """
        Create a new Hadron instance with the same values.
        The original instance remains unchanged.
        """
        return Hadron(self._irrep_row, self._flavor_structure)


def set_time_in_expr(expr: Expr, time: int | str) -> Expr:
    """
    Recursively traverse the expression to modify the tag.time value of leaf nodes

    Args:
        expr: The expression to modify
        time: The new time value to set

    Returns:
        The modified expression
    """
    time = normalize_time_key(time, name="time")
    replacements = {}

    # Traverse all sub-expressions in the expression
    for sub_expr in sp.preorder_traversal(expr):
        if isinstance(sub_expr, HadronIrrepRow):
            # Create a new Tag, only modify the time value
            new_tag = Tag(sub_expr.tag.tag, time)
            # Create a new HadronIrrepRow instance
            new_row = HadronIrrepRow(
                sub_expr.hadron_name,
                sub_expr.momentum,
                sub_expr.irrep_name,
                sub_expr.row_idx,
                sub_expr.parity,
                new_tag,
                sub_expr.dagger,
                quark_symmetry=dict(sub_expr.quark_symmetry),
                quark_exchange=sub_expr.quark_exchange,
            )
            replacements[sub_expr] = new_row
        elif isinstance(sub_expr, HadronFlavorStructure):
            # For MesonFlavorStructure, create a new instance
            new_meson = HadronFlavorStructure(sub_expr.flavor_str, time,
                                             dict(sub_expr.quark_symmetry))
            replacements[sub_expr] = new_meson

    # Apply replacements and return the new expression
    return expr.xreplace(replacements)


def set_time_in_list(hadron_list: List[Hadron], time: int | str) -> List[Hadron]:
    """
    Apply set_time_in_expr function to each expression in the list, and set_time function to each Hadron
    """
    hadron_list_new = []
    for i in range(len(hadron_list)):
        hadron = hadron_list[i]
        hadron_list_new.append(hadron.set_time(time))
    return hadron_list_new


def operator_conjugate(expr: Expr) -> Expr:
    if isinstance(expr, Add):
        terms = Add.make_args(expr)
        return Add(*[operator_conjugate(term) for term in terms])
    elif isinstance(expr, Mul):
        factors = Mul.make_args(expr)
        return Mul(*[operator_conjugate(factor) for factor in factors[::-1]])
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        return Pow(operator_conjugate(base), exp)
    elif isinstance(expr, HadronIrrepRow):
        return expr.conjugate()
    elif isinstance(expr, HadronFlavorStructure):
        return expr.conjugate()
    else:
        return expr.conjugate()


def set_dagger_in_list(hadron_list: List[Hadron], dagger: bool) -> List[Hadron]:
    """
    对列表中的每个表达式应用set_dagger_in_expr函数，并且对每个Hadron调用设置dagger方法

    Args:
        hadron_list: 要处理的哈德龙列表
        dagger: 要设置的dagger值

    Returns:
        处理后的哈德龙列表
    """
    hadron_list_new = []
    for i in range(len(hadron_list)):
        hadron = hadron_list[i]
        if dagger:
            hadron_list_new.append(hadron.conjugate())
        else:
            hadron_list_new.append(hadron.copy())
    return hadron_list_new


def _finish_correlator(expression: Expr) -> Expr:
    """Canonical numeric Wick polynomials need no general algebra search.

    The preceding evaluated Add/Mul constructors already collect numeric
    coefficients and identical monomials. Preserve general simplification for
    symbolic coefficients, powers, and other algebra outside this form.
    """
    for term in Add.make_args(expression):
        for factor in Mul.make_args(term):
            if not isinstance(factor, Diagram) and not factor.is_Number:
                return sp.simplify(expression)
    return expression


def _prepare_hadrons(hadrons, time_slice_list, dagger_list, *, degenerate):
    if time_slice_list is None:
        time_slice_list = [i for i in range(len(hadrons))]
    else:
        try:
            time_slice_list = list(time_slice_list)
        except TypeError as exc:
            raise TypeError("time_slice_list must be a sequence") from exc
        if len(time_slice_list) != len(hadrons):
            raise ValueError("time_slice_list must have one entry per operator slot")
    if dagger_list is None:
        dagger_list = [False for _ in range(len(hadrons))]
        # 默认把最后一个设为True
        if len(dagger_list) >= 2:
            dagger_list[-1] = True
    else:
        try:
            dagger_list = list(dagger_list)
        except TypeError as exc:
            raise TypeError("dagger_list must be a sequence") from exc
        if len(dagger_list) != len(hadrons):
            raise ValueError("dagger_list must have one entry per operator slot")
    if not isinstance(degenerate, (bool, np.bool_)):
        raise TypeError("degenerate must be a bool")
    degenerate = bool(degenerate)
    time_slice_list = [None if value is None else normalize_time_key(
        value, name="time_slice_list entry") for value in time_slice_list]
    if any(not isinstance(value, (bool, np.bool_)) for value in dagger_list):
        raise TypeError("dagger_list entries must be bools")
    dagger_list = [bool(value) for value in dagger_list]

    for i in range(len(hadrons)):
        if time_slice_list[i] is not None:
            hadrons[i] = set_time_in_list(hadrons[i], time_slice_list[i])
        hadrons[i] = set_dagger_in_list(hadrons[i], dagger_list[i])
    return degenerate


def gen_correlator(
    hadrons: List[List[Hadron]],
    time_slice_list: Sequence[int | str | None] | None = None,
    dagger_list: Sequence[bool] | None = None,
    *,
    degenerate: bool = True,
    structured: bool = False,
):
    """
    Calculate correlation functions for multiple Hadron lists

    Args:
        hadrons: List containing multiple lists of Hadron objects
        time_slice_list: List of time slices; a None entry preserves the
            individual particle times in that projected multi-particle operator.
        dagger_list: List of dagger values to apply to hadrons
        structured: Keep compact single-hadron axes and Wick templates for
            metadata expansion instead of constructing a symbolic matrix.

    Returns:
        Object matrix, or StructuredCorrelator when structured=True.
    """
    if not isinstance(structured, (bool, np.bool_)):
        raise TypeError("structured must be a bool")
    degenerate = _prepare_hadrons(hadrons, time_slice_list, dagger_list, degenerate=degenerate)
    if structured:
        from .structured import StructuredCorrelator
        return StructuredCorrelator.from_hadrons(hadrons, degenerate=degenerate)
    cache = {}
    exchange_cache = {}
    from .quark_exchange_matrix import scalar_exchange_flavors
    scalar_flavors = scalar_exchange_flavors(tuple(tuple((h.irrep_row, h.flavor_structure) for h in axis)
                                                  for axis in hadrons))
    result_matrix = np.ndarray(
        shape=tuple([len(hadrons[i]) for i in range(len(hadrons))]), dtype=object
    )
    # Create all possible index combinations
    indices_ranges = [range(len(hadrons[i])) for i in range(len(hadrons))]
    indices_combinations = list(product(*indices_ranges))
    # Iterate through all index combinations
    for indices in indices_combinations:
        # Build corresponding hadron_tuple
        hadron_tuple = tuple(hadrons[i][indices[i]] for i in range(len(hadrons)))
        position_wavefnc = Mul(*(h.irrep_row for h in hadron_tuple)).expand()
        flavor_wavefnc = Mul(*(h.flavor_structure for h in hadron_tuple)).expand()
        result = S(0)
        terms = Add.make_args(position_wavefnc)
        for term in terms:
            if term == 0:
                continue
            insersion_list = []
            factors = word_factors(term)
            coeff = S(1)
            for factor in factors:
                if isinstance(factor, HadronIrrepRow):
                    insersion_list.append(factor)
                else:
                    coeff *= factor
            # Convert to a hashable form suitable for dictionary key
            contraction_key = str(flavor_wavefnc.expand())
            if contraction_key not in cache:
                term_of_result = diagram_simplify(
                    quark_contract(
                        flavor_wavefnc, np.arange(len(insersion_list)), degenerate=degenerate
                    )
                )
                cache[contraction_key] = term_of_result
            else:
                term_of_result = cache[contraction_key]
            term_of_result = diagram_vertice_replace(
                term_of_result, {i: v for i, v in enumerate(insersion_list)}
            )
            result += term_of_result * coeff

        # Store hadron_tuple in corresponding position in result
        from .quark_exchange_matrix import reduce_matrix_exchange
        result = reduce_matrix_exchange(result, cache=exchange_cache, scalar_flavors=scalar_flavors)
        result_matrix[indices] = _finish_correlator(result)
    return result_matrix


__all__ = [
    "Hadron",
    "operator_conjugate",
    "set_time_in_expr",
    "set_time_in_list",
    "set_dagger_in_list",
    "gen_correlator",
]
