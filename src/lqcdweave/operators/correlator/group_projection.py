import sympy as sp
import numpy as np
from sympy import Expr, Matrix, Pow, Mul, Add
from numbers import Integral
from typing import Sequence
from itertools import product

from lqcdweave.symmetry import (
    irrep_matrices, inverse, reference_frame, project_row, stabilizer,
)
from lqcdweave.symmetry.sympy_utils import find_linear_independent_exprs
from .spatial_structure import HadronIrrepRow


def operator_transform(expr: Expr, group_element, time=None) -> Expr:
    if isinstance(expr, list):
        return [operator_transform(item, group_element) for item in expr]
    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array(
            [operator_transform(item, group_element) for item in flattened],
            dtype=object,
        )
        return result.reshape(original_shape)
    elif isinstance(expr, dict):
        return {k: operator_transform(v, group_element) for k, v in expr.items()}
    elif isinstance(expr, tuple):
        return tuple(operator_transform(item, group_element) for item in expr)
    elif isinstance(expr, Matrix):
        return Matrix(operator_transform(expr.tolist(), group_element))
    elif isinstance(expr, Add):
        terms = Add.make_args(expr)
        return Add(*[operator_transform(term, group_element, time) for term in terms])
    elif isinstance(expr, Mul):
        factors = Mul.make_args(expr)
        return Mul(
            *[operator_transform(factor, group_element, time) for factor in factors]
        )
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        return Pow(operator_transform(base, group_element, time), exp)
    elif isinstance(expr, HadronIrrepRow):
        return expr.transform(group_element, time)
    elif hasattr(expr, "transform"):
        return expr.transform(group_element, time)
    else:
        return expr


def expr_little_group_projection(expr, irrep_name, row_idx, parity=None, p_ref=False):
    """
    Project expression to little group irrep

    Args:
        expr: Expression to project
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        p_ref: Whether to return reference momentum info

    Returns:
        Projected expression
    """
    from .spatial_structure import HadronIrrepRow

    momentum = np.array([0, 0, 0], dtype=int)
    terms = Add.make_args(expr)
    factors = Mul.make_args(terms[0])
    for factor in factors:
        if isinstance(factor, HadronIrrepRow):
            momentum += np.array(factor.momentum, dtype=int)
    ref_p, rotation = reference_frame(momentum)
    p_str = ",".join(str(int(x)) for x in momentum)
    ref_p_str = ",".join(map(str, ref_p))
    expr_tmp = operator_transform(expr, inverse(rotation))
    projected_irrep_row = project_row(expr_tmp, ref_p, irrep_name, 0, operator_transform, parity)
    if p_ref:
        return projected_irrep_row, p_str, ref_p_str
    result = project_row(projected_irrep_row, ref_p, irrep_name, row_idx, operator_transform, parity)
    result = operator_transform(result, rotation)
    return result


def multi_exprs_little_group_projection(
    expr_list, irrep_name, row_idx, parity=None, single_result=False
):
    """
    Project multiple expressions to little group irrep

    Args:
        expr_list: List of expressions to project
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        single_result: Whether to return single result

    Returns:
        List of projected expressions
    """
    result_expr_list = []
    for expr in expr_list:
        projected_expr, p_str, ref_p_str = expr_little_group_projection(
            expr, irrep_name, row_idx, parity, p_ref=True
        )
        if projected_expr != 0:
            if single_result:
                ref_p = tuple(map(int, ref_p_str.split(",")))
                _, rotation = reference_frame(tuple(map(int, p_str.split(","))))
                return [operator_transform(project_row(projected_expr, ref_p, irrep_name,
                                                       row_idx, operator_transform, parity), rotation)]
            else:
                result_expr_list.append(projected_expr.expand())
    # return result_expr_list
    tmp = find_linear_independent_exprs(result_expr_list)
    result = tmp.copy()
    for i in range(len(tmp)):
        ref_p = tuple(map(int, ref_p_str.split(",")))
        _, rotation = reference_frame(tuple(map(int, p_str.split(","))))
        result[i] = operator_transform(project_row(tmp[i], ref_p, irrep_name, row_idx,
                                                  operator_transform, parity), rotation)
    return result


def hadron_little_group_projection(
    hadron_irreps: Sequence,
    irrep_name: str,
    row_idx: int,
    parity: int | None = None,
    single_result: bool = False,
) -> list[Expr]:
    """
    Project hadron irreps to little group irrep

    Args:
        hadron_irreps: List of HadronIrrep objects
        irrep_name: Irrep name
        row_idx: Row index
        parity: Parity
        single_result: Whether to return single result

    Returns:
        List of projected expressions
    """
    from .spatial_structure import HadronIrrep

    if isinstance(hadron_irreps, (str, bytes)):
        raise TypeError("hadron_irreps must be a sequence of HadronIrrep")
    try:
        hadron_irreps = tuple(hadron_irreps)
    except TypeError as exc:
        raise TypeError("hadron_irreps must be a sequence of HadronIrrep") from exc
    if not hadron_irreps:
        raise ValueError("at least one HadronIrrep is required")
    if any(not isinstance(hadron_irrep, HadronIrrep) for hadron_irrep in hadron_irreps):
        raise TypeError("hadron_irreps entries must be HadronIrrep")
    if isinstance(row_idx, bool) or not isinstance(row_idx, Integral):
        raise TypeError("row_idx must be an integer")
    if row_idx < 0:
        raise ValueError("row_idx must be non-negative")
    row_idx = int(row_idx)
    if parity is not None:
        if isinstance(parity, bool) or not isinstance(parity, Integral):
            raise TypeError("parity must be -1, 1, or None")
        if int(parity) not in (-1, 1):
            raise ValueError("parity must be -1, 1, or None")
        parity = int(parity)
    if not isinstance(single_result, bool):
        raise TypeError("single_result must be a bool")

    momentum_structure = tuple(
        hadron_irreps[i].momentum for i in range(len(hadron_irreps))
    )
    momentum_total = [
        sum([momentum_structure[i][j] for i in range(len(momentum_structure))])
        for j in range(len(momentum_structure[0]))
    ]
    little_group = stabilizer(momentum_total)

    momentum_structure_dict = {}
    rotation = irrep_matrices((0, 0, 0), "T1u")
    for key in little_group.keys():
        new_momentum_structure = [
            tuple(rotation[key] @ Matrix(momentum_structure[i]))
            for i in range(len(momentum_structure))
        ]
        momentum_structure_dict[tuple(new_momentum_structure)] = None
    exprs_mul_rows = []
    for momentum_structure in momentum_structure_dict.keys():
        hadrons_list = []
        for idx in range(len(hadron_irreps)):
            hadron_irrep = hadron_irreps[idx]
            new_hadron_irrep = HadronIrrep(
                hadron_irrep.hadron_name,
                list(momentum_structure[idx]),
                hadron_irrep.irrep_name,
                hadron_irrep.parity,
                hadron_irrep.tag,
                hadron_irrep.dagger,
            )
            hadrons_list.append(
                [new_hadron_irrep[j] for j in range(new_hadron_irrep.lenth)]
            )
        for expr in list(product(*hadrons_list)):
            exprs_mul_rows.append(Mul(*expr))
    # for i, expr in enumerate(exprs_mul_rows):
    #     print(f"{i}:", expr)
    return multi_exprs_little_group_projection(
        exprs_mul_rows, irrep_name, row_idx, parity, single_result
    )
