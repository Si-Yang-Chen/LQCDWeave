"""Exact sparse-spin baryon elimination, with dense mode GEMMs.

Only immutable axis recipes are cached. Spin entries are supplied by the
binding after basis/source transformations; no operator coefficient belongs
here. This implements the mathematical sparse sum independently of CLQCD.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern


@dataclass(frozen=True)
class SparseSpinRecipe:
    operand_order: tuple
    permutations: tuple
    mode_permutation: tuple
    free_shapes: tuple
    output_permutation: tuple
    modes: tuple
    chunk: int
    pair_groups: bool


@dataclass(frozen=True)
class BatchedSparseSpinRecipe:
    scalar: SparseSpinRecipe
    lane: object
    width: int
    operand_lanes: tuple
    output_permutation: tuple


def sparse_spin_pattern_supported(entries):
    """Conservative dispatch by exact support, never by a weight threshold.

    Up to six exact nonzeros include the DR H parity rows even when all
    four third-spin values occur. Larger supports retain the previously
    measured four-pair/two-third-spin regime. Group count changes arithmetic,
    not scratch residency: each group's temporaries are released in turn.
    """
    pairs = {(j, k) for _, j, k, _ in entries}
    return len(entries) <= 6 or (len(pairs) <= 4 and len({k for _, k in pairs}) <= 2)


@lru_cache(maxsize=256)
def sparse_spin_recipe(equation, shapes, dtypes, kernel_path):
    """Keep the scalar route and recognize one leading retained lane."""
    scalar = _scalar_sparse_spin_recipe(equation, shapes, dtypes, kernel_path)
    if scalar is not None:
        return scalar
    equation = parse_axis_pattern(equation)
    labels, output = equation.inputs, equation.output
    if len(labels) != 5 or len(shapes) != 5 or len(labels[0]) != 3:
        return None
    # A retained coordinate is the leading axis of Phi or of at least two
    # legs. A leading free axis in only one leg remains on the scalar route.
    candidates = {labels[1][0]} if len(labels[1]) == 4 else set()
    leading = [text[0] for text in labels[2:] if text]
    candidates.update(name for name in leading if leading.count(name) >= 2)
    candidates.intersection_update(output)
    if len(candidates) != 1:
        return None
    lane = next(iter(candidates))
    present = tuple(lane in text for text in labels)
    if present[0] or not any(present[1:]):
        return None
    if any(text[0] != lane for text, has_lane in zip(labels[1:], present[1:]) if has_lane):
        return None
    if any(len(text) != len(shape) for text, shape in zip(labels, shapes)):
        return None
    width = next((shape[0] for shape, has_lane in zip(shapes[1:], present[1:]) if has_lane), 0)
    if width <= 0 or any(shape[0] != width for shape, has_lane in zip(shapes[1:], present[1:]) if has_lane):
        return None
    stripped_labels = tuple(text[1:] if has_lane else text for text, has_lane in zip(labels, present))
    stripped_shapes = tuple(shape[1:] if has_lane else shape for shape, has_lane in zip(shapes, present))
    stripped_output = tuple(name for name in output if name != lane)
    if len(stripped_output) != len(output) - 1:
        return None
    stripped_equation = AxisPattern(stripped_labels, stripped_output)
    base = _scalar_sparse_spin_recipe(stripped_equation, stripped_shapes, dtypes, kernel_path)
    if base is None:
        return None
    # Recover the scalar kernel's grouped free-axis order, then prepend lane.
    grouped = [None] * len(stripped_output)
    for output_index, grouped_index in enumerate(base.output_permutation):
        grouped[grouped_index] = stripped_output[output_index]
    grouped = (lane, *grouped)
    return BatchedSparseSpinRecipe(base, lane, width, present,
                                   tuple(grouped.index(name) for name in output))


def _scalar_sparse_spin_recipe(equation, shapes, dtypes, kernel_path):
    """Recognize three independent legs within the explicit path's workspace.

    Other joins, mixed precision, traces, and shared free labels retain the
    original fixed-path expression. No path is searched or replaced.
    """
    if len(shapes) != 5 or len(set(dtypes)) != 1:
        return None
    if np.dtype(dtypes[0]) not in (np.dtype('complex64'), np.dtype('complex128')):
        return None
    equation = parse_axis_pattern(equation)
    labels, output = equation.inputs, equation.output
    if len(labels) != 5 or len(labels[0]) != 3 or len(labels[1]) != 3:
        return None
    if shapes[0] != (4, 4, 4):
        return None
    if any(len(text) != len(shape) or len(set(text)) != len(text)
           for text, shape in zip(labels, shapes)):
        return None
    spin, mode = labels[:2]
    if len(set(spin + mode)) != 6 or set(spin + mode).intersection(output):
        return None
    dimensions = {}
    for text, shape in zip(labels, shapes):
        for name, size in zip(text, shape):
            if size <= 0 or (name in dimensions and dimensions[name] != size):
                return None
            dimensions[name] = size
    order, permutations, free_shapes, mode_order, free_labels = [], [], [], [], []
    for spin_label in spin:
        matches = [index for index in range(2, 5) if spin_label in labels[index]]
        if len(matches) != 1:
            return None
        index = matches[0]
        text = labels[index]
        mode_matches = [name for name in mode if name in text]
        if len(mode_matches) != 1 or len(set(text).intersection(spin)) != 1:
            return None
        local_mode = mode_matches[0]
        free = tuple(name for name in text if name not in spin + mode)
        order.append(index)
        mode_order.append(mode.index(local_mode))
        permutations.append(tuple(text.index(name) for name in (spin_label, local_mode) + free))
        free_shapes.append(tuple(dimensions[name] for name in free))
        free_labels.extend(free)
    if len(set(order)) != 3 or len(set(mode_order)) != 3:
        return None
    if len(set(free_labels)) != len(free_labels) or set(free_labels) != set(output):
        return None
    if len(set(output)) != len(output):
        return None
    modes = tuple(shapes[1][index] for index in mode_order)
    f0, f1, f2 = (math.prod(shape) for shape in free_shapes)
    chunk = max(1, f0 // 4)
    first = f0 * modes[1] * modes[2]
    accumulator = f0 * modes[2] * f1
    scratch = max(2 * accumulator + first,
                  accumulator + 2 * first,
                  accumulator + first + 2 * modes[0] * f0,
                  accumulator + chunk * f1 * (modes[2] + f2))
    paired_scratch = max(3 * accumulator + 2 * first + 2 * modes[0] * f0,
                         2 * accumulator + chunk * f1 * f2 + 2 * modes[2] * f2)
    # Replay only label elimination of the provided path to recover its
    # already-budgeted largest intermediate. This is not a path search.
    pending = [set(text) for text in labels]
    largest = math.prod(dimensions[name] for name in output)
    for step in kernel_path:
        if len(step) != 2 or len(set(step)) != 2 or any(i < 0 or i >= len(pending) for i in step):
            return None
        selected = set().union(*(pending[index] for index in step))
        remaining = [value for index, value in enumerate(pending) if index not in step]
        needed = set(output).union(*remaining)
        retained = selected.intersection(needed)
        largest = max(largest, math.prod(dimensions[name] for name in retained))
        pending = remaining + [retained]
    if len(pending) != 1 or scratch > largest:
        return None
    return SparseSpinRecipe(
        tuple(order), tuple(permutations), tuple(mode_order), tuple(free_shapes),
        tuple(free_labels.index(name) for name in output), modes, chunk,
        paired_scratch <= largest,
    )


def prepare_sparse_spin_elemental(recipe, elemental, backend):
    """Pack current Phi before upstream legs exist; no IO or persistent cache."""
    if isinstance(recipe, BatchedSparseSpinRecipe):
        base = recipe.scalar
        first_mode, second_mode, third_mode = base.mode_permutation
        if recipe.operand_lanes[1]:
            return backend.ascontiguousarray(
                elemental.transpose((0, first_mode + 1, third_mode + 1, second_mode + 1))
            ).reshape(recipe.width, base.modes[0], -1)
        recipe = base
    first_mode, second_mode, third_mode = recipe.mode_permutation
    return backend.ascontiguousarray(
        elemental.transpose((first_mode, third_mode, second_mode))
    ).reshape(recipe.modes[0], -1)


def contract_sparse_spin(recipe, entries, operands, backend, *, prepared_elemental=None):
    """Eliminate three local spin indices without materializing G times Phi.

    Grouping by the third spin avoids one large final GEMM per nonzero. For
    each (second, third) spin pair, mix the small first propagator slices
    before the dense mode sum. First mode intermediates are released per
    pair, final GEMMs are chunked, and only the complete result survives into
    the caller's snapshot phase. Pack Phi as (first, third, second) modes once
    so every first GEMM already has the layout required by the second GEMM.
    The scalar route keeps its two-dimensional GEMMs. A shared leading lane
    uses batched matmul with singleton views for static operands.
    """
    if isinstance(recipe, BatchedSparseSpinRecipe):
        return _contract_batched_sparse_spin(recipe, entries, operands, backend,
                                             prepared_elemental=prepared_elemental)
    dtype = operands[0].dtype
    modes = recipe.modes
    free = tuple(math.prod(shape) for shape in recipe.free_shapes)
    legs = tuple(
        operands[index].transpose(permutation).reshape(4, modes[leg], free[leg])
        for leg, (index, permutation) in enumerate(zip(recipe.operand_order, recipe.permutations))
    )
    elemental = (prepare_sparse_spin_elemental(recipe, operands[1], backend)
                 if prepared_elemental is None else prepared_elemental)
    result = backend.zeros(free, dtype=dtype)
    groups = {}
    for i, j, k, weight in entries:
        groups.setdefault(k, {}).setdefault(j, []).append((i, weight))
    if len(groups) > 2 and recipe.pair_groups:
        # Pack two exact third-spin groups into one final GEMM. Each pair's
        # mode result is written directly into its columns, so no independent
        # per-group accumulator or broadcast operand remains resident.
        grouped = tuple(groups.items())
        for offset in range(0, len(grouped), 2):
            batch = grouped[offset:offset+2]
            accumulator = backend.empty((free[0], free[1], len(batch)*modes[2]), dtype=dtype)
            for slot, (k, group) in enumerate(batch):
                target = accumulator[..., slot*modes[2]:(slot+1)*modes[2]]
                for term, (j, first_spin) in enumerate(group.items()):
                    mixed = None
                    for i, weight in first_spin:
                        contribution = legs[0][i] * np.dtype(dtype).type(weight)
                        if mixed is None:
                            mixed = contribution
                        else:
                            mixed += contribution
                        del contribution
                    packed = (mixed.T @ elemental).reshape(-1, modes[1])
                    del mixed
                    contribution = (packed @ legs[1][j]).reshape(free[0], modes[2], free[1])
                    del packed
                    if term == 0:
                        target[...] = contribution.transpose(0, 2, 1)
                    else:
                        target += contribution.transpose(0, 2, 1)
                    del contribution
                del target
            last = backend.concatenate(tuple(legs[2][k] for k, _ in batch), axis=0)
            for start in range(0, free[0], recipe.chunk):
                stop = min(start + recipe.chunk, free[0])
                left = accumulator[start:stop].reshape(-1, len(batch)*modes[2])
                part = (left @ last).reshape(stop-start, free[1], free[2])
                result[start:stop] += part
                del left, part
            del last, accumulator
        grouped_shape = tuple(size for shape in recipe.free_shapes for size in shape)
        return result.reshape(grouped_shape).transpose(recipe.output_permutation)
    for k, group in groups.items():
        accumulator = None
        for j, first_spin in group.items():
            mixed = None
            for i, weight in first_spin:
                contribution = legs[0][i] * np.dtype(dtype).type(weight)
                if mixed is None:
                    mixed = contribution
                else:
                    mixed += contribution
                del contribution
            packed = (mixed.T @ elemental).reshape(-1, modes[1])
            del mixed
            contribution = (packed @ legs[1][j]).reshape(free[0], modes[2], free[1])
            del packed
            if accumulator is None:
                accumulator = contribution
            else:
                accumulator += contribution
            del contribution
        for start in range(0, free[0], recipe.chunk):
            stop = min(start + recipe.chunk, free[0])
            left = accumulator[start:stop].transpose(0, 2, 1).reshape(-1, modes[2])
            part = (left @ legs[2][k]).reshape(stop - start, free[1], free[2])
            result[start:stop] += part
            del left, part
        del accumulator
    grouped_shape = tuple(size for shape in recipe.free_shapes for size in shape)
    return result.reshape(grouped_shape).transpose(recipe.output_permutation)


def _contract_batched_sparse_spin(recipe, entries, operands, backend, *, prepared_elemental):
    """Eliminate the three mode axes with shared-lane batched matmul.

    A static operand gets a singleton leading dimension, so NumPy/CuPy
    broadcast its view in matmul. No Python loop iterates over lanes.
    """
    base = recipe.scalar
    dtype = operands[0].dtype
    modes = base.modes
    free = tuple(math.prod(shape) for shape in base.free_shapes)
    legs = []
    for leg, (index, permutation) in enumerate(zip(base.operand_order, base.permutations)):
        value = operands[index]
        if recipe.operand_lanes[index]:
            value = value.transpose((0,) + tuple(axis + 1 for axis in permutation))
            value = value.reshape(recipe.width, 4, modes[leg], free[leg])
        else:
            value = value.transpose(permutation).reshape(1, 4, modes[leg], free[leg])
        legs.append(value)
    elemental = (prepare_sparse_spin_elemental(recipe, operands[1], backend)
                 if prepared_elemental is None else prepared_elemental)
    result = backend.zeros((recipe.width, *free), dtype=dtype)
    groups = {}
    for i, j, k, weight in entries:
        groups.setdefault(k, {}).setdefault(j, []).append((i, weight))
    for k, group in groups.items():
        accumulator = None
        for j, first_spin in group.items():
            mixed = None
            for i, weight in first_spin:
                contribution = legs[0][:, i] * np.dtype(dtype).type(weight)
                if mixed is None:
                    mixed = contribution
                else:
                    mixed += contribution
            # (B,F0,I) @ (B or static,I,K*J) -> (B,F0,K,J).
            packed = (mixed.transpose(0, 2, 1) @ elemental).reshape(-1, free[0] * modes[2], modes[1])
            # (B,F0*K,J) @ (B or static,J,F1) -> (B,F0,K,F1).
            contribution = (packed @ legs[1][:, j]).reshape(-1, free[0], modes[2], free[1])
            if accumulator is None:
                accumulator = contribution
            else:
                accumulator += contribution
            del mixed, packed, contribution
        for start in range(0, free[0], base.chunk):
            stop = min(start + base.chunk, free[0])
            left = accumulator[:, start:stop].transpose(0, 1, 3, 2).reshape(
                accumulator.shape[0], (stop - start) * free[1], modes[2])
            # (B,chunk*F1,K) @ (B or static,K,F2).
            part = (left @ legs[2][:, k]).reshape(-1, stop - start, free[1], free[2])
            result[:, start:stop] += part
            del left, part
        del accumulator
    grouped_shape = (recipe.width,) + tuple(size for shape in base.free_shapes for size in shape)
    return result.reshape(grouped_shape).transpose(recipe.output_permutation)


__all__ = ['sparse_spin_recipe', 'sparse_spin_pattern_supported', 'contract_sparse_spin',
           'prepare_sparse_spin_elemental']
