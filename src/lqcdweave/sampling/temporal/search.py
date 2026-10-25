"""Exact, streaming time-selection search with replaceable rules and ranking."""

from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
from itertools import combinations
from numbers import Integral
from typing import TypeVar

import numpy as np

from lqcdweave.contracts.time_selection import (
    TimeSelection, TimeSelectionRecord, positive_integer,
)
from .criteria import Criterion, covers_all_differences, is_legal

Score = TypeVar("Score")
DEFAULT_CRITERIA = (covers_all_differences,)


class SearchLimitExceeded(RuntimeError):
    """Search is incomplete: neither absence nor optimality has been proved."""


def _dimensions(Lt: int, Nt: int) -> tuple[int, int]:
    Lt, Nt = positive_integer(Lt, "Lt"), positive_integer(Nt, "Nt")
    if Nt > Lt:
        raise ValueError("Nt must not exceed Lt")
    return Lt, Nt


def iter_legal_schemes(
    Lt: int,
    Nt: int,
    loop: int,
    *,
    criteria: Iterable[Criterion] = DEFAULT_CRITERIA,
    max_candidates: int | None = 1_000_000,
) -> Iterator[TimeSelection]:
    """Yield all legal offset subsets, in lexicographic order, including shifts.

    Exhausting this iterator proves completeness. A budget exception leaves only
    a partial enumeration. ``None`` removes the candidate limit. An incompatible
    Nt/loop yields no schemes; invalid lattice dimensions raise ValueError.
    """
    Lt, Nt = _dimensions(Lt, Nt)
    loop = positive_integer(loop, "loop")
    if Lt % loop:
        raise ValueError("loop must divide Lt")
    if max_candidates is not None:
        max_candidates = positive_integer(max_candidates, "max_candidates")
    criteria = tuple(criteria)
    if Nt * loop % Lt:
        return
    size = Nt * loop // Lt
    # This necessary bound applies ONLY when full coverage is explicitly required.
    if any(rule is covers_all_differences for rule in criteria):
        if size * (size - 1) + 1 < loop:
            return
    for checked, offsets in enumerate(combinations(range(loop), size), start=1):
        if max_candidates is not None and checked > max_candidates:
            raise SearchLimitExceeded(
                f"loop={loop}: exceeded {max_candidates} candidates; search incomplete"
            )
        selection = TimeSelection(Lt, loop, offsets)
        if is_legal(selection, criteria):
            yield selection


def find_min_loop(
    Lt: int,
    Nt: int,
    *,
    criteria: Iterable[Criterion] = DEFAULT_CRITERIA,
    max_candidates: int | None = 1_000_000,
) -> int | None:
    """Return the smallest divisor of Lt with a legal witness, or None if none.

    The candidate budget is per loop. An unfinished smaller-loop search raises;
    it must never be silently skipped when claiming a minimum.
    """
    Lt, Nt = _dimensions(Lt, Nt)
    criteria = tuple(criteria)
    for loop in range(1, Lt + 1):
        if Lt % loop == 0:
            schemes = iter_legal_schemes(
                Lt, Nt, loop, criteria=criteria, max_candidates=max_candidates
            )
            if next(schemes, None) is not None:
                return loop
    return None


def select_best(
    schemes: Iterable[TimeSelection],
    *,
    key: Callable[[TimeSelection], Score],
    maximize: bool = False,
) -> TimeSelection | None:
    """Rank the entire input using a comparable key (scalar or tuple).

    Return None on empty input; exact ties keep the first scheme. Exceptions,
    including incomplete enumeration, propagate rather than returning a partial
    optimum. Caller-provided numeric keys must be finite (in particular not NaN).
    """
    best = None
    for scheme in schemes:
        score = key(scheme)
        if best is None or (score > best_score if maximize else score < best_score):
            best, best_score = scheme, score
    return best


def select_t(
    Lt: int,
    Nt: int,
    *,
    key: Callable[[TimeSelection], Score],
    loop: int | None = None,
    criteria: Iterable[Criterion] = DEFAULT_CRITERIA,
    maximize: bool = False,
    max_candidates: int | None = 1_000_000,
    cfg: str | int | None = None,
) -> TimeSelection | TimeSelectionRecord | None:
    """Optimize a template, then optionally realize a reproducible cfg shift.

    With cfg, preserve the template's slot order after a periodic translation;
    callers save these actual times and reuse the saved record thereafter.
    """
    seed = None
    if cfg is not None:
        if isinstance(cfg, bool) or not isinstance(cfg, (str, Integral)) or str(cfg) == "":
            raise ValueError("cfg must be a nonempty string or integer")
        positive_integer(Lt, "Lt")
        seed = int.from_bytes(sha256(str(cfg).encode("utf-8")).digest(), "big")
    criteria = tuple(criteria)
    if loop is None:
        loop = find_min_loop(Lt, Nt, criteria=criteria, max_candidates=max_candidates)
    if loop is None:
        return None
    best = select_best(
        iter_legal_schemes(Lt, Nt, loop, criteria=criteria, max_candidates=max_candidates),
        key=key,
        maximize=maximize,
    )
    if best is None or seed is None:
        return best
    return realize_time_selection(best, cfg=cfg)


def realize_time_selection(template: TimeSelection, *, cfg: str | int) -> TimeSelectionRecord:
    """Realize one cfg's ordered shift from a template without repeating search.

    Uses the same stable PCG64 mapping as ``select_t(..., cfg=...)``. Save once
    and reload that actual record for production, recovery and reconstruction.
    """
    if not isinstance(template, TimeSelection):
        raise TypeError("template must be a TimeSelection search template")
    if isinstance(cfg, bool) or not isinstance(cfg, (str, Integral)) or str(cfg) == "":
        raise ValueError("cfg must be a nonempty string or integer")
    seed = int.from_bytes(sha256(str(cfg).encode("utf-8")).digest(), "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    shift = int(rng.integers(template.Lt))
    return TimeSelectionRecord(template.Lt, tuple((t + shift) % template.Lt for t in template.times))
