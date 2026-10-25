from __future__ import annotations

from itertools import permutations
import inspect

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    BaryonTerm,
    BaryonVertex,
    ContractionPlan,
    TensorSpec,
)
from lqcdweave.contraction import compile_baryon_terms as RootCompileBaryonTerms
from lqcdweave.contraction.planning.api import (
    ContractionPlanCache,
    compile_baryon_terms,
    compile_baryon_terms_cached,
)


_KEYS = ("S", "E", "R", "F", "P0", "P1", "P2")


def _term(keys=_KEYS, source_slots=(0, 1, 2), coefficient=1):
    return BaryonTerm(
        source=BaryonVertex(keys[0], keys[1]),
        sink=BaryonVertex(keys[2], keys[3]),
        propagators=tuple(keys[4:]),
        source_slots=source_slots,
        coefficient=coefficient,
    )


def _specs(keys=_KEYS, *, wrong_rank=None):
    shapes = [(1, 1, 1)] * 4 + [(1, 1, 1, 1)] * 3
    if wrong_rank is not None:
        shapes[wrong_rank] = (1, 1)
    return {
        key: TensorSpec(shape, np.dtype(np.complex128))
        for key, shape in zip(keys, shapes, strict=True)
    }


class _TrapBindingValue:
    def __getattribute__(self, name):
        if name in {"shape", "dtype", "__array__"}:
            raise AssertionError(f"P inspected numerical binding attribute {name}")
        return object.__getattribute__(self, name)


def test_cached_compiler_signature_names_cache_and_keeps_greedy_default() -> None:
    signature = inspect.signature(compile_baryon_terms_cached)

    assert tuple(signature.parameters) == ("cache", "specs", "terms", "bindings", "optimize")
    assert signature.parameters["optimize"].default == "greedy"


def test_public_compile_baryon_terms_keeps_legacy_signature_and_returns_plan() -> None:
    assert RootCompileBaryonTerms is compile_baryon_terms
    plan = compile_baryon_terms(_specs(), (_term(source_slots=(2, 0, 1)),))

    assert isinstance(plan, ContractionPlan)
    assert len(plan.terms) == 1


def test_cached_compiler_never_reads_binding_values_and_rebinds_keys() -> None:
    cache = ContractionPlanCache()
    terms = (_term(source_slots=(2, 0, 1), coefficient=2j),)
    bindings = {key: _TrapBindingValue() for key in _KEYS}

    plan, rebound = compile_baryon_terms_cached(
        cache=cache, specs=_specs(), terms=terms, bindings=bindings, optimize="greedy"
    )

    assert isinstance(plan, ContractionPlan)
    assert tuple(rebound) == tuple(range(7))
    assert tuple(rebound.values()) == tuple(bindings.values())
    assert cache.stats == {"hits": 0, "misses": 1, "entries": 1, "max_entries": 8}


def test_cached_compiler_reuses_structure_with_fresh_binding_keys() -> None:
    cache = ContractionPlanCache()
    first_keys = tuple(f"first-{index}" for index in range(7))
    second_keys = tuple(f"second-{index}" for index in range(7))
    first_bindings = {key: object() for key in first_keys}
    second_bindings = {key: object() for key in second_keys}

    first_plan, _ = compile_baryon_terms_cached(
        cache, _specs(first_keys), (_term(first_keys),), first_bindings
    )
    second_plan, second_rebound = compile_baryon_terms_cached(
        cache, _specs(second_keys), (_term(second_keys),), second_bindings
    )

    assert first_plan is second_plan
    assert tuple(second_rebound.values()) == tuple(second_bindings.values())
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1


@pytest.mark.parametrize("source_slots", tuple(permutations(range(3))))
def test_compiler_accepts_each_valid_source_slot_permutation(source_slots) -> None:
    plan = compile_baryon_terms(_specs(), (_term(source_slots=source_slots),))

    assert isinstance(plan, ContractionPlan)
    assert len(plan.terms) == 1


def test_shape_validation_uses_specs_before_touching_bindings() -> None:
    bindings = {key: _TrapBindingValue() for key in _KEYS}

    with pytest.raises(ValueError, match="rank 3"):
        compile_baryon_terms_cached(
            ContractionPlanCache(), _specs(wrong_rank=0), (_term(),), bindings
        )


def test_cached_compiler_rejects_missing_and_extra_bindings() -> None:
    specs = _specs()
    terms = (_term(),)

    with pytest.raises(KeyError, match="missing bindings"):
        compile_baryon_terms_cached(None, specs, terms, {})

    with pytest.raises(KeyError, match="unknown input"):
        compile_baryon_terms_cached(
            None, specs, terms, {key: object() for key in (*_KEYS, "extra")}
        )

