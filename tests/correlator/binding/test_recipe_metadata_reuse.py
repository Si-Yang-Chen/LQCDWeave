"""Immutable recipe definitions can be reused without sharing mutable views."""
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.operators.correlator.api import Tag, Hadron, Flavor
from lqcdweave.operators.single_hadron.operator import InsertionRow


def test_definition_reuse_preserves_independent_operator_and_mode_views():
    vertices = RecipeVertex(object(), momenta=((0, 0, 0),))
    row = vertices.symbol('op_00_gamma5_A1', tag=Tag(0, 'sink'))[0]
    original = InsertionRow.__call__
    calls = []
    def counted(self, *args):
        calls.append(args)
        return original(self, *args)
    with patch.object(InsertionRow, '__call__', counted):
        first = vertices(row)
        expected = first.operator.terms
        first.operator.name = 'mutated'
        first.operator.parts.append('mutated')
        first.operator._explicit = ()
        selected = first.with_mode_indices(((0,), (1,)))
        second = vertices(row)
    assert len(calls) == 1
    assert first is not second and first.operator is not second.operator
    assert second.operator.name == row.hadron_name and second.operator.parts == []
    assert second.operator.terms == expected
    assert selected.mode_indices == ((0,), (1,)) and second.mode_indices is None
    assert second.reader is None and second.key is None


def test_ordered_momentum_instances_and_dagger_remain_separate():
    first = RecipeVertex(object(), momenta=((0, 0, 0), (0, 0, 1)))
    reordered = RecipeVertex(object(), momenta=((0, 0, 1), (0, 0, 0)))
    row = first.symbol('op_00_gamma5_A1', (0, 0, 1), irrep='A2', tag=Tag(0, 'sink'))[0]
    a, b = first(row), reordered(row)
    assert {term.momentum for term in a.operator.terms} == {1}
    assert {term.momentum for term in b.operator.terms} == {0}
    assert all(term.momentum_vector == (0, 0, 1) for term in a.operator.terms + b.operator.terms)
    source = Hadron(row, Flavor('uc')).conjugate().irrep_row
    assert first(source).dagger is True and first(row).dagger is False
    assert len(first._operator_terms) == 2 and len(reordered._operator_terms) == 1


def test_cached_terms_keep_numerical_vertex_equal_with_distinct_readers():
    from lqcdweave.adapters.elemental import ArrayElemental
    from lqcdweave.operators.single_hadron.api import meson55_preset
    components = tuple(row['component'] for row in meson55_preset(version='v2'))
    data = np.arange(len(components)*4, dtype=np.complex64).reshape(len(components), 1, 1, 2, 2)
    reader = ArrayElemental({'v2v': data}, components, cfg='cfg-a', momentum_list=((0, 0, 0),), phase_convention='midpoint')
    vertices = RecipeVertex(reader)
    row = vertices.symbol('op_00_gamma5_A1', tag=Tag(0, 'sink'))[0]
    initial, repeated = vertices(row), vertices(row)
    np.testing.assert_array_equal(initial.get(0), repeated.get(0))
    other = ArrayElemental({'v2v': data+1}, components, cfg='cfg-b', momentum_list=((0, 0, 0),), phase_convention='midpoint')
    rebound = repeated.with_reader(other)
    np.testing.assert_array_equal(vertices(row).get(0), initial.get(0))
    assert rebound.reader is other and vertices(row).reader is reader
