"""Dense scene validation retains no unused parsed-term catalog."""
import unittest
from unittest.mock import patch
import weakref

from lqcdweave.contracts.correlator import scene
from lqcdweave.contracts.correlator.api import (
    AxisPattern, BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
)


class SceneValidationMemoryTests(unittest.TestCase):
    def test_dense_parses_are_released_before_checking_the_next_term(self):
        original = scene.parse_term
        references, previous_live = [], []

        class ParseOwner:
            def __init__(self, parsed):
                self.parsed = parsed

        def parse(*args):
            previous_live.append(sum(ref() is not None for ref in references))
            owner = ParseOwner(original(*args))
            references.append(weakref.ref(owner))
            return owner

        terms = tuple(ContractionTerm(('A',), AxisPattern(((index,),), ())) for index in range(32))
        with patch.object(scene, 'parse_term', side_effect=parse) as checked:
            problem = ContractionProblem({'A': TensorSpec((2,), 'float64')}, terms,
                                         {'A': SourceSpec('A', 16)})
        self.assertEqual(checked.call_count, len(terms))
        self.assertEqual(problem.terms, terms)
        self.assertEqual(previous_live, [0] * len(terms))
        self.assertTrue(all(ref() is None for ref in references))

    def test_reused_layout_keeps_actual_input_keys_and_terms(self):
        pattern = AxisPattern(((0, 1), (1, 2)), (0, 2))
        specs = {'A': TensorSpec((2, 3), 'float64'), 'B': TensorSpec((3, 4), 'float64'),
                 'C': TensorSpec((2, 3), 'float64')}
        sources = {key: SourceSpec(key, 256) for key in specs}
        terms = (ContractionTerm(('A', 'B'), pattern), ContractionTerm(('C', 'B'), pattern)) * 16
        with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
            problem = ContractionProblem(specs, terms, sources)
        self.assertEqual(checked.call_count, 1)
        self.assertEqual(problem.terms, terms)
        self.assertEqual(problem.terms[1].inputs, ('C', 'B'))
        self.assertEqual(set(problem.specs), {'A', 'B', 'C'})
        # A separate construction independently checks its own shapes.
        with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
            ContractionProblem(specs, terms, sources)
        self.assertEqual(checked.call_count, 1)

    def test_reuse_does_not_hide_late_shape_key_or_coefficient_errors(self):
        pattern = AxisPattern(((0, 1), (1, 2)), (0, 2))
        specs = {'A': TensorSpec((2, 3), 'float64'), 'B': TensorSpec((3, 4), 'float64'),
                 'C': TensorSpec((2, 5), 'float64')}
        sources = {key: SourceSpec(key, 256) for key in specs}
        valid = ContractionTerm(('A', 'B'), pattern)
        for final, error, message in (
            (ContractionTerm(('C', 'B'), pattern), ValueError, r'term 17: label 1 has incompatible dimensions'),
            (ContractionTerm(('missing', 'B'), pattern), KeyError, r'term 17: no TensorSpec.*missing'),
            (ContractionTerm(('A', 'B'), pattern, 2), ValueError, 'physical scene coefficients'),
        ):
            with self.subTest(final=final), self.assertRaisesRegex(error, message):
                ContractionProblem(specs, (valid,)*17+(final,), sources)

    def test_equal_but_distinct_equations_are_not_identity_reused(self):
        first = ContractionTerm(('A',), AxisPattern(((0,),), ()))
        second = ContractionTerm(('A',), AxisPattern(((0,),), ()))
        self.assertIsNot(first.subscripts, second.subscripts)
        with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
            ContractionProblem({'A': TensorSpec((2,), 'float64')}, (first, second)*8,
                               {'A': SourceSpec('A', 16)})
        self.assertEqual(checked.call_count, 2)

    def test_extended_records_and_keys_keep_the_original_checks(self):
        class ExtendedSpec(TensorSpec):
            pass
        class ExtendedTerm(ContractionTerm):
            pass
        class ExtendedKey(str):
            pass
        cases = ((TensorSpec, ContractionTerm, ExtendedKey('A')),
                 (ExtendedSpec, ContractionTerm, 'A'), (TensorSpec, ExtendedTerm, 'A'))
        for spec, term, key in cases:
            value = term((key,), 'i->')
            with self.subTest(spec=spec, term=term, key=type(key)):
                with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
                    ContractionProblem({'A': spec((2,), 'float64')}, (value,)*3,
                                       {'A': SourceSpec('A', 16)})
                self.assertEqual(checked.call_count, 3)

    def test_layout_proof_catalog_is_bounded(self):
        terms = tuple(ContractionTerm(('A',), AxisPattern(((index,),), ())) for index in range(129))
        with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
            ContractionProblem({'A': TensorSpec((2,), 'float64')}, terms+(terms[-1],)*3,
                               {'A': SourceSpec('A', 16)})
        self.assertEqual(checked.call_count, 132)

    def test_baryon_joins_still_keep_each_actual_parsed_term(self):
        term = ContractionTerm(('g', 'phi'), AxisPattern(((0, 1, 2), (3, 4, 5)), ()))
        specs = {'g': TensorSpec((4, 4, 4), 'float64'), 'phi': TensorSpec((2, 2, 2), 'float64')}
        sources = {'g': SourceSpec('g', 512), 'phi': SourceSpec('phi', 64)}
        join = BaryonJoinSpec(0, 'source', 'g', 'phi', (0, 1, 2), (3, 4, 5))
        with patch.object(scene, '_validate_dense_terms', side_effect=AssertionError('dense-only proof')):
            with patch.object(scene, 'parse_term', wraps=scene.parse_term) as checked:
                ContractionProblem(specs, (term, term), sources, baryon_joins=(join,))
        self.assertEqual(checked.call_count, 2)

    def test_late_invalid_dense_term_still_reports_its_index(self):
        terms = ((ContractionTerm(('A',), 'i->'),) * 31
                 + (ContractionTerm(('A',), 'ij->'),))
        with self.assertRaisesRegex(ValueError, r'term 31, operand 0.*rank 1'):
            ContractionProblem({'A': TensorSpec((2,), 'float64')}, terms,
                               {'A': SourceSpec('A', 16)})


if __name__ == '__main__':
    unittest.main()
