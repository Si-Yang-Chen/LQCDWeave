"""Integer connectivity semantics, independent of einsum spelling."""
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionTerm, TensorSpec, axis_ids, parse_term,
)


class IntegerIndicesTests(unittest.TestCase):
    def test_external_integer_normalization_keeps_order_and_freezes_mutable_input(self):
        class ExternalIndex:
            def __index__(self): return 1 << 130

        class IntegerSubclass(int):
            pass

        external = [np.int64(2), ExternalIndex(), IntegerSubclass(5)]
        pattern = AxisPattern((external,), (np.uint64(2),))
        external[0] = 99
        self.assertEqual(pattern.inputs, ((2, 1 << 130, 5),))
        self.assertEqual(pattern.output, (2,))
        self.assertTrue(all(type(axis) is int for axis in pattern.inputs[0] + pattern.output))
        self.assertEqual(axis_ids(iter((np.uint64(7), 3))), (7, 3))
        self.assertEqual(axis_ids((3, 0, 1 << 130)), (3, 0, 1 << 130))
        with self.assertRaises(ValueError):
            axis_ids((np.int64(-1),))

    def test_scalar_output_still_normalizes_inputs_and_stops_at_first_bad_axis(self):
        for bad, error in ((True, TypeError), (-1, ValueError), (1.0, TypeError)):
            visited = []

            def axes():
                for value in (np.int64(3), bad, 9):
                    visited.append(value)
                    yield value

            with self.subTest(bad=bad), self.assertRaises(error):
                AxisPattern((axes(),), ())
            self.assertEqual(len(visited), 2)

    def test_external_string_is_absent_from_parsed_record(self):
        source = ContractionTerm(("a",), "ij->ji", 2)
        parsed = parse_term(source, {"a": TensorSpec((2, 3), "float64")}, 0)
        self.assertIsInstance(parsed.source.subscripts, AxisPattern)
        self.assertEqual(parsed.source.subscripts,
                         AxisPattern(((ord("i"), ord("j")),),
                                     (ord("j"), ord("i"))))
        self.assertEqual(axis_ids("ij"), (ord("i"), ord("j")))
        self.assertEqual(parsed.source.coefficient, 2)
        self.assertEqual(parsed.output_shape, (3, 2))

    def test_aligned_open_axis_and_reduction(self):
        pattern = AxisPattern(((1000, 9), (1000, 9)), (1000,))
        self.assertEqual(pattern.output_shape(((2, 3), (2, 3))), (2,))
        self.assertEqual(pattern.normalized(), AxisPattern(((0, 1), (0, 1)), (0,)))

    def test_diagonal_scalar_and_output_order(self):
        self.assertEqual(AxisPattern(((5, 5),), ()).output_shape(((3, 3),)), ())
        self.assertEqual(AxisPattern(((), ()), ()).output_shape(((), ())), ())
        self.assertEqual(AxisPattern(((5, 7),), (7, 5)).output_shape(((2, 3),)), (3, 2))

    def test_hyperedge_and_many_ids(self):
        self.assertEqual(AxisPattern(((7,), (7,), (7,)), ()).dimensions(((2,),)*3), {7: 2})
        axes = tuple(range(100))
        self.assertEqual(AxisPattern((axes,), axes).output_shape(((1,)*100,)), (1,)*100)

    def test_rank_and_extent_errors(self):
        for pattern, shapes in (
            (AxisPattern(((0, 0),), ()), ((2, 3),)),
            (AxisPattern(((0,), (0,)), ()), ((0,), (2,))),
            (AxisPattern(((0,),), (0,)), ((2, 3),)),
            (AxisPattern(((0,),), (0,)), ()),
        ):
            with self.subTest(pattern=pattern, shapes=shapes), self.assertRaises(ValueError):
                pattern.dimensions(shapes)

    def test_invalid_ids_and_outputs(self):
        for inputs, output, error in (
            (((True,),), (), TypeError), (((-1,),), (), ValueError),
            (((1.0,),), (), TypeError), (("i",), (), TypeError),
            (((0,),), (0, 0), ValueError), (((0,),), (1,), ValueError),
            ((), (), ValueError),
        ):
            with self.subTest(inputs=inputs, output=output), self.assertRaises(error):
                AxisPattern(inputs, output)


if __name__ == "__main__":
    unittest.main()
