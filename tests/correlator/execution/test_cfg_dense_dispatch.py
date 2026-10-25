"""Direct cfg dispatch retains current values, views and mapping extensions."""
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import Action, TensorSpec
from lqcdweave.contraction.execution.bound_program import BoundProgram
from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.operations.contract import bind_dense_action, bind_dense_call
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView, _ResultOwner, bind_result_views,
)


class CfgDenseDispatchTests(unittest.TestCase):
    def program(self, equation='ij,jk->ik', operands=('left', 'right'), shapes=((2, 3), (3, 2))):
        action = Action('contract', 'out', operands, equation)
        catalog = {key: TensorSpec(shape, 'float64') for key, shape in zip(operands, shapes)}
        call = bind_dense_call(action, bind_dense_action(action, catalog, np), np,
                               shared_residency=True)
        return BoundProgram((call,), ('out',), np, False, None, shared_residency=True)

    def test_fresh_bindings_values_and_ownership_without_mapping_forwarders(self):
        program = self.program()
        outputs = []
        references = []
        for offset, scale in ((20, 1), (70, -3)):
            left = np.arange(6.).reshape(2, 3) * scale
            right = np.arange(6.).reshape(3, 2)
            expected = left @ right
            arrays = {offset: left, offset + 1: right}
            node = _NodeView({'left': offset, 'right': offset + 1, 'out': offset + 2}, arrays)
            # Output collection uses ordinary mapping access; isolate the
            # dispatch itself to prove there is no per-operand forwarding.
            with patch.object(_NodeView, '__getitem__', side_effect=AssertionError('forwarded')):
                program.calls[0]((node, None, *node._dense_call_context()))
            np.testing.assert_array_equal(arrays[offset + 2], expected)
            actual, = program.run(SimpleNamespace(), node)
            outputs.append(actual)
            references.append(weakref.ref(left))
            arrays.clear()
            del node, left, right
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        np.testing.assert_array_equal(outputs[1], -3 * outputs[0])

    def test_deferred_duplicate_operand_and_subsequent_replacement(self):
        program = self.program('ij,ij->', ('left', 'left'), ((2, 3), (2, 3)))
        source = np.arange(6.).reshape(2, 3)
        owner = _ResultOwner(source)
        _, layout = bind_result_views(source.shape, source.dtype,
            (('left', (slice(None), slice(None)), ()),))[0]
        deferred = _DeferredResultView(owner, layout)
        arrays = {1: deferred}
        node = _NodeView({'left': 1, 'out': 2}, arrays)
        value, = program.run(None, node)
        self.assertIsNot(type(arrays[1]), _DeferredResultView)
        self.assertTrue(np.shares_memory(arrays[1], source))
        np.testing.assert_allclose(value, np.sum(source * source))
        arrays[1] = source * 2
        value, = program.run(None, node)
        np.testing.assert_allclose(value, 4 * np.sum(source * source))

    def test_mapping_extension_and_kernel_failure_keep_existing_semantics(self):
        class TracedView(_NodeView):
            def __getitem__(self, key):
                self.reads.append(key)
                return super().__getitem__(key)
        node = TracedView({'left': 1, 'right': 2, 'out': 3},
                          {1: np.ones((2, 3)), 2: np.ones((3, 2))})
        node.reads = []
        self.program().run(None, node)
        self.assertEqual(node.reads, ['left', 'right', 'out'])
        action = Action('contract', 'out', ('left', 'right'), 'ij,jk->ik')
        def fail(operands):
            raise RuntimeError('kernel failure')
        call = bind_dense_call(action, fail, np, shared_residency=True)
        native = _NodeView(node.binding, node.arrays)
        previous = node.arrays[3]
        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
            call((native, None, *native._dense_call_context()))
        self.assertIs(node.arrays[3], previous)


if __name__ == '__main__':
    unittest.main()
