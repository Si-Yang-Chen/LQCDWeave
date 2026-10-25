"""Trusted current numeric IDs gather raw inputs without retaining owners."""
from dataclasses import replace
from types import MappingProxyType
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import SourcePreparation, TensorPreparation, TensorSpec
from lqcdweave.contraction.execution.cfg_runtime import _LocalLoader
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import bind_prepared_loader, _NumericNodeBinding
from lqcdweave.contraction.execution.preparation_kernel import FusedSpinPreparation
from .test_cfg_template_preparation import _backend, _spin, _template


class _Blocks(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def _fixture(width=9, *, backend=None, dtype=np.float64, repeated=None):
    dtype = np.dtype(dtype)
    spin_dtype = np.float32 if dtype.itemsize in (4, 8) and dtype != np.float64 else np.float64
    spin = np.eye(4, dtype=spin_dtype)
    closure = {f'leaf-{index}': _spin(spin, raw=f'raw-{index % 2}')
               for index in range(width)}
    leaves = tuple(closure) if repeated is None else tuple(repeated)
    closure['stack'] = SourcePreparation(('raw-0', 'raw-1'), recipe=TensorPreparation('stack', leaves))
    specs = {key: TensorSpec((4,4,2,3), dtype) for key in closure}
    specs['stack'] = TensorSpec((len(leaves),4,4,2,3), dtype)
    tables, item_specs, bindings, raw_sets = {}, {}, [], []
    for point in range(2):
        nodes = _NumericNodeBinding({key: point*1000+index+1 for index,key in enumerate(closure)})
        for key, prep in closure.items():
            recipe = prep.recipe
            inputs = ((f'{recipe.inputs[0]}:{point}',) if recipe.kind == 'spin_spatial' else
                      tuple(str(nodes[item]) for item in leaves))
            blocks = inputs if recipe.kind == 'spin_spatial' else tuple(f'raw-{lane}:{point}' for lane in range(2))
            global_prep = replace(prep, blocks=blocks, recipe=replace(recipe, inputs=inputs))
            tables[str(nodes[key])] = global_prep
            item_specs[str(nodes[key])] = specs[key]
        bindings.append(nodes)
        values = {}
        for lane in range(2):
            base = np.arange(12, dtype=np.float64).reshape(2,6)+point*20+lane*4
            if dtype.kind == 'c': base = base+1j*(base+3)
            values[f'raw-{lane}:{point}'] = np.array(base, dtype=dtype)[:,::2]
        raw_sets.append(values)
    loader = bind_prepared_loader(None, MappingProxyType(tables), MappingProxyType(item_specs),
        backend=_backend() if backend is None else backend)
    bound = loader.bind_template_preparations(_template(closure,specs))
    return loader,bound,bindings,raw_sets,leaves,spin


def _inputs_only(program, blocks, destination=None, *, inputs=None):
    return inputs


class PreparationInputGatherCpuTests(unittest.TestCase):
    def test_single_leaf_native_mapping_skips_text_ids_and_keeps_current_unary_inputs(self):
        class NumericOnly(dict):
            def __getitem__(self, key):
                if type(key) is str:
                    raise AssertionError('native unary preparation rebuilt a text node ID')
                return super().__getitem__(key)

        loader, bound, nodes, raw, _, _ = _fixture(1)
        # The public protocol fallback still has its original text identities;
        # native single-leaf roles must consume the already built numeric index.
        loader._template_raw_inputs = NumericOnly(loader._template_raw_inputs)
        with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
            for point in range(2):
                for key in ('leaf-0', 'stack'):
                    current = _Blocks(raw[point])
                    values = loader.materialize_template_preparation(bound, key, nodes[point], current)
                    self.assertIsInstance(values, tuple)
                    self.assertEqual(len(values), 1)
                    self.assertIs(values[0], raw[point][f'raw-0:{point}'])
                    self.assertEqual(current.reads, [f'raw-0:{point}'])
                    current[f'raw-0:{point}'] = np.full((2, 3), 100 + point)
                    self.assertIs(loader.materialize_template_preparation(
                        bound, key, nodes[point], current)[0], current[f'raw-0:{point}'])

    def test_single_leaf_native_fallback_keeps_text_id_and_raw_key_errors(self):
        loader, bound, nodes, raw, _, _ = _fixture(1)
        unknown = _NumericNodeBinding(nodes[0])
        unknown['leaf-0'] = 99999
        with self.assertRaises(KeyError) as caught:
            loader.materialize_template_preparation(bound, 'leaf-0', unknown, raw[0])
        self.assertEqual(caught.exception.args, ('99999',))
        missing = _Blocks(raw[0])
        del missing['raw-0:0']
        with self.assertRaises(KeyError) as caught:
            loader.materialize_template_preparation(bound, 'leaf-0', nodes[0], missing)
        self.assertEqual(caught.exception.args, ('raw-0:0',))
        self.assertEqual(missing.reads, ['raw-0:0'])
        with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
            ordinary = loader.materialize_template_preparation(bound, 'leaf-0', dict(nodes[0]), raw[0])
            native = loader.materialize_template_preparation(bound, 'leaf-0', nodes[0], raw[0])
        self.assertIs(ordinary[0], native[0])

    def test_low_reuse_native_roles_gather_current_inputs_in_original_order(self):
        for width in (6,9,16,21,27):
            with self.subTest(width=width):
                loader,bound,nodes,raw,leaves,_ = _fixture(width)
                for point in range(2):
                    local = _LocalLoader(loader,nodes[point],bound,preparation_keys=('stack',))
                    self.assertIsNone(local.instance_preparations)
                    current = _Blocks(raw[point])
                    with patch.object(FusedSpinPreparation,'materialize',_inputs_only):
                        result = local.materialize('stack',None,current)
                        expected_keys = tuple(f'raw-{index%2}:{point}' for index in range(width))
                        self.assertEqual(current.reads,list(expected_keys))
                        self.assertTrue(all(value is raw[point][key] for value,key in zip(result,expected_keys,strict=True)))
                        replacement = np.full((2,3),99+point)
                        current[f'raw-0:{point}'] = replacement
                        result = local.materialize('stack',None,current)
                        self.assertIs(result[0],replacement)

    def test_explicit_repeated_leaf_scalar_and_pair_getters_preserve_expansion(self):
        for leaves in (('leaf-0',)*16, ('leaf-0','leaf-1')*8):
            with self.subTest(unique=len(set(leaves))):
                loader,bound,nodes,raw,_,_ = _fixture(2,repeated=leaves)
                current = _Blocks(raw[0])
                with patch.object(FusedSpinPreparation,'materialize',_inputs_only):
                    values = loader.materialize_template_preparation(bound,'stack',nodes[0],current)
                self.assertEqual(current.reads,[f'raw-{i}:0' for i in range(len(set(leaves)))])
                self.assertEqual(len(values),16)
                self.assertTrue(all(value is raw[0][f'raw-{int(key[-1])}:0'] for value,key in zip(values,leaves,strict=True)))

    def test_external_numeric_mapping_and_debug_keep_original_resolver(self):
        loader,bound,nodes,raw,_,_ = _fixture()
        with patch.object(FusedSpinPreparation,'materialize',_inputs_only):
            expected = loader.materialize_template_preparation(bound,'stack',dict(nodes[0]),raw[0])
            actual = loader.materialize_template_preparation(bound,'stack',nodes[0],raw[0])
            self.assertTrue(all(a is b for a,b in zip(expected,actual,strict=True)))
            with execution_debug(True):
                self.assertIsNone(loader.materialize_template_preparation(bound,'stack',nodes[0],raw[0]))

    def test_first_missing_local_global_and_raw_identity_errors_are_preserved(self):
        loader,bound,nodes,raw,_,_ = _fixture()
        missing = _NumericNodeBinding(nodes[0]); del missing['leaf-0']
        with self.assertRaises(KeyError) as caught:
            loader.materialize_template_preparation(bound,'stack',missing,raw[0])
        self.assertEqual(caught.exception.args,('leaf-0',))
        unknown = _NumericNodeBinding(nodes[0]); unknown['leaf-0'] = 99999
        with self.assertRaises(KeyError) as caught:
            loader.materialize_template_preparation(bound,'stack',unknown,raw[0])
        self.assertEqual(caught.exception.args,('99999',))
        blocks = _Blocks(raw[0]); del blocks['raw-0:0']
        with self.assertRaises(KeyError) as caught:
            loader.materialize_template_preparation(bound,'stack',nodes[0],blocks)
        self.assertEqual(caught.exception.args,('raw-0:0',))
        self.assertEqual(blocks.reads,['raw-0:0'])

    def test_current_input_arrays_are_not_retained_by_descriptor_or_loader(self):
        loader,bound,nodes,raw,_,_ = _fixture()
        references = [weakref.ref(value) for values in raw for value in values.values()]
        with patch.object(FusedSpinPreparation,'materialize',_inputs_only):
            for point in range(2):
                result = loader.materialize_template_preparation(bound,'stack',nodes[point],raw[point])
                self.assertEqual(len(result),9)
        del result,raw
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        other,_,_,_,_,_ = _fixture()
        with self.assertRaisesRegex(ValueError,'another loader scope'):
            other.materialize_template_preparation(bound,'stack',nodes[0],{})


class PreparationInputGatherCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError as error:
            raise unittest.SkipTest('requires CuPy') from error
        cupy.cuda.runtime.getDeviceProperties(0)
        cls.cp = cupy

    def test_current_strided_inputs_match_original_debug_and_numpy_on_nondefault_stream(self):
        cp = self.cp
        stream = cp.cuda.Stream(non_blocking=True)
        for dtype in (np.float32,np.float64,np.complex64,np.complex128):
            with self.subTest(dtype=np.dtype(dtype).name), stream:
                loader,bound,nodes,host_sets,leaves,spin = _fixture(backend=cp,dtype=dtype)
                outputs = []
                for point in range(2):
                    blocks = {key: cp.asarray(value) for key,value in host_sets[point].items()}
                    if point:
                        blocks = {key:value[:,::-1] for key,value in blocks.items()}
                    current = _LocalLoader(loader,nodes[point],bound,preparation_keys=('stack',))
                    self.assertIsNone(current.instance_preparations)
                    actual = current.materialize('stack',None,blocks)
                    original = loader.materialize_template_preparation(bound,'stack',dict(nodes[point]),blocks)
                    with execution_debug(True):
                        audited = current.materialize('stack',None,blocks)
                    host = {key:(value[:,::-1] if point else value) for key,value in host_sets[point].items()}
                    expected = np.stack([np.einsum('ab,ij->abij',spin,host[f'raw-{int(leaf.rsplit("-",1)[1])%2}:{point}'])
                                         for leaf in leaves]).astype(dtype)
                    stream.synchronize()
                    for value in (actual,original,audited):
                        np.testing.assert_array_equal(cp.asnumpy(value),expected)
                    outputs.append(actual)
                self.assertNotEqual(outputs[0].data.ptr,outputs[1].data.ptr)


if __name__ == '__main__':
    unittest.main()
