"""Minimum task precision is fixed before planning, reads and contractions."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, DiagramBindings, DiagramInputs,
    ExpandedCorrelator, OutputNode, OutputRecipe, SourcePreparation, SourceSpec,
    StorageBlock, TensorPreparation, TensorSpec, TermContribution, describe_coordinates,
)
from lqcdweave.contracts.correlator.precision import minimum_contraction_dtype
from lqcdweave.contraction.expansion.precision import normalize_contraction_precision
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contraction.execution.direct import execute_direct_contraction
from lqcdweave.contraction.results.api import assemble_correlator, bind_output_assembly


def example():
    rng = np.random.default_rng(1294)
    a = (rng.normal(size=(3,5)) + 1j*rng.normal(size=(3,5))).astype('complex64')
    b = rng.normal(size=(5,2)) + 1j*rng.normal(size=(5,2))
    values = {'a':a,'b':b}
    specs = {k:TensorSpec(v.shape,v.dtype) for k,v in values.items()}
    sources = {k:SourceSpec(k,v.nbytes,snapshot_bytes=v.nbytes) for k,v in values.items()}
    blocks = {k:StorageBlock(k,k,v.nbytes,host_bytes=v.nbytes) for k,v in values.items()}
    preps = {k:SourcePreparation((k,),dtype=v.dtype.name,
             recipe=TensorPreparation('identity',(k,))) for k,v in values.items()}
    class Loader:
        calls = 0
        def read_block(self,key,block):
            self.calls+=1;return values[key]
        def close_block(self,key,value):pass
        def materialize(self,key,source,raw):return raw[key]
    loader = Loader()
    loader.specs,loader.preparations=specs,preps
    coords,batch = describe_coordinates()
    inputs = DiagramInputs(ContractionProblem(specs,(ContractionTerm(('a','b'),'ab,bc->ac'),),sources),
        blocks,preps,loader,coords,batch,('term',))
    root=OutputNode('add',children=(OutputNode('term',TermContribution('term',1.25-.4j)),
                                  OutputNode('constant',.125j)))
    return ExpandedCorrelator(None,inputs,OutputRecipe(root,('term',),coords),DiagramBindings(),coords,batch),values,loader


class MinimumPrecisionTests(unittest.TestCase):
    def test_declared_precision_overrides_inputs_without_reading(self):
        expanded, values, loader = example()
        for dtype in ('complex64', 'complex128'):
            normalized = normalize_contraction_precision(expanded, dtype=dtype)
            self.assertEqual(loader.calls, 0)
            self.assertEqual({s.dtype.name for s in normalized.tensor_specs.values()}, {dtype})
            self.assertIs(normalized, normalize_contraction_precision(normalized))
            coefficient = normalized.output_recipe.root.children[0].value.coefficient
            self.assertEqual(coefficient, np.dtype(dtype).type(1.25-.4j).item())
        self.assertEqual(values['b'].dtype, np.dtype('complex128'))
        with self.assertRaisesRegex(TypeError, 'cannot discard complex'):
            normalize_contraction_precision(expanded, dtype='float32')

    def test_declared_constant_only_task_converts_before_output_arithmetic(self):
        expanded, _, _ = example()
        root = OutputNode('constant_array', ArrayConstant.from_array(
            np.array([1.123456789+.3j], dtype='complex128')), shape=(1,))
        expanded = replace(expanded, inputs=None,
                           output_recipe=replace(expanded.output_recipe, root=root, term_ids=()))
        normalized = normalize_contraction_precision(expanded, dtype='complex64')
        self.assertEqual(normalized.output_recipe.root.value.array().dtype, np.dtype('complex64'))
        actual = bind_output_assembly(normalized.output_recipe)(())
        np.testing.assert_array_equal(actual, np.array([1.123456789+.3j], dtype='complex64'))

    def test_identity_converts_before_conjugation(self):
        from lqcdweave.contraction.execution.preparation import _prepare_identity
        raw = np.array([1.123456789+.3j], dtype='complex128')
        def conjugate(value, *, out):
            self.assertEqual(value.dtype, np.dtype('complex64'))
            return np.conjugate(value, out=out)
        backend = SimpleNamespace(empty=np.empty, conjugate=conjugate)
        recipe = TensorPreparation('identity', ('raw',), conjugate=True, compute_dtype='complex64')
        actual = _prepare_identity(backend, recipe, {'raw': raw}, None, TensorSpec((1,), 'complex64'))
        np.testing.assert_array_equal(actual, raw.astype('complex64').conj())

    def test_component_precision_and_complex_values(self):
        for inputs,expected in ((['complex64','complex128'],'complex64'),
                                (['float32','complex128'],'complex64'),
                                (['float64','complex64'],'complex64'),
                                (['float64','float32'],'float32'),
                                (['int64','complex128'],'complex128')):
            self.assertEqual(minimum_contraction_dtype(inputs),np.dtype(expected))
        with self.assertRaises(TypeError):minimum_contraction_dtype(['float16','float64'])

    def test_metadata_only_idempotent_and_preserves_raw_storage(self):
        expanded,values,loader=example()
        normalized=normalize_contraction_precision(expanded)
        self.assertEqual(loader.calls,0)
        self.assertIs(normalized,normalize_contraction_precision(normalized))
        self.assertTrue(all(s.dtype==np.dtype('complex64') for s in normalized.tensor_specs.values()))
        self.assertEqual(normalized.inputs.blocks,expanded.inputs.blocks)
        self.assertEqual(normalized.output_recipe.dtype,'complex64')
        self.assertEqual(normalized.inputs.preparations['b'].recipe.compute_dtype,'complex64')
        self.assertEqual(values['b'].dtype,np.dtype('complex128'))

    def test_real_inputs_keep_complex_output_coefficients(self):
        expanded,_,_=example()
        specs={k:replace(s,dtype='float32') for k,s in expanded.tensor_specs.items()}
        expanded=replace(expanded,inputs=replace(expanded.inputs,
            problem=replace(expanded.problem,specs=specs)))
        self.assertEqual(normalize_contraction_precision(expanded).output_recipe.dtype,'complex64')

    def test_real_inputs_keep_archived_complex_array_constants(self):
        expanded,_,_=example()
        specs={k:replace(s,dtype='float32') for k,s in expanded.tensor_specs.items()}
        value=ArrayConstant.from_array(np.array([1j],dtype='complex64'))
        expanded=replace(expanded,inputs=replace(expanded.inputs,
            problem=replace(expanded.problem,specs=specs)),
            output_recipe=replace(expanded.output_recipe,
                root=OutputNode('constant_array',value,shape=(1,))))
        normalized=normalize_contraction_precision(expanded)
        self.assertEqual(normalized.output_recipe.dtype,'complex64')
        np.testing.assert_array_equal(bind_output_assembly(normalized.output_recipe)(()),[1j])

    def test_fixed_path_and_both_assemblers_match_independent_single_precision(self):
        expanded,values,_=example()
        normalized=normalize_contraction_precision(expanded)
        path=plan_direct_execution(normalized,memory_limit=1<<26)
        result=execute_direct_contraction(normalized,path,backend=np)
        product=values['a'] @ values['b'].astype('complex64')
        # Inputs/GEMM stay c64; the established output policy evaluates small
        # products and sums in c128, then rounds the final field once.
        expected=((1.25-.4j)*product.astype('complex128')+.125j).astype('complex64')
        actual=assemble_correlator(normalized,result.term_values)
        self.assertEqual(actual.dtype,np.dtype('complex64'))
        np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-6)
        for debug in (False,True):
            bound=bind_output_assembly(normalized.output_recipe,debug=debug)
            np.testing.assert_array_equal(bound((product,)),expected)

    def test_cpu_native_workflow_and_cfg_output_stay_single_precision(self):
        from test_structured_correlator import fixture,task
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        bind,_,_=fixture()
        rng=np.random.default_rng(33)
        raw=(rng.normal(size=(2,2,4,4,3,3))+1j*rng.normal(size=(2,2,4,4,3,3))).astype('complex64')
        prop=ArrayPropagator({'vsv':raw},time_convention='absolute',cfg='cfg',data_identity='single')
        selected=calculate(task(structured=True,flavor='I0'),vertex_map=bind,
            propagator_map={'S^q':prop,r'S^q_\mathrm{local}':prop},configuration='cfg',usedNe=3,
            optimize='none',execute=False)
        self.assertEqual({s.dtype.name for s in selected.expanded.tensor_specs.values()},{'complex64'})
        result=selected.run(backend=np)
        self.assertEqual(np.asarray(result.value).dtype,np.dtype('complex64'))
        from lqcdweave.contraction.expansion.api import input_readers
        complete=complete_calculation(selected,readers=input_readers(selected.inputs))
        self.assertTrue(all(np.dtype(f.dtype)==np.dtype('complex64') for f in complete.plan.output_fields))
        np.testing.assert_allclose(complete.run(backend=np).value,result.value,rtol=3e-6,atol=3e-6)

    def test_archive_metadata_and_device_output_description_keep_dtype(self):
        from lqcdweave.storage.contraction_plans.api import encode_plan_value,decode_plan_value
        from lqcdweave.contraction.execution.output_program import describe_device_output_fields
        expanded,_,_=example()
        normalized=normalize_contraction_precision(expanded)
        for value in (normalized.output_recipe,normalized.inputs.preparations['b']):
            self.assertEqual(decode_plan_value(encode_plan_value(value)),value)
        fields=describe_device_output_fields(normalized.output_recipe,normalized.problem)
        self.assertTrue(all(np.dtype(dtype)==np.dtype('complex64') for _,_,dtype in fields))

    def test_coordinate_scan_and_powers_keep_single_precision(self):
        from lqcdweave.contraction.results.api import TermValue
        coordinates,_=describe_coordinates(time_pairs=({'sink':0,'source':1},{'sink':1,'source':0}))
        term=OutputNode('diagram_sum',(('term',1.25-.4j),))
        root=OutputNode('power',2,(term,))
        recipe=OutputRecipe(root,('term',),coordinates,dtype='complex64')
        inputs=(np.complex64(.7+.3j),np.complex64(-.2+.1j))
        values=tuple(TermValue('term',point.key,value) for point,value in zip(coordinates.points,inputs))
        actual=assemble_correlator(SimpleNamespace(output_recipe=recipe,coordinates=coordinates),values)
        expected=np.power(np.complex64(1.25-.4j)*np.asarray(inputs),2,dtype='complex64')
        self.assertEqual(actual.dtype,np.dtype('complex64'))
        np.testing.assert_allclose(actual,expected,rtol=5e-7,atol=1e-8)

    def test_matrix_zero_outputs_use_the_selected_dtype_in_both_assemblers(self):
        coordinates,_=describe_coordinates()
        recipe=OutputRecipe(OutputNode('zeros','complex128',shape=(2,3)),(),coordinates,dtype='complex64')
        for debug in (False,True):
            value=bind_output_assembly(recipe,debug=debug)(())
            self.assertEqual(value.dtype,np.dtype('complex64'))
            np.testing.assert_array_equal(value,np.zeros((2,3),dtype='complex64'))

    def test_raw_mixed_plan_is_rejected_before_any_input_read(self):
        expanded,_,loader=example()
        path=plan_direct_execution(expanded,memory_limit=1<<26)
        with self.assertRaisesRegex(ValueError,'mixed-precision'):
            execute_direct_contraction(expanded,path,backend=np)
        self.assertEqual(loader.calls,0)

    def test_uniform_opaque_preparations_need_no_precision_conversion(self):
        expanded,values,loader=example()
        preps={k:replace(p,recipe=None,dtype=None) for k,p in expanded.inputs.preparations.items()}
        loader.preparations=preps
        with self.assertRaisesRegex(NotImplementedError,'described preparation recipe'):
            normalize_contraction_precision(replace(expanded,
                inputs=replace(expanded.inputs,preparations=preps)))
        self.assertEqual(loader.calls,0)
        values['a']=values['a'].astype('complex128')
        specs={k:TensorSpec(v.shape,v.dtype) for k,v in values.items()}
        loader.specs,loader.preparations=specs,preps
        sources={k:SourceSpec(k,v.nbytes,snapshot_bytes=v.nbytes) for k,v in values.items()}
        blocks={k:StorageBlock(k,k,v.nbytes,host_bytes=v.nbytes) for k,v in values.items()}
        expanded=replace(expanded,inputs=replace(expanded.inputs,
            problem=replace(expanded.problem,specs=specs,sources=sources),preparations=preps,blocks=blocks))
        normalized=normalize_contraction_precision(expanded)
        self.assertEqual(normalized.output_recipe.dtype,'complex128')
        self.assertEqual(loader.calls,0)
        result=execute_direct_contraction(normalized,plan_direct_execution(normalized,memory_limit=1<<26),backend=np)
        expected=(1.25-.4j)*(values['a']@values['b'])+.125j
        np.testing.assert_allclose(assemble_correlator(normalized,result.term_values),expected,rtol=2e-12,atol=2e-12)

    def test_low_level_batches_and_matrix_kernels_reject_mixed_inputs(self):
        from correlator.planning.test_dense_batches import DenseBatchTests
        from correlator.execution.test_matrix_kernels import MatrixKernelTests
        from lqcdweave.contraction.planning.api import describe_dense_batch
        from lqcdweave.contraction.execution.operations.dense_batch import bind_dense_batch
        from lqcdweave.contraction.execution.api import contract_meson_matrix_pc,contract_meson_loop_pc
        actions,specs,profile=DenseBatchTests().fixture()
        saved=describe_dense_batch(actions,(0,1,2,3),specs,profile)
        mixed=dict(specs)
        mixed['b0']=replace(mixed['b0'],dtype='complex64')
        for backend in ('numpy','cupy'):
            self.assertIsNone(describe_dense_batch(actions,(0,1,2,3),mixed,profile,backend=backend))
        with self.assertRaisesRegex(ValueError,'mixed-precision'):
            bind_dense_batch(saved,mixed,np)
        arrays=list(MatrixKernelTests().fixture(3))
        arrays[0]=arrays[0].astype('complex64')
        with self.assertRaisesRegex(ValueError,'mixed-precision'):
            contract_meson_matrix_pc(*arrays)
        with self.assertRaisesRegex(ValueError,'mixed-precision'):
            contract_meson_loop_pc(arrays[0],arrays[2],arrays[4])

    def test_emitted_preparation_uses_single_precision_with_wider_raw_data(self):
        from correlator.execution.test_fused_preparation import _HostKernelBackend
        from lqcdweave.contraction.execution.preparation_kernel import FusedSpinPreparation
        backend=_HostKernelBackend()
        self.addCleanup(backend.close)
        rng=np.random.default_rng(521)
        random=lambda shape:(rng.normal(size=shape)+1j*rng.normal(size=shape))
        raw=random((3,2))[::-1]
        spin=random((4,4)).astype('complex64')
        profile=tuple(random(shape).astype('complex64') for shape in ((2,3),(2,2),(2,)))
        for rows in ((),profile):
            recipe=TensorPreparation('spin_spatial',('raw',),
                tuple(ArrayConstant.from_array(a) for a in (spin,*rows)),
                conjugate=True,permutation=(1,0),configuration='cfg' if rows else None,
                compute_dtype='complex64')
            program=FusedSpinPreparation.spin(backend,recipe,(spin,*rows),
                TensorSpec((4,4,2,3),'complex64'))
            value=raw.astype('complex64')
            if rows:
                left,right,weights=rows
                projected=np.zeros(value.shape,dtype='complex64')
                for p in range(len(weights)):
                    projected+=weights[p]*left[p].conj()[:,None]*value*right[p][None,:]
                value=projected
            expected=spin[:,:,None,None]*value.conj().T[None,None,:,:]
            actual=program.materialize({'raw':raw})
            self.assertEqual(actual.dtype,np.dtype('complex64'))
            np.testing.assert_allclose(actual,expected,rtol=5e-6,atol=5e-6)


if __name__=='__main__':unittest.main()
