"""Explicit spatial/spin rounding boundaries, independent of gamma names."""
from dataclasses import replace
from types import MappingProxyType
import unittest
import numpy as np
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, TensorPreparation, TensorSpec, SourcePreparation, ContractionTerm, ContractionProblem, SourceSpec)
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.contraction.planning.api import ContractionState, CostProfile
from lqcdweave.storage.contraction_plans.api import _encode_plan_value, _decode_plan_value

class SpinSpatialFactorTests(unittest.TestCase):
    def test_small_required_wide_reduction_fits_original_workspace(self):
        from lqcdweave.contraction.execution.reduction import bind_precise_reduction
        from lqcdweave.contracts.correlator.api import AxisPattern
        pattern=AxisPattern(((0,1),(2,1,0)),(2,))
        rng=np.random.default_rng(261032)
        values=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
                     for shape in ((3,3),(2,3,3)))
        run=bind_precise_reduction(pattern,tuple(v.shape for v in values),
            (np.complex64,)*2,np,force_matrix=True)
        self.assertIsNotNone(run)
        self.assertLessEqual(run.workspace_bytes,sum(v.nbytes for v in values)+2*np.dtype('complex64').itemsize)
        expected=np.einsum(values[0].astype(np.complex128),pattern.inputs[0],
            values[1].astype(np.complex128),pattern.inputs[1],pattern.output).astype(np.complex64)
        np.testing.assert_array_equal(run(*values),expected)

    def test_spatial_variants_resolve_each_registered_input_once(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import lqcdweave.contraction.expansion.monomial_spin as monomial
        def constant(value):
            return ArrayConstant.from_array(np.asarray(value, dtype=np.complex128))
        identity = np.eye(4)
        phase = identity * 1j
        profile = np.array([1., 2.])
        first = TensorPreparation('spin_spatial', ('raw',),
            (constant(identity), constant(np.eye(2)), constant(np.ones((2, 3))), constant(profile)),
            configuration='cfg', compute_dtype='complex64')
        recipes = {'a': first, 'phase': replace(first, constants=(constant(phase), *first.constants[1:])),
                   'profile': replace(first, constants=(*first.constants[:3], constant(profile[::-1]))),
                   'cast': replace(first, compute_dtype='complex128'),
                   'parent': replace(first, inputs=('other-raw',)),
                   'dagger': replace(first, conjugate=True, permutation=(1, 0))}
        registry = SimpleNamespace(specs={}, sources={}, preparations={})
        for key, recipe in recipes.items():
            registry.specs[key] = TensorSpec((4, 4, 2, 3), np.complex128)
            registry.sources[key] = SourceSpec(key, 0)
            registry.preparations[key] = SourcePreparation(recipe.inputs, recipe=recipe)
        variants = monomial.MonomialPreparations(registry)
        with patch.object(monomial, 'unit_spin', wraps=monomial.unit_spin) as probe:
            outputs = {key: variants.separate(key) for key in recipes}
            for repeat in range(10):
                self.assertEqual({key: variants.separate(key) for key in recipes}, outputs)
            self.assertEqual(probe.call_count, len(recipes))
        self.assertEqual(outputs['a'], outputs['phase'])
        self.assertEqual(len(set(outputs.values())), 5)
        # No identity can leak from the previous registry binding.
        fresh = monomial.MonomialPreparations(registry)
        with patch.object(monomial, 'unit_spin', wraps=monomial.unit_spin) as probe:
            fresh.separate('a')
            self.assertEqual(probe.call_count, 1)

    def test_spatial_profile_preserves_original_order_dagger_and_casts(self):
        rng=np.random.default_rng(261029)
        for dtype in (np.complex64,np.complex128):
            spin=np.array([[0,0,0,1j],[0,0,-1,0],[0,-1j,0,0],[1,0,0,0]],dtype=dtype)
            constants=(spin,rng.normal(size=(3,2)).astype(dtype),rng.normal(size=(3,3)).astype(dtype),np.array([.7,-.2,.4],dtype))
            full=TensorPreparation('spin_spatial',('raw',),tuple(ArrayConstant.from_array(x) for x in constants),conjugate=True,permutation=(1,0),configuration='cfg',compute_dtype=np.dtype(dtype).name)
            spatial=replace(full,spatial_only=True)
            preps={'full':full,'spatial':spatial}
            for repeat in range(2):
                raw=(rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))).astype(dtype)[:,::-1]
                arrays={}
                for key,recipe in preps.items():
                    shape=(4,4,3,2) if key=='full' else (3,2)
                    loader=bind_prepared_loader(None,MappingProxyType({key:SourcePreparation(('raw',),recipe=recipe)}),MappingProxyType({key:TensorSpec(shape,dtype)}),backend=np)
                    arrays[key]=loader.materialize(key,None,{'raw':raw})
                for row,column in enumerate((3,2,1,0)):
                    np.testing.assert_array_equal(arrays['full'][row,column],spin[row,column]*arrays['spatial'])
                self.assertFalse(np.shares_memory(arrays['spatial'],raw))
            self.assertEqual(_decode_plan_value(_encode_plan_value(spatial)),spatial)

    def test_explicit_sum_barrier_blocks_narrow_gram_and_replays_wide_terminal(self):
        shapes=((2,65),(3,65),(2,2),(2,3))
        specs={str(i):TensorSpec(shape,np.complex64) for i,shape in enumerate(shapes)}
        term=ContractionTerm(tuple(specs),'ak,bk,ca,db->cd')
        axes=term.subscripts.inputs
        barrier=(((axes[0][1],),(axes[0][0],axes[1][0])),)
        term=replace(term,sum_barriers=barrier)
        problem=ContractionProblem(specs,(term,),{k:SourceSpec(k,0) for k in specs})
        state=ContractionState.initial(problem)
        profile=CostProfile()
        for key in specs:state=state.load(key,profile,1<<24)
        with self.assertRaisesRegex(ValueError,'coupled spin'):
            state.contract(0,(0,1),profile,1<<24)
        actions=[]
        for pair in ((0,2),(0,1),(0,1)):
            state=state.contract(0,pair,profile,1<<24);actions.append(state.actions[-1])
        self.assertTrue(actions[-1].wide_accumulation)
        self.assertFalse(any(a.wide_accumulation for a in actions[:-1]))
        rng=np.random.default_rng(261030)
        arrays={key:(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64) for key,shape in zip(specs,shapes)}
        for action in actions:
            value,_=bind_dense_action(action,state.catalog,np)(tuple(arrays[k] for k in action.operands));arrays[action.tensor]=value
        a,b=(arrays[k] for k in actions[-1].operands)
        equation=actions[-1].equation.normalized()
        expected=np.einsum(a.astype(np.complex128),equation.inputs[0],b.astype(np.complex128),equation.inputs[1],equation.output).astype(np.complex64)
        np.testing.assert_array_equal(arrays[actions[-1].tensor],expected)
        self.assertEqual(_decode_plan_value(_encode_plan_value(actions[-1])),actions[-1])
        self.assertEqual(_decode_plan_value(_encode_plan_value(term)),term)
        legacy=_encode_plan_value(ContractionTerm(tuple(specs),'ak,bk,ca,db->cd'))
        del legacy['value']['sum_barriers']
        self.assertFalse(_decode_plan_value(legacy).sum_barriers)

    def test_unconstrained_gram_is_not_substituted_into_a_constrained_term(self):
        specs={str(i):TensorSpec(shape,np.complex64) for i,shape in enumerate(((2,65),(3,65),(2,2),(2,3)))}
        plain=ContractionTerm(tuple(specs),'ak,bk,ca,db->cd')
        axes=plain.subscripts.inputs
        protected=replace(plain,sum_barriers=(((axes[0][1],),(axes[0][0],axes[1][0])),))
        state=ContractionState.initial(ContractionProblem(specs,(plain,protected),{k:SourceSpec(k,0) for k in specs}))
        for key in specs:state=state.load(key,CostProfile(),1<<24)
        before=state.pending[1]
        state=state.contract(0,(0,1),CostProfile(),1<<24)
        self.assertEqual(state.pending[1],before)
        self.assertEqual(len(state.pending[0].factors),3)

    def test_public_narrow_i0_fixed_archive_preserves_values(self):
        self._check_public_narrow_archive('numpy')

    def _check_public_narrow_archive(self, backend):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
        from lqcdweave.contraction.correlator import calculate,save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        from test_structured_correlator import task
        rng=np.random.default_rng(261031)
        def random(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        component=ElementalComponent(0)
        elemental=ArrayElemental({'v2v':random((1,1,2,3,3))},(component,),cfg='cfg',phase_convention='anchor')
        propagator=ArrayPropagator({'vsv':random((2,2,4,4,3,3))},time_convention='absolute',cfg='cfg')
        operators={name:Operator.from_terms(name,(OperatorTerm(mask,component,0,1),)) for name,mask in (('z',0),('a',3))}
        vertices={}
        def bind(row):
            key=row.hadron_name,row.dagger
            if key not in vertices:vertices[key]=Meson(elemental,operators[row.hadron_name],row.dagger,phase_convention='anchor')
            return vertices[key]
        options=dict(vertex_map=bind,configuration='cfg',usedNe=3,
            propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator},memory_limit=1<<26,host_memory_limit=1<<26)
        selected=calculate(task(structured=True,flavor='I0'),backend=backend,execute=False,time_axes={0:0,1:1},
            search={'deadline_seconds':1.,'max_states':1000,'component_workers':1},**options)
        self.assertTrue(selected.path.plan.state.actions)
        pairs=({0:0,1:1},{0:1,1:0})
        expected=calculate(task(structured=False,flavor='I0'),optimize='none',time_pairs=pairs,**options).value
        coordinates,_=describe_coordinates(time_pairs=pairs)
        complete=complete_calculation(selected,coordinates=coordinates,readers=selected.inputs.loader.input_readers)
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'phase.json.gz';save_calculation_plan(archive,complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',side_effect=AssertionError('archive searched')):
                loaded=calculate(None,plan_archive=archive,configuration='cfg',vertex_map=elemental,
                    propagator_map=options['propagator_map'],memory_limit=1<<26,host_memory_limit=1<<26,execute=False)
                first=loaded.run().value;second=loaded.run().value
        np.testing.assert_allclose(first,expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_array_equal(first,second)


class SpinSpatialCuPyTests(unittest.TestCase):
    def test_public_narrow_i0_fixed_archive_matches_independent_cpu(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        check=SpinSpatialFactorTests('test_public_narrow_i0_fixed_archive_preserves_values')
        check._check_public_narrow_archive('cupy')
