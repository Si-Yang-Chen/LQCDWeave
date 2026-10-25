"""Catalog -> common physical plan -> replay -> existing output assembly."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import sympy as sp

from test_structured_correlator import fixture,task
from lqcdweave.contracts.correlator.api import DiagramBindings,describe_coordinates
from lqcdweave.contracts.execution import TermValue
from lqcdweave.contraction.expansion.api import describe_meson_matrix,bind_meson_matrix_catalog,MatrixCatalogBinder
from lqcdweave.contraction.planning.api import plan_matrix_actions,CostProfile
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.operators.correlator.api import (
    remove_disconneted_diagram,split_meson_mode_pairs,Hadron,HadronIrrep,Tag,Flavor,gen_correlator,
)
from lqcdweave.contraction.correlator import calculate


class MatrixInputTests(unittest.TestCase):
    def setUp(self):
        self.bind,self.elemental,self.propagator = fixture()
        self.bindings = DiagramBindings(vertex_map=self.bind,configuration='cfg',usedNe=3,
            propagator_map={'S^q':self.propagator,r'S^q_\mathrm{local}':self.propagator})

    def describe(self,request,**options):
        catalog = describe_meson_matrix(request,bindings=self.bindings)
        return bind_meson_matrix_catalog(catalog,bindings=self.bindings,**options)

    def run_expanded(self,expanded):
        if expanded.inputs is None:
            return assemble_correlator(expanded,())
        inputs = expanded.inputs
        plan = plan_matrix_actions(inputs.problem,inputs.matrix_kernels,inputs.blocks,inputs.preparations,
            profile=CostProfile(),memory_limit=1 << 30,reserved_host_bytes=inputs.assembly_host_bytes)
        values = replay_scene_storage(plan,inputs.loader,io_overlap=False)
        return assemble_correlator(expanded,tuple(TermValue(term,expanded.coordinates.points[0].key,value)
            for term,value in zip(inputs.term_ids,values,strict=True)))

    def reference(self,request,**options):
        return calculate(request,vertex_map=self.bind,propagator_map=self.bindings.propagator_map,
            configuration='cfg',usedNe=3,optimize='none',memory_limit=1 << 30,**options).value

    def test_charged_I0_I1_complex_coefficients_and_pc(self):
        for flavor in ('charged','I0','I1'):
            for boundaries in (None,(0,1,2,3)):
                with self.subTest(flavor=flavor,boundaries=boundaries):
                    request = task(structured=True,flavor=flavor)
                    if boundaries:
                        request = split_meson_mode_pairs(request,partitions=boundaries)
                    expanded = self.describe(request,row_block=2)
                    actual = self.run_expanded(expanded)
                    np.testing.assert_allclose(actual,self.reference(request),rtol=3e-12,atol=3e-12)

    def test_whole_matrix_catalog_closes_private_factors_before_global_planning(self):
        from lqcdweave.contraction.planning.meson_core import plan_meson_whole
        for flavor, factorized in ((flavor, factorized) for flavor in ('charged', 'I0', 'I1')
                                    for factorized in (False, True)):
            with self.subTest(flavor=flavor, factorized=factorized):
                request = task(structured=True, flavor=flavor)
                expanded = self.describe(request, _whole_vertices=True, _factorize_spin=factorized)
                inputs = expanded.inputs
                assert inputs.problem.meson_cores
                assert all(term.kernel is not None for term in inputs.problem.terms)
                plan = plan_meson_whole(inputs.problem, inputs.blocks, inputs.preparations,
                    profile=CostProfile(), memory_limit=1 << 30,
                    reserved_host_bytes=inputs.assembly_host_bytes)
                values = replay_scene_storage(plan, inputs.loader, io_overlap=False)
                actual = assemble_correlator(expanded, tuple(TermValue(term,
                    expanded.coordinates.points[0].key, value)
                    for term, value in zip(inputs.term_ids, values, strict=True)))
                np.testing.assert_allclose(actual, self.reference(request), rtol=3e-12, atol=3e-12)

    def test_no_scalar_or_component_pair_expansion_and_no_reads(self):
        request = split_meson_mode_pairs(task(structured=True,flavor='I0'),partitions=(0,1,2,3))
        with (patch('lqcdweave.contraction.expansion.structured._Compiler.task',side_effect=AssertionError('cell')),
              patch.object(self.elemental,'block',side_effect=AssertionError('IO'))):
            expanded = self.describe(request)
        self.assertEqual(len(expanded.inputs.problem.terms),3**4+2*3**2)
        self.assertEqual(sum(core.kind=='connected' for core in expanded.inputs.matrix_kernels),3**4)
        self.assertTrue(all(core.mode_block and core.partitions is None
                            for core in expanded.inputs.matrix_kernels))
        self.assertEqual(expanded.output_recipe.root.kind,'add')

    def test_unequal_physical_blocks_keep_pc_and_dagger_numerics(self):
        boundaries = (0,1,3)
        request = split_meson_mode_pairs(task(structured=True,flavor='I0'),partitions=boundaries)
        expanded = self.describe(request)
        self.assertEqual(len(expanded.problem.terms),2**4+2*2**2)
        widths = {size for spec in expanded.problem.specs.values()
                  if spec.description is not None and spec.description.role=='propagator'
                  for size in spec.shape[-2:]}
        self.assertEqual(widths,{1,2})
        self.assertFalse(any(len(spec.shape)==3 and spec.shape[-2:]==(3,3)
                             for spec in expanded.problem.specs.values()))
        scalar = split_meson_mode_pairs(task(structured=False,flavor='I0'),partitions=boundaries)
        expected = self.reference(scalar)
        actual = self.run_expanded(expanded)
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)

    def test_linear_rows_mixing_physical_modes_stay_compact(self):
        a = HadronIrrep('a',(0,0,0),'A_1',-1,Tag(11,0))[0]
        b = HadronIrrep('a',(0,0,0),'A_1',-1,Tag(12,0))[0]
        axis = [Hadron((a+b)/sp.sqrt(2), Flavor('uu')),
                Hadron((a-b)/sp.sqrt(2), Flavor('uu'))]
        request = gen_correlator([axis.copy(),axis.copy()], [0,1], [False,True],
                                 structured=True,degenerate=True)
        original = self.bind
        def bind(row):
            pair = ((0,), (1,2)) if row.tag.tag == 11 else ((1,2), (0,))
            return original(row).with_mode_indices(pair[::-1] if row.dagger else pair)
        self.bind = bind
        self.bindings = replace(self.bindings, vertex_map=bind)
        with (patch('lqcdweave.contraction.expansion.structured._Compiler.task',
                    side_effect=AssertionError('scalar fallback')),
              patch.object(self.elemental,'block',side_effect=AssertionError('early IO'))):
            expanded = self.describe(request)
        self.assertEqual(len(expanded.correlator.groups), 4)
        self.assertTrue(expanded.inputs.matrix_kernels)
        self.assertTrue(all(child.kind == 'scatter' for child in expanded.output_recipe.root.children))
        np.testing.assert_allclose(self.run_expanded(expanded), self.reference(request),
                                   rtol=3e-12,atol=3e-12)

    def test_multiple_groups_reuse_computation_and_inputs(self):
        full = task(structured=True,flavor='I0')
        connected = remove_disconneted_diagram(full,[r'S^q_\mathrm{local}'])
        request = {'full':full,'connected':connected,'disconnected':full-connected,'copy':full}
        expanded = self.describe(request)
        self.assertEqual(len(expanded.inputs.problem.terms),3)
        actual = self.run_expanded(expanded)
        expected = self.reference(request)
        for key in request:
            np.testing.assert_allclose(actual[key],expected[key],rtol=3e-12,atol=3e-12)

    def test_partitioned_and_unpartitioned_outputs_share_one_delivery(self):
        full=task(structured=True,flavor='I0')
        request={'full':full,'pc':split_meson_mode_pairs(full,partitions=(0,1,3))}
        expanded=self.describe(request)
        self.assertEqual(len(expanded.problem.terms),3+2**4+2*2**2)
        actual=self.run_expanded(expanded)
        ordinary=task(structured=False,flavor='I0')
        expected=self.reference({'full':ordinary,'pc':split_meson_mode_pairs(ordinary,partitions=(0,1,3))})
        for key in request:
            np.testing.assert_allclose(actual[key],expected[key],rtol=3e-12,atol=3e-12)

    def test_direct_binding_describes_each_component_once_without_diagrams(self):
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
        from lqcdweave.contraction.expansion.state_graph import lower_state_graph
        from lqcdweave.operators.correlator.api import Diagram
        full = task(structured=True,flavor='I0')
        request = {'full':full,'copy':full,'disconnected':full-remove_disconneted_diagram(
            full,[r'S^q_\mathrm{local}'])}
        described = describe_meson_matrix(request,bindings=self.bindings)
        describe = Meson.describe_factors
        with (patch.object(LegacyDiagramBinding,'describe',side_effect=AssertionError('scalar expansion')),
              patch.object(Diagram,'__init__',side_effect=AssertionError('representative graph')),
              patch('lqcdweave.contraction.expansion.matrix_inputs.lower_state_graph',
                    wraps=lower_state_graph) as lower,
              patch.object(Meson,'describe_factors',autospec=True,side_effect=describe) as factors):
            expanded = bind_meson_matrix_catalog(described,bindings=self.bindings)
        self.assertEqual(lower.call_count,len(described.catalog.networks))
        self.assertEqual(factors.call_count,len(described.catalog.components))
        self.assertEqual(len(expanded.inputs.problem.terms),3)
        actual = self.run_expanded(expanded)
        expected = self.reference(request)
        for name in request:
            np.testing.assert_allclose(actual[name],expected[name],rtol=3e-12,atol=3e-12)

    def test_direct_binding_checks_actual_endpoint_basis_before_io(self):
        element = self.elemental.describe_block
        propagator = self.propagator.describe_block
        def element_metadata(*args,**kwargs):
            metadata = element(*args,**kwargs)
            return replace(metadata,axis_bindings={axis:('basis-A',tuple(range(size)))
                for axis,size in zip(metadata.axes,metadata.shape,strict=True)})
        def propagator_metadata(*args,**kwargs):
            metadata = propagator(*args,**kwargs)
            return replace(metadata,axis_bindings={
                'source_mode':('basis-B',tuple(range(metadata.shape[-1]))),
                'sink_mode':('basis-A',tuple(range(metadata.shape[-2])))})
        with (patch.object(self.elemental,'describe_block',side_effect=element_metadata),
              patch.object(self.propagator,'describe_block',side_effect=propagator_metadata),
              patch.object(self.elemental,'block',side_effect=AssertionError('early IO')),
              patch.object(self.propagator,'block',side_effect=AssertionError('early IO'))):
            with self.assertRaisesRegex(ValueError,'actual dataset ID or ordered coordinates'):
                self.describe(task(structured=True))

    def test_reusable_binder_rechecks_reader_configuration(self):
        described = describe_meson_matrix(task(structured=True),bindings=self.bindings)
        binder = MatrixCatalogBinder(described,bindings=self.bindings)
        binder.bind()
        with (patch.object(self.propagator,'cfg','another-cfg'),
              patch.object(self.propagator,'block',side_effect=AssertionError('early IO'))):
            with self.assertRaisesRegex(ValueError,'configuration disagrees'):
                binder.bind()

    def test_loader_does_not_retain_construction_registry(self):
        import gc
        import weakref
        from lqcdweave.contraction.expansion.physical_inputs import PhysicalInputRegistry
        registries = []
        def registry(bindings, **kwargs):
            value = PhysicalInputRegistry(bindings, **kwargs)
            registries.append(weakref.ref(value))
            return value
        request = task(structured=True,flavor='I0')
        with patch('lqcdweave.contraction.expansion.matrix_inputs.PhysicalInputRegistry',
                   side_effect=registry):
            expanded = self.describe(request)
        gc.collect()
        self.assertEqual(len(registries),1)
        self.assertIsNone(registries[0]())
        np.testing.assert_allclose(self.run_expanded(expanded),self.reference(request),
                                   rtol=3e-12,atol=3e-12)

    def test_reusable_binder_keeps_jobs_and_coefficients_and_rebinds_actual_times(self):
        from lqcdweave.contraction.expansion.matrix_inputs import _catalog_jobs
        request = split_meson_mode_pairs(task(structured=True,flavor='I0'),partitions=(0,1,2,3))
        described = describe_meson_matrix(request,bindings=self.bindings)
        with patch('lqcdweave.contraction.expansion.matrix_inputs._catalog_jobs', wraps=_catalog_jobs) as build:
            binder = MatrixCatalogBinder(described,bindings=self.bindings,row_block=2)
            bound = []
            for times in ({0:1,1:0},{0:0,1:1}):
                coordinates,_ = describe_coordinates(time_axes=times)
                with patch.object(self.elemental,'read_slab',side_effect=AssertionError('early IO')):
                    bound.append(binder.bind(coordinates=coordinates))
                np.testing.assert_allclose(self.run_expanded(bound[-1]),self.reference(request,time_axes=times),
                                           rtol=3e-12,atol=3e-12)
            self.assertEqual(build.call_count,1)
        first,second = bound
        self.assertIsNot(first.inputs.loader,second.inputs.loader)
        self.assertIsNot(first.identity_token,second.identity_token)
        self.assertNotEqual(first.coordinates,second.coordinates)
        constants = [key for key in first.problem.sources if key.startswith('matrix:coefficient:')]
        self.assertTrue(constants)
        from lqcdweave.contraction.execution.api import bind_prepared_loader
        first_loader=bind_prepared_loader(first.inputs.loader,first.inputs.loader.preparations,first.inputs.loader.specs)
        second_loader=bind_prepared_loader(second.inputs.loader,second.inputs.loader.preparations,second.inputs.loader.specs)
        for key in constants:
            a = first_loader.materialize(key,first.problem.sources[key],{})
            b = second_loader.materialize(key,second.problem.sources[key],{})
            self.assertIs(a,b)
            self.assertFalse(a.flags.writeable)
        # Later binding does not replace the first coordinate's materializer.
        np.testing.assert_allclose(self.run_expanded(first),self.reference(request,time_axes={0:1,1:0}),
                                   rtol=3e-12,atol=3e-12)

    def test_mixed_flavor_rectangles_and_zero_wick_blocks(self):
        rows = [HadronIrrep(name,(0,0,0),'A_1',-1,Tag(0,0))[0] for name in ('z','a')]
        axis = [Hadron(rows[0],Flavor('ud')),Hadron(rows[1],Flavor('uu')+Flavor('dd'))]
        request = gen_correlator([axis.copy(),axis.copy()],[0,1],[False,True],structured=True,degenerate=True)
        request = split_meson_mode_pairs(request,partitions=(0,1,2,3))
        actual = self.run_expanded(self.describe(request))
        np.testing.assert_allclose(actual,self.reference(request),rtol=3e-12,atol=3e-12)
        self.assertTrue(np.all(actual[0,1]==0))
        self.assertTrue(np.all(actual[1,0]==0))

    def test_coordinate_iterator_binds_on_demand_without_retaining_history(self):
        import gc
        import weakref
        request=split_meson_mode_pairs(task(structured=True,flavor='I0'),partitions=(0,1,2,3))
        described=describe_meson_matrix(request,bindings=self.bindings)
        binder=MatrixCatalogBinder(described,bindings=self.bindings,row_block=2)
        pairs=({0:1,1:0},{0:0,1:1},{0:1,1:0})
        coordinates,_=describe_coordinates(time_pairs=pairs)
        with patch.object(binder,'bind',wraps=binder.bind) as bind:
            iterator=binder.iter_bind(coordinates)
            self.assertEqual(bind.call_count,0)
            for i,times in enumerate(pairs):
                with patch.object(self.elemental,'block',side_effect=AssertionError('early IO')):
                    expanded=next(iterator)
                self.assertEqual(bind.call_count,i+1)
                self.assertEqual(expanded.coordinates.points,(coordinates.points[i],))
                self.assertFalse(expanded.coordinates.scan_shape)
                np.testing.assert_allclose(self.run_expanded(expanded),self.reference(request,time_axes=times),
                                           rtol=3e-12,atol=3e-12)
                previous=weakref.ref(expanded)
                del expanded
                # The spy records return-independent call arguments, not outputs.
                gc.collect()
                self.assertIsNone(previous())
            with self.assertRaises(StopIteration):
                next(iterator)

    def test_coordinate_iterator_requires_explicit_coordinate_contract(self):
        binder=MatrixCatalogBinder(describe_meson_matrix(task(structured=True),bindings=self.bindings),
                                   bindings=self.bindings)
        with self.assertRaisesRegex(TypeError,'CoordinateSpec'):
            binder.iter_bind([{0:1,1:0}])

    def test_single_source_loop_and_fixed_rebinding(self):
        row = HadronIrrep('a',(0,0,0),'A_1',-1,Tag(0,0))[0]
        request = gen_correlator([[Hadron(row,(Flavor('uu')+Flavor('dd'))/sp.sqrt(2))]],
            [1],[True],structured=True,degenerate=True)
        request = split_meson_mode_pairs(request,partitions=(0,1,2,3))
        coordinates,_ = describe_coordinates(time_axes={1:0})
        actual = self.run_expanded(self.describe(request,coordinates=coordinates))
        expected = self.reference(request,time_axes={1:0})
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)

    def test_zero_only_request_and_scan_rejection(self):
        rows = [HadronIrrep(name,(0,0,0),'A_1',-1,Tag(0,0))[0] for name in ('z','a')]
        request = gen_correlator([[Hadron(rows[0],Flavor('ud'))],[Hadron(rows[1],Flavor('uu'))]],
            [0,1],[False,True],structured=True,degenerate=True)
        expanded = self.describe(request)
        self.assertIsNone(expanded.inputs)
        np.testing.assert_equal(self.run_expanded(expanded),np.zeros((1,1),dtype=complex))
        coords,_ = describe_coordinates(time_axes={0:[0,1],1:0})
        with self.assertRaises(NotImplementedError):
            self.describe(task(structured=True),coordinates=coords)

    def test_kernel_directory_alignment_is_checked(self):
        inputs = self.describe(task(structured=True,flavor='I0')).inputs
        with self.assertRaises(ValueError):
            replace(inputs,matrix_kernels=inputs.matrix_kernels[:1])

    def test_single_time_pair_preserves_requested_coordinate_axis(self):
        request = task(structured=True,flavor='I0')
        coordinates,_ = describe_coordinates(time_pairs=[{0:1,1:0}])
        expanded = self.describe(request,coordinates=coordinates)
        actual = self.run_expanded(expanded)
        expected = self.reference(request,time_pairs=[{0:1,1:0}])
        self.assertEqual(actual.shape,(2,2,1))
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)


if __name__ == '__main__':unittest.main()
