"""Pure saved preparation descriptions recover without binding callbacks."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, TensorPreparation, DiagramBindings,
)
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion.api import describe_meson_matrix, bind_meson_matrix_catalog
from lqcdweave.contraction.execution.api import PreparedSourceLoader, bind_prepared_loader
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


class PreparationRecipeTests(unittest.TestCase):
    def test_constant_roundtrip_preserves_complex_bytes_and_ownership(self):
        value = np.array([[1+2j, -3j]], dtype='>c16')
        constant = ArrayConstant.from_array(value)
        restored = decode_plan_value(encode_plan_value(constant))
        value[:] = 0
        result = restored.array()
        np.testing.assert_array_equal(result, [[1+2j, -3j]])
        self.assertEqual(result.dtype, np.dtype('>c16'))
        self.assertFalse(result.flags.writeable)
        with self.assertRaises(ValueError):
            result.flags.writeable = True

    def test_unknown_or_malformed_preparation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown'):
            TensorPreparation('arbitrary_callback')
        with self.assertRaisesRegex(ValueError, 'byte extent'):
            ArrayConstant((4,), 'complex128', '00')
        with self.assertRaisesRegex(ValueError, 'each axis'):
            TensorPreparation('spin_spatial', ('raw',),
                (ArrayConstant.from_array(np.eye(4)),), permutation=(0,0))

    def test_ordinary_dense_sources_have_archivable_preparations(self):
        from lqcdweave.contraction.correlator import calculate
        binding, _, propagator = fixture()
        calculation = calculate(task(structured=False), vertex_map=binding,
            propagator_map={'S^q': propagator}, configuration='cfg', usedNe=3,
            optimize='none', execute=False, memory_limit=1 << 30)
        loader = calculation.inputs.loader
        # All source transforms are saved descriptions; this path must not
        # call the former Meson materializer during numerical execution.
        self.assertTrue(all(p.recipe is not None for p in calculation.inputs.preparations.values()))
        restored = decode_plan_value(encode_plan_value(calculation.inputs.preparations))
        physical = loader
        recovered = bind_prepared_loader(physical, restored, calculation.inputs.problem.specs)
        replacement = replace(calculation.inputs, loader=recovered, preparations=restored)
        expanded = replace(calculation.expanded, inputs=replacement)
        path = replace(calculation.path, expanded=expanded)
        expected = calculation.run().value
        with patch.object(Meson, 'materialize_unweighted_factor', side_effect=AssertionError('closure')):
            actual = replace(calculation, expanded=expanded, path=path).run().value
        np.testing.assert_allclose(actual, expected)

    def test_matrix_stack_recovers_from_pure_records(self):
        binding, _, propagator = fixture()
        bindings = DiagramBindings(vertex_map=binding, configuration='cfg', usedNe=3,
            propagator_map={'S^q':propagator, r'S^q_\mathrm{local}':propagator})
        expanded = bind_meson_matrix_catalog(
            describe_meson_matrix(task(structured=True, flavor='I0'), bindings=bindings),
            bindings=bindings)
        original = expanded.inputs.loader
        preps = decode_plan_value(encode_plan_value(original.preparations))
        from lqcdweave.contraction.expansion.api import restore_input_loader
        requests = decode_plan_value(encode_plan_value(expanded.inputs.read_requests))
        # Reconstruct from reader APIs and records alone, without the original
        # registry, vertex bindings or any materialization closure.
        _, elemental, _ = fixture()
        readers = {request.reader_role: propagator if request.family == 'propagator' else elemental
                   for request in requests.values()}
        readers.update({alias.reader_role: propagator if alias.family == 'propagator' else elemental
                        for request in requests.values() for alias in request.aliases})
        recovered_physical = restore_input_loader(
            requests, readers, preps, original.specs, configuration='cfg',
            constants=original.constant_values,
        )
        self.assertEqual(recovered_physical.preparations, preps)
        self.assertEqual(recovered_physical.specs, original.specs)
        self.assertEqual(recovered_physical.constant_values, original.constant_values)
        self.assertFalse(hasattr(recovered_physical, 'materialize'))
        recovered = bind_prepared_loader(
            recovered_physical, recovered_physical.preparations, recovered_physical.specs,
        )
        original_prepared = bind_prepared_loader(
            original, original.preparations, original.specs,
            constant_values=original.constant_values,
        )
        for key, prep in expanded.inputs.preparations.items():
            self.assertIsNotNone(prep.recipe)
            raw = {block: original.read_block(block, expanded.inputs.blocks[block]) for block in prep.blocks}
            expected = original_prepared.materialize(key, None, raw)
            with patch.object(Meson, 'materialize_unweighted_factor', side_effect=AssertionError('closure')):
                actual = recovered.materialize(key, None, raw)
            np.testing.assert_allclose(actual, expected)

    def test_restored_inputs_reject_incompatible_metadata_before_reading(self):
        from lqcdweave.contraction.expansion.api import restore_input_loader
        from lqcdweave.contracts.correlator.api import SourceReadRequest, TensorAxisMetadata
        from unittest.mock import Mock
        metadata = TensorAxisMetadata(('mode',),(3,),'complex128','input','selected',
            axis_bindings={'mode':('basis', (0,1,2))})
        request = SourceReadRequest('propagator:q','propagator',(0,0,'vsv'),(),metadata)
        reader = Mock(cfg='cfg')
        reader.describe_block.return_value = replace(metadata,shape=(2,),axis_bindings={})
        with self.assertRaisesRegex(ValueError,'axes, shape'):
            restore_input_loader({'raw':request},{'propagator:q':reader},{},{},configuration='cfg')
        reader.block.assert_not_called()

    def test_restored_read_aliases_cannot_split_a_shared_input(self):
        from lqcdweave.contraction.expansion.api import restore_input_loader
        from lqcdweave.contracts.correlator.api import SourceReadRequest, TensorAxisMetadata
        from unittest.mock import Mock
        metadata = TensorAxisMetadata(('mode',),(3,),'complex128','input','selected')
        alias = SourceReadRequest('propagator:local','propagator',(0,0,'vsv'),(),metadata)
        request = SourceReadRequest('propagator:q','propagator',(0,0,'vsv'),(),metadata,aliases=(alias,))
        a, b = Mock(cfg='cfg'), Mock(cfg='cfg')
        a.describe_block.return_value = metadata
        b.describe_block.return_value = replace(metadata,data_identity='different-input')
        with self.assertRaisesRegex(ValueError,'aliases'):
            restore_input_loader({'raw':request},{'propagator:q':a,'propagator:local':b},{},{},configuration='cfg')
        a.block.assert_not_called()
        b.block.assert_not_called()

    def test_saved_inputs_reject_changed_momentum_order_before_reading(self):
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.expansion.api import restore_input_loader
        binding, elemental, propagator = fixture()
        elemental.momenta = [[0,0,0]]
        calculation = calculate(task(structured=False), vertex_map=binding,
            propagator_map={'S^q':propagator}, configuration='cfg', usedNe=3,
            optimize='none', execute=False, memory_limit=1 << 30)
        inputs = calculation.inputs
        roles = inputs.loader.input_readers
        original = elemental.momenta
        elemental.momenta = np.asarray(original)
        restore_input_loader(inputs.read_requests,roles,
            inputs.loader.preparations,inputs.loader.specs,configuration='cfg')
        original[0][0] = 1
        elemental.momenta = original
        with patch.object(elemental, 'block', side_effect=AssertionError('numeric read')):
            with self.assertRaisesRegex(ValueError, 'conventions'):
                restore_input_loader(inputs.read_requests,roles,
                    inputs.loader.preparations,inputs.loader.specs,configuration='cfg')

    def test_dense_factor_description_matches_existing_dagger_and_profile(self):
        from lqcdweave.operators.single_hadron.api import SpectralProfile
        from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.contracts.correlator.api import SourcePreparation, TensorSpec
        _, elemental, _ = fixture()
        profile = SpectralProfile(np.array([[[1,2,3],[3,2,1]]],complex), cfg='cfg')
        operator = Operator.from_terms('profile', [OperatorTerm(3,ElementalComponent(0),0,profile=profile)])
        for dagger in (False,True):
            meson = Meson(elemental,operator,dagger,phase_convention='anchor')
            descriptor, = meson.describe_factors(0,'v2v',usedNe=3)
            raw = meson.read_factor_block(descriptor)
            expected = meson.materialize_unweighted_factor(descriptor,raw)
            recipe = decode_plan_value(encode_plan_value(meson.describe_factor_preparation(descriptor,'raw')))
            loader = PreparedSourceLoader(None, {'value':SourcePreparation(('raw',), recipe=recipe)},
                {'value':TensorSpec(expected.shape,expected.dtype)})
            actual = loader.materialize('value',None,{'raw':raw})
            np.testing.assert_allclose(actual,expected)

            # The device binding must run dagger/profile/spin preparation on
            # its supplied backend, without mutating the caller's cache.
            from types import SimpleNamespace
            from unittest.mock import Mock
            backend=SimpleNamespace(asarray=Mock(wraps=np.asarray),
                zeros_like=np.zeros_like,multiply=Mock(wraps=np.multiply),
                einsum=Mock(wraps=np.einsum))
            bound=loader.with_backend(backend)
            rebound=bound.materialize('value',None,{'raw':raw})
            np.testing.assert_allclose(rebound,expected)
            self.assertIs(loader.backend,np)
            self.assertIsNot(bound.constants,loader.constants)
            self.assertTrue(backend.asarray.called)
            self.assertTrue(backend.multiply.called)
            backend.einsum.assert_not_called()

    def test_saved_stack_rejects_scalar_shape_before_io(self):
        from lqcdweave.contraction.expansion.api import restore_input_loader
        from lqcdweave.contracts.correlator.api import SourcePreparation, TensorSpec
        from unittest.mock import Mock
        reader = Mock()
        preps = {'a':SourcePreparation((), recipe=TensorPreparation('stack', ('a',)))}
        # A shape mismatch is detected even before visiting a self reference.
        with self.assertRaisesRegex(ValueError, 'shape'):
            restore_input_loader({}, {'reader':reader}, preps, {'a':TensorSpec((), 'complex128')}, debug=True)
        reader.block.assert_not_called()

    def test_saved_stack_requires_known_members(self):
        from lqcdweave.contraction.expansion.api import restore_input_loader
        from lqcdweave.contracts.correlator.api import SourcePreparation, TensorSpec
        prep = SourcePreparation((), recipe=TensorPreparation('stack', ('missing',)))
        with self.assertRaisesRegex(ValueError, 'unknown stack member'):
            restore_input_loader({}, {}, {'a':prep}, {'a':TensorSpec((1,3), 'complex128')}, debug=True)


if __name__ == '__main__':
    unittest.main()
